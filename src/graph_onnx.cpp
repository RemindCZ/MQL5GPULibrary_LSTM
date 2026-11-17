#include "graph_fit.h"
#include "onnx_writer.h"
#include <algorithm>
#include <iomanip>
#include <locale>
#include <sstream>

namespace graphfit {
using namespace dnonnx;
namespace {
struct Graph {
    Proto graph;
    int counter=0;
    std::string id() {return "v"+std::to_string(counter++);}
    std::string tensor(std::initializer_list<int> dims,const Floats& x) {
        auto name=id();initializer(graph,name,dims,1,x.data(),x.size()*sizeof(float));return name;
    }
    std::string integers(const std::vector<int64_t>& x) {
        auto name=id();initializer(graph,name,{static_cast<int>(x.size())},7,x.data(),x.size()*sizeof(int64_t));return name;
    }
    std::string op(const std::string& kind,const std::vector<std::string>& args,
                   std::initializer_list<std::pair<std::string,int>> attrs={},const std::string& dest="") {
        Proto n;const auto out=dest.empty()?id():dest;
        for(const auto& a:args)n.text(1,a);
        n.text(2,out);n.text(4,kind);
        for(const auto& kv:attrs) {
            Proto a;a.text(1,kv.first);a.number(3,kv.second);a.number(20,2);n.message(5,a);
        }
        graph.message(1,n);return out;
    }
    std::string reshape(const std::string& x,const std::vector<int64_t>& shape) {
        return op("Reshape",{x,integers(shape)});
    }
    std::string affine(const std::string& x,int in,int out,const Floats& w,const Floats& b) {
        Floats transposed(w.size());
        for(int i=0;i<in;i++)for(int j=0;j<out;j++)transposed[i*out+j]=w[j*in+i];
        return op("Add",{op("MatMul",{x,tensor({in,out},transposed)}),tensor({out},b)});
    }
    std::string recurrent(const std::string& x,const dn::LayerState& l,int steps) {
        int in=l.input,h=l.hidden,gates=l.type==2?4:l.type==1?3:1;
        Floats w(gates*h*in),r(gates*h*h),b(2*gates*h);
        // ONNX LSTM uses IOFC. Our GRU weights update the candidate by z;
        // ONNX weights the old state by z, so negate the update gate affine.
        const int iofc[]={1,3,0,2};
        for(int gate=0;gate<gates;gate++)for(int j=0;j<h;j++) {
            int row=gate*h+j,source=(l.type==2?iofc[gate]:gate)*h+j;
            if(l.type==1 && gate==2) {
                std::copy_n(l.candidate_x.data()+j*in,in,w.data()+row*in);
                std::copy_n(l.candidate_h.data()+j*h,h,r.data()+row*h);
                b[row]=l.bias_x[j];b[gates*h+row]=l.bias_h[j];
            } else {
                float sign=l.type==1 && gate==0?-1.f:1.f;
                const float* v=l.weights.data()+source*(in+h);
                for(int i=0;i<h;i++)r[row*h+i]=sign*v[i];
                for(int i=0;i<in;i++)w[row*in+i]=sign*v[h+i];
                b[row]=sign*l.bias[source];
            }
        }
        std::vector<std::string> args{x,tensor({1,gates*h,in},w),tensor({1,gates*h,h},r),tensor({1,2*gates*h},b)};
        auto seq=l.type==1?op("GRU",args,{{"hidden_size",h},{"linear_before_reset",1}}):
                           op(l.type==2?"LSTM":"RNN",args,{{"hidden_size",h}});
        return reshape(seq,{steps,1,h});
    }
};
}
std::vector<uint8_t> export_graph(const dn::ModelState& net,const std::string& key,const Head& head,
                                  const Floats& feature_scale,const Floats& temperature,const Floats& probe) {
    int h=net.output;
    Graph g;g.graph.text(2,"DNForecast sequence graph");
    g.graph.message(11,value("features",{1,net.sequence,net.input}));
    g.graph.message(12,value("forecast",{1,h,6}));
    auto input=g.reshape(g.op("Div",{"features",g.tensor({net.input},feature_scale)}),{net.sequence,1,net.input});
    std::vector<std::string> values;
    for(const auto& l:net.layers) {
        std::vector<std::string> parents;
        for(int p:l.node.parents)parents.push_back(p<0?input:values.at(p));
        auto x=parents.front();
        if(parents.size()>1)x=l.node.merge?g.op("Sum",parents):g.op("Concat",parents,{{"axis",2}});
        if(l.type<3)x=g.recurrent(x,l,net.sequence);
        if(l.type==3) {
            x=g.affine(x,l.input,l.hidden,l.weights,l.bias);
            const char* activations[]={"Identity","Tanh","Relu","Sigmoid"};
            if(l.node.activation)x=g.op(activations[l.node.activation],{x});
        }
        values.push_back(x);
    }
    auto last=g.reshape(g.op("Gather",{values.at(net.output_node),g.integers({net.sequence-1})},{{"axis",0}}),{1,net.output_input});
    auto raw=g.affine(last,net.output_input,h,net.weights,net.bias);
    auto point=g.op("Mul",{raw,g.tensor({h},head.scale)});
    auto offset=[&](const Floats& q){return g.op("Add",{point,g.tensor({h},q)});};
    auto low=offset(head.low),med=offset(head.median),high=offset(head.high);
    auto prob=g.op("Sigmoid",{g.op("Div",{med,g.tensor({h},temperature)})});
    std::vector<std::string> fields;
    for(const auto& v:{point,low,med,high,prob,g.tensor({h},head.variance)})fields.push_back(g.reshape(v,{1,h,1}));
    g.op("Concat",fields,{{"axis",2}},"forecast");
    Proto model,opset;model.number(1,7);model.text(2,"DNForecast");model.message(7,g.graph);
    opset.number(2,11);model.message(8,opset);
    std::ostringstream expected;expected.imbue(std::locale::classic());expected<<std::setprecision(9);
    for(float f:probe)expected<<f<<' ';
    metadata(model,"dn3.contract",key);metadata(model,"dn3.zero_probe",expected.str());
    metadata(model,"dn3.kind","sequence-graph");return model.bytes;
}
}
