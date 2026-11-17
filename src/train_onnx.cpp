#include "train_network.h"
#include <cstring>
#include <iomanip>
#include <locale>
#include <sstream>

#include "onnx_writer.h"
namespace dnfit {
using namespace dnonnx;
std::vector<uint8_t> export_onnx(const Net& net,const std::string& contract) {
    const int h=net.horizons, inputs=net.inputs();
    Proto graph;
    graph.text(2,"DNForecast Time Delay Network");
    graph.message(11,value("features",{1,net.steps,net.markets*3}));
    graph.message(12,value("forecast",{1,h,6}));
    auto tensor=[&](const char* name,std::initializer_list<int> dims,const Vec& v) {
        initializer(graph,name,dims,1,v.data(),v.size()*sizeof(float));
    };
    initializer(graph,"times",{static_cast<int>(net.taps.size())},7,net.taps.data(),net.taps.size()*sizeof(int64_t));
    tensor("mean",{inputs},net.center); tensor("scale",{inputs},net.scale);
    tensor("w1",{inputs,hidden},net.w1); tensor("b1",{hidden},net.b1);
    tensor("w2",{hidden,2*h},net.w2); tensor("b2",{2*h},net.b2);
    tensor("target_scale",{h},net.target_scale);
    tensor("low",{h},net.low); tensor("med",{h},net.median); tensor("high",{h},net.high);
    tensor("variance",{1,h},net.variance);
    auto node=[&](const char* op,std::initializer_list<const char*> in,std::initializer_list<const char*> out,
                  int axis=-1,bool axes=false) {
        Proto p;
        for (auto s:in) p.text(1,s);
        for (auto s:out) p.text(2,s);
        p.text(4,op);
        if (axis>=0) {
            Proto a; a.text(1,axes?"axes":"axis");
            a.number(axes?8:3,axis); a.number(20,axes?7:2); p.message(5,a);
        }
        graph.message(1,p);
    };
    node("Gather",{"features","times"},{"tapped"},1); node("Flatten",{"tapped"},{"flat"},1);
    node("Sub",{"flat","mean"},{"shifted"}); node("Div",{"shifted","scale"},{"normalized"});
    node("MatMul",{"normalized","w1"},{"linear1"}); node("Add",{"linear1","b1"},{"preactivation"});
    node("Tanh",{"preactivation"},{"hidden"}); node("MatMul",{"hidden","w2"},{"linear2"});
    node("Add",{"linear2","b2"},{"heads"}); node("Split",{"heads"},{"normalized_point","logit"},1);
    node("Mul",{"normalized_point","target_scale"},{"point"});
    node("Add",{"point","low"},{"q10"}); node("Add",{"point","med"},{"q50"}); node("Add",{"point","high"},{"q90"});
    node("Sigmoid",{"logit"},{"prob"});
    node("Unsqueeze",{"point"},{"p"},2,true); node("Unsqueeze",{"q10"},{"l"},2,true);
    node("Unsqueeze",{"q50"},{"m"},2,true); node("Unsqueeze",{"q90"},{"u"},2,true);
    node("Unsqueeze",{"prob"},{"d"},2,true); node("Unsqueeze",{"variance"},{"v"},2,true);
    node("Concat",{"p","l","m","u","d","v"},{"forecast"},2);
    Proto model, opset;
    model.number(1,7); model.text(2,"DNForecast"); model.message(7,graph);
    opset.number(2,11); model.message(8,opset);
    Vec input(inputs),state(hidden),head(2*h);
    for (int j=0;j<inputs;++j) input[j]=-net.center[j]/net.scale[j];
    net.forward(input.data(),state,head);
    std::ostringstream probe; probe.imbue(std::locale::classic()); probe<<std::setprecision(9);
    for (float f:net.predict(head)) probe<<f<<' ';
    metadata(model,"dn3.contract",contract); metadata(model,"dn3.zero_probe",probe.str());
    metadata(model,"dn3.kind","tdnn"); metadata(model,"dn3.status","research");
    return model.bytes;
}
}
