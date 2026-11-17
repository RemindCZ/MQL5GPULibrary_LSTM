#include "graph_fit.h"
#include "DNForecast.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <thread>

namespace graphfit {
Config read(const std::filesystem::path& path,int features,int horizons) {
    std::ifstream file(path);std::string line,backend,junk;
    Config c;
    if(!std::getline(file,line))throw std::invalid_argument("Missing network configuration");
    std::istringstream first(line);first.imbue(std::locale::classic());
    if(!(first>>backend>>c.batch>>c.rate) || (first>>junk) || (backend!="cpu" && backend!="cuda") ||
       c.batch<1 || c.batch>4096 || !(c.rate>0 && c.rate<=1))
        throw std::invalid_argument("Invalid backend, learning rate or minibatch");
    c.cuda=backend=="cuda";std::vector<int> widths;std::vector<dn::NodeSpec> specs;
    const std::vector<std::string> types={"LSTM","GRU","RNN","Dense","Identity"},acts={"linear","tanh","relu","sigmoid"};
    int64_t parameters=0;
    while(std::getline(file,line)) {
        if(line.find_first_not_of(" \t\r")==std::string::npos)continue;
        std::istringstream row(line);row.imbue(std::locale::classic());std::string type,act,merge;
        Layer l{};
        if(!(row>>type>>l.width>>act>>l.dropout>>merge) || c.layers.size()>=1024)
            throw std::invalid_argument("Invalid layer or maximum graph depth exceeded");
        l.type=int(std::find(types.begin(),types.end(),type)-types.begin());
        l.node.activation=int(std::find(acts.begin(),acts.end(),act)-acts.begin());l.node.merge=merge=="sum";
        if(l.type>=5 || l.node.activation>=4 || (l.type!=3 && l.node.activation) || (merge!="concat" && merge!="sum"))
            throw std::invalid_argument("Unknown layer, merge or activation");
        while(row>>junk) {
            size_t consumed=0;int parent=std::stoi(junk,&consumed);
            if(consumed!=junk.size())throw std::invalid_argument("Invalid parent index");l.node.parents.push_back(parent);
        }
        int in=dn::merged_width(l.node,features,widths);if(l.type==4 && l.width==0)l.width=in;
        if(!dn::layer(in,l.width,l.dropout) || (l.type==4 && l.width!=in))throw std::invalid_argument("Invalid layer shape or dropout");
        int64_t fan=int64_t(in)+(l.type<3?l.width:0);
        int gates=l.type==0?4:l.type==1?3:l.type==4?0:1;
        parameters+=(fan+1)*gates*l.width+(l.type==1?l.width:0);
        if(parameters>16777216)throw std::invalid_argument("Network exceeds the 64 MiB MT5 model limit");
        c.layers.push_back(l);widths.push_back(l.width);specs.push_back(l.node);
    }
    if(!dn::all_reach_output(specs,int(specs.size())-1))throw std::invalid_argument("Every layer must reach the final layer");
    if(parameters+(int64_t(widths.back())+1)*horizons>16777216)throw std::invalid_argument("Output exceeds the MT5 model limit");
    return c;
}
Floats Head::forecast(const double* raw) const {
    Floats result;
    for(size_t j=0;j<scale.size();j++) {
        float p=float(raw[j])*scale[j],m=p+median[j];
        result.insert(result.end(),{p,p+low[j],m,p+high[j],float(1/(1+std::exp(-double(m)/temperature[j]))),variance[j]});
    }
    return result;
}
static void verify(const std::vector<uint8_t>& model,const std::string& key,int steps,int markets,int h,
                   const double* window,const Floats& expected) {
    if(model.size()>64*1024*1024)throw std::runtime_error("ONNX exceeds 64 MiB MT5 limit");
    if(!(DN3_GetCapabilities()&2))throw std::runtime_error("Graph export verification requires ONNX Runtime CPU");
    struct Session {int64_t id;~Session(){if(id>0)DN3_CloseSession(id);}};
    Session s{DN3_OpenSession(steps,markets,h,1,0,DN3_ONNX_CPU,key.data(),int(key.size()),model.data(),int(model.size()))};
    auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(60);
    auto wait=[&](){if(std::chrono::steady_clock::now()>deadline)throw std::runtime_error("ONNX verification timed out");
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));};
    while(DN3_StatusOf(s.id)==DN3_PENDING)wait();
    if(DN3_StatusOf(s.id)!=DN3_OK){char err[512]{};DN3_Error(s.id,0,err,sizeof(err));throw std::runtime_error(std::string("ONNX: ")+err);}
    auto job=DN3_Submit(s.id,1,window,(steps+1)*markets);
    std::vector<double> y(6*h);int need=0,rc;int64_t origin=0;
    while((rc=DN3_TryRead(s.id,job,y.data(),int(y.size()),&need,&origin))==DN3_PENDING)wait();
    if(rc || need!=6*h || origin!=1)throw std::runtime_error("ONNX forecast failed verification");
    for(size_t i=0;i<y.size();i++)
        if(!std::isfinite(expected[i]) || std::abs(y[i]-expected[i])>3e-7+3e-4*std::abs(expected[i]))
            throw std::runtime_error("Exported graph differs from training engine");
}
std::vector<uint8_t> fit(const Config& config,const std::string& key,int steps,int markets,int h,int epochs,
                         const std::vector<int64_t>& times,const std::vector<double>& closes,
                         std::vector<double>& metrics,std::vector<double>& trace) {
    int rows=int(times.size()),f=3*markets,stride=steps*f;
    std::vector<int> widths;
    for(const auto& l:config.layers) {
        int in=dn::merged_width(l.node,f,widths);
        if(int64_t(steps)*std::max(in,l.width)*6>std::numeric_limits<int>::max() ||
           (config.cuda && int64_t(config.batch)*steps*std::max(in,l.width)*4>std::numeric_limits<int>::max()))
            throw std::invalid_argument("Sequence, minibatch and layer width exceed backend indexing limits");
        widths.push_back(l.width);
    }
    int a=rows*55/100,b=rows*70/100,c=rows*85/100,end=rows-h;
    if(steps<16 || h>steps/2 || a-h-steps<64 || b-h-a<32 || c-h-b<32 || end-c<32)
        throw std::invalid_argument("Insufficient history for purged fit, selection, calibration and test blocks");
    for(int i=0;i<rows;i++)if(times[i]<=0 || (i && times[i]<=times[i-1]))throw std::invalid_argument("Timestamps must increase");
    if(int64_t(end-steps)*stride>134217728)throw std::invalid_argument("Training windows exceed the 1 GiB host buffer; reduce bars or lookback");
    Head head;head.scale.resize(h);
    Floats scales(f,1),encoded(stride);std::vector<double> sq(f),ysq(h);
    std::vector<double> data(size_t(end-steps)*stride),labels(size_t(end-steps)*h);
    auto input=[&](int i){return data.data()+size_t(i-steps)*stride;};
    auto target=[&](int i){return labels.data()+size_t(i-steps)*h;};
    auto price=[&](int i){return closes[i*markets];};
    for(int i=0;i<rows;i++)for(int m=0;m<markets;m++){
        double p=closes[i*markets+m];
        if(!std::isfinite(p)||p<0||(m==0&&p==0))throw std::invalid_argument("Invalid close price");
    }
    for(int i=steps;i<end;i++) {
        if(DN3_Encode(steps,markets,&closes[(i-steps)*markets],(steps+1)*markets,encoded.data(),stride))
            throw std::invalid_argument("Invalid price window");
        std::copy(encoded.begin(),encoded.end(),input(i));
        for(int j=0;j<h;j++){
            double y=target(i)[j]=std::log(price(i+j+1))-std::log(price(i));
            if(i<a-h)ysq[j]+=y*y;
        }
        if(i<a-h)for(int k=0;k<f;k+=3){double r=encoded[stride-f+k];sq[k]+=r*r;}
    }
    int count=a-h-steps;
    for(int k=0;k<f;k+=3)scales[k]=float(std::sqrt(std::max(1e-12,sq[k]/count)));
    for(int j=0;j<h;j++)head.scale[j]=float(std::sqrt(std::max(1e-12,ysq[j]/count)));
    for(size_t i=0;i<data.size();i++)data[i]=float(data[i])/scales[i%f];
    for(size_t i=0;i<labels.size();i++)labels[i]/=head.scale[i%h];
    auto engine=(config.cuda?gpu:cpu)(config,steps,f,h,data.data(),labels.data(),count);
    auto predictions=[&](int first,int stop){
        std::vector<double> y(size_t(stop-first)*h);
        for(int i=first;i<stop;i+=config.batch)engine->predict(input(i),std::min(config.batch,stop-i),y.data()+size_t(i-first)*h);
        for(double v:y)if(!std::isfinite(v))throw std::runtime_error("Non-finite network output");
        return y;
    };
    double best=std::numeric_limits<double>::infinity();
    for(int epoch=1;epoch<=epochs;epoch++) {
        std::cerr<<"epoch "<<epoch<<"/"<<epochs<<std::endl;
        engine->epoch();auto selected=predictions(a,b-h);double loss=0;
        for(size_t j=0;j<selected.size();j++){double e=selected[j]-target(a)[j];loss+=e*e;}
        if(loss<best){engine->save_best();best=loss;}
    }
    if(!std::isfinite(best))throw std::runtime_error("Training diverged");
    engine->restore_best();
    auto calibration=predictions(b,c-h);
    for(int j=0;j<h;j++) {
        Floats residual;double squares=0,rv=0;
        for(int i=b;i<c-h;i++) {
            float e=float((target(i)[j]-calibration[(i-b)*h+j])*head.scale[j]);
            residual.push_back(e);squares+=double(e)*e;
            for(int k=0;k<=j;k++){double r=std::log(price(i+k+1))-std::log(price(i+k));rv+=r*r;}
        }
        std::sort(residual.begin(),residual.end());
        auto q=[&](double level){double pos=level*(residual.size()-1);size_t k=size_t(pos);
            return float(residual[k]+(pos-k)*(residual[std::min(k+1,residual.size()-1)]-residual[k]));};
        head.low.push_back(q(.1));head.median.push_back(q(.5));head.high.push_back(q(.9));
        head.variance.push_back(float(std::max(1e-12,rv/residual.size())));
        head.temperature.push_back(float(std::max(1e-6,.5513289*std::sqrt(squares/residual.size()))));
    }
    auto test=predictions(c,end);metrics.assign(h*3,0);trace.clear();
    for(int i=c;i<end;i++) {
        auto out=head.forecast(test.data()+(i-c)*h);
        for(int j=0;j<h;j++) {
            double y=target(i)[j]*head.scale[j],e=out[j*6]-y;
            metrics[j*3]+=e*e;metrics[j*3+1]+=y*y;metrics[j*3+2]+=out[j*6+1]<=y && y<=out[j*6+3];
        }
        for(int k:{0,1,3,4})trace.push_back(out[(h-1)*6+k]);
    }
    for(double& m:metrics)m/=end-c;
    std::vector<double> zero(stride),raw(h);engine->predict(zero.data(),1,raw.data());
    auto model=export_graph(engine->weights(),key,head,scales,head.temperature,head.forecast(raw.data()));
    verify(model,key,steps,markets,h,&closes[(c-steps)*markets],head.forecast(test.data()));
    return model;
}
}
