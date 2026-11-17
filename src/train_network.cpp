#include "train_network.h"
#include "DNForecast.h"
#include "DNForecastTrain.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>

namespace dnfit {
float sigmoid(float x) { return 1.0f/(1.0f+std::exp(-std::clamp(x,-75.f,75.f))); }
void Net::forward(const float* x,Vec& state,Vec& head) const {
    state=b1;
    for (int j=0;j<inputs();++j)
        for (int k=0;k<hidden;++k) state[k]+=x[j]*w1[j*hidden+k];
    for (float& v:state) v=std::tanh(v);
    head=b2;
    for (int k=0;k<hidden;++k)
        for (int q=0;q<2*horizons;++q) head[q]+=state[k]*w2[k*2*horizons+q];
}
Vec Net::predict(const Vec& head) const {
    Vec values;
    for (int h=0;h<horizons;++h) {
        const float p=head[h]*target_scale[h];
        values.insert(values.end(),{p,p+low[h],p+median[h],p+high[h],sigmoid(head[h+horizons]),variance[h]});
    }
    return values;
}
float quantile(const Vec& sorted,float fraction) {
    float pos=(sorted.size()-1)*fraction;
    size_t i=static_cast<size_t>(pos);
    return sorted[i]+(pos-i)*(sorted[std::min(i+1,sorted.size()-1)]-sorted[i]);
}
Net fit(int steps,int markets,int h,int epochs,const double* prices,int rows,double* report) {
    Net n{}; n.steps=steps; n.markets=markets; n.horizons=h;
    for (int lag:{0,1,2,3,7,15,steps/2,steps-1}) {
        int tap=steps-1-lag;
        if (std::find(n.taps.begin(),n.taps.end(),tap)==n.taps.end()) n.taps.push_back(tap);
    }
    const int dim=static_cast<int>(n.taps.size())*3*markets;
    const int split1=rows*55/100, split2=rows*70/100, split3=rows*85/100;
    const int end=rows-h, train_end=split1-h;
    if (train_end-steps<64 || split2-h-split1<32 || split3-h-split2<32 || end-split3<32)
        throw std::invalid_argument("Insufficient closed bars for separated fit, selection, calibration and test periods");
    if (int64_t(rows)*(dim+h)>16000000) throw std::invalid_argument("Reduce the number of bars or symbols");
    for (int i=0;i<rows*markets;++i)
        if (!std::isfinite(prices[i]) || prices[i]<0 || (i%markets==0 && prices[i]<=0))
            throw std::invalid_argument("Invalid close price");
    Vec x(static_cast<size_t>(end)*dim), y(static_cast<size_t>(end)*h), encoded(steps*markets*3);
    n.center.assign(dim,0); n.scale.resize(dim); n.target_scale.resize(h);
    std::vector<double> sums(dim),sq(dim),ysq(h);
    for (int i=steps;i<end;++i) {
        if (DN3_Encode(steps,markets,prices+(i-steps)*markets,(steps+1)*markets,encoded.data(),static_cast<int>(encoded.size())))
            throw std::invalid_argument("Invalid training window");
        float* row=x.data()+i*dim;
        for (auto t:n.taps) row=std::copy_n(encoded.data()+t*3*markets,3*markets,row);
        for (int q=0;q<h;++q) {
            float val=static_cast<float>(std::log(prices[(i+q+1)*markets])-std::log(prices[i*markets]));
            y[i*h+q]=val;
            if (i<train_end) ysq[q]+=double(val)*val;
        }
        if (i<train_end) for (int j=0;j<dim;++j) {
            double v=x[i*dim+j]; sums[j]+=v; sq[j]+=v*v;
        }
    }
    const int count=train_end-steps;
    for (int j=0;j<dim;++j) {
        double mean=sums[j]/count;
        n.center[j]=static_cast<float>(mean);
        n.scale[j]=static_cast<float>(std::sqrt(std::max(1.e-8,sq[j]/count-mean*mean)));
    }
    for (int i=steps;i<end;++i) for (int j=0;j<dim;++j)
        x[i*dim+j]=(x[i*dim+j]-n.center[j])/n.scale[j];
    for (int q=0;q<h;++q) n.target_scale[q]=static_cast<float>(std::sqrt(std::max(1.e-12,ysq[q]/count)));
    std::mt19937 rng(37);
    std::uniform_real_distribution<float> initial(-1.f/std::sqrt(float(dim)),1.f/std::sqrt(float(dim)));
    n.w1.resize(dim*hidden); for (float& w:n.w1) w=initial(rng);
    n.b1.assign(hidden,0); n.w2.assign(hidden*2*h,0); n.b2.assign(2*h,0);
    Vec state(hidden),head(2*h),delta(2*h),back(hidden);
    std::vector<int> indices(count); std::iota(indices.begin(),indices.end(),steps);
    Net best; double best_loss=std::numeric_limits<double>::infinity();
    for (int epoch=0;epoch<epochs;++epoch) {
        std::shuffle(indices.begin(),indices.end(),rng);
        float rate=.008f/std::sqrt(float(1+epoch));
        for (int i:indices) {
            const float* row=x.data()+i*dim;
            n.forward(row,state,head);
            for (int q=0;q<h;++q) {
                delta[q]=std::clamp(head[q]-y[i*h+q]/n.target_scale[q],-3.f,3.f)/h;
                delta[h+q]=(sigmoid(head[h+q])-(y[i*h+q]>0 ? 1.f : 0.f))/h;
            }
            for (int k=0;k<hidden;++k) {
                float grad=0;
                for (int q=0;q<2*h;++q) {
                    float& w=n.w2[k*2*h+q]; grad+=w*delta[q]; w-=rate*state[k]*delta[q];
                }
                back[k]=grad*(1.f-state[k]*state[k]); n.b1[k]-=rate*back[k];
            }
            for (int j=0;j<dim;++j) for (int k=0;k<hidden;++k)
                n.w1[j*hidden+k]-=rate*std::clamp(row[j]*back[k],-3.f,3.f);
            for (int q=0;q<2*h;++q) n.b2[q]-=rate*delta[q];
        }
        double loss=0;
        for (int i=split1;i<split2-h;++i) {
            n.forward(x.data()+i*dim,state,head);
            for (int q=0;q<h;++q) { double e=head[q]-y[i*h+q]/n.target_scale[q]; loss+=e*e; }
        }
        if (loss<best_loss) { best=n; best_loss=loss; }
    }
    if (!std::isfinite(best_loss)) throw std::runtime_error("Network did not converge");
    n=std::move(best);
    std::vector<Vec> residual(h); std::vector<double> realized(h);
    for (int i=split2;i<split3-h;++i) {
        n.forward(x.data()+i*dim,state,head);
        double rv=0;
        for (int q=0;q<h;++q) {
            residual[q].push_back(y[i*h+q]-head[q]*n.target_scale[q]);
            double r=std::log(prices[(i+q+1)*markets])-std::log(prices[(i+q)*markets]);
            rv+=r*r; realized[q]+=rv;
        }
    }
    for (int q=0;q<h;++q) {
        auto& v=residual[q]; std::sort(v.begin(),v.end());
        n.low.push_back(quantile(v,.1f)); n.median.push_back(quantile(v,.5f)); n.high.push_back(quantile(v,.9f));
        n.variance.push_back(static_cast<float>(std::max(1.e-12,realized[q]/v.size())));
    }
    std::fill_n(report,3*h,0.);
    for (int i=split3;i<end;++i) {
        n.forward(x.data()+i*dim,state,head); const auto output=n.predict(head);
        for (int q=0;q<h;++q) {
            const double truth=y[i*h+q], e=output[6*q]-truth;
            report[3*q]+=e*e; report[3*q+1]+=truth*truth;
            report[3*q+2]+=output[6*q+1]<=truth && truth<=output[6*q+3];
        }
    }
    for (int q=0;q<3*h;++q) report[q]/=end-split3;
    return n;
}
void verify_export(const Net& net,const std::string& key,const std::vector<uint8_t>& bytes,
                   const double* window,int64_t origin) {
    if (!(DN3_GetCapabilities() & 2)) return; // The portable training-only build can still write ONNX.
    const int count=(net.steps+1)*net.markets;
    struct Session { int64_t id; ~Session(){if(id>0) DN3_CloseSession(id);} };
    Session session{DN3_OpenSession(net.steps,net.markets,net.horizons,1,0,DN3_ONNX_CPU,key.data(),
        static_cast<int>(key.size()),bytes.data(),static_cast<int>(bytes.size()))};
    if(session.id<=0) throw std::runtime_error("Cannot open ONNX verification session");
    auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    while(DN3_StatusOf(session.id)==DN3_PENDING && std::chrono::steady_clock::now()<deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if(DN3_StatusOf(session.id)!=DN3_OK) {
        char reason[512]{}; DN3_Error(session.id,0,reason,sizeof(reason));
        throw std::runtime_error(std::string("ONNX model verification failed: ")+reason);
    }
    auto job=DN3_Submit(session.id,origin,window,count);
    if(job<=0) throw std::runtime_error("Cannot submit ONNX verification window");
    std::vector<double> output(net.horizons*DN3_FIELDS); int needed=0,status; int64_t returned=0;
    do {
        status=DN3_TryRead(session.id,job,output.data(),static_cast<int>(output.size()),&needed,&returned);
        if(status==DN3_PENDING) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while(status==DN3_PENDING && std::chrono::steady_clock::now()<deadline);
    DN3_Release(session.id,job);
    if(status!=0 || returned!=origin || needed!=static_cast<int>(output.size())) throw std::runtime_error("ONNX prediction verification failed");
    Vec encoded(net.steps*3*net.markets), x, state, head;
    if(DN3_Encode(net.steps,net.markets,window,count,encoded.data(),static_cast<int>(encoded.size()))) throw std::runtime_error("Invalid verification data");
    for(auto t:net.taps) for(int j=0;j<3*net.markets;++j) {
        auto i=x.size(); x.push_back((encoded[t*3*net.markets+j]-net.center[i])/net.scale[i]);
    }
    net.forward(x.data(),state,head); auto expected=net.predict(head);
    for(size_t i=0;i<output.size();++i)
        if(std::abs(output[i]-expected[i])>1.e-8+2.e-4*std::abs(expected[i])) throw std::runtime_error("Native and ONNX predictions differ");
}
}
int DNFit_Train(int steps,int markets,int h,int epochs,const char* contract,int contract_size,
                const int64_t* times,int rows,const double* closes,int close_count,uint8_t* model,
                int capacity,double* report,int report_count,char* error,int error_capacity) {
    if (!error || error_capacity<1) return -1;
    error[0]=0;
    try {
        if (steps<16 || steps>1024 || markets<1 || markets>16 || h<1 || h>64 || h>steps/2 ||
            epochs<1 || epochs>64 || rows<1 || rows>100000 || !closes || !times || close_count!=rows*markets ||
            !contract || contract_size<1 || contract_size>2048 || std::memchr(contract,0,contract_size) ||
            !model || capacity<1 || !report || report_count<3*h)
            throw std::invalid_argument("Invalid dimensions, contract or output buffers");
        const std::string key(contract,contract_size);
        const std::string prefix="dn3-v1|target-trading-bars|";
        const auto sep=key.find('|',prefix.size());
        if (key.compare(0,prefix.size(),prefix)!=0 || sep==std::string::npos || sep+1>=key.size() ||
            std::count(key.begin()+sep+1,key.end(),',')+1!=markets)
            throw std::invalid_argument("Invalid market/timeframe contract");
        for (int i=0;i<rows;++i) if (times[i]<=0 || (i && times[i]<=times[i-1]))
            throw std::invalid_argument("Closed bar timestamps must increase strictly");
        auto net=dnfit::fit(steps,markets,h,epochs,closes,rows,report);
        auto bytes=dnfit::export_onnx(net,key);
        if (bytes.size()>static_cast<size_t>(capacity)) throw std::length_error("Model buffer too small");
        dnfit::verify_export(net,key,bytes,closes+(rows-steps-1)*markets,times[rows-1]);
        std::copy(bytes.begin(),bytes.end(),model); return static_cast<int>(bytes.size());
    } catch (const std::exception& e) {
        size_t n=std::min(std::strlen(e.what()),static_cast<size_t>(error_capacity-1));
        std::memcpy(error,e.what(),n); error[n]=0; return -1;
    } catch (...) { return -1; }
}
