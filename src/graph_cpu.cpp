#include "graph_fit.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace graphfit {
namespace {
float sigmoid(float x){return 1.f/(1.f+std::exp(-x));}
float activate(int a,float x){return a==1?std::tanh(x):a==2?std::max(0.f,x):a==3?sigmoid(x):x;}
float derivative(int a,float y){return a==1?1-y*y:a==2?float(y>0):a==3?y*(1-y):1;}
float dot(const float* a,const float* b,int n){return std::inner_product(a,a+n,b,0.f);}
struct Cache {
    Floats input,state,output,cell,gates,mask,above;
};
using Field=Floats dn::LayerState::*;
const Field fields[]={&dn::LayerState::weights,&dn::LayerState::bias,&dn::LayerState::candidate_x,
    &dn::LayerState::candidate_h,&dn::LayerState::bias_x,&dn::LayerState::bias_h};
class Cpu final:public Engine {
    dn::ModelState net,best,grad,mom,vel;
    int steps,features,h,batch,rows,updates=0;
    double rate;
    const double *training,*labels;
    std::mt19937 rng{37};
    std::vector<int> order;
    std::vector<Cache> cache;
    Floats raw,zeros;
    static int gates(int type){return type==2?4:type==1?2:type==4?0:1;}
    void init(Floats& a,int count,int fan) {
        std::uniform_real_distribution<float> rand(-1.f/std::sqrt(float(fan)),1.f/std::sqrt(float(fan)));
        a.resize(count);for(auto& v:a)v=rand(rng);
    }
    void zero(dn::ModelState& m) {
        for(auto& l:m.layers)for(auto field:fields)std::fill((l.*field).begin(),(l.*field).end(),0);
        std::fill(m.weights.begin(),m.weights.end(),0);std::fill(m.bias.begin(),m.bias.end(),0);
    }
    void forward(const double* x,bool train,double* result) {
        raw.resize(steps*features);
        std::transform(x,x+raw.size(),raw.begin(),[](double value){return static_cast<float>(value);});
        for(size_t j=0;j<net.layers.size();j++) {
            const auto& l=net.layers[j];auto& c=cache[j];int I=l.input,H=l.hidden,G=gates(l.type);
            c.input.assign(steps*I,0);int offset=0;
            for(int parent:l.node.parents) {
                int w=parent<0?features:net.layers[parent].hidden;
                const auto& source=parent<0?raw:cache[parent].output;
                for(int t=0;t<steps;t++)for(int k=0;k<w;k++)c.input[t*I+offset+k]+=source[t*w+k];
                if(!l.node.merge)offset+=w;
            }
            c.state.resize(steps*H);c.output.resize(steps*H);c.mask.assign(steps*H,1);
            c.cell.assign(steps*H,0);c.gates.resize(steps*(G+2)*H);
            for(int t=0;t<steps;t++) {
                const float* in=c.input.data()+t*I;
                const float* prev=t?c.state.data()+(t-1)*H:zeros.data();
                float* g=c.gates.data()+t*(G+2)*H;
                for(int q=0;q<G*H;q++) {
                    const float* w=l.weights.data()+q*(I+(l.type<3?H:0));
                    g[q]=l.bias[q]+(l.type<3?dot(prev,w,H):0)+dot(in,w+(l.type<3?H:0),I);
                }
                for(int k=0;k<H;k++) {
                    int pos=t*H+k;float y=0;
                    if(l.type==4)y=in[k];
                    else if(l.type==3 || l.type==0)y=activate(l.type==0?1:l.node.activation,g[k]);
                    else if(l.type==2) {
                        g[k]=sigmoid(g[k]);g[H+k]=sigmoid(g[H+k]);g[2*H+k]=std::tanh(g[2*H+k]);g[3*H+k]=sigmoid(g[3*H+k]);
                        c.cell[pos]=g[k]*(t?c.cell[pos-H]:0)+g[H+k]*g[2*H+k];
                        y=g[3*H+k]*std::tanh(c.cell[pos]);
                    } else {
                        float z=g[k]=sigmoid(g[k]),r=g[H+k]=sigmoid(g[H+k]);
                        float nh=g[3*H+k]=dot(prev,l.candidate_h.data()+k*H,H)+l.bias_h[k];
                        float n=g[2*H+k]=std::tanh(dot(in,l.candidate_x.data()+k*I,I)+l.bias_x[k]+r*nh);
                        y=(1-z)*prev[k]+z*n;
                    }
                    if(train && l.dropout>0)c.mask[pos]=std::bernoulli_distribution(1-l.dropout)(rng)?1/(1-l.dropout):0;
                    c.state[pos]=y;c.output[pos]=y*c.mask[pos];
                }
            }
        }
        int width=net.output_input;
        const float* last=cache.back().output.data()+(steps-1)*width;
        for(int k=0;k<h;k++)result[k]=net.bias[k]+dot(last,net.weights.data()+k*width,width);
    }
    // Backpropagate through one sample, accumulating into the current minibatch.
    void backward(const double* prediction,const double* target) {
        for(size_t j=0;j<cache.size();j++)cache[j].above.assign(cache[j].state.size(),0);
        int width=net.output_input;auto& last=cache.back();
        for(int k=0;k<h;k++) {
            float d=static_cast<float>(2*(prediction[k]-target[k])/h);grad.bias[k]+=d;
            for(int a=0;a<width;a++) {
                grad.weights[k*width+a]+=d*last.output[(steps-1)*width+a];
                last.above[(steps-1)*width+a]+=d*net.weights[k*width+a];
            }
        }
        for(int j=static_cast<int>(net.layers.size())-1;j>=0;j--) {
            const auto& l=net.layers[j];auto& p=grad.layers[j];auto& c=cache[j];
            int I=l.input,H=l.hidden,G=gates(l.type);
            Floats dh(H),dc(H),previous(H),dx(I),dg(G*H);
            for(int t=steps-1;t>=0;t--) {
                const float* prev=t?c.state.data()+(t-1)*H:zeros.data();
                const float* x=c.input.data()+t*I;const float* g=c.gates.data()+t*(G+2)*H;
                std::fill(previous.begin(),previous.end(),0);std::fill(dx.begin(),dx.end(),0);
                for(int k=0;k<H;k++) {
                    int pos=t*H+k;float d=dh[k]+c.above[pos]*c.mask[pos];
                    if(l.type==4){dx[k]=d;continue;}
                    if(l.type==0 || l.type==3)dg[k]=d*derivative(l.type==0?1:l.node.activation,c.state[pos]);
                    else if(l.type==2) {
                        float f=g[k],i=g[H+k],v=g[2*H+k],o=g[3*H+k],tc=std::tanh(c.cell[pos]);
                        float cell=dc[k]+d*o*(1-tc*tc);
                        dg[k]=cell*(t?c.cell[pos-H]:0)*f*(1-f);
                        dg[H+k]=cell*v*i*(1-i);dg[2*H+k]=cell*i*(1-v*v);
                        dg[3*H+k]=d*tc*o*(1-o);dc[k]=cell*f;
                    } else {
                        float z=g[k],r=g[H+k],n=g[2*H+k],dn=d*z*(1-n*n);
                        dg[k]=d*(n-prev[k])*z*(1-z);dg[H+k]=dn*g[3*H+k]*r*(1-r);
                        previous[k]+=d*(1-z);p.bias_x[k]+=dn;p.bias_h[k]+=dn*r;
                        for(int a=0;a<I;a++){p.candidate_x[k*I+a]+=dn*x[a];dx[a]+=dn*l.candidate_x[k*I+a];}
                        for(int a=0;a<H;a++){p.candidate_h[k*H+a]+=dn*r*prev[a];previous[a]+=dn*r*l.candidate_h[k*H+a];}
                    }
                }
                for(int q=0;q<G*H;q++) {
                    int r=l.type<3?H:0;int base=q*(I+r);float d=dg[q];p.bias[q]+=d;
                    for(int a=0;a<r;a++){p.weights[base+a]+=d*prev[a];previous[a]+=d*l.weights[base+a];}
                    for(int a=0;a<I;a++){p.weights[base+r+a]+=d*x[a];dx[a]+=d*l.weights[base+r+a];}
                }
                dh.swap(previous);int offset=0;
                for(int parent:l.node.parents) {
                    int w=parent<0?features:net.layers[parent].hidden;
                    if(parent>=0)for(int a=0;a<w;a++)cache[parent].above[t*w+a]+=dx[offset+a];
                    if(!l.node.merge)offset+=w;
                }
            }
        }
    }
    void update(int count) {
        ++updates;double alpha=rate*std::sqrt(1-std::pow(.999,updates))/(1-std::pow(.9,updates));
        auto step=[&](Floats& w,const Floats& g,Floats& m,Floats& v){
            for(size_t i=0;i<w.size();i++){
                float d=std::clamp(g[i]/count,-5.f,5.f);m[i]=.9f*m[i]+.1f*d;v[i]=.999f*v[i]+.001f*d*d;
                w[i]-=float(alpha*m[i]/(std::sqrt(v[i])+1e-8));
            }
        };
        for(size_t i=0;i<net.layers.size();i++)for(auto f:fields)
            step(net.layers[i].*f,grad.layers[i].*f,mom.layers[i].*f,vel.layers[i].*f);
        step(net.weights,grad.weights,mom.weights,vel.weights);step(net.bias,grad.bias,mom.bias,vel.bias);
    }
public:
    Cpu(const Config& c,int s,int f,int out,const double* x,const double* y,int n):
        steps(s),features(f),h(out),batch(c.batch),rows(n),rate(c.rate),training(x),labels(y),order(n) {
        net.graph=true;net.input=f;net.sequence=s;net.output=h;std::vector<int> widths;
        for(const auto& node:c.layers) {
            dn::LayerState l;l.type=node.type<3?2-node.type:node.type;l.node=node.node;l.dropout=node.dropout;
            l.input=dn::merged_width(l.node,f,widths);l.hidden=node.width;
            int I=l.input,H=l.hidden,g=gates(l.type)*H,fan=I+(l.type<3?H:0);
            init(l.weights,fan*g,fan);l.bias.assign(g,0);
            if(l.type==1){init(l.candidate_x,H*I,I);init(l.candidate_h,H*H,H);l.bias_x.resize(H);l.bias_h.resize(H);}
            widths.push_back(H);net.layers.push_back(std::move(l));
        }
        net.output_node=int(widths.size())-1;net.output_input=widths.back();
        init(net.weights,h*widths.back(),widths.back());net.bias.resize(h);
        zeros.resize(*std::max_element(widths.begin(),widths.end()));cache.resize(widths.size());
        grad=net;zero(grad);mom=vel=grad;std::iota(order.begin(),order.end(),0);
    }
    void epoch() override {
        std::shuffle(order.begin(),order.end(),rng);std::vector<double> out(h);
        for(int begin=0;begin<rows;begin+=batch) {
            zero(grad);int end=std::min(rows,begin+batch);
            for(int i=begin;i<end;i++) {
                int n=order[i];forward(training+size_t(n)*steps*features,true,out.data());backward(out.data(),labels+n*h);
            }
            update(end-begin);
        }
    }
    void predict(const double* x,int n,double* y) override {
        for(int i=0;i<n;i++)forward(x+size_t(i)*steps*features,false,y+i*h);
    }
    dn::ModelState weights() override{return net;}
    void save_best() override{best=net;}
    void restore_best() override{net=best;}
};
}
std::unique_ptr<Engine> cpu(const Config& c,int s,int f,int h,const double* x,const double* y,int n) {
    return std::make_unique<Cpu>(c,s,f,h,x,y,n);
}
}
