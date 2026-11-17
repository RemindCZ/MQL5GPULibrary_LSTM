#include "DNForecast.h"
#include "DNForecastTrain.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
static void require(bool ok,const char* reason) { if (!ok) throw std::runtime_error(reason); }
int main() {
    constexpr int rows=640, steps=32, markets=2, h=3;
    std::vector<int64_t> times(rows); std::vector<double> prices(rows*markets), stats(3*h);
    const std::string key="dn3-v1|target-trading-bars|3600|BTCUSD,ETHUSD";
    for (int i=0;i<rows;++i) {
        times[i]=1600000000LL+3600LL*i; prices[2*i]=32000*std::exp(.02*std::sin(i*.06)+.003*std::cos(i*.17));
        prices[2*i+1]=(i%37<4)?0:1900*std::exp(.04*std::sin((i+1)*.06));
    }
    std::vector<uint8_t> bytes(1024*1024); char error[256]{};
    auto fit=[&]() { return DNFit_Train(steps,markets,h,3,key.data(),static_cast<int>(key.size()),times.data(),rows,
        prices.data(),rows*markets,bytes.data(),static_cast<int>(bytes.size()),stats.data(),3*h,error,sizeof(error)); };
    int size=fit(); require(size>0,error);
    std::vector<uint8_t> original(bytes.begin(),bytes.begin()+size);
    for (int j=0;j<h;++j) require(std::isfinite(stats[3*j]) && stats[3*j+1]>0 && stats[3*j+2]>=0 && stats[3*j+2]<=1,"invalid holdout");
    if (DN3_GetCapabilities()&2) {
        const auto session=DN3_OpenSession(steps,markets,h,2,0,1,key.data(),static_cast<int>(key.size()),bytes.data(),size);
        require(session>0,"open"); auto until=std::chrono::steady_clock::now()+std::chrono::seconds(20);
        while (DN3_StatusOf(session)==1 && std::chrono::steady_clock::now()<until) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (DN3_StatusOf(session)!=0) {DN3_Error(session,0,error,sizeof(error)); throw std::runtime_error(error);}
        double mse[h]{};
        for (int i=rows*85/100;i<rows-h;++i) {
            auto job=DN3_Submit(session,times[i],&prices[(i-steps)*markets],(steps+1)*markets); require(job>0,"submit");
            double result[6*h]; int needed=0,code; int64_t origin=0;
            do {code=DN3_TryRead(session,job,result,6*h,&needed,&origin); if(code==1) std::this_thread::yield();}
            while(code==1 && std::chrono::steady_clock::now()<until);
            require(code==0 && needed==6*h && origin==times[i],"ONNX prediction");
            for(int j=0;j<h;++j) { double e=result[6*j]-std::log(prices[(i+j+1)*markets]/prices[i*markets]); mse[j]+=e*e; }
            DN3_Release(session,job);
        }
        DN3_CloseSession(session);
        for(int j=0;j<h;++j) require(std::abs(mse[j]/(rows-h-rows*85/100)-stats[3*j])<1.e-10,"ONNX/holdout mismatch");
    }
    prices[(rows-1)*markets]*=1.2;
    require(fit()==size && std::equal(original.begin(),original.end(),bytes.begin()),"test labels influenced fit");
    times[10]=times[9]; require(fit()<0 && error[0],"unordered data accepted");
    std::cout<<"Native fit, holdout isolation and production runtime checks passed\n";
}
