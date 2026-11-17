#include "DNForecastTrain.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
namespace fs=std::filesystem;
namespace {
std::vector<std::string> fields(std::string line) {
    if (!line.empty() && line.back()=='\r') line.pop_back();
    std::vector<std::string> result; size_t start=0;
    for (;;) {
        auto comma=line.find(',',start); result.push_back(line.substr(start,comma-start));
        if (comma==std::string::npos) return result;
        start=comma+1;
    }
}
long long integer(const std::string& text) {
    size_t end=0; auto v=std::stoll(text,&end);
    if (end!=text.size()) throw std::invalid_argument("Invalid integer");
    return v;
}
void save_new(const fs::path& dest,const uint8_t* data,int n) {
    fs::path work;
    for (int i=0;;++i) {
        work=dest; work+=".tmp-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+"-"+std::to_string(i);
        if (fs::create_directory(work)) break;
    }
    auto staged=work/"model";
    try {
        std::ofstream f(staged,std::ios::binary); f.exceptions(std::ios::failbit|std::ios::badbit);
        f.write(reinterpret_cast<const char*>(data),n); f.close();
#ifdef _WIN32
        if (!MoveFileExW(staged.c_str(),dest.c_str(),MOVEFILE_WRITE_THROUGH))
            throw std::runtime_error("Cannot publish model. Choose an unused output filename.");
#else
        fs::create_hard_link(staged,dest);
#endif
    } catch (...) { std::error_code ec; fs::remove(staged,ec); fs::remove(work,ec); throw; }
    fs::remove(staged); fs::remove(work);
}
}
#ifdef _WIN32
int wmain(int argc,wchar_t** argv) {
#else
int main(int argc,char** argv) {
#endif
    try {
        if (argc<4 || argc>7) {
            std::cerr<<"DNForecast-Train bars.csv new-model.onnx timeframe_seconds [steps=96] [horizons=6] [epochs=8]\n"
                       "CSV header: close_time,available_at,TARGET[,DRIVER...]; timestamps in seconds; blank auxiliary = missing\n";
            return 2;
        }
        fs::path dest(argv[2]);
        if (fs::exists(dest)) throw std::runtime_error("Model already exists");
        auto number=[&](int arg,int fallback,int maximum) {
            long long x=arg<argc?integer(fs::path(argv[arg]).string()):fallback;
            if (x<1 || x>maximum) throw std::invalid_argument("Invalid training parameter");
            return static_cast<int>(x);
        };
        int seconds=number(3,3600,31536000), steps=number(4,96,1024), h=number(5,6,64), epochs=number(6,8,64);
        std::ifstream f{fs::path(argv[1])}; std::string line;
        if (!std::getline(f,line)) throw std::runtime_error("Cannot read input CSV");
        if (line.compare(0,3,"\xef\xbb\xbf")==0) line.erase(0,3);
        auto names=fields(line); int markets=static_cast<int>(names.size())-2;
        if (markets<1 || markets>16 || names[0]!="close_time" || names[1]!="available_at")
            throw std::invalid_argument("Invalid CSV header");
        std::string contract="dn3-v1|target-trading-bars|"+std::to_string(seconds)+"|";
        std::set<std::string> unique;
        for (size_t j=2;j<names.size();++j) {
            if (names[j].empty() || names[j].find_first_of("|\"")!=std::string::npos || !unique.insert(names[j]).second)
                throw std::invalid_argument("Invalid or duplicate symbol");
            contract+=(j>2?",":"")+names[j];
        }
        std::vector<int64_t> times; std::vector<double> prices;
        while (std::getline(f,line)) {
            auto values=fields(line);
            if (values.size()!=names.size() || times.size()>=100000) throw std::invalid_argument("Invalid CSV row or row limit");
            auto t=integer(values[0]);
            if (integer(values[1])!=t) throw std::invalid_argument("Expected observations available at bar close");
            times.push_back(t);
            for (size_t j=2;j<values.size();++j) {
                size_t end=0; const auto& s=values[j];
                double v=s.empty()?0.:std::stod(s,&end);
                if (end!=s.size()) throw std::invalid_argument("Invalid close price");
                prices.push_back(v);
            }
        }
        if (f.bad()) throw std::runtime_error("CSV read error");
        std::vector<uint8_t> model(1024*1024); std::vector<double> metrics(h*3); char error[512]{};
        int size=DNFit_Train(steps,markets,h,epochs,contract.data(),static_cast<int>(contract.size()),times.data(),
            static_cast<int>(times.size()),prices.data(),static_cast<int>(prices.size()),model.data(),static_cast<int>(model.size()),
            metrics.data(),static_cast<int>(metrics.size()),error,sizeof(error));
        if (size<=0) throw std::runtime_error(error);
        save_new(dest,model.data(),size);
        std::cout<<"horizon,test_mse,last_price_mse,coverage80\n";
        for (int q=0;q<h;++q) std::cout<<q+1<<','<<metrics[3*q]<<','<<metrics[3*q+1]<<','<<metrics[3*q+2]<<'\n';
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
    return 0;
}
