#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "forecast_model.h"
#include <cmath>
#include <filesystem>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace forecast {
namespace {
struct LegacyDll {
    HMODULE module=nullptr;
    int handle=0;
    void (*free_handle)(int)=nullptr;
    ~LegacyDll() {
        if (handle && free_handle) free_handle(handle);
        if (module) FreeLibrary(module);
    }
    template<class Fn> Fn symbol(const char* name) {
        auto fn=GetProcAddress(module,name);
        if (!fn) throw std::runtime_error(std::string("Missing legacy export ")+name);
        return reinterpret_cast<Fn>(fn);
    }
};

class Legacy final:public Model {
    Shape s;
    LegacyDll dll;
    std::vector<double> mean,scale;
    double divisor=0;
    using Predict=int(*)(int,const double*,int,int,int,double*);
    Predict predict_batch=nullptr;
    std::unique_ptr<Model> reference;
public:
    Legacy(Shape shape,const std::string& contract,const std::vector<uint8_t>& bytes):
        s(shape),mean(s.markets),scale(s.markets),reference(load_model(s,DN3_REFERENCE,contract,{})) {
        std::istringstream stream(std::string(bytes.begin(),bytes.end()));
        stream.imbue(std::locale::classic());
        std::string magic,description;
        std::getline(stream,magic);std::getline(stream,description);
        if (magic!="DN3V2" || description!=contract || !(stream>>divisor) || !std::isfinite(divisor) || divisor<=0)
            throw std::runtime_error("Invalid checkpoint header or market contract");
        for (double& v:mean) if (!(stream>>v) || !std::isfinite(v)) throw std::runtime_error("Invalid mean");
        for (double& v:scale) if (!(stream>>v) || !std::isfinite(v) || v<=0) throw std::runtime_error("Invalid scale");
        if (stream.get()!='\n') throw std::runtime_error("Invalid checkpoint separator");
        std::string checkpoint(std::istreambuf_iterator<char>(stream),{});
        if (checkpoint.empty() || checkpoint.find('\0')!=std::string::npos) throw std::runtime_error("Invalid checkpoint");
        HMODULE self=nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                reinterpret_cast<LPCWSTR>(&DN3_GetAbiVersion),&self)) throw std::runtime_error("Unknown module path");
        std::wstring path(32768,L'\0');
        const auto n=GetModuleFileNameW(self,path.data(),static_cast<DWORD>(path.size()));
        if (!n || n>=path.size()) throw std::runtime_error("Invalid module path");
        path.resize(n);
        auto sibling=std::filesystem::path(path).parent_path()/L"MQL5GPULibrary_LSTM.dll";
        dll.module=LoadLibraryExW(sibling.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!dll.module) throw std::runtime_error("Legacy DLL or CUDA dependencies unavailable");
        auto version=dll.symbol<int(*)()>("DN_GetVersion");
        if (version()!=20300) throw std::runtime_error("Legacy ABI 2.3 required");
        dll.free_handle=dll.symbol<void(*)(int)>("DN_Free");
        predict_batch=dll.symbol<Predict>("DN_PredictBatch");
        dll.handle=dll.symbol<int(*)()>("DN_Create")();
        if (dll.handle<=0 || !dll.symbol<int(*)(int,const char*)>("DN_LoadState")(dll.handle,checkpoint.c_str()) ||
            dll.symbol<int(*)(int)>("DN_GetInputDim")(dll.handle)!=s.markets ||
            dll.symbol<int(*)(int)>("DN_GetOutputDim")(dll.handle)!=s.horizons ||
            dll.symbol<int(*)(int)>("DN_GetSequenceLength")(dll.handle)!=s.steps) throw std::runtime_error("Invalid legacy checkpoint shape or weights");
    }
    std::vector<double> predict(const std::vector<float>& input) override {
        std::vector<double> x(s.steps*s.markets),y(s.horizons);
        for (size_t i=0;i<x.size();++i) {
            if (!input[i*3+1]) throw std::runtime_error("Legacy model requires complete driver history");
            x[i]=(input[3*i]-mean[i%s.markets])/scale[i%s.markets];
        }
        if (!predict_batch(dll.handle,x.data(),1,static_cast<int>(x.size()),0,y.data()))
            throw std::runtime_error("Legacy GPU prediction failed");
        auto out=reference->predict(input);
        for (int h=0;h<s.horizons;++h) out[h*DN3_FIELDS]=y[h]/divisor;
        return out;
    }
};
}
std::unique_ptr<Model> load_legacy(Shape s,const std::string& contract,const std::vector<uint8_t>& bytes) {
    return std::make_unique<Legacy>(s,contract,bytes);
}
}
#endif
