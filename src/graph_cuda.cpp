#include "graph_fit.h"
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "MQL5GPULibrary.h"
#include <sstream>
namespace graphfit {
namespace {
struct Module {
    HMODULE h;
    explicit Module(HMODULE value):h(value){if(!h)throw std::runtime_error("CUDA engine or NVIDIA runtime unavailable");}
    ~Module(){FreeLibrary(h);}
    template<typename T>T get(const char* name) const {
        auto f=GetProcAddress(h,name);if(!f)throw std::runtime_error(std::string("Missing CUDA export ")+name);
        return reinterpret_cast<T>(f);
    }
};
std::filesystem::path library() {
    wchar_t path[32768];auto n=GetModuleFileNameW(nullptr,path,32768);
    if(!n || n>=32768)throw std::runtime_error("Unknown trainer path");
    auto dir=std::filesystem::path(path).parent_path();
    auto packed=dir/L"DNForecast-CUDA"/L"MQL5GPULibrary_LSTM.dll";
    return std::filesystem::is_regular_file(packed)?packed:dir/L"MQL5GPULibrary_LSTM.dll";
}
struct Api {
    Module dll{LoadLibraryExW(library().c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)};
#define ENTRY(name) decltype(&DN_##name) name=dll.get<decltype(&DN_##name)>("DN_" #name)
    ENTRY(Create);ENTRY(Free);ENTRY(GetVersion);ENTRY(GetError);ENTRY(BeginGraph);ENTRY(AddGraphNode);
    ENTRY(SetGraphOutput);ENTRY(SetMiniBatchSize);ENTRY(LoadBatch);ENTRY(Train);ENTRY(PredictBatch);
    ENTRY(SnapshotWeights);ENTRY(RestoreWeights);ENTRY(SaveState);ENTRY(GetState);
#undef ENTRY
    int h=0;
    Api() {
        if(GetVersion()!=20300)throw std::runtime_error("CUDA engine ABI 2.3 required");
        h=Create();check(h>0);
    }
    ~Api(){if(h)Free(h);}
    void check(bool ok) {
        if(ok)return;
        short msg[512]{};GetError(msg,512);
        std::wstring w(reinterpret_cast<wchar_t*>(msg));
        throw std::runtime_error("CUDA: "+std::string(w.begin(),w.end()));
    }
};
class Cuda final:public Engine {
    Api a;int sample;double rate;
public:
    Cuda(const Config& c,int steps,int features,int h,const double* x,const double* y,int rows):
        sample(steps*features),rate(c.rate) {
        a.check(a.BeginGraph(a.h,features,steps)!=0);
        int index=0;
        for(const auto& l:c.layers) {
            a.check(a.AddGraphNode(a.h,l.type,l.width,l.node.activation,l.dropout,l.node.parents.data(),
                                    static_cast<int>(l.node.parents.size()),l.node.merge)==index++);
        }
        a.check(a.SetGraphOutput(a.h,index-1,h)!=0);
        a.check(a.SetMiniBatchSize(a.h,c.batch)!=0);
        a.check(a.LoadBatch(a.h,x,y,rows,sample,h,0)!=0);
    }
    void epoch() override {a.check(a.Train(a.h,1,0,rate,0)!=0);}
    void predict(const double* x,int n,double* y) override {a.check(a.PredictBatch(a.h,x,n,sample,0,y)!=0);}
    void save_best() override {a.check(a.SnapshotWeights(a.h)!=0);}
    void restore_best() override {a.check(a.RestoreWeights(a.h)!=0);}
    dn::ModelState weights() override {
        int n=a.SaveState(a.h);a.check(n>1);
        std::vector<char> data(n);a.check(a.GetState(a.h,data.data(),n)!=0);
        std::istringstream input(data.data());return dn::read_checkpoint(input);
    }
};
}
std::string device() {
    Module driver(LoadLibraryExW(L"nvcuda.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32));
    using Init=int(WINAPI*)(unsigned);using Get=int(WINAPI*)(int*,int);using Name=int(WINAPI*)(char*,int,int);
    int ordinal;char name[256]{};
    if(driver.get<Init>("cuInit")(0) || driver.get<Get>("cuDeviceGet")(&ordinal,0) ||
       driver.get<Name>("cuDeviceGetName")(name,sizeof(name),ordinal))throw std::runtime_error("No usable NVIDIA GPU");
    Api a; return name;
}
std::unique_ptr<Engine> gpu(const Config& c,int s,int f,int h,const double* x,const double* y,int n) {
    return std::make_unique<Cuda>(c,s,f,h,x,y,n);
}
}
#else
namespace graphfit {
std::string device(){throw std::runtime_error("CUDA graph engine requires Windows x64");}
std::unique_ptr<Engine> gpu(const Config&,int,int,int,const double*,const double*,int){device();return {};}
}
#endif
