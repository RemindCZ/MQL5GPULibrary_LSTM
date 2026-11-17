#include "forecast_model.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#ifdef DN3_ORT
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <filesystem>
#define ORT_API_MANUAL_INIT
#endif
#include <onnxruntime_cxx_api.h>
#include <array>
#include <locale>
#include <sstream>
#endif

namespace forecast {
bool valid_window(int steps, int markets) {
    return steps >= 16 && steps <= 1024 && markets >= 1 && markets <= 16;
}

bool encode(int steps, int markets, const double* closes, float* output) {
    for (int m=0; m<markets; ++m) {
        int age=steps;
        double previous=closes[m];
        for (int t=0; t<=steps; ++t) {
            const double price=closes[t*markets+m];
            if (!std::isfinite(price) || price<0 || (m==0 && price==0)) return false;
            if (t>0) {
                const bool available=price>0 && previous>0;
                float* channel=output+3*((t-1)*markets+m);
                channel[0]=available ? static_cast<float>(std::log(price)-std::log(previous)) : 0;
                channel[1]=available ? 1.0f : 0.0f;
                age=price>0 ? 0 : std::min(steps,age+1);
                channel[2]=static_cast<float>(age)/steps;
            } else if (price>0) age=0;
            previous=price;
        }
    }
    return true;
}

class Reference final : public Model {
    Shape s;
public:
    explicit Reference(Shape shape):s(shape) {}
    std::vector<double> predict(const std::vector<float>& input) override {
        // Re-scale past cumulative returns by variance observed strictly before each window.
        std::vector<double> sums(s.steps+1), variance(s.steps+1);
        constexpr int warmup=8;
        constexpr double floor=1e-16;
        for (int t=0;t<s.steps;++t) {
            const double r=input[t*3*s.markets];
            sums[t+1]=sums[t]+r;
            variance[t+1]=t<warmup ? (variance[t]*t+r*r)/(t+1) : .94*variance[t]+.06*r*r;
        }
        std::vector<double> result(s.outputs()), samples;
        samples.reserve(s.steps);
        for (int h=1;h<=s.horizons;++h) {
            samples.clear();
            const double v=std::max(floor,variance.back())*h;
            for (int t=warmup;t+h<=s.steps;++t) {
                const double scale=std::sqrt(v/(h*std::max(floor,variance[t])));
                samples.push_back((sums[t+h]-sums[t])*scale);
            }
            std::sort(samples.begin(),samples.end());
            auto quantile=[&](double q) {
                const double pos=q*(samples.size()-1);
                const auto lower=static_cast<size_t>(pos);
                return samples[lower]+(pos-lower)*(samples[static_cast<size_t>(std::ceil(pos))]-samples[lower]);
            };
            double* row=result.data()+(h-1)*DN3_FIELDS;
            row[DN3_POINT]=0;
            row[DN3_Q10]=quantile(.1); row[DN3_Q50]=quantile(.5); row[DN3_Q90]=quantile(.9);
            row[DN3_P_UP]=.5; row[DN3_VARIANCE]=v;
        }
        return result;
    }
};

void check_output(const std::vector<double>& result) {
    for (size_t i=0;i<result.size();i+=DN3_FIELDS) {
        const auto r=result.data()+i;
        if (!std::all_of(r,r+DN3_FIELDS,[](double x){return std::isfinite(x);}) ||
            r[DN3_Q10]>r[DN3_Q50] || r[DN3_Q50]>r[DN3_Q90] ||
            r[DN3_P_UP]<0 || r[DN3_P_UP]>1 || r[DN3_VARIANCE]<=0)
            throw std::runtime_error("Invalid distribution in model output");
    }
}

#ifdef DN3_ORT
static bool init_ort() {
#ifdef _WIN32
    // MT5 can have its own inference engine. Bind the SDK next to this DLL explicitly.
    static const auto runtime=[]() -> std::shared_ptr<void> {
        HMODULE self=nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                reinterpret_cast<LPCWSTR>(&DN3_GetAbiVersion),&self)) return {};
        std::wstring path(32768,L'\0');
        const auto n=GetModuleFileNameW(self,path.data(),static_cast<DWORD>(path.size()));
        if (!n || n>=path.size()) return {};
        path.resize(n);
        const auto file=std::filesystem::path(path).parent_path()/L"onnxruntime.dll";
        auto dll=LoadLibraryExW(file.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!dll) return {};
        std::shared_ptr<void> owner(dll,[](void* h){FreeLibrary(static_cast<HMODULE>(h));});
        auto base=reinterpret_cast<decltype(&OrtGetApiBase)>(GetProcAddress(dll,"OrtGetApiBase"));
        if (!base) return {};
        const OrtApi* api=base()->GetApi(ORT_API_VERSION);
        if (!api) return {};
        Ort::InitApi(api);
        return owner;
    }();
    return static_cast<bool>(runtime);
#else
    return true;
#endif
}
class Onnx final : public Model {
    Shape s;
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    std::array<int64_t,3> input_shape;
    std::unique_ptr<Reference> reference;
    std::vector<int> tasks;
    static void tensor(const Ort::TypeInfo& type, const std::vector<int64_t>& shape) {
        auto t=type.GetTensorTypeAndShapeInfo();
        if (t.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || t.GetShape()!=shape)
            throw std::runtime_error("Incorrect ONNX tensor type or dimensions");
    }
public:
    Onnx(Shape shape,int backend,const std::string& contract,const std::vector<uint8_t>& bytes):
        s(shape),input_shape{1,s.steps,s.markets*3} {
        static Ort::Env env(ORT_LOGGING_LEVEL_WARNING,"DNForecast");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetInterOpNumThreads(1);
        if (backend==DN3_ONNX_CUDA) {
            OrtCUDAProviderOptions cuda{};
            options.AppendExecutionProvider_CUDA(cuda);
        }
        session=Ort::Session(env,bytes.data(),bytes.size(),options);
        Ort::AllocatorWithDefaultOptions alloc;
        if (session.GetInputCount()!=1 || session.GetOutputCount()!=1 ||
            std::string(session.GetInputNameAllocated(0,alloc).get())!="features" ||
            std::string(session.GetOutputNameAllocated(0,alloc).get())!="forecast")
            throw std::runtime_error("Expected one input features and one output forecast");
        tensor(session.GetInputTypeInfo(0),{1,s.steps,3*s.markets});
        tensor(session.GetOutputTypeInfo(0),{1,s.horizons,DN3_FIELDS});
        auto meta=session.GetModelMetadata();
        auto description=meta.LookupCustomMetadataMapAllocated("dn3.contract",alloc);
        if (!description || description.get()!=contract) throw std::runtime_error("Market or timeframe contract mismatch");
        auto expected=meta.LookupCustomMetadataMapAllocated("dn3.zero_probe",alloc);
        if (!expected) throw std::runtime_error("Missing reference tensor");
        std::istringstream stream(expected.get());
        stream.imbue(std::locale::classic());
        const auto actual=predict(std::vector<float>(s.features(),0));
        check_output(actual);
        for (double x:actual) {
            double ref;
            if (!(stream>>ref) || !std::isfinite(ref) || std::abs(x-ref)>1e-8+1e-4*std::abs(ref))
                throw std::runtime_error("Reference tensor mismatch");
        }
        std::string extra;
        if (stream>>extra) throw std::runtime_error("Extra reference values");
        auto policy=meta.LookupCustomMetadataMapAllocated("dn3.tasks",alloc);
        if (policy) {
            std::istringstream selected(policy.get());
            for (int h=0;h<s.horizons;++h) {
                int mask;
                if (!(selected>>mask) || mask<0 || mask>15) throw std::runtime_error("Invalid per-horizon task mask");
                tasks.push_back(mask);
            }
            if (selected>>extra) throw std::runtime_error("Unexpected extra task masks");
            if (std::any_of(tasks.begin(),tasks.end(),[](int mask){return mask!=15;}))
                reference=std::make_unique<Reference>(s);
        }
    }
    std::vector<double> predict(const std::vector<float>& input) override {
        auto tensor=Ort::Value::CreateTensor<float>(memory,const_cast<float*>(input.data()),input.size(),
                                                     input_shape.data(),input_shape.size());
        const char* in="features"; const char* out="forecast";
        auto result=session.Run(Ort::RunOptions{nullptr},&in,&tensor,1,&out,1);
        const float* values=result.front().GetTensorData<float>();
        std::vector<double> output(values,values+s.outputs());
        if (reference) {
            const auto base=reference->predict(input);
            constexpr int group[DN3_FIELDS]={1,2,2,2,4,8};
            for (int h=0;h<s.horizons;++h)
                for (int f=0;f<DN3_FIELDS;++f)
                    if (!(tasks[h]&group[f])) output[h*DN3_FIELDS+f]=base[h*DN3_FIELDS+f];
        }
        return output;
    }
};
#endif

int capabilities() {
    int result=1;
#ifdef _WIN32
    result|=8;
#endif
#ifdef DN3_ORT
    if (!init_ort()) return result;
    result|=2;
    for (const auto& provider:Ort::GetAvailableProviders())
        if (provider=="CUDAExecutionProvider") result|=4;
#endif
    return result;
}
std::unique_ptr<Model> load_model(Shape shape,int backend,const std::string& contract,
                                  const std::vector<uint8_t>& bytes) {
    if (backend==DN3_REFERENCE) return std::make_unique<Reference>(shape);
    if (backend==DN3_LEGACY) {
#ifdef _WIN32
        extern std::unique_ptr<Model> load_legacy(Shape,const std::string&,const std::vector<uint8_t>&);
        return load_legacy(shape,contract,bytes);
#else
        throw std::runtime_error("Legacy CUDA DLL requires Windows x64");
#endif
    }
#ifdef DN3_ORT
    if (!init_ort()) throw std::runtime_error("Compatible ONNX Runtime unavailable alongside DNForecast");
    return std::make_unique<Onnx>(shape,backend,contract,bytes);
#else
    throw std::runtime_error("Build with DN3_ORT_ROOT to use ONNX");
#endif
}
}
