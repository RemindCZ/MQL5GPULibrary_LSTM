#include <windows.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using MqlBool = int;
constexpr MqlBool MQL_TRUE = 1;
constexpr MqlBool MQL_FALSE = 0;

class DllApi {
public:
    explicit DllApi(const std::string& dllPath) {
        module_ = LoadLibraryA(dllPath.c_str());
        if (!module_) {
            throw std::runtime_error("Nepodarilo se nacist DLL: " + dllPath);
        }
        LoadFunctions();
    }

    ~DllApi() {
        if (module_) {
            FreeLibrary(module_);
        }
    }

    int Create() const { return DN_Create_(); }
    void Free(int h) const { DN_Free_(h); }

    MqlBool SetSequenceLength(int h, int seq) const { return DN_SetSequenceLength_(h, seq); }
    MqlBool SetMiniBatchSize(int h, int mbs) const { return DN_SetMiniBatchSize_(h, mbs); }
    MqlBool AddLayerEx(int h, int in, int out, int act, int ln, double drop) const {
        return DN_AddLayerEx_(h, in, out, act, ln, drop);
    }
    MqlBool SetOutputDim(int h, int out) const { return DN_SetOutputDim_(h, out); }
    MqlBool LoadBatch(int h, const double* X, const double* T, int batch, int in, int out, int l) const {
        return DN_LoadBatch_(h, X, T, batch, in, out, l);
    }
    MqlBool PredictBatch(int h, const double* X, int batch, int in, int l, double* Y) const {
        return DN_PredictBatch_(h, X, batch, in, l, Y);
    }
    int SaveState(int h) const { return DN_SaveState_(h); }
    MqlBool GetState(int h, char* buf, int maxLen) const { return DN_GetState_(h, buf, maxLen); }
    MqlBool LoadState(int h, const char* buf) const { return DN_LoadState_(h, buf); }

    std::wstring GetErrorText() const {
        std::vector<short> buf(512, 0);
        DN_GetError_(buf.data(), static_cast<int>(buf.size()));
        std::wstring text;
        for (short c : buf) {
            if (c == 0) {
                break;
            }
            text.push_back(static_cast<wchar_t>(c));
        }
        return text;
    }

private:
    template <typename T>
    T LoadSymbol(const char* name) {
        auto* symbol = reinterpret_cast<T>(GetProcAddress(module_, name));
        if (!symbol) {
            throw std::runtime_error(std::string("Chybi export: ") + name);
        }
        return symbol;
    }

    void LoadFunctions() {
        DN_Create_ = LoadSymbol<CreateFn>("DN_Create");
        DN_Free_ = LoadSymbol<FreeFn>("DN_Free");
        DN_SetSequenceLength_ = LoadSymbol<SetSequenceLengthFn>("DN_SetSequenceLength");
        DN_SetMiniBatchSize_ = LoadSymbol<SetMiniBatchSizeFn>("DN_SetMiniBatchSize");
        DN_AddLayerEx_ = LoadSymbol<AddLayerExFn>("DN_AddLayerEx");
        DN_SetOutputDim_ = LoadSymbol<SetOutputDimFn>("DN_SetOutputDim");
        DN_LoadBatch_ = LoadSymbol<LoadBatchFn>("DN_LoadBatch");
        DN_PredictBatch_ = LoadSymbol<PredictBatchFn>("DN_PredictBatch");
        DN_SaveState_ = LoadSymbol<SaveStateFn>("DN_SaveState");
        DN_GetState_ = LoadSymbol<GetStateFn>("DN_GetState");
        DN_LoadState_ = LoadSymbol<LoadStateFn>("DN_LoadState");
        DN_GetError_ = LoadSymbol<GetErrorFn>("DN_GetError");
    }

    using CreateFn = int(__stdcall*)();
    using FreeFn = void(__stdcall*)(int);
    using SetSequenceLengthFn = MqlBool(__stdcall*)(int, int);
    using SetMiniBatchSizeFn = MqlBool(__stdcall*)(int, int);
    using AddLayerExFn = MqlBool(__stdcall*)(int, int, int, int, int, double);
    using SetOutputDimFn = MqlBool(__stdcall*)(int, int);
    using LoadBatchFn = MqlBool(__stdcall*)(int, const double*, const double*, int, int, int, int);
    using PredictBatchFn = MqlBool(__stdcall*)(int, const double*, int, int, int, double*);
    using SaveStateFn = int(__stdcall*)(int);
    using GetStateFn = MqlBool(__stdcall*)(int, char*, int);
    using LoadStateFn = MqlBool(__stdcall*)(int, const char*);
    using GetErrorFn = void(__stdcall*)(short*, int);

    HMODULE module_ = nullptr;
    CreateFn DN_Create_ = nullptr;
    FreeFn DN_Free_ = nullptr;
    SetSequenceLengthFn DN_SetSequenceLength_ = nullptr;
    SetMiniBatchSizeFn DN_SetMiniBatchSize_ = nullptr;
    AddLayerExFn DN_AddLayerEx_ = nullptr;
    SetOutputDimFn DN_SetOutputDim_ = nullptr;
    LoadBatchFn DN_LoadBatch_ = nullptr;
    PredictBatchFn DN_PredictBatch_ = nullptr;
    SaveStateFn DN_SaveState_ = nullptr;
    GetStateFn DN_GetState_ = nullptr;
    LoadStateFn DN_LoadState_ = nullptr;
    GetErrorFn DN_GetError_ = nullptr;
};

class TestRunner {
public:
    void Run(const std::string& name, const std::function<void()>& test) {
        try {
            test();
            std::cout << "[PASS] " << name << "\n";
        } catch (const std::exception& ex) {
            ++failed_;
            std::cerr << "[FAIL] " << name << " -> " << ex.what() << "\n";
        }
    }

    int FailedCount() const { return failed_; }

private:
    int failed_ = 0;
};

void ExpectTrue(bool value, const std::string& msg) {
    if (!value) {
        throw std::runtime_error(msg);
    }
}

void ExpectEq(int actual, int expected, const std::string& msg) {
    if (actual != expected) {
        throw std::runtime_error(msg + " (actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected) + ")");
    }
}

void ExpectContains(const std::string& text, const std::string& token, const std::string& msg) {
    if (text.find(token) == std::string::npos) {
        throw std::runtime_error(msg + " (text='" + text + "')");
    }
}

std::string Narrow(const std::wstring& ws) {
    std::string out;
    out.reserve(ws.size());
    for (wchar_t c : ws) {
        out.push_back((c >= 0 && c <= 0x7F) ? static_cast<char>(c) : "?"[0]);
    }
    return out;
}

void TestRejectsInvalidHandle(const DllApi& api) {
    constexpr int invalidHandle = -123;
    ExpectEq(api.SetSequenceLength(invalidHandle, 8), MQL_FALSE, "DN_SetSequenceLength musi odmitnout neplatny handle");
    ExpectEq(api.SetMiniBatchSize(invalidHandle, 4), MQL_FALSE, "DN_SetMiniBatchSize musi odmitnout neplatny handle");
    ExpectEq(api.AddLayerEx(invalidHandle, 2, 4, 0, 0, 0.0), MQL_FALSE, "DN_AddLayerEx musi odmitnout neplatny handle");
    ExpectEq(api.SetOutputDim(invalidHandle, 1), MQL_FALSE, "DN_SetOutputDim musi odmitnout neplatny handle");
    ExpectEq(api.SaveState(invalidHandle), 0, "DN_SaveState musi odmitnout neplatny handle");
}

void TestMinimalInferencePipeline(const DllApi& api) {
    int h = api.Create();
    ExpectTrue(h > 0, "DN_Create selhal (zkontroluj CUDA device a ovladace)");

    struct HandleGuard {
        const DllApi& api;
        int handle;
        ~HandleGuard() { api.Free(handle); }
    } guard{api, h};

    ExpectEq(api.SetSequenceLength(h, 2), MQL_TRUE, "DN_SetSequenceLength selhal");
    ExpectEq(api.SetMiniBatchSize(h, 2), MQL_TRUE, "DN_SetMiniBatchSize selhal");
    ExpectEq(api.AddLayerEx(h, 1, 4, 0, 0, 0.0), MQL_TRUE, "DN_AddLayerEx selhal");
    ExpectEq(api.SetOutputDim(h, 1), MQL_TRUE, "DN_SetOutputDim selhal");

    // batch=2, in=2 => seq_len=2, feature_dim=1
    const std::vector<double> X{0.10, 0.20, 0.30, 0.40};
    const std::vector<double> T{0.0, 1.0};

    ExpectEq(api.LoadBatch(h, X.data(), T.data(), 2, 2, 1, 2), MQL_TRUE,
             "DN_LoadBatch selhal: " + Narrow(api.GetErrorText()));

    std::vector<double> Y(2, 0.0);
    ExpectEq(api.PredictBatch(h, X.data(), 2, 2, 2, Y.data()), MQL_TRUE,
             "DN_PredictBatch selhal: " + Narrow(api.GetErrorText()));

    for (double v : Y) {
        ExpectTrue(std::isfinite(v), "Predikce obsahuje NaN/Inf");
    }
}

void TestStateSerialization(const DllApi& api) {
    int h = api.Create();
    ExpectTrue(h > 0, "DN_Create selhal");

    struct HandleGuard {
        const DllApi& api;
        int handle;
        ~HandleGuard() { api.Free(handle); }
    } guard{api, h};

    ExpectEq(api.AddLayerEx(h, 1, 2, 0, 0, 0.0), MQL_TRUE, "DN_AddLayerEx selhal");
    ExpectEq(api.SetOutputDim(h, 1), MQL_TRUE, "DN_SetOutputDim selhal");

    int stateLen = api.SaveState(h);
    ExpectTrue(stateLen > 0, "DN_SaveState vratil prazdny buffer");

    std::vector<char> state(static_cast<size_t>(stateLen), '\0');
    ExpectEq(api.GetState(h, state.data(), static_cast<int>(state.size())), MQL_TRUE,
             "DN_GetState selhal");

    std::string stateText(state.data());
    ExpectContains(stateText, "LSTM_V1", "Serializace musi obsahovat hlavicku LSTM_V1");

    ExpectEq(api.LoadState(h, "BROKEN_STATE"), MQL_FALSE,
             "DN_LoadState musi odmitnout poskozeny vstup");

    ExpectEq(api.LoadState(h, stateText.c_str()), MQL_TRUE,
             "DN_LoadState selhal nad validnim stavem");
}

} // namespace

int main(int argc, char** argv) {
    const std::string dllPath = (argc > 1) ? argv[1] : "MQL5GPULibrary_LSTM.dll";

    TestRunner runner;
    DllApi api(dllPath);

    runner.Run("Invalid handle guard clauses", [&] { TestRejectsInvalidHandle(api); });
    runner.Run("Minimal load + predict", [&] { TestMinimalInferencePipeline(api); });
    runner.Run("State serialization roundtrip", [&] { TestStateSerialization(api); });

    if (runner.FailedCount() != 0) {
        std::cerr << "\nNeuspesnych testu: " << runner.FailedCount() << "\n";
        return 1;
    }

    std::cout << "\nVsechny testy probehly uspesne.\n";
    return 0;
}
