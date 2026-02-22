# MQL5GPULibrary_LSTM

`MQL5GPULibrary_LSTM` is a Windows x64 DLL that exposes a CUDA-accelerated, multi-layer LSTM engine to MetaTrader 5 (MQL5). It is designed for low-latency usage in trading systems where training can run asynchronously while an EA or script continues to process market events.

---

## Table of Contents

1. [What this library provides](#what-this-library-provides)
2. [Runtime and platform requirements](#runtime-and-platform-requirements)
3. [Installation in MetaTrader 5](#installation-in-metatrader-5)
4. [High-level API lifecycle](#high-level-api-lifecycle)
5. [Exported API reference](#exported-api-reference)
6. [Data layout and tensor shapes](#data-layout-and-tensor-shapes)
7. [Asynchronous training model](#asynchronous-training-model)
8. [Model persistence (save/load state)](#model-persistence-saveload-state)
9. [Error handling and diagnostics](#error-handling-and-diagnostics)
10. [MQL5 usage example](#mql5-usage-example)
11. [Build from source](#build-from-source)
12. [Performance and stability recommendations](#performance-and-stability-recommendations)
13. [Known limitations](#known-limitations)
14. [License](#license)

---

## What this library provides

### Core capabilities

- CUDA-accelerated computation on NVIDIA GPUs.
- Multi-layer LSTM topology with configurable layer dimensions.
- Optional dropout parameter at layer configuration level.
- Gradient clipping support.
- Batch prediction API.
- Asynchronous training in a worker thread.
- Model snapshot/restore functions.
- In-memory serialization/deserialization of model state.

### Intended use

This DLL is intended to be called from MQL5 using `#import`. Typical usage is:

1. Create an instance.
2. Configure sequence length, mini-batch size, layers, output size, and optimization settings.
3. Load training data.
4. Start asynchronous training.
5. Poll training state and retrieve results.
6. Run inference and/or save model state.
7. Free instance resources.

---

## Runtime and platform requirements

- **OS:** Windows 64-bit.
- **Trading platform:** MetaTrader 5 64-bit.
- **GPU:** NVIDIA GPU with CUDA support.
- **CUDA runtime:** Must match the DLL/toolkit compatibility requirements in your environment.

At runtime (on the machine running MT5), required CUDA DLLs must be discoverable (e.g., CUDA runtime / cuBLAS / cuRAND binaries compatible with this build).

---

## Installation in MetaTrader 5

1. Copy `MQL5GPULibrary_LSTM.dll` into your MT5 data folder under:
   - `MQL5\Libraries`
2. In MetaTrader, enable **Allow DLL imports** for your EA/script.
3. Declare imports in MQL5 using exact signatures (see [Exported API reference](#exported-api-reference)).

---

## High-level API lifecycle

Recommended lifecycle for each network handle:

1. `DN_Create`
2. `DN_SetSequenceLength`
3. `DN_SetMiniBatchSize`
4. `DN_AddLayerEx` (one or more times)
5. `DN_SetOutputDim`
6. `DN_SetGradClip` (optional)
7. `DN_LoadBatch`
8. `DN_TrainAsync` and status polling via `DN_GetTrainingStatus`
9. `DN_PredictBatch` and/or state functions (`DN_SaveState`, `DN_GetState`, `DN_LoadState`)
10. `DN_StopTraining` (if needed)
11. `DN_Free`

> Tip: Keep input dimensions consistent across layer definitions and batch loading to avoid validation failures.

---

## Exported API reference

> Calling convention: `__stdcall`.
>
> Boolean return type uses `MQL_BOOL` (`1 = true`, `0 = false`).

### 1) Instance management

#### `int DN_Create()`
Creates a new network instance and returns a positive handle.

- **Returns:**
  - `>0`: valid handle
  - `0`: creation failed

#### `void DN_Free(int h)`
Stops running training (if any), waits for worker completion, and releases all resources for handle `h`.

---

### 2) Network configuration

#### `bool DN_SetSequenceLength(int h, int seq_len)`
Sets sequence length used by the network.

#### `bool DN_SetMiniBatchSize(int h, int mbs)`
Sets mini-batch size.

#### `bool DN_AddLayerEx(int h, int in, int out, int act, int ln, double drop)`
Adds one LSTM layer.

- `in`: input size for this layer
- `out`: hidden size for this layer
- `act`: activation selector (reserved/implementation-dependent)
- `ln`: layer normalization selector (reserved/implementation-dependent)
- `drop`: dropout value

#### `bool DN_SetOutputDim(int h, int out_dim)`
Initializes/updates output layer dimension.

#### `bool DN_SetGradClip(int h, double clip)`
Sets gradient clipping threshold.

---

### 3) Data I/O and inference

#### `bool DN_LoadBatch(int h, const double* X, const double* T, int batch, int in, int out, int l)`
Loads one training batch into the network.

- `X`: input tensor buffer (double)
- `T`: target tensor buffer (double)
- `batch`: batch size
- `in`: input feature size
- `out`: output feature size
- `l`: sequence length

#### `bool DN_PredictBatch(int h, const double* X, int batch, int in, int l, double* Y)`
Runs prediction for a batch.

- `X`: input buffer
- `Y`: output buffer written by DLL

---

### 4) Training control (async)

#### `bool DN_TrainAsync(int h, int epochs, double target_mse, double lr, double wd)`
Starts asynchronous training in a worker thread.

- `epochs`: max training epochs
- `target_mse`: stop criterion (target loss)
- `lr`: learning rate
- `wd`: weight decay

#### `int DN_GetTrainingStatus(int h)`
Returns current training state.

- `0`: idle
- `1`: training
- `2`: completed
- `-1`: error / invalid handle

#### `void DN_GetTrainingResult(int h, double* out_mse, int* out_epochs)`
Fetches latest training result values.

- `out_mse`: final/last MSE
- `out_epochs`: number of processed epochs

#### `void DN_StopTraining(int h)`
Requests training stop.

---

### 5) Weights, diagnostics, and model state

#### `bool DN_SnapshotWeights(int h)`
Stores an internal snapshot of current weights.

#### `bool DN_RestoreWeights(int h)`
Restores previously snapshotted weights.

#### `int DN_GetLayerCount(int h)`
Returns number of configured LSTM layers.

#### `double DN_GetLayerWeightNorm(int h, int l)`
Diagnostic placeholder (current implementation returns default value).

#### `double DN_GetGradNorm(int h)`
Diagnostic placeholder (current implementation returns default value).

#### `int DN_SaveState(int h)`
Serializes model into internal string buffer.

- **Returns:** required byte length including null terminator, or `0` on failure.

#### `bool DN_GetState(int h, char* buf, int max_len)`
Copies serialized model text into caller buffer.

- Must be called with buffer size at least value returned by `DN_SaveState`.

#### `bool DN_LoadState(int h, const char* buf)`
Loads model from serialized state text.

#### `void DN_GetError(short* buf, int len)`
Copies last error message into UTF-16-like short buffer (null-terminated).

---

## Data layout and tensor shapes

Input and target buffers are passed as flat `double*` arrays from MQL5.
Internally, the library converts to float and executes GPU operations.

For reliable integration, keep these dimensions consistent:

- `seq_len` from configuration should match `l` used in `DN_LoadBatch` and `DN_PredictBatch`.
- `in` in `DN_LoadBatch` / `DN_PredictBatch` should match model input configuration.
- `out` in `DN_LoadBatch` should match configured output dimension.
- Output buffer `Y` must be preallocated by caller with enough capacity for the full predicted batch.

---

## Asynchronous training model

- `DN_TrainAsync` starts training in a background worker.
- Poll `DN_GetTrainingStatus` from MQL5 timer/event loop.
- Use `DN_GetTrainingResult` when status is `2` (completed) or when stopping.
- Call `DN_StopTraining` for cooperative cancellation.
- `DN_Free` also ensures training is stopped and resources are cleaned safely.

This design allows MT5 logic (ticks, order management, indicators) to continue while training runs.

---

## Model persistence (save/load state)

Typical save workflow:

1. `int n = DN_SaveState(h);`
2. Allocate `char buffer[n];`
3. `DN_GetState(h, buffer, n);`
4. Persist `buffer` as text (file/database/global variable).

Typical load workflow:

1. Read previously stored text into null-terminated `char*`.
2. Call `DN_LoadState(h, textBuffer);`

Use this mechanism to transfer trained weights between sessions without retraining from scratch.

---

## Error handling and diagnostics

- Most API calls return `false` (`0`) on failure.
- On failure, call `DN_GetError` to retrieve the last error message.
- Error messages include CUDA/cuBLAS/cuRAND failures and some validation/runtime issues.
- `DN_GetTrainingStatus` can return `-1` for invalid handles / error state access.

Minimal error retrieval pattern in MQL5:

```mq5
short errBuf[512];
ArrayInitialize(errBuf, 0);
DN_GetError(errBuf, 512);
string msg = ShortArrayToString(errBuf);
Print("DLL error: ", msg);
```

---

## MQL5 usage example

```mq5
#import "MQL5GPULibrary_LSTM.dll"
int  DN_Create();
void DN_Free(int h);
bool DN_SetSequenceLength(int h, int seq_len);
bool DN_SetMiniBatchSize(int h, int mbs);
bool DN_AddLayerEx(int h, int in, int out, int act, int ln, double drop);
bool DN_SetOutputDim(int h, int out_dim);
bool DN_SetGradClip(int h, double clip);
bool DN_LoadBatch(int h, const double &X[], const double &T[], int batch, int in, int out, int l);
bool DN_TrainAsync(int h, int epochs, double target_mse, double lr, double wd);
int  DN_GetTrainingStatus(int h);
void DN_GetTrainingResult(int h, double &mse, int &epochs_done);
void DN_StopTraining(int h);
bool DN_PredictBatch(int h, const double &X[], int batch, int in, int l, double &Y[]);
#import

int g_net = 0;

double g_last_mse = 0.0;
int    g_last_epochs = 0;

int OnInit()
{
   g_net = DN_Create();
   if(g_net <= 0)
      return INIT_FAILED;

   if(!DN_SetSequenceLength(g_net, 32)) return INIT_FAILED;
   if(!DN_SetMiniBatchSize(g_net, 64)) return INIT_FAILED;

   // Example topology: input=16 -> hidden=64 -> hidden=32 -> output=1
   if(!DN_AddLayerEx(g_net, 16, 64, 0, 0, 0.1)) return INIT_FAILED;
   if(!DN_AddLayerEx(g_net, 64, 32, 0, 0, 0.1)) return INIT_FAILED;
   if(!DN_SetOutputDim(g_net, 1)) return INIT_FAILED;

   if(!DN_SetGradClip(g_net, 1.0)) return INIT_FAILED;

   // TODO: fill XTrain and TTrain with your data and call DN_LoadBatch(...)
   // if(!DN_LoadBatch(g_net, XTrain, TTrain, 64, 16, 1, 32)) return INIT_FAILED;

   if(!DN_TrainAsync(g_net, 2000, 0.001, 0.0005, 0.0001))
      return INIT_FAILED;

   EventSetTimer(1);
   return INIT_SUCCEEDED;
}

void OnTimer()
{
   int status = DN_GetTrainingStatus(g_net);
   if(status == 2)
   {
      DN_GetTrainingResult(g_net, g_last_mse, g_last_epochs);
      Print("Training completed. MSE=", DoubleToString(g_last_mse, 8),
            " epochs=", g_last_epochs);
      EventKillTimer();
   }
}

void OnDeinit(const int reason)
{
   DN_StopTraining(g_net);
   DN_Free(g_net);
}
```

---

## Build from source

Project files included:

- `MQL5GPULibrary_LSTM.sln`
- `MQL5GPULibrary_LSTM.vcxproj`
- `kernel.cu`

Recommended build steps:

1. Open `MQL5GPULibrary_LSTM.sln` in Visual Studio (x64 toolchain).
2. Ensure CUDA build customization is installed (project references CUDA 13.0 props/targets).
3. Select **Release | x64**.
4. Verify CUDA toolkit path and environment.
5. Build the solution to produce `MQL5GPULibrary_LSTM.dll`.

---

## Performance and stability recommendations

- Use a fixed, realistic sequence length and batch size based on your strategy latency budget.
- Avoid reallocating network topology every tick; configure once and reuse handle.
- Keep training batches dimensionally consistent.
- Use asynchronous training + status polling instead of blocking calls from `OnTick`.
- Always stop/free handles on deinitialization to prevent leaked resources.

---

## Known limitations

- Diagnostic norm functions currently return placeholder values:
  - `DN_GetLayerWeightNorm` returns `0.0`
  - `DN_GetGradNorm` returns `0.0`
- API is Windows-focused (`__stdcall`, Win32 DLL export model).
- CUDA runtime compatibility must match deployment machine.

---

## License

Projekt je licencovaný pod MIT licencí. Podrobnosti jsou v `LICENSE.txt`.

## Přiložený indikátor pro MT5

V repozitáři je i ukázkový indikátor `MQL5/Indicators/LSTMTrendStart.mq5`, který používá tuto DLL pro detekci začínajícího trendu:

- trénuje model nad posledním oknem historických dat (returns)
- periodicky spouští asynchronní přeučení
- na každé nové svíčce predikuje trend score
- vykreslí šipku nahoru/dolů při překročení prahu `InpSignalThreshold`

### Nasazení v MetaTrader 5

1. Zkopírujte `MQL5GPULibrary_LSTM.dll` do `MQL5\Libraries`.
2. Zkopírujte `MQL5/Indicators/LSTMTrendStart.mq5` do `MQL5\Indicators` (datová složka terminálu).
3. V MetaEditoru indikátor zkompilujte.
4. V MT5 zapněte **Allow DLL imports**.
5. Přidejte indikátor do grafu a upravte inputy (sekvence, retrain interval, threshold) podle trhu/timeframu.
