# MQL5GPULibrary_LSTM

CUDA recurrent neural networks for MetaTrader 5: stacked **LSTM, GRU and tanh RNN** layers, a linear output head, AdamW training, dropout, gradient clipping and asynchronous progress reporting.

Author: **Tomáš Bělák** · [MIT license](LICENSE.txt) · [Česky](docs/README.cs.md)

![LSTM](LSTM.gif)

## Build and test

Requires **64-bit Windows**, Visual Studio 2022 with C++ tools, a compatible NVIDIA CUDA Toolkit, an NVIDIA GPU and driver, CMake 3.24+, and Python 3 for tests. The locally verified configuration is CUDA 13.0, MSVC 19.44 and RTX 3060 (compute capability 8.6).

```powershell
cmake -S . -B build -A x64 -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --output-on-failure
```

Set `CMAKE_CUDA_ARCHITECTURES` for the deployment GPU. Output: `build/Release/MQL5GPULibrary_LSTM.dll`. The existing Visual Studio solution also supports Debug and Release DLL builds. See [build and verification instructions](docs/BUILDING.md) for CUDA version overrides, dependencies and CPU-only checks.

## Install in MT5

1. Build the DLL from this source revision and copy it to the terminal data directory, `MQL5/Libraries`.
2. Copy `MQL5/Include/MQL5GPULibrary.mqh` to `MQL5/Include`.
3. Copy the indicators and `MQL5/Scripts/LSTM_SmokeTest.mq5` into their corresponding terminal directories. `LSTM_Pure_MTF_EA.mq5` is an Expert Advisor; place it under `MQL5/Experts` when using it as an EA.
4. Make the CUDA runtime, cuBLAS and cuRAND DLLs available to the terminal, enable DLL imports, and compile the desired MQL5 source in MetaEditor.
5. Run `LSTM_SmokeTest` to verify synthetic-data training and prediction before using an indicator.

The smoke script makes no trades. The indicators and EA are examples; model validation and trading performance are separate from library correctness. Generated DLLs are local build artifacts, not source-controlled releases.

## API essentials

Use the shared [MQL5 include](MQL5/Include/MQL5GPULibrary.mqh) or [C header](include/MQL5GPULibrary.h). All original `DN_*` signatures are retained.

```cpp
int h = DN_Create();                         // positive handle, 0 on failure
DN_SetSequenceLength(h, 30);
DN_SetMiniBatchSize(h, 64);
DN_AddLayerEx(h, 8, 64, 0, 0, 0.10);        // 8 features; act=0 selects LSTM
DN_AddLayerEx(h, 0, 32, 0, 0, 0.05);        // infer input width from previous layer
DN_SetOutputDim(h, 2);
// Check every return value; allocate buffers before these calls:
// DN_LoadBatch(h, X, T, samples, 30*8, 2, 0);
// DN_TrainAsync(h, 100, 0.001, 0.0008, 0.0);
// Poll status/progress, then predict after training has finished.
DN_Free(h);
```

- Public inputs are flat sample-major doubles: `X[sample, timestep, feature]`. Targets and predictions are `[sample, output]`. Internal computation uses float32.
- `in = sequence_length * features`, `layout = 0`, `ln = 0`.
- `DN_AddLayerEx` selects **0=LSTM, 1=GRU, 2=RNN**. It does not select an activation function. Layer normalization is not implemented.
- `DN_Train` blocks. `DN_TrainAsync` returns after reserving the model for a worker.
- Prediction, mutation and serialization on a busy model return failure promptly. Progress, status, result polling and stop requests remain available.
- `DN_GetProgressAll` outputs **lr, mse, best_mse**, in that order after its four integer outputs.
- `DN_GetError` copies the last library error as UTF-16. Read it immediately after failure; the diagnostic is shared across handles.

See [the API contract](docs/API.md) for buffer lengths, lifecycle, state formats and migration notes.

## What is tested

Portable tests validate dimension overflow, malformed checkpoint handling and consistency of all 38 exported signatures across CUDA, C and MQL5. GPU tests exercise all recurrent types, stacked networks, CPU reference equations and numerical gradient norms, serialization rollback, asynchronous lifecycle and final loss reporting. Tests run against the newly built DLL; they do not claim to verify trading profitability or MetaEditor compilation.

GitHub Actions runs portable checks on Windows and Linux. GPU checks are an explicit workflow for a Windows runner labeled `cuda`; hosted CPU checks do not substitute for GPU verification.

## Repository map

| Path | Purpose |
| --- | --- |
| `kernel.cu` | CUDA kernels, recurrent layers, training and DLL exports |
| `include/` | C ABI, shared dimension validation and CPU checkpoint parser |
| `MQL5/Include/` | One shared DLL import declaration for all examples |
| `MQL5/Indicators/`, `MQL5/Scripts/` | MT5 examples and smoke test |
| `tests/`, `tools/check_api.py` | CPU, ABI and GPU regression checks |
| `docs/` | API, build and Czech usage documentation |
| `CMakeLists.txt`, `*.sln`, `*.vcxproj` | Build entry points |

[Changes](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [Implementation guide](func_docu.md)

![Poděkování Aničce](ann.svg)
