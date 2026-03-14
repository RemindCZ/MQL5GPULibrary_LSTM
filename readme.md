# MQL5GPULibrary_LSTM

> **Author of this documentation:** Tomáš Bělák  
> **Scope:** production-grade technical documentation for CUDA-accelerated LSTM training/inference from MetaTrader 5 (MQL5) via DLL bridge.

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-brightgreen.svg)
![MT5](https://img.shields.io/badge/MetaTrader%205-DLL%20API-blue.svg)

---

## 1. Executive Summary

`MQL5GPULibrary_LSTM` is a hybrid quantitative-engineering project that exposes a CUDA-backed recurrent neural network runtime (LSTM-centric, with additional RNN/GRU layer support) to **MetaTrader 5** through a DLL API designed for low-latency practical use.

The project solves a common integration bottleneck in algorithmic trading systems:

- MQL5 is excellent for strategy orchestration, execution logic, and chart-native tooling,
- while high-throughput deep learning training is better suited to C++/CUDA.

This repository bridges those worlds with:

1. **Handle-based model lifecycle management** (`DN_Create`, `DN_Free`),
2. **Flexible network construction** (`DN_AddLayerEx`, `DN_AddGRULayer`, `DN_AddRNNLayer`),
3. **Synchronous and asynchronous training** (`DN_Train`, `DN_TrainAsync`),
4. **Detailed progress telemetry** (`DN_GetProgress*`, `DN_GetProgressAll`),
5. **State persistence and rollback tools** (`DN_SaveState`, `DN_GetState`, `DN_LoadState`, snapshots),
6. **Batch inference API suitable for indicator- or EA-level integration** (`DN_PredictBatch`).

---

## 2. Architecture and Design Goals

### 2.1 Primary design goals

The runtime is designed around five practical constraints from real trading-system environments:

- **Deterministic ownership:** each model instance is represented by an integer handle, minimizing cross-context ambiguity.
- **Operational resilience:** every mutating API call can be validated and followed by centralized error retrieval (`DN_GetError`).
- **UI responsiveness in MT5:** async training allows heavy GPU workloads without freezing chart logic or user interactions.
- **Production telemetry:** rich progress metadata (epochs, minibatches, best MSE, ETA, gradient norm) enables informed runtime decisions.
- **Experiment safety:** snapshots and state serialization reduce iteration risk and support reproducible workflows.

### 2.2 High-level component map

- **`kernel.cu`**: CUDA/C++ core implementing model graph, training loop, telemetry, and exported DLL boundary.
- **`MQL5/Indicators/*.mq5`**: integration layer + practical examples of usage in MT5 indicator context.
- **`docs/`**: static visualization and interactive documentation assets.
- **Solution/project files (`.sln`, `.vcxproj`)**: reproducible build entry points for Windows toolchain.

---

## 3. API Model and Lifecycle Contract

### 3.1 Lifecycle primitives

```cpp
int DN_Create();
void DN_Free(int h);
```

- `DN_Create` allocates and registers a new model context and returns a positive handle on success.
- `DN_Free` must be called exactly once per valid handle to release associated resources.

**Operational rule:** in MQL5, treat the handle as a managed resource; reset it to an invalid value after `DN_Free` to prevent accidental reuse.

### 3.2 Configuration stage

```cpp
MQL_BOOL DN_SetSequenceLength(int h, int seq_len);
MQL_BOOL DN_SetMiniBatchSize(int h, int mbs);
MQL_BOOL DN_AddLayerEx(int h, int in, int out, int act, int ln, double drop);
MQL_BOOL DN_AddGRULayer(int h, int in, int out, double drop);
MQL_BOOL DN_AddRNNLayer(int h, int in, int out, double drop);
MQL_BOOL DN_SetGradClip(int h, double clip);
MQL_BOOL DN_SetOutputDim(int h, int out_dim);
```

#### Configuration invariants

- `seq_len > 0`
- `mbs > 0`
- `0.0 <= drop < 1.0`
- `out_dim` must match target tensor dimensionality.

For canonical LSTM architecture stacks, `DN_AddLayerEx` should be considered the primary constructor.

### 3.3 Data loading and prediction

```cpp
MQL_BOOL DN_LoadBatch(int h, const double* X, const double* T,
                      int batch, int in, int out, int l);
MQL_BOOL DN_PredictBatch(int h, const double* X,
                         int batch, int in, int l, double* Y);
```

#### Tensor-shape contract (critical)

- `in = seq_len * feature_dim`
- `len(X) = batch * in`
- `len(T) = batch * out`
- `len(Y) = batch * out_dim`
- `in % seq_len == 0` must hold

Violating these assumptions typically produces immediate API failure and a meaningful error message retrievable via `DN_GetError`.

---

## 4. Training Engine Modes

### 4.1 Synchronous training

```cpp
MQL_BOOL DN_Train(int h, int epochs, double target_mse, double lr, double wd);
```

Use synchronous mode when:

- integrating in offline optimization scripts,
- deterministic blocking is acceptable,
- simple control flow is preferred.

### 4.2 Asynchronous training

```cpp
MQL_BOOL DN_TrainAsync(int h, int epochs, double target_mse, double lr, double wd);
int DN_GetTrainingStatus(int h);
void DN_GetTrainingResult(int h, double* out_mse, int* out_epochs);
void DN_StopTraining(int h);
```

Use asynchronous mode when:

- the indicator/EA must keep processing ticks or UI events,
- long training runs require progress visibility,
- strategy architecture enforces non-blocking behavior.

#### Recommended status machine semantics

- `0` → idle
- `1` → training in progress
- `2` → completed
- `-1` → error

---

## 5. Progress Telemetry (Operational Intelligence)

```cpp
int DN_GetProgressEpoch(int h);
int DN_GetProgressTotalEpochs(int h);
int DN_GetProgressMiniBatch(int h);
int DN_GetProgressTotalMiniBatches(int h);
double DN_GetProgressLR(int h);
double DN_GetProgressMSE(int h);
double DN_GetProgressBestMSE(int h);
double DN_GetProgressGradNorm(int h);
int DN_GetProgressTotalSteps(int h);
double DN_GetProgressPercent(int h);
double DN_GetProgressElapsedSec(int h);
double DN_GetProgressETASec(int h);
MQL_BOOL DN_GetProgressAll(
    int h,
    int* epoch, int* total_epochs,
    int* minibatch, int* total_minibatches,
    double* mse, double* best_mse,
    double* lr, double* grad_norm,
    double* percent, double* elapsed_sec, double* eta_sec
);
```

### Telemetry interpretation guide

- **`mse`**: current optimization state, noisy at batch granularity.
- **`best_mse`**: stability anchor; evaluate convergence versus this metric.
- **`grad_norm`**: early warning for exploding/unstable optimization.
- **`eta_sec`**: operational estimate for UI and task scheduling.

### Polling best practices in MT5

- Poll from `OnTimer` (e.g., 100–500 ms), not every tick.
- Prefer `DN_GetProgressAll` to reduce call overhead and improve consistency.
- Persist sampled telemetry for post-mortem model diagnostics.

---

## 6. State Persistence, Rollback, and Diagnostics

```cpp
int DN_GetLayerCount(int h);
double DN_GetLayerWeightNorm(int h, int l);
double DN_GetGradNorm(int h);

int DN_SaveState(int h);
MQL_BOOL DN_GetState(int h, char* buf, int max_len);
MQL_BOOL DN_LoadState(int h, const char* buf);

MQL_BOOL DN_SnapshotWeights(int h);
MQL_BOOL DN_RestoreWeights(int h);

void DN_GetError(short* buf, int len);
```

### Practical production patterns

- **Checkpointing policy:** call `DN_SaveState` at milestone convergence points and export state string externally.
- **Safe experimentation:** `DN_SnapshotWeights` before high-risk hyperparameter changes; restore when divergence is detected.
- **Failure triage:** on any `MQL_FALSE`, immediately call `DN_GetError` and log the returned message with context.

---

## 7. Full MT5 Integration Workflow

### 7.1 Initialization (`OnInit`)

1. `DN_Create`
2. Configure sequence length and minibatch
3. Build layer stack
4. Set output dimension
5. Load training batch
6. Optionally snapshot
7. Start `DN_TrainAsync`

### 7.2 Runtime loop (`OnTimer`)

1. Query status (`DN_GetTrainingStatus`)
2. Read telemetry (`DN_GetProgressAll`)
3. Update chart UI / logs
4. If completed, collect result (`DN_GetTrainingResult`)
5. Optionally run inference (`DN_PredictBatch`)

### 7.3 Deinitialization (`OnDeinit`)

1. If running, request stop (`DN_StopTraining`)
2. Release handle (`DN_Free`)
3. Zero/reset local handle variable

---

## 8. Build and Deployment

### 8.1 Build prerequisites

- Windows with Visual Studio capable of opening `.sln`
- NVIDIA CUDA Toolkit compatible with your compiler toolset
- x64 target environment

### 8.2 Build process

1. Open `MQL5GPULibrary_LSTM.sln`
2. Select `Release | x64`
3. Build solution
4. Copy resulting DLL into terminal data folder `MQL5\Libraries`

### 8.3 MT5 deployment

1. Copy indicator files from `MQL5/Indicators` into `MQL5\Indicators`
2. Compile in MetaEditor
3. Enable DLL imports in MT5 settings
4. Attach indicator to chart and verify initialization logs

---

## 9. Performance and Stability Recommendations

### Hyperparameters

- Start conservatively (`lr`, `epochs`) and tune incrementally.
- Use `target_mse` for controlled early stopping.
- Apply gradient clipping (`DN_SetGradClip`) when gradient norm spikes.

### Data pipeline

- Validate shape arithmetic before every `DN_LoadBatch` call.
- Keep feature normalization strategy consistent between train and inference.
- Use fixed seeds and deterministic preprocessing where possible.

### Concurrency and resource hygiene

- Isolate each strategy/model with a unique handle.
- Do not share mutable buffers across asynchronous model operations.
- Always stop and free cleanly on chart unload or strategy reset.

---

## 10. Repository Map

- `kernel.cu` — CUDA DLL implementation, core training/inference runtime.
- `MQL5/Indicators/LSTM_RealTimePredictor.mq5` — primary practical MT5 integration example.
- `MQL5/Indicators/Examples/LSTM_PatternCompletion_Demo.mq5` — demo-oriented usage.
- `docs/index.html`, `docs/app.js`, `docs/lstm-flow.svg` — static documentation artifacts.
- `MQL5GPULibrary_LSTM.sln`, `MQL5GPULibrary_LSTM.vcxproj` — build orchestration.
- `LICENSE.txt` — licensing terms.

---

## 11. Troubleshooting Matrix

1. **Any API returns failure (`MQL_FALSE`)**  
   → call `DN_GetError`, log immediately, include context (handle, operation, dimensions).

2. **Training does not converge**  
   → reduce `lr`, inspect `grad_norm`, enable/adjust clipping, validate target scaling.

3. **Status remains idle after async start**  
   → verify initialization order and successful `DN_TrainAsync` return code.

4. **Prediction output malformed**  
   → re-check `in`, `seq_len`, `out_dim`, and allocated output buffer length.

5. **MT5 instability during long sessions**  
   → audit handle lifecycle and deinitialization discipline, reduce polling frequency.

---

## 12. License

This project is distributed under the **MIT License**. See `LICENSE.txt` for the full legal text.

---

### A small personal note
![Poděkování Aničce](ann.svg)

Děkuji Aničce za pomoc při nočním přemýšlení nad zbytečně složitým kódem. 💙
