# MQL5GPULibrary_LSTM

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Open Source](https://img.shields.io/badge/Open%20Source-Yes-brightgreen.svg)

`MQL5GPULibrary_LSTM` is an open-source CUDA DLL for MetaTrader 5 that implements a multi-layer LSTM network with asynchronous training, dropout, checkpointing, model-state serialization, and real-time progress telemetry.

The current runtime is aligned with **LSTM DLL v1.4.0** and is compatible with newer indicator patterns (for example `LSTM_RealTimePredictor.mq5 v2.1`) that rely on unified progress polling through `DN_GetProgressAll()`.

🎬 **Video walkthrough:** https://youtu.be/N_wc1mEvQP4

---

## Repository Contents

- `kernel.cu` — main DLL implementation (CUDA + cuBLAS + cuRAND).
- `MQL5/Indicators/LSTMTrendStart.mq5` — sample MT5 indicator.
- `docs/index.html`, `docs/app.js`, `docs/lstm-flow.svg` — interactive documentation assets.
- `MQL5GPULibrary_LSTM.sln`, `MQL5GPULibrary_LSTM.vcxproj` — Visual Studio build configuration.
- `unit_tests/` — C++ executable API unit test proposal for the DLL exports.

---


## Pattern Completion Demo (MQL5)

A new indicator example is available at `MQL5/Indicators/Examples/LSTM_PatternCompletion_Demo.mq5`.

It demonstrates a pattern-completion workflow:

- Candle windows are converted into symbolic features (body, upper wick, lower wick, optional direction flag).
- A training dataset is built from sliding windows over recent chart history (`SEQ_LEN` input, `PRED_K` future bars).
- Targets are regression values in `[0..1]`: `bullish_score` and `bearish_score = 1 - bullish_score`.
- The indicator runs short asynchronous training rounds with the existing `DN_*` DLL API and then performs inference on the latest window.
- Two lines are plotted in a separate indicator window: BullishScore and BearishScore.

This is a demonstration of the data flow and DLL integration, not a trading system.

---
## Runtime Profile (from `kernel.cu`)

The runtime currently provides:

- **Asynchronous training** on a dedicated background worker.
- **Lock-free telemetry** suitable for frequent UI polling (`OnTimer`).
- **Column-major matrix contract** for all GEMM paths (cuBLAS-native).
- **Persistent GPU buffers** for loaded dataset tensors.
- **Handle-based model management** for multiple models in one process.

---

## Core Features

1. **Multiple independent models** via handles (`DN_Create` / `DN_Free`).
2. **Configurable sequence and mini-batch sizes** (`DN_SetSequenceLength`, `DN_SetMiniBatchSize`).
3. **Stacked LSTM architecture with output projection** (`DN_AddLayerEx`, `DN_SetOutputDim`).
4. **Layer-level dropout** during training.
5. **Adam-like optimization** with gradient clipping and LR scheduling.
6. **Asynchronous training state machine** (`TS_IDLE`, `TS_TRAINING`, `TS_COMPLETED`, `TS_ERROR`).
7. **Weight checkpointing** (`DN_SnapshotWeights`, `DN_RestoreWeights`).
8. **Text-based model serialization** (`DN_SaveState`, `DN_GetState`, `DN_LoadState`) with `LSTM_V1` header.
9. **Real-time progress metrics** through lock-free getters (`DN_GetProgress*`, `DN_GetProgressAll`).
10. **Centralized error channel** (`DN_GetError(short*...)`).

---

## Mathematical Contract and Memory Layout

The entire runtime uses **column-major layout**.

### LSTM layer weights

- `W` shape: `[input_size + hidden_size, 4 * hidden_size]`
- Forward pass: `gates = W^T * hx`
- Backward pass:
  - `dW += hx * dg^T`
  - `dhx = W * dg`

### Output projection weights

- `W_out` shape: `[hidden_last, out_dim]`
- Forward pass: `Y = W_out^T * h_last`
- Backward pass:
  - `dW_out = h_last * dY^T`
  - `dh_last = W_out * dY`

If dimensions do not match the expected contract, the API call fails and detailed diagnostics can be retrieved via `DN_GetError`.

---

## DLL API Reference

All exports use `__stdcall` and return `MQL_BOOL` where applicable (`1 = success`, `0 = fail`).

---

### 1) Lifecycle

#### `int DN_Create();`
Creates a new model instance and returns its handle.

**Example**
```cpp
int h = DN_Create();
if(h <= 0) {
  // Handle creation failed
}
```

#### `void DN_Free(int h);`
Releases all CPU/GPU resources associated with a handle.

**Example**
```cpp
DN_Free(h);
h = 0;
```

---

### 2) Model Configuration

#### `MQL_BOOL DN_SetSequenceLength(int h, int seq_len);`
Sets sequence length used to interpret flattened input vectors.

- Must be called before loading training/prediction batches.
- `seq_len` must be positive.

**Example**
```cpp
if(!DN_SetSequenceLength(h, 32)) {
  // Read message via DN_GetError
}
```

#### `MQL_BOOL DN_SetMiniBatchSize(int h, int mbs);`
Defines mini-batch size used by the training runtime.

**Example**
```cpp
DN_SetMiniBatchSize(h, 64);
```

#### `MQL_BOOL DN_AddLayerEx(int h, int in, int out, int act, int ln, double drop);`
Adds one LSTM layer.

- `in`: input dimension to this layer.
- `out`: hidden dimension (layer output size).
- `act`: activation configuration (implementation-defined).
- `ln`: layer-normalization toggle/config flag (implementation-defined).
- `drop`: dropout rate in `[0,1)` for training.

**Example**
```cpp
DN_AddLayerEx(h, 8, 64, 0, 0, 0.10); // First LSTM layer
DN_AddLayerEx(h, 64, 64, 0, 0, 0.10); // Stacked layer
```

#### `MQL_BOOL DN_SetGradClip(int h, double clip);`
Sets gradient clipping threshold for training stability.

**Example**
```cpp
DN_SetGradClip(h, 1.0);
```

#### `MQL_BOOL DN_SetOutputDim(int h, int out_dim);`
Sets final projection dimension (model output width).

**Example**
```cpp
DN_SetOutputDim(h, 1); // e.g., single-value regression
```

---

### 3) Data Loading and Inference

#### `MQL_BOOL DN_LoadBatch(int h, const double* X, const double* T, int batch, int in, int out, int l);`
Uploads one training dataset batch to GPU buffers.

- `X` size must be `batch * in`.
- `T` size must be `batch * out`.
- `in` must satisfy `in % seq_len == 0`.
- `l` corresponds to sequence length for validation path compatibility.

**Example**
```cpp
// batch=128, seq_len=32, feature_dim=8 => in=256
DN_LoadBatch(h, X, T, 128, 256, 1, 32);
```

#### `MQL_BOOL DN_PredictBatch(int h, const double* X, int batch, int in, int l, double* Y);`
Runs batched forward inference.

- `Y` output size must be `batch * model_out_dim`.

**Example**
```cpp
DN_PredictBatch(h, X_live, 32, 256, 32, Y_pred);
```

---

### 4) Checkpointing

#### `MQL_BOOL DN_SnapshotWeights(int h);`
Stores a restorable weight snapshot (in-memory checkpoint).

#### `MQL_BOOL DN_RestoreWeights(int h);`
Restores the last snapshot created by `DN_SnapshotWeights`.

**Example**
```cpp
DN_SnapshotWeights(h);
// ... run training experiment
if(validation_worse)
  DN_RestoreWeights(h);
```

---

### 5) Asynchronous Training

#### `MQL_BOOL DN_TrainAsync(int h, int epochs, double target_mse, double lr, double wd);`
Starts asynchronous training and returns immediately.

- `epochs`: maximum epochs.
- `target_mse`: optional early-stop threshold.
- `lr`: initial learning rate.
- `wd`: weight decay.

**Example**
```cpp
DN_TrainAsync(h, 200, 1e-4, 1e-3, 1e-5);
```

#### `int DN_GetTrainingStatus(int h);`
Returns current training state (`0`, `1`, `2`, `-1`).

#### `void DN_GetTrainingResult(int h, double* out_mse, int* out_epochs);`
Returns final training metrics after completion.

#### `void DN_StopTraining(int h);`
Requests graceful asynchronous stop.

**Example**
```cpp
while(DN_GetTrainingStatus(h) == 1) {
  Sleep(100);
}
double final_mse = 0.0;
int epochs_done = 0;
DN_GetTrainingResult(h, &final_mse, &epochs_done);
```

---

### 6) Progress Reporting (v1.4.0)

Lock-free telemetry getters for frequent polling:

- `int DN_GetProgressEpoch(int h);`
- `int DN_GetProgressTotalEpochs(int h);`
- `int DN_GetProgressMiniBatch(int h);`
- `int DN_GetProgressTotalMiniBatches(int h);`
- `double DN_GetProgressLR(int h);`
- `double DN_GetProgressMSE(int h);`
- `double DN_GetProgressBestMSE(int h);`
- `double DN_GetProgressGradNorm(int h);`
- `int DN_GetProgressTotalSteps(int h);`
- `double DN_GetProgressPercent(int h);`
- `double DN_GetProgressElapsedSec(int h);`
- `double DN_GetProgressETASec(int h);`
- `MQL_BOOL DN_GetProgressAll(...);`

`DN_GetProgressAll()` is the recommended primary polling API for modern indicators because it minimizes call overhead and synchronization pressure.

**Example (conceptual)**
```cpp
// Prefer one bulk call in OnTimer for UI refresh
DN_GetProgressAll(h, &epoch, &epochs, &mb, &mb_total,
                  &mse, &best_mse, &lr, &grad_norm,
                  &percent, &elapsed, &eta);
```

---

### 7) Diagnostics

#### `int DN_GetLayerCount(int h);`
Returns number of configured LSTM layers.

#### `double DN_GetLayerWeightNorm(int h, int l);`
Returns norm of layer weights (`l` = layer index).

#### `double DN_GetGradNorm(int h);`
Returns current gradient norm.

#### `void DN_GetError(short* buf, int len);`
Reads latest error message into caller-provided buffer.

**Example**
```cpp
short err[512];
DN_GetError(err, 512);
```

---

### 8) Serialization

#### `int DN_SaveState(int h);`
Serializes current model state internally and returns payload length.

#### `MQL_BOOL DN_GetState(int h, char* buf, int max_len);`
Copies serialized state to caller buffer.

#### `MQL_BOOL DN_LoadState(int h, const char* buf);`
Loads serialized state (expects valid `LSTM_V1` payload).

**Example**
```cpp
int n = DN_SaveState(h);
char* blob = new char[n + 1];
if(DN_GetState(h, blob, n + 1)) {
  // Persist blob to file / database
}
// Later:
DN_LoadState(h2, blob);
```

---

## Training State Machine

- `0` → `TS_IDLE`
- `1` → `TS_TRAINING`
- `2` → `TS_COMPLETED`
- `-1` → `TS_ERROR`

---

## Recommended MT5 Workflow

1. `DN_Create`
2. `DN_SetSequenceLength`, `DN_SetMiniBatchSize`
3. `DN_AddLayerEx` (all LSTM layers)
4. `DN_SetOutputDim`
5. `DN_LoadBatch`
6. Optional `DN_SnapshotWeights`
7. `DN_TrainAsync`
8. Poll `DN_GetTrainingStatus` + `DN_GetProgressAll`
9. `DN_GetTrainingResult`
10. `DN_PredictBatch`
11. `DN_Free`

---

## Data Shape Rules

- `in = seq_len * feature_dim`
- `X` length = `batch * in`
- `T` length = `batch * out`
- `Y` length = `batch * model_out_dim`

Required constraint:

- `in % seq_len == 0`

---

## Build and Deployment

### Build

1. Open `MQL5GPULibrary_LSTM.sln` in Visual Studio.
2. Ensure CUDA Toolkit is installed.
3. Build with configuration `Release | x64`.
4. Copy the resulting DLL to `MQL5\Libraries` in your MT5 data directory.

### MT5 Deployment

1. Place DLL into `MQL5\Libraries`.
2. Place `LSTMTrendStart.mq5` into `MQL5\Indicators`.
3. Compile in MetaEditor.
4. Enable DLL imports in MT5 settings.

---

## Interactive Documentation

Open `docs/index.html` in your browser.

---

## Official Links

- https://remind.cz/DLL/MQL5GPULibrary_LSTM.html
- https://remind.cz/

- <img width="2560" height="1080" alt="image" src="https://github.com/user-attachments/assets/3a7391f5-f7e0-4e15-b1f3-2e05890276fd" />
<img width="374" height="423" alt="image" src="https://github.com/user-attachments/assets/a3c5c639-3f5d-48c7-a5bf-264a75878d54" />
<img width="361" height="363" alt="image" src="https://github.com/user-attachments/assets/9492f727-f41f-4ed3-b9bd-fc82c8c0a96a" />
<img width="382" height="348" alt="image" src="https://github.com/user-attachments/assets/4f69ab82-1751-4b96-ae96-7bb1156e70f1" />





---

## License

MIT (`LICENSE.txt`).
