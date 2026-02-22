# MQL5GPULibrary_LSTM

MQL5GPULibrary_LSTM is a 64-bit CUDA-accelerated DLL for MetaTrader 5 that provides a multi-layer LSTM model with asynchronous training, dropout support, model checkpointing, and state serialization. The repository also includes an MT5 indicator example (`MQL5/Indicators/LSTMTrendStart.mq5`) that demonstrates practical usage for financial time series prediction.

## Project Scope

This repository contains:

- `kernel.cu`: the CUDA/C++ source implementing the DLL API and model runtime.
- `MQL5/Indicators/LSTMTrendStart.mq5`: a reference MT5 indicator using asynchronous training and prediction.
- `docs/index.html`, `docs/app.js`, `docs/lstm-flow.svg`: interactive visual documentation.
- Visual Studio solution/project files for building the DLL (`MQL5GPULibrary_LSTM.sln`, `.vcxproj`).

## Current Runtime Profile

The implementation in `kernel.cu` is currently labeled as LSTM DLL `v1.3.0` and is designed around:

- CUDA streams and cuBLAS/cuRAND integration.
- Column-major matrix layout for all GEMM operations.
- Multi-layer LSTM stack and a linear output projection layer.
- Persistent GPU buffers for loaded training data.
- Asynchronous training through a dedicated worker thread.
- Thread-safe handle-based model management.

## Core Features

### 1. Multi-handle lifecycle

You can create and manage multiple independent model instances through integer handles returned by `DN_Create`.

### 2. Configurable sequence and mini-batch policy

- Sequence length is configurable via `DN_SetSequenceLength`.
- Mini-batch size is configurable via `DN_SetMiniBatchSize`.

### 3. Layered architecture

- Add one or more LSTM layers with `DN_AddLayerEx`.
- Build or rebuild the output projection via `DN_SetOutputDim`.

### 4. Dropout-enabled training path

Dropout is supported per LSTM layer and applied in forward training mode, with explicit dropout mask usage in backward propagation.

### 5. Optimizer and schedule

Training internally uses Adam-style moments with:

- bias correction terms,
- gradient clipping,
- warmup phase,
- cosine learning-rate decay with floor.

### 6. Asynchronous training state machine

Training can run in a background thread while MQL5 polls status. State values:

- `0`: idle
- `1`: training
- `2`: completed
- `-1`: error

### 7. Snapshot and restore

`DN_SnapshotWeights` and `DN_RestoreWeights` allow temporary best-weight checkpointing without full serialization.

### 8. Text serialization

- `DN_SaveState` computes required serialized payload size.
- `DN_GetState` copies the serialized text into caller buffer.
- `DN_LoadState` restores model architecture and parameters.

Serialized models currently use the `LSTM_V1` textual header.

### 9. Unified error message channel

`DN_GetError` returns the latest error message through a `short*` buffer for logging on the MQL5 side.

## Mathematical and Memory Layout Contract

The code follows cuBLAS-native column-major layout for all matrices.

Important implications:

- LSTM gate weight matrix shape is treated as `[input_plus_hidden x 4*hidden]`.
- Output layer weight matrix shape is `[hidden_last x out_dim]`.
- Host-side arrays received from MQL5 are transformed to GPU float buffers.
- Input sequences are transposed to timestep-major GPU layout for recurrent processing.

If you provide incorrect dimensions, calls fail and details can be retrieved via `DN_GetError`.

## DLL API Reference

All exports are `__stdcall` and `MQL_BOOL` returns follow:

- `1` success
- `0` failure

### Handle management

#### `int DN_Create();`

Creates a model instance and returns a positive handle. Returns `0` on failure.

#### `void DN_Free(int h);`

Releases model resources. If asynchronous training is active, stop/join flow is handled before destruction.

### Configuration

#### `MQL_BOOL DN_SetSequenceLength(int h, int seq_len);`

Sets sequence length. Effective value is clamped to at least `1`.

#### `MQL_BOOL DN_SetMiniBatchSize(int h, int mbs);`

Sets mini-batch size. Effective value is clamped to at least `1`.

#### `MQL_BOOL DN_AddLayerEx(int h, int in, int out, int act, int ln, double drop);`

Adds an LSTM layer.

- `in`: expected input size (can be auto-bound later if needed).
- `out`: hidden size.
- `drop`: dropout probability for the layer.
- `act` and `ln`: currently compatibility parameters and not active in core math path.

#### `MQL_BOOL DN_SetGradClip(int h, double clip);`

Sets gradient clipping threshold used during updates.

#### `MQL_BOOL DN_SetOutputDim(int h, int out_dim);`

Creates or reinitializes output projection based on last LSTM hidden dimension.

### Data loading and prediction

#### `MQL_BOOL DN_LoadBatch(int h, const double* X, const double* T, int batch, int in, int out, int l);`

Loads the full training dataset into persistent GPU memory.

- `X` logical shape: `[batch x in]`
- `T` logical shape: `[batch x out]`
- `in` must be divisible by configured sequence length.
- Internally, feature dimension is inferred as `in / seq_len`.
- `l` is currently accepted for compatibility.

#### `MQL_BOOL DN_PredictBatch(int h, const double* X, int batch, int in, int l, double* Y);`

Runs inference.

- `X` shape: `[batch x in]`
- `Y` output shape: `[batch x out_dim_of_model]`
- `in % seq_len == 0` is required.
- `l` is currently compatibility-only.

### Asynchronous training

#### `MQL_BOOL DN_TrainAsync(int h, int epochs, double target_mse, double lr, double wd);`

Starts background training using the last successful `DN_LoadBatch` dataset.

Behavior notes:

- Returns immediately.
- Fails if training is already running.
- Uses shuffled mini-batches and internal schedule.
- Can stop early when `target_mse > 0` and reached during periodic full-dataset evaluation.

#### `int DN_GetTrainingStatus(int h);`

Returns async state (`0`, `1`, `2`, `-1`). Safe for polling loops.

#### `void DN_GetTrainingResult(int h, double* out_mse, int* out_epochs);`

Returns final measured full-train MSE and completed epoch count from the latest async run.

#### `void DN_StopTraining(int h);`

Sets cooperative stop flag. Caller should continue polling until non-running state.

### Diagnostics

#### `int DN_GetLayerCount(int h);`

Returns total active layer count (all LSTM layers plus output layer when present).

#### `double DN_GetLayerWeightNorm(int h, int l);`

Currently exposed as placeholder; currently returns `0.0`.

#### `double DN_GetGradNorm(int h);`

Currently exposed as placeholder; currently returns `0.0`.

### Checkpoint and serialization

#### `MQL_BOOL DN_SnapshotWeights(int h);`

Saves current in-memory weights as checkpoint snapshot.

#### `MQL_BOOL DN_RestoreWeights(int h);`

Restores previously snapshotted weights.

#### `int DN_SaveState(int h);`

Serializes model into internal text buffer and returns required byte size (including null terminator).

#### `MQL_BOOL DN_GetState(int h, char* buf, int max_len);`

Copies serialized state to caller buffer.

#### `MQL_BOOL DN_LoadState(int h, const char* buf);`

Loads model from serialized text format.

### Error retrieval

#### `void DN_GetError(short* buf, int len);`

Copies latest error message into a caller-provided short buffer.

## Training Workflow Recommendation

A robust MQL5 workflow:

1. Create handle with `DN_Create`.
2. Set sequence length and mini-batch size.
3. Add all LSTM layers.
4. Set output dimension.
5. Load training dataset (`DN_LoadBatch`).
6. Optionally snapshot baseline weights.
7. Start async training (`DN_TrainAsync`).
8. Poll `DN_GetTrainingStatus` in `OnTimer` or `OnCalculate`.
9. On completion, call `DN_GetTrainingResult`.
10. Run inference with `DN_PredictBatch`.
11. Free handle in deinitialization (`DN_Free`).

Always validate return values and read `DN_GetError` immediately on failure.

## Data Shape Rules

Let:

- `batch` = number of samples,
- `seq_len` = sequence length,
- `feature_dim` = features per timestep,
- `in = seq_len * feature_dim`,
- `out` = target dimension.

Required flat array sizes:

- `X`: `batch * in`
- `T`: `batch * out`
- `Y`: `batch * out` (for prediction output when model out-dim equals `out`)

Mismatch between these rules and configured model dimensions is a common source of errors.

## Threading and Safety Notes

- Each model instance contains its own synchronization primitives.
- Async worker acquires exclusive model lock during training.
- Status/result/stop interface is lock-free from caller perspective (atomic-based).
- `DN_Free` is designed to avoid leaving worker threads running.

## MetaTrader 5 Deployment

1. Copy `MQL5GPULibrary_LSTM.dll` to `MQL5\Libraries` in your MT5 Data Folder.
2. Copy `MQL5/Indicators/LSTMTrendStart.mq5` to `MQL5\Indicators`.
3. Compile indicator in MetaEditor.
4. Enable DLL imports in MT5 settings.
5. Attach indicator to chart and configure parameters.

## Build from Source

1. Open `MQL5GPULibrary_LSTM.sln` in Visual Studio.
2. Ensure CUDA Toolkit is installed and integrated with Visual Studio build customizations.
3. Select `Release | x64`.
4. Build the DLL and deploy it to MT5 Libraries folder.

## Interactive Documentation

Open `docs/index.html` in a web browser to inspect the included visual explanation of LSTM data and flow.

## Typical Failure Sources

- DLL imports disabled in MT5.
- Missing or incompatible CUDA runtime/toolkit components.
- Invalid dimensional arguments (`in`, `out`, `seq_len`, `batch`).
- GPU out-of-memory due to large batch, sequence, or model dimensions.
- Calling training before successful data load.

## MQL5 Error Retrieval Example

```mq5
short err[];
ArrayResize(err, 512);
DN_GetError(err, 512);
Print("DLL error: ", ShortArrayToString(err));
```

## License

MIT License. See `LICENSE.txt`.
