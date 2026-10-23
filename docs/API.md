# DLL API contract

The authoritative declarations are [MQL5GPULibrary.h](../include/MQL5GPULibrary.h) and [MQL5GPULibrary.mqh](../MQL5/Include/MQL5GPULibrary.mqh), with 45 exports in version 2.3.0. ABI: x64 Windows, C linkage, `__stdcall`, `int` booleans (1 success, 0 failure), double buffers at the boundary, float32 on the GPU. Handle, node-ID, size, status and telemetry functions have the distinct return values documented below.

## Graph API (2.3.0)

`DN_GetVersion()` returns `20300`. `DN_BeginGraph(h, features, sequence)` clears the previous architecture/data and starts a graph with fixed input width. Add nodes in topological order using `DN_AddGraphNode(h, type, width, activation, dropout, parents, parent_count, merge)`.

- Public types: **0=LSTM, 1=GRU, 2=RNN, 3=Dense, 4=Identity**.
- Activation: Dense supports **0=linear, 1=tanh, 2=ReLU, 3=sigmoid**. Other types require 0; their internal recurrent activations are unchanged.
- Parents: external input **-1**, or IDs of earlier nodes; caller supplies `parent_count` valid integers (1–64).
- Merge: **0=concatenate** in parent order; **1=sum**, requiring equal parent widths.
- Identity: width 0 infers merged width; any explicit width must match. Other widths must be positive.
- Dropout: finite `0 <= dropout < 1`, including after Dense/Identity outputs.
- Return: **node ID >= 0 on success, -1 on failure**. Zero is a successful first node.

`DN_SetGraphOutput(h, node, outputs)` creates a linear head from the last timestep of the selected node. Every created node must contribute to that output. Cycles, forward references, disconnected branches and incompatible sums are rejected. The limit is 1024 nodes, plus tensor-index and available-memory limits. `DN_GetInputDim`, `DN_GetOutputDim`, and `DN_GetSequenceLength` return the configured dimensions (zero on error; input/head may be zero before configuration).

Dense/Identity operate per timestep, not across time. Use a recurrent node or engineered lag features to incorporate temporal history. `DN_SetOutputDim` selects the final graph node; sequential AddLayer calls are rejected in graph mode. Building a topology is not an all-or-nothing replacement transaction: configure a separate handle to retain an active model on a builder failure. The MQL/Python helper syntax and examples are in [the manual](MANUAL.cs.md).

## Ownership and concurrency

`DN_Create` returns a positive handle or zero. Release it with `DN_Free`; repeated release is harmless. Free removes the handle from the registry, requests cancellation, joins the worker and waits for any active model operation. Free is blocking and must complete before unloading the DLL.

Operations requiring the model lock fail promptly when the model is in use, closed, or training. Different handles do not hold the global registry lock while doing GPU work. Status, progress, result polling and cancellation take only the short registry lookup plus atomic reads/writes. Serialize application operations on a given handle, including the SaveState/GetState pair.

Training states: `0` idle, `1` training, `2` completed (including a requested stop), `-1` failed or invalid handle. Cancellation is checked between minibatches. The final loss is evaluated on the current weights before completion is reported. `DN_StopTraining` is a request; wait for status to change or call `DN_Free`. Read final results after status is 2 or -1. While training, the result fields are race-free but are not a coherent completed result.

C++ exceptions are caught at the exported boundary and in the worker. Failure does not guarantee rollback of a partially completed training run. `DN_GetError(short*, int)` writes a terminated UTF-16 message, truncating to `len-1`; the last diagnostic is global and may be replaced by another handle's error.

## Configuration

| Function | Contract |
| --- | --- |
| `DN_SetSequenceLength` | Positive timestep count. A change invalidates loaded training data. |
| `DN_SetMiniBatchSize` | Positive sample count. Tensor products are checked before use. |
| `DN_AddLayerEx` | `in>=0`, `out>0`, `0<=drop<1`; `act`: 0=LSTM, 1=GRU, 2=RNN; `ln=0`. |
| `DN_AddGRULayer`, `DN_AddRNNLayer` | Same dimensions/dropout rules. |
| `DN_SetOutputDim` | Positive linear-head width; recurrent layers must already exist. Replaces the head and invalidates loaded data. |
| `DN_SetGradClip` | Finite nonnegative float32-representable threshold; zero disables clipping. |

An input width of zero binds lazily on the first valid data load or prediction. Explicit widths must match the input feature count or preceding hidden width. Appending a recurrent layer invalidates the output head and loaded batch; configure the output again. Incompatible input data is rejected, rather than silently reinitializing learned weights. At most 1024 recurrent layers are accepted.

Dropout applies to outputs between layers and to the output head input during training, not to the recurrent hidden state. Inference disables dropout. Each prediction starts with zero hidden/cell states; state is not carried across calls. GRU uses `h=(1-z)*h_previous+z*candidate` and a reset gate applied after the hidden candidate projection. This convention differs from some other libraries.

## Tensor layout and validation

Only `layout=0` is supported:

- `X[s,t,f]` is at `X[(s*seq_len+t)*features+f]`.
- Pass `in=seq_len*features`, `batch>0`, `out>0`.
- Input length is `batch*in`; target length is `batch*out`.
- Prediction output length is `batch*configured_output_dim`.
- Targets and predictions use `Y[s,o] = Y[s*out+o]`.

Inputs/targets must be finite and representable as float32. Dimensions are validated against signed kernel indexing limits before allocation. The legacy pointer ABI cannot inspect the actual array length: the caller must allocate every required element and supply valid pointers. `DN_LoadState` similarly requires a readable NUL-terminated buffer. Never pass untrusted raw pointers.

`DN_LoadBatch` copies data before returning. Training can then run independently of caller buffers. A validation failure leaves the existing data/model untouched. A later allocation/upload failure can invalidate the loaded batch; reload before training. Loading data with a different output width is rejected if a head is already configured.

## Training and telemetry

`DN_Train[Async](h, epochs, target_mse, lr, wd)` requires loaded data, positive epochs and learning rate, and finite nonnegative target MSE/weight decay. A zero MSE target disables early stopping. The scheduler restarts per training call; Adam moments and its bias-correction step continue across calls/data reloads. Loading a checkpoint or restoring weights resets Adam.

Full-dataset MSE is measured at epoch 1, every 50 epochs, the last scheduled epoch and final completion/cancellation. Early stopping is checked at those evaluation points. MSE failure or non-finite gradients/loss fails the training run. The final result corresponds to the returned weights, including cancellation between periodic evaluations.

`DN_GetProgressAll` arguments after the handle:

```text
epoch, total_epochs, minibatch, total_minibatches,
lr, mse, best_mse, grad_norm, percent, elapsed_sec, eta_sec
```

Outputs are independent atomic samples, not an indivisible snapshot. `grad_norm` is the global norm before clipping, including GRU candidate tensors. `total_steps` counts optimizer updates. Completion sets percent to 100 and ETA to zero, including an explicitly stopped run; the reported epochs show how far it reached. Poll on a timer, for example every 200 ms.

## Checkpoints and snapshots

1. Call `DN_SaveState(h)`; it returns the required byte count **including NUL**, or zero.
2. Allocate that many `char` bytes, then call `DN_GetState(h, buffer, size)`.
3. Store the ASCII text; pass a NUL-terminated copy to `DN_LoadState`.

New sequential saves use `SEQMODEL_V3`; graph saves use `SEQMODEL_V4`, adding external input width, node activations, merge modes, parent lists and output-node selection. Both use the classic locale and float32 round-trip precision. Checkpoints contain architecture, dropout and weights/biases, not optimizer moments, training data, RNG state or preprocessing parameters. Save normalization and feature ordering separately. V4 appends on-disk Dense=3 and Identity=4; V1–V3 compatibility is unchanged.

The parser validates version, positive dimensions, layer connectivity, tensor counts, finite weights and end-of-file before allocating GPU state. Loading builds replacement layers and commits only when every upload succeeds. Invalid or truncated state preserves the current model. Successful loading clears training data, snapshots, cached serialization, optimizer state and telemetry; call LoadBatch to train again.

Legacy `LSTM_V1` and LSTM/RNN `SEQMODEL_V2` are accepted. V2 GRU is rejected because it lacks the separated candidate tensors required by this implementation. On-disk IDs are **0=RNN, 1=GRU, 2=LSTM**; these historical IDs are intentionally different from `DN_AddLayerEx` selectors.

`DN_SnapshotWeights` / `DN_RestoreWeights` provide an in-memory weight rollback. Restore requires a complete snapshot and resets Adam moments. It is not a full training-session checkpoint. Snapshot and restore are unavailable during training.

## Migration from earlier sources

- Copy the new shared MQL include along with indicator sources.
- Nonzero layout/layer-normalization flags and invalid dimensions now fail explicitly.
- Handle busy failures by polling training status and retrying when idle; operations no longer wait for training while blocking all handles.
- Wrong input feature counts no longer silently regenerate the first layer.
- Correct the documented ProgressAll order to `lr, mse, best_mse`.
- Rebuild the DLL. An older binary can have different behavior despite identical function names.
