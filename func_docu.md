# Implementation guide

For the public interface and migration rules, see [docs/API.md](docs/API.md). This guide maps the implementation without duplicating its signatures.

## Data path

MQL5 provides sample-major double arrays. LoadBatch copies and converts them into float32 buffers. Training gathers a shuffled minibatch and transposes its sequence into timestep-major storage. PredictBatch performs the same layout conversion without shuffling. Within a timestep, each sample is a column in a cuBLAS matrix.

The recurrent stack evaluates each timestep from a zero initial state. LSTM maintains cell and hidden caches; GRU stores update/reset/candidate values; RNN uses tanh. Layer outputs are gathered for the next layer. Only the final timestep feeds the linear output head. Training applies output dropout and backpropagates through every timestep and layer.

## Components

- `GPUMemory<T>` owns CUDA allocations and preserves an existing allocation when growth fails. Counts are validated before multiplication into bytes.
- `GPUContext` owns a nonblocking stream, cuBLAS handle and cuRAND generator. Runtime zeroing and kernels share a stream.
- `RecurrentLayer` defines shared weights/caches and output access, including dropout. Concrete implementations provide forward, backward, update and snapshot operations.
- `SequenceModel` owns architecture, batches, optimizer counters, worker and atomic telemetry. The model mutex serializes GPU mutations; the registry mutex only protects handle lookup/removal.
- `include/model_validation.h` checks dimensions and numeric parameters using bounded integer arithmetic.
- `include/checkpoint.h` parses text into CPU values before GPU allocation. The model commits replacement layers only after successful uploads.
- `include/MQL5GPULibrary.h` is compiled against the exports; `tools/check_api.py` compares the C and MQL declarations.

## Matrix conventions

Weights are column-major. A recurrent gate matrix has shape `[input+hidden, gates]`, where concatenated input order is **hidden then input**. LSTM gates are `[forget,input,candidate,output]`; GRU update/reset gates are `[update,reset]`, followed by separate input and hidden candidate matrices. RNN has one tanh projection. Output weights are `[last_hidden,output]`.

Forward uses `W^T * input`; weight gradients use `input * output_gradient^T`; input gradients use `W * output_gradient`. GRU gate gradients must interleave update/reset blocks per sample, and input gradients must include the separate candidate projection. Tests compare the CPU recurrence and numerical gradient norm against the DLL to guard these conventions.

## Optimization and observability

MSE gradients average across all minibatch output elements. Global clipping includes all trainable tensors. AdamW updates follow the per-call warmup/cosine schedule; Adam moments persist while weights persist. Checkpoint load and snapshot restore reset optimizer moments. Final MSE is measured from final weights, including after early cancellation. Errors and non-finite results propagate to the DLL return value/training status.

GPU inference/training checks live in `tests/test_dll.py`; portable dimension/parser checks live in `tests/validation_tests.cpp`. See [BUILDING](docs/BUILDING.md) for exact commands and the separate MetaEditor validation step.
