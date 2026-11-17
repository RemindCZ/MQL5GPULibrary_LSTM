# Changelog

## 3.3.0

- Build branched LSTM, GRU, RNN, Dense and Identity networks in the Studio layer editor. Persist the graph, learning rate, batch size and backend with experiments and schedules.
- Train custom graphs on CPU or NVIDIA CUDA. The offline Windows release includes the CUDA training engine and its user-mode runtime libraries.
- Select the epoch on a chronological validation block, calibrate residual intervals and check the exported ONNX against the fitted model before making it available to MT5.

## 3.2.0

- Request aligned closed bars from a connected MT5 terminal through a read-only Expert Advisor. The browser controls symbols, timeframe, sample size and training parameters.
- Schedule native training independently of the browser. Restore schedules after restarting Studio and skip unchanged snapshots.
- Inspect held-out forecasts, calibrated intervals, model skill, trade returns and mark-to-market equity. Configure direction, signal threshold, position size and per-side costs.
- Export chronological holdout predictions through the native trainer and the optional DNFit_TrainTrace ABI.

## 3.1.0

- Fit a time-delay neural network directly from an MT5 chart, verify its ONNX graph and attach the resulting forecast.
- Train in a browser with DNForecast Studio on a Windows host, retain completed jobs and download models. The Windows client and training server run without Python.
- Include a native CSV trainer, training DLL, C header, import library and Czech instructions in the standalone distribution.

## 3.0.0

- Add a portable asynchronous forecasting ABI with bounded queues, per-session scheduling, terminal job states and explicit ownership.
- Add an MT5 closed-bar indicator, C and Python clients, ONNX CPU inference and a Windows adapter for CUDA 2.3 checkpoints.
- Add offline ridge, GRU and causal TCN training, calibrated residual bands, a forecast/outcome ledger and quote-based execution replay. Publish the trained research model, validation metrics and Czech deployment manual.

## 2.3.0 — unreleased

- Add an acyclic graph builder with LSTM/GRU/RNN/Dense/Identity nodes, arbitrary widths, concatenation, residual sums, dense activations and multi-output linear heads.
- Implement graph forward/backward propagation, shared-parent gradient accumulation, dropout and AdamW for pointwise layers, with independent numerical-gradient tests.
- Add version/shape introspection, 45 synchronized exports and validated V4 graph checkpoints while preserving V1–V3 readers and the legacy sequential ABI.
- Add the cross-asset MT5 forecast indicator: closed-bar alignment, training-only normalization, purged holdout, persistence comparison, candidate promotion, future curves, residual bands and prospective CSV logging.
- Add a standard-library Python client and standalone CSV training/inference with saved preprocessing metadata.
- Compile the new indicator and smoke script in MetaEditor; add packaging/installation tools and a comprehensive Czech PDF manual.

## 2.2.0 — unreleased

### Correctness

- Implement shared recurrent output access so GRU and RNN are concrete, compilable layer types.
- Pack GRU update/reset gradients per sample and include the candidate projection in propagated input gradients.
- Apply dropout to GRU/RNN outputs as well as LSTM; order runtime buffer zeroing on the compute stream.
- Evaluate final MSE from final weights, propagate evaluation errors, detect non-finite gradients/loss and report the complete gradient norm.
- Restart the learning-rate schedule per training call while keeping Adam steps/moments consistent across data reloads. Reset Adam on snapshot restore.

### Runtime and persistence

- Validate tensor sizes, dropout, finite inputs, flags and training parameters before GPU use; reject incompatible feature dimensions without resetting weights.
- Release the global handle registry lock before acquiring a model lock. Reject busy operations, guard worker joins and make result fields atomic.
- Contain exceptions at DLL and worker boundaries; provide actionable validation diagnostics.
- Parse checkpoints on the CPU, validate counts/connectivity/values, and commit GPU replacements transactionally. Preserve models on load failure and serialize floats with round-trip precision.
- Retain V1 LSTM and V2 LSTM/RNN reading; reject incomplete legacy V2 GRU states explicitly.

### Repository

- Add CMake builds, portable tests, GPU integration/numerical regressions and CI workflows.
- Fix Visual Studio Debug to produce a DLL, align CUDA runtime dependencies and make toolkit/architecture configurable.
- Add a public C header and shared MQL include used by every example, plus a synthetic MT5 smoke script.
- Replace inaccurate API documentation with verified shape, progress, concurrency and checkpoint contracts; add Czech usage instructions.

ABI function signatures are retained. See [migration notes](docs/API.md#migration-from-earlier-sources) for stricter validation and busy behavior.
