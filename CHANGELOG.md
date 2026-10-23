# Changelog

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
