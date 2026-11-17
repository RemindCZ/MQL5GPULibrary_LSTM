# DNForecast 3.0 verification

Windows x64, Xeon E5-2680 v4, MSVC 19.44 and ONNX Runtime CPU 1.23.2. The combined build passed the C executable, both ABI contracts, native validation, financial dataset tests, asynchronous runtime, CUDA tests and version 2.3 checkpoint parity. The standalone reference build passed all five CTest targets without an ONNX SDK or CUDA. The optional ONNX authoring test separately exercised all six fields of a mixed per-horizon policy.

MetaEditor compiled both indicators and both smoke scripts without errors or warnings. The isolated MT5 demo terminal passed the reference and ONNX smoke tests. On the AUDUSD H1 demo chart, all four markets supplied 97 closed prices. The Python client and the sixth-horizon indicator buffer both produced 0.6738341392575763 from an anchor of 0.67379. The chart returned lower and upper prices of 0.6726279116985494 and 0.674967693327198. This was a live inference and chart integration test with order placement disabled.

The October execution sample and its limitations are documented in [the report](research/ICTrading-2026-10.json). The broker's numeric bar and tick timestamps were used together. In the live observation the broker clock ran approximately two hours ahead of Unix wall-clock UTC. Historical close publication times were assumed. An independently verified historical UTC mapping across daylight-saving transitions, eight-hour model-swap soak and ONNX CUDA provider parity remain unverified.

## Version 2.3 validation record

Environment: Windows x64, NVIDIA GeForce RTX 3060 (compute capability 8.6), CUDA Toolkit 13.0.48, MSVC 19.44.35228, CMake 3.30.0-rc4, Python 3.14.7.

| Check | Result |
| --- | --- |
| CMake Release CUDA DLL build | Passed |
| Original Visual Studio solution, Debug x64 DLL build | Passed for the preceding runtime-fix revision; the graph release was verified with CMake Release |
| CTest `validation` | Passed: dimensions, overflow, parser rejection and legacy formats |
| CTest `api_contract` | Passed: all 45 C/CUDA/MQL signatures agree |
| CTest `financial_series` | Passed: chronological purge, train-only scaling, persistence scoring and invalid input rejection |
| CTest `gpu_api` | Passed: 14 integration tests |
| NVIDIA Compute Sanitizer memcheck | Passed: 14 tests, **0 errors**, child processes included |
| `git diff --check` | Passed |
| IC Trading MetaEditor: cross-asset indicator and smoke script | Both compiled to EX5: **0 errors, 0 warnings** |
| Standalone CSV training, save and reload | Passed on explicitly synthetic data; restored forecast prices match exactly |
| Package installation in isolated mock terminal data folder | Passed: all 7 MQL5 files match, implicit overwrite refused, explicit upgrade accepted |
| Czech PDF | 21 pages rendered and visually reviewed; all 45 export declarations included |
| Live MT5 chart / broker-market performance | Not run; no claim of predictive skill or profitability on IC Trading history |

The GPU tests include independent scalar reference recurrences for LSTM/GRU/RNN, finite-difference gradient norms, a stacked RNN-to-GRU gradient case, a recurrent fork (RNN feeding GRU and LSTM) with shared-parent gradient accumulation and sigmoid projection, Dense tanh/ReLU branches, Identity sum, concat including raw input, dropout and repeated training, exact V3/V4 float32 checkpoint round trips, malformed-state preservation, invalid tensors, snapshots, final MSE and stop/free behavior. The standalone test invokes both CLI scripts with paths containing spaces, training-only metadata and checkpoint reload.

Commands are in [BUILDING.md](BUILDING.md). CUDA 13 warns that its `shared` runtime option is deprecated in favor of `hybrid`; the build retains the shared setting for compatibility with existing CMake/Visual Studio CUDA integration. The original Debug project also reports existing size_t-to-int conversion warnings in recurrent kernels; entry-point tensor validation bounds those counts.

These checks exercise small deterministic fixtures and runtime contracts. They are not a long-duration performance benchmark, full training-quality evaluation, or verification of trading strategy behavior. GPU workflow availability depends on the repository owner configuring the documented self-hosted runner.
