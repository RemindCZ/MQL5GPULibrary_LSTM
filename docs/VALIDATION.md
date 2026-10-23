# Local validation record

Environment: Windows x64, NVIDIA GeForce RTX 3060 (compute capability 8.6), CUDA Toolkit 13.0.48, MSVC 19.44.35228, CMake 3.30.0-rc4, Python 3.14.7.

| Check | Result |
| --- | --- |
| CMake Release CUDA DLL build | Passed |
| Original Visual Studio solution, Debug x64 DLL build | Passed |
| CTest `validation` | Passed: dimensions, overflow, parser rejection and legacy formats |
| CTest `api_contract` | Passed: all 38 C/CUDA/MQL signatures agree |
| CTest `gpu_api` | Passed: 10 integration tests |
| NVIDIA Compute Sanitizer memcheck | Passed: 10 tests, **0 errors** |
| `git diff --check` | Passed |
| MetaEditor compilation / live MT5 integration | Not run; MetaEditor was not available in the checked installation directories |

The GPU tests include independent scalar reference recurrences for LSTM/GRU/RNN, finite-difference gradient norms, a stacked RNN-to-GRU gradient case, dropout and repeated training, exact float32 checkpoint round trips, malformed-state preservation, invalid tensors, snapshots, final MSE and stop/free behavior.

Commands are in [BUILDING.md](BUILDING.md). CUDA 13 warns that its `shared` runtime option is deprecated in favor of `hybrid`; the build retains the shared setting for compatibility with existing CMake/Visual Studio CUDA integration. The original Debug project also reports existing size_t-to-int conversion warnings in recurrent kernels; entry-point tensor validation bounds those counts.

These checks exercise small deterministic fixtures and runtime contracts. They are not a long-duration performance benchmark, full training-quality evaluation, or verification of trading strategy behavior. GPU workflow availability depends on the repository owner configuring the documented self-hosted runner.
