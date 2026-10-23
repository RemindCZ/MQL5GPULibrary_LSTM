# Build and verification

## Windows CUDA DLL

Install Visual Studio 2022 with Desktop development with C++, Windows SDK and CUDA build integration. Use a CUDA toolkit compatible with the compiler and deployment driver. Install CMake 3.24+ and Python 3 for tests.

```powershell
cmake -S . -B build -A x64 -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --output-on-failure
```

Use the architecture supported by the deployment GPU; `86` is for the tested RTX 3060. Multiple architectures can be passed as a quoted semicolon-separated list. CUDA libraries are linked through CMake imported targets. Debug builds use `--config Debug`.

The original solution remains supported:

```powershell
msbuild MQL5GPULibrary_LSTM.sln /m /p:Configuration=Release /p:Platform=x64 /p:DN_CudaVersion=13.0
```

`CudaCodeGeneration` can override the Visual Studio default `compute_86,sm_86`. Debug now builds a DLL as well. C++17 is explicitly enabled for CUDA compilation. Do not mix a prebuilt DLL from another revision with these sources when validating a change.

The MT5 process must find CUDA runtime, cuBLAS and cuRAND runtime libraries from the installed toolkit. CUDA 13 can place runtime DLLs under `bin/x64`. Restart MT5 after changing its environment. Keep libraries from the same compatible toolkit installation.

## Portable tests without a GPU

```sh
cmake -S . -B build-cpu -DDN_BUILD_CUDA=OFF
cmake --build build-cpu --config Release
ctest --test-dir build-cpu -C Release --output-on-failure
```

These compile and execute the same dimension validation and checkpoint parser used by the DLL. Python checks all exported C/CUDA/MQL signatures and shared includes. They do not build or validate the CUDA kernels. GitHub Actions runs this configuration on Windows and Linux.

## GPU regression tests

The default CUDA CMake configuration registers `gpu_api` with CTest. A missing GPU/runtime fails this test rather than silently skipping it. It checks scalar CPU reference recurrences and finite-difference gradient norms, multi-output and stacked networks, exact checkpoint round trips, malformed-state rollback, validation, stop/free and concurrency behavior.

Standalone execution:

```powershell
python tests/test_dll.py build/Release/MQL5GPULibrary_LSTM.dll "$env:CUDA_PATH/bin"
```

Optional memory checking using NVIDIA Compute Sanitizer:

```powershell
& "$env:CUDA_PATH/bin/compute-sanitizer.bat" --tool memcheck --error-exitcode 1 python tests/test_dll.py build/Release/MQL5GPULibrary_LSTM.dll "$env:CUDA_PATH/bin"
```

`Windows GPU checks` is manual and requires an available self-hosted Windows x64 runner with label `cuda`, compiler, toolkit, Python and GPU. It does not automatically execute pull-request code on a private machine.

## MetaEditor verification

Install the shared include in the terminal data folder, compile all modified MQL5 examples in MetaEditor, then run `LSTM_SmokeTest` with DLL imports enabled. CTest validates the DLL and import contracts, not the MetaEditor compiler or live terminal behavior. Use synthetic/demo data for integration checks.
