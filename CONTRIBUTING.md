# Contributing

Keep exported `DN_*` signatures compatible with both the C header and shared MQL include. When changing a contract, update `docs/API.md` and the changelog. Do not commit generated binaries, local model states or trading/account data.

Before submitting changes:

1. Run the portable CMake/CTest configuration from [BUILDING](docs/BUILDING.md).
2. For CUDA/runtime changes, build the DLL and run the GPU tests. Add a regression that exercises the observed failure; numerical changes should use independent reference equations where possible.
3. For MQL changes, compile affected sources in MetaEditor and describe that validation separately from CTest.
4. Record toolchain, GPU and commands in the PR. State explicitly which checks could not run.

Use bounded shape arithmetic before allocations/kernel launches. Preserve live model state on checkpoint parse failures. Avoid holding the handle registry mutex during GPU work, model locking or thread joins. Keep runtime operations on their model's CUDA stream.

CI portable checks run on Windows/Linux without a GPU. The manual GPU workflow requires a configured Windows runner with the `cuda` label; do not expose a self-hosted runner to untrusted automatic pull-request execution.
