# Unit testy pro `kernel.cu`

Tato slozka obsahuje navrh **spustitelnych C++ unit testu** nad exportovanou DLL API (`DN_*`).

## Co testy pokryvaji

- Osetreni neplatneho handle (`DN_SetSequenceLength`, `DN_AddLayerEx`, `DN_SaveState`, ...).
- Minimalni pipeline `Create -> AddLayer -> LoadBatch -> PredictBatch`.
- Serializaci stavu (`DN_SaveState` / `DN_GetState` / `DN_LoadState`) vcetne negativniho testu na poskozeny vstup.

## Build (Windows)

```powershell
cd unit_tests
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

## Spusteni

```powershell
.\build\Release\kernel_api_tests.exe ..\x64\Release\MQL5GPULibrary_LSTM.dll
```

Pokud cestu nezadas, test runner se pokusi nacist `MQL5GPULibrary_LSTM.dll` z aktualni slozky.

> Poznamka: testy vyzaduji dostupne CUDA runtime prostredi a GPU kompatibilni se stavajicim DLL.
