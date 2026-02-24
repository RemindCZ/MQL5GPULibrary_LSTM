# MQL5GPULibrary_LSTM

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Open Source](https://img.shields.io/badge/Open%20Source-Yes-brightgreen.svg)

Open-source CUDA DLL pro MetaTrader 5, která implementuje vícevrstvou LSTM síť s asynchronním tréninkem, dropoutem, checkpointy, serializací stavu a **průběžným progress reportingem (v1.4.0)**.

## Co je v repozitáři

- `kernel.cu` – hlavní implementace DLL API (CUDA + cuBLAS + cuRAND).
- `MQL5/Indicators/LSTMTrendStart.mq5` – ukázkový indikátor pro MT5.
- `docs/index.html`, `docs/app.js`, `docs/lstm-flow.svg` – interaktivní dokumentace.
- `MQL5GPULibrary_LSTM.sln`, `MQL5GPULibrary_LSTM.vcxproj` – build konfigurace pro Visual Studio.

## Runtime profil (aktuálně dle `kernel.cu`)

Aktuální hlavička jádra deklaruje:

- **LSTM DLL v1.4.0**
- asynchronní trénink na background vlákně,
- lock-free telemetry průběhu tréninku,
- column-major kontrakt pro všechny GEMM operace,
- persistentní GPU buffery pro načtený dataset,
- handle-based správu více modelů v jednom procesu.

## Hlavní vlastnosti

1. **Více nezávislých modelů přes handly** (`DN_Create` / `DN_Free`).
2. **Konfigurovatelná sekvence a mini-batch** (`DN_SetSequenceLength`, `DN_SetMiniBatchSize`).
3. **Multi-layer LSTM + output projekce** (`DN_AddLayerEx`, `DN_SetOutputDim`).
4. **Dropout v tréninku** na úrovni jednotlivých LSTM vrstev.
5. **Adam-like optimalizace** (gradient clipping + schedule s warmup/cosine decay).
6. **Asynchronní tréninkový stavový automat** (`TS_IDLE`, `TS_TRAINING`, `TS_COMPLETED`, `TS_ERROR`).
7. **Checkpointy vah** (`DN_SnapshotWeights`, `DN_RestoreWeights`).
8. **Textová serializace modelu** (`DN_SaveState`, `DN_GetState`, `DN_LoadState`) s hlavičkou `LSTM_V1`.
9. **Progress reporting v reálném čase** přes lock-free API (`DN_GetProgress*`, `DN_GetProgressAll`).
10. **Centrální chybový kanál** (`DN_GetError(short*...)`).

## Matematický kontrakt a layout paměti

Celý runtime používá **column-major** (nativní cuBLAS konvence).

- LSTM váhy: `W` má tvar `[input_size + hidden_size, 4*hidden_size]`.
  - forward: `gates = W^T * hx`
  - backward: `dW += hx * dg^T`, `dhx = W * dg`
- Output váhy: `W_out` má tvar `[hidden_last, out_dim]`.
  - forward: `Y = W_out^T * h_last`
  - backward: `dW_out = h_last * dY^T`, `dh_last = W_out * dY`

Při rozměrovém nesouladu API vrací failure a detail je dostupný přes `DN_GetError`.

## DLL API reference

Všechny exporty jsou `__stdcall`, návratový typ `MQL_BOOL`: `1 = success`, `0 = fail`.

### 1) Životní cyklus

- `int DN_Create();`
- `void DN_Free(int h);`

### 2) Konfigurace modelu

- `MQL_BOOL DN_SetSequenceLength(int h, int seq_len);`
- `MQL_BOOL DN_SetMiniBatchSize(int h, int mbs);`
- `MQL_BOOL DN_AddLayerEx(int h, int in, int out, int act, int ln, double drop);`
- `MQL_BOOL DN_SetGradClip(int h, double clip);`
- `MQL_BOOL DN_SetOutputDim(int h, int out_dim);`

### 3) Data a inference

- `MQL_BOOL DN_LoadBatch(int h, const double* X, const double* T, int batch, int in, int out, int l);`
- `MQL_BOOL DN_PredictBatch(int h, const double* X, int batch, int in, int l, double* Y);`

### 4) Checkpointing

- `MQL_BOOL DN_SnapshotWeights(int h);`
- `MQL_BOOL DN_RestoreWeights(int h);`

### 5) Asynchronní trénink

- `MQL_BOOL DN_TrainAsync(int h, int epochs, double target_mse, double lr, double wd);`
- `int DN_GetTrainingStatus(int h);`
- `void DN_GetTrainingResult(int h, double* out_mse, int* out_epochs);`
- `void DN_StopTraining(int h);`

### 6) Progress reporting (v1.4.0)

Lock-free telemetry (vhodná pro polling v `OnTimer`):

- `int DN_GetProgressEpoch(int h);`
- `int DN_GetProgressTotalEpochs(int h);`
- `int DN_GetProgressMiniBatch(int h);`
- `int DN_GetProgressTotalMiniBatches(int h);`
- `double DN_GetProgressLR(int h);`
- `double DN_GetProgressMSE(int h);`
- `double DN_GetProgressBestMSE(int h);`
- `double DN_GetProgressGradNorm(int h);`
- `int DN_GetProgressTotalSteps(int h);`
- `double DN_GetProgressPercent(int h);`
- `double DN_GetProgressElapsedSec(int h);`
- `double DN_GetProgressETASec(int h);`
- `MQL_BOOL DN_GetProgressAll(...);` – bulk getter všech hlavních metrik jedním voláním.

### 7) Diagnostika

- `int DN_GetLayerCount(int h);`
- `double DN_GetLayerWeightNorm(int h, int l);`
- `double DN_GetGradNorm(int h);`
- `void DN_GetError(short* buf, int len);`

### 8) Serializace

- `int DN_SaveState(int h);`
- `MQL_BOOL DN_GetState(int h, char* buf, int max_len);`
- `MQL_BOOL DN_LoadState(int h, const char* buf);`

## Stavový automat tréninku

- `0` = `TS_IDLE`
- `1` = `TS_TRAINING`
- `2` = `TS_COMPLETED`
- `-1` = `TS_ERROR`

## Doporučený workflow v MT5

1. `DN_Create`
2. `DN_SetSequenceLength`, `DN_SetMiniBatchSize`
3. `DN_AddLayerEx` (všechny LSTM vrstvy)
4. `DN_SetOutputDim`
5. `DN_LoadBatch`
6. volitelně `DN_SnapshotWeights`
7. `DN_TrainAsync`
8. poll `DN_GetTrainingStatus` + `DN_GetProgressAll`
9. `DN_GetTrainingResult`
10. `DN_PredictBatch`
11. `DN_Free`

## Pravidla tvarů dat

- `in = seq_len * feature_dim`
- `X` má velikost `batch * in`
- `T` má velikost `batch * out`
- `Y` má velikost `batch * out_dim_modelu`

`in % seq_len == 0` je povinná podmínka pro load/predict cestu.

## Build a nasazení

### Build

1. Otevři `MQL5GPULibrary_LSTM.sln` ve Visual Studiu.
2. Ujisti se, že je nainstalovaný CUDA Toolkit.
3. Build konfigurace: `Release | x64`.
4. Zkopíruj DLL do MT5 `MQL5\Libraries`.

### MT5 deployment

1. DLL do `MQL5\Libraries`
2. `LSTMTrendStart.mq5` do `MQL5\Indicators`
3. Kompilace v MetaEditoru
4. Povolit DLL importy v MT5

## Interaktivní dokumentace

Otevři `docs/index.html` v prohlížeči.

## Oficiální odkazy

- https://remind.cz/DLL/MQL5GPULibrary_LSTM.html
- https://remind.cz/

## License

MIT (`LICENSE.txt`).
