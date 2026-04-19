# MQL5GPULibrary_LSTM

CUDA DLL knihovna pro trénink a inferenci **LSTM** modelu z prostředí **MetaTrader 5 (MQL5)**.

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-brightgreen.svg)
![MT5](https://img.shields.io/badge/MetaTrader%205-DLL%20API-blue.svg)
![LSTM](https://github.com/RemindCZ/MQL5GPULibrary_LSTM/blob/master/LSTM.gif)

---

## Co je v nové verzi jádra (`kernel.cu`)

Aktuální `kernel.cu` implementuje čistě LSTM pipeline se zaměřením na:

- asynchronní trénink (`DN_TrainAsync`) s bezpečným předáním locku,
- lock-free progress telemetry (`DN_GetProgress*`, `DN_GetProgressAll`),
- robustnější chybové hlášení (`Network not found` vs `Network busy`),
- snapshot/restore vah,
- serializaci stavu modelu do textového bufferu (`DN_SaveState`, `DN_GetState`, `DN_LoadState`),
- interní optimalizace AdamW + gradient clipping.

> Pozn.: Aktuální exportované API v `kernel.cu` je LSTM-only. Historické reference na GRU/RNN už neplatí.

---

## Architektura

- **`kernel.cu`** – CUDA/C++ jádro, trénink, inference, serializace, DLL exporty.
- **`MQL5/Indicators/*.mq5`** – příklady integrace v MT5.
- **`MQL5GPULibrary_LSTM.sln` + `.vcxproj`** – build pro Visual Studio.
- **`docs/`** – statická dokumentace a vizualizace.

---

## Životní cyklus modelu (doporučený postup)

1. `DN_Create` – vytvoření handle.
2. Nastavení parametrů (`DN_SetSequenceLength`, `DN_SetMiniBatchSize`, vrstvy, output).
3. `DN_LoadBatch` – nahrání trénovacích dat.
4. Trénink (`DN_Train` nebo `DN_TrainAsync`).
5. Monitoring přes `DN_GetTrainingStatus` + `DN_GetProgressAll`.
6. Predikce přes `DN_PredictBatch`.
7. `DN_Free` – uvolnění handle.

---

## Exportované API (aktuální)

### 1) Handle a lifecycle

```cpp
int DN_Create();
void DN_Free(int h);
```

- `DN_Create()` vrací `>0` při úspěchu, jinak `0`.
- `DN_Free(h)` korektně zastaví worker thread a uvolní síť.

### 2) Konfigurace sítě

```cpp
MQL_BOOL DN_SetSequenceLength(int h, int seq_len);
MQL_BOOL DN_SetMiniBatchSize(int h, int mbs);
MQL_BOOL DN_AddLayerEx(int h, int in, int out, int act, int ln, double drop);
MQL_BOOL DN_SetGradClip(int h, double clip);
MQL_BOOL DN_SetOutputDim(int h, int out_dim);
```

Důležité:

- `seq_len` a `mbs` jsou interně clampované minimálně na `1`.
- `DN_AddLayerEx` aktuálně mapuje na LSTM vrstvu (`in`, `out`, `drop`); `act` a `ln` jsou v této verzi nevyužité.
- `DN_SetOutputDim` musí být kompatibilní s cílovým výstupem v `DN_LoadBatch`.

### 3) Data, trénink, inference

```cpp
MQL_BOOL DN_LoadBatch(int h, const double* X,
                      const double* T, int batch, int in, int out, int l);

MQL_BOOL DN_Train(int h, int epochs,
                  double target_mse, double lr, double wd,
                  double* out_mse, int* out_epochs);

MQL_BOOL DN_TrainAsync(int h, int epochs,
                       double target_mse, double lr, double wd);

MQL_BOOL DN_PredictBatch(int h, const double* X,
                         int batch, int in, int l, double* Y);
```

Datový kontrakt:

- `in = seq_len * feature_dim`
- `X` má délku `batch * in`
- `T` má délku `batch * out`
- `Y` má délku `batch * out_dim`
- `in % seq_len == 0` musí platit.

Poznámka k parametru `l`: v aktuálním jádru je přítomen kvůli kompatibilitě signatury, ale není použit při výpočtu.

### 4) Async stav a výsledky

```cpp
int  DN_GetTrainingStatus(int h);
void DN_GetTrainingResult(int h, double* out_mse, int* out_epochs);
void DN_StopTraining(int h);
```

Stavy:

- `0` = idle
- `1` = training
- `2` = completed
- `-1` = error

### 5) Progress telemetry

```cpp
int    DN_GetProgressEpoch(int h);
int    DN_GetProgressTotalEpochs(int h);
int    DN_GetProgressMiniBatch(int h);
int    DN_GetProgressTotalMiniBatches(int h);
double DN_GetProgressLR(int h);
double DN_GetProgressMSE(int h);
double DN_GetProgressBestMSE(int h);
double DN_GetProgressGradNorm(int h);
int    DN_GetProgressTotalSteps(int h);
double DN_GetProgressPercent(int h);
double DN_GetProgressElapsedSec(int h);
double DN_GetProgressETASec(int h);

MQL_BOOL DN_GetProgressAll(
    int h,
    int* out_epoch, int* out_total_epochs,
    int* out_mb, int* out_total_mb,
    double* out_lr, double* out_mse, double* out_best_mse,
    double* out_grad_norm, double* out_pct,
    double* out_elapsed_sec, double* out_eta_sec);
```

Pro MT5 je nejpraktičtější periodicky volat `DN_GetProgressAll` z `OnTimer`.

### 6) Diagnostika, snapshot a stav

```cpp
MQL_BOOL DN_SnapshotWeights(int h);
MQL_BOOL DN_RestoreWeights(int h);

int    DN_GetLayerCount(int h);
double DN_GetLayerWeightNorm(int h, int l);
double DN_GetGradNorm(int h);

int      DN_SaveState(int h);
MQL_BOOL DN_GetState(int h, char* buf, int max_len);
MQL_BOOL DN_LoadState(int h, const char* buf);

void DN_GetError(short* buf, int len);
```

---

## Thread-safety a lockování

- Mutační operace nad sítí používají exkluzivní lock (`net_mtx`).
- Progress gettery a status gettery jsou lock-free (atomiky).
- Při běžícím tréninku může konfigurace vracet chybu `Network ... is busy (training in progress)`.

---

## Build a nasazení

### Požadavky

- Windows x64
- Visual Studio (solution build)
- NVIDIA CUDA Toolkit
- MT5 s povolenými DLL importy

### Build

1. Otevři `MQL5GPULibrary_LSTM.sln`.
2. Zvol `Release | x64`.
3. Build solution.
4. Zkopíruj výslednou DLL do `MQL5\Libraries`.

### MT5 integrace

1. Zkopíruj `.mq5` soubory z `MQL5/Indicators`.
2. Přelož v MetaEditoru.
3. Zapni *Allow DLL imports*.

---

## Praktické doporučení

- Po každém `MQL_FALSE` hned volat `DN_GetError` a logovat text chyby.
- Před rizikovým tréninkem uložit `DN_SnapshotWeights`.
- U dlouhého tréninku používat `DN_TrainAsync` + timer polling progressu.
- Po `DN_Free` vždy v MQL5 handle vynulovat (např. `h = 0`).
