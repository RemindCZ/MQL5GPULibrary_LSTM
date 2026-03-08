# MQL5GPULibrary_LSTM — Dokumentace kernelu 2.0.0

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-brightgreen.svg)
![MT5](https://img.shields.io/badge/MetaTrader%205-DLL%20API-blue.svg)

Tento repozitář obsahuje CUDA DLL knihovnu pro MetaTrader 5 s vícevstvým LSTM modelem a asynchronním trénováním.
Tato dokumentace je kompletně připravena pro **kernel v2.0.0** (`kernel.cu`).

---

## 1. Co přináší verze 2.0.0

Kernel 2.0.0 sjednocuje trénování, inferenci, diagnostiku a telemetrii do jedné stabilní C API vrstvy exportované z DLL.

Hlavní body:

- více modelů paralelně přes handle (`DN_Create` / `DN_Free`),
- asynchronní trénování s průběžným pollingem (`DN_TrainAsync` + `DN_GetProgressAll`),
- checkpointing vah (`DN_SnapshotWeights`, `DN_RestoreWeights`),
- serializace stavu (`DN_SaveState`, `DN_GetState`, `DN_LoadState`),
- centralizované chybové hlášky (`DN_GetError`),
- standardizovaný stavový automat trénování.

---

## 2. Rychlý start (MT5 workflow)

Doporučené pořadí volání API:

1. `DN_Create`
2. `DN_SetSequenceLength`
3. `DN_SetMiniBatchSize`
4. `DN_AddLayerEx` (pro všechny LSTM vrstvy)
5. `DN_SetOutputDim`
6. `DN_LoadBatch`
7. volitelně `DN_SnapshotWeights`
8. `DN_TrainAsync`
9. polling: `DN_GetTrainingStatus` + `DN_GetProgressAll`
10. `DN_GetTrainingResult`
11. `DN_PredictBatch`
12. `DN_Free`

---

## 3. API přehled (kernel 2.0.0)

Všechny exporty používají `__stdcall`. Funkce s návratem `MQL_BOOL` vrací:

- `1` = úspěch
- `0` = chyba

### 3.1 Lifecycle

```cpp
int DN_Create();
void DN_Free(int h);
```

- `DN_Create` vrací kladný handle modelu.
- `DN_Free` uvolní CPU/GPU prostředky handle.

### 3.2 Konfigurace modelu

```cpp
MQL_BOOL DN_SetSequenceLength(int h, int seq_len);
MQL_BOOL DN_SetMiniBatchSize(int h, int mbs);
MQL_BOOL DN_AddLayerEx(int h, int in, int out, int act, int ln, double drop);
MQL_BOOL DN_SetGradClip(int h, double clip);
MQL_BOOL DN_SetOutputDim(int h, int out_dim);
```

Důležité zásady:

- `seq_len > 0`
- dropout (`drop`) patří do intervalu `[0, 1)`
- `DN_SetOutputDim` musí odpovídat cílovému rozměru `T`

### 3.3 Data a inference

```cpp
MQL_BOOL DN_LoadBatch(int h, const double* X, const double* T, int batch, int in, int out, int l);
MQL_BOOL DN_PredictBatch(int h, const double* X, int batch, int in, int l, double* Y);
```

Kontrakt tvarů:

- `in = seq_len * feature_dim`
- `len(X) = batch * in`
- `len(T) = batch * out`
- `len(Y) = batch * out_dim_modelu`
- musí platit: `in % seq_len == 0`

### 3.4 Trénování

```cpp
MQL_BOOL DN_Train(int h, int epochs, double target_mse, double lr, double wd);
MQL_BOOL DN_TrainAsync(int h, int epochs, double target_mse, double lr, double wd);
int DN_GetTrainingStatus(int h);
void DN_GetTrainingResult(int h, double* out_mse, int* out_epochs);
void DN_StopTraining(int h);
```

- `DN_Train` je synchronní varianta.
- `DN_TrainAsync` vrací okamžitě a trénuje na pozadí.
- `DN_StopTraining` vyžádá korektní ukončení asynchronního běhu.

### 3.5 Progress telemetry

```cpp
int DN_GetProgressEpoch(int h);
int DN_GetProgressTotalEpochs(int h);
int DN_GetProgressMiniBatch(int h);
int DN_GetProgressTotalMiniBatches(int h);
double DN_GetProgressLR(int h);
double DN_GetProgressMSE(int h);
double DN_GetProgressBestMSE(int h);
double DN_GetProgressGradNorm(int h);
int DN_GetProgressTotalSteps(int h);
double DN_GetProgressPercent(int h);
double DN_GetProgressElapsedSec(int h);
double DN_GetProgressETASec(int h);
MQL_BOOL DN_GetProgressAll(int h,
                           int* epoch, int* total_epochs,
                           int* minibatch, int* total_minibatches,
                           double* mse, double* best_mse,
                           double* lr, double* grad_norm,
                           double* percent, double* elapsed_sec, double* eta_sec);
```

Doporučení: v `OnTimer` preferujte jeden hromadný dotaz přes `DN_GetProgressAll` místo mnoha jednotlivých callů.

### 3.6 Diagnostika a serializace

```cpp
int DN_GetLayerCount(int h);
double DN_GetLayerWeightNorm(int h, int l);
double DN_GetGradNorm(int h);
void DN_GetError(short* buf, int len);

int DN_SaveState(int h);
MQL_BOOL DN_GetState(int h, char* buf, int max_len);
MQL_BOOL DN_LoadState(int h, const char* buf);

MQL_BOOL DN_SnapshotWeights(int h);
MQL_BOOL DN_RestoreWeights(int h);
```

- serializační formát používá hlavičku `LSTM_V1`,
- při chybě vždy čtěte `DN_GetError`.

---

## 4. Stavový automat trénování

`DN_GetTrainingStatus` vrací:

- `0` → `TS_IDLE`
- `1` → `TS_TRAINING`
- `2` → `TS_COMPLETED`
- `-1` → `TS_ERROR`

Příklad bezpečného pollingu:

```cpp
while(DN_GetTrainingStatus(h) == 1) {
    // Sleep / čekání mezi tick-y
}

double mse = 0.0;
int epochs_done = 0;
DN_GetTrainingResult(h, &mse, &epochs_done);
```

---

## 5. Praktická doporučení pro stabilní provoz

- Vždy nastavte `seq_len` před načtením dat.
- Pokud experimentujete, dělejte checkpoint před dlouhým trénováním.
- V UI používejte periodický polling (`OnTimer`), ne agresivní polling v každém ticku.
- Po chybě ihned načtěte diagnostiku přes `DN_GetError`.
- Při použití více modelů držte striktní mapování `handle -> dataset -> konfigurace`.

---

## 6. Build a nasazení

### Build (Visual Studio + CUDA)

1. Otevřete `MQL5GPULibrary_LSTM.sln`.
2. Zkontrolujte nainstalovaný CUDA Toolkit.
3. Build konfigurace: `Release | x64`.
4. Vzniklé DLL zkopírujte do `MQL5\Libraries` v datové složce MT5.

### Nasazení v MetaTrader 5

1. Uložte DLL do `MQL5\Libraries`.
2. Zkopírujte indikátor (např. z `MQL5/Indicators/`) do `MQL5\Indicators`.
3. Zkompilujte v MetaEditoru.
4. V MT5 povolte DLL imports.

---

## 7. Obsah repozitáře

- `kernel.cu` — implementace kernelu 2.0.0 (CUDA/C++).
- `MQL5/Indicators/` — integrační indikátory pro MT5.
- `docs/index.html`, `docs/app.js`, `docs/lstm-flow.svg` — interaktivní dokumentace.
- `MQL5GPULibrary_LSTM.sln`, `MQL5GPULibrary_LSTM.vcxproj` — build konfigurace pro Visual Studio.

---

## 8. Licence

MIT (`LICENSE.txt`).
