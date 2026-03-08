# MQL5GPULibrary_LSTM — Dokumentace `kernel.cu` v2.1.0

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-brightgreen.svg)
![MT5](https://img.shields.io/badge/MetaTrader%205-DLL%20API-blue.svg)

Tento dokument je zcela nová technická dokumentace pro **LSTM DLL v2.1.0** (`kernel.cu`) se zaměřením na:

- **asynchronní trénování** (`DN_TrainAsync`),
- **reporting průběhu tréninku** (`DN_GetProgress*`, `DN_GetProgressAll`),
- bezpečné řízení životního cyklu modelu v prostředí **MetaTrader 5 + MQL5**.

---

## 1) Co je nové ve verzi 2.1.0

Verze 2.1.0 sjednocuje robustní GPU trénink, správu stavu modelu a telemetrii do jedné DLL API vrstvy.

Nejdůležitější body:

1. **Async trénování na pozadí** bez blokování hlavní MQL5 logiky (`DN_TrainAsync`).
2. **Jemnozrnný progress reporting** (epocha, minibatch, ETA, elapsed, best MSE, grad norm).
3. **Hromadné čtení telemetrie** jedním voláním (`DN_GetProgressAll`) pro nižší overhead.
4. **Více současných modelů** přes handle architekturu (`DN_Create` / `DN_Free`).
5. **Snapshot/restore vah** pro rollback experimentů (`DN_SnapshotWeights`, `DN_RestoreWeights`).
6. **Serializace stavu** pro persistenci (`DN_SaveState`, `DN_GetState`, `DN_LoadState`).
7. **Centralizované chybové hlášení** (`DN_GetError`) použitelné v každé fázi pipeline.

---

## 2) Architektura použití v MT5

Každý model je reprezentován integer handlem. Můžete provozovat více handle instancí paralelně (např. různé symboly/timeframy).

**Doporučený lifecycle:**

1. `DN_Create`
2. `DN_SetSequenceLength`
3. `DN_SetMiniBatchSize`
4. `DN_AddLayerEx` (opakovaně pro stack LSTM vrstev)
5. `DN_SetOutputDim`
6. `DN_LoadBatch`
7. (volitelně) `DN_SnapshotWeights`
8. `DN_TrainAsync` nebo `DN_Train`
9. polling přes `DN_GetTrainingStatus` + `DN_GetProgressAll`
10. `DN_GetTrainingResult`
11. `DN_PredictBatch`
12. `DN_Free`

---

## 3) API reference (v2.1.0)

> Všechny exporty používají `__stdcall`.
>
> Typ `MQL_BOOL`: `1 = success`, `0 = fail`.

### 3.1 Lifecycle

```cpp
int DN_Create();
void DN_Free(int h);
```

- `DN_Create` vrací kladný handle, při chybě `<= 0`.
- `DN_Free` vždy volejte při ukončení práce s modelem.

### 3.2 Konfigurace modelu

```cpp
MQL_BOOL DN_SetSequenceLength(int h, int seq_len);
MQL_BOOL DN_SetMiniBatchSize(int h, int mbs);
MQL_BOOL DN_AddLayerEx(int h, int in, int out, int act, int ln, double drop);
MQL_BOOL DN_AddGRULayer(int h, int in, int out, double drop);
MQL_BOOL DN_AddRNNLayer(int h, int in, int out, double drop);
MQL_BOOL DN_SetGradClip(int h, double clip);
MQL_BOOL DN_SetOutputDim(int h, int out_dim);
```

Konfigurační zásady:

- `seq_len > 0`
- `mbs > 0`
- `drop` v intervalu `<0, 1)`
- `out_dim` musí odpovídat cílovému rozměru v `T`

> Pro čisté LSTM workflow používejte primárně `DN_AddLayerEx`.

### 3.3 Nahrání dat + inference

```cpp
MQL_BOOL DN_LoadBatch(int h, const double* X, const double* T,
                      int batch, int in, int out, int l);
MQL_BOOL DN_PredictBatch(int h, const double* X,
                         int batch, int in, int l, double* Y);
```

Datový kontrakt:

- `in = seq_len * feature_dim`
- `len(X) = batch * in`
- `len(T) = batch * out`
- `len(Y) = batch * out_dim`
- musí platit `in % seq_len == 0`

### 3.4 Trénování (sync + async)

```cpp
MQL_BOOL DN_Train(int h, int epochs, double target_mse, double lr, double wd);
MQL_BOOL DN_TrainAsync(int h, int epochs, double target_mse, double lr, double wd);
int DN_GetTrainingStatus(int h);
void DN_GetTrainingResult(int h, double* out_mse, int* out_epochs);
void DN_StopTraining(int h);
```

Použití:

- `DN_Train` = blokující (synchronní) varianta.
- `DN_TrainAsync` = neblokující varianta na pozadí.
- `DN_StopTraining` = bezpečný požadavek na ukončení async běhu.

### 3.5 Progress Reporting API

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
MQL_BOOL DN_GetProgressAll(
    int h,
    int* epoch, int* total_epochs,
    int* minibatch, int* total_minibatches,
    double* mse, double* best_mse,
    double* lr, double* grad_norm,
    double* percent, double* elapsed_sec, double* eta_sec
);
```

Doporučení pro produkční MT5:

- polling dělejte v `OnTimer` (např. 100–500 ms), ne v každém ticku,
- preferujte **`DN_GetProgressAll`** před sérií jednotlivých getterů,
- ETA je orientační, ale velmi užitečná pro UI/UX tréninku.

### 3.6 Diagnostika, stav, snapshoty

```cpp
int DN_GetLayerCount(int h);
double DN_GetLayerWeightNorm(int h, int l);
double DN_GetGradNorm(int h);

int DN_SaveState(int h);
MQL_BOOL DN_GetState(int h, char* buf, int max_len);
MQL_BOOL DN_LoadState(int h, const char* buf);

MQL_BOOL DN_SnapshotWeights(int h);
MQL_BOOL DN_RestoreWeights(int h);

void DN_GetError(short* buf, int len);
```

- `DN_SaveState` + `DN_GetState` + `DN_LoadState` = serializace modelu.
- `DN_SnapshotWeights`/`DN_RestoreWeights` = rychlý rollback vah mezi experimenty.
- Při libovolném `MQL_FALSE` vždy ihned čtěte `DN_GetError`.

---

## 4) Trénovací stavový automat

`DN_GetTrainingStatus` vrací:

- `0` → `TS_IDLE`
- `1` → `TS_TRAINING`
- `2` → `TS_COMPLETED`
- `-1` → `TS_ERROR`

Bezpečný polling pattern:

```cpp
int st = DN_GetTrainingStatus(h);
while(st == 1) {
    // čekání řízené timerem / Sleep
    st = DN_GetTrainingStatus(h);
}

if(st == 2) {
    double mse = 0.0;
    int epochs_done = 0;
    DN_GetTrainingResult(h, &mse, &epochs_done);
}
```

---

## 5) Doporučené async workflow pro MQL5

### 5.1 Inicializace (OnInit)

- vytvořte handle,
- nakonfigurujte topologii,
- načtěte batch,
- spusťte `DN_TrainAsync`.

### 5.2 Periodický monitoring (OnTimer)

- čtěte `DN_GetTrainingStatus`,
- čtěte `DN_GetProgressAll`,
- aktualizujte panel / Comment / log,
- při `TS_COMPLETED` načtěte `DN_GetTrainingResult`.

### 5.3 Deinitializace (OnDeinit)

- volejte `DN_StopTraining` (pokud ještě běží),
- následně `DN_Free`.

---

## 6) Parametry tréninku a tuning

Parametry `DN_Train*`:

- `epochs` – max počet epoch,
- `target_mse` – cílová chyba pro early stop,
- `lr` – learning rate,
- `wd` – weight decay.

Praktická doporučení:

- začněte konzervativně (`lr` nižší, rozumný `epochs`),
- sledujte `best_mse` + `grad_norm` (detekce nestability),
- při explozivních gradientech nastavte `DN_SetGradClip`.

---

## 7) Build a deployment

### Build (Visual Studio + CUDA Toolkit)

1. Otevřete `MQL5GPULibrary_LSTM.sln`.
2. Ověřte kompatibilní CUDA Toolkit.
3. Build: `Release | x64`.
4. DLL zkopírujte do `MQL5\Libraries` v datové složce MT5.

### Deployment v MetaTrader 5

1. Umístěte DLL do `MQL5\Libraries`.
2. Umístěte indikátor z `MQL5/Indicators` do `MQL5\Indicators`.
3. Zkompilujte v MetaEditoru.
4. V MT5 povolte DLL imports.

---

## 8) Struktura repozitáře

- `kernel.cu` — implementace LSTM DLL v2.1.0.
- `MQL5/Indicators/` — integrační MQL5 indikátory a příklady.
- `docs/` — statická dokumentační aplikace.
- `MQL5GPULibrary_LSTM.sln`, `MQL5GPULibrary_LSTM.vcxproj` — build konfigurace.

---

## 9) Troubleshooting checklist

1. Po každé chybě (`MQL_FALSE`) čtěte `DN_GetError`.
2. Ověřte správné dimenze `X/T/Y` a `seq_len`.
3. Ujistěte se, že je dostupná CUDA kompatibilní GPU konfigurace.
4. Při paralelních modelech striktně oddělte handle + data.
5. Při dlouhých trénincích používejte snapshoty a průběžný progress polling.

---

## 10) Licence

MIT — viz `LICENSE.txt`.
