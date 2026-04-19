# `kernel.cu` – funkční dokumentace (aktuální stav)

Tento dokument je přepsán podle současného obsahu `kernel.cu` a popisuje **reálně exportované** API a chování interního LSTM runtime.

---

## 1) Základní konvence

### 1.1 Matice a cuBLAS

Celé jádro používá **column-major** layout (nativní pro cuBLAS):

- prvek `(i, j)` je na indexu `i + j * rows`,
- leading dimension `lda = počet řádků`.

To je kritické pro interpretaci vah LSTM i output vrstvy.

### 1.2 Základní typy

- `MQL_BOOL` = `int`, kde:
  - `MQL_TRUE = 1`
  - `MQL_FALSE = 0`

- `TrainingState`:
  - `TS_IDLE = 0`
  - `TS_TRAINING = 1`
  - `TS_COMPLETED = 2`
  - `TS_ERROR = -1`

---

## 2) Error handling

Interní chyby jsou ukládány globálně přes `SetError(...)` a čteny exportem:

```cpp
void DN_GetError(short* buf, int len);
```

### Chování

- `DN_GetError` kopíruje poslední wide-string chybu do `short*` bufferu.
- Při chybě CUDA/cuBLAS/cuRAND se nastavuje i interní status (`g_last_cuda`, `g_last_cublas`, `g_last_curand`).
- K typickým textům patří:
  - `Network X not found`
  - `Network X is busy (training in progress)`
  - CUDA launch/runtime chyby

---

## 3) Handle management

### `int DN_Create()`

Vytvoří `LSTMNet`, zaregistruje jej do mapy handle→instance a vrátí kladný handle.

**Návrat:**
- `>0` úspěch
- `0` chyba (včetně overflow handle čítače)

### `void DN_Free(int h)`

- odstraní síť z globální mapy,
- zavolá `StopTraining()`,
- počká na worker (`JoinWorker()`),
- korektně doběhne destrukce zdrojů.

---

## 4) Konfigurace modelu

### `MQL_BOOL DN_SetSequenceLength(int h, int seq_len)`

- Interně: `seq_len = max(1, seq_len)`.

### `MQL_BOOL DN_SetMiniBatchSize(int h, int mbs)`

- Interně: `mini_batch_size = max(1, mbs)`.

### `MQL_BOOL DN_AddLayerEx(int h, int in, int out, int act, int ln, double drop)`

Aktuálně mapuje na `AddLayer(...)`, který volá `AddLSTMLayer(in, out, drop)`.

- `in` = vstupní dimenze vrstvy
- `out` = hidden size vrstvy
- `drop` = dropout rate
- `act`, `ln` = v této verzi nejsou v logice využité (zůstávají kvůli API kompatibilitě)

### `MQL_BOOL DN_SetGradClip(int h, double clip)`

Nastaví `grad_clip` pro globální clipping gradientu během update.

### `MQL_BOOL DN_SetOutputDim(int h, int out_dim)`

Vytvoří/reinicializuje output vrstvu nad poslední LSTM vrstvou.

**Precondition:** musí existovat aspoň jedna LSTM vrstva.

---

## 5) Práce s daty

### `MQL_BOOL DN_LoadBatch(int h, const double* X, const double* T, int batch, int in, int out, int l)`

Načte trénovací dataset do GPU.

### Datový kontrakt

- `in = seq_len * feature_dim`
- `X` délka = `batch * in`
- `T` délka = `batch * out`
- `in % seq_len == 0`

### Co se děje uvnitř

1. Kontrola vstupů a kontextu.
2. `in -> feature_dim` výpočet a validace.
3. `BindInputIfNeeded` + `BindIntermediateLayers` (lazy rebinding dimenzí).
4. Kontrola/rebuild output vrstvy na `out`.
5. Přenos `double` dat na GPU + převod `double -> float` kernely.
6. Příprava `ones` bufferu a redukčních bufferů.
7. Reset Adam momentů všech vrstev (`mW/vW/mb/vb`) asynchronně na streamu.
8. Reset training step interních proměnných (`b1_pow`, `b2_pow`, `step`, `last_full_train_mse`).

### Poznámka

Parametr `l` je v aktuální implementaci signatury, ale není použit při výpočtu.

---

## 6) Trénink

### 6.1 Sync

```cpp
MQL_BOOL DN_Train(int h, int epochs,
                  double target_mse, double lr, double wd,
                  double* out_mse, int* out_epochs);
```

- Vyžaduje načtený dataset + vrstvy + output.
- Volá `TrainSync_Locked`.
- Nastaví stav `TS_TRAINING`, po doběhu `TS_COMPLETED` nebo `TS_ERROR`.
- Při úspěchu plní `out_mse`, `out_epochs` (pokud nejsou `nullptr`).

### 6.2 Async

```cpp
MQL_BOOL DN_TrainAsync(int h, int epochs,
                       double target_mse, double lr, double wd);
```

- Start worker threadu přes `StartTrainingAsync_Locked`.
- Používá handshake `promise/future`, aby nedošlo k deadlocku při předání locku.
- Ošetřen fail konstrukce `std::thread` (`std::system_error`) → `TS_ERROR`.

### 6.3 Stop/result/status

```cpp
int  DN_GetTrainingStatus(int h);
void DN_GetTrainingResult(int h, double* out_mse, int* out_epochs);
void DN_StopTraining(int h);
```

- `DN_StopTraining` nastaví stop flag; trénink skončí při nejbližší kontrolní podmínce.
- `DN_GetTrainingStatus` vrací hodnotu `TrainingState`.

---

## 7) Progress API (lock-free)

Exporty:

- `DN_GetProgressEpoch`
- `DN_GetProgressTotalEpochs`
- `DN_GetProgressMiniBatch`
- `DN_GetProgressTotalMiniBatches`
- `DN_GetProgressLR`
- `DN_GetProgressMSE`
- `DN_GetProgressBestMSE`
- `DN_GetProgressGradNorm`
- `DN_GetProgressTotalSteps`
- `DN_GetProgressPercent`
- `DN_GetProgressElapsedSec`
- `DN_GetProgressETASec`
- `DN_GetProgressAll`

### Doporučení

V MT5 pollovat přes `OnTimer`, ne přes každý tick. Nejpraktičtější je `DN_GetProgressAll`.

---

## 8) Inference

### `MQL_BOOL DN_PredictBatch(int h, const double* X, int batch, int in, int l, double* Y)`

### Tok uvnitř

1. Kontrola vstupu a vazeb dimenzí.
2. Převod `double -> float`.
3. Transpozice do timestep layoutu (`kTransposeToTimestep`).
4. `ForwardFull(..., training=false)`.
5. Převod výsledku `float -> double` a kopie do host `Y`.

**Pozn.:** parametr `l` je nevyužitý, zachován pro kompatibilitu API.

---

## 9) Snapshot, diagnostika a stav modelu

### Snapshot

```cpp
MQL_BOOL DN_SnapshotWeights(int h);
MQL_BOOL DN_RestoreWeights(int h);
```

- Uloží/obnoví „best“ kopie vah všech LSTM vrstev i output vrstvy.

### Diagnostika

```cpp
int    DN_GetLayerCount(int h);
double DN_GetLayerWeightNorm(int h, int l);
double DN_GetGradNorm(int h);
```

- `DN_GetLayerCount`: `#lstm_layers + output_layer`.
- `DN_GetLayerWeightNorm`: L2 norma vah vrstvy.
- `DN_GetGradNorm`: globální L2 norma gradientů přes všechny vrstvy.

### Serializace

```cpp
int      DN_SaveState(int h);
MQL_BOOL DN_GetState(int h, char* buf, int max_len);
MQL_BOOL DN_LoadState(int h, const char* buf);
```

#### Formát

- Header: `LSTM_V1`
- Dále `SEQ_LEN`, počet vrstev, parametry každé vrstvy, output vrstva.
- `DN_SaveState` ukládá do interního `serialize_buf` a vrací potřebnou délku včetně `\0`.
- `DN_GetState` vrátí `MQL_FALSE`, pokud je buffer malý nebo stav není uložen.

---

## 10) Locking model a souběh

- Konfigurační a mutační exporty používají exkluzivní lock (`FindAndLockExclusive`).
- Pokud síť neexistuje: explicitní chyba „not found“.
- Pokud síť právě trénuje: explicitní chyba „busy“.
- Progress/status API běží bez locku přes atomiky.

---

## 11) Důležité rozdíly proti starší dokumentaci

- Exportované API je **LSTM-only** (žádné `DN_AddGRULayer`, `DN_AddRNNLayer`).
- `DN_Train` má out parametry (`out_mse`, `out_epochs`) v signatuře.
- `DN_GetProgressAll` má pořadí argumentů:
  `lr, mse, best_mse, grad_norm, pct, elapsed, eta`.
- Parametry `act`, `ln`, `layout/l` jsou aktuálně kompatibilitní a výpočetně se nevyužívají.

---

## 12) Rychlý checklist pro MT5

1. Vytvoř handle (`DN_Create`).
2. Nastav `seq_len`, `mbs`, vrstvy, output.
3. Zavolej `DN_LoadBatch` s validními dimenzemi.
4. Spusť `DN_TrainAsync`.
5. Polluj `DN_GetTrainingStatus` + `DN_GetProgressAll` v timeru.
6. Po dokončení `DN_GetTrainingResult` a případně `DN_PredictBatch`.
7. Na konci vždy `DN_Free`.
