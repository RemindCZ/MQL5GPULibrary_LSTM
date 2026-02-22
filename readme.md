# MQL5GPULibrary_LSTM

`MQL5GPULibrary_LSTM` je 64bit DLL knihovna pro MetaTrader 5, která poskytuje LSTM síť akcelerovanou přes NVIDIA CUDA. V repozitáři je i ukázkový indikátor `MQL5/Indicators/LSTMTrendStart.mq5`, který knihovnu používá pro predikci očekávaného pohybu (Expected Move %) a kreslí adaptivní prahy do samostatného okna.

---

## Co je v repozitáři

- `MQL5GPULibrary_LSTM.dll` – zkompilovaná DLL pro použití v MT5.
- `kernel.cu` – zdrojový kód DLL (exportované funkce API).
- `MQL5/Indicators/LSTMTrendStart.mq5` – referenční indikátor s asynchronním trénováním.

---

## Rychlé nasazení v MT5

1. Zkopírujte `MQL5GPULibrary_LSTM.dll` do `MQL5\Libraries` (Data Folder vašeho MT5).
2. Zkopírujte `MQL5/Indicators/LSTMTrendStart.mq5` do `MQL5\Indicators`.
3. Otevřete MetaEditor a indikátor zkompilujte.
4. V MT5 povolte **Allow DLL imports**.
5. Přidejte indikátor do grafu a nastavte inputy (`InpSeqLen`, `InpTrainBars`, `InpEpochs`, `InpTargetMSE`, prahy apod.).

---

## Jak funguje ukázkový indikátor `LSTMTrendStart`

Indikátor:

- pracuje v `indicator_separate_window` a vykresluje:
  - `ExpectedMove %` (hlavní křivka),
  - `Thr Low` (dolní adaptivní práh),
  - `Thr High` (horní adaptivní práh);
- používá multi-symbol vstupy (hlavní symbol + `InpExtraSymbol1` + `InpExtraSymbol2`);
- připravuje sekvenční data (`InpSeqLen`) a target přes budoucí rozsah (`InpHorizonH`);
- trénuje asynchronně (`DN_TrainAsync`) a průběh hlídá přes `DN_GetTrainingStatus`;
- používá snapshot/restore vah (`DN_SnapshotWeights`, `DN_RestoreWeights`) pro bezpečnější retrain;
- obsahuje robustní debug režim (`InpSuperDebug`) včetně limitu logů.

Doporučení:

- Pro první test používejte menší `InpTrainBars` a `InpEpochs`.
- Pokud se data mezi symboly hůře zarovnávají, ponechte `InpAlignExact=false`.
- Prahy dolaďujte přes `InpThrMinPct`, `InpThrLowPercentile`, `InpThrHighPercentile`.

---

## API DLL (aktuální exporty)

> V DLL jsou návratové typy `MQL_BOOL` (0/1), v MQL5 importu se běžně mapují jako `int`.

### Životní cyklus sítě

1. `DN_Create`
2. `DN_SetSequenceLength`
3. `DN_SetMiniBatchSize`
4. `DN_AddLayerEx` (jedna nebo více vrstev)
5. volitelně `DN_SetGradClip`
6. `DN_LoadBatch`
7. `DN_TrainAsync` + polling přes `DN_GetTrainingStatus`
8. `DN_PredictBatch`
9. `DN_Free`

### Exportované funkce

```cpp
int      DN_Create();
void     DN_Free(int h);

MQL_BOOL DN_SetSequenceLength(int h, int seq_len);
MQL_BOOL DN_SetMiniBatchSize(int h, int mbs);
MQL_BOOL DN_AddLayerEx(int h, int in, int out, int act, int ln, double drop);
MQL_BOOL DN_SetGradClip(int h, double clip);
MQL_BOOL DN_SetOutputDim(int h, int out_dim);

MQL_BOOL DN_LoadBatch(int h, const double* X, const double* T,
                      int batch, int in, int out, int l);
MQL_BOOL DN_PredictBatch(int h, const double* X,
                         int batch, int in, int l, double* Y);

MQL_BOOL DN_SnapshotWeights(int h);
MQL_BOOL DN_RestoreWeights(int h);

double   DN_GetGradNorm(int h);
double   DN_GetLayerWeightNorm(int h, int l);
int      DN_GetLayerCount(int h);

MQL_BOOL DN_TrainAsync(int h, int epochs, double target_mse, double lr, double wd);
int      DN_GetTrainingStatus(int h);   // 0=idle, 1=running, 2=completed, -1=error
void     DN_GetTrainingResult(int h, double* out_mse, int* out_epochs);
void     DN_StopTraining(int h);

int      DN_SaveState(int h);
MQL_BOOL DN_GetState(int h, char* buf, int max_len);
MQL_BOOL DN_LoadState(int h, const char* buf);

void     DN_GetError(short* buf, int len);
```

---

## Důležité poznámky k datům

- `seq_len` musí souhlasit mezi konfigurací (`DN_SetSequenceLength`) a daty (`DN_LoadBatch`, `DN_PredictBatch`).
- Vstupní dimenze `in` musí odpovídat první vrstvě v `DN_AddLayerEx`.
- Výstupní dimenze targetu `out` musí odpovídat modelu.
- Buffery `X`, `T`, `Y` v MQL5 musí být správně předalokované.

---

## Diagnostika chyb

Při chybě volejte `DN_GetError`:

```mq5
short err[];
ArrayResize(err, 512);
DN_GetError(err, 512);
Print("DLL error: ", ShortArrayToString(err));
```

Typické problémy:

- nepovolené DLL importy v MT5,
- chybějící/nekompatibilní CUDA runtime knihovny,
- nevalidní rozměry batch/sekvence,
- nedostatek GPU paměti.

---

## Build ze zdroje

1. Otevřete `MQL5GPULibrary_LSTM.sln` ve Visual Studiu.
2. Zkontrolujte nainstalovanou CUDA toolchain (projekt používá CUDA build customizations).
3. Build konfigurace: **Release | x64**.
4. Zkompilujte DLL a výsledný soubor nasaďte do `MQL5\Libraries`.

---

## Licence

MIT, viz `LICENSE.txt`.
