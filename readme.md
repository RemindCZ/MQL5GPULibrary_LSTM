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

## Comprehensive DLL Function Documentation (English)

This section provides a complete, implementation-aligned reference for all exported DLL functions used from MQL5.

### Calling convention and return semantics

- All exports use `__stdcall`.
- `MQL_BOOL` is an integer (`1 = success`, `0 = failure`).
- Handle-returning functions return `0` on failure.
- `DN_GetTrainingStatus` returns one of:
  - `0` = idle
  - `1` = training in progress
  - `2` = training completed
  - `-1` = error or invalid handle

### Model lifecycle and threading model

The DLL supports multiple independent models via integer handles.

1. Create model: `DN_Create`.
2. Configure sequence and mini-batch policy.
3. Add one or more LSTM layers.
4. Define output layer dimension.
5. Load training batch.
6. Start asynchronous training and poll status.
7. Run inference (`DN_PredictBatch`).
8. Optionally save/load model state.
9. Release handle with `DN_Free`.

Important runtime behavior:

- Asynchronous training runs in a worker thread.
- Training lock ownership is exclusive while training executes.
- `DN_GetTrainingStatus`, `DN_GetTrainingResult`, and `DN_StopTraining` are lock-free from caller perspective.

### Data layout contract for `X`, `T`, and `Y`

The API accepts host-side `double*` arrays and internally converts to GPU float tensors.

- Let:
  - `batch` = number of samples
  - `seq_len` = timesteps per sample
  - `feature_dim` = features per timestep
  - `in = seq_len * feature_dim`
  - `out` = output dimension
- Required linear lengths:
  - `X`: `batch * in`
  - `T`: `batch * out`
  - `Y`: `batch * out`
- `DN_LoadBatch` and `DN_PredictBatch` validate `in % seq_len == 0`.

### Detailed function reference

#### `int DN_Create();`

Creates a new network instance and returns a positive handle.

- Returns `0` if CUDA context initialization fails.
- Each handle maps to one internal `LSTMNet` instance.

#### `void DN_Free(int h);`

Safely destroys the model instance.

- Requests training stop if worker thread is active.
- Waits for synchronization points to avoid race conditions.
- Invalid handles are ignored safely.

#### `MQL_BOOL DN_SetSequenceLength(int h, int seq_len);`

Sets sequence length used by load/train/predict routines.

- Effective value is clamped to at least `1`.
- Must match the shape assumption used in `in` parameters.

#### `MQL_BOOL DN_SetMiniBatchSize(int h, int mbs);`

Configures mini-batch size for training.

- Effective value is clamped to at least `1`.
- Affects optimizer step scheduling and memory buffers.

#### `MQL_BOOL DN_AddLayerEx(int h, int in, int out, int act, int ln, double drop);`

Adds one LSTM layer.

- `in`: expected input size for the layer definition.
- `out`: hidden size of the LSTM layer.
- `drop`: dropout probability for that layer.
- `act` and `ln` are currently accepted for compatibility, but not used by the current implementation.

Operational note:

- During `DN_LoadBatch`/`DN_PredictBatch`, layer input dimensions are auto-bound to actual upstream dimensions when needed.

#### `MQL_BOOL DN_SetGradClip(int h, double clip);`

Sets gradient clipping value used during optimizer updates.

- Applied inside layer update kernels.
- Useful for controlling exploding gradients.

#### `MQL_BOOL DN_SetOutputDim(int h, int out_dim);`

Creates/updates output projection layer based on last LSTM hidden size.

- Requires at least one LSTM layer to exist.
- Reinitializes output weights when output dimension changes.

#### `MQL_BOOL DN_LoadBatch(int h, const double* X, const double* T, int batch, int in, int out, int l);`

Loads full training dataset batch into persistent GPU buffers.

- `X` and `T` must be non-null and preallocated with valid lengths.
- `l` is a layout parameter reserved for compatibility; current implementation does not branch by this argument.
- Rebuilds bindings when feature dimension differs from existing layer input shape.
- Ensures output layer shape equals requested `out`.

On success, this function sets the internal training dataset used by `DN_TrainAsync`.

#### `MQL_BOOL DN_PredictBatch(int h, const double* X, int batch, int in, int l, double* Y);`

Runs forward inference on provided input batch.

- Converts host `double` input to internal GPU float.
- Executes full forward pass through LSTM stack and output layer.
- Converts prediction back to host `double` output array.
- `l` is currently reserved and not used in dispatch logic.

#### `MQL_BOOL DN_SnapshotWeights(int h);`

Stores current model weights as "best" snapshot.

- Captures all LSTM layers and output layer.
- Intended for checkpointing before risky retraining.

#### `MQL_BOOL DN_RestoreWeights(int h);`

Restores model weights from previously saved snapshot.

- Restores all compatible layers.
- Returns failure if snapshots are not available for required components.

#### `MQL_BOOL DN_TrainAsync(int h, int epochs, double target_mse, double lr, double wd);`

Starts asynchronous training over the last batch loaded by `DN_LoadBatch`.

- Preconditions:
  - Batch data already loaded.
  - At least one LSTM layer.
  - Output layer initialized.
- Uses shuffled mini-batches, Adam-style updates, learning-rate schedule, optional early stop (`target_mse > 0`).
- Returns `0` immediately if training is already running.

Training completes when:

- all epochs finish,
- `target_mse` is reached during periodic full-dataset evaluation,
- or stop flag is set via `DN_StopTraining`.

#### `int DN_GetTrainingStatus(int h);`

Reads current async training state.

- Non-blocking polling call.
- Safe to call frequently from indicator timer/update loops.

#### `void DN_GetTrainingResult(int h, double* out_mse, int* out_epochs);`

Retrieves final metrics from latest async run.

- `out_mse`: final full-train MSE.
- `out_epochs`: number of epochs completed.
- Either pointer may be null.

#### `void DN_StopTraining(int h);`

Sets cooperative stop flag for active training worker.

- Stop is observed at epoch/mini-batch boundaries.
- Use with status polling until non-running state is returned.

#### `int DN_GetLayerCount(int h);`

Returns total number of active layers.

- Includes all LSTM layers plus output layer if created.

#### `double DN_GetLayerWeightNorm(int h, int l);`
#### `double DN_GetGradNorm(int h);`

Diagnostic placeholders in current build.

- Currently return `0.0`.
- Exposed for forward-compatible tooling/API stability.

#### `int DN_SaveState(int h);`

Serializes current model into internal text buffer and returns required byte count including null terminator.

- Returns `0` on failure.
- Must be followed by `DN_GetState` to copy payload out.

#### `MQL_BOOL DN_GetState(int h, char* buf, int max_len);`

Copies previously serialized state into caller buffer.

- Requires successful `DN_SaveState` before call.
- Fails if buffer is too small.
- Writes null-terminated text payload.

#### `MQL_BOOL DN_LoadState(int h, const char* buf);`

Loads model from serialized text format.

- Expected header token: `LSTM_V1`.
- Recreates architecture and copies all weights/biases to GPU.

#### `void DN_GetError(short* buf, int len);`

Returns last DLL error message as UTF-16-like short buffer.

- Provide writable array and positive length.
- Result is null-terminated.
- Typical errors include CUDA launch/runtime failures, dimension mismatches, and out-of-memory conditions.

### Recommended robust usage pattern in MQL5

1. Build/configure architecture once at startup.
2. Validate every return code (`== 1`).
3. On failure, immediately call `DN_GetError` and log message.
4. During asynchronous training:
   - call `DN_TrainAsync`,
   - poll `DN_GetTrainingStatus` from `OnTimer`/`OnCalculate`,
   - read metrics with `DN_GetTrainingResult` on completion,
   - optionally checkpoint via `DN_SnapshotWeights`.
5. Always call `DN_Free` in deinitialization path.

---


## Interaktivní HTML dokumentace

K repozitáři byla doplněna vizuální dokumentace vysvětlující funkci LSTM v DLL:

- `docs/index.html`
- `docs/app.js`
- `docs/lstm-flow.svg`

Dokumentaci otevřete v prohlížeči přes `docs/index.html`.

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
