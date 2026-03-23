# MQL5GPULibrary_LSTM

> **Author of this documentation:** Tomáš Bělák  
> **Scope:** production-grade technical documentation for CUDA-accelerated LSTM training/inference from MetaTrader 5 (MQL5) via DLL bridge.

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-brightgreen.svg)
![MT5](https://img.shields.io/badge/MetaTrader%205-DLL%20API-blue.svg)
![LSTM](https://github.com/RemindCZ/MQL5GPULibrary_LSTM/blob/master/LSTM.gif)

---

## 1. Executive Summary

`MQL5GPULibrary_LSTM` is a hybrid quantitative-engineering project that exposes a CUDA-backed recurrent neural network runtime (LSTM-centric, with additional RNN/GRU layer support) to **MetaTrader 5** through a DLL API designed for low-latency practical use.

The project solves a common integration bottleneck in algorithmic trading systems:

- MQL5 is excellent for strategy orchestration, execution logic, and chart-native tooling,
- while high-throughput deep learning training is better suited to C++/CUDA.

This repository bridges those worlds with:

1. **Handle-based model lifecycle management** (`DN_Create`, `DN_Free`),
2. **Flexible network construction** (`DN_AddLayerEx`, `DN_AddGRULayer`, `DN_AddRNNLayer`),
3. **Synchronous and asynchronous training** (`DN_Train`, `DN_TrainAsync`),
4. **Detailed progress telemetry** (`DN_GetProgress*`, `DN_GetProgressAll`),
5. **State persistence and rollback tools** (`DN_SaveState`, `DN_GetState`, `DN_LoadState`, snapshots),
6. **Batch inference API suitable for indicator- or EA-level integration** (`DN_PredictBatch`).

---

## 2. Architecture and Design Goals

### 2.1 Primary design goals

The runtime is designed around five practical constraints from real trading-system environments:

- **Deterministic ownership:** each model instance is represented by an integer handle, minimizing cross-context ambiguity.
- **Operational resilience:** every mutating API call can be validated and followed by centralized error retrieval (`DN_GetError`).
- **UI responsiveness in MT5:** async training allows heavy GPU workloads without freezing chart logic or user interactions.
- **Production telemetry:** rich progress metadata (epochs, minibatches, best MSE, ETA, gradient norm) enables informed runtime decisions.
- **Experiment safety:** snapshots and state serialization reduce iteration risk and support reproducible workflows.

### 2.2 High-level component map

- **`kernel.cu`**: CUDA/C++ core implementing model graph, training loop, telemetry, and exported DLL boundary.
- **`MQL5/Indicators/*.mq5`**: integration layer + practical examples of usage in MT5 indicator context.
- **`docs/`**: static visualization and interactive documentation assets.
- **Solution/project files (`.sln`, `.vcxproj`)**: reproducible build entry points for Windows toolchain.

---

## 3. API Model and Lifecycle Contract

### 3.1 Lifecycle primitives

```cpp
int DN_Create();
void DN_Free(int h);
```

- `DN_Create` allocates and registers a new model context and returns a positive handle on success.
- `DN_Free` must be called exactly once per valid handle to release associated resources.

**Operational rule:** in MQL5, treat the handle as a managed resource; reset it to an invalid value after `DN_Free` to prevent accidental reuse.

### 3.2 Configuration stage

```cpp
MQL_BOOL DN_SetSequenceLength(int h, int seq_len);
MQL_BOOL DN_SetMiniBatchSize(int h, int mbs);
MQL_BOOL DN_AddLayerEx(int h, int in, int out, int act, int ln, double drop);
MQL_BOOL DN_AddGRULayer(int h, int in, int out, double drop);
MQL_BOOL DN_AddRNNLayer(int h, int in, int out, double drop);
MQL_BOOL DN_SetGradClip(int h, double clip);
MQL_BOOL DN_SetOutputDim(int h, int out_dim);
```

#### Configuration invariants

- `seq_len > 0`
- `mbs > 0`
- `0.0 <= drop < 1.0`
- `out_dim` must match target tensor dimensionality.

For canonical LSTM architecture stacks, `DN_AddLayerEx` should be considered the primary constructor.

### 3.3 Data loading and prediction

```cpp
MQL_BOOL DN_LoadBatch(int h, const double* X, const double* T,
                      int batch, int in, int out, int l);
MQL_BOOL DN_PredictBatch(int h, const double* X,
                         int batch, int in, int l, double* Y);
```

#### Tensor-shape contract (critical)

- `in = seq_len * feature_dim`
- `len(X) = batch * in`
- `len(T) = batch * out`
- `len(Y) = batch * out_dim`
- `in % seq_len == 0` must hold

Violating these assumptions typically produces immediate API failure and a meaningful error message retrievable via `DN_GetError`.

---

## 4. Training Engine Modes

### 4.1 Synchronous training

```cpp
MQL_BOOL DN_Train(int h, int epochs, double target_mse, double lr, double wd);
```

Use synchronous mode when:

- integrating in offline optimization scripts,
- deterministic blocking is acceptable,
- simple control flow is preferred.

### 4.2 Asynchronous training

```cpp
MQL_BOOL DN_TrainAsync(int h, int epochs, double target_mse, double lr, double wd);
int DN_GetTrainingStatus(int h);
void DN_GetTrainingResult(int h, double* out_mse, int* out_epochs);
void DN_StopTraining(int h);
```

Use asynchronous mode when:

- the indicator/EA must keep processing ticks or UI events,
- long training runs require progress visibility,
- strategy architecture enforces non-blocking behavior.

#### Recommended status machine semantics

- `0` → idle
- `1` → training in progress
- `2` → completed
- `-1` → error

---

## 5. Progress Telemetry (Operational Intelligence)

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

### Telemetry interpretation guide

- **`mse`**: current optimization state, noisy at batch granularity.
- **`best_mse`**: stability anchor; evaluate convergence versus this metric.
- **`grad_norm`**: early warning for exploding/unstable optimization.
- **`eta_sec`**: operational estimate for UI and task scheduling.

### Polling best practices in MT5

- Poll from `OnTimer` (e.g., 100–500 ms), not every tick.
- Prefer `DN_GetProgressAll` to reduce call overhead and improve consistency.
- Persist sampled telemetry for post-mortem model diagnostics.

---

## 6. State Persistence, Rollback, and Diagnostics

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

### Practical production patterns

- **Checkpointing policy:** call `DN_SaveState` at milestone convergence points and export state string externally.
- **Safe experimentation:** `DN_SnapshotWeights` before high-risk hyperparameter changes; restore when divergence is detected.
- **Failure triage:** on any `MQL_FALSE`, immediately call `DN_GetError` and log the returned message with context.

---

## 7. Full MT5 Integration Workflow

### 7.1 Initialization (`OnInit`)

1. `DN_Create`
2. Configure sequence length and minibatch
3. Build layer stack
4. Set output dimension
5. Load training batch
6. Optionally snapshot
7. Start `DN_TrainAsync`

### 7.2 Runtime loop (`OnTimer`)

1. Query status (`DN_GetTrainingStatus`)
2. Read telemetry (`DN_GetProgressAll`)
3. Update chart UI / logs
4. If completed, collect result (`DN_GetTrainingResult`)
5. Optionally run inference (`DN_PredictBatch`)

### 7.3 Deinitialization (`OnDeinit`)

1. If running, request stop (`DN_StopTraining`)
2. Release handle (`DN_Free`)
3. Zero/reset local handle variable

---

## 8. Build and Deployment

### 8.1 Build prerequisites

- Windows with Visual Studio capable of opening `.sln`
- NVIDIA CUDA Toolkit compatible with your compiler toolset
- x64 target environment

### 8.2 Build process

1. Open `MQL5GPULibrary_LSTM.sln`
2. Select `Release | x64`
3. Build solution
4. Copy resulting DLL into terminal data folder `MQL5\Libraries`

### 8.3 MT5 deployment

1. Copy indicator files from `MQL5/Indicators` into `MQL5\Indicators`
2. Compile in MetaEditor
3. Enable DLL imports in MT5 settings
4. Attach indicator to chart and verify initialization logs

---

## 9. Performance and Stability Recommendations

### Hyperparameters

- Start conservatively (`lr`, `epochs`) and tune incrementally.
- Use `target_mse` for controlled early stopping.
- Apply gradient clipping (`DN_SetGradClip`) when gradient norm spikes.

### Data pipeline

- Validate shape arithmetic before every `DN_LoadBatch` call.
- Keep feature normalization strategy consistent between train and inference.
- Use fixed seeds and deterministic preprocessing where possible.

### Concurrency and resource hygiene

- Isolate each strategy/model with a unique handle.
- Do not share mutable buffers across asynchronous model operations.
- Always stop and free cleanly on chart unload or strategy reset.

---

## 10. Repository Map

- `kernel.cu` — CUDA DLL implementation, core training/inference runtime.
- `MQL5/Indicators/LSTM_RealTimePredictor.mq5` — primary practical MT5 integration example.
- `MQL5/Indicators/Examples/LSTM_PatternCompletion_Demo.mq5` — demo-oriented usage.
- `docs/index.html`, `docs/app.js`, `docs/lstm-flow.svg` — static documentation artifacts.
- `MQL5GPULibrary_LSTM.sln`, `MQL5GPULibrary_LSTM.vcxproj` — build orchestration.
- `LICENSE.txt` — licensing terms.

---

## 11. Troubleshooting Matrix

1. **Any API returns failure (`MQL_FALSE`)**  
   → call `DN_GetError`, log immediately, include context (handle, operation, dimensions).

2. **Training does not converge**  
   → reduce `lr`, inspect `grad_norm`, enable/adjust clipping, validate target scaling.

3. **Status remains idle after async start**  
   → verify initialization order and successful `DN_TrainAsync` return code.

4. **Prediction output malformed**  
   → re-check `in`, `seq_len`, `out_dim`, and allocated output buffer length.

5. **MT5 instability during long sessions**  
   → audit handle lifecycle and deinitialization discipline, reduce polling frequency.

---

## 12. License

This project is distributed under the **MIT License**. See `LICENSE.txt` for the full legal text.

---

### A small personal note
![Poděkování Aničce](ann.svg)


# MQL5GPULibrary_LSTM

> Autor této dokumentace: Tomáš Bělák
> Rozsah: produkční technická dokumentace pro CUDA-akcelerovaný trénink a inferenci LSTM ze systému MetaTrader 5 (MQL5) prostřednictvím DLL rozhraní.

![Licence: MIT](https://img.shields.io/badge/Licence-MIT-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-Aktivn%C3%AD-brightgreen.svg)
![MT5](https://img.shields.io/badge/MetaTrader%205-DLL%20API-blue.svg)

---

## 1. Shrnutí projektu

`MQL5GPULibrary_LSTM` je hybridní kvantitativně-inženýrský projekt, který zpřístupňuje CUDA-akcelerovaný běhový modul rekurentních neuronových sítí (s důrazem na LSTM, s dodatečnou podporou vrstev RNN/GRU) platformě **MetaTrader 5** skrze DLL API navržené pro praktické využití s nízkou latencí.

Projekt řeší běžné integrační úzké místo v systémech algoritmického obchodování:

- MQL5 vyniká v orchestraci strategií, exekuční logice a nástrojích nativně svázaných s grafy,
- zatímco vysokovýkonný trénink hlubokých sítí je vhodnější pro C++/CUDA.

Toto úložiště propojuje oba světy prostřednictvím:

1. **Správy životního cyklu modelu založené na handle** (`DN_Create`, `DN_Free`),
2. **Flexibilní konstrukce sítě** (`DN_AddLayerEx`, `DN_AddGRULayer`, `DN_AddRNNLayer`),
3. **Synchronního i asynchronního tréninku** (`DN_Train`, `DN_TrainAsync`),
4. **Podrobné telemetrie průběhu** (`DN_GetProgress*`, `DN_GetProgressAll`),
5. **Nástrojů pro persistenci stavu a návrat k předchozímu stavu** (`DN_SaveState`, `DN_GetState`, `DN_LoadState`, snímky),
6. **API pro dávkovou inferenci vhodného k integraci na úrovni indikátorů či EA** (`DN_PredictBatch`).

---

## 2. Architektura a cíle návrhu

### 2.1 Primární cíle návrhu

Běhový modul je navržen s ohledem na pět praktických omezení reálných obchodních systémů:

- **Deterministické vlastnictví:** každá instance modelu je reprezentována celočíselným handle, čímž se minimalizuje nejednoznačnost napříč kontexty.
- **Provozní odolnost:** každé volání API měnící stav lze ověřit a následně získat centralizovanou informaci o chybě (`DN_GetError`).
- **Odezva UI v MT5:** asynchronní trénink umožňuje náročné GPU úlohy bez zamrznutí logiky grafu nebo uživatelských interakcí.
- **Produkční telemetrie:** bohatá metadata o průběhu (epochy, mini-dávky, nejlepší MSE, odhadovaný čas dokončení, norma gradientu) umožňují informovaná rozhodnutí za běhu.
- **Bezpečnost experimentů:** snímky a serializace stavu snižují riziko při iteracích a podporují reprodukovatelné pracovní postupy.

### 2.2 Mapa hlavních komponent

- **`kernel.cu`**: jádro v CUDA/C++ implementující graf modelu, trénovací smyčku, telemetrii a exportované rozhraní DLL.
- **`MQL5/Indicators/*.mq5`**: integrační vrstva a praktické příklady použití v kontextu indikátorů MT5.
- **`docs/`**: statické vizualizace a interaktivní dokumentační podklady.
- **Soubory řešení a projektu (`.sln`, `.vcxproj`)**: reprodukovatelné vstupní body sestavení pro nástrojový řetězec Windows.

---

## 3. Model API a kontrakt životního cyklu

### 3.1 Základní operace životního cyklu

```cpp
int DN_Create();
void DN_Free(int h);
```

- `DN_Create` alokuje a zaregistruje nový kontext modelu a při úspěchu vrací kladný handle.
- `DN_Free` musí být voláno právě jednou pro každý platný handle, aby se uvolnily přidružené prostředky.

**Provozní pravidlo:** v MQL5 zacházejte s handle jako se spravovaným prostředkem; po volání `DN_Free` jej resetujte na neplatnou hodnotu, abyste předešli nechtěnému opětovnému použití.

### 3.2 Fáze konfigurace

```cpp
MQL_BOOL DN_SetSequenceLength(int h, int seq_len);
MQL_BOOL DN_SetMiniBatchSize(int h, int mbs);
MQL_BOOL DN_AddLayerEx(int h, int in, int out, int act, int ln, double drop);
MQL_BOOL DN_AddGRULayer(int h, int in, int out, double drop);
MQL_BOOL DN_AddRNNLayer(int h, int in, int out, double drop);
MQL_BOOL DN_SetGradClip(int h, double clip);
MQL_BOOL DN_SetOutputDim(int h, int out_dim);
```

#### Invarianty konfigurace

- `seq_len > 0`
- `mbs > 0`
- `0.0 <= drop < 1.0`
- `out_dim` musí odpovídat dimenzi cílového tenzoru.

Pro kanonické zásobníky architektury LSTM by měl být `DN_AddLayerEx` považován za primární konstruktor vrstvy.

### 3.3 Načítání dat a predikce

```cpp
MQL_BOOL DN_LoadBatch(int h, const double* X, const double* T,
int batch, int in, int out, int l);
MQL_BOOL DN_PredictBatch(int h, const double* X,
int batch, int in, int l, double* Y);
```

#### Kontrakt tvaru tenzorů (kritické)

- `in = seq_len * feature_dim`
- `len(X) = batch * in`
- `len(T) = batch * out`
- `len(Y) = batch * out_dim`
- Musí platit `in % seq_len == 0`

Porušení těchto předpokladů typicky vede k okamžitému selhání API a smysluplné chybové zprávě získatelné přes `DN_GetError`.

---

## 4. Režimy trénovacího jádra

### 4.1 Synchronní trénink

```cpp
MQL_BOOL DN_Train(int h, int epochs, double target_mse, double lr, double wd);
```

Synchronní režim použijte v případech, kdy:

- provádíte integraci v offline optimalizačních skriptech,
- je přijatelné deterministické blokování,
- preferujete jednoduchý řídicí tok.

### 4.2 Asynchronní trénink

```cpp
MQL_BOOL DN_TrainAsync(int h, int epochs, double target_mse, double lr, double wd);
int DN_GetTrainingStatus(int h);
void DN_GetTrainingResult(int h, double* out_mse, int* out_epochs);
void DN_StopTraining(int h);
```

Asynchronní režim použijte v případech, kdy:

- indikátor/EA musí pokračovat ve zpracování ticků nebo UI událostí,
- dlouhé trénovací běhy vyžadují viditelnost průběhu,
- architektura strategie vyžaduje neblokující chování.

#### Doporučená sémantika stavového automatu

- `0` → nečinný
- `1` → trénink probíhá
- `2` → dokončeno
- `-1` → chyba

---

## 5. Telemetrie průběhu (provozní inteligence)

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

### Průvodce interpretací telemetrie

- **`mse`**: aktuální stav optimalizace, na úrovni dávek zašuměný.
- **`best_mse`**: stabilizační kotva; konvergenci vyhodnocujte vůči této metrice.
- **`grad_norm`**: včasné varování před explodující/nestabilní optimalizací.
- **`eta_sec`**: provozní odhad pro UI a plánování úloh.

### Doporučené postupy pro polling v MT5

- Dotazujte se z `OnTimer` (např. 100–500 ms), nikoli při každém ticku.
- Preferujte `DN_GetProgressAll` pro snížení režie volání a zvýšení konzistence.
- Vzorkovaná telemetrická data uchovávejte pro následnou diagnostiku modelu.

---

## 6. Persistence stavu, návrat k předchozímu stavu a diagnostika

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

### Praktické produkční vzory

- **Politika checkpointů:** volání `DN_SaveState` v milnících konvergence a export stavového řetězce externě.
- **Bezpečné experimentování:** `DN_SnapshotWeights` před rizikovými změnami hyperparametrů; obnovení při detekci divergence.
- **Řešení selhání:** při jakémkoli `MQL_FALSE` okamžitě zavolejte `DN_GetError` a zaznamenanou zprávu uložte do logu s kontextem (handle, operace, dimenze).

---

## 7. Kompletní pracovní postup integrace v MT5

### 7.1 Inicializace (`OnInit`)

1. `DN_Create`
2. Konfigurace délky sekvence a velikosti mini-dávky
3. Sestavení zásobníku vrstev
4. Nastavení výstupní dimenze
5. Načtení trénovací dávky
6. Volitelně vytvoření snímku
7. Spuštění `DN_TrainAsync`

### 7.2 Běhová smyčka (`OnTimer`)

1. Dotaz na stav (`DN_GetTrainingStatus`)
2. Čtení telemetrie (`DN_GetProgressAll`)
3. Aktualizace UI grafu / logů
4. Po dokončení získání výsledku (`DN_GetTrainingResult`)
5. Volitelně spuštění inference (`DN_PredictBatch`)

### 7.3 Deinicializace (`OnDeinit`)

1. Pokud běží trénink, vyžádání zastavení (`DN_StopTraining`)
2. Uvolnění handle (`DN_Free`)
3. Vynulování / reset lokální proměnné handle

---

## 8. Sestavení a nasazení

### 8.1 Předpoklady pro sestavení

- Windows s Visual Studio schopným otevřít `.sln`
- NVIDIA CUDA Toolkit kompatibilní s vaší sadou nástrojů kompilátoru
- Cílové prostředí x64

### 8.2 Proces sestavení

1. Otevřete `MQL5GPULibrary_LSTM.sln`
2. Zvolte konfiguraci `Release | x64`
3. Sestavte řešení
4. Zkopírujte výslednou DLL do složky terminálových dat `MQL5\Libraries`

### 8.3 Nasazení v MT5

1. Zkopírujte soubory indikátorů z `MQL5/Indicators` do `MQL5\Indicators`
2. Zkompilujte v MetaEditoru
3. Povolte import DLL v nastavení MT5
4. Připojte indikátor ke grafu a ověřte inicializační logy

---

## 9. Doporučení pro výkon a stabilitu

### Hyperparametry

- Začněte konzervativně (`lr`, `epochs`) a laďte postupně.
- Používejte `target_mse` pro řízené předčasné zastavení.
- Aplikujte ořezávání gradientu (`DN_SetGradClip`), jakmile zaznamenáte výkyvy normy gradientu.

### Datový pipeline

- Před každým voláním `DN_LoadBatch` ověřte aritmetiku tvarů.
- Udržujte konzistentní strategii normalizace příznaků mezi tréninkem a inferencí.
- Kde je to možné, používejte pevné seedové hodnoty a deterministické předzpracování.

### Souběžnost a hygiena prostředků

- Každou strategii/model izolujte s unikátním handle.
- Nesdílejte mutabilní buffery napříč asynchronními operacemi modelu.
- Při uvolnění grafu nebo resetu strategie vždy čistě zastavte a uvolněte prostředky.

---

## 10. Mapa úložiště

- `kernel.cu` — implementace CUDA DLL, jádro běhového modulu pro trénink a inferenci.
- `MQL5/Indicators/LSTM_RealTimePredictor.mq5` — primární praktický příklad integrace v MT5.
- `MQL5/Indicators/Examples/LSTM_PatternCompletion_Demo.mq5` — demonstrační ukázka použití.
- `docs/index.html`, `docs/app.js`, `docs/lstm-flow.svg` — statické dokumentační podklady.
- `MQL5GPULibrary_LSTM.sln`, `MQL5GPULibrary_LSTM.vcxproj` — orchestrace sestavení.
- `LICENSE.txt` — licenční podmínky.

---

## 11. Matice řešení problémů

1. **Jakékoli API vrátí selhání (`MQL_FALSE`)**
→ zavolejte `DN_GetError`, okamžitě zalogujte, uveďte kontext (handle, operace, dimenze).

2. **Trénink nekonverguje**
→ snižte `lr`, zkontrolujte `grad_norm`, povolte/upravte ořezávání, ověřte škálování cílových hodnot.

3. **Stav zůstává nečinný po spuštění asynchronního tréninku**
→ ověřte pořadí inicializace a úspěšný návratový kód `DN_TrainAsync`.

4. **Výstup predikce je nesprávně formátovaný**
→ znovu zkontrolujte `in`, `seq_len`, `out_dim` a délku alokovaného výstupního bufferu.

5. **Nestabilita MT5 při dlouhých sezeních**
→ proveďte audit životního cyklu handle a disciplíny deinicializace, snižte frekvenci pollingu.

---

## 12. Licence

Tento projekt je distribuován pod licencí **MIT**. Úplné znění licence naleznete v souboru `LICENSE.txt`.

---

### Malá osobní poznámka
![Poděkování Aničce](ann.svg)
