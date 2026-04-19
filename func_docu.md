# `kernel.cu` – extrémně precizní dokumentace funkcí a parametrů

> Zdroj: `kernel.cu` v projektu `MQL5GPULibrary_LSTM`.
> 
> Cíl tohoto dokumentu: vytvořit praktickou „mapu“ všech zásadních funkcí, jejich parametrů, datových toků a návratových hodnot – od CUDA kernelů přes vrstvy modelu až po exportované DLL API (`DN_*`) pro MQL5.

---

## 1) Konvence a základní typy

## 1.1 Datové rozložení matic
- Celý soubor používá **column-major** rozložení (nativní pro cuBLAS).
- Vzorec indexace: prvek `(i, j)` je na `base[i + j * rows]`.
- V dokumentaci níže jsou rozměry zapisovány jako `[rows x cols]`.

## 1.2 Základní typy
- `MQL_BOOL` (`int`) – logický návrat pro DLL API (`MQL_TRUE=1`, `MQL_FALSE=0`).
- `TrainingState`:
  - `TS_IDLE = 0`
  - `TS_TRAINING = 1`
  - `TS_COMPLETED = 2`
  - `TS_ERROR = -1`

---

## 2) Chybové a pomocné funkce

## 2.1 `SetError(const wchar_t* fmt, ...)`
**Účel:** uloží poslední chybové hlášení do globálního `g_last_err_w` (thread-safe přes mutex).

**Parametry:**
- `fmt`: formátovací řetězec (wide char).
- `...`: variadické argumenty jako u `printf`.

**Poznámka:** volaná z `CUDA_CHECK_*`, `CUBLAS_CHECK_*`, `CURAND_CHECK_*` maker i z logiky tříd.

## 2.2 `RoundUpEven(size_t n)`
**Účel:** zaokrouhlí `n` na nejbližší sudé číslo nahoru.

**Parametry:**
- `n`: vstupní velikost.

**Návrat:** sudé `size_t`.

## 2.3 `toU32(size_t v)`
**Účel:** bezpečný převod na `unsigned` s saturací na `UINT_MAX`.

**Parametry:**
- `v`: vstupní velikost.

**Návrat:** `unsigned`.

---

## 3) GPU RAII struktury

## 3.1 `template<typename T> struct GPUMemory`
Wrapper nad `cudaMalloc/cudaFree` s evidencí `count/capacity`.

### Metody

### `MQL_BOOL alloc(size_t n)`
- Alokuje/resize GPU buffer na `n` prvků typu `T`.
- Reuse, pokud `n <= capacity`.

**Parametry:**
- `n`: počet prvků (ne bajtů).

**Návrat:**
- `MQL_TRUE`: alokace/resize OK.
- `MQL_FALSE`: chyba (typicky OOM; vyplní globální error stav).

### `MQL_BOOL zero()`
- Synchronní nulování celého aktivního bufferu (`count*sizeof(T)`).

### `MQL_BOOL zeroAsync(cudaStream_t s)`
- Asynchronní nulování v proudu `s`.

**Parametry:**
- `s`: CUDA stream, kde běží memset.

### `void free()`
- Uvolní alokaci (`cudaFree`), resetuje metadata.

### `size_t bytes() const`
- Vrací aktivní velikost v bajtech (`count*sizeof(T)`).

## 3.2 `struct GPUContext`
Per-thread GPU kontext (stream + cuBLAS + cuRAND).

### `MQL_BOOL Init(int device = 0)`
**Parametry:**
- `device`: CUDA device index.

**Co dělá:**
1. `cudaSetDevice`
2. `cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)`
3. `cublasCreate + cublasSetStream`
4. `curandCreateGenerator + seed + curandSetStream`

### `void Destroy()`
- Bezpečné zrušení `curand`, `cublas`, streamu.

---

## 4) Training progress (lock-free čtení pro MQL5)

## 4.1 `struct TrainingProgress`
Sada atomických polí pro online reporting.

### `void Reset()`
- Reset všech counters a metrik na výchozí hodnoty.

### `void UpdateTiming()`
- Aktualizuje elapsed/ETA podle `progress_pct` a času od startu.

---

## 5) Device helper funkce

## 5.1 `warpReduceSumFloat(float v)`
- Warp-level sum redukce přes `__shfl_down_sync`.

## 5.2 `d_sigmoid(float x)`
- Device sigmoid: `1 / (1 + exp(-x))`.

---

## 6) CUDA kernely – detailní parametry

Níže je každá kernel funkce z `kernel.cu` se semantics parametrů.

## 6.1 Konverzní kernely

### `kCopyD2F_vec4(int n, const double* in, float* out)`
**Účel:** převod `double -> float` vektorově po 4, grid-stride loop.

**Parametry:**
- `n`: počet prvků.
- `in`: vstupní pole (`double`, délka `n`).
- `out`: výstupní pole (`float`, délka `n`).

### `kCopyF2D_vec4(int n, const float* in, double* out)`
Stejné jako výše, opačný směr `float -> double`.

## 6.2 LSTM gate kernely

### `kLSTMGatesForward(...)`
```cpp
(int hidden_size, int batch,
 const float* gates_raw, const float* c_prev,
 float* c_new, float* h_new,
 float* f_cache, float* i_cache, float* g_cache, float* o_cache)
```
**Účel:** z preaktivací gate spočítá `f,i,g,o`, nový cell state `c_new` a hidden state `h_new`.

**Parametry:**
- `hidden_size`: počet neuronů v hidden stavu.
- `batch`: mini-batch size.
- `gates_raw`: sloučené preaktivace 4 bran (`[4H x B]` v major uspořádání podle implementace).
- `c_prev`: předchozí cell state `[H x B]`.
- `c_new`: nový cell state `[H x B]`.
- `h_new`: nový hidden state `[H x B]`.
- `f_cache/i_cache/g_cache/o_cache`: cache aktivací pro backward.

### `kLSTMGatesBackward(...)`
```cpp
(int hidden_size, int batch,
 const float* dh, const float* dc_next,
 const float* c_prev, const float* c_cur,
 const float* f_cache, const float* i_cache,
 const float* g_cache, const float* o_cache,
 float* dc_prev_out, float* dgates_raw)
```
**Účel:** spočítá gradienty LSTM bran + `dc_prev`.

**Parametry navíc:**
- `dh`: gradient z hidden větve pro aktuální čas.
- `dc_next`: gradient cell state z budoucího času.
- `c_cur`: aktuální `c_t`.
- `dc_prev_out`: gradient do `c_{t-1}`.
- `dgates_raw`: gradienty preaktivací všech 4 bran.

## 6.3 GRU gate kernely

### `kGRUGatesForward(...)`
```cpp
(int hidden_size, int batch,
 const float* zr_raw, const float* nx_raw, const float* nh_raw,
 const float* h_prev,
 float* h_new,
 float* z_cache, float* r_cache, float* n_cache, float* nh_cache)
```
**Účel:** GRU forward: výpočet `z`, `r`, kandidáta `n` a `h_new`.

### `kGRUGatesBackward(...)`
```cpp
(int hidden_size, int batch,
 const float* dh,
 const float* z_cache, const float* r_cache,
 const float* n_cache, const float* nh_cache,
 const float* h_prev,
 float* dh_prev_direct,
 float* dz_raw, float* dr_raw, float* dnx_raw, float* dnh_raw)
```
**Účel:** GRU backward pro lokální derivace a větvení gradientu.

## 6.4 Simple RNN kernely

### `kRNNForward(int hidden_size, int batch, const float* preact, float* h_new)`
- `h_new = tanh(preact)`.

### `kRNNBackward(int hidden_size, int batch, const float* dh, const float* h_cur, float* dpreact)`
- `dpreact = dh * (1 - h^2)`.

## 6.5 Tensor utility kernely

### `kAddBiasInplace(int rows, int cols, float* A, const float* bias)`
- Přičte bias po řádcích: `A[row + col*rows] += bias[row]`.

### `kConcatHX(int hidden_size, int input_size, int batch, const float* h_prev, const float* x, float* hx)`
- Slepí `[h_prev; x]` do `hx` pro gate GEMM.

### `kSplitDHX(int hidden_size, int input_size, int batch, const float* dhx, float* dh_prev, float* dx)`
- Rozdělí gradient z concatenace zpět na `dh_prev` a `dx`.
- `dh_prev`/`dx` mohou být `nullptr` (podle potřeby větve).

### `kBiasLinear(int rows, int cols, float* A, const float* bias)`
- Funkčně ekvivalentní `kAddBiasInplace`; používá se pro lineární output.

### `kAddInplace(int n, float* dst, const float* src)`
- `dst[i] += src[i]`.

## 6.6 Loss / norm / optimizer kernely

### `kMSEGrad(int n, const float* y, const float* t, float* d)`
- Gradient MSE: `d = 2*(y - t)/n`.

### `kMSEReduceWarp(int n, const float* y, const float* t, float* out_sum)`
- Redukuje součet kvadratické chyby `sum((y-t)^2)` do `out_sum`.

### `kL2NormReduceWarp(int n, const float* buf, float* out_sum)`
- Redukuje `sum(buf^2)` pro L2 norm diagnostiku/clipping.

### `kAdamW(...)`
```cpp
(int n, float* p, float* m, float* v,
 const float* g,
 float lr, float b1, float b2, float eps,
 float wd, float c1, float c2, float clip_val)
```
**Účel:** update parametrů AdamW (s bias-correction přes `c1/c2`).

**Parametry:**
- `p`: parametry.
- `m`, `v`: 1. a 2. moment.
- `g`: gradient.
- `lr`: learning rate.
- `b1`, `b2`: decay momentů.
- `eps`: numerická stabilita.
- `wd`: weight decay.
- `c1`, `c2`: bias-correction koeficienty.
- `clip_val`: v kódu nepoužito (globální clipping se řeší jinde).

### `kScaleGradients(int n, float* grad, float scale)`
- Uniformní škálování gradientů (globální norm clipping).

## 6.7 Sekvenční/mini-batch kernely

### `kGatherTimesteps(int hidden_size, int batch, int seq_len, float* dst, const float* const* src_ptrs)`
- Sbalí timestep buffery (i ne-kontiguální) do jednoho contiguous bloku.

### `kDropoutForward(int n, float* A, const float* rand_vals, float drop_rate, unsigned char* mask)`
- Inverted dropout: zachované prvky škáluje `1/(1-drop_rate)`, ostatní nulují.

### `kDropoutBackward(int n, float* dA, const unsigned char* mask, float drop_rate)`
- Backward přes stejnou masku + inverted scaling.

### `kTransposeToTimestep(int batch, int seq_len, int feat, const float* src, float* dst)`
- Převod z `[batch, seq, feat]` do timestep-major layoutu používaného vrstvami.

### `kGatherTransposeSeq(int mb_size, int seq_len, int feat, const int* indices, const float* src, float* dst)`
- Gather z globálního datasetu podle mini-batch indexů + současná transpozice do timestep-major.

### `kGatherRows(int dim, int batch, const float* src, const int* indices, float* dst)`
- Gather řádků (`indices[b]`) ze zdroje do minibatch bufferu.

### `kMinMaxReduce256(int n, const float* buf, float* out_min, float* out_max)`
- Block-level min/max reduce (256 vláken/blok), mezivýstup po blocích.

---

## 7) Abstrakce rekurentních vrstev

## 7.1 `enum class SequenceLayerType`
- `RNN`, `GRU`, `LSTM`, `ATTENTION`.

## 7.2 `class RecurrentLayer` (abstraktní rozhraní)

### Povinné metody implementací
- `Forward(...)`
- `Backward(...)`
- `Update(...)`
- `SaveBest()` / `RestoreBest()`
- `GetOutputH(...)`
- `GetInputSize()`, `GetHiddenSize()`, `GetDropoutRate()`, `GetType()`, `GetGateDim()`
- `Init(...)`, `InitFromData(...)`, `FreeAll()`

### Společná GPU pole (v base)
- Parametry: `W`, `b`.
- AdamW state: `mW`, `vW`, `mb`, `vb`.
- Gradienty: `dW`, `db`.
- Snapshot: `W_best`, `b_best`.
- Cache (časové kroky): `hx_cache`, `f/i/g/o_cache`, `c_cache`, `h_cache`.
- Dropout: `dropout_mask`, `dropout_mask_valid`, `h_drop_cache`, `dropout_rand`.

---

## 8) Konkrétní vrstvy

## 8.1 `LSTMLayer`

### `Init(in_sz, hid_sz, drop_rate)`
- Alokace všech parametrů.
- Inicializace vah Gauss (`stddev ~ sqrt(2/(concat+gate_dim))`).
- Bias: forget gate přednastaven na `1.0`.

### `InitFromData(in_sz, hid_sz, drop_rate)`
- Alokuje struktury bez náhodné inicializace vah (následně se typicky nahrávají serializovaná data).

### `Forward(...)`
**Klíčové parametry:**
- `X`: vstup v timestep-major formátu.
- `seq_len`, `batch`.
- `training`: zapnutí dropout větve.
- `curand_gen`: generátor random pro dropout masky.

### `Backward(...)`
- `dh_last`: gradient ze ztráty do posledního kroku vrstvy.
- `dh_above_seq`: gradient ze „shora“ (další vrstva) pro všechny kroky.
- `ones_ptr`: pomocný vektor jedniček (např. pro bias GEMV/GEMM postupy).

### `Update(lr, c1, c2, wd, clip, stream)`
- AdamW update parametrů vrstvy.

### `SaveBest()` / `RestoreBest()`
- Snapshot a rollback nejlepších vah.

## 8.2 `GRULayer`
Metodicky stejné API jako `LSTMLayer`, ale vnitřní gate logika je GRU (`z`, `r`, `n`) a odpovídající cache buffery.

## 8.3 `RNNLayer`
Nejjednodušší rekurentní varianta (`tanh`), stejné lifecycle API (`Init/Forward/Backward/Update/Snapshot`).

## 8.4 `AttentionLayer` (stub)
- V souboru je definovaný základ, ale attention není hlavní produkční výpočetní větev jako LSTM/GRU/RNN.

---

## 9) Output vrstva (`OutputLayer`)

## Hlavní metody
- `Init(int in_d, int out_d)` – alokace + random init.
- `InitFromData(int in_d, int out_d)` – alokace bez random init.
- `Forward(cublasHandle_t, cudaStream_t, const float* H, float* Y, int batch)` – lineární projekce.
- `Backward(cublasHandle_t, cudaStream_t, const float* dY, const float* H, float* dH, int batch)` – gradienty output vrstvy.
- `Update(float lr, float c1, float c2, float wd, float clip, cudaStream_t)` – AdamW update.
- `SaveBest()/RestoreBest()` – snapshot management.

**Důležité parametry:**
- `H`: vstup hidden representation `[in_dim x batch]`.
- `Y`: predikce `[out_dim x batch]`.
- `dY`, `dH`: gradienty output/input.

---

## 10) Learning-rate scheduler

## `class LRScheduler` (abstrakce)
- `virtual float GetLR(int step, float base_lr) const = 0;`

## `class CosineWarmupScheduler` (implementace)
- Dynamické LR s warmup + cosine decay.

**Parametry GetLR:**
- `step`: aktuální trénovací krok.
- `base_lr`: základní LR.

---

## 11) `SequenceModel` – orchestrace sítě

Třída spravuje:
- vektor rekurentních vrstev,
- output vrstvu,
- trénink/predikci,
- progress reporting,
- async thread lifecycle,
- serializaci stavu modelu.

## 11.1 Konfigurační metody
- `SetGradClip(float v)` – nastaví globální norm clipping.
- `SetSequenceLength(int sl)` – min. 1.
- `SetMiniBatchSize(int mbs)` – min. 1.
- `AddLayer(...)` / `AddGRULayer(...)` / `AddRNNLayer(...)` – přidání vrstev.
- `SetOutputLayer(int out_dim)` – output projekce.

## 11.2 Data a inference
- `LoadBatch(const double* X, const double* T, int batch, int in, int out, int l)`
  - nahraje a převede trénovací data do GPU bufferů.
- `PredictBatch(const double* X, double* Y, int batch, int in, int out, int l)`
  - jednorázová inference batch vstupu.

## 11.3 Trénink
- `Train(int epochs, double lr, double b1, double b2, double eps, double wd, int patience)`
- `TrainAsync(...)`
- `StopTraining()`

## 11.4 Diagnostika a progress
- `GetStatus()`, `GetResult(...)`
- `GetProgress*()` family (epoch, minibatch, LR, MSE, best MSE, grad norm, ETA...)
- `GetLayerCount()`, `GetLayerWeightNorm(...)`, `GetGradNorm()`

## 11.5 Snapshot/serializace
- `SnapshotWeights()`, `RestoreWeights()`
- `SaveState()`, `GetState(...)`, `LoadState(...)`

---

## 12) Globální mapování instancí modelů

- `g_nets: map<int, shared_ptr<SequenceModel>>`.
- `g_id`: generátor handle ID.
- Synchronizace přes `g_map_mtx`.

Pomocné funkce:
- `FindAndLockExclusive(int h, std::unique_lock<std::shared_mutex>& lk)`
- `FindNetNoLock(int h)`
- `ComputeDeviceL2Norm(cudaStream_t, const float* buf, int n)`

---

## 13) Exportované DLL API (`DN_*`) – přesná referenční část

Následující funkce jsou **externě volatelné z MQL5**.

## 13.1 Lifecycle

### `int DN_Create()`
- Vytvoří `SequenceModel`, vrátí handle (`>0`) nebo `0` při chybě.

### `void DN_Free(int h)`
- Uvolní model pro daný handle.

## 13.2 Topologie a konfigurace

### `MQL_BOOL DN_SetSequenceLength(int h, int seq_len)`
- Nastavení délky sekvence.

### `MQL_BOOL DN_SetMiniBatchSize(int h, int mbs)`
- Nastavení velikosti minibatche.

### `MQL_BOOL DN_AddLayerEx(int h, int in, int out, int act, double drop)`
- Přidání vrstvy (default LSTM branch dle interní logiky).

**Parametry:**
- `h`: handle modelu.
- `in`: vstupní dimenze vrstvy.
- `out`: hidden dimenze vrstvy.
- `act`: identifikátor typu/aktivace (v implementaci se mapuje na typ vrstvy).
- `drop`: dropout rate (`0..1`).

### `MQL_BOOL DN_AddGRULayer(int h, int in, int out, double drop)`
- Přidá GRU vrstvu.

### `MQL_BOOL DN_AddRNNLayer(int h, int in, int out, double drop)`
- Přidá simple RNN vrstvu.

### `MQL_BOOL DN_SetGradClip(int h, double clip)`
- Nastaví globální gradient clipping.

### `MQL_BOOL DN_SetOutputDim(int h, int out_dim)`
- Nastaví výstupní dimenzi lineární hlavy.

## 13.3 Data / inference / snapshot

### `MQL_BOOL DN_LoadBatch(int h, const double* X, const double* T, int batch, int in, int out, int l)`
- Nahraje trénovací batch (`X`, `T`) a metadata dimenzí.

### `MQL_BOOL DN_PredictBatch(int h, const double* X, double* Y, int batch, int in, int out, int l)`
- Predikce bez tréninku.

### `MQL_BOOL DN_SnapshotWeights(int h)`
- Uloží snapshot aktuálních vah.

### `MQL_BOOL DN_RestoreWeights(int h)`
- Obnoví poslední snapshot.

## 13.4 Trénink sync/async

### `MQL_BOOL DN_Train(int h, int epochs, double lr, double b1, double b2, double eps, double wd, int patience)`
- Blokující trénink.

### `MQL_BOOL DN_TrainAsync(int h, int epochs, double lr, double b1, double b2, double eps, double wd, int patience)`
- Neblokující trénink na background vlákně.

### `int DN_GetTrainingStatus(int h)`
- Vrací `TrainingState` (`-1/0/1/2`).

### `void DN_GetTrainingResult(int h, double* out_mse, int* out_epochs_done)`
- Vrátí agregovaný výsledek tréninku.

### `void DN_StopTraining(int h)`
- Nastaví stop flag (graceful stop).

## 13.5 Progress API

Jednotlivé getters:
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

### `MQL_BOOL DN_GetProgressAll(...)`
Vrací vše v jednom volání přes ukazatele:
```cpp
(int h,
 int* out_epoch, int* out_total_epochs,
 int* out_mb, int* out_total_mb,
 double* out_lr, double* out_mse, double* out_best_mse,
 double* out_grad_norm, double* out_pct,
 double* out_elapsed_sec, double* out_eta_sec)
```

## 13.6 Diagnostika a stav

### `int DN_GetLayerCount(int h)`
- Počet vrstev + output layer.

### `double DN_GetLayerWeightNorm(int h, int l)`
- L2 norma vah vybrané vrstvy.

### `double DN_GetGradNorm(int h)`
- Celková gradientová norma.

### `int DN_SaveState(int h)`
- Serializuje stav modelu interně (do string bufferu v objektu).

### `MQL_BOOL DN_GetState(int h, char* buf, int max_len)`
- Zkopíruje serializovaný stav do volajícího bufferu.

### `MQL_BOOL DN_LoadState(int h, const char* buf)`
- Deserializuje stav modelu ze stringu.

### `void DN_GetError(short* buf, int len)`
- Vrátí poslední wide-char chybovou zprávu (zkrácenou na `len`).

---

## 14) Doporučení pro bezpečné použití API

1. Volat vždy v pořadí: `DN_Create -> konfigurace vrstev -> DN_SetOutputDim -> DN_LoadBatch`.
2. Dimenze `in/out/l/seq_len` musí odpovídat fyzickému layoutu dat v MQL5.
3. Při async tréninku pravidelně pollovat `DN_GetTrainingStatus` + `DN_GetProgressAll`.
4. Před shutdown volat `DN_StopTraining` (pokud běží async) a následně `DN_Free`.
5. Po chybě číst `DN_GetError` kvůli diagnostice CUDA/cuBLAS/cuRAND.

---

## 15) Quick reference (zkrácený checklist)

- **Kernely pro data:** `kCopy*`, `kTransposeToTimestep`, `kGather*`.
- **Kernely pro recurrent math:** `kLSTMGates*`, `kGRUGates*`, `kRNN*`.
- **Kernely pro trénink:** `kMSEGrad`, `kMSEReduceWarp`, `kL2NormReduceWarp`, `kScaleGradients`, `kAdamW`.
- **Externí API:** `DN_*` (create/free/config/load/predict/train/progress/state/error).

Tento soubor je navržen jako „single source of truth“ pro vývojáře MQL5 i C++/CUDA integraci.
