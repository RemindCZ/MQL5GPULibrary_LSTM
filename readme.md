# MQL5GPULibrary_LSTM

MQL5GPULibrary_LSTM je dynamická knihovna pro Windows x64, která umožňuje trénování a inferenci LSTM sítí přes CUDA přímo z prostředí MetaTrader 5 (MQL5).

Knihovna je navržená pro nízkou latenci a průběžný provoz. Trénování je možné spustit asynchronně, takže terminál může dál zpracovávat tick data a strategii.

## Hlavní vlastnosti

- akcelerace výpočtů na NVIDIA GPU přes CUDA
- asynchronní trénování ve worker vlákně
- vícevstvé LSTM s konfigurací rozměrů po vrstvách
- gradient clipping
- snapshot a obnova vah
- serializace a deserializace stavu modelu

## Požadavky

- Windows 64-bit
- MetaTrader 5 64-bit
- NVIDIA GPU
- nainstalovaný CUDA runtime kompatibilní s použitou verzí DLL

V prostředí, kde běží MT5, musí být dostupné příslušné CUDA knihovny (například `cudart64_*.dll`, `cublas64_*.dll`, `curand64_*.dll`).

## Instalace

1. Zkopírujte `MQL5GPULibrary_LSTM.dll` do složky `MQL5\Libraries` v datovém adresáři MetaTraderu.
2. V MT5 zapněte povolení DLL importů v nastavení Expert Advisors.
3. V MQL5 skriptu nebo EA použijte `#import` se správnými signaturami funkcí.

## Exportované API

Níže je přehled exportů z DLL podle aktuální implementace.

### Správa instance

- `int DN_Create()`
- `void DN_Free(int h)`

### Konfigurace

- `bool DN_SetSequenceLength(int h, int seq_len)`
- `bool DN_SetMiniBatchSize(int h, int mbs)`
- `bool DN_AddLayerEx(int h, int in, int out, int act, int ln, double drop)`
- `bool DN_SetGradClip(int h, double clip)`

### Data a inference

- `bool DN_LoadBatch(int h, const double* X, const double* T, int batch, int in, int out, int l)`
- `bool DN_PredictBatch(int h, const double* X, int batch, int in, int l, double* Y)`

### Trénování

- `bool DN_TrainAsync(int h, int epochs, double target_mse, double lr, double wd)`
- `int DN_GetTrainingStatus(int h)`
- `void DN_GetTrainingResult(int h, double* out_mse, int* out_epochs)`
- `void DN_StopTraining(int h)`

Stavy trénování:

- `0` nečinný stav
- `1` běží trénování
- `2` trénování dokončeno
- `-1` chyba nebo neplatný handle

### Stav modelu a diagnostika

- `bool DN_SnapshotWeights(int h)`
- `bool DN_RestoreWeights(int h)`
- `int DN_GetLayerCount(int h)`
- `double DN_GetLayerWeightNorm(int h, int l)`
- `double DN_GetGradNorm(int h)`
- `int DN_SaveState(int h)`
- `bool DN_GetState(int h, char* buf, int max_len)`
- `bool DN_LoadState(int h, const char* buf)`
- `void DN_GetError(short* buf, int len)`

Poznámka: některé diagnostické funkce mohou vracet pouze výchozí hodnoty podle aktuální verze implementace.

## Základní příklad pro MQL5

```mq5
#import "MQL5GPULibrary_LSTM.dll"
int  DN_Create();
void DN_Free(int h);
bool DN_TrainAsync(int h, int epochs, double target_mse, double lr, double wd);
int  DN_GetTrainingStatus(int h);
void DN_StopTraining(int h);
#import

int g_net = 0;

int OnInit()
{
   g_net = DN_Create();
   if(g_net <= 0)
      return INIT_FAILED;

   if(!DN_TrainAsync(g_net, 2000, 0.001, 0.0005, 0.0001))
      return INIT_FAILED;

   EventSetTimer(1);
   return INIT_SUCCEEDED;
}

void OnTimer()
{
   int status = DN_GetTrainingStatus(g_net);
   if(status == 2)
   {
      Print("Trénování dokončeno");
      EventKillTimer();
   }
}

void OnDeinit(const int reason)
{
   DN_StopTraining(g_net);
   DN_Free(g_net);
}
```

## Build ze zdroje

Projekt obsahuje řešení pro Visual Studio a CUDA (`.sln`, `.vcxproj`, `kernel.cu`).

Doporučený postup:

1. Otevřete `MQL5GPULibrary_LSTM.sln` ve Visual Studio.
2. Zvolte konfiguraci `Release` a platformu `x64`.
3. Ověřte nastavení cesty k CUDA Toolkit.
4. Proveďte build DLL.

## Licence

Projekt je licencovaný pod MIT licencí. Podrobnosti jsou v `LICENSE.txt`.
