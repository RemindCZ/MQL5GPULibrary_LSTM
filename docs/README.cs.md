# MQL5GPULibrary_LSTM — český návod

Autor: **Tomáš Bělák**. Knihovna propojuje MetaTrader 5 s CUDA implementací LSTM, GRU a RNN. Podporuje více vrstev, AdamW, dropout, ořezávání gradientů, asynchronní trénink a ukládání vah.

## Sestavení

Potřebujete 64bitová Windows, Visual Studio 2022 s C++ nástroji, kompatibilní NVIDIA CUDA Toolkit a ovladač, NVIDIA GPU, CMake 3.24+ a Python 3 pro testy.

```powershell
cmake -S . -B build -A x64 -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --output-on-failure
```

Architekturu `86` upravte podle GPU; odpovídá zde ověřené RTX 3060. DLL vznikne v `build/Release`. [Podrobnosti sestavení a testů](BUILDING.md).

## Instalace do MT5

1. DLL zkopírujte do `MQL5/Libraries` v datové složce terminálu.
2. Společný soubor `MQL5/Include/MQL5GPULibrary.mqh` zkopírujte do `MQL5/Include`.
3. Indikátory a skript `LSTM_SmokeTest.mq5` umístěte do odpovídajících složek. Soubor `LSTM_Pure_MTF_EA.mq5` patří při použití jako EA do `MQL5/Experts`.
4. Zajistěte dostupnost CUDA runtime, cuBLAS a cuRAND pro proces terminálu, povolte import DLL a zdroje zkompilujte v MetaEditoru.
5. Spusťte smoke test se syntetickými daty. Skript neprovádí obchody.

## Důležité podmínky rozhraní

- `DN_AddLayerEx`: `act=0` LSTM, `1` GRU, `2` RNN. `ln` musí být nula; layer normalization není implementována.
- `layout=0`; vstupní pole má pořadí vzorek → čas → příznak. `in=délka_sekvence*počet_příznaků`.
- Rozměry musí být kladné, dropout v intervalu 0 až méně než 1, hodnoty konečné a reprezentovatelné jako float32.
- Během tréninku jsou dostupné stav, telemetrie, výsledky a požadavek na zastavení. Predikce, změny a serializace stejného modelu vracejí neúspěch bez čekání.
- Pořadí desetinných výstupů `DN_GetProgressAll` začíná **lr, mse, best_mse**.
- `DN_Free` požádá o zastavení a počká na dokončení pracovníka. Poté handle nepoužívejte.
- Neplatný checkpoint zachová původní model. Úspěšné načtení obnoví váhy a architekturu, ale vymaže trénovací data a stav optimalizátoru.

[Úplný kontrakt API](API.md) · [Přehled oprav](../CHANGELOG.md)

Testy knihovny neověřují výnosnost obchodní strategie. Způsob normalizace, časové rozdělení trénovacích/testovacích dat a validaci modelu je potřeba řešit v konkrétní aplikaci.
