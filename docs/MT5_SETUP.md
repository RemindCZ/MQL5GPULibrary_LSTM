# Automatická instalace DNForecast do MT5

[Stáhnout instalátor DNForecast 3.0.0 pro Windows x64](https://github.com/RemindCZ/MQL5GPULibrary_LSTM/releases/download/v3.0.0/DNForecast-3.0.0-MT5-Setup.exe)

Spusťte EXE dvojklikem pod stejným uživatelem Windows, který používá MetaTrader 5. Instalace probíhá bez průvodce, volby složky a stahování dalších souborů. Podporované prostředí je Windows 10 nebo 11 x64 s klasickým terminálem MT5. Instalátor používá součást Windows .NET Framework 4.8. Po dokončení se okno samo zavře.

V Navigátoru MT5 zvolte u **Indikátory → Obnovit** a přidejte **DNForecast_Async** na graf AUDUSD H1. Při prvním použití potvrďte import DLL pro tento indikátor. Pro výzkumnou TCN nastavte `InpModelFile` na `DNForecast\AUDUSD-H1-TCN.onnx`. Funkčnost knihovny a modelu můžete ověřit spuštěním skriptu **DNForecast_Smoke**. Indikátor potřebuje tržní historii podle zvolených symbolů.

## Výběr terminálů

- Běžná instalace vyhledá datové složky aktuálního uživatele v `%APPDATA%\MetaQuotes\Terminal`. Ověří vazbu souboru `origin.txt` na existující `terminal64.exe` a složku `MQL5`. Při více platných instalacích obslouží všechny.
- Přenosné terminály najde také podle běžících procesů aktuální relace a registrovaných instalačních složek. Pro samostatnou přenosnou kopii můžete umístit EXE vedle jejího `terminal64.exe`. Při dvojkliku odtud instaluje pouze do této kopie.
- Před prvním spuštěním platformy mohou datové složky ještě chybět. Okno instalátoru v tom případě počká, až uživatel poprvé otevře MT5.
- Standardní datová složka potřebuje pouze oprávnění aktuálního uživatele. Přenosná kopie vyžaduje zápis do své složky.

[Datové složky a režim portable podle MetaQuotes](https://www.metatrader5.com/en/terminal/help/start_advanced/start)

## Obsah a aktualizace

Instalátor kontroluje SHA-256 vestavěného balíčku i jednotlivých instalovaných souborů. DLL a ONNX Runtime umisťuje do `MQL5\Libraries`, indikátor do `Indicators`, ověřovací skript do `Scripts` a importy do `Include`. Model, jeho metadata, české PDF a licenční texty jsou v `MQL5\Files\DNForecast`. Chybějící závislosti Visual C++ pro ONNX doplní přímo vedle DLL. Použije dostupnou vyhovující nebo novější verzi z daného umístění či Windows.

Při opakovaném spuštění ověří existující soubory. Shodné soubory si zachovají čas změny. Nahrazené verze uloží ve složce **DNForecast.Backups** v příslušné datové složce terminálu. Pokud MT5 drží měněný soubor otevřený, instalátor jej zkusí znovu po uvolnění. Okno můžete ponechat otevřené až do běžného zavření MT5. Rozpracované změny po chybě vrátí zpět. Po výpadku dokončí obnovu následující spuštění instalátoru.

Protokoly jsou v `%LOCALAPPDATA%\Remind\DNForecast\Logs`. Obsahují cílové cesty a výsledky instalace. Instalace se týká souborů programu. O přidání na konkrétní graf a povolení DLL rozhoduje uživatel přímo v MT5.

## Bezobslužné nasazení

```powershell
$setup = '.\DNForecast-3.0.0-MT5-Setup.exe'
$p = Start-Process $setup -ArgumentList '/quiet' -Wait -PassThru
$p.ExitCode
```

Parametr `/target "C:\cesta k datům MT5"` určí jedinou datovou složku obsahující MQL5. `/quiet` potlačí okno, `/log "C:\cesta\setup.log"` zvolí protokol. S `/no-wait` lze zamčený soubor vyřešit opakováním z vlastního nasazovacího systému. Kód 0 znamená dokončeno, 1 chybu, 2 žádný nalezený terminál, 32 soubor používaný jiným procesem, 1223 ukončení uživatelem.

## Sestavení instalátoru

Vývojář potřebuje publikovaný ZIP v3.0.0, kompilátor .NET Framework x64 a distribuovatelné knihovny Visual C++ 2022 x64. Build ověří SHA-256 zdrojového ZIPu. Vlastní umístění Microsoft.VC143.CRT lze zadat přes `-CrtDirectory`.

```powershell
.\tools\build_mt5_installer.ps1 -ReleaseZip .\DNForecast-3.0.0-win-x64.zip
.\tests\test_mt5_installer.ps1 -Setup .\output\mt5-installer\DNForecast-3.0.0-MT5-Setup.exe -ReleaseZip .\DNForecast-3.0.0-win-x64.zip
```
