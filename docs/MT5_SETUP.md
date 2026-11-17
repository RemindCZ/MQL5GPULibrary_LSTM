# Automatická instalace DNForecast do MT5

[Stáhnout instalátor DNForecast 3.1.0 pro Windows x64](https://github.com/RemindCZ/MQL5GPULibrary_LSTM/releases/download/v3.1.0/DNForecast-3.1.0-MT5-Setup.exe)

Spusťte EXE dvojklikem pod stejným uživatelem Windows, který používá MetaTrader 5. Vyberte instalaci MT5, zkontrolujte zobrazenou datovou cestu a klikněte na **Nainstalovat**. Tlačítkem **Vybrat jinou složku…** můžete určit vlastní umístění. Po potvrzení instalátor provede všechny kroky offline. Podporované prostředí je Windows 10 nebo 11 x64 s klasickým terminálem MT5. Instalátor používá součást Windows .NET Framework 4.8. Po dokončení klikněte na **Zavřít**.

V Navigátoru MT5 zvolte u **Indikátory → Obnovit** a přidejte **DNForecast_Async** na graf AUDUSD H1. Při prvním použití potvrďte import DLL pro tento indikátor. Pro výzkumnou TCN nastavte `InpModelFile` na `DNForecast\AUDUSD-H1-TCN.onnx`. Funkčnost knihovny a modelu můžete ověřit spuštěním skriptu **DNForecast_Smoke**. Indikátor potřebuje tržní historii podle zvolených symbolů.

Vlastní model natrénujete skriptem **DNForecast_Train** na grafu svého symbolu. Webové rozhraní spustíte tlačítkem **Otevřít studio** po instalaci nebo souborem `MQL5\Libraries\DNForecast-Studio.exe`. [Nativní trénink a samostatný server](Native-Studio.cs.txt)

## Výběr terminálů

- Při dvojkliku instalátor vyhledá datové složky aktuálního uživatele v `%APPDATA%\MetaQuotes\Terminal`. Každý profil musí mít složku `MQL5` a soubor `origin.txt` odkazující na existující `terminal64.exe` ve složce Program Files, v `%LOCALAPPDATA%\Programs` nebo v instalační složce evidované ve Windows. Jeden nalezený terminál je předvybraný. Při více terminálech vyberte jeden ze seznamu. První zápis provede až tlačítko **Nainstalovat**.
- Pro přenosný terminál nebo vlastní instalaci mimo uvedená umístění otevřete v příslušném MT5 **Soubor → Otevřít složku dat**. V instalátoru zvolte **Vybrat jinou složku…** a vyberte tuto složku nebo její podsložku `MQL5`.
- Před prvním spuštěním platformy mohou datové složky ještě chybět. Spusťte MT5, otevřete jeho datovou složku a vyberte ji v instalátoru.
- Standardní datová složka potřebuje pouze oprávnění aktuálního uživatele. Přenosná kopie vyžaduje zápis do své složky.

[Datové složky a režim portable podle MetaQuotes](https://www.metatrader5.com/en/terminal/help/start_advanced/start)

## Obsah a aktualizace

Instalátor kontroluje SHA-256 vestavěného balíčku i jednotlivých instalovaných souborů. DLL a ONNX Runtime umisťuje do `MQL5\Libraries`, indikátor do `Indicators`, ověřovací skript do `Scripts` a importy do `Include`. Model, jeho metadata, české PDF a licenční texty jsou v `MQL5\Files\DNForecast`. Chybějící závislosti Visual C++ pro ONNX doplní přímo vedle DLL. Použije dostupnou vyhovující nebo novější verzi z daného umístění či Windows.

Při opakovaném spuštění ověří existující soubory. Shodné soubory si zachovají čas změny. Nahrazené verze uloží ve složce **DNForecast.Backups** v příslušné datové složce terminálu. Pokud MT5 drží měněný soubor otevřený, instalátor jej zkusí znovu po uvolnění. Okno můžete ponechat otevřené až do běžného zavření MT5. Rozpracované změny po chybě vrátí zpět. Po výpadku dokončí obnovu následující spuštění instalátoru.

Protokoly jsou v `%LOCALAPPDATA%\Remind\DNForecast\Logs`. Obsahují cílové cesty a výsledky instalace. Instalace se týká souborů programu. O přidání na konkrétní graf a povolení DLL rozhoduje uživatel přímo v MT5.

## Bezobslužné nasazení

```powershell
$setup = '.\DNForecast-3.1.0-MT5-Setup.exe'
$p = Start-Process $setup -ArgumentList '/quiet' -Wait -PassThru
$p.ExitCode
```

V tichém režimu se při jediném nalezeném profilu použije tento profil. Při několika profilech zadejte cíl výslovně. Parametr `/target "C:\cesta k datům MT5"` určí jedinou datovou složku obsahující MQL5. `/quiet` potlačí okno, `/log "C:\cesta\setup.log"` zvolí protokol. S `/no-wait` lze zamčený soubor vyřešit opakováním z vlastního nasazovacího systému. Kód 0 znamená dokončeno, 1 chybu, 2 žádný nalezený terminál, 3 více profilů bez určeného cíle, 32 soubor používaný jiným procesem, 1223 ukončení uživatelem.

## Sestavení instalátoru

Vývojář potřebuje ZIP v3.1.0 a kompilátor .NET Framework x64. ZIP už obsahuje distribuovatelné knihovny Visual C++ x64. Pro kontrolu zveřejněného ZIPu předejte `-ExpectedZipSha256` s otiskem vydání. Skript `package_forecast.ps1` přijímá při vytváření ZIPu volitelný `-CrtDirectory`.

```powershell
.\tools\build_mt5_installer.ps1 -ReleaseZip .\DNForecast-3.1.0-win-x64.zip
.\tests\test_mt5_installer.ps1 -Setup .\output\mt5-installer\DNForecast-3.1.0-MT5-Setup.exe -ReleaseZip .\DNForecast-3.1.0-win-x64.zip
```
