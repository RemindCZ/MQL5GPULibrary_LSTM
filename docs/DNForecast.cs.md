# DNForecast 3.2
Asynchronní predikce finančních časových řad pro MetaTrader 5, Python a C. Technický manuál • Remind / Tomáš Bělák

## Instalace ve Windows a první graf
Spusťte DNForecast-3.2.0-MT5-Setup.exe a vyberte datovou složku terminálu. Při ruční instalaci rozbalte DNForecast-3.2.0-win-x64.zip. V MT5 zvolte Soubor → Otevřít složku dat a zkopírujte její cestu. Z kořene rozbaleného vydání spusťte v PowerShellu instalační příkaz. Parametr -Force dovoluje přepsání souborů předchozí instalace. Během aktualizace DLL ukončete terminál.

```
.\tools\Install-MT5.ps1 -TerminalDataPath 'C:\cesta\ke\složce dat'
```

Po spuštění MT5 povolte import DLL pro DNForecast_Smoke a spusťte jej z Navigátoru na grafu. Zpráva „DN3 smoke PASS“ v záložce Experti ověřuje plochou cenovou řadu v referenci i dvě hodnoty přibalené TCN proti samostatnému klientovi. Vyžaduje distribuovaný model v MQL5/Files/DNForecast.

Na graf AUDUSD H1 přidejte DNForecast_Async. Výchozí střed je poslední uzavřená cena. Dvě krajní linie ukazují empirické kvantily 0,10 a 0,90 předpokládaného výnosu. Graf se přepočítá při nové uzavřené svíčce. Povolte posun grafu pro místo za poslední svíčkou. Křivka spojuje odhady pro jednotlivé budoucí obchodní svíčky.

InpDrivers určuje přesné symboly v pořadí, oddělené čárkou. Prázdný řetězec znamená samotný cílový trh. InpSteps je počet minulých výnosů (16–1024). InpHorizons je počet budoucích svíček (1–64, nejvýše polovina okna). InpJournal zapíná zápis prospektivního CSV do MQL5/Files. InpCuda požaduje kompatibilní CUDA provider u ONNX.

Model z balíčku vyberte zadáním DNForecast\AUDUSD-H1-TCN.onnx do InpModelFile. Vyžaduje AUDUSD H1, 96 kroků, 6 horizontů a pomocné trhy NZDUSD, USDJPY, USDCAD v tomto pořadí. Shoda symbolů včetně přípon chrání před záměnou vstupů. Pro jiného brokera lze vytrénovat balíček s jeho jmény. Instalátor před nahrazením souborů vytvoří zálohu.

## Webové studio a autonomní trénink

V přihlášeném terminálu přetáhněte DNForecast_Bridge z Navigátoru mezi experty na jeden otevřený graf. Konektor pouze čte historii. V instalátoru zvolte Otevřít studio nebo spusťte MQL5/Libraries/DNForecast-Studio.exe. V záhlaví uvidíte server brokera a stav MT5.

Na kartě Experiment zadejte cílový symbol, například BTCUSD, časový rámec H1 a 4096 uzavřených svíček. Do pomocných trhů lze přidat přesné názvy brokera oddělené čárkou. Po spuštění získá server snímek požadovaného rozsahu. Pomocnou cenu použije jen tehdy, pokud odpovídající svíčka byla prokazatelně uzavřená v okamžiku rozhodnutí. Chybějící pozorování mají masku.

Zvolte délku vstupu, horizont a počet epoch. CPU natrénuje síť a ověří export ONNX. Chronologické bloky mají poměr 55 % pro učení, 15 % pro výběr epochy, 15 % pro kalibraci a 15 % pro test. Počátky s budoucím cílem přesahujícím hranici jejich bloku jsou vynechány.

V Automatizaci zvolte interval kontroly a aktivujte plán. Po zavření karty plán pracuje dál, pokud běží server a přihlášený terminál s konektorem. Při výpadku čeká na spojení. Po restartu Studio načte nastavení z disku. Kontrolní otisk historie a jméno brokera umožňují přeskočit již zpracovaný snímek. Samostatný server vedle volby --data podporuje --mt5-data s cestou k profilu obsahujícímu MQL5.

Graf ceny zarovnává odhad s časem jeho cíle. Zlepšení MSE je 1 − MSE sítě / MSE poslední ceny. Pokrytí udává podíl skutečných cen mezi Q10 a Q90. Obchodní simulace porovnává výnos na nejdelším horizontu H se zvoleným prahem. Na signál reaguje až na příští uzavírací ceně a pozici drží H svíček. Poplatek účtuje při vstupu i výstupu podle hodnoty pozice. Kapitál se oceňuje na každé uzavřené svíčce. Jde o historickou simulaci při zadaných nákladech. Skutečná exekuce, financování a změny spreadu mohou průběh podstatně změnit.

Výsledky obsahují ONNX, snímek vstupu, metriky pro každý horizont, úplnou testovací řadu a přehled obchodů. Stažený ONNX připojte na cílový graf indikátorem DNForecast_Async se stejnými symboly, délkou vstupu a horizontem.

## Datová smlouva a interpretace výstupu
Nejstarší řádek matice je první. Vstup jsou uzavírací ceny [S+1,M], kde S je délka okna a M počet trhů. Sloupec 0 je kladná cena cíle. U pomocného trhu nula značí nedostupnou kotaci. Vstup musí být konečný. Klient drží původ každé predikce odděleně od okamžiku vydání.

DN3_Encode počítá stejné příznaky pro trénink, DLL i živý graf. Pro každý trh vytváří přirozený logaritmický výnos, masku přítomnosti obou po sobě jdoucích cen a stáří poslední dostupné ceny v počtu cílových svíček dělené S (nejvýše 1). Po mezeře první dostupná cena obnoví stáří, výnos bude znám až z dalšího dostupného páru. Knihovna používá obchodní svíčky cíle, proto může výnos po víkendu obsahovat cenový skok.

Každý horizont má šest hodnot typu double v pořadí point, q10, q50, q90, p_up, variance. První čtyři jsou kumulativní logaritmické výnosy. Pro kresbu převeďte hodnotu r na cenu C × exp(r). p_up je pravděpodobnost přísně kladného výnosu. Variance má jednotku čtverce logaritmického výnosu. Bodová predikce a medián mohou být různé.

Vestavěná reference vrací nulový bodový výnos a p_up=0,5. Variance vychází z EWMA se součinitelem 0,94. Kvantily používají minulé kumulativní výnosy příslušného horizontu přepočtené podle volatility známé před jejich počátkem. U klidné řady zůstává malé kladné minimum variance. Pokrytí pásma se musí změřit pro daný trh a období.

Živý indikátor odebírá pouze uzavřené svíčky. Při výpadku dostupnosti pomocného trhu přenese masku. Historické soubory MT5 typicky neobsahují okamžik, kdy broker poprvé zveřejnil konečnou hodnotu. Export proto označuje předpoklad dostupnosti při uzávěrce. Prospektivní deník indikátoru zaznamenává serverový čas otevření referenční svíčky a okamžik vydání. Samostatný přehrávač používá čas uzavření. Před společným vyhodnocením musejí používat stejnou časovou osu.

## C ABI a životní cyklus požadavku
Deklarace typů a přesné argumenty jsou v include/DNForecast.h, importy MT5 v DNForecast.mqh. Ve Windows x64 odpovídá MQL long typu int64_t. Všechna pole jsou plochá a délky určují počet prvků příslušného typu. Text kontraktu se předává v UTF-8 bez ukončovací nuly. DN3_GetAbiVersion vrací 30000.

DN3_OpenSession přijímá délku okna, počet trhů, horizonty, kapacitu, režim latest, backend, kontrakt a obsah modelu. Vrátí kladné ID relace nebo záporný kód. Vstupní bajty si DLL zkopíruje. Načtení, ověření tvarů a zkušební inference proběhnou mimo vlákno grafu. DN3_StatusOf vrací 1 během načítání, 0 po připravení nebo zápornou chybu. DN3_Error s job=0 poskytne důvod selhání modelu.

DN3_Submit přijímá kladný identifikátor původu a přesný počet cen. Zkopíruje je a vrátí kladné ID úlohy. ID má význam uvnitř dané relace. Do úspěšného načtení modelu nelze odesílat. Každá relace drží omezený počet dosud neuvolněných požadavků. Sdílený plánovač obsluhuje relace cyklicky až čtyřmi pracovními vlákny. Jednu relaci zpracovává sériově.

DN3_TryRead vrací 1 při čekání, 0 při výsledku, −3 při malém bufferu. Zapisuje potřebnou délku a původ. Výsledek lze číst opakovaně do DN3_Release. Kódy −1, −2, −4, −5, −6 a −7 označují po řadě chybný vstup, plnou frontu, zrušení, nahrazení novějším požadavkem, chybu výpočtu a neznámý identifikátor. U plné fronty nejprve uvolněte dokončené výsledky.

DN3_Cancel zruší čekající úlohu. Rozběhnutý výpočet může fyzicky skončit později. DN3_Release uvolní výsledek a bezpečně snese opakované volání. V režimu latest je starší nezahájená práce při přijetí nové označena −5, její ID lze zkontrolovat a uvolnit. Uzavřete všechny relace přes DN3_CloseSession před vyložením DLL. Poslední uzavření vyčká na dokončení pracovníků. DN3_Metrics vrací osm čítačů v pořadí veřejné hlavičky včetně času výpočtu poslední úlohy v mikrosekundách.

## Modely, trénink a kompatibilita

Distribuce obsahuje nativní CPU trenér, skript DNForecast_Train a webové DNForecast Studio. Pro trénink přímo na grafu BTCUSD H1 můžete spustit skript ze sekce Skripty. DLL vystavuje DNFit_Train pro běžné klienty. Rozšiřující rozhraní DNFit_TrainTrace přijímá navíc výstupní pole a jeho délku v hodnotách double. Pro každý testovací počátek zapisuje čtyři hodnoty nejdelšího horizontu v pořadí bodový logaritmický výnos, Q10, Q90 a pravděpodobnost růstu. Počet počátků odpovídá rows − H − floor(0,85 × rows). Úplná deklarace je v include/DNForecastTrain.h.

Výchozí reference funguje na CPU. Součástí vydání pro Windows je volitelně načítaný ONNX Runtime CPU 1.23.2 včetně licencí dodavatelů. Pro CUDA provider sestavte nebo instalujte odpovídající GPU distribuci ONNX Runtime, CUDA a cuDNN. Dostupné rozhraní zjistíte z DN3_GetCapabilities. Skutečné načtení konkrétního modelu ovládá stav relace. Původní nativní DLL 2.3 používá vlastní CUDA závislosti.

ONNX vstup features je float32 [1,S,3M], výstup forecast je float32 [1,H,6]. Jedna samostatná ONNX zpráva obsahuje váhy, metadata dn3.contract a dn3.zero_probe. Kontrakt má tvar dn3-v1|target-trading-bars|3600|AUDUSD,NZDUSD,USDJPY,USDCAD. Referenční tenzor obsahuje 6H desetinných hodnot výstupu pro nulový příznakový vstup. DLL zkontroluje kontrakt, statické tvary, referenční hodnoty, konečnost, pořadí kvantilů, pravděpodobnost a kladnou varianci. U příznaků normalizovaných specificky pro model patří transformace dovnitř grafu.

Knihovna podporuje vlastní topologie nad operátory zvoleného ONNX backendu. Přiložený trénink nabízí ridge, GRU a reziduální TCN s dilatacemi 1, 2, 4, 8, 16, 32. Lineární a neuronová verze dávají přímé odhady na všechny horizonty. Kvantilové posuny reziduí vznikají na samostatném kalibračním období. Všechny části používají stejnou nativní transformaci DN3_Encode. Přiložená TCN má status research.

Volitelný výzkumný trénink v PyTorch spouštějte mimo terminál. Chronologické bloky obsahují 55 % dat pro trénink a po 15 % pro výběr epochy, kalibraci a test. Na hranicích se vynechají počátky s přesahem budoucího cíle. Report obsahuje SHA-256 dat a modelu, rozsahy času, shodu PyTorch s vydaným modelem a ztráty po horizontech.

Pro samostatný výběr jednotlivých výstupů použijte select_tasks.py. Za vstupním a novým názvem ONNX následuje jedna maska na každý horizont. Součet bitů 1 (bod), 2 (tři kvantily), 4 (směr), 8 (variance) vybírá výstupy sítě. Zbývající veličiny dodá reference. Maska 2 tak dovolí otestovat cenové pásmo sítě při ponechání poslední ceny jako středové čáry. Politika je součástí nového hashe modelu.

```
python examples/train_forecast.py DLL bars.csv model.onnx --symbols AUDUSD,NZDUSD,USDJPY,USDCAD --kind tcn
python examples/pack_legacy.py adresar_modelu model.dn2 --seconds 3600
```

Adaptér DN3V2 přečte checkpoint a škálování vytvořené původním train_csv.py. Vyžaduje MQL5GPULibrary_LSTM.dll vedle DNForecast.dll a dostupné knihovny CUDA. Nativní výstup přepočte původním target_scale. Kvantily, p_up a variance zůstávají referenční. DLL 2.3 a její původní dokumentace jsou v repozitáři dále dostupné.

## Samostatný provoz a test obchodního pravidla
Pythonový klient pro inferenci používá ctypes ze standardní knihovny. Uvolnění zdrojů zajistí with Session(...). Příklad v C přilinkujte proti DNForecast.lib a při spuštění umístěte DLL vedle aplikace. Volitelný výzkumný skript train_forecast.py v repozitáři používá NumPy, PyTorch a onnx. Python export historie vyžaduje MetaTrader5 pro Windows a připojený terminál.

```
with Session(Library(dll), ["AUDUSD"], steps=96, horizons=6) as s:
    forecast = s.predict(origin, closes)
    sixth_price = closes[-1] * math.exp(forecast[5].point)
```

CSV pro přehrání obsahuje close_time, available_at a sloupce instrumentů v deklarovaném pořadí. Časy jsou celé sekundy na jedné časové ose. Každý řádek představuje zmrazený snímek při rozhodnutí. Prázdná buňka je chybějící pomocný trh. Skript replay_forecast.py vytvoří SQLite deník s tabulkami forecasts a outcomes. Opakované spuštění nad stejnými daty nepřidá duplikáty. Oprava již vydaného vstupu vyžaduje oddělenou verzi deníku. Horizont na konci testu bez dozrálé cílové ceny zůstane bez výsledku.

backtest_strategy.py bere přesné ID série z přehrávače, odložený počáteční čas, horizont a samostatný CSV tok time_msc,bid,ask. Pevné pravidlo porovnává velikost bodového signálu s posledním známým spreadem, oboustrannými náklady a vstupním prahem. Nakupuje na ask, prodává na bid. Plnění nastává až na další pozdější kotaci. Vstup po vypršení signálu se vynechá. Test dovoluje nejvýše jednu otevřenou pozici s počáteční nominální hodnotou nejvýše 1× vlastní kapitál.

Výstup zahrnuje čistý výnos, maximální propad z průběžné likvidační hodnoty, obrat, komise a počty obchodů. Zadávejte komisi a skluz v bazických bodech za stranu. Předpokládané plnění celého objemu, financování přes noc, přesná velikost lotu, měnová konverze účtu a brokerův margin call vyžadují před nasazením další ověření. Historický přehrávač pracuje s virtuálním kapitálem v kotovací měně.

## Výsledky a hranice ověření
Distribuovaná TCN byla vyhodnocena na 2 799 pozdějších historických oknech. MSE logaritmického výnosu v šesté hodině dosáhla 1,0054násobku reference poslední ceny. Do referenčního režimu byla proto jako výchozí volba zařazena poslední známá cena. Výběr TCN v indikátoru je výslovnou volbou uživatele.

Navazující simulace použila říjnové kotace AUDUSD 2026 z terminálu IC Trading včetně skutečného bid a ask. Export od 24. září do 1. listopadu zahrnoval 648 uzavřených H1 svíček pro každý ze čtyř trhů a 1 983 619 kotací včetně úvodní historie. V říjnu vzniklo 529 předpovědí a 523 plně dozrálých šestihodinových výsledků. MSE činila 0,000003416 u TCN a 0,000003425 u reference. Pokrytí pásma TCN bylo 83,17 %, reference 74,19 %.

Při komisi 0,5 bp a skluzu 0,5 bp za stranu pravidlo s šestihodinovým horizontem nevstoupilo ani jednou. Čistý simulovaný výnos i propad byly nulové. Údaje o komisi a skluzu jsou scénářové předpoklady. Toto období je nyní součástí vyhodnocení. Jakákoli nová volba pravidla potřebuje další oddělené období a následné průběžné pozorování.

Měření malé TCN na CPU ve Windows použilo 10 000 požadavků pro každou z konfigurací 1, 8 a 32 relací. Při jedné relaci byl p99 odeslání 0,0246 ms a p95 do převzetí výsledku 0,430 ms. U 32 relací byl p95 do převzetí 1,657 ms. Měří se od volání DLL v Pythonu do převzetí výsledku. Čas vykreslení v MT5 závisí také na časovači, terminálu a dostupnosti dat.

Přesné hodnoty testu jsou v docs/research/ICTrading-2026-10.json a models/AUDUSD-H1-TCN.json. Reprodukci zajišťují docs/research/export_snapshots.py a evaluate_execution.py. Licenční informace k ONNX Runtime jsou v distribučním balíčku. Výzkumné zdroje, soutěž referencí, sezónní variance a experiment se zpožděnou aktualizací vah jsou v ARCHITEKTURA.cs.md a NAVRH_PREDIKCI.cs.md.
