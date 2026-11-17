# Návrh asynchronní knihovny pro predikci finančních časových řad v MT5

Výzkumný návrh pro MQL5GPULibrary_LSTM, 17. listopadu 2026. Výchozí vydání DLL 2.3.0.

Na zde popsaný pilot navazuje [architektura knihovny, výzkum teorie her a ověření na roce 2025](ARCHITEKTURA.cs.md). Obsahuje smlouvy nového rozhraní, datový tok a podmínky pro nasazení jednotlivých modelů.

Pro příští verzi doporučuji **oddělit model od zpracování požadavků**. Každý graf odešle uzavřené pozorování, dál obsluhuje terminál a později převezme výsledek označený časem svíčky a verzí modelu. Stejné výpočetní jádro použije samostatná aplikace v Pythonu nebo C++. Jako první novou síť bych otestoval malou **kauzální časovou konvoluční síť (TCN)** se společným vstupem pro několik trhů. Vítězem pro konkrétní instrument a horizont se stane model, který prokáže přínos na pozdějších kotacích a stihne požadovanou latenci.

## Co dnes skutečně dělá DLL a indikátor

| Místo | Zjištění z verze 2.3.0 | Důsledek pro návrh |
| --- | --- | --- |
| [PredictBatch](../kernel.cu#L2819) a [GPUContext](../kernel.cu#L232) | Každá predikce konstruuje objekt se streamem, cuBLAS a cuRAND, přiděluje dočasné GPU buffery, převádí double na float, mění uspořádání vstupu a před návratem volá `cudaStreamSynchronize`. | Vyčlenit prostředky pro inferenci, držet je po dobu relace a měřit cenu jednotlivých kroků. |
| [Zamykání modelu](../kernel.cu#L3008) | `DN_TrainAsync` má pracovní vlákno. `DN_PredictBatch` získává výlučný zámek modelu pomocí `try_to_lock`. Trénink i jiné použití téhož modelu mohou vrátit stav obsazeno. Vrstvy používají měnitelné mezivýsledky. | Oddělit neměnné váhy a pracovní prostor každého současného výpočtu. Pouhé oslabení zámku by vytvořilo datový závod. |
| [Indikátor CrossAsset](../MQL5/Indicators/LSTM_CrossAsset_Forecast.mq5) | Už zpracovává uzavřené svíčky, drží aktivní a kandidátní model, přímo predikuje několik horizontů a porovnává je s nulovým výnosem. Validace celé dávky i živá predikce probíhají synchronně při obsluze časovače. `Collect` při potřebě nové predikce načítá dlouhou historii. `Paint` inicializuje celé tři buffery a překresluje graf také z `OnCalculate`. | Krátké obsluhy událostí, průběžně doplňovaná historie a překreslení jen změněných bodů. Samotná rychlejší síť tyto náklady neodstraní. |
| [Příprava dat](../python/financial_series.py) a [validace](VALIDATION.md) | Existuje časové dělení s odstupem a škálování z tréninku. Pásmo indikátoru vychází z RMSE reziduí. Repozitář dokládá kontrolu GPU výpočtů, gradientů, checkpointů a kompilace MQL5. | Použít existující numerické testy jako ochranu kompatibility. Pro hospodářský význam doplnit opakované časové řezy, kalibraci a prospektivní záznam u brokera. |

### Naměřená latence verze 2.3.0

Windows x64, Intel Xeon E5-2680 v4, RTX 3060 s 12 GiB VRAM, ovladač 610.88. Přímo přes C ABI jsem změřil výchozí graf LSTM 32 + GRU 16 → Dense 24, čtyři trhy, 24 časových kroků a šest výstupů. Stejný handle, vstup a výstup v předem připravených polích `ctypes`, syntetické konečné hodnoty, bez souběžného tréninku. Pro každou velikost dávky 20 zahřívacích volání a dvě série po 100 měřeních. Čas na hostiteli od vstupu po návrat zahrnuje přenosy. Měření nezahrnuje terminál ani vykreslení.

| Oken v jednom volání | Medián série 1 / 2 | 95. percentil série 1 / 2 | Oken za sekundu série 1 / 2 |
| ---: | ---: | ---: | ---: |
| 1 | 4,350 / 4,327 ms | 9,556 / 8,447 ms | 171 / 201 |
| 8 | 4,278 / 4,329 ms | 8,120 / 8,524 ms | 1 616 / 1 594 |
| 32 | 5,007 / 4,328 ms | 9,078 / 8,584 ms | 5 567 / 6 821 |

Osm připravených oken se stejnými vahami mělo přibližně stejnou mediánovou latenci jako jedno. To podporuje společné zpracování souběžně doručených oken. Pro osamocený graf by čekání na naplnění dávky přidalo prodlevu. Podíl inicializace, kopírování a samotných kernelů určí profilování. SHA-256 měřené DLL je `70b8ae0825acb83d144ad42fd67b6074dac8f7f08ee78ad8e368d1bca301ca9a`. [Reprodukční skript](research/benchmark_predict.py)

## Pilot na skutečných hodinových cenách AUDUSD

Použil jsem veřejný archiv hodinových OHLC z kotací Dukascopy, který vydavatel odvozuje z M30. Čtyři soubory pokrývají AUDUSD, NZDUSD, USDJPY a USDCAD. Jde o historické zavírací ceny v UTC. Archiv neobsahuje čas skutečného doručení svíček do MT5 ani párové exekuční kotace IC Trading. Přesné SHA-256 všech vstupů kontroluje reprodukční skript. [Popis a původ datové sady](https://huggingface.co/datasets/Yllvar/fx-historical-data) · [Reprodukce pilotu](research/audusd_h1_pilot.py)

Trénink proběhl na **4 621** oknech roku 2023, následný test na **4 627** oknech roku 2024 bez přeučování. Každé okno má 24 po sobě jdoucích hodinových výnosů a šest přímých kumulativních cílových výnosů. Vstupy potřebují kompletní známou historii všech čtyř trhů. Pro vyčíslení chyby se ověřují pouze budoucí ceny cíle. Víkendy a mezery uvnitř okna či vyhodnocovaného horizontu se vynechávají. Škálování používá jedinečné vstupní řádky pouze z tréninku. Poslední trénovací cíl je uzavřen před hranicí testu.

**Poměr MSE vůči předpovědi poslední známé ceny.** Hodnota 1 znamená shodnou chybu, nižší je lepší. Závorky v posledním sloupci jsou 95% intervaly pro šestou hodinu z 2 000 párových výběrů po kalendářních týdnech. Popisují variabilitu testovacího období při pevném modelu. U sítí uvádím jednotlivé tréninky.

| Model a opakování | 1 h | 3 h | 6 h | Interval pro 6 h |
| --- | ---: | ---: | ---: | --- |
| Poslední cena | 1,0000 | 1,0000 | 1,0000 | referenční hodnota |
| Ridge pouze AUDUSD | 1,0031 | 1,0008 | 0,9980 | [0,9926, 1,0041] |
| Ridge všechny čtyři měny | 1,0084 | 1,0065 | 1,0010 | [0,9926, 1,0107] |
| Výchozí LSTM/GRU, běh 1 | 1,0200 | 1,0113 | 1,0155 | [0,9994, 1,0337] |
| Výchozí LSTM/GRU, běh 2 | 1,0492 | 1,0145 | 1,0138 | [1,0009, 1,0291] |
| Výchozí LSTM/GRU, běh 3 | 1,0234 | 1,0097 | 1,0125 | [0,9982, 1,0307] |
| TCN, seed 37 | 1,0023 | 1,0020 | 1,0007 | [0,9872, 1,0155] |
| TCN, seed 38 | 0,9999 | 1,0015 | 1,0001 | [0,9901, 1,0116] |
| TCN, seed 39 | 1,0043 | 1,0035 | 1,0044 | [0,9960, 1,0138] |

Ridge používá lineární regresi se stejným oknem a penalizací `lambda=1` ve ztrátě zprůměrované přes vzorky. Nativní síť používá výchozí topologii repozitáře. Současná DLL nemá veřejný seed, proto její čísla mezi spuštěními kolísají. Pilotní TCN má 16 kanálů, čtyři reziduální bloky s dilatacemi 1/2/4/8, levé doplnění a bodový výstup. Modely mají šest epoch a batch 128. U TCN se cíle škálují po horizontech. Porovnání tedy zahrnuje konkrétní trénovací postup i architekturu. Delší 96kroková síť a kvantilový výstup popsané níže jsou samostatné budoucí experimenty.

**Rozhodnutí z pilotu.** Současná síť měla na šestihodinovém horizontu ve všech třech hlavních bězích vyšší chybu než poslední cena. TCN byla blíž referenci, ale všechny její šestihodinové intervaly zahrnují jedničku. Zlepšení ridge na samotném AUDUSD o 0,2 % má stejnou nejistotu. Přidání tří měn v lineárním modelu bodový výsledek zhoršilo. Například TCN se seedem 37 měla v prvních třech čtvrtletích poměry 1,0019 / 1,0115 / 1,0128 a ve čtvrtém 0,9786. Vybrat jen poslední čtvrtletí by vedlo k jiné volbě. Pro dnešní nasazení směrové křivky je proto obhajitelnou referencí poslední cena. O přidané hodnotě kvantilů, dalších vstupů nebo jiné historie rozhodne následující samostatná studie.

Pro opakování uložit čtyři soubory `<SYMBOL>_H1.parquet` z uvedeného archivu do stejné složky. Skript vyžaduje NumPy a DuckDB, přepínač `--tcn` také PyTorch. Měření použilo DuckDB 1.4.2 a PyTorch 2.7.1 CPU. Pro nativní část se použije vydaná 64bitová DLL a její závislosti CUDA. Oba skripty se spouštějí z kořene projektu.

```powershell
python docs/research/audusd_h1_pilot.py --cache tmp --dll build/Release/MQL5GPULibrary_LSTM.dll --tcn
python docs/research/benchmark_predict.py build/Release/MQL5GPULibrary_LSTM.dll
```

## Které architektury mají smysl porovnat

U směnných kurzů se výsledek může změnit s obdobím, horizontem i dostupností vstupů v okamžiku předpovědi. Výzkum na skutečně dostupných datech ukazuje vliv revizí a volby období. Novější studie osmi měnových párů zkoumala, kdy vyšší složitost pomůže při použití měnových fundamentů. Obě práce dávají důvod zařadit silnou jednoduchou referenci. Výsledek pro hodinové mezitržní výnosy musí dodat vlastní experiment. [Fed, real-time exchange-rate forecasts](https://www.federalreserve.gov/econres/ifdp/exchange-rate-forecasting-the-errors-we39ve-really-made.htm) · [Virtue or Mirage?](https://www.federalreserve.gov/econres/feds/virtue-or-mirage-complexity-in-exchange-rate-prediction.htm)

| Kandidát | Proč ho měřit | Podmínka pro další práci |
| --- | --- | --- |
| Poslední cena, ridge regrese a pásmo empirických kvantilů | Nulový očekávaný pohyb, levná lineární predikce z týchž zpožděných vstupů a přímá reference pro nejistotu. | Všechny dostanou stejné okamžiky rozhodnutí a shodnou dostupnost dat. Lineární model může být nasazeným vítězem. |
| Současný větvený LSTM/GRU a samostatná malá GRU | Změří přínos nové sítě vůči již fungujícímu kódu a cenu složitější rekurence. | U GRU porovnat celé okno s průběžným stavem pouze při shodné definici historie a resetu. |
| Malá vícekanálová TCN | Kauzální konvoluce spojují známou historii různých délek, časové kroky lze během výpočtu paralelizovat. Běžné operátory umožňují porovnat CPU a GPU. | První nový prototyp. Ověřit zvlášť přínos dalších trhů a delší paměti. [Bai, Kolter a Koltun](https://arxiv.org/abs/1803.01271) |
| PatchTST | Umožňuje levněji vyhodnotit delší historii po úsecích. Původní verze zpracovává kanály nezávisle a sdílí jejich váhy. | Přidat až jako samostatný experiment s dlouhým oknem. Pro vliv jiných měn explicitně definovat a porovnat společnou výstupní vrstvu. [Nie a kol.](https://arxiv.org/abs/2211.14730) |
| Chronos-2 | Předtrénovaná modelová reference podporuje více řad a doprovodné proměnné. Má smysl i bez místního trénování velkého modelu. | Offline porovnání s pevnou verzí vah, měřením paměti a kontrolou překryvu předtrénování s testem. Přednost má období doložitelně po uzávěře tréninku. [Autoři Chronos-2](https://arxiv.org/abs/2510.15821) |

Jednoduchá lineární řešení v dlouhodobých benchmarkových úlohách dokázala překonat několik tehdejších Transformerů. Tyto výsledky ani publikované výsledky PatchTST neurčují vítěze na cenách AUDUSD. První sada experimentů proto zahrnuje TCN, lineární regresi a současnou síť. Přidání většího modelu má smysl, až když jeho kvalita vyváží vyšší náklady. [Zeng a kol.](https://arxiv.org/abs/2205.13504)

### První konfigurace TCN

Vstup má tvar `[dávka, 96 uzavřených svíček, příznaky]`. Výchozí cílový graf je AUDUSD H1. NZDUSD, USDJPY a USDCAD jsou kandidátní vysvětlující trhy. XAUUSD hodnotit v odděleném pokusu kvůli odlišným přestávkám seance. Základem jsou logaritmické výnosy, minulé rozpětí high–low a realizovaná variabilita dostupná do času rozhodnutí. Spread, čas od poslední kotace, masky chybějících hodnot a čas seance přidávat po skupinách. Budoucí ceny doprovodných trhů model nedostává.

1. Lineární projekce v každém čase smíchá příznaky trhů do 32 kanálů. Šest reziduálních bloků má každý jednu kauzální konvoluci šířky 3 a dilatace postupně 1, 2, 4, 8, 16 a 32. Použít levé doplnění, ReLU a případně normalizaci pouze přes kanály téhož času. Receptivní pole je `1 + 2 × (1+2+4+8+16+32) = 127` kroků a pokrývá zvolené okno. Šířky 16 a 32 a okna 24 a 96 tvoří malý předem daný rozsah experimentů.
2. Z poslední reprezentace vyrobí společná výstupní vrstva šest **přímých** předpovědí. Pro horizont `h` je cílem `log(C[t+h]/C[t])`. Trénovací větev pro srovnání s dnešní DLL předpovídá podmíněný průměr s MSE. Pravděpodobnostní větev poskytne kvantily 0,10 / 0,50 / 0,90 pomocí pinball loss. Vzdálenosti od mediánu parametrizovat kladnými hodnotami, aby se meze nekřížily. Každý horizont má stanovenou vlastní škálu z tréninku. Při hodnocení se výsledky vracejí do původních jednotek.
3. Střed křivky je `C[t] × exp(q50[h])`. Dolní a horní pásmo se převádějí stejně. Pásmo je bodové pro jednotlivé horizonty. Současné pokrytí celé budoucí dráhy by potřebovalo samostatný test. Šířka pásma má samostatný užitek i v období, kdy se směr výnosu nedaří předvídat. Výzkum realizované volatility na skutečných kurzech poskytuje důvod hodnotit tyto dvě schopnosti samostatně. [Andersen a kol.](https://www.nber.org/papers/w8160)

Když vyhraje síť jen se samotným AUDUSD, ostatní symboly může konkrétní model vypustit. Když uspěje pouze předpověď rozpětí, graf může zobrazit poslední cenu jako referenci a označené ověřené pásmo. Každý horizont má přiřazený vlastní stav ověření.

## Data a ověření predikce

**Jednotkou pozorování je okamžik, kdy šlo prognózu skutečně vydat.** Ukládat symbol a jeho brokerovou příponu, timeframe, typ ceny, čas začátku a konce svíčky, čas dostupnosti a čas posledního ticku, pokud je známý. Zachovat původní serverový čas a auditovat převod na UTC včetně změn času. Pro historická OHLC, která nemají čas přijetí, označit předpokládané zpoždění a zopakovat pokus se zpožděním pomocného trhu o další svíčku. Standardizaci, prahy i režimy volatility fitovat jen z minulosti.

Při chybějící cílové svíčce vydat stav čekání. Pro volitelné vstupy definovat maximální stáří a předat modelu masku a skutečné stáří. Vrátí-li se pomocný trh po přestávce, nepřipsat kumulovaný pohyb jediné společné hodině bez příznaku mezery. Definovat, zda horizont znamená příštích šest obchodních svíček nebo šest hodin UTC. Pro první živý režim doporučuji obchodní svíčky cíle a předem známý kalendář seancí. Na obou stranách rozhraní musí vyjít stejná časová osa i po víkendu.

**Navržený experiment na kotacích IC Trading**

1. Zmrazit historii a popis symbolů, přednostně několik let H1 se změnami trendu, volatility a spreadů. Před zobrazením chyb zaregistrovat primární cíl AUDUSD, šestou budoucí svíčku, pořadí kandidátů, povolené příznaky a rozpočet ladění. Doložit podíl použitelných svíček a předpovědí v každé seanci.
2. Pokud rozsah historie dovolí, použít nejméně 24 měsíců na počáteční učení. V posouvaných řezech použít pozdější oddělená období pro výběr konfigurace a kalibraci pásem a následný tříměsíční blok pro ověření. Na každé hranici odstranit původy, jejichž poslední cílová svíčka zasahuje do další části. Posledních alespoň šest měsíců vyhradit jednorázovému finálnímu testu zamčeného kandidáta. Případná další volba modelu vyžaduje nový budoucí úsek. [Hyndman a Athanasopoulos, časová validace](https://otexts.com/fpp3/tscv.html)
3. Porovnat shodné okamžiky a také dostupnost vůči všem vhodným cílovým svíčkám. Seřadit varianty od samotného cíle přes jednotlivé doprovodné měny po celý koš. Stejnou matici příznaků dostane ridge, GRU i TCN. Rozdíl cílového kurzu a koše je přímý test přínosu dalších trhů. Pro nové NN evidovat rozptyl nejméně tří předem daných inicializací, identitu dat, konfigurací a checkpointů.
4. Pro střed s MSE hodnotit poměr k nulovému výnosu po horizontech. Pro medián MAE, pro tři kvantily průměrné pinball loss, pro 80% pásmo pokrytí a šířku podle horizontu, seance a režimu. Referenční pásmo sestrojit z minulých reziduí nebo z EWMA škálovaných výnosů. Uvést počet předpovědí. Týdenní blokový bootstrap rozdílů ztrát zachová překryv sousedních prognóz uvnitř bloku. Ověřit citlivost k délce bloku a používat stejné časové bloky napříč trhy. [Gneiting a Raftery](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)
5. Kandidát pro směrovou křivku musí na primárním horizontu v oddělených datech porazit předem zvolenou nejsilnější jednoduchou referenci. Navržená hranice je horní 95% mez párového rozdílu ztráty pod nulou a žádný soustavný rozpad v jednotlivých řezech. U pásma požadovat nižší kvantilovou ztrátu a zkontrolovat nominální pokrytí s ohledem na počet nezávislých bloků. Pokud přínos zůstane nerozhodný, nasadit levnější ověřenou variantu a dále sbírat předpovědi. Případný obchodní test má vlastní předem daná pravidla, první proveditelnou bid/ask kotaci, komisi, skluz a podle délky držení financování.

## Asynchronní provoz od grafu po výsledek

**MT5** zachytí novou uzavřenou svíčku, doplní kruhovou historii a odešle kopii vstupu. **Plánovač DLL** přiřadí identifikátor a naplánuje práci podle modelu a zbývajícího času. **Výpočetní vlákno** uloží výsledek do paměti ve vlastnictví DLL. **MT5** při další události přečte hotový výsledek a vykreslí pouze změněné body. Ukazatel na pole MQL5 se po návratu z volání nikdy nepoužije a pracovní vlákna nevolají MQL5 ani API grafu.

| Část | Navržená smlouva |
| --- | --- |
| C ABI | Přidat verzované operace pro otevření relace, odeslání vstupu, `TryRead`, zrušení požadavku, uvolnění výsledku a uzavření relace. Pevné šířky, délky polí, kapacita výstupu, verze schématu a chybový kód pro konkrétní požadavek. Stejné smlouvy v C, MQL5 a Pythonu. Při příliš malém výstupu vrátit potřebnou délku a výsledek ponechat k dalšímu čtení. |
| Identita a pořadí | Každý požadavek nese relaci, instrument, timeframe, začátek i čas dostupnosti poslední svíčky a generaci vah. Výsledek patří přesně tomuto vstupu. Každý graf odmítne starší generaci nebo čas. Stejné ID požadavku má stabilní koncový stav i při opakovaném čtení a přerušení. |
| Zatížení | Fronta i počet nezpracovaných výsledků mají pevný paměťový limit. Živý graf smí sloučit dosud nezahájené požadavky pro stejnou sérii ve prospěch novějšího. Každý přeskočený požadavek dostane jednoznačný stav. Dávkové přehrávání zachová pořadová čísla a vrací `QUEUE_FULL` s možností opakování. Výsledky mají dokumentovanou platnost a způsob uvolnění. |
| CPU a GPU | Jedna instance výpočetního jádra uvnitř procesu terminálu sdílená jednotlivými grafy, dlouhodobá relace, malý pool pracovních prostorů. GPU používá předem přidělené buffery, omezený pool page-locked host paměti a událost dokončení. Profiler rozdělí čekání, H2D, výpočet a D2H. CPU běží v pracovním vlákně s omezeným počtem vláken ONNX Runtime, aby jeden graf nezabral všechna jádra. [NVIDIA CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) · [ONNX Runtime I/O Binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html) |
| Dávkování | Seskupovat jen okna stejné generace modelu, schématu, tvaru a výpočetního backendu. Shodná topologie sama nestačí. Při volném zařízení a jediném živém požadavku začít ihned. Pro souběžné příchody zkusit plánovač bez čekání a s omezeným čekáním 0,25–1 ms, měřit p99. Offline fronta má menší prioritu. |
| Změna vah a chyby | Připravit a zahřát kandidáta vedle aktivní verze, ověřit manifest a referenční vstupy, poté atomicky přepnout. Rozpracované požadavky drží původní váhy do dokončení. Při selhání kandidáta dál obsluhovat aktivní verzi. Při chybě GPU přepnout jen na ověřený CPU ekvivalent, jinak vrátit chybu a označit graf. Při odpojení indikátoru ukončit odběr, uvolnit zdroje až po dokončení práce. Poslední uzavření musí bezpečně ukončit vlákna před vyložením DLL. |

**Volba výpočetního jádra.** Pro nové modely použít jako první přenositelný formát ONNX, trénovat offline a nasadit přes C/C++ ONNX Runtime. Jádro pro CPU musí jít načíst i na stroji bez ovladače NVIDIA. CUDA provider a dnešní nativní knihovnu načítat jako volitelné moduly. Balíček ponese deklarovaný opset, tvary tensorů, identitu vah, popis příznaků, transformace, kalibraci, kalendář, povolené backendy a referenční vstup/výstup. Nepodporované operátory oznámit při importu. MQL5 má vlastní ONNX podporu, kterou lze použít jako nezávislou kontrolu číselné shody pro podporované modely. [ONNX v MQL5](https://www.mql5.com/en/docs/onnx) · [CUDA provider a jeho závislosti](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

**Hranice modulů.** Stávající knihovnu a její exporty 2.3 ponechat jako kompatibilní modul pro stávající indikátory. Novou Windows x64 DLL vystavit přes samostatnou hlavičku C, soubor importů MQL5 a klienta Python `ctypes`. Instalátor do MT5 přidá jen zvolený backend. Kód sdílený tréninkem a živým během bude řešit pořadí symbolů, časová okna a transformace. Ovladač tréninku poběží mimo vlákno terminálu, výchozí v samostatném procesu. Při souběhu na jediné GPU sledovat dopad tréninku na p99 živé inference a trénink omezit, odložit nebo poslat na druhé zařízení. Priorita streamu sama nezaručí dodržení termínu. Nezávislé přeučení spojit se změnou generace až po validaci celého balíčku včetně škálování.

Libovolnou topologii realizovat dvojí cestou. Pro dnešní modely zachovat programové skládání DAG a slučování větví. U nových sítí přijímat uživatelské ONNX grafy v rozsahu zveřejněných operátorů a tvarů. Před zveřejněním výsledku porovnat referenci trénovacího frameworku s CPU, GPU a MT5. Příklad počáteční tolerance FP32 je `atol=1e-5, rtol=1e-4` v jednotkách výstupu, doplněná tolerancí v bodech konkrétní kotace. FP16, CUDA Graphs, TensorRT nebo průběžná cache TCN přijdou na řadu podle profilu konkrétního úzkého místa. U CUDA Graphs ověřit stabilní tvary, adresy a omezení souběhu relace. [ONNX Runtime CUDA Graphs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#using-cuda-graphs-preview)

V MT5 indikátory stejného symbolu sdílejí vlákno. Je-li předchozí událost časovače ve frontě nebo se zpracovává, terminál další nezařadí. Dokumentace pro živý provoz uvádí praktickou mez timeru přibližně 10–16 ms. Proto odděleně měřit dobu návratu DLL, dobu dokončení práce a prodlevu do překreslení. Při klidném trhu může časovač běžet pomalu, při rozpracované predikci přejít na jemnější kontrolu. `OnCalculate` může hotový výsledek převzít při dřívějším ticku. Zapsat skutečný okamžik vydání předpovědi a při další svíčce původní záznam nepřepisovat. [Vlákna programů MQL5](https://www.mql5.com/en/docs/runtime/running) · [Události časovače](https://www.mql5.com/en/docs/event_handlers/ontimer)

## Pořadí implementace a přejímací zkoušky

| Krok | Výstup pro uživatele | Podmínka dokončení |
| --- | --- | --- |
| 1. Referenční přehrávač a měření | Historická sada IC Trading, záznam dostupnosti, deterministické přehrávání stejných událostí v MT5 a samostatně, základní modely a profil verze 2.3. | Shodné vstupní okno, jednotky výstupu a původ předpovědi. Test se záměrnou změnou všech budoucích kotací musí ponechat dříve vydané predikce shodné. |
| 2. První svislý řez | Jeden stávající model, nové rozhraní odeslat/převzít, jedna relace v MT5 a jeden klient Pythonu. Zahrnout inkrementální načítání a částečné překreslení. | Asynchronní výsledek souhlasí se synchronním. Ověřit odpojení grafu za běhu, plnou frontu, opakované čtení, příliš malý buffer, pozdní výsledek, zrušení a výměnu modelu během práce. |
| 3. Výběr sítě | Přímá TCN, jednoduché reference, stávající síť, export ONNX a časová studie na shodných kotacích. | Předem daný výběrový protokol, oddělený závěrečný test, výsledky po horizontech, pokrytí a náklady CPU/GPU. Zdokumentovat i výsledek, kdy vyhraje jednoduchá reference. |
| 4. Více grafů a instalace | Sdílený plánovač, dávkování při souběhu, proces bez CUDA, manifest modelu, příklady C/Python a indikátor s mediánem a pásmem. | Testy na čistých Windows x64 s GPU i bez něj, kontrola všech stávajících exportů a přesně těch starých checkpointů, které podporuje verze 2.3. Kompilace MetaEditorem, lokální Strategy Tester a dlouhý běh na grafech. [Formáty checkpointů](API.md#checkpoints-and-snapshots) |
| 5. Prospektivní provoz | Vedle dnešní křivky zaznamenávat uzamčenou novou verzi, čas vydání, cenu při vydání, verzi a později skutečnou hodnotu. | Po předem určeném období vyhodnotit reálnou dostupnost, latenci a chyby. Umět obnovit předchozí balíček vah a nastavení. |

**Počáteční výkonnostní cíle pro krok 2 a 4.** Na uvedené sestavě Xeon a RTX 3060 navrhuji pro zahřátý běh, čtyři trhy a nejvýše 96 kroků p99 do 0,5 ms pro odeslání z grafu, p95 do 10 ms od odeslání po připravený výsledek u jednoho již načteného malého modelu a p95 do 100 ms od dostupnosti všech vstupů po vykreslení na H1. Jde o cíle, jejichž dosažení má potvrdit měření. Primární přínos na H1 je spolehlivě uvolnit vlákno grafu i při souběhu. Měřit také studený start, p50/p95/p99, počet výsledků za sekundu, RAM/VRAM a podíl prošlých či sloučených požadavků.

Pro výběr konfigurace použít 1, 8 a 32 grafů, rozlišit shodné a různé váhy, jednotlivé a současné uzávěrky, dávky 1/8/32 a souběžné přeučení. Porovnat CPU FP32, CUDA FP32 a současnou DLL na stejném vstupu. Nejméně 10 000 požadavků na konfiguraci pro p99 a osmihodinový běh včetně pravidelných výměn modelů. Podmínkou vydání je omezená paměť po zahřátí, nulový počet výsledků připsaných jiné svíčce a zaznamenaný konečný stav každého přijatého požadavku. Ověřit, že poslední klient může DLL bezpečně vyložit i po chybě zařízení.
