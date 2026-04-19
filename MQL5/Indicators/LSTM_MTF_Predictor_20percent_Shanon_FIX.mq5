//+------------------------------------------------------------------+
//| LSTM_Pure_MTF_Predictor.mq5 v5.23 |
//| Pure LSTM architecture with Multi-Timeframe (MTF) inputs |
//| GPU-optimized via MQL5GPULibrary_LSTM.dll |
//| |
//| LICENCE: MIT License |
//| Kód můžeš brát, upravovat, rozbíjet a používat jak chceš. |
//| Pokud z toho zbohatneš, pošli mi pohled. Pokud proděláš kalhoty, |
//| nestěžuj si u mě. Trh je jak tekutý písek – jakmile se začneš |
//| moc mrskat, stáhne tě to dolů. |
//| |
//| KONVENCE (series indexování - bacha na to, je to občas o nervy): |
//| index 0 = nejnovější bar (formující, nepoužívá se, kecá) |
//| index 1 = poslední uzavřený bar (jistota) |
//| větší index = starší bar (historie) |
//+------------------------------------------------------------------+
#property copyright "Tomáš Bělák Remind"
#property link "https://remind.cz/"
#property description "Pure MTF LSTM v5.90: připravené na produkci"
#property version "5.90"
#property strict

#property indicator_separate_window
#property indicator_minimum 0
#property indicator_maximum 100
#property indicator_level1 50
#property indicator_level2 30
#property indicator_level3 70
#property indicator_levelcolor clrDimGray
#property indicator_levelstyle STYLE_DOT

#property indicator_buffers 3
#property indicator_plots 3

// --- Vykreslování: Linka pravděpodobnosti ---
#property indicator_label1 "BUY Prob (%)"
#property indicator_type1 DRAW_LINE
#property indicator_color1 clrLime
#property indicator_style1 STYLE_SOLID
#property indicator_width1 2

// --- Vykreslování: Šipky nahoru/dolů ---
#property indicator_label2 "Cross Up 30"
#property indicator_type2 DRAW_ARROW
#property indicator_color2 clrDeepSkyBlue
#property indicator_style2 STYLE_SOLID
#property indicator_width2 1

#property indicator_label3 "Cross Down 70"
#property indicator_type3 DRAW_ARROW
#property indicator_color3 clrTomato
#property indicator_style3 STYLE_SOLID
#property indicator_width3 1

//+------------------------------------------------------------------+
//| DLL Import - Křemík je vlastně jen chytře uspořádaný písek, |
//| do kterého pouštíme blesky. Tady to komunikuje s GPU. |
//+------------------------------------------------------------------+
#import "MQL5GPULibrary_LSTM.dll"
int DN_Create();
void DN_Free(int h);
int DN_SetSequenceLength(int h, int seq_len);
int DN_SetMiniBatchSize(int h, int mbs);
int DN_AddLayerEx(int h, int in_sz, int out_sz, int act, int ln, double drop);
int DN_SetOutputDim(int h, int out_dim);
int DN_SetGradClip(int h, double clip);
int DN_LoadBatch(int h, const double &X[], const double &T[], int batch, int in_dim, int out_dim, int layout);
int DN_TrainAsync(int h, int epochs, double target_mse, double lr, double wd);
int DN_GetTrainingStatus(int h);
void DN_GetTrainingResult(int h, double &out_mse, int &out_epochs);
void DN_StopTraining(int h);
int DN_PredictBatch(int h, const double &X[], int batch, int in_dim, int layout, double &Y[]);
int DN_SnapshotWeights(int h);
int DN_RestoreWeights(int h);
int DN_GetLayerCount(int h);
double DN_GetLayerWeightNorm(int h, int layer);
double DN_GetGradNorm(int h);
int DN_SaveState(int h);
int DN_GetState(int h, char &buf[], int max_len);
int DN_LoadState(int h, const char &buf[]);
void DN_GetError(short &buf[], int len);

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
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
int DN_GetProgressAll(int h, int &epoch, int &total_epochs, int &mb, int &total_mb,
                      double &lr, double &mse, double &best_mse, double &grad_norm,
                      double &pct, double &elapsed_sec, double &eta_sec);
#import

//+------------------------------------------------------------------+
//| Konstanty - ať nemáme v kódu magická čísla. |
//+------------------------------------------------------------------+
#define FEAT_PER_TF 18
#define NUM_TFS 3
#define FEAT_PER_BAR (FEAT_PER_TF * NUM_TFS)
#define OUTPUT_DIM 2
#define MODEL_MAGIC 0x4C53544D // Hex pro "LSTM"
#define MODEL_META_VER 4
#define LN2_CONST 0.6931471805599453
#define ARROW_UP 233
#define ARROW_DOWN 234

//+------------------------------------------------------------------+
//| Vstupní parametry (Inputs) |
//| Tyhle věci může uživatel rozbít v nastavení indikátoru. |
//+------------------------------------------------------------------+
input group "=== Multi-Timeframe Inputs ==="
input ENUM_TIMEFRAMES InpTF1 = PERIOD_CURRENT;
input ENUM_TIMEFRAMES InpTF2 = PERIOD_H1;
input ENUM_TIMEFRAMES InpTF3 = PERIOD_H4;

input group "=== LSTM Architecture ==="
input int InpLookback = 30; // Jak daleko do minulosti síť čučí
input int InpHiddenSize1 = 128;
input int InpHiddenSize2 = 64;
input int InpHiddenSize3 = 32; // 0 = vypnuto
input double InpDropout = 0.15; // Zapomínání, ať se to nepřeucí jak cvičená opice

input group "=== Shannon Entropy ==="
input int InpEntropyFastPeriod = 12;
input int InpEntropySlowPeriod = 32;
input int InpEntropyPriceStep = 1;
input bool InpUseEntropyFilter = true;
input double InpEntropyChaosLevel = 0.65;
input double InpEntropyFilterPower = 0.60;
input bool InpEntropyFilterCrossSignals = false; // Bacha, filtruje to i šipky

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
input group "=== Training (GPU) ==="
input int InpTrainBars = 3000;
input int InpInitialEpochs = 300;
input int InpRetrainEpochs = 80;
input int InpRetrainInterval = 200;
input double InpLearningRate = 0.0008;
input double InpWeightDecay = 0.0001;
input double InpTargetMSE = 0.005;
input int InpMiniBatch = 64;

input group "=== Train/Test Split & OOS Graduation ==="
input double InpTestPct = 20.0;
input bool InpShowSplitLine = true;
input double InpOOSPassThreshold = 20.0;
input int InpOOSRetrainEpochs = 50;
input double InpOOSRetrainLR = 0.0004;
input int InpMaxGraduations = 10;
input bool InpAutoGraduate = true;
input int InpMinOOSSamples = 30;
input int InpGradCooldownBars = 5;

input group "=== Prediction ==="
input int InpPredictBatch = 512; // Kolik toho nasypeme GPU najednou (bágl)
input int InpMaxPredictBars = 2000;
input int InpPredictAhead = 5; // Na kolik svíček dopředu hádáme
input bool InpUsePlotShift = true;

input group "=== Display ==="
input color InpBullColor = clrLime;
input color InpBearColor = clrRed;
input color InpNoiseColor = clrDimGray;
input int InpInfoCorner = 0;
input int InpProgressWidth = 280;
input int InpProgressHeight = 18;

input group "=== Advanced ==="
input double InpGradClip = 5.0;
input bool InpAutoRetrain = true;
input bool InpSaveModel = true;
input string InpModelPrefix = "REMIND";
input bool InpVerboseLog = false; // Kdo to má pak číst v terminálu...

//+------------------------------------------------------------------+
//| Stavy tréninku - abychom věděli, co ten model zrovna dělá. |
//+------------------------------------------------------------------+
enum ENUM_TRAIN_PHASE
  {
   PHASE_IDLE = 0,
   PHASE_INITIAL_TRAIN,
   PHASE_OOS_EVALUATE,
   PHASE_OOS_RETRAIN,
   PHASE_PERIODIC_RETRAIN
  };

//+------------------------------------------------------------------+
//| Historie promocí (Graduation) modelu |
//+------------------------------------------------------------------+
struct GradHistoryEntry
  {
   int               cycle;
   int               oldBoundary;
   int               newBoundary;
   double            oosAccuracy;
   int               oosSamples;
   double            mseBefore;
   double            mseAfter;
   datetime          timestamp;
  };

//+------------------------------------------------------------------+
//| Globální proměnné (Ano, vím, že jsou zlo, ale tady se to hodí) |
//+------------------------------------------------------------------+
string g_CrossObjPrefix = "MTF_X_";
string g_InfoPrefix = "MTF_I_";
string g_ProgPrefix = "MTF_P_";
string g_SplitPrefix = "MTF_S_";

// Buffery pro indikátor
double g_ProbPct[];
double g_CrossUp30[];
double g_CrossDown70[];

// Stav sítě a tréninku
int g_NetHandle = 0;
bool g_ModelReady = false;
bool g_IsTraining = false;
int g_LastTrainBar = 0;
int g_TotalBars = 0;
double g_LastMSE = 0.0;
int g_TotalEpochs = 0;
double g_BestMSE = 1e10;
int g_TargetEpochs = 0;
int g_CurrentEpochs = 0;
datetime g_TrainStartTime = 0;

// Statistiky z úspěšnosti (accuracy)
int g_TrainCorrect = 0;
int g_TrainTotal = 0;
int g_TestCorrect = 0;
int g_TestTotal = 0;
int g_AccuracyTotalEligible = 0;
double g_CoveragePct = 0.0;

// OOS hranice
int g_TestBoundary = 0;
int g_FrozenTestBoundary = 0;

double g_ATRMean = 0.0;
datetime g_LastClosedBarTime = 0;

// GUI hodnoty pro progress bar (tohle se sype z DLLka)
int g_ProgEpoch = 0;
int g_ProgTotalEpochs = 0;
int g_ProgMB = 0;
int g_ProgTotalMB = 0;
int g_ProgTotalSteps = 0;
double g_ProgLR = 0.0;
double g_ProgMSE = 0.0;
double g_ProgBestMSE = 0.0;
double g_ProgGradNorm = 0.0;
double g_ProgPercent = 0.0;
double g_ProgElapsedSec = 0.0;
double g_ProgETASec = 0.0;

// Různé keše, ať to nepočítáme pořád dokola
double g_ATRCache[];
int g_ATRCacheStart = -1;
int g_ATRCachedBars = 0;
int g_LastPredictedTo = -1;

double g_EstVRAM_MB = 0;

bool g_NeedImmediatePredict = false;
bool g_LoadedFromFile = false;
string g_ModelFilePath = "";

// Shannon displej
double g_LastEntropyFastTF1 = 0.0;
double g_LastEntropySlowTF1 = 0.0;
double g_LastEntropyDeltaTF1 = 0.0;

// Promoce (Graduation) - když se model začne tvářit chytře v OOS
ENUM_TRAIN_PHASE g_TrainPhase = PHASE_IDLE;
int g_GraduationCount = 0;
double g_LastOOSAccuracy = 0.0;
bool g_OOSPassed = false;
int g_PreGradBoundary = 0;
int g_PostGradBoundary = 0;
bool g_PendingGraduation = false;

datetime g_LastGradTime = 0;
int g_LastGradBarCount = 0;
int g_BarsSinceLastGrad = 0;

GradHistoryEntry g_GradHistory[];
int g_GradHistoryCount = 0;

// Cache pro OOS retrain
datetime g_CachedTime[];
double g_CachedHigh[];
double g_CachedLow[];
double g_CachedClose[];
int g_CachedRatesTotal = 0;

// Cache entropie pro inference smyčku
double g_EntropyFastCache[];
double g_EntropySlowCache[];
int g_EntropyCacheSize = 0;
int g_EntropyCacheStartBar = -1;

//+------------------------------------------------------------------+
//| Posunutí grafu dopředu o predikované svíčky |
//+------------------------------------------------------------------+
int GetPlotShift()
  {
   return InpUsePlotShift ? InpPredictAhead : 0;
  }

//+------------------------------------------------------------------+
//| Kde nám končí In-Sample a začíná Out-of-Sample? |
//+------------------------------------------------------------------+
int GetActiveBoundary()
  {
   return (g_FrozenTestBoundary > 0) ? g_FrozenTestBoundary : g_TestBoundary;
  }

//+------------------------------------------------------------------+
//| Výpočet hranice testovacích dat. Zbytek si necháme na hraní. |
//+------------------------------------------------------------------+
int ComputeTestBoundary(int rates_total)
  {
   int minBar = InpPredictAhead + 1;
   int maxBar = rates_total - 1 - InpLookback;
   if(maxBar < minBar)
      return minBar;
   int totalRange = maxBar - minBar + 1;
   int testCount = (int)MathRound(totalRange * InpTestPct / 100.0);
   testCount = MathMax(testCount, 1);
   testCount = MathMin(testCount, totalRange - 1);
   return minBar + testCount;
  }

//+------------------------------------------------------------------+
//| Inicializace - jdeme na to. Snad se nám ten pískový hrad |
//| nesesype hned na startu. |
//+------------------------------------------------------------------+
int OnInit()
  {
   Print("=== Pure MTF LSTM v5.23 (reviewed & hardened) ===");
// Náhoda je blbec, tak to zkusíme aspoň trošku zrandomizovat
   MathSrand((int)((long)TimeCurrent() ^ (long)GetTickCount()));
   SetIndexBuffer(0, g_ProbPct, INDICATOR_DATA);
   SetIndexBuffer(1, g_CrossUp30, INDICATOR_DATA);
   SetIndexBuffer(2, g_CrossDown70, INDICATOR_DATA);
   ArraySetAsSeries(g_ProbPct, true);
   ArraySetAsSeries(g_CrossUp30, true);
   ArraySetAsSeries(g_CrossDown70, true);
   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetDouble(2, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(1, PLOT_ARROW, ARROW_UP);
   PlotIndexSetInteger(2, PLOT_ARROW, ARROW_DOWN);
   int shift = GetPlotShift();
   PlotIndexSetInteger(0, PLOT_SHIFT, shift);
   PlotIndexSetInteger(1, PLOT_SHIFT, shift);
   PlotIndexSetInteger(2, PLOT_SHIFT, shift);
   if(shift > 0)
     {
      if(!ChartGetInteger(0, CHART_SHIFT))
         ChartSetInteger(0, CHART_SHIFT, true);
      double shiftSize = ChartGetDouble(0, CHART_SHIFT_SIZE);
      if(shiftSize < 10.0)
         ChartSetDouble(0, CHART_SHIFT_SIZE, 10.0);
     }
   IndicatorSetString(INDICATOR_SHORTNAME,
                      StringFormat("MTF-LSTM-ENT(%s,%s,%s) PA%d T/T%.0f%% G%.0f%%",
                                   EnumToString(InpTF1), EnumToString(InpTF2), EnumToString(InpTF3),
                                   InpPredictAhead, InpTestPct, InpOOSPassThreshold));
   IndicatorSetString(INDICATOR_LEVELTEXT, 0, "50% — WAIT zone");
   IndicatorSetString(INDICATOR_LEVELTEXT, 1, "30% — SELL zone");
   IndicatorSetString(INDICATOR_LEVELTEXT, 2, "70% — BUY zone");
// Křísíme GPU
   if(!InitNetwork())
     {
      Print("FATAL: Network init failed. GPU asi stávkuje.");
      return INIT_FAILED;
     }
   g_ModelFilePath = BuildModelFileName();
   Print("Model file: ", g_ModelFilePath);
   g_LoadedFromFile = false;
   if(InpSaveModel && FileIsExist(g_ModelFilePath, FILE_COMMON))
     {
      if(LoadModel())
        {
         Print("Model loaded — skippuju učení, jdeme rovnou na věc.");
         g_ModelReady = true;
         g_LoadedFromFile = true;
         g_NeedImmediatePredict = true;
         g_LastTrainBar = 0;
        }
      else
         Print("Model load failed — no nic, trénujeme od nuly.");
     }
// Vynulujeme stavy
   g_LastClosedBarTime = 0;
   g_LastPredictedTo = -1;
   g_TestBoundary = 0;
   if(!g_LoadedFromFile)
      g_FrozenTestBoundary = 0;
   g_TrainPhase = PHASE_IDLE;
   g_GraduationCount = 0;
   g_LastOOSAccuracy = 0.0;
   g_OOSPassed = false;
   g_PendingGraduation = false;
   g_LastGradTime = 0;
   g_LastGradBarCount = 0;
   g_BarsSinceLastGrad = 0;
   g_GradHistoryCount = 0;
   ArrayResize(g_GradHistory, 0);
   g_EntropyCacheSize = 0;
   g_EntropyCacheStartBar = -1;
// Nakreslíme tabulky
   CreateInfoPanel();
   CreateProgressBar();
   EventSetMillisecondTimer(100);
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
//| Úklid - Vysypeme z botiček poslední zrnka písku a jdeme. |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   if(g_NetHandle > 0 && g_IsTraining)
     {
      DN_StopTraining(g_NetHandle);
      Sleep(500); // Dáme tomu chvilku na vydechnutí
     }
   if(InpSaveModel && g_ModelReady && g_NetHandle > 0)
      SaveModel();
   if(g_NetHandle > 0)
     {
      DN_Free(g_NetHandle);
      g_NetHandle = 0;
     }
   CleanupObjects();
  }

//+------------------------------------------------------------------+
//| Kategorizace svíčky - přežvýkáme cenu na jednoduché vzory |
//+------------------------------------------------------------------+
int EncodeCandle(double open, double high, double low, double close)
  {
   double range = high - low;
   if(range < _Point)
      return 4; // Doji jak prase
   double bodyAbs = MathAbs(close - open);
   double bodyRatio = bodyAbs / range;
   double upperShadow = high - MathMax(open, close);
   double lowerShadow = MathMin(open, close) - low;
   bool bullish = (close >= open);
   if(bodyRatio < 0.10)
      return 4;
   int result;
   if(bullish)
     {
      if(bodyRatio > 0.70)
         result = 8; // Full bull
      else
         if(bodyRatio > 0.30)
            result = 7;
         else
            if(lowerShadow > upperShadow)
               result = 5; // Pinbar up
            else
               result = 6;
     }
   else
     {
      if(bodyRatio > 0.70)
         result = 0; // Full bear
      else
         if(bodyRatio > 0.30)
            result = 1;
         else
            if(upperShadow > lowerShadow)
               result = 2; // Pinbar down
            else
               result = 3;
     }
// Pro jistotu, kdyby se matematika zbláznila
   if(result < 0 || result > 8)
      result = 4;
   return result;
  }

//+------------------------------------------------------------------+
//| Featury pro model (Momentum, pozice, ATR, atd.) |
//+------------------------------------------------------------------+
double GetMomentumFeature(ENUM_TIMEFRAMES tf, int sh, int back)
  {
   int bars = iBars(_Symbol, tf);
   if(sh < 0 || sh + back >= bars)
      return 0.0;
   double c0 = iClose(_Symbol, tf, sh);
   double c1 = iClose(_Symbol, tf, sh + back);
   if(MathAbs(c1) < _Point)
      return 0.0;
   return MathMax(-2.0, MathMin(2.0, (c0 - c1) / c1));
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double GetRangePositionFeature(ENUM_TIMEFRAMES tf, int sh, int wnd)
  {
   int bars = iBars(_Symbol, tf);
   if(sh < 0 || sh + wnd >= bars)
      return 0.5;
   double hi = -DBL_MAX;
   double lo = DBL_MAX;
   for(int i = 0; i < wnd; i++)
     {
      double h = iHigh(_Symbol, tf, sh + i);
      double l = iLow(_Symbol, tf, sh + i);
      if(h > hi)
         hi = h;
      if(l < lo)
         lo = l;
     }
   double c = iClose(_Symbol, tf, sh);
   double rng = hi - lo;
   if(rng < _Point)
      return 0.5;
   return MathMax(0.0, MathMin(1.0, (c - lo) / rng));
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double GetAtrRatioFeature(ENUM_TIMEFRAMES tf, int sh, int period)
  {
   int bars = iBars(_Symbol, tf);
   if(sh < 0 || sh + period + 1 >= bars)
      return 1.0;
   double sum = 0.0;
   int cnt = 0;
   for(int i = 0; i < period; i++)
     {
      int b = sh + i;
      if(b + 1 >= bars)
         break;
      double h = iHigh(_Symbol, tf, b);
      double l = iLow(_Symbol, tf, b);
      double pc = iClose(_Symbol, tf, b + 1);
      double tr = MathMax(h - l, MathMax(MathAbs(h - pc), MathAbs(l - pc)));
      sum += tr;
      cnt++;
     }
   if(cnt <= 0)
      return 1.0;
   double atr = sum / cnt;
   double base = MathMax(g_ATRMean, _Point * 100);
   return MathMax(0.0, MathMin(5.0, atr / base));
  }

//+------------------------------------------------------------------+
//| Entropie. Měříme, jak moc se ten trh sype jako písek mezi prsty. |
//+------------------------------------------------------------------+
int GetEntropyStateAtShift(ENUM_TIMEFRAMES tf, int sh, int priceStepPoints)
  {
   int bars = iBars(_Symbol, tf);
   if(sh < 0 || sh + 1 >= bars)
      return 0;
   double c0 = iClose(_Symbol, tf, sh);
   double c1 = iClose(_Symbol, tf, sh + 1);
   double delta = c0 - c1;
   double thr = priceStepPoints * _Point;
   if(delta > thr)
      return 1;
   if(delta < -thr)
      return 2;
   return 0; // Flat
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CalculateShannonEntropyTF(ENUM_TIMEFRAMES tf, int sh, int period, int priceStepPoints)
  {
   int bars = iBars(_Symbol, tf);
   if(period <= 1 || sh < 0 || sh + period + 1 >= bars)
      return 0.5;
   int cntFlat = 0, cntUp = 0, cntDown = 0;
   for(int i = 0; i < period; i++)
     {
      int st = GetEntropyStateAtShift(tf, sh + i, priceStepPoints);
      if(st == 1)
         cntUp++;
      else
         if(st == 2)
            cntDown++;
         else
            cntFlat++;
     }
   double pFlat = (double)cntFlat / period;
   double pUp = (double)cntUp / period;
   double pDown = (double)cntDown / period;
   double H = 0.0;
   if(pFlat > 0.0)
      H -= pFlat * MathLog(pFlat) / LN2_CONST;
   if(pUp > 0.0)
      H -= pUp * MathLog(pUp) / LN2_CONST;
   if(pDown > 0.0)
      H -= pDown * MathLog(pDown) / LN2_CONST;
   double Hmax = MathLog(3.0) / LN2_CONST;
   if(Hmax <= 0.0)
      return 0.5;
   double nH = H / Hmax;
   return MathMax(0.0, MathMin(1.0, nH));
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string EntropyModeText(double e)
  {
   if(e < 0.35)
      return "TREND";
   if(e < 0.65)
      return "TRANSITION";
   return "CHAOTIC";
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double ApplyEntropyConfidenceFilter(double pBull, double entropySlow)
  {
   if(!InpUseEntropyFilter)
      return pBull;
// Pokud je trh aspoň trochu čitelný, necháme to být
   if(entropySlow <= InpEntropyChaosLevel)
      return pBull;
// Když to lítá náhodně, stahujeme pravděpodobnost zpátky k 50 %
   double excess = (entropySlow - InpEntropyChaosLevel) / MathMax(1e-8, (1.0 - InpEntropyChaosLevel));
   excess = MathMax(0.0, MathMin(1.0, excess));
   double strength = MathMax(0.0, MathMin(1.0, InpEntropyFilterPower));
   double mix = excess * strength;
   return pBull * (1.0 - mix) + 0.5 * mix;
  }

//+------------------------------------------------------------------+
//| Skládáme data z více timeframeů |
//+------------------------------------------------------------------+
bool IsTFBarValid(ENUM_TIMEFRAMES tf, int sh, datetime refTime)
  {
   if(sh < 0)
      return false;
   datetime barTime = iTime(_Symbol, tf, sh);
   long tfSeconds = PeriodSeconds(tf);
   if(tfSeconds <= 0)
      return true;
// Kontrola, jestli ten bar není 10 let starý kvůli dírám v datech
   return (MathAbs((long)barTime - (long)refTime) <= tfSeconds * 2);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void GetTFData(ENUM_TIMEFRAMES tf, datetime t, double &feats[], int offset)
  {
   int sh = iBarShift(_Symbol, tf, t, false);
   if(sh < 0 || !IsTFBarValid(tf, sh, t))
     {
      for(int i = 0; i < FEAT_PER_TF; i++)
         feats[offset + i] = 0.0;
      return;
     }
   double o = iOpen(_Symbol, tf, sh), h = iHigh(_Symbol, tf, sh),
          l = iLow(_Symbol, tf, sh), c = iClose(_Symbol, tf, sh);
   long vol = iTickVolume(_Symbol, tf, sh);
   long prevVol = (sh + 1 < iBars(_Symbol, tf)) ? iTickVolume(_Symbol, tf, sh + 1) : 0;
   double range = h - l;
   if(range < _Point)
      range = _Point;
   int sym = EncodeCandle(o, h, l, c);
// One-hot encoding svíčky
   for(int k = 0; k < 9; k++)
      feats[offset + k] = (k == sym) ? 1.0 : 0.0;
   feats[offset + 9] = (c - o) / range;
   feats[offset + 10] = ((h - MathMax(o, c)) - (MathMin(o, c) - l)) / range;
   feats[offset + 11] = MathMax(-2.0, MathMin(2.0, (prevVol > 0) ? (double)(vol - prevVol) / prevVol : 0.0));
   feats[offset + 12] = GetMomentumFeature(tf, sh, 5);
   feats[offset + 13] = GetAtrRatioFeature(tf, sh, 14);
   feats[offset + 14] = GetRangePositionFeature(tf, sh, 20);
   double eFast = CalculateShannonEntropyTF(tf, sh, InpEntropyFastPeriod, InpEntropyPriceStep);
   double eSlow = CalculateShannonEntropyTF(tf, sh, InpEntropySlowPeriod, InpEntropyPriceStep);
   double eDiff = eFast - eSlow;
   feats[offset + 15] = eFast;
   feats[offset + 16] = eSlow;
   feats[offset + 17] = MathMax(-1.0, MathMin(1.0, eDiff));
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void ExtractMTFFeatures(int startBar, int count, const datetime &time[], int rates_total, double &feats[])
  {
   ArrayResize(feats, count * FEAT_PER_BAR);
   ArrayInitialize(feats, 0.0);
   for(int i = 0; i < count; i++)
     {
      int bar = startBar + i, off = i * FEAT_PER_BAR;
      if(bar < 1 || bar >= rates_total)
         continue;
      datetime t = time[bar];
      GetTFData(InpTF1, t, feats, off);
      GetTFData(InpTF2, t, feats, off + FEAT_PER_TF);
      GetTFData(InpTF3, t, feats, off + FEAT_PER_TF * 2);
     }
  }

//+------------------------------------------------------------------+
//| Nastavení sítě a odhad paměti. Aby to nebouchlo. |
//+------------------------------------------------------------------+
bool InitNetwork()
  {
   g_NetHandle = DN_Create();
   if(g_NetHandle == 0)
     {
      Print("DN_Create fail: ", GetDLLError());
      return false;
     }
   DN_SetSequenceLength(g_NetHandle, InpLookback);
   DN_SetMiniBatchSize(g_NetHandle, InpMiniBatch);
   DN_SetGradClip(g_NetHandle, InpGradClip);
   if(!DN_AddLayerEx(g_NetHandle, FEAT_PER_BAR, InpHiddenSize1, 0, 0, InpDropout))
     { Print("L1 fail"); DN_Free(g_NetHandle); g_NetHandle = 0; return false; }
   if(!DN_AddLayerEx(g_NetHandle, 0, InpHiddenSize2, 0, 0, InpDropout * 0.5))
     { Print("L2 fail"); DN_Free(g_NetHandle); g_NetHandle = 0; return false; }
   if(InpHiddenSize3 > 0)
      if(!DN_AddLayerEx(g_NetHandle, 0, InpHiddenSize3, 0, 0, 0.0))
        { Print("L3 fail"); DN_Free(g_NetHandle); g_NetHandle = 0; return false; }
   if(!DN_SetOutputDim(g_NetHandle, OUTPUT_DIM))
     { Print("OutDim fail"); DN_Free(g_NetHandle); g_NetHandle = 0; return false; }
   EstimateVRAM();
   Print(StringFormat("Network: %d->LSTM(%d)->LSTM(%d)%s->%d",
                      FEAT_PER_BAR, InpHiddenSize1, InpHiddenSize2,
                      InpHiddenSize3 > 0 ? StringFormat("->LSTM(%d)", InpHiddenSize3) : "", OUTPUT_DIM));
   return true;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void EstimateVRAM()
  {
   int H1 = InpHiddenSize1, H2 = InpHiddenSize2, H3 = InpHiddenSize3;
   int lastH = (H3 > 0) ? H3 : H2;
   double w1 = (double)(FEAT_PER_BAR + H1) * 4 * H1 * 4;
   double w2 = (double)(H1 + H2) * 4 * H2 * 4;
   double w3 = (H3 > 0) ? (double)(H2 + H3) * 4 * H3 * 4 : 0.0;
   double wo = (double)lastH * OUTPUT_DIM * 4;
   double wm = (w1 + w2 + w3 + wo) * 3.0;
   double dm = (double)InpTrainBars * InpLookback * FEAT_PER_BAR * 4.0 + (double)InpTrainBars * OUTPUT_DIM * 4.0;
   double cm = (double)InpLookback * InpMiniBatch * (H1 + H2) * 4.0 * 7.0;
   if(H3 > 0)
      cm += (double)InpLookback * InpMiniBatch * H3 * 4.0 * 7.0;
   double pm = (double)InpPredictBatch * InpLookback * FEAT_PER_BAR * 8.0 + (double)InpPredictBatch * OUTPUT_DIM * 8.0;
   g_EstVRAM_MB = (wm + dm + cm * 1.5 + pm) / (1024.0 * 1024.0);
  }

//+------------------------------------------------------------------+
//| Různé utility |
//+------------------------------------------------------------------+
void InvalidateCache()
  {
   g_ATRCacheStart = -1;
   g_ATRCachedBars = 0;
   g_LastPredictedTo = -1;
   g_EntropyCacheSize = 0;
   g_EntropyCacheStartBar = -1;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void ShuffleIntArray(int &arr[], int n)
  {
   for(int i = n - 1; i > 0; i--)
     {
      int j = (int)(MathRand() % (i + 1));
      int t = arr[i];
      arr[i] = arr[j];
      arr[j] = t;
     }
  }

//+------------------------------------------------------------------+
//| Vykreslí dělicí čáru mezi učením a realitou (OOS) |
//+------------------------------------------------------------------+
void UpdateSplitLine(const datetime &time[], int rates_total)
{
   long chart_id    = ChartID();
   string lineName  = g_SplitPrefix + "VLine";
   string labelName = g_SplitPrefix + "Label";

   int activeBoundary = GetActiveBoundary();
   if(!InpShowSplitLine || activeBoundary <= 0 || activeBoundary >= rates_total)
   {
      ObjectDelete(chart_id, lineName);
      ObjectDelete(chart_id, labelName);
      return;
   }

   datetime splitTime = time[activeBoundary];

   if(ObjectFind(chart_id, lineName) < 0)
      ObjectCreate(chart_id, lineName, OBJ_VLINE, 0, splitTime, 0);
   else
      ObjectSetInteger(chart_id, lineName, OBJPROP_TIME, splitTime);

   ObjectSetInteger(chart_id, lineName, OBJPROP_COLOR,      clrOrangeRed);
   ObjectSetInteger(chart_id, lineName, OBJPROP_STYLE,      STYLE_DASH);
   ObjectSetInteger(chart_id, lineName, OBJPROP_WIDTH,      1);
   ObjectSetInteger(chart_id, lineName, OBJPROP_BACK,       true);
   ObjectSetInteger(chart_id, lineName, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(chart_id, lineName, OBJPROP_HIDDEN,     true);
   ObjectSetInteger(chart_id, lineName, OBJPROP_TIMEFRAMES, OBJ_ALL_PERIODS);
   ObjectSetString( chart_id, lineName, OBJPROP_TOOLTIP,
                    StringFormat("Train/Test split [Grad #%d]", g_GraduationCount));

   if(ObjectFind(chart_id, labelName) < 0)
   {
      ObjectCreate(chart_id, labelName, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(chart_id, labelName, OBJPROP_CORNER,     CORNER_LEFT_UPPER);
      ObjectSetInteger(chart_id, labelName, OBJPROP_XDISTANCE,  15);
      ObjectSetInteger(chart_id, labelName, OBJPROP_YDISTANCE,  405);
      ObjectSetInteger(chart_id, labelName, OBJPROP_FONTSIZE,   8);
      ObjectSetString( chart_id, labelName, OBJPROP_FONT,       "Arial");
      ObjectSetInteger(chart_id, labelName, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(chart_id, labelName, OBJPROP_HIDDEN,     true);
      ObjectSetInteger(chart_id, labelName, OBJPROP_TIMEFRAMES, OBJ_ALL_PERIODS);
   }

   ObjectSetInteger(chart_id, labelName, OBJPROP_COLOR, clrOrangeRed);

   int oosFirst = InpPredictAhead + 1;
   int oosLast  = activeBoundary - 1;
   int oosCount = oosLast - oosFirst + 1;
   ObjectSetString(chart_id, labelName, OBJPROP_TEXT,
                   StringFormat("OOS [bar %d..%d] (%d bars) | boundary %d | Grad #%d",
                                oosFirst, oosLast, oosCount,
                                activeBoundary, g_GraduationCount));
}

//+------------------------------------------------------------------+
//| HLAVNÍ SMYČKA - Písek v přesýpacích hodinách se sem sype. |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
// Otočíme pole prdelí dopředu, ať se nám to líp indexuje
   ArraySetAsSeries(time, true);
   ArraySetAsSeries(open, true);
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   ArraySetAsSeries(close, true);
   ArraySetAsSeries(tick_volume, true);
   int minBars = InpLookback + InpPredictAhead + MathMax(100, InpEntropySlowPeriod + 20);
   if(rates_total < minBars)
      return 0; // Málo dat, jdeme na párek.
   g_TotalBars = rates_total;
   g_TestBoundary = ComputeTestBoundary(rates_total);
   CacheRateData(time, high, low, close, rates_total);
   bool newClosedBar = false;
   if(rates_total >= 2)
     {
      datetime t = time[1];
      if(t != g_LastClosedBarTime)
        {
         g_LastClosedBarTime = t;
         newClosedBar = true;
         g_LastPredictedTo = -1;
         g_BarsSinceLastGrad++;
        }
     }
   if(prev_calculated == 0)
     {
      ArrayInitialize(g_ProbPct, EMPTY_VALUE);
      ArrayInitialize(g_CrossUp30, EMPTY_VALUE);
      ArrayInitialize(g_CrossDown70, EMPTY_VALUE);
     }
   if(g_LoadedFromFile && g_LastTrainBar == 0)
      g_LastTrainBar = rates_total;
// Start úvodního učení (pokud nemáme model načtený z disku)
   if(!g_ModelReady && !g_IsTraining && !g_LoadedFromFile)
     {
      if(StartTraining(rates_total, time, high, low, close, true))
        {
         g_IsTraining = true;
         g_TrainPhase = PHASE_INITIAL_TRAIN;
        }
     }
// Periodické přeučování (aby model úplně nezblbnul, když se trh změní)
   if(g_ModelReady && InpAutoRetrain && !g_IsTraining && newClosedBar
      && g_TrainPhase == PHASE_IDLE && !g_PendingGraduation)
     {
      if(rates_total - g_LastTrainBar >= InpRetrainInterval)
        {
         if(StartTraining(rates_total, time, high, low, close, false))
           {
            g_IsTraining = true;
            g_TrainPhase = PHASE_PERIODIC_RETRAIN;
           }
        }
     }
// Predikce, když se model zrovna neučí
   if(g_ModelReady && !g_IsTraining)
     {
      if(newClosedBar || prev_calculated == 0 || g_NeedImmediatePredict)
        {
         BulkPredictAndEvaluate(rates_total, time, high, low, close);
         UpdateCrossSignals(rates_total, time);
         g_NeedImmediatePredict = false;
         if(InpAutoGraduate
            && g_TrainPhase == PHASE_IDLE
            && !g_PendingGraduation
            && g_BarsSinceLastGrad >= InpGradCooldownBars)
           {
            CheckAndTriggerGraduation(rates_total, time, high, low, close);
           }
        }
     }
// 0. bar (formující) kecá, radši mu nevěříme
   g_ProbPct[0] = EMPTY_VALUE;
   g_CrossUp30[0] = EMPTY_VALUE;
   g_CrossDown70[0] = EMPTY_VALUE;
   UpdateSplitLine(time, rates_total);
   UpdateInfoPanel();
   UpdateProgressBar();
   return rates_total;
  }

//+------------------------------------------------------------------+
//| Ukládáme si data bokem, abychom při OOS retrainu měli co žrát. |
//+------------------------------------------------------------------+
void CacheRateData(const datetime &time[], const double &high[],
                   const double &low[], const double &close[], int rates_total)
{
   g_CachedRatesTotal = rates_total;

   if(ArraySize(g_CachedTime) != rates_total)
   {
      ArrayResize(g_CachedTime,  rates_total);
      ArrayResize(g_CachedHigh,  rates_total);
      ArrayResize(g_CachedLow,   rates_total);
      ArrayResize(g_CachedClose, rates_total);
   }

   // Nejprve vypneme AsSeries flag na cílových polích,
   // aby ArrayCopy kopírovalo fyzicky index za indexem
   ArraySetAsSeries(g_CachedTime,  false);
   ArraySetAsSeries(g_CachedHigh,  false);
   ArraySetAsSeries(g_CachedLow,   false);
   ArraySetAsSeries(g_CachedClose, false);

   ArrayCopy(g_CachedTime,  time,  0, 0, rates_total);
   ArrayCopy(g_CachedHigh,  high,  0, 0, rates_total);
   ArrayCopy(g_CachedLow,   low,   0, 0, rates_total);
   ArrayCopy(g_CachedClose, close, 0, 0, rates_total);

   // Nyní nastavíme AsSeries=true — fyzická data jsou zkopírována
   // ve stejném pořadí jako zdrojová série (index 0 = nejnovější),
   // flag jen řekne MQL5 jak číst indexy
   ArraySetAsSeries(g_CachedTime,  true);
   ArraySetAsSeries(g_CachedHigh,  true);
   ArraySetAsSeries(g_CachedLow,   true);
   ArraySetAsSeries(g_CachedClose, true);
}

//+------------------------------------------------------------------+
//| Kešování entropie, ať to CPU neodpálí při BulkPredict. |
//+------------------------------------------------------------------+
void PrecomputeEntropyCache(int startBar, int count, const datetime &time[], int rates_total)
  {
   if(count <= 0)
      return;
   bool needRecompute = (g_EntropyCacheStartBar != startBar) || (g_EntropyCacheSize < count);
   if(!needRecompute)
      return;
   ArrayResize(g_EntropyFastCache, count);
   ArrayResize(g_EntropySlowCache, count);
   g_EntropyCacheStartBar = startBar;
   g_EntropyCacheSize = count;
   for(int i = 0; i < count; i++)
     {
      int barIdx = startBar + i;
      if(barIdx < 0 || barIdx >= rates_total)
        {
         g_EntropyFastCache[i] = 0.5;
         g_EntropySlowCache[i] = 0.5;
         continue;
        }
      datetime tBar = time[barIdx];
      int shTF1 = iBarShift(_Symbol, InpTF1, tBar, false);
      if(shTF1 >= 0 && IsTFBarValid(InpTF1, shTF1, tBar))
        {
         g_EntropyFastCache[i] = CalculateShannonEntropyTF(InpTF1, shTF1, InpEntropyFastPeriod, InpEntropyPriceStep);
         g_EntropySlowCache[i] = CalculateShannonEntropyTF(InpTF1, shTF1, InpEntropySlowPeriod, InpEntropyPriceStep);
        }
      else
        {
         g_EntropyFastCache[i] = 0.5;
         g_EntropySlowCache[i] = 0.5;
        }
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double GetCachedEntropyFast(int barIdx)
  {
   if(g_EntropyCacheStartBar < 0)
      return 0.5;
   int idx = barIdx - g_EntropyCacheStartBar;
   if(idx < 0 || idx >= g_EntropyCacheSize)
      return 0.5;
   return g_EntropyFastCache[idx];
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double GetCachedEntropySlow(int barIdx)
  {
   if(g_EntropyCacheStartBar < 0)
      return 0.5;
   int idx = barIdx - g_EntropyCacheStartBar;
   if(idx < 0 || idx >= g_EntropyCacheSize)
      return 0.5;
   return g_EntropySlowCache[idx];
  }

// Sloučená funkce - původně tu byly dvě dělající to samé (GetEntropySlowAtBar / GetEntropySlowForBar)
double GetEntropySlowForBar(int barIdx, const datetime &time[], int rates_total)
  {
   if(g_EntropyCacheStartBar >= 0)
     {
      int idx = barIdx - g_EntropyCacheStartBar;
      if(idx >= 0 && idx < g_EntropyCacheSize)
         return g_EntropySlowCache[idx];
     }
// Fallback, kdyby náhodou cache selhala
   if(barIdx < 0 || barIdx >= rates_total)
      return 0.5;
   datetime tBar = time[barIdx];
   int shTF1 = iBarShift(_Symbol, InpTF1, tBar, false);
   if(shTF1 >= 0 && IsTFBarValid(InpTF1, shTF1, tBar))
      return CalculateShannonEntropyTF(InpTF1, shTF1, InpEntropySlowPeriod, InpEntropyPriceStep);
   return 0.5;
  }

//+------------------------------------------------------------------+
//| Graduation = když se síť osvědčí v OOS datech, posuneme hranici. |
//| Takový diplom pro úspěšný křemík. |
//+------------------------------------------------------------------+
void CheckAndTriggerGraduation(int rates_total, const datetime &time[], const double &high[], const double &low[], const double &close[])
  {
   if(g_GraduationCount >= InpMaxGraduations)
     {
      if(InpVerboseLog)
         Print("Max graduations reached: ", g_GraduationCount);
      return;
     }
   if(g_TestTotal < InpMinOOSSamples)
     {
      if(InpVerboseLog)
         Print(StringFormat("Not enough OOS samples: %d < %d", g_TestTotal, InpMinOOSSamples));
      return;
     }
   int activeBoundary = GetActiveBoundary();
   int oosBarCount = activeBoundary - (InpPredictAhead + 1);
   if(oosBarCount < InpMinOOSSamples)
     {
      if(InpVerboseLog)
         Print(StringFormat("OOS region too small to graduate: %d bars", oosBarCount));
      return;
     }
   double oosAcc = (g_TestTotal > 0) ? (double)g_TestCorrect / g_TestTotal * 100.0 : 0.0;
   g_LastOOSAccuracy = oosAcc;
   if(oosAcc >= InpOOSPassThreshold)
     {
      Print(StringFormat("=== OOS GRADUATION TRIGGERED #%d === accuracy: %.1f%% >= %.1f%% (%d/%d) | boundary: %d",
                         g_GraduationCount + 1, oosAcc, InpOOSPassThreshold, g_TestCorrect, g_TestTotal, activeBoundary));
      g_OOSPassed = true;
      g_PendingGraduation = true;
      g_PreGradBoundary = activeBoundary;
      if(StartOOSRetrain(rates_total, time, high, low, close))
        {
         g_IsTraining = true;
         g_TrainPhase = PHASE_OOS_RETRAIN;
        }
      else
        {
         Print("OOS retrain failed to start — jen posunu hranici a jedeme dál.");
         PerformBoundaryGraduation(rates_total);
         g_PendingGraduation = false;
        }
     }
   else
     {
      if(InpVerboseLog)
         Print(StringFormat("OOS accuracy %.1f%% < %.1f%% — no graduation, učíme se dál.", oosAcc, InpOOSPassThreshold));
     }
  }

//+------------------------------------------------------------------+
//| Nastartuje GPU učení speciálně pro OOS data (před promocí) |
//+------------------------------------------------------------------+
bool StartOOSRetrain(int rates_total, const datetime &time[], const double &high[], const double &low[], const double &close[])
  {
   if(g_NetHandle == 0)
      return false;
   int activeBoundary = GetActiveBoundary();
   int oosMinBar = InpPredictAhead + 1;
   int oosMaxBar = activeBoundary - 1;
   if(oosMaxBar < oosMinBar)
     {
      Print("StartOOSRetrain: empty OOS region (křemík nemá co žrát).");
      return false;
     }
   int candidateCnt = oosMaxBar - oosMinBar + 1;
   Print(StringFormat("OOS retrain region: bars [%d..%d] = %d bars", oosMinBar, oosMaxBar, candidateCnt));
   int candidates[];
   ArrayResize(candidates, candidateCnt);
   for(int i = 0; i < candidateCnt; i++)
      candidates[i] = oosMinBar + i;
   ShuffleIntArray(candidates, candidateCnt);
   int maxSamples = MathMin(candidateCnt, InpTrainBars);
// Uložíme zálohu váh, kdyby to dopadlo bledě
   DN_SnapshotWeights(g_NetHandle);
   g_TargetEpochs = InpOOSRetrainEpochs;
   g_CurrentEpochs = 0;
   g_TrainStartTime = TimeCurrent();
// Reset UI ukazatelů
   g_ProgEpoch = 0;
   g_ProgTotalEpochs = g_TargetEpochs;
   g_ProgMB = 0;
   g_ProgTotalMB = 0;
   g_ProgTotalSteps = 0;
   g_ProgLR = InpOOSRetrainLR;
   g_ProgMSE = 0.0;
   g_ProgBestMSE = 0.0;
   g_ProgGradNorm = 0.0;
   g_ProgPercent = 0.0;
   g_ProgElapsedSec = 0.0;
   g_ProgETASec = 0.0;
   if(g_ATRMean < _Point)
      ComputeATRMean(rates_total, close, high, low);
   int featNewest = candidates[0], featOldest = candidates[0];
   for(int s = 1; s < maxSamples; s++)
     {
      if(candidates[s] < featNewest)
         featNewest = candidates[s];
      if(candidates[s] > featOldest)
         featOldest = candidates[s];
     }
   featOldest = MathMin(featOldest + InpLookback - 1, rates_total - 1);
   int featRange = featOldest - featNewest + 1;
   double barFeats[];
   ExtractMTFFeatures(featNewest, featRange, time, rates_total, barFeats);
   int sampleBars[];
   ArrayResize(sampleBars, maxSamples);
   for(int s = 0; s < maxSamples; s++)
      sampleBars[s] = candidates[s];
   int inDim = InpLookback * FEAT_PER_BAR;
   double X[];
   BuildSequenceInput(barFeats, featNewest, featRange, sampleBars, maxSamples, InpLookback, X);
   double T[];
   ArrayResize(T, maxSamples * OUTPUT_DIM);
   ArrayInitialize(T, 0.0);
   int validSamples = 0;
   for(int s = 0; s < maxSamples; s++)
     {
      int predBar = sampleBars[s];
      int tgtBar = predBar - InpPredictAhead;
      if(tgtBar < 1 || tgtBar < oosMinBar || predBar >= rates_total)
         continue;
      double pc = close[predBar], cc = close[tgtBar];
      if(pc < _Point)
         pc = _Point;
      double ret = (cc - pc) / pc;
      double atr = GetCachedATR(predBar);
      if(atr < _Point)
         atr = _Point;
      double normRet = ret / (atr / pc);
      double pBull = 1.0 / (1.0 + MathExp(-normRet * 3.0));
      T[s * OUTPUT_DIM + 0] = pBull;
      T[s * OUTPUT_DIM + 1] = 1.0 - pBull;
      validSamples++;
     }
   if(validSamples < 10)
     {
      Print("StartOOSRetrain: too few valid samples: ", validSamples);
      DN_RestoreWeights(g_NetHandle);
      return false;
     }
   if(!DN_LoadBatch(g_NetHandle, X, T, maxSamples, inDim, OUTPUT_DIM, 0))
     {
      Print("OOS LoadBatch fail: ", GetDLLError());
      DN_RestoreWeights(g_NetHandle);
      return false;
     }
   if(!DN_TrainAsync(g_NetHandle, g_TargetEpochs, InpTargetMSE, InpOOSRetrainLR, InpWeightDecay))
     {
      Print("OOS TrainAsync fail: ", GetDLLError());
      DN_RestoreWeights(g_NetHandle);
      return false;
     }
   Print(StringFormat("OOS Retrain started: %d samples (%d valid) x %d epochs (LR=%.6f) | OOS bars [%d..%d]",
                      maxSamples, validSamples, g_TargetEpochs, InpOOSRetrainLR, oosMinBar, oosMaxBar));
   return true;
  }

//+------------------------------------------------------------------+
//| Zapsání diplomu (Posunutí hranice) |
//+------------------------------------------------------------------+
void PerformBoundaryGraduation(int rates_total)
  {
   int oldBoundary = GetActiveBoundary();
   int oosFirst = InpPredictAhead + 1;
   int oldOOSCount = oldBoundary - oosFirst;
   if(oldOOSCount <= 0)
     {
      Print("PerformBoundaryGraduation: no OOS bars to graduate");
      return;
     }
   int newOOSCount = (int)MathRound(oldOOSCount * InpTestPct / 100.0);
   newOOSCount = MathMax(newOOSCount, InpMinOOSSamples);
   newOOSCount = MathMin(newOOSCount, oldOOSCount - 1);
   if(newOOSCount <= 0)
      newOOSCount = 1;
   int newBoundary = oosFirst + newOOSCount;
   if(newBoundary >= oldBoundary)
      newBoundary = oldBoundary - 1;
   if(newBoundary < oosFirst + 1)
      newBoundary = oosFirst + 1;
   int graduatedBars = oldBoundary - newBoundary;
   int newOOSBars = newBoundary - oosFirst;
   GradHistoryEntry entry;
   entry.cycle = g_GraduationCount + 1;
   entry.oldBoundary = oldBoundary;
   entry.newBoundary = newBoundary;
   entry.oosAccuracy = g_LastOOSAccuracy;
   entry.oosSamples = g_TestTotal;
   entry.mseBefore = g_LastMSE;
   entry.mseAfter = g_ProgMSE > 0 ? g_ProgMSE : g_LastMSE;
   entry.timestamp = TimeCurrent();
   g_GradHistoryCount++;
   ArrayResize(g_GradHistory, g_GradHistoryCount);
   g_GradHistory[g_GradHistoryCount - 1] = entry;
   g_FrozenTestBoundary = newBoundary;
   g_GraduationCount++;
   g_LastGradTime = TimeCurrent();
   g_LastGradBarCount = g_TotalBars;
   g_BarsSinceLastGrad = 0;
// Vynulujeme statistiky pro nový cyklus
   g_TrainCorrect = 0;
   g_TrainTotal = 0;
   g_TestCorrect = 0;
   g_TestTotal = 0;
   g_AccuracyTotalEligible = 0;
   g_CoveragePct = 0.0;
   g_OOSPassed = false;
   Print(StringFormat("=== GRADUATION #%d COMPLETE ===", g_GraduationCount));
   Print(StringFormat(" boundary: %d -> %d (shifted by %d bars)", oldBoundary, newBoundary, graduatedBars));
   Print(StringFormat(" graduated %d bars into training | new OOS: %d bars [%d..%d]", graduatedBars, newOOSBars, oosFirst, newBoundary - 1));
   Print(StringFormat(" OOS accuracy was: %.1f%% (%d/%d)", g_LastOOSAccuracy, g_TestCorrect, g_TestTotal));
  }

//+------------------------------------------------------------------+
//| Časovač - Koukáme, jestli už to GPU sežvýkalo |
//+------------------------------------------------------------------+
void OnTimer()
{
   if(!g_IsTraining || g_NetHandle == 0)
      return;

   PollGPUProgress();

   int st = DN_GetTrainingStatus(g_NetHandle);

   // Status 1 = ještě maká
   if(st == 1)
   {
      UpdateInfoPanel();
      UpdateProgressBar();
      return;
   }

   g_IsTraining = false;

   double mse = 0.0;
   int    ep  = 0;
   DN_GetTrainingResult(g_NetHandle, mse, ep);
   g_LastMSE      = mse;
   g_CurrentEpochs = ep;
   g_TotalEpochs  += ep;

   // Status 2 = hotovo
   if(st == 2)
   {
      Print(StringFormat("Training done [%s]: MSE=%.6f ep=%d elapsed=%.1fs",
                         PhaseToString(g_TrainPhase), mse, ep, g_ProgElapsedSec));

      if(mse < g_BestMSE)
      {
         g_BestMSE = mse;
         DN_SnapshotWeights(g_NetHandle);
      }

      switch(g_TrainPhase)
      {
         case PHASE_INITIAL_TRAIN:
         case PHASE_PERIODIC_RETRAIN:
            g_ModelReady  = true;
            g_LastTrainBar = g_TotalBars;
            InvalidateCache();
            g_TrainCorrect = 0;
            g_TrainTotal   = 0;
            g_TestCorrect  = 0;
            g_TestTotal    = 0;
            g_AccuracyTotalEligible = 0;
            g_CoveragePct  = 0.0;
            g_TrainPhase   = PHASE_IDLE;
            if(InpSaveModel)
               SaveModel();
            g_NeedImmediatePredict = true;
            break;

         case PHASE_OOS_RETRAIN:
         {
            g_ModelReady   = true;
            g_LastTrainBar = g_TotalBars;
            InvalidateCache();

            int ratesForGrad = (g_CachedRatesTotal > 0)
                               ? g_CachedRatesTotal
                               : g_TotalBars;
            if(ratesForGrad <= 0)
            {
               Print("ERROR: Cannot perform graduation - no rate data cached");
               g_PendingGraduation = false;
               g_TrainPhase = PHASE_IDLE;
               break;
            }
            PerformBoundaryGraduation(ratesForGrad);
            g_TrainPhase        = PHASE_IDLE;
            g_PendingGraduation = false;
            if(InpSaveModel)
               SaveModel();
            g_NeedImmediatePredict = true;
            break;
         }

         default:
            g_TrainPhase = PHASE_IDLE;
            break;
      }

      // OPRAVA: ChartSetSymbolPeriod způsoboval reinicializaci celého indikátoru
      // Stačí pouze překreslit graf
      ChartRedraw();
   }
   else
   {
      Print("Training error [", PhaseToString(g_TrainPhase), "]: ", GetDLLError());

      if(g_TrainPhase == PHASE_OOS_RETRAIN)
      {
         DN_RestoreWeights(g_NetHandle);
         Print("OOS retrain failed — weights restored, graduation cancelled");
         g_PendingGraduation = false;
      }
      else
      {
         DN_RestoreWeights(g_NetHandle);
      }
      g_TrainPhase = PHASE_IDLE;
   }

   UpdateInfoPanel();
   UpdateProgressBar();
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string PhaseToString(ENUM_TRAIN_PHASE phase)
  {
   switch(phase)
     {
      case PHASE_IDLE:
         return "IDLE";
      case PHASE_INITIAL_TRAIN:
         return "INITIAL";
      case PHASE_OOS_EVALUATE:
         return "OOS_EVAL";
      case PHASE_OOS_RETRAIN:
         return "OOS_RETRAIN";
      case PHASE_PERIODIC_RETRAIN:
         return "PERIODIC";
      default:
         return "UNKNOWN";
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void PollGPUProgress()
  {
   if(g_NetHandle == 0)
      return;
   if(!DN_GetProgressAll(g_NetHandle, g_ProgEpoch, g_ProgTotalEpochs,
                         g_ProgMB, g_ProgTotalMB, g_ProgLR, g_ProgMSE, g_ProgBestMSE,
                         g_ProgGradNorm, g_ProgPercent, g_ProgElapsedSec, g_ProgETASec))
      return;
   g_ProgTotalSteps = DN_GetProgressTotalSteps(g_NetHandle);
   g_CurrentEpochs = g_ProgEpoch;
   if(g_ProgMSE > 0)
      g_LastMSE = g_ProgMSE;
   if(g_ProgBestMSE > 0 && g_ProgBestMSE < 1e10)
      g_BestMSE = g_ProgBestMSE;
  }

//+------------------------------------------------------------------+
//| Hlavní trénovací funkce - cpeme křemíku čísla lopatou. |
//+------------------------------------------------------------------+
bool StartTraining(int rates_total, const datetime &time[],
                   const double &high[], const double &low[],
                   const double &close[], bool initial)
{
   if(g_NetHandle == 0)
      return false;

   g_TargetEpochs  = initial ? InpInitialEpochs : InpRetrainEpochs;
   g_CurrentEpochs = 0;
   g_TrainStartTime = TimeCurrent();

   g_ProgEpoch       = 0;
   g_ProgTotalEpochs = g_TargetEpochs;
   g_ProgMB          = 0;
   g_ProgTotalMB     = 0;
   g_ProgTotalSteps  = 0;
   g_ProgLR          = InpLearningRate;
   g_ProgMSE         = 0.0;
   g_ProgBestMSE     = 0.0;
   g_ProgGradNorm    = 0.0;
   g_ProgPercent     = 0.0;
   g_ProgElapsedSec  = 0.0;
   g_ProgETASec      = 0.0;

   ComputeATRMean(rates_total, close, high, low);

   g_TestBoundary = ComputeTestBoundary(rates_total);

   // OPRAVA: FrozenTestBoundary měníme POUZE při initial train
   // nebo pokud ještě nebyl nastaven — graduace by se jinak smazaly
   if(initial || g_FrozenTestBoundary == 0)
      g_FrozenTestBoundary = g_TestBoundary;

   if(initial)
   {
      g_GraduationCount   = 0;
      g_BarsSinceLastGrad = InpGradCooldownBars;
      g_GradHistoryCount  = 0;
      ArrayResize(g_GradHistory, 0);
   }

   int trainMinBar = g_FrozenTestBoundary + InpPredictAhead;
   int trainMaxBar = rates_total - 1 - InpLookback;

   if(trainMaxBar < trainMinBar)
   {
      Print(StringFormat("StartTraining: not enough train bars (trainMin=%d, trainMax=%d, boundary=%d)",
                         trainMinBar, trainMaxBar, g_FrozenTestBoundary));
      return false;
   }

   int candidateCnt = trainMaxBar - trainMinBar + 1;
   int candidates[];
   ArrayResize(candidates, candidateCnt);
   for(int i = 0; i < candidateCnt; i++)
      candidates[i] = trainMinBar + i;

   ShuffleIntArray(candidates, candidateCnt);
   int maxSamples = MathMin(candidateCnt, InpTrainBars);
   int inDim      = InpLookback * FEAT_PER_BAR;

   int featNewest = candidates[0], featOldest = candidates[0];
   for(int s = 1; s < maxSamples; s++)
   {
      if(candidates[s] < featNewest) featNewest = candidates[s];
      if(candidates[s] > featOldest) featOldest = candidates[s];
   }
   featOldest = MathMin(featOldest + InpLookback - 1, rates_total - 1);
   int featRange = featOldest - featNewest + 1;

   double barFeats[];
   ExtractMTFFeatures(featNewest, featRange, time, rates_total, barFeats);

   int sampleBars[];
   ArrayResize(sampleBars, maxSamples);
   for(int s = 0; s < maxSamples; s++)
      sampleBars[s] = candidates[s];

   double X[];
   BuildSequenceInput(barFeats, featNewest, featRange,
                      sampleBars, maxSamples, InpLookback, X);

   double T[];
   ArrayResize(T, maxSamples * OUTPUT_DIM);
   ArrayInitialize(T, 0.0);

   for(int s = 0; s < maxSamples; s++)
   {
      int predBar = sampleBars[s];
      int tgtBar  = predBar - InpPredictAhead;
      if(tgtBar < g_FrozenTestBoundary || tgtBar < 1 || predBar >= rates_total)
         continue;

      double pc = close[predBar], cc = close[tgtBar];
      if(pc < _Point) pc = _Point;

      double ret     = (cc - pc) / pc;
      double atr     = GetCachedATR(predBar);
      if(atr < _Point) atr = _Point;

      double normRet = ret / (atr / pc);
      double pBull   = 1.0 / (1.0 + MathExp(-normRet * 3.0));

      T[s * OUTPUT_DIM + 0] = pBull;
      T[s * OUTPUT_DIM + 1] = 1.0 - pBull;
   }

   if(!DN_LoadBatch(g_NetHandle, X, T, maxSamples, inDim, OUTPUT_DIM, 0))
   {
      Print("LoadBatch fail: ", GetDLLError());
      return false;
   }
   if(!DN_TrainAsync(g_NetHandle, g_TargetEpochs, InpTargetMSE,
                     InpLearningRate, InpWeightDecay))
   {
      Print("TrainAsync fail: ", GetDLLError());
      return false;
   }

   int testBars = g_FrozenTestBoundary - (InpPredictAhead + 1);
   Print(StringFormat(
      "Training [%s]: %d samples x %d epochs | TRAIN bars [%d..%d] | TEST bars [%d..%d] (%d bars)",
      initial ? "INITIAL" : "PERIODIC",
      maxSamples, g_TargetEpochs, trainMinBar, trainMaxBar,
      InpPredictAhead + 1, g_FrozenTestBoundary - 1, testBars));

   UpdateProgressBar();
   return true;
}

//+------------------------------------------------------------------+
//| Predikce ve velkém - necháme síť vyplivnout odhady a ohodnotíme |
//| si, jak moc nás tahá za nos. |
//+------------------------------------------------------------------+
void BulkPredictAndEvaluate(int rates_total, const datetime &time[], const double &high[], const double &low[], const double &close[])
{
   if(!g_ModelReady || g_NetHandle == 0 || g_IsTraining)
      return;

   g_TrainCorrect = 0;
   g_TrainTotal   = 0;
   g_TestCorrect  = 0;
   g_TestTotal    = 0;
   g_AccuracyTotalEligible = 0;
   g_CoveragePct  = 0.0;

   int activeBoundary = GetActiveBoundary();
   int newestPredBar  = 1;
   int oldestPredBar  = MathMin(rates_total - 1 - InpLookback,
                                newestPredBar + InpMaxPredictBars - 1);
   if(oldestPredBar < newestPredBar)
      return;

   int fullStartPred   = newestPredBar;
   int fullLastPred    = oldestPredBar;
   int fullTotalPredict = fullLastPred - fullStartPred + 1;
   if(fullTotalPredict <= 0)
      return;

   if(g_ATRMean < _Point)
      ComputeATRMean(rates_total, close, high, low);

   // Ochrana: count nesmí přetéct za konec pole
   int safeCount = MathMin(fullTotalPredict + InpLookback + 10,
                           rates_total - fullStartPred);
   BulkComputeATRWilder(fullStartPred, safeCount, 14, high, low, close, rates_total);
   PrecomputeEntropyCache(fullStartPred, fullTotalPredict, time, rates_total);

   int featNewest  = fullStartPred;
   int featOldest  = MathMin(fullLastPred + InpLookback - 1, rates_total - 1);
   int featRange   = featOldest - featNewest + 1;

   double barFeats[];
   ExtractMTFFeatures(featNewest, featRange, time, rates_total, barFeats);

   int batchSize   = InpPredictBatch;
   int inDim       = InpLookback * FEAT_PER_BAR;
   int drawLastPred = fullLastPred;
   if(g_LastPredictedTo != -1)
      drawLastPred = MathMin(fullLastPred, g_LastPredictedTo + 2);

   for(int batchStart = 0; batchStart < fullTotalPredict; batchStart += batchSize)
   {
      int curBatch = MathMin(batchSize, fullTotalPredict - batchStart);

      int sampleBars[];
      ArrayResize(sampleBars, curBatch);
      for(int i = 0; i < curBatch; i++)
         sampleBars[i] = fullStartPred + batchStart + i;

      double X[];
      BuildSequenceInput(barFeats, featNewest, featRange, sampleBars, curBatch, InpLookback, X);

      double Y[];
      ArrayResize(Y, curBatch * OUTPUT_DIM);
      if(!DN_PredictBatch(g_NetHandle, X, curBatch, inDim, 0, Y))
      {
         if(InpVerboseLog)
            Print("PredictBatch fail: ", GetDLLError());
         break;
      }

      for(int i = 0; i < curBatch; i++)
      {
         int barIdx = sampleBars[i];
         if(barIdx < 1 || barIdx >= rates_total)
            continue;

         double pBull = MathMax(0.0, MathMin(1.0, Y[i * OUTPUT_DIM + 0]));
         double pBear = MathMax(0.0, MathMin(1.0, Y[i * OUTPUT_DIM + 1]));
         double total = pBull + pBear;
         if(total > 0.001)
         {
            pBull /= total;
            pBear /= total;
         }
         else
         {
            pBull = 0.5;
            pBear = 0.5;
         }

         double eFast = GetCachedEntropyFast(barIdx);
         double eSlow = GetCachedEntropySlow(barIdx);
         double eDiff = eFast - eSlow;

         pBull = ApplyEntropyConfidenceFilter(pBull, eSlow);
         pBear = 1.0 - pBull;

         if(barIdx == 1)
         {
            g_LastEntropyFastTF1  = eFast;
            g_LastEntropySlowTF1  = eSlow;
            g_LastEntropyDeltaTF1 = eDiff;
         }

         if(barIdx <= drawLastPred)
            g_ProbPct[barIdx] = 100.0 * pBull;

         g_AccuracyTotalEligible++;

         double predDir = pBull - pBear;
         if(MathAbs(predDir) <= 0.1)
            continue; // neutrální zóna

         int evalBar = barIdx - InpPredictAhead;
         if(evalBar < 1 || evalBar >= barIdx)
            continue;

         double actualMove = close[evalBar] - close[barIdx];
         bool correct = (predDir > 0.0 && actualMove > 0.0) ||
                        (predDir < 0.0 && actualMove < 0.0);

         // OPRAVA: vyšší barIdx = starší bar = training data
         // nižší barIdx (blíže k 0) = novější bar = OOS/test data
         if(activeBoundary > 0 && barIdx >= activeBoundary)
         {
            // Starší bary — training region
            g_TrainTotal++;
            if(correct)
               g_TrainCorrect++;
         }
         else
         {
            // Novější bary — OOS/test region
            g_TestTotal++;
            if(correct)
               g_TestCorrect++;
         }
      }
   }

   g_LastPredictedTo = drawLastPred;
   int used = g_TrainTotal + g_TestTotal;
   g_CoveragePct = (g_AccuracyTotalEligible > 0)
                   ? (100.0 * used / g_AccuracyTotalEligible)
                   : 0.0;

   if(InpVerboseLog)
      Print(StringFormat("BulkPredict: train=%d/%d test=%d/%d coverage=%.1f%% boundary=%d grad=%d",
                         g_TrainCorrect, g_TrainTotal,
                         g_TestCorrect,  g_TestTotal,
                         g_CoveragePct, activeBoundary, g_GraduationCount));
}



//+------------------------------------------------------------------+
//| Vykreslení signálů (šipek) podle toho, kde se to kříží |
//+------------------------------------------------------------------+
void UpdateCrossSignals(int rates_total, const datetime &time[])
{
   ArrayInitialize(g_CrossUp30,   EMPTY_VALUE);
   ArrayInitialize(g_CrossDown70, EMPTY_VALUE);
   DeleteCrossObjects();

   int subWindow = GetMySubWindow();
   if(subWindow < 1)
      subWindow = 1;

   if(rates_total < 3)
      return;

   int shiftBars  = GetPlotShift();
   int bufferSize = ArraySize(g_CrossUp30); // skutečná velikost bufferu

   int maxSearchBar = MathMin(rates_total - 2,
                              InpMaxPredictBars + InpLookback + 10);

   int validBars[];
   int validCount = 0;
   ArrayResize(validBars, maxSearchBar + 1);

   for(int i = 1; i <= maxSearchBar; i++)
   {
      if(i < bufferSize && g_ProbPct[i] != EMPTY_VALUE)
      {
         validBars[validCount] = i;
         validCount++;
      }
   }

   if(validCount < 2)
      return;

   for(int vi = 0; vi < validCount - 1; vi++)
   {
      int iCurr = validBars[vi];
      int iPrev = validBars[vi + 1];

      if(iPrev - iCurr > 3)
         continue;

      double curr = g_ProbPct[iCurr];
      double prev = g_ProbPct[iPrev];

      if(curr == EMPTY_VALUE || prev == EMPTY_VALUE)
         continue;

      double diff = curr - prev;
      if(MathAbs(diff) < 1e-10)
         continue;

      // Entropie filtruje šipky pokud je to žádáno
      if(InpUseEntropyFilter && InpEntropyFilterCrossSignals)
      {
         double eSlow = GetEntropySlowForBar(iCurr, time, rates_total);
         if(eSlow > InpEntropyChaosLevel)
            continue;
      }

      datetime tCurr = time[iCurr];
      datetime tPrev = (iPrev < rates_total) ? time[iPrev] : time[iCurr];

      // === CROSS UP přes 30 ===
      if(prev < 30.0 && curr >= 30.0)
      {
         double k = (diff != 0.0) ? (30.0 - prev) / diff : 0.5;
         k = MathMax(0.0, MathMin(1.0, k));
         datetime tCross  = InterpolateTime(tPrev, tCurr, k);
         datetime tPlaced = ShiftTimeByBars(tCross, shiftBars);

         string name = g_CrossObjPrefix + "UP30_" + IntegerToString(iCurr);
         CreateCrossArrowObject(name, subWindow, tPlaced, 30.0,
                                clrDeepSkyBlue, ARROW_UP,
                                StringFormat("Cross UP 30 | bar=%d | %.2f->%.2f",
                                             iCurr, prev, curr));
         // Bezpečná kontrola hranic bufferu
         if(iCurr >= 1 && iCurr < bufferSize && iCurr < rates_total)
            g_CrossUp30[iCurr] = 30.0;
      }

      // === CROSS DOWN přes 70 ===
      if(prev > 70.0 && curr <= 70.0)
      {
         double k = (diff != 0.0) ? (70.0 - prev) / diff : 0.5;
         k = MathMax(0.0, MathMin(1.0, k));
         datetime tCross  = InterpolateTime(tPrev, tCurr, k);
         datetime tPlaced = ShiftTimeByBars(tCross, shiftBars);

         string name = g_CrossObjPrefix + "DN70_" + IntegerToString(iCurr);
         CreateCrossArrowObject(name, subWindow, tPlaced, 70.0,
                                clrTomato, ARROW_DOWN,
                                StringFormat("Cross DOWN 70 | bar=%d | %.2f->%.2f",
                                             iCurr, prev, curr));
         // Bezpečná kontrola hranic bufferu
         if(iCurr >= 1 && iCurr < bufferSize && iCurr < rates_total)
            g_CrossDown70[iCurr] = 70.0;
      }
   }

   // Formující bar (0) vždy prázdný
   if(bufferSize > 0)
   {
      g_CrossUp30[0]   = EMPTY_VALUE;
      g_CrossDown70[0] = EMPTY_VALUE;
   }

   ChartRedraw();
}

//+------------------------------------------------------------------+
//| Příprava dávky pro vstup do LSTM |
//+------------------------------------------------------------------+
void BuildSequenceInput(const double &barFeatures[], int featureStartBar, int totalFeatBars,
                        const int &sampleBars[], int nSamples, int seqLen, double &X[])
  {
   int inDim = seqLen * FEAT_PER_BAR;
   ArrayResize(X, nSamples * inDim);
   for(int s = 0; s < nSamples; s++)
     {
      int tgtBar = sampleBars[s];
      for(int t = 0; t < seqLen; t++)
        {
         int bar = tgtBar + (seqLen - 1 - t);
         int cIdx = bar - featureStartBar;
         int xOff = s * inDim + t * FEAT_PER_BAR;
         if(cIdx >= 0 && cIdx < totalFeatBars)
           {
            int fOff = cIdx * FEAT_PER_BAR;
            for(int f = 0; f < FEAT_PER_BAR; f++)
               X[xOff + f] = barFeatures[fOff + f];
           }
         else
           {
            for(int f = 0; f < FEAT_PER_BAR; f++)
               X[xOff + f] = 0.0;
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| Výpočet ATR (Wilder) ve velkém |
//+------------------------------------------------------------------+
void BulkComputeATRWilder(int startBar, int count, int period,
                          const double &high[], const double &low[],
                          const double &close[], int rates_total)
  {
   if(count <= 0 || period <= 0)
      return;
   if(g_ATRCacheStart == startBar && g_ATRCachedBars >= count)
      return;
   ArrayResize(g_ATRCache, count);
   g_ATRCacheStart = startBar;
   g_ATRCachedBars = count;
   ArrayInitialize(g_ATRCache, _Point * 100);
   int oldestBar = startBar + count - 1;
   if(oldestBar > rates_total - 2)
      oldestBar = rates_total - 2;
   if(startBar < 1 || oldestBar < startBar)
      return;
   double sumTR = 0.0;
   int trCount = 0;
   int initEndBar = oldestBar;
   for(int b = oldestBar; b >= startBar && trCount < period; b--)
     {
      if(b + 1 >= rates_total)
         continue;
      double tr = MathMax(high[b] - low[b],
                          MathMax(MathAbs(high[b] - close[b + 1]),
                                  MathAbs(low[b] - close[b + 1])));
      sumTR += tr;
      trCount++;
      initEndBar = b;
     }
   if(trCount <= 0)
      return;
   double atr = sumTR / trCount;
   int ci = initEndBar - startBar;
   if(ci >= 0 && ci < count)
      g_ATRCache[ci] = atr;
   for(int bar = initEndBar - 1; bar >= startBar; bar--)
     {
      if(bar + 1 >= rates_total)
         continue;
      double tr = MathMax(high[bar] - low[bar],
                          MathMax(MathAbs(high[bar] - close[bar + 1]),
                                  MathAbs(low[bar] - close[bar + 1])));
      atr = (atr * (period - 1) + tr) / period;
      ci = bar - startBar;
      if(ci >= 0 && ci < count)
         g_ATRCache[ci] = atr;
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double GetCachedATR(int barIdx)
  {
   if(g_ATRCacheStart < 0)
      return _Point * 100;
   int idx = barIdx - g_ATRCacheStart;
   if(idx < 0 || idx >= g_ATRCachedBars)
      return _Point * 100;
   return MathMax(g_ATRCache[idx], _Point);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void ComputeATRMean(int rates_total, const double &close[], const double &high[], const double &low[])
  {
   int n = MathMin(InpTrainBars, rates_total - InpLookback - 10);
   if(n <= 0)
     {
      g_ATRMean = _Point * 100;
      return;
     }
   BulkComputeATRWilder(1, n, 14, high, low, close, rates_total);
   double sum = 0.0;
   int cnt = 0;
   for(int i = 0; i < n; i++)
     {
      int bar = 1 + i;
      if(bar >= rates_total - 1)
         break;
      sum += GetCachedATR(bar);
      cnt++;
     }
   g_ATRMean = (cnt > 0) ? sum / cnt : _Point * 100;
   if(g_ATRMean < _Point)
      g_ATRMean = _Point;
  }

//+------------------------------------------------------------------+
//| Tvorba jména souboru. Ať se nám to na disku nepopere. |
//+------------------------------------------------------------------+
string BuildModelFileName()
  {
   string sym = _Symbol;
   StringReplace(sym, ".", "");
   StringReplace(sym, "/", "");
   StringReplace(sym, "#", "");
   StringReplace(sym, " ", "");
   string tf1 = TFToShortString(InpTF1);
   string tf2 = TFToShortString(InpTF2);
   string tf3 = TFToShortString(InpTF3);
   string layers = StringFormat("LSTM%dx%d", InpHiddenSize1, InpHiddenSize2);
   if(InpHiddenSize3 > 0)
      layers += "x" + IntegerToString(InpHiddenSize3);
   return StringFormat("%s_%s_%s-%s-%s_PA%d_L%d_%s_F%dx%d_TB%d_TT%.0f_ENT%d-%d_V523.lstm",
                       InpModelPrefix, sym, tf1, tf2, tf3, InpPredictAhead, InpLookback,
                       layers, FEAT_PER_BAR, OUTPUT_DIM, InpTrainBars, InpTestPct,
                       InpEntropyFastPeriod, InpEntropySlowPeriod);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string TFToShortString(ENUM_TIMEFRAMES tf)
  {
   if(tf == PERIOD_CURRENT)
      tf = (ENUM_TIMEFRAMES)_Period;
   switch(tf)
     {
      case PERIOD_M1:
         return "M1";
      case PERIOD_M2:
         return "M2";
      case PERIOD_M3:
         return "M3";
      case PERIOD_M4:
         return "M4";
      case PERIOD_M5:
         return "M5";
      case PERIOD_M6:
         return "M6";
      case PERIOD_M10:
         return "M10";
      case PERIOD_M12:
         return "M12";
      case PERIOD_M15:
         return "M15";
      case PERIOD_M20:
         return "M20";
      case PERIOD_M30:
         return "M30";
      case PERIOD_H1:
         return "H1";
      case PERIOD_H2:
         return "H2";
      case PERIOD_H3:
         return "H3";
      case PERIOD_H4:
         return "H4";
      case PERIOD_H6:
         return "H6";
      case PERIOD_H8:
         return "H8";
      case PERIOD_H12:
         return "H12";
      case PERIOD_D1:
         return "D1";
      case PERIOD_W1:
         return "W1";
      case PERIOD_MN1:
         return "MN1";
      default:
         return "TF" + IntegerToString((int)tf);
     }
  }

//+------------------------------------------------------------------+
//| UI panely - ať to trochu vypadá |
//+------------------------------------------------------------------+
void CreateInfoPanel()
{
   // MakeLabel nyní používá ChartID() interně,
   // takže stačí jen zavolat — funguje v hlavním i subokně
   MakeLabel(g_InfoPrefix + "T",
             "Pure MTF LSTM v5.23 (reviewed)",
             15, 35, clrDodgerBlue, 10, true);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void UpdateInfoPanel()
  {
   string st;
   color sc;
   if(g_IsTraining)
     {
      if(g_TrainPhase == PHASE_OOS_RETRAIN)
        {
         st = "GPU OOS Retrain...";
         sc = clrMagenta;
        }
      else
        {
         st = "GPU Training...";
         sc = clrYellow;
        }
     }
   else
      if(g_ModelReady)
        {
         st = "Ready";
         sc = clrLime;
        }
      else
        {
         st = "Waiting";
         sc = clrGray;
        }
   if(g_ModelReady && g_LoadedFromFile && g_TotalEpochs == 0)
      st += " (loaded)";
   MakeLabel(g_InfoPrefix + "St", "Status: " + st, 15, 55, sc, 9, false);
   string arch = StringFormat("MTF(%d)->LSTM(%d->%d", FEAT_PER_BAR, InpHiddenSize1, InpHiddenSize2);
   if(InpHiddenSize3 > 0)
      arch += "->" + IntegerToString(InpHiddenSize3);
   arch += ")->P(bull,bear)";
   MakeLabel(g_InfoPrefix + "Ar", arch, 15, 72, clrWhite, 8, false);
   double dispMSE = (g_IsTraining && g_ProgMSE > 0) ? g_ProgMSE : g_LastMSE;
   double dispBest = (g_IsTraining && g_ProgBestMSE > 0 && g_ProgBestMSE < 1e10) ? g_ProgBestMSE : g_BestMSE;
   MakeLabel(g_InfoPrefix + "MSE", StringFormat("MSE: %.6f (best: %.6f)", dispMSE, dispBest), 15, 89, clrSilver, 9, false);
   MakeLabel(g_InfoPrefix + "Ep", StringFormat("Epochs: %d | VRAM: ~%.0f MB", g_TotalEpochs, g_EstVRAM_MB), 15, 106, clrSilver, 9, false);
   double trainAcc = (g_TrainTotal > 0) ? (double)g_TrainCorrect / g_TrainTotal * 100.0 : 0.0;
   color trainClr = (trainAcc > 55) ? clrLime : (trainAcc > 50) ? clrYellow : clrOrangeRed;
   string trainStr = (g_TrainTotal == 0) ? "Train accuracy: --" : StringFormat("Train accuracy: %.1f%% (%d/%d)", trainAcc, g_TrainCorrect, g_TrainTotal);
   MakeLabel(g_InfoPrefix + "TrainAcc", trainStr, 15, 123, trainClr, 9, false);
   double testAcc = (g_TestTotal > 0) ? (double)g_TestCorrect / g_TestTotal * 100.0 : 0.0;
   color testClr = (testAcc > 55) ? clrLime : (testAcc > 50) ? clrYellow : clrOrangeRed;
   int activeBoundary = GetActiveBoundary();
   int oosBarCount = activeBoundary - (InpPredictAhead + 1);
   string testStr = (g_TestTotal == 0) ? StringFormat("OOS accuracy (%d bars): --", oosBarCount) : StringFormat("OOS accuracy: %.1f%% (%d/%d) [%d OOS bars]", testAcc, g_TestCorrect, g_TestTotal, oosBarCount);
   MakeLabel(g_InfoPrefix + "TestAcc", testStr, 15, 140, testClr, 9, true);
   MakeLabel(g_InfoPrefix + "Coverage", StringFormat("Coverage: %.1f%% (%d/%d)", g_CoveragePct, g_TrainTotal + g_TestTotal, g_AccuracyTotalEligible), 15, 157, clrLightSteelBlue, 8, false);
   string gradStr;
   color gradClr;
   if(g_GraduationCount > 0)
     {
      gradStr = StringFormat("Graduations: %d/%d | Last OOS: %.1f%% | Thr: %.0f%% | CD: %d/%d",
                             g_GraduationCount, InpMaxGraduations, g_LastOOSAccuracy, InpOOSPassThreshold, g_BarsSinceLastGrad, InpGradCooldownBars);
      gradClr = clrMagenta;
     }
   else
     {
      gradStr = StringFormat("Graduations: 0/%d | Threshold: %.0f%% | Cooldown: %d bars", InpMaxGraduations, InpOOSPassThreshold, InpGradCooldownBars);
      gradClr = clrDimGray;
     }
   MakeLabel(g_InfoPrefix + "Grad", gradStr, 15, 174, gradClr, 8, false);
   MakeLabel(g_InfoPrefix + "TFTitle", "── Multi-Timeframe Inputs ──", 15, 191, clrLightSteelBlue, 9, true);
   MakeLabel(g_InfoPrefix + "TFVal", StringFormat("TF1: %s | TF2: %s | TF3: %s", EnumToString(InpTF1), EnumToString(InpTF2), EnumToString(InpTF3)), 15, 208, clrSilver, 8, false);
   int pshift = GetPlotShift();
   string shiftStr = (pshift > 0) ? StringFormat("Plot shift: +%d bars ->", pshift) : "Plot shift: OFF";
   MakeLabel(g_InfoPrefix + "Shift", shiftStr, 15, 225, pshift > 0 ? clrCornflowerBlue : clrDimGray, 8, false);
   string splitStr = StringFormat("Boundary: bar %d | Train: bars [%d+] | OOS: bars [%d..%d]", activeBoundary, activeBoundary, InpPredictAhead + 1, activeBoundary - 1);
   MakeLabel(g_InfoPrefix + "Split", splitStr, 15, 242, clrOrangeRed, 8, false);
   string entStr = StringFormat("Entropy TF1 fast/slow/delta: %.2f / %.2f / %.2f [%s]", g_LastEntropyFastTF1, g_LastEntropySlowTF1, g_LastEntropyDeltaTF1, EntropyModeText(g_LastEntropySlowTF1));
   color entClr = (g_LastEntropySlowTF1 < 0.35) ? clrLime : (g_LastEntropySlowTF1 < 0.65) ? clrYellow : clrOrangeRed;
   MakeLabel(g_InfoPrefix + "Entropy", entStr, 15, 259, entClr, 8, true);
   string entFilt = StringFormat("Entropy filter: %s | chaos>=%.2f | power=%.2f", InpUseEntropyFilter ? "ON" : "OFF", InpEntropyChaosLevel, InpEntropyFilterPower);
   MakeLabel(g_InfoPrefix + "EntropyF", entFilt, 15, 276, clrSilver, 8, false);
   if(g_ModelReady)
     {
      double pbPct = g_ProbPct[1];
      if(pbPct == EMPTY_VALUE)
         pbPct = 50.0;
      string predStr;
      color predClr;
      if(pbPct > 55.0)
        {
         predStr = StringFormat("BULLISH %.0f%%", pbPct);
         predClr = InpBullColor;
        }
      else
         if(pbPct < 45.0)
           {
            predStr = StringFormat("BEARISH %.0f%%", 100.0 - pbPct);
            predClr = InpBearColor;
           }
         else
           {
            predStr = StringFormat("NEUTRAL %.0f%%", pbPct);
            predClr = InpNoiseColor;
           }
      MakeLabel(g_InfoPrefix + "Pred", StringFormat("Next %d bars: %s", InpPredictAhead, predStr), 15, 293, predClr, 10, true);
     }
   else
      MakeLabel(g_InfoPrefix + "Pred", "Čekám na model...", 15, 293, clrGray, 9, false);
   MakeLabel(g_InfoPrefix + "SrcBar", StringFormat("Source: bar[1] (%s)", TimeToString(g_LastClosedBarTime, TIME_DATE | TIME_MINUTES)), 15, 313, clrDarkGray, 8, false);
   string shortName = g_ModelFilePath;
   if(StringLen(shortName) > 50)
      shortName = "..." + StringSubstr(shortName, StringLen(shortName) - 47);
   MakeLabel(g_InfoPrefix + "File", "File: " + shortName, 15, 330, clrDarkGray, 7, false);
   if(g_GradHistoryCount > 0)
     {
      GradHistoryEntry last = g_GradHistory[g_GradHistoryCount - 1];
      string histStr = StringFormat("Last: #%d b%d->%d acc%.0f%% (%d smp) @%s", last.cycle, last.oldBoundary, last.newBoundary, last.oosAccuracy, last.oosSamples, TimeToString(last.timestamp, TIME_MINUTES));
      MakeLabel(g_InfoPrefix + "GradHist", histStr, 15, 347, clrDarkMagenta, 7, false);
     }
   else
      MakeLabel(g_InfoPrefix + "GradHist", "", 15, 347, clrDarkGray, 7, false);
   if(g_GraduationCount > 0)
     {
      int remainingOOS = activeBoundary - (InpPredictAhead + 1);
      if(remainingOOS < InpMinOOSSamples)
         MakeLabel(g_InfoPrefix + "GradWarn", StringFormat("OOS exhausted (%d bars < %d min)", remainingOOS, InpMinOOSSamples), 15, 364, clrOrangeRed, 8, true);
      else
         MakeLabel(g_InfoPrefix + "GradWarn", StringFormat("Next OOS region: %d bars available", remainingOOS), 15, 364, clrDimGray, 7, false);
     }
   else
      MakeLabel(g_InfoPrefix + "GradWarn", "", 15, 364, clrDarkGray, 7, false);
   ChartRedraw();
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void MakeLabel(string name, string text, int x, int y, color clr, int sz, bool bold)
{
   // Vždy kreslíme do hlavního okna (0), ne do subokna indikátoru
   long chart_id = ChartID();

   if(ObjectFind(chart_id, name) < 0)
   {
      ObjectCreate(chart_id, name, OBJ_LABEL, 0, 0, 0); // window=0 = hlavní okno
      ObjectSetInteger(chart_id, name, OBJPROP_CORNER,     InpInfoCorner);
      ObjectSetInteger(chart_id, name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(chart_id, name, OBJPROP_HIDDEN,     true);
      // Důležité: zajistíme viditelnost ve všech periodách
      ObjectSetInteger(chart_id, name, OBJPROP_TIMEFRAMES, OBJ_ALL_PERIODS);
   }

   ObjectSetInteger(chart_id, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(chart_id, name, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(chart_id, name, OBJPROP_COLOR,     clr);
   ObjectSetInteger(chart_id, name, OBJPROP_FONTSIZE,  sz);
   ObjectSetString( chart_id, name, OBJPROP_FONT,      bold ? "Arial Bold" : "Arial");
   ObjectSetString( chart_id, name, OBJPROP_TEXT,      text);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CreateProgressBar()
{
   long chart_id = ChartID();
   int yOff = 435;

   // --- Background ---
   string bg = g_ProgPrefix + "BG";
   if(ObjectFind(chart_id, bg) < 0)
      ObjectCreate(chart_id, bg, OBJ_RECTANGLE_LABEL, 0, 0, 0); // window=0
   ObjectSetInteger(chart_id, bg, OBJPROP_CORNER,       InpInfoCorner);
   ObjectSetInteger(chart_id, bg, OBJPROP_XDISTANCE,    15);
   ObjectSetInteger(chart_id, bg, OBJPROP_YDISTANCE,    yOff);
   ObjectSetInteger(chart_id, bg, OBJPROP_XSIZE,        InpProgressWidth);
   ObjectSetInteger(chart_id, bg, OBJPROP_YSIZE,        InpProgressHeight);
   ObjectSetInteger(chart_id, bg, OBJPROP_BGCOLOR,      C'40,40,50');
   ObjectSetInteger(chart_id, bg, OBJPROP_BORDER_COLOR, clrDimGray);
   ObjectSetInteger(chart_id, bg, OBJPROP_BORDER_TYPE,  BORDER_FLAT);
   ObjectSetInteger(chart_id, bg, OBJPROP_SELECTABLE,   false);
   ObjectSetInteger(chart_id, bg, OBJPROP_HIDDEN,       true);
   ObjectSetInteger(chart_id, bg, OBJPROP_TIMEFRAMES,   OBJ_ALL_PERIODS);

   // --- Fill bar ---
   string fi = g_ProgPrefix + "Fill";
   if(ObjectFind(chart_id, fi) < 0)
      ObjectCreate(chart_id, fi, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(chart_id, fi, OBJPROP_CORNER,     InpInfoCorner);
   ObjectSetInteger(chart_id, fi, OBJPROP_XDISTANCE,  15);
   ObjectSetInteger(chart_id, fi, OBJPROP_YDISTANCE,  yOff);
   ObjectSetInteger(chart_id, fi, OBJPROP_XSIZE,      0);
   ObjectSetInteger(chart_id, fi, OBJPROP_YSIZE,      InpProgressHeight);
   ObjectSetInteger(chart_id, fi, OBJPROP_BGCOLOR,    clrDodgerBlue);
   ObjectSetInteger(chart_id, fi, OBJPROP_BORDER_TYPE,BORDER_FLAT);
   ObjectSetInteger(chart_id, fi, OBJPROP_WIDTH,      0);
   ObjectSetInteger(chart_id, fi, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(chart_id, fi, OBJPROP_HIDDEN,     true);
   ObjectSetInteger(chart_id, fi, OBJPROP_TIMEFRAMES, OBJ_ALL_PERIODS);

   // --- Procenta text ---
   string tx = g_ProgPrefix + "Txt";
   if(ObjectFind(chart_id, tx) < 0)
      ObjectCreate(chart_id, tx, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(chart_id, tx, OBJPROP_CORNER,     InpInfoCorner);
   ObjectSetInteger(chart_id, tx, OBJPROP_XDISTANCE,  15 + InpProgressWidth / 2);
   ObjectSetInteger(chart_id, tx, OBJPROP_YDISTANCE,  yOff + 2);
   ObjectSetInteger(chart_id, tx, OBJPROP_FONTSIZE,   9);
   ObjectSetString( chart_id, tx, OBJPROP_FONT,       "Arial Bold");
   ObjectSetInteger(chart_id, tx, OBJPROP_COLOR,      clrWhite);
   ObjectSetInteger(chart_id, tx, OBJPROP_ANCHOR,     ANCHOR_CENTER);
   ObjectSetInteger(chart_id, tx, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(chart_id, tx, OBJPROP_HIDDEN,     true);
   ObjectSetInteger(chart_id, tx, OBJPROP_TIMEFRAMES, OBJ_ALL_PERIODS);

   // --- Detail text (Ep/MB/ETA) ---
   string dt = g_ProgPrefix + "Det";
   if(ObjectFind(chart_id, dt) < 0)
      ObjectCreate(chart_id, dt, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(chart_id, dt, OBJPROP_CORNER,     InpInfoCorner);
   ObjectSetInteger(chart_id, dt, OBJPROP_XDISTANCE,  15);
   ObjectSetInteger(chart_id, dt, OBJPROP_YDISTANCE,  yOff + InpProgressHeight + 4);
   ObjectSetInteger(chart_id, dt, OBJPROP_FONTSIZE,   8);
   ObjectSetString( chart_id, dt, OBJPROP_FONT,       "Arial");
   ObjectSetInteger(chart_id, dt, OBJPROP_COLOR,      clrSilver);
   ObjectSetInteger(chart_id, dt, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(chart_id, dt, OBJPROP_HIDDEN,     true);
   ObjectSetInteger(chart_id, dt, OBJPROP_TIMEFRAMES, OBJ_ALL_PERIODS);

   // --- MSE text ---
   string ms = g_ProgPrefix + "MSE";
   if(ObjectFind(chart_id, ms) < 0)
      ObjectCreate(chart_id, ms, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(chart_id, ms, OBJPROP_CORNER,     InpInfoCorner);
   ObjectSetInteger(chart_id, ms, OBJPROP_XDISTANCE,  15);
   ObjectSetInteger(chart_id, ms, OBJPROP_YDISTANCE,  yOff + InpProgressHeight + 19);
   ObjectSetInteger(chart_id, ms, OBJPROP_FONTSIZE,   8);
   ObjectSetString( chart_id, ms, OBJPROP_FONT,       "Arial");
   ObjectSetInteger(chart_id, ms, OBJPROP_COLOR,      clrGold);
   ObjectSetInteger(chart_id, ms, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(chart_id, ms, OBJPROP_HIDDEN,     true);
   ObjectSetInteger(chart_id, ms, OBJPROP_TIMEFRAMES, OBJ_ALL_PERIODS);

   // --- LR / GradNorm text ---
   string lr = g_ProgPrefix + "LR";
   if(ObjectFind(chart_id, lr) < 0)
      ObjectCreate(chart_id, lr, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(chart_id, lr, OBJPROP_CORNER,     InpInfoCorner);
   ObjectSetInteger(chart_id, lr, OBJPROP_XDISTANCE,  15);
   ObjectSetInteger(chart_id, lr, OBJPROP_YDISTANCE,  yOff + InpProgressHeight + 34);
   ObjectSetInteger(chart_id, lr, OBJPROP_FONTSIZE,   8);
   ObjectSetString( chart_id, lr, OBJPROP_FONT,       "Arial");
   ObjectSetInteger(chart_id, lr, OBJPROP_COLOR,      clrCornflowerBlue);
   ObjectSetInteger(chart_id, lr, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(chart_id, lr, OBJPROP_HIDDEN,     true);
   ObjectSetInteger(chart_id, lr, OBJPROP_TIMEFRAMES, OBJ_ALL_PERIODS);

   // --- Title ---
   string tt = g_ProgPrefix + "Title";
   if(ObjectFind(chart_id, tt) < 0)
      ObjectCreate(chart_id, tt, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(chart_id, tt, OBJPROP_CORNER,     InpInfoCorner);
   ObjectSetInteger(chart_id, tt, OBJPROP_XDISTANCE,  15);
   ObjectSetInteger(chart_id, tt, OBJPROP_YDISTANCE,  yOff - 16);
   ObjectSetInteger(chart_id, tt, OBJPROP_FONTSIZE,   9);
   ObjectSetString( chart_id, tt, OBJPROP_FONT,       "Arial Bold");
   ObjectSetInteger(chart_id, tt, OBJPROP_COLOR,      clrDodgerBlue);
   ObjectSetInteger(chart_id, tt, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(chart_id, tt, OBJPROP_HIDDEN,     true);
   ObjectSetInteger(chart_id, tt, OBJPROP_TIMEFRAMES, OBJ_ALL_PERIODS);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void UpdateProgressBar()
{
   long chart_id = ChartID();
   bool show = g_IsTraining ||
               (g_TrainStartTime > 0 && TimeCurrent() - g_TrainStartTime < 5);

   string names[] = {"BG","Fill","Txt","Det","MSE","LR","Title"};
   for(int i = 0; i < ArraySize(names); i++)
      ObjectSetInteger(chart_id, g_ProgPrefix + names[i],
                       OBJPROP_TIMEFRAMES,
                       show ? OBJ_ALL_PERIODS : OBJ_NO_PERIODS);

   if(!show)
   {
      ChartRedraw(chart_id);
      return;
   }

   double prog = g_ProgPercent / 100.0;
   prog = MathMax(0.0, MathMin(1.0, prog));
   if(prog < 0.001 && g_TargetEpochs > 0 && g_CurrentEpochs > 0)
      prog = MathMin(1.0, (double)g_CurrentEpochs / g_TargetEpochs);

   color fc;
   if(g_TrainPhase == PHASE_OOS_RETRAIN)
      fc = clrMagenta;
   else if(prog < 0.33)
      fc = clrOrangeRed;
   else if(prog < 0.66)
      fc = clrGold;
   else if(prog < 1.0)
      fc = clrDodgerBlue;
   else
      fc = clrLime;

   ObjectSetInteger(chart_id, g_ProgPrefix + "Fill",
                    OBJPROP_XSIZE,
                    MathMax(0, (int)(InpProgressWidth * prog)));
   ObjectSetInteger(chart_id, g_ProgPrefix + "Fill",
                    OBJPROP_BGCOLOR, fc);

   ObjectSetString(chart_id, g_ProgPrefix + "Txt",
                   OBJPROP_TEXT,
                   StringFormat("%.1f%%", prog * 100));

   // Detail: Ep / MB / ETA / Elapsed
   string det = "";
   if(g_ProgTotalEpochs > 0)
   {
      det = StringFormat("Ep %d/%d", g_ProgEpoch, g_ProgTotalEpochs);
      if(g_ProgTotalMB > 0)
         det += StringFormat(" | MB %d/%d", g_ProgMB, g_ProgTotalMB);
      if(g_ProgETASec > 0)
         det += " | ETA " + FormatDuration(g_ProgETASec);
      if(g_ProgElapsedSec > 0)
         det += " | " + FormatDuration(g_ProgElapsedSec);
   }
   else
      det = StringFormat("Epoch %d / %d", g_CurrentEpochs, g_TargetEpochs);
   ObjectSetString(chart_id, g_ProgPrefix + "Det", OBJPROP_TEXT, det);

   // MSE
   string mseT = "";
   if(g_ProgMSE > 0 || g_LastMSE > 0)
   {
      double dm = (g_ProgMSE > 0) ? g_ProgMSE : g_LastMSE;
      mseT = StringFormat("MSE: %.6f -> %.4f", dm, InpTargetMSE);
      double db = (g_ProgBestMSE > 0 && g_ProgBestMSE < 1e9)
                  ? g_ProgBestMSE : g_BestMSE;
      if(db < 1e9)
         mseT += StringFormat(" | Best: %.6f", db);
   }
   ObjectSetString(chart_id, g_ProgPrefix + "MSE", OBJPROP_TEXT, mseT);

   // LR + GradNorm
   string lrT = "";
   if(g_ProgLR > 0)
      lrT = StringFormat("LR: %.6f", g_ProgLR);
   if(g_ProgGradNorm > 0)
      lrT += (StringLen(lrT) > 0 ? " | " : "")
             + StringFormat("GradNorm: %.4f", g_ProgGradNorm);
   ObjectSetString(chart_id, g_ProgPrefix + "LR", OBJPROP_TEXT, lrT);

   // Title
   string ttl;
   if(g_TrainPhase == PHASE_OOS_RETRAIN)
      ttl = StringFormat("OOS Retrain -> Grad #%d (acc %.0f%%)",
                         g_GraduationCount + 1, g_LastOOSAccuracy);
   else
      ttl = StringFormat("GPU Training + Shannon (Train/Test %.0f%%/%.0f%%)",
                         100.0 - InpTestPct, InpTestPct);

   if(g_IsTraining)
      ttl += StringSubstr("....", 0, ((int)(GetTickCount() / 400)) % 4);
   else if(prog >= 1.0)
      ttl = "Training Complete!";

   ObjectSetString(chart_id, g_ProgPrefix + "Title", OBJPROP_TEXT, ttl);

   color titleClr;
   if(g_TrainPhase == PHASE_OOS_RETRAIN)
      titleClr = clrMagenta;
   else if(g_IsTraining)
      titleClr = clrYellow;
   else
      titleClr = clrLime;

   ObjectSetInteger(chart_id, g_ProgPrefix + "Title", OBJPROP_COLOR, titleClr);

   ChartRedraw(chart_id);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string FormatDuration(double seconds)
  {
   int s = (int)MathRound(seconds);
   if(s < 0)
      return "--:--";
   if(s < 60)
      return StringFormat("%ds", s);
   if(s < 3600)
      return StringFormat("%dm%02ds", s / 60, s % 60);
   return StringFormat("%dh%02dm", s / 3600, (s % 3600) / 60);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CleanupObjects()
{
   long chart_id = ChartID();
   // Mažeme vždy z chart_id, ne z anonymního 0
   ObjectsDeleteAll(chart_id, g_InfoPrefix);
   ObjectsDeleteAll(chart_id, g_ProgPrefix);
   ObjectsDeleteAll(chart_id, g_SplitPrefix);
   ObjectsDeleteAll(chart_id, g_CrossObjPrefix);
}

//+------------------------------------------------------------------+
//| Zjištění chyb z DLL - když nám to spadne na hubu |
//+------------------------------------------------------------------+
string GetDLLError()
  {
   short buf[];
   ArrayResize(buf, 512);
   ArrayInitialize(buf, 0);
   DN_GetError(buf, 512);
   string s = "";
   for(int i = 0; i < 512 && buf[i] != 0; i++)
      s += ShortToString(buf[i]);
   return (StringLen(s) == 0) ? "unknown DLL error (prostě se to seklo)" : s;
  }

//+------------------------------------------------------------------+
//| Ukládání a načítání modelu do/ze souboru |
//+------------------------------------------------------------------+
bool SaveModel()
  {
   if(g_NetHandle == 0 || StringLen(g_ModelFilePath) == 0)
      return false;
   int stateSize = DN_SaveState(g_NetHandle);
   if(stateSize <= 0)
      return false;
   char stateBuf[];
   ArrayResize(stateBuf, stateSize);
   if(!DN_GetState(g_NetHandle, stateBuf, stateSize))
      return false;
   int fh = FileOpen(g_ModelFilePath, FILE_WRITE | FILE_BIN | FILE_COMMON);
   if(fh == INVALID_HANDLE)
      return false;
   FileWriteInteger(fh, MODEL_MAGIC, INT_VALUE);
   FileWriteInteger(fh, MODEL_META_VER, INT_VALUE);
   FileWriteInteger(fh, g_FrozenTestBoundary, INT_VALUE);
   FileWriteInteger(fh, g_GraduationCount, INT_VALUE);
   FileWriteInteger(fh, stateSize, INT_VALUE);
   FileWriteArray(fh, stateBuf, 0, stateSize);
   FileClose(fh);
   Print(StringFormat("Model saved (%d bytes) -> %s | boundary=%d | grads=%d", stateSize, g_ModelFilePath, g_FrozenTestBoundary, g_GraduationCount));
   return true;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool LoadModel()
{
   if(g_NetHandle == 0 || StringLen(g_ModelFilePath) == 0)
      return false;

   int fh = FileOpen(g_ModelFilePath, FILE_READ | FILE_BIN | FILE_COMMON);
   if(fh == INVALID_HANDLE)
      return false;

   ulong fs = FileSize(fh);

   // Minimální velikost: 5x INT_VALUE (4 byty) = 20 bytů pro hlavičku
   if(fs < 20 || fs > 100000000)
   {
      FileClose(fh);
      Print("LoadModel: invalid file size: ", fs);
      return false;
   }

   int magic = FileReadInteger(fh, INT_VALUE);
   if(magic != MODEL_MAGIC)
   {
      FileClose(fh);
      Print("LoadModel: invalid magic number — tenhle soubor s námi nekamarádí.");
      return false;
   }

   int metaVer = FileReadInteger(fh, INT_VALUE);
   if(metaVer < 3)
   {
      FileClose(fh);
      Print("LoadModel: unsupported meta version: ", metaVer);
      return false;
   }

   int boundary  = FileReadInteger(fh, INT_VALUE);
   int gradCount = FileReadInteger(fh, INT_VALUE);
   int stateSize = FileReadInteger(fh, INT_VALUE);

   // Ověření: stateSize musí sedět s tím co zbývá v souboru
   ulong expectedRemaining = fs - 20; // 5 * 4 bytů hlavičky
   if(stateSize <= 0 || (ulong)stateSize > expectedRemaining)
   {
      FileClose(fh);
      Print(StringFormat("LoadModel: invalid state size: %d (remaining: %I64u)",
                         stateSize, expectedRemaining));
      return false;
   }

   char buf[];
   ArrayResize(buf, stateSize + 1);
   int bytesRead = (int)FileReadArray(fh, buf, 0, stateSize);
   FileClose(fh);

   if(bytesRead != stateSize)
   {
      Print(StringFormat("LoadModel: read %d bytes, expected %d", bytesRead, stateSize));
      return false;
   }

   buf[stateSize] = 0;

   if(!DN_LoadState(g_NetHandle, buf))
   {
      Print("LoadState fail: ", GetDLLError());
      return false;
   }

   g_FrozenTestBoundary = boundary;
   g_GraduationCount    = gradCount;
   g_BarsSinceLastGrad  = InpGradCooldownBars;

   Print(StringFormat("Model loaded (%d bytes) <- %s | boundary=%d | grads=%d",
                      stateSize, g_ModelFilePath,
                      g_FrozenTestBoundary, g_GraduationCount));
   return true;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void DeleteCrossObjects()
{
   ObjectsDeleteAll(ChartID(), g_CrossObjPrefix);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
datetime InterpolateTime(datetime t_old, datetime t_new, double k)
  {
   if(k <= 0.0)
      return t_old;
   if(k >= 1.0)
      return t_new;
   long dt = (long)t_new - (long)t_old;
   return (datetime)((long)t_old + (long)MathRound((double)dt * k));
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
datetime ShiftTimeByBars(datetime t, int barsForward)
  {
   if(barsForward <= 0)
      return t;
   int currentBarIndex = iBarShift(_Symbol, _Period, t, false);
   if(currentBarIndex >= 0)
     {
      int targetBarIndex = currentBarIndex - barsForward;
      if(targetBarIndex >= 0)
        {
         datetime targetTime = iTime(_Symbol, _Period, targetBarIndex);
         if(targetTime > 0)
            return targetTime;
        }
     }
// Fallback (o víkendech to prostě trochu kecá, no)
   int sec = PeriodSeconds(_Period);
   if(sec <= 0)
      return t;
   return (datetime)((long)t + (long)(barsForward * sec));
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CreateCrossArrowObject(const string name, const int subWindow,
                            const datetime t, const double priceLevel,
                            const color clr, const int arrowCode,
                            const string tooltip)
{
   long chart_id = ChartID();

   if(ObjectFind(chart_id, name) >= 0)
      ObjectDelete(chart_id, name);

   if(!ObjectCreate(chart_id, name, OBJ_ARROW, subWindow, t, priceLevel))
      return;

   ObjectSetInteger(chart_id, name, OBJPROP_ARROWCODE,  arrowCode);
   ObjectSetInteger(chart_id, name, OBJPROP_COLOR,      clr);
   ObjectSetInteger(chart_id, name, OBJPROP_WIDTH,      1);
   ObjectSetInteger(chart_id, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(chart_id, name, OBJPROP_HIDDEN,     true);
   ObjectSetInteger(chart_id, name, OBJPROP_BACK,       false);
   ObjectSetInteger(chart_id, name, OBJPROP_TIMEFRAMES, OBJ_ALL_PERIODS);
   ObjectSetString( chart_id, name, OBJPROP_TOOLTIP,    tooltip);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int GetMySubWindow()
  {
   long chart_id = ChartID();
   int windows_total = (int)ChartGetInteger(chart_id, CHART_WINDOWS_TOTAL);
   string myName = StringFormat("MTF-LSTM-ENT(%s,%s,%s) PA%d T/T%.0f%% G%.0f%%",
                                EnumToString(InpTF1), EnumToString(InpTF2), EnumToString(InpTF3),
                                InpPredictAhead, InpTestPct, InpOOSPassThreshold);
   for(int w = 1; w < windows_total; w++)
     {
      int ind_total = ChartIndicatorsTotal(chart_id, w);
      for(int i = 0; i < ind_total; i++)
        {
         string ind_name = ChartIndicatorName(chart_id, w, i);
         if(ind_name == myName)
            return w;
         if(StringFind(ind_name, "MTF-LSTM-ENT(") == 0)
            return w;
        }
     }
   return -1;
  }
//+------------------------------------------------------------------+



//+------------------------------------------------------------------+
// In world of uncertainty, probabillity is the only honest language.
// Zlaté pravidlo programování - uklízej po sobě. Jinak máš v kódu brzo
// víc písku než u mě na polici.
// Užívejte si, je později než čekáte!
// Konec kódu. Začátek výzkumu REMIND for you.
//+------------------------------------------------------------------+
