//+------------------------------------------------------------------+
//| LSTM_EntropyGated_Predictor.mq5                      v4.30       |
//| Entropy-first, LSTM-second architecture                          |
//| GPU-optimized via MQL5GPULibrary_LSTM.dll                        |
//|                                                                  |
//| KONVENCE (platí všude v kódu):                                   |
//|   index 0 = nejnovější bar (formující)                           |
//|   index 1 = poslední uzavřený bar                                |
//|   větší index = starší bar                                       |
//|   "budoucnost" od baru X = menší index (X - N)                  |
//|   "minulost"  od baru X = větší index (X + N)                   |
//|                                                                  |
//| FIX v4.30 (hlavní bug z v4.2):                                  |
//|  Symptom: H=1.000, NOISE pořád, žádné predikce.                 |
//|  Příčina: firstValidEntropyBar = InpEntropyWindow = 20           |
//|    → smyčka entropy začínala od indexu 20                        |
//|    → bar[1] zůstával na init hodnotě 1.0                         |
//|    → podmínka (1 < g_CalcStart=20) v UpdateInfoPanel = TRUE vždy|
//|    → zobrazovalo NOISE i když H < threshold                      |
//|  Oprava:                                                         |
//|    firstValidEntropyBar = 1 (okno jde do minulosti = větší idx) |
//|    Guard v entropy smyčce: i + window - 1 < rates_total          |
//|    UpdateInfoPanel: odstraněna falešná podmínka (1 < g_CalcStart)|
//|    Smooth state při inkrement. přepočtu: g_Entropy[calcStart+1] |
//|    (starší bar = větší index, ne calcStart-1)                    |
//+------------------------------------------------------------------+
#property copyright "Tomáš Bělák"
#property link      "https://remind.cz/"
#property description "Entropy-Gated LSTM v4.30: as-series indexing + robust entropy gating/prediction alignment"
#property version   "4.30"
#property strict

#property indicator_separate_window
#property indicator_minimum 0
#property indicator_maximum 100
#property indicator_level1  50
#property indicator_level2  30
#property indicator_level3  70
#property indicator_levelcolor clrDimGray
#property indicator_levelstyle STYLE_DOT

#property indicator_buffers 4
#property indicator_plots   1

#property indicator_label1  "BUY Prob (%)"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrLime
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2

//+------------------------------------------------------------------+
//| DLL Import                                                        |
//+------------------------------------------------------------------+
#import "MQL5GPULibrary_LSTM.dll"
int    DN_Create();
void   DN_Free(int h);
int    DN_SetSequenceLength(int h, int seq_len);
int    DN_SetMiniBatchSize(int h, int mbs);
int    DN_AddLayerEx(int h, int in_sz, int out_sz, int act, int ln, double drop);
int    DN_SetOutputDim(int h, int out_dim);
int    DN_SetGradClip(int h, double clip);
int    DN_LoadBatch(int h, const double &X[], const double &T[],
                    int batch, int in_dim, int out_dim, int layout);
int    DN_TrainAsync(int h, int epochs, double target_mse, double lr, double wd);
int    DN_GetTrainingStatus(int h);
void   DN_GetTrainingResult(int h, double &out_mse, int &out_epochs);
void   DN_StopTraining(int h);
int    DN_PredictBatch(int h, const double &X[], int batch,
                       int in_dim, int layout, double &Y[]);
int    DN_SnapshotWeights(int h);
int    DN_RestoreWeights(int h);
int    DN_GetLayerCount(int h);
double DN_GetLayerWeightNorm(int h, int layer);
double DN_GetGradNorm(int h);
int    DN_SaveState(int h);
int    DN_GetState(int h, char &buf[], int max_len);
int    DN_LoadState(int h, const char &buf[]);
void   DN_GetError(short &buf[], int len);

int    DN_GetProgressEpoch(int h);
int    DN_GetProgressTotalEpochs(int h);
int    DN_GetProgressMiniBatch(int h);
int    DN_GetProgressTotalMiniBatches(int h);
double DN_GetProgressLR(int h);
double DN_GetProgressMSE(int h);
double DN_GetProgressBestMSE(int h);
double DN_GetProgressGradNorm(int h);
int    DN_GetProgressTotalSteps(int h);
double DN_GetProgressPercent(int h);
double DN_GetProgressElapsedSec(int h);
double DN_GetProgressETASec(int h);
int    DN_GetProgressAll(int h,
                         int &epoch, int &total_epochs,
                         int &mb, int &total_mb,
                         double &lr, double &mse, double &best_mse,
                         double &grad_norm, double &pct,
                         double &elapsed_sec, double &eta_sec);
#import

//+------------------------------------------------------------------+
//| Inputs                                                            |
//+------------------------------------------------------------------+
input group "=== Entropy Gate ==="
input int      InpEntropyWindow    = 20;
input int      InpSymbolAlphabet   = 9;
input double   InpEntropyThreshold = 0.75;
input double   InpEntropyMargin    = 0.15;
input int      InpEntropySmooth    = 3;
input bool     InpGrayOnNoise      = true;

input group "=== LSTM Architecture ==="
input int      InpLookback         = 30;
input int      InpHiddenSize1      = 64;
input int      InpHiddenSize2      = 32;
input int      InpHiddenSize3      = 0;
input double   InpDropout          = 0.10;

input group "=== Training (GPU) ==="
input int      InpTrainBars        = 3000;
input int      InpInitialEpochs    = 300;
input int      InpRetrainEpochs    = 80;
input int      InpRetrainInterval  = 200;
input double   InpLearningRate     = 0.0008;
input double   InpWeightDecay      = 0.0001;
input double   InpTargetMSE        = 0.005;
input int      InpMiniBatch        = 64;

input group "=== Prediction ==="
input int      InpPredictBatch     = 512;
input int      InpMaxPredictBars   = 2000;
input int      InpPredictAhead     = 5;

input group "=== Display ==="
input color    InpBullColor        = clrLime;
input color    InpBearColor        = clrRed;
input color    InpNoiseColor       = clrDimGray;
input color    InpEntropyColor     = clrGold;
input int      InpInfoCorner       = 0;
input int      InpProgressWidth    = 280;
input int      InpProgressHeight   = 18;

input group "=== Advanced ==="
input double   InpGradClip         = 5.0;
input bool     InpAutoRetrain      = true;
input bool     InpSaveModel        = false;
input string   InpModelFile        = "lstm_entropy_v4.bin";
input bool     InpVerboseLog       = false;

//+------------------------------------------------------------------+
#define FEAT_PER_BAR   12
#define OUTPUT_DIM      2

//+------------------------------------------------------------------+
//| Globals                                                           |
//+------------------------------------------------------------------+
double g_ProbPct[];
double g_Entropy[];        // smoothed Shannon entropy, init=1.0 (max noise)
double g_RawEntropy[];
double g_SymbolSeq[];

int    g_NetHandle        = 0;
bool   g_ModelReady       = false;
bool   g_IsTraining       = false;
int    g_LastTrainBar     = 0;
int    g_TotalBars        = 0;
double g_LastMSE          = 0.0;
int    g_TotalEpochs      = 0;
double g_BestMSE          = 1e10;
int    g_TargetEpochs     = 0;
int    g_CurrentEpochs    = 0;
datetime g_TrainStartTime = 0;

int    g_CorrectPred      = 0;
int    g_TotalPred        = 0;

double g_ATRMean          = 0.0;
datetime g_LastClosedBarTime = 0;

int    g_ProgEpoch        = 0;
int    g_ProgTotalEpochs  = 0;
int    g_ProgMB           = 0;
int    g_ProgTotalMB      = 0;
int    g_ProgTotalSteps   = 0;
double g_ProgLR           = 0.0;
double g_ProgMSE          = 0.0;
double g_ProgBestMSE      = 0.0;
double g_ProgGradNorm     = 0.0;
double g_ProgPercent      = 0.0;
double g_ProgElapsedSec   = 0.0;
double g_ProgETASec       = 0.0;

// ATR cache — Wilderova rekurence
double g_ATRCache[];       // g_ATRCache[i] = ATR pro bar (g_ATRCacheStart + i)
int    g_ATRCacheStart    = -1;
int    g_ATRCachedBars    = 0;

int    g_LastPredictedTo  = -1;
int    g_CalcStart        = 0;   // nejnovější bar kde g_Entropy je skutečně spočítaná
int    g_CalcEnd          = 0;   // nejstarší bar kde g_Entropy je skutečně spočítaná

// Entropy smooth state
double g_EntropySmoothState = -1.0;

double g_EstVRAM_MB       = 0;
int    g_LowEntropyBars   = 0;
int    g_HighEntropyBars  = 0;
double g_AvgEntropyRecent = 0.0;

string g_InfoPrefix = "EL_I_";
string g_ProgPrefix = "EL_P_";

//+------------------------------------------------------------------+
//| Init                                                              |
//+------------------------------------------------------------------+
int OnInit()
  {
   Print("=== Entropy-Gated LSTM v4.30 (GPU) ===");
   Print("Konvence: index 0=nejnovejsi, vetsi index=starsi bar");

   SetIndexBuffer(0, g_ProbPct,       INDICATOR_DATA);
   SetIndexBuffer(1, g_Entropy,       INDICATOR_CALCULATIONS);
   SetIndexBuffer(2, g_RawEntropy,    INDICATOR_CALCULATIONS);
   SetIndexBuffer(3, g_SymbolSeq,     INDICATOR_CALCULATIONS);

   ArraySetAsSeries(g_ProbPct, true);
   ArraySetAsSeries(g_Entropy, true);
   ArraySetAsSeries(g_RawEntropy, true);
   ArraySetAsSeries(g_SymbolSeq, true);

   // FIX 2: g_Entropy inicializujeme na 1.0 (maximum noise).
   // Bary, pro které se entropie ještě nepočítala, tak nikdy
   // nevypadají jako "structured" (H=0 by bylo chybné).
   // PlotIndexSetDouble nastaví "prázdnou" hodnotu pro vykreslování.
   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);

   IndicatorSetString(INDICATOR_SHORTNAME,
                      StringFormat("Entropy-LSTM(%d|%.2f|%d->%d)",
                                   InpEntropyWindow, InpEntropyThreshold,
                                   InpHiddenSize1, InpHiddenSize2));
   IndicatorSetString(INDICATOR_LEVELTEXT, 0, "50% — neutral");
   IndicatorSetString(INDICATOR_LEVELTEXT, 1, "30% — bearish zone");
   IndicatorSetString(INDICATOR_LEVELTEXT, 2, "70% — bullish zone");

   if(!InitNetwork())
     {
      Print("FATAL: Network init failed");
      return INIT_FAILED;
     }

   if(InpSaveModel && FileIsExist(InpModelFile, FILE_COMMON))
     {
      if(LoadModel())
        {
         Print("Model loaded OK");
         g_ModelReady = true;
        }
     }

   g_LastClosedBarTime   = 0;
   g_EntropySmoothState  = -1.0;
   g_LastPredictedTo     = -1;
   g_CalcStart           = 0;
   g_CalcEnd             = 0;

   CreateInfoPanel();
   CreateProgressBar();
   EventSetMillisecondTimer(100);
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
//| Deinit                                                            |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   if(g_NetHandle > 0 && g_IsTraining)
     {
      DN_StopTraining(g_NetHandle);
      Sleep(500);
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
//| Symbolic candle encoding (0..8)                                   |
//|                                                                   |
//| Funguje na libovolném baru — nezávisí na směru indexů.           |
//+------------------------------------------------------------------+
int EncodeCandle(double open, double high, double low, double close)
  {
   double range = high - low;
   if(range < _Point) return 4; // doji pro nulový range

   double bodyAbs   = MathAbs(close - open);
   double bodyRatio = bodyAbs / range;
   double upperShadow = high - MathMax(open, close);
   double lowerShadow = MathMin(open, close) - low;
   bool   bullish = (close >= open);

   if(bodyRatio < 0.10) return 4; // doji

   if(bullish)
     {
      if(bodyRatio > 0.70)                return 8; // strong bull
      else if(bodyRatio > 0.30)           return 7; // medium bull
      else if(lowerShadow > upperShadow)  return 5; // weak bull, lower shadow
      else                                return 6; // weak bull, upper shadow
     }
   else
     {
      if(bodyRatio > 0.70)                return 0; // strong bear
      else if(bodyRatio > 0.30)           return 1; // medium bear
      else if(upperShadow > lowerShadow)  return 2; // weak bear, upper shadow
      else                                return 3; // weak bear, lower shadow
     }

   // Fallback (ochrana proti krajním FP anomáliím)
   return 4;
  }

//+------------------------------------------------------------------+
//| Shannon entropy (normalizovaná 0..1)                              |
//+------------------------------------------------------------------+
double ComputeShannonEntropy(const double &symbols[],
                             int startIdx,
                             int window,
                             int alphabetSize)
  {
   if(window < 2 || alphabetSize < 2) return 1.0;

   int counts[];
   ArrayResize(counts, alphabetSize);
   ArrayInitialize(counts, 0);
   int total = 0;

   for(int i = 0; i < window; i++)
     {
      int idx = startIdx + i;
      if(idx < 0 || idx >= ArraySize(symbols)) continue;
      int sym = (int)MathRound(symbols[idx]);
      sym = MathMax(0, MathMin(alphabetSize - 1, sym));
      counts[sym]++;
      total++;
     }
   if(total < 2) return 1.0;

   double H = 0.0;
   for(int i = 0; i < alphabetSize; i++)
     {
      if(counts[i] == 0) continue;
      double p = (double)counts[i] / total;
      H -= p * MathLog(p) / MathLog(2.0);
     }
   double maxH = MathLog((double)alphabetSize) / MathLog(2.0);
   if(maxH < 0.001) return 1.0;
   return MathMin(1.0, H / maxH);
  }

//+------------------------------------------------------------------+
//| EMA smoothing                                                     |
//+------------------------------------------------------------------+
double SmoothEntropy(double rawH, double prev, int period)
  {
   if(period < 2 || prev < 0.0) return rawH;
   double alpha = 2.0 / (period + 1.0);
   return alpha * rawH + (1.0 - alpha) * prev;
  }

//+------------------------------------------------------------------+
//| Entropy confidence weight (soft gate)                            |
//+------------------------------------------------------------------+
double EntropyWeight(double H, double thr, double margin)
  {
   if(margin <= 0.0) return (H < thr) ? 1.0 : 0.0;
   double w = (thr - H) / margin;
   if(w < 0.0) w = 0.0;
   if(w > 1.0) w = 1.0;
   return w;
  }

//+------------------------------------------------------------------+
//| Feature extraction — 12 features per bar                         |
//|                                                                   |
//| startBar = index nejnovějšího baru v bloku (menší = novější).    |
//| count    = počet barů směrem do minulosti (rostoucí index).       |
//| Takže blok pokrývá bary [startBar .. startBar+count-1].          |
//+------------------------------------------------------------------+
void ExtractSymbolicFeatures(int startBar, int count,
                             const double &open[],
                             const double &high[],
                             const double &low[],
                             const double &close[],
                             const long   &tick_volume[],
                             int rates_total,
                             double &feats[])
  {
   ArrayResize(feats, count * FEAT_PER_BAR);
   ArrayInitialize(feats, 0.0);

   for(int i = 0; i < count; i++)
     {
      int bar = startBar + i;                       // startBar = nejnovější, bar roste do minulosti
      int off = i * FEAT_PER_BAR;
      if(bar < 1 || bar >= rates_total - 1) continue; // přeskočit formující (0) a mimo rozsah

      double o   = open[bar];
      double h   = high[bar];
      double l   = low[bar];
      double c   = close[bar];
      double vol = (double)tick_volume[bar];
      // "předchozí" bar = starší = index bar+1 (větší index)
      double prevVol = (bar + 1 < rates_total) ? (double)tick_volume[bar + 1] : vol;
      double range = h - l;
      if(range < _Point) range = _Point;

      int sym = EncodeCandle(o, h, l, c);
      for(int k = 0; k < InpSymbolAlphabet && k < 9; k++)
         feats[off + k] = (k == sym) ? 1.0 : 0.0;

      double body    = c - o;
      double uShadow = h - MathMax(o, c);
      double lShadow = MathMin(o, c) - l;
      double vc      = (prevVol > 0) ? (vol - prevVol) / prevVol : 0.0;

      feats[off + 9]  = body / range;                           // signed body/range
      feats[off + 10] = (uShadow - lShadow) / range;           // shadow imbalance
      feats[off + 11] = MathMax(-2.0, MathMin(2.0, vc));       // volume change (clipped)
     }
  }

//+------------------------------------------------------------------+
//| Network init                                                      |
//+------------------------------------------------------------------+
bool InitNetwork()
  {
   g_NetHandle = DN_Create();
   if(g_NetHandle == 0) { Print("DN_Create fail: ", GetDLLError()); return false; }

   DN_SetSequenceLength(g_NetHandle, InpLookback);
   DN_SetMiniBatchSize(g_NetHandle, InpMiniBatch);
   DN_SetGradClip(g_NetHandle, InpGradClip);

   if(!DN_AddLayerEx(g_NetHandle, FEAT_PER_BAR, InpHiddenSize1, 0, 0, InpDropout))
     { Print("L1 fail: ", GetDLLError()); DN_Free(g_NetHandle); g_NetHandle = 0; return false; }
   if(!DN_AddLayerEx(g_NetHandle, 0, InpHiddenSize2, 0, 0, InpDropout * 0.5))
     { Print("L2 fail: ", GetDLLError()); DN_Free(g_NetHandle); g_NetHandle = 0; return false; }
   if(InpHiddenSize3 > 0)
      if(!DN_AddLayerEx(g_NetHandle, 0, InpHiddenSize3, 0, 0, 0.0))
        { Print("L3 fail: ", GetDLLError()); DN_Free(g_NetHandle); g_NetHandle = 0; return false; }
   if(!DN_SetOutputDim(g_NetHandle, OUTPUT_DIM))
     { Print("OutDim fail: ", GetDLLError()); DN_Free(g_NetHandle); g_NetHandle = 0; return false; }

   EstimateVRAM();
   Print(StringFormat("Network: %d -> LSTM(%d) -> LSTM(%d)%s -> %d",
                      FEAT_PER_BAR, InpHiddenSize1, InpHiddenSize2,
                      InpHiddenSize3 > 0 ? StringFormat(" -> LSTM(%d)", InpHiddenSize3) : "",
                      OUTPUT_DIM));
   return true;
  }

//+------------------------------------------------------------------+
void EstimateVRAM()
  {
   int H1 = InpHiddenSize1, H2 = InpHiddenSize2, H3 = InpHiddenSize3;
   int lastH = (H3 > 0) ? H3 : H2;
   double w1 = (double)(FEAT_PER_BAR + H1) * 4 * H1 * 4;
   double w2 = (double)(H1 + H2) * 4 * H2 * 4;
   double w3 = (H3 > 0) ? (double)(H2 + H3) * 4 * H3 * 4 : 0;
   double wo = (double)lastH * OUTPUT_DIM * 4;
   double weightMem = (w1 + w2 + w3 + wo) * 3;
   double dataMem   = (double)InpTrainBars * InpLookback * FEAT_PER_BAR * 4.0
                    + (double)InpTrainBars * OUTPUT_DIM * 4.0;
   double cacheMem  = (double)InpLookback * InpMiniBatch * (H1 + H2) * 4.0 * 7;
   if(H3 > 0) cacheMem += (double)InpLookback * InpMiniBatch * H3 * 4.0 * 7;
   double predMem = (double)InpPredictBatch * InpLookback * FEAT_PER_BAR * 8.0
                  + (double)InpPredictBatch * OUTPUT_DIM * 8.0;
   g_EstVRAM_MB = (weightMem + dataMem + cacheMem * 1.5 + predMem) / (1024.0 * 1024.0);
   Print(StringFormat("Est. VRAM: %.1f MB", g_EstVRAM_MB));
  }

//+------------------------------------------------------------------+
//| Invalidate caches — vždy po (re)tréninku                         |
//+------------------------------------------------------------------+
void InvalidateCache()
  {
   g_ATRCacheStart      = -1;
   g_ATRCachedBars      = 0;
   g_EntropySmoothState = -1.0;
   g_LastPredictedTo    = -1;   // přepočítáme všechny predikce
  }

//+------------------------------------------------------------------+
//| FIX 5: Fisher-Yates shuffle                                       |
//+------------------------------------------------------------------+
void ShuffleIntArray(int &arr[], int n)
  {
   MathSrand((int)(TimeCurrent() & 0x7FFFFFFF));
   for(int i = n - 1; i > 0; i--)
     {
      int j = (int)(MathRand() % (i + 1));
      int tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
     }
  }

//+------------------------------------------------------------------+
//| ═══════════════════════════════════════════════════════════════════|
//| OnCalculate                                                        |
//| ═══════════════════════════════════════════════════════════════════|
//|                                                                   |
//| Indexová konvence (připomínka):                                   |
//|   [0]            = formující bar (nikdy nepoužíváme pro trénink)  |
//|   [1]            = poslední uzavřený bar                          |
//|   [rates_total-1]= nejstarší bar                                  |
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
   int minBars = InpLookback + InpEntropyWindow + InpPredictAhead + 100;
   if(rates_total < minBars) return 0;

   g_TotalBars = rates_total;

   // Nový uzavřený bar? (detekce vždy podle time[1])
   bool newClosedBar = false;
   if(rates_total >= 2)
     {
      datetime t = time[1];
      if(t != g_LastClosedBarTime)
        {
         g_LastClosedBarTime = t;
         newClosedBar = true;
         // as-series posun indexů při novém baru => indexově kotvené inkrementy reset
         g_LastPredictedTo = -1;
         g_EntropySmoothState = -1.0;
        }
     }

   //--- STEP 1: Symbolické kódování
   //
   // Okno entropie pro bar i: symboly [i .. i+window-1] (větší index = starší).
   // Potřebujeme i + InpEntropyWindow - 1 < rates_total
   //   → i <= rates_total - InpEntropyWindow
   //   → nejnovější (nejmenší) validní i = 1  (pokud rates_total > InpEntropyWindow+1)
   //
   // OPRAVA: firstValidEntropyBar = 1, NE InpEntropyWindow.
   // Původní chyba: firstValidEntropyBar = InpEntropyWindow = 20,
   // takže smyčka začínala od indexu 20 a bar[1] zůstával na 1.0 (noise).
   int calcStart = 1;
   int needForPredict = 1 + InpMaxPredictBars + InpLookback + InpEntropyWindow + 32;
   int minBar = InpPredictAhead + 1;
   int needForTrain = (minBar + InpTrainBars - 1) + InpEntropyWindow;
   int endBar = MathMin(rates_total - InpEntropyWindow,
                        MathMax(needForPredict, needForTrain));
   if(endBar < calcStart)
      endBar = 0;

   if(prev_calculated == 0)
     {
      ArrayInitialize(g_Entropy, 1.0);
      ArrayInitialize(g_RawEntropy, 1.0);
      ArrayInitialize(g_ProbPct, 50.0);
      g_EntropySmoothState = -1.0;
     }
   g_CalcStart = (endBar >= calcStart) ? calcStart : 0;
   g_CalcEnd   = (endBar >= calcStart) ? endBar   : 0;

   // Formující bar [0] a první nevalidní starší bar hned za vypočteným rozsahem držíme v NOISE.
   g_RawEntropy[0]    = 1.0;
   g_Entropy[0]       = 1.0;
   if(endBar + 1 < rates_total)
     {
      g_RawEntropy[endBar + 1]    = 1.0;
      g_Entropy[endBar + 1]       = 1.0;
     }

   // Kódujeme symboly jen pro bary nutné pro entropy rozsah [1..endBar]
   int symStart = calcStart;
   int symEnd   = MathMin(rates_total - 1, endBar + InpEntropyWindow - 1);
   for(int i = symStart; i <= symEnd; i++)
      g_SymbolSeq[i] = (double)EncodeCandle(open[i], high[i], low[i], close[i]);

   //--- STEP 2: Entropy
   g_LowEntropyBars  = 0;
   g_HighEntropyBars = 0;
   double entropySum = 0.0;
   int    entropyCnt = 0;
   double smoothState = (endBar + 1 < rates_total) ? g_Entropy[endBar + 1] : 1.0;
   if(smoothState < 0.0) smoothState = 1.0;

   for(int i = endBar; i >= calcStart; i--)
     {
      // Okno entropie: bary [i .. i + window - 1] (do minulosti = větší indexy).
      // Guard: potřebujeme i + InpEntropyWindow - 1 < rates_total.
      // Bary na konci pole (nejstarší) nemají plné okno — přeskočíme je.
      if(i + InpEntropyWindow - 1 >= rates_total)
        {
         g_RawEntropy[i] = 1.0;
         g_Entropy[i]    = 1.0;
         continue;
        }

      double rawH = ComputeShannonEntropy(g_SymbolSeq, i, InpEntropyWindow, InpSymbolAlphabet);
      g_RawEntropy[i] = rawH;

      double smoothH = SmoothEntropy(rawH, smoothState, InpEntropySmooth);
      smoothState    = smoothH;
      g_Entropy[i]   = smoothH;
      bool lowEntropy  = (smoothH < InpEntropyThreshold);

      // Statistiky posledních 200 barů (indexy 1..200 = nejnovější uzavřené)
      if(i >= 1 && i <= 200)
        {
         if(lowEntropy) g_LowEntropyBars++;
         else           g_HighEntropyBars++;
         entropySum += smoothH;
         entropyCnt++;
        }

     }

   g_EntropySmoothState = smoothState;
   if(entropyCnt > 0)
      g_AvgEntropyRecent = entropySum / entropyCnt;

   //--- STEP 3: Training
   if(!g_ModelReady && !g_IsTraining)
     {
      if(StartTraining(rates_total, open, high, low, close, tick_volume, true))
         g_IsTraining = true;
     }
   if(g_ModelReady && InpAutoRetrain && !g_IsTraining && newClosedBar)
     {
      if(rates_total - g_LastTrainBar >= InpRetrainInterval)
         if(StartTraining(rates_total, open, high, low, close, tick_volume, false))
            g_IsTraining = true;
     }

   //--- STEP 4: Predikce
   if(g_ModelReady && !g_IsTraining)
     {
      if(newClosedBar || prev_calculated == 0)
         BulkPredict(rates_total, open, high, low, close, tick_volume);
     }

   // Formující bar [0] = vždy prázdný
   g_ProbPct[0] = EMPTY_VALUE;

   UpdateInfoPanel();
   UpdateProgressBar();
   return rates_total;
  }

//+------------------------------------------------------------------+
//| Timer                                                             |
//+------------------------------------------------------------------+
void OnTimer()
  {
   if(!g_IsTraining || g_NetHandle == 0) return;

   PollGPUProgress();
   int st = DN_GetTrainingStatus(g_NetHandle);
   if(st == 1) { UpdateInfoPanel(); UpdateProgressBar(); return; }

   g_IsTraining = false;
   double mse = 0;
   int ep = 0;
   DN_GetTrainingResult(g_NetHandle, mse, ep);
   g_LastMSE     = mse;
   g_CurrentEpochs = ep;
   g_TotalEpochs += ep;

   if(st == 2)
     {
      Print(StringFormat("Training done: MSE=%.6f ep=%d elapsed=%.1fs", mse, ep, g_ProgElapsedSec));
      if(mse < g_BestMSE) { g_BestMSE = mse; DN_SnapshotWeights(g_NetHandle); }
      g_ModelReady   = true;
      g_LastTrainBar = g_TotalBars;
      InvalidateCache();
      if(InpSaveModel) SaveModel();
     }
   else
     {
      Print("Training error: ", GetDLLError());
      DN_RestoreWeights(g_NetHandle);
     }

   UpdateInfoPanel();
   UpdateProgressBar();
  }

//+------------------------------------------------------------------+
void PollGPUProgress()
  {
   if(g_NetHandle == 0) return;
   if(!DN_GetProgressAll(g_NetHandle,
                         g_ProgEpoch, g_ProgTotalEpochs,
                         g_ProgMB, g_ProgTotalMB,
                         g_ProgLR, g_ProgMSE, g_ProgBestMSE,
                         g_ProgGradNorm, g_ProgPercent,
                         g_ProgElapsedSec, g_ProgETASec)) return;
   g_ProgTotalSteps = DN_GetProgressTotalSteps(g_NetHandle);
   g_CurrentEpochs  = g_ProgEpoch;
   if(g_ProgMSE > 0)                              g_LastMSE = g_ProgMSE;
   if(g_ProgBestMSE > 0 && g_ProgBestMSE < 1e10) g_BestMSE = g_ProgBestMSE;
  }

//+------------------------------------------------------------------+
//| ═══════════════════════════════════════════════════════════════════|
//| StartTraining                                                      |
//|                                                                   |
//| Indexová konvence (opakování):                                    |
//|   predBar = bar, od kterého predikujeme (kde máme vstupní data)  |
//|   tgtBar  = predBar - InpPredictAhead                            |
//|           = menší index = novější bar = ten, který NASTANE         |
//|             InpPredictAhead barů po predBar                       |
//|   actualMove = close[tgtBar] - close[predBar]                    |
//|              = pohyb ceny od predBar do tgtBar (novějšího)        |
//+------------------------------------------------------------------+
bool StartTraining(int rates_total,
                   const double &open[],
                   const double &high[],
                   const double &low[],
                   const double &close[],
                   const long   &tick_volume[],
                   bool initial)
  {
   if(g_NetHandle == 0) return false;

   g_TargetEpochs   = initial ? InpInitialEpochs : InpRetrainEpochs;
   g_CurrentEpochs  = 0;
   g_TrainStartTime = TimeCurrent();
   g_ProgEpoch = 0; g_ProgTotalEpochs = g_TargetEpochs;
   g_ProgMB = 0; g_ProgTotalMB = 0; g_ProgTotalSteps = 0;
   g_ProgLR = InpLearningRate; g_ProgMSE = 0; g_ProgBestMSE = 0;
   g_ProgGradNorm = 0; g_ProgPercent = 0; g_ProgElapsedSec = 0; g_ProgETASec = 0;

   ComputeATRMean(rates_total, close, high, low);

   // Hranice trénovacích barů:
   //   predBar musí mít:
   //     tgtBar = predBar - InpPredictAhead >= 1      (target = uzavřený novější bar)
   //     predBar + InpLookback - 1 <= rates_total - 2 (sekvence sahá jen do uzavřených barů)
   //
   //   → predBar <= rates_total - 1 - InpLookback   (nejstarší možný predBar)
   //   → predBar >= InpPredictAhead + 1              (nejnovější možný predBar)
   //
   //   minBar = nejnovější candidate (nejmenší index)
   //   maxBar = nejstarší candidate (největší index, s omezením InpTrainBars)

   int minBar = InpPredictAhead + 1;
   int maxBar = rates_total - 1 - InpLookback;
   if(maxBar > minBar + InpTrainBars - 1) maxBar = minBar + InpTrainBars - 1;

   if(maxBar < minBar)
     {
      Print(StringFormat("StartTraining: nedostatek barů (minBar=%d, maxBar=%d)", minBar, maxBar));
      return false;
     }

   // FIX 3: Kandidáti — jen bary s platnou entropií (>= g_CalcStart)
   //         a s low entropy (< threshold)
   int candidates[];
   ArrayResize(candidates, 0);
   for(int bar = minBar; bar <= maxBar; bar++)
     {
      // FIX 3: bar musí být >= g_CalcStart (entropie je skutečně spočítána)
      if(bar < g_CalcStart || bar > g_CalcEnd) continue;
      // FIX 3: g_RawEntropy[bar] > 0 jako další pojistka
      if(g_RawEntropy[bar] <= 0.0 || g_RawEntropy[bar] >= 1.0) continue;
      if(g_Entropy[bar] < InpEntropyThreshold)
        {
         int sz = ArraySize(candidates);
         ArrayResize(candidates, sz + 1);
         candidates[sz] = bar;
        }
     }

   int nCandidates = ArraySize(candidates);
   if(nCandidates < 50)
     {
      Print(StringFormat("Málo low-entropy vzorků: %d (potřeba 50+)", nCandidates));
      if(initial)
        {
         Print("Fallback: trénuji na všech (validních) barech");
         ArrayResize(candidates, 0);
         for(int bar = minBar; bar <= maxBar; bar++)
           {
            if(bar < g_CalcStart || bar > g_CalcEnd) continue;
            if(g_RawEntropy[bar] <= 0.0 || g_RawEntropy[bar] >= 1.0) continue;
            int sz = ArraySize(candidates);
            ArrayResize(candidates, sz + 1);
            candidates[sz] = bar;
           }
         nCandidates = ArraySize(candidates);
         if(nCandidates < 50) { Print("Stále málo: ", nCandidates); return false; }
        }
      else return false;
     }

   // FIX 5: Shuffle kandidátů → diverzitní minibatche, lepší generalizace
   ShuffleIntArray(candidates, nCandidates);

   int maxSamples = MathMin(nCandidates, InpTrainBars);
   int inDim      = InpLookback * FEAT_PER_BAR;

   // Po shufflu: candidates[0] může být libovolný bar.
   // Potřebujeme feature rozsah pokrývající všechny vzorky.
   // Najdeme min (nejnovější) a max (nejstarší) bar.
   int featNewest = candidates[0];
   int featOldest = candidates[0];
   for(int s = 1; s < maxSamples; s++)
     {
      if(candidates[s] < featNewest) featNewest = candidates[s];
      if(candidates[s] > featOldest) featOldest = candidates[s];
     }
   featOldest = MathMin(featOldest + InpLookback - 1, rates_total - 2);
   int featRange = featOldest - featNewest + 1;
   if(featRange <= 0) { Print("StartTraining: featRange <= 0"); return false; }

   double barFeats[];
   ExtractSymbolicFeatures(featNewest, featRange, open, high, low, close,
                           tick_volume, rates_total, barFeats);

   int sampleBars[];
   ArrayResize(sampleBars, maxSamples);
   for(int s = 0; s < maxSamples; s++) sampleBars[s] = candidates[s];

   double X[];
   BuildSequenceInput(barFeats, featNewest, featRange, sampleBars, maxSamples, InpLookback, X);

   // Targety:
   //   predBar = sampleBars[s]  (vstupní bar)
   //   tgtBar  = predBar - InpPredictAhead  (novější bar = menší index)
   //   actualMove = close[tgtBar] - close[predBar]  (kladné = cena rostla)
   double T[];
   ArrayResize(T, maxSamples * OUTPUT_DIM);
   ArrayInitialize(T, 0.0);

   for(int s = 0; s < maxSamples; s++)
     {
      int predBar = sampleBars[s];
      int tgtBar  = predBar - InpPredictAhead;   // menší index = novější = budoucí bar

      if(tgtBar < 1)              continue; // target musí být uzavřený
      if(predBar >= rates_total)  continue;

      double pc = close[predBar]; // cena v čase predikce
      double cc = close[tgtBar];  // cena InpPredictAhead barů "v budoucnosti"
      if(pc < _Point) pc = _Point;

      double ret     = (cc - pc) / pc;
      double atr     = GetCachedATR(predBar);
      if(atr < _Point) atr = _Point;
      double normRet = ret / (atr / pc);

      double pBull = 1.0 / (1.0 + MathExp(-normRet * 3.0));
      T[s * OUTPUT_DIM + 0] = pBull;
      T[s * OUTPUT_DIM + 1] = 1.0 - pBull;
     }

   if(!DN_LoadBatch(g_NetHandle, X, T, maxSamples, inDim, OUTPUT_DIM, 0))
     { Print("LoadBatch fail: ", GetDLLError()); return false; }
   if(!DN_TrainAsync(g_NetHandle, g_TargetEpochs, InpTargetMSE, InpLearningRate, InpWeightDecay))
     { Print("TrainAsync fail: ", GetDLLError()); return false; }

   Print(StringFormat("Training: %d vzorků x %d epoch | feature bars [%d..%d] | shuffled",
                      maxSamples, g_TargetEpochs, featNewest, featOldest));
   UpdateProgressBar();
   return true;
  }

//+------------------------------------------------------------------+
//| ═══════════════════════════════════════════════════════════════════|
//| BulkPredict                                                        |
//|                                                                   |
//| Indexová konvence:                                                |
//|   [0]   = formující bar — NIKDY nepredikujeme                    |
//|   [1]   = nejnovější uzavřený bar — VŽDY predikujeme              |
//|   větší = starší                                                  |
//|                                                                   |
//| Pro bar barIdx:                                                   |
//|   Sekvence vstupu jde do minulosti: bary [barIdx .. barIdx+L-1]  |
//|   Predikce říká: "co nastane v barIdx - InpPredictAhead?"        |
//|   evalBar = barIdx - InpPredictAhead < barIdx (novější = menší)  |
//+------------------------------------------------------------------+
void BulkPredict(int rates_total,
                 const double &open[],
                 const double &high[],
                 const double &low[],
                 const double &close[],
                 const long   &tick_volume[])
  {
   if(!g_ModelReady || g_NetHandle == 0 || g_IsTraining) return;

   int newestPredBar = 1;
   int oldestPredBar = MathMin(rates_total - 1 - InpLookback,
                               newestPredBar + InpMaxPredictBars - 1);
   if(oldestPredBar < newestPredBar)
     {
      if(InpVerboseLog) Print("BulkPredict: oldestPredBar < newestPredBar");
      return;
     }

   int startPred = newestPredBar;
   int lastPred  = oldestPredBar;
   if(g_LastPredictedTo != -1)
      lastPred = MathMin(oldestPredBar, g_LastPredictedTo + 2);

   int totalPredict = lastPred - startPred + 1;
   if(totalPredict <= 0) return;

   if(g_ATRMean < _Point) ComputeATRMean(rates_total, close, high, low);
   BulkComputeATRWilder(startPred, totalPredict + InpLookback + 10,
                        14, high, low, close, rates_total);

   for(int i = 0; i < totalPredict; i++)
     {
      int bar = startPred + i;
      if(bar >= 1 && bar < rates_total)
         g_ProbPct[bar] = 50.0;
     }

   int lstmBars[];
   ArrayResize(lstmBars, 0);

   for(int i = 0; i < totalPredict; i++)
     {
      int bar = startPred + i;
      if(bar < 1 || bar >= rates_total) continue;

      bool validEntropy = (g_CalcStart > 0 && bar >= g_CalcStart && bar <= g_CalcEnd);
      double w = validEntropy ? EntropyWeight(g_Entropy[bar], InpEntropyThreshold, InpEntropyMargin) : 0.0;
      if(w > 0.0)
        {
         int sz = ArraySize(lstmBars);
         ArrayResize(lstmBars, sz + 1);
         lstmBars[sz] = bar;
        }
     }

   int nLSTM = ArraySize(lstmBars);
   if(nLSTM == 0)
     {
      g_LastPredictedTo = lastPred;
      if(InpVerboseLog)
         Print(StringFormat("BulkPredict: %d barů => neutral only (w=0)", totalPredict));
      return;
     }

   int featNewest = lstmBars[0];
   int featOldest = MathMin(lstmBars[nLSTM - 1] + InpLookback - 1, rates_total - 1);
   int featRange  = featOldest - featNewest + 1;
   if(featRange <= 0) { Print("BulkPredict: featRange <= 0"); return; }

   uint t0 = GetTickCount();
   double barFeats[];
   ExtractSymbolicFeatures(featNewest, featRange, open, high, low, close,
                           tick_volume, rates_total, barFeats);

   int batchSize = InpPredictBatch;
   int inDim     = InpLookback * FEAT_PER_BAR;
   int processed = 0;

   for(int batchStart = 0; batchStart < nLSTM; batchStart += batchSize)
     {
      int curBatch = MathMin(batchSize, nLSTM - batchStart);
      int sampleBars[];
      ArrayResize(sampleBars, curBatch);
      for(int i = 0; i < curBatch; i++)
         sampleBars[i] = lstmBars[batchStart + i];

      double X[];
      BuildSequenceInput(barFeats, featNewest, featRange,
                         sampleBars, curBatch, InpLookback, X);

      double Y[];
      ArrayResize(Y, curBatch * OUTPUT_DIM);
      if(!DN_PredictBatch(g_NetHandle, X, curBatch, inDim, 0, Y))
        {
         if(InpVerboseLog) Print("PredictBatch fail at offset ", batchStart);
         break;
        }

      for(int i = 0; i < curBatch; i++)
        {
         int barIdx = sampleBars[i];
         if(barIdx < 1 || barIdx >= rates_total) continue;

         double rawBull = Y[i * OUTPUT_DIM + 0];
         double rawBear = Y[i * OUTPUT_DIM + 1];
         double pBull   = 1.0 / (1.0 + MathExp(-rawBull));
         double pBear   = 1.0 / (1.0 + MathExp(-rawBear));
         double total   = pBull + pBear;
         if(total > 0.001) { pBull /= total; pBear /= total; }
         else              { pBull = 0.5;    pBear = 0.5;    }

         bool validEntropy = (g_CalcStart > 0 && barIdx >= g_CalcStart && barIdx <= g_CalcEnd);
         double w = validEntropy ? EntropyWeight(g_Entropy[barIdx], InpEntropyThreshold, InpEntropyMargin) : 0.0;

         double pBullW = 0.5 + w * (pBull - 0.5);
         pBullW = MathMax(0.0, MathMin(1.0, pBullW));
         double pBearW = 1.0 - pBullW;

         g_ProbPct[barIdx] = 100.0 * pBullW;

         double predDir = pBullW - pBearW;
         if(MathAbs(predDir) > 0.1)
           {
            int evalBar = barIdx - InpPredictAhead;
            if(evalBar >= 1 && evalBar < barIdx)
              {
               double actualMove = close[evalBar] - close[barIdx];
               bool correct = (predDir > 0 && actualMove > 0) ||
                              (predDir < 0 && actualMove < 0);
               g_TotalPred++;
               if(correct) g_CorrectPred++;
              }
           }
        }
      processed += curBatch;
     }

   g_LastPredictedTo = lastPred;
   if(InpVerboseLog)
      Print(StringFormat("BulkPredict: %d/%d barů (LSTM: %d) za %d ms",
                         processed, totalPredict, nLSTM, GetTickCount() - t0));
  }

//+------------------------------------------------------------------+
//| BuildSequenceInput                                                |
//|                                                                   |
//| Pro každý vzorek s:                                               |
//|   tgtBar = sampleBars[s]  (nejnovější bar sekvence)              |
//|   Sekvence vstupů: t=0 (nejstarší) .. t=seqLen-1 (nejnovější)   |
//|   Bar pro t: tgtBar + (seqLen - 1 - t)                           |
//|   = tgtBar + seqLen - 1 = nejstarší                              |
//|   = tgtBar + 0          = nejnovější (tgtBar sám)                |
//|   Větší index = starší → tgtBar + (seqLen-1) je nejstarší v seq. |
//+------------------------------------------------------------------+
void BuildSequenceInput(const double &barFeatures[],
                        int featureStartBar,
                        int totalFeatBars,
                        const int &sampleBars[],
                        int nSamples,
                        int seqLen,
                        double &X[])
  {
   int inDim = seqLen * FEAT_PER_BAR;
   ArrayResize(X, nSamples * inDim);

   for(int s = 0; s < nSamples; s++)
     {
      int tgtBar = sampleBars[s]; // nejnovější bar sekvence (menší index)
      for(int t = 0; t < seqLen; t++)
        {
         // t=0 nejstarší, t=seqLen-1 nejnovější
         int bar   = tgtBar + (seqLen - 1 - t); // větší index = starší = t=0
         int cIdx  = bar - featureStartBar;
         int xOff  = s * inDim + t * FEAT_PER_BAR;
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
//| FIX 4: ATR — Wilderova rekurence O(count) místo O(count*period)  |
//|                                                                   |
//| startBar = nejnovější bar bloku (menší index).                   |
//| Počítáme od nejstaršího (startBar+count-1) ke startBar,          |
//| protože Wilder potřebuje starší bary jako základ.                |
//+------------------------------------------------------------------+
void BulkComputeATRWilder(int startBar, int count, int period,
                          const double &high[],
                          const double &low[],
                          const double &close[],
                          int rates_total)
  {
   if(count <= 0 || period <= 0) return;

   // Zkontroluj zda cache pokrývá celý požadovaný rozsah
   if(g_ATRCacheStart == startBar && g_ATRCachedBars >= count) return;

   int totalNeeded = count;
   ArrayResize(g_ATRCache, totalNeeded);
   g_ATRCacheStart = startBar;
   g_ATRCachedBars = totalNeeded;

   ArrayInitialize(g_ATRCache, _Point * 100);

   int oldestBar = startBar + count - 1;
   if(oldestBar > rates_total - 2) oldestBar = rates_total - 2; // bar+1 musí existovat
   if(startBar < 1 || oldestBar < startBar) return;

   // Init ATR jako SMA z TR na nejstarším baru rozsahu přes `period` hodnot k novějším.
   double sumTR = 0.0;
   int trCount = 0;
   for(int b = oldestBar; b >= startBar && trCount < period; b--)
     {
      if(b + 1 >= rates_total) continue;
      double h  = high[b];
      double l  = low[b];
      double pc = close[b + 1];
      double tr = MathMax(h - l, MathMax(MathAbs(h - pc), MathAbs(l - pc)));
      sumTR += tr;
      trCount++;
     }
   if(trCount <= 0) return;

   double atr = sumTR / trCount;
   int cOldest = oldestBar - startBar;
   if(cOldest >= 0 && cOldest < totalNeeded)
      g_ATRCache[cOldest] = atr;

   // Wilder rekurence: od staršího k novějšímu (index klesá)
   for(int bar = oldestBar - 1; bar >= startBar; bar--)
     {
      if(bar + 1 >= rates_total) continue;
      double h  = high[bar];
      double l  = low[bar];
      double pc = close[bar + 1];
      double tr = MathMax(h - l, MathMax(MathAbs(h - pc), MathAbs(l - pc)));
      atr = (atr * (period - 1) + tr) / period;

      int cIdx = bar - startBar;
      if(cIdx >= 0 && cIdx < totalNeeded)
         g_ATRCache[cIdx] = atr;
     }
  }

//+------------------------------------------------------------------+
double GetCachedATR(int barIdx)
  {
   if(g_ATRCacheStart < 0) return _Point * 100;
   int idx = barIdx - g_ATRCacheStart;
   if(idx < 0 || idx >= g_ATRCachedBars) return _Point * 100;
   return MathMax(g_ATRCache[idx], _Point);
  }

//+------------------------------------------------------------------+
void ComputeATRMean(int rates_total,
                    const double &close[],
                    const double &high[],
                    const double &low[])
  {
   int n = MathMin(InpTrainBars, rates_total - InpLookback - 10);
   if(n <= 0) { g_ATRMean = _Point * 100; return; }

   // Bary 1..n (nejnovější uzavřené)
   BulkComputeATRWilder(1, n, 14, high, low, close, rates_total);

   double sumATR = 0.0;
   int cnt = 0;
   for(int i = 0; i < n; i++)
     {
      int bar = 1 + i;
      if(bar >= rates_total - 1) break;
      sumATR += GetCachedATR(bar);
      cnt++;
     }
   g_ATRMean = (cnt > 0) ? sumATR / cnt : _Point * 100;
   if(g_ATRMean < _Point) g_ATRMean = _Point;
  }

//+------------------------------------------------------------------+
//| Info Panel                                                        |
//+------------------------------------------------------------------+
void CreateInfoPanel()
  {
   string bg = g_InfoPrefix + "BG";
   ObjectCreate(0, bg, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, bg, OBJPROP_CORNER, InpInfoCorner);
   ObjectSetInteger(0, bg, OBJPROP_XDISTANCE, 10);
   ObjectSetInteger(0, bg, OBJPROP_YDISTANCE, 30);
   ObjectSetInteger(0, bg, OBJPROP_XSIZE, 345);
   ObjectSetInteger(0, bg, OBJPROP_YSIZE, 280);
   ObjectSetInteger(0, bg, OBJPROP_BGCOLOR, C'30,30,40');
   ObjectSetInteger(0, bg, OBJPROP_BORDER_COLOR, clrDodgerBlue);
   ObjectSetInteger(0, bg, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(0, bg, OBJPROP_WIDTH, 2);
   ObjectSetInteger(0, bg, OBJPROP_BACK, false);
   ObjectSetInteger(0, bg, OBJPROP_SELECTABLE, false);
   MakeLabel(g_InfoPrefix + "T", "Entropy-Gated LSTM v4.30",
             15, 35, clrDodgerBlue, 10, true);
  }

//+------------------------------------------------------------------+
void UpdateInfoPanel()
  {
   string st; color sc;
   if(g_IsTraining)      { st = "GPU Training..."; sc = clrYellow; }
   else if(g_ModelReady) { st = "Ready";           sc = clrLime;   }
   else                  { st = "Waiting";         sc = clrGray;   }
   MakeLabel(g_InfoPrefix + "St", "Status: " + st, 15, 55, sc, 9, false);

   string arch = StringFormat("Symbolic(%d) -> LSTM(%d->%d",
                              FEAT_PER_BAR, InpHiddenSize1, InpHiddenSize2);
   if(InpHiddenSize3 > 0) arch += "->" + IntegerToString(InpHiddenSize3);
   arch += ") -> P(bull,bear)";
   MakeLabel(g_InfoPrefix + "Ar", arch, 15, 72, clrWhite, 8, false);

   double dispMSE  = (g_IsTraining && g_ProgMSE > 0) ? g_ProgMSE : g_LastMSE;
   double dispBest = (g_IsTraining && g_ProgBestMSE > 0 && g_ProgBestMSE < 1e10)
                     ? g_ProgBestMSE : g_BestMSE;
   MakeLabel(g_InfoPrefix + "MSE",
             StringFormat("MSE: %.6f (best: %.6f)", dispMSE, dispBest),
             15, 89, clrSilver, 9, false);
   MakeLabel(g_InfoPrefix + "Ep",
             StringFormat("Epochs: %d | VRAM: ~%.0f MB", g_TotalEpochs, g_EstVRAM_MB),
             15, 106, clrSilver, 9, false);

   double acc = (g_TotalPred > 0) ? (double)g_CorrectPred / g_TotalPred * 100.0 : 0;
   color  ac  = (acc > 55) ? clrLime : (acc > 50) ? clrYellow : clrOrangeRed;
   string accStr = (g_TotalPred == 0)
                   ? "Direction accuracy: -- (zatím žádné predikce)"
                   : StringFormat("Direction accuracy: %.1f%% (%d/%d)",
                                  acc, g_CorrectPred, g_TotalPred);
   MakeLabel(g_InfoPrefix + "Acc", accStr, 15, 123, ac, 9, false);

   MakeLabel(g_InfoPrefix + "EntTitle", "── Entropy-weighted confidence ──", 15, 143, clrGold, 9, true);

   double curEntropy = g_Entropy[1];
   bool   validEntropy = (g_CalcStart > 0 && 1 >= g_CalcStart && 1 <= g_CalcEnd);
   double curW = validEntropy ? EntropyWeight(curEntropy, InpEntropyThreshold, InpEntropyMargin) : 0.0;
   MakeLabel(g_InfoPrefix + "Regime",
             StringFormat("Confidence weight: %.2f", curW), 15, 160, clrLightSteelBlue, 10, true);
   MakeLabel(g_InfoPrefix + "EntVal",
             StringFormat("H=%.3f thr=%.2f margin=%.2f | alpha=%d win=%d",
                          curEntropy, InpEntropyThreshold, InpEntropyMargin,
                          InpSymbolAlphabet, InpEntropyWindow),
             15, 178, clrSilver, 8, false);

   int totalRegime = g_LowEntropyBars + g_HighEntropyBars;
   double pctStruct = (totalRegime > 0) ? (double)g_LowEntropyBars / totalRegime * 100.0 : 0;
   MakeLabel(g_InfoPrefix + "RegDist",
             StringFormat("Posl. 200: %.0f%% below thr, %.0f%% above thr (avg H=%.3f)",
                          pctStruct, 100 - pctStruct, g_AvgEntropyRecent),
             15, 194, clrDarkGray, 8, false);

   if(g_ModelReady)
     {
      double pbPct = g_ProbPct[1];
      if(pbPct == EMPTY_VALUE) pbPct = 50.0;
      double pb = pbPct / 100.0;
      double pbr = 1.0 - pb;
      string predStr; color predClr;
      if(pbPct > 55.0)       { predStr = StringFormat("BULLISH %.0f%%", pbPct);       predClr = InpBullColor; }
      else if(pbPct < 45.0)  { predStr = StringFormat("BEARISH %.0f%%", 100.0-pbPct); predClr = InpBearColor; }
      else                   { predStr = StringFormat("NEUTRAL %.0f%%", pbPct);       predClr = InpNoiseColor; }
      MakeLabel(g_InfoPrefix + "Pred",
                StringFormat("Next %d bars: %s", InpPredictAhead, predStr),
                15, 215, predClr, 10, true);
      MakeLabel(g_InfoPrefix + "PredD",
                StringFormat("P(bull)=%.3f  P(bear)=%.3f", pb, pbr),
                15, 233, clrSilver, 9, false);
     }
   else
     {
      MakeLabel(g_InfoPrefix + "Pred",  "Čekám na model...", 15, 215, clrGray, 9, false);
      MakeLabel(g_InfoPrefix + "PredD", "", 15, 233, clrGray, 8, false);
     }

   MakeLabel(g_InfoPrefix + "SrcBar",
             StringFormat("Source: bar[1] (%s) | bar0=forming",
                          TimeToString(g_LastClosedBarTime, TIME_DATE | TIME_MINUTES)),
             15, 258, clrDarkGray, 8, false);
   ChartRedraw();
  }

//+------------------------------------------------------------------+
void MakeLabel(string name, string text, int x, int y,
               color clr, int sz, bool bold)
  {
   if(ObjectFind(0, name) < 0)
     {
      ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, name, OBJPROP_CORNER, InpInfoCorner);
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
     }
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, sz);
   ObjectSetString(0, name, OBJPROP_FONT, bold ? "Arial Bold" : "Arial");
   ObjectSetString(0, name, OBJPROP_TEXT, text);
  }

//+------------------------------------------------------------------+
//| Progress Bar                                                      |
//+------------------------------------------------------------------+
void CreateProgressBar()
  {
   int yOff = 335;
   struct ObjDef { string suffix; int type; };
   // Bg, Fill, Txt, Det, MSE, LR, Title
   string bg = g_ProgPrefix + "BG";
   ObjectCreate(0, bg, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, bg, OBJPROP_CORNER, InpInfoCorner);
   ObjectSetInteger(0, bg, OBJPROP_XDISTANCE, 15);
   ObjectSetInteger(0, bg, OBJPROP_YDISTANCE, yOff);
   ObjectSetInteger(0, bg, OBJPROP_XSIZE, InpProgressWidth);
   ObjectSetInteger(0, bg, OBJPROP_YSIZE, InpProgressHeight);
   ObjectSetInteger(0, bg, OBJPROP_BGCOLOR, C'40,40,50');
   ObjectSetInteger(0, bg, OBJPROP_BORDER_COLOR, clrDimGray);
   ObjectSetInteger(0, bg, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(0, bg, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, bg, OBJPROP_HIDDEN, true);

   string fi = g_ProgPrefix + "Fill";
   ObjectCreate(0, fi, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, fi, OBJPROP_CORNER, InpInfoCorner);
   ObjectSetInteger(0, fi, OBJPROP_XDISTANCE, 15);
   ObjectSetInteger(0, fi, OBJPROP_YDISTANCE, yOff);
   ObjectSetInteger(0, fi, OBJPROP_XSIZE, 0);
   ObjectSetInteger(0, fi, OBJPROP_YSIZE, InpProgressHeight);
   ObjectSetInteger(0, fi, OBJPROP_BGCOLOR, clrDodgerBlue);
   ObjectSetInteger(0, fi, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(0, fi, OBJPROP_WIDTH, 0);
   ObjectSetInteger(0, fi, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, fi, OBJPROP_HIDDEN, true);

   string tx = g_ProgPrefix + "Txt";
   ObjectCreate(0, tx, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, tx, OBJPROP_CORNER, InpInfoCorner);
   ObjectSetInteger(0, tx, OBJPROP_XDISTANCE, 15 + InpProgressWidth / 2);
   ObjectSetInteger(0, tx, OBJPROP_YDISTANCE, yOff + 2);
   ObjectSetInteger(0, tx, OBJPROP_FONTSIZE, 9);
   ObjectSetString(0, tx, OBJPROP_FONT, "Arial Bold");
   ObjectSetInteger(0, tx, OBJPROP_COLOR, clrWhite);
   ObjectSetInteger(0, tx, OBJPROP_ANCHOR, ANCHOR_CENTER);
   ObjectSetInteger(0, tx, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, tx, OBJPROP_HIDDEN, true);

   string dt = g_ProgPrefix + "Det";
   ObjectCreate(0, dt, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, dt, OBJPROP_CORNER, InpInfoCorner);
   ObjectSetInteger(0, dt, OBJPROP_XDISTANCE, 15);
   ObjectSetInteger(0, dt, OBJPROP_YDISTANCE, yOff + InpProgressHeight + 4);
   ObjectSetInteger(0, dt, OBJPROP_FONTSIZE, 8);
   ObjectSetString(0, dt, OBJPROP_FONT, "Arial");
   ObjectSetInteger(0, dt, OBJPROP_COLOR, clrSilver);
   ObjectSetInteger(0, dt, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, dt, OBJPROP_HIDDEN, true);

   string ms = g_ProgPrefix + "MSE";
   ObjectCreate(0, ms, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, ms, OBJPROP_CORNER, InpInfoCorner);
   ObjectSetInteger(0, ms, OBJPROP_XDISTANCE, 15);
   ObjectSetInteger(0, ms, OBJPROP_YDISTANCE, yOff + InpProgressHeight + 19);
   ObjectSetInteger(0, ms, OBJPROP_FONTSIZE, 8);
   ObjectSetString(0, ms, OBJPROP_FONT, "Arial");
   ObjectSetInteger(0, ms, OBJPROP_COLOR, clrGold);
   ObjectSetInteger(0, ms, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, ms, OBJPROP_HIDDEN, true);

   string lr = g_ProgPrefix + "LR";
   ObjectCreate(0, lr, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, lr, OBJPROP_CORNER, InpInfoCorner);
   ObjectSetInteger(0, lr, OBJPROP_XDISTANCE, 15);
   ObjectSetInteger(0, lr, OBJPROP_YDISTANCE, yOff + InpProgressHeight + 34);
   ObjectSetInteger(0, lr, OBJPROP_FONTSIZE, 8);
   ObjectSetString(0, lr, OBJPROP_FONT, "Arial");
   ObjectSetInteger(0, lr, OBJPROP_COLOR, clrCornflowerBlue);
   ObjectSetInteger(0, lr, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, lr, OBJPROP_HIDDEN, true);

   string tt = g_ProgPrefix + "Title";
   ObjectCreate(0, tt, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, tt, OBJPROP_CORNER, InpInfoCorner);
   ObjectSetInteger(0, tt, OBJPROP_XDISTANCE, 15);
   ObjectSetInteger(0, tt, OBJPROP_YDISTANCE, yOff - 16);
   ObjectSetInteger(0, tt, OBJPROP_FONTSIZE, 9);
   ObjectSetString(0, tt, OBJPROP_FONT, "Arial Bold");
   ObjectSetInteger(0, tt, OBJPROP_COLOR, clrDodgerBlue);
   ObjectSetInteger(0, tt, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, tt, OBJPROP_HIDDEN, true);
  }

//+------------------------------------------------------------------+
void UpdateProgressBar()
  {
   bool show = g_IsTraining ||
               (g_TrainStartTime > 0 && TimeCurrent() - g_TrainStartTime < 5);
   string names[] = {"BG","Fill","Txt","Det","MSE","LR","Title"};
   for(int i = 0; i < ArraySize(names); i++)
      ObjectSetInteger(0, g_ProgPrefix + names[i], OBJPROP_TIMEFRAMES,
                       show ? OBJ_ALL_PERIODS : OBJ_NO_PERIODS);
   if(!show) { ChartRedraw(); return; }

   double prog = g_ProgPercent / 100.0;
   prog = MathMax(0.0, MathMin(1.0, prog));
   if(prog < 0.001 && g_TargetEpochs > 0 && g_CurrentEpochs > 0)
      prog = MathMin(1.0, (double)g_CurrentEpochs / g_TargetEpochs);

   color fc;
   if(prog < 0.33)      fc = clrOrangeRed;
   else if(prog < 0.66) fc = clrGold;
   else if(prog < 1.0)  fc = clrDodgerBlue;
   else                 fc = clrLime;

   ObjectSetInteger(0, g_ProgPrefix+"Fill", OBJPROP_XSIZE, MathMax(0,(int)(InpProgressWidth*prog)));
   ObjectSetInteger(0, g_ProgPrefix+"Fill", OBJPROP_BGCOLOR, fc);
   ObjectSetString(0,  g_ProgPrefix+"Txt", OBJPROP_TEXT, StringFormat("%.1f%%", prog*100));

   string det = "";
   if(g_ProgTotalEpochs > 0)
     {
      det = StringFormat("Ep %d/%d", g_ProgEpoch, g_ProgTotalEpochs);
      if(g_ProgTotalMB > 0)   det += StringFormat(" | MB %d/%d", g_ProgMB, g_ProgTotalMB);
      if(g_ProgETASec > 0)    det += " | ETA " + FormatDuration(g_ProgETASec);
      if(g_ProgElapsedSec > 0) det += " | " + FormatDuration(g_ProgElapsedSec);
     }
   else det = StringFormat("Epoch %d / %d", g_CurrentEpochs, g_TargetEpochs);
   ObjectSetString(0, g_ProgPrefix+"Det", OBJPROP_TEXT, det);

   string mseT = "";
   if(g_ProgMSE > 0 || g_LastMSE > 0)
     {
      double dm = (g_ProgMSE > 0) ? g_ProgMSE : g_LastMSE;
      mseT = StringFormat("MSE: %.6f -> %.4f", dm, InpTargetMSE);
      double db = (g_ProgBestMSE > 0 && g_ProgBestMSE < 1e9) ? g_ProgBestMSE : g_BestMSE;
      if(db < 1e9) mseT += StringFormat(" | Best: %.6f", db);
     }
   ObjectSetString(0, g_ProgPrefix+"MSE", OBJPROP_TEXT, mseT);

   string lrT = "";
   if(g_ProgLR > 0) lrT = StringFormat("LR: %.6f", g_ProgLR);
   if(g_ProgGradNorm > 0) lrT += (StringLen(lrT)>0?" | ":"") + StringFormat("GradNorm: %.4f", g_ProgGradNorm);
   ObjectSetString(0, g_ProgPrefix+"LR", OBJPROP_TEXT, lrT);

   string ttl = "GPU Training (entropy-gated, shuffled)";
   if(g_IsTraining) ttl += StringSubstr("....", 0, ((int)(GetTickCount()/400))%4);
   else if(prog >= 1.0) ttl = "Training Complete!";
   ObjectSetString(0, g_ProgPrefix+"Title", OBJPROP_TEXT, ttl);
   ObjectSetInteger(0, g_ProgPrefix+"Title", OBJPROP_COLOR, g_IsTraining ? clrYellow : clrLime);
   ChartRedraw();
  }

//+------------------------------------------------------------------+
string FormatDuration(double seconds)
  {
   int s = (int)MathRound(seconds);
   if(s < 0)    return "--:--";
   if(s < 60)   return StringFormat("%ds", s);
   if(s < 3600) return StringFormat("%dm%02ds", s/60, s%60);
   return StringFormat("%dh%02dm", s/3600, (s%3600)/60);
  }

//+------------------------------------------------------------------+
void CleanupObjects()
  {
   ObjectsDeleteAll(0, g_InfoPrefix);
   ObjectsDeleteAll(0, g_ProgPrefix);
  }

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
   return (StringLen(s) == 0) ? "unknown DLL error" : s;
  }

//+------------------------------------------------------------------+
bool SaveModel()
  {
   if(g_NetHandle == 0) return false;
   int sz = DN_SaveState(g_NetHandle);
   if(sz <= 0) return false;
   char buf[];
   ArrayResize(buf, sz);
   if(!DN_GetState(g_NetHandle, buf, sz)) return false;
   int fh = FileOpen(InpModelFile, FILE_WRITE | FILE_BIN | FILE_COMMON);
   if(fh == INVALID_HANDLE) return false;
   FileWriteArray(fh, buf);
   FileClose(fh);
   Print("Model saved (", sz, " bytes)");
   return true;
  }

//+------------------------------------------------------------------+
bool LoadModel()
  {
   if(g_NetHandle == 0) return false;
   int fh = FileOpen(InpModelFile, FILE_READ | FILE_BIN | FILE_COMMON);
   if(fh == INVALID_HANDLE) return false;
   ulong fs = FileSize(fh);
   if(fs == 0 || fs > 100000000) { FileClose(fh); return false; }
   char buf[];
   ArrayResize(buf, (int)fs + 1);
   FileReadArray(fh, buf);
   FileClose(fh);
   buf[(int)fs] = 0;
   if(!DN_LoadState(g_NetHandle, buf)) { Print("LoadState fail: ", GetDLLError()); return false; }
   return true;
  }

//+------------------------------------------------------------------+
/*
═══════════════════════════════════════════════════════════════════════
 CHANGELOG v4.2 → v4.3
═══════════════════════════════════════════════════════════════════════

 HLAVNÍ BUG (v4.2): H=1.000 pořád, NOISE pořád, žádné predikce
 ────────────────────────────────────────────────────────────────────
 Symptom z MT5:
   H=1.000 / 0.75 threshold — NOISE — LSTM OFF
   Posl. 200: 55% structured, 45% noise (avg H=0.746)
   → Protože avg H=0.746 < 0.75, trh JE structured, ale panel říkal NOISE.

 Příčina 1: firstValidEntropyBar = InpEntropyWindow = 20
   Smyčka entropy začínala od i=20.
   Bar[1] (nejnovější uzavřený) zůstal na ArrayInitialize hodnotě 1.0.
   g_CalcStart = 20.

 Příčina 2: podmínka v UpdateInfoPanel:
   bool curInNoise = ... || (1 < g_CalcStart) || ...
   1 < 20 = TRUE vždy → bar[1] byl vždy NOISE bez ohledu na H.

 Příčina 3 (bonus): smooth state při inkrementálním přepočtu:
   g_EntropySmoothState = g_Entropy[calcStart - 1]
   Ale "starší" bar v as-series poli = větší index = calcStart + 1.
   (calcStart - 1 je novější, ne starší!)

 Opravy:
   1. firstValidEntropyBar = 1
      Okno entropie [i..i+window-1] jde do minulosti (větší indexy).
      Pro i=1: okno = [1..20] — validní pokud rates_total > 21.
      Guard v smyčce: if(i + InpEntropyWindow - 1 >= rates_total) skip.

   2. UpdateInfoPanel: odstraněna podmínka (1 < g_CalcStart).
      g_CalcStart = 1 → podmínka by byla 1 < 1 = FALSE (neškodná),
      ale pro srozumitelnost odstraněna úplně.

   3. Smooth state: g_Entropy[calcStart + 1] (starší = větší index).

═══════════════════════════════════════════════════════════════════════
 PŘEHLED VŠECH OPRAV (v4.0 → v4.3)
═══════════════════════════════════════════════════════════════════════
 v4.1: BulkPredict rozsah, g_LastPredictedTo reset, direction accuracy
        podmínka, entropy smooth state, ATR cache v BulkPredict
 v4.2: Komentáře (menší=novější), ArrayInitialize(g_Entropy,1.0),
        guard bar>=g_CalcStart, Wilderovo ATR O(n), Fisher-Yates shuffle
 v4.3: firstValidEntropyBar=1, entropy okno guard, curInNoise oprava,
        smooth state směr indexu
═══════════════════════════════════════════════════════════════════════
*/
//+------------------------------------------------------------------+
