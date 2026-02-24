//+------------------------------------------------------------------+
//| LSTM_RealTimePredictor.mq5                           v2.1        |
//| GPU-optimized real-time LSTM price predictor                     |
//| All NN computation on GPU via MQL5GPULibrary_LSTM.dll            |
//|                                                                  |
//| v2.1 Changes:                                                    |
//|  - Uses new DN_GetProgressAll() for real-time training progress  |
//|  - Accurate ETA from GPU-side timing                             |
//|  - Live MSE, LR, grad norm display during training               |
//|  - Enhanced progress bar with detailed training metrics          |
//+------------------------------------------------------------------+
#property copyright "Tomáš Bělák"
#property link      "https://remind.cz/"
#property description "LSTM Real-Time Predictor (GPU): real-time predikce ceny pomocí LSTM sítě počítané na GPU přes MQL5GPULibrary_LSTM.dll. Zobrazuje predikovanou cenu, horní/dolní pásmo nejistoty a sílu trendu. Podporuje asynchronní trénink na GPU s živým průběhem (MSE, LR, grad norm, ETA) a dávkovou predikci pro vysoký výkon."
#property version   "2.10"
#property strict
#property indicator_chart_window
#property indicator_buffers 8
#property indicator_plots   4

#property indicator_label1  "Predicted Price"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrDodgerBlue
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2

#property indicator_label2  "Upper Band"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrLightSkyBlue
#property indicator_style2  STYLE_DOT
#property indicator_width2  1

#property indicator_label3  "Lower Band"
#property indicator_type3   DRAW_LINE
#property indicator_color3  clrLightSkyBlue
#property indicator_style3  STYLE_DOT
#property indicator_width3  1

#property indicator_label4  "Trend Strength"
#property indicator_type4   DRAW_HISTOGRAM
#property indicator_color4  clrLime,clrRed
#property indicator_style4  STYLE_SOLID
#property indicator_width4  3

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
   int    DN_TrainAsync(int h, int epochs, double target_mse,
                        double lr, double wd);
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

   // Progress monitoring (lock-free, safe during training)
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
input group "=== Architecture ==="
input int      InpLookback        = 30;
input int      InpHiddenSize1     = 96;       // Layer-1 hidden (larger for GPU)
input int      InpHiddenSize2     = 48;       // Layer-2 hidden
input int      InpHiddenSize3     = 0;        // Layer-3 hidden (0 = off)
input int      InpPredictAhead    = 5;
input double   InpDropout         = 0.10;

input group "=== Training (GPU) ==="
input int      InpTrainBars       = 5000;     // Max training bars (fills VRAM)
input int      InpInitialEpochs   = 300;
input int      InpRetrainEpochs   = 80;
input int      InpRetrainInterval = 200;
input double   InpLearningRate    = 0.0008;
input double   InpWeightDecay     = 0.0001;
input double   InpTargetMSE       = 0.005;
input int      InpMiniBatch       = 64;       // GPU mini-batch

input group "=== Prediction (GPU Batch) ==="
input int      InpPredictBatch    = 512;      // Predict this many bars per GPU call
input int      InpMaxPredictBars  = 2000;     // Max bars to predict backwards

input group "=== Display ==="
input bool     InpShowFutureLine  = true;
input bool     InpShowConfidence  = true;
input int      InpFutureBars      = 10;
input color    InpBullColor       = clrLime;
input color    InpBearColor       = clrRed;
input int      InpInfoCorner      = 0;
input int      InpProgressWidth   = 260;
input int      InpProgressHeight  = 18;

input group "=== Advanced ==="
input double   InpGradClip        = 5.0;
input bool     InpAutoRetrain     = true;
input bool     InpSaveModel       = false;
input string   InpModelFile       = "lstm_v2.bin";
input bool     InpVerboseLog      = false;    // Print debug info

//+------------------------------------------------------------------+
//| Feature constants                                                 |
//+------------------------------------------------------------------+
#define FEAT_PER_BAR  16
#define OUTPUT_DIM    3

//+------------------------------------------------------------------+
//| Globals                                                           |
//+------------------------------------------------------------------+
double g_PredictedPrice[];
double g_UpperBand[];
double g_LowerBand[];
double g_TrendStrength[];
double g_TrendColor[];
double g_PredDirection[];
double g_Confidence[];
double g_Magnitude[];

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

// GPU-side progress (updated from DLL) --------------------------------
int    g_ProgEpoch        = 0;
int    g_ProgTotalEpochs  = 0;
int    g_ProgMB           = 0;
int    g_ProgTotalMB      = 0;
int    g_ProgTotalSteps   = 0;   // FIX: was used but never declared
double g_ProgLR           = 0.0;
double g_ProgMSE          = 0.0;
double g_ProgBestMSE      = 0.0;
double g_ProgGradNorm     = 0.0;
double g_ProgPercent      = 0.0;
double g_ProgElapsedSec   = 0.0;
double g_ProgETASec       = 0.0;

// Feature cache -------------------------------------------------------
double g_FeatureCache[];
int    g_CacheStartBar   = -1;
int    g_CachedBars      = 0;
int    g_CacheValidTo    = -1;

// ATR cache -----------------------------------------------------------
double g_ATRCache[];
int    g_ATRCacheStart    = -1;
int    g_ATRCachedBars    = 0;

// Batch prediction bookkeeping ----------------------------------------
int    g_LastPredictedTo  = -1;

// VRAM estimate -------------------------------------------------------
double g_EstVRAM_MB       = 0;

string g_FuturePrefix  = "LSTM_F_";
string g_InfoPrefix    = "LSTM_I_";
string g_ProgPrefix    = "LSTM_P_";

//+------------------------------------------------------------------+
//| Init                                                              |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("=== LSTM Real-Time Predictor v2.1 (GPU + Progress) ===");

   SetIndexBuffer(0, g_PredictedPrice, INDICATOR_DATA);
   SetIndexBuffer(1, g_UpperBand,      INDICATOR_DATA);
   SetIndexBuffer(2, g_LowerBand,      INDICATOR_DATA);
   SetIndexBuffer(3, g_TrendStrength,  INDICATOR_DATA);
   SetIndexBuffer(4, g_TrendColor,     INDICATOR_COLOR_INDEX);
   SetIndexBuffer(5, g_PredDirection,  INDICATOR_CALCULATIONS);
   SetIndexBuffer(6, g_Confidence,     INDICATOR_CALCULATIONS);
   SetIndexBuffer(7, g_Magnitude,      INDICATOR_CALCULATIONS);

   for(int i = 0; i < 4; i++)
      PlotIndexSetDouble(i, PLOT_EMPTY_VALUE, 0.0);

   IndicatorSetString(INDICATOR_SHORTNAME,
      StringFormat("LSTM-GPU(%d,%d,%d)", InpLookback, InpHiddenSize1, InpPredictAhead));

   if(!InitNetwork())
   {
      Print("FATAL: Network init failed");
      return INIT_FAILED;
   }

   if(InpSaveModel && FileIsExist(InpModelFile, FILE_COMMON))
   {
      if(LoadModel()) { Print("Model loaded OK"); g_ModelReady = true; }
   }

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
   if(g_NetHandle > 0) { DN_Free(g_NetHandle); g_NetHandle = 0; }
   CleanupObjects();
}

//+------------------------------------------------------------------+
//| OnCalculate                                                       |
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
   int minBars = InpLookback + InpTrainBars + InpPredictAhead + 50;
   if(rates_total < minBars) return 0;

   g_TotalBars = rates_total;

   //--- Initial training
   if(!g_ModelReady && !g_IsTraining)
   {
      if(StartTraining(rates_total, open, high, low, close, tick_volume, true))
         g_IsTraining = true;
   }

   //--- Auto retrain
   if(g_ModelReady && InpAutoRetrain && !g_IsTraining)
   {
      if(rates_total - g_LastTrainBar >= InpRetrainInterval)
      {
         if(StartTraining(rates_total, open, high, low, close, tick_volume, false))
            g_IsTraining = true;
      }
   }

   //--- Bulk prediction
   if(g_ModelReady && !g_IsTraining)
   {
      BulkPredict(rates_total, prev_calculated, open, high, low, close, tick_volume);
   }
   else
   {
      int start = (prev_calculated == 0) ? 0 : prev_calculated - 1;
      for(int i = start; i < rates_total; i++)
      {
         g_PredictedPrice[i] = close[i];
         g_UpperBand[i]      = 0;
         g_LowerBand[i]      = 0;
         g_TrendStrength[i]  = 0;
         g_TrendColor[i]     = 0;
      }
   }

   //--- Future line
   if(g_ModelReady && InpShowFutureLine && rates_total > 0)
      DrawFuturePrediction(rates_total - 1, time, close, high, low);

   UpdateInfoPanel();
   UpdateProgressBar();
   return rates_total;
}

//+------------------------------------------------------------------+
//| Timer — polls GPU progress                                        |
//+------------------------------------------------------------------+
void OnTimer()
{
   if(!g_IsTraining || g_NetHandle == 0)
   {
      if(g_IsTraining) UpdateProgressBar();
      return;
   }

   //--- Poll GPU-side progress (lock-free, no blocking)
   PollGPUProgress();

   int st = DN_GetTrainingStatus(g_NetHandle);
   if(st == 1) // TS_TRAINING
   {
      UpdateInfoPanel();
      UpdateProgressBar();
      return;
   }

   //--- Training finished
   g_IsTraining = false;
   double mse = 0; int ep = 0;
   DN_GetTrainingResult(g_NetHandle, mse, ep);
   g_LastMSE = mse;
   g_CurrentEpochs = ep;
   g_TotalEpochs += ep;

   if(st == 2) // TS_COMPLETED
   {
      Print(StringFormat("Training done: MSE=%.6f ep=%d elapsed=%.1fs",
            mse, ep, g_ProgElapsedSec));
      if(mse < g_BestMSE)
      {
         g_BestMSE = mse;
         DN_SnapshotWeights(g_NetHandle);
      }
      g_ModelReady = true;
      g_LastTrainBar = g_TotalBars;
      g_LastPredictedTo = -1;
      InvalidateCache();
      if(InpSaveModel) SaveModel();
   }
   else // TS_ERROR
   {
      Print("Training error: ", GetDLLError());
      DN_RestoreWeights(g_NetHandle);
   }
   UpdateInfoPanel();
   UpdateProgressBar();
}

//+------------------------------------------------------------------+
//| Poll GPU progress — single DLL call gets all metrics              |
//+------------------------------------------------------------------+
void PollGPUProgress()
{
   if(g_NetHandle == 0) return;

   if(!DN_GetProgressAll(g_NetHandle,
         g_ProgEpoch, g_ProgTotalEpochs,
         g_ProgMB, g_ProgTotalMB,
         g_ProgLR, g_ProgMSE, g_ProgBestMSE,
         g_ProgGradNorm, g_ProgPercent,
         g_ProgElapsedSec, g_ProgETASec))
   {
      return; // net not found
   }

   // FIX: keep steps in a declared global
   g_ProgTotalSteps = DN_GetProgressTotalSteps(g_NetHandle);

   // Sync local tracking variables
   g_CurrentEpochs = g_ProgEpoch;
   if(g_ProgMSE > 0) g_LastMSE = g_ProgMSE;
   if(g_ProgBestMSE < g_BestMSE && g_ProgBestMSE > 0)
      g_BestMSE = g_ProgBestMSE;
}

//+------------------------------------------------------------------+
//| Init network                                                      |
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
   {
      if(!DN_AddLayerEx(g_NetHandle, 0, InpHiddenSize3, 0, 0, 0.0))
      { Print("L3 fail: ", GetDLLError()); DN_Free(g_NetHandle); g_NetHandle = 0; return false; }
   }

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
//| Estimate VRAM usage                                               |
//+------------------------------------------------------------------+
void EstimateVRAM()
{
   int H1 = InpHiddenSize1, H2 = InpHiddenSize2, H3 = InpHiddenSize3;
   int F  = FEAT_PER_BAR;
   int S  = InpLookback;
   int N  = InpTrainBars;
   int MB = InpMiniBatch;

   double w1 = (double)(F + H1) * 4 * H1 * 4;
   double w2 = (double)(H1 + H2) * 4 * H2 * 4;
   double w3 = (H3 > 0) ? (double)(H2 + H3) * 4 * H3 * 4 : 0;
   int lastH = (H3 > 0) ? H3 : H2;
   double wo = (double)lastH * OUTPUT_DIM * 4;

   double weightMem = (w1 + w2 + w3 + wo) * 3;

   double dataMem = (double)N * S * F * 4.0 + (double)N * OUTPUT_DIM * 4.0;

   double cacheMem = (double)S * MB * H1 * 4.0 * 7;
   cacheMem += (double)S * MB * H2 * 4.0 * 7;
   if(H3 > 0) cacheMem += (double)S * MB * H3 * 4.0 * 7;

   double gradMem = cacheMem * 0.5;

   double predMem = (double)InpPredictBatch * S * F * 8.0;
   predMem += (double)InpPredictBatch * OUTPUT_DIM * 8.0;

   g_EstVRAM_MB = (weightMem + dataMem + cacheMem + gradMem + predMem) / (1024.0 * 1024.0);

   Print(StringFormat("Est. VRAM: %.1f MB (W:%.1f D:%.1f C:%.1f G:%.1f P:%.1f)",
         g_EstVRAM_MB,
         weightMem / 1048576.0,
         dataMem / 1048576.0,
         cacheMem / 1048576.0,
         gradMem / 1048576.0,
         predMem / 1048576.0));
}

//+------------------------------------------------------------------+
//| Invalidate feature cache                                          |
//+------------------------------------------------------------------+
void InvalidateCache()
{
   g_CacheStartBar = -1;
   g_CachedBars    = 0;
   g_CacheValidTo  = -1;
   g_ATRCacheStart = -1;
   g_ATRCachedBars = 0;
}

//+------------------------------------------------------------------+
//| Bulk ATR computation                                              |
//+------------------------------------------------------------------+
void BulkComputeATR(int startBar, int count, int period,
                    const double &high[],
                    const double &low[],
                    const double &close[],
                    int rates_total)
{
   if(g_ATRCacheStart == startBar && g_ATRCachedBars >= count)
      return;

   ArrayResize(g_ATRCache, count);
   g_ATRCacheStart = startBar;
   g_ATRCachedBars = count;

   for(int i = 0; i < count; i++)
   {
      int bar = startBar + i;
      double sum = 0;
      int n = 0;
      for(int k = 0; k < period && bar + k < rates_total - 1; k++)
      {
         double h  = high[bar + k];
         double l  = low[bar + k];
         double pc = close[bar + k + 1];
         double tr = MathMax(h - l, MathMax(MathAbs(h - pc), MathAbs(l - pc)));
         sum += tr;
         n++;
      }
      g_ATRCache[i] = (n > 0) ? sum / n : _Point * 100;
   }
}

//+------------------------------------------------------------------+
//| Get cached ATR                                                    |
//+------------------------------------------------------------------+
double GetCachedATR(int barIdx)
{
   if(g_ATRCacheStart < 0) return _Point * 100;
   int idx = barIdx - g_ATRCacheStart;
   if(idx < 0 || idx >= g_ATRCachedBars) return _Point * 100;
   return g_ATRCache[idx];
}

//+------------------------------------------------------------------+
//| Bulk feature extraction                                           |
//+------------------------------------------------------------------+
void BulkExtractFeatures(int startBar, int count,
                         const double &open[],
                         const double &high[],
                         const double &low[],
                         const double &close[],
                         const long &tick_volume[],
                         int rates_total,
                         double &feats[])
{
   ArrayResize(feats, count * FEAT_PER_BAR);

   int atrStart = startBar;
   int atrCount = count + 30;
   if(atrStart + atrCount > rates_total)
      atrCount = rates_total - atrStart;
   BulkComputeATR(atrStart, atrCount, 14, high, low, close, rates_total);

   double volMAScratch[];
   ArrayResize(volMAScratch, count);
   for(int i = 0; i < count; i++)
   {
      int bar = startBar + i;
      double s = 0;
      int n = 0;
      for(int k = 0; k < 20 && bar + k < rates_total; k++)
      {
         s += (double)tick_volume[bar + k];
         n++;
      }
      volMAScratch[i] = (n > 0) ? s / n : 1.0;
   }

   for(int i = 0; i < count; i++)
   {
      int bar = startBar + i;
      int off = i * FEAT_PER_BAR;

      if(bar < 1 || bar >= rates_total - 1)
      {
         for(int k = 0; k < FEAT_PER_BAR; k++)
            feats[off + k] = 0;
         continue;
      }

      double o = open[bar];
      double h = high[bar];
      double l = low[bar];
      double c = close[bar];
      double prevC = close[bar + 1];
      double vol = (double)tick_volume[bar];
      double prevVol = (double)tick_volume[bar + 1];

      double range = h - l;
      if(range < _Point) range = _Point;
      if(prevC < _Point) prevC = c;
      if(prevVol < 1) prevVol = vol;

      double atr = GetCachedATR(bar);
      if(atr < _Point) atr = _Point;

      double avgATR = (g_ATRMean > _Point) ? g_ATRMean : atr;

      int fi = off;

      feats[fi++] = (o - prevC) / atr;
      feats[fi++] = (h - prevC) / atr;
      feats[fi++] = (l - prevC) / atr;
      feats[fi++] = (c - prevC) / atr;

      feats[fi++] = (c - prevC) / atr;
      feats[fi++] = (h - prevC) / atr;
      feats[fi++] = (l - prevC) / atr;

      feats[fi++] = atr / avgATR - 1.0;
      feats[fi++] = range / atr;

      double gain = 0, loss = 0;
      for(int k = 0; k < 14 && bar + k < rates_total - 1; k++)
      {
         double change = close[bar + k] - close[bar + k + 1];
         if(change > 0) gain += change;
         else loss -= change;
      }
      double rs = (loss > 0) ? gain / loss : 1.0;
      feats[fi++] = (100.0 - 100.0 / (1.0 + rs) - 50.0) / 50.0;

      double emaF = 0, emaS = 0;
      double aF = 2.0 / 13.0, aS = 2.0 / 27.0;
      for(int k = 0; k < 26 && bar + k < rates_total; k++)
      {
         if(k < 12) emaF += close[bar + k] * MathPow(1.0 - aF, k);
         emaS += close[bar + k] * MathPow(1.0 - aS, k);
      }
      emaF *= aF;
      emaS *= aS;
      feats[fi++] = (emaF - emaS) / atr;

      double hh = h, ll = l;
      for(int k = 0; k < 14 && bar + k < rates_total; k++)
      {
         if(high[bar + k] > hh) hh = high[bar + k];
         if(low[bar + k] < ll) ll = low[bar + k];
      }
      double stR = hh - ll;
      double stoch = (stR > _Point) ? (c - ll) / stR : 0.5;
      feats[fi++] = (stoch - 0.5) * 2.0;

      double vc = (prevVol > 0) ? (vol - prevVol) / prevVol : 0;
      feats[fi++] = MathMax(-3.0, MathMin(3.0, vc)) / 3.0;

      double vma = volMAScratch[i];
      double vr = (vma > 0) ? vol / vma : 1.0;
      feats[fi++] = MathMax(-2.0, MathMin(2.0, vr - 1.0));

      double body = c - o;
      double uShadow = h - MathMax(o, c);
      double lShadow = MathMin(o, c) - l;
      feats[fi++] = body / range;
      feats[fi++] = (uShadow - lShadow) / range;
   }

   g_CacheStartBar = startBar;
   g_CachedBars    = count;
   g_CacheValidTo  = startBar;
}

//+------------------------------------------------------------------+
//| Build sequence input                                              |
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
      int tgtBar = sampleBars[s];

      for(int t = 0; t < seqLen; t++)
      {
         int bar = tgtBar + (seqLen - 1 - t);
         int cacheIdx = bar - featureStartBar;

         int xOff = s * inDim + t * FEAT_PER_BAR;

         if(cacheIdx >= 0 && cacheIdx < totalFeatBars)
         {
            int fOff = cacheIdx * FEAT_PER_BAR;
            for(int f = 0; f < FEAT_PER_BAR; f++)
               X[xOff + f] = barFeatures[fOff + f];
         }
         else
         {
            for(int f = 0; f < FEAT_PER_BAR; f++)
               X[xOff + f] = 0;
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Compute normalization params                                      |
//+------------------------------------------------------------------+
void ComputeNormParams(int rates_total,
                       const double &close[],
                       const double &high[],
                       const double &low[])
{
   int n = MathMin(InpTrainBars, rates_total - InpLookback - 10);
   double sumATR = 0;
   for(int i = 0; i < n; i++)
   {
      int bar = i + 1;
      if(bar >= rates_total - 1) break;
      double sum = 0; int cnt = 0;
      for(int k = 0; k < 14 && bar + k < rates_total - 1; k++)
      {
         double h  = high[bar + k];
         double l  = low[bar + k];
         double pc = close[bar + k + 1];
         sum += MathMax(h - l, MathMax(MathAbs(h - pc), MathAbs(l - pc)));
         cnt++;
      }
      sumATR += (cnt > 0 ? sum / cnt : _Point);
   }
   g_ATRMean = (n > 0) ? sumATR / n : _Point * 100;
   if(g_ATRMean < _Point) g_ATRMean = _Point;
}

//+------------------------------------------------------------------+
//| Start training (GPU)                                              |
//+------------------------------------------------------------------+
bool StartTraining(int rates_total,
                   const double &open[],
                   const double &high[],
                   const double &low[],
                   const double &close[],
                   const long &tick_volume[],
                   bool initial)
{
   if(g_NetHandle == 0) return false;

   g_TargetEpochs   = initial ? InpInitialEpochs : InpRetrainEpochs;
   g_CurrentEpochs  = 0;
   g_TrainStartTime = TimeCurrent();

   // Reset GPU progress tracking
   g_ProgEpoch       = 0;
   g_ProgTotalEpochs = g_TargetEpochs;
   g_ProgMB          = 0;
   g_ProgTotalMB     = 0;
   g_ProgTotalSteps  = 0; // FIX: reset steps too
   g_ProgLR          = InpLearningRate;
   g_ProgMSE         = 0;
   g_ProgBestMSE     = 0;
   g_ProgGradNorm    = 0;
   g_ProgPercent     = 0;
   g_ProgElapsedSec  = 0;
   g_ProgETASec      = 0;

   ComputeNormParams(rates_total, close, high, low);

   int maxSamples = MathMin(InpTrainBars, rates_total - InpLookback - InpPredictAhead - 5);
   if(maxSamples < 50) { Print("Too few samples: ", maxSamples); return false; }

   int inDim = InpLookback * FEAT_PER_BAR;

   int firstTarget = 1 + InpPredictAhead;
   int lastTarget  = maxSamples + InpPredictAhead;
   int oldestBar   = lastTarget + InpLookback;
   int newestBar   = firstTarget;

   if(oldestBar >= rates_total - 1) oldestBar = rates_total - 2;

   int featRange = oldestBar - newestBar + 1;

   uint t0 = GetTickCount();

   double barFeats[];
   BulkExtractFeatures(newestBar, featRange, open, high, low, close,
                       tick_volume, rates_total, barFeats);

   if(InpVerboseLog)
      Print("Feature extraction: ", featRange, " bars in ",
            GetTickCount() - t0, " ms");

   int sampleBars[];
   ArrayResize(sampleBars, maxSamples);
   for(int s = 0; s < maxSamples; s++)
      sampleBars[s] = s + firstTarget;

   double X[];
   BuildSequenceInput(barFeats, newestBar, featRange,
                      sampleBars, maxSamples, InpLookback, X);

   double T[];
   ArrayResize(T, maxSamples * OUTPUT_DIM);
   ArrayInitialize(T, 0);

   for(int s = 0; s < maxSamples; s++)
   {
      int predBar = s + 1;
      int tgtBar  = predBar + InpPredictAhead;

      if(tgtBar >= rates_total - 1 || predBar >= rates_total) continue;

      double pc = close[tgtBar];
      double cc = close[predBar];
      if(pc < _Point) pc = _Point;

      double ret = (cc - pc) / pc;
      double atr = GetCachedATR(tgtBar);
      if(atr < _Point) atr = _Point;
      double normRet = ret / (atr / pc);

      T[s * OUTPUT_DIM + 0] = MathTanh(normRet * 2.0);
      T[s * OUTPUT_DIM + 1] = 1.0 / (1.0 + MathExp(-MathAbs(normRet)));

      double volRatio = atr / g_ATRMean;
      T[s * OUTPUT_DIM + 2] = 1.0 / (1.0 + volRatio);
   }

   uint t1 = GetTickCount();
   if(InpVerboseLog)
      Print("Data prep total: ", t1 - t0, " ms for ", maxSamples, " samples");

   if(!DN_LoadBatch(g_NetHandle, X, T, maxSamples, inDim, OUTPUT_DIM, 0))
   {
      Print("LoadBatch fail: ", GetDLLError());
      return false;
   }

   uint t2 = GetTickCount();
   double transferMB = ((double)maxSamples * inDim * 8.0 +
                        (double)maxSamples * OUTPUT_DIM * 8.0) / 1048576.0;
   if(InpVerboseLog)
      Print(StringFormat("GPU transfer: %.1f MB in %d ms", transferMB, t2 - t1));

   if(!DN_TrainAsync(g_NetHandle, g_TargetEpochs, InpTargetMSE,
                     InpLearningRate, InpWeightDecay))
   {
      Print("TrainAsync fail: ", GetDLLError());
      return false;
   }

   Print(StringFormat("Training started: %d samples x %d epochs (%.1f MB on GPU)",
         maxSamples, g_TargetEpochs, transferMB));

   UpdateProgressBar();
   return true;
}

//+------------------------------------------------------------------+
//| Bulk GPU-batched prediction                                       |
//+------------------------------------------------------------------+
void BulkPredict(int rates_total,
                 int prev_calculated,
                 const double &open[],
                 const double &high[],
                 const double &low[],
                 const double &close[],
                 const long &tick_volume[])
{
   if(!g_ModelReady || g_NetHandle == 0 || g_IsTraining) return;

   int newestBar = InpLookback;
   int oldestBar = rates_total - 1;

   int startPred = newestBar;
   if(prev_calculated > 0 && g_LastPredictedTo >= newestBar)
   {
      startPred = MathMax(newestBar, rates_total - (rates_total - prev_calculated) - 1);
   }

   int maxPred = MathMin(InpMaxPredictBars, rates_total - InpLookback);
   if(oldestBar - startPred > maxPred)
      startPred = oldestBar - maxPred;

   int totalPredict = oldestBar - startPred + 1;
   if(totalPredict <= 0) return;

   if(g_ATRMean < _Point)
      ComputeNormParams(rates_total, close, high, low);

   int featNewest = startPred;
   int featOldest = oldestBar + InpLookback;
   if(featOldest >= rates_total) featOldest = rates_total - 1;
   int featRange = featOldest - featNewest + 1;

   uint t0 = GetTickCount();

   double barFeats[];
   BulkExtractFeatures(featNewest, featRange, open, high, low, close,
                       tick_volume, rates_total, barFeats);

   if(InpVerboseLog)
      Print("Predict: feature extraction ", featRange, " bars in ",
            GetTickCount() - t0, " ms");

   int batchSize = InpPredictBatch;
   int inDim     = InpLookback * FEAT_PER_BAR;

   int processed = 0;
   uint tPred = GetTickCount();

   for(int batchStart = 0; batchStart < totalPredict; batchStart += batchSize)
   {
      int curBatch = MathMin(batchSize, totalPredict - batchStart);

      int sampleBars[];
      ArrayResize(sampleBars, curBatch);
      for(int i = 0; i < curBatch; i++)
         sampleBars[i] = startPred + batchStart + i;

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
         if(barIdx < 0 || barIdx >= rates_total) continue;

         double predDir  = MathMax(-1.0, MathMin(1.0,  Y[i * OUTPUT_DIM + 0]));
         double predMag  = MathMax(0.0,  MathMin(2.0,  Y[i * OUTPUT_DIM + 1]));
         double predConf = MathMax(0.0,  MathMin(1.0,  Y[i * OUTPUT_DIM + 2]));

         g_PredDirection[barIdx] = predDir;
         g_Magnitude[barIdx]     = predMag;
         g_Confidence[barIdx]    = predConf;

         double atr = GetCachedATR(barIdx);
         double move = predDir * predMag * atr;
         g_PredictedPrice[barIdx] = close[barIdx] + move;

         double bw = atr * (1.0 - predConf) * 2.0;
         g_UpperBand[barIdx] = g_PredictedPrice[barIdx] + bw;
         g_LowerBand[barIdx] = g_PredictedPrice[barIdx] - bw;

         g_TrendStrength[barIdx] = predDir * predMag * predConf * 100;
         g_TrendColor[barIdx]    = (predDir > 0) ? 0 : 1;

         if(barIdx + InpPredictAhead < rates_total &&
            MathAbs(g_PredDirection[barIdx]) >= 0.1)
         {
            int evalBar = barIdx - InpPredictAhead;
            if(evalBar >= 0 && evalBar < rates_total)
            {
               double actualMove = close[evalBar] - close[barIdx];
               bool correct = (g_PredDirection[barIdx] > 0 && actualMove > 0) ||
                              (g_PredDirection[barIdx] < 0 && actualMove < 0);
               g_TotalPred++;
               if(correct) g_CorrectPred++;
            }
         }
      }

      processed += curBatch;
   }

   g_LastPredictedTo = startPred;

   if(InpVerboseLog)
      Print(StringFormat("Predicted %d bars in %d ms (%.0f bars/sec)",
            processed, GetTickCount() - tPred,
            processed * 1000.0 / MathMax(1, GetTickCount() - tPred)));
}

//+------------------------------------------------------------------+
//| Draw future prediction                                            |
//+------------------------------------------------------------------+
void DrawFuturePrediction(int lastBar,
                          const datetime &time[],
                          const double &close[],
                          const double &high[],
                          const double &low[])
{
   ObjectsDeleteAll(0, g_FuturePrefix);
   if(!InpShowFutureLine || !g_ModelReady) return;

   double pDir  = g_PredDirection[lastBar];
   double pMag  = g_Magnitude[lastBar];
   double pConf = g_Confidence[lastBar];
   if(MathAbs(pDir) < 0.05) return;

   double curPrice = close[lastBar];
   double atr = GetCachedATR(lastBar);

   datetime curTime = time[lastBar];
   int pSec = PeriodSeconds();

   double prices[];
   datetime times[];
   ArrayResize(prices, InpFutureBars + 1);
   ArrayResize(times,  InpFutureBars + 1);

   prices[0] = curPrice;
   times[0]  = curTime;

   for(int i = 1; i <= InpFutureBars; i++)
   {
      double decay = MathExp(-0.1 * i);
      prices[i] = prices[i-1] + pDir * pMag * atr * decay / InpPredictAhead;
      times[i]  = curTime + i * pSec;
   }

   color lc = (pDir > 0) ? InpBullColor : InpBearColor;

   for(int i = 0; i < InpFutureBars; i++)
   {
      string nm = g_FuturePrefix + "L" + IntegerToString(i);
      ObjectCreate(0, nm, OBJ_TREND, 0, times[i], prices[i], times[i+1], prices[i+1]);
      ObjectSetInteger(0, nm, OBJPROP_COLOR, lc);
      ObjectSetInteger(0, nm, OBJPROP_WIDTH, 2);
      ObjectSetInteger(0, nm, OBJPROP_RAY_RIGHT, false);
      ObjectSetInteger(0, nm, OBJPROP_SELECTABLE, false);
   }

   if(InpShowConfidence)
   {
      double bw = atr * (1.0 - pConf) * 2.0;
      for(int i = 0; i < InpFutureBars; i++)
      {
         string nu = g_FuturePrefix + "U" + IntegerToString(i);
         string nd = g_FuturePrefix + "D" + IntegerToString(i);
         ObjectCreate(0, nu, OBJ_TREND, 0,
                      times[i], prices[i]+bw, times[i+1], prices[i+1]+bw);
         ObjectCreate(0, nd, OBJ_TREND, 0,
                      times[i], prices[i]-bw, times[i+1], prices[i+1]-bw);
         ObjectSetInteger(0, nu, OBJPROP_COLOR, lc);
         ObjectSetInteger(0, nu, OBJPROP_STYLE, STYLE_DOT);
         ObjectSetInteger(0, nu, OBJPROP_SELECTABLE, false);
         ObjectSetInteger(0, nd, OBJPROP_COLOR, lc);
         ObjectSetInteger(0, nd, OBJPROP_STYLE, STYLE_DOT);
         ObjectSetInteger(0, nd, OBJPROP_SELECTABLE, false);
      }
   }

   string an = g_FuturePrefix + "Arr";
   ENUM_OBJECT at = (pDir > 0) ? OBJ_ARROW_UP : OBJ_ARROW_DOWN;
   ObjectCreate(0, an, at, 0, times[InpFutureBars], prices[InpFutureBars]);
   ObjectSetInteger(0, an, OBJPROP_COLOR, lc);
   ObjectSetInteger(0, an, OBJPROP_WIDTH, 3);
   ObjectSetInteger(0, an, OBJPROP_SELECTABLE, false);

   string ln = g_FuturePrefix + "PL";
   ObjectCreate(0, ln, OBJ_TEXT, 0, times[InpFutureBars], prices[InpFutureBars]);
   ObjectSetString(0, ln, OBJPROP_TEXT,
      StringFormat("%.5f (%.0f%%)", prices[InpFutureBars], pConf * 100));
   ObjectSetInteger(0, ln, OBJPROP_COLOR, lc);
   ObjectSetInteger(0, ln, OBJPROP_FONTSIZE, 9);
   ObjectSetInteger(0, ln, OBJPROP_ANCHOR, ANCHOR_LEFT);
}

//+------------------------------------------------------------------+
//| Format seconds to readable string                                 |
//+------------------------------------------------------------------+
string FormatDuration(double seconds)
{
   int s = (int)MathRound(seconds);
   if(s < 0) return "--:--";
   if(s < 60)   return StringFormat("%ds", s);
   if(s < 3600) return StringFormat("%dm%02ds", s / 60, s % 60);
   return StringFormat("%dh%02dm", s / 3600, (s % 3600) / 60);
}

//+------------------------------------------------------------------+
//| Progress bar                                                      |
//+------------------------------------------------------------------+
void CreateProgressBar()
{
   int yOff = 235;

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

   // Extra line: LR + GradNorm
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

void UpdateProgressBar()
{
   bool show = g_IsTraining ||
               (g_TrainStartTime > 0 && TimeCurrent() - g_TrainStartTime < 5);

   string names[] = { "BG","Fill","Txt","Det","MSE","LR","Title" };
   for(int i = 0; i < ArraySize(names); i++)
   {
      string n = g_ProgPrefix + names[i];
      ObjectSetInteger(0, n, OBJPROP_TIMEFRAMES,
                       show ? OBJ_ALL_PERIODS : OBJ_NO_PERIODS);
   }

   if(!show) { ChartRedraw(); return; }

   //--- Use GPU-side progress (accurate percentage from DLL)
   double prog = g_ProgPercent / 100.0;
   prog = MathMax(0.0, MathMin(1.0, prog));

   // Fallback if GPU hasn't reported yet
   if(prog < 0.001 && g_TargetEpochs > 0 && g_CurrentEpochs > 0)
      prog = MathMin(1.0, (double)g_CurrentEpochs / g_TargetEpochs);

   color fc;
   if(prog < 0.33)      fc = clrOrangeRed;
   else if(prog < 0.66) fc = clrGold;
   else if(prog < 1.0)  fc = clrDodgerBlue;
   else                 fc = clrLime;

   int fw = (int)(InpProgressWidth * prog);
   ObjectSetInteger(0, g_ProgPrefix + "Fill", OBJPROP_XSIZE, MathMax(0, fw));
   ObjectSetInteger(0, g_ProgPrefix + "Fill", OBJPROP_BGCOLOR, fc);

   ObjectSetString(0, g_ProgPrefix + "Txt", OBJPROP_TEXT,
                   StringFormat("%.1f%%", prog * 100));

   //--- Detail line: epoch + minibatch + ETA (from GPU-side timing)
   string det;
   if(g_ProgTotalEpochs > 0)
   {
      det = StringFormat("Ep %d/%d", g_ProgEpoch, g_ProgTotalEpochs);

      if(g_ProgTotalMB > 0)
         det += StringFormat(" | MB %d/%d", g_ProgMB, g_ProgTotalMB);

      // ETA from GPU-side (much more accurate than MQL5 datetime)
      if(g_ProgETASec > 0)
         det += " | ETA " + FormatDuration(g_ProgETASec);

      if(g_ProgElapsedSec > 0)
         det += " | " + FormatDuration(g_ProgElapsedSec);
   }
   else
   {
      det = StringFormat("Epoch %d / %d", g_CurrentEpochs, g_TargetEpochs);
   }
   ObjectSetString(0, g_ProgPrefix + "Det", OBJPROP_TEXT, det);

   //--- MSE line
   string mseT = "";
   if(g_ProgMSE > 0 || g_LastMSE > 0)
   {
      double dispMSE = (g_ProgMSE > 0) ? g_ProgMSE : g_LastMSE;
      mseT = StringFormat("MSE: %.6f", dispMSE);
      if(InpTargetMSE > 0)
         mseT += StringFormat(" -> %.4f (%.0f%%)",
                  InpTargetMSE, MathMin(100.0, InpTargetMSE / MathMax(1e-12, dispMSE) * 100));

      double dispBest = (g_ProgBestMSE > 0 && g_ProgBestMSE < 1e10) ?
                         g_ProgBestMSE : g_BestMSE;
      if(dispBest < 1e9)
         mseT += StringFormat(" | Best: %.6f", dispBest);
   }
   ObjectSetString(0, g_ProgPrefix + "MSE", OBJPROP_TEXT, mseT);

   //--- LR + Grad norm line
   string lrT = "";
   if(g_ProgLR > 0)
      lrT = StringFormat("LR: %.6f", g_ProgLR);
   if(g_ProgGradNorm > 0)
   {
      if(StringLen(lrT) > 0) lrT += " | ";
      lrT += StringFormat("GradNorm: %.4f", g_ProgGradNorm);
   }
   if(g_ProgTotalSteps > 0)
   {
      if(StringLen(lrT) > 0) lrT += " | ";
      lrT += StringFormat("Step: %d", g_ProgTotalSteps); // FIX: use cached steps
   }
   ObjectSetString(0, g_ProgPrefix + "LR", OBJPROP_TEXT, lrT);

   //--- Title
   string ttl = "GPU Training";
   if(g_IsTraining)
   {
      int d = ((int)(GetTickCount() / 400)) % 4;
      ttl += StringSubstr("....", 0, d);
   }
   else if(prog >= 1.0)
      ttl = "Training Complete!";

   ObjectSetString(0, g_ProgPrefix + "Title", OBJPROP_TEXT, ttl);
   ObjectSetInteger(0, g_ProgPrefix + "Title", OBJPROP_COLOR,
                    g_IsTraining ? clrYellow : clrLime);

   ChartRedraw();
}

//+------------------------------------------------------------------+
//| Info panel                                                        |
//+------------------------------------------------------------------+
void CreateInfoPanel()
{
   string bg = g_InfoPrefix + "BG";
   ObjectCreate(0, bg, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, bg, OBJPROP_CORNER, InpInfoCorner);
   ObjectSetInteger(0, bg, OBJPROP_XDISTANCE, 10);
   ObjectSetInteger(0, bg, OBJPROP_YDISTANCE, 30);
   ObjectSetInteger(0, bg, OBJPROP_XSIZE, 290);
   ObjectSetInteger(0, bg, OBJPROP_YSIZE, 195);
   ObjectSetInteger(0, bg, OBJPROP_BGCOLOR, C'30,30,40');
   ObjectSetInteger(0, bg, OBJPROP_BORDER_COLOR, clrDodgerBlue);
   ObjectSetInteger(0, bg, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(0, bg, OBJPROP_WIDTH, 2);
   ObjectSetInteger(0, bg, OBJPROP_BACK, false);
   ObjectSetInteger(0, bg, OBJPROP_SELECTABLE, false);

   MakeLabel(g_InfoPrefix + "T", "LSTM-GPU Predictor v2.1",
             15, 35, clrDodgerBlue, 11, true);
}

void UpdateInfoPanel()
{
   string st;
   color sc;
   if(g_IsTraining) { st = "GPU Training..."; sc = clrYellow; }
   else if(g_ModelReady) { st = "Ready"; sc = clrLime; }
   else { st = "Waiting"; sc = clrGray; }

   MakeLabel(g_InfoPrefix + "St", "Status: " + st, 15, 55, sc, 9, false);

   string arch = StringFormat("LSTM(%d->%d->%d", FEAT_PER_BAR, InpHiddenSize1, InpHiddenSize2);
   if(InpHiddenSize3 > 0) arch += "->" + IntegerToString(InpHiddenSize3);
   arch += ")->" + IntegerToString(OUTPUT_DIM);
   MakeLabel(g_InfoPrefix + "Ar", arch, 15, 72, clrWhite, 9, false);

   // Show live MSE during training from GPU
   double dispMSE = g_LastMSE;
   double dispBest = g_BestMSE;
   if(g_IsTraining)
   {
      if(g_ProgMSE > 0) dispMSE = g_ProgMSE;
      if(g_ProgBestMSE > 0 && g_ProgBestMSE < 1e10) dispBest = g_ProgBestMSE;
   }
   MakeLabel(g_InfoPrefix + "MSE",
      StringFormat("MSE: %.6f (best: %.6f)", dispMSE, dispBest),
      15, 89, clrSilver, 9, false);

   MakeLabel(g_InfoPrefix + "Ep",
      StringFormat("Epochs: %d | VRAM: ~%.0f MB", g_TotalEpochs, g_EstVRAM_MB),
      15, 106, clrSilver, 9, false);

   double acc = (g_TotalPred > 0) ? (double)g_CorrectPred / g_TotalPred * 100 : 0;
   color ac = (acc > 55) ? clrLime : (acc > 50) ? clrYellow : clrOrangeRed;
   MakeLabel(g_InfoPrefix + "Acc",
      StringFormat("Accuracy: %.1f%% (%d/%d)", acc, g_CorrectPred, g_TotalPred),
      15, 123, ac, 9, false);

   int lb = g_TotalBars - 1;
   if(lb >= 0 && g_ModelReady)
   {
      double dir = g_PredDirection[lb];
      double conf = g_Confidence[lb];
      double mag = g_Magnitude[lb];
      string pt; color pc;

      if(MathAbs(dir) < 0.1) { pt = "NEUTRAL"; pc = clrGray; }
      else if(dir > 0) { pt = StringFormat("BULLISH (%.0f%%)", conf * 100); pc = clrLime; }
      else { pt = StringFormat("BEARISH (%.0f%%)", conf * 100); pc = clrOrangeRed; }

      MakeLabel(g_InfoPrefix + "Pr",
         StringFormat("Next %d bars: %s", InpPredictAhead, pt),
         15, 147, pc, 10, true);

      MakeLabel(g_InfoPrefix + "Sg",
         StringFormat("Dir: %.3f | Mag: %.3f | Conf: %.3f", dir, mag, conf),
         15, 167, clrSilver, 9, false);

      MakeLabel(g_InfoPrefix + "GP",
         StringFormat("Batch: %d | Train: %d bars",
                      InpPredictBatch, InpTrainBars),
         15, 184, clrDarkGray, 8, false);
   }
   else
   {
      MakeLabel(g_InfoPrefix + "Pr", "Awaiting model...", 15, 147, clrGray, 9, false);
      MakeLabel(g_InfoPrefix + "Sg", "", 15, 167, clrGray, 9, false);
      MakeLabel(g_InfoPrefix + "GP", "", 15, 184, clrGray, 8, false);
   }

   ChartRedraw();
}

//+------------------------------------------------------------------+
//| Label helper                                                      |
//+------------------------------------------------------------------+
void MakeLabel(string name, string text, int x, int y,
               color clr, int sz, bool bold)
{
   if(ObjectFind(0, name) < 0)
   {
      ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, name, OBJPROP_CORNER, InpInfoCorner);
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   }
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, sz);
   ObjectSetString(0, name, OBJPROP_FONT, bold ? "Arial Bold" : "Arial");
   ObjectSetString(0, name, OBJPROP_TEXT, text);
}

//+------------------------------------------------------------------+
//| Cleanup                                                           |
//+------------------------------------------------------------------+
void CleanupObjects()
{
   ObjectsDeleteAll(0, g_FuturePrefix);
   ObjectsDeleteAll(0, g_InfoPrefix);
   ObjectsDeleteAll(0, g_ProgPrefix);
}

//+------------------------------------------------------------------+
//| DLL error                                                         |
//+------------------------------------------------------------------+
string GetDLLError()
{
   short buf[];
   ArrayResize(buf, 512);
   DN_GetError(buf, 512);
   string s = "";
   for(int i = 0; i < 512 && buf[i] != 0; i++)
      s += ShortToString(buf[i]);
   return s;
}

//+------------------------------------------------------------------+
//| Save / Load model                                                 |
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
   if(!DN_LoadState(g_NetHandle, buf))
   { Print("LoadState fail: ", GetDLLError()); return false; }
   return true;
}
//+------------------------------------------------------------------+
