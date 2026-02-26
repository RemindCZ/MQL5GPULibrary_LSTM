//+------------------------------------------------------------------+
//| LSTM_PatternCompletion_Demo.mq5                                  |
//| Demonstrates pattern-completion training using candle features.   |
//| Builds a window of SeqLen bars (series arrays: 0=current forming) |
//| Trains small LSTM via MQL5GPULibrary_LSTM.dll and predicts        |
//| bullish/bearish completion scores in [0..1].                      |
//|                                                                  |
//| IMPORTANT INDEXING (MQL5 series arrays):                          |
//|   index 0 = current forming bar                                   |
//|   index 1 = last closed bar                                       |
//|   index increases into the past                                   |
//|                                                                  |
//| This demo:                                                       |
//|   - trains from past windows that END at bar_end >= 1+PredK       |
//|   - target looks "forward" K bars (towards the present)           |
//|     meaning indices decrease: bar_end-1, bar_end-2, ...           |
//+------------------------------------------------------------------+
#property strict
#property indicator_separate_window
#property indicator_buffers 2
#property indicator_plots   2
#property indicator_minimum 0.0
#property indicator_maximum 1.0

#property indicator_label1  "BullishScore"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrLime
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2

#property indicator_label2  "BearishScore"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrTomato
#property indicator_style2  STYLE_SOLID
#property indicator_width2  2

#import "MQL5GPULibrary_LSTM.dll"
   int    DN_Create();
   void   DN_Free(int h);
   int    DN_SetSequenceLength(int h, int seq_len);
   int    DN_SetMiniBatchSize(int h, int mbs);
   int    DN_AddLayerEx(int h, int in_sz, int out_sz, int act, int ln, double drop);
   int    DN_SetOutputDim(int h, int out_dim);

   int    DN_LoadBatch(int h, const double &X[], const double &T[],
                       int batch, int in_dim, int out_dim, int layout);

   int    DN_TrainAsync(int h, int epochs, double target_mse,
                        double lr, double wd);
   int    DN_GetTrainingStatus(int h);
   void   DN_StopTraining(int h);

   int    DN_GetProgressAll(int h,
             int &epoch, int &total_epochs,
             int &mb, int &total_mb,
             double &lr, double &mse, double &best_mse,
             double &grad_norm, double &pct,
             double &elapsed_sec, double &eta_sec);

   int    DN_PredictBatch(int h, const double &X[], int batch,
                          int in_dim, int layout, double &Y[]);

   void   DN_GetError(short &buf[], int len);
#import

// -------------------- Inputs --------------------
input group "Pattern Encoding"
input int      InpSeqLen            = 20;     // Sequence length (bars)
input int      InpPredK             = 5;      // Future bars to score (towards present)
input bool     InpUseDirectionFlag  = true;   // Add bullish/bearish flag
input bool     InpNormalizeFeatures = true;   // Min-max normalize per window
input double   InpNormEps           = 1e-8;   // Epsilon for safe division

input group "Training"
input int      InpHistoryBars       = 1500;   // Training history depth (bars into past)
input int      InpMaxSamples        = 1200;   // Max training samples
input int      InpHiddenSize1       = 32;     // LSTM layer 1 hidden size
input int      InpHiddenSize2       = 16;     // LSTM layer 2 hidden size (0=off)
input double   InpDropout           = 0.05;   // Dropout rate
input int      InpTrainEpochs       = 25;     // Training epochs
input double   InpTargetMSE         = 0.01;   // Target MSE (DLL uses MSE)
input double   InpLearningRate      = 0.001;  // Learning rate
input double   InpWeightDecay       = 0.0001; // Weight decay
input int      InpRetrainEveryBars  = 20;     // Retrain interval (new bars)
input int      InpMiniBatch         = 32;     // Mini-batch size

// -------------------- Globals --------------------
double g_BullishBuffer[];
double g_BearishBuffer[];

int      g_NetHandle      = 0;
bool     g_ModelReady     = false;
bool     g_IsTraining     = false;
int      g_BarsSinceTrain = 0;
datetime g_LastClosedTime = 0;   // time[1] (last closed bar)

// GPU progress
int      g_ProgEpoch       = 0;
int      g_ProgTotalEpochs = 0;
int      g_ProgMB          = 0;
int      g_ProgTotalMB     = 0;
double   g_ProgLR          = 0.0;
double   g_ProgMSE         = 0.0;
double   g_ProgBestMSE     = 0.0;
double   g_ProgGradNorm    = 0.0;
double   g_ProgPercent     = 0.0;
double   g_ProgElapsedSec  = 0.0;
double   g_ProgETASec      = 0.0;

// -------------------- Helpers --------------------
int FeatureDim()
{
   return (InpUseDirectionFlag ? 4 : 3);
}

string GetDLLError()
{
   short buf[];
   ArrayResize(buf, 512);
   ArrayInitialize(buf, 0);
   DN_GetError(buf, 512);

   string s = "";
   for(int i = 0; i < ArraySize(buf) && buf[i] != 0; i++)
      s += ShortToString(buf[i]);

   if(StringLen(s) == 0)
      return "unknown DLL error";
   return s;
}

bool IsFiniteValue(double v)
{
   return (MathIsValidNumber(v) && v > -1e300 && v < 1e300);
}

double SafeDiv(double num, double den, double eps)
{
   if(!IsFiniteValue(num))
      num = 0.0;
   if(!IsFiniteValue(den) || MathAbs(den) <= eps)
      den = (den >= 0.0 ? eps : -eps);

   double out = num / den;
   if(!IsFiniteValue(out))
      return 0.0;
   return out;
}

string FormatDuration(double seconds)
{
   int s = (int)MathRound(seconds);
   if(s < 0) return "--";
   if(s < 60)   return StringFormat("%ds", s);
   if(s < 3600) return StringFormat("%dm%02ds", s / 60, s % 60);
   return StringFormat("%dh%02dm", s / 3600, (s % 3600) / 60);
}

// Sigmoid keeps outputs sane even if linear head is unbounded.
// (Your DLL output layer is linear.)
double Sigmoid(double x)
{
   if(!IsFiniteValue(x)) return 0.5;
   // clamp for numeric sanity
   if(x > 60.0)  return 1.0;
   if(x < -60.0) return 0.0;
   return 1.0 / (1.0 + MathExp(-x));
}

// -------------------- Network init --------------------
bool InitNetwork()
{
   g_NetHandle = DN_Create();
   if(g_NetHandle <= 0)
   {
      Print("DN_Create failed: ", GetDLLError());
      return false;
   }

   if(!DN_SetSequenceLength(g_NetHandle, InpSeqLen))
   {
      Print("DN_SetSequenceLength failed: ", GetDLLError());
      DN_Free(g_NetHandle); g_NetHandle = 0;
      return false;
   }

   if(!DN_SetMiniBatchSize(g_NetHandle, InpMiniBatch))
   {
      Print("DN_SetMiniBatchSize failed: ", GetDLLError());
      DN_Free(g_NetHandle); g_NetHandle = 0;
      return false;
   }

   // Layer 1: input_dim -> hidden1
   if(!DN_AddLayerEx(g_NetHandle, FeatureDim(), InpHiddenSize1, 0, 0, InpDropout))
   {
      Print("DN_AddLayerEx L1 failed: ", GetDLLError());
      DN_Free(g_NetHandle); g_NetHandle = 0;
      return false;
   }

   // Layer 2: hidden1 -> hidden2 (optional)
   if(InpHiddenSize2 > 0)
   {
      if(!DN_AddLayerEx(g_NetHandle, 0, InpHiddenSize2, 0, 0, InpDropout * 0.5))
      {
         Print("DN_AddLayerEx L2 failed: ", GetDLLError());
         DN_Free(g_NetHandle); g_NetHandle = 0;
         return false;
      }
   }

   // Output: 2 values (bullish score, bearish score)
   if(!DN_SetOutputDim(g_NetHandle, 2))
   {
      Print("DN_SetOutputDim failed: ", GetDLLError());
      DN_Free(g_NetHandle); g_NetHandle = 0;
      return false;
   }

   Print(StringFormat("Network: feat=%d -> LSTM(%d)%s -> 2",
         FeatureDim(), InpHiddenSize1,
         (InpHiddenSize2 > 0 ? StringFormat(" -> LSTM(%d)", InpHiddenSize2) : "")));

   return true;
}

// -------------------- Feature builder --------------------
// Build features for a window that ENDS at bar_end (series index).
// Window covers bars: bar_end, bar_end+1, ..., bar_end+SeqLen-1 (older)
// Output flat[] length = SeqLen*feat_dim, ordered oldest->newest (chronological).
bool BuildWindowFeatures(const int rates_total,
                         const double &open[],
                         const double &high[],
                         const double &low[],
                         const double &close[],
                         const int bar_end,
                         double &flat[])
{
   const int feat_dim = FeatureDim();
   const int in_dim = InpSeqLen * feat_dim;
   ArrayResize(flat, in_dim);

   // Need full window
   int oldest = bar_end + (InpSeqLen - 1);
   if(oldest >= rates_total) return false;
   if(bar_end < 0) return false;

   // Window hi/lo for normalization
   double win_hi = -1e300;
   double win_lo =  1e300;

   for(int i = 0; i < InpSeqLen; i++)
   {
      int idx = bar_end + i;
      double h = high[idx];
      double l = low[idx];
      if(h > win_hi) win_hi = h;
      if(l < win_lo) win_lo = l;
   }

   double win_range = win_hi - win_lo;
   if(!IsFiniteValue(win_range) || win_range <= InpNormEps)
      win_range = InpNormEps;

   // Fill chronological order: oldest first.
   // oldest index = bar_end+SeqLen-1, newest = bar_end
   for(int t = 0; t < InpSeqLen; t++)
   {
      int idx = bar_end + (InpSeqLen - 1 - t); // maps t=0 -> oldest
      double o = open[idx];
      double h = high[idx];
      double l = low[idx];
      double c = close[idx];

      double body  = SafeDiv(c - o, win_range, InpNormEps);
      double upper = SafeDiv(h - MathMax(o, c), win_range, InpNormEps);
      double lower = SafeDiv(MathMin(o, c) - l, win_range, InpNormEps);

      int base = t * feat_dim;
      flat[base + 0] = body;
      flat[base + 1] = upper;
      flat[base + 2] = lower;
      if(InpUseDirectionFlag)
         flat[base + 3] = (c >= o ? 1.0 : 0.0);
   }

   // Optional min-max normalization per channel inside this window
   if(InpNormalizeFeatures)
   {
      for(int f = 0; f < feat_dim; f++)
      {
         if(InpUseDirectionFlag && f == 3) // binary channel, keep as is
            continue;

         double mn =  1e300;
         double mx = -1e300;
         for(int t = 0; t < InpSeqLen; t++)
         {
            double v = flat[t * feat_dim + f];
            if(v < mn) mn = v;
            if(v > mx) mx = v;
         }

         double den = mx - mn;
         if(!IsFiniteValue(den) || den <= InpNormEps)
            den = InpNormEps;

         for(int t = 0; t < InpSeqLen; t++)
         {
            int pos = t * feat_dim + f;
            flat[pos] = SafeDiv(flat[pos] - mn, den, InpNormEps);
         }
      }
   }

   return true;
}

// -------------------- Target builder --------------------
// Target looks "forward" K bars from bar_end towards present (smaller indices).
// future bars are: bar_end-1, bar_end-2, ..., bar_end-K
void BuildTarget(const int rates_total,
                 const double &open[],
                 const double &close[],
                 const int bar_end,
                 double &bullish,
                 double &bearish)
{
   int up = 0;
   int count = 0;

   for(int k = 1; k <= InpPredK; k++)
   {
      int idx = bar_end - k;
      if(idx < 1) break;             // avoid 0 (forming) and negative
      if(idx >= rates_total) continue;

      if(close[idx] > open[idx]) up++;
      count++;
   }

   if(count > 0)
   {
      bullish = (double)up / (double)count;
      bearish = 1.0 - bullish;
   }
   else
   {
      bullish = 0.5;
      bearish = 0.5;
   }

   bullish = MathMin(1.0, MathMax(0.0, bullish));
   bearish = MathMin(1.0, MathMax(0.0, bearish));
}

// -------------------- Training set builder --------------------
// Builds samples from windows ending at bar_end in [min_end .. max_end].
// bar_end must satisfy: bar_end >= 1+PredK (enough future closed bars)
// and bar_end+SeqLen-1 < rates_total (enough past bars).
// X layout expected by DLL: row-major [batch, in_dim] where in_dim=SeqLen*feat_dim.
// Each row is chronological timestep order (oldest->newest) which we already ensure.
bool BuildTrainingSet(const int rates_total,
                      const double &open[],
                      const double &high[],
                      const double &low[],
                      const double &close[],
                      double &X[],
                      double &T[],
                      int &batch,
                      int &in_dim)
{
   const int feat_dim = FeatureDim();
   in_dim = InpSeqLen * feat_dim;
   batch = 0;

   // Max possible end index so that window has enough past bars:
   // bar_end + (SeqLen-1) <= rates_total-1  => bar_end <= rates_total-SeqLen
   int max_end = rates_total - InpSeqLen;
   // Min end index so that we have K future closed bars (towards present):
   // bar_end - K >= 1  => bar_end >= 1+K
   int min_end = 1 + InpPredK;

   if(max_end < min_end) return false;

   // Limit by history depth: end indices from min_end..max_end, but only last InpHistoryBars
   // "History depth" counts bars into past from the most recent usable end.
   // The most recent usable end is min(max_end, some big), but in series indexing,
   // smaller index = more recent. So "most recent end" is actually min_end, not max_end.
   // We'll instead cap the OLDEST end (largest index) by history depth from min_end.
   int oldest_allowed = min_end + MathMax(0, InpHistoryBars - 1);
   if(oldest_allowed > max_end) oldest_allowed = max_end;

   // candidate end indices: bar_end in [min_end .. oldest_allowed]
   int possible = oldest_allowed - min_end + 1;
   if(possible <= 0) return false;

   int wanted = MathMin(possible, InpMaxSamples);

   // Use the most recent 'wanted' ends: min_end .. min_end+wanted-1
   int start_end = min_end;
   int end_end   = min_end + wanted - 1;

   ArrayResize(X, wanted * in_dim);
   ArrayResize(T, wanted * 2);

   double sample[];
   for(int s = 0; s < wanted; s++)
   {
      int bar_end = start_end + s;

      if(!BuildWindowFeatures(rates_total, open, high, low, close, bar_end, sample))
         continue;

      int xoff = batch * in_dim;
      for(int i = 0; i < in_dim; i++)
         X[xoff + i] = sample[i];

      double bull = 0.5, bear = 0.5;
      BuildTarget(rates_total, open, close, bar_end, bull, bear);

      T[batch * 2 + 0] = bull;
      T[batch * 2 + 1] = bear;
      batch++;
   }

   if(batch <= 0) return false;

   if(batch < wanted)
   {
      ArrayResize(X, batch * in_dim);
      ArrayResize(T, batch * 2);
   }

   return true;
}

// -------------------- Training control --------------------
bool StartTraining(const int rates_total,
                   const double &open[],
                   const double &high[],
                   const double &low[],
                   const double &close[])
{
   if(g_NetHandle <= 0)
      return false;

   double X[];
   double T[];
   int batch = 0;
   int in_dim = 0;

   if(!BuildTrainingSet(rates_total, open, high, low, close, X, T, batch, in_dim))
   {
      Print("Training set build failed: insufficient data (rates_total=", rates_total, ")");
      return false;
   }

   Print(StringFormat("Training: %d samples, in_dim=%d (seq=%d x feat=%d)",
                      batch, in_dim, InpSeqLen, FeatureDim()));

   // layout=0: DLL expects row-major flat, and internally transposes to timestep-major.
   if(!DN_LoadBatch(g_NetHandle, X, T, batch, in_dim, 2, 0))
   {
      Print("DN_LoadBatch failed: ", GetDLLError());
      return false;
   }

   if(!DN_TrainAsync(g_NetHandle, InpTrainEpochs, InpTargetMSE, InpLearningRate, InpWeightDecay))
   {
      Print("DN_TrainAsync failed: ", GetDLLError());
      return false;
   }

   g_IsTraining   = true;
   g_ProgPercent  = 0.0;
   g_ProgMSE      = 0.0;
   g_ProgBestMSE  = 0.0;
   return true;
}

// Predict on latest closed bar (index=1) using window ending at bar_end=1+PredK
// Why 1+PredK? Because target definition needs K future closed bars; for live prediction
// we only need the window; but we want consistency with training distribution.
// For "completion" we score how the pattern tends to resolve; so we end at last closed bar 1,
// and don't require target existence. We'll end at bar_end=1 for live prediction.
// (We keep the window shape identical.)
bool PredictLatest(const int rates_total,
                   const double &open[],
                   const double &high[],
                   const double &low[],
                   const double &close[],
                   double &bull,
                   double &bear)
{
   bull = 0.5;
   bear = 0.5;

   int bar_end = 1; // last closed bar
   int oldest  = bar_end + (InpSeqLen - 1);
   if(oldest >= rates_total) return false;

   double X[];
   if(!BuildWindowFeatures(rates_total, open, high, low, close, bar_end, X))
      return false;

   int in_dim = InpSeqLen * FeatureDim();

   double Y[];
   ArrayResize(Y, 2);
   ArrayInitialize(Y, 0.0);

   if(!DN_PredictBatch(g_NetHandle, X, 1, in_dim, 0, Y))
   {
      Print("DN_PredictBatch failed: ", GetDLLError());
      return false;
   }

   // Map linear outputs -> (0..1) via sigmoid, then renormalize
   double b0 = Sigmoid(Y[0]);
   double b1 = Sigmoid(Y[1]);

   double sum = b0 + b1;
   if(sum > InpNormEps)
   {
      bull = b0 / sum;
      bear = b1 / sum;
   }
   else
   {
      bull = 0.5;
      bear = 0.5;
   }

   if(!IsFiniteValue(bull)) bull = 0.5;
   if(!IsFiniteValue(bear)) bear = 0.5;

   bull = MathMin(1.0, MathMax(0.0, bull));
   bear = MathMin(1.0, MathMax(0.0, bear));
   return true;
}

// -------------------- UI --------------------
void UpdateProgressComment()
{
   string line = "";

   if(g_IsTraining)
   {
      line = StringFormat("Training: Ep %d/%d | MB %d/%d | MSE: %.6f | %.1f%% | LR: %.6f | GradN: %.4f | Elapsed: %s | ETA: %s",
                          g_ProgEpoch, g_ProgTotalEpochs,
                          g_ProgMB, g_ProgTotalMB,
                          g_ProgMSE,
                          g_ProgPercent,
                          g_ProgLR,
                          g_ProgGradNorm,
                          FormatDuration(g_ProgElapsedSec),
                          FormatDuration(g_ProgETASec));
   }
   else if(g_ModelReady)
   {
      int left = MathMax(0, InpRetrainEveryBars - g_BarsSinceTrain);
      double mse_show = (g_ProgBestMSE > 0.0 ? g_ProgBestMSE : g_ProgMSE);
      line = StringFormat("Model ready | Best MSE: %.6f | Retrain in %d bars", mse_show, left);
   }
   else
   {
      line = "Waiting for data...";
   }

   Comment(line);
}

// -------------------- Events --------------------
int OnInit()
{
   SetIndexBuffer(0, g_BullishBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, g_BearishBuffer, INDICATOR_DATA);

   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   IndicatorSetString(INDICATOR_SHORTNAME, "LSTM Pattern Completion (GPU)");

   if(!InitNetwork())
      return INIT_FAILED;

   EventSetMillisecondTimer(250);

   Print(StringFormat("PatternCompletion: seq=%d pred_k=%d feat=%d hist=%d maxS=%d",
                      InpSeqLen, InpPredK, FeatureDim(), InpHistoryBars, InpMaxSamples));

   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   Comment("");

   if(g_NetHandle > 0)
   {
      if(g_IsTraining)
      {
         DN_StopTraining(g_NetHandle);
         Sleep(300);
      }
      DN_Free(g_NetHandle);
      g_NetHandle = 0;
   }
}

void OnTimer()
{
   if(!g_IsTraining || g_NetHandle <= 0)
      return;

   DN_GetProgressAll(g_NetHandle,
                     g_ProgEpoch, g_ProgTotalEpochs,
                     g_ProgMB, g_ProgTotalMB,
                     g_ProgLR, g_ProgMSE, g_ProgBestMSE,
                     g_ProgGradNorm, g_ProgPercent,
                     g_ProgElapsedSec, g_ProgETASec);

   int st = DN_GetTrainingStatus(g_NetHandle);

   UpdateProgressComment();

   if(st == 1) // TS_TRAINING
      return;

   g_IsTraining = false;

   if(st == 2) // TS_COMPLETED
   {
      g_ModelReady     = true;
      g_BarsSinceTrain = 0;
      Print(StringFormat("Training complete: MSE=%.6f best=%.6f elapsed=%.1fs",
                         g_ProgMSE, g_ProgBestMSE, g_ProgElapsedSec));
   }
   else // TS_ERROR
   {
      Print("Training error: ", GetDLLError());
   }

   UpdateProgressComment();
}

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
   // Need enough bars for:
   // - training: bar_end up to rates_total-SeqLen, and bar_end min=1+PredK
   // - prediction: window ending at bar_end=1 => oldest=1+SeqLen-1
   int min_bars = (1 + InpSeqLen - 1) + 1; // simplest: rates_total > SeqLen
   if(rates_total < MathMax(min_bars, InpSeqLen + InpPredK + 10))
      return 0;

   // Ensure buffers are series to match chart indexing (0=current)
   ArraySetAsSeries(g_BullishBuffer, true);
   ArraySetAsSeries(g_BearishBuffer, true);

   int start = (prev_calculated > 0 ? prev_calculated - 1 : 0);
   start = MathMax(start, 0);
   for(int i = start; i < rates_total; i++)
   {
      g_BullishBuffer[i] = EMPTY_VALUE;
      g_BearishBuffer[i] = EMPTY_VALUE;
   }

   // New closed bar detection: use time[1] in series arrays
   datetime last_closed = time[1];
   bool new_closed_bar = (g_LastClosedTime != last_closed);
   if(new_closed_bar)
   {
      g_LastClosedTime = last_closed;
      g_BarsSinceTrain++;

      if(!g_IsTraining && (!g_ModelReady || g_BarsSinceTrain >= InpRetrainEveryBars))
      {
         if(StartTraining(rates_total, open, high, low, close))
            Print(StringFormat("Training started (bars since last=%d)", g_BarsSinceTrain));
      }
   }

   // Predict on last closed bar
   if(g_ModelReady && !g_IsTraining)
   {
      double bull = 0.5, bear = 0.5;
      if(PredictLatest(rates_total, open, high, low, close, bull, bear))
      {
         g_BullishBuffer[1] = bull;
         g_BearishBuffer[1] = bear;

         // optional continuity on forming bar:
         g_BullishBuffer[0] = bull;
         g_BearishBuffer[0] = bear;
      }
   }

   UpdateProgressComment();
   return rates_total;
}
//+------------------------------------------------------------------+
