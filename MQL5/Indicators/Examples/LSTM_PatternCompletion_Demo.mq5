//+------------------------------------------------------------------+
//| LSTM_PatternCompletion_Demo.mq5                                  |
//| Demonstrates pattern-completion training using candle tokens.    |
//| The indicator builds a symbolic sequence from OHLC, trains a     |
//| small LSTM model through MQL5GPULibrary_LSTM.dll, and predicts   |
//| bullish/bearish completion scores in [0..1].                     |
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

input group "Pattern Encoding"
input int      InpSeqLen            = 20;     // Sequence length (bars)
input int      InpPredK             = 5;      // Future bars to score
input bool     InpUseDirectionFlag  = true;   // Add bullish/bearish flag
input bool     InpNormalizeFeatures = true;   // Min-max normalize per window
input double   InpNormEps           = 1e-8;   // Epsilon for safe division

input group "Training"
input int      InpHistoryBars       = 1500;   // Training history depth
input int      InpMaxSamples        = 1200;   // Max training samples
input int      InpHiddenSize1       = 32;     // LSTM layer 1 hidden size
input int      InpHiddenSize2       = 16;     // LSTM layer 2 hidden size (0=off)
input double   InpDropout           = 0.05;   // Dropout rate
input int      InpTrainEpochs       = 25;     // Training epochs
input double   InpTargetMSE         = 0.01;   // Target MSE
input double   InpLearningRate      = 0.001;  // Learning rate
input double   InpWeightDecay       = 0.0001; // Weight decay
input int      InpRetrainEveryBars  = 20;     // Retrain interval (new bars)
input int      InpMiniBatch         = 32;     // Mini-batch size

//+------------------------------------------------------------------+
//| Globals                                                           |
//+------------------------------------------------------------------+
double g_BullishBuffer[];
double g_BearishBuffer[];

int      g_NetHandle          = 0;
bool     g_ModelReady         = false;
bool     g_IsTraining         = false;
int      g_BarsSinceTrain     = 0;
datetime g_LastBarTime        = 0;

// GPU progress
int      g_ProgEpoch          = 0;
int      g_ProgTotalEpochs    = 0;
int      g_ProgMB             = 0;
int      g_ProgTotalMB        = 0;
double   g_ProgLR             = 0.0;
double   g_ProgMSE            = 0.0;
double   g_ProgBestMSE        = 0.0;
double   g_ProgGradNorm       = 0.0;
double   g_ProgPercent        = 0.0;
double   g_ProgElapsedSec     = 0.0;
double   g_ProgETASec         = 0.0;

//+------------------------------------------------------------------+
//| Feature dimension per bar                                         |
//+------------------------------------------------------------------+
int FeatureDim()
{
   return (InpUseDirectionFlag ? 4 : 3);
}

//+------------------------------------------------------------------+
//| Get DLL error string                                              |
//+------------------------------------------------------------------+
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

//+------------------------------------------------------------------+
//| Safe value check                                                  |
//+------------------------------------------------------------------+
bool IsFiniteValue(double v)
{
   return (MathIsValidNumber(v) && v > -1e300 && v < 1e300);
}

//+------------------------------------------------------------------+
//| Safe division                                                     |
//+------------------------------------------------------------------+
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

//+------------------------------------------------------------------+
//| Format seconds to readable string                                 |
//+------------------------------------------------------------------+
string FormatDuration(double seconds)
{
   int s = (int)MathRound(seconds);
   if(s < 0) return "--";
   if(s < 60)   return StringFormat("%ds", s);
   if(s < 3600) return StringFormat("%dm%02ds", s / 60, s % 60);
   return StringFormat("%dh%02dm", s / 3600, (s % 3600) / 60);
}

//+------------------------------------------------------------------+
//| Initialize LSTM network                                           |
//+------------------------------------------------------------------+
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

   Print(StringFormat("Network: %d -> LSTM(%d) -> LSTM(%d) -> 2",
         FeatureDim(), InpHiddenSize1,
         InpHiddenSize2 > 0 ? InpHiddenSize2 : InpHiddenSize1));

   return true;
}

//+------------------------------------------------------------------+
//| Build features for one window of InpSeqLen bars                   |
//| chrono_start = chronological index (0 = oldest available bar)     |
//| Output: flat[InpSeqLen * FeatureDim] — oldest first               |
//+------------------------------------------------------------------+
bool BuildWindowFeatures(const int rates_total,
                         const double &open[],
                         const double &high[],
                         const double &low[],
                         const double &close[],
                         const int chrono_start,
                         double &flat[])
{
   const int feat_dim = FeatureDim();
   const int in_dim = InpSeqLen * feat_dim;
   ArrayResize(flat, in_dim);

   //--- Find window range for normalization
   double win_hi = -1e300;
   double win_lo =  1e300;

   for(int i = 0; i < InpSeqLen; i++)
   {
      // Convert chronological index to series-indexed arrays
      // In OnCalculate, arrays are NOT as-series by default
      // chrono_start+i is direct index into the arrays
      int idx = chrono_start + i;
      if(idx < 0 || idx >= rates_total)
         return false;
      if(high[idx] > win_hi) win_hi = high[idx];
      if(low[idx]  < win_lo) win_lo = low[idx];
   }

   double win_range = win_hi - win_lo;
   if(!IsFiniteValue(win_range) || win_range <= InpNormEps)
      win_range = InpNormEps;

   //--- Extract features for each bar in window
   for(int i = 0; i < InpSeqLen; i++)
   {
      int idx = chrono_start + i;
      double o = open[idx];
      double h = high[idx];
      double l = low[idx];
      double c = close[idx];

      double body  = SafeDiv(c - o, win_range, InpNormEps);
      double upper = SafeDiv(h - MathMax(o, c), win_range, InpNormEps);
      double lower = SafeDiv(MathMin(o, c) - l, win_range, InpNormEps);

      int base = i * feat_dim;
      flat[base + 0] = body;
      flat[base + 1] = upper;
      flat[base + 2] = lower;
      if(InpUseDirectionFlag)
         flat[base + 3] = (c >= o ? 1.0 : 0.0);
   }

   //--- Optional min-max normalization per feature channel
   if(InpNormalizeFeatures)
   {
      for(int f = 0; f < feat_dim; f++)
      {
         // Skip binary direction flag
         if(InpUseDirectionFlag && f == 3)
            continue;

         double mn =  1e300;
         double mx = -1e300;
         for(int i = 0; i < InpSeqLen; i++)
         {
            double v = flat[i * feat_dim + f];
            if(v < mn) mn = v;
            if(v > mx) mx = v;
         }

         double den = mx - mn;
         if(!IsFiniteValue(den) || den <= InpNormEps)
            den = InpNormEps;

         for(int i = 0; i < InpSeqLen; i++)
         {
            int pos = i * feat_dim + f;
            flat[pos] = SafeDiv(flat[pos] - mn, den, InpNormEps);
         }
      }
   }

   return true;
}

//+------------------------------------------------------------------+
//| Build target for K bars starting at future_start                  |
//| Uses direct array indexing (arrays are NOT as-series)             |
//+------------------------------------------------------------------+
void BuildTarget(const int rates_total,
                 const double &open[],
                 const double &close[],
                 const int future_start,
                 double &bullish,
                 double &bearish)
{
   int up = 0;
   int count = 0;
   for(int i = 0; i < InpPredK; i++)
   {
      int idx = future_start + i;
      if(idx < 0 || idx >= rates_total)
         continue;
      if(close[idx] > open[idx])
         up++;
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

//+------------------------------------------------------------------+
//| Build training dataset from history                               |
//| X: [batch, seq_len * feat_dim] row-major, oldest timestep first  |
//| T: [batch, 2] bullish/bearish scores                             |
//+------------------------------------------------------------------+
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

   // Need at least: InpSeqLen bars for input + InpPredK bars for target
   // Plus some margin. We use bar indices [0..rates_total-1] directly.
   // Last usable window start: rates_total - InpSeqLen - InpPredK
   int last_window_start = rates_total - InpSeqLen - InpPredK;
   if(last_window_start < 0)
      return false;

   // Limit history depth
   int first_window_start = MathMax(0, last_window_start - InpHistoryBars + 1);

   int possible = last_window_start - first_window_start + 1;
   if(possible <= 0)
      return false;

   int wanted = MathMin(possible, InpMaxSamples);

   // Take the most recent 'wanted' samples
   int actual_first = last_window_start - wanted + 1;
   if(actual_first < first_window_start)
      actual_first = first_window_start;

   wanted = last_window_start - actual_first + 1;
   if(wanted <= 0)
      return false;

   ArrayResize(X, wanted * in_dim);
   ArrayResize(T, wanted * 2);

   double sample[];
   for(int s = 0; s < wanted; s++)
   {
      int win_start = actual_first + s;

      if(!BuildWindowFeatures(rates_total, open, high, low, close,
                              win_start, sample))
         continue;

      int xoff = batch * in_dim;
      for(int i = 0; i < in_dim; i++)
         X[xoff + i] = sample[i];

      double bull = 0.5, bear = 0.5;
      int future_start = win_start + InpSeqLen;
      BuildTarget(rates_total, open, close, future_start, bull, bear);

      T[batch * 2 + 0] = bull;
      T[batch * 2 + 1] = bear;
      batch++;
   }

   if(batch <= 0)
      return false;

   // Trim arrays if some samples were skipped
   if(batch < wanted)
   {
      ArrayResize(X, batch * in_dim);
      ArrayResize(T, batch * 2);
   }

   return true;
}

//+------------------------------------------------------------------+
//| Start async training                                              |
//+------------------------------------------------------------------+
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

   if(!BuildTrainingSet(rates_total, open, high, low, close,
                        X, T, batch, in_dim))
   {
      Print("Training set build failed: insufficient data");
      return false;
   }

   Print(StringFormat("Training: %d samples, in_dim=%d (seq=%d x feat=%d)",
         batch, in_dim, InpSeqLen, FeatureDim()));

   // layout=0: standard row-major flat format
   if(!DN_LoadBatch(g_NetHandle, X, T, batch, in_dim, 2, 0))
   {
      Print("DN_LoadBatch failed: ", GetDLLError());
      return false;
   }

   if(!DN_TrainAsync(g_NetHandle, InpTrainEpochs, InpTargetMSE,
                     InpLearningRate, InpWeightDecay))
   {
      Print("DN_TrainAsync failed: ", GetDLLError());
      return false;
   }

   g_IsTraining = true;
   g_ProgPercent = 0;
   g_ProgMSE = 0;
   return true;
}

//+------------------------------------------------------------------+
//| Predict latest bar                                                |
//+------------------------------------------------------------------+
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

   // Build window ending at the last completed bar
   // Last completed bar index = rates_total - 2
   // (index rates_total-1 is current forming bar)
   int last_completed = rates_total - 2;
   if(last_completed < InpSeqLen - 1)
      return false;

   int win_start = last_completed - InpSeqLen + 1;
   if(win_start < 0)
      return false;

   double X[];
   if(!BuildWindowFeatures(rates_total, open, high, low, close,
                           win_start, X))
      return false;

   int in_dim = InpSeqLen * FeatureDim();

   double Y[];
   ArrayResize(Y, 2);

   // layout=0: standard format
   if(!DN_PredictBatch(g_NetHandle, X, 1, in_dim, 0, Y))
   {
      Print("DN_PredictBatch failed: ", GetDLLError());
      return false;
   }

   bull = Y[0];
   bear = Y[1];

   if(!IsFiniteValue(bull)) bull = 0.5;
   if(!IsFiniteValue(bear)) bear = 0.5;

   bull = MathMin(1.0, MathMax(0.0, bull));
   bear = MathMin(1.0, MathMax(0.0, bear));

   // Normalize to sum = 1
   double sum = bull + bear;
   if(sum > InpNormEps)
   {
      bull = bull / sum;
      bear = bear / sum;
   }

   return true;
}

//+------------------------------------------------------------------+
//| Update comment with training progress                             |
//+------------------------------------------------------------------+
void UpdateProgressComment()
{
   string line = "";

   if(g_IsTraining)
   {
      line = StringFormat("Training: Ep %d/%d | MB %d/%d | "
                          "MSE: %.6f | %.1f%% | "
                          "LR: %.6f | GradN: %.4f | "
                          "Elapsed: %s | ETA: %s",
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
      line = StringFormat("Model ready | Last MSE: %.6f | "
                          "Retrain in %d bars",
                          g_ProgBestMSE > 0 ? g_ProgBestMSE : g_ProgMSE,
                          MathMax(0, InpRetrainEveryBars - g_BarsSinceTrain));
   }
   else
   {
      line = "Waiting for data...";
   }

   Comment(line);
}

//+------------------------------------------------------------------+
//| OnInit                                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   SetIndexBuffer(0, g_BullishBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, g_BearishBuffer, INDICATOR_DATA);

   // Do NOT set as series — use direct chronological indexing
   // (consistent with how OnCalculate provides arrays)
   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, EMPTY_VALUE);

   IndicatorSetString(INDICATOR_SHORTNAME, "LSTM Pattern Completion");

   if(!InitNetwork())
      return INIT_FAILED;

   EventSetMillisecondTimer(250);

   Print(StringFormat("PatternCompletion: seq=%d pred_k=%d feat=%d hist=%d",
         InpSeqLen, InpPredK, FeatureDim(), InpHistoryBars));

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| OnDeinit                                                          |
//+------------------------------------------------------------------+
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

//+------------------------------------------------------------------+
//| OnTimer — poll GPU training progress                              |
//+------------------------------------------------------------------+
void OnTimer()
{
   if(!g_IsTraining || g_NetHandle <= 0)
      return;

   //--- Read all progress in one DLL call (lock-free)
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

   //--- Training finished
   g_IsTraining = false;

   if(st == 2) // TS_COMPLETED
   {
      g_ModelReady = true;
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
   int min_bars = InpSeqLen + InpPredK + 50;
   if(rates_total < min_bars)
      return 0;

   //--- Initialize empty values for new bars
   int start = (prev_calculated > 0) ? prev_calculated - 1 : 0;
   for(int i = start; i < rates_total; i++)
   {
      g_BullishBuffer[i] = EMPTY_VALUE;
      g_BearishBuffer[i] = EMPTY_VALUE;
   }

   //--- Detect new bar (time[] is chronological: [0]=oldest, [last]=newest)
   datetime newest_time = time[rates_total - 1];
   bool new_bar = (g_LastBarTime != newest_time);
   if(new_bar)
   {
      g_LastBarTime = newest_time;
      g_BarsSinceTrain++;

      //--- Trigger training if needed
      if(!g_IsTraining && (!g_ModelReady || g_BarsSinceTrain >= InpRetrainEveryBars))
      {
         if(StartTraining(rates_total, open, high, low, close))
            Print(StringFormat("Training started (%d bars since last)",
                  g_BarsSinceTrain));
      }
   }

   //--- Predict on latest completed bar
   if(g_ModelReady && !g_IsTraining)
   {
      double bull = 0.5, bear = 0.5;
      if(PredictLatest(rates_total, open, high, low, close, bull, bear))
      {
         // Write to the last completed bar (rates_total - 2)
         int last_completed = rates_total - 2;
         if(last_completed >= 0 && last_completed < rates_total)
         {
            g_BullishBuffer[last_completed] = bull;
            g_BearishBuffer[last_completed] = bear;
         }

         // Also write to current forming bar for visual continuity
         g_BullishBuffer[rates_total - 1] = bull;
         g_BearishBuffer[rates_total - 1] = bear;
      }
   }

   UpdateProgressComment();
   return rates_total;
}
//+------------------------------------------------------------------+
