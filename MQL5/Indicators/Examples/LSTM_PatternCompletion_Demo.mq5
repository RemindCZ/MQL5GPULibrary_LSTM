//+------------------------------------------------------------------+
//| LSTM_PatternCompletion_Demo.mq5                                  |
//| Demonstrates pattern-completion training using candle tokens.    |
//| The indicator builds a symbolic sequence from OHLC, trains a     |
//| small LSTM model through MQL5GPULibrary_LSTM.dll, and predicts   |
//| bullish/bearish completion scores in [0..1].                     |
//|                                                                  |
//| How to run:                                                       |
//| 1) Copy indicator to MQL5/Indicators/Examples and compile in MT5 |
//| 2) Ensure MQL5GPULibrary_LSTM.dll is available to terminal       |
//| 3) Attach indicator to a chart and watch progress in Comment()   |
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
   int    DN_AddLayerEx(int h, int in_sz, int out_sz, int act, int ln, double drop);
   int    DN_SetOutputDim(int h, int out_dim);
   int    DN_LoadBatch(int h, const double &X[], const double &T[], int batch, int in_dim, int out_dim, int layout);
   int    DN_TrainAsync(int h, int epochs, double target_mse, double lr, double wd);
   int    DN_GetTrainingStatus(int h);
   int    DN_GetProgressAll(int h,
             int &epoch, int &total_epochs,
             int &mb, int &total_mb,
             double &lr, double &mse, double &best_mse,
             double &grad_norm, double &pct,
             double &elapsed_sec, double &eta_sec);
   int    DN_PredictBatch(int h, const double &X[], int batch, int in_dim, int layout, double &Y[]);
   void   DN_GetError(short &buf[], int len);
#import

input group "Pattern Encoding"
input int      InpSeqLen            = 20;
input int      InpPredK             = 5;
input bool     InpUseDirectionFlag  = true;
input bool     InpNormalizeFeatures = true;
input double   InpNormEps           = 1e-8;

input group "Training"
input int      InpHistoryBars       = 1500;
input int      InpMaxSamples        = 1200;
input int      InpHiddenSize1       = 32;
input int      InpHiddenSize2       = 16;
input double   InpDropout           = 0.05;
input int      InpTrainEpochs       = 25;
input double   InpTargetMSE         = 0.01;
input double   InpLearningRate      = 0.001;
input double   InpWeightDecay       = 0.0001;
input int      InpRetrainEveryBars  = 20;

double g_BullishBuffer[];
double g_BearishBuffer[];

int      g_NetHandle          = 0;
bool     g_ModelReady         = false;
bool     g_IsTraining         = false;
int      g_BarsSinceTrain     = 0;
datetime g_LastBarTime        = 0;

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
int FeatureDim()
{
   return (InpUseDirectionFlag ? 4 : 3);
}

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
bool IsFiniteValue(double v)
{
   return (MathIsValidNumber(v) && v > -DBL_MAX && v < DBL_MAX);
}

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
      return false;
   }

   if(!DN_AddLayerEx(g_NetHandle, FeatureDim(), InpHiddenSize1, 0, 0, InpDropout))
   {
      Print("DN_AddLayerEx L1 failed: ", GetDLLError());
      return false;
   }

   if(InpHiddenSize2 > 0)
   {
      if(!DN_AddLayerEx(g_NetHandle, 0, InpHiddenSize2, 0, 0, InpDropout * 0.5))
      {
         Print("DN_AddLayerEx L2 failed: ", GetDLLError());
         return false;
      }
   }

   if(!DN_SetOutputDim(g_NetHandle, 2))
   {
      Print("DN_SetOutputDim failed: ", GetDLLError());
      return false;
   }

   return true;
}

//+------------------------------------------------------------------+
int ToSeriesShift(const int rates_total, const int chrono_index)
{
   return rates_total - 1 - chrono_index;
}

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

   double win_hi = -DBL_MAX;
   double win_lo =  DBL_MAX;

   for(int i = 0; i < InpSeqLen; i++)
   {
      int sh = ToSeriesShift(rates_total, chrono_start + i);
      if(sh < 0 || sh >= rates_total)
         return false;
      if(high[sh] > win_hi) win_hi = high[sh];
      if(low[sh]  < win_lo) win_lo = low[sh];
   }

   double win_range = win_hi - win_lo;
   if(!IsFiniteValue(win_range) || win_range <= InpNormEps)
      win_range = InpNormEps;

   for(int i = 0; i < InpSeqLen; i++)
   {
      int sh = ToSeriesShift(rates_total, chrono_start + i);
      double o = open[sh];
      double h = high[sh];
      double l = low[sh];
      double c = close[sh];

      double body = SafeDiv(c - o, win_range, InpNormEps);
      double upper = SafeDiv(h - MathMax(o, c), win_range, InpNormEps);
      double lower = SafeDiv(MathMin(o, c) - l, win_range, InpNormEps);

      int base = i * feat_dim;
      flat[base + 0] = body;
      flat[base + 1] = upper;
      flat[base + 2] = lower;
      if(InpUseDirectionFlag)
         flat[base + 3] = (c >= o ? 1.0 : 0.0);
   }

   if(InpNormalizeFeatures)
   {
      for(int f = 0; f < feat_dim; f++)
      {
         if(InpUseDirectionFlag && f == 3)
            continue;

         double mn = DBL_MAX;
         double mx = -DBL_MAX;
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
            int idx = i * feat_dim + f;
            flat[idx] = SafeDiv(flat[idx] - mn, den, InpNormEps);
         }
      }
   }

   return true;
}

//+------------------------------------------------------------------+
void BuildTarget(const int rates_total,
                 const double &open[],
                 const double &close[],
                 const int future_start_chrono,
                 double &bullish,
                 double &bearish)
{
   int up = 0;
   for(int i = 0; i < InpPredK; i++)
   {
      int sh = ToSeriesShift(rates_total, future_start_chrono + i);
      if(sh >= 0 && sh < rates_total && close[sh] > open[sh])
         up++;
   }

   bullish = (double)up / (double)InpPredK;
   bearish = 1.0 - bullish;

   bullish = MathMin(1.0, MathMax(0.0, bullish));
   bearish = MathMin(1.0, MathMax(0.0, bearish));
}

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

   const int closed_bars = rates_total - 1;
   if(closed_bars <= InpSeqLen + InpPredK)
      return false;

   int chrono_start = MathMax(0, closed_bars - InpHistoryBars);
   int chrono_end_exclusive = closed_bars;
   int possible = chrono_end_exclusive - chrono_start - (InpSeqLen + InpPredK) + 1;
   if(possible <= 0)
      return false;

   int wanted = MathMin(possible, InpMaxSamples);
   int first_sample = chrono_end_exclusive - (InpSeqLen + InpPredK) - wanted + 1;
   if(first_sample < chrono_start)
      first_sample = chrono_start;

   ArrayResize(X, wanted * in_dim);
   ArrayResize(T, wanted * 2);

   double sample[];
   for(int s = 0; s < wanted; s++)
   {
      int win_chrono = first_sample + s;
      if(!BuildWindowFeatures(rates_total, open, high, low, close, win_chrono, sample))
         break;

      int xoff = s * in_dim;
      for(int i = 0; i < in_dim; i++)
         X[xoff + i] = sample[i];

      double bull = 0.5;
      double bear = 0.5;
      BuildTarget(rates_total, open, close, win_chrono + InpSeqLen, bull, bear);
      T[s * 2 + 0] = bull;
      T[s * 2 + 1] = bear;
      batch++;
   }

   if(batch <= 0)
      return false;

   if(batch < wanted)
   {
      ArrayResize(X, batch * in_dim);
      ArrayResize(T, batch * 2);
   }

   return true;
}

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
   if(!BuildTrainingSet(rates_total, open, high, low, close, X, T, batch, in_dim))
   {
      Print("Training set build skipped: insufficient bars");
      return false;
   }

   if(!DN_LoadBatch(g_NetHandle, X, T, batch, in_dim, 2, InpSeqLen))
   {
      Print("DN_LoadBatch failed: ", GetDLLError());
      return false;
   }

   if(!DN_TrainAsync(g_NetHandle, InpTrainEpochs, InpTargetMSE, InpLearningRate, InpWeightDecay))
   {
      Print("DN_TrainAsync failed: ", GetDLLError());
      return false;
   }

   g_IsTraining = true;
   return true;
}

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

   const int closed_bars = rates_total - 1;
   if(closed_bars < InpSeqLen)
      return false;

   int last_chrono = closed_bars - 1;
   int win_start = last_chrono - InpSeqLen + 1;
   if(win_start < 0)
      return false;

   double X[];
   if(!BuildWindowFeatures(rates_total, open, high, low, close, win_start, X))
      return false;

   double Y[];
   ArrayResize(Y, 2);

   if(!DN_PredictBatch(g_NetHandle, X, 1, ArraySize(X), InpSeqLen, Y))
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

   double sum = bull + bear;
   if(sum > InpNormEps)
   {
      bull = bull / sum;
      bear = bear / sum;
   }

   return true;
}

//+------------------------------------------------------------------+
void UpdateProgressComment()
{
   string line = StringFormat("PatternCompletion epoch=%d/%d mb=%d/%d mse=%.6f pct=%.1f eta=%.1fs",
                              g_ProgEpoch, g_ProgTotalEpochs,
                              g_ProgMB, g_ProgTotalMB,
                              g_ProgMSE,
                              g_ProgPercent,
                              g_ProgETASec);
   Comment(line);
}

//+------------------------------------------------------------------+
int OnInit()
{
   SetIndexBuffer(0, g_BullishBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, g_BearishBuffer, INDICATOR_DATA);
   ArraySetAsSeries(g_BullishBuffer, true);
   ArraySetAsSeries(g_BearishBuffer, true);
   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, EMPTY_VALUE);

   IndicatorSetString(INDICATOR_SHORTNAME, "LSTM Pattern Completion Demo");

   if(!InitNetwork())
      return INIT_FAILED;

   EventSetTimer(1);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   Comment("");

   if(g_NetHandle > 0)
   {
      DN_Free(g_NetHandle);
      g_NetHandle = 0;
   }
}

//+------------------------------------------------------------------+
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

   if(st == 1)
      return;

   g_IsTraining = false;
   if(st == 2)
   {
      g_ModelReady = true;
      g_BarsSinceTrain = 0;
      Print("Pattern completion training complete.");
   }
   else
   {
      Print("Pattern completion training error: ", GetDLLError());
   }
}

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

   int start = (prev_calculated > 0 ? prev_calculated - 1 : 0);
   for(int i = start; i < rates_total; i++)
   {
      g_BullishBuffer[i] = EMPTY_VALUE;
      g_BearishBuffer[i] = EMPTY_VALUE;
   }

   bool new_bar = (g_LastBarTime != time[0]);
   if(new_bar)
   {
      g_LastBarTime = time[0];
      g_BarsSinceTrain++;

      if(!g_IsTraining && (!g_ModelReady || g_BarsSinceTrain >= InpRetrainEveryBars))
      {
         if(StartTraining(rates_total, open, high, low, close))
            Print("Pattern completion training started.");
      }
   }

   if(g_ModelReady && !g_IsTraining)
   {
      double bull = 0.5;
      double bear = 0.5;
      if(PredictLatest(rates_total, open, high, low, close, bull, bear))
      {
         g_BullishBuffer[0] = bull;
         g_BearishBuffer[0] = bear;
      }
   }

   if(!g_IsTraining)
      Comment("PatternCompletion ready");

   return rates_total;
}
//+------------------------------------------------------------------+
