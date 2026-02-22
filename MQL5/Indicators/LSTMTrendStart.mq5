#property copyright "MIT"
#property version   "1.00"
#property strict
#property indicator_separate_window
#property indicator_buffers 3
#property indicator_plots   3

#property indicator_label1  "TrendScore"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrDodgerBlue
#property indicator_width1  2

#property indicator_label2  "UpStart"
#property indicator_type2   DRAW_ARROW
#property indicator_color2  clrLime
#property indicator_width2  1

#property indicator_label3  "DownStart"
#property indicator_type3   DRAW_ARROW
#property indicator_color3  clrTomato
#property indicator_width3  1

input int      InpSequenceLength      = 32;
input int      InpHiddenSize          = 24;
input int      InpHorizonBars         = 4;
input int      InpTrainBars           = 700;
input int      InpRetrainEveryNBars   = 25;
input int      InpTrainEpochs         = 80;
input double   InpTargetMSE           = 0.00001;
input double   InpLearningRate         = 0.002;
input double   InpWeightDecay          = 0.00001;
input double   InpSignalThreshold      = 0.0015;
input bool     InpEnableTraining       = true;
input bool     InpVerboseLog           = true;

#import "MQL5GPULibrary_LSTM.dll"
int  DN_Create();
void DN_Free(int h);
bool DN_SetSequenceLength(int h, int seq_len);
bool DN_SetMiniBatchSize(int h, int mbs);
bool DN_AddLayerEx(int h, int in, int out, int act, int ln, double drop);
bool DN_SetGradClip(int h, double clip);
bool DN_LoadBatch(int h, const double &X[], const double &T[], int batch, int in, int out, int l);
bool DN_PredictBatch(int h, const double &X[], int batch, int in, int l, double &Y[]);
bool DN_TrainAsync(int h, int epochs, double target_mse, double lr, double wd);
int  DN_GetTrainingStatus(int h);
void DN_GetTrainingResult(int h, double &out_mse, int &out_epochs);
void DN_StopTraining(int h);
void DN_GetError(short &buf[], int len);
#import

double g_scoreBuffer[];
double g_upBuffer[];
double g_downBuffer[];

int      g_net              = 0;
datetime g_lastBarTime      = 0;
int      g_newBarsCounter   = 0;
int      g_lastTrainStatus  = 0;

string LastDllError()
{
   short raw[256];
   ArrayInitialize(raw, 0);
   DN_GetError(raw, ArraySize(raw));

   string msg = "";
   for(int i = 0; i < ArraySize(raw); ++i)
   {
      if(raw[i] == 0)
         break;
      msg += ShortToString(raw[i]);
   }
   return msg;
}

bool BuildTrainingSet(const double &close[], int rates_total, double &X[], double &T[], int &samples)
{
   int usable = MathMin(InpTrainBars, rates_total - InpHorizonBars - 2);
   samples = usable - InpSequenceLength - InpHorizonBars + 1;
   if(samples < 8)
      return false;

   ArrayResize(X, samples * InpSequenceLength);
   ArrayResize(T, samples);

   int startShift = usable + InpHorizonBars;

   for(int s = 0; s < samples; ++s)
   {
      int endShift = startShift - s;
      int base = s * InpSequenceLength;

      for(int k = 0; k < InpSequenceLength; ++k)
      {
         int idxOld = endShift - (InpSequenceLength - k);
         int idxNew = idxOld - 1;

         if(idxOld < 1 || idxNew < 0)
            return false;

         double prev = close[idxOld];
         double next = close[idxNew];
         double r = (prev != 0.0 ? (next - prev) / prev : 0.0);
         X[base + k] = r;
      }

      int nowIdx = endShift - 1;
      int futIdx = nowIdx - InpHorizonBars;
      if(nowIdx < 0 || futIdx < 0)
         return false;

      double c0 = close[nowIdx];
      double c1 = close[futIdx];
      T[s] = (c0 != 0.0 ? (c1 - c0) / c0 : 0.0);
   }

   return true;
}

bool TrainModel(const double &close[], int rates_total)
{
   if(!InpEnableTraining)
      return true;

   if(DN_GetTrainingStatus(g_net) == 1)
      return true;

   double X[];
   double T[];
   int samples = 0;

   if(!BuildTrainingSet(close, rates_total, X, T, samples))
      return false;

   if(!DN_LoadBatch(g_net, X, T, samples, 1, 1, InpSequenceLength))
   {
      Print("DN_LoadBatch failed: ", LastDllError());
      return false;
   }

   if(!DN_TrainAsync(g_net, InpTrainEpochs, InpTargetMSE, InpLearningRate, InpWeightDecay))
   {
      Print("DN_TrainAsync failed: ", LastDllError());
      return false;
   }

   if(InpVerboseLog)
      PrintFormat("LSTM training started. Samples=%d, seq=%d", samples, InpSequenceLength);

   return true;
}

bool PredictCurrent(const double &close[], int rates_total, double &score)
{
   if(rates_total <= InpSequenceLength + 2)
      return false;

   double X[];
   ArrayResize(X, InpSequenceLength);

   int endShift = 1;
   for(int k = 0; k < InpSequenceLength; ++k)
   {
      int idxOld = endShift + (InpSequenceLength - k);
      int idxNew = idxOld - 1;
      if(idxOld >= rates_total || idxNew >= rates_total)
         return false;

      double prev = close[idxOld];
      double next = close[idxNew];
      X[k] = (prev != 0.0 ? (next - prev) / prev : 0.0);
   }

   double Y[1];
   if(!DN_PredictBatch(g_net, X, 1, 1, InpSequenceLength, Y))
   {
      Print("DN_PredictBatch failed: ", LastDllError());
      return false;
   }

   score = Y[0];
   return true;
}

int OnInit()
{
   SetIndexBuffer(0, g_scoreBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, g_upBuffer, INDICATOR_DATA);
   SetIndexBuffer(2, g_downBuffer, INDICATOR_DATA);

   ArraySetAsSeries(g_scoreBuffer, true);
   ArraySetAsSeries(g_upBuffer, true);
   ArraySetAsSeries(g_downBuffer, true);

   PlotIndexSetInteger(1, PLOT_ARROW, 241);
   PlotIndexSetInteger(2, PLOT_ARROW, 242);

   g_net = DN_Create();
   if(g_net <= 0)
   {
      Print("DN_Create failed: ", LastDllError());
      return INIT_FAILED;
   }

   if(!DN_SetSequenceLength(g_net, InpSequenceLength) ||
      !DN_SetMiniBatchSize(g_net, 1) ||
      !DN_AddLayerEx(g_net, 1, InpHiddenSize, 2, 1, 0.0) ||
      !DN_AddLayerEx(g_net, InpHiddenSize, 1, 0, 0, 0.0) ||
      !DN_SetGradClip(g_net, 1.0))
   {
      Print("LSTM configuration failed: ", LastDllError());
      return INIT_FAILED;
   }

   EventSetTimer(1);
   IndicatorSetString(INDICATOR_SHORTNAME, "LSTM Trend Start (DLL)");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   if(g_net > 0)
   {
      DN_StopTraining(g_net);
      DN_Free(g_net);
      g_net = 0;
   }
}

void OnTimer()
{
   if(g_net <= 0)
      return;

   int status = DN_GetTrainingStatus(g_net);
   if(status != g_lastTrainStatus)
   {
      g_lastTrainStatus = status;
      if(status == 1 && InpVerboseLog)
         Print("LSTM training running...");
      else if(status == 2)
      {
         double mse = 0.0;
         int epochs = 0;
         DN_GetTrainingResult(g_net, mse, epochs);
         PrintFormat("LSTM training complete. epochs=%d mse=%.8f", epochs, mse);
      }
      else if(status == -1)
         Print("LSTM training error: ", LastDllError());
   }
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
   if(rates_total < InpSequenceLength + InpHorizonBars + 20)
      return 0;

   if(prev_calculated == 0)
   {
      ArrayInitialize(g_scoreBuffer, EMPTY_VALUE);
      ArrayInitialize(g_upBuffer, EMPTY_VALUE);
      ArrayInitialize(g_downBuffer, EMPTY_VALUE);
      g_lastBarTime = 0;
   }

   bool isNewBar = (time[0] != g_lastBarTime);
   if(isNewBar)
   {
      g_lastBarTime = time[0];
      g_newBarsCounter++;

      if(g_newBarsCounter == 1 || (InpRetrainEveryNBars > 0 && g_newBarsCounter % InpRetrainEveryNBars == 0))
         TrainModel(close, rates_total);
   }

   double score = 0.0;
   if(PredictCurrent(close, rates_total, score))
   {
      g_scoreBuffer[1] = score;
      g_upBuffer[1] = EMPTY_VALUE;
      g_downBuffer[1] = EMPTY_VALUE;

      if(score > InpSignalThreshold)
         g_upBuffer[1] = score;
      else if(score < -InpSignalThreshold)
         g_downBuffer[1] = score;
   }

   return rates_total;
}
