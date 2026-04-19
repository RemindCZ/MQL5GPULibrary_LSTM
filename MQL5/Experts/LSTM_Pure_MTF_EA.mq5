//+------------------------------------------------------------------+
//| LSTM_Pure_MTF_EA.mq5 v1.00                                       |
//| Pure LSTM EA s MTF featurami + Shannon filtrem                  |
//| GPU-optimized via MQL5GPULibrary_LSTM.dll                       |
//|                                                                  |
//| LICENCE: MIT License                                             |
//| Když to vydělá, super. Když to nevyjde, aspoň jsme si sáhli      |
//| na statistiku místo věštění z kávové sedliny.                    |
//+------------------------------------------------------------------+
#property copyright "Tomáš Bělák Remind"
#property link      "https://remind.cz/"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>

//+------------------------------------------------------------------+
//| DLL Import - Křemík + blesky + trpělivost = inference            |
//+------------------------------------------------------------------+
#import "MQL5GPULibrary_LSTM.dll"
int DN_Create();
void DN_Free(int h);
int DN_SetSequenceLength(int h, int seq_len);
int DN_SetMiniBatchSize(int h, int mbs);
int DN_AddLayerEx(int h, int in_sz, int out_sz, int act, int ln, double drop);
int DN_SetOutputDim(int h, int out_dim);
int DN_SetGradClip(int h, double clip);
int DN_PredictBatch(int h, const double &X[], int batch, int in_dim, int layout, double &Y[]);
int DN_SaveState(int h);
int DN_GetState(int h, char &buf[], int max_len);
int DN_LoadState(int h, const char &buf[]);
void DN_GetError(short &buf[], int len);
#import

//+------------------------------------------------------------------+
//| Konstanty a enumy                                                |
//+------------------------------------------------------------------+
#define FEAT_PER_TF 18
#define NUM_TFS 3
#define FEAT_PER_BAR (FEAT_PER_TF * NUM_TFS)
#define OUTPUT_DIM 2
#define MODEL_MAGIC 0x4C53544D
#define MODEL_META_VER 1
#define LN2_CONST 0.6931471805599453
#define MIN_MODEL_STATE_BYTES 64

enum ENUM_LOG_LEVEL
  {
   LOG_DBG = 0,
   LOG_OK,
   LOG_FAIL
  };

enum ENUM_OPPOSITE_ACTION
  {
   OPP_IGNORE = 0,
   OPP_CLOSE,
   OPP_REVERSE
  };

enum ENUM_POSITION_MODE
  {
   POS_ONE_TRADE = 0,
   POS_ALLOW_REVERSE
  };

//+------------------------------------------------------------------+
//| Inputy                                                           |
//+------------------------------------------------------------------+
input group "=== Multi-Timeframe Inputs ==="
input ENUM_TIMEFRAMES InpTF1 = PERIOD_CURRENT;
input ENUM_TIMEFRAMES InpTF2 = PERIOD_H1;
input ENUM_TIMEFRAMES InpTF3 = PERIOD_H4;
input int InpLookback = 30;

input group "=== LSTM Architecture ==="
input int InpHiddenSize1 = 128;
input int InpHiddenSize2 = 64;
input int InpHiddenSize3 = 32;
input double InpDropout = 0.15;
input double InpGradClip = 5.0;
input int InpMiniBatch = 1;

input group "=== Shannon Entropy ==="
input int InpEntropyFastPeriod = 12;
input int InpEntropySlowPeriod = 32;
input int InpEntropyPriceStep = 1;
input bool InpUseEntropyFilter = true;
input double InpEntropyChaosLevel = 0.65;
input double InpEntropyFilterPower = 0.60;

input group "=== Trading Logic ==="
input double InpBuyThresholdPct = 70.0;
input double InpSellThresholdPct = 30.0;
input double InpDeadZonePct = 4.0;
input double InpMinConfidence = 0.12;
input ENUM_POSITION_MODE InpPositionMode = POS_ONE_TRADE;
input ENUM_OPPOSITE_ACTION InpOppositeSignalAction = OPP_REVERSE;

input group "=== Risk Management ==="
input double InpFixedLot = 0.10;
input bool InpUseRiskPercent = false;
input double InpRiskPercent = 1.0;
input int InpATRPeriod = 14;
input double InpATRSLMultiplier = 1.5;
input double InpATRTPMultiplier = 2.2;
input double InpRiskRewardRatio = 1.6;
input int InpMaxSpreadPoints = 40;
input int InpDeviationPoints = 20;
input long InpMagicNumber = 590501;

input group "=== Model / Prediction ==="
input bool InpSaveModel = true;
input string InpModelPrefix = "REMIND";
input bool InpInferenceOnly = true;
input bool InpAttemptRetrain = false; // Zatím jen architektura hooku

input group "=== Advanced ==="
input bool InpVerboseLog = false;
input int InpTimerSeconds = 0;
input int InpTradeCooldownSec = 5;

//+------------------------------------------------------------------+
//| Struktury                                                        |
//+------------------------------------------------------------------+
struct ModelMetaEA
  {
   int               magic;
   int               metaVersion;
   int               featPerBar;
   int               outputDim;
   int               lookback;
   int               hidden1;
   int               hidden2;
   int               hidden3;
   int               tf1;
   int               tf2;
   int               tf3;
   int               entropyFast;
   int               entropySlow;
   int               entropyStep;
   int               stateSize;
  };

struct PredictionSnapshot
  {
   datetime          sourceBarTime;
   double            pBull;
   double            pBear;
   double            entropyFast;
   double            entropySlow;
   double            entropyDelta;
   bool              valid;
  };

//+------------------------------------------------------------------+
//| Globální stav                                                    |
//+------------------------------------------------------------------+
CTrade g_Trade;
int g_NetHandle = 0;
bool g_ModelReady = false;
bool g_LoadedFromFile = false;
string g_ModelFilePath = "";
datetime g_LastProcessedClosedBar = 0;
datetime g_LastTradeTime = 0;
double g_ATRMean = 0.0;
PredictionSnapshot g_LastPred;

//+------------------------------------------------------------------+
//| Utility log                                                      |
//+------------------------------------------------------------------+
void LogMessage(ENUM_LOG_LEVEL level, string msg)
  {
   if(level == LOG_DBG && !InpVerboseLog)
      return;
   string pfx = "[DBG] ";
   if(level == LOG_OK)
      pfx = "[OK] ";
   else if(level == LOG_FAIL)
      pfx = "[FAIL] ";
   Print(pfx + msg);
  }

//+------------------------------------------------------------------+
//| DLL chyba                                                        |
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
   return (StringLen(s) == 0) ? "unknown DLL error (GPU se dnes tváří tajemně)" : s;
  }

//+------------------------------------------------------------------+
//| Vstupní sanity check                                             |
//+------------------------------------------------------------------+
bool ValidateInputs()
  {
   if(InpLookback < 8 || InpLookback > 256)
     {
      LogMessage(LOG_FAIL, "InpLookback musí být 8..256");
      return false;
     }
   if(InpHiddenSize1 < 4 || InpHiddenSize2 < 4 || InpHiddenSize3 < 0)
     {
      LogMessage(LOG_FAIL, "Hidden sizes jsou mimo rozumný rozsah");
      return false;
     }
   if(InpBuyThresholdPct <= 50.0 || InpBuyThresholdPct >= 99.0 || InpSellThresholdPct >= 50.0 || InpSellThresholdPct <= 1.0)
     {
      LogMessage(LOG_FAIL, "Buy/Sell threshold invalid (čekáme >50 a <50)");
      return false;
     }
   if(InpDeadZonePct < 0.0 || InpDeadZonePct > 20.0)
      return false;
   if(InpMinConfidence < 0.0 || InpMinConfidence > 0.49)
      return false;
   if(InpATRPeriod < 2)
      return false;
   return true;
  }

//+------------------------------------------------------------------+
//| Nový uzavřený bar?                                               |
//+------------------------------------------------------------------+
bool IsNewClosedBar(datetime &closedBarTime)
  {
   datetime t[];
   ArraySetAsSeries(t, true);
   if(CopyTime(_Symbol, PERIOD_CURRENT, 1, 1, t) != 1)
      return false;
   closedBarTime = t[0];
   if(closedBarTime <= 0)
      return false;
   if(closedBarTime == g_LastProcessedClosedBar)
      return false;
   return true;
  }

//+------------------------------------------------------------------+
//| Candle encoding                                                  |
//+------------------------------------------------------------------+
int EncodeCandle(double open, double high, double low, double close)
  {
   double range = high - low;
   if(range < _Point)
      return 4;
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
         result = 8;
      else if(bodyRatio > 0.30)
         result = 7;
      else if(lowerShadow > upperShadow)
         result = 5;
      else
         result = 6;
     }
   else
     {
      if(bodyRatio > 0.70)
         result = 0;
      else if(bodyRatio > 0.30)
         result = 1;
      else if(upperShadow > lowerShadow)
         result = 2;
      else
         result = 3;
     }
   if(result < 0 || result > 8)
      result = 4;
   return result;
  }

//+------------------------------------------------------------------+
//| Feature extraction helpers                                       |
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
   return 0;
  }

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
      else if(st == 2)
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
   return MathMax(0.0, MathMin(1.0, H / Hmax));
  }

double ApplyEntropyConfidenceFilter(double pBull, double entropySlow)
  {
   if(!InpUseEntropyFilter)
      return pBull;
   if(entropySlow <= InpEntropyChaosLevel)
      return pBull;
   double excess = (entropySlow - InpEntropyChaosLevel) / MathMax(1e-8, (1.0 - InpEntropyChaosLevel));
   excess = MathMax(0.0, MathMin(1.0, excess));
   double strength = MathMax(0.0, MathMin(1.0, InpEntropyFilterPower));
   double mix = excess * strength;
   return pBull * (1.0 - mix) + 0.5 * mix;
  }

bool IsTFBarValid(ENUM_TIMEFRAMES tf, int sh, datetime refTime)
  {
   if(sh < 0)
      return false;
   datetime barTime = iTime(_Symbol, tf, sh);
   long tfSeconds = PeriodSeconds(tf);
   if(tfSeconds <= 0)
      return true;
   return (MathAbs((long)barTime - (long)refTime) <= tfSeconds * 2);
  }

void GetTFData(ENUM_TIMEFRAMES tf, datetime t, double &feats[], int offset)
  {
   int sh = iBarShift(_Symbol, tf, t, false);
   if(sh < 0 || !IsTFBarValid(tf, sh, t))
     {
      for(int i = 0; i < FEAT_PER_TF; i++)
         feats[offset + i] = 0.0;
      return;
     }

   double o = iOpen(_Symbol, tf, sh), h = iHigh(_Symbol, tf, sh), l = iLow(_Symbol, tf, sh), c = iClose(_Symbol, tf, sh);
   long vol = iTickVolume(_Symbol, tf, sh);
   long prevVol = (sh + 1 < iBars(_Symbol, tf)) ? iTickVolume(_Symbol, tf, sh + 1) : 0;
   double range = h - l;
   if(range < _Point)
      range = _Point;

   int sym = EncodeCandle(o, h, l, c);
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
   feats[offset + 15] = eFast;
   feats[offset + 16] = eSlow;
   feats[offset + 17] = MathMax(-1.0, MathMin(1.0, eFast - eSlow));
  }

bool BuildSequenceInput(datetime sourceBarTime, double &X[])
  {
   int inDim = InpLookback * FEAT_PER_BAR;
   ArrayResize(X, inDim);
   ArrayInitialize(X, 0.0);
   for(int step = 0; step < InpLookback; step++)
     {
      int shMain = iBarShift(_Symbol, PERIOD_CURRENT, sourceBarTime, false) + step;
      if(shMain < 1)
         return false;
      datetime t = iTime(_Symbol, PERIOD_CURRENT, shMain);
      int rowOff = step * FEAT_PER_BAR;
      GetTFData(InpTF1, t, X, rowOff);
      GetTFData(InpTF2, t, X, rowOff + FEAT_PER_TF);
      GetTFData(InpTF3, t, X, rowOff + FEAT_PER_TF * 2);
     }
   return true;
  }

//+------------------------------------------------------------------+
//| Model init/load/save                                             |
//+------------------------------------------------------------------+
void FillModelMeta(ModelMetaEA &m, int stateSize)
  {
   m.magic = MODEL_MAGIC;
   m.metaVersion = MODEL_META_VER;
   m.featPerBar = FEAT_PER_BAR;
   m.outputDim = OUTPUT_DIM;
   m.lookback = InpLookback;
   m.hidden1 = InpHiddenSize1;
   m.hidden2 = InpHiddenSize2;
   m.hidden3 = InpHiddenSize3;
   m.tf1 = (int)InpTF1;
   m.tf2 = (int)InpTF2;
   m.tf3 = (int)InpTF3;
   m.entropyFast = InpEntropyFastPeriod;
   m.entropySlow = InpEntropySlowPeriod;
   m.entropyStep = InpEntropyPriceStep;
   m.stateSize = stateSize;
  }

bool IsModelMetaCompatible(const ModelMetaEA &m)
  {
   if(m.magic != MODEL_MAGIC || m.metaVersion != MODEL_META_VER)
      return false;
   if(m.featPerBar != FEAT_PER_BAR || m.outputDim != OUTPUT_DIM || m.lookback != InpLookback)
      return false;
   if(m.hidden1 != InpHiddenSize1 || m.hidden2 != InpHiddenSize2 || m.hidden3 != InpHiddenSize3)
      return false;
   if(m.tf1 != (int)InpTF1 || m.tf2 != (int)InpTF2 || m.tf3 != (int)InpTF3)
      return false;
   if(m.entropyFast != InpEntropyFastPeriod || m.entropySlow != InpEntropySlowPeriod || m.entropyStep != InpEntropyPriceStep)
      return false;
   if(m.stateSize < MIN_MODEL_STATE_BYTES)
      return false;
   return true;
  }

string TFToShortString(ENUM_TIMEFRAMES tf)
  {
   if(tf == PERIOD_CURRENT)
      tf = (ENUM_TIMEFRAMES)_Period;
   return EnumToString(tf);
  }

string BuildModelFileName()
  {
   string sym = _Symbol;
   StringReplace(sym, ".", "");
   StringReplace(sym, "/", "");
   StringReplace(sym, "#", "");
   StringReplace(sym, " ", "");
   string layers = StringFormat("LSTM%dx%d", InpHiddenSize1, InpHiddenSize2);
   if(InpHiddenSize3 > 0)
      layers += "x" + IntegerToString(InpHiddenSize3);
   return StringFormat("%s_%s_%s-%s-%s_EA_L%d_%s_F%dx%d_ENT%d-%d_V100.lstm",
                       InpModelPrefix, sym, TFToShortString(InpTF1), TFToShortString(InpTF2), TFToShortString(InpTF3),
                       InpLookback, layers, FEAT_PER_BAR, OUTPUT_DIM, InpEntropyFastPeriod, InpEntropySlowPeriod);
  }

bool InitNetwork()
  {
   g_NetHandle = DN_Create();
   if(g_NetHandle == 0)
     {
      LogMessage(LOG_FAIL, "DN_Create fail: " + GetDLLError());
      return false;
     }
   DN_SetSequenceLength(g_NetHandle, InpLookback);
   DN_SetMiniBatchSize(g_NetHandle, InpMiniBatch);
   DN_SetGradClip(g_NetHandle, InpGradClip);

   if(!DN_AddLayerEx(g_NetHandle, FEAT_PER_BAR, InpHiddenSize1, 0, 0, InpDropout))
      return false;
   if(!DN_AddLayerEx(g_NetHandle, 0, InpHiddenSize2, 0, 0, InpDropout * 0.5))
      return false;
   if(InpHiddenSize3 > 0 && !DN_AddLayerEx(g_NetHandle, 0, InpHiddenSize3, 0, 0, 0.0))
      return false;
   if(!DN_SetOutputDim(g_NetHandle, OUTPUT_DIM))
      return false;

   LogMessage(LOG_OK, StringFormat("Network ready: %d -> [%d,%d,%d] -> %d", FEAT_PER_BAR, InpHiddenSize1, InpHiddenSize2, InpHiddenSize3, OUTPUT_DIM));
   return true;
  }

bool SaveModel()
  {
   if(g_NetHandle <= 0 || !g_ModelReady || StringLen(g_ModelFilePath) == 0)
      return false;
   int stateSize = DN_SaveState(g_NetHandle);
   if(stateSize <= 0)
      return false;

   char stateBuf[];
   ArrayResize(stateBuf, stateSize);
   if(!DN_GetState(g_NetHandle, stateBuf, stateSize))
      return false;

   ModelMetaEA meta;
   FillModelMeta(meta, stateSize);

   int fh = FileOpen(g_ModelFilePath, FILE_WRITE | FILE_BIN | FILE_COMMON);
   if(fh == INVALID_HANDLE)
      return false;

   FileWriteInteger(fh, meta.magic, INT_VALUE);
   FileWriteInteger(fh, meta.metaVersion, INT_VALUE);
   FileWriteInteger(fh, meta.featPerBar, INT_VALUE);
   FileWriteInteger(fh, meta.outputDim, INT_VALUE);
   FileWriteInteger(fh, meta.lookback, INT_VALUE);
   FileWriteInteger(fh, meta.hidden1, INT_VALUE);
   FileWriteInteger(fh, meta.hidden2, INT_VALUE);
   FileWriteInteger(fh, meta.hidden3, INT_VALUE);
   FileWriteInteger(fh, meta.tf1, INT_VALUE);
   FileWriteInteger(fh, meta.tf2, INT_VALUE);
   FileWriteInteger(fh, meta.tf3, INT_VALUE);
   FileWriteInteger(fh, meta.entropyFast, INT_VALUE);
   FileWriteInteger(fh, meta.entropySlow, INT_VALUE);
   FileWriteInteger(fh, meta.entropyStep, INT_VALUE);
   FileWriteInteger(fh, meta.stateSize, INT_VALUE);
   FileWriteArray(fh, stateBuf, 0, stateSize);
   FileClose(fh);

   LogMessage(LOG_OK, StringFormat("Model saved (%d bytes): %s", stateSize, g_ModelFilePath));
   return true;
  }

bool LoadModel()
  {
   if(g_NetHandle <= 0 || StringLen(g_ModelFilePath) == 0)
      return false;

   int fh = FileOpen(g_ModelFilePath, FILE_READ | FILE_BIN | FILE_COMMON);
   if(fh == INVALID_HANDLE)
      return false;

   ulong fs = FileSize(fh);
   if(fs < 60 || fs > 100000000)
     {
      FileClose(fh);
      LogMessage(LOG_FAIL, "LoadModel: invalid file size");
      return false;
     }

   ModelMetaEA meta;
   meta.magic = FileReadInteger(fh, INT_VALUE);
   meta.metaVersion = FileReadInteger(fh, INT_VALUE);
   meta.featPerBar = FileReadInteger(fh, INT_VALUE);
   meta.outputDim = FileReadInteger(fh, INT_VALUE);
   meta.lookback = FileReadInteger(fh, INT_VALUE);
   meta.hidden1 = FileReadInteger(fh, INT_VALUE);
   meta.hidden2 = FileReadInteger(fh, INT_VALUE);
   meta.hidden3 = FileReadInteger(fh, INT_VALUE);
   meta.tf1 = FileReadInteger(fh, INT_VALUE);
   meta.tf2 = FileReadInteger(fh, INT_VALUE);
   meta.tf3 = FileReadInteger(fh, INT_VALUE);
   meta.entropyFast = FileReadInteger(fh, INT_VALUE);
   meta.entropySlow = FileReadInteger(fh, INT_VALUE);
   meta.entropyStep = FileReadInteger(fh, INT_VALUE);
   meta.stateSize = FileReadInteger(fh, INT_VALUE);

   if(!IsModelMetaCompatible(meta))
     {
      FileClose(fh);
      LogMessage(LOG_FAIL, "LoadModel: metadata mismatch (jiná architektura nebo TF)");
      return false;
     }

   char buf[];
   ArrayResize(buf, meta.stateSize + 1);
   int br = (int)FileReadArray(fh, buf, 0, meta.stateSize);
   FileClose(fh);
   if(br != meta.stateSize)
      return false;
   buf[meta.stateSize] = 0;

   if(!DN_LoadState(g_NetHandle, buf))
     {
      LogMessage(LOG_FAIL, "DN_LoadState fail: " + GetDLLError());
      return false;
     }

   LogMessage(LOG_OK, StringFormat("Model loaded (%d bytes): %s", meta.stateSize, g_ModelFilePath));
   return true;
  }

//+------------------------------------------------------------------+
//| Inference                                                        |
//+------------------------------------------------------------------+
bool ComputePrediction(datetime barTime, PredictionSnapshot &out)
  {
   out.valid = false;
   if(!g_ModelReady || g_NetHandle <= 0)
      return false;

   double X[];
   if(!BuildSequenceInput(barTime, X))
     {
      LogMessage(LOG_DBG, "BuildSequenceInput fail");
      return false;
     }

   double Y[];
   ArrayResize(Y, OUTPUT_DIM);
   if(!DN_PredictBatch(g_NetHandle, X, 1, InpLookback * FEAT_PER_BAR, 0, Y))
     {
      LogMessage(LOG_FAIL, "DN_PredictBatch fail: " + GetDLLError());
      return false;
     }

   double pBull = MathMax(0.0, MathMin(1.0, Y[0]));
   double pBear = MathMax(0.0, MathMin(1.0, Y[1]));
   double total = pBull + pBear;
   if(total > 1e-9)
     {
      pBull /= total;
      pBear /= total;
     }
   else
     {
      pBull = 0.5;
      pBear = 0.5;
     }

   int sh = iBarShift(_Symbol, InpTF1 == PERIOD_CURRENT ? PERIOD_CURRENT : InpTF1, barTime, false);
   if(sh < 0)
      sh = 1;
   double eFast = CalculateShannonEntropyTF(InpTF1 == PERIOD_CURRENT ? PERIOD_CURRENT : InpTF1, sh, InpEntropyFastPeriod, InpEntropyPriceStep);
   double eSlow = CalculateShannonEntropyTF(InpTF1 == PERIOD_CURRENT ? PERIOD_CURRENT : InpTF1, sh, InpEntropySlowPeriod, InpEntropyPriceStep);

   pBull = ApplyEntropyConfidenceFilter(pBull, eSlow);
   pBull = MathMax(0.0, MathMin(1.0, pBull));

   out.sourceBarTime = barTime;
   out.pBull = pBull;
   out.pBear = 1.0 - pBull;
   out.entropyFast = eFast;
   out.entropySlow = eSlow;
   out.entropyDelta = eFast - eSlow;
   out.valid = true;
   return true;
  }

//+------------------------------------------------------------------+
//| Money management                                                 |
//+------------------------------------------------------------------+
double GetATRValue()
  {
   int hATR = iATR(_Symbol, PERIOD_CURRENT, InpATRPeriod);
   if(hATR == INVALID_HANDLE)
      return 0.0;
   double b[];
   ArraySetAsSeries(b, true);
   int copied = CopyBuffer(hATR, 0, 1, 1, b);
   IndicatorRelease(hATR);
   if(copied != 1)
      return 0.0;
   return MathMax(b[0], _Point * 5.0);
  }

void UpdateATRMean()
  {
   int sample = MathMin(200, iBars(_Symbol, PERIOD_CURRENT) - 2);
   if(sample < 20)
     {
      g_ATRMean = _Point * 100;
      return;
     }
   double sum = 0.0;
   int cnt = 0;
   for(int i = 1; i <= sample; i++)
     {
      double h = iHigh(_Symbol, PERIOD_CURRENT, i);
      double l = iLow(_Symbol, PERIOD_CURRENT, i);
      double pc = iClose(_Symbol, PERIOD_CURRENT, i + 1);
      double tr = MathMax(h - l, MathMax(MathAbs(h - pc), MathAbs(l - pc)));
      sum += tr;
      cnt++;
     }
   g_ATRMean = (cnt > 0) ? MathMax(sum / cnt, _Point) : (_Point * 100);
  }

double NormalizeVolumeBySymbol(double vol)
  {
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(step <= 0.0)
      step = minLot;
   vol = MathMax(minLot, MathMin(maxLot, vol));
   vol = MathFloor(vol / step) * step;
   return NormalizeDouble(vol, 2);
  }

double ComputeLotByRisk(double slPoints)
  {
   if(!InpUseRiskPercent || InpRiskPercent <= 0.0)
      return NormalizeVolumeBySymbol(InpFixedLot);

   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskMoney = balance * InpRiskPercent / 100.0;
   if(riskMoney <= 0.0 || slPoints <= 0.0)
      return NormalizeVolumeBySymbol(InpFixedLot);

   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tickValue <= 0.0 || tickSize <= 0.0)
      return NormalizeVolumeBySymbol(InpFixedLot);

   double moneyPerPointPerLot = tickValue * (_Point / tickSize);
   if(moneyPerPointPerLot <= 0.0)
      return NormalizeVolumeBySymbol(InpFixedLot);

   double lots = riskMoney / (slPoints * moneyPerPointPerLot);
   return NormalizeVolumeBySymbol(lots);
  }

bool IsSpreadOK()
  {
   double spreadPts = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID)) / _Point;
   if(spreadPts > InpMaxSpreadPoints)
     {
      LogMessage(LOG_DBG, StringFormat("Spread high: %.1f > %d", spreadPts, InpMaxSpreadPoints));
      return false;
     }
   return true;
  }

//+------------------------------------------------------------------+
//| Pozice                                                           |
//+------------------------------------------------------------------+
int FindPositionTypeByMagic()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(!PositionSelectByTicket(PositionGetTicket(i)))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      return (int)PositionGetInteger(POSITION_TYPE);
     }
   return -1;
  }

bool CloseOurPositions()
  {
   bool allOk = true;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      if(!g_Trade.PositionClose(ticket, InpDeviationPoints))
        {
         allOk = false;
         LogMessage(LOG_FAIL, StringFormat("Close fail #%I64u rc=%d", ticket, g_Trade.ResultRetcode()));
        }
     }
   return allOk;
  }

//+------------------------------------------------------------------+
//| Obchodní logika                                                  |
//+------------------------------------------------------------------+
int EvaluateSignal(const PredictionSnapshot &p)
  {
   if(!p.valid)
      return 0;
   double pbPct = p.pBull * 100.0;
   double conf = MathAbs(p.pBull - 0.5);

   if(conf < InpMinConfidence)
      return 0;
   if(pbPct >= 50.0 - InpDeadZonePct && pbPct <= 50.0 + InpDeadZonePct)
      return 0;
   if(pbPct >= InpBuyThresholdPct)
      return 1;
   if(pbPct <= InpSellThresholdPct)
      return -1;
   return 0;
  }

bool BuildOrderPrices(bool isBuy, double &price, double &sl, double &tp, double &slPoints)
  {
   double atr = GetATRValue();
   if(atr <= 0.0)
      return false;

   price = isBuy ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double slDist = atr * MathMax(0.1, InpATRSLMultiplier);
   double tpDist = atr * MathMax(0.1, InpATRTPMultiplier);

   if(InpRiskRewardRatio > 0.01)
      tpDist = slDist * InpRiskRewardRatio;

   sl = isBuy ? (price - slDist) : (price + slDist);
   tp = isBuy ? (price + tpDist) : (price - tpDist);

   int stopLevel = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   int freezeLevel = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL);
   double minDist = MathMax(stopLevel, freezeLevel) * _Point;

   if(MathAbs(price - sl) < minDist)
      sl = isBuy ? (price - minDist) : (price + minDist);
   if(MathAbs(tp - price) < minDist)
      tp = isBuy ? (price + minDist) : (price - minDist);

   price = NormalizeDouble(price, _Digits);
   sl = NormalizeDouble(sl, _Digits);
   tp = NormalizeDouble(tp, _Digits);
   slPoints = MathAbs(price - sl) / _Point;
   return (slPoints > 0.0);
  }

bool OpenSignalTrade(int signal, const PredictionSnapshot &p)
  {
   if(signal == 0)
      return false;
   if(!IsSpreadOK())
      return false;

   if(g_LastTradeTime > 0 && (TimeCurrent() - g_LastTradeTime) < InpTradeCooldownSec)
      return false;

   bool isBuy = (signal > 0);
   double price, sl, tp, slPoints;
   if(!BuildOrderPrices(isBuy, price, sl, tp, slPoints))
     {
      LogMessage(LOG_FAIL, "BuildOrderPrices fail");
      return false;
     }

   double lots = ComputeLotByRisk(slPoints);
   if(lots <= 0.0)
      return false;

   g_Trade.SetExpertMagicNumber(InpMagicNumber);
   g_Trade.SetDeviationInPoints(InpDeviationPoints);

   string cmt = StringFormat("LSTM_EA pBull=%.1f%% eS=%.2f", p.pBull * 100.0, p.entropySlow);
   bool ok = isBuy ? g_Trade.Buy(lots, _Symbol, price, sl, tp, cmt)
                   : g_Trade.Sell(lots, _Symbol, price, sl, tp, cmt);

   if(!ok)
     {
      LogMessage(LOG_FAIL, StringFormat("Order fail rc=%d err=%d", g_Trade.ResultRetcode(), GetLastError()));
      return false;
     }

   g_LastTradeTime = TimeCurrent();
   LogMessage(LOG_OK, StringFormat("%s %.2f lot | pBull=%.1f%% pBear=%.1f%%",
                                   isBuy ? "BUY" : "SELL", lots, p.pBull * 100.0, p.pBear * 100.0));
   return true;
  }

void ProcessTradingDecision(const PredictionSnapshot &p)
  {
   int signal = EvaluateSignal(p);
   int posType = FindPositionTypeByMagic();

   if(posType == -1)
     {
      if(signal != 0)
         OpenSignalTrade(signal, p);
      return;
     }

   bool posBuy = (posType == POSITION_TYPE_BUY);
   bool opposite = (signal > 0 && !posBuy) || (signal < 0 && posBuy);
   bool sameDir = (signal > 0 && posBuy) || (signal < 0 && !posBuy);

   if(sameDir)
     {
      LogMessage(LOG_DBG, "Pozice už běží stejným směrem, nepřiléváme olej do jednoho hrnce.");
      return;
     }

   if(opposite)
     {
      if(InpOppositeSignalAction == OPP_IGNORE)
         return;
      if(InpOppositeSignalAction == OPP_CLOSE || InpOppositeSignalAction == OPP_REVERSE)
        {
         if(!CloseOurPositions())
            return;
         if(InpOppositeSignalAction == OPP_REVERSE || InpPositionMode == POS_ALLOW_REVERSE)
            OpenSignalTrade(signal, p);
        }
      return;
     }

   // signal == 0 -> defaultně nic, pozici držíme
  }

//+------------------------------------------------------------------+
//| Lifecycle                                                        |
//+------------------------------------------------------------------+
int OnInit()
  {
   if(!ValidateInputs())
      return INIT_PARAMETERS_INCORRECT;

   g_ModelFilePath = BuildModelFileName();
   UpdateATRMean();

   g_Trade.SetExpertMagicNumber(InpMagicNumber);
   g_Trade.SetDeviationInPoints(InpDeviationPoints);

   if(!InitNetwork())
      return INIT_FAILED;

   if(LoadModel())
     {
      g_ModelReady = true;
      g_LoadedFromFile = true;
      LogMessage(LOG_OK, "Model načten ze souboru, jdeme na věc.");
     }
   else
     {
      g_ModelReady = false;
      LogMessage(LOG_FAIL, "Model soubor nenalezen / nekompatibilní. EA je v režimu WAIT.");
      if(!InpInferenceOnly && InpAttemptRetrain)
         LogMessage(LOG_DBG, "Retrain hook připraven, ale v této verzi schválně vypnutý.");
     }

   if(InpTimerSeconds > 0)
      EventSetTimer(InpTimerSeconds);

   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   if(InpSaveModel && g_ModelReady)
      SaveModel();
   if(g_NetHandle > 0)
     {
      DN_Free(g_NetHandle);
      g_NetHandle = 0;
     }
  }

void RunBarCycle()
  {
   datetime closedBarTime;
   if(!IsNewClosedBar(closedBarTime))
      return;

   g_LastProcessedClosedBar = closedBarTime;
   UpdateATRMean();

   if(!g_ModelReady)
     {
      LogMessage(LOG_DBG, "Model není ready, obchodování stojí.");
      return;
     }

   PredictionSnapshot p;
   if(!ComputePrediction(closedBarTime, p))
      return;

   g_LastPred = p;
   LogMessage(LOG_DBG, StringFormat("bar=%s pBull=%.2f%% eFast=%.2f eSlow=%.2f",
                                    TimeToString(closedBarTime, TIME_DATE | TIME_MINUTES),
                                    p.pBull * 100.0, p.entropyFast, p.entropySlow));

   ProcessTradingDecision(p);
  }

void OnTick()
  {
   RunBarCycle();
  }

void OnTimer()
  {
   RunBarCycle();
  }
//+------------------------------------------------------------------+
