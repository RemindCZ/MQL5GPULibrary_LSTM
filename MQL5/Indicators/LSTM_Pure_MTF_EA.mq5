//+------------------------------------------------------------------+
//| LSTM_Pure_MTF_EA.mq5 v2.10                                       |
//| Pure LSTM EA s MTF featurami + Shannon filtrem                   |
//| GPU-optimized via MQL5GPULibrary_LSTM.dll                        |
//|                                                                  |
//| LICENCE: MIT License                                             |
//| Copyright: Tomáš Bělák – Remind (https://remind.cz/)            |
//+------------------------------------------------------------------+
#property copyright "Tomáš Bělák Remind"
#property link      "https://remind.cz/"
#property version   "2.10"
#property strict

#include <Trade/Trade.mqh>

//+------------------------------------------------------------------+
//| DLL Import – signatury dle aktuálního exportu MQL5GPULibrary_LSTM |
//+------------------------------------------------------------------+
#import "MQL5GPULibrary_LSTM.dll"
   int    DN_Create();
   void   DN_Free(int h);
   int    DN_SetSequenceLength(int h, int seq_len);
   int    DN_SetMiniBatchSize(int h, int mbs);
   // act: typ vrstvy (0=LSTM), ln: layer-norm flag, drop: dropout rate
   int    DN_AddLayerEx(int h, int in_sz, int out_sz, int act, int ln, double drop);
   int    DN_SetOutputDim(int h, int out_dim);
   int    DN_SetGradClip(int h, double clip);
   // DN_PredictBatch: layout=0 => [batch, out_dim] (flat), Y musí mít batch*out_dim prvků
   int    DN_PredictBatch(int h, const double &X[], int batch, int in_dim, int layout, double &Y[]);
   int    DN_SaveState(int h);
   int    DN_GetState(int h, char &buf[], int max_len);
   int    DN_LoadState(int h, const char &buf[]);
   void   DN_GetError(short &buf[], int len);
#import

//+------------------------------------------------------------------+
//| Konstanty                                                        |
//+------------------------------------------------------------------+
#define FEAT_PER_TF           18
#define NUM_TFS               3
#define FEAT_PER_BAR          (FEAT_PER_TF * NUM_TFS)   // = 54
#define OUTPUT_DIM            2
#define LN2_CONST             0.6931471805599453
#define MIN_MODEL_STATE_BYTES 64
#define DLL_ERR_BUF_LEN       512
#define FILE_SIZE_MAX         100000000

enum ENUM_MODEL_CONST
{
   MODEL_MAGIC    = 0x4C53544D,
   MODEL_META_VER = 1
};

//+------------------------------------------------------------------+
//| Enumy                                                            |
//+------------------------------------------------------------------+
enum ENUM_LOG_LEVEL
{
   LOG_DBG  = 0,
   LOG_OK   = 1,
   LOG_FAIL = 2
};

enum ENUM_OPPOSITE_ACTION
{
   OPP_IGNORE  = 0,
   OPP_CLOSE   = 1,
   OPP_REVERSE = 2
};

enum ENUM_POSITION_MODE
{
   POS_ONE_TRADE     = 0,
   POS_ALLOW_REVERSE = 1
};

//+------------------------------------------------------------------+
//| Vstupní parametry                                                |
//+------------------------------------------------------------------+
input group "=== Multi-Timeframe ==="
input ENUM_TIMEFRAMES InpTF1       = PERIOD_CURRENT;
input ENUM_TIMEFRAMES InpTF2       = PERIOD_H1;
input ENUM_TIMEFRAMES InpTF3       = PERIOD_H4;
input int             InpLookback  = 30;

input group "=== LSTM Architektura ==="
input int    InpHiddenSize1 = 128;
input int    InpHiddenSize2 = 64;
input int    InpHiddenSize3 = 32;
input double InpDropout     = 0.15;
input double InpGradClip    = 5.0;
input int    InpMiniBatch   = 1;

input group "=== Shannon Entropie ==="
input int    InpEntropyFastPeriod  = 12;
input int    InpEntropySlowPeriod  = 32;
input int    InpEntropyPriceStep   = 1;
input bool   InpUseEntropyFilter   = true;
input double InpEntropyChaosLevel  = 0.65;
input double InpEntropyFilterPower = 0.60;

input group "=== Obchodní logika ==="
input double               InpBuyThresholdPct       = 70.0;
input double               InpSellThresholdPct      = 30.0;
input double               InpDeadZonePct           = 4.0;
input double               InpMinConfidence         = 0.12;
input ENUM_POSITION_MODE   InpPositionMode          = POS_ONE_TRADE;
input ENUM_OPPOSITE_ACTION InpOppositeSignalAction  = OPP_REVERSE;

input group "=== Risk Management ==="
input double InpFixedLot          = 0.10;
input bool   InpUseRiskPercent    = false;
input double InpRiskPercent       = 1.0;
input int    InpATRPeriod         = 14;
input double InpATRSLMultiplier   = 1.5;
input double InpATRTPMultiplier   = 2.2;
input double InpRiskRewardRatio   = 1.6;
input int    InpMaxSpreadPoints   = 40;
input int    InpDeviationPoints   = 20;
input long   InpMagicNumber       = 590501;

input group "=== Model ==="
input bool   InpSaveModel      = true;
input string InpModelPrefix    = "REMIND";
input bool   InpInferenceOnly  = true;
input bool   InpAttemptRetrain = false;

input group "=== Pokročilé ==="
input bool InpVerboseLog       = false;
input int  InpTimerSeconds     = 0;
input int  InpTradeCooldownSec = 5;

//+------------------------------------------------------------------+
//| Struktury                                                        |
//+------------------------------------------------------------------+

// Metadata ukládaná spolu se stavem modelu do souboru
struct ModelMetaEA
{
   int magic;
   int metaVersion;
   int featPerBar;
   int outputDim;
   int lookback;
   int hidden1;
   int hidden2;
   int hidden3;
   int tf1;
   int tf2;
   int tf3;
   int entropyFast;
   int entropySlow;
   int entropyStep;
   int stateSize;

   void Reset()
   {
      magic = metaVersion = featPerBar = outputDim = lookback = 0;
      hidden1 = hidden2 = hidden3 = 0;
      tf1 = tf2 = tf3 = 0;
      entropyFast = entropySlow = entropyStep = stateSize = 0;
   }
};

// Snapshot posledního predikčního výsledku
struct PredictionSnapshot
{
   datetime sourceBarTime;
   double   pBull;
   double   pBear;
   double   entropyFast;
   double   entropySlow;
   double   entropyDelta;
   bool     valid;

   void Reset()
   {
      sourceBarTime = 0;
      pBull = pBear = 0.5;
      entropyFast = entropySlow = entropyDelta = 0.0;
      valid = false;
   }
};

//+------------------------------------------------------------------+
//| Globální stav EA                                                 |
//+------------------------------------------------------------------+
CTrade             g_Trade;
int                g_NetHandle              = 0;
bool               g_ModelReady             = false;
bool               g_LoadedFromFile         = false;
string             g_ModelFilePath          = "";
datetime           g_LastProcessedClosedBar = 0;
datetime           g_LastTradeTime          = 0;
double             g_ATRMean                = 0.0;
int                g_ATRHandle              = INVALID_HANDLE;  // cachovaný ATR handle
PredictionSnapshot g_LastPred;

//+------------------------------------------------------------------+
//| Sekce: Utility                                                   |
//+------------------------------------------------------------------+

//--- Logování s filtrováním dle úrovně
void LogMessage(ENUM_LOG_LEVEL level, const string msg)
{
   if(level == LOG_DBG && !InpVerboseLog)
      return;

   string prefix;
   switch(level)
   {
      case LOG_OK:   prefix = "[OK]   "; break;
      case LOG_FAIL: prefix = "[FAIL] "; break;
      default:       prefix = "[DBG]  "; break;
   }
   Print(prefix + msg);
}

//--- Přečtení chybového textu z DLL
string GetDLLError()
{
   short buf[];
   ArrayResize(buf, DLL_ERR_BUF_LEN);
   ArrayInitialize(buf, 0);
   DN_GetError(buf, DLL_ERR_BUF_LEN);

   string s = "";
   for(int i = 0; i < DLL_ERR_BUF_LEN && buf[i] != 0; i++)
      s += ShortToString(buf[i]);

   return (StringLen(s) > 0) ? s : "neznámá DLL chyba";
}

//--- Ořez double do <low, high>
double Clamp(double value, double low, double high)
{
   return MathMax(low, MathMin(high, value));
}

//--- Kontrola finity čísla (NaN/Inf guard)
bool IsFinite(const double v)
{
   return MathIsValidNumber(v);
}

//+------------------------------------------------------------------+
//| Sekce: Validace vstupů                                           |
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
   if(InpBuyThresholdPct <= 50.0 || InpBuyThresholdPct >= 99.0)
   {
      LogMessage(LOG_FAIL, "BuyThreshold musí být (50, 99)");
      return false;
   }
   if(InpSellThresholdPct >= 50.0 || InpSellThresholdPct <= 1.0)
   {
      LogMessage(LOG_FAIL, "SellThreshold musí být (1, 50)");
      return false;
   }
   if(InpDeadZonePct < 0.0 || InpDeadZonePct > 20.0)
   {
      LogMessage(LOG_FAIL, "DeadZonePct musí být 0..20");
      return false;
   }
   if(InpMinConfidence < 0.0 || InpMinConfidence > 0.49)
   {
      LogMessage(LOG_FAIL, "MinConfidence musí být 0..0.49");
      return false;
   }
   if(InpATRPeriod < 2)
   {
      LogMessage(LOG_FAIL, "ATRPeriod musí být >= 2");
      return false;
   }
   if(InpBuyThresholdPct <= InpSellThresholdPct)
   {
      LogMessage(LOG_FAIL, "BuyThreshold musí být větší než SellThreshold");
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
//| Sekce: Detekce nového uzavřeného baru                            |
//+------------------------------------------------------------------+
bool IsNewClosedBar(datetime &outClosedBarTime)
{
   datetime times[];
   ArraySetAsSeries(times, true);
   if(CopyTime(_Symbol, PERIOD_CURRENT, 1, 1, times) != 1)
      return false;

   outClosedBarTime = times[0];
   if(outClosedBarTime <= 0 || outClosedBarTime == g_LastProcessedClosedBar)
      return false;

   return true;
}

//+------------------------------------------------------------------+
//| Sekce: Feature engineering                                       |
//+------------------------------------------------------------------+

//--- One-hot enkódování svíčky (0..8)
int EncodeCandle(double open, double high, double low, double close)
{
   double range = high - low;
   if(range < _Point)
      return 4; // doji / neurčito

   double bodyAbs    = MathAbs(close - open);
   double bodyRatio  = bodyAbs / range;
   double upperShadow = high - MathMax(open, close);
   double lowerShadow = MathMin(open, close) - low;
   bool   bullish    = (close >= open);

   if(bodyRatio < 0.10)
      return 4;

   if(bullish)
   {
      if(bodyRatio > 0.70) return 8;                              // silná bull svíčka
      if(bodyRatio > 0.30) return 7;                              // střední bull
      return (lowerShadow > upperShadow) ? 5 : 6;                // hammer / spinning bull
   }
   else
   {
      if(bodyRatio > 0.70) return 0;                              // silná bear svíčka
      if(bodyRatio > 0.30) return 1;                              // střední bear
      return (upperShadow > lowerShadow) ? 2 : 3;                // shooting star / spinning bear
   }
}

//--- Momentum: normalizovaná změna close za 'back' barů
double GetMomentumFeature(ENUM_TIMEFRAMES tf, int shift, int back)
{
   if(shift < 0 || shift + back >= iBars(_Symbol, tf))
      return 0.0;

   double c0 = iClose(_Symbol, tf, shift);
   double c1 = iClose(_Symbol, tf, shift + back);
   if(MathAbs(c1) < _Point)
      return 0.0;

   return Clamp((c0 - c1) / c1, -2.0, 2.0);
}

//--- Pozice close v N-barovém rangi (0 = dno, 1 = vrchol)
double GetRangePositionFeature(ENUM_TIMEFRAMES tf, int shift, int window)
{
   int bars = iBars(_Symbol, tf);
   if(shift < 0 || shift + window >= bars)
      return 0.5;

   double hi = -DBL_MAX, lo = DBL_MAX;
   for(int i = 0; i < window; i++)
   {
      hi = MathMax(hi, iHigh(_Symbol, tf, shift + i));
      lo = MathMin(lo, iLow (_Symbol, tf, shift + i));
   }

   double range = hi - lo;
   if(range < _Point)
      return 0.5;

   return Clamp((iClose(_Symbol, tf, shift) - lo) / range, 0.0, 1.0);
}

//--- Poměr aktuálního ATR vůči globálnímu průměru
double GetAtrRatioFeature(ENUM_TIMEFRAMES tf, int shift, int period)
{
   int bars = iBars(_Symbol, tf);
   if(shift < 0 || shift + period + 1 >= bars)
      return 1.0;

   double sum = 0.0;
   int    cnt = 0;
   for(int i = 0; i < period; i++)
   {
      int b = shift + i;
      if(b + 1 >= bars) break;
      double h  = iHigh (_Symbol, tf, b);
      double l  = iLow  (_Symbol, tf, b);
      double pc = iClose(_Symbol, tf, b + 1);
      sum += MathMax(h - l, MathMax(MathAbs(h - pc), MathAbs(l - pc)));
      cnt++;
   }
   if(cnt <= 0) return 1.0;

   double atr  = sum / cnt;
   double base = MathMax(g_ATRMean, _Point * 100.0);
   return Clamp(atr / base, 0.0, 5.0);
}

//--- Diskrétní stav pohybu ceny pro entropii (0=flat, 1=up, 2=down)
int GetEntropyStateAtShift(ENUM_TIMEFRAMES tf, int shift, int priceStepPoints)
{
   if(shift < 0 || shift + 1 >= iBars(_Symbol, tf))
      return 0;

   double delta = iClose(_Symbol, tf, shift) - iClose(_Symbol, tf, shift + 1);
   double thr   = priceStepPoints * _Point;

   if(delta >  thr) return 1;
   if(delta < -thr) return 2;
   return 0;
}

//--- Shannonova entropie pohybu ceny přes 'period' barů, normalizovaná na [0,1]
double CalculateShannonEntropyTF(ENUM_TIMEFRAMES tf, int shift, int period, int priceStepPoints)
{
   if(period <= 1 || shift < 0 || shift + period + 1 >= iBars(_Symbol, tf))
      return 0.5;

   int cntFlat = 0, cntUp = 0, cntDown = 0;
   for(int i = 0; i < period; i++)
   {
      int st = GetEntropyStateAtShift(tf, shift + i, priceStepPoints);
      if     (st == 1) cntUp++;
      else if(st == 2) cntDown++;
      else             cntFlat++;
   }

   double pFlat = (double)cntFlat / period;
   double pUp   = (double)cntUp   / period;
   double pDown = (double)cntDown / period;

   double H = 0.0;
   if(pFlat > 0.0) H -= pFlat * MathLog(pFlat) / LN2_CONST;
   if(pUp   > 0.0) H -= pUp   * MathLog(pUp)   / LN2_CONST;
   if(pDown > 0.0) H -= pDown * MathLog(pDown)  / LN2_CONST;

   double Hmax = MathLog(3.0) / LN2_CONST;
   if(Hmax <= 0.0) return 0.5;

   return Clamp(H / Hmax, 0.0, 1.0);
}

//--- Oslabení signálu při vysoké entropii (chaos)
double ApplyEntropyConfidenceFilter(double pBull, double entropySlow)
{
   if(!InpUseEntropyFilter || entropySlow <= InpEntropyChaosLevel)
      return pBull;

   double excess   = Clamp((entropySlow - InpEntropyChaosLevel) /
                           MathMax(1e-8, 1.0 - InpEntropyChaosLevel), 0.0, 1.0);
   double strength = Clamp(InpEntropyFilterPower, 0.0, 1.0);
   double mix      = excess * strength;

   return pBull * (1.0 - mix) + 0.5 * mix;
}

//--- Kontrola, zda je bar na daném TF časově konzistentní s referenčním časem
bool IsTFBarValid(ENUM_TIMEFRAMES tf, int shift, datetime refTime)
{
   if(shift < 0) return false;
   long tfSeconds = PeriodSeconds(tf);
   if(tfSeconds <= 0) return true;
   return (MathAbs((long)iTime(_Symbol, tf, shift) - (long)refTime) <= tfSeconds * 2);
}

//--- Naplnění FEAT_PER_TF features pro jeden TF a jeden timestep
void ExtractTFFeatures(ENUM_TIMEFRAMES tf, datetime refTime, double &feats[], int offset)
{
   int sh = iBarShift(_Symbol, tf, refTime, false);
   if(sh < 0 || !IsTFBarValid(tf, sh, refTime))
   {
      ArrayFill(feats, offset, FEAT_PER_TF, 0.0);
      return;
   }

   double o    = iOpen (_Symbol, tf, sh);
   double h    = iHigh (_Symbol, tf, sh);
   double l    = iLow  (_Symbol, tf, sh);
   double c    = iClose(_Symbol, tf, sh);
   int    bars = iBars(_Symbol, tf);
   long   vol     = iTickVolume(_Symbol, tf, sh);
   long   prevVol = (sh + 1 < bars) ? iTickVolume(_Symbol, tf, sh + 1) : 0;

   double range = MathMax(h - l, _Point);

   // One-hot enkódování svíčky (features 0..8)
   int sym = EncodeCandle(o, h, l, c);
   for(int k = 0; k < 9; k++)
      feats[offset + k] = (k == sym) ? 1.0 : 0.0;

   // Normalizované tělo a stín (features 9..10)
   feats[offset + 9]  = (c - o) / range;
   feats[offset + 10] = ((h - MathMax(o, c)) - (MathMin(o, c) - l)) / range;

   // Objemová změna (feature 11)
   feats[offset + 11] = Clamp((prevVol > 0) ? (double)(vol - prevVol) / prevVol : 0.0, -2.0, 2.0);

   // Momentum, ATR ratio, range position (features 12..14)
   feats[offset + 12] = GetMomentumFeature    (tf, sh, 5);
   feats[offset + 13] = GetAtrRatioFeature     (tf, sh, 14);
   feats[offset + 14] = GetRangePositionFeature(tf, sh, 20);

   // Shannonova entropie fast/slow + delta (features 15..17)
   double eFast = CalculateShannonEntropyTF(tf, sh, InpEntropyFastPeriod, InpEntropyPriceStep);
   double eSlow = CalculateShannonEntropyTF(tf, sh, InpEntropySlowPeriod, InpEntropyPriceStep);
   feats[offset + 15] = eFast;
   feats[offset + 16] = eSlow;
   feats[offset + 17] = Clamp(eFast - eSlow, -1.0, 1.0);
}

//--- Sestavení celé vstupní sekvence X[lookback * FEAT_PER_BAR]
bool BuildSequenceInput(datetime sourceBarTime, double &X[])
{
   int inDim = InpLookback * FEAT_PER_BAR;
   ArrayResize(X, inDim);
   ArrayInitialize(X, 0.0);

   int baseShift = iBarShift(_Symbol, PERIOD_CURRENT, sourceBarTime, false);
   if(baseShift < 1)
   {
      LogMessage(LOG_DBG, "BuildSequenceInput: baseShift < 1");
      return false;
   }

   for(int step = 0; step < InpLookback; step++)
   {
      int sh = baseShift + step;
      if(sh < 1) return false;

      datetime t   = iTime(_Symbol, PERIOD_CURRENT, sh);
      int      off = step * FEAT_PER_BAR;

      ExtractTFFeatures(InpTF1, t, X, off);
      ExtractTFFeatures(InpTF2, t, X, off + FEAT_PER_TF);
      ExtractTFFeatures(InpTF3, t, X, off + FEAT_PER_TF * 2);
   }
   return true;
}

//+------------------------------------------------------------------+
//| Sekce: ATR Mean (baseline pro normalizaci)                       |
//+------------------------------------------------------------------+
void UpdateATRMean()
{
   int sample = MathMin(200, iBars(_Symbol, PERIOD_CURRENT) - 2);
   if(sample < 20)
   {
      g_ATRMean = _Point * 100.0;
      return;
   }

   double sum = 0.0;
   int    cnt = 0;
   for(int i = 1; i <= sample; i++)
   {
      double h  = iHigh (_Symbol, PERIOD_CURRENT, i);
      double l  = iLow  (_Symbol, PERIOD_CURRENT, i);
      double pc = iClose(_Symbol, PERIOD_CURRENT, i + 1);
      sum += MathMax(h - l, MathMax(MathAbs(h - pc), MathAbs(l - pc)));
      cnt++;
   }
   g_ATRMean = (cnt > 0) ? MathMax(sum / cnt, _Point) : _Point * 100.0;
}

//+------------------------------------------------------------------+
//| Sekce: Model – metadata                                          |
//+------------------------------------------------------------------+

void FillModelMeta(ModelMetaEA &m, int stateSize)
{
   m.magic       = MODEL_MAGIC;
   m.metaVersion = MODEL_META_VER;
   m.featPerBar  = FEAT_PER_BAR;
   m.outputDim   = OUTPUT_DIM;
   m.lookback    = InpLookback;
   m.hidden1     = InpHiddenSize1;
   m.hidden2     = InpHiddenSize2;
   m.hidden3     = InpHiddenSize3;
   m.tf1         = (int)InpTF1;
   m.tf2         = (int)InpTF2;
   m.tf3         = (int)InpTF3;
   m.entropyFast = InpEntropyFastPeriod;
   m.entropySlow = InpEntropySlowPeriod;
   m.entropyStep = InpEntropyPriceStep;
   m.stateSize   = stateSize;
}

bool IsModelMetaCompatible(const ModelMetaEA &m)
{
   if(m.magic      != (int)MODEL_MAGIC   || m.metaVersion != (int)MODEL_META_VER) return false;
   if(m.featPerBar != FEAT_PER_BAR       || m.outputDim   != OUTPUT_DIM)          return false;
   if(m.lookback   != InpLookback)                                                 return false;
   if(m.hidden1    != InpHiddenSize1     || m.hidden2     != InpHiddenSize2)       return false;
   if(m.hidden3    != InpHiddenSize3)                                              return false;
   if(m.tf1        != (int)InpTF1        || m.tf2         != (int)InpTF2)          return false;
   if(m.tf3        != (int)InpTF3)                                                 return false;
   if(m.entropyFast != InpEntropyFastPeriod)                                       return false;
   if(m.entropySlow != InpEntropySlowPeriod)                                       return false;
   if(m.entropyStep != InpEntropyPriceStep)                                        return false;
   if(m.stateSize   <  MIN_MODEL_STATE_BYTES)                                      return false;
   return true;
}

//--- Zápis metadat do otevřeného souboru
void WriteModelMeta(int fh, const ModelMetaEA &m)
{
   FileWriteInteger(fh, m.magic,       INT_VALUE);
   FileWriteInteger(fh, m.metaVersion, INT_VALUE);
   FileWriteInteger(fh, m.featPerBar,  INT_VALUE);
   FileWriteInteger(fh, m.outputDim,   INT_VALUE);
   FileWriteInteger(fh, m.lookback,    INT_VALUE);
   FileWriteInteger(fh, m.hidden1,     INT_VALUE);
   FileWriteInteger(fh, m.hidden2,     INT_VALUE);
   FileWriteInteger(fh, m.hidden3,     INT_VALUE);
   FileWriteInteger(fh, m.tf1,         INT_VALUE);
   FileWriteInteger(fh, m.tf2,         INT_VALUE);
   FileWriteInteger(fh, m.tf3,         INT_VALUE);
   FileWriteInteger(fh, m.entropyFast, INT_VALUE);
   FileWriteInteger(fh, m.entropySlow, INT_VALUE);
   FileWriteInteger(fh, m.entropyStep, INT_VALUE);
   FileWriteInteger(fh, m.stateSize,   INT_VALUE);
}

//--- Čtení metadat z otevřeného souboru
void ReadModelMeta(int fh, ModelMetaEA &m)
{
   m.magic       = FileReadInteger(fh, INT_VALUE);
   m.metaVersion = FileReadInteger(fh, INT_VALUE);
   m.featPerBar  = FileReadInteger(fh, INT_VALUE);
   m.outputDim   = FileReadInteger(fh, INT_VALUE);
   m.lookback    = FileReadInteger(fh, INT_VALUE);
   m.hidden1     = FileReadInteger(fh, INT_VALUE);
   m.hidden2     = FileReadInteger(fh, INT_VALUE);
   m.hidden3     = FileReadInteger(fh, INT_VALUE);
   m.tf1         = FileReadInteger(fh, INT_VALUE);
   m.tf2         = FileReadInteger(fh, INT_VALUE);
   m.tf3         = FileReadInteger(fh, INT_VALUE);
   m.entropyFast = FileReadInteger(fh, INT_VALUE);
   m.entropySlow = FileReadInteger(fh, INT_VALUE);
   m.entropyStep = FileReadInteger(fh, INT_VALUE);
   m.stateSize   = FileReadInteger(fh, INT_VALUE);
}

//+------------------------------------------------------------------+
//| Sekce: Model – filename, init, save, load                        |
//+------------------------------------------------------------------+

string TFToShortString(ENUM_TIMEFRAMES tf)
{
   if(tf == PERIOD_CURRENT) tf = (ENUM_TIMEFRAMES)_Period;
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

   return StringFormat("%s_%s_%s-%s-%s_EA_L%d_%s_F%dx%d_ENT%d-%d_V210.lstm",
                       InpModelPrefix, sym,
                       TFToShortString(InpTF1), TFToShortString(InpTF2), TFToShortString(InpTF3),
                       InpLookback, layers,
                       FEAT_PER_BAR, OUTPUT_DIM,
                       InpEntropyFastPeriod, InpEntropySlowPeriod);
}

//--- Sestavení sítě přes DLL API
bool InitNetwork()
{
   g_NetHandle = DN_Create();
   if(g_NetHandle == 0)
   {
      LogMessage(LOG_FAIL, "DN_Create selhal: " + GetDLLError());
      return false;
   }

   DN_SetSequenceLength(g_NetHandle, InpLookback);
   DN_SetMiniBatchSize (g_NetHandle, InpMiniBatch);
   DN_SetGradClip      (g_NetHandle, InpGradClip);

   // Vrstva 1: vstup (FEAT_PER_BAR) → hidden1
   if(!DN_AddLayerEx(g_NetHandle, FEAT_PER_BAR, InpHiddenSize1, 0, 0, InpDropout))
   {
      LogMessage(LOG_FAIL, "Přidání vrstvy 1 selhalo: " + GetDLLError());
      return false;
   }

   // Vrstva 2: hidden1 → hidden2  (in_sz = InpHiddenSize1, ne 0)
   if(!DN_AddLayerEx(g_NetHandle, InpHiddenSize1, InpHiddenSize2, 0, 0, InpDropout * 0.5))
   {
      LogMessage(LOG_FAIL, "Přidání vrstvy 2 selhalo: " + GetDLLError());
      return false;
   }

   // Volitelná vrstva 3: hidden2 → hidden3
   if(InpHiddenSize3 > 0 &&
      !DN_AddLayerEx(g_NetHandle, InpHiddenSize2, InpHiddenSize3, 0, 0, 0.0))
   {
      LogMessage(LOG_FAIL, "Přidání vrstvy 3 selhalo: " + GetDLLError());
      return false;
   }

   if(!DN_SetOutputDim(g_NetHandle, OUTPUT_DIM))
   {
      LogMessage(LOG_FAIL, "SetOutputDim selhal: " + GetDLLError());
      return false;
   }

   LogMessage(LOG_OK, StringFormat("Síť připravena: %d → [%d, %d, %d] → %d",
                                   FEAT_PER_BAR, InpHiddenSize1, InpHiddenSize2,
                                   InpHiddenSize3, OUTPUT_DIM));
   return true;
}

//--- Uložení stavu modelu + metadat do souboru
bool SaveModel()
{
   if(g_NetHandle <= 0 || !g_ModelReady || StringLen(g_ModelFilePath) == 0)
      return false;

   int stateSize = DN_SaveState(g_NetHandle);
   if(stateSize <= 0)
   {
      LogMessage(LOG_FAIL, "DN_SaveState vrátil 0: " + GetDLLError());
      return false;
   }

   char stateBuf[];
   ArrayResize(stateBuf, stateSize);
   if(!DN_GetState(g_NetHandle, stateBuf, stateSize))
   {
      LogMessage(LOG_FAIL, "DN_GetState selhal: " + GetDLLError());
      return false;
   }

   ModelMetaEA meta;
   FillModelMeta(meta, stateSize);

   int fh = FileOpen(g_ModelFilePath, FILE_WRITE | FILE_BIN | FILE_COMMON);
   if(fh == INVALID_HANDLE)
   {
      LogMessage(LOG_FAIL, "FileOpen (write) selhal pro: " + g_ModelFilePath);
      return false;
   }

   WriteModelMeta(fh, meta);
   FileWriteArray(fh, stateBuf, 0, stateSize);
   FileClose(fh);

   LogMessage(LOG_OK, StringFormat("Model uložen (%d bytů): %s", stateSize, g_ModelFilePath));
   return true;
}

//--- Načtení stavu modelu ze souboru
bool LoadModel()
{
   if(g_NetHandle <= 0 || StringLen(g_ModelFilePath) == 0)
      return false;

   int fh = FileOpen(g_ModelFilePath, FILE_READ | FILE_BIN | FILE_COMMON);
   if(fh == INVALID_HANDLE)
      return false;

   ulong fs = FileSize(fh);
   if(fs < 60 || fs > (ulong)FILE_SIZE_MAX)
   {
      FileClose(fh);
      LogMessage(LOG_FAIL, StringFormat("LoadModel: neplatná velikost souboru (%I64u bytů)", fs));
      return false;
   }

   ModelMetaEA meta;
   meta.Reset();
   ReadModelMeta(fh, meta);

   if(!IsModelMetaCompatible(meta))
   {
      FileClose(fh);
      LogMessage(LOG_FAIL, "LoadModel: nekompatibilní metadata (jiná architektura nebo TF)");
      return false;
   }

   char buf[];
   ArrayResize(buf, meta.stateSize + 1);
   int br = (int)FileReadArray(fh, buf, 0, meta.stateSize);
   FileClose(fh);

   if(br != meta.stateSize)
   {
      LogMessage(LOG_FAIL, StringFormat("LoadModel: přečteno %d bytů, očekáváno %d", br, meta.stateSize));
      return false;
   }
   buf[meta.stateSize] = 0;

   if(!DN_LoadState(g_NetHandle, buf))
   {
      LogMessage(LOG_FAIL, "DN_LoadState selhal: " + GetDLLError());
      return false;
   }

   LogMessage(LOG_OK, StringFormat("Model načten (%d bytů): %s", meta.stateSize, g_ModelFilePath));
   return true;
}

//+------------------------------------------------------------------+
//| Sekce: Inference                                                 |
//+------------------------------------------------------------------+
bool ComputePrediction(datetime barTime, PredictionSnapshot &out)
{
   out.Reset();
   if(!g_ModelReady || g_NetHandle <= 0)
      return false;

   double X[];
   if(!BuildSequenceInput(barTime, X))
   {
      LogMessage(LOG_DBG, "BuildSequenceInput selhal");
      return false;
   }

   int inDim = InpLookback * FEAT_PER_BAR;
   if(ArraySize(X) != inDim)
   {
      LogMessage(LOG_FAIL, StringFormat("ComputePrediction: neočekávaná velikost X (%d != %d)", ArraySize(X), inDim));
      return false;
   }

   double Y[];
   if(ArrayResize(Y, OUTPUT_DIM) != OUTPUT_DIM)
      return false;
   ArrayInitialize(Y, 0.0);

   // batch=1, layout=0 => Y je flat [batch, OUTPUT_DIM]
   if(!DN_PredictBatch(g_NetHandle, X, 1, inDim, 0, Y))
   {
      LogMessage(LOG_FAIL, "DN_PredictBatch selhal: " + GetDLLError());
      return false;
   }

   if(ArraySize(Y) < OUTPUT_DIM || !IsFinite(Y[0]) || !IsFinite(Y[1]))
   {
      LogMessage(LOG_FAIL, "ComputePrediction: nevalidní výstup Y");
      return false;
   }

   // Stabilní softmax z logitů -> pravděpodobnosti
   double z0 = Y[0];
   double z1 = Y[1];
   double zMax = MathMax(z0, z1);
   double e0 = MathExp(z0 - zMax);
   double e1 = MathExp(z1 - zMax);
   double den = e0 + e1;

   double pBull = 0.5;
   double pBear = 0.5;
   if(IsFinite(den) && den > 1e-12)
   {
      pBull = e0 / den;
      pBear = e1 / den;
   }

   // Entropie na primárním TF
   ENUM_TIMEFRAMES tf1Resolved = (InpTF1 == PERIOD_CURRENT) ? PERIOD_CURRENT : InpTF1;
   int sh = iBarShift(_Symbol, tf1Resolved, barTime, false);
   if(sh < 0) sh = 1;

   double eFast = CalculateShannonEntropyTF(tf1Resolved, sh, InpEntropyFastPeriod, InpEntropyPriceStep);
   double eSlow = CalculateShannonEntropyTF(tf1Resolved, sh, InpEntropySlowPeriod, InpEntropyPriceStep);

   pBull = Clamp(ApplyEntropyConfidenceFilter(pBull, eSlow), 0.0, 1.0);

   out.sourceBarTime = barTime;
   out.pBull         = pBull;
   out.pBear         = 1.0 - pBull;
   out.entropyFast   = eFast;
   out.entropySlow   = eSlow;
   out.entropyDelta  = eFast - eSlow;
   out.valid         = true;
   return true;
}

//+------------------------------------------------------------------+
//| Sekce: Money management                                          |
//+------------------------------------------------------------------+

//--- ATR z cachovaného iATR handle (bar[1])
double GetATRValue()
{
   if(g_ATRHandle == INVALID_HANDLE)
   {
      LogMessage(LOG_FAIL, "ATR handle není inicializován");
      return 0.0;
   }

   double buf[];
   ArraySetAsSeries(buf, true);
   int copied = CopyBuffer(g_ATRHandle, 0, 1, 1, buf);
   if(copied != 1 || ArraySize(buf) < 1 || !IsFinite(buf[0]))
      return 0.0;

   return MathMax(buf[0], _Point * 5.0);
}

//--- Normalizace lotů dle brokerských limitů
double NormalizeLots(double vol)
{
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double step   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(step <= 0.0) step = minLot;

   vol = Clamp(vol, minLot, maxLot);
   vol = MathFloor(vol / step) * step;
   return NormalizeDouble(vol, 2);
}

//--- Výpočet lotsizes dle risk %
double ComputeLotByRisk(double slPoints)
{
   if(!InpUseRiskPercent || InpRiskPercent <= 0.0 || slPoints <= 0.0)
      return NormalizeLots(InpFixedLot);

   double balance   = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskMoney = balance * InpRiskPercent / 100.0;
   if(riskMoney <= 0.0)
      return NormalizeLots(InpFixedLot);

   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tickValue <= 0.0 || tickSize <= 0.0)
      return NormalizeLots(InpFixedLot);

   double moneyPerPointPerLot = tickValue * (_Point / tickSize);
   if(moneyPerPointPerLot <= 0.0)
      return NormalizeLots(InpFixedLot);

   return NormalizeLots(riskMoney / (slPoints * moneyPerPointPerLot));
}

//--- Kontrola spreadu
bool IsSpreadAcceptable()
{
   double spreadPts = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID)) / _Point;
   if(spreadPts > InpMaxSpreadPoints)
   {
      LogMessage(LOG_DBG, StringFormat("Spread příliš vysoký: %.1f > %d bodů", spreadPts, InpMaxSpreadPoints));
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
//| Sekce: Správa pozic                                              |
//+------------------------------------------------------------------+

//--- Pomocná: ověření, zda pozice patří tomuto EA
bool IsOurPosition(ulong ticket)
{
   if(!PositionSelectByTicket(ticket))                              return false;
   if(PositionGetString(POSITION_SYMBOL)    != _Symbol)            return false;
   if(PositionGetInteger(POSITION_MAGIC)    != InpMagicNumber)     return false;
   return true;
}

//--- Vrátí typ otevřené pozice EA, nebo -1 pokud žádná
int FindOurPositionType()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!IsOurPosition(ticket)) continue;
      return (int)PositionGetInteger(POSITION_TYPE);
   }
   return -1;
}

//--- Uzavření všech pozic EA na daném symbolu
bool CloseOurPositions()
{
   bool allOk = true;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!IsOurPosition(ticket)) continue;

      if(!g_Trade.PositionClose(ticket, InpDeviationPoints))
      {
         LogMessage(LOG_FAIL, StringFormat("Zavření pozice #%I64u selhalo, retcode=%d",
                                           ticket, g_Trade.ResultRetcode()));
         allOk = false;
      }
   }
   return allOk;
}

//+------------------------------------------------------------------+
//| Sekce: Obchodní logika                                           |
//+------------------------------------------------------------------+

//--- Vyhodnocení signálu (+1 buy, -1 sell, 0 bez akce)
int EvaluateSignal(const PredictionSnapshot &p)
{
   if(!p.valid) return 0;

   double pbPct = p.pBull * 100.0;
   double conf  = MathAbs(p.pBull - 0.5);

   if(conf < InpMinConfidence)                                             return 0;
   if(pbPct >= 50.0 - InpDeadZonePct && pbPct <= 50.0 + InpDeadZonePct)  return 0;
   if(pbPct >= InpBuyThresholdPct)                                        return  1;
   if(pbPct <= InpSellThresholdPct)                                       return -1;
   return 0;
}

//--- Výpočet cen pro pokyn (price, SL, TP) + ověření stop-levelů
bool BuildOrderPrices(bool isBuy, double &price, double &sl, double &tp, double &slPoints)
{
   double atr = GetATRValue();
   if(atr <= 0.0)
   {
      LogMessage(LOG_FAIL, "ATR = 0, nelze počítat SL/TP");
      return false;
   }

   price = isBuy ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                 : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(!IsFinite(price) || price <= 0.0)
   {
      LogMessage(LOG_FAIL, "Nevalidní cena pro otevření obchodu");
      return false;
   }

   double slDist = atr * MathMax(0.1, InpATRSLMultiplier);
   double tpDist = (InpRiskRewardRatio > 0.01)
                   ? slDist * InpRiskRewardRatio
                   : atr * MathMax(0.1, InpATRTPMultiplier);

   sl = isBuy ? price - slDist : price + slDist;
   tp = isBuy ? price + tpDist : price - tpDist;

   // Přizpůsobení stop-levelům brokera
   int    stopLevel   = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   int    freezeLevel = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL);
   double minDist     = MathMax(stopLevel, freezeLevel) * _Point;

   if(MathAbs(price - sl) < minDist) sl = isBuy ? price - minDist : price + minDist;
   if(MathAbs(tp - price) < minDist) tp = isBuy ? price + minDist : price - minDist;

   price    = NormalizeDouble(price, _Digits);
   sl       = NormalizeDouble(sl,    _Digits);
   tp       = NormalizeDouble(tp,    _Digits);

   if(!IsFinite(sl) || !IsFinite(tp) || sl <= 0.0 || tp <= 0.0)
      return false;

   slPoints = MathAbs(price - sl) / _Point;
   if(!IsFinite(slPoints) || slPoints <= 0.0)
      return false;

   return true;
}

//--- Otevření pokynu dle signálu
bool OpenSignalTrade(int signal, const PredictionSnapshot &p)
{
   if(signal == 0)           return false;
   if(!IsSpreadAcceptable()) return false;

   if(g_LastTradeTime > 0 &&
      (TimeCurrent() - g_LastTradeTime) < InpTradeCooldownSec)
      return false;

   bool   isBuy = (signal > 0);
   double price, sl, tp, slPoints;
   if(!BuildOrderPrices(isBuy, price, sl, tp, slPoints))
      return false;

   double lots = ComputeLotByRisk(slPoints);
   if(!IsFinite(lots) || lots <= 0.0)
   {
      LogMessage(LOG_FAIL, "Nevalidní lotsize");
      return false;
   }

   string comment = StringFormat("LSTM_EA pBull=%.1f%% eS=%.2f", p.pBull * 100.0, p.entropySlow);

   ResetLastError();
   bool ok = isBuy ? g_Trade.Buy (lots, _Symbol, price, sl, tp, comment)
                   : g_Trade.Sell(lots, _Symbol, price, sl, tp, comment);

   if(!ok)
   {
      LogMessage(LOG_FAIL, StringFormat("Pokyn selhal, retcode=%d lastErr=%d",
                                        g_Trade.ResultRetcode(), GetLastError()));
      return false;
   }

   if(g_Trade.ResultDeal() == 0 && g_Trade.ResultOrder() == 0)
   {
      LogMessage(LOG_FAIL, StringFormat("Pokyn bez deal/order, retcode=%d", g_Trade.ResultRetcode()));
      return false;
   }

   g_LastTradeTime = TimeCurrent();
   LogMessage(LOG_OK, StringFormat("%s %.2f lot | pBull=%.1f%% pBear=%.1f%%",
                                   isBuy ? "BUY" : "SELL", lots,
                                   p.pBull * 100.0, p.pBear * 100.0));
   return true;
}

//--- Rozhodnutí o obchodu (otevření / uzavření / reverz / nic)
void ProcessTradingDecision(const PredictionSnapshot &p)
{
   int signal  = EvaluateSignal(p);
   int posType = FindOurPositionType();

   // Bez otevřené pozice
   if(posType == -1)
   {
      if(signal != 0)
         OpenSignalTrade(signal, p);
      return;
   }

   bool posBuy   = (posType == POSITION_TYPE_BUY);
   bool sameDir  = (signal > 0 &&  posBuy) || (signal < 0 && !posBuy);
   bool opposite = (signal > 0 && !posBuy) || (signal < 0 &&  posBuy);

   if(sameDir)
   {
      LogMessage(LOG_DBG, "Pozice běží stejným směrem, přeskakuji.");
      return;
   }

   if(opposite)
   {
      if(InpOppositeSignalAction == OPP_IGNORE)
         return;

      // OPP_CLOSE nebo OPP_REVERSE: nejdříve zavřít existující pozici
      if(!CloseOurPositions())
         return;

      // OPP_REVERSE: otevřít novou pozici v opačném směru
      if(InpOppositeSignalAction == OPP_REVERSE || InpPositionMode == POS_ALLOW_REVERSE)
         OpenSignalTrade(signal, p);
   }
   // signal == 0: pozici držíme
}

//+------------------------------------------------------------------+
//| Sekce: Lifecycle EA                                              |
//+------------------------------------------------------------------+
int OnInit()
{
   if(!ValidateInputs())
      return INIT_PARAMETERS_INCORRECT;

   g_ModelFilePath = BuildModelFileName();
   g_LastPred.Reset();
   UpdateATRMean();

   // Inicializace cachovaného ATR handle
   g_ATRHandle = iATR(_Symbol, PERIOD_CURRENT, InpATRPeriod);
   if(g_ATRHandle == INVALID_HANDLE)
   {
      LogMessage(LOG_FAIL, "Inicializace iATR selhala");
      return INIT_FAILED;
   }

   g_Trade.SetExpertMagicNumber(InpMagicNumber);
   g_Trade.SetDeviationInPoints(InpDeviationPoints);

   if(!InitNetwork())
      return INIT_FAILED;

   if(LoadModel())
   {
      g_ModelReady     = true;
      g_LoadedFromFile = true;
      LogMessage(LOG_OK, "Model načten ze souboru.");
   }
   else
   {
      g_ModelReady = false;
      LogMessage(LOG_FAIL, "Model nebyl nalezen nebo je nekompatibilní. EA čeká (WAIT režim).");
      if(!InpInferenceOnly && InpAttemptRetrain)
         LogMessage(LOG_DBG, "Retrain hook je připraven, ale v této verzi záměrně vypnutý.");
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

   if(g_ATRHandle != INVALID_HANDLE)
   {
      IndicatorRelease(g_ATRHandle);
      g_ATRHandle = INVALID_HANDLE;
   }
}

void RunBarCycle()
{
   datetime closedBarTime;
   if(!IsNewClosedBar(closedBarTime))
      return;

   // Zaznamenat nový bar až po úspěšné detekci
   g_LastProcessedClosedBar = closedBarTime;
   UpdateATRMean();

   if(!g_ModelReady)
   {
      LogMessage(LOG_DBG, "Model není připraven, obchodování pozastaveno.");
      return;
   }

   PredictionSnapshot p;
   if(!ComputePrediction(closedBarTime, p))
      return;

   g_LastPred = p;
   LogMessage(LOG_DBG, StringFormat("bar=%s pBull=%.2f%% eFast=%.3f eSlow=%.3f",
                                    TimeToString(closedBarTime, TIME_DATE | TIME_MINUTES),
                                    p.pBull * 100.0, p.entropyFast, p.entropySlow));

   ProcessTradingDecision(p);
}

void OnTick()  { RunBarCycle(); }
void OnTimer() { RunBarCycle(); }
//+------------------------------------------------------------------+
