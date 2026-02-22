#property strict
#property indicator_separate_window
#property indicator_buffers 4
#property indicator_plots   3
#property version "1.31"

// Plot 0: Expected Move %
#property indicator_label1  "ExpectedMove %"
#property indicator_type1   DRAW_LINE
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2
#property indicator_color1  clrDodgerBlue

// Plot 1: Low Threshold
#property indicator_label2  "Thr Low"
#property indicator_type2   DRAW_LINE
#property indicator_style2  STYLE_DOT
#property indicator_width2  1
#property indicator_color2  clrLime

// Plot 2: High Threshold
#property indicator_label3  "Thr High"
#property indicator_type3   DRAW_LINE
#property indicator_style3  STYLE_DOT
#property indicator_width3  1
#property indicator_color3  clrRed

// ============================================================================
// DLL API — asynchronní NN
// ============================================================================
#import "MQL5GPULibrary_LSTM.dll"
int    DN_Create();
void   DN_Free(int h);
int    DN_SetSequenceLength(int h, int seq_len);
int    DN_SetMiniBatchSize(int h, int mbs);
int    DN_AddLayerEx(int h, int in_dim, int out_dim, int act, int use_ln, double dropout);
int    DN_SetGradClip(int h, double clip);
int    DN_LoadBatch(int h, const double &X[], const double &T[], int batch, int in_dim, int out_dim, int layout);
int    DN_PredictBatch(int h, const double &X[], int batch, int in_dim, int layout, double &out_Y[]);
int    DN_SnapshotWeights(int h);
int    DN_RestoreWeights(int h);
double DN_GetGradNorm(int h);
void   DN_GetError(short &buf[], int len);

// Async
int    DN_TrainAsync(int h, int epochs, double target_mse, double lr, double wd);
int    DN_GetTrainingStatus(int h); // 0=IDLE, 1=RUNNING, 2=COMPLETED, -1=ERROR
void   DN_GetTrainingResult(int h, double &out_mse, int &out_epochs);
void   DN_StopTraining(int h);
#import

// ============================================================================
// Inputs
// ============================================================================
input group "═══ Symboly ═══"
input string InpExtraSymbol1 = "EURUSD";
input string InpExtraSymbol2 = "XAUUSD";

input group "═══ Zarovnání času ═══"
input bool   InpAlignExact   = false;  // exact=true často selže na M15 (BTC 24/7 vs FX seance)

input group "═══ Sekvence a target ═══"
input int    InpSeqLen       = 32;     // délka sekvence
input int    InpHorizonH     = 12;     // horizont pro future range (barů dopředu)
input int    InpHiddenSize   = 64;
input double InpDropout      = 0.0;

input group "═══ Trénování ═══"
input int    InpTrainBars    = 12000;  // počet trénovacích vzorků (cílený)
input int    InpEpochs       = 4000;
input double InpLR           = 0.002;
input double InpWD           = 0.00005;
input double InpTargetMSE    = 0.0005;
input double InpGradClip     = 3.0;
input int    InpMiniBatch    = 32;
input double InpTargetClamp  = 2.5;    // clamp z-score targetu (po log1p)

input group "═══ Prahy a zobrazení ═══"
input int    InpHistoryLenThr     = 200;  // historie pro adaptivní prahy
input double InpThrMinPct         = 0.15; // minimální práh v %
input double InpThrLowPercentile  = 0.50; // percentil low (0..1)
input double InpThrHighPercentile = 0.85; // percentil high (0..1)
input int    InpCalcHistory       = 600;  // kolik barů zpět dopočítat křivku
input bool   InpShowComment       = true;

input group "═══ Filtrace multi-symbol kvality ═══"
input bool   InpFilterFrozenOtherSymbols = true; // když EURUSD/XAUUSD vrací pořád stejný bar index, sample přeskočit
input int    InpMinShiftChanges          = 3;    // min. počet změn shiftu v sekvenci (na každém extra symbolu)

input group "═══ Multi-symbol sampling ═══"
input int    InpAttemptMult = 4; // kolikrát víc pokusů než train_bars (kvůli FX víkendům/seancím)

input group "═══ Anti-spam (retry) ═══"
input int    InpTrainRetrySec = 30; // minimální čas mezi pokusy spustit trénink po failu

input group "═══ SUPER-DEBUG ═══"
input bool   InpSuperDebug     = true;    // zapnout detailní debug
input int    InpDbgMaxLines    = 140;     // max řádků na jeden pokus (anti-spam)
input bool   InpDbgDumpOnce    = false;   // pokud true, po prvním failu už další dumpy nepojedou
input int    InpDbgShiftPrintLimit = 12;  // kolik prvních iBarShift řádků tisknout v PrepareData (anti-spam)

// ============================================================================
// Buffers
// ============================================================================
double g_buf_move_pct[];
double g_buf_thr_low[];
double g_buf_thr_high[];
double g_buf_raw[];

// ============================================================================
// State
// ============================================================================
int    g_net = 0;
bool   g_trained = false;
bool   g_async_training = false;
bool   g_init_done = false;

uint   g_train_start_tick = 0;
bool   g_is_retrain = false;

double g_t_mean = 0.0;
double g_t_std  = 1.0;

// Threshold history
double g_abs_hist[];
int    g_abs_count = 0;

// Symbols
string g_sym_main;
string g_sym1;
string g_sym2;

// Rates cache
MqlRates g_r_main[];
MqlRates g_r_1[];
MqlRates g_r_2[];

// Volume normalization
double g_vmax_main = 1.0;
double g_vmax_1    = 1.0;
double g_vmax_2    = 1.0;

// Retry control
datetime g_last_train_attempt = 0;

// ============================================================================
// SUPER-DEBUG infra
// ============================================================================
int  g_dbg_lines_left = 0;
bool g_dbg_failed_once = false;
int  g_dbg_shift_print_left = 0;

ulong NowUS() { return (ulong)GetMicrosecondCount(); }

void DbgResetBudget()
{
   g_dbg_lines_left = InpDbgMaxLines;
   g_dbg_shift_print_left = InpDbgShiftPrintLimit;
}

bool DbgCan()
{
   if(!InpSuperDebug) return false;
   if(InpDbgDumpOnce && g_dbg_failed_once) return false;
   return (g_dbg_lines_left > 0);
}

void DbgPrint(const string tag, const string msg)
{
   if(!DbgCan()) return;
   Print(tag, " ", msg);
   g_dbg_lines_left--;
}

string TFToStr(ENUM_TIMEFRAMES tf)
{
   if(tf==PERIOD_M1)  return "M1";
   if(tf==PERIOD_M5)  return "M5";
   if(tf==PERIOD_M15) return "M15";
   if(tf==PERIOD_M30) return "M30";
   if(tf==PERIOD_H1)  return "H1";
   if(tf==PERIOD_H4)  return "H4";
   if(tf==PERIOD_D1)  return "D1";
   return IntegerToString((int)tf);
}

string TimeToStrSec(datetime t) { return TimeToString(t, TIME_DATE|TIME_MINUTES|TIME_SECONDS); }

void MarkDbgFail() { g_dbg_failed_once = true; }

// ============================================================================
// Error helpers
// ============================================================================
string GetDLLError()
{
   short buf[];
   ArrayResize(buf, 512);
   DN_GetError(buf, 512);
   return ShortArrayToString(buf);
}

// ============================================================================
// Safe wrappers for debug
// ============================================================================
int SafeCopyRates(const string sym, ENUM_TIMEFRAMES tf, int start, int count, MqlRates &out[])
{
   ResetLastError();
   int got = CopyRates(sym, tf, start, count, out);
   int le = GetLastError();
   DbgPrint("[DBG]", StringFormat("CopyRates(%s,%s,start=%d,count=%d) => got=%d, err=%d", sym, TFToStr(tf), start, count, got, le));
   return got;
}

int SafeBars(const string sym, ENUM_TIMEFRAMES tf)
{
   ResetLastError();
   int b = Bars(sym, tf);
   int le = GetLastError();
   DbgPrint("[DBG]", StringFormat("Bars(%s,%s) => %d, err=%d", sym, TFToStr(tf), b, le));
   return b;
}

// iBarShift je drahý a logy snadno zaspamují terminál.
// Tiskneme jen omezeně (InpDbgShiftPrintLimit) a jen pokud je SUPER-DEBUG.
int SafeBarShiftLimited(const string sym, ENUM_TIMEFRAMES tf, datetime t, bool exact)
{
   ResetLastError();
   int sh = iBarShift(sym, tf, t, exact);
   int le = GetLastError();

   if(DbgCan() && g_dbg_shift_print_left > 0)
   {
      DbgPrint("[DBG]", StringFormat("iBarShift(%s,%s,time=%s,exact=%s) => sh=%d, err=%d",
               sym, TFToStr(tf), TimeToStrSec(t), (exact?"true":"false"), sh, le));
      g_dbg_shift_print_left--;
   }
   return sh;
}

// ============================================================================
// Threshold history
// ============================================================================
void ResetThrHistory()
{
   g_abs_count = 0;
   ArrayResize(g_abs_hist, InpHistoryLenThr);
}

void PushThrSample(double move_pct, int counter)
{
   double v = MathAbs(move_pct);
   int idx = (g_abs_count < InpHistoryLenThr) ? g_abs_count : (counter % InpHistoryLenThr);
   if(ArraySize(g_abs_hist) != InpHistoryLenThr)
      ArrayResize(g_abs_hist, InpHistoryLenThr);
   g_abs_hist[idx] = v;
   if(g_abs_count < InpHistoryLenThr) g_abs_count++;
}

double PercentileFromHist(double p)
{
   int n = MathMin(g_abs_count, InpHistoryLenThr);
   if(n < 20) return InpThrMinPct;

   double tmp[];
   ArrayResize(tmp, n);
   for(int i=0;i<n;i++) tmp[i] = g_abs_hist[i];
   ArraySort(tmp);

   int idx = (int)MathFloor((n - 1) * p);
   idx = MathMax(0, MathMin(idx, n-1));
   double thr = tmp[idx];
   if(thr < InpThrMinPct) thr = InpThrMinPct;
   return thr;
}

// ============================================================================
// Denorm: raw -> log1p(range) -> range -> %
// ============================================================================
double DenormExpectedMovePct(double raw)
{
   double z = raw * g_t_std + g_t_mean;   // log(1+range)
   double range = MathExp(z) - 1.0;
   if(range < 0) range = 0;
   return 100.0 * range;
}

// ============================================================================
// PrepareData
// Robust fix:
// - extend main lookback window by InpAttemptMult to survive FX closed periods
// - compute max iBarShift over required main times for s1,s2
// - load s1,s2 deep enough (max_shift+3) so shifts are always in-range
// Features per symbol (4): [logret_seq, range/close, body/close, vol_norm]
// Target: log1p(future_range(main, H bars)) -> z-score + clamp
// Optional: filter sequences where EURUSD/XAUUSD shifts don't change enough.
// ============================================================================
bool PrepareData(double &X[], double &T[], int &samples, int train_bars)
{
   DbgResetBudget();
   ulong t0 = NowUS();

   const int seq = InpSeqLen;
   const int H   = InpHorizonH;
   const int feat_per_sym = 4;
   const int sym_count = 3;
   const int feat_total = feat_per_sym * sym_count;
   const int idim = seq * feat_total;

   bool exact = InpAlignExact;

   DbgPrint("[DBG]", "================ SUPER-DEBUG: PrepareData BEGIN ================");
   DbgPrint("[DBG]", StringFormat("Symbols: main=%s, s1=%s, s2=%s | TF=%s | exact=%s",
                                 g_sym_main, g_sym1, g_sym2, TFToStr(_Period), (exact?"true":"false")));

   // --- NEW: attempt multiplier (FX closed periods) ---
   int attempt_mult = MathMax(1, InpAttemptMult);
   int want_attempts = train_bars * attempt_mult;

   int b0 = SafeBars(g_sym_main, _Period);
   int max_possible_attempts = b0 - (seq + H + 3);
   if(max_possible_attempts < train_bars)
   {
      DbgPrint("[FAIL]", StringFormat("Not enough bars for main. bars=%d need_at_least=%d",
                                     b0, (train_bars + seq + H + 3)));
      MarkDbgFail();
      return false;
   }

   int max_attempt = MathMin(want_attempts, max_possible_attempts);
   int need_main   = max_attempt + seq + H + 3;

   DbgPrint("[DBG]", StringFormat("Params: train_bars=%d, seq=%d, H=%d, idim=%d", train_bars, seq, H, idim));
   DbgPrint("[DBG]", StringFormat("attempt_mult=%d => want_attempts=%d, max_attempt=%d", attempt_mult, want_attempts, max_attempt));
   DbgPrint("[DBG]", StringFormat("need_main=%d", need_main));

   // Ensure symbols in MarketWatch
   SymbolSelect(g_sym1, true);
   SymbolSelect(g_sym2, true);

   if(b0 < need_main)
   {
      DbgPrint("[FAIL]", StringFormat("Not enough bars for main lookback. b0=%d need_main=%d", b0, need_main));
      MarkDbgFail();
      return false;
   }

   ArraySetAsSeries(g_r_main, true);
   int got0 = SafeCopyRates(g_sym_main, _Period, 0, need_main, g_r_main);
   if(got0 < need_main)
   {
      DbgPrint("[FAIL]", StringFormat("CopyRates main insufficient. got0=%d need_main=%d", got0, need_main));
      MarkDbgFail();
      return false;
   }

   // Compute max shifts required for s1/s2 over the main time window.
   int max_sh1 = -1;
   int max_sh2 = -1;

   for(int i=0; i<need_main; i++)
   {
      datetime tm = g_r_main[i].time;

      int sh1 = SafeBarShiftLimited(g_sym1, _Period, tm, exact);
      int sh2 = SafeBarShiftLimited(g_sym2, _Period, tm, exact);

      if(sh1 > max_sh1) max_sh1 = sh1;
      if(sh2 > max_sh2) max_sh2 = sh2;
   }

   if(max_sh1 < 0 || max_sh2 < 0)
   {
      DbgPrint("[FAIL]", StringFormat("iBarShift max failed: max_sh1=%d max_sh2=%d", max_sh1, max_sh2));
      DbgPrint("[DBG]", "Tip: stáhni historii EURUSD/XAUUSD pro tento timeframe, nebo zkontroluj název symbolu u brokera.");
      MarkDbgFail();
      return false;
   }

   int need1 = max_sh1 + 3;
   int need2 = max_sh2 + 3;

   DbgPrint("[DBG]", StringFormat("Computed depth for others: max_sh1=%d max_sh2=%d => need1=%d need2=%d",
                                 max_sh1, max_sh2, need1, need2));

   int b1 = SafeBars(g_sym1, _Period);
   int b2 = SafeBars(g_sym2, _Period);
   if(b1 < need1 || b2 < need2)
   {
      DbgPrint("[FAIL]", StringFormat("Not enough bars for others. b1=%d need1=%d | b2=%d need2=%d", b1, need1, b2, need2));
      DbgPrint("[DBG]", "Tip: otevři graf EURUSD a XAUUSD na stejném TF a nech terminál stáhnout historii.");
      MarkDbgFail();
      return false;
   }

   ArraySetAsSeries(g_r_1, true);
   ArraySetAsSeries(g_r_2, true);

   int got1 = SafeCopyRates(g_sym1, _Period, 0, need1, g_r_1);
   int got2 = SafeCopyRates(g_sym2, _Period, 0, need2, g_r_2);
   if(got1 < need1 || got2 < need2)
   {
      DbgPrint("[FAIL]", StringFormat("CopyRates others insufficient. got1=%d need1=%d | got2=%d need2=%d", got1, need1, got2, need2));
      MarkDbgFail();
      return false;
   }

   // vmax over actual loaded ranges
   g_vmax_main = 1.0; for(int i=0;i<ArraySize(g_r_main);i++) g_vmax_main = MathMax(g_vmax_main, (double)g_r_main[i].tick_volume);
   g_vmax_1    = 1.0; for(int i=0;i<ArraySize(g_r_1);i++)    g_vmax_1    = MathMax(g_vmax_1,    (double)g_r_1[i].tick_volume);
   g_vmax_2    = 1.0; for(int i=0;i<ArraySize(g_r_2);i++)    g_vmax_2    = MathMax(g_vmax_2,    (double)g_r_2[i].tick_volume);
   DbgPrint("[OK]", StringFormat("vmax: main=%.0f s1=%.0f s2=%.0f", g_vmax_main, g_vmax_1, g_vmax_2));

   // Build up to train_bars samples (with possible filtering)
   samples = 0;

   ArrayResize(X, train_bars * idim);
   ArrayResize(T, train_bars);

   // --- targets precompute on main
   double sum=0.0, sum2=0.0;
   double tmpRawT[];
   ArrayResize(tmpRawT, max_attempt);

   int attempts = 0;

   for(int s_try=0; s_try<max_attempt; s_try++)
   {
      // Chronological indexing:
      // i_ch_end increases with s_try; map to series index via (need_main-1)-i_ch_end.
      int i_ch_end = (seq - 1) + s_try;
      int i_series_end = (need_main - 1) - i_ch_end;
      if(i_series_end < 0 || i_series_end >= need_main) break;

      double c0 = g_r_main[i_series_end].close;
      if(c0 <= 0) break;

      double mxH=-1e100, mnL=1e100;
      for(int j=1; j<=H; j++)
      {
         int i_ch_f = i_ch_end + j;
         int i_series_f = (need_main - 1) - i_ch_f;
         if(i_series_f < 0) { mxH=-1e100; break; }
         mxH = MathMax(mxH, g_r_main[i_series_f].high);
         mnL = MathMin(mnL, g_r_main[i_series_f].low);
      }
      if(mxH < -1e50) break;

      double range = (mxH - mnL) / c0;
      if(range < 0) range = 0;
      double z = MathLog(1.0 + range);

      tmpRawT[s_try] = z;
      sum += z; sum2 += z*z;
      attempts++;
   }

   if(attempts < train_bars)
   {
      DbgPrint("[FAIL]", StringFormat("Not enough attempts for training. attempts=%d required=%d", attempts, train_bars));
      MarkDbgFail();
      return false;
   }

   g_t_mean = sum / attempts;
   g_t_std  = MathSqrt(MathMax(1e-10, sum2/attempts - g_t_mean*g_t_mean));

   int idx0_series_end = (need_main - 1) - ((seq - 1) + 0);
   DbgPrint("[DBG]", StringFormat("Target sample0: time_end=%s c0=%.5f z=%.6f",
                                 TimeToStrSec(g_r_main[idx0_series_end].time),
                                 g_r_main[idx0_series_end].close, tmpRawT[0]));
   DbgPrint("[OK]", StringFormat("Target stats: mean=%.8f std=%.8f (attempts=%d)", g_t_mean, g_t_std, attempts));

   // 2nd pass: build X/T samples
   for(int s_try=0; s_try<attempts && samples < train_bars; s_try++)
   {
      // optional: detect frozen shifts on extra symbols within this sequence
      int last_sh1 = INT_MAX, last_sh2 = INT_MAX;
      int chg1 = 0, chg2 = 0;

      double zn = (tmpRawT[s_try] - g_t_mean) / g_t_std;
      if(zn > InpTargetClamp) zn = InpTargetClamp;
      if(zn < -InpTargetClamp) zn = -InpTargetClamp;

      int i_ch_end = (seq - 1) + s_try;
      int i_ch_start = i_ch_end - (seq - 1);

      int shM_arr[], sh1_arr[], sh2_arr[];
      ArrayResize(shM_arr, seq);
      ArrayResize(sh1_arr, seq);
      ArrayResize(sh2_arr, seq);

      bool ok = true;

      // collect shifts for all t
      for(int t=0; t<seq; t++)
      {
         int i_ch = i_ch_start + t;
         int i_series = (need_main - 1) - i_ch;
         if(i_series < 0 || i_series >= need_main) { ok=false; break; }

         datetime tm = g_r_main[i_series].time;

         int shM = iBarShift(g_sym_main, _Period, tm, exact);
         int sh1 = iBarShift(g_sym1,     _Period, tm, exact);
         int sh2 = iBarShift(g_sym2,     _Period, tm, exact);

         if(InpSuperDebug && s_try==0 && t < 3 && DbgCan())
         {
            DbgPrint("[DBG]", StringFormat("t=%d time=%s shM=%d sh1=%d sh2=%d", t, TimeToStrSec(tm), shM, sh1, sh2));
         }

         if(shM < 0 || sh1 < 0 || sh2 < 0) { ok=false; break; }

         // CRITICAL: bounds check vs actually loaded arrays
         if(shM >= ArraySize(g_r_main) || sh1 >= ArraySize(g_r_1) || sh2 >= ArraySize(g_r_2))
         {
            DbgPrint("[FAIL]", StringFormat("Shift OOR: shM=%d/%d sh1=%d/%d sh2=%d/%d time=%s",
               shM, ArraySize(g_r_main), sh1, ArraySize(g_r_1), sh2, ArraySize(g_r_2), TimeToStrSec(tm)));
            ok=false; break;
         }

         shM_arr[t] = shM;
         sh1_arr[t] = sh1;
         sh2_arr[t] = sh2;

         if(InpFilterFrozenOtherSymbols)
         {
            if(last_sh1 != INT_MAX && sh1 != last_sh1) chg1++;
            if(last_sh2 != INT_MAX && sh2 != last_sh2) chg2++;
            last_sh1 = sh1;
            last_sh2 = sh2;
         }
      }

      if(!ok) continue;

      if(InpFilterFrozenOtherSymbols)
      {
         if(chg1 < InpMinShiftChanges || chg2 < InpMinShiftChanges)
         {
            // likely weekend / FX frozen mapping -> skip
            continue;
         }
      }

      // fill X
      for(int t=0; t<seq; t++)
      {
         int shM = shM_arr[t];
         int sh1 = sh1_arr[t];
         int sh2 = sh2_arr[t];

         double cM = g_r_main[shM].close;
         double c1 = g_r_1[sh1].close;
         double c2 = g_r_2[sh2].close;
         if(cM<=0||c1<=0||c2<=0) { ok=false; break; }

         // returns computed within sequence (no sh+1!)
         double retM=0.0, ret1=0.0, ret2=0.0;
         if(t > 0)
         {
            double cMp = g_r_main[shM_arr[t-1]].close;
            double c1p = g_r_1[sh1_arr[t-1]].close;
            double c2p = g_r_2[sh2_arr[t-1]].close;
            if(cMp>0) retM = MathLog(cM / cMp);
            if(c1p>0) ret1 = MathLog(c1 / c1p);
            if(c2p>0) ret2 = MathLog(c2 / c2p);
         }

         // ranges and body
         double f_m1 = (g_r_main[shM].high - g_r_main[shM].low) / cM;
         double f_m2 = (g_r_main[shM].close - g_r_main[shM].open) / cM;
         double f_m3 = (g_vmax_main>0)? (double)g_r_main[shM].tick_volume / g_vmax_main : 0;

         double f_11 = (g_r_1[sh1].high - g_r_1[sh1].low) / c1;
         double f_12 = (g_r_1[sh1].close - g_r_1[sh1].open) / c1;
         double f_13 = (g_vmax_1>0)? (double)g_r_1[sh1].tick_volume / g_vmax_1 : 0;

         double f_21 = (g_r_2[sh2].high - g_r_2[sh2].low) / c2;
         double f_22 = (g_r_2[sh2].close - g_r_2[sh2].open) / c2;
         double f_23 = (g_vmax_2>0)? (double)g_r_2[sh2].tick_volume / g_vmax_2 : 0;

         int off = samples*idim + t*feat_total;
         X[off+0]=retM;  X[off+1]=f_m1; X[off+2]=f_m2; X[off+3]=f_m3;
         X[off+4]=ret1;  X[off+5]=f_11; X[off+6]=f_12; X[off+7]=f_13;
         X[off+8]=ret2;  X[off+9]=f_21; X[off+10]=f_22; X[off+11]=f_23;
      }

      if(!ok) continue;

      T[samples] = zn;
      samples++;
   }

   DbgPrint("[DBG]", StringFormat("Sampling result: valid=%d / attempts=%d (%.1f%%)",
                                 samples, attempts,
                                 (attempts>0 ? 100.0*(double)samples/(double)attempts : 0.0)));

   if(samples < train_bars)
   {
      DbgPrint("[FAIL]", StringFormat("Filtering removed too many samples. got=%d required=%d", samples, train_bars));
      DbgPrint("[DBG]", "Tip: zvyš InpAttemptMult, vypni InpFilterFrozenOtherSymbols nebo sniž InpMinShiftChanges.");
      MarkDbgFail();
      return false;
   }

   ArrayResize(X, samples * idim);

   ulong t1 = NowUS();
   DbgPrint("[OK]", StringFormat("PrepareData OK. samples=%d idim=%d time=%.3f ms",
                                 samples, idim, (double)(t1-t0)/1000.0));
   DbgPrint("[DBG]", "================ SUPER-DEBUG: PrepareData END ==================");
   return true;
}

// ============================================================================
// Async training orchestration
// ============================================================================
bool StartAsyncTraining(int train_bars, int epochs, bool retrain)
{
   if(g_async_training) return false;

   double X[], T[];
   int samples = 0;

   if(!PrepareData(X, T, samples, train_bars))
   {
      Print("[FAIL] PrepareData selhala (multi-symbol / target range)");
      return false;
   }

   int feat_total = 4 * 3;
   int idim = InpSeqLen * feat_total;

   if(InpSuperDebug)
      Print("[DBG] LoadBatch: samples=", samples, " idim=", idim, " out_dim=1 layout=0");

   if(!DN_LoadBatch(g_net, X, T, samples, idim, 1, 0))
   {
      Print("[FAIL] DN_LoadBatch chyba: ", GetDLLError());
      return false;
   }

   if(retrain) DN_SnapshotWeights(g_net);

   g_train_start_tick = GetTickCount();
   g_is_retrain = retrain;

   double lr = retrain ? InpLR * 0.5 : InpLR;

   if(!DN_TrainAsync(g_net, epochs, InpTargetMSE, lr, InpWD))
   {
      Print("[FAIL] DN_TrainAsync selhal: ", GetDLLError());
      if(retrain) DN_RestoreWeights(g_net);
      return false;
   }

   g_async_training = true;
   EventSetTimer(1);
   Print(retrain ? "[OK] Spouštím RETRAIN na pozadí..." : "[OK] Spouštím PRVNÍ TRÉNINK na pozadí...");
   return true;
}

void CheckTrainingStatus()
{
   if(!g_async_training)
   {
      EventKillTimer();
      return;
   }

   int st = DN_GetTrainingStatus(g_net);

   if(st == 1)
   {
      if(InpShowComment)
      {
         uint elapsed = (GetTickCount() - g_train_start_tick) / 1000;
         Comment("LSTM: trénink běží... ", IntegerToString((int)elapsed), " s");
      }
      return;
   }

   EventKillTimer();
   g_async_training = false;

   if(st == 2)
   {
      double mse; int ep;
      DN_GetTrainingResult(g_net, mse, ep);
      double dur = (GetTickCount() - g_train_start_tick) / 1000.0;

      Print(g_is_retrain ? "[OK] RETRAIN" : "[OK] TRÉNINK",
            " Hotovo: ", ep, " epoch, MSE=", DoubleToString(mse, 6),
            " (", DoubleToString(dur, 1), "s)");

      DN_SnapshotWeights(g_net);

      g_trained = true;
      ResetThrHistory();
      ChartRedraw(0);
   }
   else
   {
      Print("[FAIL] Trénink selhal: ", GetDLLError());
      if(g_is_retrain) DN_RestoreWeights(g_net);
      g_trained = false;
      if(InpShowComment) Comment("LSTM: trénink selhal");
   }
}

// ============================================================================
// Build input for inference at bar index
// Robustly load enough depth for s1/s2 based on required times
// ============================================================================
bool BuildInputAtShift(double &X[], int shift_main, int &idim_out)
{
   const int seq = InpSeqLen;
   const int feat_total = 4 * 3;
   const int idim = seq * feat_total;
   idim_out = idim;

   bool exact = InpAlignExact;

   int need_main = shift_main + seq + 5;
   if(Bars(g_sym_main, _Period) < need_main) return false;

   ArraySetAsSeries(g_r_main, true);
   if(CopyRates(g_sym_main, _Period, 0, need_main, g_r_main) < need_main) return false;

   int max_sh1 = -1, max_sh2 = -1;
   for(int t=0; t<seq; t++)
   {
      int shMain = shift_main + (seq - 1 - t);
      if(shMain < 0 || shMain >= ArraySize(g_r_main)) return false;
      datetime tm = g_r_main[shMain].time;

      int sh1 = iBarShift(g_sym1, _Period, tm, exact);
      int sh2 = iBarShift(g_sym2, _Period, tm, exact);
      if(sh1 < 0 || sh2 < 0) return false;

      if(sh1 > max_sh1) max_sh1 = sh1;
      if(sh2 > max_sh2) max_sh2 = sh2;
   }

   int need1 = max_sh1 + 3;
   int need2 = max_sh2 + 3;

   if(Bars(g_sym1, _Period) < need1) return false;
   if(Bars(g_sym2, _Period) < need2) return false;

   ArraySetAsSeries(g_r_1, true);
   ArraySetAsSeries(g_r_2, true);

   if(CopyRates(g_sym1, _Period, 0, need1, g_r_1) < need1) return false;
   if(CopyRates(g_sym2, _Period, 0, need2, g_r_2) < need2) return false;

   g_vmax_main=1; for(int i=0;i<ArraySize(g_r_main);i++) g_vmax_main=MathMax(g_vmax_main,(double)g_r_main[i].tick_volume);
   g_vmax_1=1;    for(int i=0;i<ArraySize(g_r_1);i++)    g_vmax_1   =MathMax(g_vmax_1,   (double)g_r_1[i].tick_volume);
   g_vmax_2=1;    for(int i=0;i<ArraySize(g_r_2);i++)    g_vmax_2   =MathMax(g_vmax_2,   (double)g_r_2[i].tick_volume);

   ArrayResize(X, idim);

   int shM_arr[], sh1_arr[], sh2_arr[];
   ArrayResize(shM_arr, seq);
   ArrayResize(sh1_arr, seq);
   ArrayResize(sh2_arr, seq);

   for(int t=0; t<seq; t++)
   {
      int shMain = shift_main + (seq - 1 - t);
      datetime tm = g_r_main[shMain].time;

      int shM = iBarShift(g_sym_main, _Period, tm, exact);
      int sh1 = iBarShift(g_sym1,     _Period, tm, exact);
      int sh2 = iBarShift(g_sym2,     _Period, tm, exact);
      if(shM < 0 || sh1 < 0 || sh2 < 0) return false;

      if(shM >= ArraySize(g_r_main) || sh1 >= ArraySize(g_r_1) || sh2 >= ArraySize(g_r_2)) return false;

      shM_arr[t]=shM; sh1_arr[t]=sh1; sh2_arr[t]=sh2;
   }

   for(int t=0; t<seq; t++)
   {
      int shM = shM_arr[t], sh1 = sh1_arr[t], sh2 = sh2_arr[t];

      double cM = g_r_main[shM].close;
      double c1 = g_r_1[sh1].close;
      double c2 = g_r_2[sh2].close;
      if(cM<=0||c1<=0||c2<=0) return false;

      double retM=0.0, ret1=0.0, ret2=0.0;
      if(t > 0)
      {
         double cMp = g_r_main[shM_arr[t-1]].close;
         double c1p = g_r_1[sh1_arr[t-1]].close;
         double c2p = g_r_2[sh2_arr[t-1]].close;
         if(cMp>0) retM = MathLog(cM / cMp);
         if(c1p>0) ret1 = MathLog(c1 / c1p);
         if(c2p>0) ret2 = MathLog(c2 / c2p);
      }

      double f_m1 = (g_r_main[shM].high - g_r_main[shM].low) / cM;
      double f_m2 = (g_r_main[shM].close - g_r_main[shM].open) / cM;
      double f_m3 = (g_vmax_main>0)? (double)g_r_main[shM].tick_volume / g_vmax_main : 0;

      double f_11 = (g_r_1[sh1].high - g_r_1[sh1].low) / c1;
      double f_12 = (g_r_1[sh1].close - g_r_1[sh1].open) / c1;
      double f_13 = (g_vmax_1>0)? (double)g_r_1[sh1].tick_volume / g_vmax_1 : 0;

      double f_21 = (g_r_2[sh2].high - g_r_2[sh2].low) / c2;
      double f_22 = (g_r_2[sh2].close - g_r_2[sh2].open) / c2;
      double f_23 = (g_vmax_2>0)? (double)g_r_2[sh2].tick_volume / g_vmax_2 : 0;

      int off = t*feat_total;
      X[off+0]=retM;  X[off+1]=f_m1;  X[off+2]=f_m2;  X[off+3]=f_m3;
      X[off+4]=ret1;  X[off+5]=f_11;  X[off+6]=f_12;  X[off+7]=f_13;
      X[off+8]=ret2;  X[off+9]=f_21;  X[off+10]=f_22; X[off+11]=f_23;
   }

   return true;
}

// ============================================================================
// Predict
// ============================================================================
bool PredictForBar(int bar_idx, double &out_raw, double &out_move_pct)
{
   if(!g_trained) return false;
   if(g_async_training) return false;

   int shift_main = bar_idx + 1;
   double X[];
   int idim = 0;

   if(!BuildInputAtShift(X, shift_main, idim)) return false;

   double Y[];
   ArrayResize(Y, 1);
   if(!DN_PredictBatch(g_net, X, 1, idim, 0, Y)) return false;

   out_raw = Y[0];
   out_move_pct = DenormExpectedMovePct(out_raw);
   return true;
}

// ============================================================================
// History calc
// ============================================================================
void CalculateHistory(int rates_total)
{
   if(!g_trained || g_async_training) return;

   int max_bar = rates_total - 5;
   int calc_bars = MathMin(InpCalcHistory, max_bar);
   if(calc_bars <= 0) return;

   ResetThrHistory();

   int counter = 0;
   for(int bar = calc_bars; bar >= 0; bar--)
   {
      double raw, mv;
      if(!PredictForBar(bar, raw, mv))
      {
         g_buf_move_pct[bar] = EMPTY_VALUE;
         g_buf_raw[bar]      = EMPTY_VALUE;
         g_buf_thr_low[bar]  = EMPTY_VALUE;
         g_buf_thr_high[bar] = EMPTY_VALUE;
         continue;
      }

      counter++;
      PushThrSample(mv, counter);

      double thrL = PercentileFromHist(InpThrLowPercentile);
      double thrH = PercentileFromHist(InpThrHighPercentile);

      g_buf_move_pct[bar] = mv;
      g_buf_raw[bar]      = raw;
      g_buf_thr_low[bar]  = thrL;
      g_buf_thr_high[bar] = thrH;
   }
}

// ============================================================================
// Comment
// ============================================================================
void UpdateComment(double move_pct, double thrL, double thrH)
{
   if(!InpShowComment) return;
   if(g_async_training) return;

   string state = "LOW MOVE";
   if(move_pct >= thrH) state = "HIGH MOVE";
   else if(move_pct >= thrL) state = "MOVE OK";

   string txt =
      "ExpectedMove: " + DoubleToString(move_pct, 3) + " %\n" +
      "ThrLow: " + DoubleToString(thrL, 3) + " % | ThrHigh: " + DoubleToString(thrH, 3) + " %\n" +
      "State: " + state + "\n" +
      "Inputs: " + g_sym_main + " + " + g_sym1 + " + " + g_sym2 + " | TF=" + TFToStr(_Period);

   Comment(txt);
}

// ============================================================================
// Events
// ============================================================================
int OnInit()
{
   g_sym_main = _Symbol;
   g_sym1 = InpExtraSymbol1;
   g_sym2 = InpExtraSymbol2;

   SymbolSelect(g_sym1, true);
   SymbolSelect(g_sym2, true);

   SetIndexBuffer(0, g_buf_move_pct, INDICATOR_DATA);
   SetIndexBuffer(1, g_buf_thr_low,  INDICATOR_DATA);
   SetIndexBuffer(2, g_buf_thr_high, INDICATOR_DATA);
   SetIndexBuffer(3, g_buf_raw,      INDICATOR_CALCULATIONS);

   ArraySetAsSeries(g_buf_move_pct, true);
   ArraySetAsSeries(g_buf_thr_low,  true);
   ArraySetAsSeries(g_buf_thr_high, true);
   ArraySetAsSeries(g_buf_raw,      true);

   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetDouble(2, PLOT_EMPTY_VALUE, EMPTY_VALUE);

   ResetThrHistory();

   g_net = DN_Create();
   if(g_net <= 0)
   {
      Print("[FAIL] DN_Create: ", GetDLLError());
      return INIT_FAILED;
   }

   DN_SetSequenceLength(g_net, InpSeqLen);
   DN_SetMiniBatchSize(g_net, InpMiniBatch);
   DN_SetGradClip(g_net, InpGradClip);
   DN_AddLayerEx(g_net, 0, InpHiddenSize, 0, 0, InpDropout);

   // Kick off training if enough main bars exist (others will be validated inside PrepareData)
   int attempt_mult = MathMax(1, InpAttemptMult);
   int need_hint = (InpTrainBars * attempt_mult) + InpSeqLen + InpHorizonH + 10;

   if(Bars(g_sym_main, _Period) >= need_hint)
   {
      g_last_train_attempt = TimeCurrent();
      if(!StartAsyncTraining(InpTrainBars, InpEpochs, false))
         Print("[FAIL] StartAsyncTraining failed at OnInit.");
   }
   else
   {
      Print("[DBG] Not enough bars yet on main. Need ~", need_hint, " bars (main).");
   }

   g_init_done = true;
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
   Comment("");
}

void OnTimer()
{
   CheckTrainingStatus();
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
   if(!g_init_done || g_net <= 0) return prev_calculated;
   if(g_async_training) return prev_calculated;

   // If not trained yet, try to start training when enough bars exist (with cooldown)
   if(!g_trained)
   {
      if((TimeCurrent() - g_last_train_attempt) >= InpTrainRetrySec)
      {
         g_last_train_attempt = TimeCurrent();
         StartAsyncTraining(InpTrainBars, InpEpochs, false);
      }
      return prev_calculated;
   }

   // Initial history calculation
   if(prev_calculated == 0)
      CalculateHistory(rates_total);

   // Current prediction
   double raw0, mv0;
   if(!PredictForBar(0, raw0, mv0)) return rates_total;

   static int counter = 0;
   counter++;
   PushThrSample(mv0, counter);

   double thrL = PercentileFromHist(InpThrLowPercentile);
   double thrH = PercentileFromHist(InpThrHighPercentile);

   g_buf_move_pct[0] = mv0;
   g_buf_raw[0]      = raw0;
   g_buf_thr_low[0]  = thrL;
   g_buf_thr_high[0] = thrH;

   UpdateComment(mv0, thrL, thrH);

   return rates_total;
}
//+------------------------------------------------------------------+
