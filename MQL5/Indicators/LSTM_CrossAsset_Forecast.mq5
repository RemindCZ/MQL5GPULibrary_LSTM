// Multi-symbol, direct multi-horizon forecast. No trading operations.
#property copyright "Tomas Belak / RemindCZ"
#property version "2.30"
#property strict
#property indicator_chart_window
#property indicator_buffers 3
#property indicator_plots 3
#property indicator_label1 "Future forecast"
#property indicator_type1 DRAW_LINE
#property indicator_color1 clrDeepSkyBlue
#property indicator_width1 2
#property indicator_label2 "Validation error upper"
#property indicator_type2 DRAW_LINE
#property indicator_color2 clrSlateGray
#property indicator_style2 STYLE_DOT
#property indicator_label3 "Validation error lower"
#property indicator_type3 DRAW_LINE
#property indicator_color3 clrSlateGray
#property indicator_style3 STYLE_DOT
#include <MQL5GPULibrary.mqh>
#include <MQL5GPUTopology.mqh>

input group "Markets (attach to the target chart; suggested AUDUSD H1)"
input string InpDrivers="NZDUSD,USDJPY,USDCAD"; // Exact broker names; commodities can be added
input ENUM_TIMEFRAMES InpTimeframe=PERIOD_H1;
input int InpHistoryBars=2500;
input int InpSequence=24;
input int InpHorizon=6;
input group "Graph: type:width:activation:dropout:merge:parents;..."
input string InpTopology="lstm:32:linear:0.10:concat:-1;gru:16:linear:0.10:concat:-1;dense:24:tanh:0:concat:0,1";
input int InpEpochs=60;
input int InpBatch=64;
input double InpLearningRate=0.001;
input int InpRetrainBars=24;
input group "Out-of-sample validation"
input double InpValidationFraction=0.20;
input bool InpRequireSkill=true;
input double InpMaximumMSERatio=0.98; // Model / zero-return baseline, lower is better
input double InpBandMultiplier=1.96; // Residual RMSE multiplier, not a calibrated confidence level
input double InpMaxAbsForecastReturn=0.10; // Hide implausible predictions rather than clipping them
input bool InpWriteForecastCSV=true; // Prospective forecasts only, MQL5/Files
input string InpInstance="default";

double Forecast[],Upper[],Lower[];
string Symbols[],Status="Waiting for history";
int Active=0,Candidate=0,FeatureCount=0,ValidationCount=0;
double ActiveMean[],ActiveStd[],CandidateMean[],CandidateStd[],ResidualRMSE[];
double ValidationX[],ValidationT[],PendingRMSE[];
double LastPrediction[],LastAnchor=0,LastScore=1;
datetime LastOrigin=0,LastAttempt=0,LastLogged=0;
bool HavePrediction=false,ActiveSkilled=false;
bool OriginalChartShift=false,ChartShiftChanged=false,LabelCreated=false,StatusLabelCreated=false;
ulong LastRetry=0;
string LabelName,StatusName,LogFile;

int FindTime(const MqlRates &rates[],const datetime wanted)
{
   int lo=0,hi=ArraySize(rates)-1;
   while(lo<=hi) { int mid=(lo+hi)/2; if(rates[mid].time==wanted) return mid;
      if(rates[mid].time<wanted) lo=mid+1; else hi=mid-1; }
   return -1;
}

// Return rows correspond exactly to target bar timestamps, with all markets present.
// A row uses that bar's close and the previous target bar's close only.
bool Collect(double &features[],double &prices[],datetime &times[],const int requested)
{
   MqlRates target[];
   int n=CopyRates(Symbols[0],InpTimeframe,1,requested,target);
   if(n<InpSequence+InpHorizon+100) { Status="Loading target history"; return false; }
   ArrayResize(features,n*FeatureCount); ArrayInitialize(features,0);
   bool valid[]; ArrayResize(valid,n); ArrayInitialize(valid,true); valid[0]=false;
   for(int s=0;s<FeatureCount;s++)
   {
      MqlRates rates[];
      int copied=CopyRates(Symbols[s],InpTimeframe,target[0].time,target[n-1].time,rates);
      if(copied<2) { Status="Loading history: "+Symbols[s]; return false; }
      for(int i=1;i<n;i++)
      {
         int a=FindTime(rates,target[i].time),b=FindTime(rates,target[i-1].time);
         if(a<0 || b<0 || rates[a].close<=0 || rates[b].close<=0) { valid[i]=false; continue; }
         features[i*FeatureCount+s]=MathLog(rates[a].close/rates[b].close);
         if(!MathIsValidNumber(features[i*FeatureCount+s])) valid[i]=false;
      }
   }
   ArrayResize(prices,n); ArrayResize(times,n);
   for(int i=0;i<n;i++) { prices[i]=target[i].close; times[i]=target[i].time;
      if(!valid[i]) for(int f=0;f<FeatureCount;f++) features[i*FeatureCount+f]=EMPTY_VALUE; }
   return true;
}

bool WindowValid(const double &rows[],const int end)
{
   if(end<InpSequence) return false;
   for(int t=end-InpSequence+1;t<=end;t++)
      for(int f=0;f<FeatureCount;f++) if(rows[t*FeatureCount+f]==EMPTY_VALUE) return false;
   return true;
}

void Encode(const double &rows[],const int end,const double &mean[],const double &scale[],double &dest[],const int offset)
{
   for(int t=0;t<InpSequence;t++) for(int f=0;f<FeatureCount;f++)
      dest[offset+t*FeatureCount+f]=(rows[(end-InpSequence+1+t)*FeatureCount+f]-mean[f])/scale[f];
}

bool PrepareTraining(const double &rows[],const double &prices[],double &x[],double &y[])
{
   int n=ArraySize(prices),last=n-1-InpHorizon;
   int cut=InpSequence+(int)((last-InpSequence+1)*(1.0-InpValidationFraction));
   // All training labels finish strictly before the first validation origin.
   int train_end=cut-InpHorizon-1;
   if(train_end<InpSequence+64 || last-cut<32) { Status="Insufficient history for purged validation"; return false; }
   ArrayResize(CandidateMean,FeatureCount); ArrayInitialize(CandidateMean,0);
   ArrayResize(CandidateStd,FeatureCount); ArrayInitialize(CandidateStd,0);
   int rows_used=0;
   for(int i=1;i<=train_end;i++)
   {
      if(rows[i*FeatureCount]==EMPTY_VALUE) continue;
      rows_used++;
      for(int f=0;f<FeatureCount;f++) CandidateMean[f]+=rows[i*FeatureCount+f];
   }
   if(rows_used<64) { Status="Too few aligned training rows"; return false; }
   for(int f=0;f<FeatureCount;f++) CandidateMean[f]/=rows_used;
   for(int i=1;i<=train_end;i++) if(rows[i*FeatureCount]!=EMPTY_VALUE)
      for(int f=0;f<FeatureCount;f++) CandidateStd[f]+=MathPow(rows[i*FeatureCount+f]-CandidateMean[f],2);
   for(int f=0;f<FeatureCount;f++) CandidateStd[f]=MathMax(1e-8,MathSqrt(CandidateStd[f]/rows_used));
   int train_count=0; ValidationCount=0;
   for(int i=InpSequence;i<=last;i++) if(WindowValid(rows,i))
   { if(i<=train_end) train_count++; else if(i>=cut) ValidationCount++; }
   if(train_count<64 || ValidationCount<32) { Status="Too few synchronized windows; check market sessions"; return false; }
   int width=InpSequence*FeatureCount;
   ArrayResize(x,train_count*width); ArrayResize(y,train_count*InpHorizon);
   ArrayResize(ValidationX,ValidationCount*width); ArrayResize(ValidationT,ValidationCount*InpHorizon);
   int tr=0,va=0;
   for(int i=InpSequence;i<=last;i++)
   {
      if(!WindowValid(rows,i)) continue;
      if(i<=train_end)
      {
         Encode(rows,i,CandidateMean,CandidateStd,x,tr*width);
         for(int h=1;h<=InpHorizon;h++) y[tr*InpHorizon+h-1]=100.0*MathLog(prices[i+h]/prices[i]);
         tr++;
      }
      else if(i>=cut)
      {
         Encode(rows,i,CandidateMean,CandidateStd,ValidationX,va*width);
         for(int h=1;h<=InpHorizon;h++) ValidationT[va*InpHorizon+h-1]=100.0*MathLog(prices[i+h]/prices[i]);
         va++;
      }
   }
   return true;
}

bool StartCandidate(const double &rows[],const double &prices[])
{
   double x[],y[];
   if(!PrepareTraining(rows,prices,x,y)) return false;
   Candidate=DN_Create();
   if(Candidate<=0) { Status=GPUErrorText(); return false; }
   string reason;
   bool ok=GPUBuildTopology(Candidate,FeatureCount,InpSequence,InpHorizon,InpTopology,reason)
      && DN_SetMiniBatchSize(Candidate,InpBatch) && DN_SetGradClip(Candidate,5.0)
      && DN_LoadBatch(Candidate,x,y,ArraySize(y)/InpHorizon,InpSequence*FeatureCount,InpHorizon,0)
      && DN_TrainAsync(Candidate,InpEpochs,0,InpLearningRate,0.0001);
   if(!ok) { Status="Training setup: "+reason+" "+GPUErrorText(); DN_Free(Candidate); Candidate=0; return false; }
   Status="Training candidate; active forecast remains available";
   return true;
}

void CheckCandidate()
{
   if(Candidate<=0) return;
   int state=DN_GetTrainingStatus(Candidate);
   if(state==1) { Status=StringFormat("Training %.1f%% | active validation ratio %.3f",DN_GetProgressPercent(Candidate),LastScore); return; }
   if(state!=2) { Status="Training failed: "+GPUErrorText(); DN_Free(Candidate); Candidate=0; return; }
   double predicted[]; ArrayResize(predicted,ArraySize(ValidationT));
   if(!DN_PredictBatch(Candidate,ValidationX,ValidationCount,InpSequence*FeatureCount,0,predicted))
   { Status="Validation failed: "+GPUErrorText(); DN_Free(Candidate); Candidate=0; return; }
   double loss=0,baseline=0;
   ArrayResize(PendingRMSE,InpHorizon); ArrayInitialize(PendingRMSE,0);
   for(int i=0;i<ArraySize(predicted);i++)
   {
      if(!MathIsValidNumber(predicted[i])) { Status="Non-finite validation output"; DN_Free(Candidate); Candidate=0; return; }
      double e=predicted[i]-ValidationT[i]; loss+=e*e; baseline+=ValidationT[i]*ValidationT[i];
      PendingRMSE[i%InpHorizon]+=e*e;
   }
   double ratio=baseline>1e-20 ? loss/baseline : DBL_MAX;
   bool skilled=ratio<InpMaximumMSERatio;
   if(skilled || !InpRequireSkill)
   {
      if(Active>0) DN_Free(Active);
      Active=Candidate; Candidate=0; ActiveSkilled=skilled; LastScore=ratio;
      ArrayCopy(ActiveMean,CandidateMean); ArrayCopy(ActiveStd,CandidateStd);
      ArrayResize(ResidualRMSE,InpHorizon);
      for(int h=0;h<InpHorizon;h++) ResidualRMSE[h]=MathSqrt(PendingRMSE[h]/ValidationCount)/100.0;
      LastOrigin=0; HavePrediction=false;
      Status=StringFormat("Model accepted | validation MSE / baseline %.3f | %d windows",ratio,ValidationCount);
   }
   else
   {
      Status=StringFormat("Candidate rejected: MSE / baseline %.3f; %s",ratio,Active>0 ? "retaining prior model" : "showing persistence baseline");
      DN_Free(Candidate); Candidate=0;
   }
}

void PredictLatest(const double &rows[],const double &prices[],const datetime &times[])
{
   int end=ArraySize(prices)-1;
   if(!WindowValid(rows,end)) { HavePrediction=false; Status="Latest cross-market bars are incomplete; forecast hidden"; return; }
   if(times[end]==LastOrigin && HavePrediction) return;
   ArrayResize(LastPrediction,InpHorizon);
   if(Active>0)
   {
      double x[]; ArrayResize(x,InpSequence*FeatureCount);
      Encode(rows,end,ActiveMean,ActiveStd,x,0);
      if(!DN_PredictBatch(Active,x,1,ArraySize(x),0,LastPrediction)) { Status=GPUErrorText(); HavePrediction=false; return; }
      for(int h=0;h<InpHorizon;h++)
      {
         LastPrediction[h]/=100.0;
         if(!MathIsValidNumber(LastPrediction[h]) || MathAbs(LastPrediction[h])>InpMaxAbsForecastReturn)
         { Status="Forecast outside configured plausibility limit; hidden"; HavePrediction=false; return; }
      }
   }
   else ArrayInitialize(LastPrediction,0.0);
   LastAnchor=prices[end]; LastOrigin=times[end]; HavePrediction=true;
   if(InpWriteForecastCSV && LastLogged!=LastOrigin)
   {
      int file=FileOpen(LogFile,FILE_READ|FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_SHARE_READ,',');
      if(file!=INVALID_HANDLE)
      {
         if(FileSize(file)==0) FileWrite(file,"origin_bar_open","symbol","horizon_bars","anchor_close","forecast_close","model","validation_mse_ratio");
         FileSeek(file,0,SEEK_END);
         for(int h=0;h<InpHorizon;h++) FileWrite(file,TimeToString(LastOrigin,TIME_DATE|TIME_MINUTES),_Symbol,h+1,
            DoubleToString(LastAnchor,_Digits),DoubleToString(LastAnchor*MathExp(LastPrediction[h]),_Digits),
            Active>0 ? (ActiveSkilled ? "validated" : "unvalidated") : "persistence",DoubleToString(LastScore,6));
         FileClose(file); LastLogged=LastOrigin;
      }
   }
}

void Paint()
{
   if(ArraySize(Forecast)<InpHorizon+2) return;
   ArrayInitialize(Forecast,EMPTY_VALUE); ArrayInitialize(Upper,EMPTY_VALUE); ArrayInitialize(Lower,EMPTY_VALUE);
   if(HavePrediction)
   {
      Forecast[InpHorizon+1]=LastAnchor;
      for(int h=1;h<=InpHorizon;h++)
      {
         int index=InpHorizon+1-h;
         Forecast[index]=LastAnchor*MathExp(LastPrediction[h-1]);
         if(Active>0 && ArraySize(ResidualRMSE)==InpHorizon)
         {
            double band=InpBandMultiplier*ResidualRMSE[h-1];
            Upper[index]=LastAnchor*MathExp(LastPrediction[h-1]+band);
            Lower[index]=LastAnchor*MathExp(LastPrediction[h-1]-band);
         }
      }
   }
   PlotIndexSetInteger(0,PLOT_LINE_COLOR,Active>0 ? (ActiveSkilled ? clrDeepSkyBlue : clrOrange) : clrGray);
   string text="Cross-asset "+_Symbol+" | closed bars | "+IntegerToString(InpHorizon)+" horizons";
   ObjectSetString(0,LabelName,OBJPROP_TEXT,text);
   ObjectSetString(0,StatusName,OBJPROP_TEXT,Status);
   ChartRedraw();
}

void OnTimer()
{
   CheckCandidate();
   datetime current=iTime(_Symbol,InpTimeframe,0);
   if(current==0) return;
   if(GetTickCount64()-LastRetry<1000) { Paint(); return; }
   LastRetry=GetTickCount64();
   double rows[],prices[]; datetime times[];
   // Rebuild only on new information or initial history download.
   bool need_forecast=LastOrigin!=iTime(_Symbol,InpTimeframe,1) || !HavePrediction;
   bool need_train=Candidate<=0 && (LastAttempt==0 || iBarShift(_Symbol,InpTimeframe,LastAttempt)>=InpRetrainBars);
   if((need_forecast || need_train) && Collect(rows,prices,times,InpHistoryBars))
   {
      if(need_train && StartCandidate(rows,prices)) LastAttempt=current;
      PredictLatest(rows,prices,times);
   }
   Paint();
}

int OnInit()
{
   if(StringLen(InpInstance)<1 || StringLen(InpInstance)>32) return INIT_PARAMETERS_INCORRECT;
   for(int i=0;i<StringLen(InpInstance);i++)
   {
      ushort c=StringGetCharacter(InpInstance,i);
      if(!((c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' && c<='9') || c=='_' || c=='-'))
      { Print("InpInstance accepts letters, digits, underscore and hyphen only"); return INIT_PARAMETERS_INCORRECT; }
   }
   if(_Period!=InpTimeframe || InpSequence<2 || InpSequence>512 || InpHorizon<1 || InpHorizon>64 ||
      InpHistoryBars<256 || InpHistoryBars>100000 || InpEpochs<1 || InpBatch<1 || InpRetrainBars<1 ||
      InpValidationFraction<0.1 || InpValidationFraction>0.4 || InpLearningRate<=0 ||
      InpMaximumMSERatio<=0 || InpBandMultiplier<0 || InpMaxAbsForecastReturn<=0)
   { Print("Invalid parameters; chart timeframe must match InpTimeframe"); return INIT_PARAMETERS_INCORRECT; }
   if(!MQLInfoInteger(MQL_DLLS_ALLOWED)) { Print("Enable DLL imports for this indicator"); return INIT_FAILED; }
   if(DN_GetVersion()<20300) { Print("DLL version 2.3.0 or newer required"); return INIT_FAILED; }
   string drivers[]; int count=StringSplit(InpDrivers,',',drivers);
   if(count<1 || count>15) return INIT_PARAMETERS_INCORRECT;
   FeatureCount=count+1; ArrayResize(Symbols,FeatureCount); Symbols[0]=_Symbol;
   if((long)InpHistoryBars*InpSequence*FeatureCount>20000000)
   { Print("Reduce history, sequence or markets: indicator input budget is 20 million values"); return INIT_PARAMETERS_INCORRECT; }
   for(int i=0;i<count;i++) { StringTrimLeft(drivers[i]); StringTrimRight(drivers[i]); Symbols[i+1]=drivers[i]; }
   for(int i=0;i<FeatureCount;i++)
   {
      for(int j=0;j<i;j++) if(Symbols[i]==Symbols[j]) { Print("Duplicate symbol: ",Symbols[i]); return INIT_PARAMETERS_INCORRECT; }
      if(!SymbolSelect(Symbols[i],true)) { Print("Unavailable broker symbol: ",Symbols[i],"; set exact names including suffixes"); return INIT_FAILED; }
   }
   SetIndexBuffer(0,Forecast,INDICATOR_DATA); SetIndexBuffer(1,Upper,INDICATOR_DATA); SetIndexBuffer(2,Lower,INDICATOR_DATA);
   ArraySetAsSeries(Forecast,true); ArraySetAsSeries(Upper,true); ArraySetAsSeries(Lower,true);
   for(int i=0;i<3;i++) { PlotIndexSetInteger(i,PLOT_SHIFT,InpHorizon); PlotIndexSetDouble(i,PLOT_EMPTY_VALUE,EMPTY_VALUE); }
   IndicatorSetString(INDICATOR_SHORTNAME,"CrossAsset Forecast ["+InpInstance+"]");
   LabelName="DN_CrossAsset_"+InpInstance+"_"+IntegerToString(ChartID());
   StatusName=LabelName+"_status";
   if(ObjectFind(0,LabelName)>=0 || ObjectFind(0,StatusName)>=0)
   { Print("Choose a unique InpInstance on this chart"); return INIT_PARAMETERS_INCORRECT; }
   string safe_symbol=_Symbol;
   StringReplace(safe_symbol,"/","_"); StringReplace(safe_symbol,"\\","_"); StringReplace(safe_symbol,":","_");
   LogFile="CrossAsset_"+safe_symbol+"_"+EnumToString(InpTimeframe)+"_"+InpInstance+".csv";
   if(!ObjectCreate(0,LabelName,OBJ_LABEL,0,0,0)) return INIT_FAILED;
   LabelCreated=true;
   if(!ObjectCreate(0,StatusName,OBJ_LABEL,0,0,0)) return INIT_FAILED;
   StatusLabelCreated=true;
   ObjectSetInteger(0,StatusName,OBJPROP_CORNER,CORNER_LEFT_UPPER);
   ObjectSetInteger(0,StatusName,OBJPROP_XDISTANCE,12); ObjectSetInteger(0,StatusName,OBJPROP_YDISTANCE,44);
   ObjectSetInteger(0,StatusName,OBJPROP_COLOR,clrSlateGray); ObjectSetInteger(0,StatusName,OBJPROP_FONTSIZE,10);
   ObjectSetInteger(0,LabelName,OBJPROP_CORNER,CORNER_LEFT_UPPER);
   ObjectSetInteger(0,LabelName,OBJPROP_XDISTANCE,12); ObjectSetInteger(0,LabelName,OBJPROP_YDISTANCE,24);
   ObjectSetInteger(0,LabelName,OBJPROP_COLOR,clrDeepSkyBlue); ObjectSetInteger(0,LabelName,OBJPROP_FONTSIZE,10);
   if(!EventSetTimer(1)) { Print("Cannot start forecast timer"); return INIT_FAILED; }
   OriginalChartShift=(bool)ChartGetInteger(0,CHART_SHIFT);
   ChartSetInteger(0,CHART_SHIFT,true);
   ChartShiftChanged=true;
   return INIT_SUCCEEDED;
}

int OnCalculate(const int rates_total,const int prev_calculated,const datetime &time[],const double &open[],
   const double &high[],const double &low[],const double &close[],const long &tick_volume[],const long &volume[],const int &spread[])
{
   if(prev_calculated==0) { ArrayInitialize(Forecast,EMPTY_VALUE); ArrayInitialize(Upper,EMPTY_VALUE); ArrayInitialize(Lower,EMPTY_VALUE); }
   // Never move a stale forecast to a new origin while cross-market history is loading.
   if(LastOrigin!=iTime(_Symbol,InpTimeframe,1)) HavePrediction=false;
   Paint();
   return rates_total;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   if(Candidate>0) { DN_StopTraining(Candidate); DN_Free(Candidate); }
   if(Active>0) DN_Free(Active);
   if(LabelCreated) ObjectDelete(0,LabelName);
   if(StatusLabelCreated) ObjectDelete(0,StatusName);
   if(ChartShiftChanged) ChartSetInteger(0,CHART_SHIFT,OriginalChartShift);
}
