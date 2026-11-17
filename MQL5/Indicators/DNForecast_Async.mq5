#property copyright "Remind / Tomáš Bělák"
#property version "3.00"
#property indicator_chart_window
#property indicator_buffers 3
#property indicator_plots 3
#property indicator_label1 "Forecast"
#property indicator_type1 DRAW_LINE
#property indicator_color1 clrDeepSkyBlue
#property indicator_width1 2
#property indicator_label2 "Q10"
#property indicator_type2 DRAW_LINE
#property indicator_color2 clrSlateGray
#property indicator_label3 "Q90"
#property indicator_type3 DRAW_LINE
#property indicator_color3 clrSlateGray
#include <DNForecast.mqh>

input string InpDrivers="NZDUSD,USDJPY,USDCAD"; // Exact broker symbols, comma-separated
input int InpSteps=96;
input int InpHorizons=6;
input string InpModelFile=""; // ONNX in MQL5/Files. Empty selects the CPU reference
input bool InpCuda=false;
input bool InpJournal=true;
input bool InpDiagnostics=false;

double PointLine[],LowLine[],HighLine[],Window[],Result[];
string Symbols[],Generation="reference-3.0",Journal;
long SessionId=0,JobId=0;
datetime WindowEnd=0,Submitted=0,ChartBar=0;
int Markets=0;
bool Failed=false;
ulong Retry=0;
bool OldShift=false,ShiftSet=false;
string LastDiagnostic="";

void Diagnostic(const string message)
{
   if(InpDiagnostics && message!=LastDiagnostic) {Print(message);LastDiagnostic=message;}
}

int OnInit()
{
   if(InpSteps<16 || InpSteps>1024 || InpHorizons<1 || InpHorizons>64 || InpHorizons>InpSteps/2 ||
      (InpCuda && InpModelFile=="") || !MQLInfoInteger(MQL_DLLS_ALLOWED)) return INIT_PARAMETERS_INCORRECT;
   if(DN3_GetAbiVersion()!=30000) return INIT_FAILED;
   string drivers[];
   int n=InpDrivers=="" ? 0 : StringSplit(InpDrivers,',',drivers);
   if(n<0 || n>15) return INIT_PARAMETERS_INCORRECT;
   Markets=n+1; ArrayResize(Symbols,Markets); Symbols[0]=_Symbol;
   string names=_Symbol;
   for(int i=0;i<n;i++)
   {
      StringTrimLeft(drivers[i]); StringTrimRight(drivers[i]); Symbols[i+1]=drivers[i];
      for(int j=0;j<=i;j++) if(Symbols[j]==drivers[i]) return INIT_PARAMETERS_INCORRECT;
      if(!SymbolSelect(drivers[i],true)) {Print("Unavailable symbol ",drivers[i]); return INIT_FAILED;}
      names+=","+drivers[i];
   }
   uchar key[],model[],empty_key[],digest[];
   int keylen=StringToCharArray("dn3-v1|target-trading-bars|"+IntegerToString(PeriodSeconds())+"|"+names,key,0,-1,CP_UTF8)-1;
   int size=0,backend=0;
   if(InpModelFile!="")
   {
      int f=FileOpen(InpModelFile,FILE_READ|FILE_BIN);
      if(f==INVALID_HANDLE) {Print("Cannot open model in MQL5/Files: ",InpModelFile);return INIT_FAILED;}
      ulong bytes=FileSize(f);
      if(bytes<1 || bytes>67108864) {FileClose(f);return INIT_FAILED;}
      size=(int)bytes;
      if(ArrayResize(model,size)!=size || FileReadArray(f,model,0,size)!=(uint)size)
      {FileClose(f);return INIT_FAILED;}
      FileClose(f);
      if(CryptEncode(CRYPT_HASH_SHA256,model,empty_key,digest)!=32) return INIT_FAILED;
      Generation=""; for(int k=0;k<32;k++) Generation+=StringFormat("%02x",digest[k]);
      backend=(size>=6 && CharArrayToString(model,0,6,CP_UTF8)=="DN3V2\n") ? 3 : (InpCuda ? 2 : 1);
   }
   else ArrayResize(model,1);
   SessionId=DN3_OpenSession(InpSteps,Markets,InpHorizons,4,1,backend,key,keylen,model,size);
   if(SessionId<=0) {Print("Open failed ",SessionId);return INIT_FAILED;}
   SetIndexBuffer(0,PointLine); SetIndexBuffer(1,LowLine); SetIndexBuffer(2,HighLine);
   ArraySetAsSeries(PointLine,true);ArraySetAsSeries(LowLine,true);ArraySetAsSeries(HighLine,true);
   for(int i=0;i<3;i++) {PlotIndexSetInteger(i,PLOT_SHIFT,InpHorizons);PlotIndexSetDouble(i,PLOT_EMPTY_VALUE,EMPTY_VALUE);}
   IndicatorSetString(INDICATOR_SHORTNAME,InpModelFile=="" ? "DNForecast | last price + empirical bands" : "DNForecast | ONNX "+StringSubstr(Generation,0,8));
   ArrayResize(Result,InpHorizons*6);
   ArrayResize(Window,(InpSteps+1)*Markets);
   // Distinct files per attachment keep writers independent, including on chart restart.
   Journal=StringFormat("DNForecast_%I64d_%I64d_%I64d.csv",ChartID(),(long)TimeLocal(),SessionId);
   OldShift=(bool)ChartGetInteger(0,CHART_SHIFT);
   if(!EventSetMillisecondTimer(40)) return INIT_FAILED;
   ChartSetInteger(0,CHART_SHIFT,true);ShiftSet=true;
   return INIT_SUCCEEDED;
}

bool Collect(const datetime latest)
{
   bool append=WindowEnd>0 && iTime(_Symbol,_Period,2)==WindowEnd;
   int count=append ? 1 : InpSteps+1;
   MqlRates target[];
   if(CopyRates(_Symbol,_Period,1,count,target)!=count || target[count-1].time!=latest)
   {Diagnostic("Waiting for target history");return false;}
   double tail[];ArrayResize(tail,count*Markets);ArrayInitialize(tail,0.);
   for(int i=0;i<count;i++) {if(target[i].close<=0) return false;tail[i*Markets]=target[i].close;}
   for(int m=1;m<Markets;m++)
   {
      MqlRates bars[];
      int total=CopyRates(Symbols[m],_Period,target[0].time,latest,bars);
      datetime forming=iTime(Symbols[m],_Period,0);
      // Never freeze an initial history download as a real market closure.
      if(forming==0 || (total<=0 && !SeriesInfoInteger(Symbols[m],_Period,SERIES_SYNCHRONIZED)))
      {Diagnostic(StringFormat("Waiting for %s history (copied %d, current %I64d)",Symbols[m],total,(long)forming));return false;}
      int k=0;
      for(int i=0;i<count;i++)
      {
         while(k<total && bars[k].time<target[i].time) k++;
         if(k<total && bars[k].time==target[i].time && bars[k].time<forming)
            tail[i*Markets+m]=bars[k].close;
      }
   }
   if(append)
   {
      for(int i=0;i<InpSteps*Markets;i++) Window[i]=Window[i+Markets];
      ArrayCopy(Window,tail,InpSteps*Markets);
   }
   else ArrayCopy(Window,tail);
   WindowEnd=latest;
   return true;
}

void Poll()
{
   if(Failed || SessionId<=0 || ArraySize(PointLine)<InpHorizons+3) return;
   int state=DN3_StatusOf(SessionId);
   if(state==1) return;
   if(state!=0) {Print("Model load failed: ",DN3Error(SessionId,0));Failed=true;return;}
   datetime latest=iTime(_Symbol,_Period,1);
   if(latest<=0) return;
   if(JobId>0)
   {
      int needed=0; long origin=0;
      int rc=DN3_TryRead(SessionId,JobId,Result,ArraySize(Result),needed,origin);
      if(rc==1) return;
      if(rc!=0) Print("Forecast failed: ",rc," ",DN3Error(SessionId,JobId));
      DN3_Release(SessionId,JobId);JobId=0;
      if(rc==0 && origin==(long)latest && needed==InpHorizons*6)
      {
         Diagnostic("Forecast ready for "+TimeToString(latest));
         int base=InpHorizons+1;
         double anchor=Window[InpSteps*Markets];
         PointLine[base]=anchor; LowLine[base]=anchor; HighLine[base]=anchor;
         int file=INVALID_HANDLE;
         if(InpJournal) file=FileOpen(Journal,FILE_READ|FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_SHARE_READ,',');
         if(file!=INVALID_HANDLE)
         {
            if(FileSize(file)==0) FileWrite(file,"symbol","seconds","origin_open_server","issued_server","generation","h","anchor","point","q10","q50","q90","p_up","variance");
            FileSeek(file,0,SEEK_END);
         }
         datetime issued=TimeTradeServer();
         for(int h=1;h<=InpHorizons;h++)
         {
            int p=(h-1)*6;
            PointLine[base-h]=anchor*MathExp(Result[p]);
            LowLine[base-h]=anchor*MathExp(Result[p+1]);
            HighLine[base-h]=anchor*MathExp(Result[p+3]);
            if(file!=INVALID_HANDLE) FileWrite(file,_Symbol,PeriodSeconds(),origin,(long)issued,Generation,h,anchor,
               Result[p],Result[p+1],Result[p+2],Result[p+3],Result[p+4],Result[p+5]);
         }
         if(file!=INVALID_HANDLE) FileClose(file);
         ChartRedraw();
      }
   }
   if(Submitted==latest || GetTickCount64()<Retry) return;
   Retry=GetTickCount64()+250;
   if(WindowEnd!=latest && !Collect(latest)) return;
   JobId=DN3_Submit(SessionId,(long)latest,Window,ArraySize(Window));
   if(JobId>0) {Submitted=latest;Diagnostic("Submitted "+TimeToString(latest));}
   else {Print("Submit failed ",JobId);JobId=0;}
}
void OnTimer() {Poll();}

int OnCalculate(const int total,const int previous,const datetime &times[],const double &open[],
   const double &high[],const double &low[],const double &close[],const long &vol[],const long &realvol[],const int &spread[])
{
   ArraySetAsSeries(times,true);
   if(total==0) return 0;
   if(previous==0 || times[0]!=ChartBar)
   {
      int clear=previous==0 ? total : MathMin(total,InpHorizons+3+MathMax(1,total-previous));
      for(int i=0;i<clear;i++) {PointLine[i]=EMPTY_VALUE;LowLine[i]=EMPTY_VALUE;HighLine[i]=EMPTY_VALUE;}
      if(previous==0) {WindowEnd=0;Submitted=0;}
      ChartBar=times[0];
   }
   Poll(); // Dependent indicators can receive calculation events without timer events.
   return total;
}
void OnDeinit(const int reason)
{
   EventKillTimer();
   if(SessionId>0) DN3_CloseSession(SessionId);
   if(ShiftSet) ChartSetInteger(0,CHART_SHIFT,OldShift);
}
