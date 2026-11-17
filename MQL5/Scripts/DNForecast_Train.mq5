#property copyright "Remind / Tomáš Bělák"
#property version "3.01"
#property script_show_inputs
#property description "Fit a neural forecast on this chart's closed bars and attach the trained model."
#include <DNForecast.mqh>
#import "DNForecastTrain.dll"
int DNFit_Train(int steps,int markets,int horizons,int epochs,const uchar &contract[],int contract_size,
                const long &times[],int rows,const double &prices[],int price_count,uchar &model[],int capacity,
                double &report[],int report_count,uchar &error[],int error_capacity);
#import

input int InpBars=4096;         // Closed bars to request
input string InpDrivers="";     // Additional broker symbols, comma separated
input int InpSteps=96;          // Lookback in bars
input int InpHorizons=6;        // Forecast length in bars
input int InpEpochs=8;          // Training epochs
input bool InpAttach=true;      // Display the trained model on this chart
input bool InpStudioHistoryOnly=false; // Export closed bars for the browser studio instead

bool History(string &symbols[],long &times[],double &prices[])
{
   MqlRates target[];
   int copied=-1;
   for(int attempt=0;attempt<12 && !IsStopped();attempt++)
   {
      copied=CopyRates(_Symbol,_Period,1,InpBars,target);
      if(copied>=InpBars || (copied>=512 && SeriesInfoInteger(_Symbol,_Period,SERIES_SYNCHRONIZED))) break;
      Sleep(500);
   }
   if(copied<512) {Print("Training needs at least 512 closed target bars, received ",copied);return false;}
   int markets=ArraySize(symbols);
   ArrayResize(times,copied); ArrayResize(prices,copied*markets); ArrayInitialize(prices,0.);
   for(int i=0;i<copied;i++) {times[i]=(long)target[i].time; prices[i*markets]=target[i].close;}
   for(int m=1;m<markets;m++)
   {
      MqlRates bars[];
      int n=-1; datetime forming=0;
      for(int attempt=0;attempt<12 && !IsStopped();attempt++)
      {
         n=CopyRates(symbols[m],_Period,target[0].time,target[copied-1].time,bars);
         forming=iTime(symbols[m],_Period,0);
         if(n>0 && forming>0 && SeriesInfoInteger(symbols[m],_Period,SERIES_SYNCHRONIZED)) break;
         Sleep(500);
      }
      if(n<=0 || forming<=0 || !SeriesInfoInteger(symbols[m],_Period,SERIES_SYNCHRONIZED))
      {Print("History unavailable for ",symbols[m]);return false;}
      int j=0,matched=0;
      for(int i=0;i<copied;i++)
      {
         while(j<n && bars[j].time<target[i].time) j++;
         if(j<n && bars[j].time==target[i].time && bars[j].time<forming)
         {prices[i*markets+m]=bars[j].close;matched++;}
      }
      if(!matched) {Print("No overlapping closed bars for ",symbols[m]);return false;}
   }
   return !IsStopped();
}

bool Probe(const double &prices[],const uchar &key[],const int keylen,const uchar &model[],const int markets,const long origin)
{
   long session=DN3_OpenSession(InpSteps,markets,InpHorizons,2,1,1,key,keylen,model,ArraySize(model));
   if(session<=0) {Print("Verification session failed: ",session);return false;}
   ulong until=GetTickCount64()+30000; int state=1;
   while(state==1 && !IsStopped() && GetTickCount64()<until) {Sleep(20);state=DN3_StatusOf(session);}
   int n=(InpSteps+1)*markets; double window[],out[]; ArrayResize(out,InpHorizons*6);
   ArrayCopy(window,prices,0,ArraySize(prices)-n,n);
   long job=0,returned=0; int needed=0,rc=-1;
   if(state==0) job=DN3_Submit(session,origin,window,n);
   if(job>0)
   {
      do {rc=DN3_TryRead(session,job,out,ArraySize(out),needed,returned); if(rc==1) Sleep(20);}
      while(rc==1 && !IsStopped() && GetTickCount64()<until);
   }
   bool ok=(rc==0 && needed==ArraySize(out) && returned==origin);
   if(!ok) Print("Trained ONNX verification failed: ",DN3Error(session,job>0?job:0));
   if(job>0) DN3_Release(session,job); DN3_CloseSession(session);
   return ok;
}

string Digest(const uchar &data[])
{
   uchar empty[],hash[]; string text="";
   if(CryptEncode(CRYPT_HASH_SHA256,data,empty,hash)!=32) return "";
   for(int i=0;i<32;i++) text+=StringFormat("%02x",hash[i]);
   return text;
}

string SafeSymbol()
{
   string safe="";
   for(int i=0;i<StringLen(_Symbol);i++)
   {
      ushort c=StringGetCharacter(_Symbol,i);
      safe+=((c>=48 && c<=57)||(c>=65 && c<=90)||(c>=97 && c<=122)) ? StringSubstr(_Symbol,i,1) : "_";
   }
   return safe;
}

bool ExportHistory(const string names,const long &times[],const double &prices[],const int markets)
{
   int rows=ArraySize(times);
   string stem="DNForecast\\History\\"+SafeSymbol()+"-"+IntegerToString(PeriodSeconds())+"s-"+
               IntegerToString(times[rows-1])+"-"+IntegerToString(GetMicrosecondCount());
   string temp=stem+".tmp", path=stem+".csv";
   int f=FileOpen(temp,FILE_WRITE|FILE_TXT|FILE_ANSI,0,CP_UTF8);
   if(f==INVALID_HANDLE) return false;
   bool ok=FileWriteString(f,"close_time,available_at,"+names+"\r\n")>0;
   datetime forming=iTime(_Symbol,_Period,0);
   for(int i=0;i<rows && ok && !IsStopped();i++)
   {
      long available=i+1<rows ? times[i+1] : (long)forming;
      if(available<=times[i]) {ok=false;break;}
      string stamp=IntegerToString(available),line=stamp+","+stamp;
      for(int j=0;j<markets;j++)
         line+=","+(prices[i*markets+j]>0 ? DoubleToString(prices[i*markets+j],-16) : "");
      ok=FileWriteString(f,line+"\r\n")>0;
   }
   FileFlush(f);FileClose(f);
   if(!ok || IsStopped() || !FileMove(temp,0,path,0)) {FileDelete(temp);return false;}
   Print("DNForecast Studio history ready: ",path);
   return true;
}

bool SameFile(const string path,const int size,const string hash)
{
   int f=FileOpen(path,FILE_READ|FILE_BIN|FILE_SHARE_READ);
   if(f==INVALID_HANDLE) return false;
   uchar data[]; bool ok=false;
   if(FileSize(f)==(ulong)size && ArrayResize(data,size)==size)
      ok=(FileReadArray(f,data)==(uint)size && Digest(data)==hash);
   FileClose(f); return ok;
}

void OnStart()
{
   if(!MQLInfoInteger(MQL_DLLS_ALLOWED)) {Print("Allow DLL imports for DNForecast_Train.");return;}
   if(InpBars<512 || InpBars>100000 || InpSteps<16 || InpSteps>1024 || InpHorizons<1 ||
      InpHorizons>64 || InpHorizons>InpSteps/2 || InpEpochs<1 || InpEpochs>64)
   {Print("Invalid bar count, model dimensions or epochs.");return;}
   string symbols[],drivers[],names=_Symbol,driver_list="";
   int count=InpDrivers=="" ? 0 : StringSplit(InpDrivers,',',drivers);
   if(count<0 || count>15) {Print("Maximum 15 auxiliary symbols.");return;}
   ArrayResize(symbols,count+1); symbols[0]=_Symbol;
   for(int i=0;i<count;i++)
   {
      StringTrimLeft(drivers[i]); StringTrimRight(drivers[i]); symbols[i+1]=drivers[i];
      for(int j=0;j<=i;j++) if(symbols[j]==drivers[i]) {Print("Duplicate symbol ",drivers[i]);return;}
      if(drivers[i]=="" || !SymbolSelect(drivers[i],true)) {Print("Unknown driver ",drivers[i]);return;}
      names+=","+drivers[i]; driver_list+=(i==0?"":",")+drivers[i];
   }
   Print("Training ",names," on ",EnumToString(_Period),". Reading closed bars.");
   long times[]; double prices[],report[]; uchar key[],model[],error[];
   if(!History(symbols,times,prices)) return;
   int rows=ArraySize(times),markets=ArraySize(symbols);
   if(InpStudioHistoryOnly) {if(!ExportHistory(names,times,prices,markets)) Print("History export failed: ",GetLastError());return;}
   int keylen=StringToCharArray("dn3-v1|target-trading-bars|"+IntegerToString(PeriodSeconds())+"|"+names,key,0,-1,CP_UTF8)-1;
   ArrayResize(model,1048576); ArrayResize(error,512); ArrayResize(report,3*InpHorizons);
   int size=DNFit_Train(InpSteps,markets,InpHorizons,InpEpochs,key,keylen,times,rows,prices,ArraySize(prices),
                        model,ArraySize(model),report,ArraySize(report),error,ArraySize(error));
   if(size<=0) {Print("Training failed: ",CharArrayToString(error,0,-1,CP_UTF8));return;}
   ArrayResize(model,size);
   if(IsStopped() || !Probe(prices,key,keylen,model,markets,times[rows-1])) return;
   string hash=Digest(model); if(hash=="") {Print("Cannot hash model");return;}
   string base="DNForecast\\"+SafeSymbol()+"-"+EnumToString(_Period)+"-"+StringSubstr(hash,0,16);
   string path=base+".onnx", tmp=base+"-"+IntegerToString(ChartID())+"-"+IntegerToString(GetMicrosecondCount())+".tmp";
   if(!FileIsExist(path))
   {
      int f=FileOpen(tmp,FILE_WRITE|FILE_BIN);
      if(f==INVALID_HANDLE) {Print("Cannot create model: ",GetLastError());return;}
      uint written=FileWriteArray(f,model); FileFlush(f); FileClose(f);
      bool committed=written==(uint)size && FileMove(tmp,0,path,0);
      if(!committed) FileDelete(tmp);
   }
   if(!SameFile(path,size,hash)) {Print("Model could not be saved and verified: ",path);return;}
   PrintFormat("DNForecast TRAIN PASS symbol=%s bars=%d test=%d model=%s",_Symbol,rows,rows-InpHorizons-rows*85/100,path);
   string reportpath=base+"-"+IntegerToString(ChartID())+"-"+IntegerToString(GetMicrosecondCount())+".csv";
   int file=FileOpen(reportpath,FILE_WRITE|FILE_CSV|FILE_ANSI,',');
   if(file!=INVALID_HANDLE) FileWrite(file,"horizon","test_count","test_first_open","test_last_open","test_mse","last_price_mse","coverage80");
   for(int q=0;q<InpHorizons;q++)
   {
      if(file!=INVALID_HANDLE) FileWrite(file,q+1,rows-InpHorizons-rows*85/100,times[rows*85/100],times[rows-InpHorizons-1],
                                         report[3*q],report[3*q+1],report[3*q+2]);
      PrintFormat("H%d test MSE %.9g / last price %.9g, coverage %.3f",q+1,report[3*q],report[3*q+1],report[3*q+2]);
   }
   if(file!=INVALID_HANDLE) FileClose(file); else Print("Could not save report, see Experts log");
   if(!InpAttach) return;
   int indicator=iCustom(_Symbol,_Period,"DNForecast_Async",driver_list,InpSteps,InpHorizons,path,false,true,true);
   if(indicator==INVALID_HANDLE || !ChartIndicatorAdd(0,0,indicator))
      Print("Model saved. Attach DNForecast_Async manually. InpModelFile=",path,", error ",GetLastError());
   else {ChartRedraw();Print("DNForecast trained indicator attached: ",path);}
   if(indicator!=INVALID_HANDLE) IndicatorRelease(indicator);
}
