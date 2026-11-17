#property copyright "Remind / Tomáš Bělák"
#property version "3.20"
#property description "Read-only history connection for DNForecast Studio. Attach once to a chart."
#define BASE "DNForecast\\Bridge\\"
int owner=INVALID_HANDLE;
ulong last_seen=0,started=0;
string current="";

bool Publish(const string name,const string body)
{
   string temp=BASE+name+".tmp";
   int f=FileOpen(temp,FILE_WRITE|FILE_TXT|FILE_ANSI,0,CP_UTF8);
   if(f==INVALID_HANDLE) return false;
   bool ok=FileWriteString(f,body)>0;
   FileFlush(f);FileClose(f);
   return ok && FileMove(temp,0,BASE+name,FILE_REWRITE);
}

void Heartbeat()
{
   string state=TerminalInfoInteger(TERMINAL_CONNECTED) ? "online" : "offline";
   string info="DNBRIDGE1\n"+state+"\n"+AccountInfoString(ACCOUNT_SERVER)+"\n";
   for(int i=0;i<SymbolsTotal(true) && i<512;i++) info+=SymbolName(i,true)+"\n";
   if(Publish("status.txt",info)) last_seen=GetTickCount64();
}

int OnInit()
{
   owner=FileOpen(BASE+"owner.lock",FILE_BIN|FILE_WRITE);
   if(owner==INVALID_HANDLE) {Print("DNForecast Bridge already active or Files directory unavailable.");return INIT_FAILED;}
   if(!EventSetTimer(1)) return INIT_FAILED;
   Heartbeat();Print("DNForecast Bridge ready.");return INIT_SUCCEEDED;
}
void OnDeinit(const int reason)
{
   EventKillTimer();
   if(owner!=INVALID_HANDLE) {FileDelete(BASE+"status.txt");FileClose(owner);}
}

ENUM_TIMEFRAMES Frame(const int seconds)
{
   ENUM_TIMEFRAMES frames[]={PERIOD_M1,PERIOD_M5,PERIOD_M15,PERIOD_M30,PERIOD_H1,PERIOD_H4,PERIOD_D1};
   for(int i=0;i<ArraySize(frames);i++) if(PeriodSeconds(frames[i])==seconds) return frames[i];
   return PERIOD_CURRENT;
}

bool ValidId(const string name)
{
   if(StringLen(name)!=36 || StringSubstr(name,32)!=".req") return false;
   for(int i=0;i<32;i++) if(StringFind("0123456789abcdef",StringSubstr(name,i,1))<0) return false;
   return true;
}

// Return zero while MT5 is synchronizing, -1 for a permanent error, and 1 after publication.
int Snapshot(const string id,string &problem)
{
   int f=FileOpen(BASE+id+".req",FILE_READ|FILE_TXT|FILE_ANSI|FILE_SHARE_READ,0,CP_UTF8);
   if(f==INVALID_HANDLE) return 0;
   string request=FileReadString(f);FileClose(f);
   string args[],symbols[];
   if(StringSplit(request,'|',args)!=3) {problem="Invalid history request";return -1;}
   ENUM_TIMEFRAMES frame=Frame((int)StringToInteger(args[0]));
   int count=(int)StringToInteger(args[1]), markets=StringSplit(args[2],',',symbols);
   if(frame==PERIOD_CURRENT || count<512 || count>20000 || markets<1 || markets>8) {problem="Unsupported dimensions";return -1;}
   if(!TerminalInfoInteger(TERMINAL_CONNECTED)) {problem="MT5 disconnected";return 0;}
   for(int m=0;m<markets;m++) if(!SymbolSelect(symbols[m],true)) {problem="Unknown broker symbol: "+symbols[m];return -1;}
   // Include the current bar ONLY for its opening timestamp, the availability time of its predecessor.
   MqlRates target[];
   int n=CopyRates(symbols[0],frame,0,count+1,target);
   if(n!=count+1 || !SeriesInfoInteger(symbols[0],frame,SERIES_SYNCHRONIZED))
   {problem="Target history incomplete: "+symbols[0]+" ("+IntegerToString(MathMax(0,n-1))+"/"+IntegerToString(count)+")";return 0;}
   double prices[];ArrayResize(prices,count*markets);ArrayInitialize(prices,0.);
   for(int i=0;i<count;i++) prices[i*markets]=target[i].close;
   for(int m=1;m<markets;m++)
   {
      MqlRates bars[];
      int copied=CopyRates(symbols[m],frame,target[0].time,target[count].time,bars);
      if(copied<2 || !SeriesInfoInteger(symbols[m],frame,SERIES_SYNCHRONIZED))
      {problem="History unavailable: "+symbols[m];return 0;}
      int j=0,matched=0;
      for(int i=0;i<count;i++)
      {
         while(j<copied && bars[j].time<target[i].time) j++;
         if(j+1<copied && bars[j].time==target[i].time && bars[j+1].time<=target[i+1].time)
         {prices[i*markets+m]=bars[j].close;matched++;}
      }
      if(!matched) {problem="No synchronous closed bars: "+symbols[m];return -1;}
   }
   string temp=BASE+id+".part";
   f=FileOpen(temp,FILE_WRITE|FILE_TXT|FILE_ANSI,0,CP_UTF8);
   if(f==INVALID_HANDLE) {problem="Cannot create history";return -1;}
   bool ok=FileWriteString(f,"close_time,available_at,"+args[2]+"\r\n")>0;
   for(int i=0;i<count && ok && !IsStopped();i++)
   {
      string stamp=IntegerToString((long)target[i+1].time),row=stamp+","+stamp;
      for(int m=0;m<markets;m++) row+=","+(prices[i*markets+m]>0 ? DoubleToString(prices[i*markets+m],-16) : "");
      ok=FileWriteString(f,row+"\r\n")>0;
   }
   FileFlush(f);FileClose(f);
   if(!ok || IsStopped() || !FileIsExist(BASE+id+".req") || !FileMove(temp,0,BASE+id+".csv",0))
   {FileDelete(temp);problem="History publication interrupted";return -1;}
   return 1;
}

void OnTimer()
{
   if(GetTickCount64()-last_seen>=3000) Heartbeat();
   if(current=="")
   {
      string name;
      long find=FileFindFirst(BASE+"*.req",name);
      if(find==INVALID_HANDLE) return;
      do {if(ValidId(name)) {current=StringSubstr(name,0,32);break;}} while(FileFindNext(find,name));
      FileFindClose(find);
      if(current=="") return;
      started=GetTickCount64();
   }
   if(!FileIsExist(BASE+current+".req")) {current="";return;}
   if(FileIsExist(BASE+current+".csv") || FileIsExist(BASE+current+".err"))
   {FileDelete(BASE+current+".req");current="";return;}
   string problem="MT5 history synchronization timed out";
   int result=Snapshot(current,problem);
   if(result==0 && GetTickCount64()-started<90000) return;
   if(result!=1 && !Publish(current+".err",problem)) return;
   FileDelete(BASE+current+".req");current="";
}
