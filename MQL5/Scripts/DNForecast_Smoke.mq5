#property copyright "Remind / Tomáš Bělák"
#property version "3.00"
#include <DNForecast.mqh>

bool CheckPackagedModel()
{
   int f=FileOpen("DNForecast\\AUDUSD-H1-TCN.onnx",FILE_BIN|FILE_READ);
   if(f==INVALID_HANDLE) return false;
   uchar model[],key[];
   int size=(int)FileSize(f);
   if(size<=0 || size>67108864 || FileReadArray(f,model,0,size)!=(uint)size) {FileClose(f);return false;}
   FileClose(f);
   int n=StringToCharArray("dn3-v1|target-trading-bars|3600|AUDUSD,NZDUSD,USDJPY,USDCAD",key,0,-1,CP_UTF8)-1;
   long s=DN3_OpenSession(96,4,6,2,0,1,key,n,model,size);
   if(s<=0) return false;
   int status=1;
   for(int i=0;i<2500 && status==1;i++) {status=DN3_StatusOf(s);if(status==1) Sleep(2);}
   double x[388],out[36];ArrayInitialize(x,1.0);
   long job=DN3_Submit(s,12345,x,388),origin=0;int needed=0;
   status=1;
   for(int i=0;i<1000 && status==1;i++) {status=DN3_TryRead(s,job,out,36,needed,origin);if(status==1) Sleep(2);}
   bool ok=status==0 && origin==12345 && needed==36 && MathAbs(out[30]+0.000022690554)<1e-8
           && MathAbs(out[34]-0.503736496)<1e-5;
   if(!ok) Print("ONNX diagnostic: ",DN3Error(s,0));
   DN3_CloseSession(s);
   return ok;
}

void OnStart()
{
   uchar key[],model[1];
   int n=StringToCharArray("dn3-v1|target-trading-bars|3600|AUDUSD",key,0,-1,CP_UTF8)-1;
   long session=DN3_OpenSession(16,1,2,2,0,0,key,n,model,0);
   if(session<=0) {Print("DN3 smoke FAIL open ",session);return;}
   int state=1;
   for(int i=0;i<1000 && state==1;i++) {state=DN3_StatusOf(session); if(state==1) Sleep(2);}
   double prices[17],output[12];
   for(int i=0;i<17;i++) prices[i]=1;
   long job=DN3_Submit(session,12345,prices,17),origin=0;
   int size=0;state=1;
   for(int i=0;i<1000 && state==1;i++) {state=DN3_TryRead(session,job,output,12,size,origin);if(state==1) Sleep(2);}
   bool ok=state==0 && size==12 && origin==12345 && output[0]==0 && output[1]==0 && output[4]==0.5 && output[5]>0;
   DN3_Release(session,job);DN3_CloseSession(session);
   if(ok) ok=CheckPackagedModel();
   Print(ok ? "DN3 smoke PASS" : "DN3 smoke FAIL");
}
