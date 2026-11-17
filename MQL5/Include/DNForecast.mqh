#ifndef DN_FORECAST_MQH
#define DN_FORECAST_MQH
#import "DNForecast.dll"
int DN3_GetAbiVersion();
int DN3_GetCapabilities();
long DN3_OpenSession(int steps,int markets,int horizons,int capacity,int latest,int backend,
                     const uchar &contract[],int contract_size,const uchar &model[],int model_size);
int DN3_StatusOf(long session);
int DN3_CloseSession(long session);
int DN3_Encode(int steps,int markets,const double &closes[],int count,float &features[],int length);
long DN3_Submit(long session,long origin,const double &closes[],int count);
int DN3_TryRead(long session,long job,double &output[],int length,int &required,long &origin);
int DN3_Cancel(long session,long job);
int DN3_Release(long session,long job);
int DN3_Error(long session,long job,uchar &text[],int length);
int DN3_Metrics(long session,long &metrics[],int count);
#import
string DN3Error(const long session,const long job)
{
   uchar buf[4096];
   int n=DN3_Error(session,job,buf,4096);
   if(n<=0 || n>4096) return "Runtime error "+IntegerToString(n);
   return CharArrayToString(buf,0,n-1,CP_UTF8);
}
#endif
