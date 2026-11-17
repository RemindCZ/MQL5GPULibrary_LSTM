/* Link DNForecast.lib (Windows x64) or -lDNForecast. */
#include "DNForecast.h"
#include <stdio.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
static void pause_ms(void) { Sleep(1); }
#else
#include <time.h>
static void pause_ms(void) { struct timespec t={0,1000000}; nanosleep(&t,0); }
#endif

int main(void) {
    const char *contract="dn3-v1|target-trading-bars|3600|AUDUSD";
    double prices[97], result[6*DN3_FIELDS];
    int length=0, state=DN3_PENDING;
    int64_t origin=0;
    for (int i=0;i<97;i++) prices[i]=1.0+i*.00001;
    int64_t s=DN3_OpenSession(96,1,6,8,0,DN3_REFERENCE,contract,(int)strlen(contract),0,0);
    if (s<=0) return 1;
    for (int i=0;i<5000 && (state=DN3_StatusOf(s))==DN3_PENDING;i++) pause_ms();
    int64_t job=state==DN3_OK ? DN3_Submit(s,100,prices,97) : 0;
    if (job>0) {
        for (int i=0;i<5000;i++) {
            state=DN3_TryRead(s,job,result,6*DN3_FIELDS,&length,&origin);
            if (state!=DN3_PENDING) break;
            pause_ms();
        }
        if (state==DN3_OK) printf("origin=%lld, h6 P(up)=%.4f\n",(long long)origin,result[5*DN3_FIELDS+DN3_P_UP]);
        DN3_Release(s,job);
    }
    DN3_CloseSession(s);
    return job>0 && origin==100 && state==DN3_OK ? 0 : 1;
}
