#ifndef MQL5_GPU_LIBRARY_LSTM_MQH
#define MQL5_GPU_LIBRARY_LSTM_MQH
// Copy this file to MQL5/Include. All examples use this shared ABI declaration.
// act: 0=LSTM, 1=GRU, 2=RNN. ln must be zero. layout must be zero.
// Progress output order: lr, mse, best_mse, grad_norm, percent, elapsed, eta.
#import "MQL5GPULibrary_LSTM.dll"
int DN_Create();
void DN_Free(int h);
int DN_SetSequenceLength(int h, int seq_len);
int DN_SetMiniBatchSize(int h, int mbs);
int DN_AddLayerEx(int h, int in, int out, int act, int ln, double drop);
int DN_AddGRULayer(int h, int in, int out, double drop);
int DN_AddRNNLayer(int h, int in, int out, double drop);
int DN_SetGradClip(int h, double clip);
int DN_SetOutputDim(int h, int out_dim);
int DN_LoadBatch(int h, const double &X[], const double &T[], int batch, int in, int out, int l);
int DN_PredictBatch(int h, const double &X[], int batch, int in, int l, double &Y[]);
int DN_SnapshotWeights(int h);
int DN_RestoreWeights(int h);
int DN_Train(int h, int epochs, double target_mse, double lr, double wd);
int DN_TrainAsync(int h, int epochs, double target_mse, double lr, double wd);
int DN_GetTrainingStatus(int h);
void DN_GetTrainingResult(int h, double &out_mse, int &out_epochs);
void DN_StopTraining(int h);
int DN_GetProgressEpoch(int h);
int DN_GetProgressTotalEpochs(int h);
int DN_GetProgressMiniBatch(int h);
int DN_GetProgressTotalMiniBatches(int h);
double DN_GetProgressLR(int h);
double DN_GetProgressMSE(int h);
double DN_GetProgressBestMSE(int h);
double DN_GetProgressGradNorm(int h);
int DN_GetProgressTotalSteps(int h);
double DN_GetProgressPercent(int h);
double DN_GetProgressElapsedSec(int h);
double DN_GetProgressETASec(int h);
int DN_GetProgressAll(int h, int &out_epoch, int &out_total_epochs, int &out_mb, int &out_total_mb, double &out_lr, double &out_mse, double &out_best_mse, double &out_grad_norm, double &out_pct, double &out_elapsed_sec, double &out_eta_sec);
int DN_GetLayerCount(int h);
double DN_GetLayerWeightNorm(int h, int l);
double DN_GetGradNorm(int h);
int DN_SaveState(int h);
int DN_GetState(int h, char &buf[], int max_len);
int DN_LoadState(int h, const char &buf[]);
void DN_GetError(short &buf[], int len);
#import
#endif
