#pragma once
/* Stable C ABI. Dimensions and buffer ownership: docs/API.md. */
#ifdef _WIN32
#  define DN_CALL __stdcall
#  ifdef DN_BUILDING_LIBRARY
#    define DN_API __declspec(dllexport)
#  else
#    define DN_API __declspec(dllimport)
#  endif
#else
#  define DN_CALL
#  define DN_API
#endif
#ifdef __cplusplus
extern "C" {
#endif

DN_API int DN_CALL DN_GetVersion(void);
DN_API int DN_CALL DN_BeginGraph(int h, int features, int sequence);
DN_API int DN_CALL DN_AddGraphNode(int h, int type, int width, int activation, double dropout, const int* parents, int parent_count, int merge);
DN_API int DN_CALL DN_SetGraphOutput(int h, int node, int outputs);
DN_API int DN_CALL DN_GetInputDim(int h);
DN_API int DN_CALL DN_GetOutputDim(int h);
DN_API int DN_CALL DN_GetSequenceLength(int h);
DN_API int DN_CALL DN_Create(void);
DN_API void DN_CALL DN_Free(int h);
DN_API int DN_CALL DN_SetSequenceLength(int h, int seq_len);
DN_API int DN_CALL DN_SetMiniBatchSize(int h, int mbs);
DN_API int DN_CALL DN_AddLayerEx(int h, int in, int out, int act, int ln, double drop);
DN_API int DN_CALL DN_AddGRULayer(int h, int in, int out, double drop);
DN_API int DN_CALL DN_AddRNNLayer(int h, int in, int out, double drop);
DN_API int DN_CALL DN_SetGradClip(int h, double clip);
DN_API int DN_CALL DN_SetOutputDim(int h, int out_dim);
DN_API int DN_CALL DN_LoadBatch(int h, const double* X, const double* T, int batch, int in, int out, int l);
DN_API int DN_CALL DN_PredictBatch(int h, const double* X, int batch, int in, int l, double* Y);
DN_API int DN_CALL DN_SnapshotWeights(int h);
DN_API int DN_CALL DN_RestoreWeights(int h);
DN_API int DN_CALL DN_Train(int h, int epochs, double target_mse, double lr, double wd);
DN_API int DN_CALL DN_TrainAsync(int h, int epochs, double target_mse, double lr, double wd);
DN_API int DN_CALL DN_GetTrainingStatus(int h);
DN_API void DN_CALL DN_GetTrainingResult(int h, double* out_mse, int* out_epochs);
DN_API void DN_CALL DN_StopTraining(int h);
DN_API int DN_CALL DN_GetProgressEpoch(int h);
DN_API int DN_CALL DN_GetProgressTotalEpochs(int h);
DN_API int DN_CALL DN_GetProgressMiniBatch(int h);
DN_API int DN_CALL DN_GetProgressTotalMiniBatches(int h);
DN_API double DN_CALL DN_GetProgressLR(int h);
DN_API double DN_CALL DN_GetProgressMSE(int h);
DN_API double DN_CALL DN_GetProgressBestMSE(int h);
DN_API double DN_CALL DN_GetProgressGradNorm(int h);
DN_API int DN_CALL DN_GetProgressTotalSteps(int h);
DN_API double DN_CALL DN_GetProgressPercent(int h);
DN_API double DN_CALL DN_GetProgressElapsedSec(int h);
DN_API double DN_CALL DN_GetProgressETASec(int h);
DN_API int DN_CALL DN_GetProgressAll(int h, int* out_epoch, int* out_total_epochs, int* out_mb, int* out_total_mb, double* out_lr, double* out_mse, double* out_best_mse, double* out_grad_norm, double* out_pct, double* out_elapsed_sec, double* out_eta_sec);
DN_API int DN_CALL DN_GetLayerCount(int h);
DN_API double DN_CALL DN_GetLayerWeightNorm(int h, int l);
DN_API double DN_CALL DN_GetGradNorm(int h);
DN_API int DN_CALL DN_SaveState(int h);
DN_API int DN_CALL DN_GetState(int h, char* buf, int max_len);
DN_API int DN_CALL DN_LoadState(int h, const char* buf);
DN_API void DN_CALL DN_GetError(short* buf, int len);

#ifdef __cplusplus
}
#endif
