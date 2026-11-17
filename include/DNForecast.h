#pragma once
#include <stdint.h>
#ifdef _WIN32
#ifdef DN3_BUILD
#define DN3_API __declspec(dllexport)
#else
#define DN3_API __declspec(dllimport)
#endif
#else
#define DN3_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif

enum DN3_Status {
    DN3_OK=0, DN3_PENDING=1, DN3_INVALID=-1, DN3_FULL=-2, DN3_BUFFER=-3,
    DN3_CANCELLED=-4, DN3_SUPERSEDED=-5, DN3_FAILED=-6, DN3_UNKNOWN=-7
};
enum DN3_Backend { DN3_REFERENCE=0, DN3_ONNX_CPU=1, DN3_ONNX_CUDA=2, DN3_LEGACY=3 };
enum DN3_Field { DN3_POINT, DN3_Q10, DN3_Q50, DN3_Q90, DN3_P_UP, DN3_VARIANCE, DN3_FIELDS };

// Ownership: all input buffers can be reused immediately after the call returns.
// Model bytes are a self-contained ONNX file; contract is UTF-8 without a trailing NUL.
// Returns a positive session ID, or a negative status. Poll DN3_StatusOf before submitting.
DN3_API int DN3_GetAbiVersion(void);
DN3_API int DN3_GetCapabilities(void);  // bits 0..3: reference, ONNX CPU, advertised CUDA provider, Windows v2 adapter
DN3_API int64_t DN3_OpenSession(int steps, int markets, int horizons, int capacity, int latest,
    int backend, const char* contract, int contract_size, const uint8_t* model, int model_size);
DN3_API int DN3_StatusOf(int64_t session);
DN3_API int DN3_CloseSession(int64_t session);
// Chronological row-major [steps+1, markets] close prices. Zero = missing auxiliary close.
// Training and production use the same transform. Output [steps, markets*3] float32.
DN3_API int DN3_Encode(int steps, int markets, const double* closes, int count, float* features, int length);
DN3_API int64_t DN3_Submit(int64_t session, int64_t origin, const double* closes, int count);
// length in doubles. required and origin are written even on DN3_BUFFER / DN3_PENDING.
DN3_API int DN3_TryRead(int64_t session, int64_t job, double* output, int length, int* required, int64_t* origin);
DN3_API int DN3_Cancel(int64_t session, int64_t job);
DN3_API int DN3_Release(int64_t session, int64_t job);
// Session loading error (job=0) or job error. Returns required UTF-8 bytes including NUL.
DN3_API int DN3_Error(int64_t session, int64_t job, char* text, int length);
// submitted, ready, superseded, cancelled, rejected-full, failed, retained, last compute microseconds.
DN3_API int DN3_Metrics(int64_t session, int64_t* counters, int count);
#ifdef __cplusplus
}
#endif
