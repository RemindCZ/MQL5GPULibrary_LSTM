#pragma once
#include <stdint.h>
#ifdef _WIN32
#ifdef DN_FIT_BUILD
#define DN_FIT_API __declspec(dllexport)
#else
#define DN_FIT_API __declspec(dllimport)
#endif
#else
#define DN_FIT_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
// Chronological closed bars, row-major closes. Auxiliary zero denotes a missing observation.
// Returns the number of ONNX bytes, or a negative code with a UTF-8 diagnostic.
// The report contains three doubles per horizon: test MSE, last-price MSE, Q10-Q90 coverage.
DN_FIT_API int DNFit_Train(int steps, int markets, int horizons, int epochs,
    const char* contract, int contract_size, const int64_t* times, int rows,
    const double* closes, int close_count, uint8_t* model, int capacity,
    double* report, int report_count, char* error, int error_capacity);
#ifdef __cplusplus
}
#endif
