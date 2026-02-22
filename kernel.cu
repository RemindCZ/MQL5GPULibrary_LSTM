// kernel.cu — LSTM DLL v1.2.0 (Async Support)
// Pure LSTM implementation with CUDA acceleration
// Features: Async training, Dropout, Multi-layer, Persistent buffers
// Build as: Dynamic Library (.dll)
// ============================================================================

#define NOMINMAX
#include <windows.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <device_launch_parameters.h>

#include <vector>
#include <memory>
#include <cmath>
#include <map>
#include <mutex>
#include <string>
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <atomic>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <utility>
#include <random>
#include <numeric>
#include <thread> // Nové pro asynchronní běh

#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "curand.lib")

typedef int MQL_BOOL;
#define MQL_TRUE  1
#define MQL_FALSE 0

#define DLL_EXPORT extern "C" __declspec(dllexport)
#define DLL_CALL   __stdcall

// Stavy tréninku pro asynchronní režim
enum TrainingState : int {
    TS_IDLE = 0,
    TS_TRAINING = 1,
    TS_COMPLETED = 2,
    TS_ERROR = -1
};

// ============================================================================
// Error handling
// ============================================================================
static std::atomic<int>       g_last_cuda{ 0 };
static std::atomic<int>       g_last_cublas{ 0 };
static std::atomic<int>       g_last_curand{ 0 };
static std::mutex             g_err_mtx;
static std::wstring           g_last_err_w;
static std::atomic<long long> g_mem_usage{ 0 };

static void SetError(const wchar_t* fmt, ...) {
    std::lock_guard<std::mutex> lk(g_err_mtx);
    wchar_t buf[1024];
    va_list ap; va_start(ap, fmt);
    _vsnwprintf_s(buf, _countof(buf), _TRUNCATE, fmt, ap);
    va_end(ap);
    g_last_err_w = buf;
}

#define CUDA_CHECK_RET(x) do{                                         \
    cudaError_t e=(x);                                                \
    if(e!=cudaSuccess){                                               \
        g_last_cuda.store((int)e);                                    \
        SetError(L"[CUDA] %S @ %S:%d", cudaGetErrorString(e),        \
                 __FILE__, __LINE__);                                 \
        return MQL_FALSE;                                             \
    }                                                                 \
}while(0)

#define CUDA_CHECK_KERNEL_RET() do{                                   \
    cudaError_t e = cudaGetLastError();                               \
    if(e!=cudaSuccess){                                               \
        g_last_cuda.store((int)e);                                    \
        SetError(L"[CUDA LAUNCH] %S", cudaGetErrorString(e));        \
        return MQL_FALSE;                                             \
    }                                                                 \
}while(0)

#define CUDA_CHECK_VOID(x) do{                                        \
    cudaError_t e=(x);                                                \
    if(e!=cudaSuccess){                                               \
        g_last_cuda.store((int)e);                                    \
        SetError(L"[CUDA VOID] %S", cudaGetErrorString(e));          \
    }                                                                 \
}while(0)

#define CUBLAS_CHECK_RET(x) do{                                       \
    cublasStatus_t s=(x);                                             \
    if(s!=CUBLAS_STATUS_SUCCESS){                                     \
        g_last_cublas.store((int)s);                                  \
        SetError(L"[CUBLAS] Status %d @ %S:%d", (int)s,              \
                 __FILE__, __LINE__);                                 \
        return MQL_FALSE;                                             \
    }                                                                 \
}while(0)

#define CURAND_CHECK_RET(x) do{                                       \
    curandStatus_t s=(x);                                             \
    if(s!=CURAND_STATUS_SUCCESS){                                     \
        g_last_curand.store((int)s);                                  \
        SetError(L"[CURAND] Status %d @ %S:%d", (int)s,              \
                 __FILE__, __LINE__);                                 \
        return MQL_FALSE;                                             \
    }                                                                 \
}while(0)

// ============================================================================
// GPUMemory RAII
// ============================================================================
template<typename T>
struct GPUMemory {
    T* ptr = nullptr;
    size_t count = 0;
    size_t capacity = 0;

    GPUMemory() = default;
    ~GPUMemory() { free(); }
    GPUMemory(const GPUMemory&) = delete;
    GPUMemory& operator=(const GPUMemory&) = delete;

    GPUMemory(GPUMemory&& o) noexcept
        : ptr(o.ptr), count(o.count), capacity(o.capacity)
    {
        o.ptr = nullptr; o.count = 0; o.capacity = 0;
    }
    GPUMemory& operator=(GPUMemory&& o) noexcept {
        if (this != &o) {
            free();
            ptr = o.ptr; count = o.count; capacity = o.capacity;
            o.ptr = nullptr; o.count = 0; o.capacity = 0;
        }
        return *this;
    }

    MQL_BOOL alloc(size_t n) {
        if (n == 0) { free(); return MQL_TRUE; }
        if (n <= capacity) { count = n; return MQL_TRUE; }
        free();
        void* p = nullptr;
        cudaError_t e = cudaMalloc(&p, n * sizeof(T));
        if (e != cudaSuccess) {
            g_last_cuda.store((int)e);
            SetError(L"OOM %llu bytes", (unsigned long long)(n * sizeof(T)));
            return MQL_FALSE;
        }
        ptr = (T*)p;
        capacity = n;
        count = n;
        g_mem_usage.fetch_add((long long)(n * sizeof(T)));
        return MQL_TRUE;
    }

    MQL_BOOL zero() {
        if (!ptr || count == 0) return MQL_TRUE;
        CUDA_CHECK_RET(cudaMemset(ptr, 0, count * sizeof(T)));
        return MQL_TRUE;
    }

    MQL_BOOL zeroAsync(cudaStream_t s) {
        if (!ptr || count == 0) return MQL_TRUE;
        CUDA_CHECK_RET(cudaMemsetAsync(ptr, 0, count * sizeof(T), s));
        return MQL_TRUE;
    }

    void free() {
        if (ptr) {
            CUDA_CHECK_VOID(cudaFree(ptr));
            g_mem_usage.fetch_sub((long long)(capacity * sizeof(T)));
            ptr = nullptr;
        }
        count = 0; capacity = 0;
    }

    size_t bytes() const { return count * sizeof(T); }
};

// ============================================================================
// Utility: round up to even (curand requirement)
// ============================================================================
static inline size_t RoundUpEven(size_t n) {
    return (n + 1) & ~(size_t)1;
}

// ============================================================================
// Warp reduction helpers
// ============================================================================
__device__ __forceinline__ float warpReduceSumFloat(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}
__device__ __forceinline__ int warpReduceSumInt(int v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

// ============================================================================
// Vectorized conversion kernels
// ============================================================================
__global__ void kCopyD2F_vec4(int n, const double* __restrict__ in,
    float* __restrict__ out)
{
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (tid >= n) return;
    if (tid + 3 < n) {
        float4 v;
        v.x = (float)in[tid];
        v.y = (float)in[tid + 1];
        v.z = (float)in[tid + 2];
        v.w = (float)in[tid + 3];
        *reinterpret_cast<float4*>(out + tid) = v;
    }
    else {
        out[tid] = (float)in[tid];
        if (tid + 1 < n) out[tid + 1] = (float)in[tid + 1];
        if (tid + 2 < n) out[tid + 2] = (float)in[tid + 2];
    }
}

__global__ void kCopyF2D_vec4(int n, const float* __restrict__ in,
    double* __restrict__ out)
{
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (tid >= n) return;
    if (tid + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(in + tid);
        out[tid] = (double)v.x;
        out[tid + 1] = (double)v.y;
        out[tid + 2] = (double)v.z;
        out[tid + 3] = (double)v.w;
    }
    else {
        out[tid] = (double)in[tid];
        if (tid + 1 < n) out[tid + 1] = (double)in[tid + 1];
        if (tid + 2 < n) out[tid + 2] = (double)in[tid + 2];
    }
}

// ============================================================================
// LSTM-specific CUDA kernels
// ============================================================================
__device__ __forceinline__ float d_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Fused LSTM gate forward
__global__ void kLSTMGatesForward(
    int hidden_size, int batch,
    const float* __restrict__ gates_raw,
    const float* __restrict__ c_prev,
    float* __restrict__ c_new,
    float* __restrict__ h_new,
    float* __restrict__ f_cache,
    float* __restrict__ i_cache,
    float* __restrict__ g_cache,
    float* __restrict__ o_cache)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = hidden_size * batch;
    if (idx >= total) return;

    int h = idx % hidden_size;
    int b = idx / hidden_size;
    int stride = 4 * hidden_size;

    float f = d_sigmoid(gates_raw[h + 0 * hidden_size + b * stride]);
    float i = d_sigmoid(gates_raw[h + 1 * hidden_size + b * stride]);
    float g = tanhf(gates_raw[h + 2 * hidden_size + b * stride]);
    float o = d_sigmoid(gates_raw[h + 3 * hidden_size + b * stride]);

    float cp = c_prev[idx];
    float cn = f * cp + i * g;
    float hn = o * tanhf(cn);

    c_new[idx] = cn;
    h_new[idx] = hn;
    f_cache[idx] = f;
    i_cache[idx] = i;
    g_cache[idx] = g;
    o_cache[idx] = o;
}

// LSTM gate backward
__global__ void kLSTMGatesBackward(
    int hidden_size, int batch,
    const float* __restrict__ dh,
    const float* __restrict__ dc_next,
    const float* __restrict__ c_prev,
    const float* __restrict__ c_cur,
    const float* __restrict__ f_cache,
    const float* __restrict__ i_cache,
    const float* __restrict__ g_cache,
    const float* __restrict__ o_cache,
    float* __restrict__ dc_prev_out,
    float* __restrict__ dgates_raw)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = hidden_size * batch;
    if (idx >= total) return;

    int h = idx % hidden_size;
    int b = idx / hidden_size;

    float o = o_cache[idx];
    float tanh_c = tanhf(c_cur[idx]);
    float do_val = dh[idx] * tanh_c;
    float dc = dh[idx] * o * (1.0f - tanh_c * tanh_c) + dc_next[idx];

    float f = f_cache[idx];
    float i = i_cache[idx];
    float g = g_cache[idx];
    float cp = c_prev[idx];

    dc_prev_out[idx] = dc * f;

    int stride = 4 * hidden_size;
    dgates_raw[h + 0 * hidden_size + b * stride] = dc * cp * f * (1.0f - f);
    dgates_raw[h + 1 * hidden_size + b * stride] = dc * g * i * (1.0f - i);
    dgates_raw[h + 2 * hidden_size + b * stride] = dc * i * (1.0f - g * g);
    dgates_raw[h + 3 * hidden_size + b * stride] = do_val * o * (1.0f - o);
}

// Add bias in-place (col-major A: [rows x cols])
__global__ void kAddBiasInplace(int rows, int cols,
    float* __restrict__ A, const float* __restrict__ bias)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols)
        A[row + col * rows] += bias[row];
}

// Concatenate [h_prev; x] per sample → col-major [concat_dim x batch]
__global__ void kConcatHX(int hidden_size, int input_size, int batch,
    const float* __restrict__ h_prev,
    const float* __restrict__ x,
    float* __restrict__ hx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int concat_dim = hidden_size + input_size;
    int total = concat_dim * batch;
    if (idx >= total) return;

    int d = idx % concat_dim;
    int b = idx / concat_dim;

    if (d < hidden_size)
        hx[idx] = h_prev[d + b * hidden_size];
    else
        hx[idx] = x[(d - hidden_size) + b * input_size];
}

// Split dhx gradient back to dh_prev and dx
__global__ void kSplitDHX(int hidden_size, int input_size, int batch,
    const float* __restrict__ dhx,
    float* __restrict__ dh_prev,
    float* __restrict__ dx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int concat_dim = hidden_size + input_size;
    int total = concat_dim * batch;
    if (idx >= total) return;

    int d = idx % concat_dim;
    int b = idx / concat_dim;

    if (d < hidden_size) {
        if (dh_prev) dh_prev[d + b * hidden_size] = dhx[idx];
    }
    else {
        if (dx) dx[(d - hidden_size) + b * input_size] = dhx[idx];
    }
}

// Dense output layer bias (linear activation)
__global__ void kBiasLinear(int rows, int cols,
    float* __restrict__ A, const float* __restrict__ bias)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols)
        A[row + col * rows] += bias[row];
}

// MSE gradient: d[i] = 2*(y[i]-t[i])/n
__global__ void kMSEGrad(int n, const float* __restrict__ y,
    const float* __restrict__ t, float* __restrict__ d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = 2.0f * (y[i] - t[i]) / (float)n;
}

// MSE reduction with warp shuffle
__global__ void kMSEReduceWarp(int n, const float* __restrict__ y,
    const float* __restrict__ t, float* __restrict__ out_sum)
{
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_val = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride_val) {
        float d = y[i] - t[i];
        sum += d * d;
    }
    sum = warpReduceSumFloat(sum);
    __shared__ float warpSums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) warpSums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        int nWarps = (blockDim.x + 31) / 32;
        float bs = (lane < nWarps) ? warpSums[lane] : 0.0f;
        bs = warpReduceSumFloat(bs);
        if (lane == 0) atomicAdd(out_sum, bs);
    }
}

// L2 norm reduction
__global__ void kL2NormReduceWarp(int n, const float* __restrict__ buf,
    float* __restrict__ out_sum)
{
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_val = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride_val) {
        float v = buf[i];
        sum += v * v;
    }
    sum = warpReduceSumFloat(sum);
    __shared__ float warpSums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) warpSums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        int nWarps = (blockDim.x + 31) / 32;
        float bs = (lane < nWarps) ? warpSums[lane] : 0.0f;
        bs = warpReduceSumFloat(bs);
        if (lane == 0) atomicAdd(out_sum, bs);
    }
}

// AdamW optimizer
__global__ void kAdamW(int n, float* __restrict__ p,
    float* __restrict__ m, float* __restrict__ v,
    const float* __restrict__ g,
    float lr, float b1, float b2, float eps,
    float wd, float c1, float c2, float clip_val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float gi = fminf(fmaxf(g[i], -clip_val), clip_val);
        float mi = b1 * m[i] + (1.0f - b1) * gi;
        float vi = b2 * v[i] + (1.0f - b2) * (gi * gi);
        m[i] = mi; v[i] = vi;
        float denom = sqrtf(vi * c2) + eps;
        float upd = (mi * c1) / denom;
        p[i] -= lr * (upd + wd * p[i]);
    }
}

// Dropout forward (inverted dropout)
__global__ void kDropoutForward(int n, float* __restrict__ A,
    const float* __restrict__ rand_vals, float drop_rate,
    unsigned char* __restrict__ mask)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        bool keep = (rand_vals[i] >= drop_rate);
        mask[i] = keep ? 1 : 0;
        A[i] = keep ? A[i] / (1.0f - drop_rate) : 0.0f;
    }
}

// Dropout backward
__global__ void kDropoutBackward(int n, float* __restrict__ dA,
    const unsigned char* __restrict__ mask, float drop_rate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dA[i] = mask[i] ? dA[i] / (1.0f - drop_rate) : 0.0f;
}

// Transpose: row-major [batch x seq_len x feat] → timestep-major
// src[b*S*F + t*F + f] → dst[f + (t*B + b)*F]
__global__ void kTransposeToTimestep(int batch, int seq_len, int feat,
    const float* __restrict__ src, float* __restrict__ dst)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq_len * feat;
    if (idx >= total) return;

    int f = idx % feat;
    int rem = idx / feat;
    int t = rem % seq_len;
    int b = rem / seq_len;

    dst[f + (t * batch + b) * feat] = src[idx];
}

// Gather + transpose sequences for mini-batch
__global__ void kGatherTransposeSeq(int mb_size, int seq_len, int feat,
    const int* __restrict__ indices,
    const float* __restrict__ src, float* __restrict__ dst)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = mb_size * seq_len * feat;
    if (idx >= total) return;

    int f = idx % feat;
    int rem = idx / feat;
    int t = rem % seq_len;
    int local_b = rem / seq_len;

    int global_b = indices[local_b];
    int src_idx = global_b * seq_len * feat + t * feat + f;
    int dst_idx = f + (t * mb_size + local_b) * feat;
    dst[dst_idx] = src[src_idx];
}

// Gather rows for targets
__global__ void kGatherRows(int dim, int batch,
    const float* __restrict__ src, const int* __restrict__ indices,
    float* __restrict__ dst)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = dim * batch;
    if (idx < n) {
        int b = idx / dim;
        int d = idx % dim;
        int src_row = indices[b];
        dst[d + b * dim] = src[d + src_row * dim];
    }
}

// MinMax reduction
__global__ void kMinMaxReduce256(int n, const float* __restrict__ buf,
    float* __restrict__ out_min, float* __restrict__ out_max)
{
    __shared__ float sh_min[256], sh_max[256];
    float lo = 1e30f, hi = -1e30f;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += 256 * gridDim.x) {
        float v = buf[i];
        lo = fminf(lo, v); hi = fmaxf(hi, v);
    }
    sh_min[threadIdx.x] = lo; sh_max[threadIdx.x] = hi;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < (unsigned)s) {
            sh_min[threadIdx.x] = fminf(sh_min[threadIdx.x],
                sh_min[threadIdx.x + s]);
            sh_max[threadIdx.x] = fmaxf(sh_max[threadIdx.x],
                sh_max[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_min[blockIdx.x] = sh_min[0];
        out_max[blockIdx.x] = sh_max[0];
    }
}

// Add two buffers element-wise: dst[i] += src[i]
__global__ void kAddInplace(int n, float* __restrict__ dst,
    const float* __restrict__ src)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

// ============================================================================
// LSTMLayer
// ============================================================================
struct LSTMLayer {
    int input_size = 0;
    int hidden_size = 0;
    bool is_valid = false;

    GPUMemory<float> W, b;           // W: [(I+H) x 4H], b: [4H]
    GPUMemory<float> mW, vW, mb, vb; // AdamW states
    GPUMemory<float> dW, db;         // accumulated gradients

    GPUMemory<float> W_best, b_best;
    bool has_snapshot = false;

    // Per-timestep caches for BPTT
    std::vector<GPUMemory<float>> hx_cache;
    std::vector<GPUMemory<float>> f_cache, i_cache, g_cache, o_cache;
    std::vector<GPUMemory<float>> c_cache, h_cache;
    GPUMemory<float> h_init, c_init;

    // Dropout on output projection (not on recurrent connection)
    bool  use_dropout = false;
    float dropout_rate = 0.0f;
    std::vector<GPUMemory<unsigned char>> dropout_mask;
    std::vector<bool> dropout_mask_valid;
    std::vector<GPUMemory<float>> h_drop_cache; // dropout-applied copy

    // Input gradient buffer (filled by Backward, used by previous layer)
    GPUMemory<float> dx_buf;

    LSTMLayer() = default;

    MQL_BOOL Init(int in_sz, int hid_sz, float drop_rate = 0.0f) {
        input_size = in_sz;
        hidden_size = hid_sz;
        is_valid = false;
        has_snapshot = false;
        use_dropout = (drop_rate > 0.0f);
        dropout_rate = drop_rate;

        if (input_size <= 0 || hidden_size <= 0) return MQL_FALSE;

        int concat_dim = input_size + hidden_size;
        int gate_dim = 4 * hidden_size;
        size_t w_size = (size_t)concat_dim * gate_dim;

        if (!W.alloc(w_size))              return MQL_FALSE;
        if (!b.alloc(gate_dim))            return MQL_FALSE;
        if (!mW.alloc(w_size) || !mW.zero()) return MQL_FALSE;
        if (!vW.alloc(w_size) || !vW.zero()) return MQL_FALSE;
        if (!mb.alloc(gate_dim) || !mb.zero()) return MQL_FALSE;
        if (!vb.alloc(gate_dim) || !vb.zero()) return MQL_FALSE;

        // Xavier init for weights
        std::vector<float> hW(w_size);
        std::mt19937 gen(std::random_device{}());
        float stddev = sqrtf(2.0f / (float)(concat_dim + gate_dim));
        std::normal_distribution<float> dist(0.0f, stddev);
        for (float& w : hW) w = dist(gen);
        CUDA_CHECK_RET(cudaMemcpy(W.ptr, hW.data(), w_size * sizeof(float),
            cudaMemcpyHostToDevice));

        // Forget gate bias = 1.0
        std::vector<float> hb(gate_dim, 0.0f);
        for (int j = 0; j < hidden_size; j++)
            hb[j] = 1.0f;
        CUDA_CHECK_RET(cudaMemcpy(b.ptr, hb.data(), gate_dim * sizeof(float),
            cudaMemcpyHostToDevice));

        is_valid = true;
        return MQL_TRUE;
    }

    MQL_BOOL InitFromData(int in_sz, int hid_sz, float drop_rate = 0.0f) {
        input_size = in_sz;
        hidden_size = hid_sz;
        is_valid = false;
        has_snapshot = false;
        use_dropout = (drop_rate > 0.0f);
        dropout_rate = drop_rate;

        if (input_size <= 0 || hidden_size <= 0) return MQL_FALSE;

        int concat_dim = input_size + hidden_size;
        int gate_dim = 4 * hidden_size;
        size_t w_size = (size_t)concat_dim * gate_dim;

        if (!W.alloc(w_size))                    return MQL_FALSE;
        if (!b.alloc(gate_dim))                  return MQL_FALSE;
        if (!mW.alloc(w_size) || !mW.zero())     return MQL_FALSE;
        if (!vW.alloc(w_size) || !vW.zero())     return MQL_FALSE;
        if (!mb.alloc(gate_dim) || !mb.zero())   return MQL_FALSE;
        if (!vb.alloc(gate_dim) || !vb.zero())   return MQL_FALSE;

        is_valid = true;
        return MQL_TRUE;
    }

    const float* GetOutputH(int t, bool training) const {
        if (training && use_dropout && dropout_rate > 0.0f &&
            t < (int)dropout_mask_valid.size() && dropout_mask_valid[t] &&
            t < (int)h_drop_cache.size() && h_drop_cache[t].ptr)
        {
            return h_drop_cache[t].ptr;
        }
        return h_cache[t].ptr;
    }

    MQL_BOOL Forward(cublasHandle_t blas_h, cudaStream_t stream_h,
        const float* X, int seq_len, int batch,
        bool training, curandGenerator_t curand_gen)
    {
        if (!is_valid || !blas_h || !stream_h || !X) return MQL_FALSE;

        int H = hidden_size;
        int I = input_size;
        int concat_dim = H + I;
        int gate_dim = 4 * H;
        size_t hb_size = (size_t)H * batch;

        if (!h_init.alloc(hb_size) || !h_init.zeroAsync(stream_h)) return MQL_FALSE;
        if (!c_init.alloc(hb_size) || !c_init.zeroAsync(stream_h)) return MQL_FALSE;

        hx_cache.resize(seq_len);
        f_cache.resize(seq_len);
        i_cache.resize(seq_len);
        g_cache.resize(seq_len);
        o_cache.resize(seq_len);
        c_cache.resize(seq_len);
        h_cache.resize(seq_len);

        if (use_dropout && training && dropout_rate > 0.0f) {
            dropout_mask.resize(seq_len);
            dropout_mask_valid.assign(seq_len, false);
            h_drop_cache.resize(seq_len);
        }
        else {
            dropout_mask_valid.assign(seq_len, false);
        }

        GPUMemory<float> gates_raw;
        if (!gates_raw.alloc((size_t)gate_dim * batch)) return MQL_FALSE;

        float alpha = 1.0f, beta_zero = 0.0f;

        for (int t = 0; t < seq_len; t++) {
            if (!hx_cache[t].alloc((size_t)concat_dim * batch)) return MQL_FALSE;
            if (!f_cache[t].alloc(hb_size)) return MQL_FALSE;
            if (!i_cache[t].alloc(hb_size)) return MQL_FALSE;
            if (!g_cache[t].alloc(hb_size)) return MQL_FALSE;
            if (!o_cache[t].alloc(hb_size)) return MQL_FALSE;
            if (!c_cache[t].alloc(hb_size)) return MQL_FALSE;
            if (!h_cache[t].alloc(hb_size)) return MQL_FALSE;

            const float* h_prev = (t == 0) ? h_init.ptr : h_cache[t - 1].ptr;
            const float* c_prev = (t == 0) ? c_init.ptr : c_cache[t - 1].ptr;
            const float* x_t = X + (size_t)t * batch * I;

            int total_concat = concat_dim * batch;
            kConcatHX << <(unsigned)((total_concat + 255) / 256), 256, 0, stream_h >> > (
                H, I, batch, h_prev, x_t, hx_cache[t].ptr);
            CUDA_CHECK_KERNEL_RET();

            CUBLAS_CHECK_RET(cublasSgemm(blas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                gate_dim, batch, concat_dim,
                &alpha, W.ptr, concat_dim, hx_cache[t].ptr, concat_dim,
                &beta_zero, gates_raw.ptr, gate_dim));

            dim3 threads(16, 16);
            dim3 blocks_bias(
                (unsigned)((batch + 15) / 16),
                (unsigned)((gate_dim + 15) / 16));
            kAddBiasInplace << <blocks_bias, threads, 0, stream_h >> > (
                gate_dim, batch, gates_raw.ptr, b.ptr);
            CUDA_CHECK_KERNEL_RET();

            int total_hb = (int)hb_size;
            kLSTMGatesForward << <(unsigned)((total_hb + 255) / 256), 256, 0, stream_h >> > (
                H, batch, gates_raw.ptr, c_prev,
                c_cache[t].ptr, h_cache[t].ptr,
                f_cache[t].ptr, i_cache[t].ptr,
                g_cache[t].ptr, o_cache[t].ptr);
            CUDA_CHECK_KERNEL_RET();

            if (use_dropout && training && dropout_rate > 0.0f && curand_gen) {
                size_t rand_count = RoundUpEven(hb_size);
                GPUMemory<float> rand_buf;
                if (!rand_buf.alloc(rand_count)) return MQL_FALSE;
                CURAND_CHECK_RET(curandGenerateUniform(curand_gen,
                    rand_buf.ptr, rand_count));

                if (!dropout_mask[t].alloc(hb_size))   return MQL_FALSE;
                if (!h_drop_cache[t].alloc(hb_size))   return MQL_FALSE;

                CUDA_CHECK_RET(cudaMemcpyAsync(h_drop_cache[t].ptr,
                    h_cache[t].ptr, hb_size * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_h));

                kDropoutForward << <(unsigned)((total_hb + 255) / 256), 256, 0, stream_h >> > (
                    total_hb, h_drop_cache[t].ptr, rand_buf.ptr,
                    dropout_rate, dropout_mask[t].ptr);
                CUDA_CHECK_KERNEL_RET();
                dropout_mask_valid[t] = true;
            }
        }

        return MQL_TRUE;
    }

    MQL_BOOL Backward(cublasHandle_t blas_h, cudaStream_t stream_h,
        const float* dh_last,
        const float* dh_above_seq,
        int seq_len, int batch,
        const float* ones_ptr,
        bool used_dropout_on_h_last)
    {
        if (!is_valid || !blas_h || !stream_h) return MQL_FALSE;

        int H = hidden_size;
        int I = input_size;
        int concat_dim = H + I;
        int gate_dim = 4 * H;
        size_t hb_size = (size_t)H * batch;

        if (!dW.alloc((size_t)concat_dim * gate_dim) || !dW.zeroAsync(stream_h))
            return MQL_FALSE;
        if (!db.alloc(gate_dim) || !db.zeroAsync(stream_h))
            return MQL_FALSE;

        GPUMemory<float> dh_cur, dc_cur, dgates_raw, dhx, dh_prev_tmp;
        if (!dh_cur.alloc(hb_size) || !dh_cur.zeroAsync(stream_h)) return MQL_FALSE;
        if (!dc_cur.alloc(hb_size) || !dc_cur.zeroAsync(stream_h)) return MQL_FALSE;
        if (!dgates_raw.alloc((size_t)gate_dim * batch)) return MQL_FALSE;
        if (!dhx.alloc((size_t)concat_dim * batch))      return MQL_FALSE;
        if (!dh_prev_tmp.alloc(hb_size))    return MQL_FALSE;

        GPUMemory<float> dh_inj;
        if (dh_above_seq) {
            if (!dh_inj.alloc(hb_size)) return MQL_FALSE;
        }

        if (!dx_buf.alloc((size_t)I * batch * seq_len) || !dx_buf.zeroAsync(stream_h))
            return MQL_FALSE;

        if (dh_last) {
            CUDA_CHECK_RET(cudaMemcpyAsync(dh_cur.ptr, dh_last,
                hb_size * sizeof(float), cudaMemcpyDeviceToDevice, stream_h));

            if (used_dropout_on_h_last && use_dropout && dropout_rate > 0.0f &&
                seq_len > 0 &&
                (int)dropout_mask_valid.size() >= seq_len &&
                dropout_mask_valid[seq_len - 1] &&
                (int)dropout_mask.size() >= seq_len &&
                dropout_mask[seq_len - 1].ptr)
            {
                int total_hb = (int)hb_size;
                kDropoutBackward << <(unsigned)((total_hb + 255) / 256), 256, 0, stream_h >> > (
                    total_hb, dh_cur.ptr, dropout_mask[seq_len - 1].ptr, dropout_rate);
                CUDA_CHECK_KERNEL_RET();
            }
        }

        float alpha = 1.0f, beta_zero = 0.0f, beta_one = 1.0f;

        for (int t = seq_len - 1; t >= 0; t--) {
            if (dh_above_seq) {
                const float* dh_src_t = dh_above_seq + (size_t)t * hb_size;

                CUDA_CHECK_RET(cudaMemcpyAsync(dh_inj.ptr, dh_src_t,
                    hb_size * sizeof(float), cudaMemcpyDeviceToDevice, stream_h));

                if (use_dropout && dropout_rate > 0.0f &&
                    t < (int)dropout_mask_valid.size() && dropout_mask_valid[t] &&
                    t < (int)dropout_mask.size() && dropout_mask[t].ptr)
                {
                    int total_hb = (int)hb_size;
                    kDropoutBackward << <(unsigned)((total_hb + 255) / 256), 256, 0, stream_h >> > (
                        total_hb, dh_inj.ptr, dropout_mask[t].ptr, dropout_rate);
                    CUDA_CHECK_KERNEL_RET();
                }

                int total_hb = (int)hb_size;
                kAddInplace << <(unsigned)((total_hb + 255) / 256), 256, 0, stream_h >> > (
                    total_hb, dh_cur.ptr, dh_inj.ptr);
                CUDA_CHECK_KERNEL_RET();
            }

            const float* c_prev = (t == 0) ? c_init.ptr : c_cache[t - 1].ptr;

            int total_hb = (int)hb_size;
            kLSTMGatesBackward << <(unsigned)((total_hb + 255) / 256), 256, 0, stream_h >> > (
                H, batch,
                dh_cur.ptr, dc_cur.ptr,
                c_prev, c_cache[t].ptr,
                f_cache[t].ptr, i_cache[t].ptr,
                g_cache[t].ptr, o_cache[t].ptr,
                dc_cur.ptr,
                dgates_raw.ptr);
            CUDA_CHECK_KERNEL_RET();

            CUBLAS_CHECK_RET(cublasSgemm(blas_h, CUBLAS_OP_N, CUBLAS_OP_T,
                concat_dim, gate_dim, batch,
                &alpha, hx_cache[t].ptr, concat_dim, dgates_raw.ptr, gate_dim,
                &beta_one, dW.ptr, concat_dim));

            CUBLAS_CHECK_RET(cublasSgemv(blas_h, CUBLAS_OP_N, gate_dim, batch,
                &alpha, dgates_raw.ptr, gate_dim, ones_ptr, 1,
                &beta_one, db.ptr, 1));

            CUBLAS_CHECK_RET(cublasSgemm(blas_h, CUBLAS_OP_N, CUBLAS_OP_N,
                concat_dim, batch, gate_dim,
                &alpha, W.ptr, concat_dim, dgates_raw.ptr, gate_dim,
                &beta_zero, dhx.ptr, concat_dim));

            float* dx_t = dx_buf.ptr + (size_t)t * batch * I;
            int total_concat = concat_dim * batch;
            kSplitDHX << <(unsigned)((total_concat + 255) / 256), 256, 0, stream_h >> > (
                H, I, batch, dhx.ptr, dh_prev_tmp.ptr, dx_t);
            CUDA_CHECK_KERNEL_RET();

            if (t > 0) {
                CUDA_CHECK_RET(cudaMemcpyAsync(dh_cur.ptr, dh_prev_tmp.ptr,
                    hb_size * sizeof(float), cudaMemcpyDeviceToDevice, stream_h));
            }
        }

        return MQL_TRUE;
    }

    MQL_BOOL Update(float lr, float c1, float c2, float wd, float clip,
        cudaStream_t stream_h)
    {
        if (!is_valid || !stream_h) return MQL_FALSE;
        const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;

        int concat_dim = input_size + hidden_size;
        int gate_dim = 4 * hidden_size;
        int nW = concat_dim * gate_dim;

        kAdamW << <(unsigned)((nW + 255) / 256), 256, 0, stream_h >> > (
            nW, W.ptr, mW.ptr, vW.ptr, dW.ptr,
            lr, b1, b2, eps, wd, c1, c2, clip);
        CUDA_CHECK_KERNEL_RET();

        kAdamW << <(unsigned)((gate_dim + 255) / 256), 256, 0, stream_h >> > (
            gate_dim, b.ptr, mb.ptr, vb.ptr, db.ptr,
            lr, b1, b2, eps, 0.0f, c1, c2, clip);
        CUDA_CHECK_KERNEL_RET();

        return MQL_TRUE;
    }

    MQL_BOOL SaveBest() {
        if (!is_valid) return MQL_FALSE;
        int concat_dim = input_size + hidden_size;
        int gate_dim = 4 * hidden_size;
        size_t wSize = (size_t)concat_dim * gate_dim;
        if (!W_best.alloc(wSize) || !b_best.alloc(gate_dim)) return MQL_FALSE;
        CUDA_CHECK_RET(cudaMemcpy(W_best.ptr, W.ptr,
            wSize * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK_RET(cudaMemcpy(b_best.ptr, b.ptr,
            gate_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        has_snapshot = true;
        return MQL_TRUE;
    }

    MQL_BOOL RestoreBest() {
        if (!is_valid || !has_snapshot) return MQL_FALSE;
        int concat_dim = input_size + hidden_size;
        int gate_dim = 4 * hidden_size;
        CUDA_CHECK_RET(cudaMemcpy(W.ptr, W_best.ptr,
            (size_t)concat_dim * gate_dim * sizeof(float),
            cudaMemcpyDeviceToDevice));
        CUDA_CHECK_RET(cudaMemcpy(b.ptr, b_best.ptr,
            gate_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        return MQL_TRUE;
    }
};

// ============================================================================
// OutputLayer (linear projection)
// ============================================================================
struct OutputLayer {
    int  in_dim = 0;
    int  out_dim = 0;
    bool is_valid = false;

    GPUMemory<float> W, b;
    GPUMemory<float> mW, vW, mb, vb;
    GPUMemory<float> dW, db;

    GPUMemory<float> W_best, b_best;
    bool has_snapshot = false;

    MQL_BOOL Init(int in_d, int out_d) {
        in_dim = in_d; out_dim = out_d;
        is_valid = false; has_snapshot = false;
        if (in_dim <= 0 || out_dim <= 0) return MQL_FALSE;

        size_t w_size = (size_t)in_dim * out_dim;
        if (!W.alloc(w_size))                    return MQL_FALSE;
        if (!b.alloc(out_dim) || !b.zero())      return MQL_FALSE;
        if (!mW.alloc(w_size) || !mW.zero())     return MQL_FALSE;
        if (!vW.alloc(w_size) || !vW.zero())     return MQL_FALSE;
        if (!mb.alloc(out_dim) || !mb.zero())    return MQL_FALSE;
        if (!vb.alloc(out_dim) || !vb.zero())    return MQL_FALSE;

        std::vector<float> hW(w_size);
        std::mt19937 gen(std::random_device{}());
        float stddev = sqrtf(2.0f / (float)(in_dim + out_dim));
        std::normal_distribution<float> dist(0.0f, stddev);
        for (float& w : hW) w = dist(gen);
        CUDA_CHECK_RET(cudaMemcpy(W.ptr, hW.data(),
            w_size * sizeof(float), cudaMemcpyHostToDevice));

        is_valid = true;
        return MQL_TRUE;
    }

    MQL_BOOL InitFromData(int in_d, int out_d) {
        in_dim = in_d; out_dim = out_d;
        is_valid = false; has_snapshot = false;
        if (in_dim <= 0 || out_dim <= 0) return MQL_FALSE;

        size_t w_size = (size_t)in_dim * out_dim;
        if (!W.alloc(w_size))                    return MQL_FALSE;
        if (!b.alloc(out_dim))                   return MQL_FALSE;
        if (!mW.alloc(w_size) || !mW.zero())     return MQL_FALSE;
        if (!vW.alloc(w_size) || !vW.zero())     return MQL_FALSE;
        if (!mb.alloc(out_dim) || !mb.zero())    return MQL_FALSE;
        if (!vb.alloc(out_dim) || !vb.zero())    return MQL_FALSE;

        is_valid = true;
        return MQL_TRUE;
    }

    MQL_BOOL Forward(cublasHandle_t blas_h, cudaStream_t stream_h,
        const float* h, float* Y, int batch)
    {
        if (!is_valid) return MQL_FALSE;
        float alpha = 1.0f, beta_val = 0.0f;
        CUBLAS_CHECK_RET(cublasSgemm(blas_h, CUBLAS_OP_T, CUBLAS_OP_N,
            out_dim, batch, in_dim, &alpha, W.ptr, in_dim, h, in_dim,
            &beta_val, Y, out_dim));
        dim3 threads(16, 16);
        dim3 blocks(
            (unsigned)((batch + 15) / 16),
            (unsigned)((out_dim + 15) / 16));
        kBiasLinear << <blocks, threads, 0, stream_h >> > (
            out_dim, batch, Y, b.ptr);
        CUDA_CHECK_KERNEL_RET();
        return MQL_TRUE;
    }

    MQL_BOOL Backward(cublasHandle_t blas_h, cudaStream_t stream_h,
        const float* dY, const float* h, float* dh_out,
        const float* ones_ptr, int batch)
    {
        if (!is_valid) return MQL_FALSE;
        float alpha = 1.0f, beta_val = 0.0f;

        if (!dW.alloc((size_t)in_dim * out_dim)) return MQL_FALSE;
        CUBLAS_CHECK_RET(cublasSgemm(blas_h, CUBLAS_OP_N, CUBLAS_OP_T,
            in_dim, out_dim, batch, &alpha, h, in_dim, dY, out_dim,
            &beta_val, dW.ptr, in_dim));

        if (!db.alloc(out_dim)) return MQL_FALSE;
        CUBLAS_CHECK_RET(cublasSgemv(blas_h, CUBLAS_OP_N, out_dim, batch,
            &alpha, dY, out_dim, ones_ptr, 1, &beta_val, db.ptr, 1));

        if (dh_out) {
            CUBLAS_CHECK_RET(cublasSgemm(blas_h, CUBLAS_OP_N, CUBLAS_OP_N,
                in_dim, batch, out_dim, &alpha, W.ptr, in_dim, dY, out_dim,
                &beta_val, dh_out, in_dim));
        }
        return MQL_TRUE;
    }

    MQL_BOOL Update(float lr, float c1, float c2, float wd, float clip,
        cudaStream_t stream_h)
    {
        if (!is_valid) return MQL_FALSE;
        const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        int nW = in_dim * out_dim;
        kAdamW << <(unsigned)((nW + 255) / 256), 256, 0, stream_h >> > (
            nW, W.ptr, mW.ptr, vW.ptr, dW.ptr,
            lr, b1, b2, eps, wd, c1, c2, clip);
        CUDA_CHECK_KERNEL_RET();
        kAdamW << <(unsigned)((out_dim + 255) / 256), 256, 0, stream_h >> > (
            out_dim, b.ptr, mb.ptr, vb.ptr, db.ptr,
            lr, b1, b2, eps, 0.0f, c1, c2, clip);
        CUDA_CHECK_KERNEL_RET();
        return MQL_TRUE;
    }

    MQL_BOOL SaveBest() {
        if (!is_valid) return MQL_FALSE;
        size_t wSize = (size_t)in_dim * out_dim;
        if (!W_best.alloc(wSize) || !b_best.alloc(out_dim)) return MQL_FALSE;
        CUDA_CHECK_RET(cudaMemcpy(W_best.ptr, W.ptr,
            wSize * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK_RET(cudaMemcpy(b_best.ptr, b.ptr,
            out_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        has_snapshot = true;
        return MQL_TRUE;
    }

    MQL_BOOL RestoreBest() {
        if (!is_valid || !has_snapshot) return MQL_FALSE;
        CUDA_CHECK_RET(cudaMemcpy(W.ptr, W_best.ptr,
            (size_t)in_dim * out_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK_RET(cudaMemcpy(b.ptr, b_best.ptr,
            out_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        return MQL_TRUE;
    }
};

// ============================================================================
// LSTMNet — Main network class (multi-layer LSTM + output)
// ============================================================================
class LSTMNet {
    std::vector<std::unique_ptr<LSTMLayer>> lstm_layers;
    std::unique_ptr<OutputLayer> output_layer;

    int seq_len = 1;

    GPUMemory<float> X_full, T_full;
    GPUMemory<float> ones;
    GPUMemory<float> mse_reduce_buf;
    GPUMemory<float> dLoss;
    GPUMemory<float> dh_buf;
    GPUMemory<float> Y_pred;
    GPUMemory<float> debug_sum;

    std::vector<GPUMemory<float>> inter_layer_ts;

    int loaded_samples = 0;
    int loaded_in_dim = 0;
    int loaded_out_dim = 0;

    unsigned long long step = 0;
    float b1_pow = 1.0f, b2_pow = 1.0f;
    float grad_clip = 5.0f;

    int mini_batch_size = 64;
    std::vector<int> host_indices;

    float base_lr = 0.001f;
    int   warmup_steps = 500;
    int   total_schedule_steps = 50000;

    double last_full_train_mse = 0.0;

    cudaStream_t      stream = nullptr;
    cublasHandle_t    blas = nullptr;
    curandGenerator_t curand_gen = nullptr;
    bool              init_ok = false;

    std::mt19937 host_rng;

    // Asynchronní správa
    std::atomic<int>  m_state{ TS_IDLE };
    std::atomic<bool> m_stop_flag{ false };
    std::thread       m_worker;
    double            m_final_mse = 0.0;
    int               m_final_epochs = 0;

    struct DebugInfo {
        float output_min = 0, output_max = 0, grad_norm = 0;
        std::vector<float> layer_weight_norms;
    };
    DebugInfo last_debug;

    float GetScheduledLR(int current_step) const {
        float lr = base_lr;
        if (current_step < warmup_steps) {
            lr = base_lr * ((float)(current_step + 1) / (float)warmup_steps);
        }
        else {
            float progress = (float)(current_step - warmup_steps) /
                (float)std::max(1, total_schedule_steps - warmup_steps);
            progress = std::min(progress, 1.0f);
            lr = base_lr * 0.5f * (1.0f + cosf(3.14159265f * progress));
            lr = std::max(lr, base_lr * 0.01f);
        }
        return lr;
    }

    float ComputeL2Norm(const float* buf, int n) {
        if (!buf || n <= 0) return 0.0f;
        if (!debug_sum.alloc(1)) return 0.0f;
        CUDA_CHECK_VOID(cudaMemsetAsync(debug_sum.ptr, 0, sizeof(float), stream));
        int threads = 256;
        int blocks = std::min((n + threads - 1) / threads, 1024);
        kL2NormReduceWarp << <blocks, threads, 0, stream >> > (n, buf, debug_sum.ptr);
        CUDA_CHECK_VOID(cudaGetLastError());
        float sum = 0.0f;
        CUDA_CHECK_VOID(cudaMemcpyAsync(&sum, debug_sum.ptr, sizeof(float),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK_VOID(cudaStreamSynchronize(stream));
        return sqrtf(sum);
    }

    void ComputeMinMax(const float* buf, int n, float& lo, float& hi) {
        lo = 0.0f; hi = 0.0f;
        if (!buf || n <= 0) return;
        int nBlocks = std::min((n + 255) / 256, 1024);
        GPUMemory<float> mm_min, mm_max;
        if (!mm_min.alloc(nBlocks) || !mm_max.alloc(nBlocks)) return;
        kMinMaxReduce256 << <nBlocks, 256, 0, stream >> > (n, buf, mm_min.ptr, mm_max.ptr);
        CUDA_CHECK_VOID(cudaGetLastError());
        std::vector<float> h_min(nBlocks), h_max(nBlocks);
        CUDA_CHECK_VOID(cudaMemcpyAsync(h_min.data(), mm_min.ptr,
            nBlocks * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK_VOID(cudaMemcpyAsync(h_max.data(), mm_max.ptr,
            nBlocks * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK_VOID(cudaStreamSynchronize(stream));
        lo = 1e30f; hi = -1e30f;
        for (int i = 0; i < nBlocks; i++) {
            lo = std::min(lo, h_min[i]);
            hi = std::max(hi, h_max[i]);
        }
    }

    MQL_BOOL ForwardFull(const float* X_timestep_major, int sl, int batch,
        bool training, float* Y_buf)
    {
        int nLayers = (int)lstm_layers.size();
        if (nLayers == 0 || !output_layer || !output_layer->is_valid)
            return MQL_FALSE;

        if ((int)inter_layer_ts.size() < std::max(0, nLayers - 1)) {
            inter_layer_ts.resize(std::max(0, nLayers - 1));
        }

        for (int li = 0; li < nLayers; li++) {
            LSTMLayer* layer = lstm_layers[li].get();

            if (li == 0) {
                if (!layer->Forward(blas, stream, X_timestep_major, sl, batch,
                    training, curand_gen))
                    return MQL_FALSE;
                continue;
            }

            LSTMLayer* prev = lstm_layers[li - 1].get();
            int H_prev = prev->hidden_size;

            size_t needed = (size_t)H_prev * sl * batch;
            GPUMemory<float>& buf = inter_layer_ts[li - 1];
            if (!buf.alloc(needed)) return MQL_FALSE;

            for (int t = 0; t < sl; t++) {
                const float* src = prev->GetOutputH(t, training);
                CUDA_CHECK_RET(cudaMemcpyAsync(
                    buf.ptr + (size_t)t * batch * H_prev,
                    src,
                    (size_t)H_prev * batch * sizeof(float),
                    cudaMemcpyDeviceToDevice,
                    stream));
            }

            if (!layer->Forward(blas, stream, buf.ptr, sl, batch,
                training, curand_gen))
                return MQL_FALSE;
        }

        LSTMLayer* last_lstm = lstm_layers.back().get();
        const float* h_last = last_lstm->GetOutputH(sl - 1, training);

        if (!output_layer->Forward(blas, stream, h_last, Y_buf, batch))
            return MQL_FALSE;

        return MQL_TRUE;
    }

    double ComputeFullTrainMSE() {
        if (loaded_samples <= 0 || !X_full.ptr || !T_full.ptr ||
            lstm_layers.empty() || !output_layer)
            return -1.0;

        int batch = loaded_samples;
        int out_dim = loaded_out_dim;
        int in_dim = loaded_in_dim;

        int eval_batch = std::min(batch, 256);

        GPUMemory<float> X_mb_ts;
        GPUMemory<float> T_mb;
        GPUMemory<int>   idx;
        GPUMemory<float> Y_eval;
        if (!X_mb_ts.alloc((size_t)eval_batch * seq_len * in_dim)) return -1.0;
        if (!T_mb.alloc((size_t)eval_batch * out_dim)) return -1.0;
        if (!idx.alloc(eval_batch)) return -1.0;
        if (!Y_eval.alloc((size_t)out_dim * eval_batch)) return -1.0;
        if (!mse_reduce_buf.alloc(1)) return -1.0;

        std::vector<int> hidx(eval_batch);
        double total_mse_sum = 0.0;
        int total_out = 0;

        for (int start = 0; start < batch; start += eval_batch) {
            int cur_batch = std::min(eval_batch, batch - start);

            for (int i = 0; i < cur_batch; i++) hidx[i] = start + i;

            CUDA_CHECK_VOID(cudaMemcpyAsync(idx.ptr, hidx.data(),
                (size_t)cur_batch * sizeof(int), cudaMemcpyHostToDevice, stream));

            int total_x = cur_batch * seq_len * in_dim;
            kGatherTransposeSeq << <(unsigned)((total_x + 255) / 256), 256, 0, stream >> > (
                cur_batch, seq_len, in_dim, idx.ptr, X_full.ptr, X_mb_ts.ptr);
            CUDA_CHECK_VOID(cudaGetLastError());

            int total_t = cur_batch * out_dim;
            kGatherRows << <(unsigned)((total_t + 255) / 256), 256, 0, stream >> > (
                out_dim, cur_batch, T_full.ptr, idx.ptr, T_mb.ptr);
            CUDA_CHECK_VOID(cudaGetLastError());

            if (!ForwardFull(X_mb_ts.ptr, seq_len, cur_batch, false, Y_eval.ptr))
                return -1.0;

            int nOut = cur_batch * out_dim;
            CUDA_CHECK_VOID(cudaMemsetAsync(mse_reduce_buf.ptr, 0,
                sizeof(float), stream));

            int threads = 256;
            int blocks = std::min((nOut + threads - 1) / threads, 1024);
            kMSEReduceWarp << <blocks, threads, 0, stream >> > (
                nOut, Y_eval.ptr, T_mb.ptr, mse_reduce_buf.ptr);
            CUDA_CHECK_VOID(cudaGetLastError());

            float sum = 0.0f;
            CUDA_CHECK_VOID(cudaMemcpyAsync(&sum, mse_reduce_buf.ptr,
                sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK_VOID(cudaStreamSynchronize(stream));

            total_mse_sum += (double)sum;
            total_out += nOut;
        }

        if (total_out <= 0) return -1.0;
        return total_mse_sum / (double)total_out;
    }

    // Vnitřní trénovací smyčka (původní TrainLoaded)
    MQL_BOOL TrainInternal(int max_epochs, double target_mse, double lr, double wd)
    {
        if (!init_ok || !blas || !stream || loaded_samples <= 0 ||
            lstm_layers.empty() || !output_layer || !output_layer->is_valid)
            return MQL_FALSE;

        base_lr = (float)lr;
        int batch = loaded_samples;
        int out_dim = loaded_out_dim;
        int in_dim = loaded_in_dim;
        int nLayers = (int)lstm_layers.size();
        int H_last = lstm_layers.back()->hidden_size;

        int nMiniBatches = (batch + mini_batch_size - 1) / mini_batch_size;
        total_schedule_steps = max_epochs * nMiniBatches;
        warmup_steps = std::min(500, std::max(1, total_schedule_steps / 10));

        GPUMemory<float> X_mb_ts, T_mb, Y_pred_mb, dLoss_mb, dh_buf_mb;
        GPUMemory<int>   mb_idx;

        size_t max_mb_x = (size_t)mini_batch_size * seq_len * in_dim;
        size_t max_mb_t = (size_t)mini_batch_size * out_dim;

        if (!X_mb_ts.alloc(max_mb_x)) return MQL_FALSE;
        if (!T_mb.alloc(max_mb_t))    return MQL_FALSE;
        if (!mb_idx.alloc(mini_batch_size)) return MQL_FALSE;
        if (!Y_pred_mb.alloc(max_mb_t))  return MQL_FALSE;
        if (!dLoss_mb.alloc(max_mb_t))   return MQL_FALSE;
        if (!dh_buf_mb.alloc((size_t)mini_batch_size * H_last)) return MQL_FALSE;

        int final_epoch = 0;
        MQL_BOOL success = MQL_TRUE;

        for (int epoch = 1; epoch <= max_epochs; ++epoch) {
            if (m_stop_flag.load()) break;

            std::shuffle(host_indices.begin(), host_indices.end(), host_rng);

            for (int mb = 0; mb < nMiniBatches; mb++) {
                int mb_start = mb * mini_batch_size;
                int mb_size = std::min(mini_batch_size, batch - mb_start);

                step++;
                b1_pow *= 0.9f;
                b2_pow *= 0.999f;
                float c1 = 1.0f / (1.0f - b1_pow);
                float c2 = 1.0f / (1.0f - b2_pow);
                float cur_lr = GetScheduledLR((int)step);

                CUDA_CHECK_RET(cudaMemcpyAsync(mb_idx.ptr,
                    host_indices.data() + mb_start,
                    (size_t)mb_size * sizeof(int),
                    cudaMemcpyHostToDevice, stream));

                int total_x = mb_size * seq_len * in_dim;
                kGatherTransposeSeq << <(unsigned)((total_x + 255) / 256), 256, 0, stream >> > (
                    mb_size, seq_len, in_dim, mb_idx.ptr, X_full.ptr, X_mb_ts.ptr);
                CUDA_CHECK_KERNEL_RET();

                int total_t = mb_size * out_dim;
                kGatherRows << <(unsigned)((total_t + 255) / 256), 256, 0, stream >> > (
                    out_dim, mb_size, T_full.ptr, mb_idx.ptr, T_mb.ptr);
                CUDA_CHECK_KERNEL_RET();

                if (!ForwardFull(X_mb_ts.ptr, seq_len, mb_size, true, Y_pred_mb.ptr))
                    return MQL_FALSE;

                int nOut = mb_size * out_dim;
                kMSEGrad << <(unsigned)((nOut + 255) / 256), 256, 0, stream >> > (
                    nOut, Y_pred_mb.ptr, T_mb.ptr, dLoss_mb.ptr);
                CUDA_CHECK_KERNEL_RET();

                LSTMLayer* last_lstm = lstm_layers.back().get();
                const float* h_for_output = last_lstm->GetOutputH(seq_len - 1, true);
                bool used_dropout_on_h = (h_for_output != last_lstm->h_cache[seq_len - 1].ptr);

                if (!output_layer->Backward(blas, stream, dLoss_mb.ptr,
                    h_for_output, dh_buf_mb.ptr, ones.ptr, mb_size))
                    return MQL_FALSE;

                for (int li = nLayers - 1; li >= 0; li--) {
                    LSTMLayer* layer = lstm_layers[li].get();
                    const float* dh_last_ptr = nullptr;
                    const float* dh_above = nullptr;
                    bool had_dropout = false;

                    if (li == nLayers - 1) {
                        dh_last_ptr = dh_buf_mb.ptr;
                        had_dropout = used_dropout_on_h;
                    }
                    else {
                        LSTMLayer* next_layer = lstm_layers[li + 1].get();
                        dh_above = next_layer->dx_buf.ptr;
                    }

                    if (!layer->Backward(blas, stream, dh_last_ptr, dh_above, seq_len, mb_size,
                        ones.ptr, had_dropout))
                        return MQL_FALSE;
                }

                for (int li = 0; li < nLayers; li++)
                    if (!lstm_layers[li]->Update(cur_lr, c1, c2, (float)wd, grad_clip, stream))
                        return MQL_FALSE;
                if (!output_layer->Update(cur_lr, c1, c2, (float)wd, grad_clip, stream))
                    return MQL_FALSE;
            }

            final_epoch = epoch;
            bool collect = (epoch == 1 || epoch == max_epochs || (epoch % 50) == 0);
            if (collect) {
                last_full_train_mse = ComputeFullTrainMSE();
                if (target_mse > 0.0 && last_full_train_mse > 0.0 &&
                    last_full_train_mse <= target_mse)
                    break;
            }
        }

        if (last_full_train_mse <= 0.0)
            last_full_train_mse = ComputeFullTrainMSE();

        m_final_mse = last_full_train_mse;
        m_final_epochs = final_epoch;
        return MQL_TRUE;
    }

public:
    std::mutex net_mtx;

    float GetOutputMin()  const { return last_debug.output_min; }
    float GetOutputMax()  const { return last_debug.output_max; }
    float GetGradNorm()   const { return last_debug.grad_norm; }
    int   GetLayerCount() const { return (int)lstm_layers.size() + (output_layer ? 1 : 0); }
    float GetLayerWeightNorm(int i) const { return 0.0f; }
    float GetLayerGradNorm(int i)   const { return 0.0f; }
    float GetLayerActMin(int i)     const { return 0.0f; }
    float GetLayerActMax(int i)     const { return 0.0f; }
    float GetLayerAliveRatio(int i) const { return 1.0f; }
    bool  IsInitOK()                const { return init_ok; }

    // Asynchronní rozhraní
    int GetStatus() const { return m_state.load(); }

    void GetResult(double& mse, int& ep) const {
        mse = m_final_mse;
        ep = m_final_epochs;
    }

    void StopTraining() {
        m_stop_flag.store(true);
    }

    MQL_BOOL StartTrainingAsync(int max_epochs, double target_mse, double lr, double wd) {
        if (m_state.load() == TS_TRAINING) return MQL_FALSE;

        // Pokud běží staré vlákno (třeba už hotové), připojíme ho
        if (m_worker.joinable()) m_worker.join();

        m_stop_flag.store(false);
        m_state.store(TS_TRAINING);

        m_worker = std::thread([this, max_epochs, target_mse, lr, wd]() {
            // V novém vlákně musíme nastavit device
            cudaSetDevice(0);

            if (TrainInternal(max_epochs, target_mse, lr, wd)) {
                m_state.store(TS_COMPLETED);
            }
            else {
                m_state.store(TS_ERROR);
            }
            });

        return MQL_TRUE;
    }

    LSTMNet() : host_rng(std::random_device{}()) {
        init_ok = false;
        if (cudaSetDevice(0) != cudaSuccess) return;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return;
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) return;
        cublasSetStream(blas, stream);
        if (curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) return;
        curandSetPseudoRandomGeneratorSeed(curand_gen, (unsigned long long)std::random_device{}());
        curandSetStream(curand_gen, stream);
        init_ok = true;
    }

    ~LSTMNet() {
        m_stop_flag.store(true);
        if (m_worker.joinable()) m_worker.join();

        lstm_layers.clear();
        output_layer.reset();
        if (curand_gen) curandDestroyGenerator(curand_gen);
        if (blas) cublasDestroy(blas);
        if (stream) cudaStreamDestroy(stream);
    }

    // Settery
    void SetGradClip(float v) { grad_clip = v; }
    void SetSequenceLength(int sl) { seq_len = std::max(1, sl); }
    void SetMiniBatchSize(int mbs) { mini_batch_size = std::max(1, mbs); }

    // Konstrukce sítě
    MQL_BOOL AddLSTMLayer(int in_dim, int hidden_size, float dropout = 0.0f) {
        if (!init_ok) return MQL_FALSE;
        auto l = std::make_unique<LSTMLayer>();
        if (in_dim > 0) {
            if (!l->Init(in_dim, hidden_size, dropout)) return MQL_FALSE;
        }
        else {
            l->input_size = 0; l->hidden_size = hidden_size;
            l->is_valid = false; l->use_dropout = (dropout > 0.0f);
            l->dropout_rate = dropout;
        }
        lstm_layers.push_back(std::move(l));
        return MQL_TRUE;
    }

    MQL_BOOL SetOutputLayer(int out_dim) {
        if (!init_ok || lstm_layers.empty()) return MQL_FALSE;
        output_layer = std::make_unique<OutputLayer>();
        return output_layer->Init(lstm_layers.back()->hidden_size, out_dim);
    }

    MQL_BOOL AddLayer(int in, int out, int act, int use_ln, float dropout) {
        return AddLSTMLayer(in, out, dropout);
    }

    MQL_BOOL BindInputIfNeeded(int in_dim) {
        if (lstm_layers.empty()) return MQL_FALSE;
        LSTMLayer* L0 = lstm_layers[0].get();
        if (!L0->is_valid || L0->input_size != in_dim) {
            // Re-init logic (zjednodušeno pro stručnost, v plném kódu zachovat free)
            L0->W.free(); L0->b.free(); L0->mW.free(); L0->vW.free();
            L0->mb.free(); L0->vb.free(); L0->dW.free(); L0->db.free();
            L0->W_best.free(); L0->b_best.free(); L0->has_snapshot = false;
            return L0->Init(in_dim, L0->hidden_size, L0->dropout_rate);
        }
        return MQL_TRUE;
    }

    MQL_BOOL BindIntermediateLayers() {
        for (int i = 1; i < (int)lstm_layers.size(); i++) {
            int prev_hidden = lstm_layers[i - 1]->hidden_size;
            LSTMLayer* cur = lstm_layers[i].get();
            if (!cur->is_valid || cur->input_size != prev_hidden) {
                cur->W.free(); cur->b.free();
                // ... free buffers ...
                if (!cur->Init(prev_hidden, cur->hidden_size, cur->dropout_rate)) return MQL_FALSE;
            }
        }
        return MQL_TRUE;
    }

    MQL_BOOL SnapshotWeights() {
        for (auto& l : lstm_layers) if (!l->SaveBest()) return MQL_FALSE;
        if (output_layer) return output_layer->SaveBest();
        return MQL_TRUE;
    }

    MQL_BOOL RestoreWeights() {
        for (auto& l : lstm_layers) if (!l->RestoreBest()) return MQL_FALSE;
        if (output_layer) return output_layer->RestoreBest();
        return MQL_TRUE;
    }

    // LoadBatch
    MQL_BOOL LoadBatch(const double* X, const double* T, int batch, int in_dim, int out_dim, int layout) {
        if (!init_ok || !blas || !stream || !X || !T) return MQL_FALSE;
        if (m_state.load() == TS_TRAINING) return MQL_FALSE; // Cannot load while training

        int feature_dim = in_dim / seq_len;
        if (feature_dim * seq_len != in_dim) return MQL_FALSE;

        if (!BindInputIfNeeded(feature_dim)) return MQL_FALSE;
        if (!BindIntermediateLayers()) return MQL_FALSE;

        if (!output_layer || !output_layer->is_valid ||
            output_layer->out_dim != out_dim ||
            output_layer->in_dim != lstm_layers.back()->hidden_size) {
            output_layer = std::make_unique<OutputLayer>();
            if (!output_layer->Init(lstm_layers.back()->hidden_size, out_dim)) return MQL_FALSE;
        }

        size_t nX = (size_t)batch * seq_len * feature_dim;
        size_t nT = (size_t)batch * out_dim;

        if (!X_full.alloc(nX) || !T_full.alloc(nT)) return MQL_FALSE;
        GPUMemory<double> tmpX, tmpT;
        if (!tmpX.alloc(nX) || !tmpT.alloc(nT)) return MQL_FALSE;

        CUDA_CHECK_RET(cudaMemcpyAsync(tmpX.ptr, X, nX * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK_RET(cudaMemcpyAsync(tmpT.ptr, T, nT * sizeof(double), cudaMemcpyHostToDevice, stream));

        int threads = 256;
        int blocksX = std::min((int)((nX + 1023) / 1024), 4096);
        int blocksT = std::min((int)((nT + 1023) / 1024), 4096);

        kCopyD2F_vec4 << <blocksX, threads, 0, stream >> > ((int)nX, tmpX.ptr, X_full.ptr);
        kCopyD2F_vec4 << <blocksT, threads, 0, stream >> > ((int)nT, tmpT.ptr, T_full.ptr);

        int max_b = std::max(batch, mini_batch_size);
        if (ones.count < (size_t)max_b) {
            if (!ones.alloc((size_t)max_b)) return MQL_FALSE;
            std::vector<float> h1((size_t)max_b, 1.0f);
            CUDA_CHECK_RET(cudaMemcpyAsync(ones.ptr, h1.data(), max_b * sizeof(float), cudaMemcpyHostToDevice, stream));
        }
        if (!mse_reduce_buf.alloc(1)) return MQL_FALSE;

        host_indices.resize(batch);
        std::iota(host_indices.begin(), host_indices.end(), 0);
        loaded_samples = batch;
        loaded_in_dim = feature_dim;
        loaded_out_dim = out_dim;

        CUDA_CHECK_RET(cudaStreamSynchronize(stream));
        b1_pow = 1.0f; b2_pow = 1.0f; step = 0;
        last_full_train_mse = 0.0;
        return MQL_TRUE;
    }

    MQL_BOOL PredictBatch(const double* X, int batch, int in_dim, int layout, double* out_Y) {
        if (m_state.load() == TS_TRAINING) return MQL_FALSE; // Block prediction while training
        if (!init_ok || !blas || !stream || !X || !out_Y) return MQL_FALSE;

        int feature_dim = in_dim / seq_len;
        if (!BindInputIfNeeded(feature_dim)) return MQL_FALSE;
        if (!BindIntermediateLayers()) return MQL_FALSE;

        int out_dim = output_layer->out_dim;
        size_t nX = (size_t)in_dim * batch;
        size_t nY = (size_t)out_dim * batch;

        GPUMemory<float> X_float, X_ts, Y_float;
        GPUMemory<double> tmpDX, outDY;

        if (!X_float.alloc(nX) || !tmpDX.alloc(nX)) return MQL_FALSE;
        CUDA_CHECK_RET(cudaMemcpyAsync(tmpDX.ptr, X, nX * sizeof(double), cudaMemcpyHostToDevice, stream));
        int threads = 256;
        kCopyD2F_vec4 << <(nX + 1023) / 1024, threads, 0, stream >> > ((int)nX, tmpDX.ptr, X_float.ptr);

        if (!X_ts.alloc(nX)) return MQL_FALSE;
        kTransposeToTimestep << <(nX + 255) / 256, 256, 0, stream >> > (batch, seq_len, feature_dim, X_float.ptr, X_ts.ptr);

        if (ones.count < (size_t)batch) {
            if (!ones.alloc((size_t)batch)) return MQL_FALSE;
            std::vector<float> h1((size_t)batch, 1.0f);
            CUDA_CHECK_RET(cudaMemcpyAsync(ones.ptr, h1.data(), batch * sizeof(float), cudaMemcpyHostToDevice, stream));
        }

        if (!Y_float.alloc(nY)) return MQL_FALSE;
        if (!ForwardFull(X_ts.ptr, seq_len, batch, false, Y_float.ptr)) return MQL_FALSE;

        if (!outDY.alloc(nY)) return MQL_FALSE;
        kCopyF2D_vec4 << <(nY + 1023) / 1024, threads, 0, stream >> > ((int)nY, Y_float.ptr, outDY.ptr);
        CUDA_CHECK_RET(cudaMemcpyAsync(out_Y, outDY.ptr, nY * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK_RET(cudaStreamSynchronize(stream));
        return MQL_TRUE;
    }

    MQL_BOOL EvalMSE(double* out_mse) {
        if (m_state.load() == TS_TRAINING) return MQL_FALSE;
        double mse = ComputeFullTrainMSE();
        if (mse < 0) return MQL_FALSE;
        if (out_mse) *out_mse = mse;
        return MQL_TRUE;
    }

    // Serialization
    std::string serialize_buf;
    MQL_BOOL Save(std::string& out_buf) {
        if (lstm_layers.empty() || !output_layer) return MQL_FALSE;
        std::stringstream ss;
        ss << std::fixed << std::setprecision(8) << "LSTM_V1\nSEQ_LEN " << seq_len << "\nLSTM_LAYERS " << lstm_layers.size() << "\n";
        for (auto& l : lstm_layers) {
            int dim = l->input_size + l->hidden_size, g = 4 * l->hidden_size;
            ss << l->input_size << " " << l->hidden_size << " " << l->dropout_rate << "\n";
            std::vector<float> Wv(dim * g), bv(g);
            cudaMemcpy(Wv.data(), l->W.ptr, Wv.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(bv.data(), l->b.ptr, bv.size() * 4, cudaMemcpyDeviceToHost);
            for (float x : Wv) ss << x << " "; ss << "\n";
            for (float x : bv) ss << x << " "; ss << "\n";
        }
        ss << "OUTPUT " << output_layer->in_dim << " " << output_layer->out_dim << "\n";
        int dim = output_layer->in_dim * output_layer->out_dim;
        std::vector<float> Wv(dim), bv(output_layer->out_dim);
        cudaMemcpy(Wv.data(), output_layer->W.ptr, dim * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(bv.data(), output_layer->b.ptr, bv.size() * 4, cudaMemcpyDeviceToHost);
        for (float x : Wv) ss << x << " "; ss << "\n";
        for (float x : bv) ss << x << " "; ss << "\n";
        out_buf = ss.str();
        return MQL_TRUE;
    }

    MQL_BOOL Load(const char* data) {
        if (!data || !init_ok) return MQL_FALSE;
        std::istringstream ss(data);
        std::string t;
        if (!(ss >> t) || t != "LSTM_V1") return MQL_FALSE;
        if (!(ss >> t) || t != "SEQ_LEN" || !(ss >> seq_len)) return MQL_FALSE;
        if (!(ss >> t) || t != "LSTM_LAYERS") return MQL_FALSE;
        int nl; if (!(ss >> nl)) return MQL_FALSE;
        lstm_layers.clear();
        for (int i = 0; i < nl; i++) {
            int in_d, hid_d; float drop;
            if (!(ss >> in_d >> hid_d >> drop)) return MQL_FALSE;
            auto l = std::make_unique<LSTMLayer>();
            if (!l->InitFromData(in_d, hid_d, drop)) return MQL_FALSE;
            int dim = in_d + hid_d, g = 4 * hid_d;
            std::vector<float> Wv(dim * g), bv(g);
            for (size_t k = 0; k < Wv.size(); k++) ss >> Wv[k];
            for (size_t k = 0; k < bv.size(); k++) ss >> bv[k];
            cudaMemcpy(l->W.ptr, Wv.data(), Wv.size() * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(l->b.ptr, bv.data(), bv.size() * 4, cudaMemcpyHostToDevice);
            lstm_layers.push_back(std::move(l));
        }
        if (!(ss >> t) || t != "OUTPUT") return MQL_FALSE;
        int oi, oo; if (!(ss >> oi >> oo)) return MQL_FALSE;
        output_layer = std::make_unique<OutputLayer>();
        if (!output_layer->InitFromData(oi, oo)) return MQL_FALSE;
        std::vector<float> Wv(oi * oo), bv(oo);
        for (size_t k = 0; k < Wv.size(); k++) ss >> Wv[k];
        for (size_t k = 0; k < bv.size(); k++) ss >> bv[k];
        cudaMemcpy(output_layer->W.ptr, Wv.data(), Wv.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(output_layer->b.ptr, bv.data(), bv.size() * 4, cudaMemcpyHostToDevice);
        return MQL_TRUE;
    }
};

// ============================================================================
// DLL Exports
// ============================================================================
static std::map<int, std::unique_ptr<LSTMNet>> g_nets;
static int        g_id = 1;
static std::mutex g_map_mtx;

static LSTMNet* FindAndLockNet(int h, std::unique_lock<std::mutex>& lk) {
    std::lock_guard<std::mutex> map_lk(g_map_mtx);
    auto it = g_nets.find(h);
    if (it == g_nets.end()) return nullptr;
    LSTMNet* net = it->second.get();
    lk = std::unique_lock<std::mutex>(net->net_mtx);
    return net;
}

DLL_EXPORT int DLL_CALL DN_Create() {
    auto net = std::make_unique<LSTMNet>();
    if (!net->IsInitOK()) return 0;
    std::lock_guard<std::mutex> l(g_map_mtx);
    int id = g_id++;
    g_nets[id] = std::move(net);
    return id;
}

DLL_EXPORT void DLL_CALL DN_Free(int h) {
    std::lock_guard<std::mutex> l(g_map_mtx);
    g_nets.erase(h);
}

DLL_EXPORT MQL_BOOL DLL_CALL DN_SetSequenceLength(int h, int seq_len) {
    std::unique_lock<std::mutex> lk;
    LSTMNet* net = FindAndLockNet(h, lk);
    return net ? net->SetSequenceLength(seq_len), MQL_TRUE : MQL_FALSE;
}

DLL_EXPORT MQL_BOOL DLL_CALL DN_SetMiniBatchSize(int h, int mbs) {
    std::unique_lock<std::mutex> lk;
    LSTMNet* net = FindAndLockNet(h, lk);
    return net ? net->SetMiniBatchSize(mbs), MQL_TRUE : MQL_FALSE;
}

DLL_EXPORT MQL_BOOL DLL_CALL DN_AddLayerEx(int h, int in, int out, int act, int ln, double drop) {
    std::unique_lock<std::mutex> lk;
    LSTMNet* net = FindAndLockNet(h, lk);
    return net ? net->AddLayer(in, out, act, ln, (float)drop) : MQL_FALSE;
}

DLL_EXPORT MQL_BOOL DLL_CALL DN_SetGradClip(int h, double clip) {
    std::unique_lock<std::mutex> lk;
    LSTMNet* net = FindAndLockNet(h, lk);
    if (!net) return MQL_FALSE;
    net->SetGradClip((float)clip);
    return MQL_TRUE;
}

DLL_EXPORT MQL_BOOL DLL_CALL DN_LoadBatch(int h, const double* X, const double* T, int batch, int in, int out, int l) {
    std::unique_lock<std::mutex> lk;
    LSTMNet* net = FindAndLockNet(h, lk);
    return net ? net->LoadBatch(X, T, batch, in, out, l) : MQL_FALSE;
}

DLL_EXPORT MQL_BOOL DLL_CALL DN_PredictBatch(int h, const double* X, int batch, int in, int l, double* Y) {
    std::unique_lock<std::mutex> lk;
    LSTMNet* net = FindAndLockNet(h, lk);
    return net ? net->PredictBatch(X, batch, in, l, Y) : MQL_FALSE;
}

DLL_EXPORT MQL_BOOL DLL_CALL DN_SnapshotWeights(int h) {
    std::unique_lock<std::mutex> lk;
    LSTMNet* net = FindAndLockNet(h, lk);
    return net ? net->SnapshotWeights() : MQL_FALSE;
}

DLL_EXPORT MQL_BOOL DLL_CALL DN_RestoreWeights(int h) {
    std::unique_lock<std::mutex> lk;
    LSTMNet* net = FindAndLockNet(h, lk);
    return net ? net->RestoreWeights() : MQL_FALSE;
}

// --- Asynchronní exporty ---

DLL_EXPORT MQL_BOOL DLL_CALL DN_TrainAsync(int h, int epochs, double target_mse, double lr, double wd) {
    std::unique_lock<std::mutex> lk;
    LSTMNet* net = FindAndLockNet(h, lk);
    return net ? net->StartTrainingAsync(epochs, target_mse, lr, wd) : MQL_FALSE;
}

DLL_EXPORT int DLL_CALL DN_GetTrainingStatus(int h) {
    std::unique_lock<std::mutex> lk;
    LSTMNet* net = FindAndLockNet(h, lk);
    return net ? net->GetStatus() : -1;
}

DLL_EXPORT void DLL_CALL DN_GetTrainingResult(int h, double* out_mse, int* out_epochs) {
    std::unique_lock<std::mutex> lk;
    LSTMNet* net = FindAndLockNet(h, lk);
    if (net) {
        double m; int e;
        net->GetResult(m, e);
        if (out_mse) *out_mse = m;
        if (out_epochs) *out_epochs = e;
    }
}

DLL_EXPORT void DLL_CALL DN_StopTraining(int h) {
    std::unique_lock<std::mutex> lk;
    LSTMNet* net = FindAndLockNet(h, lk);
    if (net) net->StopTraining();
}

// --- Stavové funkce a error ---
DLL_EXPORT int DLL_CALL DN_GetLayerCount(int h) {
    std::unique_lock<std::mutex> lk;
    LSTMNet* net = FindAndLockNet(h, lk);
    return net ? net->GetLayerCount() : 0;
}
DLL_EXPORT double DLL_CALL DN_GetLayerWeightNorm(int h, int l) { return 0.0; }
DLL_EXPORT double DLL_CALL DN_GetGradNorm(int h) { return 0.0; }

DLL_EXPORT int DLL_CALL DN_SaveState(int h) {
    std::unique_lock<std::mutex> lk;
    LSTMNet* net = FindAndLockNet(h, lk);
    if (!net) return 0;
    if (!net->Save(net->serialize_buf)) return 0;
    return (int)net->serialize_buf.size() + 1;
}

DLL_EXPORT MQL_BOOL DLL_CALL DN_GetState(int h, char* buf, int max_len) {
    std::unique_lock<std::mutex> lk;
    LSTMNet* net = FindAndLockNet(h, lk);
    if (!net) return MQL_FALSE;
    const auto& s = net->serialize_buf;
    if (s.empty() || max_len < (int)s.size() + 1) return MQL_FALSE;
    memcpy(buf, s.c_str(), s.size());
    buf[s.size()] = 0;
    return MQL_TRUE;
}

DLL_EXPORT MQL_BOOL DLL_CALL DN_LoadState(int h, const char* buf) {
    std::unique_lock<std::mutex> lk;
    LSTMNet* net = FindAndLockNet(h, lk);
    return net ? net->Load(buf) : MQL_FALSE;
}

DLL_EXPORT void DLL_CALL DN_GetError(short* buf, int len) {
    if (!buf || len <= 0) return;
    std::lock_guard<std::mutex> lk(g_err_mtx);
    int copy_len = std::min((int)g_last_err_w.size(), len - 1);
    for (int i = 0; i < copy_len; i++)
        buf[i] = (short)g_last_err_w[(size_t)i];
    buf[copy_len] = 0;
}

BOOL APIENTRY DllMain(HMODULE, DWORD, LPVOID) { return TRUE; }