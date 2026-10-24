// Included by kernel.cu after the recurrent layer definitions.
__global__ void kPointwise(int n, int width, float* y, const float* bias, int activation) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = y[i] + bias[i % width];
    y[i] = activation == 1 ? tanhf(v) : activation == 2 ? fmaxf(0.0f, v) :
        activation == 3 ? d_sigmoid(v) : v;
}
__global__ void kPointwiseGrad(int n, float* gradient, const float* output, int activation) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float y = output[i];
    gradient[i] *= activation == 1 ? 1.0f-y*y : activation == 2 ? (y > 0 ? 1.0f : 0.0f) :
        activation == 3 ? y*(1.0f-y) : 1.0f;
}
__global__ void kGraphSlice(int columns, int width, int merged, int offset,
    const float* source, float* destination, bool scatter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= columns * width) return;
    int j = (i / width) * merged + offset + i % width;
    if (scatter) destination[i] += source[j];
    else destination[j] = source[i];
}

struct PointwiseLayer final : RecurrentLayer {
    int activation;
    bool identity;
    explicit PointwiseLayer(int act = 0, bool pass = false) : activation(act), identity(pass) {}
    int GetInputSize() const override { return input_size; }
    int GetHiddenSize() const override { return hidden_size; }
    float GetDropoutRate() const override { return dropout_rate; }
    SequenceLayerType GetType() const override { return static_cast<SequenceLayerType>(identity ? 4 : 3); }
    int GetGateDim() const override { return identity ? 0 : hidden_size; }
    void FreeAll() override {
        W.free(); b.free(); mW.free(); vW.free(); mb.free(); vb.free();
        dW.free(); db.free(); W_best.free(); b_best.free(); dx_buf.free();
        hx_cache.clear(); h_cache.clear(); h_drop_cache.clear();
        dropout_mask.clear(); dropout_mask_valid.clear(); dropout_rand.free();
        is_valid = has_snapshot = false;
    }
    MQL_BOOL Init(int in, int out, float drop = 0) override {
        FreeAll();
        if (!dn::layer(in, out, drop) || in <= 0 || (identity && in != out)) return MQL_FALSE;
        input_size = in; hidden_size = out; dropout_rate = drop; use_dropout = drop > 0;
        if (!identity) {
            size_t n = (size_t)in * out;
            if (!W.alloc(n) || !b.alloc(out) || !b.zero() || !mW.alloc(n) || !mW.zero() ||
                !vW.alloc(n) || !vW.zero() || !mb.alloc(out) || !mb.zero() ||
                !vb.alloc(out) || !vb.zero()) return MQL_FALSE;
            std::mt19937 rng(std::random_device{}());
            std::normal_distribution<float> distribution(0, sqrtf(2.0f / (in+out)));
            std::vector<float> weights(n);
            for (float& v : weights) v = distribution(rng);
            CUDA_CHECK_RET(cudaMemcpy(W.ptr, weights.data(), n*sizeof(float), cudaMemcpyHostToDevice));
        }
        is_valid = true;
        return MQL_TRUE;
    }
    MQL_BOOL InitFromData(int in, int out, float drop = 0) override { return Init(in, out, drop); }
    MQL_BOOL Forward(cublasHandle_t blas, cudaStream_t stream, const float* x,
        int sl, int batch, bool training, curandGenerator_t generator) override {
        if (!is_valid) return MQL_FALSE;
        hx_cache.resize(sl); h_cache.resize(sl);
        float one = 1, zero = 0;
        for (int t=0; t<sl; ++t) {
            size_t nx = (size_t)input_size*batch, ny = (size_t)hidden_size*batch;
            if (!hx_cache[t].alloc(nx) || !h_cache[t].alloc(ny)) return MQL_FALSE;
            CUDA_CHECK_RET(cudaMemcpyAsync(hx_cache[t].ptr, x+t*nx, nx*sizeof(float), cudaMemcpyDeviceToDevice, stream));
            if (identity) {
                CUDA_CHECK_RET(cudaMemcpyAsync(h_cache[t].ptr, x+t*nx, nx*sizeof(float), cudaMemcpyDeviceToDevice, stream));
            } else {
                CUBLAS_CHECK_RET(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, hidden_size, batch, input_size,
                    &one, W.ptr, input_size, hx_cache[t].ptr, input_size, &zero, h_cache[t].ptr, hidden_size));
                kPointwise<<<toU32((ny+255)/256),256,0,stream>>>((int)ny, hidden_size, h_cache[t].ptr, b.ptr, activation);
                CUDA_CHECK_KERNEL_RET();
            }
        }
        return ApplyOutputDropout(sl, batch, training, stream, generator);
    }
    MQL_BOOL Backward(cublasHandle_t blas, cudaStream_t stream, const float* last,
        const float* above, int sl, int batch, const float* ones) override {
        size_t nx = (size_t)input_size*batch, ny = (size_t)hidden_size*batch;
        if (!dx_buf.alloc(nx*sl)) return MQL_FALSE;
        if (!identity && (!dW.alloc(W.count) || !dW.zeroAsync(stream) ||
            !db.alloc(b.count) || !db.zeroAsync(stream))) return MQL_FALSE;
        GPUMemory<float> gradient;
        if (!gradient.alloc(ny)) return MQL_FALSE;
        float one = 1, zero = 0;
        for (int t=0; t<sl; ++t) {
            if (above) CUDA_CHECK_RET(cudaMemcpyAsync(gradient.ptr, above+t*ny, ny*sizeof(float), cudaMemcpyDeviceToDevice, stream));
            else if (!gradient.zeroAsync(stream)) return MQL_FALSE;
            if (last && t==sl-1) kAddInplace<<<toU32((ny+255)/256),256,0,stream>>>((int)ny,gradient.ptr,last);
            if (identity) {
                CUDA_CHECK_RET(cudaMemcpyAsync(dx_buf.ptr+t*nx,gradient.ptr,nx*sizeof(float),cudaMemcpyDeviceToDevice,stream));
            } else {
                kPointwiseGrad<<<toU32((ny+255)/256),256,0,stream>>>((int)ny,gradient.ptr,h_cache[t].ptr,activation);
                CUDA_CHECK_KERNEL_RET();
                CUBLAS_CHECK_RET(cublasSgemm(blas,CUBLAS_OP_N,CUBLAS_OP_T,input_size,hidden_size,batch,
                    &one,hx_cache[t].ptr,input_size,gradient.ptr,hidden_size,&one,dW.ptr,input_size));
                CUBLAS_CHECK_RET(cublasSgemv(blas,CUBLAS_OP_N,hidden_size,batch,&one,gradient.ptr,hidden_size,
                    ones,1,&one,db.ptr,1));
                CUBLAS_CHECK_RET(cublasSgemm(blas,CUBLAS_OP_N,CUBLAS_OP_N,input_size,batch,hidden_size,
                    &one,W.ptr,input_size,gradient.ptr,hidden_size,&zero,dx_buf.ptr+t*nx,input_size));
            }
        }
        CUDA_CHECK_KERNEL_RET();
        return MQL_TRUE;
    }
    MQL_BOOL Update(float lr,float c1,float c2,float wd,float clip,cudaStream_t stream) override {
        if (!identity) {
            kAdamW<<<toU32((W.count+255)/256),256,0,stream>>>((int)W.count,W.ptr,mW.ptr,vW.ptr,dW.ptr,lr,.9f,.999f,1e-8f,wd,c1,c2,clip);
            kAdamW<<<toU32((b.count+255)/256),256,0,stream>>>((int)b.count,b.ptr,mb.ptr,vb.ptr,db.ptr,lr,.9f,.999f,1e-8f,0,c1,c2,clip);
            CUDA_CHECK_KERNEL_RET();
        }
        return MQL_TRUE;
    }
    MQL_BOOL SaveBest() override {
        if (!is_valid) return MQL_FALSE;
        if (!identity) {
            if (!W_best.alloc(W.count) || !b_best.alloc(b.count)) return MQL_FALSE;
            CUDA_CHECK_RET(cudaMemcpy(W_best.ptr,W.ptr,W.bytes(),cudaMemcpyDeviceToDevice));
            CUDA_CHECK_RET(cudaMemcpy(b_best.ptr,b.ptr,b.bytes(),cudaMemcpyDeviceToDevice));
        }
        has_snapshot = true;
        return MQL_TRUE;
    }
    MQL_BOOL RestoreBest() override {
        if (!has_snapshot) return MQL_FALSE;
        if (!identity) {
            CUDA_CHECK_RET(cudaMemcpy(W.ptr,W_best.ptr,W.bytes(),cudaMemcpyDeviceToDevice));
            CUDA_CHECK_RET(cudaMemcpy(b.ptr,b_best.ptr,b.bytes(),cudaMemcpyDeviceToDevice));
        }
        return MQL_TRUE;
    }
};
