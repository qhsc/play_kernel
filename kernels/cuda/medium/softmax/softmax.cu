#include <cuda_runtime.h>
#include <array>
#include <cassert>
#include <cfloat>

#include <torch/extension.h>

#define WARP_SIZE              32
#define FULL_MASK              0xffffffff
#define LDST128BITS(val)       (reinterpret_cast<int4 *>(&(val))[0])
#define LDST128BITS_CONST(val) (reinterpret_cast<const int4 *>(&(val))[0])

class WarpReduceMax {
   public:
    __device__ __forceinline__ static float call(float val) {
#pragma unroll
        for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
            val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val, mask));
        }
        return val;
    }
    __device__ __forceinline__ static float default_value() {
        return -FLT_MAX;
    }
};

class WarpReduceSum {
   public:
    __device__ __forceinline__ static float call(float val) {
#pragma unroll
        for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
            val += __shfl_xor_sync(FULL_MASK, val, mask);
        }
        return val;
    }
    __device__ __forceinline__ static float default_value() {
        return 0.0f;
    }
};

template <const int kNumThreads, typename WarpReduceOp>
__device__ float cta_reduce(float val) {
    static_assert(kNumThreads <= WARP_SIZE * WARP_SIZE, "too many threads per block");
    static_assert(kNumThreads % WARP_SIZE == 0, "kNumThreads must be divisible by WARP_SIZE");

    const auto kNumWarps = kNumThreads / WARP_SIZE;
    const auto warp = threadIdx.x / WARP_SIZE;
    const auto lane = threadIdx.x % WARP_SIZE;
    const auto tid = threadIdx.x;

    // warp-level reduce
    val = WarpReduceOp::call(val);

    // store to shared memory
    __shared__ std::array<float, kNumWarps> smem;
    if (lane == 0) {
        smem[warp] = val;
    }

    // CTA level reduce from shared memory, all warps will do reduce
    __syncthreads();
    val = WarpReduceOp::call(lane < kNumWarps ? smem[lane] : WarpReduceOp::default_value());
    // broadcast through warp
    val = __shfl_sync(FULL_MASK, val, 0, WARP_SIZE);
    return val;
}

template <const int kNumThreads, const int ElemsPerThread>
__global__ void safe_softmax_f32_kernel(float *out, const float *x, const unsigned M, const unsigned K) {
    const auto k_idx = threadIdx.x * ElemsPerThread;
    static_assert(ElemsPerThread % 4 == 0, "ElemsPerThread must be divisible by 4");

    for (unsigned m_idx = blockIdx.x; m_idx < M; m_idx += blockDim.x) {
        auto x_ptr = x + m_idx * K;
        auto out_ptr = out + m_idx * K;

        std::array<float, ElemsPerThread> x_elems;
        float local_max = -FLT_MAX;
        for (unsigned idx = 0; idx < ElemsPerThread and idx + k_idx < K; idx += 4) {
            LDST128BITS(x_elems[idx]) = LDST128BITS_CONST(x_ptr[idx + k_idx]);
            local_max = fmaxf(local_max, x_elems[idx]);
            local_max = fmaxf(local_max, x_elems[idx + 1]);
            local_max = fmaxf(local_max, x_elems[idx + 2]);
            local_max = fmaxf(local_max, x_elems[idx + 3]);
        }

        float max_val = cta_reduce<kNumThreads, WarpReduceMax>(local_max);
        float exp_sum_local = 0.0f;
        for (unsigned idx = 0; idx < ElemsPerThread and idx + k_idx < K; idx += 4) {
            x_elems[idx] = expf(x_elems[idx] - max_val);
            x_elems[idx + 1] = expf(x_elems[idx + 1] - max_val);
            x_elems[idx + 2] = expf(x_elems[idx + 2] - max_val);
            x_elems[idx + 3] = expf(x_elems[idx + 3] - max_val);
            exp_sum_local += x_elems[idx] + x_elems[idx + 1] + x_elems[idx + 2] + x_elems[idx + 3];
        }

        float exp_sum = cta_reduce<kNumThreads, WarpReduceSum>(exp_sum_local);
        float exp_sum_inv = 1.0f / exp_sum;
        for (unsigned idx = 0; idx < ElemsPerThread and idx + k_idx < K; idx += 4) {
            x_elems[idx] *= exp_sum_inv;
            x_elems[idx + 1] *= exp_sum_inv;
            x_elems[idx + 2] *= exp_sum_inv;
            x_elems[idx + 3] *= exp_sum_inv;
            LDST128BITS(out_ptr[idx + k_idx]) = LDST128BITS(x_elems[idx]);
        }
    }
}

torch::Tensor softmax_f32(const torch::Tensor &x) {
    const int M = x.size(0);
    const int K = x.size(1);
    assert(K % 4 == 0);

    torch::Tensor y = torch::empty_like(x);

    if (K / 4 <= 256) {
        constexpr int Threads = 256;
        constexpr int ElemsPerThread = 4;
        safe_softmax_f32_kernel<Threads, ElemsPerThread>
            <<<std::min(M, 1024), Threads>>>(y.data_ptr<float>(), x.data_ptr<float>(), M, K);
    } else if (K / 4 <= 512) {
        constexpr int Threads = 512;
        constexpr int ElemsPerThread = 4;
        safe_softmax_f32_kernel<Threads, ElemsPerThread>
            <<<std::min(M, 1024), Threads>>>(y.data_ptr<float>(), x.data_ptr<float>(), M, K);
    } else if (K / 4 <= 1024) {
        constexpr int Threads = 1024;
        constexpr int ElemsPerThread = 4;
        safe_softmax_f32_kernel<Threads, ElemsPerThread>
            <<<std::min(M, 1024), Threads>>>(y.data_ptr<float>(), x.data_ptr<float>(), M, K);
    } else if (K / 8 <= 1024) {
        constexpr int Threads = 1024;
        constexpr int ElemsPerThread = 8;
        safe_softmax_f32_kernel<Threads, ElemsPerThread>
            <<<std::min(M, 1024), Threads>>>(y.data_ptr<float>(), x.data_ptr<float>(), M, K);
    } else if (K / 12 <= 1024) {
        constexpr int Threads = 1024;
        constexpr int ElemsPerThread = 12;
        safe_softmax_f32_kernel<Threads, ElemsPerThread>
            <<<std::min(M, 1024), Threads>>>(y.data_ptr<float>(), x.data_ptr<float>(), M, K);
    } else if (K / 16 <= 1024) {
        constexpr int Threads = 1024;
        constexpr int ElemsPerThread = 16;
        safe_softmax_f32_kernel<Threads, ElemsPerThread>
            <<<std::min(M, 1024), Threads>>>(y.data_ptr<float>(), x.data_ptr<float>(), M, K);
    } else {
        throw std::runtime_error("K is bigger than 1024*16");
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_f32", &softmax_f32, "Softmax F32 (CUDA)");
}