#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <array>

#include <torch/extension.h>

#define WARP_SIZE          32
#define FULL_MASK          0xffffffff
#define INT4(val)          (reinterpret_cast<int4 *>(&(val))[0])
#define FLOAT4(val)        (reinterpret_cast<float4 *>(&(val))[0])
#define FLOAT4_CONST(val)  (reinterpret_cast<const float4 *>(&(val))[0])
#define HALF2(val)         (reinterpret_cast<half2 *>(&(val))[0])
#define BFLOAT2(val)       (reinterpret_cast<__nv_bfloat162 *>(&(val))[0])
#define LDST128(val)       (reinterpret_cast<int4 *>(&(val))[0])
#define LDST128_CONST(val) (reinterpret_cast<const int4 *>(&(val))[0])

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (auto mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(FULL_MASK, val, mask);
    }
    return val;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_bf16_f32(__nv_bfloat16 val) {
    float v_f32 = __bfloat162float(val);
#pragma unroll
    for (auto mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        v_f32 += __shfl_xor_sync(FULL_MASK, v_f32, mask);
    }
    return v_f32;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_fp8_e4m3_f16(__nv_fp8_storage_t val) {
    half v_f16 = __nv_cvt_fp8_to_halfraw(val, __NV_E4M3);
#pragma unroll
    for (auto mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        v_f16 = __hadd(v_f16, __shfl_xor_sync(FULL_MASK, v_f16, mask));
    }
    return v_f16;
}

template <const int kNumThreads>
__global__ void reduce_sum_f32_kernel(const float *x, float *out, int N) {
    const int kNumWarps = kNumThreads / WARP_SIZE;
    static_assert(kNumWarps <= WARP_SIZE, "too many threads per block");

    auto tid = threadIdx.x;
    auto idx = blockIdx.x * kNumThreads + tid;
    auto warp = tid / WARP_SIZE;
    auto lane = tid % WARP_SIZE;

    __shared__ std::array<float, kNumWarps> smem;

    // warp-level reduce
    float val = int(idx) < N ? x[idx] : 0.0f;
    val = warp_reduce_sum_f32<WARP_SIZE>(val);
    if (lane == 0) {
        smem[warp] = val;
    }

    // CTA level reduce from shared memory using first warp
    __syncthreads();
    if (warp == 0) {
        val = lane < kNumWarps ? smem[lane] : 0.0f;
        val = warp_reduce_sum_f32<WARP_SIZE>(val);

        // write result to output
        if (tid == 0) {
            atomicAdd(out, val);
        }
    }
}

template <const int kNumThreads>
__global__ void reduce_sum_f32x4_kernel(const float *x, float *out, int N) {
    const int kNumWarps = kNumThreads / WARP_SIZE;
    static_assert(kNumWarps <= WARP_SIZE, "too many threads per block");

    auto tid = threadIdx.x;
    auto idx = (blockIdx.x * kNumThreads + tid) << 2;
    auto warp = tid / WARP_SIZE;
    auto lane = tid % WARP_SIZE;

    __shared__ std::array<float, kNumWarps> smem;

    // thread reduce 4 elements
    float4 f4{0.0f, 0.0f, 0.0f, 0.0f};
    if (int(idx) < N) {
        f4 = FLOAT4_CONST(x[idx]);
    }
    float val = f4.x + f4.y + f4.z + f4.w;

    // warp-level reduce
    val = warp_reduce_sum_f32<WARP_SIZE>(val);
    if (lane == 0) {
        smem[warp] = val;
    }

    // CTA level reduce from shared memory using first warp
    __syncthreads();
    if (warp == 0) {
        val = lane < kNumWarps ? smem[lane] : 0.0f;
        val = warp_reduce_sum_f32<WARP_SIZE>(val);

        // write result to output
        if (tid == 0) {
            atomicAdd(out, val);
        }
    }
}

template <const int kNumThreads>
__global__ void reduce_sum_bf16x8_f32_kernel(const __nv_bfloat16 *x, float *y, int N) {
    const int kNumWarps = kNumThreads / WARP_SIZE;
    static_assert(kNumWarps <= WARP_SIZE, "too many threads per block");

    auto tid = threadIdx.x;
    auto idx = (blockIdx.x * kNumThreads + tid) << 3;
    auto warp = tid / WARP_SIZE;
    auto lane = tid % WARP_SIZE;

    __shared__ std::array<float, kNumWarps> smem;

    std::array<__nv_bfloat16, 8> bf16_8;
    if (int(idx) < N) {
        LDST128(bf16_8) = LDST128_CONST(x[idx]);
    }

    // thread reduce 8 elements
    float val = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (i + int(idx) < N) {
            val += __bfloat162float(bf16_8[i]);
        }
    }

    // warp-level reduce
    val = warp_reduce_sum_f32<WARP_SIZE>(val);
    if (lane == 0) {
        smem[warp] = val;
    }

    // CTA level reduce from shared memory using first warp
    __syncthreads();
    if (warp == 0) {
        val = lane < kNumWarps ? smem[lane] : 0.0f;
        val = warp_reduce_sum_f32<WARP_SIZE>(val);

        if (tid == 0) {
            atomicAdd(y, val);
        }
    }
}

torch::Tensor reduce_sum_f32(const torch::Tensor &x) {
    const int N = x.numel();
    const int kNumThreads = 1024;
    const int kNumBlocks = (N + kNumThreads - 1) / kNumThreads;

    torch::Tensor out = torch::zeros({1}, x.options());
    reduce_sum_f32_kernel<kNumThreads><<<kNumBlocks, kNumThreads>>>(x.data_ptr<float>(), out.data_ptr<float>(), N);
    return out;
}

torch::Tensor reduce_sum_f32x4(const torch::Tensor &x) {
    const int N = x.numel();
    const int kNumThreads = 1024;
    const int kNumBlocks = (N + kNumThreads - 1) / kNumThreads / 4;

    torch::Tensor out = torch::zeros({1}, x.options());
    reduce_sum_f32x4_kernel<kNumThreads><<<kNumBlocks, kNumThreads>>>(x.data_ptr<float>(), out.data_ptr<float>(), N);
    return out;
}

torch::Tensor reduce_sum_bf16x8_f32(const torch::Tensor &x) {
    const int N = x.numel();
    const int kNumThreads = 1024;
    const int kNumBlocks = (N + kNumThreads - 1) / kNumThreads / 8;

    auto opt = x.options();
    torch::Tensor out = torch::zeros({1}, opt.dtype(torch::kFloat32));
    reduce_sum_bf16x8_f32_kernel<kNumThreads>
        <<<kNumBlocks, kNumThreads>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), out.data_ptr<float>(), N);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_sum_f32", &reduce_sum_f32, "Reduce Sum F32 (CUDA)");
    m.def("reduce_sum_f32x4", &reduce_sum_f32x4, "Reduce Sum F32x4 (CUDA)");
    m.def("reduce_sum_bf16x8_f32", &reduce_sum_bf16x8_f32, "Reduce Sum BF16x8 F32 (CUDA)");
}
