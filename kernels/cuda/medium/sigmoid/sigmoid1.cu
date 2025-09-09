#include <cuda.h>
#include <iostream>
#include <torch/extension.h>
#include <torch/types.h>

#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)

#define FLOAT4(v) (reinterpret_cast<float4 *>(&(v))[0])

__global__ void sigmoid_f32x4_kernel(float *x, float *y, const size_t N)
{
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    auto x_idx = idx * 4; // 128 bits = float32 * 4
    if (x_idx >= N)
    {
        return;
    }

    float4 x4 = FLOAT4(x[x_idx]);
    float4 y4;

    x4.x = fminf(fmaxf(x4.x, MIN_EXP_F32), MAX_EXP_F32);
    x4.y = fminf(fmaxf(x4.y, MIN_EXP_F32), MAX_EXP_F32);
    x4.z = fminf(fmaxf(x4.z, MIN_EXP_F32), MAX_EXP_F32);
    x4.w = fminf(fmaxf(x4.w, MIN_EXP_F32), MAX_EXP_F32);

    y4.x = 1.0f / (1.0f + expf(-x4.x));
    y4.y = 1.0f / (1.0f + expf(-x4.y));
    y4.z = 1.0f / (1.0f + expf(-x4.z));
    y4.w = 1.0f / (1.0f + expf(-x4.w));

    FLOAT4(y[x_idx]) = y4;
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                       \
    if (((T).options().dtype() != (th_type)))                      \
    {                                                              \
        std::cout << "Tensor Info:" << (T).options() << std::endl; \
        throw std::runtime_error("values must be " #th_type);      \
    }

#define DEFINE_HOST_ENTRY_FUNC(dtype, type, packed_type, pack_size) \
    void sigmoid_##packed_type(torch::Tensor x, torch::Tensor y)    \
    {                                                               \
        CHECK_TORCH_TENSOR_DTYPE(x, dtype)                          \
        CHECK_TORCH_TENSOR_DTYPE(y, dtype)                          \
        size_t N = x.numel();                                       \
        dim3 block(512 / pack_size);                                \
        dim3 grid((N + 512 - 1) / 512);                             \
        sigmoid_##packed_type##_kernel<<<grid, block>>>(            \
            reinterpret_cast<type *>(x.data_ptr()),                 \
            reinterpret_cast<type *>(y.data_ptr()), N);             \
    };

DEFINE_HOST_ENTRY_FUNC(torch::kFloat32, float, f32x4, 4)

#define STRINGFY(str) #str
#define BINDING_TORCH_EXTENSION(func) \
    m.def(STRINGFY(func), &func, STRINGFY(func));

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    BINDING_TORCH_EXTENSION(sigmoid_f32x4)
}