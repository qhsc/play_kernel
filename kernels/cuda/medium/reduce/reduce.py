import time

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="reduce_lib",
    sources=["reduce.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++20"],
)


def run_benchmark(
    perf_func: callable,
    values: torch.Tensor,
    tag: str,
    warmup: int = 10,
    iters: int = 1000,
):
    for i in range(warmup):
        out = perf_func(values)  # warmup
    torch.cuda.synchronize()
    start = time.time()
    for i in range(iters):
        out = perf_func(values)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.item()
    if tag.startswith("i8"):
        print(f"{out_info:>25}: {out_val:<15}, time:{mean_time:.8f}ms")
    else:
        print(f"{out_info:>25}: {out_val:<15.8f}, time:{mean_time:.8f}ms")
    return out, mean_time


Ss = [1024, 2048, 4096]
Ks = [1024, 2048, 4096]
SKs = [(S, K) for S in Ss for K in Ks]

for S, K in SKs:
    print("-" * 80)
    print(" " * 40 + f"S={S}, K={K}")
    values = torch.randn((S, K)).cuda().float()

    run_benchmark(torch.sum, values, "f32f32_th")
    run_benchmark(lib.reduce_sum_f32, values, "f32f32")
    run_benchmark(lib.reduce_sum_f32x4, values, "f32x4f32")

    v_bf16 = values.to(torch.bfloat16)
    v_bf16_2 = v_bf16.to(torch.float32)
    run_benchmark(torch.sum, v_bf16, "bf16bf16_th")
    run_benchmark(lib.reduce_sum_bf16x8_f32, v_bf16, "bf16x8f32")

    v_fp8 = values.to(torch.float8_e4m3fn)
    # run_benchmark(torch.sum, v_fp8, "fp8fp8_th")
    run_benchmark(lib.reduce_sum_fp8e4m3x16_f32, v_fp8, "fp8x16f32")
