import torch
from torch.utils.cpp_extension import load
import triton

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="softmax_lib",
    sources=["softmax.cu"],
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


def sanity_check():
    x = torch.randn(4096, 8448, device="cuda")
    y_lib = lib.softmax_f32(x)
    y_torch = torch.softmax(x, axis=1)
    torch.testing.assert_close(y_lib, y_torch)
    print("Sanity check passed")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128 * i for i in range(2, 100, 4)],
        line_arg="provider",
        line_vals=["torch", "cuda"],
        line_names=["Torch", "cuda"],
        plot_name="Softmax Performance",
        args={"M": 4096},
    )
)
def benchmark(M: int, N: int, provider: str):
    x = torch.randn(M, N, device="cuda")
    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=1))
    else:
        ms = triton.testing.do_bench(lambda: lib.softmax_f32(x))
    gbps = 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps


if __name__ == "__main__":
    sanity_check()
    benchmark.run(print_data=True)
