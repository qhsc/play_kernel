import torch
import triton
import triton.language as tl
import time


@torch.compile
def torch_naive_softmax(x: torch.Tensor) -> torch.Tensor:
    # read K, write 1
    x_max = x.max(dim=-1, keepdim=True).values
    # read K+1, write K
    x_safe = x - x_max
    # read K, write K
    numerator = torch.exp(x_safe)
    # read K, write 1
    denominator = numerator.sum(dim=-1, keepdim=True)
    # read K+1, write K
    y = numerator / denominator

    # total read 5K, write 3K
    return y


@triton.jit
def _softmax_kernel(output_ptr, x_ptr, m, k, COLUMN_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_p = tl.num_programs(0)
    cols = tl.arange(0, COLUMN_SIZE)
    mask = cols < k
    for row_idx in tl.range(pid, m, num_p):
        row = tl.load(x_ptr + row_idx * k + cols, mask=mask, other=-float("inf"))
        row_max = tl.max(row)
        numerator = tl.exp(row - row_max)
        deminator = tl.sum(numerator)
        softmax_out = numerator / deminator

        tl.store(output_ptr + row_idx * k + cols, softmax_out, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    m, k = x.shape
    _softmax_kernel[min(m, 1024),](y, x, m, k, COLUMN_SIZE=triton.next_power_of_2(k))
    return y


def sanity_check():
    x = torch.randn(4096, 8448, device="cuda")
    y_triton = triton_softmax(x)
    y_torch = torch.softmax(x, axis=1)
    torch.testing.assert_close(y_triton, y_torch)
    y_naive = torch_naive_softmax(x)
    torch.testing.assert_close(y_naive, y_torch)
    print("Sanity check passed")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128 * i for i in range(2, 100, 4)],
        line_arg="provider",
        line_vals=["torch", "triton", "naive"],
        line_names=["Torch", "Triton", "Naive"],
        plot_name="Softmax Performance",
        args={"M": 4096},
    )
)
def benchmark(M: int, N: int, provider: str):
    x = torch.randn(M, N, device="cuda")
    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=1))
    elif provider == "triton":
        ms = triton.testing.do_bench(lambda: triton_softmax(x))
    else:
        ms = triton.testing.do_bench(lambda: torch_naive_softmax(x))
    gbps = 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps


if __name__ == "__main__":
    sanity_check()
    benchmark.run(print_data=True)
