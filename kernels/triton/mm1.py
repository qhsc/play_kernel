import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _schedule(tile_id, group_tiles, group_size, m_tiles):
    group_id = tile_id // group_tiles
    sub_id = tile_id % group_tiles

    group_m_size = min(group_size, m_tiles - group_id * group_size)
    return group_id * group_size + (sub_id % group_m_size), sub_id // group_m_size


@triton.jit
def _mm_nt_kernel(
    a_desc,
    b_desc,
    c_desc,
    m,
    n,
    k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    M_GROUP_SIZE: tl.constexpr,
    NUM_SMs: tl.constexpr,
):
    output_dtype = tl.bfloat16
    pid = tl.program_id(0)

    m_tiles = tl.cdiv(m, BLOCK_SIZE_M)
    n_tiles = tl.cdiv(n, BLOCK_SIZE_N)
    tiles = m_tiles * n_tiles
    group_tiles = M_GROUP_SIZE * n_tiles

    for tile_id in tl.range(pid, tiles, NUM_SMs, flatten=True):
        m_tile_id, n_tile_id = _schedule(
            tile_id=tile_id, group_tiles=group_tiles, group_size=M_GROUP_SIZE, m_tiles=m_tiles
        )
        m_offset = m_tile_id * BLOCK_SIZE_M
        n_offset = n_tile_id * BLOCK_SIZE_N

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in tl.range(0, k, BLOCK_SIZE_K):
            ai = a_desc.load([m_offset, ki])
            bi = b_desc.load([n_offset, ki])
            acc = tl.dot(input=ai, other=bi.T, acc=acc)

        c_desc.store([m_offset, n_offset], acc.to(output_dtype))


def mm_nt_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n, k_ = b.shape
    assert k == k_ and a.is_contiguous() and b.is_contiguous()
    c = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 256, 64
    NUM_SMs = torch.cuda.get_device_properties().multi_processor_count

    tiles = triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(n, BLOCK_SIZE_N)

    _mm_nt_kernel[(min(tiles, NUM_SMs),)](
        a_desc=TensorDescriptor.from_tensor(a, block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K]),
        b_desc=TensorDescriptor.from_tensor(b, block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K]),
        c_desc=TensorDescriptor.from_tensor(c, block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N]),
        m=m,
        n=n,
        k=k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        NUM_SMs=NUM_SMs,
        M_GROUP_SIZE=8,
        num_stages=3,
        num_warps=8,
    )
    return c


def sanity_check():
    m, n, k = 4096, 4096, 4096
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    c_ref = torch.einsum("mk,nk->mn", a, b)
    c_2 = mm_nt_triton(a, b)

    torch.testing.assert_close(c_ref, c_2)
    print(">>>>> Sanity check passed!")


def bench(m, n, k):
    import time

    def _bench(fn, name):
        ITERS = 2000
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        flops = 2 * m * n * k / 1e12 / (end - begin) * ITERS
        print(f"Bench {name}: {flops:.2f} TFLOP/s")

    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")

    torch_f = lambda: torch.einsum("mk,nk->mn", a, b)
    triton_f = lambda: mm_nt_triton(a, b)

    _bench(triton_f, "triton")
    _bench(torch_f, "torch")


if __name__ == "__main__":
    sanity_check()
    bench(4096, 4096, 4096)
