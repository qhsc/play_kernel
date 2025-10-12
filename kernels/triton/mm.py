import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import time


@triton.jit
def _schedule_tile(tile_id, group_tiles, group_size_m, m_tiles):
    group_id = tile_id // group_tiles
    sub_id = tile_id % group_tiles

    group_start_m = group_id * group_size_m
    group_size_m = min(group_size_m, m_tiles - group_start_m)
    return group_start_m + (sub_id % group_size_m), sub_id // group_size_m


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
    NUM_SM: tl.constexpr,
):
    out_dtype = tl.bfloat16

    pid = tl.program_id(axis=0)
    m_tiles, n_tiles = tl.cdiv(m, BLOCK_SIZE_M), tl.cdiv(n, BLOCK_SIZE_N)
    group_tiles = M_GROUP_SIZE * n_tiles
    mn_tiles = m_tiles * n_tiles

    for tile_id in tl.range(pid, mn_tiles, NUM_SM, flatten=True):
        m_tile_idx, n_tile_idx = _schedule_tile(
            tile_id=tile_id, group_tiles=group_tiles, group_size_m=M_GROUP_SIZE, m_tiles=m_tiles
        )
        off_m = m_tile_idx * BLOCK_SIZE_M
        off_n = n_tile_idx * BLOCK_SIZE_N

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in tl.range(0, k, BLOCK_SIZE_K):
            a_tile = a_desc.load([off_m, ki])
            b_tile = b_desc.load([off_n, ki])
            acc = tl.dot(input=a_tile, other=b_tile.T, acc=acc)

        c_desc.store([off_m, off_n], acc.to(out_dtype))


def mm_nt(a: torch.Tensor, b: torch.Tensor):
    m, k = a.shape
    n, k_ = b.shape
    assert k == k_, "Incompatible dimensions for NT Kernel"

    c = torch.empty((m, n), device=a.device, dtype=a.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
    M_GROUP_SIZE = 8

    num_sm = torch.cuda.get_device_properties().multi_processor_count
    num_tiles = triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N)

    _mm_nt_kernel[(min(num_tiles, num_sm),)](
        a_desc=TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K]),
        b_desc=TensorDescriptor.from_tensor(b, [BLOCK_N, BLOCK_K]),
        c_desc=TensorDescriptor.from_tensor(c, [BLOCK_M, BLOCK_N]),
        m=m,
        n=n,
        k=k,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        M_GROUP_SIZE=M_GROUP_SIZE,
        NUM_SM=num_sm,
        num_warps=8,
        num_stages=3,
    )
    return c


def sanity_check():
    m, n, k = 4096, 4096, 4096
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    c = mm_nt(a, b)
    c_ref = torch.einsum("mk,nk->mn", a, b)
    print(f"Sanity check {torch.allclose(c, c_ref)=}")


def bench(m, n, k):
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)

    tflops = 2 * m * n * k * 1e-12
    bytes = (m * k + n * k + m * n) * 2
    ITERS = 2000

    def bench_(fn, name):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        t = (end - begin) / ITERS
        print(f"[{name}] TFLOPS: {tflops / t:.2f}, GB/s: {bytes / t / 1e9:.2f}")

    bench_(lambda: mm_nt(a, b), "triton mm_nt")

    # bench_(lambda: torch.einsum("mk,nk->mn", a, b), "torch.einsum")


sanity_check()
bench(4096, 4096, 4096)
bench(4096, 4096, 4096)
