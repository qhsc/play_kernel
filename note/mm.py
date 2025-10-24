import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


# BE CAREFUL WITH THIS FUNCTION
@triton.jit
def _schd(tile_id, m_tiles, m_group_size, group_tiles):
    group_id = tile_id // group_tiles
    sub_group_id = tile_id % group_tiles

    start_m = group_id * m_group_size
    cur_group_size = min(m_group_size, m_tiles - start_m)
    return start_m + (sub_group_id % cur_group_size), sub_group_id // cur_group_size


@triton.jit
def _mm_kernel_nt(
    a_desc,
    b_desc,
    c_desc,
    m,
    n,
    k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_CTAs,
    M_GROUP_SIZE,
):
    m_tiles, n_tiles = tl.cdiv(m, BLOCK_SIZE_M), tl.cdiv(n, BLOCK_SIZE_N)
    tiles = m_tiles * n_tiles
    group_tiles = M_GROUP_SIZE * n_tiles

    for tile_id in tl.range(tl.program_id(0), tiles, NUM_CTAs):
        m_tile_idx, n_tile_idx = _schd(tile_id, m_tiles, M_GROUP_SIZE, group_tiles)
        m_idx = m_tile_idx * BLOCK_SIZE_M
        n_idx = n_tile_idx * BLOCK_SIZE_N

        c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32)
        for k_idx in tl.range(0, k, BLOCK_SIZE_K):
            a = a_desc.load([m_idx, k_idx])
            b = b_desc.load([n_idx, k_idx])
            c = tl.dot(input=a, other=b.T, acc=c)

        c_desc.store([m_idx, n_idx], c.to(tl.bfloat16))


def mm_nt_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n, k_ = b.shape

    assert k == k_
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 256, 64

    num_sm = torch.cuda.get_device_properties().multi_processor_count
    tiles = triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(n, BLOCK_SIZE_N)

    _mm_kernel_nt[(min(num_sm, tiles),)](
        TensorDescriptor.from_tensor(a, block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K]),
        TensorDescriptor.from_tensor(b, block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K]),
        TensorDescriptor.from_tensor(c, block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N]),
        m=m,
        n=n,
        k=k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        NUM_CTAs=num_sm,
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


if __name__ == "__main__":
    sanity_check()
