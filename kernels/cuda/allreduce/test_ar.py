import torch
import torch.distributed as dist

from ar import ParallelContext

import os
import torch

so_path = "build/lib/libmy_ar.so"
if os.path.exists(so_path):
    torch.ops.load_library(so_path)
    print(f"Loaded custom allreduce backend: {so_path}")
    custom_ops = torch.ops.my_ar
else:
    print(f"WARNING: {so_path} does not exist. Make sure to build the shared library before running this test.")


def test_cuda_ipc(rank: int, world_size: int):
    ctx = ParallelContext(rank, world_size)
    a = torch.ones((1,), device="cuda", dtype=torch.float32) + rank
    dist.all_reduce(tensor=a, group=ctx.device_group)
    print(f"rank {rank} a: {a}")

    meta_ptrs = None
    buffer_ptrs = None
    custom_ptr = None
    try:
        # eager mode reduce max data size in bytes
        max_size = 8192 * 1024
        # meta sync signal + eager mode reduce workspace
        meta_ptrs = ctx.create_shared_cuda_ipc_mem(custom_ops.meta_size() + max_size)

        rank_data = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        buffer_ptrs = ctx.create_shared_cuda_ipc_mem(max_size)

        custom_ptr = custom_ops.init_custom_ar(meta_ptrs, rank_data, rank, True)
        custom_ops.register_buffer(custom_ptr, buffer_ptrs)

        test_loop = 10
        test_sizes = [
            512,
            2560,
            4096,
            5120,
            7680,
            32768,
            262144,
            524288,
            1048576,
            2097152,
        ]
        for sz in test_sizes:
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                for _ in range(test_loop):
                    inp1 = torch.randint(1, 16, (sz,), dtype=dtype, device="cuda")
                    inp1_ref = inp1.clone()
                    out1 = torch.empty_like(inp1)

                    custom_ops.all_reduce(custom_ptr, inp1, out1, buffer_ptrs[rank], max_size)

                    dist.all_reduce(inp1_ref, group=ctx.device_group)
                    torch.testing.assert_close(out1, inp1_ref)
            if rank == 0:
                print(f"Passed {sz=} check!!!")

    finally:
        dist.barrier(group=ctx.device_group)
        if custom_ptr is not None:
            custom_ops.dispose(custom_ptr)
        if buffer_ptrs:
            ctx.free_shared_buffer(buffer_ptrs)
        if meta_ptrs:
            ctx.free_shared_buffer(meta_ptrs)

        ctx.destroy()


if __name__ == "__main__":
    import argparse

    arg = argparse.ArgumentParser()
    arg.add_argument("--world_size", type=int, default=2)
    args = arg.parse_args()

    num_device = torch.cuda.device_count()
    assert (
        args.world_size <= num_device
    ), f"world_size {args.world_size} must be less than or equal to num_device {num_device}"

    import multiprocessing as mp

    for rank in range(args.world_size):
        mp.Process(target=test_cuda_ipc, args=(rank, args.world_size)).start()
    for p in mp.active_children():
        p.join()
