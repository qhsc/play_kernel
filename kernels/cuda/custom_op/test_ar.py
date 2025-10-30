from re import L
import torch
import torch.distributed as dist
import nvtx
from ar import ParallelContext
import time
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
        max_size = 8192 * 1024 * 2
        # meta sync signal + eager mode reduce workspace
        meta_ptrs = ctx.create_shared_cuda_ipc_mem(custom_ops.meta_size() + max_size)

        rank_data = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        buffer_ptrs = ctx.create_shared_cuda_ipc_mem(max_size)

        custom_ptr = custom_ops.init_custom_ar(meta_ptrs, rank_data, rank, True)
        custom_ops.register_buffer(custom_ptr, buffer_ptrs)

        warm_up_loop = 100
        test_loop = 2000
        test_sizes = [
            6144*128* i for i in [1,2,3,4,5,6,7,8]
        ]
        
        ar_custom = lambda input, output: custom_ops.all_reduce(custom_ptr, input, output, buffer_ptrs[rank], max_size)
        ar_dist = lambda input: dist.all_reduce(input, group=ctx.device_group)
        
        rs_custom = lambda input, output: custom_ops.reduce_scatter(custom_ptr, input, output, buffer_ptrs[rank], max_size)
        rs_dist = lambda input, output: dist.reduce_scatter_tensor(output=output, input=input, group=ctx.device_group)
        
        ag_custom = lambda input, output: custom_ops.all_gather(custom_ptr, input, output, buffer_ptrs[rank], max_size)
        ag_dist = lambda input, output: dist.all_gather_into_tensor(output_tensor=output, input_tensor=input, group=ctx.device_group)
        
        def get_op(op_type: str):
            if op_type == "ar":
                return ar_custom, ar_dist
            elif op_type == "rs":
                return rs_custom, rs_dist
            elif op_type == "ag":
                return ag_custom, ag_dist
            else:
                raise ValueError(f"Invalid op type: {op_type}")
        
        for op in ["ar", "rs", "ag"]:
            if rank == 0:
                print(f"========== {op=} ==========")
            for sz in test_sizes:
                for dtype in [torch.bfloat16]:
                    input_sz = sz if op != "ag" else sz // world_size
                    input_tensor = torch.randint(1, 16, (input_sz,), dtype=dtype, device="cuda")
                    input_backup = input_tensor.clone()
                    
                    output_sz = sz if op != "rs" else sz // world_size
                    output_tensor = torch.empty((output_sz,), dtype=dtype, device="cuda")
                    output_backup = torch.empty_like(output_tensor) if op != "ar" else input_backup
                    
                    op_custom, op_dist = get_op(op)
                    for i in range(warm_up_loop):
                        op_custom(input_tensor, output_tensor)
                        if op == "ar":
                            op_dist(output_backup)
                        else:
                            op_dist(input_backup, output_backup)
                        if i==0:
                            torch.testing.assert_close(output_tensor, output_backup)
                    # if rank == 0:
                    #     print(f"Passed {op=} {sz=} check!!!")
                    
                    
                    with nvtx.annotate(f"{op=} bs={sz//6144}"):
                        torch.cuda.synchronize()
                        dist.barrier(group=ctx.device_group)
                        start_time = time.perf_counter()
                        for _ in range(test_loop):
                            op_custom(input_tensor, output_tensor)
                        torch.cuda.synchronize()
                        custom_time = (time.perf_counter() - start_time) / test_loop
                        
                        torch.cuda.synchronize()
                        time.sleep(1)
                        dist.barrier(group=ctx.device_group)
                        start_time = time.perf_counter()
                        for _ in range(test_loop):
                            if op == "ar":
                                op_dist(input_backup)
                            else:
                                op_dist(input_backup, output_backup)
                        torch.cuda.synchronize()
                        dist_time = (time.perf_counter() - start_time) / test_loop
                        
                        data_size = max(input_sz, output_sz) * dtype.itemsize / 1e6 
                        comm_data_gb = max(input_sz, output_sz) * dtype.itemsize / world_size * (world_size-1) / 1e9 
                        if op == "ar":
                            comm_data_gb *= 2
                            
                        comm_bw_custom = comm_data_gb / custom_time
                        comm_bw_dist = comm_data_gb / dist_time
                        if rank == 0:
                            print(f"bs={sz//6144} data_size={data_size:.2f} MB. custom bw: {comm_bw_custom:.2f} GB/s, {custom_time*1e6:.2f} us, dist bw: {comm_bw_dist:.2f} GB/s, {dist_time*1e6:.2f} us, time_fraction: {comm_bw_dist/comm_bw_custom:.2f}")

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
    arg.add_argument("--world-size", type=int, default=2)
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
