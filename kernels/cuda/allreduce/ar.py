import torch
import torch.distributed as dist

import ctypes

from cuda_wrapper import CudaRULibrary
from typing import List


class ParallelContext:
    def __init__(self, rank: int, world_size: int):
        torch.cuda.set_device(rank)

        torch.distributed.init_process_group(
            rank=rank, init_method="tcp://localhost:19651", world_size=world_size, backend="nccl", device_id=rank
        )

        self.rank = rank
        self.world_size = world_size
        self.device_group = dist.new_group(backend="nccl")
        self.cpu_group = dist.new_group(backend="gloo")

        self.lib = CudaRULibrary()

    def destroy(self):
        torch.distributed.destroy_process_group()

    def create_shared_cuda_ipc_mem(self, size_bytes: int, group=None):
        if group is None:
            group = self.cpu_group

        rank = dist.get_rank(group)
        word_size = dist.get_world_size(group)

        pointer = self.lib.cudaMalloc(size_bytes)
        handle = self.lib.cudaIpcGetMemHandle(pointer)

        handle_bytes = ctypes.string_at(ctypes.addressof(handle), ctypes.sizeof(handle))
        input_tesnsor = torch.ByteTensor(list(handle_bytes))
        gathered_tensors = [torch.empty_like(input_tesnsor) for _ in range(word_size)]
        dist.all_gather(gathered_tensors, input_tesnsor, group=group)

        handles = []
        handle_type = type(handle)
        for tensor in gathered_tensors:
            bytes_data = bytes(tensor.cpu().tolist())
            handle_obj = handle_type()
            ctypes.memmove(ctypes.addressof(handle_obj), bytes_data, len(bytes_data))
            handles.append(handle_obj)

        pointers: List[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer.value)  # type:ignore
            else:
                try:
                    opened_ptr = self.lib.cudaIpcOpenMemHandle(h)
                    pointers.append(opened_ptr.value)  # type:ignore
                except Exception as e:
                    print(f"Rank {rank}: Failed to open IPC handle from rank {i}: {e}")
                    raise

        dist.barrier(group=group)
        return pointers

    def free_shared_buffer(self, pointers: List[int], group=None) -> None:
        if group is None:
            group = self.cpu_group
        rank = dist.get_rank(group=group)
        if pointers and len(pointers) > rank and pointers[rank] is not None:
            self.lib.cudaFree(ctypes.c_void_p(pointers[rank]))
        dist.barrier(group=group)
