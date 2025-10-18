// Adapted from: https://github.com/vllm-project/vllm/blob/v0.8.2/csrc/custom_all_reduce.cu
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>
#include <torch/library.h>

#include "ar.cuh"

// Fake pointer type, must match fptr_t type in ops.h.
// We use this type alias to indicate when pointers are passed in as int64_t.
using fptr_t = int64_t;
static_assert(sizeof(void *) == sizeof(fptr_t));

int64_t meta_size() {
    return sizeof(my_ar::Signal);
}

void dispose(fptr_t _fa) {
    delete reinterpret_cast<my_ar::CustomAllReduce *>(_fa);  // NOLINT(performance-no-int-to-ptr)
}

fptr_t init_custom_ar(
    const std::vector<fptr_t> &fake_ipc_ptrs, torch::Tensor &rank_data, int64_t rank, bool full_nvlink) {
    int world_size = fake_ipc_ptrs.size();
    if (world_size > 8)
        throw std::invalid_argument("world size > 8 is not supported");
    if (world_size % 2 != 0)
        throw std::invalid_argument("Odd num gpus is not supported for now");
    if (rank < 0 || rank >= world_size)
        throw std::invalid_argument("invalid rank passed in");

    my_ar::Signal *ipc_ptrs[8];
    for (int i = 0; i < world_size; i++) {
        ipc_ptrs[i] = reinterpret_cast<my_ar::Signal *>(fake_ipc_ptrs[i]);  // NOLINT(performance-no-int-to-ptr)
    }
    return reinterpret_cast<fptr_t>(
        new my_ar::CustomAllReduce(ipc_ptrs, rank_data.data_ptr(), rank_data.numel(), rank, world_size, full_nvlink));
}

void register_buffer(fptr_t _fa, const std::vector<fptr_t> &fake_ipc_ptrs) {
    auto fa = reinterpret_cast<my_ar::CustomAllReduce *>(_fa);
    TORCH_CHECK(fake_ipc_ptrs.size() == fa->world_size_);
    void *ipc_ptrs[8];
    for (int i = 0; i < fake_ipc_ptrs.size(); i++) {
        ipc_ptrs[i] = reinterpret_cast<void *>(fake_ipc_ptrs[i]);
    }
    fa->register_shared_buffer(ipc_ptrs);
}

// Use vector<int64_t> to represent byte data for python binding compatibility.
std::tuple<std::vector<int64_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta(fptr_t _fa) {
    auto fa = reinterpret_cast<my_ar::CustomAllReduce *>(_fa);
    auto [handle, offsets] = fa->get_graph_buffer_ipc_meta();
    std::vector<int64_t> bytes(handle.begin(), handle.end());
    return std::make_tuple(bytes, offsets);
}

// Use vector<int64_t> to represent byte data for python binding compatibility.
void register_graph_buffers(
    fptr_t _fa, const std::vector<std::vector<int64_t>> &handles, const std::vector<std::vector<int64_t>> &offsets) {
    auto fa = reinterpret_cast<my_ar::CustomAllReduce *>(_fa);
    std::vector<std::string> bytes;
    bytes.reserve(handles.size());
    for (auto &handle : handles) {
        bytes.emplace_back(handle.begin(), handle.end());
    }
    bytes.reserve(handles.size());
    fa->register_graph_buffers(bytes, offsets);
}

bool _is_weak_contiguous(torch::Tensor &t) {
    return t.is_contiguous()
           || (t.storage().nbytes() - t.storage_offset() * t.element_size() == t.numel() * t.element_size());
}

/**
 * Performs an out-of-place allreduce and stores result in out.
 *
 * If _reg_buffer is null, assumes inp.data_ptr() is already IPC-registered.
 * Otherwise, _reg_buffer is assumed to be IPC-registered and inp is first
 * copied into _reg_buffer.
 */
void all_reduce(fptr_t _fa, torch::Tensor &inp, torch::Tensor &out, fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes) {
    auto ar_ptr = reinterpret_cast<my_ar::CustomAllReduce *>(_fa);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
    TORCH_CHECK_EQ(inp.numel(), out.numel());
    TORCH_CHECK(_is_weak_contiguous(out));
    TORCH_CHECK(_is_weak_contiguous(inp));

    auto reg_buffer = reinterpret_cast<void *>(_reg_buffer);

    if (reg_buffer) {
        // Eager mode reduce
        auto input_size = inp.numel() * inp.element_size();
        TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
        AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size, cudaMemcpyDeviceToDevice, stream));
    } else {
        // Graph capturing
        reg_buffer = inp.data_ptr();
    }

    switch (out.scalar_type()) {
    case at::ScalarType::Float:
        {
            ar_ptr->all_reduce<float>(
                stream, reinterpret_cast<float *>(reg_buffer), reinterpret_cast<float *>(out.data_ptr()), out.numel());
            break;
        }
    case at::ScalarType::Half:
        {
            ar_ptr->all_reduce<half>(
                stream, reinterpret_cast<half *>(reg_buffer), reinterpret_cast<half *>(out.data_ptr()), out.numel());
            break;
        }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16:
        {
            ar_ptr->all_reduce<nv_bfloat16>(
                stream, reinterpret_cast<nv_bfloat16 *>(reg_buffer), reinterpret_cast<nv_bfloat16 *>(out.data_ptr()),
                out.numel());
            break;
        }
#endif
    default:
        throw std::runtime_error("custom allreduce only supports float32, float16 and bfloat16");
    }
}

TORCH_LIBRARY_FRAGMENT(my_ar, m) {
    m.def("meta_size", &meta_size);
    m.def("dispose", &dispose);

    m.def(
        "init_custom_ar(int[] ipc_tensors, Tensor rank_data, "
        "int rank, bool full_nvlink) -> int");
    m.impl("init_custom_ar", torch::kCUDA, &init_custom_ar);

    m.def("register_buffer", &register_buffer);

    m.def(
        "all_reduce(int fa, Tensor inp, Tensor! out, int reg_buffer, "
        "int reg_buffer_sz_bytes) -> ()");
    m.impl("all_reduce", torch::kCUDA, &all_reduce);
}