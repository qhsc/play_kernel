#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdlib>
#include <cstring>

#define DINLINE __device__ __forceinline__

#define CHECK_CUDA(cmd)                                                                   \
    do {                                                                                  \
        cudaError_t err = cmd;                                                            \
        if (err != cudaSuccess) {                                                         \
            printf("CUDA error %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    } while (0)

namespace my_ar {

constexpr int kMaxBlocks = 36;
// Counter may overflow, but it's fine since unsigned int overflow is
// well-defined behavior.
using FlagType = uint32_t;
struct Signal {
    alignas(128) FlagType self_counter[kMaxBlocks][8];
    // Two sets of peer counters are needed for two syncs. The reason is that
    // it's possible for peer GPU block to arrive at the second sync point while
    // the current GPU block haven't passed the first sync point. Thus, peer GPU
    // may write counter+1 while current GPU is busy waiting for counter. We use
    // alternating counter array to avoid this possibility.
    alignas(128) FlagType peer_counter[2][kMaxBlocks][8];
};

struct __align__(16) RankData {
    const void *__restrict__ ptrs[8];
};

struct __align__(16) RankSignals {
    Signal *signals[8];
};

// like std::array, but aligned
template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
    T data[sz];
    using type = T;
    static constexpr int size = sz;
};

// use packed type to maximize memory efficiency
// goal: generate ld.128 and st.128 instructions
template <typename T>
struct packed_t {
    // the (P)acked type for load/store
    using P = array_t<T, 16 / sizeof(T)>;
    // the (A)ccumulator type for reduction
    using A = array_t<float, 16 / sizeof(T)>;
};

#define DINLINE __device__ __forceinline__

// scalar cast functions
DINLINE float upcast_s(half val) {
    return __half2float(val);
}

template <typename T>
DINLINE T downcast_s(float val);
template <>
DINLINE half downcast_s(float val) {
    return __float2half(val);
}

// scalar add functions
// for some reason when compiling with Pytorch, the + operator for half and
// bfloat is disabled so we call the intrinsics directly
DINLINE half &assign_add(half &a, half b) {
    a = __hadd(a, b);
    return a;
}
DINLINE float &assign_add(float &a, float b) {
    return a += b;
}

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
DINLINE float upcast_s(nv_bfloat16 val) {
    return __bfloat162float(val);
}
template <>
DINLINE nv_bfloat16 downcast_s(float val) {
    return __float2bfloat16(val);
}
DINLINE nv_bfloat16 &assign_add(nv_bfloat16 &a, nv_bfloat16 b) {
    a = __hadd(a, b);
    return a;
}
#endif

template <typename T, int N>
DINLINE array_t<T, N> &packed_assign_add(array_t<T, N> &a, array_t<T, N> b) {
#pragma unroll
    for (int i = 0; i < N; i++) {
        assign_add(a.data[i], b.data[i]);
    }
    return a;
}

template <typename T, int N>
DINLINE array_t<float, N> upcast(array_t<T, N> val) {
    if constexpr (std::is_same<T, float>::value) {
        return val;
    } else {
        array_t<float, N> out;
#pragma unroll
        for (int i = 0; i < N; i++) {
            out.data[i] = upcast_s(val.data[i]);
        }
        return out;
    }
}

template <typename O>
DINLINE O downcast(array_t<float, O::size> val) {
    if constexpr (std::is_same<typename O::type, float>::value) {
        return val;
    } else {
        O out;
#pragma unroll
        for (int i = 0; i < O::size; i++) {
            out.data[i] = downcast_s<typename O::type>(val.data[i]);
        }
        return out;
    }
}

// Load data by pack, and reduce multi GPU
template <typename P, int N_GPU, typename A>
DINLINE P packed_reduce(const P *ptrs[], int idx) {
    A tmp = upcast(ptrs[0][idx]);
#pragma unroll
    for (int i = 1; i < N_GPU; i++) {
        packed_assign_add(tmp, upcast(ptrs[i][idx]));
    }
    return downcast<P>(tmp);
}

static DINLINE void st_flag_release(FlagType *flag_addr, FlagType flag) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#else
    asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#endif
}

static DINLINE FlagType ld_flag_acquire(FlagType *flag_addr) {
    FlagType flag;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
#else
    asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;" : "=r"(flag) : "l"(flag_addr));
#endif
    return flag;
}

static DINLINE void st_flag_volatile(FlagType *flag_addr, FlagType flag) {
    asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

static DINLINE FlagType ld_flag_volatile(FlagType *flag_addr) {
    FlagType flag;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
    return flag;
}

// is_start: whether this is the very first synchronization barrier.
// need_fence: whether a memory fence is needed. If true, a release-acquire
// semantic is used to enforce memory access order before and after this
// barrier.
template <int N_GPU, bool IS_START, bool NEED_FENCE>
DINLINE void multi_gpu_barrier(const RankSignals &sync_sg, Signal *self_sg, int my_rank) {
    const auto peer_rank = threadIdx.x;
    const auto bid = blockIdx.x;

    if constexpr (!IS_START)
        __syncthreads();
    static_assert(!(IS_START && NEED_FENCE));  // Start barrier shouldn't need fence.

    // Each thread will first notify one peer rank that I have arrive,
    // and then check if that peer also give me a notify signal.
    if (peer_rank < N_GPU) {
        auto val = self_sg->self_counter[bid][my_rank] += 1;
        // write value to sync signal to mark I have arrive the sync point
        // and I have to check all other peer have arrive the sync point.
        FlagType *notify_peer_ptr = &sync_sg.signals[peer_rank]->peer_counter[val % 2][bid][my_rank];
        FlagType *check_notify_ptr = &self_sg->peer_counter[val % 2][bid][peer_rank];

        if constexpr (NEED_FENCE) {
            st_flag_release(notify_peer_ptr, val);
            while (ld_flag_acquire(check_notify_ptr) != val)
                ;
        } else {
            st_flag_volatile(notify_peer_ptr, val);
            while (ld_flag_volatile(check_notify_ptr) != val)
                ;
        }
    }

    if constexpr (IS_START or NEED_FENCE)
        __syncthreads();
}

// max 512 threads per block
// min 1 CTA per sm
template <typename T, int N_GPU>
__global__ void __launch_bounds__(512, 1) cross_device_reduce_1stage(
    RankData *data, RankSignals sg, Signal *self_sg, T *__restrict__ result, int rank, int size_packed) {
    using P = typename packed_t<T>::P;
    using A = typename packed_t<T>::A;

    multi_gpu_barrier<N_GPU, true, false>(sg, self_sg, rank);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size_packed; idx += gridDim.x * blockDim.x) {
        (reinterpret_cast<P *>(result))[idx] = packed_reduce<P, N_GPU, A>((const P **)&data->ptrs[0], idx);
    }
    multi_gpu_barrier<N_GPU, false, false>(sg, self_sg, rank);
}

template <typename P>
DINLINE P *get_tmp_buf(Signal *sg) {
    return reinterpret_cast<P *>(sg + 1);
}

template <typename T, int N_GPU>
__global__ void __launch_bounds__(512, 1) cross_device_reduce_2stage(
    RankData *data, RankSignals sg, Signal *self_sg, T *__restrict__ result, int rank, int size_packed) {
    using P = typename packed_t<T>::P;
    using A = typename packed_t<T>::A;

    const auto tid = threadIdx.x + blockDim.x * blockIdx.x;
    const auto num_threads = gridDim.x * blockDim.x;

    const auto chunk_size = size_packed / N_GPU;
    const auto last_chunk_size = size_packed - chunk_size * N_GPU + chunk_size;

    const auto chunk_start = rank * chunk_size;
    const auto cur_chunk_size = rank + 1 == N_GPU ? last_chunk_size : chunk_size;
    const auto chunk_end = chunk_start + cur_chunk_size;

    P *reduce_buf[N_GPU];
#pragma unroll
    for (int i = 0; i < N_GPU; i++) {
        reduce_buf[i] = get_tmp_buf<P>(sg.signals[i]);
    }
    auto local_reduce_buf = reduce_buf[rank] - chunk_start;  // dummy offset

    // reduce scatter
    multi_gpu_barrier<N_GPU, true, false>(sg, self_sg, rank);
    for (int idx = tid + chunk_start; idx < chunk_end; idx += num_threads) {
        // Note: the reduce order between different is always from 0 to N_GPU-1
        local_reduce_buf[idx] = packed_reduce<P, N_GPU, A>((const P **)&data->ptrs[0], idx);
    }
    multi_gpu_barrier<N_GPU, false, true>(sg, self_sg, rank);

    // all gather
    int gather_idx = tid;
    for (; gather_idx < chunk_size; gather_idx += num_threads) {
        // gather from other rank
#pragma unroll
        for (int gpu_id = 0; gpu_id < N_GPU; gpu_id++) {
            (reinterpret_cast<P *>(result))[gpu_id * chunk_size + gather_idx] = reduce_buf[gpu_id][gather_idx];
        }
    }
    for (; gather_idx < last_chunk_size; gather_idx += num_threads) {
        (reinterpret_cast<P *>(result))[N_GPU * chunk_size + gather_idx] = reduce_buf[N_GPU][gather_idx];
    }
}

using IPC_KEY = std::array<uint8_t, sizeof(cudaIpcMemHandle_t)>;
static_assert(sizeof(IPC_KEY) == sizeof(cudaIpcMemHandle_t));
static_assert(alignof(IPC_KEY) == alignof(cudaIpcMemHandle_t));

class CustomAllReduce {
   public:
    int rank_;
    int world_size_;
    bool full_nvlink_;

    Signal *self_sg_;  // pointer to Signal store on device
    RankSignals sg_;   // pointers to Shared Signals store on device

    // Stores an map from a pointer to its peer pointters from all ranks.
    // key is device data ptr, value is all rank shared IPC ptr:
    // cudaMalloc -> GetIpcMemHandle -> AllGather -> IpcOpenMemHandle
    std::unordered_map<void *, RankData *> buffers_;

    // for cudaGraph work
    RankData *d_rank_data_base_, *d_rank_data_end_;
    std::vector<void *> graph_unreg_buffers_;
    // a map from IPC handles to opened IPC pointers
    std::map<IPC_KEY, char *> ipc_handles_;

    /**
   * Signals are an array of ipc-enabled buffers from all ranks.
   * For each of the buffer, the layout is as follows:
   * | -- sizeof(Signal) -- | ------ a few MB ----- |
   * The first section is for allreduce synchronization, and the second section
   * is for storing the intermediate results required by some allreduce algos.
   * rank_data is just some device workspace buffer to holding IPC pointers.
   *
   * Note: this class does not own any device memory. Any required buffers
   * are passed in from the constructor.
   */
    CustomAllReduce(
        Signal **signals, void *rank_data, size_t rank_data_sz, int rank, int world_size, bool full_nv_link = true)
        : rank_(rank)
        , world_size_(world_size)
        , full_nvlink_(full_nv_link)
        , self_sg_(signals[rank])
        , d_rank_data_base_(reinterpret_cast<RankData *>(rank_data))
        , d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData)) {
        for (int i = 0; i < world_size; i++) {
            sg_.signals[i] = signals[i];
        }
    }

    ~CustomAllReduce() {
        for (auto [_, ptr] : ipc_handles_) {
            CHECK_CUDA(cudaIpcCloseMemHandle(ptr));
        }
    }

    void check_rank_data_capacity(size_t num = 1) {
        if (d_rank_data_base_ + num > d_rank_data_end_)
            throw std::runtime_error(
                "Rank data buffer is overflowed by " + std::to_string(d_rank_data_base_ + num - d_rank_data_end_));
    }

    /**
   * Register already-shared IPC pointers. all rank data ptrs.
   * ptrs value will be copy to device workspace.
   */
    void register_shared_buffer(void **ptrs) {
        check_rank_data_capacity();
        RankData data;
        for (int i = 0; i < world_size_; i++) {
            data.ptrs[i] = ptrs[i];
        }
        auto d_data_ptr = d_rank_data_base_++;
        CHECK_CUDA(cudaMemcpy(d_data_ptr, (void *)&data, sizeof(data), cudaMemcpyHostToDevice));
        buffers_[ptrs[rank_]] = d_data_ptr;
    }

    // the opend handle will be closed when deconstructing
    char *open_ipc_handle(const void *ipc_handle) {
        auto [it, new_handle] = this->ipc_handles_.insert({reinterpret_cast<const IPC_KEY *>(ipc_handle)[0], nullptr});
        if (new_handle) {
            char *ipc_ptr;
            CHECK_CUDA(cudaIpcOpenMemHandle(
                (void **)&ipc_ptr, *((const cudaIpcMemHandle_t *)(ipc_handle)), cudaIpcMemLazyEnablePeerAccess));
            it->second = ipc_ptr;
        }
        return it->second;
    }

    // convert all graph buffer ptrs to shared IPC handle
    // note: the IPC handle must get from base ptr, thus a offset is also needed
    std::pair<std::string, std::vector<int64_t>> get_graph_buffer_ipc_meta() {
        int num_buffers = graph_unreg_buffers_.size();
        constexpr auto handle_sz = sizeof(cudaIpcMemHandle_t);

        std::string handle_str(handle_sz * num_buffers, static_cast<char>(0));
        std::vector<int64_t> offsets(num_buffers);

        for (int i = 0; i < num_buffers; i++) {
            auto ptr = graph_unreg_buffers_[i];

            // note: must share the base address of each allocation, or we get wrong
            // address
            void *base_ptr;
            if (cuPointerGetAttribute(&base_ptr, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, (CUdeviceptr)ptr)
                != CUDA_SUCCESS) {
                throw std::runtime_error("failed to get the pointer attr");
            }
            CHECK_CUDA(cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&handle_str[i * handle_sz], base_ptr));
            offsets.push_back(((char *)ptr) - ((char *)base_ptr));
        }
        return std::make_pair(handle_str, offsets);
    }

    // Note: when registering graph buffers, we intentionally choose to not
    // deduplicate the addresses. That means if the allocator reuses some
    // addresses, they will be registered again. This is to account for the remote
    // possibility of different allocation patterns between ranks. For example,
    // rank 1 may get the same input address for the second allreduce, but rank 2
    // got a different address. IPC handles have internal reference counting
    // mechanism so overhead should be small.
    void register_graph_buffers(
        const std::vector<std::string> &all_rank_handles, const std::vector<std::vector<int64_t>> &all_rank_offsets) {
        int num_buffers = graph_unreg_buffers_.size();
        check_rank_data_capacity(num_buffers);

        std::vector<RankData> rds(num_buffers);
        for (int buffer_idx = 0; buffer_idx < num_buffers; buffer_idx++) {
            void *self_ptr = graph_unreg_buffers_[buffer_idx];
            auto &rd = rds[buffer_idx];
            for (int rank = 0; rank < world_size_; rank++) {
                if (rank != rank_) {
                    char *handle = open_ipc_handle(&all_rank_handles[rank][buffer_idx * sizeof(cudaIpcMemHandle_t)]);
                    rd.ptrs[rank] = handle + all_rank_offsets[rank][buffer_idx];
                } else {
                    rd.ptrs[rank] = self_ptr;
                }
            }
        }

        CHECK_CUDA(cudaMemcpy(d_rank_data_base_, rds.data(), sizeof(RankData) * num_buffers, cudaMemcpyHostToDevice));
        d_rank_data_base_ += num_buffers;
        graph_unreg_buffers_.clear();
    }

    /**
   * Performs allreduce, assuming input has already been registered.
   *
   * Block and grid default configs are results after careful grid search. Using
   * 36 blocks give the best or close to the best runtime on the devices I
   * tried: A100, A10, A30, T4, V100. You'll notice that NCCL kernels also only
   * take a small amount of SMs. Not quite sure the underlying reason, but my
   * guess is that too many SMs will cause contention on NVLink bus.
   */
    template <typename T>
    void all_reduce(cudaStream_t stream, T *input, T *output, int size, int threads = 512, int block_limit = 36) {
        const int pack_size = packed_t<T>::P::size;
        if (size % pack_size != 0)
            throw std::runtime_error(
                "custom allreduce currently requires input length to be multiple of " + std::to_string(pack_size));

        if (block_limit > kMaxBlocks)
            throw std::runtime_error(
                "max supported block limit is " + std::to_string(kMaxBlocks) + ". Got " + std::to_string(block_limit));

        RankData *all_rank_data;
        cudaStreamCaptureStatus status;
        CHECK_CUDA(cudaStreamIsCapturing(stream, &status));
        if (cudaStreamCaptureStatusActive == status) {
            // When capturing, the real buffer is not registered
            // But this is enough for graph capturing for recoed kernel launch config and paramter address.
            // In this case, the parameter is always d_rand_data_base_ + off.
            // In later, we will fill the real data in this address, thus when replay the graph, the kernel can correct access the real data.
            all_rank_data = d_rank_data_base_ + graph_unreg_buffers_.size();
            graph_unreg_buffers_.push_back(input);
        } else {
            auto it = buffers_.find(input);
            if (it == buffers_.end()) {
                throw std::runtime_error(
                    "buffer address:" + std::to_string(reinterpret_cast<uint64_t>(input)) + " is not registered!");
            }
            all_rank_data = it->second;
        }

        const int packed_size = size / pack_size;
        const int size_bytes = size * sizeof(T);
        const int num_blocks = std::min(block_limit, (packed_size + threads - 1) / threads);

#define LAUNCH_KERNEL(N_GPU, NAME) \
    NAME<T, N_GPU><<<num_blocks, threads, 0, stream>>>(all_rank_data, sg_, self_sg_, output, rank_, packed_size)

#define REDUCE_CASE(N_GPU)                                                                                          \
    case N_GPU:                                                                                                     \
        if (world_size_ == 2) {                                                                                     \
            LAUNCH_KERNEL(N_GPU, cross_device_reduce_1stage);                                                       \
        } else if (full_nvlink_) {                                                                                  \
            if ((world_size_ == 4 and size_bytes < 512 * 1024) or (world_size_ == 8 and size_bytes < 256 * 1024)) { \
                LAUNCH_KERNEL(N_GPU, cross_device_reduce_1stage);                                                   \
            } else {                                                                                                \
                LAUNCH_KERNEL(N_GPU, cross_device_reduce_2stage);                                                   \
            }                                                                                                       \
        }                                                                                                           \
        break;

        switch (world_size_) {
            REDUCE_CASE(2)
            REDUCE_CASE(4)
            REDUCE_CASE(6)
            REDUCE_CASE(8)
        default:
            throw std::runtime_error(
                "custom allreduce only supports num gpus in (2,4,6,8). Actual num gpus = "
                + std::to_string(world_size_));
        }
#undef REDUCE_CASE
#undef LAUNCH_KERNEL
    }
};

}  // namespace my_ar