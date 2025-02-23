#ifndef NSA_CUDA_HELPERS_CUH
#define NSA_CUDA_HELPERS_CUH

#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Constants
#define BLOCK_SIZE 256
#define WARP_SIZE 32

namespace nsa_cuda_helpers {

// Helper device function for parallel reduction within a warp
template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val, cg::thread_block_tile<WARP_SIZE>& warp) {
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Helper device function for parallel max reduction within a warp
template<typename T>
__device__ __forceinline__ T warp_reduce_max(T val, cg::thread_block_tile<WARP_SIZE>& warp) {
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

} // namespace nsa_cuda_helpers

#endif // NSA_CUDA_HELPERS_CUH 