#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "../include/nsa_types.h"
#include "../include/nsa_cuda_kernels.h"
#include "../include/nsa_cuda_helpers.cuh"

namespace cg = cooperative_groups;

// Constants
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// CUDA kernel for computing attention scores and weighted sum
__global__ void compute_branch_attention_kernel(
    const float* query,          // [head_dim]
    const float* keys,           // [num_tokens, head_dim]
    const float* values,         // [num_tokens, head_dim]
    float* attention_output,     // [head_dim]
    int32_t num_tokens,
    int32_t head_dim,
    float scaling_factor
) {
    extern __shared__ float shared_mem[];
    
    // Shared memory layout:
    // - query: head_dim floats
    // - scores: num_tokens floats (for softmax)
    // - max_score: 1 float
    // - sum_exp: 1 float
    float* query_shared = shared_mem;
    float* scores = query_shared + head_dim;
    float* max_score = scores + num_tokens;
    float* sum_exp = max_score + 1;
    
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(block);
    
    // Load query into shared memory
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        query_shared[i] = query[i];
    }
    __syncthreads();
    
    // Step 1: Compute attention scores and find max
    float thread_max = -INFINITY;
    for (int i = threadIdx.x; i < num_tokens; i += blockDim.x) {
        float score = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            score += query_shared[j] * keys[i * head_dim + j];
        }
        score *= scaling_factor;
        scores[i] = score;
        thread_max = max(thread_max, score);
    }
    
    // Warp-level reduction for max score
    thread_max = nsa_cuda_helpers::warp_reduce_max<float>(thread_max, warp);
    if (warp.thread_rank() == 0) {
        atomicMax((int*)max_score, __float_as_int(thread_max));
    }
    __syncthreads();
    
    // Step 2: Compute softmax denominators
    float thread_sum = 0.0f;
    float max_val = *max_score;
    
    for (int i = threadIdx.x; i < num_tokens; i += blockDim.x) {
        float score = scores[i];
        float exp_val = __expf(score - max_val);
        scores[i] = exp_val;  // Store exp values for later use
        thread_sum += exp_val;
    }
    
    // Warp-level reduction for sum
    thread_sum = nsa_cuda_helpers::warp_reduce_sum<float>(thread_sum, warp);
    if (warp.thread_rank() == 0) {
        atomicAdd(sum_exp, thread_sum);
    }
    __syncthreads();
    
    // Step 3: Compute weighted sum of values
    float inv_sum = 1.0f / *sum_exp;
    
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float weighted_sum = 0.0f;
        for (int j = 0; j < num_tokens; j++) {
            weighted_sum += scores[j] * values[j * head_dim + i];
        }
        attention_output[i] = weighted_sum * inv_sum;
    }
}

void nsa_compute_branch_attention_cuda(
    const float* query,
    const float* keys,
    const float* values,
    float* attention_output,
    int32_t num_tokens,
    float scaling_factor,
    cudaStream_t stream
) {
    // Calculate shared memory size
    size_t shared_mem_size = (
        MAX_HEAD_DIM +    // query
        num_tokens +      // scores
        1 +              // max_score
        1                // sum_exp
    ) * sizeof(float);
    
    dim3 grid_dim(1);  // Single block for now, could be optimized for larger head dimensions
    dim3 block_dim(BLOCK_SIZE);
    
    compute_branch_attention_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(
        query,
        keys,
        values,
        attention_output,
        num_tokens,
        MAX_HEAD_DIM,
        scaling_factor
    );
} 