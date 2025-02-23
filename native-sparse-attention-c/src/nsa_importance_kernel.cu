#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "../include/nsa_types.h"
#include "../include/nsa_cuda_kernels.h"
#include "../include/nsa_cuda_helpers.cuh"

namespace cg = cooperative_groups;

// Constants
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// CUDA kernel for computing importance scores
__global__ void compute_importance_scores_kernel(
    const float* query,           // [head_dim]
    const float* compressed_keys, // [num_blocks, head_dim]
    float* importance_scores,     // [num_blocks]
    float scaling_factor,
    int32_t num_blocks,
    int32_t head_dim
) {
    extern __shared__ float shared_mem[];
    float* query_shared = shared_mem; // [head_dim]
    
    // Load query into shared memory
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        query_shared[i] = query[i];
    }
    __syncthreads();
    
    // Each thread block processes one compressed key block
    const int32_t block_idx = blockIdx.x;
    if (block_idx >= num_blocks) return;
    
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(block);
    
    // Compute dot product between query and compressed key
    float dot_product = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        dot_product += query_shared[i] * compressed_keys[block_idx * head_dim + i];
    }
    
    // Warp-level reduction for dot product
    dot_product = nsa_cuda_helpers::warp_reduce_sum<float>(dot_product, warp);
    
    // First thread in each warp writes the result
    if (threadIdx.x % WARP_SIZE == 0) {
        importance_scores[block_idx] = dot_product * scaling_factor;
    }
}

// CUDA kernel for selecting top-k blocks using parallel selection
__global__ void select_top_blocks_kernel(
    const float* importance_scores, // [num_blocks]
    int32_t* selected_indices,     // [num_selected_blocks]
    int32_t num_blocks,
    int32_t num_selected_blocks
) {
    extern __shared__ float shared_mem[];
    float* local_scores = shared_mem;                    // [BLOCK_SIZE]
    int32_t* local_indices = (int32_t*)(local_scores + BLOCK_SIZE); // [BLOCK_SIZE]
    
    const int32_t tid = threadIdx.x;
    const int32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize local arrays
    local_scores[tid] = gid < num_blocks ? importance_scores[gid] : -INFINITY;
    local_indices[tid] = gid < num_blocks ? gid : -1;
    __syncthreads();
    
    // Parallel bitonic sort within shared memory
    for (int k = 2; k <= BLOCK_SIZE; k *= 2) {
        for (int j = k/2; j > 0; j /= 2) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                if ((tid & k) == 0) {
                    if (local_scores[tid] < local_scores[ixj]) {
                        float temp_score = local_scores[tid];
                        local_scores[tid] = local_scores[ixj];
                        local_scores[ixj] = temp_score;
                        
                        int32_t temp_idx = local_indices[tid];
                        local_indices[tid] = local_indices[ixj];
                        local_indices[ixj] = temp_idx;
                    }
                } else {
                    if (local_scores[tid] > local_scores[ixj]) {
                        float temp_score = local_scores[tid];
                        local_scores[tid] = local_scores[ixj];
                        local_scores[ixj] = temp_score;
                        
                        int32_t temp_idx = local_indices[tid];
                        local_indices[tid] = local_indices[ixj];
                        local_indices[ixj] = temp_idx;
                    }
                }
            }
            __syncthreads();
        }
    }
    
    // Write top-k results
    if (tid < num_selected_blocks && local_indices[tid] != -1) {
        selected_indices[tid] = local_indices[tid];
    }
}

// CUDA kernel for extracting selected tokens
__global__ void extract_selected_tokens_kernel(
    const float* keys,              // [seq_len, head_dim]
    const float* values,            // [seq_len, head_dim]
    const int32_t* selected_indices,// [num_selected_blocks]
    float* selected_keys,           // [num_selected_tokens, head_dim]
    float* selected_values,         // [num_selected_tokens, head_dim]
    int32_t block_length,
    int32_t block_stride,
    int32_t head_dim
) {
    const int32_t block_idx = blockIdx.x;
    const int32_t thread_idx = threadIdx.x;
    const int32_t selected_block = selected_indices[block_idx];
    const int32_t start_pos = selected_block * block_stride;
    const int32_t output_offset = block_idx * block_length * head_dim;

    // Each thread copies one element at a time
    for (int i = thread_idx; i < block_length * head_dim; i += blockDim.x) {
        int token_idx = i / head_dim;
        int feat_idx = i % head_dim;
        int input_pos = (start_pos + token_idx) * head_dim + feat_idx;
        int output_pos = output_offset + token_idx * head_dim + feat_idx;

        selected_keys[output_pos] = keys[input_pos];
        selected_values[output_pos] = values[input_pos];
    }
}

void nsa_compute_importance_scores_cuda(
    const float* query,
    const float* compressed_keys,
    float* importance_scores,
    int32_t* selected_indices,
    const NSAConfig* config,
    cudaStream_t stream
) {
    int32_t num_blocks = (config->block_length + config->block_stride - 1) / 
                        config->block_stride;
    
    // First kernel: compute importance scores
    {
        size_t shared_mem_size = config->head_dim * sizeof(float);
        dim3 grid_dim((num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block_dim(BLOCK_SIZE);
        
        compute_importance_scores_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(
            query,
            compressed_keys,
            importance_scores,
            config->scaling_factor,
            num_blocks,
            config->head_dim
        );
    }
    
    // Second kernel: select top blocks
    {
        size_t shared_mem_size = BLOCK_SIZE * (sizeof(float) + sizeof(int32_t));
        dim3 grid_dim(1); // Single block for now, could be optimized for very large sequences
        dim3 block_dim(BLOCK_SIZE);
        
        select_top_blocks_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(
            importance_scores,
            selected_indices,
            num_blocks,
            config->num_selected_blocks
        );
    }
}

void nsa_extract_selected_tokens_cuda(
    const float* keys,
    const float* values,
    const int32_t* selected_indices,
    float* selected_keys,
    float* selected_values,
    const NSAConfig* config,
    cudaStream_t stream
) {
    dim3 grid_dim(config->num_selected_blocks);
    dim3 block_dim(BLOCK_SIZE);

    extract_selected_tokens_kernel<<<grid_dim, block_dim, 0, stream>>>(
        keys,
        values,
        selected_indices,
        selected_keys,
        selected_values,
        config->block_length,
        config->block_stride,
        config->head_dim
    );
} 