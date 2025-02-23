#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "../include/nsa_types.h"
#include "../include/nsa_cuda_kernels.h"
#include "../include/nsa_cuda_helpers.cuh"

namespace cg = cooperative_groups;

// Constants for the compression kernel
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Helper device function for computing position encoding
__device__ void compute_position_encoding(
    float* pos_enc,
    int32_t pos,
    int32_t dim,
    float max_freq = 10000.0f
) {
    for (int i = 0; i < dim; i += 2) {
        float freq = expf(-logf(max_freq) * i / dim);
        pos_enc[i] = sinf(pos * freq);
        if (i + 1 < dim) {
            pos_enc[i + 1] = cosf(pos * freq);
        }
    }
}

// CUDA kernel for MLP forward pass with position encoding
__global__ void compress_block_kernel(
    const float* input_tokens,    // [seq_len, head_dim]
    float* compressed_tokens,     // [num_blocks, head_dim]
    const float* weights_1,       // [head_dim, head_dim]
    const float* bias_1,         // [head_dim]
    const float* weights_2,       // [head_dim, head_dim]
    const float* bias_2,         // [head_dim]
    int32_t seq_len,
    int32_t head_dim,
    int32_t block_length,
    int32_t block_stride
) {
    extern __shared__ float shared_mem[];
    
    // Shared memory layout:
    // - Position encodings: head_dim floats
    // - Block tokens: block_length * head_dim floats
    // - Intermediate activations: head_dim floats
    float* pos_enc = shared_mem;
    float* block_tokens = pos_enc + head_dim;
    float* intermediate = block_tokens + block_length * head_dim;
    
    const int32_t block_idx = blockIdx.x;
    const int32_t thread_idx = threadIdx.x;
    const int32_t start_pos = block_idx * block_stride;
    
    // Load block tokens into shared memory
    for (int i = thread_idx; i < block_length * head_dim; i += blockDim.x) {
        int token_idx = i / head_dim;
        int feat_idx = i % head_dim;
        int global_pos = start_pos + token_idx;
        
        if (global_pos < seq_len) {
            block_tokens[i] = input_tokens[global_pos * head_dim + feat_idx];
        } else {
            block_tokens[i] = 0.0f;
        }
    }
    __syncthreads();
    
    // First MLP layer with position encoding
    for (int i = thread_idx; i < head_dim; i += blockDim.x) {
        float sum = bias_1[i];
        
        // Add position-aware weighted sum
        for (int j = 0; j < block_length; j++) {
            // Compute position encoding for current token
            if (thread_idx == 0) {
                compute_position_encoding(pos_enc, j, head_dim);
            }
            __syncthreads();
            
            float pos_weight = pos_enc[i];
            for (int k = 0; k < head_dim; k++) {
                sum += block_tokens[j * head_dim + k] * 
                      weights_1[i * head_dim + k] * 
                      pos_weight;
            }
        }
        
        // ReLU activation
        intermediate[i] = sum > 0.0f ? sum : 0.0f;
    }
    __syncthreads();
    
    // Second MLP layer
    for (int i = thread_idx; i < head_dim; i += blockDim.x) {
        float sum = bias_2[i];
        for (int j = 0; j < head_dim; j++) {
            sum += intermediate[j] * weights_2[i * head_dim + j];
        }
        compressed_tokens[block_idx * head_dim + i] = sum;
    }
}

void nsa_compress_tokens_cuda(
    const float* input_tokens,
    float* compressed_tokens,
    const float* weights_1,
    const float* bias_1,
    const float* weights_2,
    const float* bias_2,
    const NSAConfig* config,
    cudaStream_t stream
) {
    int32_t num_blocks = (config->block_length + config->block_stride - 1) / 
                        config->block_stride;
    
    // Calculate shared memory size
    size_t shared_mem_size = (
        config->head_dim +                    // Position encodings
        config->block_length * config->head_dim + // Block tokens
        config->head_dim                      // Intermediate activations
    ) * sizeof(float);
    
    dim3 grid_dim(num_blocks);
    dim3 block_dim(BLOCK_SIZE);
    
    compress_block_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(
        input_tokens,
        compressed_tokens,
        weights_1,
        bias_1,
        weights_2,
        bias_2,
        config->block_length,
        config->head_dim,
        config->block_length,
        config->block_stride
    );
} 