#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "../include/nsa_types.h"
#include "../include/nsa_cuda_kernels.h"
#include "../include/nsa_cuda_helpers.cuh"

namespace cg = cooperative_groups;

// Constants
#define BLOCK_SIZE 256
#define NUM_BRANCHES 3

// CUDA kernel for computing gating weights
__global__ void compute_gates_kernel(
    const float* query,          // [head_dim]
    const float* gate_weights,   // [3, head_dim]
    const float* gate_bias,      // [3]
    float* gate_outputs,         // [3]
    int32_t head_dim
) {
    extern __shared__ float shared_mem[];
    float* query_shared = shared_mem;
    float* local_sums = shared_mem + head_dim;
    
    // Load query into shared memory
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        query_shared[i] = query[i];
    }
    __syncthreads();
    
    // Each thread computes partial sums for all gates
    float thread_sums[NUM_BRANCHES] = {0.0f};
    
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float q_val = query_shared[i];
        for (int j = 0; j < NUM_BRANCHES; j++) {
            thread_sums[j] += q_val * gate_weights[j * head_dim + i];
        }
    }
    
    // Store partial sums in shared memory
    for (int i = 0; i < NUM_BRANCHES; i++) {
        local_sums[threadIdx.x * NUM_BRANCHES + i] = thread_sums[i];
    }
    __syncthreads();
    
    // Reduce partial sums and apply sigmoid
    if (threadIdx.x < NUM_BRANCHES) {
        float sum = gate_bias[threadIdx.x];
        for (int i = 0; i < blockDim.x; i++) {
            sum += local_sums[i * NUM_BRANCHES + threadIdx.x];
        }
        // Sigmoid activation: 1 / (1 + exp(-x))
        gate_outputs[threadIdx.x] = 1.0f / (1.0f + __expf(-sum));
    }
}

// CUDA kernel for combining branch outputs
__global__ void combine_branch_outputs_kernel(
    const float* branch_outputs, // [3, head_dim]
    const float* gate_outputs,   // [3]
    float* final_output,        // [head_dim]
    int32_t head_dim
) {
    const int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= head_dim) return;
    
    // Load gate values
    float gates[NUM_BRANCHES];
    if (threadIdx.x < NUM_BRANCHES) {
        gates[threadIdx.x] = gate_outputs[threadIdx.x];
    }
    __syncthreads();
    
    // Compute weighted sum of branch outputs
    float sum = 0.0f;
    for (int i = 0; i < NUM_BRANCHES; i++) {
        sum += gates[i] * branch_outputs[i * head_dim + tid];
    }
    
    final_output[tid] = sum;
}

void nsa_compute_gates_cuda(
    const float* query,
    const float* gate_weights,
    const float* gate_bias,
    float* gate_outputs,
    cudaStream_t stream
) {
    size_t shared_mem_size = (MAX_HEAD_DIM + BLOCK_SIZE * NUM_BRANCHES) * sizeof(float);
    
    dim3 grid_dim(1);
    dim3 block_dim(BLOCK_SIZE);
    
    compute_gates_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(
        query,
        gate_weights,
        gate_bias,
        gate_outputs,
        MAX_HEAD_DIM
    );
}

void nsa_combine_branch_outputs_cuda(
    const float* branch_outputs,
    const float* gate_outputs,
    float* final_output,
    const NSAConfig* config,
    cudaStream_t stream
) {
    dim3 grid_dim((config->head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_dim(BLOCK_SIZE);
    
    combine_branch_outputs_kernel<<<grid_dim, block_dim, 0, stream>>>(
        branch_outputs,
        gate_outputs,
        final_output,
        config->head_dim
    );
} 