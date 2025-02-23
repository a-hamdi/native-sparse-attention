#ifndef NSA_CUDA_KERNELS_H
#define NSA_CUDA_KERNELS_H

#include "nsa_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Kernel for compressing tokens into block representations
void nsa_compress_tokens_cuda(
    const float* input_tokens,      // [seq_len, head_dim]
    float* compressed_tokens,       // [num_blocks, head_dim]
    const float* weights_1,         // compression MLP weights
    const float* bias_1,
    const float* weights_2,
    const float* bias_2,
    const NSAConfig* config,
    cudaStream_t stream
);

// Kernel for computing importance scores and selecting top blocks
void nsa_compute_importance_scores_cuda(
    const float* query,             // [head_dim]
    const float* compressed_keys,   // [num_blocks, head_dim]
    float* importance_scores,       // [num_blocks]
    int32_t* selected_indices,      // [num_selected_blocks]
    const NSAConfig* config,
    cudaStream_t stream
);

// Kernel for extracting selected tokens
void nsa_extract_selected_tokens_cuda(
    const float* keys,              // [seq_len, head_dim]
    const float* values,            // [seq_len, head_dim]
    const int32_t* selected_indices,// [num_selected_blocks]
    float* selected_keys,           // [num_selected_tokens, head_dim]
    float* selected_values,         // [num_selected_tokens, head_dim]
    const NSAConfig* config,
    cudaStream_t stream
);

// Kernel for computing attention scores and outputs for each branch
void nsa_compute_branch_attention_cuda(
    const float* query,             // [head_dim]
    const float* keys,              // [num_tokens, head_dim]
    const float* values,            // [num_tokens, head_dim]
    float* attention_output,        // [head_dim]
    int32_t num_tokens,
    float scaling_factor,
    cudaStream_t stream
);

// Kernel for computing gating weights
void nsa_compute_gates_cuda(
    const float* query,             // [head_dim]
    const float* gate_weights,      // [3, head_dim]
    const float* gate_bias,         // [3]
    float* gate_outputs,            // [3]
    cudaStream_t stream
);

// Kernel for combining branch outputs with gates
void nsa_combine_branch_outputs_cuda(
    const float* branch_outputs,    // [3, head_dim]
    const float* gate_outputs,      // [3]
    float* final_output,           // [head_dim]
    const NSAConfig* config,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // NSA_CUDA_KERNELS_H 