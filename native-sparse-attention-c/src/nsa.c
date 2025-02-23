#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../include/nsa.h"
#include "../include/nsa_cuda_kernels.h"

// Helper function to get minimum of two integers
static inline int32_t min_int32(int32_t a, int32_t b) {
    return (a < b) ? a : b;
}

// Helper function to allocate CUDA memory
static int allocate_cuda_memory(void** ptr, size_t size) {
    cudaError_t err = cudaMalloc(ptr, size);
    return err == cudaSuccess ? 0 : -1;
}

// Helper function to free CUDA memory
static void free_cuda_memory(void* ptr) {
    if (ptr) cudaFree(ptr);
}

int nsa_init(NSAContext* ctx, const NSAConfig* config) {
    if (!ctx || !config) return -1;
    
    // Copy configuration
    memcpy(&ctx->config, config, sizeof(NSAConfig));
    
    // Calculate sizes
    int32_t num_blocks = (config->block_length + config->block_stride - 1) / 
                        config->block_stride;
    
    // Allocate memory for compressed tokens
    size_t compressed_size = num_blocks * config->head_dim * sizeof(float);
    if (allocate_cuda_memory((void**)&ctx->compressed.compressed_keys, compressed_size) ||
        allocate_cuda_memory((void**)&ctx->compressed.compressed_values, compressed_size)) {
        nsa_free(ctx);
        return -1;
    }
    ctx->compressed.num_blocks = num_blocks;
    
    // Allocate memory for selected tokens
    size_t selected_size = config->num_selected_blocks * config->block_length * 
                          config->head_dim * sizeof(float);
    if (allocate_cuda_memory((void**)&ctx->selected.selected_keys, selected_size) ||
        allocate_cuda_memory((void**)&ctx->selected.selected_values, selected_size) ||
        allocate_cuda_memory((void**)&ctx->selected.block_indices, 
                           config->num_selected_blocks * sizeof(int32_t))) {
        nsa_free(ctx);
        return -1;
    }
    ctx->selected.num_selected_tokens = 0;
    
    // Allocate memory for sliding window
    size_t window_size = config->window_size * config->head_dim * sizeof(float);
    if (allocate_cuda_memory((void**)&ctx->window.window_keys, window_size) ||
        allocate_cuda_memory((void**)&ctx->window.window_values, window_size)) {
        nsa_free(ctx);
        return -1;
    }
    ctx->window.current_size = 0;
    
    // Allocate memory for MLP weights
    size_t mlp_weight_size = config->head_dim * config->head_dim * sizeof(float);
    size_t mlp_bias_size = config->head_dim * sizeof(float);
    if (allocate_cuda_memory((void**)&ctx->compression_weights_1, mlp_weight_size) ||
        allocate_cuda_memory((void**)&ctx->compression_weights_2, mlp_weight_size) ||
        allocate_cuda_memory((void**)&ctx->compression_bias_1, mlp_bias_size) ||
        allocate_cuda_memory((void**)&ctx->compression_bias_2, mlp_bias_size)) {
        nsa_free(ctx);
        return -1;
    }
    
    // Allocate memory for gating network
    size_t gate_weight_size = 3 * config->head_dim * sizeof(float);
    size_t gate_bias_size = 3 * sizeof(float);
    if (allocate_cuda_memory((void**)&ctx->gate_weights, gate_weight_size) ||
        allocate_cuda_memory((void**)&ctx->gate_bias, gate_bias_size)) {
        nsa_free(ctx);
        return -1;
    }
    
    // Allocate workspace memory
    ctx->workspace_size = nsa_get_workspace_size(config);
    if (allocate_cuda_memory(&ctx->workspace, ctx->workspace_size)) {
        nsa_free(ctx);
        return -1;
    }
    
    return 0;
}

void nsa_free(NSAContext* ctx) {
    if (!ctx) return;
    
    // Free compressed tokens memory
    free_cuda_memory(ctx->compressed.compressed_keys);
    free_cuda_memory(ctx->compressed.compressed_values);
    
    // Free selected tokens memory
    free_cuda_memory(ctx->selected.selected_keys);
    free_cuda_memory(ctx->selected.selected_values);
    free_cuda_memory(ctx->selected.block_indices);
    
    // Free sliding window memory
    free_cuda_memory(ctx->window.window_keys);
    free_cuda_memory(ctx->window.window_values);
    
    // Free MLP weights
    free_cuda_memory(ctx->compression_weights_1);
    free_cuda_memory(ctx->compression_weights_2);
    free_cuda_memory(ctx->compression_bias_1);
    free_cuda_memory(ctx->compression_bias_2);
    
    // Free gating network weights
    free_cuda_memory(ctx->gate_weights);
    free_cuda_memory(ctx->gate_bias);
    
    // Free workspace
    free_cuda_memory(ctx->workspace);
    
    // Reset all pointers
    memset(ctx, 0, sizeof(NSAContext));
}

int nsa_compute_attention(
    NSAContext* ctx,
    const float* query,
    const float* keys,
    const float* values,
    int32_t seq_len,
    float* output,
    cudaStream_t stream
) {
    if (!ctx || !query || !keys || !values || !output) return -1;
    
    // Workspace memory layout:
    float* importance_scores = (float*)ctx->workspace;
    float* branch_outputs = importance_scores + ctx->compressed.num_blocks;
    float* gate_outputs = branch_outputs + 3 * ctx->config.head_dim;
    
    // Step 1: Compress tokens
    nsa_compress_tokens_cuda(
        keys,
        ctx->compressed.compressed_keys,
        ctx->compression_weights_1,
        ctx->compression_bias_1,
        ctx->compression_weights_2,
        ctx->compression_bias_2,
        &ctx->config,
        stream
    );
    
    nsa_compress_tokens_cuda(
        values,
        ctx->compressed.compressed_values,
        ctx->compression_weights_1,
        ctx->compression_bias_1,
        ctx->compression_weights_2,
        ctx->compression_bias_2,
        &ctx->config,
        stream
    );
    
    // Step 2: Compute importance scores and select blocks
    nsa_compute_importance_scores_cuda(
        query,
        ctx->compressed.compressed_keys,
        importance_scores,
        ctx->selected.block_indices,
        &ctx->config,
        stream
    );
    
    // Step 3: Extract selected tokens
    nsa_extract_selected_tokens_cuda(
        keys,
        values,
        ctx->selected.block_indices,
        ctx->selected.selected_keys,
        ctx->selected.selected_values,
        &ctx->config,
        stream
    );
    
    // Step 4: Compute attention for each branch
    // Compressed branch
    nsa_compute_branch_attention_cuda(
        query,
        ctx->compressed.compressed_keys,
        ctx->compressed.compressed_values,
        branch_outputs,
        ctx->compressed.num_blocks,
        ctx->config.scaling_factor,
        stream
    );
    
    // Selected branch
    nsa_compute_branch_attention_cuda(
        query,
        ctx->selected.selected_keys,
        ctx->selected.selected_values,
        branch_outputs + ctx->config.head_dim,
        ctx->selected.num_selected_tokens,
        ctx->config.scaling_factor,
        stream
    );
    
    // Window branch
    nsa_compute_branch_attention_cuda(
        query,
        ctx->window.window_keys,
        ctx->window.window_values,
        branch_outputs + 2 * ctx->config.head_dim,
        ctx->window.current_size,
        ctx->config.scaling_factor,
        stream
    );
    
    // Step 5: Compute gating weights
    nsa_compute_gates_cuda(
        query,
        ctx->gate_weights,
        ctx->gate_bias,
        gate_outputs,
        stream
    );
    
    // Step 6: Combine branch outputs
    nsa_combine_branch_outputs_cuda(
        branch_outputs,
        gate_outputs,
        output,
        &ctx->config,
        stream
    );
    
    return 0;
}

int nsa_update_window(
    NSAContext* ctx,
    const float* new_keys,
    const float* new_values,
    int32_t num_new,
    cudaStream_t stream
) {
    if (!ctx || !new_keys || !new_values || num_new <= 0) return -1;
    
    int32_t window_size = ctx->config.window_size;
    int32_t head_dim = ctx->config.head_dim;
    
    // Calculate how many tokens to keep from the old window
    int32_t keep_tokens = min_int32(window_size - num_new, ctx->window.current_size);
    if (keep_tokens > 0) {
        // Shift existing tokens to make room for new ones
        cudaMemcpyAsync(
            ctx->window.window_keys,
            ctx->window.window_keys + (ctx->window.current_size - keep_tokens) * head_dim,
            keep_tokens * head_dim * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream
        );
        
        cudaMemcpyAsync(
            ctx->window.window_values,
            ctx->window.window_values + (ctx->window.current_size - keep_tokens) * head_dim,
            keep_tokens * head_dim * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream
        );
    }
    
    // Copy new tokens
    int32_t copy_tokens = min_int32(num_new, window_size - keep_tokens);
    if (copy_tokens > 0) {
        cudaMemcpyAsync(
            ctx->window.window_keys + keep_tokens * head_dim,
            new_keys + (num_new - copy_tokens) * head_dim,
            copy_tokens * head_dim * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream
        );
        
        cudaMemcpyAsync(
            ctx->window.window_values + keep_tokens * head_dim,
            new_values + (num_new - copy_tokens) * head_dim,
            copy_tokens * head_dim * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream
        );
    }
    
    ctx->window.current_size = keep_tokens + copy_tokens;
    return 0;
}

void nsa_reset(NSAContext* ctx) {
    if (!ctx) return;
    ctx->window.current_size = 0;
    ctx->selected.num_selected_tokens = 0;
}

size_t nsa_get_workspace_size(const NSAConfig* config) {
    if (!config) return 0;
    
    int32_t num_blocks = (config->block_length + config->block_stride - 1) / 
                        config->block_stride;
    
    return (
        num_blocks +                    // importance scores
        3 * config->head_dim +         // branch outputs
        3                              // gate outputs
    ) * sizeof(float);
}

NSAConfig nsa_create_default_config(int32_t head_dim, int32_t num_heads) {
    NSAConfig config;
    config.head_dim = head_dim;
    config.num_heads = num_heads;
    config.block_length = 32;
    config.block_stride = 16;
    config.selection_block_size = 64;
    config.num_selected_blocks = 16;
    config.window_size = 512;
    config.scaling_factor = 1.0f / sqrtf(head_dim);
    return config;
} 