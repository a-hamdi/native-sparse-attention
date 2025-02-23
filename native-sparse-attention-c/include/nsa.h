#ifndef NSA_H
#define NSA_H

#include "nsa_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initialize NSA context with given configuration
int nsa_init(NSAContext* ctx, const NSAConfig* config);

// Free resources associated with NSA context
void nsa_free(NSAContext* ctx);

// Compute NSA attention for a single query
int nsa_compute_attention(
    NSAContext* ctx,
    const float* query,          // [head_dim]
    const float* keys,           // [seq_len, head_dim]
    const float* values,         // [seq_len, head_dim]
    int32_t seq_len,
    float* output,              // [head_dim]
    cudaStream_t stream
);

// Update sliding window with new tokens
int nsa_update_window(
    NSAContext* ctx,
    const float* new_keys,       // [num_new, head_dim]
    const float* new_values,     // [num_new, head_dim]
    int32_t num_new,
    cudaStream_t stream
);

// Reset NSA context state (clear sliding window, etc.)
void nsa_reset(NSAContext* ctx);

// Get workspace size required for given configuration
size_t nsa_get_workspace_size(const NSAConfig* config);

// Helper function to create default configuration
NSAConfig nsa_create_default_config(
    int32_t head_dim,
    int32_t num_heads
);

#ifdef __cplusplus
}
#endif

#endif // NSA_H 