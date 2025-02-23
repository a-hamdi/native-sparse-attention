#ifndef NSA_TYPES_H
#define NSA_TYPES_H

#include <cuda_runtime.h>
#include <stdint.h>

// Constants
#define MAX_HEAD_DIM 1024  // Maximum dimension for attention heads

// Configuration structure for NSA parameters
typedef struct {
    int32_t block_length;        // l: length of compression blocks
    int32_t block_stride;        // d: stride between compression blocks
    int32_t selection_block_size;// l': size of selection blocks
    int32_t num_selected_blocks; // n: number of top blocks to select
    int32_t window_size;         // w: size of sliding window
    int32_t head_dim;           // dimension of attention heads
    int32_t num_heads;          // number of attention heads
    float scaling_factor;       // 1/sqrt(head_dim) for attention scaling
} NSAConfig;

// Structure for storing compressed token information
typedef struct {
    float* compressed_keys;     // [num_blocks, head_dim]
    float* compressed_values;   // [num_blocks, head_dim]
    int32_t num_blocks;
} CompressedTokens;

// Structure for storing selected token information
typedef struct {
    float* selected_keys;      // [num_selected_tokens, head_dim]
    float* selected_values;    // [num_selected_tokens, head_dim]
    int32_t* block_indices;    // indices of selected blocks
    int32_t num_selected_tokens;
} SelectedTokens;

// Structure for storing sliding window information
typedef struct {
    float* window_keys;        // [window_size, head_dim]
    float* window_values;      // [window_size, head_dim]
    int32_t current_size;      // current number of tokens in window
} SlidingWindow;

// Main NSA context structure
typedef struct {
    NSAConfig config;
    CompressedTokens compressed;
    SelectedTokens selected;
    SlidingWindow window;
    
    // MLP weights for compression
    float* compression_weights_1;  // First layer weights
    float* compression_weights_2;  // Second layer weights
    float* compression_bias_1;     // First layer bias
    float* compression_bias_2;     // Second layer bias
    
    // Gating network weights
    float* gate_weights;
    float* gate_bias;
    
    // Workspace memory
    void* workspace;
    size_t workspace_size;
} NSAContext;

#endif // NSA_TYPES_H 