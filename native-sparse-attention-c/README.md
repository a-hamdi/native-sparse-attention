# Native Sparse Attention (NSA) CUDA Implementation

This repository contains a high-performance CUDA implementation of the Native Sparse Attention mechanism for transformer models. The implementation is designed to be efficient for long-context modeling on modern GPU hardware and is fully differentiable to support end-to-end training.

## Features

- **Three-Branch Attention Architecture**:
  - Compressed Token Branch: Global context through block compression
  - Selected Token Branch: Fine-grained selection of important blocks
  - Sliding Window Branch: Local context preservation

- **Optimized CUDA Kernels**:
  - Efficient memory access patterns
  - Shared memory utilization
  - Warp-level primitives for parallel reductions
  - Support for multiple CUDA architectures

- **Configurable Parameters**:
  - Block length and stride for token compression
  - Selection block size and count
  - Sliding window size
  - Head dimensions and count

## Requirements

- CUDA Toolkit 11.0 or later
- CMake 3.18 or later
- C/C++ compiler with C11/C++14 support

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

### Initialization

```c
#include <nsa/nsa.h>

// Create default configuration
NSAConfig config = nsa_create_default_config(64, 8);  // head_dim=64, num_heads=8

// Or customize configuration
config.block_length = 32;
config.block_stride = 16;
config.selection_block_size = 64;
config.num_selected_blocks = 16;
config.window_size = 512;

// Initialize NSA context
NSAContext ctx;
if (nsa_init(&ctx, &config) != 0) {
    // Handle error
}
```

### Computing Attention

```c
// Prepare inputs (on GPU memory)
float* query;        // [head_dim]
float* keys;         // [seq_len, head_dim]
float* values;       // [seq_len, head_dim]
float* output;       // [head_dim]
cudaStream_t stream;

// Compute attention
if (nsa_compute_attention(&ctx, query, keys, values, seq_len, output, stream) != 0) {
    // Handle error
}
```

### Updating Sliding Window

```c
// Add new tokens to sliding window
float* new_keys;     // [num_new, head_dim]
float* new_values;   // [num_new, head_dim]
int32_t num_new = 64;

if (nsa_update_window(&ctx, new_keys, new_values, num_new, stream) != 0) {
    // Handle error
}
```

### Cleanup

```c
// Free resources
nsa_free(&ctx);
```

## Implementation Details

### 1. Compressed Token Branch

The compressed token branch uses a learnable MLP to compress sequences of tokens into compact representations:

- Input tokens are partitioned into overlapping blocks
- Each block is processed with position-aware compression
- Compressed representations maintain global context

### 2. Selected Token Branch

The selected token branch identifies and processes the most relevant blocks:

- Computes importance scores using compressed keys
- Selects top-k blocks based on scores
- Processes selected tokens with full attention

### 3. Sliding Window Branch

The sliding window branch maintains recent context:

- Fixed-size window of recent tokens
- Efficient circular buffer implementation
- Full attention over window tokens

### 4. Gating Mechanism

The three branches are combined using learned gates:

- Each branch has a dedicated gate value
- Gates are computed using the current query
- Final output is a weighted sum of branch outputs

## Performance Considerations

1. **Memory Access**:
   - Coalesced memory access patterns
   - Shared memory for frequently accessed data
   - Minimal global memory transactions

2. **Parallel Processing**:
   - Warp-level primitives for reductions
   - Block-level parallelism for token processing
   - Stream-based asynchronous execution

3. **Resource Usage**:
   - Configurable shared memory usage
   - Balanced register pressure
   - Efficient workspace memory management

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 