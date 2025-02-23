#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/nsa.h"

#define SEQ_LEN 1024
#define HEAD_DIM 256
#define NUM_HEADS 8

// Helper function to check CUDA errors
void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// Helper function to initialize random data
void init_random_data(float* data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    // Get CUDA device count
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get CUDA device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }

    printf("Found %d CUDA device(s)\n", deviceCount);

    // Get device properties
    struct cudaDeviceProp deviceProp;
    err = cudaGetDeviceProperties(&deviceProp, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Using device 0: %s\n", deviceProp.name);
    printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max shared memory per block: %zu bytes\n", deviceProp.sharedMemPerBlock);

    // Initialize CUDA device
    err = cudaSetDevice(0);
    check_cuda_error(err, "cudaSetDevice");

    // Create NSA configuration
    NSAConfig config = nsa_create_default_config(HEAD_DIM, NUM_HEADS);

    printf("NSA Configuration:\n");
    printf("- Head dimension: %d\n", config.head_dim);
    printf("- Number of heads: %d\n", config.num_heads);
    printf("- Block length: %d\n", config.block_length);
    printf("- Block stride: %d\n", config.block_stride);
    printf("- Selection block size: %d\n", config.selection_block_size);
    printf("- Number of selected blocks: %d\n", config.num_selected_blocks);
    printf("- Window size: %d\n", config.window_size);
    printf("- Scaling factor: %f\n", config.scaling_factor);

    // Initialize NSA context
    NSAContext ctx;
    if (nsa_init(&ctx, &config) != 0) {
        fprintf(stderr, "Failed to initialize NSA context\n");
        return 1;
    }

    printf("NSA context initialized successfully\n");

    // Allocate host memory
    size_t seq_size = SEQ_LEN * HEAD_DIM * sizeof(float);
    float* h_query = (float*)malloc(HEAD_DIM * sizeof(float));
    float* h_keys = (float*)malloc(seq_size);
    float* h_values = (float*)malloc(seq_size);
    float* h_output = (float*)malloc(HEAD_DIM * sizeof(float));

    if (!h_query || !h_keys || !h_values || !h_output) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return 1;
    }

    printf("Host memory allocated successfully\n");

    // Initialize random data
    init_random_data(h_query, HEAD_DIM);
    init_random_data(h_keys, SEQ_LEN * HEAD_DIM);
    init_random_data(h_values, SEQ_LEN * HEAD_DIM);

    printf("Random data initialized\n");

    // Allocate device memory
    float *d_query, *d_keys, *d_values, *d_output;
    err = cudaMalloc((void**)&d_query, HEAD_DIM * sizeof(float));
    check_cuda_error(err, "cudaMalloc query");
    err = cudaMalloc((void**)&d_keys, seq_size);
    check_cuda_error(err, "cudaMalloc keys");
    err = cudaMalloc((void**)&d_values, seq_size);
    check_cuda_error(err, "cudaMalloc values");
    err = cudaMalloc((void**)&d_output, HEAD_DIM * sizeof(float));
    check_cuda_error(err, "cudaMalloc output");

    printf("Device memory allocated successfully\n");

    // Copy data to device
    err = cudaMemcpy(d_query, h_query, HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice);
    check_cuda_error(err, "cudaMemcpy query");
    err = cudaMemcpy(d_keys, h_keys, seq_size, cudaMemcpyHostToDevice);
    check_cuda_error(err, "cudaMemcpy keys");
    err = cudaMemcpy(d_values, h_values, seq_size, cudaMemcpyHostToDevice);
    check_cuda_error(err, "cudaMemcpy values");

    printf("Data copied to device successfully\n");

    // Create CUDA stream
    cudaStream_t stream;
    err = cudaStreamCreate(&stream);
    check_cuda_error(err, "cudaStreamCreate");

    printf("CUDA stream created successfully\n");

    // Compute attention
    printf("Computing attention...\n");
    if (nsa_compute_attention(&ctx, d_query, d_keys, d_values, SEQ_LEN, d_output, stream) != 0) {
        fprintf(stderr, "Failed to compute attention\n");
        return 1;
    }

    // Synchronize stream
    err = cudaStreamSynchronize(stream);
    check_cuda_error(err, "cudaStreamSynchronize");

    printf("Attention computed successfully\n");

    // Copy result back to host
    err = cudaMemcpy(h_output, d_output, HEAD_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    check_cuda_error(err, "cudaMemcpy output");

    // Print first few elements of the output
    printf("Output (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_output[i]);
    }
    printf("\n");

    // Cleanup
    printf("Cleaning up...\n");
    cudaStreamDestroy(stream);
    cudaFree(d_query);
    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_output);
    free(h_query);
    free(h_keys);
    free(h_values);
    free(h_output);
    nsa_free(&ctx);

    printf("Test completed successfully\n");
    return 0;
} 