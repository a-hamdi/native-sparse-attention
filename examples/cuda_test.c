#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaError_t err;
    int runtime_version;
    int driver_version;
    
    err = cudaRuntimeGetVersion(&runtime_version);
    if (err != cudaSuccess) {
        printf("Failed to get CUDA runtime version: %s\n", cudaGetErrorString(err));
    } else {
        printf("CUDA Runtime Version: %d\n", runtime_version);
    }
    
    err = cudaDriverGetVersion(&driver_version);
    if (err != cudaSuccess) {
        printf("Failed to get CUDA driver version: %s\n", cudaGetErrorString(err));
    } else {
        printf("CUDA Driver Version: %d\n", driver_version);
    }
    
    int device_count;
    err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        printf("Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("Number of CUDA devices: %d\n", device_count);
    
    for (int i = 0; i < device_count; i++) {
        struct cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess) {
            printf("Failed to get properties for device %d: %s\n", i, cudaGetErrorString(err));
            continue;
        }
        
        printf("\nDevice %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %zu bytes\n", prop.totalGlobalMem);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max threads dimensions: (%d, %d, %d)\n", 
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max grid dimensions: (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Warp size: %d\n", prop.warpSize);
    }
    
    return 0;
} 