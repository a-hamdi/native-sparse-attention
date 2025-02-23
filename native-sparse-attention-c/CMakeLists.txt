cmake_minimum_required(VERSION 3.16)
project(native_sparse_attention CUDA CXX C)

# Set CUDA flags
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find CUDA package
find_package(CUDA REQUIRED)

# Set compiler flags for maximum compatibility
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=compute_30")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

# Create library target
add_library(nsa SHARED
    src/nsa.c
    src/nsa_compress_kernel.cu
    src/nsa_importance_kernel.cu
    src/nsa_attention_kernel.cu
    src/nsa_gating_kernel.cu
)

# Link against CUDA libraries
target_link_libraries(nsa
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_cudart_static_LIBRARY}
)

# Set output directories
set_target_properties(nsa PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Install rules
install(TARGETS nsa
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES
    include/nsa.h
    include/nsa_types.h
    include/nsa_cuda_kernels.h
    include/nsa_cuda_helpers.cuh
    DESTINATION include/nsa
)

# Add test executables
add_executable(test_nsa examples/test_nsa.c)
target_link_libraries(test_nsa nsa)

add_executable(cuda_test examples/cuda_test.c)
target_link_libraries(cuda_test ${CUDA_LIBRARIES}) 