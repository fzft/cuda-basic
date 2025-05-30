//
// Created by ros1 on 5/31/25.
//
/*
 * blockDim 是列的数量
 * 每个block处理一行
 * 每个线程处理一列
 *
 */

#include <cuda_runtime.h>
#include <float.h>
#include <iostream>
#include <stdio.h>

#include "common.cuh"

static __global__ void argmax_f32(const int * __restrict__ x, int * __restrict__ dst, const int ncols) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    int max_val = 0;
    int max_idx = -1;
    const int* rowx = x + row * ncols; // Pointer to the start of the current row

    for (int i = col; i < ncols; i += blockDim.x) {
        if (x[row * ncols + i] > max_val) {
            max_val = x[row * ncols + i];
            max_idx = i;
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        int other_val = __shfl_xor_sync(0xFFFFFFFF, max_val, offset);
        int other_idx = __shfl_xor_sync(0xFFFFFFFF, max_idx, offset);
        if (other_val > max_val) {
            max_val = other_val;
            max_idx = other_idx;
        }
    }

    const int lane_id = threadIdx.x % WARP_SIZE; // Get the lane ID within the warp
    const int warp_id = threadIdx.x / WARP_SIZE;

    if (warp_id == 0 && lane_id == 0) {
        // Only the first thread wargrites the result
        dst[row] = max_idx; // Store the index of the maximum value
    }
}

int main() {
    int *input, *output;
    int nrows = 1; // Number of rows
    int ncols = 256; // Number of columns
    input = (int*) malloc(nrows * ncols * sizeof(int)); // Allocate memory for input
    output = (int*) malloc(1 * sizeof(int)); // Allocate memory for output
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            input[i * ncols + j] =i + j;
        }
    }

    // we set the middle col to be the max value
    input[0 * ncols + ncols / 2] = 1000; // Set a specific value to be the maximum

    // cpu compute the sum
    int *d_input, *d_output;
    cudaMalloc((void**)&d_input, nrows * ncols * sizeof(int));
    cudaMalloc((void**)&d_output, nrows * sizeof(int));

    cudaMemcpy(d_input, input, nrows * ncols * sizeof(int), cudaMemcpyHostToDevice);
    long long start_time = clock();
    argmax_f32<<<1, 32>>>(d_input, d_output, ncols);
    long long end_time = clock();
    std::cout << "argmax_f32 time: " << (end_time - start_time) / 1000.0 << " ms" << std::endl;
    cudaMemcpy(output, d_output, nrows * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Max index: " << output[0] << std::endl;
    cudaFree(d_output);
    cudaFree(d_input);
    free(output);
    free(input);
    return 0;
}