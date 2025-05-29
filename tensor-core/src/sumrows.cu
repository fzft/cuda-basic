//
// Created by ros1 on 5/30/25.
//
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <ostream>
#include <memory>
#include <chrono>

__global__ void sum_rows_f322(const int *x, int *dst, const int ncols) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    int sum = 0;
    for (int i = col; i < ncols; i += blockDim.x) {
        sum += x[row * ncols + i];
    }

    // warp 内规约
    sum = __reduce_add_sync(0xFFFFFFFF, sum);

    // 只让每个warp的第一个线程写入共享内存
    __shared__ int shared[8]; // 假设blockDim.x <= 256, 最多8个warp
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    // 只用第一个warp进一步归约共享内存里的warp部分和
    if (warp_id == 0) {
        int block_sum = (col < (blockDim.x / 32)) ? shared[lane] : 0;
        block_sum = __reduce_add_sync(0xFFFFFFFF, block_sum);
        if (threadIdx.x == 0) {
            dst[row] = block_sum;
        }
    }
}



// CUDA kernel
__global__ void sum_rows_f32(const int *x, int *dst, const int ncols) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    int sum = 0;
    for (int i = col; i < ncols; i += blockDim.x) {
        sum += x[row * ncols + i];
    }
    printf("sum = %d\n", sum);
    sum = __reduce_add_sync(0xFFFFFFFF, sum); // Use warp-level reduction to sum values across threads

    if (col == 0) {
        dst[row] = sum;
    }
}

int main() {
    int *input, *output;
    input = (int*) malloc(1024 * sizeof(int)); // Allocate memory for input
    output = (int*) malloc(1 * sizeof(int)); // Allocate memory for output
    int nrows = 1; // Number of rows
    int ncols = 256; // Number of columns
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            input[i * ncols + j] =i + j;
        }
    }

    // cpu compute the sum
    int sum = 0;
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            sum += input[i * ncols + j];
        }
    }
    std::cout << sum << std::endl;
    int *d_input, *d_output;
    cudaMalloc((void**)&d_input, nrows * ncols * sizeof(int));
    cudaMalloc((void**)&d_output, nrows * sizeof(int));

    cudaMemcpy(d_input, input, nrows * ncols * sizeof(int), cudaMemcpyHostToDevice);
    long long start_time = clock();
    sum_rows_f32<<<1, 32>>>(d_input, d_output, ncols);
    long long end_time = clock();
    std::cout << "sum_rows_f322 time: " << (end_time - start_time) / 1000.0 << " ms" << std::endl;
    cudaMemcpy(output, d_output, nrows * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nrows; ++i) {
        printf("Sum of row %d: %d\n", i, output[i]);
    }
}