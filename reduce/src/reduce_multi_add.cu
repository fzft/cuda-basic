#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>


using namespace std;

// Error checking macro
void cudaCheckError() { 
    cudaError_t err = cudaGetLastError(); 
    if(err != cudaSuccess) { 
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl; 
        exit(-1); 
    } 
}

const int THREADS_PER_BLOCK = 256;

__device__ void warp_reduce(volatile float *shared_input, int tid) {
    if (tid < 32) {
        shared_input[tid] += shared_input[tid + 32];
        shared_input[tid] += shared_input[tid + 16];
        shared_input[tid] += shared_input[tid + 8];
        shared_input[tid] += shared_input[tid + 4];
        shared_input[tid] += shared_input[tid + 2];
        shared_input[tid] += shared_input[tid + 1];
    }
}

template<unsigned int NUM_PER_BLOCK>
__global__ void reduce(float *input, float *output) {

    int tid = threadIdx.x;
    volatile __shared__ float shared_input[THREADS_PER_BLOCK];  
    // float *input_begin = input + blockIdx.x * blockDim.x * 2; 
    // shared_input[tid] = input_begin[tid] + input_begin[tid + blockDim.x];
    // __syncthreads();
    float *input_begin = input + blockIdx.x * NUM_PER_BLOCK;
    shared_input[tid] = 0;
    for (int i = 0; i <  NUM_PER_BLOCK / THREADS_PER_BLOCK; i++) {
        shared_input[tid] += input_begin[tid + i * THREADS_PER_BLOCK];
    }
    __syncthreads();

    for (int i = blockDim.x / 2 ; i > 32; i /= 2) {
        if (tid < i) { 
            shared_input[tid] += shared_input[tid + i];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warp_reduce(shared_input, tid);
    } 

    if (tid == 0) {
        output[blockIdx.x] = shared_input[0];
    }
} 


int main() {
    const int N = 32 * 1024 * 1024;
    float *input = (float*)malloc(N * sizeof(float));
    float *d_input;
    cudaMalloc((void**)&d_input, N * sizeof(float));

    constexpr int block_num = 1024;
    constexpr int num_per_block = N / block_num;
    float *output = (float*)malloc(block_num * sizeof(float));
    float *d_output;
    cudaMalloc((void**)&d_output, block_num * sizeof(float));

    float *result = (float*)malloc( block_num * sizeof(float));

    for (int i = 0; i < N; i++) {
        input[i] = 2.0 * (float)rand() / (float)RAND_MAX - 1.0;
    }

    // cpu reduce
    for (int i = 0; i < block_num; i++) {
        float sum = 0;
        for (int j = 0; j < num_per_block; j++) {
            sum += input[i * num_per_block + j];
        }
        result[i] = sum;
    }

    cout << "cpu reduce done" << endl;
    cout << "result[0] = " << result[0] << endl;
    cout << "result[1] = " << result[1] << endl;
    cout << "result[2] = " << result[2] << endl;
    cout << "result[3] = " << result[3] << endl;

    // gpu reduce
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);
 
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(block_num);

    reduce<num_per_block><<<grid, block>>>(d_input, d_output); // 所以grid 表示最上层结构， 一个gpu表示一个grid?
    cudaMemcpy(output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    cout << "gpu reduce done" << endl;
    cout << "output[0] = " << output[0] << endl;
    cout << "output[1] = " << output[1] << endl;
    cout << "output[2] = " << output[2] << endl;
    cout << "output[3] = " << output[3] << endl;

    cudaFree(d_input);
    cudaFree(d_output);
    free(input);
    free(output);
    free(result);
    return 0;

}
