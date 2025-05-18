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

__global__ void reduce(float *input, float *output) {
    float *input_begin = input + blockIdx.x * blockDim.x;
    for (int i = 1; i < blockDim.x; i *=2) {
        if (threadIdx.x % (2 * i) == 0) {
            input_begin[threadIdx.x] += input_begin[threadIdx.x + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[blockIdx.x] = input_begin[0];
    }
} 


int main() {
    const int N = 32 * 1024 * 1024;
    float *input = (float*)malloc(N * sizeof(float));
    float *d_input;
    cudaMalloc((void**)&d_input, N * sizeof(float));

    int block_num = N / THREADS_PER_BLOCK;
    float *output = (float*)malloc(block_num * sizeof(float));
    float *d_output;
    cudaMalloc((void**)&d_output, block_num * sizeof(float));

    float *result = (float*)malloc(block_num * sizeof(float));

    for (int i = 0; i < N; i++) {
        input[i] = 2.0 * (float)rand() / (float)RAND_MAX - 1.0;
    }

    // cpu reduce
    for (int i = 0; i < block_num; i++) {
        float sum = 0;
        for (int j = 0; j < THREADS_PER_BLOCK; j++) {
            sum += input[i * THREADS_PER_BLOCK + j];
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

    reduce<<<grid, block>>>(d_input, d_output); // 所以grid 表示最上层结构， 一个gpu表示一个grid?
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
