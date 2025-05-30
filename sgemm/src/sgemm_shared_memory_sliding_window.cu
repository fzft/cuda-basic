/*
 * 矩阵乘法
 * 矩阵A(m, k) * 矩阵B(k, n) = 矩阵C(m, n)
 * 矩阵C的计算方式：C[i][j] = sum(A[i][k] * B[k][j]) for k in range(k)
 * 矩阵C的计算方式：C[i][j] = sum(A[i][k] * B[k][j]) for k in range(k)
 * 矩阵C的计算方式：C[i][j] = sum(A[i][k] * B[k][j]) for k in range(k)
 * 矩阵C的计算方式：C[i][j] = sum(A[i][k] * B[k][j]) for k in range(k)
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdlib>

using namespace std;

void random_matrix(int rows, int cols, float *h_A) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            h_A[(i) * (cols) + (j)]  = rand() / (float)RAND_MAX;
        }
    }
} 

void cpu_sgemm(float *A, float *B, float *C, const int M, const int N, const int K) {
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            for (int k = 0; k < K; k++) {
                C[m * N + n] += A[m * K + k] * B[k * N + n];
            }
        }
    }
} 

float compare_matrix(float *A, float *B, const int M, const int N) {
    int i, j;
    float max_diff = 0.0f, diff;
    int printed = 0;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            diff = abs(A[i * N + j] - B[i * N + j]);
            max_diff = max(max_diff, diff);
            if (printed == 0) {
                if (max_diff > 0.5f || max_diff < -0.5f){
                    cout << "A[" << i << "][" << j << "] = " << A[i * N + j] << ", B[" << i << "][" << j << "] = " << B[i * N + j] << ", diff = " << diff << endl;
                    printed++;
                }
            }
        }
    }
    return max_diff;
}

/*
C[y][x]= 
k=0
∑
K−1
​
 A[y][k]⋅B[k][x]
*/

template<unsigned int BLOCK_SIZE>
__global__ void sgemm_kernel(float *A, float *B, float *C, const int M, const int N, const int K) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    float *A_row_start = A + blockIdx.y * blockDim.y * K; 
    float *B_col_start = B + blockIdx.x * blockDim.x;

    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;
    for (int s = 0; s < K; s+= BLOCK_SIZE) {
        shared_A[threadIdx.y][threadIdx.x] = A_row_start[threadIdx.y * K + s + threadIdx.x];
        shared_B[threadIdx.y][threadIdx.x] = B_col_start[(threadIdx.y + s) * N + threadIdx.x];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[x + y * N] = sum;
}  

int main() {
    const int m = 512;
    const int n = 512;
    const int k = 512;
    const size_t mem_size_A = m * k * sizeof(float);
    const size_t mem_size_B = k * n * sizeof(float);
    const size_t mem_size_C = m * n * sizeof(float);

    float *h_A = (float *)malloc(mem_size_A);
    float *h_B = (float *)malloc(mem_size_B);

    float *matrix_C_host_gpu_calc = (float *)malloc(mem_size_C);
    float *matrix_C_host_cpu_calc = (float *)malloc(mem_size_C);

    random_matrix(m, k, h_A);
    random_matrix(k, n, h_B);
    memset(matrix_C_host_gpu_calc, 0, mem_size_C);
    memset(matrix_C_host_cpu_calc, 0, mem_size_C); 

    float *d_A;
    float *d_B;
    float *d_C;

    cudaMalloc((void**)&d_A, mem_size_A);
    cudaMalloc((void**)&d_B, mem_size_B);
    cudaMalloc((void**)&d_C, mem_size_C);


    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    cpu_sgemm(h_A, h_B, matrix_C_host_cpu_calc, m, n, k);

    const int threads_per_block = 16;
    dim3 block(threads_per_block, threads_per_block);
    dim3 grid((m + threads_per_block - 1) / threads_per_block, (n + threads_per_block - 1) / threads_per_block);
    

    sgemm_kernel<threads_per_block><<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    cudaMemcpy(matrix_C_host_gpu_calc, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    float diff = compare_matrix(matrix_C_host_gpu_calc, matrix_C_host_cpu_calc, m, n);
    if (diff > 0.5f || diff < -0.5f) {
        cout << "diff: " << diff << endl;
    } else {
        cout << "success" << endl;
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    return 0;
}