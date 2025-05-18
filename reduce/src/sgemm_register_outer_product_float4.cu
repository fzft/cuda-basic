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

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

/*
C[y][x]= 
k=0
∑
K−1
​
 A[y][k]⋅B[k][x]

 A: M * K
 B: K * N
 C: M * N
*/
template<unsigned int M_NUM_PER_BLOCK, unsigned int N_NUM_PER_BLOCK, unsigned int K_NUM_PER_BLOCK, unsigned int REG_NUM>
__global__ void sgemm_kernel(float *A, float *B, float *C, const int M, const int N, const int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float* A_tile_start = A + blockIdx.y * M_NUM_PER_BLOCK * K; 
    float* B_tile_start = B + blockIdx.x * N_NUM_PER_BLOCK;

    __shared__ float shared_A[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float shared_B[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK]; 

    float a_reg[REG_NUM] = {0.0f};
    float b_reg[REG_NUM] = {0.0f};
    float sum[REG_NUM][REG_NUM] = {0.0f}; 

    // 加载A和B到共享内存
    for (int s = 0; s < K; s += K_NUM_PER_BLOCK) {
        for (int i = 0; i < REG_NUM; i++) {
            FETCH_FLOAT4(shared_A[ty * REG_NUM + i][tx * REG_NUM]) = FETCH_FLOAT4(A_tile_start[K * (ty * REG_NUM + i) + tx * REG_NUM + s]);
        }

        for (int i = 0; i < REG_NUM; i++) {
            FETCH_FLOAT4(shared_B[ty * REG_NUM + i][tx * REG_NUM]) = FETCH_FLOAT4(B_tile_start[N * (ty * REG_NUM + i + s) + tx * REG_NUM]);
        }
        __syncthreads(); 

        // 外积 
        for (int k = 0; k < K_NUM_PER_BLOCK; k++) {
            a_reg[0] = shared_A[ty * REG_NUM][k]; 
            a_reg[1] = shared_A[ty * REG_NUM + 1][k];
            a_reg[2] = shared_A[ty * REG_NUM + 2][k];
            a_reg[3] = shared_A[ty * REG_NUM + 3][k];
            FETCH_FLOAT4(b_reg[0]) = FETCH_FLOAT4(shared_B[k][tx * REG_NUM]);
            for (int i = 0; i < REG_NUM; i++) {
                for (int j = 0; j < REG_NUM; j++) {
                    sum[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }

        __syncthreads();
    }

    float *C_ptr = C + blockIdx.y * N * M_NUM_PER_BLOCK + blockIdx.x * N_NUM_PER_BLOCK;
    for (int i = 0; i < REG_NUM; i++) {
        for (int j = 0; j < REG_NUM; j++) {
            C_ptr[N * (ty * REG_NUM + i) + (tx * REG_NUM + j)] = sum[i][j];
        }
    } // 一步一步分析下 sgemm_kernel 每一行代码 为什么这样做
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

    constexpr unsigned int M_NUM_PER_BLOCK = 64;
    constexpr unsigned int N_NUM_PER_BLOCK = 64;
    constexpr unsigned int K_NUM_PER_BLOCK = 64;
    constexpr unsigned int NUM_PER_THREAD = 16;
    constexpr unsigned int REG_NUM = NUM_PER_THREAD / 4;

    dim3 block(16, 16);
    dim3 grid((m / M_NUM_PER_BLOCK), (n / N_NUM_PER_BLOCK));


    sgemm_kernel<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK, K_NUM_PER_BLOCK, REG_NUM><<<grid, block>>>(d_A, d_B, d_C, m, n, k);
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