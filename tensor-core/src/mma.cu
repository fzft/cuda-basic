// Like with nvcuda::wmma there are three types of matrix tiles: A, B, and C with A @ B = C.
// A is a row-major matrix with shape M x K.
// B is a column-major matrix with shape K x N.
// C is a column-major matrix with shape M x N.

#include <iostream>
#include <cuda_runtime_api.h>
#include "common.cuh"

template<int I_, int J_, typename T>
struct tile {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = I_ * J_ / WARP_SIZE;
    T x[ne] = {0};

    static __device__ __forceinline__ int get_i(const int l) {
        if constexpr (I == 8 && (J == 4 || J == 8)) {
            return threadIdx.x / 4;
        } else if constexpr (I == 16 && J == 8) {
            return (l / 2) * 8 + threadIdx.x / 4;
        } else if constexpr (I == 16 && J == 16) {
            return ((l / 2) % 2) * 8 + threadIdx.x / 4;
        } else {
            static_assert(I == -1 && J == -1, "template specialization not implemented");
        }
    }

    static __device__ __forceinline__ int get_j(const int l) {
        if constexpr (I == 8 && J == 4) {
            return threadIdx.x % 4;
        } else if constexpr (I == 8 && J == 8) {
            return 4 * l + threadIdx.x % 4;
        } else if constexpr (I == 16 && J == 8) {
            return 2 * (threadIdx.x % 4) + l % 2;
        } else if constexpr (I == 16 && J == 16) {
            return 8 * (l / 4) + 2 * (threadIdx.x % 4) + l % 2;
        } else {
            static_assert(I == -1 && J == -1, "template specialization not implemented");
        }
    }
};



static __device__ __forceinline__ void mma(
        tile<16, 8, int> & D, const tile<16, 4, int> & A, const tile<8, 4, int> & B) {
#ifdef NEW_MMA_AVAILABLE
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
        : "r"(A.x[0]), "r"(A.x[1]), "r"(B.x[0]));
#else
    // On Turing m16n8k16 mma is not available, use 2x m8n8k16 mma instead:
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
        : "+r"(D.x[0]), "+r"(D.x[1])
        : "r"(A.x[0]), "r"(B.x[0]));
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
        : "+r"(D.x[2]), "+r"(D.x[3])
        : "r"(A.x[1]), "r"(B.x[0]));
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#else
    GGML_UNUSED(D);
    GGML_UNUSED(A);
    GGML_UNUSED(B);
    NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
}


int main() {
    return 0;
}