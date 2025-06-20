#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include <sys/time.h>
#include <time.h>

// extern "C"
__global__ void matrix_multiply(int *A, int *B, int *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // row index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // column index

    if (row < n && col < n) {
        int value = 0;
        for (int k = 0; k < n; ++k) {
            value += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = value;
    }
}
