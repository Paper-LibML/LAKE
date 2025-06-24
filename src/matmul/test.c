#include <stdio.h>
#include <cuda_runtime.h>

#define N 1000  // Tamaño del vector

__global__ void vectorAdd(int* A, int* B, int* C) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int *h_A, *h_B, *h_C;  // Punteros para los vectores en la memoria del host
    int *d_A, *d_B, *d_C;  // Punteros para los vectores en la memoria del dispositivo
    int size = N * sizeof(int);

    // Alocar memoria para los vectores en el host
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    // Inicializar los vectores de entrada
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Alocar memoria en el dispositivo
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copiar los vectores de los datos del host al dispositivo
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Definir el tamaño del bloque y el número de bloques
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Lanzar el kernel
    vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

    // Copiar el resultado de vuelta a la memoria del host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Imprimir algunos resultados
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_C[i]);
    }
    printf("\n");

    // Liberar memoria en el dispositivo
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Liberar memoria en el host
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
