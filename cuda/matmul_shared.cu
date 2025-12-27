#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

#define TILE_SIZE 32

__global__ void matMulShared(float* A, float* B, float* C, int N) {

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int tiledRow = row;
        int tiledCol = t * TILE_SIZE + threadIdx.x;

        if (tiledRow < N && tiledCol < N) {
            tileA[threadIdx.y][threadIdx.x] = A[tiledRow * N + tiledCol];
        }
        else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        tiledRow = t * TILE_SIZE + threadIdx.y;
        tiledCol = col;

        if (tiledRow < N && tiledCol < N) {
            tileB[threadIdx.y][threadIdx.x] = B[tiledRow * N + tiledCol];
        }
        else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    int N = 1024;
    int size = N * N * sizeof(float);

    vector<float> h_A(N * N, 1.0f);
    vector<float> h_B(N * N, 1.0f);
    vector<float> h_C(N * N);

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (N + TILE_SIZE - 1) / TILE_SIZE
    );

    cudaDeviceSynchronize();
    auto start = chrono::high_resolution_clock::now();

    matMulShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();

    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    chrono::duration<double, milli> elapsed = end - start;

    cout << "CUDA Shared MatMul Time: " << elapsed.count() << " ms" << endl;
    cout << "Sample Output: " << h_C[0] << endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}