#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
 
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int N = 16 * 1024 * 1024;  
    int size = N * sizeof(float);

    vector<float> h_a(N, 1.0f);
    vector<float> h_b(N, 2.0f);
    vector<float> h_c(N);

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaDeviceSynchronize();
    auto start = chrono::high_resolution_clock::now();

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();

    cudaMemcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost);

    chrono::duration<double, milli> elapsed = end - start;

    cout << "CUDA Vector Add Time: " << elapsed.count() << " ms" << endl;
    cout << "Sample Output: " << h_c[0] << endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
