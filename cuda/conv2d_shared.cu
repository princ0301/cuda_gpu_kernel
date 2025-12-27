#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

#define TILE 16
#define KERNEL 3
#define RADIUS 1

__global__ void conv2dShared(float* input, float* output,
                             float* kernel, int width, int height) {

    __shared__ float tile[TILE + 2][TILE + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * TILE + tx;
    int y = blockIdx.y * TILE + ty;

    int shared_x = tx + RADIUS;
    int shared_y = ty + RADIUS;

    if (x < width && y < height)
        tile[shared_y][shared_x] = input[y * width + x];
    else
        tile[shared_y][shared_x] = 0.0f;

    if (tx == 0 && x > 0)
        tile[shared_y][0] = input[y * width + (x - 1)];
    if (tx == TILE - 1 && x < width - 1)
        tile[shared_y][shared_x + 1] = input[y * width + (x + 1)];
    if (ty == 0 && y > 0)
        tile[0][shared_x] = input[(y - 1) * width + x];
    if (ty == TILE - 1 && y < height - 1)
        tile[shared_y + 1][shared_x] = input[(y + 1) * width + x];

    __syncthreads();

    if (x >= RADIUS && x < width - RADIUS &&
        y >= RADIUS && y < height - RADIUS) {

        float sum = 0.0f;
        for (int ky = 0; ky < KERNEL; ky++) {
            for (int kx = 0; kx < KERNEL; kx++) {
                sum += tile[ty + ky][tx + kx] *
                       kernel[ky * KERNEL + kx];
            }
        }
        output[y * width + x] = sum;
    }
}

int main() {
    int width = 512;
    int height = 512;
    int size = width * height * sizeof(float);

    vector<float> h_input(width * height, 1.0f);
    vector<float> h_output(width * height, 0.0f);

    float h_kernel[9] = {
        1,1,1,
        1,1,1,
        1,1,1
    };

    float *d_input, *d_output, *d_kernel;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_kernel, 9 * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, 9 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(TILE, TILE);
    dim3 blocks((width + TILE - 1) / TILE,
                (height + TILE - 1) / TILE);

    cudaDeviceSynchronize();
    auto start = chrono::high_resolution_clock::now();

    conv2dShared<<<blocks, threads>>>(
        d_input, d_output, d_kernel, width, height
    );

    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();

    cudaMemcpy(h_output.data(), d_output, size, cudaMemcpyDeviceToHost);

    chrono::duration<double, milli> elapsed = end - start;

    cout << "CUDA Shared Conv2D Time: " << elapsed.count() << " ms" << endl;
    cout << "Sample Output: "
         << h_output[(height / 2) * width + (width / 2)] << endl;

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return 0;
}
