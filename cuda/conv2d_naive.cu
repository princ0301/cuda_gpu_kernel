#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

__global__ void conv2d(float* input, float* output, float* kernel, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int pad = 1;

    if (x >= pad && x < width - pad && y >= pad && y < height - pad) {
        float sum = 0.0f;

        for (int ky = -pad; ky <= pad; ky++) {
            for (int kx = -pad; kx <= pad; kx++) {
                int img_x = x + kx;
                int img_y = y + ky;

                float val = input[img_y * width + img_x];
                float k = kernel[(ky + pad) * 3 + (kx + pad)];

                sum += val * k;
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
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    };

    float *d_input, *d_output, *d_kernel;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_kernel, 9 * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, 9 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    cudaDeviceSynchronize();
    auto start = chrono::high_resolution_clock::now();

    conv2d<<<blocksPerGrid, threadsPerBlock>>>(
        d_input, d_output, d_kernel, width, height
    );

    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();

    cudaMemcpy(h_output.data(), d_output, size, cudaMemcpyDeviceToHost);

    chrono::duration<double, milli> elapsed = end - start;

    cout << "CUDA Conv2D Time: " << elapsed.count() << " ms" << endl;
    cout << "Sample Output (center pixel): "
         << h_output[(height / 2) * width + (width / 2)] << endl;

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return 0;
}