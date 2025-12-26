#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

int main() {
    int width = 512;
    int height = 512;
    int kernel_size = 3;
    int pad = kernel_size / 2;

    vector<float> image(width * height, 1.0f);
    vector<float> output(width * height, 0.0f);

    float kernel[3][3] = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };

    auto start = chrono::high_resolution_clock::now();

    for (int y=pad; y<height - pad; y++) {
        for (int x=pad; x<width - pad; x++) {
            float sum = 0.0f;

            for (int ky=-pad; ky<=pad; ky++) {
                for (int kx=-pad; kx<=pad; kx++) {
                    int img_y = y + ky;
                    int img_x = x + kx;

                    sum += image[img_y * width + img_x] * kernel[ky + pad][kx + pad];
                }
            }
            output[y * width + x] = sum;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = end - start;

    cout << "CPU COnv2D Time: " << elapsed.count() << " ms" << endl;
    cout << "Sample Output: " << output[width / 2 * width + width / 2] << endl;

    return 0;
}