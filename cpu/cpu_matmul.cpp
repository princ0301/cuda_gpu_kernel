#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

int main() {
    int N = 1024;

    vector<float> A(N * N, 1.0f);
    vector<float> B(N * N, 1.0f);
    vector<float> C(N * N, 0.0f);

    auto start = chrono::high_resolution_clock::now();

    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            float sum = 0.0f;
            for (int k=0; k<N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "CPU Matrix Mul TIme: " << elapsed.count() << " seconds\n";
    cout << "Sample Output: " << C[0] << endl;

    return 0;
}