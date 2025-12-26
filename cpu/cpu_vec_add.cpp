#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

int main() {
    int N = 16 * 1024 * 1024;

    vector<float> a(N, 1.0f);
    vector<float> b(N, 2.0f);
    vector<float> c(N);

    auto start = chrono::high_resolution_clock::now();

    for (int i=0; i<N; i++) {
        c[i] = a[i] + b[i];
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = end - start;

    cout << "CPU Vector Add Time: " << elapsed.count() << " ms\n";
    cout << "Sample Output: " << c[0] << endl;

    return 0;
}