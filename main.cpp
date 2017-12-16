#include <iostream>
#include <chrono>

extern "C"
double* addVectorsGPU(double* a, double* b, int n);

double* addVectorsCPU(double *a, double *b, int n) {
    auto r = (double*) malloc(n * sizeof(double));
    for(int i = 0; i < n; ++ i)
        r[i] = a[i] + b[i];
    return r;
}

double* genRanVec(int n) {
    double* v = (double*) malloc(n * sizeof(double));
    for(int i = 0; i < n; ++ i)
        v[i] = (double) rand() / RAND_MAX;
    return v;
}

int main() {
    int n = 1<<20;
    std::cout << "double vector addition, n = " << n << "\n\n";
    double* a = genRanVec(n);
    double* b = genRanVec(n);

    auto t = std::chrono::high_resolution_clock::now();
    auto r1 = addVectorsGPU(a, b, n);
    auto gpuTimeMs = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - t).count() / 1000000;

    t = std::chrono::high_resolution_clock::now();
    auto r2 = addVectorsCPU(a, b, n);
    auto cpuTimeMs = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - t).count() / 1000000;

    std::cout << "\nTOTAL----------------\n";
    std::cout << "GPU: " << gpuTimeMs << " milliseconds\n";
    std::cout << "CPU: " << cpuTimeMs << " milliseconds\n";

    delete[] r1;
    delete[] r2;

    return 0;
}
