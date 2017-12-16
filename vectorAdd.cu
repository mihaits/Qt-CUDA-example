#include <iostream>
#include <chrono>

// helper functions for cleaner time measuring code
std::chrono::time_point<std::chrono::high_resolution_clock> now() {
    return std::chrono::high_resolution_clock::now();
}

template <typename T>
double milliseconds(T t) {
    return (double) std::chrono::duration_cast<std::chrono::nanoseconds>(t).count() / 1000000;
}

// gpu kernel function
__global__
void addKernel(double* x, double* y, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

extern "C"
double* addVectorsGPU(double* a, double* b, int n) {
    auto t1 = now();

    double* x;
    double* y;
    double* z;
    cudaMalloc(&x, n * sizeof(double));
    cudaMalloc(&y, n * sizeof(double));
    cudaMemcpy(x, a, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y, b, n * sizeof(double), cudaMemcpyHostToDevice);

    auto t2 = now();

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int blockSize = deviceProp.maxThreadsPerBlock;
    int numBlocks = (n - 1) / blockSize + 1;

    addKernel<<<numBlocks, blockSize>>>(x, y, n);
    cudaDeviceSynchronize();

    auto t3 = now();

    z = (double*) malloc(n * sizeof(double));
    cudaMemcpy(z, y, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(x);
    cudaFree(y);

    auto t4 = now();

    std::cout << "GPU time breakdown--------\n";
    std::cout << "loading into device memory: " << milliseconds(t2 - t1) << " milliseconds\n";
    std::cout << "actual addition:            " << milliseconds(t3 - t2) << " milliseconds\n";
    std::cout << "loading into host memory:   " << milliseconds(t4 - t3) << " milliseconds\n";

    return z;
}
