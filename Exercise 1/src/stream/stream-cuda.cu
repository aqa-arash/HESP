#include <chrono>

#include "../util.h"
#include "stream-util.h"

inline void stream(size_t nx, const double *__restrict__ src, double *__restrict__ dest) {
    for (int i = 0; i < nx; ++i)
        dest[i] = src[i] + 1;
}

__global__ void copyOnGPU(double *src, double *dest, size_t nx) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nx)
        dest[i] = src[i] + 1;
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    // ALLOCATE DATA
    size_t size = sizeof(double) * nx;
    // allocate _host_ arrays
    double *src, *dest;
    cudaMallocHost(&src, size);
    cudaMallocHost(&dest, size);

    // allocate _device_ arrays
    double *d_src, *d_dest;
    cudaMalloc(&d_src, size);
    cudaMalloc(&d_dest, size);

    // INITIALIZE DATA ON HOST
    initStream(src, nx);

    // COPY DATA FROM HOST TO DEVICE
    cudaMemcpy(d_src, src, size, cudaMemcpyHostToDevice);

    auto numThreadsPerBlock = 256;
    auto numBlocks = (nx + numThreadsPerBlock - 1) / numThreadsPerBlock;

    // measurement
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < nIt; ++i) {
        copyOnGPU<<<numBlocks, numThreadsPerBlock>>>(d_src, d_dest, nx);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Kernel launch error: " << cudaGetErrorString(err) << "\n";
        }
        std::swap(d_src, d_dest);
        // SYNCHRONIZE GPU
        cudaDeviceSynchronize();
    }

    auto end = std::chrono::steady_clock::now();

    // COPY DATA FROM DEVICE TO HOST
    cudaMemcpy(src, d_src, size, cudaMemcpyDeviceToHost);

    printStats(end - start, nx, nIt, streamNumReads, streamNumWrites);

    // check solution
    checkSolutionStream(src, nx, nIt);

    // de-allocate _device_ arrays
    cudaFree(d_src);
    cudaFree(d_dest);
    // de-allocate _host_ arrays
    cudaFreeHost(src);
    cudaFreeHost(dest);

    return 0;
}
