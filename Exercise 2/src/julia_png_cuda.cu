#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "lodepng/lodepng.h"
#include "../cuda-util.h"

#define WIDTH 800
#define HEIGHT 800

typedef struct {
    double re;
    double im;
} Complex;

__global__ void compute_julia(unsigned char* image, Complex c, double x_min, double x_max, double y_min, double y_max, int max_iter, int color_map, double threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) return;

    double re = x_min + (x_max - x_min) * x / (WIDTH - 1);
    double im = y_max - (y_max - y_min) * y / (HEIGHT - 1);
    Complex z = {re, im};

    int iter = 0;
    for (; iter < max_iter; iter++) {
        double z_re2 = z.re * z.re;
        double z_im2 = z.im * z.im;
        if (z_re2 + z_im2 > threshold * threshold) break;

        double new_re = z_re2 - z_im2 + c.re;
        double new_im = 2.0 * z.re * z.im + c.im;
        z.re = new_re;
        z.im = new_im;
    }

    int index = 4 * (y * WIDTH + x);
    float t = (float)iter / max_iter;

    unsigned char r, g, b;
    if (color_map == 0) {
        // Rainbow
        r = (unsigned char)(9 * (1 - t) * t * t * t * 255);
        g = (unsigned char)(15 * (1 - t) * (1 - t) * t * t * 255);
        b = (unsigned char)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
    } else {
        // Inverted
        r = (unsigned char)(255 * (1 - t));
        g = (unsigned char)(128 * (1 - t));
        b = (unsigned char)(255 * t);
    }

    image[index + 0] = r;
    image[index + 1] = g;
    image[index + 2] = b;
    image[index + 3] = 255;
}

int main(int argc, char** argv) {
    if (argc != 10) {
        fprintf(stderr, "Usage: %s c_re c_im max_iter x_min x_max y_min y_max color_map threshold\n", argv[0]);
        return 1;
    }

    // ALLOCATE DATA
    Complex c = { atof(argv[1]), atof(argv[2]) };
    int max_iter = atoi(argv[3]);
    double x_min = atof(argv[4]);
    double x_max = atof(argv[5]);
    double y_min = atof(argv[6]);
    double y_max = atof(argv[7]);
    int color_map = atoi(argv[8]);
    double threshold = atof(argv[9]);

    unsigned char* image;
    unsigned char* d_image;
    size_t image_size = WIDTH * HEIGHT * 4;

    checkCudaError(cudaMallocHost(&image, image_size));
    checkCudaError(cudaMalloc(&d_image, image_size));

    dim3 blockDim(16, 16);
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, (HEIGHT + blockDim.y - 1) / blockDim.y);

    auto start = std::chrono::steady_clock::now();
    // LAUNCH GPU KERNEL
    compute_julia<<<gridDim, blockDim>>>(d_image, c, x_min, x_max, y_min, y_max, max_iter, color_map, threshold);
    // SYNC GPU
    checkCudaError(cudaDeviceSynchronize(), true);
    auto end = std::chrono::steady_clock::now();

    // COPY DATA BACK
    checkCudaError(cudaMemcpy(image, d_image, image_size, cudaMemcpyDeviceToHost));

    // POST PROCESS DATA
    char filename[256];
    snprintf(filename, sizeof(filename), "images/julia_cuda_re%.2f_im%.2f_iter%d_map%d_thr%.1f_x%.2f-%.2f_y%.2f-%.2f.png", 
             c.re, c.im, max_iter, color_map, threshold, x_min, x_max, y_min, y_max);
    unsigned error = lodepng_encode32_file(filename, image, WIDTH, HEIGHT);
    if (error) {
        fprintf(stderr, "Error %u: %s\n", error, lodepng_error_text(error));
    } else {
        printf("PNG image written to '%s'\n", filename);
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("Elapsed time: %ld ms\n", elapsed_time);
    }

    checkCudaError(cudaFreeHost(image));
    checkCudaError(cudaFree(d_image));
    return 0;
}
