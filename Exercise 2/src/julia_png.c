#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "lodepng/lodepng.h"

#define WIDTH 800
#define HEIGHT 800
#define MAX_ITER 128
#define THRESHOLD 10.0

typedef struct {
    double re;
    double im;
} Complex;

Complex complex_square(Complex z) {
    Complex result;
    result.re = z.re * z.re - z.im * z.im;
    result.im = 2.0 * z.re * z.im;
    return result;
}

Complex complex_add(Complex z, Complex c) {
    Complex result;
    result.re = z.re + c.re;
    result.im = z.im + c.im;
    return result;
}

double complex_abs(Complex z) {
    return sqrt(z.re * z.re + z.im * z.im);
}

int julia_iterations(Complex z0, Complex c) {
    Complex z = z0;
    for (int i = 0; i < MAX_ITER; i++) {
        z = complex_add(complex_square(z), c);
        if (complex_abs(z) > THRESHOLD)
            return i;
    }
    return MAX_ITER;
}

int main() {
    unsigned char* image = (unsigned char*)malloc(WIDTH * HEIGHT * 4);
    if (!image) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    Complex c = {-0.9, 0.4};
    double x_min = -1.0, x_max = 1.0;
    double y_min = -2.0, y_max = 2.0;

    auto start = std::chrono::steady_clock::now();
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            double re = x_min + (x_max - x_min) * x / (WIDTH - 1);
            double im = y_max - (y_max - y_min) * y / (HEIGHT - 1);
            Complex z0 = {re, im};
            int iter = julia_iterations(z0, c);

            int index = 4 * (y * WIDTH + x);
            float t = (float)iter / MAX_ITER;

            // Simple rainbow-like RGB mapping
            unsigned char r = (unsigned char)(9 * (1 - t) * t * t * t * 255);
            unsigned char g = (unsigned char)(15 * (1 - t) * (1 - t) * t * t * 255);
            unsigned char b = (unsigned char)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);

            image[index + 0] = r;
            image[index + 1] = g;
            image[index + 2] = b;
            image[index + 3] = 255;  // Alpha

        }
    }
    auto end = std::chrono::steady_clock::now();

    char filename[50];
    int n = 1;

    // Find the next available filename
    do {
        snprintf(filename, sizeof(filename), "julia_%d.png", n);
        FILE* file = fopen(filename, "r");
        if (file) {
            fclose(file);
            n++;
        } else {
            break;
        }
    } while (1);

    // Add metadata to the PNG file
    char metadata[256];
    snprintf(metadata, sizeof(metadata), 
             "c = (%.2f, %.2f), max_iter = %d, threshold = %.2f, domain = [%.2f, %.2f] x [%.2f, %.2f]", 
             c.re, c.im, MAX_ITER, THRESHOLD, x_min, x_max, y_min, y_max);

    unsigned error = lodepng_encode32_file(filename, image, WIDTH, HEIGHT);
    if (error) {
        fprintf(stderr, "Error %u: %s\n", error, lodepng_error_text(error));
    } else {
        // Update file paths to include the directory
        char filepath[100];
        snprintf(filepath, sizeof(filepath), "%s", filename);

        unsigned error = lodepng_encode32_file(filepath, image, WIDTH, HEIGHT);
        if (error) {
            fprintf(stderr, "Error %u: %s\n", error, lodepng_error_text(error));
        } else {
            printf("PNG image written to '%s'\n", filepath);
            auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            printf("Elapsed time: %ld ms\n", elapsed_time);

            // Write metadata to a separate text file
            char metadata_filepath[100];
            snprintf(metadata_filepath, sizeof(metadata_filepath), "julia_%d_metadata.txt", n);
            FILE* metadata_file = fopen(metadata_filepath, "w");
            if (metadata_file) {
            fprintf(metadata_file, "%s\n", metadata);
            fclose(metadata_file);
            printf("Metadata written to '%s'\n", metadata_filepath);
            } else {
            fprintf(stderr, "Failed to write metadata to '%s'\n", metadata_filepath);
            }
        }
    }

    free(image);
    return 0;
}
