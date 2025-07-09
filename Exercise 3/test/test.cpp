#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <vector>

int main() {
    // GPU test at the beginning
    std::cout << "Testing GPU availability..." << std::endl;
    #pragma omp target
    {
        if (omp_is_initial_device()) {
            printf("Running on CPU\n");
        } else {
            printf("Running on GPU\n");
        }
    }
    int data = 0;

    #pragma omp target map(tofrom: data)
    {
        data = 42;
    }

    printf("data = %d\n", data);
    return 0;
}
