//write a quick test for the parser
#include <iostream>
#include <tuple>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <math.h>
#include <cmath>
#include "parser.hpp"
#include <chrono>
#include "cpufuncs.hpp"
// cuda includes
#include "cudafuncs.hpp"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

int main(int argc, char** argv) {
    // Check if the correct number of arguments is provided
    std::string configFile;
    if (argc != 2) {
         configFile= "config.txt";
    }
    else {    // Get the config file name from command line arguments
         configFile = argv[1];
    }
    // Test file name
    // Variables to hold parsed data
    std::vector<double> positions_old, velocities_old, masses, positions_new, velocities_new, accelerations, forces;
    
    double timeStepLength = 0.0, timeStepCount = 0.0, sigma = 0.0, epsilon = 0.0, boxSize = 0.0, cutoffRadius =0.0;
    int printInterval = 0;
    int numParticles = 0;
    int useAcc = 0;

    
    // Call the parser
    parseConfigFile(configFile, positions_old, velocities_old, masses, boxSize, cutoffRadius, timeStepLength, timeStepCount, sigma, epsilon, printInterval, useAcc);

    // Output the parsed data
    std::cout << "Parsed Data:" << std::endl;
    std::cout << "Time Step Length: " << timeStepLength << std::endl;
    std::cout << "Time Step Count: " << timeStepCount << std::endl;
    std::cout << "Sigma: " << sigma << std::endl;
    std::cout << "Epsilon: " << epsilon << std::endl;
    std::cout << "Box Size: " << boxSize << std::endl;
    std::cout<< "Cutoff Radius: "<< cutoffRadius<< std::endl;
    std::cout << "Print Interval: " << printInterval << std::endl;  
    numParticles = positions_old.size()/3;
    std::cout << "Number of particles: " << numParticles << std::endl;
    std::cout << "Use acceleration: " << useAcc << std::endl;

    // Check if the parsed data is valid
    if (sigma <= 0.0 || epsilon <= 0.0) {
    std::cerr << "Error: Invalid sigma or epsilon values. Exiting simulation." << std::endl;
    return -1;
}
// check if poisitions are valid 
if (positions_old.size() % 3 != 0) {
    std::cerr << "Error: Invalid number of position values. Exiting simulation." << std::endl;
    return -1;
}

    // the minimum x is 0.0
    // check if the positions are out of bounds
    if (boxSize > 0.000000001) { // to avoid numerical errors with very small box sizes
    for (const auto& pos : positions_old) {
        if (pos < 0.0 || pos > boxSize) {
            std::cerr << "Error: Positions are out of bounds!" << std::endl;
            return -1;
        }
    }
}

    //set box size to the maximum position + 0.5
    accelerations.resize(positions_old.size(), 0.0);
    forces.resize(positions_old.size(), 0.0);
    positions_new.resize(positions_old.size(), 0.0);
    velocities_new.resize(positions_old.size(), 0.0);
    
    //initialize the values on device
    double *positions_old_d, *velocities_old_d, *forces_d, *accelerations_d, *masses_d;
    double *positions_new_d, *velocities_new_d;
    int *cells_d;
    int *particleCell_d;
   
    // calculate the cell size and number of cells
    double cell_size;
    int num_cells;
    // if boxSize is 0.0 or cutoffRadius is 0.0, set num_cells to 1 and cell_size to boxSize
    // otherwise find the minimal divisor of cutoffRadius and boxSize
    if (boxSize == 0.0 || cutoffRadius == 0.0 || useAcc == 0) {
        num_cells = 1;
        cell_size = boxSize;
        std::cout << "Box size or cutoff radius or useAcc is zero, setting num_cells = 1, cell_size = " << cell_size << std::endl;
    } else {
        try {
            std::tie(cell_size, num_cells) = findMinimalDivisor(cutoffRadius, boxSize);
            std::cout << "Found cell_size: " << cell_size << ", with num_cells per dimension = " << num_cells << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
    
    int total_cells = num_cells * num_cells * num_cells;

    // allocate memory on device
    CUDA_CHECK( cudaMalloc(&positions_old_d, positions_old.size() * sizeof(double)));
    CUDA_CHECK( cudaMalloc(&velocities_old_d, velocities_old.size() * sizeof(double)));
    CUDA_CHECK( cudaMalloc(&forces_d, forces.size() * sizeof(double)));
    CUDA_CHECK( cudaMalloc(&accelerations_d, accelerations.size() * sizeof(double)));
    CUDA_CHECK( cudaMalloc(&masses_d, masses.size() * sizeof(double)));
    CUDA_CHECK( cudaMalloc(&positions_new_d, positions_new.size() * sizeof(double)));
    CUDA_CHECK( cudaMalloc(&velocities_new_d, velocities_new.size() * sizeof(double)));
    CUDA_CHECK( cudaMalloc(&particleCell_d, numParticles * sizeof(int)));
    CUDA_CHECK( cudaMalloc(&cells_d, total_cells * sizeof(int)));

    // copy data to device
    CUDA_CHECK( cudaMemcpy(positions_old_d, positions_old.data(), positions_old.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK( cudaMemcpy(velocities_old_d, velocities_old.data(), velocities_old.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK( cudaMemcpy(forces_d, forces.data(), forces.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK( cudaMemcpy(accelerations_d, accelerations.data(), accelerations.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK( cudaMemcpy(masses_d, masses.data(), masses.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK( cudaMemcpy(positions_new_d, positions_new.data(), positions_new.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK( cudaMemcpy(velocities_new_d, velocities_new.data(), velocities_new.size() * sizeof(double), cudaMemcpyHostToDevice));

    // cout
    std::cout << "Data copied to device" << std::endl;
    

    // prepare the device kernel launch parameters
    dim3 blockSize(256);
    dim3 gridSize((numParticles + blockSize.x - 1) / blockSize.x);
    dim3 cellGridSize((total_cells + blockSize.x - 1) / blockSize.x);
    // launch the kernel to check periodic boundaries
    resetCells<<<cellGridSize, blockSize>>>(cells_d, total_cells); // should we launch less blocks ? 
    CUDA_CHECK(cudaGetLastError());

    computeParticleCells<<<gridSize, blockSize>>>(
                positions_old_d,
                cells_d,
                particleCell_d,
                numParticles,
                num_cells,
                total_cells,
                cell_size
            );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    //loop for GPU
    std::cout << "Starting time loop for GPU ..." << std::endl;
    // time loop
    auto positions_total = std::chrono::duration<double>::zero();
    int positions_count = 0;
    auto velocities_total = std::chrono::duration<double>::zero();
    int velocities_count = 0;
    auto forces_and_accelerations_total = std::chrono::duration<double>::zero();
    int forces_and_accelerations_count = 0;


    auto total_start = std::chrono::high_resolution_clock::now();
    for (int timestep = 0; timestep < timeStepCount; ++timestep) {
        // cout
        //std::cout << "Time step: " << timestep << std::endl;
        //std::cout<< "updating positions and velocities"<< std::endl;
        auto positions_start = std::chrono::high_resolution_clock::now();
        update_positions_d<<<gridSize, blockSize>>>( positions_new_d, positions_old_d, 
            velocities_old_d, accelerations_d, timeStepLength, boxSize, numParticles);
        CUDA_CHECK(cudaGetLastError());      
        CUDA_CHECK(cudaDeviceSynchronize());
        auto positions_end = std::chrono::high_resolution_clock::now();
        positions_total += positions_end - positions_start;
        positions_count++;
        
        auto velocities_start = std::chrono::high_resolution_clock::now();
        update_velocities_d<<<gridSize, blockSize>>>(velocities_new_d, velocities_old_d,
             accelerations_d, timeStepLength, numParticles);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        auto velocities_end = std::chrono::high_resolution_clock::now();
        velocities_total += velocities_end - velocities_start;
        velocities_count++;
        //std::cout<< "Positions updated"<< std::endl;
        //std::cout<< "Update complete, swapping"<< std::endl;
        std::swap(positions_old_d, positions_new_d);
        std::swap(velocities_old_d, velocities_new_d);

        // Build up linked neighbor list
        // Reset cells to -1
        // always use the cells if (useAcc == 1) {
        if (useAcc == 1 && timestep % 10 == 0) { // reset cells every 10th timestep
            resetCells<<<cellGridSize, blockSize>>>(cells_d, total_cells); // should we launch less blocks ? 
            computeParticleCells<<<gridSize, blockSize>>>(
                positions_old_d,
                cells_d,
                particleCell_d,
                numParticles,
                num_cells,
                total_cells,
                cell_size
            );
            
            /* debugging code to print particleCell_d and cells_d{
            // Copy particleCell_d and cells_d from device to host and print them
            std::vector<int> particleCell_host(numParticles);
            std::vector<int> cells_host(total_cells);

            CUDA_CHECK(cudaMemcpy(particleCell_host.data(), particleCell_d, numParticles * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(cells_host.data(), cells_d, total_cells * sizeof(int), cudaMemcpyDeviceToHost));
            
            // Print particleCell_d with timestep
            std::cout << "timestep " << timestep << " particleCell_d: ";
            for (int i = 0; i < numParticles; ++i) {
                std::cout << particleCell_host[i] << " ";
            }
            std::cout << std::endl;

            // Print cells_d with timestep
            std::cout << "timestep " << timestep << " cells_d: ";
            for (int i = 0; i < total_cells; ++i) {
                std::cout << cells_host[i] << " ";
            }
            std::cout << std::endl;
        }*/
        }
        
        auto forces_and_accelerations_start = std::chrono::high_resolution_clock::now();
        //std::cout << " Calculating Forces and accelerations"<< std::endl;
        // update forces and accelerations
        acceleration_updater_d<<<gridSize, blockSize>>>(accelerations_d,positions_old_d, 
            forces_d, masses_d, sigma, epsilon, boxSize, cutoffRadius, numParticles, cells_d, particleCell_d, num_cells);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaDeviceSynchronize());
        auto forces_and_accelerations_end = std::chrono::high_resolution_clock::now();
        forces_and_accelerations_total += forces_and_accelerations_end - forces_and_accelerations_start;
        forces_and_accelerations_count++;
        // cout
        //std::cout << "updating velocities"<< std::endl;
        // update velocities
        auto update_velocities_start = std::chrono::high_resolution_clock::now();
        update_velocities_d<<<gridSize, blockSize>>>(velocities_new_d, velocities_old_d,
             accelerations_d, timeStepLength, numParticles);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        auto update_velocities_end = std::chrono::high_resolution_clock::now();
        velocities_total += update_velocities_end - update_velocities_start;
        velocities_count++;

        // transfer new positions and velocities to old positions
        
        std::swap(velocities_old_d, velocities_new_d);
        //std::cout<< "Loop complete"<< std::endl;
        // print to file every printInterval steps
        if (printInterval > 0 && timestep % printInterval == 0) {
            // cout
            std::cout << "writing iteration " << timestep<<" to file"<< std::endl;
            cudaMemcpy(positions_old.data(), positions_old_d, positions_old.size() * sizeof(double), cudaMemcpyDeviceToHost);
            CUDA_CHECK(cudaGetLastError());
            cudaMemcpy(velocities_old.data(), velocities_old_d, velocities_old.size() * sizeof(double), cudaMemcpyDeviceToHost);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
    
            std::string outputFile = "output/cuda-output" + std::to_string(timestep / printInterval) + ".vtk";
            writeVTKFile(outputFile, positions_old, velocities_old, masses);
        }
    }
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_elapsed = total_end - total_start;
    std::cout << "Elapsed time on GPU: " << std::chrono::duration<double>(total_elapsed).count() << " seconds" << std::endl;
    std::cout<<"Average time for updating positions: " << std::chrono::duration<double>(positions_total).count() / positions_count << " seconds" << std::endl;
    std::cout<<"Average time for updating velocities: " << std::chrono::duration<double>(velocities_total).count() / velocities_count << " seconds" << std::endl;
    std::cout<<"Average time for forces and accelerations: " << std::chrono::duration<double>(forces_and_accelerations_total).count() / forces_and_accelerations_count << " seconds" << std::endl;
    std::cout << "Total time for GPU simulation: " << std::chrono::duration<double>(total_elapsed).count() << " seconds" << std::endl;  
    
    std::cout << "Simulation complete!" << std::endl;

    double total_elapsed_seconds = std::chrono::duration<double>(total_elapsed).count();
    // Append runtime to a file for later plotting
    std::ofstream runtime_file("runtimes.txt", std::ios::app);
    if (runtime_file.is_open()) {
        runtime_file << "----------------------------------------" << std::endl;
        runtime_file << "Configuration File: " << configFile << std::endl;
        runtime_file << "Average time for updating positions: " << std::chrono::duration<double>(positions_total).count() / positions_count << " seconds" << std::endl;
        runtime_file << "Average time for updating velocities: " << std::chrono::duration<double>(velocities_total).count() / velocities_count << " seconds" << std::endl;
        runtime_file << "Average time for forces and accelerations: " << std::chrono::duration<double>(forces_and_accelerations_total).count() / forces_and_accelerations_count << " seconds" << std::endl;
        runtime_file << "Total time for GPU simulation: " << total_elapsed_seconds << " seconds" << std::endl;
        runtime_file << "----------------------------------------" << std::endl;
        runtime_file.close();
    } else {
        std::cerr << "Could not open runtimes.txt for writing!" << std::endl;
    }

    // Free device memory
    cudaFree(positions_old_d);
    cudaFree(velocities_old_d);
    cudaFree(forces_d);
    cudaFree(accelerations_d);
    cudaFree(masses_d);
    cudaFree(positions_new_d);
    cudaFree(velocities_new_d);
    cudaFree(cells_d);
    cudaFree(particleCell_d);
    
    
    return 0;
}