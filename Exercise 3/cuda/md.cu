//write a quick test for the parser

#include <iostream>
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

    
    // Call the parser
    parseConfigFile(configFile, positions_old, velocities_old, masses, boxSize, cutoffRadius, timeStepLength, timeStepCount, sigma, epsilon, printInterval);

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
    
    // print the positions and velocities
    std::cout << "Positions: ";
    for (size_t i = 0; i < positions_old.size(); ++i) {
        std::cout << positions_old[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Velocities: ";
    for (size_t i = 0; i < velocities_old.size(); ++i) {
        std::cout << velocities_old[i] << " ";
    }

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
    if (boxSize > 0.0) {
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
    
    //for each element calculate the forces
    for (size_t i = 0; i < positions_old.size(); i += 3) {
        force_updater(i, positions_old, forces, sigma, epsilon, boxSize);
        acceleration_calculator(i, forces, accelerations, masses);

    }

    //initialize the values on device
    double *positions_old_d, *velocities_old_d, *forces_d, *accelerations_d, *masses_d;
    double *positions_new_d, *velocities_new_d;
   

    // allocate memory on device
    CUDA_CHECK( cudaMalloc(&positions_old_d, positions_old.size() * sizeof(double)));
    CUDA_CHECK( cudaMalloc(&velocities_old_d, velocities_old.size() * sizeof(double)));
    CUDA_CHECK( cudaMalloc(&forces_d, forces.size() * sizeof(double)));
    CUDA_CHECK( cudaMalloc(&accelerations_d, accelerations.size() * sizeof(double)));
    CUDA_CHECK( cudaMalloc(&masses_d, masses.size() * sizeof(double)));
    CUDA_CHECK( cudaMalloc(&positions_new_d, positions_new.size() * sizeof(double)));
    CUDA_CHECK( cudaMalloc(&velocities_new_d, velocities_new.size() * sizeof(double)));


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
    dim3 gridSize((masses.size() + blockSize.x - 1) / blockSize.x);
    // launch the kernel to check periodic boundaries


    //loop for GPU
    std::cout << "Starting time loop for GPU ..." << std::endl;
    // time loop
    auto start = std::chrono::high_resolution_clock::now();
    for (int timestep = 0; timestep < timeStepCount; ++timestep) {
        // cout
        //std::cout << "Time step: " << timestep << std::endl;
        //std::cout<< "updating positions and velocities"<< std::endl;
        update_positions_d<<<gridSize, blockSize>>>( positions_new_d, positions_old_d, 
            velocities_old_d, accelerations_d, timeStepLength, boxSize, numParticles);
        CUDA_CHECK(cudaGetLastError());
        //std::cout<< "Positions updated"<< std::endl;
        
        update_velocities_d<<<gridSize, blockSize>>>(velocities_new_d, velocities_old_d,
             accelerations_d, timeStepLength, numParticles);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaDeviceSynchronize());

        //std::cout<< "Update complete, swapping"<< std::endl;
        std::swap(positions_old_d, positions_new_d);
        std::swap(velocities_old_d, velocities_new_d);
        // cout
        //std::cout << " Calculating Forces and accelerations"<< std::endl;
        // update forces and accelerations
        acceleration_updater_d<<<gridSize, blockSize>>>(accelerations_d,positions_old_d, 
            forces_d, masses_d, sigma, epsilon, boxSize, cutoffRadius, numParticles);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaDeviceSynchronize());
        // cout
        //std::cout << "updating velocities"<< std::endl;
        // update velocities
        update_velocities_d<<<gridSize, blockSize>>>(velocities_new_d, velocities_old_d,
             accelerations_d, timeStepLength, numParticles);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaDeviceSynchronize());
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
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    std::cout << "Elapsed time on GPU: " << elapsed.count() << " seconds" << std::endl;
    
    std::cout << "Simulation complete!" << std::endl;

    // Free device memory
    cudaFree(positions_old_d);
    cudaFree(velocities_old_d);
    cudaFree(forces_d);
    cudaFree(accelerations_d);
    cudaFree(masses_d);
    cudaFree(positions_new_d);
    cudaFree(velocities_new_d);
    
    
    return 0;
}