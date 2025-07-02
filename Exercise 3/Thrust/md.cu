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
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>

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


    // cout
    std::cout << "Data copied to device" << std::endl;
    
    // create all data arrays on device using thrust
    thrust::device_vector<double> positions_old_T(positions_old);
    thrust::device_vector<double> velocities_old_T(velocities_old);
    thrust::device_vector<double> forces_T(forces);
    thrust::device_vector<double> accelerations_T(accelerations);
    thrust::device_vector<double> masses_T(masses);
    thrust::device_vector<double> positions_new_T(positions_new);
    thrust::device_vector<double> velocities_new_T(velocities_new);
    thrust::device_vector<int> particleCell_T(numParticles);
    thrust::device_vector<int> cells_T(total_cells);
    // copy the data to device
    thrust::copy(positions_old.begin(), positions_old.end(), positions_old_T.begin());
    thrust::copy(velocities_old.begin(), velocities_old.end(), velocities_old_T.begin());
    thrust::copy(forces.begin(), forces.end(), forces_T.begin());   
    thrust::copy(accelerations.begin(), accelerations.end(), accelerations_T.begin());
    thrust::copy(masses.begin(), masses.end(), masses_T.begin());
    thrust::copy(positions_new.begin(), positions_new.end(), positions_new_T.begin());
    thrust::copy(velocities_new.begin(), velocities_new.end(), velocities_new_T.begin());
    thrust::fill(particleCell_T.begin(), particleCell_T.end(), -1); //
    

    thrust::fill(cells_T.begin(), cells_T.end(), -1); // fill cells with -1
    thrust_computeParticleCells(
                positions_old_T,
                cells_T,
                particleCell_T,
                numParticles,
                num_cells,
                total_cells,
                cell_size
            );




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

        thrust_update_positions(positions_new_T, positions_old_T, velocities_old_T, accelerations_T, timeStepLength, boxSize, numParticles);

        auto positions_end = std::chrono::high_resolution_clock::now();
        positions_total += positions_end - positions_start;
        positions_count++;
        
        auto velocities_start = std::chrono::high_resolution_clock::now();

        thrust_update_velocities(velocities_new_T, velocities_old_T, accelerations_T, timeStepLength, numParticles);

        auto velocities_end = std::chrono::high_resolution_clock::now();
        velocities_total += velocities_end - velocities_start;
        velocities_count++;
        //std::cout<< "Positions updated"<< std::endl;

        thrust::swap(positions_old_T, positions_new_T);
        thrust::swap(velocities_old_T, velocities_new_T);
        // Build up linked neighbor list
        // Reset cells to -1
        // always use the cells if (useAcc == 1) {
        if (useAcc == 1 && timestep % 10 == 0) { // reset cells every 10th timestep
            thrust::fill(cells_T.begin(), cells_T.end(), -1); // fill cells with -1
            thrust::fill(particleCell_T.begin(), particleCell_T.end(), -1); // fill particleCell with -1
            thrust_computeParticleCells(
                positions_old_T,
                cells_T,
                particleCell_T,
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
        thrust_acceleration_updater(accelerations_T, positions_old_T, forces_T, masses_T, sigma, epsilon, boxSize, cutoffRadius, numParticles, cells_T, particleCell_T, num_cells);

        auto forces_and_accelerations_end = std::chrono::high_resolution_clock::now();
        forces_and_accelerations_total += forces_and_accelerations_end - forces_and_accelerations_start;
        forces_and_accelerations_count++;
        // cout
        //std::cout << "updating velocities"<< std::endl;
        // update velocities
        auto update_velocities_start = std::chrono::high_resolution_clock::now();

        thrust_update_velocities(velocities_new_T, velocities_old_T, accelerations_T, timeStepLength, numParticles);

        auto update_velocities_end = std::chrono::high_resolution_clock::now();
        velocities_total += update_velocities_end - update_velocities_start;
        velocities_count++;

        // transfer new positions and velocities to old positions
        thrust::swap(velocities_old_T, velocities_new_T);
        //std::cout<< "Loop complete"<< std::endl;
        // print to file every printInterval steps
        if (printInterval > 0 && timestep % printInterval == 0) {
            // cout
            std::cout << "writing iteration " << timestep<<" to file"<< std::endl;

            thrust::copy(positions_old_T.begin(), positions_old_T.end(), positions_old.begin());
            thrust::copy(velocities_old_T.begin(), velocities_old_T.end(), velocities_old.begin());
    
            std::string outputFile = "output/cuda-output" + std::to_string(timestep / printInterval) + ".vtk";
            writeVTKFile(outputFile, positions_old, velocities_old, masses);
        }
    }
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_elapsed = total_end - total_start;
    std::cout << "Elapsed time on GPU: " << std::chrono::duration<double>(total_elapsed).count() << " seconds" << std::endl;
    std::cout << "Time for "<< positions_count<< " calls of the updating positions function: " << std::chrono::duration<double>(positions_total).count() << " seconds" << std::endl;
    std::cout << "Time for "<< velocities_count<< " calls of the updating velocities function: " << std::chrono::duration<double>(velocities_total).count() << " seconds" << std::endl;
    std::cout << "Time for "<< forces_and_accelerations_count<< " calls of the forces and accelerations function: " << std::chrono::duration<double>(forces_and_accelerations_total).count() << " seconds" << std::endl;
    std::cout << "Total time for GPU simulation: " << std::chrono::duration<double>(total_elapsed).count() << " seconds" << std::endl;  
    
    std::cout << "Simulation complete!" << std::endl;

    double total_elapsed_seconds = std::chrono::duration<double>(total_elapsed).count();
    // Append runtime to a file for later plotting
    std::ofstream runtime_file("runtimes.txt", std::ios::app);
    if (runtime_file.is_open()) {
        runtime_file << "----------------------------------------" << std::endl;
        runtime_file << "Configuration File: " << configFile << std::endl;
        runtime_file << "Time for "<< positions_count<< " calls of the updating positions function: " << std::chrono::duration<double>(positions_total).count() << " seconds" << std::endl;
        runtime_file << "Time for "<< velocities_count<< " calls of the updating velocities function: " << std::chrono::duration<double>(velocities_total).count() << " seconds" << std::endl;
        runtime_file << "Time for "<< forces_and_accelerations_count<< " calls of the forces and accelerations function: " << std::chrono::duration<double>(forces_and_accelerations_total).count() << " seconds" << std::endl;
        runtime_file << "Total time for GPU simulation: " << total_elapsed_seconds << " seconds" << std::endl;
        runtime_file << "----------------------------------------" << std::endl;
        runtime_file.close();
    } else {
        std::cerr << "Could not open runtimes.txt for writing!" << std::endl;
    }

    
    return 0;
}