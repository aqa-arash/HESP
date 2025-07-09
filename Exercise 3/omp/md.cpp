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
#include <omp.h>


int main(int argc, char** argv) {
    // GPU test at the beginning
    std::cout << "Testing GPU availability..." << std::endl;
    #pragma omp target
    {
        if (omp_is_initial_device()) {
            printf("Running on CPU\n");
            printf("Number of threads: %d\n", omp_get_max_threads());
        } else {
            printf("Running on GPU\n");
        }
    }

    #pragma omp target teams distribute parallel for
    for (int i = 0; i < 1; i++) {
        int nteams = omp_get_num_teams();
        int tlimit = omp_get_thread_limit();

        printf("Number of teams: %d, Thread limit: %d\n", nteams, tlimit);
    }
    
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
    // check if positions are valid 
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

    //empty the output directory if printInterval is greater than 0
    if (printInterval > 0) {
        std::system("rm -rf output/*");
        std::cout << "Output directory cleared." << std::endl;
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
    std::vector<int>cells(total_cells, -1);
    std::vector<int> particleCell(numParticles, 0);
    // make this run on cpu 
    computeParticleCells_h(
                positions_old,
                cells,
                particleCell,
                num_cells,
                cell_size
            );

    std::cout << "Number of cells: " << total_cells << std::endl;
    std::cout << "Cell size: " << cell_size << std::endl;
    // print the first 10 cells
    std::cout << "First 10 cells: ";
    for (int i = 0; i < std::min(10, total_cells); ++i) {
        std::cout << cells[i] << " ";
    }
    std::cout << std::endl;
    // print first ten particleCell
    std::cout << "\nFirst 10 particleCell: ";
    for (int i = 0; i < std::min(10, numParticles); ++i) {
        std::cout << particleCell[i] << " ";
    }
    std::cout << std::endl;
    
    //loop for GPU
    std::cout << "Starting time loop " << std::endl;
    // time loop
    auto positions_total = std::chrono::duration<double>::zero();
    int positions_count = 0;
    auto velocities_total = std::chrono::duration<double>::zero();
    auto total_elapsed = std::chrono::duration<double>::zero();
    int velocities_count = 0;
    auto forces_and_accelerations_total = std::chrono::duration<double>::zero();
    int forces_and_accelerations_count = 0;

    // Before the time loop: One-time data transfer to GPU
    double* pos_new = positions_new.data();
    double* pos_old = const_cast<double*>(positions_old.data());
    double* vel_new = velocities_new.data();
    double* vel_old = const_cast<double*>(velocities_old.data());
    double* acc = const_cast<double*>(accelerations.data());
    double* force = forces.data();
    double* mass = const_cast<double*>(masses.data());
    int* cell_data = cells.data();
    int* particle_cell = particleCell.data();

    int total_size = numParticles * 3;
    size_t cell_data_size = cells.size();
    int N = particleCell.size();

    // One-time GPU data transfer for the entire simulation
    #pragma omp target data \
        map(tofrom: pos_new[0:total_size], pos_old[0:total_size]) \
        map(tofrom: vel_new[0:total_size], vel_old[0:total_size]) \
        map(tofrom: acc[0:total_size], force[0:total_size]) \
        map(to: mass[0:numParticles]) \
        map(tofrom: cell_data[0:cell_data_size], particle_cell[0:N])
    {
        auto total_start = std::chrono::high_resolution_clock::now();
        // Main simulation loop (runs sequentially on GPU)
        for (int timestep = 0; timestep < timeStepCount; ++timestep) {
            
            auto positions_start = std::chrono::high_resolution_clock::now();
            
            // Position update (parallelized kernel)
            update_positions_kernel(pos_new, pos_old, vel_old, acc, timeStepLength, total_size);
            
            // Apply periodic boundaries after position calculation
            if (boxSize > 0.000000001) {
                apply_periodic_boundaries_kernel(pos_new, boxSize, numParticles);
            }
            
            auto positions_end = std::chrono::high_resolution_clock::now();
            positions_total += positions_end - positions_start;
            positions_count++;
            
            // Swap operations on GPU pointers (only pointer exchange)
            double* temp_pos = pos_old;
            pos_old = pos_new;
            pos_new = temp_pos;

            auto velocities_start = std::chrono::high_resolution_clock::now();
            
            // Velocity update (parallelized kernel)
            update_velocities_kernel(vel_new, vel_old, acc, timeStepLength, total_size);
            
            auto velocities_end = std::chrono::high_resolution_clock::now();
            velocities_total += velocities_end - velocities_start;
            velocities_count++;

            // Swap operations on GPU pointers
            double* temp_vel = vel_old;
            vel_old = vel_new;
            vel_new = temp_vel;
            
            // Update cell lists every 10 time steps
            if (useAcc == 1 && timestep % 10 == 0) {
                resetCells_kernel(cell_data, cell_data_size);
                computeParticleCells_kernel(
                    pos_old, // Use current positions
                    cell_data,
                    particle_cell,
                    N,
                    num_cells,
                    cell_size,
                    total_cells
                );
            }
            
            auto forces_and_accelerations_start = std::chrono::high_resolution_clock::now();
            
            // Calculate forces and accelerations (parallelized kernel)
            acceleration_updater_kernel(acc, pos_old, force, mass, cell_data, particle_cell,
                                    sigma, epsilon, boxSize, cutoffRadius, numParticles, num_cells);
                
            auto forces_and_accelerations_end = std::chrono::high_resolution_clock::now();
            forces_and_accelerations_total += forces_and_accelerations_end - forces_and_accelerations_start;
            forces_and_accelerations_count++;
            
            auto update_velocities_start = std::chrono::high_resolution_clock::now();
            
            // Second velocity update (parallelized kernel)
            update_velocities_kernel(vel_new, vel_old, acc, timeStepLength, total_size);
            
            auto update_velocities_end = std::chrono::high_resolution_clock::now();
            velocities_total += update_velocities_end - update_velocities_start;
            velocities_count++;

            // Swap operations on GPU pointers
            temp_vel = vel_old;
            vel_old = vel_new;
            vel_new = temp_vel;

            // VTK output at specific intervals
            if (printInterval > 0 && timestep % printInterval == 0) {
                std::cout << "writing iteration " << timestep << " to file" << std::endl;
                
                // Create temporary CPU copies for VTK output
                // (GPU data is automatically copied back to CPU for this operation)
                #pragma omp target update from(pos_old[0:total_size], vel_old[0:total_size])
                
                std::string outputFile = "output/cuda-output" + std::to_string(timestep / printInterval) + ".vtk";
                writeVTKFileNew(outputFile, pos_old, vel_old, mass, numParticles);
            }
        }
        auto total_end = std::chrono::high_resolution_clock::now();
        total_elapsed = total_end - total_start;
    } // End of target data block - data is transferred back to CPU here

    
    std::cout << "Time for "<< positions_count<< " calls of the updating positions function: " << std::chrono::duration<double>(positions_total).count() << " seconds" << std::endl;
    std::cout << "Time for "<< velocities_count<< " calls of the updating velocities function: " << std::chrono::duration<double>(velocities_total).count() << " seconds" << std::endl;
    std::cout << "Time for "<< forces_and_accelerations_count<< " calls of the forces and accelerations function: " << std::chrono::duration<double>(forces_and_accelerations_total).count() << " seconds" << std::endl;
    std::cout << "Total time for simulation: " << std::chrono::duration<double>(total_elapsed).count() << " seconds" << std::endl;  
    
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