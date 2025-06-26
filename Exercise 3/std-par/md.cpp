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
#include <algorithm>
#include <execution>
#include <ranges>
#include <thread> // for multithreading (if needed)


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
    std::vector<std::atomic<int>>cells(total_cells);
    std::vector<int> particleCell(numParticles, 0);
    // make this run on cpu 
    resetCells_h(cells); // reset cells to -1
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
    int velocities_count = 0;
    auto forces_and_accelerations_total = std::chrono::duration<double>::zero();
    int forces_and_accelerations_count = 0;

    std::thread writer_thread; // Thread for writing VTK files

    auto total_start = std::chrono::high_resolution_clock::now();
    for (int timestep = 0; timestep < timeStepCount; ++timestep) {
        // cout
        //std::cout << "Time step: " << timestep << std::endl;
        //std::cout<< "updating positions and velocities"<< std::endl;
        auto positions_start = std::chrono::high_resolution_clock::now();
        // make this run on cpu 
        update_positions( positions_new, positions_old, 
            velocities_old, accelerations, timeStepLength, boxSize, numParticles);
        auto positions_end = std::chrono::high_resolution_clock::now();
        positions_total += positions_end - positions_start;
        positions_count++;
        std::swap(positions_old, positions_new);

        // make this run on cpu 
        auto velocities_start = std::chrono::high_resolution_clock::now();
        update_velocities(velocities_new, velocities_old,
             accelerations, timeStepLength, numParticles);

        auto velocities_end = std::chrono::high_resolution_clock::now();
        velocities_total += velocities_end - velocities_start;
        velocities_count++;

        std::swap(velocities_old, velocities_new);

        //std::cout<< "Positions updated"<< std::endl;
        //std::cout<< "Update complete, swapping"<< std::endl;
        
        // Build up linked neighbor list
        // Reset cells to -1
        // always use the cells if (useAcc == 1) {
        if (useAcc == 1 && timestep % 10 == 0) { // reset cells every 10th timestep
            resetCells_h(cells); // should we launch less blocks ? 
            computeParticleCells_h(
                positions_old,
                cells,
                particleCell,
                num_cells,
                cell_size
            );
            
        }
        
        auto forces_and_accelerations_start = std::chrono::high_resolution_clock::now();
        //std::cout << " Calculating Forces and accelerations"<< std::endl;
        // update forces and accelerations
        // make this run on cpu
        acceleration_updater(accelerations,positions_old, 
            forces, masses, sigma, epsilon, boxSize, cutoffRadius, numParticles, cells, particleCell, num_cells);
            
        auto forces_and_accelerations_end = std::chrono::high_resolution_clock::now();
        forces_and_accelerations_total += forces_and_accelerations_end - forces_and_accelerations_start;
        forces_and_accelerations_count++;
        // cout
        //std::cout << "updating velocities"<< std::endl;
        // update velocities
        auto update_velocities_start = std::chrono::high_resolution_clock::now();
        update_velocities(velocities_new, velocities_old,
             accelerations, timeStepLength, numParticles);
        
        auto update_velocities_end = std::chrono::high_resolution_clock::now();
        velocities_total += update_velocities_end - update_velocities_start;
        velocities_count++;

        // transfer new positions and velocities to old positions
        
        std::swap(velocities_old, velocities_new);
        
        //std::cout<< "Loop complete"<< std::endl;
        // print to file every printInterval steps
        if (printInterval > 0 && timestep % printInterval == 0) {
            // cout
            if (writer_thread.joinable()){
                writer_thread.join(); // wait for the previous write to finish
            }
            // !!!!!!! the write can run in a separate thread !!!!!!!!!!!!
            std::cout << "writing iteration " << timestep<<" to file"<< std::endl;
            std::vector<double> positions_old_copy = positions_old;
            std::vector<double> velocities_old_copy = velocities_old;        
            std::string outputFile = "output/cuda-output" + std::to_string(timestep / printInterval) + ".vtk";
            writer_thread = std::thread(writeVTKFile, outputFile, positions_old_copy, velocities_old_copy, masses);
        }
    }
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_elapsed = total_end - total_start;
    
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
        // Join the writer thread before ending the program
    if (writer_thread.joinable()) {
        writer_thread.join(); // Wait for the final write to complete
    }
    
    return 0;
}