
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


int main(){
// write initial state to file
    std::string outputFile = "output/output0.vtk";
    writeVTKFile(outputFile, positions_old, velocities_old, masses);
    // cout
    //std::cout << "Initial state written to file " << outputFile << std::endl;
    // cout

    std::cout << "Starting time loop for CPU ..." << std::endl;
    // time loop
    auto start = std::chrono::high_resolution_clock::now();
    for (int timestep = 0; timestep < timeStepCount; ++timestep) {
        // cout
    //    std::cout << "Time step: " << timestep << std::endl;
        // cout
    //    std::cout << "updating positions and velocities"<< std::endl;
        // Update positions and velocities
        for (size_t i = 0; i < positions_old.size(); i += 3) {
            positions_new[i] = positions_old[i] + velocities_old[i] * timeStepLength + 0.5 * accelerations[i] * timeStepLength * timeStepLength;
            positions_new[i + 1] = positions_old[i + 1] + velocities_old[i + 1] * timeStepLength + 0.5 * accelerations[i + 1] * timeStepLength * timeStepLength;
            positions_new[i + 2] = positions_old[i + 2] + velocities_old[i + 2] * timeStepLength + 0.5 * accelerations[i + 2] * timeStepLength * timeStepLength;

            // check periodic boundaries
            checkPeriodicBoundaries(positions_new[i], positions_new[i + 1], positions_new[i + 2], boxSize);
            
            velocities_new[i] = velocities_old[i] + 0.5 * accelerations[i] * timeStepLength;
            velocities_new[i + 1] = velocities_old[i + 1] + 0.5 * accelerations[i + 1] * timeStepLength;
            velocities_new[i + 2] = velocities_old[i + 2] + 0.5 * accelerations[i + 2] * timeStepLength;
        }
        // cout
    //    std::cout << "Forces and accelerations"<< std::endl;
        // update forces and accelerations
        for (size_t i = 0; i < positions_old.size(); i += 3) {
            force_updater(i, positions_new, forces, sigma, epsilon, boxSize);
            acceleration_calculator(i, forces, accelerations, masses);
        }

        // transfer new velocities to old velocities
        std::swap(velocities_old, velocities_new);
        // cout
    //    std::cout << "updating velocities"<< std::endl;
        // update velocities
        for (size_t i = 0; i < positions_old.size(); i += 3) {
            velocities_new[i] = velocities_old[i] + 0.5 * accelerations[i]  * timeStepLength;
            velocities_new[i + 1] = velocities_old[i + 1] + 0.5 * accelerations[i + 1]  * timeStepLength;
            velocities_new[i + 2] = velocities_old[i + 2] + 0.5 * accelerations[i + 2]  * timeStepLength;
        }

        // transfer new positions and velocities to old positions
        std::swap(positions_old, positions_new);
        std::swap(velocities_old, velocities_new);

        // print to file every printInterval steps
        if (printInterval > 0 && timestep % printInterval == 0) {
            // cout
            std::cout << "writing iteration " << timestep<<" to file"<< std::endl;
        
            std::string outputFile = "output/output" + std::to_string(timestep / printInterval) + ".vtk";
            writeVTKFile(outputFile, positions_old, velocities_old, masses);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time on cpu: " << elapsed.count() << " seconds" << std::endl;

return 0;
}