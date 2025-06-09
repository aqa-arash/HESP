//write a quick test for the parser

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <math.h>
#include <cmath>
#include "parser.hpp"

// function to check and update periodic boundaries for each particle (can be globalized)
inline void checkPeriodicBoundaries(double & x, double & y, double & z, double boxSize) {
    x = fmod(fmod(x, boxSize) + boxSize, boxSize);
    y = fmod(fmod(y, boxSize) + boxSize, boxSize);
    z = fmod(fmod(z, boxSize) + boxSize, boxSize);
}

// function to calculate the periodic distance between two particles
// (can be globalized)
std::vector<double> periodic_distance(double x1, double y1, double z1, double x2, double y2, double z2, double boxSize) {
    std::vector<double> distances(3),pbc_distance(3);
    distances[0] = x1 - x2;
    distances[1] = y1 - y2;
    distances[2] = z1 - z2;
    // apply periodic boundary conditions
    pbc_distance[0] = distances[0] - boxSize * std::round(distances[0] / boxSize);
    pbc_distance[1] = distances[1] - boxSize * std::round(distances[1] / boxSize);
    pbc_distance[2] = distances[2] - boxSize * std::round(distances[2] / boxSize);
    return pbc_distance;
}

// function to calculate the distance size
// (can be globalized)
double distance_size(std::vector<double> & distances) {
    return std::sqrt(distances[0] * distances[0] + distances[1] * distances[1] + distances[2] * distances[2]);
}

// function to calculate the forces between two particles
// (can be globalized)
std::vector<double> ij_forces(std::vector<double> distances, double sigma, double epsilon) {
    double r = distance_size(distances);
    if (r == 0.0) {
        std::cerr << "Error: Zero distance between particles!" << std::endl;
        return {0.0, 0.0, 0.0};
    }
    else if (r>2.5*sigma){ // cut off distance
        return {0.0, 0.0, 0.0};
    }
    else {
    double r6 = std::pow(sigma / r, 6);
    double force_multiplier = 24 * epsilon * r6 * (2 * r6 - 1) /(r*r);
    std::vector<double> forces(3);
    forces[0] = force_multiplier * distances[0];
    forces[1] = force_multiplier * distances[1];
    forces[2] = force_multiplier * distances[2];
    return forces;
}
}

// function to calculate the forces for a given particle
// (can be globalized)
void force_updater (size_t particle_idx, std::vector<double>& positions, std::vector<double>& forces, double sigma, double epsilon, double boxSize) {
    forces[particle_idx] = 0.0;
    forces[particle_idx + 1] = 0.0;
    forces[particle_idx + 2] = 0.0;
    // Calculate forces for the particle at particle_idx
    for (size_t j = 0; j < positions.size(); j += 3) {
        if (j != particle_idx) {
            std::vector<double> distances = periodic_distance(positions[particle_idx], positions[particle_idx + 1], positions[particle_idx + 2],
                positions[j], positions[j + 1], positions[j + 2], boxSize);
            std::vector<double> force = ij_forces(distances, sigma, epsilon);
            forces[particle_idx] += force[0];
            forces[particle_idx + 1] += force[1];
            forces[particle_idx + 2] += force[2];
        }
    }    
    // check sanity of forces (can be deleted)
    if (std::isnan(forces[particle_idx]) || std::isnan(forces[particle_idx + 1]) || std::isnan(forces[particle_idx + 2])) {
        std::cerr << "Error: Forces are NaN for particle " << particle_idx / 3 << std::endl;
        forces[particle_idx] = 0.0;
        forces[particle_idx + 1] = 0.0;
        forces[particle_idx + 2] = 0.0;
    } else if (std::isinf(forces[particle_idx]) || std::isinf(forces[particle_idx + 1]) || std::isinf(forces[particle_idx + 2])) {
        std::cerr << "Error: Forces are Inf for particle " << particle_idx / 3 << std::endl;
        forces[particle_idx] = 0.0;
        forces[particle_idx + 1] = 0.0;
        forces[particle_idx + 2] = 0.0;
}}

// function to calculate the acceleration for a given particle
// (can be globalized)
void acceleration_calculator (int idx ,std::vector<double> & forces, std::vector<double> & acceleration, std::vector<double> mass ){
    acceleration[idx+0]= forces[idx+0]/mass[idx/3];
    acceleration[idx+1]= forces[idx+1]/mass[idx/3];
    acceleration[idx+2]= forces[idx+2]/mass[idx/3];
}





int main() {
    // Test file name
    std::string configFile = "config.txt";

    // Variables to hold parsed data
    std::vector<double> positions_old, velocities_old, masses, position_new, velocities_new, accelerations, forces;
    
    double timeStepLength = 0.0, timeStepCount = 0.0, sigma = 0.0, epsilon = 0.0, boxSize = 0.0;
    int printInterval = 0;

    
    // Call the parser
    parseConfigFile(configFile, positions_old, velocities_old, masses, boxSize, timeStepLength, timeStepCount, sigma, epsilon, printInterval);

        // Output the parsed data
    std::cout << "Parsed Data:" << std::endl;
    std::cout << "Time Step Length: " << timeStepLength << std::endl;
    std::cout << "Time Step Count: " << timeStepCount << std::endl;
    std::cout << "Sigma: " << sigma << std::endl;
    std::cout << "Epsilon: " << epsilon << std::endl;
    std::cout << "Box Size: " << boxSize << std::endl;
    std::cout << "Print Interval: " << printInterval << std::endl;  
    std::cout << "Number of particles: " << positions_old.size()/3 << std::endl;

    // Check if the parsed data is valid
    if (sigma <= 0.0 || epsilon <= 0.0) {
    std::cerr << "Error: Invalid sigma or epsilon values. Exiting simulation." << std::endl;
    return -1;
}
    // the minimum x is 0.0
    // check if the positions are out of bounds
    if (*std::min_element(positions_old.begin(), positions_old.end()) < 0.0) {
        std::cerr << "Error: Positions are out of bounds!" << std::endl;
        return -1;
    }

    //set box size to the maximum position + 0.5
    accelerations.resize(positions_old.size(), 0.0);
    forces.resize(positions_old.size(), 0.0);
    position_new.resize(positions_old.size(), 0.0);
    velocities_new.resize(positions_old.size(), 0.0);
    
    //for each element calculate the forces
    for (size_t i = 0; i < positions_old.size(); i += 3) {
        force_updater(i, positions_old, forces, sigma, epsilon, boxSize);
        acceleration_calculator(i, forces, accelerations, masses);
        // check sanity of accelerations
        if (std::isnan(accelerations[i]) || std::isnan(accelerations[i + 1]) || std::isnan(accelerations[i + 2])) {
            std::cerr << "Error: Accelerations are NaN for particle " << i / 3 << std::endl;
            accelerations[i] = 0.0;
            accelerations[i + 1] = 0.0;
            accelerations[i + 2] = 0.0;
        } else if (std::isinf(accelerations[i]) || std::isinf(accelerations[i + 1]) || std::isinf(accelerations[i + 2])) {
            std::cerr << "Error: Accelerations are Inf for particle " << i / 3 << std::endl;
            accelerations[i] = 0.0;
            accelerations[i + 1] = 0.0;
            accelerations[i + 2] = 0.0;
        }
    }

    // cout
    //std::cout << "Initial forces calculated " << std::endl;
    
    // write initial state to file
    std::string outputFile = "output/output0.vtk";
    writeVTKFile(outputFile, positions_old, velocities_old, masses);
    // cout
    //std::cout << "Initial state written to file " << outputFile << std::endl;
    // cout
    std::cout << "Starting time loop..." << std::endl;
    // time loop
    for (int timestep = 0; timestep < timeStepCount; ++timestep) {
        // cout
    //    std::cout << "Time step: " << timestep << std::endl;
        // cout
    //    std::cout << "updating positions and velocities"<< std::endl;
        // Update positions and velocities
        for (size_t i = 0; i < positions_old.size(); i += 3) {
            position_new[i] = positions_old[i] + velocities_old[i] * timeStepLength + 0.5 * accelerations[i] * timeStepLength * timeStepLength;
            position_new[i + 1] = positions_old[i + 1] + velocities_old[i + 1] * timeStepLength + 0.5 * accelerations[i + 1] * timeStepLength * timeStepLength;
            position_new[i + 2] = positions_old[i + 2] + velocities_old[i + 2] * timeStepLength + 0.5 * accelerations[i + 2] * timeStepLength * timeStepLength;

            // check periodic boundaries
            checkPeriodicBoundaries(position_new[i], position_new[i + 1], position_new[i + 2], boxSize);

            velocities_new[i] = velocities_old[i] + 0.5 * accelerations[i] * timeStepLength;
            velocities_new[i + 1] = velocities_old[i + 1] + 0.5 * accelerations[i + 1] * timeStepLength;
            velocities_new[i + 2] = velocities_old[i + 2] + 0.5 * accelerations[i + 2] * timeStepLength;
        }
        // cout
    //    std::cout << "Forces and accelerations"<< std::endl;
        // update forces and accelerations
        for (size_t i = 0; i < positions_old.size(); i += 3) {
            force_updater(i, position_new, forces, sigma, epsilon, boxSize);
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
        std::swap(positions_old, position_new);
        std::swap(velocities_old, velocities_new);

        // print to file every printInterval steps
        if (printInterval > 0 && timestep % printInterval == 0) {
            // cout
            std::cout << "writing iteration " << timestep<<" to file"<< std::endl;
        
            std::string outputFile = "output/output" + std::to_string(timestep / printInterval) + ".vtk";
            writeVTKFile(outputFile, positions_old, velocities_old, masses);
        }
    }


    return 0;
}