//import necessary libraries
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <fstream>



// function to check and update periodic boundaries for each particle (can be globalized)
void checkPeriodicBoundaries(double & x, double & y, double & z, double boxSize) {
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
std::vector<double> ij_force_calculator(std::vector<double> distances, double sigma, double epsilon) {
    double r = distance_size(distances);
    if (r == 0.0) {
        //std::cerr << "Error: Zero distance between particles!" << std::endl;
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
            std::vector<double> force = ij_force_calculator(distances, sigma, epsilon);
            forces[particle_idx] += force[0];
            forces[particle_idx + 1] += force[1];
            forces[particle_idx + 2] += force[2];
        }
    }    
}


// function to calculate the acceleration for a given particle
// (can be globalized)
void acceleration_calculator (int idx ,std::vector<double> & forces, std::vector<double> & acceleration, std::vector<double> mass ){
    acceleration[idx+0]= forces[idx+0]/mass[idx/3];
    acceleration[idx+1]= forces[idx+1]/mass[idx/3];
    acceleration[idx+2]= forces[idx+2]/mass[idx/3];
}

