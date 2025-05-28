//write a code to create a vtk file with the made up positions, velocities, and masses

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "parser.hpp"
#include <cmath>
#include <random>


int main () {

    // declare variables
    std::vector<double> positions, velocities, masses;
    double boxSize = 17.0;
    int numParticles = 3;
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-0.1, 0.1);

    // allocate memory for positions, velocities, and masses
    positions.reserve(numParticles * 3);
    velocities.reserve(numParticles * 3);
    masses.reserve(numParticles);
    
    int temp= numParticles;

    // initialize positions in a cubic grid
    double spacing = boxSize / std::cbrt(numParticles); // Spacing between particles
    for (int x = 0; x < std::cbrt(numParticles); ++x) { // check if the bound is correct 
        if (temp<1) break;
        for (int y = 0; y < std::cbrt(numParticles); ++y) {
            if (temp<1) break;
            for (int z = 0; z < std::cbrt(numParticles); ++z) {
                if (temp<1) break;
                positions.push_back(x * spacing);
                positions.push_back(y * spacing);
                positions.push_back(z * spacing);
                --temp;
            }
        }
    }
    // initialize velocities with random values
    for (size_t i = 0; i < positions.size(); ++i) {
        velocities.push_back(distribution(generator));
    }
    // initialize masses with 1 values (can be modified later)
    for (int i = 0; i < numParticles; ++i) {
         // set mass to 1.0 for all particles
        masses.push_back(1.0);
    }

    // write to file
    std::string file ="initial/initial1000.vtk";
    writeVTKFile(file, positions, velocities, masses);
    
}