#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <filesystem>
#include "parser.hpp"

// Generate points on a sphere using spherical coordinates
std::vector<double> generateSphereParticles(int numParticles, float radius) {
    std::vector<double> particles;
    for (int i = 0; i < numParticles; ++i) {
        float phi = acos(1 - 2 * (i + 0.5f) / numParticles);  // from 0 to pi
        float theta = M_PI * (1 + std::sqrt(5)) * i;           // golden angle

        float x = radius * std::sin(phi) * std::cos(theta);
        float y = radius * std::sin(phi) * std::sin(theta);
        float z = radius * std::cos(phi);

        particles.push_back(x);
        particles.push_back(y);
        particles.push_back(z);
    }
    return particles;
}


int main() {
    int numParticles = 100;
    float radius = 5.0f;

    auto particles = generateSphereParticles(numParticles, radius);
    std::vector<double> velocities(numParticles*3, 0.0); // 3 components per particle
    std::vector<double> masses(numParticles, 1.0); // 1 mass per particle, initialized to 1.0
    std::string filename = "output/sphere_particles.vtk";
     writeVTKFile( filename, particles, velocities, masses);

    std::cout << "VTK file written to " << filename << "\n";
    return 0;
}
