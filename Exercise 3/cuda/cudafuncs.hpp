#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cuda.h>


__device__ void deviceAdd(double *z, double x, double y ) {
    *z = x + y;
    }

// boundary check for the particles on device
__device__ void checkPeriodicBoundaries_d(double * x, double * y, double * z, double boxSize) {
    *x = fmod(fmod(*x, boxSize) + boxSize, boxSize);
    *y = fmod(fmod(*y, boxSize) + boxSize, boxSize);
    *z = fmod(fmod(*z, boxSize) + boxSize, boxSize);
}


// function to calculate the distance between two particles on the device
__device__ void periodic_distance_d(double * distance , double * x1, double * y1, double * z1, double * x2, double * y2, double * z2, double boxSize) {
    // calculate the distance between two particles
    double dx = *x1 - *x2;
    double dy = *y1 - *y2;
    double dz = *z1 - *z2;
    // apply periodic boundary conditions
    dx -= boxSize * round(dx / boxSize);
    dy -= boxSize * round(dy / boxSize);
    dz -= boxSize * round(dz / boxSize);
    // store the distance in the output array
    distance[0] = dx;
    distance[1] = dy;
    distance[2] = dz;
}


// function to calculate the distance size on device
__device__ void distance_size_d(double * size,double * distances) {
    // calculate the distance size
    *size = sqrt(distances[0] * distances[0] + distances[1] * distances[1] + distances[2] * distances[2]);
}


// function to calculate the forces between two particles on device
__device__ void ij_forces_d(double * forces, double * distances, double sigma, double epsilon) {
    double r = 0.0;
    distance_size_d(&r, distances);
    if (r == 0.0) {
        printf("Error: Zero distance between particles!\n");
        forces[0] = 0.0;
        forces[1] = 0.0;
        forces[2] = 0.0;
    }
    else if (r>2.5*sigma){ // cut off distance
        forces[0] = 0.0;
        forces[1] = 0.0;
        forces[2] = 0.0;
    }
    else {
    double r6 = pow(sigma / r, 6);
    double force_multiplier = 24 * epsilon * r6 * (2 * r6 - 1) /(r*r);
    forces[0] = force_multiplier * distances[0];
    forces[1] = force_multiplier * distances[1];
    forces[2] = force_multiplier * distances[2];
}
}




__global__ void update_positions_d(double * positions_new, double* positions_old , double * velocities_old, 
    double * accelerations, double dt, double boxSize, int numParticles) {
    int particle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( particle_idx >= numParticles ) return;
    
    particle_idx *= 3;
    
    positions_new[particle_idx] = positions_old[particle_idx] + velocities_old[particle_idx] * dt 
                                    + 0.5 * accelerations[particle_idx] * dt * dt;
    positions_new[particle_idx + 1] = positions_old[particle_idx + 1] + velocities_old[particle_idx + 1] * dt 
                                    + 0.5 * accelerations[particle_idx + 1] * dt * dt;
    positions_new[particle_idx + 2] = positions_old[particle_idx + 2] + velocities_old[particle_idx + 2] * dt 
                                    + 0.5 * accelerations[particle_idx + 2] * dt * dt;

    // Check periodic boundaries
    checkPeriodicBoundaries_d(positions_new + particle_idx, positions_new + particle_idx + 1, positions_new + particle_idx + 2, boxSize);
}

__global__ void update_velocities_d(double * velocities_new, double* velocities_old , double * accelerations, 
    double dt, int numParticles) {
    int particle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( particle_idx >= numParticles ) return;
    
    particle_idx *= 3;
    
    velocities_new[particle_idx] = velocities_old[particle_idx] + 0.5 * accelerations[particle_idx] * dt;
    velocities_new[particle_idx + 1] = velocities_old[particle_idx + 1] +  0.5 * accelerations[particle_idx + 1] * dt;
    velocities_new[particle_idx + 2] = velocities_old[particle_idx + 2] + 0.5 * accelerations[particle_idx + 2] * dt;
}


// function to calculate the forces for a given particle on device
__global__ void acceleration_updater_d(double * acceleration, double * positions, double * forces, double * masses,  
    double sigma, double epsilon, double boxSize, int numParticles) {
    int particle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( particle_idx >= numParticles ) return;
    
    particle_idx *= 3;
    forces[particle_idx] = 0.0;
    forces[particle_idx + 1] = 0.0;
    forces[particle_idx + 2] = 0.0;
    // Calculate forces for the particle at particle_idx
    for (size_t j = 0; j < 3*numParticles; j += 3) {
        if (j != particle_idx) {
            double distances[3];
            double ij_force[3];
            periodic_distance_d(distances, positions + particle_idx, positions + particle_idx + 1, positions + particle_idx + 2,
                positions + j, positions + j + 1, positions + j + 2, boxSize);
            ij_forces_d(ij_force , distances, sigma, epsilon);
            forces[particle_idx] += ij_force[0];
            forces[particle_idx + 1] += ij_force[1];
            forces[particle_idx + 2] += ij_force[2];
        }
    } 
    // Calculate acceleration for the particle at particle_idx
    acceleration[particle_idx] = forces[particle_idx] / masses[particle_idx / 3];
    acceleration[particle_idx + 1] = forces[particle_idx + 1] / masses[particle_idx / 3];
    acceleration[particle_idx + 2] = forces[particle_idx + 2] / masses[particle_idx / 3];
    
   
}

