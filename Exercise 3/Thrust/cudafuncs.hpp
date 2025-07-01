#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>


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
__device__ void periodic_distance_d(double * distance , const double * x1,const double * y1, const double * z1, const double * x2,const double * y2,const double * z2,const double boxSize) {
    // calculate the distance between two particles
    double dx = *x1 - *x2;
    double dy = *y1 - *y2;
    double dz = *z1 - *z2;
    if (boxSize>0.000000001){
        // apply periodic boundary conditions
        dx -= boxSize * round(dx / boxSize);
        dy -= boxSize * round(dy / boxSize);
        dz -= boxSize * round(dz / boxSize);
    }
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
__device__ void ij_forces_d(double * forces, double * distances, double sigma, double epsilon, double cutoffRadius) {
    double r = 0.0;
    distance_size_d(&r, distances);
    if (r == 0.0) {
        printf("Error: Zero distance between particles!\n");
        forces[0] = 0.0;
        forces[1] = 0.0;
        forces[2] = 0.0;
    }
    else if (cutoffRadius > 0.000000001 && r>cutoffRadius){ // cut off distance
        forces[0] = 0.0;
        forces[1] = 0.0;
        forces[2] = 0.0;
    }
    else {
    double r6 = pow(sigma / r, 6.0);
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

    if (boxSize>0.000000001){ // to prevent numerical errors 
    // Check periodic boundaries
        checkPeriodicBoundaries_d(positions_new + particle_idx, positions_new + particle_idx + 1, positions_new + particle_idx + 2, boxSize);
    }
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


// Helper function to get cell coordinates from cell ID (assuming cubic grid)
__device__ void getCellCoords(int cellId, int num_cells, int *x, int *y, int *z) {
    *z = cellId / (num_cells * num_cells);
    int rem = cellId % (num_cells * num_cells);
    *y = rem / num_cells;
    *x = rem % num_cells;
}

// Helper function to check if cell2 is neighbor of cell1 (including cell1 itself)
__device__ bool isNeighborCell(int cell1, int cell2, int num_cells) {
    int x1, y1, z1;
    int x2, y2, z2;
    getCellCoords(cell1, num_cells, &x1, &y1, &z1);
    getCellCoords(cell2, num_cells, &x2, &y2, &z2);

    int dx = abs(x1 - x2);
    int dy = abs(y1 - y2);
    int dz = abs(z1 - z2);

    // Account for periodic boundaries:
    if (dx > num_cells/2) dx = num_cells - dx;
    if (dy > num_cells/2) dy = num_cells - dy;
    if (dz > num_cells/2) dz = num_cells - dz;

    return (dx <= 1 && dy <= 1 && dz <= 1);
}

__global__ void acceleration_updater_d(
    double * acceleration, double * positions, double * forces, double * masses,  
    double sigma, double epsilon, double boxSize, double cutoffRadius, int numParticles, 
    int * cells, int * particleCell, int num_cells)  // num_cells added as param
{
    int particle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_idx >= numParticles) return;

    int particle_pos = particle_idx * 3;

    // Reset force vector for this particle
    forces[particle_pos] = 0.0;
    forces[particle_pos + 1] = 0.0;
    forces[particle_pos + 2] = 0.0;

    if (num_cells > 3) {
        // calculate cell x y z coordinates
        int cell_x, cell_y, cell_z;
        double x_P = positions[3 * particle_idx + 0];
        double y_P = positions[3 * particle_idx + 1];
        double z_P = positions[3 * particle_idx + 2];

        // Compute integer cell indices (with periodic boundary conditions)
        double cell_size = boxSize / num_cells; // Assuming cubic cells
        cell_x = ((int)(x_P / cell_size)) % num_cells;
        cell_y = ((int)(y_P / cell_size)) % num_cells;
        cell_z = ((int)(z_P / cell_size)) % num_cells;
        
        for (int x = -1; x <= 1; x++) {
            for (int y = -1; y <= 1; y++) {
                for (int z = -1; z <= 1; z++) {
                    // Calculate neighbor cell index
                    int neighbor_cell_x = (cell_x + x + num_cells) % num_cells;
                    int neighbor_cell_y = (cell_y + y + num_cells) % num_cells;
                    int neighbor_cell_z = (cell_z + z + num_cells) % num_cells;
                    int neighbor_cell_index = neighbor_cell_x + 
                                              neighbor_cell_y * num_cells + 
                                              neighbor_cell_z * num_cells * num_cells;

                    // Get the first particle in this cell
                    int j = cells[neighbor_cell_index];
                    while (j != -1) {
                        if (j != particle_idx) { // Skip self interaction
                            // Calculate distance and forces
                            double distances[3];
                            double ij_force[3];

                            periodic_distance_d(distances, positions + particle_pos, positions + particle_pos + 1, positions + particle_pos + 2,
                                                 positions + j * 3, positions + j * 3 + 1, positions + j * 3 + 2, boxSize);

                            ij_forces_d(ij_force, distances, sigma, epsilon, cutoffRadius);

                            // Accumulate forces
                            forces[particle_pos] += ij_force[0];
                            forces[particle_pos + 1] += ij_force[1];
                            forces[particle_pos + 2] += ij_force[2];
                        }
                        j = particleCell[j]; // Move to next particle in the linked list
                    }
                }
            }
        }
    }
    else {
    // Loop over all other particles to calculate forces
        for (int j = 0; j < numParticles; j++) {
            if (j == particle_idx) continue; // Skip self interaction
            int j_pos = j * 3;

            double distances[3];
            double ij_force[3];

            // Calculate periodic distance vector between particles
            periodic_distance_d(distances, positions + particle_pos, positions + particle_pos + 1, positions + particle_pos + 2,
                                    positions + j_pos, positions + j_pos + 1, positions + j_pos + 2, boxSize);

            // Calculate inter-particle forces using Lennard-Jones potential (or similar)
            ij_forces_d(ij_force , distances, sigma, epsilon, cutoffRadius);

            // Accumulate forces
            forces[particle_pos] += ij_force[0];
            forces[particle_pos + 1] += ij_force[1];
            forces[particle_pos + 2] += ij_force[2];
        }
    }

        // Calculate acceleration by dividing force by particle mass
        double mass = masses[particle_idx];
        acceleration[particle_pos]     = forces[particle_pos]     / mass;
        acceleration[particle_pos + 1] = forces[particle_pos + 1] / mass;
        acceleration[particle_pos + 2] = forces[particle_pos + 2] / mass;
}


__global__ void resetCells(int* cells, int total_cells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_cells) {
        cells[idx] = -1;
    }
}

__global__ void computeParticleCells(
    double* positions,   // Linear array: [x0, y0, z0, x1, y1, z1, ...]
    int* cells,                // Head of linked list per cell
    int* particleCell,         // Linked list: next particle for each particle
    int N,                     // Number of particles
    int num_cells,             // Number of cells per dimension (cube root of total_cells)
    int total_cells,           // Total number of cells (num_cells^3)
    double cell_size)          // Cell size (same in x, y, z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Access x, y, z of particle i
    double x = positions[3 * i + 0];
    double y = positions[3 * i + 1];
    double z = positions[3 * i + 2];

    // Compute integer cell indices (with periodic boundary conditions)
    int cx = ((int)(x / cell_size)) % num_cells;
    int cy = ((int)(y / cell_size)) % num_cells;
    int cz = ((int)(z / cell_size)) % num_cells;

    if (cx < 0) cx += num_cells;
    if (cy < 0) cy += num_cells;
    if (cz < 0) cz += num_cells;

    // Compute 1D cell index
    int cell_index = cx + cy * num_cells + cz * num_cells * num_cells; // Is this right for C? Or do we just need to be consecutive?

    // Atomically insert particle i at the front of the linked list for this cell
    particleCell[i] = atomicExch(&cells[cell_index], i);
}

/////////////////////////////////
/////////////////////////////////////////////////////////////////////////// THRUST FUNCTIONS
////////////////////////////////

 void thrust_computeParticleCells(
    const thrust::device_vector<double> & positions,   // Linear array: [x0, y0, z0, x1, y1, z1, ...]
    thrust::device_vector<int> & cells,                // Head of linked list per cell
    thrust::device_vector<int> & particleCell,         // Linked list: next particle for each particle
    int N,                     // Number of particles
    int num_cells,             // Number of cells per dimension (cube root of total_cells)
    int total_cells,           // Total number of cells (num_cells^3)
    double cell_size)          // Cell size (same in x, y, z)
{
 
        // Get raw device pointers
    const double* pos_ptr = thrust::raw_pointer_cast(positions.data());
    int* cells_ptr = thrust::raw_pointer_cast(cells.data());
    int* particle_ptr = thrust::raw_pointer_cast(particleCell.data());

    // Create counting iterator
    auto counting_iter = thrust::make_counting_iterator(0);
    thrust::for_each(thrust::cuda::par,
        counting_iter, 
        counting_iter + N, 
        [pos_ptr, cells_ptr, particle_ptr, num_cells, cell_size] __device__ (int i) {
            // Access x, y, z of particle i
            double x = pos_ptr[3 * i + 0];
            double y = pos_ptr[3 * i + 1];
            double z = pos_ptr[3 * i + 2];

            // Compute integer cell indices (with periodic boundary conditions)
            int cx = ((int)(x / cell_size)) % num_cells;
            int cy = ((int)(y / cell_size)) % num_cells;
            int cz = ((int)(z / cell_size)) % num_cells;

            if (cx < 0) cx += num_cells;
            if (cy < 0) cy += num_cells;
            if (cz < 0) cz += num_cells;

            // Compute 1D cell index
            int cell_index = cx + cy * num_cells + cz * num_cells * num_cells; // Is this right for C? Or do we just need to be consecutive?

            // Atomically insert particle i at the front of the linked list for this cell
            particle_ptr[i] = atomicExch(&cells_ptr[cell_index], i);
        }
    );}



    
void thrust_update_positions(thrust::device_vector<double> & positions_new, const thrust::device_vector<double> & positions_old,
    const thrust::device_vector<double> & velocities_old, const thrust::device_vector<double> & accelerations,
    double dt, double boxSize, int numParticles) {
    
    //get raw pointers to device vectors
    double *positions_new_ptr = thrust::raw_pointer_cast(positions_new.data());
    const double *positions_old_ptr = thrust::raw_pointer_cast(positions_old.data());
    const double *velocities_old_ptr = thrust::raw_pointer_cast(velocities_old.data());
    const double *accelerations_ptr = thrust::raw_pointer_cast(accelerations.data());


    thrust::for_each(thrust::cuda::par,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(numParticles),
        [positions_new_ptr,positions_old_ptr,velocities_old_ptr, accelerations_ptr, dt, boxSize] __device__(int particle_idx) {
            particle_idx *= 3; // Convert to linear index for 3D positions
            positions_new_ptr[particle_idx] = positions_old_ptr[particle_idx] + velocities_old_ptr[particle_idx] * dt 
                                            + 0.5 * accelerations_ptr[particle_idx] * dt * dt;
            positions_new_ptr[particle_idx + 1] = positions_old_ptr[particle_idx + 1] + velocities_old_ptr[particle_idx + 1] * dt 
                                            + 0.5 * accelerations_ptr[particle_idx + 1] * dt * dt;
            positions_new_ptr[particle_idx + 2] = positions_old_ptr[particle_idx + 2] + velocities_old_ptr[particle_idx + 2] * dt 
                                            + 0.5 * accelerations_ptr[particle_idx + 2] * dt * dt;
            if (boxSize>0.000000001){ // to prevent numerical errors 
                // Check periodic boundaries
                checkPeriodicBoundaries_d(positions_new_ptr + particle_idx, positions_new_ptr + particle_idx + 1, positions_new_ptr + particle_idx + 2, boxSize);
            }
        });                                             
}



void thrust_update_velocities( thrust::device_vector<double> & velocities_new, const thrust::device_vector<double> & velocities_old , const thrust::device_vector<double> & accelerations, 
    double dt, int numParticles) {
    //get raw pointers to device vectors
    double *velocities_new_ptr = thrust::raw_pointer_cast(velocities_new.data());
    const double *velocities_old_ptr = thrust::raw_pointer_cast(velocities_old.data());
    const double *accelerations_ptr = thrust::raw_pointer_cast(accelerations.data());

    thrust::for_each(thrust::cuda::par,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(numParticles),
        [velocities_new_ptr, velocities_old_ptr, accelerations_ptr, dt] __device__(int particle_idx) {
            particle_idx *= 3; // Convert to linear index for 3D velocities
            velocities_new_ptr[particle_idx] = velocities_old_ptr[particle_idx] + 0.5 * accelerations_ptr[particle_idx] * dt;
            velocities_new_ptr[particle_idx + 1] = velocities_old_ptr[particle_idx + 1] + 0.5 * accelerations_ptr[particle_idx + 1] * dt;
            velocities_new_ptr[particle_idx + 2] = velocities_old_ptr[particle_idx + 2] + 0.5 * accelerations_ptr[particle_idx + 2] * dt;
        });
}



void thrust_acceleration_updater(
     thrust::device_vector<double> &  acceleration,const thrust::device_vector<double> &  positions,  thrust::device_vector<double> &  forces, const thrust::device_vector<double> &  masses,  
    double sigma, double epsilon, double boxSize, double cutoffRadius, int numParticles, 
    const thrust::device_vector<int> & cells, const thrust::device_vector<int> & particleCell, int num_cells)  // num_cells added as param
{
    
    //get raw pointers to device vectors
    double *acceleration_ptr = thrust::raw_pointer_cast(acceleration.data());
    const double *positions_ptr = thrust::raw_pointer_cast(positions.data());
    double *forces_ptr = thrust::raw_pointer_cast(forces.data());
    const double *masses_ptr = thrust::raw_pointer_cast(masses.data());
    const int *cells_ptr = thrust::raw_pointer_cast(cells.data());
    const int *particleCell_ptr = thrust::raw_pointer_cast(particleCell.data());
    // Launch a parallel for_each to compute forces and accelerations
    thrust::for_each(thrust::cuda::par,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(numParticles),
        [acceleration_ptr, positions_ptr, forces_ptr, masses_ptr, sigma, epsilon, boxSize, cutoffRadius, numParticles, cells_ptr, particleCell_ptr, num_cells] __device__(int particle_idx) {
               int particle_pos = particle_idx * 3;

    // Reset force vector for this particle
    forces_ptr[particle_pos] = 0.0;
    forces_ptr[particle_pos + 1] = 0.0;
    forces_ptr[particle_pos + 2] = 0.0;
        // calculate cell x y z coordinates
        int cell_x, cell_y, cell_z;
        double x_P = positions_ptr[3 * particle_idx + 0];
        double y_P = positions_ptr[3 * particle_idx + 1];
        double z_P = positions_ptr[3 * particle_idx + 2];

        // Compute integer cell indices (with periodic boundary conditions)
        double cell_size = boxSize / num_cells; // Assuming cubic cells
        cell_x = ((int)(x_P / cell_size)) % num_cells;
        cell_y = ((int)(y_P / cell_size)) % num_cells;
        cell_z = ((int)(z_P / cell_size)) % num_cells;
        
        for (int x = -1; x <= 1; x++) {
            for (int y = -1; y <= 1; y++) {
                for (int z = -1; z <= 1; z++) {
                    // Calculate neighbor cell index
                    int neighbor_cell_x = (cell_x + x + num_cells) % num_cells;
                    int neighbor_cell_y = (cell_y + y + num_cells) % num_cells;
                    int neighbor_cell_z = (cell_z + z + num_cells) % num_cells;
                    int neighbor_cell_index = neighbor_cell_x + 
                                              neighbor_cell_y * num_cells + 
                                              neighbor_cell_z * num_cells * num_cells;

                    // Get the first particle in this cell
                    int j = cells_ptr[neighbor_cell_index];
                    while (j != -1) {
                        if (j != particle_idx) { // Skip self interaction
                            // Calculate distance and forces
                            double distances[3];
                            double ij_force[3];

                            periodic_distance_d(distances, positions_ptr + particle_pos, positions_ptr + particle_pos + 1, positions_ptr + particle_pos + 2,
                                                 positions_ptr + j * 3, positions_ptr + j * 3 + 1, positions_ptr + j * 3 + 2, boxSize);

                            ij_forces_d(ij_force, distances, sigma, epsilon, cutoffRadius);

                            // Accumulate forces
                            forces_ptr[particle_pos] += ij_force[0];
                            forces_ptr[particle_pos + 1] += ij_force[1];
                            forces_ptr[particle_pos + 2] += ij_force[2];
                        }
                        j = particleCell_ptr[j]; // Move to next particle in the linked list
                    }
                }
            }
        }
    
   
        // Calculate acceleration by dividing force by particle mass
        double mass = masses_ptr[particle_idx];
        acceleration_ptr[particle_pos]     = forces_ptr[particle_pos]     / mass;
        acceleration_ptr[particle_pos + 1] = forces_ptr[particle_pos + 1] / mass;
        acceleration_ptr[particle_pos + 2] = forces_ptr[particle_pos + 2] / mass;
        });

 
}