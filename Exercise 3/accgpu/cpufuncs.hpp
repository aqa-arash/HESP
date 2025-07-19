//import necessary libraries
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <fstream>
#include <limits>
#include <utility>

void computeParticleCells_h(
    const std::vector<double> & positions,
    std::vector<int> & cells,
    std::vector<int> & particleCell,
    int num_cells,
    double cell_size) {
    
    int N = particleCell.size();
    double* pos = const_cast<double*>(positions.data());
    int* cell_data = cells.data();
    int* particle_cell = particleCell.data();
    size_t total_cells = cells.size();

    #pragma acc data copyin(pos[0:N*3]) copy(cell_data[0:total_cells], particle_cell[0:N])
    {
        #pragma acc parallel loop
        for (int i = 0; i < N; ++i) {
            double x = pos[3 * i + 0];
            double y = pos[3 * i + 1];
            double z = pos[3 * i + 2];

            int cx = ((int)(x / cell_size)) % num_cells;
            int cy = ((int)(y / cell_size)) % num_cells;
            int cz = ((int)(z / cell_size)) % num_cells;

            if (cx < 0) cx += num_cells;
            if (cy < 0) cy += num_cells;
            if (cz < 0) cz += num_cells;

            int cell_index = cx + cy * num_cells + cz * num_cells * num_cells;

            int old_head;
            #pragma acc atomic capture
            {
                old_head = cell_data[cell_index];
                cell_data[cell_index] = i;
            }
            particle_cell[i] = old_head;
        }
    }
}


// Find the minimal d > cutoffRadius such that boxSize / d is approximately an integer
std::pair<double, int> findMinimalDivisor(double cutoffRadius, double boxSize) {
    const double epsilon = 1e-10; // numerical tolerance for approximate integer check
    double best_d = std::numeric_limits<double>::max();
    int best_n = -1;

    int max_n = static_cast<int>(boxSize / cutoffRadius); // maximum reasonable number of divisions

    for (int n = 1; n <= max_n; ++n) {
        double d = boxSize / n;

        // Skip if d does not exceed cutoff
        if (d < cutoffRadius)
            continue;

        // Check if boxSize / d is numerically close to an integer
        double approx_n = boxSize / d;
        if (std::abs(approx_n - std::round(approx_n)) < epsilon) {
            // Update if this is the smallest valid d so far
            if (d < best_d) {
                best_d = d;
                best_n = n;
            }
        }
    }

    if (best_n == -1) {
        best_d = boxSize; // If no valid d found, return boxSize as default
        best_n = 1; // and set n to 1
    }

    return {best_d, best_n};
}

// function to calculate the distance size
// (can be globalized)
double distance_size(const std::vector<double> & distances) {
    return std::sqrt(distances[0] * distances[0] + distances[1] * distances[1] + distances[2] * distances[2]);
}

///////////////////////////////// CELL FUNCTIONS /////////////////////////

// GPU-parallelized resetCells function

void resetCells_kernel(int* cell_data, size_t cell_size) {
    #pragma acc parallel loop present(cell_data[0:cell_size])
    for (size_t idx = 0; idx < cell_size; ++idx) {
        cell_data[idx] = -1;
    }
}


// GPU-parallelized computeParticleCells function

void computeParticleCells_kernel(
    const double* pos,
    int* cell_data,
    int* particle_cell,
    int N,
    int num_cells,
    double cell_size,
    size_t total_cells) {

    #pragma acc parallel loop present(pos[0:N*3], cell_data[0:total_cells], particle_cell[0:N])
    for (int i = 0; i < N; ++i) {
        // Access x, y, z of particle i
        double x = pos[3 * i + 0];
        double y = pos[3 * i + 1];
        double z = pos[3 * i + 2];

        // Compute integer cell indices
        int cx = ((int)(x / cell_size)) % num_cells;
        int cy = ((int)(y / cell_size)) % num_cells;
        int cz = ((int)(z / cell_size)) % num_cells;

        if (cx < 0) cx += num_cells;
        if (cy < 0) cy += num_cells;
        if (cz < 0) cz += num_cells;

        // Compute 1D cell index
        int cell_index = cx + cy * num_cells + cz * num_cells * num_cells;

        // ATOMIC OPERATION: Insert particle i at front of linked list
        int old_head;
        #pragma acc atomic capture
        {
            old_head = cell_data[cell_index];
            cell_data[cell_index] = i;
        }
        particle_cell[i] = old_head;
    }
}


// Helper function to get cell coordinates from cell ID (assuming cubic grid)
void getCellCoords(int cellId, int num_cells, int &x, int &y, int &z) {
    z = cellId / (num_cells * num_cells);
    int rem = cellId % (num_cells * num_cells);
    y = rem / num_cells;
    x = rem % num_cells;
}

////////////////////////////////////////////////////////////////////////
////////////// Acceleration Updater Functions /////////////////////////


// GPU-compatible helper functions
#pragma acc routine seq
double distance_size_gpu(double dx, double dy, double dz) {
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}


// function to calculate the periodic distance between two particles
#pragma acc routine seq
void periodic_distance_gpu(double x1, double y1, double z1, double x2, double y2, double z2, 
                          double boxSize, double* dx, double* dy, double* dz) {
    *dx = x1 - x2;
    *dy = y1 - y2;
    *dz = z1 - z2;
    
    if (boxSize > 0.0) {
        *dx = *dx - boxSize * round(*dx / boxSize);
        *dy = *dy - boxSize * round(*dy / boxSize);
        *dz = *dz - boxSize * round(*dz / boxSize);
    }
}


#pragma acc routine seq
void ij_forces_gpu(double dx, double dy, double dz, double sigma, double epsilon, 
                   double cutoffRadius, double* fx, double* fy, double* fz) {
    double r = distance_size_gpu(dx, dy, dz);
    
    if (r == 0.0 || (cutoffRadius > 0.000000001 && r > cutoffRadius)) {
        *fx = 0.0;
        *fy = 0.0;
        *fz = 0.0;
        return;
    }
    
    double r6 = pow(sigma / r, 6.0);
    double force_multiplier = 24 * epsilon * r6 * (2 * r6 - 1) / (r * r);
    *fx = force_multiplier * dx;
    *fy = force_multiplier * dy;
    *fz = force_multiplier * dz;
}

// GPU-compatible force and acceleration calculation kernel ACC

void acceleration_updater_kernel(
    double* acc, const double* pos, double* force, const double* mass,
    int* cell_data, int* particle_cell,
    double sigma, double epsilon, double boxSize, double cutoffRadius,
    int numParticles, int num_cells) {

    #pragma acc parallel loop present(acc[0:numParticles*3], pos[0:numParticles*3], force[0:numParticles*3], \
                                      mass[0:numParticles], cell_data[0:num_cells*num_cells*num_cells], \
                                      particle_cell[0:numParticles])
    for (int particle_idx = 0; particle_idx < numParticles; ++particle_idx) {
        int particle_pos = particle_idx * 3;

        // Reset force vector for this particle
        force[particle_pos] = 0.0;
        force[particle_pos + 1] = 0.0;
        force[particle_pos + 2] = 0.0;

        // Calculate cell coordinates
        double x_P = pos[particle_pos + 0];
        double y_P = pos[particle_pos + 1];
        double z_P = pos[particle_pos + 2];

        // Compute integer cell indices (with periodic boundary conditions)
        double cell_size = boxSize / num_cells;
        int cell_x = ((int)(x_P / cell_size)) % num_cells;
        int cell_y = ((int)(y_P / cell_size)) % num_cells;
        int cell_z = ((int)(z_P / cell_size)) % num_cells;

        // Loop through neighboring cells (3x3x3 grid)
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
                    int j = cell_data[neighbor_cell_index];
                    while (j != -1) {
                        if (j != particle_idx) { // Skip self interaction
                            // Calculate distance and forces
                            double dx, dy, dz;
                            periodic_distance_gpu(pos[particle_pos], pos[particle_pos + 1], pos[particle_pos + 2],
                                                  pos[j * 3], pos[j * 3 + 1], pos[j * 3 + 2], 
                                                  boxSize, &dx, &dy, &dz);

                            double fx, fy, fz;
                            ij_forces_gpu(dx, dy, dz, sigma, epsilon, cutoffRadius, &fx, &fy, &fz);

                            // Accumulate forces
                            force[particle_pos]     += fx;
                            force[particle_pos + 1] += fy;
                            force[particle_pos + 2] += fz;
                        }
                        j = particle_cell[j]; // Move to next particle in the linked list
                    }
                }
            }
        }

        // Calculate acceleration by dividing force by particle mass
        double particle_mass = mass[particle_idx];
        acc[particle_pos]     = force[particle_pos]     / particle_mass;
        acc[particle_pos + 1] = force[particle_pos + 1] / particle_mass;
        acc[particle_pos + 2] = force[particle_pos + 2] / particle_mass;
    }
}


// New GPU-compatible function for periodic boundaries
// function to check and update periodic boundaries for each particle (can be globalized)
#pragma acc routine seq
void checkPeriodicBoundaries_gpu(double* positions, int idx, double boxSize) {
    positions[idx]     = fmod(fmod(positions[idx],     boxSize) + boxSize, boxSize);
    positions[idx + 1] = fmod(fmod(positions[idx + 1], boxSize) + boxSize, boxSize);
    positions[idx + 2] = fmod(fmod(positions[idx + 2], boxSize) + boxSize, boxSize);
}


void update_positions_kernel(double* pos_new, const double* pos_old, 
                           const double* vel_old, const double* acc, 
                           double dt, int total_size) {
    #pragma acc parallel loop present(pos_new[0:total_size], pos_old[0:total_size], \
                                      vel_old[0:total_size], acc[0:total_size])
    for (int i = 0; i < total_size; ++i) {
        pos_new[i] = pos_old[i] + vel_old[i] * dt + 0.5 * acc[i] * dt * dt;
    }
}


void apply_periodic_boundaries_kernel(double* pos_new, double boxSize, int numParticles) {
    #pragma acc parallel loop present(pos_new[0:numParticles*3])
    for (int particle_idx = 0; particle_idx < numParticles; ++particle_idx) {
        int particle_pos = 3 * particle_idx;
        checkPeriodicBoundaries_gpu(pos_new, particle_pos, boxSize);
    }
}


// In the header file or before the main function:
void update_velocities_kernel(double* vel_new, const double* vel_old, 
                             const double* acc, double dt, int total_size) {
    #pragma acc parallel loop present(vel_new[0:total_size], vel_old[0:total_size], acc[0:total_size])
    for (int i = 0; i < total_size; ++i) {
        vel_new[i] = vel_old[i] + 0.5 * acc[i] * dt;
    }
}
