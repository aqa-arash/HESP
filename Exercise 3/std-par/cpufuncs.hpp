//import necessary libraries
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <fstream>
#include <limits>
#include <utility>
#include <algorithm>
#include <execution> // for stdparallel algorithms (if needed)
#include <numeric> // for std::iota
#include <atomic> // for atomic operations

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

// function to check and update periodic boundaries for each particle (can be globalized)
void checkPeriodicBoundaries(double * x, double * y, double * z, double boxSize) {
    *x = fmod(fmod(*x, boxSize) + boxSize, boxSize);
    *y = fmod(fmod(*y, boxSize) + boxSize, boxSize);
    *z = fmod(fmod(*z, boxSize) + boxSize, boxSize);
}


// function to calculate the periodic distance between two particles
// (can be globalized)
std::vector<double> periodic_distance(double x1, double y1, double z1, double x2, double y2, double z2, double boxSize) {
    std::vector<double> distances(3),pbc_distance(3);
    distances[0] = x1 - x2;
    distances[1] = y1 - y2;
    distances[2] = z1 - z2;
    if (boxSize>0.0){
    // apply periodic boundary conditions
    pbc_distance[0] = distances[0] - boxSize * std::round(distances[0] / boxSize);
    pbc_distance[1] = distances[1] - boxSize * std::round(distances[1] / boxSize);
    pbc_distance[2] = distances[2] - boxSize * std::round(distances[2] / boxSize);
    }
    return pbc_distance;
}

// function to calculate the distance size
// (can be globalized)
double distance_size(const std::vector<double> & distances) {
    return std::sqrt(distances[0] * distances[0] + distances[1] * distances[1] + distances[2] * distances[2]);
}




// function to calculate the forces between two particles on device
void ij_forces(std::vector<double> & forces,const std::vector<double> & distances, double sigma, double epsilon, double cutoffRadius) {
    double r = 0.0;
    r = distance_size(distances);
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

///////////////////////////////// CELL FUNCTIONS /////////////////////////

// resetCells function to reset the cells array
void resetCells_h(std::vector<std::atomic<int>> & cells) {
    int N = cells.size(); // Number of cells
    std::vector<int> range(N);
    std::iota(range.begin(), range.end(), 0); // Fill range with indices
    std::atomic<int> * cells_ptr = cells.data(); // Cast to atomic pointer for direct access
    std::for_each (std::execution::par_unseq, range.begin(),range.end(), [cells_ptr](int idx) {
        // Reset each cell to -1
        cells_ptr[idx].store(-1, std::memory_order_relaxed);
    });
    // Note: We use relaxed memory order here because we are just resetting the values
}


void computeParticleCells_h(
    const std::vector<double> & positions,   // Linear array: [x0, y0, z0, x1, y1, z1, ...]
    std::vector<std::atomic<int>> & cells,                // Head of linked list per cell
    std::vector<int> & particleCell,         // Linked list: next particle for each particle
    int num_cells,             // Number of cells per dimension (cube root of total_cells)
    double cell_size)          // Cell size (same in x, y, z)
{
    int N = particleCell.size(); // Number of particles
    std::vector<int> range(N);
    std::iota(range.begin(), range.end(), 0); // Fill range with indices    
    // // Loop over all particles
    int * particleCell_ptr = particleCell.data();
    std::atomic<int> * cells_ptr = cells.data(); // Cast to atomic pointer for direct access
    std::for_each (std::execution::par_unseq, range.begin(), range.end(), [cells_ptr, particleCell_ptr, positions, num_cells,cell_size](int i) {
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

    // !!!!!!!!! make sure in future paralelization that this is done atomically !!!!!!!!!! 
    // Insert particle i at the front of the linked list for this cell (no need for atomic operation on CPU)
    int current_head, new_head;
        do {
            current_head = cells_ptr[cell_index].load(std::memory_order_acquire);
            particleCell_ptr[i] = current_head;
            new_head = i;
        } while (!cells_ptr[cell_index].compare_exchange_weak(
            current_head, new_head, 
            std::memory_order_release, 
            std::memory_order_acquire));
    });
    
    // Note: The cells vector now contains the head of the linked list for each cell,
    // and the particleCell vector contains the next particle in the linked list for each particle.
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


void acceleration_updater(
    std::vector<double> & acceleration, const std::vector<double> & positions, std::vector<double> & forces, std::vector<double> & masses,  
    double sigma, double epsilon, double boxSize, double cutoffRadius, int numParticles, 
    std::vector<std::atomic<int>> & cells, std::vector<int> &  particleCell, int num_cells)  // num_cells added as param
{
    std::vector<int> range(numParticles);
    std::iota(range.begin(), range.end(), 0); // Fill range with indices
    double * acceleration_ptr = acceleration.data();
    double * forces_ptr = forces.data();
    std::atomic<int> * cells_ptr = cells.data(); // Cast to atomic pointer for direct access
    std::for_each (std::execution::par_unseq, range.begin(),range.end(), [cells_ptr,acceleration_ptr,forces_ptr,
    num_cells,particleCell,sigma, epsilon,boxSize,cutoffRadius,numParticles, positions, masses ](int particle_idx) {
    int particle_pos = particle_idx * 3;

    // Reset force vector for this particle
    forces_ptr[particle_pos] = 0.0;
    forces_ptr[particle_pos + 1] = 0.0;
    forces_ptr[particle_pos + 2] = 0.0;
    // calculate cell x y z coordinates
    int cell_x, cell_y, cell_z;
    double x_P = positions[particle_pos + 0];
    double y_P = positions[particle_pos + 1];
    double z_P = positions[particle_pos + 2];

    // Compute integer cell indices (with periodic boundary conditions)
    double cell_size = boxSize / num_cells; // Assuming cubic cells
    cell_x = ((int)(x_P / cell_size)) % num_cells;
    cell_y = ((int)(y_P / cell_size)) % num_cells;
    cell_z = ((int)(z_P / cell_size)) % num_cells;
    
    //  loop through neighboring cells
    // Note: We use a 3x3x3 grid of neighboring cells, including the current cell.
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
                int j = cells_ptr[neighbor_cell_index].load(std::memory_order_acquire);
                while (j != -1) {
                    if (j != particle_idx) { // Skip self interaction
                        // Calculate distance and forces
                        std::vector<double> distances(3,0.0);
                        std::vector<double> ij_force(3,0.0);

                        distances = periodic_distance( positions[particle_pos], positions [ particle_pos + 1], positions [ particle_pos + 2],
                                                positions [ j * 3], positions  [j * 3 + 1], positions [ j * 3 + 2], boxSize);

                        ij_forces(ij_force, distances, sigma, epsilon, cutoffRadius);

                        // Accumulate forces
                        forces_ptr[particle_pos] += ij_force[0];
                        forces_ptr[particle_pos + 1] += ij_force[1];
                        forces_ptr[particle_pos + 2] += ij_force[2];
                    }
                    j = particleCell[j]; // Move to next particle in the linked list
                }
            }
        }
    }
        // Calculate acceleration by dividing force by particle mass
        double mass = masses[particle_idx];
        acceleration_ptr[particle_pos]     = forces_ptr[particle_pos]     / mass;
        acceleration_ptr[particle_pos + 1] = forces_ptr[particle_pos + 1] / mass;
        acceleration_ptr[particle_pos + 2] = forces_ptr[particle_pos + 2] / mass;
});
    // Note: The forces and accelerations vectors are modified in place, so no need to
}








void update_positions(std::vector<double> & positions_new, const std::vector<double> & positions_old , const std::vector<double> & velocities_old, 
    const std::vector<double> & accelerations, double dt, double boxSize, int numParticles) {
    std::vector<int> range(numParticles);
    std::iota(range.begin(), range.end(), 0); // Fill range with indices
    double * positions_new_ptr = positions_new.data();

    std::for_each(std::execution::par_unseq, range.begin(), range.end(), [=](int particle_idx) {
        int particle_pos = 3* particle_idx;
        positions_new_ptr[particle_pos] = positions_old[particle_pos] + velocities_old[particle_pos] * dt 
                                    + 0.5 * accelerations[particle_pos] * dt * dt;
        positions_new_ptr[particle_pos + 1] = positions_old[particle_pos + 1] + velocities_old[particle_pos + 1] * dt 
                                    + 0.5 * accelerations[particle_pos + 1] * dt * dt;
        positions_new_ptr[particle_pos + 2] = positions_old[particle_pos + 2] + velocities_old[particle_pos + 2] * dt 
                                    + 0.5 * accelerations[particle_pos + 2] * dt * dt;

    if (boxSize>0.000000001){ // to prevent numerical errors 
    // Check periodic boundaries
        checkPeriodicBoundaries(positions_new_ptr + particle_pos, positions_new_ptr + particle_pos + 1, positions_new_ptr + particle_pos + 2, boxSize);
        }
        }); // end of for_each
    // Note: The checkPeriodicBoundaries function modifies the positions in place, so no need 
}



void update_velocities(std::vector<double> & velocities_new, const std::vector<double> & velocities_old , 
    const std::vector<double> & accelerations, double dt, int numParticles) {
std::vector<int> range(numParticles);
    std::iota(range.begin(), range.end(), 0); // Fill range with indices 
    double * velocities_new_ptr = velocities_new.data();   
    std::for_each (std::execution::par_unseq, range.begin(), range.end(), [=](int i) {
        int particle_idx = 3* i;
        
        velocities_new_ptr[particle_idx] = velocities_old[particle_idx] + 0.5 * accelerations[particle_idx] * dt;
        velocities_new_ptr[particle_idx + 1] = velocities_old[particle_idx + 1] +  0.5 * accelerations[particle_idx + 1] * dt;
        velocities_new_ptr[particle_idx + 2] = velocities_old[particle_idx + 2] + 0.5 * accelerations[particle_idx + 2] * dt;
    });
}