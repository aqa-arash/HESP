# DEM/MD CUDA Simulation

This project is a simple, GPU-accelerated particle simulation framework (supporting Discrete Element Method and Molecular Dynamics) implemented in C++ and CUDA.

It simulates the motion of particles in a 3D box based on physical forces and integration over time.

---

## Features

* **GPU Acceleration:** All major computations (force calculation, position/velocity updates) run on an NVIDIA GPU using CUDA kernels.
* **Force Models:** Supports multiple interaction models:
    * Lennard-Jones potential
    * Spring-Dashpot (Hertz-Mindlin contact model)
    * Spring-Dashpot + Gravity
* **Boundary Conditions:** Implements both **periodic** (wrap-around) and **fixed** (reflecting) boundary conditions.
* **Optimization:** Uses a **linked-cell list** spatial subdivision to efficiently find neighboring particles, avoiding an $O(N^2)$ brute-force calculation.
* **Configuration:** Reads all simulation parameters (particle data, box size, simulation time, force model, etc.) from a `config.txt` file.
* **Output:** Saves simulation snapshots at specified intervals as `.vtk` files in the `output/` directory. These can be visualized in software like ParaView or Visit.

---

## Usage

### 1. Dependencies

* NVIDIA CUDA Toolkit (for `nvcc` compiler)
* A C++ compiler (like `g++`)

### 2. Compile

Makefile should be functional, reach out if it fails.

*(You may need to adjust flags based on your system and CUDA version.)*

### 3. Configure

Create a `config.txt` file in the same directory to define your simulation's initial conditions and parameters.

### 4. Run

Execute the compiled binary. It will automatically look for `config.txt` or accept a path as an argument. It also takes a `.vtk` as an initial condition. The path to the `.vtk` file should be given in the config file.
