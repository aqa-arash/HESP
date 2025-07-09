#!/bin/bash -l
#SBATCH --job-name=ompCPU_bench
#SBATCH --output=logs/ompCPU_bench_%j.out
#SBATCH --error=logs/ompCPU_bench_%j.err
#SBATCH --partition=singlenode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --export=NONE

# Optional: Lade benötigte Module
module load intel
make clean 
make

# Starte Benchmark-Analyse
echo "=== Performance Analysis started ==="
bash performance_analysis_ompCPU.sh

echo "=== Performance Analysis completed ==="