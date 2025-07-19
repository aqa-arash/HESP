#!/bin/bash
#SBATCH --job-name=ompGPU_bench
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=1:30:00
#SBATCH --output=logs/ompGPU_bench_%j.out
#SBATCH --error=logs/ompGPU_bench_%j.err

# Optional: Lade benötigte Module
module purge
module load nvhpc

# Optional: Zeige Infos zur GPU
echo "Allocated GPU(s):"
nvidia-smi


# Starte Benchmark-Analyse
echo "=== Performance Analysis started ==="
bash performance_analysis_oaccGPU.sh

echo "=== Performance Analysis completed ==="