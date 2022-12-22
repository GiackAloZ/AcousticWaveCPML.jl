#!/bin/bash -l
#SBATCH --job-name="weak_acou3D_xPU"
#SBATCH --output=measurements/weak_acou3D_xPU.%j.o
#SBATCH --error=measurements/weak_acou3D_xPU.%j.e
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account class04

module load daint-gpu
module load Julia/1.7.2-CrayGNU-21.09-cuda

srun julia --project=.. -O3 --check-bounds=no --math-mode=fast -- ../scripts/benchmarks/benchmark_weak_acoustic_3D_xPU.jl