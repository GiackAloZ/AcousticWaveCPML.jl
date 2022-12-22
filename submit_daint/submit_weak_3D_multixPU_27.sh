#!/bin/bash -l
#SBATCH --job-name="weak_acou3D_multixPU_27_2-2-2"
#SBATCH --output=measurements/weak_acou3D_multixPU_27_2-2-2.%j.o
#SBATCH --error=measurements/weak_acou3D_multixPU_27_2-2-2.%j.e
#SBATCH --time=00:20:00
#SBATCH --nodes=27
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account class04

module load daint-gpu
module load Julia/1.7.2-CrayGNU-21.09-cuda

export MPICH_RDMA_ENABLED_CUDA=1
export IGG_CUDAAWARE_MPI=1

srun -n27 bash -c 'LD_PRELOAD="/usr/lib64/libcuda.so:/usr/local/cuda/lib64/libcudart.so" julia --project=.. -O3 --check-bounds=no --math-mode=fast -- ../scripts/benchmarks/benchmark_weak_acoustic_3D_multixPU.jl'