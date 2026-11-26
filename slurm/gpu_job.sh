#!/bin/bash
# One deep-learning run (M5, M7) with DDP over 2 GPUs. Pass the run_dl_v4.py arguments in ARGS:
#   sbatch --export=ALL,ARGS="--nutrient N --model M7 --seqfix --sampEnd --save-checkpoints" slurm/gpu_job.sh
# External-season runs (--external) and explain_v4.py use one GPU and plain python.
# Submit from the repository root (mkdir -p logs first). Adjust partition, account and environment to your cluster.
# The image loader holds the tile sequences in host memory, so keep --mem high.
#SBATCH --job-name=nutrinet_gpu
#SBATCH --partition=<gpu-partition>
#SBATCH --gpus=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=140gb
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
# source activate <your-environment>
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd final_runs
torchrun --standalone --nproc_per_node=2 run_dl_v4.py ${ARGS:?set ARGS}
