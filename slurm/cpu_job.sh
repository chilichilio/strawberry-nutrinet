#!/bin/bash
# One classical run (M1-M4, M6) on CPU. Pass the run_ml_v4.py arguments in ARGS:
#   sbatch --export=ALL,ARGS="--nutrient K --model M6 --seqfix --sampEnd --nested" slurm/cpu_job.sh
# N classical runs reported in the article use the deep-learning folds:
#   sbatch --export=ALL,V4_FOLD_MANIFEST=$PWD/final_runs/fold_membership_ML_vs_DL.csv,ARGS="--nutrient N --model M2 --sampEnd" slurm/cpu_job.sh
# Submit from the repository root (mkdir -p logs first). Adjust partition, account and environment to your cluster.
#SBATCH --job-name=nutrinet_cpu
#SBATCH --partition=<cpu-partition>
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
# source activate <your-environment>
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd final_runs
python run_ml_v4.py ${ARGS:?set ARGS}
