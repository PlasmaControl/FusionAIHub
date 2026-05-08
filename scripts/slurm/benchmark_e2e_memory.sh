#!/bin/bash
#SBATCH --job-name=bench_e2e
#SBATCH --output=logs/%j_bench_e2e.out
#SBATCH --error=logs/%j_bench_e2e.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

# Phase C Step 5 item 5 — memory + timing benchmark for the integrated
# TS (+ optional tangtv video) E2EFoundationModel. Reports peak GPU
# memory and median step wall time for both configs at batch=128, plus
# token counts. Synthetic input; no data loader. Brief job; not part of
# the production training pipeline.

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun pixi run python ../benchmark_e2e_memory.py \
    --batch_size 128 \
    --also_batch_256