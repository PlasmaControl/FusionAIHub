#!/bin/bash
#SBATCH -A fus187
#SBATCH -J eval_elm_lora
#SBATCH -o logs/%j_elm_lora.out
#SBATCH -e logs/%j_elm_lora.err
#SBATCH -q debug
#SBATCH -t 1:15:00
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
set -e

# ELM LoRA sample-efficiency grid: one (n_shots, seed) cell per GCD, all
# parallel. Cells are resumable (a finished cell's result.json is skipped),
# so resubmitting after a timeout completes the grid. Aggregation figure
# needs study_c_probes.py's tables first; the final render is re-runnable.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
source scripts/evaluation/slurm/_eval_env.sh

GCD=0
for N in 10 50 200; do
  for SEED in 0 1; do
    ROCR_VISIBLE_DEVICES=${GCD} python scripts/evaluation/study_c_elm_lora.py \
        --n-shots ${N} --seed ${SEED} \
        > "logs/${SLURM_JOB_ID}_lora_n${N}_s${SEED}.log" 2>&1 &
    GCD=$((GCD + 1))
  done
done
wait

python scripts/evaluation/study_c_elm_lora.py --aggregate \
    > "logs/${SLURM_JOB_ID}_lora_aggregate.log" 2>&1 || true
echo "=== elm lora grid done $(date) ==="
