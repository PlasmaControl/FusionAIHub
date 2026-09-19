#!/bin/bash
#SBATCH -A fus187
#SBATCH -J eval_studies_gpu
#SBATCH -o logs/%j_eval_studies_gpu.out
#SBATCH -e logs/%j_eval_studies_gpu.err
#SBATCH -q debug
#SBATCH -t 1:30:00
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
set -e

# GPU study bundle, run AFTER the extraction sweep (hero-shot ranking reads
# its per-shot csvs). Independent studies on separate GCDs in parallel.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
source scripts/evaluation/slurm/_eval_env.sh

ROCR_VISIBLE_DEVICES=0 python scripts/evaluation/study_b_actuator_scan.py \
    --study all > "logs/${SLURM_JOB_ID}_study_b.log" 2>&1 &

ROCR_VISIBLE_DEVICES=1 python scripts/evaluation/study_a_rollout_divergence.py \
    > "logs/${SLURM_JOB_ID}_study_a9.log" 2>&1 &

ROCR_VISIBLE_DEVICES=2 python scripts/evaluation/study_a_recon_fidelity.py \
    > "logs/${SLURM_JOB_ID}_study_a34.log" 2>&1 &

ROCR_VISIBLE_DEVICES=3 python scripts/evaluation/study_b_layerwise.py \
    > "logs/${SLURM_JOB_ID}_study_b3lw.log" 2>&1 &

ROCR_VISIBLE_DEVICES=4 python scripts/evaluation/study_b_physics_sweeps.py \
    > "logs/${SLURM_JOB_ID}_study_b4.log" 2>&1 &

# seed-0 ELM LoRA cells on the otherwise-idle GCDs (cells are resumable;
# seed-1 cells + the aggregate figure run in elm_lora.sbatch afterwards)
GCD=5
for N in 10 50 200; do
  ROCR_VISIBLE_DEVICES=${GCD} python scripts/evaluation/study_c_elm_lora.py \
      --n-shots ${N} --seed 0 \
      > "logs/${SLURM_JOB_ID}_lora_n${N}_s0.log" 2>&1 &
  GCD=$((GCD + 1))
done

wait

# Render Study B figures from whatever npz outputs landed (CPU, cheap).
python scripts/evaluation/study_b_figures.py \
    > "logs/${SLURM_JOB_ID}_study_b_figs.log" 2>&1 || true
echo "=== studies done $(date) ==="
