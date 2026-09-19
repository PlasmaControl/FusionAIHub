#!/bin/bash
#SBATCH -A fus187
#SBATCH -J eval_studies_makeup
#SBATCH -o logs/%j_studies_makeup.out
#SBATCH -e logs/%j_studies_makeup.err
#SBATCH -q debug
#SBATCH -t 1:30:00
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
set -e

# Makeup for job 5135734, where six processes died on a sys.path bug
# (eval_e2e import) fixed in commit-pending edits: a9 rollout divergence,
# b3 layerwise, b4 physics sweeps, and the LoRA cells. Adds two seed-1
# LoRA cells on the spare GCDs. Remaining afterwards: lora n200 s1 +
# aggregate figure (elm_lora.sbatch skips finished cells).
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
source scripts/evaluation/slurm/_eval_env.sh

ROCR_VISIBLE_DEVICES=0 python scripts/evaluation/study_a_rollout_divergence.py \
    > "logs/${SLURM_JOB_ID}_study_a9.log" 2>&1 &
ROCR_VISIBLE_DEVICES=1 python scripts/evaluation/study_b_layerwise.py \
    > "logs/${SLURM_JOB_ID}_study_b3lw.log" 2>&1 &
ROCR_VISIBLE_DEVICES=2 python scripts/evaluation/study_b_physics_sweeps.py \
    > "logs/${SLURM_JOB_ID}_study_b4.log" 2>&1 &

GCD=3
for CELL in "10 0" "50 0" "200 0" "10 1" "50 1"; do
  set -- ${CELL}
  ROCR_VISIBLE_DEVICES=${GCD} python scripts/evaluation/study_c_elm_lora.py \
      --n-shots $1 --seed $2 \
      > "logs/${SLURM_JOB_ID}_lora_n$1_s$2.log" 2>&1 &
  GCD=$((GCD + 1))
done
wait

python scripts/evaluation/study_b_figures.py \
    > "logs/${SLURM_JOB_ID}_study_b_figs.log" 2>&1 || true
echo "=== makeup done $(date) ==="
