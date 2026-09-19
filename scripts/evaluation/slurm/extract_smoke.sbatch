#!/bin/bash
#SBATCH -A fus187
#SBATCH -J eval_extract_smoke
#SBATCH -o logs/%j_eval_extract_smoke.out
#SBATCH -e logs/%j_eval_extract_smoke.err
#SBATCH -q debug
#SBATCH -t 0:25:00
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
set -e

# Gate before the full sweep: 2 val shots on one GCD, then a numerical
# cross-check of pooled MAE vs ckpt['metrics']. Submit from the repo root.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
source scripts/evaluation/slurm/_eval_env.sh

OUT_ROOT="data/outputs/eval_suite/e2e_stage1_best_smoke"

ROCR_VISIBLE_DEVICES=0 python scripts/evaluation/extract_latents.py \
    --split val --max-shots 2 --batch-size 8 --num-workers 5 \
    --out-root "${OUT_ROOT}"

python scripts/evaluation/check_extract_vs_ckpt.py \
    "${OUT_ROOT}/latents/val"
