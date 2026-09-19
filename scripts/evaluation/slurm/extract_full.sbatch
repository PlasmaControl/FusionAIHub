#!/bin/bash
#SBATCH -A fus187
#SBATCH -J eval_extract_full
#SBATCH -o logs/%j_eval_extract_full.out
#SBATCH -e logs/%j_eval_extract_full.err
#SBATCH -q debug
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
set -e

# Full latent/error/physics sweep: val-875 + matched train-875, 8 shard
# processes (one per GCD). Resumable — finished shots (existing npz) are
# skipped, so re-submitting after a timeout just continues.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
source scripts/evaluation/slurm/_eval_env.sh

OUT_ROOT="data/outputs/eval_suite/e2e_stage1_best"

for SPLIT in val train; do
    echo "=== split ${SPLIT} $(date) ==="
    for S in 0 1 2 3 4 5 6 7; do
        ROCR_VISIBLE_DEVICES=$S python scripts/evaluation/extract_latents.py \
            --split "${SPLIT}" --shard "$S" --n-shards 8 \
            --batch-size 8 --num-workers 5 --out-root "${OUT_ROOT}" \
            > "logs/${SLURM_JOB_ID}_extract_${SPLIT}_shard${S}.log" 2>&1 &
    done
    wait
done

python scripts/evaluation/check_extract_vs_ckpt.py "${OUT_ROOT}/latents/val"
echo "=== done $(date) ==="
