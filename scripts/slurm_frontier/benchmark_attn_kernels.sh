#!/bin/bash
# Kernel-level benchmark of attention implementations on MI250X.
# Sweeps head_dim x seq_len for 4 impls (flash_ext, sdpa_math, sdpa_flash,
# sdpa_auto). Sanity-checks whether flash-attn wins anywhere on Frontier
# before we commit to it for any production stage.
#
# Usage:
#     sbatch scripts/slurm_frontier/benchmark_attn_kernels.sh
#
#SBATCH -A fus187
#SBATCH -J attn_bench
#SBATCH -o logs/%j_attn_bench.out
#SBATCH -e logs/%j_attn_bench.err
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
set -uo pipefail

PROJECT_DIR=/lustre/orion/fus187/scratch/nchen/FusionAIHub
cd "$PROJECT_DIR"
mkdir -p logs

# shellcheck disable=SC1091
source scripts/slurm_frontier/_frontier_common.sh

OUT_DIR="profile/${SLURM_JOB_ID}_attn_bench"
mkdir -p "$OUT_DIR"
echo "[bench] outputs -> $OUT_DIR"
echo "[bench] FLASH_ATTENTION_TRITON_AMD_ENABLE=${FLASH_ATTENTION_TRITON_AMD_ENABLE}"

srun -N 1 -n 1 -c "$SLURM_CPUS_PER_TASK" \
     --gpus-per-task=1 --gpu-bind=closest \
     scripts/slurm_frontier/_srun_rank_wrapper.sh \
     scripts/training/benchmark_attn_kernels.py \
     --out_dir "$OUT_DIR" \
     --batch 4 \
     --n_heads 16 \
     --head_dims 32 64 128 \
     --seq_lens 32 128 512 2048 4096 \
     --dtype bf16

echo ""
echo "=== Done. Summary: $OUT_DIR/summary.md ==="
