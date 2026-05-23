#!/bin/bash
# Memory-ceiling probe: build E2E model at 300M params and try one
# forward+backward on a single MI250X GCD. Runs the same probe under four
# configurations to find what actually fits:
#   1) standard attention,            no grad checkpoint
#   2) sdpa attention,                no grad checkpoint
#   3) sdpa attention,                gradient checkpoint
#   4) sdpa attention + grad ckpt + K=10 rollout (stage 2 pattern)
#
# Usage: sbatch scripts/slurm_frontier/memory_probe_e2e.sh
#
#SBATCH -A fus187
#SBATCH -J mem_probe
#SBATCH -o logs/%j_mem_probe.out
#SBATCH -e logs/%j_mem_probe.err
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

D_MODEL="${D_MODEL:-1024}"
N_LAYERS="${N_LAYERS:-24}"
N_HEADS="${N_HEADS:-16}"
BATCH="${BATCH:-4}"

run_probe() {
    local label="$1"; shift
    echo ""
    echo "================================================================"
    echo "=== $label ==="
    echo "================================================================"
    srun -N 1 -n 1 -c "$SLURM_CPUS_PER_TASK" \
         --gpus-per-task=1 --gpu-bind=closest \
         scripts/slurm_frontier/_srun_rank_wrapper.sh \
         scripts/training/memory_probe_e2e.py \
         --d_model "$D_MODEL" --n_layers "$N_LAYERS" --n_heads "$N_HEADS" \
         --batch_size "$BATCH" \
         "$@" || echo "[$label] non-zero exit (likely OOM — see above)"
}

run_probe "(1) standard attn, no ckpt"           --attn_impl standard
run_probe "(2) sdpa attn, no ckpt"               --attn_impl sdpa
run_probe "(3) sdpa attn, grad ckpt"             --attn_impl sdpa --gradient_checkpoint
run_probe "(4) sdpa attn, grad ckpt, K=10 rollout" \
                                                 --attn_impl sdpa --gradient_checkpoint \
                                                 --K_rollout 10

echo ""
echo "=== Done. ==="
