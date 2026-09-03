#!/bin/bash
#SBATCH -A fus187
#SBATCH -J mem_probe
#SBATCH -o logs/%j_mem_probe.out
#SBATCH -e logs/%j_mem_probe.err
#SBATCH -t 01:30:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
set -uo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "${PROJECT_DIR}/scripts/slurm_frontier/_frontier_settings.sh" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR (${PROJECT_DIR}) is not the repo root." >&2
    echo "       cd into the FusionAIHub repo before sbatch." >&2
    exit 1
fi
cd "${PROJECT_DIR}"
mkdir -p logs

# shellcheck disable=SC1091
source scripts/slurm_frontier/_frontier_settings.sh

BATCH="${BATCH:-1}"

run_probe() {
    local label="$1"; local d_model="$2"; local n_layers="$3"
    local n_heads="$4"; local k="$5"; shift 5
    echo ""
    echo "================================================================"
    echo "=== $label  (d_model=$d_model n_layers=$n_layers n_heads=$n_heads K=$k batch=$BATCH) ==="
    echo "================================================================"
    srun -N 1 -n 1 -c "$SLURM_CPUS_PER_TASK" \
         --gpus-per-task=1 --gpu-bind=closest \
         scripts/slurm_frontier/_srun_rank_wrapper.sh \
         scripts/training/memory_probe_e2e.py \
         --d_model "$d_model" --n_layers "$n_layers" --n_heads "$n_heads" \
         --batch_size "$BATCH" --K_rollout "$k" \
         "$@" || echo "[$label] non-zero exit (likely OOM — see above)"
}

COMMON_FLAGS=(--attn_impl sdpa --gradient_checkpoint)

# Single-shot probe: does 2.68B fit at K=50?
# Prior at this exact shape: K=25 → 53.73 GB peak (optim.step-bound).
# K=50 doubles rollout activations; predicted borderline (60-65 GB peak).
run_probe "2.68B @ K=50 (d=2048 L=32)"  2048 32 32  50 "${COMMON_FLAGS[@]}"

echo ""
echo "=== Done. ==="
