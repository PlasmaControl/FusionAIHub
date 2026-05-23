#!/bin/bash
# Frontier profile launcher: run scripts/training/profile_stage1.py twice on
# one MI250X GCD — first WITHOUT flash-attn, then WITH — and diff the two
# memory.json outputs. Designed to fit in a 1-hour batch allocation.
#
# Usage:
#   sbatch scripts/slurm_frontier/profile_stage1_1x1.sh
#
# Outputs land in:
#   profile/<JOBID>_stage1_1x1/without_flash/{trace.json,top_ops.txt,memory.json}
#   profile/<JOBID>_stage1_1x1/with_flash/{trace.json,top_ops.txt,memory.json}
#   profile/<JOBID>_stage1_1x1/comparison.txt  (printed to stdout too)
#
#SBATCH -A fus187
#SBATCH -J e2e_s1_prof
#SBATCH -o logs/%j_e2e_s1_prof.out
#SBATCH -e logs/%j_e2e_s1_prof.err
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

# ─── Profile settings ────────────────────────────────────────────────────
# Match canonical stage-1 model + modality mix so timings transfer to the
# 8x8 production run. Batch deliberately small to fit one MI250X GCD with
# full TS + video + spectro at n_layers=26.
DATA_DIR="${DATA_DIR:-/lustre/orion/fus187/proj-shared/foundation_model}"
STATS_PATH="${STATS_PATH:-/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt}"
LENGTHS_CACHE_DIR="${LENGTHS_CACHE_DIR:-runs/profile_stage1_lengths_cache}"
mkdir -p "$LENGTHS_CACHE_DIR"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_FILES="${MAX_FILES:-15}"
N_LAYERS="${N_LAYERS:-26}"
D_MODEL="${D_MODEL:-256}"
N_HEADS="${N_HEADS:-8}"
PROFILE_WAIT="${PROFILE_WAIT:-3}"
PROFILE_WARMUP="${PROFILE_WARMUP:-3}"
PROFILE_ACTIVE="${PROFILE_ACTIVE:-15}"

PROF_ROOT="profile/${SLURM_JOB_ID}_stage1_1x1"
mkdir -p "$PROF_ROOT/without_flash" "$PROF_ROOT/with_flash"
echo "[profile/1x1] outputs -> $PROF_ROOT"
echo "[profile/1x1] n_layers=$N_LAYERS d_model=$D_MODEL n_heads=$N_HEADS \
batch=$BATCH_SIZE active_steps=$PROFILE_ACTIVE max_files=$MAX_FILES"

run_profile() {
    local out_dir="$1"
    local extra_flag="$2"
    local label="$3"
    echo ""
    echo "=== [$label] starting profile run ==="
    srun -N 1 -n 1 -c "$SLURM_CPUS_PER_TASK" \
         --gpus-per-task=1 --gpu-bind=closest \
         scripts/slurm_frontier/_srun_rank_wrapper.sh \
         scripts/training/profile_stage1.py \
         --data_dir "$DATA_DIR" \
         --stats_path "$STATS_PATH" \
         --lengths_cache_dir "$LENGTHS_CACHE_DIR" \
         --output_dir "$out_dir" \
         --batch_size "$BATCH_SIZE" \
         --num_workers "$NUM_WORKERS" \
         --max_files "$MAX_FILES" \
         --d_model "$D_MODEL" \
         --n_layers "$N_LAYERS" \
         --n_heads "$N_HEADS" \
         --profile_wait "$PROFILE_WAIT" \
         --profile_warmup "$PROFILE_WARMUP" \
         --profile_active "$PROFILE_ACTIVE" \
         --use_video tangtv \
         --use_spectro ece co2 bes \
         $extra_flag
}

# Order matters: run WITHOUT first so MIOpen kernel cache is identical for
# both runs (flash-attn doesn't touch MIOpen, but other ops do).
run_profile "$PROF_ROOT/without_flash" ""              "no-flash"
run_profile "$PROF_ROOT/with_flash"    "--use_flash_attn" "flash"

echo ""
echo "=== Comparison ==="
python scripts/slurm_frontier/_compare_profiles.py \
    "$PROF_ROOT/without_flash/memory.json" \
    "$PROF_ROOT/with_flash/memory.json" \
    | tee "$PROF_ROOT/comparison.txt"

echo ""
echo "=== Done ==="
echo "Open traces in chrome://tracing or Perfetto:"
echo "  $PROF_ROOT/without_flash/trace.json"
echo "  $PROF_ROOT/with_flash/trace.json"
