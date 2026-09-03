#!/bin/bash
#SBATCH -A fus187
#SBATCH -J eval_s1_p1
#SBATCH -o logs/%j_eval_e2e_stage1_phase1.out
#SBATCH -e logs/%j_eval_e2e_stage1_phase1.err
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e

# Phase-1 Stage-1 evaluator (metrics only — no plots).
# Loads a frozen Stage 1 checkpoint, runs K=1 prediction shot-sharded across
# 8 GPUs of one node, writes per-window / per-shot / top-bottom CSV.gz tables.
#
# Submit from the repo root. Checkpoint and splits are passed as ARG1/ENV.
#
# Smoke (val only, 10 shots per rank, ~5 min wall):
#   EVAL_SPLITS=val EVAL_MAX_SHOTS=10 \
#     sbatch scripts/slurm_frontier/eval_e2e_stage1_phase1.sh \
#         /lustre/orion/fus187/proj-shared/models/e2e_stage1/e2e_stage1_best.pt
#
# Full val-only run:
#   EVAL_SPLITS=val \
#     sbatch scripts/slurm_frontier/eval_e2e_stage1_phase1.sh \
#         /lustre/orion/fus187/proj-shared/models/e2e_stage1/e2e_stage1_best.pt
#
# Full train + val:
#   EVAL_SPLITS="train val" -t 2:00:00 \
#     sbatch scripts/slurm_frontier/eval_e2e_stage1_phase1.sh \
#         /lustre/orion/fus187/proj-shared/models/e2e_stage1/e2e_stage1_best.pt
#
# Optional env vars:
#   EVAL_SPLITS       default "val"; pass "train val" for both splits.
#   EVAL_MAX_SHOTS    default 0 (= all shots in shard); positive int caps it.
#   EVAL_BATCH_SIZE   default 128.
#   EVAL_NUM_WORKERS  default 4.
#   EVAL_TOP_N        default 5.
#   EVAL_BOTTOM_N     default 5.
#   EVAL_OUTPUT_DIR   default eval_runs/stage1_phase1_<ckpt-stem>_<jobid>.

CHECKPOINT="${1:-${EVAL_CHECKPOINT:-}}"
if [ -z "$CHECKPOINT" ]; then
    echo "Usage: sbatch $0 <checkpoint_path>" >&2
    echo "   or  EVAL_CHECKPOINT=<path> sbatch $0" >&2
    exit 1
fi
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "${PROJECT_DIR}/scripts/slurm_frontier/_frontier_common.sh" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR (${PROJECT_DIR}) is not the repo root." >&2
    echo "       cd into the FusionAIHub repo before sbatch." >&2
    exit 1
fi
cd "${PROJECT_DIR}"
mkdir -p logs

# Distinct port from production stage1 (29500), stage2 (29502),
# stage1-smoke (29510), stage2-smoke (29512).
export MASTER_PORT=29520
source scripts/slurm_frontier/_frontier_common.sh

# ── Defaults / env overrides ─────────────────────────────────────────
EVAL_SPLITS="${EVAL_SPLITS:-val}"
EVAL_MAX_SHOTS="${EVAL_MAX_SHOTS:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-6}"
EVAL_PREFETCH_FACTOR="${EVAL_PREFETCH_FACTOR:-4}"
EVAL_TOP_N="${EVAL_TOP_N:-5}"
EVAL_BOTTOM_N="${EVAL_BOTTOM_N:-5}"

CKPT_STEM="$(basename "$CHECKPOINT" .pt)"
DEFAULT_OUT="eval_runs/stage1_phase1_${CKPT_STEM}_${SLURM_JOB_ID}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$DEFAULT_OUT}"
mkdir -p "${EVAL_OUTPUT_DIR}"

echo "[eval_s1_p1] checkpoint    : $CHECKPOINT"
echo "[eval_s1_p1] output_dir    : $EVAL_OUTPUT_DIR"
echo "[eval_s1_p1] splits        : $EVAL_SPLITS"
echo "[eval_s1_p1] max_shots     : $EVAL_MAX_SHOTS  (0 = all)"
echo "[eval_s1_p1] batch_size    : $EVAL_BATCH_SIZE"
echo "[eval_s1_p1] num_workers   : $EVAL_NUM_WORKERS"
echo "[eval_s1_p1] prefetch_fact : $EVAL_PREFETCH_FACTOR"
echo "[eval_s1_p1] world_size   : $SLURM_NTASKS (= $SLURM_JOB_NUM_NODES nodes × $SLURM_NTASKS_PER_NODE GPUs)"

# ── Per-node sampler (same pattern as training jobs) ─────────────────
SAMPLER_LOG="logs/${SLURM_JOB_ID}_sampler.log"
srun --overlap -N "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 -c 1 \
     scripts/slurm_frontier/_node_sampler.sh > "$SAMPLER_LOG" 2>&1 &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT

# ── Run ──────────────────────────────────────────────────────────────
# Each rank handles a shot-shard (rank N gets files[N::world_size]).
# No plotting in Phase 1 — just CSV.gz tables + config.json.
srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK \
     --gpus-per-task=1 --gpu-bind=closest \
     scripts/slurm_frontier/_srun_rank_wrapper.sh \
     scripts/training/eval_e2e_phase1.py \
     --checkpoint "$CHECKPOINT" \
     --data_dir /lustre/orion/fus187/proj-shared/foundation_model \
     --stats_path /lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt \
     --output_dir "$EVAL_OUTPUT_DIR" \
     --splits $EVAL_SPLITS \
     --val_fraction 0.1 \
     --seed 42 \
     --chunk_duration_s 0.05 \
     --step_size_s 0.01 \
     --warmup_s 1.0 \
     --batch_size $EVAL_BATCH_SIZE \
     --num_workers $EVAL_NUM_WORKERS \
     --prefetch_factor $EVAL_PREFETCH_FACTOR \
     --max_shots $EVAL_MAX_SHOTS \
     --top_n $EVAL_TOP_N \
     --bottom_n $EVAL_BOTTOM_N \
     --log_every 20

echo "[eval_s1_p1] outputs in: $EVAL_OUTPUT_DIR"
ls -lah "$EVAL_OUTPUT_DIR"
