#!/bin/bash
#SBATCH -A fus187
#SBATCH -J eval_s2_p1
#SBATCH -o logs/%j_eval_e2e_stage2_phase1.out
#SBATCH -e logs/%j_eval_e2e_stage2_phase1.err
#SBATCH -t 4:00:00
#SBATCH -p extended
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e

# Phase-1 Stage-2 evaluator (metrics + PASS/FAIL gates — no plots).
# Loads a frozen Stage 2 delta-rollout checkpoint, runs K-step
# autoregressive rollout (K autodetected from ckpt['args']['K_max'])
# shot-sharded across 8 GPUs of one node, writes per-window /
# per-shot / top-bottom CSV.gz tables + summary.md with PASS/FAIL on
# the four Stage-2 gates (model<copy@k=1, model<copy@k=K, dir_cos>0,
# mag_ratio in [0.3, 3.0]).
#
# Walltime budget: K-step rollout is ~K× per-window backbone cost, so
# Stage 2 K=10 runs ~5-10× longer than Stage 1's Phase 1 (1h base
# → 4h here). For d=1024 also override EVAL_BATCH_SIZE downward.
#
# Submit from the repo root. Checkpoint passed as ARG1/ENV.
#
# Smoke (val only, 10 shots per rank, ~30 min wall):
#   EVAL_SPLITS=val EVAL_MAX_SHOTS=10 \
#     sbatch scripts/slurm_frontier/eval_e2e_stage2_phase1.sh \
#         /lustre/orion/fus187/proj-shared/models/e2e_stage2_delta_48L/e2e_stage2_delta_best.pt
#
# Full val-only:
#   sbatch scripts/slurm_frontier/eval_e2e_stage2_phase1.sh \
#         /lustre/orion/fus187/proj-shared/models/e2e_stage2_delta_48L/e2e_stage2_delta_best.pt
#
# d=1024 Stage 2: needs smaller batch + more nodes; mirror the d=1024
# Stage 1 sbatch tuning from project-stage1-d1024-eval-config.md:
#   EVAL_BATCH_SIZE=32 EVAL_NUM_WORKERS=0 \
#     sbatch -N 8 scripts/slurm_frontier/eval_e2e_stage2_phase1.sh \
#         /lustre/orion/fus187/proj-shared/models/e2e_stage2_delta_d1024_48L/e2e_stage2_delta_best.pt
#
# Override the K horizon (e.g. evaluate a mid-curriculum checkpoint at
# the K it has actually been trained to) via EVAL_K — autodetect uses
# ckpt['args']['K_max'] by default.
#
# Optional env vars (all forward to the Python script):
#   EVAL_SPLITS       default "val"; pass "train val" for both.
#   EVAL_MAX_SHOTS    default 0 (= all shots in shard).
#   EVAL_BATCH_SIZE   default 128 (use 32-64 for d=1024).
#   EVAL_NUM_WORKERS  default 6.
#   EVAL_K            default 0 (autodetect from checkpoint).
#   EVAL_TOP_N        default 5.
#   EVAL_BOTTOM_N     default 5.
#   EVAL_OUTPUT_DIR   default eval_runs/stage2_phase1_<ckpt-stem>_<jobid>.

CHECKPOINT="${1:-${EVAL_CHECKPOINT:-}}"
if [ -z "$CHECKPOINT" ]; then
    echo "Usage: sbatch $0 <stage2_checkpoint_path>" >&2
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

# Distinct port from Stage 1 eval Phase 1 (29520), so Stage 1 + Stage 2
# eval jobs can run in parallel on different nodes without colliding.
export MASTER_PORT=29525
source scripts/slurm_frontier/_frontier_common.sh

# ── Defaults / env overrides ─────────────────────────────────────────
EVAL_SPLITS="${EVAL_SPLITS:-val}"
EVAL_MAX_SHOTS="${EVAL_MAX_SHOTS:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-6}"
EVAL_PREFETCH_FACTOR="${EVAL_PREFETCH_FACTOR:-4}"
EVAL_TOP_N="${EVAL_TOP_N:-5}"
EVAL_BOTTOM_N="${EVAL_BOTTOM_N:-5}"
EVAL_K="${EVAL_K:-0}"

CKPT_STEM="$(basename "$CHECKPOINT" .pt)"
DEFAULT_OUT="eval_runs/stage2_phase1_${CKPT_STEM}_${SLURM_JOB_ID}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$DEFAULT_OUT}"
mkdir -p "${EVAL_OUTPUT_DIR}"

echo "[eval_s2_p1] checkpoint    : $CHECKPOINT"
echo "[eval_s2_p1] output_dir    : $EVAL_OUTPUT_DIR"
echo "[eval_s2_p1] splits        : $EVAL_SPLITS"
echo "[eval_s2_p1] max_shots     : $EVAL_MAX_SHOTS  (0 = all)"
echo "[eval_s2_p1] batch_size    : $EVAL_BATCH_SIZE"
echo "[eval_s2_p1] num_workers   : $EVAL_NUM_WORKERS"
echo "[eval_s2_p1] K (0=auto)    : $EVAL_K"
echo "[eval_s2_p1] world_size    : $SLURM_NTASKS (= $SLURM_JOB_NUM_NODES nodes × $SLURM_NTASKS_PER_NODE GPUs)"

# Per-node sampler for memory/GPU telemetry (same pattern as training).
SAMPLER_LOG="logs/${SLURM_JOB_ID}_sampler.log"
srun --overlap -N "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 -c 1 \
     scripts/slurm_frontier/_node_sampler.sh > "$SAMPLER_LOG" 2>&1 &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT

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
     --K $EVAL_K \
     --log_every 20

echo "[eval_s2_p1] outputs in: $EVAL_OUTPUT_DIR"
ls -lah "$EVAL_OUTPUT_DIR"
