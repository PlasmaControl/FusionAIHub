#!/bin/bash
# Frontier DDP launcher: train_e2e Stage1 — 1 node × 1 GCD (single-GPU smoke / dev)
#
# Usage:
#   sbatch scripts/slurm_frontier/train_e2e_stage1_1x1.sh
#
# Common env overrides:
#   SMOKE=1                 # short test: MAX_STEPS=20, MAX_FILES=4, freq logs
#   MAX_STEPS=<int>         # total optimizer steps
#   MAX_FILES=<int>         # cap on training shots (debug)
#   BATCH_SIZE=<int>        # per-rank batch size (default 16)
#   NUM_WORKERS=<int>       # DataLoader workers per rank (default 4)
#   DATA_DIR=<path>         # override data root
#   CHECKPOINT_DIR=<path>   # override checkpoint dir
#   MASTER_PORT=<int>       # override port (default 29500)
#
# Override resource shape on the CLI (sbatch flags beat #SBATCH directives):
#   sbatch -N 8 -t 12:00:00 scripts/slurm_frontier/train_e2e_stage1_1x1.sh
#
#SBATCH -A fus187
#SBATCH -J e2e_s1_1x1
#SBATCH -o logs/%j_e2e_s1_1x1.out
#SBATCH -e logs/%j_e2e_s1_1x1.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lustre/orion/fus187/scratch/nchen/FusionAIHub}"
cd "$PROJECT_DIR"
mkdir -p logs

# Per-stage MASTER_PORT default (overridable). Must be set BEFORE sourcing
# _frontier_common.sh, since that script only fills in if unset.
export MASTER_PORT="${MASTER_PORT:-29500}"

# shellcheck disable=SC1091
source scripts/slurm_frontier/_frontier_common.sh

# ─── Resource shape (taken from SLURM allocation, never hard-coded) ──────
NODES="${SLURM_JOB_NUM_NODES:-1}"
TOTAL_RANKS="${SLURM_NTASKS:-$((NODES * 1))}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-7}"

# ─── SMOKE=1 overrides for end-to-end smoke testing ──────────────────────
if [ "${SMOKE:-0}" = "1" ]; then
    MAX_STEPS="${MAX_STEPS:-20}"
    MAX_FILES="${MAX_FILES:-4}"
    NUM_WORKERS="${NUM_WORKERS:-2}"
    LOG_EVERY="${LOG_EVERY:-2}"
    VAL_EVERY="${VAL_EVERY:-10}"
    VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-2}"
    SMOKE_BANNER="[SMOKE] "
else
    MAX_STEPS="${MAX_STEPS:-1000}"
    NUM_WORKERS="${NUM_WORKERS:-4}"
    LOG_EVERY="${LOG_EVERY:-50}"
    VAL_EVERY="${VAL_EVERY:-200}"
    VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-20}"
    SMOKE_BANNER=""
fi

MAX_FILES_FLAG=""
[ -n "${MAX_FILES:-}" ] && MAX_FILES_FLAG="--max_files $MAX_FILES"

# ─── Stage-specific defaults & init/resume flags ─────────────────────────
BATCH_SIZE="${BATCH_SIZE:-16}"
D_MODEL="${D_MODEL:-256}"
N_LAYERS="${N_LAYERS:-8}"
N_HEADS="${N_HEADS:-8}"
DATA_DIR="${DATA_DIR:-/lustre/orion/fus187/proj-shared/foundation_model}"
STATS_PATH="${STATS_PATH:-data/preprocessing_stats.pt}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-runs/e2e_stage1_frontier}"
mkdir -p "$CHECKPOINT_DIR"
# ISOLATE the lengths cache per-run (default = this run's own checkpoint dir).
# CRITICAL: the trainer's default --lengths_cache_dir is the SHARED
# foundation_model_meta dir, and the cache filename (lengths_e2e_stage1_train.pt)
# is fixed. A small-file-list 1x1 run (overfit/smoke) writing there OVERWRITES
# the ALL-shots production cache → production jobs then cold-scan 7878 files at
# 64 ranks → NCCL-watchdog crash (happened 2026-07-10). Keep 1x1 caches local.
LENGTHS_CACHE_DIR="${LENGTHS_CACHE_DIR:-$CHECKPOINT_DIR}"
mkdir -p "$LENGTHS_CACHE_DIR"

# ─── Opt-in spectrogram / FSQ code-head flags (all default OFF → the plain
#     TS-only smoke behaves exactly as before; only set these for an FSQ run) ──
SPECTRO_FLAGS=""
[ -n "${USE_SPECTRO:-}" ]      && SPECTRO_FLAGS="$SPECTRO_FLAGS --use_spectro ${USE_SPECTRO}"
[ -n "${USE_VIDEO:-}" ]        && SPECTRO_FLAGS="$SPECTRO_FLAGS --use_video ${USE_VIDEO}"
[ -n "${SPECTRO_PATCH_F:-}" ]  && SPECTRO_FLAGS="$SPECTRO_FLAGS --spectro_patch_f ${SPECTRO_PATCH_F}"
[ -n "${SPECTRO_PATCH_T:-}" ]  && SPECTRO_FLAGS="$SPECTRO_FLAGS --spectro_patch_t ${SPECTRO_PATCH_T}"
if [ "${SPEC_FSQ:-0}" = "1" ]; then
    : "${SPEC_FSQ_CODEC_DIR:?SPEC_FSQ=1 requires SPEC_FSQ_CODEC_DIR}"
    SPECTRO_FLAGS="$SPECTRO_FLAGS --spec_fsq --spec_fsq_codec_dir ${SPEC_FSQ_CODEC_DIR}"
    [ -n "${SPEC_CODE_CLASS_WEIGHT:-}" ]   && SPECTRO_FLAGS="$SPECTRO_FLAGS --spec_code_class_weight ${SPEC_CODE_CLASS_WEIGHT}"
    [ -n "${SPEC_CODE_WEIGHT_BATCHES:-}" ] && SPECTRO_FLAGS="$SPECTRO_FLAGS --spec_code_weight_batches ${SPEC_CODE_WEIGHT_BATCHES}"
fi
[ -n "${EXTRA_FLAGS:-}" ]      && SPECTRO_FLAGS="$SPECTRO_FLAGS ${EXTRA_FLAGS}"

# Auto-resume from latest checkpoint if it exists.
LATEST="$CHECKPOINT_DIR/e2e_stage1_latest.pt"
RESUME_FLAG=""
if [ -f "$LATEST" ]; then
    RESUME_FLAG="--resume_checkpoint $LATEST"
    echo "[stage1] auto-resume from $LATEST"
fi

TRAIN_SHOTS_FLAG=""
[ -n "${TRAIN_SHOTS_YAML:-}" ] && TRAIN_SHOTS_FLAG="--train_shots_yaml $TRAIN_SHOTS_YAML"
echo "${SMOKE_BANNER}[stage1/1x1] nodes=$NODES total_ranks=$TOTAL_RANKS \
batch=$BATCH_SIZE steps=$MAX_STEPS"
echo "${SMOKE_BANNER}[stage1/1x1] master=$MASTER_ADDR:$MASTER_PORT data=$DATA_DIR"

# ─── Optional GPU+CPU profiling sidecar (PROFILE=1) ──────────────────────
PROF_PID=""
if [ "${PROFILE:-0}" = "1" ]; then
    PROF_DIR="${PROF_DIR:-profile/${SLURM_JOB_ID}_$(basename "$0" .sh)}"
    mkdir -p "$PROF_DIR"
    echo "[profile] sampling rocm-smi + mpstat (1 Hz) -> $PROF_DIR"
    srun --overlap --jobid="$SLURM_JOB_ID" \
         -N "$NODES" -n "$NODES" --ntasks-per-node=1 \
         --gpus-per-task=0 --cpus-per-task=2 \
         scripts/slurm_frontier/_profile_node.sh "$PROF_DIR" &
    PROF_PID=$!
fi
trap '[ -n "${PROF_PID:-}" ] && kill "$PROF_PID" 2>/dev/null; true' EXIT

srun --overlap -N "$NODES" -n "$TOTAL_RANKS" -c "$CPUS_PER_TASK" \
     --gpus-per-task=1 --gpu-bind=closest \
     scripts/slurm_frontier/_srun_rank_wrapper.sh \
     scripts/training/train_e2e_stage1.py \
     $RESUME_FLAG $MAX_FILES_FLAG $TRAIN_SHOTS_FLAG \
--data_dir "$DATA_DIR" \
--stats_path "$STATS_PATH" \
--checkpoint_dir "$CHECKPOINT_DIR" \
--lengths_cache_dir "$LENGTHS_CACHE_DIR" \
--val_fraction 0.1 \
--seed 42 \
--chunk_duration_s 0.05 \
--prediction_horizon_s "${PRED_HORIZON:-0.05}" \
--step_size_s 0.01 \
--warmup_s 1.0 \
--d_model "$D_MODEL" \
--n_layers "$N_LAYERS" \
--n_heads "$N_HEADS" \
--dropout 0.1 \
--lr "${LR:-1e-4}" \
--min_lr 1e-6 \
--warmup_steps "${WARMUP_STEPS:-2000}" \
--weight_decay 0.1 \
--grad_clip 5.0 \
--batch_size "$BATCH_SIZE" \
--num_workers "$NUM_WORKERS" \
--max_steps "$MAX_STEPS" \
--log_every "$LOG_EVERY" \
--val_every "$VAL_EVERY" \
--val_max_batches "$VAL_MAX_BATCHES" \
$SPECTRO_FLAGS