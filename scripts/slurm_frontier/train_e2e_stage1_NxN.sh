#!/bin/bash
# Frontier DDP launcher: train_e2e Stage1 — N nodes × 8 GCDs (production multi-node; default N=4, override with `sbatch -N <X>`)
#
# Usage:
#   sbatch scripts/slurm_frontier/train_e2e_stage1_NxN.sh
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
#   sbatch -N 8 -t 12:00:00 scripts/slurm_frontier/train_e2e_stage1_NxN.sh
#
#SBATCH -A fus187
#SBATCH -J e2e_s1_NxN
#SBATCH -o logs/%j_e2e_s1_NxN.out
#SBATCH -e logs/%j_e2e_s1_NxN.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
set -uo pipefail

PROJECT_DIR=/lustre/orion/fus187/scratch/nchen/FusionAIHub
cd "$PROJECT_DIR"
mkdir -p logs

# Per-stage MASTER_PORT default (overridable). Must be set BEFORE sourcing
# _frontier_common.sh, since that script only fills in if unset.
export MASTER_PORT="${MASTER_PORT:-29500}"

# shellcheck disable=SC1091
source scripts/slurm_frontier/_frontier_common.sh

# ─── Resource shape (taken from SLURM allocation, never hard-coded) ──────
NODES="${SLURM_JOB_NUM_NODES:-4}"
TOTAL_RANKS="${SLURM_NTASKS:-$((NODES * 8))}"
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

# Auto-resume from latest checkpoint if it exists.
LATEST="$CHECKPOINT_DIR/e2e_stage1_latest.pt"
RESUME_FLAG=""
if [ -f "$LATEST" ]; then
    RESUME_FLAG="--resume_checkpoint $LATEST"
    echo "[stage1] auto-resume from $LATEST"
fi

TRAIN_SHOTS_FLAG=""
[ -n "${TRAIN_SHOTS_YAML:-}" ] && TRAIN_SHOTS_FLAG="--train_shots_yaml $TRAIN_SHOTS_YAML"
echo "${SMOKE_BANNER}[stage1/NxN] nodes=$NODES total_ranks=$TOTAL_RANKS \
batch=$BATCH_SIZE steps=$MAX_STEPS"
echo "${SMOKE_BANNER}[stage1/NxN] master=$MASTER_ADDR:$MASTER_PORT data=$DATA_DIR"

srun -N "$NODES" -n "$TOTAL_RANKS" -c "$CPUS_PER_TASK" \
     --gpus-per-task=1 --gpu-bind=closest \
     scripts/slurm_frontier/_srun_rank_wrapper.sh \
     scripts/training/train_e2e_stage1.py \
     $RESUME_FLAG $MAX_FILES_FLAG $TRAIN_SHOTS_FLAG \
--data_dir "$DATA_DIR" \
--stats_path "$STATS_PATH" \
--checkpoint_dir "$CHECKPOINT_DIR" \
--val_fraction 0.1 \
--seed 42 \
--chunk_duration_s 0.05 \
--prediction_horizon_s 0.05 \
--step_size_s 0.01 \
--warmup_s 1.0 \
--d_model "$D_MODEL" \
--n_layers "$N_LAYERS" \
--n_heads "$N_HEADS" \
--dropout 0.1 \
--lr 1e-4 \
--min_lr 1e-6 \
--warmup_steps 2000 \
--weight_decay 0.1 \
--grad_clip 5.0 \
--batch_size "$BATCH_SIZE" \
--num_workers "$NUM_WORKERS" \
--max_steps "$MAX_STEPS" \
--log_every "$LOG_EVERY" \
--val_every "$VAL_EVERY" \
--val_max_batches "$VAL_MAX_BATCHES"