#!/bin/bash
# Frontier DDP launcher: train_e2e Stage3 — 1 node × 1 GCD (single-GPU smoke / dev)
#
# Usage:
#   sbatch scripts/slurm_frontier/train_e2e_stage3_1x1.sh
#
# Common env overrides:
#   SMOKE=1                 # short test: MAX_STEPS=20, MAX_FILES=4, freq logs
#   MAX_STEPS=<int>         # total optimizer steps
#   MAX_FILES=<int>         # cap on training shots (debug)
#   BATCH_SIZE=<int>        # per-rank batch size (default 16)
#   NUM_WORKERS=<int>       # DataLoader workers per rank (default 4)
#   DATA_DIR=<path>         # override data root
#   CHECKPOINT_DIR=<path>   # override checkpoint dir
#   MASTER_PORT=<int>       # override port (default 29504)
#
# Override resource shape on the CLI (sbatch flags beat #SBATCH directives):
#   sbatch -N 8 -t 12:00:00 scripts/slurm_frontier/train_e2e_stage3_1x1.sh
#
#SBATCH -A fus187
#SBATCH -J e2e_s3_1x1
#SBATCH -o logs/%j_e2e_s3_1x1.out
#SBATCH -e logs/%j_e2e_s3_1x1.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
set -uo pipefail

PROJECT_DIR=/lustre/orion/fus187/scratch/nchen/FusionAIHub
cd "$PROJECT_DIR"
mkdir -p logs

# Per-stage MASTER_PORT default (overridable). Must be set BEFORE sourcing
# _frontier_common.sh, since that script only fills in if unset.
export MASTER_PORT="${MASTER_PORT:-29504}"

# shellcheck disable=SC1091
source scripts/slurm_frontier/_frontier_common.sh
conda activate "$CONDA_ENV_PATH"

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
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
K_MIN="${K_MIN:-2}"
K_MAX="${K_MAX:-4}"
N_CURRICULUM_BLOCKS="${N_CURRICULUM_BLOCKS:-2}"
CURRICULUM_STEPS="${CURRICULUM_STEPS:-$((MAX_STEPS / 2))}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16.0}"
POOL_SIZE="${POOL_SIZE:-50}"
BUFFER_SIZE="${BUFFER_SIZE:-500}"
BUFFER_REFRESH_PERIOD="${BUFFER_REFRESH_PERIOD:-50}"
BUFFER_REFRESH_FRACTION="${BUFFER_REFRESH_FRACTION:-0.1}"
D_MODEL="${D_MODEL:-256}"
N_LAYERS="${N_LAYERS:-8}"
N_HEADS="${N_HEADS:-8}"
DATA_DIR="${DATA_DIR:-/lustre/orion/fus187/proj-shared/foundation_model}"
STATS_PATH="${STATS_PATH:-data/preprocessing_stats.pt}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-runs/e2e_stage3_frontier}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/e2e_stage2_delta_frontier/e2e_stage2_delta_best.pt}"
mkdir -p "$CHECKPOINT_DIR"

INIT_FLAG=""
[ -f "$INIT_CHECKPOINT" ] && INIT_FLAG="--init_checkpoint $INIT_CHECKPOINT"

LATEST="$CHECKPOINT_DIR/e2e_stage3_latest.pt"
RESUME_FLAG=""
[ -f "$LATEST" ] && RESUME_FLAG="--resume_checkpoint $LATEST"

NO_AMP_FLAG=""
[ "${NO_AMP:-0}" = "1" ] && NO_AMP_FLAG="--no_amp"

USE_DISP_FLAG="--use_displacement_loss"
[ "${NO_DISPLACEMENT_LOSS:-0}" = "1" ] && USE_DISP_FLAG=""
echo "${SMOKE_BANNER}[stage3/1x1] nodes=$NODES total_ranks=$TOTAL_RANKS \
batch=$BATCH_SIZE steps=$MAX_STEPS K=[$K_MIN,$K_MAX]"
echo "${SMOKE_BANNER}[stage3/1x1] master=$MASTER_ADDR:$MASTER_PORT data=$DATA_DIR"

srun -N "$NODES" -n "$TOTAL_RANKS" -c "$CPUS_PER_TASK" \
     --gpus-per-task=1 --gpu-bind=closest \
     scripts/slurm_frontier/_srun_rank_wrapper.sh \
     scripts/training/train_e2e_stage3.py \
     $INIT_FLAG $RESUME_FLAG $MAX_FILES_FLAG $NO_AMP_FLAG $USE_DISP_FLAG \
--data_dir "$DATA_DIR" \
--stats_path "$STATS_PATH" \
--checkpoint_dir "$CHECKPOINT_DIR" \
--val_fraction 0.1 \
--seed 42 \
--chunk_duration_s 0.05 \
--step_size_s 0.01 \
--warmup_s 1.0 \
--d_model "$D_MODEL" \
--n_layers "$N_LAYERS" \
--n_heads "$N_HEADS" \
--dropout 0.1 \
--lora_rank "$LORA_RANK" \
--lora_alpha "$LORA_ALPHA" \
--K_min "$K_MIN" \
--K_max "$K_MAX" \
--n_curriculum_blocks "$N_CURRICULUM_BLOCKS" \
--curriculum_steps "$CURRICULUM_STEPS" \
--pool_size "$POOL_SIZE" \
--buffer_size "$BUFFER_SIZE" \
--buffer_refresh_period "$BUFFER_REFRESH_PERIOD" \
--buffer_refresh_fraction "$BUFFER_REFRESH_FRACTION" \
--lr 3e-5 \
--min_lr 1e-7 \
--warmup_steps 200 \
--weight_decay 0.01 \
--grad_clip 5.0 \
--cos_weight 0.3 \
--mag_weight 0.1 \
--min_disp_norm 0.01 \
--batch_size "$BATCH_SIZE" \
--num_workers "$NUM_WORKERS" \
--max_steps "$MAX_STEPS" \
--log_every "$LOG_EVERY" \
--val_every "$VAL_EVERY" \
--val_batch_size "$VAL_BATCH_SIZE"