#!/bin/bash
# Frontier DDP launcher: train_e2e Stage2 Extended — 1 node × 1 GCD (single-GPU smoke / dev)
#
# Usage:
#   sbatch scripts/slurm_frontier/train_e2e_stage2_extended_1x1.sh
#
# Common env overrides:
#   SMOKE=1                 # short test: MAX_STEPS=20, MAX_FILES=4, freq logs
#   MAX_STEPS=<int>         # total optimizer steps
#   MAX_FILES=<int>         # cap on training shots (debug)
#   BATCH_SIZE=<int>        # per-rank batch size (default 4)
#   NUM_WORKERS=<int>       # DataLoader workers per rank (default 4)
#   DATA_DIR=<path>         # override data root
#   CHECKPOINT_DIR=<path>   # override checkpoint dir
#   MASTER_PORT=<int>       # override port (default 29503)
#
# Override resource shape on the CLI (sbatch flags beat #SBATCH directives):
#   sbatch -N 8 -t 12:00:00 scripts/slurm_frontier/train_e2e_stage2_extended_1x1.sh
#
#SBATCH -A fus187
#SBATCH -J e2e_s2e_1x1
#SBATCH -o logs/%j_e2e_s2e_1x1.out
#SBATCH -e logs/%j_e2e_s2e_1x1.err
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
export MASTER_PORT="${MASTER_PORT:-29503}"

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
BATCH_SIZE="${BATCH_SIZE:-4}"
CURRICULUM_KS="${CURRICULUM_KS:-2,3,4}"
BLOCK_STEPS="${BLOCK_STEPS:-$((MAX_STEPS / 3))}"
GRAD_CHECKPOINT_EVERY="${GRAD_CHECKPOINT_EVERY:-2}"
D_MODEL="${D_MODEL:-256}"
N_LAYERS="${N_LAYERS:-8}"
N_HEADS="${N_HEADS:-8}"
MAE_WEIGHT="${MAE_WEIGHT:-1.0}"
COS_WEIGHT="${COS_WEIGHT:-0.3}"
MAG_WEIGHT="${MAG_WEIGHT:-0.1}"
MIN_DISP_NORM="${MIN_DISP_NORM:-0.01}"
DATA_DIR="${DATA_DIR:-/lustre/orion/fus187/proj-shared/foundation_model}"
STATS_PATH="${STATS_PATH:-data/preprocessing_stats.pt}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-runs/e2e_stage2_ext_frontier}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/e2e_stage2_delta_frontier/e2e_stage2_delta_best.pt}"
mkdir -p "$CHECKPOINT_DIR"

INIT_FLAG=""
[ -f "$INIT_CHECKPOINT" ] && INIT_FLAG="--init_checkpoint $INIT_CHECKPOINT"

LATEST="$CHECKPOINT_DIR/e2e_stage2_ext_latest.pt"
RESUME_FLAG=""
[ -f "$LATEST" ] && RESUME_FLAG="--resume_checkpoint $LATEST"

NO_AMP_FLAG=""
[ "${NO_AMP:-0}" = "1" ] && NO_AMP_FLAG="--no_amp"

NO_DISP_FLAG=""
[ "${NO_DISPLACEMENT_LOSS:-0}" = "1" ] && NO_DISP_FLAG="--no_displacement_loss"
echo "${SMOKE_BANNER}[stage2_extended/1x1] nodes=$NODES total_ranks=$TOTAL_RANKS \
batch=$BATCH_SIZE steps=$MAX_STEPS Ks=$CURRICULUM_KS"
echo "${SMOKE_BANNER}[stage2_extended/1x1] master=$MASTER_ADDR:$MASTER_PORT data=$DATA_DIR"

srun -N "$NODES" -n "$TOTAL_RANKS" -c "$CPUS_PER_TASK" \
     --gpus-per-task=1 --gpu-bind=closest \
     scripts/slurm_frontier/_srun_rank_wrapper.sh \
     scripts/training/train_e2e_stage2_extended.py \
     $INIT_FLAG $RESUME_FLAG $MAX_FILES_FLAG $NO_AMP_FLAG $NO_DISP_FLAG \
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
--curriculum_Ks "$CURRICULUM_KS" \
--block_steps "$BLOCK_STEPS" \
--mae_weight "$MAE_WEIGHT" \
--cos_weight "$COS_WEIGHT" \
--mag_weight "$MAG_WEIGHT" \
--min_disp_norm "$MIN_DISP_NORM" \
--grad_checkpoint_every "$GRAD_CHECKPOINT_EVERY" \
--lr 1e-5 \
--min_lr 1e-7 \
--warmup_steps 500 \
--weight_decay 0.01 \
--grad_clip 5.0 \
--batch_size "$BATCH_SIZE" \
--num_workers "$NUM_WORKERS" \
--max_steps "$MAX_STEPS" \
--log_every "$LOG_EVERY" \
--val_every "$VAL_EVERY" \
--val_max_batches "$VAL_MAX_BATCHES"