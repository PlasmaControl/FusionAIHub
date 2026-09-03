#!/bin/bash
# Frontier launcher — POC: generative spectrogram head + resize-conv video.
# From-scratch Stage-1 (single-window) run on a SUBSET of shots to validate
# (a) the flow-matching SpectrogramFlowHead recovers coherent modes (TVR ↑
# off the documented ~0.15 collapse floor) and (b) the resize-conv video
# decoder removes the 12×12 checkerboard — before committing to the full
# ~10-day 1024/48L retrain.  See plan: dapper-pondering-backus.md.
#
# Usage:
#   sbatch scripts/slurm_frontier/train_e2e_stage1_poc_genvid.sh           # full POC (4N)
#   SMOKE=1 sbatch -N 1 scripts/slurm_frontier/train_e2e_stage1_poc_genvid.sh   # quick smoke
#
# Env overrides: SMOKE, MAX_STEPS, MAX_FILES, BATCH_SIZE, D_MODEL, N_LAYERS,
#                NUM_WORKERS, MASTER_PORT, CHECKPOINT_DIR, DATA_DIR.
#
#SBATCH -A fus187
#SBATCH -J e2e_poc_genvid
#SBATCH -o logs/%j_e2e_poc_genvid.out
#SBATCH -e logs/%j_e2e_poc_genvid.err
#SBATCH -t 08:00:00
#SBATCH -p batch
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
set -uo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "${PROJECT_DIR}/scripts/slurm_frontier/_frontier_common.sh" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR (${PROJECT_DIR}) is not the repo root." >&2
    exit 1
fi
cd "${PROJECT_DIR}"
mkdir -p logs

# Distinct from Stage 1 d=1024 (29515), Stage 2 delta (29503), ext (29504).
export MASTER_PORT="${MASTER_PORT:-29530}"
# shellcheck disable=SC1091
source scripts/slurm_frontier/_frontier_common.sh

NODES="${SLURM_JOB_NUM_NODES:-1}"
TOTAL_RANKS="${SLURM_NTASKS:-$((NODES * 1))}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-7}"

# ─── POC scale (overridable) ─────────────────────────────────────────────
if [ "${SMOKE:-0}" = "1" ]; then
    MAX_STEPS="${MAX_STEPS:-20}"
    MAX_FILES="${MAX_FILES:-8}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    NUM_WORKERS="${NUM_WORKERS:-2}"
    LOG_EVERY="${LOG_EVERY:-2}"
    VAL_EVERY="${VAL_EVERY:-10}"
    VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-2}"
    BANNER="[SMOKE] "
else
    MAX_STEPS="${MAX_STEPS:-4000}"
    # ≥ ranks after the ~60% video-presence filter: the DistributedTwoLevel
    # sampler shards files across ranks (needs n_files ≥ ranks). 400 → ~60 val
    # files, safe up to 64 ranks; 200 broke at 32 ranks (val 30 < 32).
    MAX_FILES="${MAX_FILES:-400}"
    BATCH_SIZE="${BATCH_SIZE:-32}"
    NUM_WORKERS="${NUM_WORKERS:-4}"
    LOG_EVERY="${LOG_EVERY:-50}"
    VAL_EVERY="${VAL_EVERY:-250}"
    VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-40}"
    BANNER=""
fi

D_MODEL="${D_MODEL:-512}"
N_LAYERS="${N_LAYERS:-12}"
N_HEADS="${N_HEADS:-8}"

# Optional full-frequency spectro patch (SPECTRO_PATCH_F=512 SPECTRO_PATCH_T=4).
# Empty → registry default (32/64, 8). Changing the patch is a from-scratch
# architecture change, so pair with a fresh CHECKPOINT_DIR.
SPECTRO_PATCH_FLAGS=""
[ -n "${SPECTRO_PATCH_F:-}" ] && SPECTRO_PATCH_FLAGS="$SPECTRO_PATCH_FLAGS --spectro_patch_f $SPECTRO_PATCH_F"
[ -n "${SPECTRO_PATCH_T:-}" ] && SPECTRO_PATCH_FLAGS="$SPECTRO_PATCH_FLAGS --spectro_patch_t $SPECTRO_PATCH_T"

# ── Mode-prediction POC knobs (default off → unchanged genvid POC) ──
# SPEC_MASK=1: predict the mode mask from BACKBONE TOKENS (dice-only loss).
# SPEC_INPUT_COND unset: NO persistence prior → the model must PREDICT modes,
#   not copy the input — the whole point of the learnability test.
# SPEC_MASK_LAMBDA: mask weight (first-class → shapes the backbone from scratch).
# SPEC_FLOW_LAMBDA=0: drop the (L2, collapsing) flow objective for the POC.
# USE_VIDEO="": drop tangtv for speed (spectro + profiles + actuators suffice
#   to test whether ECE mode dynamics are learnable beyond persistence).
SPEC_MASK_FLAG=""
[ -n "${SPEC_MASK:-}" ] && SPEC_MASK_FLAG="--spec_mask"
SPEC_INPUT_COND_FLAG=""
[ -n "${SPEC_INPUT_COND:-}" ] && SPEC_INPUT_COND_FLAG="--spec_input_cond"
SPEC_INPUT_FEAT_FLAG=""
[ -n "${SPEC_INPUT_FEAT:-}" ] && SPEC_INPUT_FEAT_FLAG="--spec_input_feat"
# Explicit shot lists (default empty → glob+max_files split). Used for the
# single-shot overfit (train==val==200729) to test "can it FIT modes".
TRAIN_SHOTS_FLAG=""
[ -n "${TRAIN_SHOTS_YAML:-}" ] && TRAIN_SHOTS_FLAG="--train_shots_yaml ${TRAIN_SHOTS_YAML}"
VAL_SHOTS_FLAG=""
[ -n "${VAL_SHOTS_YAML:-}" ] && VAL_SHOTS_FLAG="--val_shots_yaml ${VAL_SHOTS_YAML}"
USE_VIDEO="${USE_VIDEO-tangtv}"
VIDEO_FLAG=""
[ -n "$USE_VIDEO" ] && VIDEO_FLAG="--use_video $USE_VIDEO"
USE_SPECTRO="${USE_SPECTRO:-ece co2}"
DATA_DIR="${DATA_DIR:-/lustre/orion/fus187/proj-shared/foundation_model}"
STATS_PATH="${STATS_PATH:-/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt}"
# SMOKE writes to a SEPARATE dir so its (possibly stale-architecture) tiny
# checkpoints can never be auto-resumed by the full POC run.
_POC_DIR_TAG="genvid"; [ "${SMOKE:-0}" = "1" ] && _POC_DIR_TAG="genvid_smoke"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/lustre/orion/fus187/proj-shared/models/e2e_poc_${_POC_DIR_TAG}}"
# POC-specific length / video-presence cache (the shared meta cache is keyed
# to the full production file set; --max_files uses a different subset).
LENGTHS_CACHE_DIR="${LENGTHS_CACHE_DIR:-${CHECKPOINT_DIR}/cache}"
mkdir -p "$CHECKPOINT_DIR" "$LENGTHS_CACHE_DIR"

# Auto-resume from latest if present (latest.pt saves each val).
LATEST="$CHECKPOINT_DIR/e2e_stage1_latest.pt"
RESUME_FLAG=""
if [ -f "$LATEST" ]; then
    RESUME_FLAG="--resume_checkpoint $LATEST"
    echo "[poc] auto-resume from $LATEST"
fi

echo "${BANNER}[poc/genvid] nodes=$NODES ranks=$TOTAL_RANKS d_model=$D_MODEL \
n_layers=$N_LAYERS batch=$BATCH_SIZE steps=$MAX_STEPS files=$MAX_FILES"
echo "${BANNER}[poc/genvid] master=$MASTER_ADDR:$MASTER_PORT ckpt=$CHECKPOINT_DIR"

# Per-node GPU/CPU sampler sidecar → logs/<jobid>_sampler.log lines:
#   "<ts> <host> ram=used/total_PCT% gpu_busy=PCT% vram=PCT%". ~50ms/60s.
SAMPLER_LOG="logs/${SLURM_JOB_ID}_sampler.log"
srun --overlap -N "$NODES" --ntasks-per-node=1 -c 1 \
     scripts/slurm_frontier/_node_sampler.sh > "$SAMPLER_LOG" 2>&1 &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT

srun --overlap -N "$NODES" -n "$TOTAL_RANKS" -c "$CPUS_PER_TASK" \
     --gpus-per-task=1 --gpu-bind=closest \
     scripts/slurm_frontier/_srun_rank_wrapper.sh \
     scripts/training/train_e2e_stage1.py \
     $RESUME_FLAG \
     --data_dir "$DATA_DIR" \
     --stats_path "$STATS_PATH" \
     --checkpoint_dir "$CHECKPOINT_DIR" \
     --lengths_cache_dir "$LENGTHS_CACHE_DIR" \
     --max_files "$MAX_FILES" \
     ${TRAIN_SHOTS_FLAG} \
     ${VAL_SHOTS_FLAG} \
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
     --backbone_grad_checkpoint \
     --lr 3e-4 \
     --min_lr 1e-6 \
     --warmup_steps 300 \
     --weight_decay 0.1 \
     --grad_clip 5.0 \
     --batch_size "$BATCH_SIZE" \
     --num_workers "$NUM_WORKERS" \
     --max_steps "$MAX_STEPS" \
     --log_every "$LOG_EVERY" \
     --val_every "$VAL_EVERY" \
     --val_max_batches "$VAL_MAX_BATCHES" \
     ${VIDEO_FLAG} \
     --use_spectro ${USE_SPECTRO} \
     --video_resize_conv \
     --spec_generative \
     --spec_flow_steps 6 \
     --spec_flow_lambda "${SPEC_FLOW_LAMBDA:-1.0}" \
     --spec_mask_lambda "${SPEC_MASK_LAMBDA:-0.0}" \
     --spec_mae_lambda "${SPEC_MAE_LAMBDA:-1.0}" \
     --spec_mask_loss "${SPEC_MASK_LOSS:-dice}" \
     ${SPEC_MASK_FLAG} \
     ${SPEC_INPUT_COND_FLAG} \
     ${SPEC_INPUT_FEAT_FLAG} \
     --collapse_aware_best \
     $SPECTRO_PATCH_FLAGS \
     --no_amp_val
