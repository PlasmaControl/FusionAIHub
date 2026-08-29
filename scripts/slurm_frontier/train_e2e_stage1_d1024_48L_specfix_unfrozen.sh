#!/bin/bash
#SBATCH -A fus187
#SBATCH -J e2e_s1_specfix_unfroz
#SBATCH -o logs/%j_e2e_stage1_specfix_unfrozen.out
#SBATCH -e logs/%j_e2e_stage1_specfix_unfrozen.err
#SBATCH -t 12:00:00
#SBATCH -p extended
#SBATCH -N 8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --mail-user=ps9551@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL
set -e

# Spec-fix fine-tune, FULL-MODEL UNFROZEN (2026-06-15). The three
# frozen-backbone runs (inv-stem, then +freq-stem +squared weights)
# all plateaued at the identical blurry ~17% of GT temporal variance
# for spectrograms — strong evidence the frozen backbone is the
# binding constraint (its tokens don't carry fine mode structure).
# This run removes ALL freezing: the whole 1.4B model (backbone +
# tokenizers + heads + the specfix modules) is trainable, cold-started
# from the converged Stage 1 best.pt with a fresh optimizer.
#
# Architecture / loss are the proven-shape specfix stack (freq-stem
# encoder, inv-stem decoder, per-bin SQUARED weights, 16ch/3x3 refine).
#
# Memory: full unfrozen 1.4B + Adam states is what PRODUCTION Stage 1
# trained at batch=32 + --backbone_grad_checkpoint (fit in 64 GB). The
# specfix modules add a little head-side activation; batch stays 32 +
# gc. If the first step OOMs, drop batch_size to 16.
#
# Risk: unfreezing can regress the already-good TS/video modalities.
# lr 5e-5 (10x below production 5e-4) + short warmup keeps updates
# gentle to limit catastrophic forgetting while letting the backbone
# adapt enough to encode modes. Separate checkpoint dir; original
# Stage 1 best.pt untouched.

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "${PROJECT_DIR}/scripts/slurm_frontier/_frontier_common.sh" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR (${PROJECT_DIR}) is not the repo root." >&2
    exit 1
fi
cd "${PROJECT_DIR}"

SOURCE_BEST="/lustre/orion/fus187/proj-shared/models/e2e_stage1_d1024_48L/e2e_stage1_best.pt"
# Fresh dir → resume logic cold-inits from Stage 1 best.pt (no stale
# latest.pt to resume). Chain successors resume from THIS dir's latest.
CHECKPOINT_DIR="/lustre/orion/fus187/proj-shared/models/e2e_stage1_d1024_48L_specfix_unfrozen"
mkdir -p logs "${CHECKPOINT_DIR}"

if [ ! -f "${SOURCE_BEST}" ]; then
    echo "ERROR: source best.pt not found: ${SOURCE_BEST}" >&2
    exit 1
fi

# Distinct port — _specfix used 29517, smoke 29519.
export MASTER_PORT=29520
source scripts/slurm_frontier/_frontier_common.sh
# No MIOPEN_FIND_MODE override: refine uses proven 16ch/3x3 shapes,
# freq-stem is a matmul (no MIOpen), inv-stem tunes in minutes.
# 2026-06-15 memory history (unfrozen 1.4B + Adam states):
#   batch 32        -> clean GPU OOM at step 1 (4809897/98)
#   batch 24 + expandable_segments -> "expandable_segments not
#       supported on this platform" (ROCm no-op!), ran to step ~100
#       then a rank SIGKILLed at ~150 — fragmentation-induced alloc
#       failure at the memory edge (4810152).
# Resolution: drop the unsupported knob, batch 16 for a large margin
# (~41 GB est. of 64) that absorbs fragmentation peaks. Throughput
# cost accepted — a run that finishes beats one that OOM-kills.

RESUME_FLAG=""
INIT_FLAG=""
LATEST_CKPT="${CHECKPOINT_DIR}/e2e_stage1_latest.pt"
if [ -f "${LATEST_CKPT}" ]; then
    echo "[specfix-unfrozen] resuming chain from ${LATEST_CKPT}"
    RESUME_FLAG="--resume_checkpoint ${LATEST_CKPT}"
else
    echo "[specfix-unfrozen] cold-init from ${SOURCE_BEST}"
    INIT_FLAG="--init_checkpoint ${SOURCE_BEST}"
fi

SAMPLER_LOG="logs/${SLURM_JOB_ID}_sampler.log"
srun --overlap -N "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 -c 1 \
     scripts/slurm_frontier/_node_sampler.sh > "$SAMPLER_LOG" 2>&1 &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT

# NO --freeze_whole_run / --freeze_*_steps: the entire model trains.
srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK \
     --gpus-per-task=1 --gpu-bind=closest \
     scripts/slurm_frontier/_srun_rank_wrapper.sh \
     scripts/training/train_e2e_stage1.py \
     --data_dir /lustre/orion/fus187/proj-shared/foundation_model \
     --stats_path /lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt \
     --checkpoint_dir "${CHECKPOINT_DIR}" \
     --val_fraction 0.1 \
     --seed 42 \
     --chunk_duration_s 0.05 \
     --prediction_horizon_s 0.05 \
     --step_size_s 0.01 \
     --warmup_s 1.0 \
     --d_model 1024 \
     --n_layers 48 \
     --n_heads 8 \
     --dropout 0.1 \
     --lr 5e-5 \
     --min_lr 1e-6 \
     --warmup_steps 500 \
     --weight_decay 0.1 \
     --grad_clip 5.0 \
     --batch_size 16 \
     --num_workers 6 \
     --max_steps 10000 \
     --log_every 50 \
     --val_every 590 \
     --val_max_batches 100 \
     --use_video tangtv \
     --use_spectro ece co2 bes \
     --no_amp_val \
     --backbone_grad_checkpoint \
     --spec_per_bin_loss \
     --spec_per_bin_weight_clamp 20.0 \
     --spec_per_bin_weight_power 2.0 \
     --spec_inv_stem \
     --spec_inv_stem_ch 64 \
     --spec_freq_stem \
     --spec_freq_stem_hidden 128 \
     --spectro_seam_refine \
     --video_seam_refine \
     --seam_refine_hidden_ch 16 \
     --spectro_refine_kernel 3 \
     --video_refine_kernel 1 3 3 \
     ${INIT_FLAG} \
     ${RESUME_FLAG}
