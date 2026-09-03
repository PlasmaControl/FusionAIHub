#!/bin/bash
#SBATCH --job-name=bc_stage1
#SBATCH --output=logs/%j_bc_stage1.out
#SBATCH --error=logs/%j_bc_stage1.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=33
#SBATCH --mem-per-cpu=16G

# Combined Phase B + Phase C Stage 1 — single-step pretraining of TS,
# tangtv video, AND ECE / CO2 / BES spectrograms in one run.
#
# Mirror of train_e2e_stage1.sh with three additions:
#   --use_video tangtv                — adds the 300-token tangtv
#                                       diagnostic in the diagnostic prefix.
#   --use_spectro ece co2 bes         — adds 3 × spectrogram diagnostics
#                                       (192 + 96 + 192 = 480 tokens)
#                                       between fast_ts and video.
#   --init_checkpoint <Phase A best>  — warm-starts TS + actuator weights
#                                       from e2e_stage1_best.pt. Video and
#                                       spectrogram tokenizers + heads init
#                                       from scratch (their keys are
#                                       declared in allowed_missing_prefixes).
#   --freeze_ts_steps 5000
#   --freeze_backbone_steps 5000      — backbone + TS modules held fixed
#                                       for 5 k steps so the freshly-
#                                       initialised video and spectrogram
#                                       modules can settle without
#                                       perturbing the Phase A-trained
#                                       backbone. Video and spectro
#                                       modules train throughout.
#
# Token budget:
#   slow_ts (273) + fast_ts (80) + spectro (480) + video (300) + actuators (45)
#   = 1178 tokens (8.8x attention cost vs Phase A TS-only).
# Memory at batch 256 estimated > 40 GB → expect to need batch_size = 64
# on Stellar A100 40 GB. See docs/spectrogram_tokenizer_plan.md §"Memory".
#
# Output: runs/bc_stage1/. Does not touch runs/e2e_stage1/, so the
# Phase A pipeline (Stage 2b chain + Stage 2 Extended) is unaffected.

export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1

# ── Snapshot Phase A Stage 1 best ──────────────────────────────────
# Snapshotted at job start so a future Phase A retraining cannot
# silently change what this combined run warm-started from.
PHASE_A_BEST="runs/e2e_stage1/e2e_stage1_best.pt"
SNAPSHOT="runs/e2e_stage1/e2e_stage1_best_bc_stage1_init.${SLURM_JOB_ID}.pt"

if [ ! -f "$PHASE_A_BEST" ]; then
    echo "ERROR: $PHASE_A_BEST does not exist." >&2
    echo "Phase A Stage 1 must produce a best checkpoint first." >&2
    exit 1
fi
cp "$PHASE_A_BEST" "$SNAPSHOT"
echo "Snapshot: $SNAPSHOT"

# ── Auto-resume across 24 h walls ─────────────────────────────────
# If a *_latest.pt exists in the BC-Stage 1 checkpoint dir from a
# previous submission, resume from it; the trainer's resume path
# overrides --init_checkpoint, so passing both unconditionally is safe.
# train_e2e_stage1.py hardcodes the basename "e2e_stage1_latest.pt" —
# under --checkpoint_dir runs/bc_stage1 that lands at the path below.
LATEST="runs/bc_stage1/e2e_stage1_latest.pt"
RESUME_FLAG=""
if [ -f "$LATEST" ]; then
    RESUME_FLAG="--resume_checkpoint $LATEST"
    echo "Auto-resume from $LATEST"
fi

srun pixi run python ../training/train_e2e_stage1.py \
    $RESUME_FLAG \
    --init_checkpoint "$SNAPSHOT" \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path /projects/EKOLEMEN/foundation_model/preprocessing_stats.pt \
    --checkpoint_dir runs/bc_stage1 \
    --val_fraction 0.1 \
    --seed 42 \
    \
    --chunk_duration_s 0.05 \
    --prediction_horizon_s 0.05 \
    --step_size_s 0.01 \
    --warmup_s 1.0 \
    \
    --d_model 256 \
    --n_layers 8 \
    --n_heads 8 \
    --dropout 0.1 \
    \
    --lr 1e-4 \
    --min_lr 1e-6 \
    --warmup_steps 4000 \
    --weight_decay 0.1 \
    --grad_clip 5.0 \
    \
    --batch_size 128 \
    --num_workers 16 \
    --max_steps 672000 \
    --log_every 50 \
    --val_every 4000 \
    --val_max_batches 100 \
    \
    --use_video tangtv \
    --use_spectro ece co2 bes \
    --freeze_ts_steps 5000 \
    --freeze_backbone_steps 5000