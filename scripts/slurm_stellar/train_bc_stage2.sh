#!/bin/bash
#SBATCH --job-name=bc_stage2
#SBATCH --output=logs/%j_bc_stage2.out
#SBATCH --error=logs/%j_bc_stage2.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=32G

# Combined Phase B + Phase C Stage 2b — displacement-loss K=1→10
# fine-tuning of TS, tangtv video, AND ECE / CO2 / BES spectrograms.
#
# Mirror of train_e2e_stage2_delta.sh with two additions:
#   --use_video tangtv          — adds the 300-token tangtv diagnostic
#                                 in the diagnostic prefix.
#   --use_spectro ece co2 bes   — adds 480 spectrogram tokens (ECE 192,
#                                 CO2 96, BES 192) between fast_ts and
#                                 video. Spectrograms train under
#                                 MAE-only loss (displacement deferred
#                                 per the spectrogram plan's Open
#                                 Decision #3 until reconstruction
#                                 quality is validated).
#
# Init checkpoint prefers BC-Stage 1 best (with both video and
# spectrogram modules trained); falls back to BC-Stage 1 latest, then
# Phase A Stage 1 best (TS-only — video and spectrogram keys missing
# but accepted via allowed_missing_prefixes; tokenizer + head start
# from scratch). Output: runs/bc_stage2_delta/.
#
# Loss recipe: TS keeps the standard alpha*MAE + beta*(1-cos) + gamma*|log mag|
# Stage 2b loss with weights 1.0 / 0.3 / 0.1; video and spectrograms
# get MAE only.

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# ── Snapshot init checkpoint ───────────────────────────────────────
BC_STAGE1_BEST="runs/bc_stage1/e2e_stage1_best.pt"
PHASE_A_BEST="runs/e2e_stage1/e2e_stage1_best.pt"
if [ -f "$BC_STAGE1_BEST" ]; then
    INIT_SRC="$BC_STAGE1_BEST"
    INIT_LABEL="bc_stage1_best"
elif [ -f "$PHASE_A_BEST" ]; then
    INIT_SRC="$PHASE_A_BEST"
    INIT_LABEL="phase_a_stage1_best"
    echo "WARNING: BC-Stage 1 best not yet produced; falling back to"
    echo "         Phase A Stage 1 best. Video and spectrogram modules"
    echo "         will start from scratch (allowed_missing_prefixes"
    echo "         accepts those keys)."
else
    echo "ERROR: neither $BC_STAGE1_BEST nor $PHASE_A_BEST exists." >&2
    exit 1
fi
SNAPSHOT="runs/bc_stage2_delta/init_${INIT_LABEL}.${SLURM_JOB_ID}.pt"
mkdir -p runs/bc_stage2_delta
cp "$INIT_SRC" "$SNAPSHOT"
echo "Init source: $INIT_SRC"
echo "Snapshot:    $SNAPSHOT"

# ── Auto-resume across 24 h walls ─────────────────────────────────
LATEST="runs/bc_stage2_delta/e2e_stage2_delta_latest.pt"
RESUME_FLAG=""
if [ -f "$LATEST" ]; then
    RESUME_FLAG="--resume_checkpoint $LATEST"
    echo "Auto-resume from $LATEST"
fi

srun pixi run python ../training/train_e2e_stage2_delta.py \
    $RESUME_FLAG \
    --init_checkpoint "$SNAPSHOT" \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path /projects/EKOLEMEN/foundation_model/preprocessing_stats.pt \
    --checkpoint_dir runs/bc_stage2_delta \
    --val_fraction 0.1 \
    --seed 42 \
    \
    --chunk_duration_s 0.05 \
    --step_size_s 0.01 \
    --warmup_s 1.0 \
    \
    --d_model 256 \
    --n_layers 8 \
    --n_heads 8 \
    --dropout 0.1 \
    \
    --K_max 10 \
    --curriculum_steps 322000 \
    \
    --mae_weight 1.0 \
    --cos_weight 0.3 \
    --mag_weight 0.1 \
    --min_disp_norm 0.01 \
    \
    --lr 5e-4 \
    --min_lr 1e-6 \
    --warmup_steps 500 \
    --weight_decay 0.1 \
    --grad_clip 5.0 \
    \
    --batch_size 64 \
    --num_workers 8 \
    --max_steps 322000 \
    --log_every 50 \
    --val_every 500 \
    --val_max_batches 20 \
    \
    --use_video tangtv \
    --use_spectro ece co2 bes