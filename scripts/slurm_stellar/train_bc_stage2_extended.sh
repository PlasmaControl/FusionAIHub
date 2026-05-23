#!/bin/bash
#SBATCH --job-name=bc_s2ext
#SBATCH --output=logs/%j_bc_stage2_ext.out
#SBATCH --error=logs/%j_bc_stage2_ext.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=32G

# Combined Phase B + Phase C Stage 2 Extended — full-backprop
# K={10,20,40,80} displacement-loss fine-tuning of TS, tangtv video,
# AND ECE / CO2 / BES spectrograms.
#
# Mirror of train_e2e_stage2_extended.sh with two additions:
#   --use_video tangtv          — adds the 300-token tangtv diagnostic
#                                 in the diagnostic prefix.
#   --use_spectro ece co2 bes   — adds 480 spectrogram tokens (ECE 192,
#                                 CO2 96, BES 192) between fast_ts and
#                                 video. Spectrograms train under
#                                 MAE-only loss (displacement deferred
#                                 per the spectrogram plan's Open
#                                 Decision #3 until reconstruction
#                                 quality is validated). Video also
#                                 trains under MAE-only.
#
# Init checkpoint preference order:
#   1. BC-Stage 2 (delta) best — preferred; both video and spectrogram
#      modules already curriculum-trained through K=10.
#   2. BC-Stage 1 best — modules trained at K=1; Extended will adapt
#      them to longer rollouts.
#   3. Phase A Stage 2 Extended best — TS-only; video and spectrogram
#      keys are missing-by-design and accepted via
#      allowed_missing_prefixes; both modalities start from scratch.
#   4. Phase A Stage 1 best — same as 3 but earlier.
#
# Token budget at full BC config: ~1180 tokens (398 TS + 480 spectro
# + 300 video). Memory at K=80 with grad_checkpoint_every=1 dominates
# FFN per-layer cost; ~2× over Phase A Extended at the same batch.
# Default batch=32 leaves headroom on Stellar A100 40 GB; tune up if
# the first val pass fits comfortably.
#
# Output: runs/bc_stage2_ext/. Does not touch runs/e2e_stage2_ext/, so
# the Phase A Extended pipeline continues unaffected.

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# ── Snapshot init checkpoint with fallback chain ──────────────────────
BC_STAGE2_BEST="runs/bc_stage2_delta/e2e_stage2_delta_best.pt"
BC_STAGE1_BEST="runs/bc_stage1/e2e_stage1_best.pt"
PHASE_A_S2EXT_BEST="runs/e2e_stage2_ext/e2e_stage2_ext_best.pt"
PHASE_A_S1_BEST="runs/e2e_stage1/e2e_stage1_best.pt"
if [ -f "$BC_STAGE2_BEST" ]; then
    INIT_SRC="$BC_STAGE2_BEST"
    INIT_LABEL="bc_stage2_delta_best"
elif [ -f "$BC_STAGE1_BEST" ]; then
    INIT_SRC="$BC_STAGE1_BEST"
    INIT_LABEL="bc_stage1_best"
    echo "WARNING: BC-Stage 2 (delta) best not yet produced; falling"
    echo "         back to BC-Stage 1 best."
elif [ -f "$PHASE_A_S2EXT_BEST" ]; then
    INIT_SRC="$PHASE_A_S2EXT_BEST"
    INIT_LABEL="phase_a_stage2_ext_best"
    echo "WARNING: BC checkpoints not yet produced; falling back to"
    echo "         Phase A Stage 2 Extended best. Video and spectrogram"
    echo "         modules will start from scratch (allowed_missing_prefixes"
    echo "         accepts those keys)."
elif [ -f "$PHASE_A_S1_BEST" ]; then
    INIT_SRC="$PHASE_A_S1_BEST"
    INIT_LABEL="phase_a_stage1_best"
    echo "WARNING: no BC checkpoint and no Phase A Extended best; falling"
    echo "         back to Phase A Stage 1 best. Video and spectrogram"
    echo "         modules will start from scratch."
else
    echo "ERROR: no init checkpoint found. Need at least one of:" >&2
    echo "  $BC_STAGE2_BEST" >&2
    echo "  $BC_STAGE1_BEST" >&2
    echo "  $PHASE_A_S2EXT_BEST" >&2
    echo "  $PHASE_A_S1_BEST" >&2
    exit 1
fi

mkdir -p runs/bc_stage2_ext
SNAPSHOT="runs/bc_stage2_ext/init_${INIT_LABEL}.${SLURM_JOB_ID}.pt"
cp "$INIT_SRC" "$SNAPSHOT"
echo "Init source: $INIT_SRC"
echo "Snapshot:    $SNAPSHOT"

# ── Auto-resume across 24 h walls ──────────────────────────────────────
LATEST="runs/bc_stage2_ext/e2e_stage2_ext_latest.pt"
RESUME_FLAG=""
if [ -f "$LATEST" ]; then
    RESUME_FLAG="--resume_checkpoint $LATEST"
    echo "Auto-resume from $LATEST"
fi

srun pixi run python ../training/train_e2e_stage2_extended.py \
    $RESUME_FLAG \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path /scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt \
    --checkpoint_dir runs/bc_stage2_ext \
    --init_checkpoint "$SNAPSHOT" \
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
    --curriculum_Ks 10,20,40,80 \
    --block_steps 80500 \
    \
    --mae_weight 1.0 \
    --cos_weight 0.3 \
    --mag_weight 0.1 \
    --min_disp_norm 0.01 \
    \
    --grad_checkpoint_every 1 \
    \
    --lr 1e-5 \
    --min_lr 1e-7 \
    --warmup_steps 500 \
    --weight_decay 0.01 \
    --grad_clip 5.0 \
    \
    --batch_size 32 \
    --num_workers 8 \
    --max_steps 322000 \
    --log_every 50 \
    --val_every 5000 \
    --val_max_batches 20 \
    --tf_anneal_steps 40000 \
    \
    --use_video tangtv \
    --use_spectro ece co2 bes
