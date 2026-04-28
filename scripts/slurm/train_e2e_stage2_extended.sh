#!/bin/bash
#SBATCH --job-name=e2e_s2ext
#SBATCH --output=logs/%j_e2e_stage2_ext.out
#SBATCH --error=logs/%j_e2e_stage2_ext.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=32G

# Extended Stage 2 — full-backprop K={10,20,40,80} displacement-loss
# fine-tuning, initialised from Stage 2b best. No LoRA, nothing frozen;
# gradient checkpointing every 10 rollout steps keeps K=80 tractable on
# a 40 GB A100 with bf16 autocast.

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# ── Snapshot Stage 2b best ──────────────────────────────────────────
STAGE2B_BEST="runs/e2e_stage2_delta/e2e_stage2_delta_best.pt"
SNAPSHOT="runs/e2e_stage2_delta/e2e_stage2_delta_best_stage2ext_init.${SLURM_JOB_ID}.pt"

if [ ! -f "$STAGE2B_BEST" ]; then
    echo "ERROR: $STAGE2B_BEST does not exist." >&2
    echo "Stage 2b must produce at least one validation checkpoint first." >&2
    exit 1
fi
cp "$STAGE2B_BEST" "$SNAPSHOT"
echo "Snapshot: $SNAPSHOT"

# Auto-resume: pick up from a previous run's *_latest.pt if present.
LATEST="runs/e2e_stage2_ext/e2e_stage2_ext_latest.pt"
RESUME_FLAG=""
if [ -f "$LATEST" ]; then
    RESUME_FLAG="--resume_checkpoint $LATEST"
    echo "Auto-resume from $LATEST"
fi

srun pixi run python ../training/train_e2e_stage2_extended.py \
    $RESUME_FLAG \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path /scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt \
    --checkpoint_dir runs/e2e_stage2_ext \
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
    --block_steps 48000 \
    \
    --mae_weight 1.0 \
    --cos_weight 0.3 \
    --mag_weight 0.1 \
    --min_disp_norm 0.01 \
    \
    --grad_checkpoint_every 10 \
    \
    --lr 1e-5 \
    --min_lr 1e-7 \
    --warmup_steps 500 \
    --weight_decay 0.01 \
    --grad_clip 5.0 \
    \
    --batch_size 128 \
    --num_workers 8 \
    --max_steps 193000 \
    --log_every 50 \
    --val_every 500 \
    --val_max_batches 20