#!/bin/bash
#SBATCH --job-name=e2e_s2b
#SBATCH --output=logs/%j_e2e_stage2_delta.out
#SBATCH --error=logs/%j_e2e_stage2_delta.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=32G

# Stage 2b: displacement-loss fine-tuning, initialised from Stage 1 best
# (not Stage 2 best — the plain-MAE Stage 2 sat in a copy-like local
# minimum; Stage 2b tries to escape it with a loss that directly rewards
# predicting the displacement direction and magnitude).

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# ── Snapshot Stage 1 best ────────────────────────────────────────────
STAGE1_BEST="runs/e2e_stage1/e2e_stage1_best.pt"
SNAPSHOT="runs/e2e_stage1/e2e_stage1_best_stage2delta_init.${SLURM_JOB_ID}.pt"

if [ ! -f "$STAGE1_BEST" ]; then
    echo "ERROR: $STAGE1_BEST does not exist." >&2
    exit 1
fi
cp "$STAGE1_BEST" "$SNAPSHOT"
echo "Snapshot: $SNAPSHOT"

srun pixi run python ../training/train_e2e_stage2_delta.py \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path /scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt \
    --checkpoint_dir runs/e2e_stage2_delta \
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
    --K_max 10 \
    --curriculum_steps 20000 \
    \
    --mae_weight 1.0 \
    --cos_weight 0.3 \
    --mag_weight 0.1 \
    --min_disp_norm 0.01 \
    \
    --lr 5e-4 \
    --min_lr 1e-6 \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --grad_clip 5.0 \
    \
    --batch_size 512 \
    --num_workers 8 \
    --max_steps 40000 \
    --log_every 50 \
    --val_every 500 \
    --val_max_batches 20