#!/bin/bash
#SBATCH --job-name=e2e_stage3
#SBATCH --output=logs/%j_e2e_stage3.out
#SBATCH --error=logs/%j_e2e_stage3.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=32G

# Stage 3b long-rollout LoRA fine-tuning with displacement loss.
# ResearchPlan.MD §4.3: 8-block stepwise curriculum K ∈ {10,20,...,80}
# (5k steps each), pushforward, lightweight replay buffer, LoRA on
# backbone attention layers. Base Stage 2b weights frozen.
#
# Differences from the initial Stage 3 run:
#   - Inits from Stage 2b best (escaped the copy minimum) rather than
#     the plain-MAE Stage 2 best.
#   - --use_displacement_loss adds cos+log-mag terms to the final-step
#     training loss. With heads frozen, these gradients flow *only*
#     through the LoRA attention adapters — pushing attention routing
#     to produce tokens whose decoded signal has the correct
#     displacement direction and magnitude.

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# ── Snapshot Stage 2 best ─────────────────────────────────────────────
STAGE2B_BEST="runs/e2e_stage2_delta/e2e_stage2_delta_best.pt"
SNAPSHOT="runs/e2e_stage2_delta/e2e_stage2_delta_best_stage3init.${SLURM_JOB_ID}.pt"

if [ ! -f "$STAGE2B_BEST" ]; then
    echo "ERROR: $STAGE2B_BEST does not exist." >&2
    echo "Stage 2b must produce at least one validation checkpoint before Stage 3b." >&2
    exit 1
fi
STAGE2_BEST="$STAGE2B_BEST"

cp "$STAGE2_BEST" "$SNAPSHOT"
echo "Snapshot: $SNAPSHOT"

srun pixi run python ../training/train_e2e_stage3.py \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path /scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt \
    --checkpoint_dir runs/e2e_stage3 \
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
    --lora_rank 16 \
    --lora_alpha 16.0 \
    \
    --K_min 10 \
    --K_max 80 \
    --n_curriculum_blocks 8 \
    --curriculum_steps 40000 \
    \
    --pool_size 200 \
    --buffer_size 10000 \
    --buffer_refresh_period 50 \
    --buffer_refresh_fraction 0.1 \
    \
    --lr 3e-5 \
    --min_lr 1e-7 \
    --warmup_steps 200 \
    --weight_decay 0.01 \
    --grad_clip 5.0 \
    \
    --use_displacement_loss \
    --cos_weight 0.3 \
    --mag_weight 0.1 \
    --min_disp_norm 0.01 \
    \
    --batch_size 32 \
    --num_workers 8 \
    --max_steps 40000 \
    --log_every 50 \
    --val_every 500 \
    --val_batch_size 8
