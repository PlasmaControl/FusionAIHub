#!/bin/bash
#SBATCH --job-name=e2e_stage2
#SBATCH --output=logs/%j_e2e_stage2.out
#SBATCH --error=logs/%j_e2e_stage2.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=32G

# Stage 2 short-rollout fine-tuning of the end-to-end foundation model.
# ResearchPlan.MD §4.2: stepwise curriculum K = 1..K_max, full backprop
# through all K steps, bf16 autocast on CUDA, best-checkpoint gating on
# sum-of-per-step model MAE, per-step MAE called out at steps 1 / K_max/2 /
# K_max in each validation log.

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# ── Init checkpoint snapshot ─────────────────────────────────────────────
# Stage 1 job(s) keep overwriting ``e2e_stage1_best.pt`` on each val
# improvement. Snapshot the current best under a Stage-2-job-specific
# filename so our init does not drift mid-run. If no Stage 1 best exists
# yet, abort before burning the GPU.

STAGE1_BEST="runs/e2e_stage1/e2e_stage1_best.pt"
SNAPSHOT="runs/e2e_stage1/e2e_stage1_best_stage2init.${SLURM_JOB_ID}.pt"

if [ ! -f "$STAGE1_BEST" ]; then
    echo "ERROR: $STAGE1_BEST does not exist." >&2
    echo "Wait for a Stage 1 validation to land before submitting Stage 2." >&2
    exit 1
fi

cp "$STAGE1_BEST" "$SNAPSHOT"
echo "Snapshot: $SNAPSHOT"

srun pixi run python ../training/train_e2e_stage2.py \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path /scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt \
    --checkpoint_dir runs/e2e_stage2 \
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
    --lr 3e-5 \
    --min_lr 1e-6 \
    --warmup_steps 200 \
    --weight_decay 0.1 \
    --grad_clip 5.0 \
    \
    --batch_size 16 \
    --num_workers 8 \
    --max_steps 40000 \
    --log_every 50 \
    --val_every 500 \
    --val_max_batches 20