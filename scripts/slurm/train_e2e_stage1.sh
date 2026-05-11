#!/bin/bash
#SBATCH --job-name=e2e_stage1
#SBATCH --output=logs/%j_e2e_stage1.out
#SBATCH --error=logs/%j_e2e_stage1.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=18
#SBATCH --mem-per-cpu=32G

# Stage 1 single-step pretraining of the end-to-end foundation model.
# ResearchPlan.MD §4.1 + user directives: warmup_s=1.0, step_size_s=0.01.
# Full shot list (glob + 10% val split), d_model=256, n_layers=8,
# cosine LR schedule with linear warmup, best-model checkpointing,
# pred_delta/tgt_delta logged at each validation.

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Auto-resume: if a *_latest.pt exists in the checkpoint dir, pass it as
# --resume_checkpoint. Stage 1 has no --init_checkpoint path; on first
# submission there's nothing to resume, so the flag is simply omitted.
LATEST="runs/e2e_stage1_ddp/e2e_stage1_latest.pt"
RESUME_FLAG=""
if [ -f "$LATEST" ]; then
    RESUME_FLAG="--resume_checkpoint $LATEST"
    echo "Auto-resume from $LATEST"
else
    echo "Fresh start (no previous $LATEST)."
fi

srun pixi run torchrun --standalone --nproc_per_node=2 \
    ../training/train_e2e_stage1.py \
    $RESUME_FLAG \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path /scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt \
    --checkpoint_dir runs/e2e_stage1_ddp \
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
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --grad_clip 5.0 \
    \
    --batch_size 256 \
    --num_workers 8 \
    --max_steps 336000 \
    --log_every 50 \
    --val_every 2000 \
    --val_max_batches 50