#!/bin/bash
#SBATCH --job-name=e2e_stage1
#SBATCH --output=logs/%j_e2e_stage1.out
#SBATCH --error=logs/%j_e2e_stage1.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=17
#SBATCH --mem-per-cpu=32G

# Stage 1 single-step pretraining of the end-to-end foundation model.
# ResearchPlan.MD §4.1 + user directives: warmup_s=1.0, step_size_s=0.01.
# Full shot list (glob + 10% val split), d_model=256, n_layers=8,
# cosine LR schedule with linear warmup, best-model checkpointing,
# pred_delta/tgt_delta logged at each validation.

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun pixi run python ../training/train_e2e_stage1.py \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path /scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt \
    --checkpoint_dir runs/e2e_stage1 \
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
    --lr 5e-4 \
    --min_lr 1e-6 \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --grad_clip 5.0 \
    \
    --batch_size 512 \
    --num_workers 16 \
    --max_steps 200000 \
    --log_every 50 \
    --val_every 2000 \
    --val_max_batches 50