#!/bin/bash
#SBATCH -A fus187
#SBATCH -J e2e_stage1
#SBATCH -o logs/%j_e2e_stage1.out
#SBATCH -e logs/%j_e2e_stage1.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
set -e

cd /lustre/orion/fus187/scratch/nchen/FusionAIHub
mkdir -p logs runs/e2e_stage1

export MASTER_PORT=29500
source scripts/slurm_frontier/_frontier_common.sh
conda activate "$CONDA_ENV_PATH"

srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK \
     --gpus-per-task=1 --gpu-bind=closest \
     scripts/slurm_frontier/_srun_rank_wrapper.sh \
     scripts/training/train_e2e_stage1.py \
     --data_dir /lustre/orion/fus187/proj-shared/foundation_model \
     --stats_path data/preprocessing_stats.pt \
     --checkpoint_dir runs/e2e_stage1 \
     --val_fraction 0.1 \
     --seed 42 \
     --chunk_duration_s 0.05 \
     --prediction_horizon_s 0.05 \
     --step_size_s 0.01 \
     --warmup_s 1.0 \
     --d_model 256 \
     --n_layers 8 \
     --n_heads 8 \
     --dropout 0.1 \
     --lr 1e-4 \
     --min_lr 1e-6 \
     --warmup_steps 2000 \
     --weight_decay 0.1 \
     --grad_clip 5.0 \
     --batch_size 16 \
     --num_workers 4 \
     --max_steps 50000 \
     --log_every 50 \
     --val_every 500 \
     --val_max_batches 20
