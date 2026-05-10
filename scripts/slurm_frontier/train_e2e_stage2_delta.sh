#!/bin/bash
#SBATCH -A fus187
#SBATCH -J e2e_stage2_delta
#SBATCH -o logs/%j_e2e_stage2_delta.out
#SBATCH -e logs/%j_e2e_stage2_delta.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
set -e

cd /lustre/orion/fus187/scratch/nchen/FusionAIHub
mkdir -p logs runs/e2e_stage2_delta

export MASTER_PORT=29502
source scripts/slurm_frontier/_frontier_common.sh

srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK \
     --gpus-per-task=1 --gpu-bind=closest \
     scripts/slurm_frontier/_srun_rank_wrapper.sh \
     scripts/training/train_e2e_stage2_delta.py \
     --data_dir /lustre/orion/fus187/proj-shared/foundation_model \
     --stats_path data/preprocessing_stats.pt \
     --checkpoint_dir runs/e2e_stage2_delta \
     --val_fraction 0.1 \
     --seed 42 \
     --chunk_duration_s 0.05 \
     --step_size_s 0.01 \
     --warmup_s 1.0 \
     --d_model 256 \
     --n_layers 8 \
     --n_heads 8 \
     --dropout 0.1 \
     --K_max 10 \
     --curriculum_steps 25000 \
     --mae_weight 1.0 \
     --cos_weight 0.3 \
     --mag_weight 0.1 \
     --min_disp_norm 0.01 \
     --lr 5e-4 \
     --min_lr 1e-6 \
     --warmup_steps 500 \
     --weight_decay 0.1 \
     --grad_clip 5.0 \
     --batch_size 8 \
     --num_workers 4 \
     --max_steps 50000 \
     --log_every 50 \
     --val_every 500 \
     --val_max_batches 20
