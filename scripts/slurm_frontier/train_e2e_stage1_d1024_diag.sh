#!/bin/bash
#SBATCH -A fus187
#SBATCH -J e2e_stage1_d1024_diag
#SBATCH -o logs/%j_e2e_stage1_d1024_diag.out
#SBATCH -e logs/%j_e2e_stage1_d1024_diag.err
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e

# Diagnostic d=1024 run to test whether disabling the AWS-OFI-NCCL
# plugin also fixes the 256MB BROADCAST hang that killed 4700730/31.
# max_steps=1 → trainer loads checkpoint, runs DDP wrap (which
# broadcasts the offending 256MB tensor across ranks), then exits at
# the loop guard since current_step >> max_steps. If the broadcast
# completes, the plugin was the cause for d=1024 too.

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"

CHECKPOINT_DIR="/lustre/orion/fus187/proj-shared/models/e2e_stage1_d1024_48L_diag"
PROD_LATEST="/lustre/orion/fus187/proj-shared/models/e2e_stage1_d1024_48L/e2e_stage1_latest.pt"
mkdir -p logs "${CHECKPOINT_DIR}"

# Distinct port to avoid collision with held d=1024 chain (29515).
export MASTER_PORT=29517
source scripts/slurm_frontier/_frontier_common.sh

srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK \
     --gpus-per-task=1 --gpu-bind=closest \
     scripts/slurm_frontier/_srun_rank_wrapper.sh \
     scripts/training/train_e2e_stage1.py \
     --data_dir /lustre/orion/fus187/proj-shared/foundation_model \
     --stats_path /lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt \
     --checkpoint_dir "${CHECKPOINT_DIR}" \
     --val_fraction 0.1 \
     --seed 42 \
     --chunk_duration_s 0.05 \
     --prediction_horizon_s 0.05 \
     --step_size_s 0.01 \
     --warmup_s 1.0 \
     --d_model 1024 \
     --n_layers 48 \
     --n_heads 8 \
     --dropout 0.1 \
     --lr 5e-4 \
     --min_lr 1e-6 \
     --warmup_steps 4000 \
     --weight_decay 0.1 \
     --grad_clip 5.0 \
     --batch_size 32 \
     --num_workers 6 \
     --max_steps 1 \
     --log_every 1 \
     --val_every 590 \
     --val_max_batches 100 \
     --use_video tangtv \
     --use_spectro ece co2 bes \
     --no_amp_val \
     --backbone_grad_checkpoint \
     --resume_checkpoint "${PROD_LATEST}"
