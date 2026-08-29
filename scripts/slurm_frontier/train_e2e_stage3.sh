#!/bin/bash
#SBATCH -A fus187
#SBATCH -J e2e_stage3
#SBATCH -o logs/%j_e2e_stage3.out
#SBATCH -e logs/%j_e2e_stage3.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
set -e

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "${PROJECT_DIR}/scripts/slurm_frontier/_frontier_settings.sh" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR (${PROJECT_DIR}) is not the repo root." >&2
    echo "       cd into the FusionAIHub repo before sbatch." >&2
    exit 1
fi
cd "${PROJECT_DIR}"
mkdir -p logs runs/e2e_stage3

export MASTER_PORT=29504
source scripts/slurm_frontier/_frontier_settings.sh

srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK \
     --gpus-per-task=1 --gpu-bind=closest \
     scripts/slurm_frontier/_srun_rank_wrapper.sh \
     scripts/training/train_e2e_stage3.py \
     --data_dir /lustre/orion/fus187/proj-shared/foundation_model \
     --stats_path data/preprocessing_stats.pt \
     --checkpoint_dir runs/e2e_stage3 \
     --val_fraction 0.1 \
     --seed 42 \
     --chunk_duration_s 0.05 \
     --step_size_s 0.01 \
     --warmup_s 1.0 \
     --d_model 256 \
     --n_layers 8 \
     --n_heads 8 \
     --dropout 0.1 \
     --lora_rank 16 \
     --lora_alpha 16.0 \
     --K_min 2 \
     --K_max 4 \
     --n_curriculum_blocks 2 \
     --curriculum_steps 25000 \
     --pool_size 50 \
     --buffer_size 500 \
     --buffer_refresh_period 50 \
     --buffer_refresh_fraction 0.1 \
     --use_displacement_loss \
     --cos_weight 0.3 \
     --mag_weight 0.1 \
     --min_disp_norm 0.01 \
     --lr 3e-5 \
     --min_lr 1e-7 \
     --warmup_steps 200 \
     --weight_decay 0.01 \
     --grad_clip 5.0 \
     --batch_size 16 \
     --val_batch_size 8 \
     --num_workers 4 \
     --max_steps 50000 \
     --log_every 50 \
     --val_every 500
