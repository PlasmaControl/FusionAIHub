#!/bin/bash
#SBATCH -A fus187
#SBATCH -J e2e_stage2_ext
#SBATCH -o logs/%j_e2e_stage2_ext.out
#SBATCH -e logs/%j_e2e_stage2_ext.err
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
mkdir -p logs runs/e2e_stage2_extended

export MASTER_PORT=29503
source scripts/slurm_frontier/_frontier_settings.sh

srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK \
     --gpus-per-task=1 --gpu-bind=closest \
     scripts/slurm_frontier/_srun_rank_wrapper.sh \
     scripts/training/train_e2e_stage2_extended.py \
     --data_dir /lustre/orion/fus187/proj-shared/foundation_model \
     --stats_path data/preprocessing_stats.pt \
     --checkpoint_dir runs/e2e_stage2_extended \
     --val_fraction 0.1 \
     --seed 42 \
     --chunk_duration_s 0.05 \
     --step_size_s 0.01 \
     --warmup_s 1.0 \
     --d_model 256 \
     --n_layers 8 \
     --n_heads 8 \
     --dropout 0.1 \
     --curriculum_Ks 2,3,4 \
     --block_steps 16667 \
     --mae_weight 1.0 \
     --cos_weight 0.3 \
     --mag_weight 0.1 \
     --min_disp_norm 0.01 \
     --grad_checkpoint_every 2 \
     --lr 1e-5 \
     --min_lr 1e-7 \
     --warmup_steps 500 \
     --weight_decay 0.01 \
     --grad_clip 5.0 \
     --batch_size 4 \
     --num_workers 4 \
     --max_steps 50000 \
     --log_every 50 \
     --val_every 500 \
     --val_max_batches 20
