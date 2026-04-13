#!/bin/bash
#SBATCH --job-name=cer_rot_reconstruction
#SBATCH --output=logs/%j_cer_rot_reconstruction.out
#SBATCH --error=logs/%j_cer_rot_reconstruction.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=10G

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun pixi run python ../training/cer_rot_profile_reconstruction.py \
    --signal "cer_rot" \
    --d_model 32 \
    --n_tokens 16 \
    --batch_size 512 \
    --num_workers 8 \
    --epochs 200 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --warmup_epochs 5 \
    --min_lr 0.0 \
    --checkpoint_dir runs \
    --stats_path /projects/EKOLEMEN/foundation_model/preprocessing_stats.pt