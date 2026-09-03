#!/bin/bash
#SBATCH --job-name=train_unimodal
#SBATCH --output=logs/%j_train_unimodal.out
#SBATCH --error=logs/%j_train_unimodal.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun pixi run python scripts/train_unimodal_autoencoder.py \
    --signal "ece" \
    --d_model 16 \
    --batch_size 3 \
    --num_workers 4 \
    --epochs 200 \
    --lr 0.001 \
    --weight_decay 0.05 \
    --warmup_epochs 5 \
    --min_lr 0.0 \
    --checkpoint_dir runs
