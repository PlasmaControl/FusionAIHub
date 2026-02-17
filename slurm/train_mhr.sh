#!/bin/bash
#SBATCH --job-name=train_mhr
#SBATCH --output=logs/%j_train_mhr.out
#SBATCH --error=logs/%j_train_mhr.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun pixi run python scripts/train_unimodal_autoencoder.py \
    --signal "mhr" \
    --d_model 16 \
    --batch_size 128 \
    --num_workers 4 \
    --epochs 300 \
    --lr 0.001 \
    --n_fft 256 \
    --hop_length 256 \
    --chunk_duration_s 0.05 \
    --log_interval 20 \
    --checkpoint_dir runs \