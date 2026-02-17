#!/bin/bash
#SBATCH --job-name=train_co2
#SBATCH --output=logs/%j_train_co2.out
#SBATCH --error=logs/%j_train_co2.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun python scripts/train_unimodal_autoencoder.py \
    --signal "co2" \
    --d_model 16 \
    --batch_size 24 \
    --num_workers 2 \
    --epochs 100 \
    --lr 0.001 \
    --n_fft 256 \
    --hop_length 128 \
    --log_interval 5 \
    --checkpoint_dir runs