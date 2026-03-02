#!/bin/bash
#SBATCH --job-name=train_mhr
#SBATCH --output=logs/%j_train_mhr.out
#SBATCH --error=logs/%j_train_mhr.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun pixi run torchrun \
    --standalone \
    --nproc_per_node=2 \
    scripts/training/train_unimodal_autoencoder.py \
    --signal mhr \
    --data_dir /scratch/gpfs/EKOLEMEN/big_d3d_data/dummy_foundation_model_data \
    --d_model 64 \
    --model_kwargs '{"n_layers": 6, "kernel_size": [2, 3, 3], "stride": [1, 2, 2], "base_channels": 4}' \
    --batch_size 128 \
    --num_workers 4 \
    --epochs 300 \
    --lr 0.001 \
    --n_fft 256 \
    --hop_length 256 \
    --chunk_duration_s 0.05 \
    --log_interval 20 \
    --checkpoint_dir runs/mhr_spectrogram
