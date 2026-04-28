#!/bin/bash
#SBATCH --job-name=ece_dw_ft
#SBATCH --output=logs/%j_train_ece_conv_dw_ft.out
#SBATCH --error=logs/%j_train_ece_conv_dw_ft.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun pixi run torchrun \
    --standalone \
    --nproc_per_node=2 \
    scripts/training/train_unimodal_autoencoder.py \
    -- \
    --signal mhr \
    --model spectrogram_conv \
    --data_dir /scratch/gpfs/EKOLEMEN/big_d3d_data/dummy_foundation_model_data \
    --d_model 128 \
    --model_kwargs '{"variant":"depthwise_freq_time"}' \
    --batch_size 32 \
    --num_workers 8 \
    --epochs 300 \
    --lr 0.001 \
    --n_fft 256 \
    --hop_length 256 \
    --chunk_duration_s 0.1 \
    --plot_channel 20 \
    --plot_indices 50 \
    --log_interval 1 \
    --checkpoint_dir runs/ece_spectrogram_conv_dw_ft
