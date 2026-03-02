#!/bin/bash
#SBATCH --job-name=train_co2
#SBATCH --output=logs/%j_train_co2.out
#SBATCH --error=logs/%j_train_co2.err
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
    -- \
    --signal co2 \
    --data_dir /scratch/gpfs/EKOLEMEN/big_d3d_data/dummy_foundation_model_data \
    --d_model 256 \
    --model_kwargs '{"n_enc_layers": 4, "n_dec_layers": 2, "n_heads": 4, "patch_h": 8, "patch_w": 8}' \
    --batch_size 24 \
    --num_workers 4 \
    --epochs 3000 \
    --lr 0.001 \
    --n_fft 256 \
    --hop_length 128 \
    --chunk_duration_s 0.1 \
    --log_interval 5 \
    --checkpoint_dir runs/co2_spectrogram
