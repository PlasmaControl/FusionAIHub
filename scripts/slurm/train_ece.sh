#!/bin/bash
#SBATCH --job-name=train_ece
#SBATCH --output=logs/%j_train_ece.out
#SBATCH --error=logs/%j_train_ece.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3G

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun pixi run torchrun \
    --nproc_per_node=1 \
    --rdzv_id=ece \
    --rdzv_backend=c10d \
    scripts/training/train_unimodal_autoencoder.py \
    --signal ece \
    --data_dir /scratch/gpfs/EKOLEMEN/big_d3d_data/dummy_foundation_model_data \
    --d_model 16 \
    --batch_size 16 \
    --num_workers 8 \
    --epochs 300 \
    --lr 0.001 \
    --n_fft 256 \
    --hop_length 256 \
    --chunk_duration_s 0.05 \
    --log_interval 20 \
    --checkpoint_dir runs \
    # --resume