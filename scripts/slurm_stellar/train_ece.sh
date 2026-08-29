#!/bin/bash
#SBATCH --job-name=train_ece
#SBATCH --output=logs/%j_train_ece.out
#SBATCH --error=logs/%j_train_ece.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8G

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd /scratch/gpfs/nc1514/FusionAIHub

srun pixi run python scripts/training/train_unimodal_autoencoder.py \
    --signal "ece" \
    --d_model 64 \
    --batch_size 4 \
    --num_workers 2 \
    --epochs 50 \
    --lr 1e-3 \
    --weight_decay 0.05 \
    --warmup_epochs 5 \
    --val_split 0.2 \
    --chunk_duration_s 0.2 \
    --n_fft 256 \
    --hop_length 256 \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path data/preprocessing_stats.pt \
    --checkpoint_dir runs/ece
