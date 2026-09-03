#!/bin/bash
#SBATCH --job-name=train_multimodal
#SBATCH --output=logs/%j_train_multimodal.out
#SBATCH --error=logs/%j_train_multimodal.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=8G

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd /scratch/gpfs/nc1514/FusionAIHub

srun pixi run python scripts/training/train_multimodal_predictor.py \
    --token_dir data/tokens \
    --checkpoint_dir runs \
    --output_dir runs/multimodal \
    --d_model 64 \
    --n_heads 8 \
    --n_layers 6 \
    --batch_size 32 \
    --num_workers 4 \
    --epochs 50 \
    --lr 1e-3 \
    --weight_decay 0.05 \
    --val_split 0.2
