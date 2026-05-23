#!/bin/bash
#SBATCH --job-name=video_ae
#SBATCH --output=logs/%j_video_ae.out
#SBATCH --error=logs/%j_video_ae.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=32G

# Standalone tangtv autoencoder validation. Trains the tube-patch
# VideoTokenizer + VideoOutputHead end-to-end on masked MAE for ~5k
# steps to validate the per-patch token capacity before Step 5
# integration into the full E2E foundation model.
#
# Default patch (3, 12, 12) over input (3, 120, 360) -> 300 tokens
# per camera per 50 ms window. Each token reconstructs one disjoint
# 2 x 3 x 12 x 12 region.
#
# This job is intentionally short (4 h wall) and disjoint from the
# Phase A pipeline — it does not touch e2e_stage{1,2_delta,2_ext,3}
# checkpoints or runs/. Output goes to runs/video_ae/.

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun pixi run python ../training/train_video_ae.py \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --checkpoint_dir runs/video_ae \
    --max_steps 5000 \
    --batch_size 256 \
    --num_workers 8 \
    --lr 1e-3 \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --log_every 50 \
    --val_every 500 \
    --patch_size 3 12 12 \
    --val_fraction 0.05 \
    --seed 42