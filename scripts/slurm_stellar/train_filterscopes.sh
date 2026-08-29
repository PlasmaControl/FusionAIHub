#!/bin/bash
#SBATCH --job-name=filterscopes_reconstruction
#SBATCH --output=logs/%j_filterscopes_reconstruction.out
#SBATCH --error=logs/%j_filterscopes_reconstruction.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=17
#SBATCH --mem-per-cpu=8G

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun pixi run python ../training/filterscopes_reconstruction.py \
    --signal "filterscopes" \
    --d_model 16 \
    --n_tokens 32 \
    --batch_size 2048 \
    --num_workers 16 \
    --epochs 200 \
    --lr 1e-4 \
    --weight_decay 0.3 \
    --warmup_epochs 5 \
    --min_lr 0.0 \
    --checkpoint_dir runs \
    --stats_path /projects/EKOLEMEN/foundation_model/preprocessing_stats.pt
