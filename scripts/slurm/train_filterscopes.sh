#!/bin/bash
#SBATCH --job-name=filterscopes_reconstruction
#SBATCH --output=logs/%j_filterscopes_reconstruction.out
#SBATCH --error=logs/%j_filterscopes_reconstruction.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=16G

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun pixi run python ../training/filterscopes_reconstruction.py \
    --signal "filterscopes" \
    --d_model 512 \
    --batch_size 2048 \
    --num_workers 8 \
    --epochs 200 \
    --lr 1e-3 \
    --weight_decay 0.05 \
    --warmup_epochs 5 \
    --min_lr 0.0 \
    --checkpoint_dir runs \
    --stats_path /scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt
