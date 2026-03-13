#!/bin/bash
#SBATCH --job-name=fast_time_series_reconstruction
#SBATCH --output=logs/%j_fast_time_series_reconstruction.out
#SBATCH --error=logs/%j_fast_time_series_reconstruction.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=17
#SBATCH --mem-per-cpu=8G

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun pixi run python ../training/fast_time_series_reconstruction.py \
    --signal "filterscopes" \
    --d_model 512 \
    --batch_size 2048 \
    --num_workers 16 \
    --epochs 200 \
    --lr 1e-2 \
    --weight_decay 0.05 \
    --warmup_epochs 5 \
    --min_lr 0.0 \
    --checkpoint_dir runs \
    --stats_path /scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt
