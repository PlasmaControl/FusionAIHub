#!/bin/bash
#SBATCH --job-name=ts_core_temp_reconstruction
#SBATCH --output=logs/%j_ts_core_temp_reconstruction.out
#SBATCH --error=logs/%j_ts_core_temp_reconstruction.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=10G

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun pixi run python ../training/ts_core_temp_profile_reconstruction.py \
    --signal "ts_core_temp" \
    --d_model 512 \
    --n_tokens 4 \
    --batch_size 512 \
    --num_workers 8 \
    --epochs 200 \
    --lr 1e-4 \
    --weight_decay 0.3 \
    --warmup_epochs 5 \
    --min_lr 0.0 \
    --checkpoint_dir runs \
    --stats_path /projects/EKOLEMEN/foundation_model/preprocessing_stats.pt
