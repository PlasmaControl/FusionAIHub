#!/bin/bash
#SBATCH --job-name=train_co2_mae
#SBATCH --output=logs/%j_train_co2_mae.out
#SBATCH --error=logs/%j_train_co2_mae.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

module load pixi

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun pixi run python scripts/training/spectrogram_reconstruction.py \
    --signal co2 \
    --model spectrogram_mae \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path data/preprocessing_stats.pt \
    --shot_min 200000 \
    --shot_max 200500 \
    --mask_ratio 0.5 \
    --d_model 256 \
    --n_tokens 0 \
    --batch_size 16 \
    --num_workers 4 \
    --epochs 500 \
    --lr 3e-4 \
    --weight_decay 1e-4 \
    --warmup_epochs 10 \
    --min_lr 1e-6 \
    --n_fft 256 \
    --hop_length 128 \
    --log_interval 5 \
    --num_plots 4 \
    --checkpoint_dir runs/co2_mae
