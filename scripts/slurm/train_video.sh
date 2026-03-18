#!/bin/bash
#SBATCH --job-name=train_video
#SBATCH --output=logs/%j_train_video.out
#SBATCH --error=logs/%j_train_video.err
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=aj17@princeton.edu

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun pixi run python ../training/train_video_reconstruction.py \
    --signal tangtv \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model/ \
    --file_glob "*_processed.h5" \
    --shuffle \
    --model video \
    --stats_path /scratch/gpfs/aj17/runs/preprocessing_stats.pt \
    --n_fft 1024 \
    --hop_length 256 \
    --clip_seconds 0.5 \
    --target_fps 50 \
    --image_size 128 \
    --n_channels 2 \
    --n_tokens 64 \
    --d_model 512 \
    --batch_size 16 \
    --num_workers 4 \
    --epochs 500 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --min_lr 0.0 \
    --checkpoint_dir /scratch/gpfs/aj17/runs/ \
    --num_plots 0 \
    --log_interval 1
