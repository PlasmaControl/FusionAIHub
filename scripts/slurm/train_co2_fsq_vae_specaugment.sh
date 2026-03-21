#!/bin/bash
#SBATCH --job-name=train_co2_fsq_vae_specaugment
#SBATCH --output=logs/%j_train_co2_fsq_vae_specaugment.out
#SBATCH --error=logs/%j_train_co2_fsq_vae_specaugment.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64G

module load pixi

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun pixi run python scripts/training/spectrogram_reconstruction.py \
    --signal co2 \
    --model spectrogram_fsq_vae \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path data/preprocessing_stats.pt \
    --fsq_levels 8 5 5 5 5 \
    --patch_h 4 --patch_w 16 \
    --per_channel_patch \
    --d_model 256 \
    --n_tokens 0 \
    --batch_size 16 \
    --num_workers 2 \
    --epochs 500 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --scheduler cosine \
    --warmup_epochs 20 \
    --min_lr 1e-6 \
    --n_fft 256 \
    --hop_length 128 \
    --freq_mask_param 8 \
    --time_mask_param 100 \
    --n_freq_masks 2 \
    --n_time_masks 2 \
    --loss_weighting variance \
    --grad_clip 1.0 \
    --log_interval 5 \
    --num_plots 4 \
    --checkpoint_dir runs/co2_fsq_vae_specaugment \
    --resume
