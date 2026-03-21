#!/bin/bash
#SBATCH --job-name=co2_fsq_vae_p8_matched
#SBATCH --output=logs/%j_co2_fsq_vae_p8_matched.out
#SBATCH --error=logs/%j_co2_fsq_vae_p8_matched.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64G

module load pixi

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Matched to CNN baseline: same 501 shots, same preprocessing (default log_standardize)
srun pixi run python scripts/training/spectrogram_reconstruction.py \
    --signal co2 \
    --model spectrogram_fsq_vae \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path data/preprocessing_stats.pt \
    --shot_min 200000 \
    --shot_max 200500 \
    --fsq_levels 8 5 5 5 5 \
    --patch_h 8 --patch_w 8 \
    --d_model 256 \
    --n_tokens 0 \
    --batch_size 16 \
    --num_workers 2 \
    --epochs 500 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --scheduler none \
    --n_fft 256 \
    --hop_length 128 \
    --log_interval 5 \
    --num_plots 4 \
    --checkpoint_dir runs/co2_fsq_vae_p8_matched
