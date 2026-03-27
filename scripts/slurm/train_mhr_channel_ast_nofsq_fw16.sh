#!/bin/bash
#SBATCH --job-name=mhr_chast_nofsq_fw16
#SBATCH --output=logs/%j_mhr_chast_nofsq_fw16.out
#SBATCH --error=logs/%j_mhr_chast_nofsq_fw16.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64G

module load pixi

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Channel-AST, no FSQ, no merge, frame_width=16 — MHR (C=6, fixed dead channels)
# Token count: 6 × ceil(1954/16) = 6 × 123 = 738 tokens × 256 d_model
# Per-frame embed: Linear(128*16=2048, 256) = 8:1
srun pixi run python scripts/training/spectrogram_reconstruction.py \
    --signal mhr \
    --model spectrogram_channel_ast_fsq \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path data/preprocessing_stats.pt \
    --fsq_levels \
    --frame_width 16 \
    --time_conv_kernel 7 \
    --d_model 256 \
    --n_tokens 0 \
    --batch_size 8 \
    --num_workers 2 \
    --epochs 500 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --scheduler none \
    --n_fft 256 \
    --hop_length 128 \
    --log_interval 5 \
    --num_plots 4 \
    --checkpoint_dir runs/mhr_channel_ast_nofsq_fw16_v2
