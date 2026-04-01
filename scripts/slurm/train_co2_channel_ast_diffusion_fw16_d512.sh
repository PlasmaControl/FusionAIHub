#!/bin/bash
#SBATCH --job-name=co2_diff_fw16_d512
#SBATCH --output=logs/%j_co2_diff_fw16_d512.out
#SBATCH --error=logs/%j_co2_diff_fw16_d512.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64G

module load pixi

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Channel-AST-Diffusion, fw=16, d_model=512 — CO2 (C=4)
# Per-frame embed: Linear(128*16=2048, 512) = 4:1 (was 8:1 at d_model=256)
# Token count: 4 x 123 = 492 tokens x 512 d_model
srun pixi run python scripts/training/spectrogram_reconstruction.py \
    --signal co2 \
    --model spectrogram_channel_ast_diffusion \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path data/preprocessing_stats.pt \
    --frame_width 16 \
    --time_conv_kernel 7 \
    --d_model 512 \
    --n_tokens 0 \
    --eval_steps 20 \
    --batch_size 8 \
    --num_workers 2 \
    --epochs 500 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --scheduler none \
    --grad_clip 1.0 \
    --n_fft 256 \
    --hop_length 128 \
    --log_interval 5 \
    --num_plots 4 \
    --checkpoint_dir runs/co2_channel_ast_diffusion_fw16_d512 \
    --resume
