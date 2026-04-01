#!/bin/bash
#SBATCH --job-name=co2_chast_gan_fw16
#SBATCH --output=logs/%j_co2_chast_gan_fw16.out
#SBATCH --error=logs/%j_co2_chast_gan_fw16.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64G

module load pixi

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Channel-AST-GAN (PatchGAN + R3GAN), fw=16, d_model=512 — CO2 (C=4)
# Generator: Channel-AST autoencoder (no FSQ, continuous)
# Discriminator: PatchGAN per-channel, no normalization
# Loss: L1 + 0.1 * RpGAN adversarial + R1 + R2 gradient penalties
# D optimizer: Adam(β₁=0, β₂=0.9, lr=2e-4) per R3GAN
srun pixi run python scripts/training/spectrogram_reconstruction.py \
    --signal co2 \
    --model spectrogram_channel_ast_gan \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path data/preprocessing_stats.pt \
    --frame_width 16 \
    --time_conv_kernel 7 \
    --d_model 512 \
    --n_tokens 0 \
    --adv_weight 0.1 \
    --gp_gamma 10.0 \
    --d_lr 2e-4 \
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
    --checkpoint_dir runs/co2_channel_ast_gan_fw16 \
    --resume
