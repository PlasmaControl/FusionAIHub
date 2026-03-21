#!/bin/bash
#SBATCH --job-name=co2_cnxt_bn32_s4c4
#SBATCH --output=logs/%j_co2_cnxt_bn32_s4c4.out
#SBATCH --error=logs/%j_co2_cnxt_bn32_s4c4.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64G

module load pixi

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Two-stage bottleneck:
#   Stage 1: ConvNeXt stem_stride=4 → 32×489 spatial, then 1×1 conv 256→32 channels
#   Stage 2: compress_stride=4 (two stride-2 convs) → 8×123 spatial, d_model=256
# Final tokens: 8×123 = 984 tokens of d_model=256
# ConvNeXt processes at full 32×489 resolution before compression
srun pixi run python scripts/training/spectrogram_reconstruction.py \
    --signal co2 \
    --model spectrogram_cnn_perceiver \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path data/preprocessing_stats.pt \
    --shot_min 200000 \
    --shot_max 200500 \
    --convnext_dims 256 \
    --convnext_depths 6 \
    --stem_stride 4 \
    --bottleneck_dim 32 \
    --compress_stride 4 \
    --d_model 256 \
    --n_tokens 0 \
    --batch_size 8 \
    --num_workers 2 \
    --epochs 500 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --scheduler cosine \
    --warmup_epochs 10 \
    --min_lr 1e-6 \
    --grad_clip 1.0 \
    --n_fft 256 \
    --hop_length 128 \
    --log_interval 5 \
    --num_plots 4 \
    --checkpoint_dir runs/co2_convnext_bn32_s4_compress4
