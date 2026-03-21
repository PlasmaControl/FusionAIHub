#!/bin/bash
#SBATCH --job-name=co2_cnn1d_v2
#SBATCH --output=logs/%j_co2_cnn1d_v2.out
#SBATCH --error=logs/%j_co2_cnn1d_v2.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64G

module load pixi

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# 1D ConvNeXt with hierarchical frequency-reducing stem
# Stem: Conv2d(4,64,(4,2),stride=(4,2)) -> Conv2d(64,128,(4,1),stride=(4,1))
#       -> Conv2d(128,256,(8,1),stride=(8,1))
# Frequency: 128 -> 32 -> 8 -> 1 (gradual, not one-shot)
# 6 ConvNeXt 1D blocks for temporal context
# Token count: 977 tokens x 32 channels (same as AST-FSQ)
srun pixi run python scripts/training/spectrogram_reconstruction.py \
    --signal co2 \
    --model spectrogram_cnn1d \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path data/preprocessing_stats.pt \
    --shot_min 200000 \
    --shot_max 200500 \
    --convnext_dims 256 \
    --convnext_depths 6 \
    --stem_dims 64 128 \
    --frame_width 2 \
    --bottleneck_dim 32 \
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
    --checkpoint_dir runs/co2_cnn1d_v2
