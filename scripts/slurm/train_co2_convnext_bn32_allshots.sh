#!/bin/bash
#SBATCH --job-name=co2_convnext_bn32_all
#SBATCH --output=logs/%j_co2_convnext_bn32_all.out
#SBATCH --error=logs/%j_co2_convnext_bn32_all.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64G

module load pixi

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# All good shots (>= 200000), excludes degenerate early shots
srun pixi run python scripts/training/spectrogram_reconstruction.py \
    --signal co2 \
    --model spectrogram_cnn_perceiver \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path data/preprocessing_stats.pt \
    --shot_min 200000 \
    --convnext_dims 64 128 256 \
    --convnext_depths 2 2 6 \
    --stem_stride 4 \
    --bottleneck_dim 32 \
    --d_model 32 \
    --n_tokens 0 \
    --batch_size 16 \
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
    --checkpoint_dir runs/co2_convnext_bn32_allshots
