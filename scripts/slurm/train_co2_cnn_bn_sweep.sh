#!/bin/bash
# Usage: sbatch --export=BN_DIM=4 scripts/slurm/train_co2_cnn_bn_sweep.sh
#SBATCH --job-name=co2_cnn_bn
#SBATCH --output=logs/%j_co2_cnn_bn%x.out
#SBATCH --error=logs/%j_co2_cnn_bn%x.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64G

module load pixi

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

BN_DIM=${BN_DIM:?"Set BN_DIM via --export=BN_DIM=<value>"}

srun pixi run python scripts/training/spectrogram_reconstruction.py \
    --signal co2 \
    --model spectrogram_cnn \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path data/preprocessing_stats.pt \
    --shot_min 200000 \
    --shot_max 200500 \
    --cnn_dims 64 128 \
    --bottleneck_dim "$BN_DIM" \
    --d_model 256 \
    --n_tokens 0 \
    --batch_size 16 \
    --num_workers 2 \
    --epochs 500 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --warmup_epochs 5 \
    --scheduler cosine \
    --n_fft 256 \
    --hop_length 128 \
    --log_interval 5 \
    --num_plots 4 \
    --checkpoint_dir "runs/co2_cnn_bn${BN_DIM}"
