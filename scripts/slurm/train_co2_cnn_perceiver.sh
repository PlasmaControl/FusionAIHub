#!/bin/bash
#SBATCH --job-name=co2_cnn_perceiver
#SBATCH --output=logs/%j_co2_cnn_perceiver.out
#SBATCH --error=logs/%j_co2_cnn_perceiver.err
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
    --model spectrogram_cnn_perceiver \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path data/preprocessing_stats.pt \
    --cnn_dims 64 128 \
    --d_model 256 \
    --n_tokens 16 \
    --n_heads 4 \
    --n_self_layers 2 \
    --n_dec_self_layers 2 \
    --dropout 0.1 \
    --batch_size 16 \
    --num_workers 2 \
    --epochs 500 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --scheduler cosine \
    --warmup_epochs 10 \
    --min_lr 1e-6 \
    --n_fft 256 \
    --hop_length 128 \
    --log_interval 5 \
    --num_plots 4 \
    --checkpoint_dir runs/co2_cnn_perceiver
