#!/bin/bash
#SBATCH --job-name=train_mhr_gate_512
#SBATCH --output=logs/%j_train_mhr_gate_512.out
#SBATCH --error=logs/%j_train_mhr_gate_512.err
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32G

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun pixi run torchrun \
    --standalone \
    --nproc_per_node=1 \
    scripts/training/train_unimodal_autoencoder.py \
    -- \
    --signal mhr \
    --model spectrogram_tf_only \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --batch_size 2 \
    --num_workers 2 \
    --epochs 50 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --warmup_epochs 0 \
    --min_lr 1e-5 \
    --n_fft 1024 \
    --hop_length 256 \
    --chunk_duration_s 0.25 \
    --log_interval 3 \
    --num_plots 1 \
    --val_split 0.05 \
    --use_wandb \
    --use_metrics \
    --use_gated_target \
    --checkpoint_dir runs/mhr_gate_512_1gpu \
    --model_kwargs '{"hidden_dim": 128, "latent_dim": 4, "freq_dim": 16, "lstm_hidden": 96, "lstm_layers": 1, "lstm_on": false}'
