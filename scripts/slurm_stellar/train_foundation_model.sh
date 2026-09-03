#!/bin/bash
#SBATCH --job-name=fm_fusion
#SBATCH --output=logs/%j_fm_fusion.out
#SBATCH --error=logs/%j_fm_fusion.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=32G

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun pixi run python ../training/train_foundation_model.py \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model/ \
    --stats_path /projects/EKOLEMEN/foundation_model/preprocessing_stats.pt \
    --ae_checkpoint_dir /projects/EKOLEMEN/foundation_model/ \
    --checkpoint_dir runs/foundation_model \
    --d_model 256 \
    --n_latent 128 \
    --encoder_layers 1 \
    --processor_layers 2 \
    --decoder_layers 3 \
    --dynamics_layers 3 \
    --dynamics_type cross_attention \
    --ema_decay 0.996 \
    --encode_loss_weight 0.0 \
    --rollout_loss_weight 2.0 \
    --signal_loss_weight 0.1 \
    --delta_loss_weight 1.0 \
    --n_heads 8 \
    --dropout 0.1 \
    --batch_size 64 \
    --num_workers 8 \
    --prefetch_factor 4 \
    --epochs 500 \
    --encoder_lr 1e-5 \
    --dynamics_lr 1e-3 \
    --weight_decay 0.05 \
    --warmup_epochs 5 \
    --min_lr 1e-6 \
    --steps_per_epoch 0 \
    --plot_every 1 \
    --rollout_start 1 \
    --rollout_ramp_epochs 30 \
    --rollout_noise_std 0.1 \
    --teacher_forcing_start 0.5 \
    --teacher_forcing_epochs 40 \
    --context_noise_std 0.1 \
    --context_drop_rate 0.1 \
    --warmup_s 1.0