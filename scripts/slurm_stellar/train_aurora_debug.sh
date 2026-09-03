#!/bin/bash
#SBATCH --job-name=aurora_debug
#SBATCH --output=logs/%j_aurora_debug.out
#SBATCH --error=logs/%j_aurora_debug.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=4G

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun pixi run python ../training/train_aurora.py \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model/ \
    --stats_path /projects/EKOLEMEN/foundation_model/preprocessing_stats.pt \
    --ae_checkpoint_dir /projects/EKOLEMEN/foundation_model/ \
    --ae_token_stats_path /projects/EKOLEMEN/foundation_model/ae_token_stats.pt \
    --checkpoint_dir runs/aurora_debug \
    --d_model 128 \
    --n_latent 64 \
    --encoder_cross_layers 2 \
    --encoder_self_layers 2 \
    --backbone_blocks 8 \
    --decoder_layers 2 \
    --n_heads 4 \
    --mlp_ratio 2.0 \
    --dropout 0.1 \
    --max_files 500 \
    --batch_size 16 \
    --num_workers 4 \
    --prefetch_factor 2 \
    --pretrain_epochs 50 \
    --finetune_epochs 30 \
    --pretrain_lr 1e-4 \
    --finetune_lr 3e-5 \
    --weight_decay 0.05 \
    --warmup_epochs 5 \
    --min_lr 1e-6 \
    --max_rollout 8 \
    --rollout_ramp_epochs 15 \
    --plot_every 5 \
    --warmup_s 1.0 \
    --recon_weight 0.0 \
    --delta_weight 1.0 \
    --step_diversity_weight 1.0
