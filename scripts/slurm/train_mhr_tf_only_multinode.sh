#!/bin/bash
#SBATCH --job-name=train_mhr_tf_only
#SBATCH --output=logs/%j_train_mhr_tf_only.out
#SBATCH --error=logs/%j_train_mhr_tf_only.err
#SBATCH --time=12:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1


nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Head node: $head_node ($head_node_ip)"

srun pixi run torchrun \
    --nnodes=2 \
    --nproc_per_node=2 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:29500 \
    scripts/training/train_unimodal_autoencoder.py \
    -- \
    --signal mhr \
    --model spectrogram_tf_only \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --batch_size 10 \
    --num_workers 4 \
    --epochs 300 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --warmup_epochs 0 \
    --min_lr 1e-5 \
    --n_fft 256 \
    --hop_length 256 \
    --chunk_duration_s 0.2 \
    --log_interval 1 \
    --num_plots 1 \
    --val_split 0.05 \
    --use_wandb \
    --use_metrics \
    --loss_type l1_ssim \
    --checkpoint_dir runs/mhr_tf_only \
    --model_kwargs '{"hidden_dim": 128, "latent_dim": 4, "freq_dim": 16, "lstm_hidden": 96, "lstm_layers": 1}'
