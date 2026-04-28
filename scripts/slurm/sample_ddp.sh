#!/bin/bash
#SBATCH --job-name=cond_imgnet_pretrain_h200
#SBATCH --partition=ailab
#SBATCH --output=logs/%j_cond_imgnet_pretrain_h200.out
#SBATCH --error=logs/%j_cond_imgnet_pretrain_h200.err
#SBATCH --time=72:00:00
#SBATCH --account=kolemen
#SBATCH --qos=gpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=13G

source .venv/bin/activate
export OMP_NUM_THREADS=1
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500

srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=$SLURM_GPUS_PER_NODE \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  --rdzv_backend=c10d \
  train_tarflow.py --dataset=imagenet --img_size=128 --channel_size=3 \
  --patch_size=4 --channels=1024 --blocks=8 --layers_per_block=8 \
  --noise_std=0.15 --batch_size=380 --epochs=320 --lr=1e-4 --nvp --cfg=0 --drop_label=0.1 \
  --sample_freq=20 --logdir=runs/imagenet128-cond-pretrain-32 --compile --accum_steps=2 --num_workers=10 \
  --resume=runs/imagenet128-cond-pretrain-32/imagenet_model_4_1024_8_8_0.15.pth