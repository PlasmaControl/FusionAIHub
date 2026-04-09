#!/bin/bash
#SBATCH --job-name=train_cer_rot_rocm
#SBATCH --output=logs/%j_train_cer_rot_rocm.out
#SBATCH --error=logs/%j_train_cer_rot_rocm.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTORCH_ROCM_ARCH=gfx90a

cd /scratch/gpfs/nc1514/FusionAIHub
source .venv-rocm/bin/activate

srun python scripts/training/train_unimodal_autoencoder.py \
    --signal "cer_rot" \
    --d_model 64 \
    --batch_size 8 \
    --num_workers 16 \
    --epochs 50 \
    --lr 1e-3 \
    --weight_decay 0.05 \
    --warmup_epochs 5 \
    --val_split 0.2 \
    --chunk_duration_s 0.2 \
    --n_fft 256 \
    --hop_length 256 \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path data/preprocessing_stats.pt \
    --checkpoint_dir runs/cer_rot
