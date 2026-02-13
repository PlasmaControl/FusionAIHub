#!/bin/bash
#SBATCH --job-name=train_unimodal
#SBATCH --output=logs/%j_train_unimodal.out
#SBATCH --error=logs/%j_train_unimodal.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

EPOCHS=50
D_MODEL=256
BATCH_SIZE=2
NUM_WORKERS=4
LR=0.001
WEIGHT_DECAY=0.05
WARMUP_EPOCHS=5
MIN_LR=0.0
CHECKPOINT_DIR=runs

# Spectrograms + fast time series (no profiles or videos)
SIGNALS=(mhr ece co2 d_alpha gas ech pin tin)

for SIGNAL in "${SIGNALS[@]}"; do
    echo "============================================"
    echo "Training signal: ${SIGNAL}"
    echo "============================================"
    srun pixi run python train_unimodal_autoencoder.py \
        --signal "${SIGNAL}" \
        --d_model "${D_MODEL}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --warmup_epochs "${WARMUP_EPOCHS}" \
        --min_lr "${MIN_LR}" \
        --checkpoint_dir "${CHECKPOINT_DIR}"
done

echo "All modalities complete."
