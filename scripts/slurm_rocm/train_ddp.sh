#!/bin/bash
# 2-GPU DDP launcher for ROCm on della-milan.
# Usage:
#   SIGNAL=ece bash scripts/slurm_rocm/train_ddp.sh
# Env:
#   SIGNAL          required signal name (matches MODEL_REGISTRY entry)
#   BATCH_SIZE      per-GPU batch size (default: 4)
#   EPOCHS          (default: 2)
#   D_MODEL         (default: 64)
#   MASTER_PORT     (default: 29500)
#   GPUS            comma list of GPU IDs to use (default: "0,1")
#
#SBATCH --job-name=train_ddp_rocm
#SBATCH --output=logs/%j_train_ddp_rocm.out
#SBATCH --error=logs/%j_train_ddp_rocm.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G
set -uo pipefail

: "${SIGNAL:?SIGNAL env var required}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-2}"
D_MODEL="${D_MODEL:-64}"
GPUS="${GPUS:-0,1}"
NPROC=$(awk -F, '{print NF}' <<<"$GPUS")
MASTER_PORT="${MASTER_PORT:-29500}"

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTORCH_ROCM_ARCH=gfx90a
export HIP_VISIBLE_DEVICES="$GPUS"
export PATH=/opt/rocm-7.2.1/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib:${LD_LIBRARY_PATH:-}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT
mkdir -p logs

cd /scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub
source .venv-rocm/bin/activate

echo "[ddp] signal=$SIGNAL gpus=$GPUS nproc=$NPROC batch=$BATCH_SIZE epochs=$EPOCHS"

torchrun \
    --nnodes=1 \
    --nproc_per_node="$NPROC" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    scripts/training/train_unimodal_autoencoder.py \
    --signal "$SIGNAL" \
    --d_model "$D_MODEL" \
    --batch_size "$BATCH_SIZE" \
    --num_workers 16 \
    --epochs "$EPOCHS" \
    --lr 1e-3 \
    --weight_decay 0.05 \
    --warmup_epochs 5 \
    --val_split 0.2 \
    --chunk_duration_s 0.2 \
    --n_fft 256 \
    --hop_length 256 \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path data/preprocessing_stats.pt \
    --checkpoint_dir "runs/${SIGNAL}_ddp"
