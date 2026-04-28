#!/bin/bash
# 2-GPU DDP launcher for E2E Stage 2 on AMD MI210 (della-milan).
# Usage:
#   bash scripts/slurm_rocm/train_e2e_stage2_ddp.sh
# Env overrides:
#   GPUS              (default: "0,1")
#   BATCH_SIZE        per-rank, (default: 8 — bf16 rollouts are heavier than stage1)
#   MAX_STEPS         (default: 1000 smoke; bump for prod)
#   K_MAX             (default: 10)
#   CURRICULUM_STEPS  (default: matches MAX_STEPS / 2)
#   D_MODEL N_LAYERS N_HEADS  (default: 256 / 8 / 8)
#   MAX_FILES         (default: unset = all)
#   INIT_CHECKPOINT   path to stage1 best (default: runs/e2e_stage1/e2e_stage1_best.pt)
#   NO_AMP=1          disable bf16 autocast
#   MASTER_PORT       (default: 29501)
#
#SBATCH --job-name=e2e_stage2_ddp_rocm
#SBATCH --output=logs/%j_e2e_stage2_ddp.out
#SBATCH --error=logs/%j_e2e_stage2_ddp.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
set -uo pipefail

GPUS="${GPUS:-0,1}"
NPROC=$(awk -F, '{print NF}' <<<"$GPUS")
MASTER_PORT="${MASTER_PORT:-29501}"

BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-1000}"
K_MAX="${K_MAX:-10}"
CURRICULUM_STEPS="${CURRICULUM_STEPS:-$((MAX_STEPS / 2))}"
D_MODEL="${D_MODEL:-256}"
N_LAYERS="${N_LAYERS:-8}"
N_HEADS="${N_HEADS:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_EVERY="${LOG_EVERY:-50}"
VAL_EVERY="${VAL_EVERY:-200}"
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-20}"

DATA_DIR="${DATA_DIR:-/scratch/gpfs/EKOLEMEN/foundation_model}"
STATS_PATH="${STATS_PATH:-data/preprocessing_stats.pt}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-runs/e2e_stage2_ddp}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/e2e_stage1/e2e_stage1_best.pt}"

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

INIT_FLAG=""
if [ -f "$INIT_CHECKPOINT" ]; then
    INIT_FLAG="--init_checkpoint $INIT_CHECKPOINT"
    echo "[ddp_stage2] init from $INIT_CHECKPOINT"
else
    echo "[ddp_stage2] WARNING: $INIT_CHECKPOINT not found — random init"
fi

MAX_FILES_FLAG=""
if [ -n "${MAX_FILES:-}" ]; then
    MAX_FILES_FLAG="--max_files $MAX_FILES"
fi

NO_AMP_FLAG=""
if [ "${NO_AMP:-0}" = "1" ]; then
    NO_AMP_FLAG="--no_amp"
fi

echo "[ddp_stage2] gpus=$GPUS nproc=$NPROC batch=$BATCH_SIZE steps=$MAX_STEPS K_max=$K_MAX d_model=$D_MODEL"

torchrun \
    --nnodes=1 \
    --nproc_per_node="$NPROC" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    scripts/training/train_e2e_stage2.py \
    $INIT_FLAG \
    $MAX_FILES_FLAG \
    $NO_AMP_FLAG \
    --data_dir "$DATA_DIR" \
    --stats_path "$STATS_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --val_fraction 0.1 \
    --seed 42 \
    --chunk_duration_s 0.05 \
    --step_size_s 0.01 \
    --warmup_s 1.0 \
    --d_model "$D_MODEL" \
    --n_layers "$N_LAYERS" \
    --n_heads "$N_HEADS" \
    --dropout 0.1 \
    --K_max "$K_MAX" \
    --curriculum_steps "$CURRICULUM_STEPS" \
    --lr 3e-5 \
    --min_lr 1e-6 \
    --warmup_steps 200 \
    --weight_decay 0.1 \
    --grad_clip 5.0 \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --max_steps "$MAX_STEPS" \
    --log_every "$LOG_EVERY" \
    --val_every "$VAL_EVERY" \
    --val_max_batches "$VAL_MAX_BATCHES"
