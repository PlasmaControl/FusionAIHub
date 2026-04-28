#!/bin/bash
# 2-GPU DDP launcher for E2E Stage 3 (LoRA + displacement loss + replay
# buffer) on AMD MI210.
#
#SBATCH --job-name=e2e_stage3_ddp_rocm
#SBATCH --output=logs/%j_e2e_stage3_ddp.out
#SBATCH --error=logs/%j_e2e_stage3_ddp.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
set -uo pipefail

GPUS="${GPUS:-0,1}"
NPROC=$(awk -F, '{print NF}' <<<"$GPUS")
MASTER_PORT="${MASTER_PORT:-29504}"

BATCH_SIZE="${BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-1000}"
K_MIN="${K_MIN:-2}"
K_MAX="${K_MAX:-4}"
N_CURRICULUM_BLOCKS="${N_CURRICULUM_BLOCKS:-2}"
CURRICULUM_STEPS="${CURRICULUM_STEPS:-$((MAX_STEPS / 2))}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16.0}"
POOL_SIZE="${POOL_SIZE:-50}"
BUFFER_SIZE="${BUFFER_SIZE:-500}"
BUFFER_REFRESH_PERIOD="${BUFFER_REFRESH_PERIOD:-50}"
BUFFER_REFRESH_FRACTION="${BUFFER_REFRESH_FRACTION:-0.1}"
D_MODEL="${D_MODEL:-256}"
N_LAYERS="${N_LAYERS:-8}"
N_HEADS="${N_HEADS:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_EVERY="${LOG_EVERY:-50}"
VAL_EVERY="${VAL_EVERY:-200}"

DATA_DIR="${DATA_DIR:-/scratch/gpfs/EKOLEMEN/foundation_model}"
STATS_PATH="${STATS_PATH:-data/preprocessing_stats.pt}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-runs/e2e_stage3_ddp}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/e2e_stage2_delta/e2e_stage2_delta_best.pt}"

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
fi

LATEST="$CHECKPOINT_DIR/e2e_stage3_latest.pt"
RESUME_FLAG=""
if [ -f "$LATEST" ]; then
    RESUME_FLAG="--resume_checkpoint $LATEST"
fi

MAX_FILES_FLAG=""
if [ -n "${MAX_FILES:-}" ]; then
    MAX_FILES_FLAG="--max_files $MAX_FILES"
fi

NO_AMP_FLAG=""
if [ "${NO_AMP:-0}" = "1" ]; then
    NO_AMP_FLAG="--no_amp"
fi

USE_DISP_FLAG="--use_displacement_loss"
if [ "${NO_DISPLACEMENT_LOSS:-0}" = "1" ]; then
    USE_DISP_FLAG=""
fi

echo "[ddp_stage3] gpus=$GPUS nproc=$NPROC batch=$BATCH_SIZE steps=$MAX_STEPS K=[$K_MIN,$K_MAX]"

torchrun \
    --nnodes=1 \
    --nproc_per_node="$NPROC" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    scripts/training/train_e2e_stage3.py \
    $INIT_FLAG \
    $RESUME_FLAG \
    $MAX_FILES_FLAG \
    $NO_AMP_FLAG \
    $USE_DISP_FLAG \
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
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --K_min "$K_MIN" \
    --K_max "$K_MAX" \
    --n_curriculum_blocks "$N_CURRICULUM_BLOCKS" \
    --curriculum_steps "$CURRICULUM_STEPS" \
    --pool_size "$POOL_SIZE" \
    --buffer_size "$BUFFER_SIZE" \
    --buffer_refresh_period "$BUFFER_REFRESH_PERIOD" \
    --buffer_refresh_fraction "$BUFFER_REFRESH_FRACTION" \
    --lr 3e-5 \
    --min_lr 1e-7 \
    --warmup_steps 200 \
    --weight_decay 0.01 \
    --grad_clip 5.0 \
    --cos_weight 0.3 \
    --mag_weight 0.1 \
    --min_disp_norm 0.01 \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --max_steps "$MAX_STEPS" \
    --log_every "$LOG_EVERY" \
    --val_every "$VAL_EVERY" \
    --val_batch_size "$VAL_BATCH_SIZE"
