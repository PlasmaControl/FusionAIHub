#!/bin/bash
# Train all priority unimodal autoencoders sequentially.
# Usage: bash scripts/training/train_all_unimodal.sh
#
# For SLURM job arrays, set SLURM_ARRAY_TASK_ID to select a signal index.

set -euo pipefail

DATA_DIR="${DATA_DIR:-/scratch/gpfs/EKOLEMEN/big_d3d_data/dummy_foundation_model_data}"
STATS_PATH="${STATS_PATH:-data/preprocessing_stats.pt}"
RUN_DIR="${RUN_DIR:-runs}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
D_MODEL="${D_MODEL:-64}"
VAL_SPLIT="${VAL_SPLIT:-0.2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-3}"

# Priority signals and their model types
SIGNALS=(
    "ece"
    "co2"
    "pin"
    "tin"
    "ech"
    "mse"
    "ts_core_density"
    "filterscopes"
)

# If running as SLURM array, only run the indexed signal
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    IDX=$SLURM_ARRAY_TASK_ID
    if [ "$IDX" -ge "${#SIGNALS[@]}" ]; then
        echo "SLURM_ARRAY_TASK_ID=$IDX exceeds number of signals (${#SIGNALS[@]})"
        exit 1
    fi
    SIGNALS=("${SIGNALS[$IDX]}")
fi

for SIGNAL in "${SIGNALS[@]}"; do
    echo "================================================"
    echo "Training: ${SIGNAL}"
    echo "================================================"

    CHECKPOINT_DIR="${RUN_DIR}/${SIGNAL}"

    python scripts/training/train_unimodal_autoencoder.py \
        --signal "$SIGNAL" \
        --d_model "$D_MODEL" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --val_split "$VAL_SPLIT" \
        --num_workers "$NUM_WORKERS" \
        --lr "$LR" \
        --data_dir "$DATA_DIR" \
        --stats_path "$STATS_PATH" \
        --checkpoint_dir "$CHECKPOINT_DIR"

    echo "Completed: ${SIGNAL}"
    echo ""
done

echo "All unimodal training complete."
