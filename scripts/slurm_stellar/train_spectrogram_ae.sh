#!/bin/bash
#SBATCH --job-name=spectro_ae
#SBATCH --output=logs/%j_spectrogram_ae.out
#SBATCH --error=logs/%j_spectrogram_ae.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=32G

# Standalone spectrogram autoencoder validation (Phase B Step 6).
# Trains SpectrogramTokenizer + SpectrogramOutputHead end-to-end on
# masked MAE for ~5k steps to validate that per-patch tokens reconstruct
# the modality's spectrogram structure before Step 5 integration.
#
# Per-modality: ECE 40 ch / patch (F=32, T=8) / 192 tok / 40x compression
#               CO2  4 ch / patch (F=64, T=8) /  96 tok /  8x
#               BES 16 ch / patch (F=32, T=8) / 192 tok / 16x
#
# Usage (positional arg):
#   sbatch train_spectrogram_ae.sh ece
#   sbatch train_spectrogram_ae.sh co2
#   sbatch train_spectrogram_ae.sh bes
#
# Output goes to runs/spectrogram_ae_<modality>/ relative to scripts/slurm/.
# This job is intentionally short (4 h wall) and disjoint from the
# Phase A/B production pipelines.

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

MODALITY="${1:-}"
if [ -z "$MODALITY" ]; then
    echo "Usage: sbatch $0 <ece|co2|bes>" >&2
    exit 1
fi
case "$MODALITY" in
    ece|co2|bes) ;;
    *) echo "Modality must be one of {ece, co2, bes}; got '$MODALITY'" >&2; exit 1 ;;
esac

CHECKPOINT_DIR="runs/spectrogram_ae_${MODALITY}"

echo "Modality:        $MODALITY"
echo "Checkpoint dir:  $CHECKPOINT_DIR"

srun pixi run python ../training/train_spectrogram_ae.py \
    --modality "$MODALITY" \
    --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
    --stats_path /projects/EKOLEMEN/foundation_model/preprocessing_stats.pt \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --max_steps 5000 \
    --batch_size 128 \
    --num_workers 8 \
    --lr 1e-3 \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --log_every 50 \
    --val_every 500 \
    --val_fraction 0.05 \
    --seed 42