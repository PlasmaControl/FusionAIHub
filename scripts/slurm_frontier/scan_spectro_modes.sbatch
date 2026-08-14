#!/bin/bash
#SBATCH -A fus187
#SBATCH -J scan_modes
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 1
# Rank all shots by per-modality spectrogram mode activity (co2/bes/mhr/ece).
# CPU-only, ProcessPoolExecutor across the node's cores. See
# scripts/data_preparation/scan_spectro_modes.py. Env: WORKERS, N_WINDOWS,
# MODALITIES, OUT_DIR, MAX_SHOTS.
FMH=/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub
cd "$FMH"
source scripts/slurm_frontier/_frontier_common.sh
export PYTHONPATH="$FMH/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
WORKERS="${WORKERS:-56}"
N_WINDOWS="${N_WINDOWS:-40}"
OUT_DIR="${OUT_DIR:-eval_runs/spectro_mode_scan}"
MODALITIES="${MODALITIES:-ece co2 bes mhr}"
TARGET="${TARGET:-modes}"
MAX_SHOTS_FLAG=""
[ -n "${MAX_SHOTS:-}" ] && MAX_SHOTS_FLAG="--max_shots ${MAX_SHOTS}"
EXTRA=""
[ "$TARGET" = "elm" ] && EXTRA="--target elm --prom_k ${PROM_K:-5.0} --refractory_ms ${REFRACTORY_MS:-1.0}"
echo "[scan_modes] host=$(hostname) target=$TARGET workers=$WORKERS n_windows=$N_WINDOWS out=$OUT_DIR mods=$MODALITIES"
python scripts/data_preparation/scan_spectro_modes.py \
    --workers "$WORKERS" --n_windows "$N_WINDOWS" --out "$OUT_DIR" \
    --modalities $MODALITIES $MAX_SHOTS_FLAG $EXTRA
echo "=== SCAN DONE target=$TARGET (exit $?) ==="
