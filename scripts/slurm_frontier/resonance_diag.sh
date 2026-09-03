#!/bin/bash
#SBATCH -A fus187
#SBATCH -J resonance_diag
#SBATCH -o logs/%j_resonance_diag.out
#SBATCH -e logs/%j_resonance_diag.err
#SBATCH -t 0:40:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e

# RESONANCE DIAGNOSTIC — T1 (mode energy) vs T2 (roughness/realization bits) for
# the ece SpectrogramTokenizer.proj resonance. Per mode-active window, compares
# GT-path proj-absmax vs predicted-path proj-absmax of the SAME window's ridge.
# READ-ONLY on all model dirs; writes only to eval_runs/resonance_diag.
#
# Usage: sbatch scripts/slurm_frontier/resonance_diag.sh [ckpt]

FMH=/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub
cd "$FMH"
mkdir -p logs eval_runs/resonance_diag

CKPT="${1:-/lustre/orion/fus187/proj-shared/models/e2e_g3fix_anneal/e2e_stage1_beta6.0_step3000.pt}"

export MASTER_PORT=29594
source scripts/slurm_frontier/_frontier_common.sh

# Persistent shared MIOpen kernel cache (same as the render jobs — reuse compiled
# kernels; this is a 1-GPU short job, not 64-rank training).
export MIOPEN_USER_DB_PATH="/lustre/orion/fus187/proj-shared/ps9551/.miopen_eval_cache"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

export PYTHONPATH="$FMH/src:$FMH/scripts/training:$FMH/analysis/mode_audit:$PYTHONPATH"
export EXTRA_DATA_DIR="${EXTRA_DATA_DIR:-/lustre/orion/fus187/proj-shared/additional_data}"
export SHOT="${SHOT:-200729}"
export BATCH="${BATCH:-16}"
export MAX_WIN="${MAX_WIN:-64}"
export OUT_DIR="${OUT_DIR:-$FMH/eval_runs/resonance_diag}"
export CACHE_DIR="${CACHE_DIR:-$FMH/eval_runs/resonance_diag/cache}"

echo "[resonance_diag] ckpt    : $CKPT"
echo "[resonance_diag] shot    : $SHOT   batch=$BATCH  max_win=$MAX_WIN"
echo "[resonance_diag] out_dir : $OUT_DIR"

python analysis/mode_audit/resonance_diag.py "$CKPT"

echo "[resonance_diag] result in: $OUT_DIR/{resonance_diag.json,spatial_spectrum.png}"
