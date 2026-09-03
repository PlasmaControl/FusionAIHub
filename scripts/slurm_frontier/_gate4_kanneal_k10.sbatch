#!/bin/bash
#SBATCH -A fus187
#SBATCH -J g4_kanneal_k10
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"; cd "${PROJECT_DIR}"; mkdir -p logs analysis/mode_audit
export MASTER_PORT=29582
source scripts/slurm_frontier/_frontier_common.sh
export MIOPEN_USER_DB_PATH="/lustre/orion/fus187/proj-shared/ps9551/.miopen_eval_cache"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"; mkdir -p "$MIOPEN_USER_DB_PATH"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# ── K=10 GATE (step-5000, e2e_g3fix_kanneal_v2) — EXACT block-0 baseline protocol ──
# Reproduce the established EXIT INSTRUMENT (EXPERIMENTS.md:724): argmax paired
# counterfactual, 200729, n=256, k∈{0,10,39}, DOSES 0,±2σ, β=6, K=40 gate@10.
# Matches g4_block0_5010348 (the paired denominator) byte-for-env.
CKPT="/lustre/orion/fus187/proj-shared/models/e2e_g3fix_kanneal_v2/e2e_stage1_latest.pt"
export SHOT=200729
export K=40
export K_GATE=10
export DOSES="0,2,-2"
export ACT=pin
export BATCH=8
export MAX_WIN=256
export FEEDBACK_MODE=argmax
export TEMP=1.0
export DESC_ANCHOR_BETA=6.0
export OUT_DIR="$PROJECT_DIR/eval_runs/gate4_kanneal_v2_k10_step5000"
export CACHE_DIR="$PROJECT_DIR/eval_runs/gate4_cache"

python analysis/mode_audit/gate4_kprobe.py "$CKPT"
echo "[g4_kanneal_k10] done"
