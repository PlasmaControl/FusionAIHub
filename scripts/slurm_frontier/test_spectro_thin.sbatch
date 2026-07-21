#!/bin/bash
#SBATCH -A fus187
#SBATCH -J spectro_thin
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
#SBATCH -t 1:30:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gres=gpu:1
# Objective thin-pattern reconstruction test (encode->decode round-trip),
# OLD (512,4) vs NEW (64,32 + positional embeddings) at equal 24-token budget.
# All test hyperparameters pass through as args, e.g.:
#   sbatch scripts/slurm_frontier/test_spectro_thin.sbatch --base_ch 64 --steps 6000 \
#          --freq_pe_ch 24 --time_pe_ch 8 --out_dir eval_runs/spectro_thin_test/iterN
FMH=/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub
cd "$FMH"
source scripts/slurm_frontier/_frontier_common.sh
# _frontier_common sets a COLD per-job /tmp MIOpen cache -> every conv shape
# recompiles from scratch, which stalls this test at step 1 for tens of minutes
# (the flow U-Net's 512xT convs are the worst). Redirect to the SHARED persistent
# cache (same one the fast eval jobs use) so compiled kernels are reused across
# runs, and use FAST find-mode (heuristic, skips exhaustive kernel benchmarking).
# MIOPEN_SHARED=1 -> reuse the persistent warm cache (fast when kernels already
# compiled). Default off: _frontier_common's fresh per-job /tmp cache avoids
# reusing a possibly-poisoned kernel entry (FIND_MODE=2 stalled job 4927974).
if [ -n "${MIOPEN_SHARED:-}" ]; then
    export MIOPEN_USER_DB_PATH="/lustre/orion/fus187/proj-shared/ps9551/.miopen_eval_cache"
    export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
    mkdir -p "$MIOPEN_USER_DB_PATH"
fi
# MIOPEN_FAST=1 -> FIND_MODE=2 (fast heuristic compile) — can pick a kernel that
# stalls at RUNTIME (job 4927974). Off by default.
[ -n "${MIOPEN_FAST:-}" ] && export MIOPEN_FIND_MODE=2
export PYTHONPATH="$FMH/src${PYTHONPATH:+:$PYTHONPATH}"
echo "[spectro_thin] host=$(hostname) gpu=${ROCR_VISIBLE_DEVICES:-?} args=$*"
python scripts/training/test_spectro_pattern_reconstruction.py "$@"
echo "=== SPECTRO THIN TEST DONE (exit $?) ==="
