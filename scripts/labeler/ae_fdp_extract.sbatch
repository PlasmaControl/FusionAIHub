#!/bin/bash
#SBATCH --job-name=ae-fdp-extract
#SBATCH --account=nstx-u
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=17G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%j.out

# Staged .BCI files -> compact CO2 chords (scripts/labeler/ae_fdp_label.py extract), no network.
# The stage subcommand submits this per batch with LIST=<root>/batches/bNNNN.csv; the infer job
# runs after it. STAGE_DIR reads another stage (D3D/fdp_stage for the equivalence test) and
# KEEP=1 keeps the staged files (always kept outside <root>/stage).
#   LIST=... [STAGE_DIR=...] [KEEP=1] sbatch scripts/labeler/ae_fdp_extract.sbatch
# Each worker holds ~1.6 GB (four float64 chords of ~17 M samples), so 8 workers fit in 17G. 8 CPUs
# backfill into a full cpu partition (2026-11-03); a 2,000-shot batch takes ~8 min.
set -euo pipefail
REPO=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub
export LABELER_ROOT="${LABELER_ROOT:-$REPO/.cache/labeler}"
export AE_FDP_ROOT="${AE_FDP_ROOT:-$LABELER_ROOT/ae_fdp_1ms}"
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export SYS_D3_DELIM=';' PTDATA_LOC=1 PYTHONUNBUFFERED=1
unset BEARER_TOKEN PTDATA_JSON_INDEX_DIR PTDATA_JSON_INDEX_PATTERN PTDATA_PLUGIN_LIB
cd "$REPO"
echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -Is): list $LIST, root $AE_FDP_ROOT"
ARGS=(--root "$AE_FDP_ROOT" extract --list "$LIST" --workers "${SLURM_CPUS_PER_TASK:-8}")
[ -n "${STAGE_DIR:-}" ] && ARGS+=(--stage-dir "$STAGE_DIR")
[ -n "${KEEP:-}" ] && ARGS+=(--keep-inputs)
"$REPO/.pixi/envs/fdp/bin/python" -u scripts/labeler/ae_fdp_label.py "${ARGS[@]}" 2>&1 \
    | grep -v -E "^Cannot open file: pelican://|XRD_PLUGINCONFDIR|warnings.warn"
echo "finished at $(date -Is)"
