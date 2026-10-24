#!/bin/bash
#SBATCH --job-name=faith-process
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=360G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/gpfs/EKOLEMEN/D3D/runs/slurm/%A_%a.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/D3D/runs/slurm/%A_%a.out

# Staged FDP views -> raw <shot>.h5 -> FAITH corpus <shot>_processed.h5, on stellarai CPU nodes.
#
#   1. scripts/data_fetching_fdp/stage_shots_fdp.py (login node, network) fills $STAGE/<shot>/
#   2. this job: fetch_raw_fdp.py --stage-dir extracts each staged shot from local files
#      (no network; ~40 s/shot), then the unchanged prepare_data.py resamples it (~70 s/shot,
#      15 GB peak RSS on 185955, hence PREP_WORKERS=20 on a 370 GB node).
#
# Array task i takes every N-th pending shot (staged, not yet processed), so it can be
# resubmitted while staging is still running. The processed file is written into a
# per-task temporary directory and moved into place only when prepare_data reports success,
# so a killed job never leaves a truncated <shot>_processed.h5 that would be skipped later.
# The intermediate raw <shot>.h5 (~16 GB: every node keeps its own float64 time base) is
# deleted once processed, unless KEEP_RAW=1; it can be regenerated from the stage in ~40 s.
#
#   sbatch scripts/slurm_stellarai/process_staged.sbatch          # from the repo root
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
D3D="${D3D_ROOT:-/scratch/gpfs/EKOLEMEN/D3D}"
STAGE="${STAGE_DIR:-$D3D/fdp_stage}"
RAW="${RAW_DIR:-$D3D/d3d_time_series_data}"
OUT="${OUT_DIR:-$D3D/signals}"
SHOT_FILE="${SHOT_FILE:-$D3D/shot_lists/recommender_v1.txt}"
EXTRACT_WORKERS="${EXTRACT_WORKERS:-32}"
PREP_WORKERS="${PREP_WORKERS:-20}"
KEEP_RAW="${KEEP_RAW:-0}"
FDP_PY="$REPO/.pixi/envs/fdp/bin/python"
PREP_PY="$REPO/.pixi/envs/default/bin/python"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
NTASK="${SLURM_ARRAY_TASK_COUNT:-1}"
TAG="${SLURM_ARRAY_JOB_ID:-local}_${TASK}"
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset BEARER_TOKEN

cd "$REPO"
mkdir -p "$RAW" "$OUT" "$D3D/runs/process" "$D3D/runs/prepare"
LIST="$D3D/runs/process/$TAG.txt"
grep -v '^#' "$SHOT_FILE" | awk 'NF' | while read -r s; do
  if [ -d "$STAGE/$s" ] && [ ! -f "$OUT/${s}_processed.h5" ]; then echo "$s"; fi
done | awk -v n="$NTASK" -v i="$TASK" '(NR - 1) % n == i' > "$LIST"
N=$(wc -l < "$LIST")
echo "task $TASK/$NTASK on $(hostname) at $(date -Is): $N shots"
[ "$N" -eq 0 ] && exit 0

"$FDP_PY" -u scripts/data_fetching_fdp/fetch_raw_fdp.py --shot-file "$LIST" \
  --stage-dir "$STAGE" --out-dir "$RAW" --workers "$EXTRACT_WORKERS" 2>&1 \
  | grep -v "^Cannot open file: pelican://"   # libd3's index probe; harmless locally

SHOTS=$(while read -r s; do if [ -f "$RAW/$s.h5" ]; then echo "$s"; fi; done < "$LIST" | paste -sd, -)
[ -z "$SHOTS" ] && { echo "no shot extracted"; exit 1; }
TMP="$OUT/.tmp_$TAG"
mkdir -p "$TMP"
"$PREP_PY" -u scripts/data_preparation/prepare_data.py \
  "modalities.input_data_path=$RAW" "modalities.output_data_path=$TMP" \
  "modalities.num_workers=$PREP_WORKERS" "shot_list.shots=[$SHOTS]" \
  "hydra.run.dir=$D3D/runs/prepare/$TAG" > "$D3D/runs/prepare/$TAG.log" 2>&1 || true
grep -E "Done\.|ERROR" "$D3D/runs/prepare/$TAG.log" | tail -20 || true

# prepare_data logs a failed shot as "shot <n>: <error>"; anything it did not name is complete.
FAILED=$(grep -oE "shot [0-9]+: " "$D3D/runs/prepare/$TAG.log" | grep -oE "[0-9]+" | sort -u || true)
moved=0
for f in "$TMP"/*_processed.h5; do
  [ -e "$f" ] || continue
  s=$(basename "$f" _processed.h5)
  if grep -qx "$s" <<<"$FAILED"; then rm -f "$f"; continue; fi
  mv "$f" "$OUT/"; moved=$((moved + 1))
  [ "$KEEP_RAW" = 1 ] || rm -f "$RAW/$s.h5"
done
rmdir "$TMP" 2>/dev/null || true
echo "task $TASK done at $(date -Is): $moved processed, failed: ${FAILED:-none}"
