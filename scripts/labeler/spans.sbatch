#!/bin/bash
#SBATCH --job-name=spans
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=22G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out

# One editor's span suggestions over the frozen cohort's non-blind shots:
#   python -m labeler.events.spans --event $EVENT --workers $WORKERS
# merges into $LABELER_ROOT/suggestions/<method>/v1/<event>_suggest_<method>_v1.csv; a
# shot already there is skipped. The corpus and the raw cache are only read. WORKERS
# is one a CPU unless set; more than that keeps the CPUs busy through slow reads.
#
#   pilot:      the full run's shape on the queue's first 20 shots, e.g.
#               EVENT=high_confinement_mode LIMIT=20 sbatch --cpus-per-task=4 --mem=3G <this>
#   full run:   EVENT=sawtooth_oscillation sbatch <this>
#               EVENT=high_confinement_mode sbatch --cpus-per-task=4 --mem=3G <this>
#               EVENT=edge_localized_mode sbatch --cpus-per-task=4 --mem=3G <this>
#               each with --mem reset to its pilot's MaxRSS x 1.2
#   extension:  EVENT=... WINDOWS=population sbatch --time=02:00:00 <this>
#   gate:       python -m labeler.jobstats --job-id <id> --cpu-only
# The tearing-mode editor's `window` method reads nothing: run it on the login node.
#
# ------------------------------------------ sizing (measured 2026-09-26, login node)
#   one worker, shot 202713: ece_sawtooth 3.7 s and 2.24 GB max RSS (the whole ECE
#   array is read); dalpha_lh 5.5 s and 0.50 GB; elm_clock, H-mode gate included,
#   4.2 s and 0.50 GB. Three 2024-2025 cohort shots read cold: ece_sawtooth 2.6-2.9 s
#   at 87 % CPU, dalpha_lh and elm_clock 0.7-1.0 s at 99-100 %.
#   cohort: 450 non-blind shots.
#   cpus-per-task=8   450 x 6 s / 8 workers = 5.6 min; the other two on 4 CPUs,
#                     450 x 1 s / 4 = 1.9 min, so start-up stays a small part of it.
#   mem=22G           8 x 2.24 GB x 1.2 for the sawteeth; 3G for the others (4 x 0.5 x 1.2).
#   time=00:30:00     5.6 min x 5 for cold GPFS reads of 600 MB of ECE a shot.
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub-round-two}"
ENV=/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker
ROOT="$("$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LABELER_ROOT="$ROOT" PYTHONPATH="$REPO/src"
export LABELER_LABEL_TABLES="${LABELER_LABEL_TABLES:-/scratch/gpfs/nc1514/FusionAIHub/data/events}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
: "${EVENT:?set EVENT to the editor, e.g. EVENT=sawtooth_oscillation}"
echo "job ${SLURM_JOB_ID:-none} $EVENT on $(hostname) at $(date -Is), root $ROOT"
srun "$ENV/bin/python" -m labeler.events.spans --event "$EVENT" \
    --windows "${WINDOWS:-cohort}" --workers "${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}" \
    ${LIMIT:+--limit "$LIMIT"}
echo "finished at $(date -Is)"
