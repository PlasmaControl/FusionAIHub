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
# merges into $LABELER_ROOT/suggestions/<method>/<v>/<event>_suggest_<method>_<v>.csv,
# <v> VERSION (`--version`), v1 unless set: v1's tables predate the dead-filterscope
# and ramp-up rules, and v2's, written beside them, have them. A table whose meta
# records other rules than the method's now is refused, never rewritten: name a new
# VERSION. A shot already in the table is skipped unless FORCE is set. The corpus
# and the raw cache are only read (Ip too: the ELM and sawtooth drafts start where
# it reaches 0.8 of its flat-top). WORKERS is one a CPU unless set; more keeps the
# CPUs busy through slow reads. SHOTS names shots; LIMIT takes the queue's first N.
# GOLD=1 also scores the drafts on the event's gold-tier roster shots (`--gold`) and
# writes the score into the table's meta ("gold"); a later run without it keeps the
# score. FORCE changes what a page open on the table shows for unsaved shots; saved
# labels are never touched.
#
#   pilot:      the full run's shape on the queue's first 20 shots, e.g.
#               EVENT=high_confinement_mode VERSION=v2 LIMIT=20 \
#                 sbatch --cpus-per-task=4 --mem=3G <this>
#   full run:   the current rules in their own tables, v1's untouched (a v1 table
#               refuses a rerun under them), e.g.
#               EVENT=sawtooth_oscillation VERSION=v2 GOLD=1 \
#                 sbatch --cpus-per-task=4 --mem=9G --time=01:00:00 <this>
#               EVENT=high_confinement_mode VERSION=v2 \
#                 sbatch --cpus-per-task=4 --mem=3300M <this>
#               EVENT=edge_localized_mode VERSION=v2 GOLD=1 \
#                 sbatch --cpus-per-task=4 --mem=3400M <this>
#               each with --mem reset to its pilot's MaxRSS x 1.2
#   redo:       every shot of a version again, its rules unchanged, with the gold
#               score, e.g.
#               EVENT=sawtooth_oscillation VERSION=v2 FORCE=1 GOLD=1 sbatch <this>
#               EVENT=edge_localized_mode VERSION=v2 FORCE=1 GOLD=1 \
#                 sbatch --cpus-per-task=4 --mem=3400M <this>
#               then rebuild the stores of a page open on them (review_build.sbatch,
#               FORCE=1)
#   extension:  EVENT=... VERSION=v2 WINDOWS=population sbatch --time=02:00:00 <this>
#   gate:       python -m labeler.jobstats --job-id <id> --cpu-only
# The tearing-mode editor's `window` method reads nothing: run it on the login node.
#
# ------------------------------------------ sizing (measured 2026-09-26, login node)
#   one worker, shot 202713: ece_sawtooth 3.7 s and 2.24 GB max RSS (the whole ECE
#   array is read); dalpha_lh 5.5 s and 0.50 GB; elm_clock, H-mode gate included,
#   4.2 s and 0.50 GB. Three 2024-2025 cohort shots read cold: ece_sawtooth 2.6-2.9 s
#   at 87 % CPU, dalpha_lh and elm_clock 0.7-1.0 s at 99-100 %.
#   cohort: 450 non-blind shots.
#   2026-09-27, with the Ip start rule, shot 189061: ece_sawtooth 2.3 s and 2.24 GB,
#   elm_clock 3.0 s and 0.96 GB (0.95 GB without the Ip read: the shot's own inputs).
#   cpus-per-task=8   450 x 6 s / 8 workers = 5.6 min; the other two on 4 CPUs,
#                     450 x 1 s / 4 = 1.9 min, so start-up stays a small part of it.
#   mem=22G           8 x 2.24 GB x 1.2 for the sawtooth; 3G for H-mode (4 x 0.5 x 1.2),
#                     5G for ELMs (4 x 0.96 x 1.2 on 189061).
#   time=00:30:00     5.6 min x 5 for cold GPFS reads of 600 MB of ECE a shot.
#   2026-09-29, v2 (round three), 4 workers on 4 CPUs. Pilots (LIMIT=20): ELM job
#   2944528 47 s, 2.42 GB MaxRSS; H-mode 2944529 47 s, 2.39 GB; sawtooth 2944530 19 s,
#   7.91 GB at 11G (4 x 2.24 GB does not fit 5G). Full runs, 450 shots: ELM 2944539
#   8 min 43 s, 85 % CPU, 2.70 GiB of 3G; H-mode 2944541 8 min 43 s, 85 %, 2.67 GiB of
#   3G; sawtooth 2944540 6 min 9 s, 80 %, 7.45 GiB of 10G. sacct's MaxRSS x 1.2,
#   rounded up, gives the sizes above: ELMs 3400M (2944539, 2,831,572K x 1.2 =
#   3318 MiB), H-mode 3300M (2944541, 2,800,852K x 1.2 = 3282 MiB), sawtooth 9G
#   (2944540, 7,807,552K x 1.2 = 8.93 GiB). jobstats, which samples every 30 s,
#   read 46-57 % of these peaks.
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
ENV=/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker
ROOT="$("$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LABELER_ROOT="$ROOT" PYTHONPATH="$REPO/src"
export LABELER_LABEL_TABLES="${LABELER_LABEL_TABLES:-/scratch/gpfs/nc1514/FusionAIHub/data/events}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE LABELER_NO_FETCH=1
cd "$REPO"
: "${EVENT:?set EVENT to the editor, e.g. EVENT=sawtooth_oscillation}"
echo "job ${SLURM_JOB_ID:-none} $EVENT ${VERSION:-v1} on $(hostname)" \
    "at $(date -Is), root $ROOT"
srun "$ENV/bin/python" -m labeler.events.spans --event "$EVENT" \
    --windows "${WINDOWS:-cohort}" --workers "${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}" \
    ${LIMIT:+--limit "$LIMIT"} ${SHOTS:+--shots $SHOTS} ${FORCE:+--force} \
    ${GOLD:+--gold} ${VERSION:+--version "$VERSION"}
echo "finished at $(date -Is)"
