#!/bin/bash
#SBATCH --job-name=review-build
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out

# One editor's review rows for every roster shot that has none, ahead of the review:
#   python -m labeler.events.review.build --event $EVENT --workers $WORKERS
# writes $LABELER_ROOT/spectrograms/<event>/<shot>.h5, which the page otherwise builds
# on first open. Nothing is fetched here: fill the raw cache on the login node first,
#   python -m labeler.events.raw --event edge_localized_mode   (and high_confinement_mode)
# or the ELM D-alpha is FS01 and old shots have no CO2. Exits 1 if any shot failed.
# WORKERS is one a CPU unless set; more keep the CPUs busy through slow reads.
#
#   pilot:      the first 20 roster shots in the full run's shape, e.g.
#               R=/scratch/gpfs/nc1514/FusionAIHub/data/events/neoclassical_tearing_mode
#               SHOTS="$(sed -n 2,21p $R/shots.csv | cut -d, -f1 | xargs)" \
#                 EVENT=neoclassical_tearing_mode sbatch <this>
#   full run:   EVENT=neoclassical_tearing_mode sbatch <this>
#               WORKERS=12 EVENT=sawtooth_oscillation SHOTS="<the cohort's 450>" \
#                 sbatch --cpus-per-task=4 --mem=6G <this>
#               WORKERS=6 EVENT=high_confinement_mode sbatch --cpus-per-task=2 --mem=3G <this>
#               EVENT=edge_localized_mode sbatch --cpus-per-task=4 --mem=2G <this>
#               each with --mem reset to its pilot's MaxRSS x 1.2
#   rebuild:    EVENT=neoclassical_tearing_mode FORCE=1 sbatch <this>  (--force: rebuilds built shots)
#               after a panel change, e.g. the ELM D-alpha rows and the sawtooth SXR chords:
#               EVENT=edge_localized_mode FORCE=1 sbatch --cpus-per-task=4 --mem=2G <this>
#               WORKERS=12 EVENT=sawtooth_oscillation FORCE=1 SHOTS="<the cohort's 450>" \
#                 sbatch --cpus-per-task=4 --mem=6G <this>
#               a store is only the page's rows: labels, drafts and history are untouched
#   gate:       python -m labeler.jobstats --job-id <id> --cpu-only
#
# ------------------------------------------ sizing (measured 2026-09-26, login node)
#   one shot (202713), one worker, fetching off: tearing modes 5.1 s and 1.12 GB max
#   RSS, sawteeth 2.6 s and 0.51 GB, ELMs 1.2 s and 0.35 GB, H-mode 0.5 s and 0.32 GB.
#   Sawteeth since the ECE medians (2026-09-28, pilot job 2943760, 20 shots): 12
#   workers on 4 CPUs, 30 s, 69 % CPU with start-up, 4.92 GB MaxRSS (0.41 GB a
#   worker), so --mem=6G; 8 workers on 4 CPUs reached 55 %. With the Thomson Te row
#   (pilot job 2943818): 26 s, 83 % CPU by sacct, 5.23 GB MaxRSS, still --mem=6G;
#   the full rebuild (job 2943840, 450 shots) 7 min 51 s, 89.7 % CPU, 85 % of 6G.
#   Tearing modes since the n-coloured row (2026-09-28, pilot job 2943811, 20 shots):
#   8 workers on 8 CPUs, 28 s, 40 % CPU (start-up and a half-empty last round are most
#   of 20 shots; each shot is 5 s), 7.19 GB MaxRSS (0.90 GB a worker). The full
#   rebuild (job 2943826, 450 shots): 4 min 51 s, 72.9 % CPU, and 5.7 GiB by jobstats
#   (63 % of 9G) against sacct's 6.96 GiB, which sums the forked workers' shared
#   pages, so --mem=8G.
#   Two 2024 cohort shots read cold: tearing modes 86-90 % CPU and ELMs 99 %, but
#   sawteeth 56 % and H-mode 43-48 %, waiting on GPFS: two or three workers a CPU.
#   roster: 450 cohort shots (plus the 10 gold sawtooth shots, outside the corpus).
#   cpus-per-task=8   450 x 5.1 s / 8 workers = 4.8 min; the others 1-2 min on 2-4 CPUs.
#   mem=8G            tearing modes' 5.7 GiB by jobstats x 1.4; sawteeth 12 x 0.41 x 1.2,
#                     H-mode 6 x 0.32 x 1.2 and ELMs 4 x 0.35 x 1.2, rounded up.
#   time=00:30:00     4.8 min x 6 for cold GPFS reads.
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
ENV=/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker
ROOT="$("$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LABELER_ROOT="$ROOT" PYTHONPATH="$REPO/src"
export LABELER_LABEL_TABLES="${LABELER_LABEL_TABLES:-/scratch/gpfs/nc1514/FusionAIHub/data/events}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE LABELER_NO_FETCH=1
cd "$REPO"
: "${EVENT:?set EVENT to the editor, e.g. EVENT=neoclassical_tearing_mode}"
echo "job ${SLURM_JOB_ID:-none} $EVENT on $(hostname) at $(date -Is), root $ROOT"
srun "$ENV/bin/python" -m labeler.events.review.build --event "$EVENT" \
    --workers "${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}" ${SHOTS:+--shots $SHOTS} \
    ${FORCE:+--force}
echo "finished at $(date -Is)"
