#!/bin/bash
#SBATCH --job-name=ae-seg-train
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out

# The AE segmentation model, then its one scoring on the test shots:
#   python -m labeler.ae.seg.train     the reviewed pseudo-masks
#       ($LABELER_ROOT/segmentation/alfven_eigenmode/pseudo-v1, built on the login node by
#       `python -m labeler.ae.seg.pseudo`), the owner's mask decisions
#       ($LABELER_LABEL_TABLES/alfven_eigenmode/review/masks.jsonl, if there are any), the review
#       store's AE rows and the chosen ae_xpower model's split -> $LABELER_ROOT/models/ae_seg/v1/
#       {model.pt, split.csv, training.json}
#   python -m labeler.ae.seg.evaluate  the test shots, against the recipe and TokEye baselines
#       -> evaluation.{json,md} beside the model
# Nothing is fetched. CPU only: SegNet has ~0.28 M parameters, and a GPU would idle far below
# the 70 % gate.
#
#   pilot:      PILOT=12 sbatch <this>   (12 shots, 2 epochs, into runs/ae_seg/pilot; 3 test shots)
#   v2:         VERSION=v2 PILOT=12 sbatch <this>, then VERSION=v2 sbatch --mem=<from that
#               pilot> <this>: pseudo-v2 over 0-250 kHz, v3's snapshot and test shots, into
#               models/ae_seg/v2 (a pilot into runs/ae_seg/pilot-v2), scored over the owner's
#               whole windows (round three, §2.5)
#   v3:         VERSION=v3 PILOT=12 sbatch --mem=4G <this>, then VERSION=v3 sbatch
#               --mem=2100M --time=01:30:00 <this>: pseudo-v3 (built on the login node by
#               `python -m labeler.ae.seg.pseudo --version v3`, refused unless its rules.json
#               gate passed), v2's split and test shots, into models/ae_seg/v3 (a pilot
#               into runs/ae_seg/pilot-v3); its test is a second use of v2's 60 test shots
#               (S2 fix)
#   v4:         VERSION=v4 PILOT=12 sbatch --mem=4G <this>, then VERSION=v4 sbatch --mem=2G
#               <this>: pseudo-v4 (SegNet v1 with the band at 60-250 kHz, built on the login
#               node by `python -m labeler.ae.seg.pseudo --version v4`), the live labels and
#               v1's split (refused unless it is models/ae_seg/v1/split.csv), into
#               models/ae_seg/v4 (a pilot into runs/ae_seg/pilot-v4), scored over 0-2 s; its
#               test is a second use of v1's test shots
#   full run:   sbatch --mem=<from the pilot> <this>
#   gate:       python -m labeler.jobstats --job-id <id> --cpu-only
#
# ------------------------------------------ sizing (measured 2026-09-26, login node, 4 threads)
#   pilot, 8 train + 4 val shots, 2 epochs: 5.3-5.8 s an epoch, 15.5 s wall, 335 % of 4 threads
#   (84 %), max RSS 1.39 GB.
#   probe, 16 train + 4 val shots, 12 epochs: ~120 s; validation Dice 0.90 at its threshold.
#   88 train shots x 8 crops = 88 batches of 8, ~48 s an epoch; 40 epochs at most: ~32 min
#   (patience 8 usually stops it sooner). evaluate, 3 test shots: 5.3 s, 1.18 GB, 258 %; the 58
#   test shots take ~2 min.
#   mem=2G         1.39 GB x 1.25, plus ~0.1 GB for 104 shots' inputs; reset it from the pilot's
#                  training.json `peak_rss_mb` x 1.25, so jobstats, which samples every 30 s,
#                  still sees 70 % or more.
#   time=01:30:00  32 min x 2.8.
# v3 (estimates from SegNet v2's full run, not measured on v3): v2 trained and scored 104 + 16
#   shots and 60 test shots in ~32 min at --mem=2100M, max RSS 1.88 GiB; pseudo-v3 has v2's
#   grid, shots and split, so v3 should need the same: --mem=2100M, --time=01:30:00. The pilot
#   runs at 4G so it cannot run out; reset the full run from its training.json `peak_rss_mb`.
# v4 (an estimate, not measured): v1's shots, split and inputs (the store's 257 bins); only the
#   scored band is wider, 60-250 kHz, so v1's 2G and 01:30:00; the pilot at 4G, and reset the
#   full run from its training.json `peak_rss_mb` x 1.25.
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
ENV=/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker
ROOT="$("$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LABELER_ROOT="$ROOT" PYTHONPATH="$REPO/src"
export LABELER_LABEL_TABLES="${LABELER_LABEL_TABLES:-/scratch/gpfs/nc1514/FusionAIHub/data/events}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE LABELER_NO_FETCH=1
cd "$REPO"
VERSION="${VERSION:-v1}"
echo "job ${SLURM_JOB_ID:-none} ($VERSION) on $(hostname) at $(date -Is), root $ROOT"
MODELS="$ROOT/models/ae_seg/$VERSION"
PILOT_DIR="$ROOT/runs/ae_seg/pilot"
if [ "$VERSION" != v1 ]; then PILOT_DIR="$PILOT_DIR-$VERSION"; fi
if [ -n "${PILOT:-}" ]; then MODELS="$PILOT_DIR"; fi
srun "$ENV/bin/python" -m labeler.ae.seg.train --version "$VERSION" ${PILOT:+--pilot "$PILOT"}
srun "$ENV/bin/python" -m labeler.ae.seg.evaluate --version "$VERSION" --models "$MODELS" \
    ${PILOT:+--limit 3}
echo "finished at $(date -Is)"
