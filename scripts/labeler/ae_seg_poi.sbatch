#!/bin/bash
#SBATCH --job-name=ae-seg-poi
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=7500M
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out

# The segmentation model's points of interest, and a JPEG per shot:
#   python -m labeler.ae.seg.poi --workers 4                  every AE180 shot, from its store
#   python -m labeler.ae.seg.poi --from-corpus --shots ...    new shots, from the corpus's CO2
# writes $LABELER_ROOT/poi/alfven_eigenmode/ae_seg-v1/poi.csv (a shot drawn again replaces its
# rows) and $LABELER_ROOT/gallery/alfven_eigenmode/ae_seg-v1/<shot>.jpg. Nothing is fetched.
#
#   pilot:      SHOTS="170659 172011 170795 170719 172008 172012 172018 171999" sbatch <this>
#               At least 8 shots: two shots per worker, all four workers active.
#   full run:   sbatch --dependency=afterok:<the ae_seg_train job> <this>
#   new shots:  FROM_CORPUS=1 SHOTS="<shots>" sbatch <this>
#   v2:         VERSION=v2 sbatch --dependency=afterok:<the v2 ae_seg_train job> <this>
#               -> poi/alfven_eigenmode/ae_seg-v2 and gallery/alfven_eigenmode/ae_seg-v2
#   v3:         VERSION=v3 SHOTS="170659 172011 170795 170719 172008 172012 172018 171999"
#               sbatch <this> (pilot), then VERSION=v3 sbatch <this>
#               -> poi/alfven_eigenmode/ae_seg-v3 and gallery/alfven_eigenmode/ae_seg-v3
#   v4:         VERSION=v4 SHOTS="170659 172011 170795 170719 172008 172012 172018 171999"
#               sbatch <this> (pilot), then VERSION=v4 sbatch <this>: v1's AE180 shots over
#               60-250 kHz -> poi/alfven_eigenmode/ae_seg-v4 and gallery/alfven_eigenmode/ae_seg-v4
#   gate:       python -m labeler.jobstats --job-id <id> --cpu-only
#
# ------------------------------------------ sizing (measured 2026-09-29, v2's full run 2945169)
#   SegNet v2 over 0-250 kHz and whole records, 180 shots, 4 workers: 2 min 50 s, CPU 80.6 %;
#   sacct MaxRSS 6,342,728K, 99.9 % of 6200M (jobstats, sampling every 30 s, read 80.3 %).
#   mem=7500M     6194 MiB x 1.2 is 7433 MiB; rounded up.
#   v3 (an estimate from v2's run, not measured): the same network, band and shots, so the
#   same 6,342,728K at 7500M and ~2 min 50 s; check it on the v3 pilot's MaxRSS.
#   v4 (an estimate, not measured): v1's network, shots and store rows (v1's full run
#   2942540: 5,240,864K), so the header's 7500M holds it; check it on the v4 pilot's MaxRSS.
# before v2 (measured 2026-09-26, login node):
#   probe, 2 store shots, 2 workers: 8.7 s wall with ~4 s of start-up, largest process 1.08 GB;
#   1 corpus shot: 7.9 s, 1.37 GB.
#   mem=6200M     full run 2942540: sacct MaxRSS 5,240,864K at 5G, at the limit;
#                use × 1.2 headroom. Size from workers × one worker's peak plus the parent.
#                The 2-shot pilot under-predicted: 2700M ran out of memory (2942538).
#   time=00:30:00   ~180 shots x ~2 s / 4 workers = 90 s, x 20 for cold reads and start-up.
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
# SHOTS is left unquoted on purpose: it is a list of shot numbers.
# shellcheck disable=SC2086
srun "$ENV/bin/python" -m labeler.ae.seg.poi --version "$VERSION" \
    --workers "${SLURM_CPUS_PER_TASK:-4}" ${FROM_CORPUS:+--from-corpus} ${SHOTS:+--shots $SHOTS}
echo "finished at $(date -Is)"
