#!/bin/bash
#SBATCH --job-name=ae-xpower-posthoc
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2100M
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
# Memory: sacct MaxRSS 1823 MiB for the pilot (2943489, LIMIT=3) and 1771 MiB for the full
# run (2943497); 2100M is the larger plus about 15 %.

# The post-hoc record of a cross-validated version (v2), after its one test: computed
# from the same model, threshold and frames the test scored; no decision depends on
# it. It recomputes P(AE) from models/ae_xpower/$VERSION/<candidate>/model.pt for the
# split's test shots, with every check the test makes of its inputs, and refuses
# unless evaluation.json names that model, chosen.json, split, labels and source table
# (by sha256). It writes only models/ae_xpower/$VERSION/posthoc.{json,md}: (a) the MHD
# false-positive rate by place (lead-in, inside AE, after AE, shots with no present
# frame) for every method, with SELDNet differences; (b) the present and MHD shares in
# the out-of-fold and test frames; (c) the seed study (ae_xpower_cv.sbatch with SEED=),
# "not run" until one seed has all five folds; (d) the second look at v1's test shots.
# evaluation.* is never written; evaluate --test still refuses a second scoring. A
# pilot (LIMIT, the first N test shots, 20 or fewer) writes only to
# runs/ae_xpower/posthoc/$VERSION/. Run it again after the seed study for its (c).
#
#   pilot:      LIMIT=3 sbatch <this>   (exempt from the gate but reported)
#   full run:   sbatch <this>
#   after the study:  sbatch --dependency=afterok:<seed array ids> <this>
#   gate:       python -m labeler.jobstats --job-id <id> --cpu-only
#
# ------------------------------------------ sizing (measured 2026-09-28, E = job 2943080)
#   E, v2's evaluate --test on the same 60 test shots, every method, 1 thread: 5 min
#   8 s, sacct MaxRSS 2392 MiB. This loads the same model, SELDNet and frames, skips
#   the whole-window pass, and reads the 120 out-of-fold shots' saved frames (a few MB).
#   cpus=1         E's.
#   mem=2800M      2392 MiB plus 15 % is 2751 MiB; rounded up to 2800M.
#   time=00:30:00  E's 5 min 8 s, x 5.8.
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub-round-two}"
ENV=/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker
ROOT="$("$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LABELER_ROOT="$ROOT" PYTHONPATH="$REPO/src"
export LABELER_LABEL_TABLES="${LABELER_LABEL_TABLES:-/scratch/gpfs/nc1514/FusionAIHub/data/events}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
VERSION="${VERSION:-v2}"
echo "job ${SLURM_JOB_ID:-none} ($VERSION${LIMIT:+ limit $LIMIT}) on $(hostname) at $(date -Is), root $ROOT"
srun "$ENV/bin/python" -m labeler.ae.xpower.posthoc --version "$VERSION" \
    ${LIMIT:+--limit "$LIMIT"}
echo "finished at $(date -Is)"
