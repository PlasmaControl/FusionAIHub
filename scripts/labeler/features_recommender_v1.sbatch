#!/bin/bash
#SBATCH --job-name=lm-features-rv1
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out

# labeler `features` for the recommender_v1 shots that have no
# <shot>_features.h5 yet (the list `shot_design corpus select --verify-flattop`
# writes as recommender_v1_pending_features.txt). Once these exist,
# `shot_design corpus select --finalize` verifies rule (d) (Ip flat-top >= 1 s)
# on measured `ip` instead of the pulse-length proxy.
#
# ---------------------------------------------------------------- sizing (measured 2026-09-07, login node)
#   1 worker,  5 shots: 23.7 s wall (4.7 s/shot), 131 % of one cpu, process tree 2.65 GB
#   4 workers, 12 shots: 19.8 s wall (1.65 s/shot), 273 % of one cpu, process tree 5.3 GB
#   -> ~0.68 cpu and ~1.1 GB per worker + ~1 GB main; EFIT resolves from the archive (no fdp)
#   8 workers: ~5.4 cpu -> --cpus-per-task=6 (~90 % util); ~10 GB -> --mem=12G (~85 %);
#   352 shots * ~0.85 s = ~5 min -> --time=00:20:00.
#   The pixi launcher (~1 GB RSS) is bypassed: the env's python is called directly with the
#   activation variable the fdp feature needs (LD_LIBRARY_PATH for GLIBCXX_3.4.29).

set -euo pipefail
REPO=/scratch/gpfs/nc1514/FusionAIHub
ROOT="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
ENV="$REPO/.pixi/envs/labelmaker"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LABELER_ROOT="$ROOT" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -Is)"
srun "$ENV/bin/python" -m labeler.run features \
    --models d3d_tearing_onset_cnn1d d3d_tearing_time_to_event_dsm d3d_elm_time_to_event_dsm d3d_ae_activity_seldnet \
    --shot-file "${SHOT_FILE:-$ROOT/recommender_v1_pending_features.txt}" \
    --workers 8 --timeout 600
echo "finished at $(date -Is)"
