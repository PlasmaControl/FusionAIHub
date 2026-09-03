#!/bin/bash
#SBATCH -A fus187
#SBATCH -J build_cache
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 1
# Regenerate the full lengths + video-presence caches (CPU-only parallel scan) so
# the production run finds a WARM cache (avoids the ~33-min in-job rescan / NCCL
# watchdog). --use_video tangtv (the HDF5 group; the trainer maps the split
# tangtv_lower/upper -> "tangtv" so cameras_key matches). Matches the production
# file list via resolve_shot_files (all shots, val_fraction 0.1, seed 42).
FMH=/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub
cd "$FMH"
source scripts/slurm_frontier/_frontier_common.sh
export PYTHONPATH="$FMH/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
META=/lustre/orion/fus187/proj-shared/foundation_model_meta
# --use_video tangtv → video-present cache (4430/464). NO_VIDEO_FILTER=1 → skip the
# video filter → full ALL-shots list (7878/875), matching a --no_video_presence_filter run.
VIDEO_FILTER_ARG="--use_video ${BUILD_CACHE_CAMERAS:-tangtv}"
[ -n "${NO_VIDEO_FILTER:-}" ] && VIDEO_FILTER_ARG=""
echo "[build_cache] host=$(hostname) regenerating lengths caches -> $META  (video_filter='${VIDEO_FILTER_ARG:-<none/all-shots>}')"
python scripts/build_dataset_cache.py \
    --data_dir /lustre/orion/fus187/proj-shared/foundation_model \
    --val_fraction 0.1 --seed 42 \
    ${VIDEO_FILTER_ARG} \
    --cache_dir "$META" --video_cache_dir "$META" \
    --num_workers "${INDEXING_WORKERS:-56}"
echo "=== BUILD CACHE DONE (exit $?) ==="
