#!/bin/bash
#SBATCH -A fus187
#SBATCH -J scan_vidchan
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 1
# Per-channel tangtv liveness scan → per-divertor valid shot lists.
FMH=/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub
cd "$FMH"
source scripts/slurm_frontier/_frontier_common.sh
export PYTHONPATH="$FMH/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
echo "[scan_vidchan] host=$(hostname)"
python scripts/data_preparation/scan_video_channels.py --workers "${WORKERS:-56}"
echo "=== VIDEO CHANNEL SCAN DONE (exit $?) ==="
