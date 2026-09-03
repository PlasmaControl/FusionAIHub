#!/bin/bash
#SBATCH -A fus187
#SBATCH -J video_recon
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gres=gpu:1
# Video encoder/decoder reconstruction test at the FIXED budget (300 tokens @
# d_model 1024). Variants: deconv | resize | flow | flow_nope. All args pass
# through, e.g.:
#   sbatch scripts/slurm_frontier/test_video_recon.sbatch \
#     --variants deconv,resize,flow,flow_nope --base_ch 64 --steps 4000 \
#     --flow_steps 12 --out_dir eval_runs/video_test/sweep1
FMH=/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub
cd "$FMH"
source scripts/slurm_frontier/_frontier_common.sh
export PYTHONPATH="$FMH/src${PYTHONPATH:+:$PYTHONPATH}"
echo "[video_recon] host=$(hostname) args=$*"
python scripts/training/test_video_reconstruction.py "$@"
echo "=== VIDEO RECON TEST DONE (exit $?) ==="
