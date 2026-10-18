#!/bin/bash
#SBATCH -A fus187
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
# =====================================================================================
# ONE rollout curve point on the FIXED 12-shot protocol (do not change the shot list mid-curve):
#   3 shots pinned into train+val (190729/190735/190736) + 9 held-out; ALL 15 modalities
#   (eval_dynamics' default 11-name list silently drops cer_rot/filterscopes/mirnov/mse);
#   K0=20 seed frames, 80 predicted frames, T=1.0, dynamics_latest.pt (never best.pt).
# Shards the 12 shots over 8 GCDs (~50 min). The npz archive records the checkpoint step, and
# OUT_DIR carries the job id so successive points never mix.
#   sbatch -J roll15_ss75full -N 1 --ntasks-per-node=8 --cpus-per-task=7 --gres=gpu:8 \
#          -t 02:00:00 -p batch --export=ALL,RUN=ss75full scripts/slurm_frontier/rollout15_examples.sh
#   then: scontrol update JobId=<id> Partition=extended,batch   (g1's QOS node cap is used by trainers)
# Score (static GT pairs screened; near-1 vs persistence + motion ratio + timing corr, all three):
#   python analysis/rollout_npz_score.py --split 190729,190735,190736 <OUT_DIR ...>
# Overrides: CKPT (path), TEMPERATURE, SHOT.
# =====================================================================================
set -uo pipefail
# REPO_DIR: repository root (other clusters set it, see scripts/slurm_stellarai/).
cd "${REPO_DIR:-/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub}"
RUN=${RUN:-ss75full}
export IGNITE_EVAL_MODALITIES="bes,cer_rot,cer_ti,co2,ece,filterscopes,mhr,mirnov,mse,tangtv_lower,tangtv_upper,ts_core_density,ts_core_temp,ts_tangential_density,ts_tangential_temp"
export SHOT=${SHOT:-"190729,190735,190736,192238,203727,192042,200328,190872,203860,192945,193650,203213"}
export CKPT=${CKPT:-/lustre/orion/fus187/proj-shared/models/ignite_prod_v4/runs/${RUN}/dynamics_latest.pt}
export OUT_DIR=${OUT_DIR:-/lustre/orion/fus187/proj-shared/models/ignite_prod_v4/runs/${RUN}/rollout15_examples_${SLURM_JOB_ID}}
export CACHE_DIR=${CACHE_DIR:-/lustre/orion/fus187/proj-shared/models/ignite_prod_v4/frame_codes}
export SPLIT_SEED=42 VAL_N=438 TEMPERATURE=${TEMPERATURE:-1.0}
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-7}
exec bash scripts/slurm_frontier/eval_dynamics.sh
