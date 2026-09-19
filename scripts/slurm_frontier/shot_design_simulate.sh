#!/bin/bash
#SBATCH -A fus187
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -J sd-simulate
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH -c 7
#SBATCH -t 01:00:00
#SBATCH --output=/lustre/orion/fus187/proj-shared/nchen/shot_design/runs/slurm/%j.out
# STUB. The IGNITE what-if rollout job. Its header is fixed here (one GCD, debug QOS --
# a demo-length job) so the house-style check covers it from the start; the body is
# Task D4's.
source "$(dirname "$0")/_shot_design_common.sh"
echo "filled in by Task D4"
exit 1
