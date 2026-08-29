#!/bin/bash
#SBATCH -A fus187
#SBATCH -J ignite_skill_vs_step
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
# Rollout skill vs TRAINING STEP — the exposure-bias regression test (the Phase-1
# acceptance instrument). Rolls every distinct-step checkpoint of each run dir over the
# same shots with the same seed and plots token skill (accuracy minus persistence)
# against step; the documented bp128 pathology is that this curve INVERTS with more
# training, and a CTF / self-forcing arm must stop it inverting.
#
# GATE ON THE DECODED PER-BAND ROWS, not the token rows: token skill was MEASURED blind to
# the inversion on bp128 (flat +0.249 -> +0.259 where the decoded mhr TM band went
# +0.153 -> -0.862). A flat TOKEN curve is not evidence that an arm stopped inverting.
#
# SINGLE GCD on purpose: this is eval, one shot at a time, and the harness holds one
# model at a time. Keep -t <= 02:00:00 — longer walltimes are rejected here.
#
# Env overrides:
#   RUN_DIRS  space-separated checkpoint dirs -> one repeated --run_dir per entry (the arms)
#   CACHE_DIR frame-code cache the shots are read from (MUST match the arms' tokenizer)
#   SHOTS     comma-separated shot list
#   OUT_DIR   destination for skill_vs_step.json + skill_vs_step.png
#   FLAGS     extra harness flags, e.g. "--global_pool --top_p 0.9 --best_of_n 4"
#             (decoded metrics need --bp_root/--codec_root/--data_dir; the harness defaults
#              already point at the bp128 edges, the current codecs and additional_data)
#
# Submit from the repo root:
#   sbatch scripts/slurm_frontier/ignite_skill_vs_step.sh
#   RUN_DIRS="/path/base /path/ctf" OUT_DIR=data/outputs/ignite_skill_vs_step/phase1 \
#     sbatch scripts/slurm_frontier/ignite_skill_vs_step.sh
set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
source scripts/slurm_frontier/_frontier_settings.sh

BP_ROOT="/lustre/orion/fus187/proj-shared/models/ignite_bp128"
RUN_DIRS="${RUN_DIRS:-${BP_ROOT}/runs/bp128_d512L8}"
# NOTE the default cache is the EXTENDED bp128 cache, not ${BP_ROOT}/frame_codes: the
# canonical cache was built before 199597 was encoded and does NOT contain it (the shot the
# documented +0.153 -> -0.862 inversion was measured on). This is the DURABLE proj-shared
# copy -- the original lives in data/outputs/, which is purge-by-atime scratch, and the
# headline regression test must not depend on that. Rebuild recipe + bin-edges
# compatibility record: PROVENANCE.txt in that directory, and
# scripts/evaluation/ignite_build_bp_cache.py.
CACHE_DIR="${CACHE_DIR:-/lustre/orion/fus187/proj-shared/nchen/ignite_bp128_frame_codes_ext}"
SHOTS="${SHOTS:-199597}"
OUT_DIR="${OUT_DIR:-data/outputs/ignite_skill_vs_step/baseline}"
FLAGS="${FLAGS:-}"
mkdir -p logs "${OUT_DIR}"

ARMS=()
for d in ${RUN_DIRS}; do ARMS+=(--run_dir "${d}"); done

export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
echo "[ignite_skill_vs_step] host=$(hostname) runs=${RUN_DIRS} cache=${CACHE_DIR} \
shots=${SHOTS} out=${OUT_DIR} flags=${FLAGS:-none}"
srun -N 1 -n 1 -c "${SLURM_CPUS_PER_TASK:-7}" --gpus-per-task=1 --gpu-bind=closest \
     python -u scripts/evaluation/ignite_skill_vs_step.py \
     "${ARMS[@]}" \
     --cache_dir "${CACHE_DIR}" \
     --shots "${SHOTS}" \
     --out_dir "${OUT_DIR}" ${FLAGS}
