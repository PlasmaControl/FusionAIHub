#!/bin/bash
#SBATCH -A fus187
#SBATCH -J embed_text_pilot
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -t 00:30:00
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
# PILOT: embed a small --limit 40 shots with Qwen3-Embedding-8B, one process per
# GCD, self-sharded via SLURM_PROCID/SLURM_NTASKS (no torchrun/DDP -- each rank
# embeds an independent shot slice and writes its own shard file). Sanity-checks
# the pipeline end to end (embed -> merge -> verify) before the full sweep.
#
# Compute nodes are OFFLINE. Weights must already be cached under HF_HOME below;
# pre-download from a LOGIN node first, e.g.:
#   HF_HOME=/lustre/orion/fus187/proj-shared/foundation_model/hf_cache \
#     python -c "from transformers import AutoModel, AutoTokenizer; \
#     AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-8B'); \
#     AutoModel.from_pretrained('Qwen/Qwen3-Embedding-8B')"
set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
source scripts/slurm_frontier/_frontier_settings.sh
mkdir -p logs

export HF_HOME=/lustre/orion/fus187/proj-shared/foundation_model/hf_cache
export HF_HUB_OFFLINE=1

PYTHON="${PROJECT_DIR}/.pixi/envs/frontier/bin/python"

OUT="/lustre/orion/fus187/scratch/nchen/text_embed_pilot/text_embeddings_pilot.h5"
SHARD_DIR="/lustre/orion/fus187/scratch/nchen/text_embed_pilot/text_embeddings_pilot_shards"
mkdir -p "$(dirname "${OUT}")"

echo "[embed_text_pilot] host=$(hostname) nodes=${SLURM_JOB_NUM_NODES} world_size=${SLURM_NTASKS} out=${OUT}"
srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_NTASKS" -c "$SLURM_CPUS_PER_TASK" \
     --gpus-per-task=1 --gpu-bind=closest \
     scripts/slurm_frontier/_srun_rank_wrapper.sh \
     scripts/data_preparation/embed_shot_text.py \
     --out "${OUT}" --shard_dir "${SHARD_DIR}" --limit 40

echo "[embed_text_pilot] merge host=$(hostname)"
srun -N1 -n1 "${PYTHON}" -u scripts/data_preparation/embed_shot_text.py \
     --out "${OUT}" --shard_dir "${SHARD_DIR}" --merge

echo "[embed_text_pilot] verify host=$(hostname)"
set +e
srun -N1 -n1 "${PYTHON}" -u scripts/data_preparation/embed_shot_text.py \
     --out "${OUT}" --shard_dir "${SHARD_DIR}" --verify
VERIFY_RC=$?
set -e
echo "=== embed_text_pilot DONE (verify exit ${VERIFY_RC}) ==="
exit "${VERIFY_RC}"
