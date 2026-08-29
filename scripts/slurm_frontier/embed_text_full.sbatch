#!/bin/bash
#SBATCH -A fus187
#SBATCH -J embed_text_full
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
# FULL sweep: embed every per-shot text bundle with Qwen3-Embedding-8B, one
# process per GCD (32 ranks total), self-sharded via SLURM_PROCID/SLURM_NTASKS
# (no torchrun/DDP). Default batch partition/QOS.
#
# RESUMABLE: each rank's shard file is skip-on-complete (a shard already fully
# written is left alone on resubmission), so a re-`sbatch` after a timeout just
# continues the unfinished shards. `--merge` is idempotent -- safe to rerun.
# IMPORTANT: resubmission MUST use the SAME node/task count (-N / --ntasks-per-node)
# as the original submission. The shard layout is shots[rank::world_size], so a
# DIFFERENT world size reshuffles which rank owns which shot and --merge will abort
# with a duplicate-shot assertion.
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

echo "[embed_text_full] host=$(hostname) nodes=${SLURM_JOB_NUM_NODES} world_size=${SLURM_NTASKS}"
srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_NTASKS" -c "$SLURM_CPUS_PER_TASK" \
     --gpus-per-task=1 --gpu-bind=closest \
     scripts/slurm_frontier/_srun_rank_wrapper.sh \
     scripts/data_preparation/embed_shot_text.py

echo "[embed_text_full] merge host=$(hostname)"
srun -N1 -n1 "${PYTHON}" -u scripts/data_preparation/embed_shot_text.py --merge

echo "[embed_text_full] verify host=$(hostname)"
set +e
srun -N1 -n1 "${PYTHON}" -u scripts/data_preparation/embed_shot_text.py --verify
VERIFY_RC=$?
set -e
echo "=== embed_text_full DONE (verify exit ${VERIFY_RC}) ==="
exit "${VERIFY_RC}"
