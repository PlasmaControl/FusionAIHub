#!/bin/bash
# Environment hook for Princeton StellarAI (NVIDIA B200), sourced by
# scripts/slurm_frontier/train_dynamics.sh through CLUSTER_ENV.
#
# The `stellarai` pixi environment (pyproject.toml) carries torch 2.11 + CUDA 12.8 wheels,
# which bundle the CUDA runtime and NCCL, so no module load is needed. Install it once on
# a login node:   pixi install -e stellarai
# Like Frontier, the hook runs --frozen under a lock so concurrent jobs never re-resolve.
_STELLARAI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_STELLARAI_DIR}/../.." && pwd)"
export PATH="$HOME/.pixi/bin:$PATH"
_PIXI_LOCK="${_REPO_ROOT}/.pixi/.shell-hook.lock"
mkdir -p "$(dirname "${_PIXI_LOCK}")"
_hook_out="$(flock -w 300 "${_PIXI_LOCK}" \
    pixi shell-hook -e stellarai --frozen --manifest-path "${_REPO_ROOT}/pyproject.toml")"
eval "${_hook_out}"

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
if [ -n "${SLURM_NODELIST:-}" ]; then
    MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)"
else
    MASTER_ADDR="127.0.0.1"
fi
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29500}"
# GPU binding for train_dynamics.sh and eval_dynamics.sh. --gpu-bind=closest (the Frontier default) exposes the 4 GPUs
# of a task's socket here, so local ranks 4-7 hit "invalid device ordinal" (job 2855); binding one
# GPU per task (map_gpu) gets through set_device but NCCL's first all-reduce then fails with the
# same error (job 2864). Expose all 8 GPUs to every rank; _DDPState picks cuda:LOCAL_RANK.
export SRUN_GPU_BIND="${SRUN_GPU_BIND:-none}"
