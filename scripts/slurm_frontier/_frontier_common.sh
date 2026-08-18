# Frontier-common environment for ROCm DDP jobs.
# Source from every Frontier SLURM script BEFORE activating the venv.
# Sets modules, RCCL/NCCL knobs, MIOpen cache, and MASTER_ADDR/PORT.
#
# Frontier hardware reminders (see docs.olcf.ornl.gov):
#   - 4x MI250X = 8 GCDs per node, each appears as a separate GPU.
#   - HSN is Slingshot via libfabric/cxi; RCCL needs hsn0 + kdreg2.
#   - MIOpen cache in $HOME is slow & contended; redirect to /tmp.

# shellcheck shell=bash

module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load rocm/7.1.1
module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"

# Pixi env activation (replaces the old conda env). One-time setup:
#   pixi install -e frontier
# Each SLURM script then sources this file to get the env on PATH.
export PATH="$HOME/.pixi/bin:$PATH"
# Resolve manifest relative to this script so the file works for any clone of the repo.
_FRONTIER_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_FRONTIER_REPO_ROOT="$(cd "${_FRONTIER_COMMON_DIR}/../.." && pwd)"
# --frozen trusts pixi.lock and skips the metadata refresh that otherwise
# hits the rattler cache on lustre. Without it, concurrent SLURM jobs race
# for `.cache/rattler/cache/repodata/*.shards-cache-v1` locks and some
# fail to activate the env (rank wrapper then hits `python: not found`,
# exit 127). See logs/4613942 + 4614164 .err. All required packages are
# already installed on disk under .pixi/envs/frontier/, so the refresh
# adds no value at job-runtime.
# shellcheck disable=SC1091,SC2046
# SERIALIZE the shell-hook across concurrently-starting jobs. 2026-08-17: launching 5-7 jobs
# within seconds of each other had each of them run `pixi shell-hook` against the same
# Lustre-backed env; they raced and STRIPPED FILES FROM 62 PACKAGES (torch/bin emptied ->
# `import torch` died with "Unable to find torch_shm_manager", every job failed). --frozen alone
# is not enough. flock makes the hook mutually exclusive; it is read-only in the normal case so
# the lock is held only briefly.
_PIXI_LOCK="${_FRONTIER_REPO_ROOT}/.pixi/.shell-hook.lock"
mkdir -p "$(dirname "${_PIXI_LOCK}")"
_hook_out="$(flock -w 300 "${_PIXI_LOCK}" \
    pixi shell-hook -e frontier --frozen --manifest-path "${_FRONTIER_REPO_ROOT}/pyproject.toml")"
eval "${_hook_out}"

# AWS-OFI-NCCL plugin (built at ~/aws-ofi-nccl/install; sources from
# github.com/aws/aws-ofi-nccl). Routes NCCL/RCCL collectives through
# libfabric/cxi over Slingshot HSN instead of TCP sockets. Validated by
# smoke 4615534: plugin loaded as v10, NET/OFI selected provider=cxi,
# 4 NICs per rank, libfabric 1.22, HIP 7.1. See memory/project-aws-ofi-rccl-plugin.md.
# Must come AFTER `pixi shell-hook` because the hook overwrites LD_LIBRARY_PATH.
# DISABLED 2026-05-27 for post-maintenance NCCL hang diagnosis (jobs
# 4700720/21 d=256 ALLREDUCE timeout, 4700730/31 d=1024 BROADCAST timeout —
# all same code as 4642538 that ran 10h fine 2026-05-25).
# export LD_LIBRARY_PATH="$HOME/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH"

# Performance / correctness knobs
export PYTORCH_ROCM_ARCH=gfx90a
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HSA_FORCE_FINE_GRAIN_PCIE=1

# RCCL over Slingshot HSN
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_NET_GDR_LEVEL=3
export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_DEFAULT_CQ_SIZE=131072

# NCCL collective-timeout diagnostics. Stage-1 chained job 4581029 died at
# 04h46m elapsed when one rank stopped participating in a BROADCAST and
# the 10-minute watchdog timeout terminated all 64 ranks. The visible
# error message recommended FlightRecorder for stack traces, but it was
# disabled — so we never learned which rank stalled. Enable both knobs
# so next time we get the culprit rank.
#   - TORCH_FR_BUFFER_SIZE: per-rank ring buffer of recent collective
#     ops (new name; TORCH_NCCL_TRACE_BUFFER_SIZE is the deprecated
#     alias, kept emitting a deprecation warning at every rank init).
#     2048 entries is enough for a few minutes of history at our cadence.
#   - TORCH_NCCL_DUMP_ON_TIMEOUT: writes the buffer to disk when the
#     watchdog fires.
# Cost when no timeout occurs: negligible (a few hundred KB per rank).
export TORCH_FR_BUFFER_SIZE=2048
export TORCH_NCCL_DUMP_ON_TIMEOUT=1

# MIOpen kernel cache: per-job, node-local
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-${SLURM_JOB_ID:-local}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

# Distributed master endpoint derived from SLURM allocation
if [ -n "${SLURM_NODELIST:-}" ]; then
    MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)"
else
    MASTER_ADDR="127.0.0.1"
fi
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29500}"
