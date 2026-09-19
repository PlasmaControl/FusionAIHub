# Frontier-common environment for ROCm DDP jobs.
# Source from every Frontier SLURM script BEFORE activating the venv.
# Sets modules, RCCL/NCCL knobs, MIOpen cache, and MASTER_ADDR/PORT.
#
# Frontier hardware reminders (see docs.olcf.ornl.gov):
#   - 4x MI250X = 8 GCDs per node, each appears as a separate GPU.
#   - HSN is Slingshot via libfabric/cxi; RCCL needs hsn0 + kdreg2.
#   - MIOpen cache in $HOME is slow & contended; redirect to /tmp.
#
# NOTE 2026-08-29: this file was purged from Lustre scratch (atime purge) and
# was also missing from HEAD; recreated from ps9551's production clone at
# proj-shared (flock fix included) plus the rccl-net-plugin block below.

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

# RCCL over Slingshot HSN. The OLCF rccl-net-plugin module routes RCCL
# collectives through libfabric/cxi (aws-ofi-nccl 1.19.2, prebuilt against
# rocm/7.1.1) instead of TCP sockets, and sets the HPE-recommended NCCL/FI_CXI
# env (NCCL_CROSS_NIC=1, GDR_LEVEL=PHB, all 4 hsn NICs, kdreg2, ...).
# Replaces the hand-built ~/aws-ofi-nccl plugin (validated by smoke 4615534,
# then DISABLED 2026-05-27 for post-maintenance NCCL hang diagnosis — jobs
# 4700720/21 ALLREDUCE and 4700730/31 BROADCAST timeouts). Production ran on
# TCP sockets ever since. The module is the OLCF-supported successor; if
# hangs recur, RCCL_PLUGIN=0 restores the socket path as an escape hatch.
# MUST come after `pixi shell-hook` — the hook overwrites LD_LIBRARY_PATH,
# and the plugin is discovered via the lib dir this module prepends.
if [ "${RCCL_PLUGIN:-1}" = "1" ]; then
    module load rccl-net-plugin/1.0
    # The module also sets FI_CXI_RDZV_PROTO=alt_read, which is only valid
    # when the job requested `#SBATCH --network=disable_rdzv_get`. Opt in with
    # RCCL_ALT_RDZV=1 alongside that sbatch flag; otherwise strip the alt-read
    # rendezvous knobs so the default protocol is used (plugin still active).
    if [ "${RCCL_ALT_RDZV:-0}" != "1" ]; then
        unset FI_CXI_RDZV_PROTO FI_CXI_RDZV_EAGER_SIZE \
              FI_CXI_RDZV_THRESHOLD FI_CXI_RDZV_GET_MIN
    fi
    # Multi-node: fail loudly if the plugin fails to initialize — a silent
    # fallback to TCP sockets is exactly the regression this block fixes.
    # (Single-node jobs skip this: the HSN fabric isn't configured for them
    # and the plugin never initializes there by design.)
    if [ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 ]; then
        export NCCL_NET="OFI"
    fi
else
    export NCCL_SOCKET_IFNAME=hsn0
    export NCCL_NET_GDR_LEVEL=3
    export FI_MR_CACHE_MONITOR=kdreg2
    export FI_CXI_DEFAULT_CQ_SIZE=131072
fi

# Performance / correctness knobs
export PYTORCH_ROCM_ARCH=gfx90a
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HSA_FORCE_FINE_GRAIN_PCIE=1

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
