# shellcheck shell=bash
# Sourced by every Frontier SLURM wrapper. Wrappers cd to the FusionAIHub
# repo root before sourcing, so $PWD = repo root here.

module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load rocm/7.1.1
module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"

PIXI_ENV="$PWD/.pixi/envs/frontier"
export PATH="${PIXI_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${PIXI_ENV}/lib:${LD_LIBRARY_PATH}"
export CONDA_PREFIX="${PIXI_ENV}"

# Performance / correctness knobs
export PYTORCH_ROCM_ARCH=gfx90a
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HSA_FORCE_FINE_GRAIN_PCIE=1

# flash-attn 2 on ROCm: main_perf branch requires this at IMPORT time to
# take the Triton-AMD (aiter) path; otherwise it tries `flash_attn_2_cuda`.
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

# RCCL over Slingshot HSN. The OLCF rccl-net-plugin module routes RCCL
# collectives through libfabric/cxi (aws-ofi-nccl 1.19.2, prebuilt against
# rocm/7.1.1) instead of TCP sockets, and sets the HPE-recommended NCCL/FI_CXI
# env (NCCL_CROSS_NIC=1, GDR_LEVEL=PHB, all 4 hsn NICs, kdreg2, ...).
# Without it, inter-node allreduce runs over TCP on hsn0 — the path production
# silently fell back to after the hand-built ~/aws-ofi-nccl plugin was
# disabled 2026-05-27 for hang triage (see _frontier_common.sh history).
# RCCL_PLUGIN=0 restores that socket path as a hang-triage escape hatch.
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

# MIOpen kernel cache: per-job, node-local
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

# Distributed master endpoint
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)"
export MASTER_PORT=29500
