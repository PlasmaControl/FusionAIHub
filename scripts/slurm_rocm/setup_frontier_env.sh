#!/bin/bash
# Build & install flash-attention 2 (Triton backend) for OLCF Frontier (MI250X / gfx90a).
#
# Run from the repo root on a Frontier LOGIN node:
#     pixi run -e frontier setup-flash-attn
#
# Builds entirely on the login node — no SLURM allocation, no GPU. The Triton
# backend (FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE) replaces the multi-hour
# Composable Kernel template/hipcc compile with a quick pure-Python install
# (~2-5 min). Triton kernels are JIT-compiled at first use, so no GPU is
# needed at build time.
#
# A separate `verify-flash-attn` pixi task tests the install on a GPU; run it
# from inside any SLURM allocation that has --gpus.
#
# Prerequisite: `pixi install -e frontier` has been run once.
set -euo pipefail

PROJECT_DIR=/lustre/orion/fus187/scratch/nchen/FusionAIHub
FLASH_ATTN_SHA=5301a359f59ef8fa10f211618d9f7a69716a8898
FLASH_ATTN_URL="https://github.com/ROCm/flash-attention.git"
FLASH_ATTN_LOCAL="${PROJECT_DIR}/.build/flash-attention"
ROCM_MODULE=rocm/7.1.1

cd "$PROJECT_DIR"

echo "=== Ensuring local flash-attention checkout ==="
mkdir -p "$(dirname "${FLASH_ATTN_LOCAL}")"
if [ ! -d "${FLASH_ATTN_LOCAL}/.git" ]; then
    echo "    cloning ${FLASH_ATTN_URL} -> ${FLASH_ATTN_LOCAL}"
    git clone --filter=blob:none "${FLASH_ATTN_URL}" "${FLASH_ATTN_LOCAL}"
fi
pushd "${FLASH_ATTN_LOCAL}" >/dev/null
HAVE_SHA="$(git rev-parse HEAD 2>/dev/null || echo none)"
if [ "${HAVE_SHA}" != "${FLASH_ATTN_SHA}" ]; then
    echo "    fetching + checking out ${FLASH_ATTN_SHA}"
    git fetch origin "${FLASH_ATTN_SHA}"
    git checkout -q "${FLASH_ATTN_SHA}"
fi
echo "    initializing submodules"
git submodule update --init --recursive
popd >/dev/null

# Locate the pixi env's python. We bypass `pixi run` / `pixi install` because
# both re-resolve the lock file on every invocation (slow on PyPI sockets,
# and pixi/uv hangs on autofs locks under contention).
PIXI_PY="${PROJECT_DIR}/.pixi/envs/frontier/bin/python"
if [ ! -x "$PIXI_PY" ]; then
    echo "ERROR: frontier pixi env not provisioned at $PIXI_PY." >&2
    echo "       Run \`pixi install -e frontier\` first." >&2
    exit 1
fi

# Module load on the login node. The Triton backend doesn't strictly require
# the ROCm module at build time (Triton compiles kernels JIT at first call,
# inside whatever ROCm environment the runtime uses), but we load it for
# consistency with the runtime environment.
# shellcheck disable=SC1091
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load PrgEnv-gnu "${ROCM_MODULE}" craype-accel-amd-gfx90a

# Triton backend — no Composable Kernel, no hipcc template explosion.
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export PYTORCH_ROCM_ARCH=gfx90a

echo ""
echo "=== Installing flash-attn 2 (Triton backend) on login node ==="
echo "    source     = ${FLASH_ATTN_LOCAL}"
echo "    pinned SHA = ${FLASH_ATTN_SHA}"
echo "    python     = ${PIXI_PY}"
echo "    FLASH_ATTENTION_TRITON_AMD_ENABLE=${FLASH_ATTENTION_TRITON_AMD_ENABLE}"
"$PIXI_PY" -m pip install --no-build-isolation -v "${FLASH_ATTN_LOCAL}"

echo ""
echo "=== Login-node install complete ==="
echo "Test the install on a GPU from inside a SLURM allocation:"
echo "    salloc -A fus187 -t 00:10:00 -N 1 --gpus=1"
echo "    pixi run -e frontier verify-flash-attn"
