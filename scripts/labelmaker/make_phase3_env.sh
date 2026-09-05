#!/bin/bash
# Build the Phase 3 training venv on group storage. Idempotent: re-running
# re-resolves the requirements into the same prefix.
#
#     bash scripts/labelmaker/make_phase3_env.sh
#
# The venv lives under $LABELMAKER_ROOT/envs, never under /scratch/gpfs/nc1514
# (near quota), and so does uv's cache, so wheels hardlink into the venv
# instead of being copied across filesystems.
set -euo pipefail

UV=${UV:-/home/nc1514/.local/bin/uv}
ROOT=${LABELMAKER_ROOT:-/scratch/gpfs/EKOLEMEN/nc1514/labelmaker}
VENV=${VENV:-$ROOT/envs/phase3}
REQ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/phase3-requirements.txt"

export UV_CACHE_DIR=${UV_CACHE_DIR:-$ROOT/envs/.uv-cache}
mkdir -p "$ROOT/envs"

"$UV" venv --python 3.11 "$VENV"
"$UV" pip install --python "$VENV/bin/python" -r "$REQ"

"$VENV/bin/python" -c "import torch, sklearn, sksurv, tabpfn; \
print('torch', torch.__version__, 'sklearn', sklearn.__version__, \
'sksurv', sksurv.__version__, 'tabpfn', tabpfn.__version__)"
echo "phase3 venv ready: $VENV"
