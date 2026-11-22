#!/usr/bin/env bash
# Builds /models/envs/ignite-mast: TokaMark's scorer and this repository's model code in one
# interpreter (Python 3.12). It starts from the exact package versions of the TokaMind
# environment, so TokaMark behaves as it does in the replication, then adds what IGNITE needs.
# Run once, with no arguments. It never replaces an existing environment.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ENV=/models/envs/ignite-mast

if [ -e "$ENV" ]; then
    echo "make_env.sh: $ENV already exists; not touching it. Move it away yourself to rebuild." >&2
    exit 1
fi

mkdir -p /models/envs
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

uv venv "$ENV" --python 3.12
uv pip freeze --python /models/tokamind/.venv/bin/python | grep -v -e '^-e ' -e ' @ file://' > "$TMP/base.txt"
uv pip install --python "$ENV/bin/python" -r "$TMP/base.txt"
uv pip install --python "$ENV/bin/python" einops==0.8.2 x-transformers==2.23.5 \
    vector-quantize-pytorch==1.31.0 "flash-attn-4[cu13]==4.0.0b40" pytest

# The three source trees enter through a .pth file. An editable install would write
# *.egg-info under /models/tokamind/, which must stay untouched.
SITE=$("$ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
printf '%s\n' /models/tokamind/tokamark/src /models/tokamind/repo/src "$REPO/src" > "$SITE/ignite_mast.pth"

uv pip freeze --python "$ENV/bin/python" > "$REPO/scripts/mast/requirements-node13.txt"
