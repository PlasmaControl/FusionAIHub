#!/usr/bin/env bash
# Rootless Ollama for shot_design. Reuses the existing binary, model store and keypair home.
# Usage: scripts/shot_design/serve_llm.sh [--host HOST] [--port PORT]
# Only the operator starts this script; tests substitute a fake binary and temporary paths.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

HOST=127.0.0.1
PORT=11434
while [ $# -gt 0 ]; do
  case "$1" in
    --host) [ $# -ge 2 ] || { echo "--host needs a value" >&2; exit 2; }; HOST="$2"; shift 2 ;;
    --port) [ $# -ge 2 ] || { echo "--port needs a value" >&2; exit 2; }; PORT="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
URL="http://$HOST:$PORT"
# Any HTTP response counts as occupied, including an error from a non-Ollama server.
# Nothing has been started or written when we refuse a second instance.
if curl --noproxy '*' --connect-timeout 1 --max-time 2 -s -o /dev/null "$URL/api/version"; then
  echo "a server is already listening on $URL; not starting a second one" >&2
  exit 2
fi

# ONE Pixi invocation resolves every setting and the interpreter for endpoint publication.
# The absolute manifest reuses the main checkout's environment; PYTHONPATH selects this source.
PYOUT="$(pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -c '
import sys
from shot_design import config
cfg = config.load_yaml("llm.yaml")
root = "/scratch/gpfs/EKOLEMEN/nc1514/shot-recommender"
print(sys.executable)
print(config.load_paths().data_root)
print(cfg.get("ollama_bin_dir", root + "/bin/ollama"))
print(cfg.get("ollama_models_dir", root + "/models/ollama"))
print(cfg.get("ollama_home_dir", root + "/ollama_home"))
for tag in dict.fromkeys(cfg["models"].values()):
    print(tag)
')"
mapfile -t SETTINGS <<<"$PYOUT"
SHOT_DESIGN_PY="${SETTINGS[0]}"
DATA_ROOT="${SETTINGS[1]}"
BIN_DIR="${SETTINGS[2]}"
export OLLAMA_MODELS="${SETTINGS[3]}"
OLLAMA_HOME_DIR="${SETTINGS[4]}"
MODELS=("${SETTINGS[@]:5}")
OLLAMA="$BIN_DIR/bin/ollama"
EP="$DATA_ROOT/llm/endpoint.json"
if [ ! -x "$OLLAMA" ]; then
  echo "Ollama binary missing or not executable: $OLLAMA" >&2
  exit 2
fi
export OLLAMA_HOST="$HOST:$PORT"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-24h}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-2}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-16384}"
export LD_LIBRARY_PATH="$BIN_DIR/lib/ollama${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$DATA_ROOT/llm" "$OLLAMA_MODELS" "$OLLAMA_HOME_DIR"
STARTED="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
WROTE_EP=
cleanup() {
  # Compare both URL and run start: stopping this run must preserve a replacement's handle,
  # including a replacement using the same URL. Parse JSON rather than depending on formatting.
  if [ -n "$WROTE_EP" ]; then
    "$SHOT_DESIGN_PY" - "$EP" "$URL" "$STARTED" <<'PY'
import json
import sys
from pathlib import Path
path, url, started = sys.argv[1:]
p = Path(path)
try:
    doc = json.loads(p.read_text())
    if doc.get("url") == url and doc.get("started") == started:
        p.unlink()
except (OSError, ValueError, AttributeError):
    pass
PY
  fi
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Set HOME only for the Ollama child; leave this shell and Pixi's home unchanged.
env HOME="$OLLAMA_HOME_DIR" "$OLLAMA" serve &
SERVER_PID=$!
for _ in {1..60}; do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "our Ollama process exited" >&2; exit 1; }
  if curl --noproxy '*' --connect-timeout 1 --max-time 2 -fs "$URL/api/version" >/dev/null; then
    break
  fi
  sleep 1
done
curl --noproxy '*' --connect-timeout 1 --max-time 2 -fs "$URL/api/version" >/dev/null || {
  echo "Ollama did not start" >&2; exit 1;
}
for m in "${MODELS[@]}"; do
  # Installed tags require no registry access or downloads. Retain pull for a missing tag.
  if ! env HOME="$OLLAMA_HOME_DIR" "$OLLAMA" show "$m" >/dev/null 2>&1; then
    echo "pulling missing model $m"
    env HOME="$OLLAMA_HOME_DIR" "$OLLAMA" pull "$m"
  fi
done
kill -0 "$SERVER_PID" 2>/dev/null || { echo "our Ollama process exited" >&2; exit 1; }
VERSION="$(env HOME="$OLLAMA_HOME_DIR" "$OLLAMA" --version 2>/dev/null | tr -d '\n')"
"$SHOT_DESIGN_PY" - "$EP" "$URL" "$VERSION" "$STARTED" "${MODELS[@]}" <<'PY'
import json
import os
import socket
import sys
import tempfile
from pathlib import Path

ep, url, version, started, *models = sys.argv[1:]
doc = {"url": url, "models": models, "host": socket.gethostname(),
       "job_id": os.environ.get("SLURM_JOB_ID"), "started": started, "version": version}
with tempfile.NamedTemporaryFile(mode="w", dir=Path(ep).parent, suffix=".part", delete=False) as f:
    tmp = Path(f.name)
    json.dump(doc, f, indent=1)
try:
    os.replace(tmp, ep)
finally:
    tmp.unlink(missing_ok=True)
PY
WROTE_EP=1
echo "ready: $URL  (endpoint file $EP; models ${MODELS[*]})"
wait "$SERVER_PID"
