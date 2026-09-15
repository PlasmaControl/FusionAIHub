#!/usr/bin/env bash
# Run on the login node after serve_llm.sbatch publishes <data_root>/llm/endpoint.json.
# Pass --dry-run --limit 5 to preview; the client is CPU-side and the endpoint holds the GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
exec pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m shot_design blurb --all "$@"
