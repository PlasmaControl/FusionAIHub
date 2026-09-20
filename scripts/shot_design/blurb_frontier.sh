#!/usr/bin/env bash
# Backfill blurbs on Frontier via Gemini Flash (agy). Runs on the LOGIN NODE, not sbatch:
# agy needs the network and the OAuth cache in ~/.gemini, neither of which a compute node
# has. Pass --dry-run --limit 5 to preview a few shots first. WORKERS sets the thread pool
# size (default 8); each in-flight call is one ~5-10 s `agy` subprocess.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/slurm_frontier/_shot_design_common.sh
"$PY" -m shot_design blurb --workers "${WORKERS:-8}" "$@"
