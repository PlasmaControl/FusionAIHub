#!/bin/bash
#
# Fast per-shot fetch: stage once, then read locally in two passes.
#
# Why two passes in two processes rather than one process?  The MDSplus pass
# wants PTDATA_JSON_INDEX_DIR *absent*, so that PTDATA-backed TDI expression
# nodes read the staged objects instead of the federation (measured on ECE
# TECEF01: 18.9 s indexed vs 1.34 s staged).  The PTDATA pointname pass wants
# it *present*.  Toggling that variable inside one process leaves the ptdata
# plugin in a broken state -- it opens sockets to the origin, bypasses the
# Princeton cache and deadlocks in futex.  Two processes, each with a clean
# environment, avoids the problem entirely.
#
# Usage:
#   ./fetch_shot_fast.sh <shot> <output_dir> [stage_parent]
#
# Run it from anywhere; it wraps itself in `pixi run fdp run` inside the FDP
# project.  Staging defaults to /dev/shm (RAM), so nothing is written to
# Stellar's disks, and it is removed when the shot finishes.

set -u

SHOT="${1:?usage: fetch_shot_fast.sh <shot> <output_dir> [stage_parent]}"
OUTPUT_DIR="${2:?usage: fetch_shot_fast.sh <shot> <output_dir> [stage_parent]}"
STAGE_PARENT="${3:-/dev/shm/fdp_stage_$$}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FDP_PROJECT="${FDP_PROJECT:-/scratch/gpfs/EKOLEMEN/ps9551/fdp-demo}"
ENV_DIR="${FDP_PROJECT}/.pixi/envs/default"
CONFIG="${CONFIG:-${SCRIPT_DIR}/config_fdp.yaml}"
# Reads are independent, so they scale with processes. 8 is a safe default on a
# 96-core node; the staged objects are in RAM so there is no disk contention.
WORKERS="${WORKERS:-8}"

STAGE_DIR="${STAGE_PARENT}/${SHOT}"
mkdir -p "${OUTPUT_DIR}" "${STAGE_DIR}"

# Pass 3 deliberately keeps its retry on. Its own failures are the pointnames
# absent from the shot index, and the index is not authoritative about absence:
# retrying all 152 of them takes 5.6 s (median 2 ms -- they fail on a local
# routing check, not a round trip) and recovers MPI79FA322D, which the reference
# file has and which matches it byte for byte. Roughly a second per signal
# gained.
#
# Pass 2 cannot read PTDATA-backed TDI expression nodes at all: it runs with
# PTDATA_JSON_INDEX_DIR unset, and that variable is what routes the read, so
# retrying them in pass 2 recovers nothing however many attempts it makes (the
# NB beam CURRENT/VOLTAGE/PERVEANCE/F0 nodes are the ones this loses). It hands
# them to pass 3 through this manifest instead, which has the index.
RETRY_MANIFEST="${STAGE_DIR}/retry_pass2.json"

# Pelican cache. FDP_PREFER_SHERPA=1 selects the Princeton cache, as the
# d3d-cache module does. Off by default: since tokens come from the origin's own
# issuer (fdp login, fdp >= 0.12), sherpa rejects them ("token appears valid
# but was rejected by the server", 2026-10-19) and the director's cache works.
export BEARER_TOKEN_FILE="${BEARER_TOKEN_FILE:-${HOME}/.fdp/token}"
if [ "${FDP_PREFER_SHERPA:-0}" = "1" ]; then
    export PELICAN_CLIENT_PREFERREDCACHES="https://sherpa.princeton.edu:8443"
    export PELICAN_NEAREST_CACHE="https://sherpa.princeton.edu:8443"
else
    unset PELICAN_CLIENT_PREFERREDCACHES PELICAN_NEAREST_CACHE
fi

cleanup() { rm -rf "${STAGE_PARENT}"; }
trap cleanup EXIT

echo "shot ${SHOT}: staging to ${STAGE_DIR}, output ${OUTPUT_DIR}"
START=$(date +%s)

# ---- pass 1: stage everything (needs the federation index) -----------------
"${ENV_DIR}/bin/fdp" run "${ENV_DIR}/bin/python" -u "${SCRIPT_DIR}/fetch_shot.py" \
    --shot "${SHOT}" --config "${CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    --stage-dir "${STAGE_PARENT}" --keep-stage --stage-only
STAGE_RC=$?
if [ ${STAGE_RC} -ne 0 ]; then
    echo "shot ${SHOT}: staging failed (rc=${STAGE_RC})"
    exit 1
fi

# ---- pass 2: MDSplus, index removed so TDI nodes read staged objects -------
"${ENV_DIR}/bin/fdp" run "${SCRIPT_DIR}/_without_index.sh" \
    "${ENV_DIR}/bin/python" -u "${SCRIPT_DIR}/fetch_shot.py" \
    --shot "${SHOT}" --config "${CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    --stage-dir "${STAGE_PARENT}" --keep-stage --reuse-stage --no-scan \
    --workers "${WORKERS}" --skip-tree ptdata --no-retry \
    --write-retry-manifest "${RETRY_MANIFEST}"
MDS_RC=$?

# ---- pass 3: PTDATA pointnames, index present, staged objects reused -------
# Bounded: through the director's cache its TDI retry phase stalled for >25 min on
# shot 192945 (2026-10-19) after the pointnames had been merged into the file, so a
# timeout loses at most retried signals, never the PTDATA already written.
timeout "${PASS3_TIMEOUT:-1200}" \
"${ENV_DIR}/bin/fdp" run "${ENV_DIR}/bin/python" -u "${SCRIPT_DIR}/fetch_shot.py" \
    --shot "${SHOT}" --config "${CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    --stage-dir "${STAGE_PARENT}" --keep-stage --reuse-stage --no-scan \
    --workers "${WORKERS}" --only-tree ptdata \
    --retry-manifest "${RETRY_MANIFEST}"
PT_RC=$?

ELAPSED=$(( $(date +%s) - START ))
echo "shot ${SHOT}: both passes done in ${ELAPSED}s (mds rc=${MDS_RC}, ptdata rc=${PT_RC})"

if [ -s "${OUTPUT_DIR}/${SHOT}.h5" ]; then
    exit 0
fi
exit 1
