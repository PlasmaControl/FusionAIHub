#!/bin/bash
#
# FDP campaign on the StellarAI login node, with resume.
#
# StellarAI compute nodes have no route to the FDP: no DNS for osg-htc.org, and the
# proxy/default module forbids every FDP host (tested 2026-10-19). So the fetch runs
# on the login node, niced, a few shots at a time. That is cheap: a shot is ~4 min of
# staging and ~10-20 min of PTDATA retries, both network-bound, and ~10 s of CPU.
#
# Usage:
#   OUTPUT_DIR=/scratch/gpfs/... SHOT_LIST=shots.txt nohup ./fetch_login_batches.sh > fdp_login.log 2>&1 &
#   OUTPUT_DIR=/scratch/gpfs/... SHOT_START=200000 SHOT_END=200100 nohup ./fetch_login_batches.sh ...
#
# Resume: shots in $OUTPUT_DIR/.completed_shots are skipped; failures go to
# .failed_shots. Stops before the next shot once the token has under 30 min left.

set -u
cd "$(dirname "$0")" || exit 1
: "${OUTPUT_DIR:?set OUTPUT_DIR (one <shot>.h5 per shot, ~17 GB each: 16.6 GB measured on 192945)}"
CONCURRENT="${CONCURRENT:-2}"      # shots in flight; each stages ~5 GB in /dev/shm
export WORKERS="${WORKERS:-4}"     # read processes per shot
export FDP_PROJECT="${FDP_PROJECT:-/scratch/gpfs/EKOLEMEN/ps9551/fdp-demo}"
TOKEN="${BEARER_TOKEN_FILE:-$HOME/.fdp/token}"

mkdir -p "${OUTPUT_DIR}/logs"
COMPLETED="${OUTPUT_DIR}/.completed_shots"; FAILED="${OUTPUT_DIR}/.failed_shots"
touch "${COMPLETED}" "${FAILED}"

if [ -n "${SHOT_LIST:-}" ]; then
    shots=$(grep -vE '^\s*(#|$)' "${SHOT_LIST}")
else
    shots=$(seq "${SHOT_START:?set SHOT_LIST or SHOT_START/SHOT_END}" "${SHOT_END:?}")
fi
todo=$(comm -23 <(echo "${shots}" | sort -u) <(sort -u "${COMPLETED}"))
echo "[fdp] $(date '+%F %T') $(echo "${todo}" | grep -c .) shots to fetch, ${CONCURRENT} at a time -> ${OUTPUT_DIR}"

token_minutes() {
    python3 -I -c 'import base64,json,sys,time
t=open(sys.argv[1]).read().strip().split(".")[1]; t+="="*(-len(t)%4)
print(int((json.loads(base64.urlsafe_b64decode(t))["exp"]-time.time())//60))' "${TOKEN}" 2>/dev/null || echo -1
}

fetch_one() {
    shot=$1
    left=$(token_minutes)
    if [ "${left}" -lt 30 ]; then
        echo "[fdp] $(date '+%T') ${shot}: token has ${left} min left -- run 'fdp login', then restart"
        return 255   # stops xargs
    fi
    log="${OUTPUT_DIR}/logs/${shot}.log"
    if nice -n 19 ./fetch_shot_fast.sh "${shot}" "${OUTPUT_DIR}" > "${log}" 2>&1 \
            && [ -s "${OUTPUT_DIR}/${shot}.h5" ]; then
        ( flock -x 9; echo "${shot}" >> "${COMPLETED}" ) 9>"${COMPLETED}.lock"
        echo "[fdp] $(date '+%T') ${shot}: ok ($(tail -1 "${log}" | grep -oE '[0-9]+s' | head -1))"
    else
        ( flock -x 9; echo "${shot}" >> "${FAILED}" ) 9>"${FAILED}.lock"
        echo "[fdp] $(date '+%T') ${shot}: FAILED, see ${log}"
    fi
}
export -f fetch_one token_minutes
export OUTPUT_DIR COMPLETED FAILED TOKEN

echo "${todo}" | grep . | xargs -P "${CONCURRENT}" -I{} bash -c 'fetch_one "$@"' _ {}
echo "[fdp] $(date '+%F %T') done: $(wc -l < "${COMPLETED}") completed, $(sort -u "${FAILED}" | wc -l) failed"
