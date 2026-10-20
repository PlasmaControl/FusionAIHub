#!/bin/bash
#
# Submit FDP shot fetches as SLURM job arrays, in batches, with resume.
#
# Ported from the omega submit_read_mds_batches.sh: same batching, throttling
# and .completed_shots/.failed_shots resume behaviour. The Globus stage is gone
# (the fetch runs on the cluster that stores the output) and so is the local
# staging cache (reads are range requests against a Pelican cache).
#
# Usage:
#   ./submit_fetch_batches.sh
#   nohup ./submit_fetch_batches.sh > submit_fdp.log 2>&1 &

# ============================================
# Configuration -- SET OUTPUT_DIR BEFORE USE
# ============================================
# Left blank on purpose so the filesystem is a deliberate choice. One <shot>.h5
# per shot, ~6-7 GB each for the full config.
OUTPUT_DIR=""

# The pixi project that provides the FDP stack (fdp, toksearch, toksearch_d3d,
# ptdata 2.x, mdsplus-xrdcl, libfdpio2, h5py). Its `fdp run` supplies the
# federation environment; see the guide in that directory.
FDP_PROJECT="/scratch/gpfs/EKOLEMEN/ps9551/fdp-demo"

# Mode: "range" or "list"
MODE="range"

# For range mode:
SHOT_START=200000
SHOT_END=200100

# For list mode (one shot number per line, '#' comments allowed):
SHOT_LIST_FILE="shots_to_process.txt"

# Signal configs, resolved relative to this directory.
CONFIG_FILES="config_fdp.yaml"

# Where each job stages its objects. /dev/shm is RAM: nothing reaches Stellar's
# disks and the files vanish with the job. A shot needs ~5 GB, so keep
# MAX_SUBMIT_LIMIT x 5 GB comfortably inside a node's memory, or point this at a
# filesystem instead.
STAGE_PARENT="/dev/shm"

# Batch settings
BATCH_SIZE=1000
MAX_SUBMIT_LIMIT=25

# State files
STATE_FILE=".submission_state"
COMPLETED_FILE=".completed_shots"
FAILED_FILE=".failed_shots"

# ============================================
# Main script
# ============================================
cd "$(dirname "$0")" || exit 1

if [ -z "${OUTPUT_DIR}" ]; then
    echo "ERROR: set OUTPUT_DIR at the top of this script."
    echo "       Left blank on purpose: output grows ~6-7 GB per shot, so the"
    echo "       filesystem should be a deliberate choice."
    exit 1
fi

if [ ! -d "${FDP_PROJECT}" ]; then
    echo "ERROR: FDP project not found: ${FDP_PROJECT}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" jobs
touch "${COMPLETED_FILE}" "${FAILED_FILE}"

echo "========================================="
echo "FDP Batch Data Fetcher"
echo "========================================="
echo "Mode:            ${MODE}"
echo "Configs:         ${CONFIG_FILES}"
echo "Output dir:      ${OUTPUT_DIR}"
echo "FDP project:     ${FDP_PROJECT}"
echo "Batch size:      ${BATCH_SIZE}"
echo "Max concurrent:  ${MAX_SUBMIT_LIMIT}"

for config in ${CONFIG_FILES}; do
    if [ ! -f "${config}" ]; then
        echo "ERROR: config file not found: ${config}"
        exit 1
    fi
done

if [ ! -f "$HOME/.fdp/token" ] && [ -z "${BEARER_TOKEN}" ]; then
    echo "ERROR: no FDP bearer token (~/.fdp/token or \$BEARER_TOKEN)."
    echo "       Tokens are JWTs with a limited lifetime, so a long campaign"
    echo "       can outlive one; a wholesale batch failure with 401/403 means"
    echo "       it expired."
    exit 1
fi

# Smallest end-to-end check of token, director and cache before submitting
# anything: it costs one small object read.
echo -n "Checking FDP access... "
if (cd "${FDP_PROJECT}" && pixi run fdp ls /fdp-d3d/archives/index/json >/dev/null 2>&1); then
    echo "ok"
else
    echo "FAILED"
    echo "ERROR: 'fdp ls /fdp-d3d/archives/index/json' did not succeed."
    echo "       Check the token first, then the environment; see the FDP guide."
    exit 1
fi

# ---- build the work list, skipping completed shots ----
SHOT_LIST=$(mktemp)

if [ "${MODE}" = "range" ]; then
    echo "Shot range:      ${SHOT_START} to ${SHOT_END}"
    for shot in $(seq ${SHOT_START} ${SHOT_END}); do
        grep -q "^${shot}$" "${COMPLETED_FILE}" 2>/dev/null && continue
        echo ${shot} >> "${SHOT_LIST}"
    done
elif [ "${MODE}" = "list" ]; then
    echo "Shot list file:  ${SHOT_LIST_FILE}"
    if [ ! -f "${SHOT_LIST_FILE}" ]; then
        echo "ERROR: shot list file not found: ${SHOT_LIST_FILE}"
        rm -f "${SHOT_LIST}"
        exit 1
    fi
    while IFS= read -r shot; do
        [[ -z "$shot" || "$shot" =~ ^[[:space:]]*# ]] && continue
        grep -q "^${shot}$" "${COMPLETED_FILE}" 2>/dev/null && continue
        echo ${shot} >> "${SHOT_LIST}"
    done < "${SHOT_LIST_FILE}"
else
    echo "ERROR: invalid MODE '${MODE}'. Must be 'range' or 'list'"
    rm -f "${SHOT_LIST}"
    exit 1
fi

TOTAL_SHOTS=$(wc -l < "${SHOT_LIST}")
echo "========================================="

if [ "${TOTAL_SHOTS}" -eq 0 ]; then
    echo "No shots to process (all completed or none in range)"
    rm -f "${SHOT_LIST}"
    exit 0
fi

echo "Total shots to process: ${TOTAL_SHOTS}"

# ---- submit in batches, throttled on queue depth ----
BATCH_NUM=0
SHOT_INDEX=0

while [ ${SHOT_INDEX} -lt "${TOTAL_SHOTS}" ]; do
    BATCH_NUM=$((BATCH_NUM + 1))
    BATCH_FILE="batch_${BATCH_NUM}.txt"

    sed -n "$((SHOT_INDEX + 1)),$((SHOT_INDEX + BATCH_SIZE))p" \
        "${SHOT_LIST}" > "${BATCH_FILE}"
    BATCH_SHOTS=$(wc -l < "${BATCH_FILE}")

    while true; do
        RUNNING_JOBS=$(squeue -u "$USER" -h -t running,pending -r | wc -l)
        [ "${RUNNING_JOBS}" -lt "${MAX_SUBMIT_LIMIT}" ] && break
        echo "Queue full: ${RUNNING_JOBS}/${MAX_SUBMIT_LIMIT} | Completed: \
$(wc -l < "${COMPLETED_FILE}")/${TOTAL_SHOTS} | Failed: $(wc -l < "${FAILED_FILE}")"
        sleep 30
    done

    echo "Submitting batch ${BATCH_NUM} with ${BATCH_SHOTS} shots..."
    JOB_ID=$(sbatch --parsable \
        --array=1-${BATCH_SHOTS} \
        --output=jobs/job_%A_%a.out \
        --error=jobs/job_%A_%a.err \
        --export=ALL,BATCH_FILE=${BATCH_FILE},CONFIG_FILES="${CONFIG_FILES}",OUTPUT_DIR=${OUTPUT_DIR},FDP_PROJECT=${FDP_PROJECT},COMPLETED_FILE=${COMPLETED_FILE},FAILED_FILE=${FAILED_FILE},STAGE_PARENT=${STAGE_PARENT} \
        fetch_shot.sh)

    echo "Submitted batch ${BATCH_NUM} as job ${JOB_ID}"
    echo "${BATCH_NUM}:${BATCH_FILE}:${JOB_ID}" >> "${STATE_FILE}"

    SHOT_INDEX=$((SHOT_INDEX + BATCH_SHOTS))
    sleep 2
done

echo ""
echo "========================================="
echo "All batches submitted (${BATCH_NUM} total). Monitoring..."
echo "========================================="

while true; do
    RUNNING_JOBS=$(squeue -u "$USER" -h -t running,pending -r | wc -l)
    echo "Jobs running: ${RUNNING_JOBS} | Completed: \
$(wc -l < "${COMPLETED_FILE}")/${TOTAL_SHOTS} | Failed: $(wc -l < "${FAILED_FILE}")"
    [ "${RUNNING_JOBS}" -eq 0 ] && { echo "All jobs finished"; break; }
    sleep 60
done

echo ""
echo "========================================="
echo "Final Summary"
echo "========================================="
echo "Total shots requested:  ${TOTAL_SHOTS}"
echo "Successfully completed: $(wc -l < "${COMPLETED_FILE}")"
echo "Failed:                 $(wc -l < "${FAILED_FILE}")"
echo "========================================="

rm -f "${SHOT_LIST}" "${STATE_FILE}" batch_*.txt
echo "Done at: $(date)"
