#!/bin/bash

# ============================================
# Configuration
# ============================================
# Choose mode: "range" or "list"
MODE="list"  # or "list"

# For range mode:
SHOT_START=200700
SHOT_END=200800

# For list mode (one shot number per line):
SHOT_LIST_FILE="shots_to_process.txt"

# Common configuration
CONFIG_FILES="config_atlas.yaml config_chiron.yaml"  # Process both servers
OUTPUT_DIR="/cscratch/steinerp/database/data"
NODE_PATHS_DIR="/cscratch/steinerp/database/node_paths"  # Deprecated but kept for compatibility

# Batch settings
BATCH_SIZE=1000
MAX_SUBMIT_LIMIT=25

# State files
STATE_FILE=".submission_state"
COMPLETED_FILE=".completed_shots"
FAILED_FILE=".failed_shots"

# ============================================
# Main Script
# ============================================

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}
mkdir -p jobs

# Initialize tracking files if they don't exist
touch ${COMPLETED_FILE}
touch ${FAILED_FILE}

echo "========================================="
echo "MDSPlus Batch Data Fetcher"
echo "========================================="
echo "Mode: ${MODE}"
echo "Config files: ${CONFIG_FILES}"

if [ "${MODE}" = "range" ]; then
    echo "Shot range: ${SHOT_START} to ${SHOT_END}"
elif [ "${MODE}" = "list" ]; then
    echo "Shot list file: ${SHOT_LIST_FILE}"
else
    echo "ERROR: Invalid MODE '${MODE}'. Must be 'range' or 'list'"
    exit 1
fi

# Verify all config files exist
for config in ${CONFIG_FILES}; do
    if [ ! -f "${config}" ]; then
        echo "ERROR: Config file not found: ${config}"
        exit 1
    fi
done

echo "Output directory: ${OUTPUT_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo "Max concurrent jobs: ${MAX_SUBMIT_LIMIT}"
echo "========================================="

# Generate shot list based on mode
SHOT_LIST=$(mktemp)

if [ "${MODE}" = "range" ]; then
    # Range mode: generate sequence
    for shot in $(seq ${SHOT_START} ${SHOT_END}); do
        # Skip if already completed
        if grep -q "^${shot}$" ${COMPLETED_FILE} 2>/dev/null; then
            continue
        fi
        echo ${shot} >> ${SHOT_LIST}
    done

elif [ "${MODE}" = "list" ]; then
    # List mode: read from file
    if [ ! -f "${SHOT_LIST_FILE}" ]; then
        echo "ERROR: Shot list file not found: ${SHOT_LIST_FILE}"
        rm -f ${SHOT_LIST}
        exit 1
    fi

    # Read shots from file, skip already completed
    while IFS= read -r shot; do
        # Skip empty lines and comments
        [[ -z "$shot" || "$shot" =~ ^[[:space:]]*# ]] && continue

        # Skip if already completed
        if grep -q "^${shot}$" ${COMPLETED_FILE} 2>/dev/null; then
            continue
        fi

        echo ${shot} >> ${SHOT_LIST}
    done < "${SHOT_LIST_FILE}"
fi

TOTAL_SHOTS=$(wc -l < ${SHOT_LIST})

if [ ${TOTAL_SHOTS} -eq 0 ]; then
    echo "No shots to process (all completed or none in range)"
    rm -f ${SHOT_LIST}
    exit 0
fi

echo "Total shots to process: ${TOTAL_SHOTS}"

# Split into batches
BATCH_NUM=0
SHOT_INDEX=0

while [ ${SHOT_INDEX} -lt ${TOTAL_SHOTS} ]; do
    BATCH_NUM=$((BATCH_NUM + 1))
    BATCH_FILE="batch_${BATCH_NUM}.txt"

    # Extract batch
    START_LINE=$((SHOT_INDEX + 1))
    END_LINE=$((SHOT_INDEX + BATCH_SIZE))

    sed -n "${START_LINE},${END_LINE}p" ${SHOT_LIST} > ${BATCH_FILE}

    BATCH_SHOTS=$(wc -l < ${BATCH_FILE})

    # Wait if queue is full
    while true; do
        RUNNING_JOBS=$(squeue -u $USER -h -t running,pending -r | wc -l)

        if [ ${RUNNING_JOBS} -lt ${MAX_SUBMIT_LIMIT} ]; then
            break
        fi

        # Get completion stats
        COMPLETED_COUNT=$(wc -l < ${COMPLETED_FILE})
        FAILED_COUNT=$(wc -l < ${FAILED_FILE})

        echo "Queue full: ${RUNNING_JOBS}/${MAX_SUBMIT_LIMIT} | Completed: ${COMPLETED_COUNT}/${TOTAL_SHOTS} | Failed: ${FAILED_COUNT}"
        sleep 30
    done

    # Submit batch as job array
    echo "Submitting batch ${BATCH_NUM} with ${BATCH_SHOTS} shots..."

    JOB_ID=$(sbatch --parsable \
        --array=1-${BATCH_SHOTS} \
        --output=jobs/job_%A_%a.out \
        --error=jobs/job_%A_%a.err \
        --export=ALL,BATCH_FILE=${BATCH_FILE},CONFIG_FILES="${CONFIG_FILES}",OUTPUT_DIR=${OUTPUT_DIR},NODE_PATHS_DIR=${NODE_PATHS_DIR},COMPLETED_FILE=${COMPLETED_FILE},FAILED_FILE=${FAILED_FILE} \
        read_mds.sh)

    echo "Submitted batch ${BATCH_NUM} as job ${JOB_ID}"

    # Save state
    echo "${BATCH_NUM}:${BATCH_FILE}:${JOB_ID}" >> ${STATE_FILE}

    SHOT_INDEX=$((SHOT_INDEX + BATCH_SHOTS))

    # Brief pause between submissions
    sleep 2
done

echo ""
echo "========================================="
echo "All batches submitted (${BATCH_NUM} batches total)"
echo "Monitoring progress..."
echo "========================================="

# Monitor until completion
while true; do
    RUNNING_JOBS=$(squeue -u $USER -h -t running,pending -r | wc -l)
    COMPLETED_COUNT=$(wc -l < ${COMPLETED_FILE})
    FAILED_COUNT=$(wc -l < ${FAILED_FILE})

    echo "Jobs running: ${RUNNING_JOBS} | Completed: ${COMPLETED_COUNT}/${TOTAL_SHOTS} | Failed: ${FAILED_COUNT}"

    if [ ${RUNNING_JOBS} -eq 0 ]; then
        echo "All jobs finished"
        break
    fi

    sleep 60
done

echo ""
echo "========================================="
echo "Final Summary"
echo "========================================="
echo "Total shots requested: ${TOTAL_SHOTS}"
echo "Successfully completed: $(wc -l < ${COMPLETED_FILE})"
echo "Failed: $(wc -l < ${FAILED_FILE})"
echo "========================================="

# Cleanup
rm -f ${SHOT_LIST}
rm -f ${STATE_FILE}
rm -f batch_*.txt

echo "Done at: $(date)"
