#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

module load mdsplus

# Configuration
CHUNK_SIZE=100

# Globus configuration
ENABLE_GLOBUS=true  # Set to false to disable Globus transfer
GLOBUS_SOURCE_ENDPOINT="20749357-d221-43c6-bbc4-79691e6776b8"
GLOBUS_DEST_ENDPOINT="544b12dc-cb3d-11e9-939b-02ff96a5aa76"
GLOBUS_DEST_PATH="/scratch/gpfs/EKOLEMEN/big_d3d_data/d3d_time_series_data/"

# Get shot number
SHOT_NUMBER=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${BATCH_FILE})

if [ -z "${SHOT_NUMBER}" ]; then
    echo "ERROR: Could not get shot number for task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "========================================="
echo "Job started at: $(date)"
echo "Shot number: ${SHOT_NUMBER}"
echo "Config files: ${CONFIG_FILES}"
echo "Chunk size: ${CHUNK_SIZE}"
echo "========================================="

OUTPUT_FILE="${OUTPUT_DIR}/${SHOT_NUMBER}.h5"
TOTAL_FAILED_CHUNKS=0

# Process each config file sequentially
for CONFIG_FILE in ${CONFIG_FILES}; do
    echo ""
    echo "========================================="
    echo "Processing config: ${CONFIG_FILE}"
    echo "========================================="

    if [ ! -f "${CONFIG_FILE}" ]; then
        echo "ERROR: Config file not found: ${CONFIG_FILE}"
        TOTAL_FAILED_CHUNKS=$((TOTAL_FAILED_CHUNKS + 1))
        continue
    fi

    # Extract server
    SERVER=$(grep "^server:" ${CONFIG_FILE} | cut -d: -f2- | xargs)
    echo "Server: ${SERVER}"

    # Create flat list: each line is "tree_name|signal_line"
    TMP_FLAT_LIST=$(mktemp)

    awk '
    /^  [a-zA-Z0-9_]+:$/ {
        current_tree = $1
        sub(/:$/, "", current_tree)
        next
    }
    /^    - / {
        if (current_tree != "") {
            print current_tree "|" $0
        }
    }
    ' ${CONFIG_FILE} > ${TMP_FLAT_LIST}

    TOTAL_SIGNALS=$(wc -l < ${TMP_FLAT_LIST})
    NUM_CHUNKS=$(( (TOTAL_SIGNALS + CHUNK_SIZE - 1) / CHUNK_SIZE ))

    echo "Total signals: ${TOTAL_SIGNALS}"
    echo "Processing in ${NUM_CHUNKS} chunks"
    echo "========================================="

    FAILED_CHUNKS=0

    for (( chunk=0; chunk<NUM_CHUNKS; chunk++ )); do
        CHUNK_NUM=$((chunk + 1))
        START_LINE=$(( chunk * CHUNK_SIZE + 1 ))
        END_LINE=$(( (chunk + 1) * CHUNK_SIZE ))

        echo ""
        echo "Processing chunk ${CHUNK_NUM}/${NUM_CHUNKS} (signals ${START_LINE}-${END_LINE})..."

        # Extract chunk of signals
        CHUNK_DATA=$(sed -n "${START_LINE},${END_LINE}p" ${TMP_FLAT_LIST})

        if [ -z "${CHUNK_DATA}" ]; then
            echo "  Chunk is empty, skipping..."
            continue
        fi

        # Count signals in chunk
        CHUNK_SIGNAL_COUNT=$(echo "${CHUNK_DATA}" | wc -l)
        echo "  Chunk contains ${CHUNK_SIGNAL_COUNT} signals"

        # Create config for this chunk
        CONFIG_BASE=$(basename ${CONFIG_FILE} .yaml)
        CONFIG_FILE_CHUNK="config_${CONFIG_BASE}_${SHOT_NUMBER}_chunk${CHUNK_NUM}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yml"

        cat > "${CONFIG_FILE_CHUNK}" << EOF
shot_numbers:
  - ${SHOT_NUMBER}

trees:
EOF

        # Group signals by tree and add to config
        echo "${CHUNK_DATA}" | awk -F'|' '
        {
            tree = $1
            signal = $2
            if (tree != current_tree) {
                if (current_tree != "") {
                    # Print accumulated signals for previous tree
                    for (i = 0; i < sig_count; i++) {
                        print signals[i]
                    }
                }
                # Start new tree
                current_tree = tree
                print "  " tree ":"
                sig_count = 0
            }
            signals[sig_count++] = signal
        }
        END {
            # Print last tree signals
            if (sig_count > 0) {
                for (i = 0; i < sig_count; i++) {
                    print signals[i]
                }
            }
        }
        ' >> "${CONFIG_FILE_CHUNK}"

        # Add output file and server
        cat >> "${CONFIG_FILE_CHUNK}" << EOF

out_filename: ${OUTPUT_FILE}
server: ${SERVER}
EOF

        # Run read_mds
        echo "  Running read_mds..."
        read_mds -c ${CONFIG_FILE_CHUNK}
        EXIT_CODE=$?

        if [ ${EXIT_CODE} -eq 0 ]; then
            echo "  ✓ Chunk ${CHUNK_NUM}/${NUM_CHUNKS} completed successfully"
            rm -f ${CONFIG_FILE_CHUNK}
        else
            echo "  ✗ Chunk ${CHUNK_NUM}/${NUM_CHUNKS} FAILED (exit code: ${EXIT_CODE})"
            echo "  Config preserved: ${CONFIG_FILE_CHUNK}"
            FAILED_CHUNKS=$((FAILED_CHUNKS + 1))
        fi
    done

    rm -f ${TMP_FLAT_LIST}

    echo ""
    echo "========================================="
    echo "Config ${CONFIG_FILE} summary:"
    echo "  Total signals: ${TOTAL_SIGNALS}"
    echo "  Total chunks: ${NUM_CHUNKS}"
    echo "  Failed chunks: ${FAILED_CHUNKS}"
    echo "========================================="

    TOTAL_FAILED_CHUNKS=$((TOTAL_FAILED_CHUNKS + FAILED_CHUNKS))
done

# Overall summary
echo ""
echo "========================================="
echo "Overall processing summary for shot ${SHOT_NUMBER}:"
echo "  Configs processed: ${CONFIG_FILES}"
echo "  Total failed chunks: ${TOTAL_FAILED_CHUNKS}"
echo "========================================="

# Check overall success
if [ ${TOTAL_FAILED_CHUNKS} -eq 0 ]; then
    if [ -f "${OUTPUT_FILE}" ] && [ -s "${OUTPUT_FILE}" ]; then
        echo "SUCCESS: All configs completed, output file: ${OUTPUT_FILE}"

        (
            flock -x 200
            if ! grep -q "^${SHOT_NUMBER}$" ${COMPLETED_FILE} 2>/dev/null; then
                echo "${SHOT_NUMBER}" >> ${COMPLETED_FILE}
            fi
        ) 200>${COMPLETED_FILE}.lock

        # ============================================
        # GLOBUS TRANSFER SECTION
        # ============================================
        if [ "${ENABLE_GLOBUS}" = true ]; then
            echo ""
            echo "========================================="
            echo "Starting Globus transfer..."

            OUTPUT_FILENAME=$(basename "${OUTPUT_FILE}")
            GLOBUS_SOURCE_PATH="${OUTPUT_FILE#/cscratch/}"

            echo "Transferring: ${OUTPUT_FILENAME}"
            echo "Source path: ${GLOBUS_SOURCE_PATH}"
            echo "Dest path: ${GLOBUS_DEST_PATH}${OUTPUT_FILENAME}"

            TRANSFER_TASK_ID=$(globus transfer \
                --preserve-mtime \
                --label "Auto-transfer ${OUTPUT_FILENAME} $(date +%Y%m%d-%H%M%S)" \
                --jmespath 'task_id' \
                --format unix \
                --notify off \
                "${GLOBUS_SOURCE_ENDPOINT}:${GLOBUS_SOURCE_PATH}" \
                "${GLOBUS_DEST_ENDPOINT}:${GLOBUS_DEST_PATH}${OUTPUT_FILENAME}")

            TRANSFER_EXIT_CODE=$?
            echo "Transfer exit code: ${TRANSFER_EXIT_CODE}"

            if [ ${TRANSFER_EXIT_CODE} -eq 0 ]; then
                echo "Transfer submitted: Task ID ${TRANSFER_TASK_ID}"
                echo "Waiting for transfer to complete..."

                globus task wait "${TRANSFER_TASK_ID}" --timeout 7200 --polling-interval 30

                if [ $? -eq 0 ]; then
                    echo "✓ Transfer completed successfully!"
                    echo "Deleting local file to free up space..."

                    rm -f "${OUTPUT_FILE}"

                    if [ $? -eq 0 ]; then
                        echo "✓ Local file deleted: ${OUTPUT_FILE}"

                        TRANSFER_LOG="${OUTPUT_DIR}/globus_transfers.log"
                        echo "$(date '+%Y-%m-%d %H:%M:%S') | ${SHOT_NUMBER} | ${OUTPUT_FILENAME} | TRANSFERRED_AND_DELETED" >> ${TRANSFER_LOG}
                    else
                        echo "✗ WARNING: Could not delete local file"
                    fi
                else
                    echo "✗ Transfer failed or timed out"
                    echo "Local file preserved: ${OUTPUT_FILE}"
                fi
            else
                echo "✗ Transfer submission failed with exit code ${TRANSFER_EXIT_CODE}"
                echo "Check: endpoint IDs, paths, and activation status"
            fi
            echo "========================================="
        else
            echo ""
            echo "========================================="
            echo "Globus transfer disabled - file retained locally"
            echo "File location: ${OUTPUT_FILE}"
            echo "========================================="
        fi
        # ============================================
        # END GLOBUS TRANSFER SECTION
        # ============================================

        echo "Job completed successfully at: $(date)"
        exit 0
    else
        echo "ERROR: Output file missing or empty: ${OUTPUT_FILE}"
        TOTAL_FAILED_CHUNKS=1
    fi
fi

echo "ERROR: ${TOTAL_FAILED_CHUNKS} chunk(s) failed for shot ${SHOT_NUMBER}"

(
    flock -x 200
    if ! grep -q "^${SHOT_NUMBER}$" ${FAILED_FILE} 2>/dev/null; then
        echo "${SHOT_NUMBER}" >> ${FAILED_FILE}
    fi
) 200>${FAILED_FILE}.lock

echo "Job failed at: $(date)"
exit 1
