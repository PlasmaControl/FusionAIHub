#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#
# SLURM array worker: fetches one shot from the FDP.
#
# One array task per shot, the shot number read from BATCH_FILE by task id.
# The work itself is delegated to fetch_shot_fast.sh, which stages the shot's
# objects and then reads them locally in two further passes; see that script
# for why the passes are separate processes.
#
# Expects these to be exported by submit_fetch_batches.sh:
#   BATCH_FILE, CONFIG_FILES, OUTPUT_DIR, COMPLETED_FILE, FAILED_FILE,
#   FDP_PROJECT, STAGE_PARENT (optional)
#
# --mem must cover the staging area as well as the process: staging defaults to
# /dev/shm, which is RAM, and a shot needs ~5 GB of it.  Reduce concurrency per
# node rather than raising this, or point STAGE_PARENT at a filesystem.

set -o pipefail

SHOT_NUMBER=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${BATCH_FILE}")

if [ -z "${SHOT_NUMBER}" ]; then
    echo "ERROR: no shot number for array task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Per-job staging directory, so concurrent array tasks cannot collide and a
# killed job leaves nothing behind that another job would reuse.
STAGE="${STAGE_PARENT:-/dev/shm}/fdp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "========================================="
echo "Job started at: $(date)"
echo "Shot number:    ${SHOT_NUMBER}"
echo "Configs:        ${CONFIG_FILES}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "Staging:        ${STAGE}"
echo "========================================="

export FDP_PROJECT
for config in ${CONFIG_FILES}; do
    export CONFIG="${SCRIPT_DIR}/${config}"
    break   # fetch_shot_fast.sh takes one config; loop kept for clarity
done

"${SCRIPT_DIR}/fetch_shot_fast.sh" "${SHOT_NUMBER}" "${OUTPUT_DIR}" "${STAGE}"
EXIT_CODE=$?

rm -rf "${STAGE}"

OUTPUT_FILE="${OUTPUT_DIR}/${SHOT_NUMBER}.h5"

if [ ${EXIT_CODE} -eq 0 ] && [ -s "${OUTPUT_FILE}" ]; then
    echo "SUCCESS: ${OUTPUT_FILE}"
    (
        flock -x 200
        if ! grep -q "^${SHOT_NUMBER}$" "${COMPLETED_FILE}" 2>/dev/null; then
            echo "${SHOT_NUMBER}" >> "${COMPLETED_FILE}"
        fi
    ) 200>"${COMPLETED_FILE}.lock"
    echo "Job completed at: $(date)"
    exit 0
fi

echo "ERROR: shot ${SHOT_NUMBER} failed (exit ${EXIT_CODE})"
(
    flock -x 200
    if ! grep -q "^${SHOT_NUMBER}$" "${FAILED_FILE}" 2>/dev/null; then
        echo "${SHOT_NUMBER}" >> "${FAILED_FILE}"
    fi
) 200>"${FAILED_FILE}.lock"

echo "Job failed at: $(date)"
exit 1
