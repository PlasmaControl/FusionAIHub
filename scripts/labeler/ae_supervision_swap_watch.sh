#!/bin/bash
# Poll a submitted array, moving >30-minute pending tasks to shared GPU 0.
# Usage: bash scripts/labeler/ae_supervision_swap_watch.sh JOB_ID SUBMIT_ISO_TIME [HEAD_PID]
set -euo pipefail
JOB_ID=${1:?array job ID required}
DEADLINE=$(( $(date -d "${2:?submit timestamp required}" +%s) + 1800 ))
REPO=/scratch/gpfs/nc1514/FusionAIHub-r4-aeswap
export TMPDIR=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4/tmp/aeswap
export LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
export LABELER_LABEL_TABLES=/scratch/gpfs/nc1514/FusionAIHub/data/events
export LABELER_NO_FETCH=1 PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT="$LABELER_ROOT/round4/aeswap"
PYTHON=${AESWAP_PYTHON:-$LABELER_ROOT/envs/phase3/bin/python}
cd "$REPO"
ARMS=(legacy dense threeway)
HEAD_PID=${3:-}
HEAD_EXTERNAL=0
FALLBACK_STARTED=0
if [[ -n $HEAD_PID ]]; then
    HEAD_EXTERNAL=1
    FALLBACK_STARTED=1
fi

read_queue() {
    # Query the user queue: a finished array may no longer be a valid job ID.
    squeue -h -r -u "$(id -un)" -o '%i %t %R' | \
        awk -v id="$JOB_ID" '$1 == id || index($1, id "_") == 1'
}

while true; do
    date -Is
    if ! QUEUE=$(read_queue); then
        echo "queue query failed; keeping active training and retrying" >&2
        sleep 60
        continue
    fi
    echo "$QUEUE"
    COMPLETE=0
    for TASK in {0..8}; do
        ARM=${ARMS[$((TASK % 3))]}
        SEED=$((TASK / 3))
        if [[ -f "$OUT/models/$ARM/seed-$SEED/run.json" ]]; then
            COMPLETE=$((COMPLETE + 1))
        fi
    done
    echo "completed $COMPLETE / 9"
    if (( COMPLETE == 9 )); then
        if [[ -n $HEAD_PID && $HEAD_EXTERNAL == 0 ]]; then
            wait "$HEAD_PID"
        fi
        exit 0
    fi
    if (( FALLBACK_STARTED == 0 && $(date +%s) > DEADLINE )); then
        # Pending array elements can disappear into a compressed sacct record
        # after cancellation. Include previously cancelled, incomplete tasks
        # so restarting this watcher safely resumes the fallback.
        PENDING=()
        for TASK in {0..8}; do
            ID="${JOB_ID}_${TASK}"
            ARM=${ARMS[$((TASK % 3))]}
            SEED=$((TASK / 3))
            STATE=$(awk -v id="$ID" '$1 == id {print $2}' <<< "$QUEUE")
            if [[ ! -f "$OUT/models/$ARM/seed-$SEED/run.json" ]] && \
                [[ $STATE == PD || -z $STATE ]]; then
                PENDING+=("$ID")
            fi
        done
        FALLBACK_TASKS=()
        for ID in "${PENDING[@]}"; do
            # Restrict cancellation to pending state, avoiding a start-time race.
            STATE=$(read_queue | \
                awk -v id="$ID" '$1 == id {print $2}')
            if [[ $STATE == PD ]]; then
                scancel -t PENDING "$ID"
                sleep 1
            fi
            STATE=$(read_queue | \
                awk -v id="$ID" '$1 == id {print $2}')
            if [[ -z $STATE ]]; then
                FALLBACK_TASKS+=("${ID##*_}")
                echo "fallback $ID absent from queue after cancellation at $(date -Is)"
            fi
        done
        FALLBACK_STARTED=1
        if (( ${#FALLBACK_TASKS[@]} > 0 )); then
            (
                for TASK in "${FALLBACK_TASKS[@]}"; do
                    ARM=${ARMS[$((TASK % 3))]}
                    SEED=$((TASK / 3))
                    [[ ! -f "$OUT/models/$ARM/seed-$SEED/run.json" ]] || continue
                    echo "head start $ARM seed $SEED $(date -Is)"
                    CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
                        scripts/labeler/ae_supervision_swap.py train \
                        --supervision "$ARM" --seed "$SEED" \
                        > "$OUT/head-$ARM-seed$SEED.log" 2>&1
                    bash "$LABELER_ROOT/scratch/bin/tmpsweep.sh"
                    echo "head finished $ARM seed $SEED $(date -Is)"
                done
            ) > "$OUT/head-sequence.log" 2>&1 &
            HEAD_PID=$!
            echo "head sequence PID $HEAD_PID: ${FALLBACK_TASKS[*]}"
        fi
    fi
    if [[ -n $HEAD_PID ]] && ! kill -0 "$HEAD_PID" 2>/dev/null; then
        if (( HEAD_EXTERNAL == 0 )); then
            wait "$HEAD_PID" || exit 1
        fi
        HEAD_PID=
    fi
    if [[ -z $QUEUE && -z $HEAD_PID ]]; then
        echo "no active training, but incomplete runs; inspect logs" >&2
        exit 1
    fi
    # Poll every three minutes; individual sleeps stay within one minute.
    for _ in 1 2 3; do sleep 60; done
done
