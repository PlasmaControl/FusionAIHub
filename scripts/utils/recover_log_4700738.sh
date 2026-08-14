#!/bin/bash
# Background watchdog: poll job 4700738, copy its open log fds from
# the compute node's /proc until the job exits. The directory entry
# for these logs was lost on lustre but slurm_script + srun on
# frontier10185 still hold the inode open via fd 1/2.
#
# Re-copying every 60s gives us a near-real-time mirror at the
# original paths; the final iteration after the job state turns
# non-RUNNING captures the last lines flushed before slurmstepd
# closes the fds (after which the orphan inode is reclaimed by
# lustre and the content is gone forever).
#
# Designed to be launched with nohup + disown so it survives the
# user's session closing.

set -u

PARENT_JOB=4700738
NODE=frontier10185
PID=410768  # slurm_script PID on frontier10185 (verified 2026-06-01)
REPO=/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub
DST_OUT="${REPO}/logs/4700738_e2e_stage1_d1024_48L.out"
DST_ERR="${REPO}/logs/4700738_e2e_stage1_d1024_48L.err"
WATCH_LOG="${REPO}/logs/recovery_watch_4700738.log"

cp_once() {
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$NODE" "
        if [ -e /proc/${PID}/fd/1 ]; then cp /proc/${PID}/fd/1 ${DST_OUT}; fi
        if [ -e /proc/${PID}/fd/2 ]; then cp /proc/${PID}/fd/2 ${DST_ERR}; fi
    " 2>>"$WATCH_LOG"
}

echo "[$(date)] recovery watchdog started for job ${PARENT_JOB} on node ${NODE} (slurm_script pid ${PID})" >> "$WATCH_LOG"

while true; do
    state=$(squeue -j "$PARENT_JOB" -h -o "%T" 2>/dev/null)
    if [ -z "$state" ]; then
        echo "[$(date)] job ${PARENT_JOB} no longer in queue — final copy attempt" >> "$WATCH_LOG"
        cp_once
        break
    fi
    if [ "$state" != "RUNNING" ]; then
        echo "[$(date)] job ${PARENT_JOB} state=${state} — final copy attempt" >> "$WATCH_LOG"
        cp_once
        break
    fi
    cp_once
    sleep 60
done

# Sizes after final copy.
echo "[$(date)] done. Final sizes:" >> "$WATCH_LOG"
ls -la "$DST_OUT" "$DST_ERR" >> "$WATCH_LOG" 2>&1
