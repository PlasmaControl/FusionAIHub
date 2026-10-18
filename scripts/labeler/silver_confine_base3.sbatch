#!/bin/bash
#SBATCH --job-name=silver-confine-base3
#SBATCH --partition=gpu
#SBATCH --qos=gpu-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=7300M
#SBATCH --time=00:06:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labeler/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labeler/runs/slurm/%j.out

# confine-ours base3 (2026-10-14: five run-day folds, three seeds each averaged; macro F1
# 0.909 on confine-cnn's protocol, outputs/labeler/confinement/ours/protocol_base3.json)
# over a named shot list, for the review page's silver drafts, as ONE batched GPU job:
#   scripts/labeler/confinement_ours_shots.py label --variant base3 \
#       --shots-file $SHOTS_FILE --name $NAME --out-dir $OUT_DIR --device cuda \
#       --workers $WORKERS --batch $BATCH --check $CHECK
#   -> $OUT_DIR/$NAME.csv + $NAME.meta.json (shot, category, t_start, t_end, confidence,
#      attrs: the layout of extend_confine_ours/roster_review.csv); a shot with no
#      fetched zero-d record also gets $LABELER_ROOT/round4/conf/zerod_disk/<shot>.npz
# WORKERS processes (one a CPU) read and assemble the shots (HDF5 reads, 1 ms binning)
# ahead of the GPU; the GPU takes to_input's clipping and masks and runs the 15 networks
# over BATCH shots at a time, each shot as if alone (forward_alone); CHECK shots are
# recomputed one at a time on the GPU and on the CPU and the difference goes in the meta.
# The default writes $LABELER_LABEL_TABLES/confinement/extend_confine_base3/silver_shots.csv, whose
# directory sorts before extend_confine_ours, so on the review page it outranks the
# 2026-10-03 roster_review.csv where both hold a shot. A curated shot is read by the
# fold that never saw it, any other shot by the mean of the five folds.
# Nothing is fetched here (LABELER_NO_FETCH=1; the script reads disk only): fetch the
# zero-d records first, on the login node under fdp, e.g.
#   flock $R/scratch/<session>/raw_cache.lock pixi run --frozen --no-install -e labeler \
#       fdp run python scripts/labeler/confinement_zerod_fetch.py --shots <shots> --pace 1
# The GPU is sampled every 2 s into $LABELER_ROOT/runs/slurm/<job>.nvsmi.csv; the log
# stamps the shot loop's start and end in the same clock.
#
#   pilot:    SHOTS_FILE=<csv of at most 20 shots> NAME=pilot \
#               OUT_DIR=$LABELER_ROOT/runs/silver_infer/confine_base3/pilot5 \
#               sbatch --mem=12G <this>     (BATCH=16: 20 shots fill one batch)
#   full run: sbatch <this>     (the 450 silver shots, $LABELER_LABEL_TABLES/silver/silver_shots.csv)
#   gate:     cd /scratch/gpfs/nc1514/FusionAIHub && PYTHONPATH=$PWD/src pixi run --frozen \
#               --no-install -e labeler python -m labeler.jobstats --job-id <id> [--pilot]
#
# ------------------------------------------ sizing (2026-10-15/16)
# Full run 2965499 (450 shots, batch 64, --mem 3100M = 1.2 x a 20-shot pilot's 2,592,948 KiB)
# died OUT_OF_MEMORY before its first batch. Cause: eight silver shots (156785, 158019,
# 156790, 158023, 156793, 156796, 176077, 176087) have no fetched zero-d record, so
# assemble_disk -> disk_record bins their raw CO2 trace (15,000,000 samples, a 521 MB raw
# file): zerod.bin_stats peaks at 428 MB and the read at 242 MB, 607 MB in all (tracemalloc,
# shot 156785; a worker's RSS goes 211 -> 735 MB and back to 190 MB). Six of the eight sit
# among the first 32 shots, so up to six workers held that at once, on top of the networks'
# process (~1.2 GB) and the other workers' ~190 MB each. The 20-shot pilots (2965489,
# 2965494) read only 2 of these heavy shots, not six at once, and their batch of 64 never
# filled. A worker's death also left the pool hanging (the job
# sat at 0 % GPU to its limit); read_shots now stops at once (BrokenProcessPool -> kill the
# workers, SystemExit; diag job 2965717: exit 1 after 1.4 s) and reads at most 4 x workers
# shots ahead (the old pool.map submitted all 450).
# Sizing run (the worst case on purpose): pilot 2965722, 20 shots = those 8 heavy shots
# (all eight read at once, one on each of the 8 workers) + 12 others, 8 CPUs, 8 readers,
# batch 16 (one full batch of 16 + 4), check 20, float32: 19 s, sacct MaxRSS of the step
# 6,189,156 KiB (the sum of the processes' RSS, shared pages counted once a process).
#   --mem 7300M = 1.2 x 6,189,156 KiB = 7,253 MiB, rounded up. The full run can still reach
#   the worst case: eight reader peaks of 744 MiB at once (diag job 2965717) plus the
#   ~1.2 GB networks process is about 7,100 MiB (the pilot's sampled 6.2 GB sum is lower),
#   so 7300M leaves a margin of about 3 %, not the 20 % of 1.2 x the pilot.
#   --time 6 min: the whole run is expected in about a minute (the 8-worker read of all 450
#   shots alone took 15.9 s, diag job 2965717), so a hang is cut off near the 5-minute bound.
# Full run 2965724 (450 shots, batch 16, --mem 7300M): 25 s (shot loop 13.3 s: reads 9.5 s,
# networks 2.5 s, segments 1.3 s; set-up 2 s; check 4.6 s), step MaxRSS 4,931,616 KiB, the
# kernel's cgroup peak 4,882 MiB; 433 of 450 shots labelled, 1,641 segments. jobstats
# (plain, the GPU job's gate; --cpu-only is for CPU jobs) has no CPU or GPU rows for a
# 25 s job (it samples every 30 s) and reads
# cpu_mem 66 % (< 70 %): the 7300M is the worst case above, which the real order of the
# shots did not reach. A rerun of the same list could ask 5900M (1.2 x 4,931,616 KiB), but
# if the heavy shots line up the pilot's 6.2 GB can recur, so 7300M stays the default.
# GPU rule: not 50 % busy (the 15 small U-Nets take 0.07-0.4 s a batch; pilot 2965722's
# nvidia-smi over the 5.2 s shot loop: 2 0 0 14 12 12 %), so this job runs under the other
# branch, a tiny model with the whole list in one batched job of at most 5 minutes.
set -euo pipefail
REPO=/scratch/gpfs/nc1514/FusionAIHub
ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labeler
PY=$ROOT/envs/phase3/bin/python   # torch 2.14 + CUDA 12.6
export LABELER_ROOT=$ROOT PYTHONPATH=$REPO/src
export LABELER_LABEL_TABLES="${LABELER_LABEL_TABLES:-/scratch/gpfs/EKOLEMEN/nc1514/events}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE LABELER_NO_FETCH=1
export XDG_CACHE_HOME=$ROOT/runs/slurm/cache/${SLURM_JOB_ID:-none}
SHOTS_FILE=${SHOTS_FILE:-$LABELER_LABEL_TABLES/silver/silver_shots.csv}
NAME=${NAME:-silver_shots}
OUT_DIR=${OUT_DIR:-$LABELER_LABEL_TABLES/confinement/extend_confine_base3}
VARIANT=${VARIANT:-base3}
CPUS=${SLURM_CPUS_PER_TASK:-8}
WORKERS=${WORKERS:-$CPUS}
BATCH=${BATCH:-16}
CHECK=${CHECK:-20}
NVSMI=$ROOT/runs/slurm/${SLURM_JOB_ID:-none}.nvsmi.csv
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader \
    -l 2 > "$NVSMI" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true; rm -rf "$XDG_CACHE_HOME"' EXIT
cd "$REPO"
echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -Is); git $(git rev-parse --short HEAD)"
echo "variant $VARIANT, shots $SHOTS_FILE ($(($(wc -l < "$SHOTS_FILE") - 1)) rows) -> $OUT_DIR/$NAME.csv"
echo "cpus $CPUS, workers $WORKERS, batch $BATCH, check $CHECK; GPU samples -> $NVSMI"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
set +e
srun --cpus-per-task="$CPUS" "$PY" -u scripts/labeler/confinement_ours_shots.py label \
    --variant "$VARIANT" --shots-file "$SHOTS_FILE" --name "$NAME" --out-dir "$OUT_DIR" \
    --device cuda --workers "$WORKERS" --batch "$BATCH" --check "$CHECK"
RC=$?
set -e
echo "finished at $(date -Is), exit $RC"
# the job's peak memory as the kernel charged it (cgroup v1; sacct's MaxRSS sums the processes' RSS)
CG=$(sed -n 's/^[0-9]*:memory:\(.*\)$/\1/p' /proc/self/cgroup)
CG=${CG%%/step_*}
[ -r "/sys/fs/cgroup/memory$CG/memory.max_usage_in_bytes" ] && \
    echo "cgroup $CG peak $(( $(cat /sys/fs/cgroup/memory$CG/memory.max_usage_in_bytes) / 1048576 )) MiB" || true
exit $RC
