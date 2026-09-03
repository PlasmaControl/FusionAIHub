#!/bin/bash
# Post-job efficiency report for e2e_stage1 training jobs.
# Usage: scripts/slurm_frontier/job_report.sh <jobid> [<jobid> ...]
#
# Reads SLURM accounting (sacct + seff) and the training log under logs/
# to report:
#   * job state / elapsed
#   * CPU+mem efficiency (seff)
#   * training throughput (wall s/step, median 50-step compute pace,
#     compute / wall efficiency)
#   * validation passes and best val_loss
#   * fault patterns
#   * GPU utilization (only if a logs/<jobid>_gpu.log sampler ran)

set -u

report_one() {
    local JOB="$1"
    local ERR="logs/${JOB}_e2e_stage1.err"
    local OUT="logs/${JOB}_e2e_stage1.out"
    local GPU="logs/${JOB}_gpu.log"

    echo "============================================================"
    echo " Job ${JOB}"
    echo "============================================================"

    sacct -j "$JOB" -o JobID%-18,State,Elapsed,TotalCPU,CPUTime,MaxRSS,NTasks,NNodes,Partition \
        2>/dev/null | head -10

    echo
    echo "-- Derived CPU + memory --"
    sacct -j "$JOB" -P -n -o JobID,Elapsed,TotalCPU,CPUTime,MaxRSS,NTasks,NNodes \
        2>/dev/null | awk -F'|' '
        function tsec(t,    p, d, rest, q, n) {
            if (t == "" || t == "INVALID" || t == "Unknown") return 0
            if (t ~ /-/) { split(t, p, "-"); d=p[1]; rest=p[2] } else { d=0; rest=t }
            sub(/\.[0-9]+$/, "", rest)
            n = split(rest, q, ":")
            if (n == 3) return d*86400 + q[1]*3600 + q[2]*60 + q[3]
            else if (n == 2) return d*86400 + q[1]*60 + q[2]
            else            return d*86400 + q[1]+0
        }
        function memk(m,    v, u) {
            if (m == "" || m == "0") return 0
            if (m ~ /[KMGT]$/) {
                u = substr(m, length(m), 1)
                v = substr(m, 1, length(m)-1) + 0
            } else { v = m + 0; u = "K" }
            if (u == "T") return v*1024*1024*1024
            if (u == "G") return v*1024*1024
            if (u == "M") return v*1024
            return v
        }
        # Pick the srun step (.0) — the real workload step that has both
        # TotalCPU and MaxRSS populated. The job-level row aggregates
        # CPUTime across the whole allocation but has no TotalCPU/MaxRSS.
        $1 ~ /\.0$/ {
            elap_s = tsec($2); tc_s = tsec($3); ct_s = tsec($4); rss_k = memk($5)
            ntasks = $6; nnodes = $7
            if (ct_s > 0)
                printf "  CPU efficiency:    %.1f%%  (TotalCPU=%s / CPUTime=%s)\n", tc_s*100/ct_s, $3, $4
            if (rss_k > 0)
                printf "  Peak task RSS:     %.2f GB  (max single-task RSS across %s tasks on %s nodes)\n", rss_k/1024/1024, ntasks, nnodes
        }
        '

    if [ ! -f "$ERR" ]; then
        echo
        echo "-- log $ERR not found --"
        echo
        return
    fi

    echo
    echo "-- Training throughput --"
    local TMP
    TMP=$(mktemp)
    grep -E "INFO \[rank0\] step [0-9]+/" "$ERR" | while read -r line; do
        local ts_str step ts
        ts_str=$(echo "$line" | awk '{print $1" "$2}' | cut -d, -f1)
        step=$(echo "$line" | grep -oE "step [0-9]+" | awk '{print $2}')
        ts=$(date -d "$ts_str" +%s 2>/dev/null) || continue
        echo "$ts $step"
    done > "$TMP"

    if [ -s "$TMP" ]; then
        local first last ft lt fs ls dt ds wall
        first=$(head -1 "$TMP"); last=$(tail -1 "$TMP")
        ft=${first% *}; fs=${first#* }
        lt=${last% *};  ls=${last#* }
        dt=$((lt - ft)); ds=$((ls - fs))
        if [ "$ds" -gt 0 ]; then
            wall=$(awk -v d="$dt" -v s="$ds" 'BEGIN{printf "%.2f", d/s}')
            echo "  steps ${fs} -> ${ls}  (${ds} steps in ${dt} s)"
            echo "  wall step time: ${wall} s/step"

            awk '
                NR>1 && $2-prev_s==50 && $1-prev_ts<600 { print ($1-prev_ts)/50 }
                { prev_ts=$1; prev_s=$2 }
            ' "$TMP" | sort -n | awk -v wall="$wall" '
                { vals[NR]=$1 }
                END {
                    if (NR==0) exit
                    m = (NR%2==1) ? vals[int(NR/2)+1] : (vals[NR/2]+vals[NR/2+1])/2
                    printf "  median 50-step compute pace: %.2f s/step (%d windows)\n", m, NR
                    if (wall+0 > 0) printf "  throughput efficiency: %.1f%% (compute / wall)\n", m*100/wall
                }'
        else
            echo "  (only one step line in log)"
        fi
    else
        echo "  (no step lines logged)"
    fi
    rm -f "$TMP"

    echo
    echo "-- Validation --"
    local nval
    nval=$(grep -cE "Validation \(MAE" "$ERR" 2>/dev/null || true)
    echo "  passes: ${nval:-0}"
    grep -E "new best val_loss" "$ERR" 2>/dev/null | sed 's/^/  /' | tail -5 || true

    echo
    echo "-- Faults / errors --"
    local f
    f=$(grep -cE "Memory access|HIP error|CUDA error|OOM-Killer|out of memory|^Killed| Killed |Traceback" \
        "$ERR" "$OUT" 2>/dev/null | awk -F: 'BEGIN{s=0} {s+=$2} END{print s}')
    echo "  fault-pattern lines: ${f:-0}"
    if [ "${f:-0}" -gt 0 ]; then
        grep -mE -m3 "Memory access|HIP error|CUDA error|OOM-Killer|out of memory|^Killed| Killed |Traceback" \
            "$ERR" "$OUT" 2>/dev/null | sed 's/^/    /'
    fi

    echo
    echo "-- Sampler (per-node, every 60s) --"
    local SAMPLER="logs/${JOB}_sampler.log"
    if [ -f "$SAMPLER" ]; then
        awk '
            # Sampler line format:
            #   <ts> <host> ram=USED/TOTAL_GB_PCT% gpu_busy=PCT% vram=PCT%
            function num(s) { gsub(/[^0-9]/, "", s); return s+0 }
            $0 ~ /ram=.*gpu_busy=.*vram=/ {
                # Extract numbers from each label
                for (i = 1; i <= NF; i++) {
                    if (match($i, /^ram=/))     ram = num($i)
                    if (match($i, /^gpu_busy=/)) gpu = num($i)
                    if (match($i, /^vram=/))    vram = num($i)
                }
                rsum += ram; gsum += gpu; vsum += vram; n++
                if (ram > rmax) rmax = ram
                if (gpu > gmax) gmax = gpu
                if (vram > vmax) vmax = vram
                # also collect p95 arrays
                rvals[n] = ram; gvals[n] = gpu; vvals[n] = vram
            }
            END {
                if (n == 0) { print "  (sampler log empty or unparseable)"; exit }
                printf "  samples: %d  (across all nodes, combined)\n", n
                printf "  Host RAM:   mean %.0f%%   peak %.0f%%\n", rsum/n, rmax
                printf "  GPU busy:   mean %.0f%%   peak %.0f%%\n", gsum/n, gmax
                printf "  VRAM used:  mean %.0f%%   peak %.0f%%\n", vsum/n, vmax
            }' "$SAMPLER"
    else
        echo "  (no $SAMPLER — sampler block in train_e2e_stage1.sh writes one)"
    fi
    echo
}

if [ $# -eq 0 ]; then
    echo "usage: $0 <jobid> [<jobid> ...]" >&2
    exit 1
fi

for JOB in "$@"; do
    report_one "$JOB"
done
