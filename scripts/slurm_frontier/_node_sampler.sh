#!/bin/bash
# Per-node sampler for SLURM training jobs.
#
# Designed to be launched as a side srun step via --overlap so it runs
# concurrently with the main srun without stealing GPUs. Writes one line
# per node per SAMPLER_INTERVAL seconds (default 60) with:
#   timestamp host ram=used/total_GB_PCT% gpu_busy=PCT% vram=PCT%
#
# Cost: rocm-smi + free + awk = ~50ms per sample; at 60s interval that is
# ~0.08% of one CPU per node. Negligible vs training workload.
#
# Output stream goes to the file the launcher redirects stdout to —
# typically logs/${SLURM_JOB_ID}_sampler.log.

ROCM_SMI="${ROCM_SMI:-/opt/rocm-7.1.1/bin/rocm-smi}"
INTERVAL="${SAMPLER_INTERVAL:-60}"

while :; do
    ts=$(date +%FT%T)
    host=$(hostname -s)

    ram=$(free -g | awk '/^Mem:/ {printf "%d/%d_GB_%d%%", $3, $2, $3*100/$2}')

    # Mean GPU busy% across the 8 GCDs visible on this node.
    gpu=$("$ROCM_SMI" --showuse 2>/dev/null | awk '
        /GPU use \(%\)/ { sum += $NF; n++ }
        END { if (n) printf "%.0f", sum/n; else print "NA" }
    ')

    # Mean VRAM utilization across GCDs. rocm-smi --showmeminfo vram emits
    #   GPU[N]: VRAM Total Memory (B): <bytes>
    #   GPU[N]: VRAM Total Used Memory (B): <bytes>
    # one pair per GCD. Compute used/total per GCD then average.
    vram=$("$ROCM_SMI" --showmeminfo vram 2>/dev/null | awk '
        /VRAM Total Used Memory \(B\)/ { used [jdx++] = $NF; next }
        /VRAM Total Memory \(B\)/      { total[idx++] = $NF }
        END {
            for (k = 0; k < idx && k < jdx; k++) {
                if (total[k]+0 > 0) { pct += used[k]*100.0/total[k]; n++ }
            }
            if (n) printf "%.0f", pct/n; else print "NA"
        }
    ')

    echo "$ts $host ram=$ram gpu_busy=${gpu}% vram=${vram}%"
    sleep "$INTERVAL"
done
