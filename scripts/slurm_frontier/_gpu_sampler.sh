#!/bin/bash
# Background GPU sampler for Frontier jobs: source _shot_design_common.sh first, then
#   bash "$REPO/scripts/slurm_frontier/_gpu_sampler.sh" "$ROOT/runs/slurm/$JOB_TAG.gpu.csv" &
# and `trap 'kill %1' EXIT` so it dies with the job. `labeler.jobstats.parse_frontier`
# reads the file back: it is the only GPU utilisation OLCF offers, since there is no
# `jobstats` here. Name the file after the id `sacct` reports (for an array element that
# is <arrayjobid>_<taskid>, NOT $SLURM_JOB_ID), or the gate will not find it.
#
# ONE rocm-smi call per sample, `--showuse --showmeminfo vram --csv`, which prints
#   device,GPU use (%),GFX Activity,VRAM Total Memory (B),VRAM Total Used Memory (B)
#   card0,0,203885505,68702699520,10977280
# behind a WARNING line and a blank one -- so the columns are found BY NAME and the rows
# by their `card` prefix. A fixed `NR==2` would read the blank line and record 0 % for
# the whole run. GPU use is averaged over the sampled GCDs and the memory summed, so a
# job given more than one GCD is measured on all of them.
#
# `-d` is not optional. rocm-smi ignores ROCR_VISIBLE_DEVICES and reports every card on
# the node, so on an exclusive node a one-GCD job would be averaged against seven idle
# siblings and read ~1/8 of the utilisation it actually got -- which is exactly the number
# the gate then judges. `-d` takes one or more indices, so the comma-separated
# ROCR_VISIBLE_DEVICES becomes a space-separated list. Unset (an interactive node, or a
# job that was given the whole node) means sample everything.
out="$1"; echo "epoch_s,gpu_pct,vram_used_mb,vram_total_mb" > "$out"
devices=()
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
  devices=(-d ${ROCR_VISIBLE_DEVICES//,/ })
fi
while true; do
  sample=$(rocm-smi "${devices[@]}" --showuse --showmeminfo vram --csv 2>/dev/null | awk -F, '
    /^device/ {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /GPU use/)                  use_col = i
        if ($i ~ /VRAM Total Memory/)        tot_col = i
        if ($i ~ /VRAM Total Used Memory/)   mem_col = i
      }
      next
    }
    /^card/ { use += $use_col; mem += $mem_col; tot += $tot_col; n++ }
    END { if (n) printf "%.0f,%.0f,%.0f", use / n, mem / 1048576, tot / 1048576 }
  ')
  echo "$(date +%s),${sample:-0,0,0}" >> "$out"; sleep 30
done
