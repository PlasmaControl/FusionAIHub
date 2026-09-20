### Task B4: `labeler.jobstats` Frontier backend

**Files:**
- Modify: `src/labeler/jobstats.py` (read it fully first: `JobStats`, `parse_jobstats`, `has_utilisation`, `__main__`)
- Create: `scripts/slurm_frontier/_gpu_sampler.sh`
- Test: `tests/labeler/test_jobstats_frontier.py`

**Interfaces:**
- Produces: `parse_frontier(sacct_text: str, rocm_samples: str | None) -> JobStats`; CLI `python -m labeler.jobstats <jobid>` auto-detects Frontier when `jobstats` is not on PATH and `sacct` is.
- `_gpu_sampler.sh` writes `$ROOT/runs/slurm/<jobid>.gpu.csv` with lines `epoch_s,gpu_pct,vram_used_mb,vram_total_mb` every 30 s (`rocm-smi --showuse --showmemuse --csv`).

- [ ] **Step 1: Failing tests**

```python
# tests/labeler/test_jobstats_frontier.py
from labeler import jobstats

SACCT = """JobID|Elapsed|AllocCPUS|ReqMem|MaxRSS|TotalCPU|State
123|01:00:00|56|100G||56:00:00|COMPLETED
123.0|01:00:00|56||40G|50:24:00|COMPLETED
"""
GPU = "epoch_s,gpu_pct,vram_used_mb,vram_total_mb\n1,80,32000,65536\n2,60,32000,65536\n"

def test_parse_frontier_reads_cpu_and_gpu_utilisation():
    s = jobstats.parse_frontier(SACCT, GPU)
    assert s.job_id == "123"
    assert round(s.cpu_pct) == 90          # 50.4 h TotalCPU / (1 h * 56 cores)
    assert round(s.cpu_mem_pct) == 40      # 40G / 100G
    assert round(s.gpu_pct) == 70          # mean of samples
    assert round(s.gpu_mem_pct) == 49      # 32000/65536
    assert jobstats.has_utilisation(s)

def test_parse_frontier_without_gpu_samples_is_cpu_only():
    s = jobstats.parse_frontier(SACCT, None)
    assert s.gpu_pct is None and s.gpu_mem_pct is None
```
Adapt field names to the real `JobStats` attributes after reading the class; keep the numbers.

- [ ] **Step 2: Run to fail** → `AttributeError: parse_frontier`.

- [ ] **Step 3: Implement `parse_frontier` and the auto-detect**

Parse `sacct -P -o JobID,Elapsed,AllocCPUS,ReqMem,MaxRSS,TotalCPU,State` output: batch row supplies Elapsed/AllocCPUS/ReqMem; the `.0` step supplies MaxRSS/TotalCPU (use the max over steps). Reuse `parse_duration`, `parse_size`, `parse_req_mem`. GPU: mean of `gpu_pct`, and `max(vram_used)/vram_total`. In `main()`, when `shutil.which("jobstats") is None and shutil.which("sacct")`, run the `sacct` command above for the job id and read `$SHOT_DESIGN_DATA_ROOT/runs/slurm/<jobid>.gpu.csv` if present.

- [ ] **Step 4: Sampler script**

```bash
#!/bin/bash
# Background GPU sampler for Frontier jobs: source _shot_design_common.sh first, then
#   bash "$REPO/scripts/slurm_frontier/_gpu_sampler.sh" "$ROOT/runs/slurm/$SLURM_JOB_ID.gpu.csv" &
out="$1"; echo "epoch_s,gpu_pct,vram_used_mb,vram_total_mb" > "$out"
while true; do
  use=$(rocm-smi --showuse --csv 2>/dev/null | awk -F, 'NR==2{print $2}')
  mem=$(rocm-smi --showmemuse --csv 2>/dev/null | awk -F, 'NR==2{print $2}')
  tot=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, 'NR==2{print int($2/1048576)}')
  echo "$(date +%s),${use:-0},${mem:-0},${tot:-0}" >> "$out"; sleep 30
done
```
Add the sampler line (backgrounded, killed at exit with `trap 'kill %1' EXIT`) to `shot_design_encode.sh` from B3.

- [ ] **Step 5: Run to pass, commit**

```bash
git add src/labeler/jobstats.py scripts/slurm_frontier/_gpu_sampler.sh scripts/slurm_frontier/shot_design_encode.sh tests/labeler/test_jobstats_frontier.py
git commit -m "labeler: jobstats Frontier backend from sacct + rocm-smi samples"
```

