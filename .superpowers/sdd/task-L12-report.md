# Task L12 report

Implemented in `/scratch/gpfs/nc1514/FusionAIHub-L`, branch `recommender-L12`.
**Two GPU pilots ran, each the first 20 shots. No production job was submitted.**
Both allocations completed, but both gates are **FAIL (exempt)** and both have
15 successful shots / 5 shot errors. The six-worker pilot is almost twice as
fast; neither configuration meets the production utilisation requirements.

Throughout this report:

```bash
REPO=/scratch/gpfs/nc1514/FusionAIHub-L
ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
cd "$REPO"
```

## Deliverables and scope decisions

- `scripts/labelmaker/tokeye_masks.sbatch`: one node, task and GPU per element,
  CPU binding, explicit chunk/rank/world, phase3 CUDA Python, single-thread
  numerical libraries, readonly text, no concurrent index writes, AMP and
  240-second shot timeout. The README at its top gives the measured sizing.
- `scripts/labelmaker/tokeye_text_subset.sh`: login-node CPU pre-pass over the
  whole list, plus a private shared copy of the unchanged site jobstats client.
- `scripts/labelmaker/tokeye_masks_afterok.sbatch`: CPU rebuild followed by
  `jobstats_check.py` under the labelmaker pixi environment, with raw captures
  and `runs/slurm/jobstats.json`. Numbered `srun` steps preserve payload MaxRSS.
- `tests/labelmaker/test_tokeye_sbatch.py`: parses the required flags/ordering,
  checks prefetch >= workers and CPUs = workers + 2, checks staging/accounting
  steps, and runs shell validation/invalid-resource cases without submissions.
- `docs/LABELMAKER.md`, “Running the events job”: the three commands, polling,
  gates, failure handling, measurements and binding no-production stop rule.
- Additive driver instrumentation retains per-PID prep/tail VmHWM in
  `totals.worker_rss_gib`; the existing peak-worker field still covers both pools.
  No model, mask, event-detection or scheduling algorithm was changed.
- **Write-boundary decision:** the canonical `$ROOT/events_index.parquet` is
  outside the permitted directories. Added `--index-out` for rebuilds only;
  the companion writes `$ROOT/events/events_index.parquet`. The canonical
  index remains untouched and existing consumers of it remain stale until a
  later task authorises publishing there. Tests pin this distinction.
- Corpus, checkpoints and label tables were read-only. Runtime products,
  accounting captures and caches are under `$ROOT/{masks,events,text,runs}`;
  working files are in this worktree or `/tmp`. No production sizing was tested
  by padding time, reserving dummy memory or lowering thresholds.

Read before implementation: the brief, plan §8, spec Appendix A5, driver and
both CLI help pages, L10 Re-review 2, CPU encode precedent and old AE sbatch.
Current Slurm documentation was fetched with Context7 and checked against the
[whole-array dependency documentation](https://slurm.schedmd.com/job_array.html),
[CPU binding documentation](https://slurm.schedmd.com/cpu_management.html) and
[scontrol reference](https://slurm.schedmd.com/scontrol.html).

## Exact execution sequence and every job ID

### Pre-pass 1: login node, no SLURM job

```bash
mkdir -p "$ROOT/runs/slurm/l12"
head -20 "$ROOT/recommender_v1.txt" > "$ROOT/runs/slurm/l12/pilot20.txt"
REPO=$PWD SHOT_FILE="$ROOT/recommender_v1.txt" /usr/bin/time -v \
  bash scripts/labelmaker/tokeye_text_subset.sh \
  > "$ROOT/runs/slurm/l12/prepass.log" 2>&1
```

Exit 0: covered the whole 500-shot list; zero added records because the subset
already existed. Wall 11.21 s, user 2.43 s, system 0.50 s. Both subsequent run
JSONs have `text_subset_missing=[]`, `text_subset=readonly` and `index=skipped`.

### Pilot 1: array 2931065, element 2931065_0

```bash
REPO=$PWD SHOT_FILE="$ROOT/runs/slurm/l12/pilot20.txt" N_CHUNKS=1 \
  PREP_WORKERS=6 PREFETCH=6 TILE_BATCH=96 \
  sbatch --parsable --nodelist=stellar-m01g5 --array=0-0%1 \
  --cpus-per-task=8 --mem=24G --time=00:20:00 \
  scripts/labelmaker/tokeye_masks.sbatch
```

The original script is preserved at `runs/slurm/l12/pilot1.sbatch`.
Initial requests were exploratory: L10's worker-memory envelope, six workers,
96-tile cap and prefetch equal to workers. Final defaults were resized later.

Queue investigation on the **same pending job**, with no additional pilot:

```bash
scontrol update JobId=2931065 ReqNodeList=
scontrol update JobId=2931065 Gres=gpu:1
```

The first removed the pin. The second printed `2931065_0: Access/permission
denied`; no retry was made. Subsequent state nevertheless displayed
`TresPerNode=gres/gpu:1` as well as the existing per-task request, and sacct
confirmed exactly one allocated GPU. The original request is preserved in
`l12/pilot1-before-gres-update.txt`; the running allocation is in
`l12/pilot1-running-job.txt`. The final script declares both consistent
one-GPU constraints explicitly. The precise cause of the queue delay was not
established, so the request change is not claimed as a proven scheduler fix.

A **test-only** scheduling query was also made, not a submitted job:

```bash
REPO=$PWD SHOT_FILE="$ROOT/runs/slurm/l12/pilot20.txt" \
  sbatch --test-only --nodelist=stellar-m01g5 --gres=gpu:1 --array=0-0%1 \
  --cpus-per-task=8 --mem=24G --time=00:20:00 \
  scripts/labelmaker/tokeye_masks.sbatch
```

It predicted hypothetical `Job 2931066` at `2026-09-13T22:27:40` on m01g5.
Immediately querying `scontrol show job 2931066` returned Invalid job id: this
was not an allocation, submission or third pilot and has no gate. The real
job started much earlier, at **2026-09-13 19:38:20 EDT**.

Cold-node evidence: `pilot1-node-before.txt` shows m01g5 idle, zero allocated
CPUs/memory, 498990 MB free of 500000 MB. The allocation's initial nvidia-smi
reports **A100-PCIE-40GB, 40960 MiB total, 0 MiB used, 0% GPU utilisation**.
No model/corpus warm-up ran there. Shared GPFS caches were not flushed or
claimed cold; this is a fresh compute-node/model allocation, not a claim of
cluster-wide cache eviction. The GPU UUID ends `ecd966cdac0e`.

Polled `squeue` every 20 seconds through departure (trace
`l12/2931065.squeue.log`, LEFT_QUEUE at 19:41:15). Sacct: **COMPLETED, 0:0,
161 seconds**, end 19:41:01.

### Companion 1: 2931072, afterok rebuild + initial gate

Submitted only after observing pilot 1 leave squeue:

```bash
REPO=$PWD sbatch --parsable --dependency=afterok:2931065 \
  scripts/labelmaker/tokeye_masks_afterok.sbatch 2931065 --pilot
```

At this revision the script requested serial / 1 CPU / 2G / 10 minutes. Site
policy routed it to **stellar-debug/all**, node stellar-i10n19. It completed
19:41:53–19:47:08, **315 s**, exit 0. Rebuilt **130 rows from 18/18 event files**:
15 successful pilot shots plus three pre-existing event files, as the driver's
whole-root rebuild contract requires.

Its initial gate was **FAIL (exempt)** with CPU/GPU undetermined because
`jobstats` is absent from compute-node PATH. The 300-second wait could not fix
that. This first verdict is preserved verbatim at
`l12/pilot1-compute-gate-undetermined.json`, and stdout at `2931072.out`.
After observing 2931072 leave squeue (`l12/2931072.squeue.log`), the required
module gate and the companion's own gate ran on the login node:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src PIXI_CACHE_DIR=/tmp/l12-pixi \
  pixi run --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
  -e labelmaker python -m labelmaker.jobstats --job-id 2931065 --pilot \
  --preserve-dir "$ROOT/runs/slurm" --out "$ROOT/runs/slurm/jobstats.json"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src PIXI_CACHE_DIR=/tmp/l12-pixi \
  pixi run --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
  -e labelmaker python -m labelmaker.jobstats --job-id 2931072 --pilot \
  --preserve-dir "$ROOT/runs/slurm" --out "$ROOT/runs/slurm/jobstats.json"
```

Corrected pilot verdict: **FAIL (exempt)**, CPU 27.1%, CPU-memory 19.6%,
GPU 6.9%, GPU-memory 94.7%. Companion verdict: **FAIL (exempt)**, CPU 1.2%,
sampled memory 1.0%, GPUs N/A. Both CLI exits are 0 because of `--pilot`, not
because thresholds passed. Gate logs: `l12/pilot1-gate.log` and
`l12/pilot1-companion-gate.log`.

### Diagnosis, companion repair and pre-pass 2

Inference wall time is not CUDA-kernel time: `masks.infer` includes host tiling,
CPU conversion/concatenation and stitching. The GPU figure is 6.9% despite
76.027 s in that phase (49.8% of driver wall). Prep wait is 21.3%, describe
19.3%, and tail wait only 1.6%. Removing prep wait alone cannot turn 6.9% into
70%; the host work inside and between inference calls needs profiling and
restructuring. No CUDA-event/transfer profile was collected, so these timings
do not prove an exact split of kernel, transfer and host-stitch costs.

All **182 saved blocks contain only 10–55 tiles**, below the 96 cap (derived
from saved `_meta.n_cols` and `masks.n_tiles`; `pilot1/block_sizes.json`). Thus
raising the batch cap would do no additional batching on this sample. GPU
memory already meets its gate, while torch peak allocated memory is only
17.647 GiB: the 37.9-GB device-memory peak includes memory not counted as live
torch tensors, such as allocator cache. It is not evidence of GPU compute use.
The tail pool is well overlapped and prefetch is not below the worker count.

CPU 27.1% on eight cores suggested trying fewer reserved workers/cores. The
single permitted adjustment tests **1 prep worker / prefetch 2 / 3 CPUs / 9G**,
keeping the same shots, passes, model, AMP, batch and one tail worker. Nine GiB
exceeds pilot 1's 6.438-GiB MaxRSS by 39.8%; no below-peak memory sizing.

SSH probes to stellar-amd and stellar-vis2 stalled and were terminated; no
remote gate ran. Instead the pre-pass now stages the **unchanged** four site
client files from `/usr/local/jobstats` under `runs/slurm/jobstats-client`,
private permissions (directory 2700 inherited setgid, files 600, executable
700). The site external-database setting is disabled. Byte hashes of originals
and copies are preserved at `l12/jobstats-client.sha256`; no client/config is
committed or installed into an environment. The companion prepends this path.
Numbered srun steps also remedy the first companion's batch-only RSS ambiguity.

```bash
REPO=$PWD SHOT_FILE="$ROOT/recommender_v1.txt" /usr/bin/time -v \
  bash scripts/labelmaker/tokeye_text_subset.sh \
  > "$ROOT/runs/slurm/l12/prepass2.log" 2>&1
```

Exit 0, again all 500 shots / zero added records, 2.61 s wall, 95% CPU.

### Pilot 2: array 2931084, element 2931084_0

```bash
REPO=$PWD SHOT_FILE="$ROOT/runs/slurm/l12/pilot20.txt" N_CHUNKS=1 \
  PREP_WORKERS=1 PREFETCH=2 TILE_BATCH=96 \
  sbatch --parsable --nodelist=stellar-m01g5 --array=0-0%1 \
  --cpus-per-task=3 --mem=9G --time=00:04:00 \
  scripts/labelmaker/tokeye_masks.sbatch
```

Script snapshot: `l12/pilot2.sbatch` (includes `--gres=gpu:1`). Node m01g5,
same A100 with initial 0 MiB used / 0% GPU. Fresh model process, but the same
node and shots had run earlier, so this second result is **not an independently
cold-filesystem replicate**. That possible warm-cache advantage did not prevent
its slowdown.

The four-minute request was an underestimate: 161 s × 1.3 rounded up did not
allow for the reduced-worker slowdown. At observed ~20 s per shot, attempted
once:

```bash
scontrol update JobId=2931084 TimeLimit=00:10:00
```

Slurm denied it, and the limit stayed four minutes. No retry or replacement
job was submitted. The cluster's existing **OverTimeLimit=5 min** allowed the
job to finish at **19:58:23**, after **303 s**, despite its four-minute request.
Sacct reports **COMPLETED / 0:0**, not TIMEOUT. This was discovered while
monitoring; production sizing does **not** depend on this grace period.
Trace: `l12/2931084.squeue.log` through LEFT_QUEUE.

### Companion 2: 2931085, corrected afterok rebuild + gate

Submitted after observing 2931084 leave squeue:

```bash
REPO=$PWD sbatch --parsable --dependency=afterok:2931084 \
  scripts/labelmaker/tokeye_masks_afterok.sbatch 2931084 --pilot
```

Snapshot: `l12/pilot2-afterok.sbatch`. Requested serial / 1 CPU / 512M /
6 minutes; site routed to stellar-debug/all on stellar-i10n19. **COMPLETED /
0:0, 7 s wall**. Rebuilt 130 rows from the same 18 files and successfully ran
the staged site client inside the labelmaker pixi environment. GPU gate:
**FAIL (exempt)**, CPU 39.2%, CPU-memory 35.6%, GPU 3.8%, GPU-memory 94.7%.

After polling this companion out of squeue (`l12/2931085.squeue.log`), ran:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src PIXI_CACHE_DIR=/tmp/l12-pixi \
  pixi run --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
  -e labelmaker python -m labelmaker.jobstats --job-id 2931084 --pilot \
  --preserve-dir "$ROOT/runs/slurm" --out "$ROOT/runs/slurm/jobstats.json"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src PIXI_CACHE_DIR=/tmp/l12-pixi \
  pixi run --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
  -e labelmaker python -m labelmaker.jobstats --job-id 2931085 --pilot \
  --preserve-dir "$ROOT/runs/slurm" --out "$ROOT/runs/slurm/jobstats.json"
```

Logs: `l12/pilot2-gate.log`, `l12/pilot2-companion-gate.log`. The companion's
own gate is **FAIL (exempt)**: CPU undetermined, memory 52.0% via sacct, GPUs
N/A. Its seven-second run is too short for sampled jobstats: raw output falls
back to seff (CPU 57.14%, memory 52.03%). Those seff figures are disclosed,
not silently substituted into the gate. The job was not padded to get samples.

## Measurements and gate verdicts

The produced and read ledger is **`$ROOT/runs/slurm/jobstats.json`**, with all
four actual job IDs. Each has preserved `<id>.jobstats.txt` and `<id>.sacct.txt`
in that directory. All four final verdicts are `passed=false, exempt=true`.

| Job | CPU | CPU-memory used by gate | GPU | GPU-memory | Verdict |
|---|---:|---:|---:|---:|---|
| 2931065_0 | 27.1% | 19.6%, sampled | 6.9% | 94.7% | FAIL (exempt) |
| 2931072 | 1.2% | 1.0%, sampled | N/A | N/A | FAIL (exempt) |
| 2931084_0 | 39.2% | 35.6%, sampled | 3.8% | 94.7% | FAIL (exempt) |
| 2931085 | undetermined | 52.0%, sacct fallback | N/A | N/A | FAIL (exempt) |

Both memory measurements, with units retained as printed:

| Job | Request | jobstats sampled memory | sacct MaxRSS | MaxRSS in GiB | Gate's sacct % |
|---|---:|---|---:|---:|---:|
| 2931065_0 | 24G | 4.7 GB / 19.6% | 6750340K (.0) | 6.438 | 26.8% |
| 2931072 | 2G | 21.1 MB / 1.0% | 400244K (.batch) | 0.382 | undetermined* |
| 2931084_0 | 9G | 3.2 GB / 35.6% | 4135400K (.0) | 3.944 | 43.8% |
| 2931085 | 512M | unavailable: too short | 272804K (.0) | 0.260 | 52.0% |

\* L11 refuses a batch-only MaxRSS as the fallback; the first companion did
run its payload in `.batch`, so the raw peak is recorded here regardless. Its
arithmetic ratio is 19.09%, but that is not the tool's accepted measurement.
The corrected companion uses numbered steps. Its seff memory is 266.41 MB,
derived from sacct, not a second sampled measurement. A missing sampled
measurement is explicitly unavailable, not zero.

Each GPU reports maximum **37.9 GB / 40 GB (94.7%)** device memory, versus
**17.647440 GiB** torch peak allocated. Host VmHWM is not cgroup memory: shared
library pages make summing workers inappropriate as the allocation charge.

| Driver measurement | Pilot 1 | Pilot 2 |
|---|---:|---:|
| Requested / returned shots | 20 / 20 | 20 / 20 |
| Successful / errored shots | 15 / 5 | 15 / 5 |
| Tiles (includes errored shots' inferred blocks) | 7478 | 7478 |
| Completed blocks in successful products | 182 | 182 |
| Events in successful products | 14073 | 14073 |
| Driver elapsed | 152.72 s | 299.22 s |
| SLURM elapsed | 161 s | 303 s |
| Driver tiles/s | 48.97 | 24.99 |
| Tiles / inference-phase second (not kernel benchmark) | 98.36 | 98.67 |
| Amortised wall / attempted shot | 7.64 s | 14.96 s |
| prep_wait_s | 32.528 | 184.106 |
| infer_s | 76.027 | 75.789 |
| describe_s | 29.537 | 26.513 |
| finish_s (overlapped) | 36.231 | 36.064 |
| tail_wait_s | 2.401 | 2.440 |
| Parent VmHWM | 2.589 GiB | 2.310 GiB |
| Worst worker VmHWM | 2.431 GiB | 2.450 GiB |

Per-worker VmHWM from the run JSON:

| Pilot | Pool | PID | GiB |
|---|---|---|---:|
| 1 | prep | 940428 | 0.951 |
| 1 | prep | 940455 | 0.951 |
| 1 | prep | 940456 | 0.951 |
| 1 | prep | 940457 | 0.951 |
| 1 | prep | 940458 | 0.951 |
| 1 | prep | 940459 | 0.951 |
| 1 | tail | 940425 | 2.431 |
| 2 | prep | 941558 | 0.951 |
| 2 | tail | 941555 | 2.450 |

Per-shot driver wall times below include each shot's overlapped finish time.
They therefore do not sum to the driver's elapsed time; the amortised figures
above are elapsed/20. Both runs have exactly the same status for every shot.
Full per-shot prep/infer/describe/finish/tail-wait timings are in each preserved
`pilot{1,2}/run.json` and `measurements.json`.

| Shot | Status in both | Pilot 1 wall s | Pilot 2 wall s | Prep wait 1 s | Prep wait 2 s |
|---|---|---:|---:|---:|---:|
| 185786 | ok | 13.37 | 16.53 | 2.809 | 7.464 |
| 185955 | ok | 11.04 | 20.28 | 1.614 | 10.654 |
| 185956 | ok | 10.87 | 19.63 | 1.600 | 10.628 |
| 185962 | error | 8.22 | 17.17 | 1.492 | 10.811 |
| 185980 | error | 8.23 | 17.08 | 1.708 | 10.740 |
| 185982 | error | 8.26 | 16.89 | 1.673 | 10.719 |
| 185996 | ok | 10.44 | 19.10 | 1.591 | 10.591 |
| 186015 | ok | 10.91 | 19.83 | 1.506 | 10.731 |
| 186034 | ok | 10.90 | 19.95 | 1.521 | 10.808 |
| 186036 | ok | 10.62 | 19.66 | 1.538 | 10.835 |
| 186055 | ok | 8.30 | 14.03 | 1.537 | 7.474 |
| 186059 | ok | 11.00 | 19.82 | 1.708 | 10.737 |
| 186090 | error | 6.04 | 11.63 | 1.704 | 7.442 |
| 186114 | ok | 7.85 | 13.55 | 1.501 | 7.306 |
| 186116 | ok | 8.28 | 13.91 | 1.570 | 7.411 |
| 186118 | ok | 8.59 | 14.22 | 1.570 | 7.469 |
| 186194 | ok | 8.41 | 13.97 | 1.510 | 7.247 |
| 186196 | error | 6.24 | 11.62 | 1.454 | 7.155 |
| 186223 | ok | 8.11 | 13.99 | 1.326 | 7.336 |
| 186224 | ok | 10.37 | 19.52 | 1.596 | 10.549 |

## Per-source completion and errors

All 20 expected source paths were checked in each pilot. **Only 15 files exist;
five are missing**, for **185962, 185980, 185982, 186090, 186196**. Thus the
literal 20-source-file target was not achieved. Missing files are not counted
as zero-error observations. The existing 15 source files contain only their
own pilot's run ID; copies from both pilots are preserved before overwrite.
Source-count tables are identical across pilots:

| Source | ok rows per pilot | skipped rows per pilot | error rows per pilot |
|---|---:|---:|---:|
| actuator | 75 | 0 | 0 |
| dalpha_lh | 0 | 15 | 0 |
| ece_sawtooth | 15 | 0 | 0 |
| elm_clock | 15 | 0 | 0 |
| nbi_counter | 0 | 15 | 0 |
| qh_flattop | 0 | 15 | 0 |
| qh_proxy | 15 | 0 | 0 |
| text | 15 | 0 | 0 |
| tokeye_track | 182 | 74 | 0 |
| tokeye_transient | 15 | 0 | 0 |

These are producer/block rows, not event counts or independent shots.
Zero source error rows do **not** contradict the five shot-level errors: those
shots never wrote their sources file. There are no database source rows for
these shots; absence from a positive-only curated table is not a negative.

Skip reasons per pilot: dalpha_lh 15 (co2 absent, `(4,1)` data); nbi_counter 15
(no canonical Ip in the corpus path); qh_flattop 15 (no Ip-derived flat-top);
tokeye_track 30 (fallback not needed) + 44 (group absent). Each of the 15
successful shots has ece_sawtooth, elm_clock, qh_proxy, text and transient
producer rows marked ok. An ok proxy row does not repair its skipped flat-top
source or establish a QH event.

The five shot errors are `ValueError: t1_s must not exceed t_cov1_s`.
Four show `6.14391994539734` versus `6.143147945404053`; 186090 shows
`6.144175945395114` versus the same coverage end. This is an existing event
coverage-boundary failure, reproduced with both scheduling settings. It was
not patched as part of L12's scheduling/measurement task. It must be repaired
and the five missing products completed before production expansion.

## Diagnosis outcome and unsubmitted production sizing

One worker is **prep-bound**: prep wait grows by 151.578 s, while total wall
grows by 146.50 s and inference remains effectively unchanged. It saves about
6% in sacct total CPU time (395.505 -> 370.404 CPU-s) but nearly doubles GPU
reservation time. Its modest CPU-efficiency increase still misses 70% and
GPU utilisation falls. Keep the **completed six-worker measurement** as the
sizing basis. No third pilot is permitted or submitted.

**Proposal only — do not submit under the current gates:** remaining 480 shots,
**8 contiguous chunks × 60 shots × 1 task/GPU, array 0-7%2**. At most two
simultaneous GPU tasks means <=2 nodes and 2 GPUs, within gpu-stellar's measured
`gres/gpu=8,node=2` and `MaxJobsPU=2` limits. Per task: **8 CPUs = 6 prep + 2,
prefetch 6, tail 1, batch 96, AMP, 9G, 00:11:00**.

- Memory: max recorded pilot-1 RSS 6.438 GiB × 1.30 = 8.37 -> 9G. Keep both
  measurements visible: jobstats 4.7 GB predicts only **52.2% sampled memory**,
  whereas sacct 6.438/9 predicts 71.5%. The gate uses the sampled number, so
  this is a **memory MISS**, not a claimed pass from cherry-picking MaxRSS.
- Time: 161/20 × 60 = 483 s per task, ×1.30 = 627.9 s -> **11 minutes**.
  Four waves at concurrency two imply **32.2 minutes expected GPU-phase wall**,
  about **41.9 minutes with the requested 30% margin** (44 minutes if every
  element uses its fully rounded request), excluding queue time and pre/post
  CPU work. Same contention/shot-mix assumptions as the pilot; not measured
  production throughput. Existing five shot errors and the narrow first-20
  sample limit confidence in extrapolation to the remaining years/modalities.
- Expected GPU-task gates: **CPU ~27.1% MISS; sampled CPU-memory ~52.2% MISS;
  GPU ~6.9% MISS; GPU-memory ~94.7% likely MET** on similar A100-40GB blocks.
  Larger/different blocks could change both memory footprints. Batch 96 has
  no benefit beyond 55 tiles on this sample; it is not a general VRAM guarantee.
- The afterok companion remains **1 CPU / 512M / 6 minutes**. This retains
  31% headroom over the first 0.382-GiB peak; the corrected 0.260-GiB peak
  projects only 52% memory. CPU is undetermined by jobstats on the 7-second
  pilot (seff 57.14%); for 480 files its sampled CPU efficiency is unmeasured.
  Its **CPU gate may be undetermined or missed; memory likely missed**. Do not
  pad the job or shrink below the larger observed peak just to manufacture a
  pass. A future design could separate the accounting wait onto the login
  node, while still measuring/gating the useful CPU rebuild.

What would fix the blockers: first repair the coverage clipping and wire the
missing Ip/flat-top producers required by the evidence contract. Profile host
stitching/serialization and device transfers, then consider preparing complete
batches ahead of GPU work and pooling compatible blocks across shots. That is
an untested architectural change requiring output-identity validation, not a
claim that more prep workers or a larger batch flag will meet 70%. The two
measured configurations do not establish an all-four-gates production size.

The remaining-shot list was prepared and verified as exactly 480 lines under
`runs/slurm/l12/remaining480.txt`. The exact future command sequence below
was **not executed** (its initial pre-pass is included to make the recipe
self-contained). It must omit `--pilot` for production:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub-L
export REPO=$PWD
export LABELMAKER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
ROOT=$LABELMAKER_ROOT
mkdir -p "$ROOT/runs/slurm/l12"
tail -n +21 "$ROOT/recommender_v1.txt" > "$ROOT/runs/slurm/l12/remaining480.txt"

# 1. Login-node pre-pass for the whole 500-shot list and site client staging.
SHOT_FILE="$ROOT/recommender_v1.txt" bash scripts/labelmaker/tokeye_text_subset.sh

# 2. PROPOSAL ONLY: the remaining 480 shots, eight chunks, two concurrent tasks.
JOBID=$(SHOT_FILE="$ROOT/runs/slurm/l12/remaining480.txt" N_CHUNKS=8 \
  PREP_WORKERS=6 PREFETCH=6 TILE_BATCH=96 \
  sbatch --parsable --partition=gpu --qos=gpu-stellar --nodes=1 --ntasks=1 \
  --gpus-per-task=1 --gres=gpu:1 --cpus-per-task=8 --mem=9G \
  --time=00:11:00 --array=0-7%2 scripts/labelmaker/tokeye_masks.sbatch)

# 3. One dependent rebuild + production gate over every element, no exemption.
CHECKID=$(sbatch --parsable --partition=serial --nodes=1 --ntasks=1 \
  --cpus-per-task=1 --mem=512M --time=00:06:00 \
  --dependency=afterok:"$JOBID" scripts/labelmaker/tokeye_masks_afterok.sbatch "$JOBID")

# Observe completion and gate the companion too; do not omit this CPU job.
while [[ -n $(squeue -h -j "$CHECKID" -o %i) ]]; do sleep 20; done
PYTHONPATH="$REPO/src" pixi run --manifest-path \
  /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
  python -m labelmaker.jobstats --job-id "$CHECKID" --wait-for-data 300 \
  --preserve-dir "$ROOT/runs/slurm" --out "$ROOT/runs/slurm/jobstats.json"
```

On a failed GPU array, afterok cannot run: manually gate that array after it
leaves squeue and cancel its pending companion. No failed/absent gate permits
expansion. The five failed first-20 shots remain a separate prerequisite and
are not disguised as part of the “remaining 480” completion claim.

## Verification and reproducibility

Final full suite, on the final functional code:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src PIXI_CACHE_DIR=/tmp/l12-pixi \
  pixi run --frozen --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
  -e labelmaker python -m pytest tests/labelmaker -q -W error
```

**1431 passed, 2 skipped in 125.01 s, exit 0.** Log:
`$ROOT/runs/slurm/l12/final-suite.log` (also `/tmp/l12-final-suite.log`). A
third-party XRootD `client/finalize.py:46` Torch FutureWarning is printed during
interpreter shutdown after pytest restores warning filters; it is not a test
failure. Pixi's non-frozen, explicitly requested gate commands also print the
existing `httpx` extra warning; gate results are preserved separately.

**The two known MAIN-checkout `test_events_databases` failures caused by the
user's uncommitted CSV move were not fixed or touched.** With this worktree's
PYTHONPATH and committed CSVs they do not reproduce: targeted database suite
**38 passed in 2.39 s** (`/tmp/l12-databases.log`). The initial unrestricted-
thread baseline was stopped after 18 minutes when the newer full suite had
already passed; no baseline pass is claimed for that stopped run.

Test-first evidence: missing scripts/index option caused six failures before
implementation (`/tmp/l12-red.log`); per-PID RSS assertions failed in all three
pool combinations before the mapping existed (`/tmp/l12-rss-red.log`); client
staging and numbered srun regression assertions each failed before their fixes
(`/tmp/l12-client-red.log`, `/tmp/l12-srun-red.log`). Final scheduler tests:
**6 passed**. Bash syntax is checked by those tests. The tiny golden-data
helper's two pre-existing ruff violations were corrected (context-managed
file open and f-string) without regenerating data or touching database tests.

Ruff command (labelmaker pixi environment):
`ruff check src/labelmaker/events/driver.py scripts/labelmaker tests/labelmaker`
— **All checks passed**. `git diff --check` is clean.

Pilot data were read back from their JSON, not inferred from log headlines.
`runs/slurm/l12/pilot{1,2}/` preserves each `run.json`, all 15 existing source
parquets, and `measurements.json`; missing source paths are listed explicitly.
The measurement script is preserved at `runs/slurm/l12/measure_pilot.py`.
Original driver JSONs are `runs/events/tokeye-2931065_c0of1_r0.json` and
`runs/events/tokeye-2931084_c0of1_r0.json`. Their git SHA is the pre-commit base
3993bbc: the pilots ran the L12 worktree edits before committing, with unchanged
model/event logic. Final commits carry the measured evidence and required
`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` trailer.

Final checklist: three scripts exist and satisfy the brief's execution flags;
exactly two <=20-shot GPU pilots ran; every actual job was gated after completion;
jobstats.json was produced and read; all four GPU utilisation metrics, both
host-memory measurements (or explicit unavailability for the short CPU job),
per-worker RSS, timings, source completion and errors are above; suite and ruff
are green; production is a proposal only. The branch and worktree are retained.
