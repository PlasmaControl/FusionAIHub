# L14-perf report

Worktree `/scratch/gpfs/nc1514/FusionAIHub-L14perf`, branch `recommender-L14perf`.
The original profile used `3cced76004183f1b3eef4a485b9e294a207ab9e5`;
resumed work started at `38a3c13`, incorporating merge base `67f47e3` and C3fix.
The L12 report is gone; comparison numbers come from the binding brief and
`scripts/labelmaker/tokeye_masks.sbatch` header. Runtime products use
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/l14perf/`; the prescribed gate
captures use `runs/slurm/`. One A100 pilot was submitted in the original loop; the authorized second
pilot is recorded in the Fix loop below. Production remains report-only.

## Deliverable 1: profile before implementation

Completed before any production source or test change. Instrumentation script:
`.superpowers/sdd/profile_l14perf.py`; unmodified production source at `3cced76`.
Login node `stellar-vis2`, V100S GPU 1, first three L12 shots
185786, 185955, 185956; both passes; 6 prep workers, prefetch 6, batch cap 96,
AMP. Tail workers **0** for attribution of write costs in this process;
this differs from the overlapped L12/pilot tail and is not a utilization pilot.
Timed CUDA sections synchronize for attribution; their sum is an instrumented
wall split, not uninstrumented throughput. One forward attempted too large a
V100S allocation and used the existing halving retry. Failed-forward time and
concatenation/control overhead remain in `infer_s`'s residual (~1.59 s).
No allocation was resized to manipulate a gate.

| Phase, 38 completed blocks / 1,170 tiles | Before seconds |
|---|---:|
| Prepared-future wait | 6.635684 |
| Host tiling | 0.637067 |
| H2D | 0.280518 |
| Successful forward, synchronized wall | 6.976396 |
| D2H | 1.717956 |
| Host stitch | 7.611698 |
| Describe | 5.263547 |
| Mask write | 2.278039 |
| Events write | 0.230823 |
| Sources write | 0.063069 |
| Total inference, inclusive | 18.814418 |

Per-block table is the `blocks` array in `profile-before/profile.json`.
`torch.profiler` CPU+CUDA trace is `profile-before/trace.json`; operator table
is `profile-before/operators.txt`. CUDA profiler reports pageable D2H 0.617 s
and H2D 0.267 s (includes model setup); these are device activity, not the
synchronized host-wall values above. Nested profiler rows must not be summed.
Source and raw artifacts live under the root above. Command, exit **0**:

```bash
L14_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/l14perf/profile-before
printf 'Resolved root: %s\n' "$L14_ROOT"
mkdir -p "$L14_ROOT/text"
cp /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/text/logs_subset.jsonl "$L14_ROOT/text/"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=1 XDG_CACHE_HOME="$L14_ROOT/cache" \
CUDA_CACHE_PATH="$L14_ROOT/cache/cuda" \
/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/envs/phase3/bin/python -u \
.superpowers/sdd/profile_l14perf.py \
--shot-file /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/l12/pilot20.txt \
--limit 3 --passes wide zoom --root "$L14_ROOT" \
--corpus /scratch/gpfs/EKOLEMEN/foundation_model \
--unet /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/tokeye/big_tf_unet_251210.pt \
--device cuda --tile-batch 96 --amp --prep-workers 6 --prefetch 6 \
--tail-workers 0 --text-subset readonly --no-index --run-id profile-before \
> "$L14_ROOT/profile.log" 2>&1
```

L12 comparison baseline: pilot 2931065_0, A100, 6 prep/6 prefetch/1 tail,
20 shots, 7,478 tiles/152.72 s = 48.97 tiles/s; CPU 27.1%, sampled host memory
4.7 GB/24 GB (19.6%), GPU 6.9%, device memory 37.9 GB/40 GB (94.7%),
sacct MaxRSS 6,750,340K (6.438 GiB), torch peak allocated 17.647 GiB.

## Design rulings and progress

- Ruling: preserve sparse float32 coherent probabilities at threshold-lit pixels
  in the compact result, alongside packed masks and summaries. The brief's
  packed-only suggestion cannot preserve probability-weighted track centroids,
  mean probabilities, or percentile confidence. Components do not grow the
  mask, so unlit coherent and all transient probability floats can be omitted.
  Cost: transfer reduction depends on coherent occupancy; it is not a universal
  32-fold reduction. Identity takes precedence over that approximate estimate.
- Ruling: retain the legacy full-probability inference API as the independent
  `process_shot` reference; optimized driver paths must compare against it.
- Ruling: separate readonly Python/checkpoint inputs from sbatch output ROOT;
  current L12 script incorrectly ties these to the writable root. A plain
  ROOT override must be honored and the resolved root printed before writes.
- Resumed implementation, identity verification, pilot, suites and documentation
  are completed below; the full pilot utilization gate remains FAIL (exempt).

## Resumed deliverable 1: WIP tests and writable root

Resumed at `38a3c13` (current reference includes C3fix). The L12 report is gone;
only the quoted brief/header numbers above are used. Initial worktree was clean.
Both WIP tests were run first, with the real test opted in: **2 failed, exit 1**.
The sbatch test exposed ROOT losing to LABELMAKER_ROOT. The real test exposed
an unsupported `process_shot(index=False)` argument before reaching inference.
Both tests are retained: the first guards output isolation, and the second is
needed for real checkpoint identity. Removed the stale reference-only index
argument and strengthened the real test to compare complete NPZ file bytes and
hashes, in addition to array bytes and exact events/sources frames (including
`intervals` and `min_gap_s`). Compact/pooled API checks await deliverable 2;
the opt-in skip is not identity evidence.

The sbatch now resolves and prints ROOT before execution, lets explicit ROOT
win over activation, and uses independent RUNTIME_ROOT / PHASE3_PYTHON / UNET
inputs. Scheduler contract tests updated for the independent Python path.
Commands (from this worktree, common exports on every pixi command):

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 TMPDIR=/tmp/l14perf
L14PERF_REAL_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/l14perf/identity-wip OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m pytest tests/labelmaker/test_tokeye_sbatch.py::test_scratch_root_separates_outputs_from_readonly_runtime tests/labelmaker/test_l14perf_real_identity.py -q -W error -p no:cacheprovider -s
# Initial: 2 failed, exit 1.
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m pytest tests/labelmaker/test_tokeye_sbatch.py tests/labelmaker/test_l14perf_real_identity.py -q -W error -p no:cacheprovider
# After fix: 7 passed, 1 opt-in skip, exit 0.
```

## Deliverable 2: compact inference and scheduling

Implemented float64 overlap accumulation on the inference device, followed by
float32 conversion, the unchanged 0.2 threshold, packbits in NumPy's bit order,
and float64-divide/float32-cast row and column fractions. The compact payload
includes row-major sparse coherent float32 values. CPU description reconstructs
only the coherent probability map; transient probabilities are unnecessary.
The full-probability `process_shot` inference branch remains the oracle.
Byte identity is conditional: byte-identical when no OOM halving occurs or
when it occurs identically. Different allocation pressure can cause different
forward batch shapes and therefore different AMP rounding.

At TILE 512 / STRIDE 448, at most two tiles overlap any column; division by
two is exact power-of-two scaling, so float32 versus float64 overlap
accumulation is indistinguishable here. Float64 matches the oracle's intent
and provides extra protection; these tests do not prove it is necessary.

Input buffers pool consecutive tiles across channels, passes and shots. CUDA
uses pinned host buffers, nonblocking transfers on a copy stream, readiness
events and record_stream ownership. Device results are compacted on a separate
stream while later forward work runs. Description and finishing can run in the
tail pool; ordered, bounded tail payloads preserve inferred work through a pool
restart. Prep work, waits and memory are measured; sizing follows the AFTER
profile. `--probs-on-host` selects the original schedule and full probabilities.

### Additional CUDA identity finding and required batching constraint

The requested CPU checks did not expose CUDA AMP batch-shape rounding. An
additional one-real-shot V100S check at batch cap 96/AMP found that unrestricted
cross-block forward batches changed thresholded pixels, even though compact
per-block inference was byte-identical to the CUDA oracle. The original and
compact GPU NPZ both hashed to
`8f3a11e12382d68d8e1098497759e5bc0032b1153c4770b7064984d7c1280370`;
unrestricted pooling produced
`9f462bc147a87dcadc176ca83c7491ad7ffd8bc13e74b38ad193f4d8125ac38f`.
The first mismatch was `mhr_00_wide_coh_packed`; counts also differed in the
stored row/column summaries. The unweakened comparison failed, exit 1.
No threshold, probability rounding or tolerance was changed.

Consequently CUDA pools full **input-transfer** batches but reconstructs the
oracle's per-block forward groups on the device. At that historical snapshot CPU retained full cross-block
forward batches; the Fix loop defaults both devices to reference boundaries
and makes CPU cross-block forwards opt-in. This is a deliberate deviation from full CUDA forward pooling
to enforce output identity. It follows PyTorch's documented lack of bitwise
equivalence across batch shapes:
[PyTorch numerical accuracy](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/numerical_accuracy.md).
Copy-stream synchronization follows
[PyTorch CUDA semantics](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/cuda.md),
retrieved with the find-docs skill and Context7 (`library` then `docs`).

### Identity evidence

All synthetic stage comparisons use the CURRENT `pipeline.process_shot`, both
passes, the 14-block fixture, and worker/prefetch pairs (0,1), (1,4), (2,2).
Tests compare complete NPZ bytes and exact events/sources frames after dropping
only run_id/written_at. Sources include intervals JSON and min_gap_s; an
additional gapped-filterscope fixture asserts disjoint intervals explicitly.
The tail-stage test makes any parent-process describe call fail, so identical
outputs also prove description moved to the spawned tail. Three synthetic shots
produce 42 tiles in eight batches of five plus a final two. A separate transfer
probe keeps three full input batches while preserving forward groups 2,3,2,2.
GPU tests cover overlap edges and probabilities one float32 step around the
threshold, plus stream ownership and full-probability equivalence.

Initial compact, pooled and tail stage tests each failed before their required
implementation. After implementation all 12 synthetic stage/settings cases
passed, with these hashes at worktree HEAD `48a5ba4` (uncommitted implementation):

| Product | SHA-256 |
|---|---|
| Synthetic masks NPZ file | `49aebe37b731b8c7f8cc2ca3635904aa6fa69fdf13df974e7eb30be2c65a125e` |
| Synthetic normalized events frame | `7ec5d898589fa00c513c2c893b74a2d6dfbca310e2786d4e2d54fe9a63298bdf` |
| Synthetic normalized sources frame | `6aabdbf4fce6f0f81e4ccd071a9c0ba3b173bd251fbcf8da887d6ad241f2ae1f` |
| Real CPU masks NPZ file | `a31d87e0ed4ac71884b2292d1c8fe2f6826834a271d9e0ad41b42d898d6e71ff` |
| Real CPU masks array digest | `bd5a1d705e2ae656d6497502ca2934cf659cf3b35bcd68e8e660cb036fd8c169` |
| Real CPU normalized events frame | `137b1b43b6fb8fa58e6d6d548cf84d4f20be87834e9dd55d6a6847e39558097f` |
| Real CPU normalized sources frame | `057f0fe0714ddf77392eb5c56b473cec2ebcd0eded92d13bfd85c101e18ac209` |

Real shot 185786 has 10 blocks / 1,663 CPU events. Checkpoint SHA-256:
`4afc3948ba53af40cb2787441b251757f8c77e764784e0aaa2602af0471229a9`.
The fresh oracle plus compact comparison passed in 1006.55 s, exit 0.
Later CPU schedules reuse those completed oracle files at the same HEAD,
asserting their git_sha before comparison; the default test invocation always
runs a fresh oracle. The optional reuse avoids another eight-minute CPU oracle
forward without weakening a product comparison. Frame digests are SHA-256 of
`to_json(orient='table', double_precision=15)` after the two-column drop.
NPZ digests hash the actual file bytes; array digests additionally hash sorted
key names, dtypes, shapes and bytes.

Exact commands, from the worktree (common exports shown once; no pixi install):

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 TMPDIR=/tmp/l14perf
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m pytest tests/labelmaker/test_l14perf_identity.py -q -W error -p no:cacheprovider -s
L14PERF_STAGES=compact L14PERF_REAL_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/l14perf/identity-compact pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m pytest tests/labelmaker/test_l14perf_real_identity.py -q -W error -p no:cacheprovider -s
L14PERF_STAGES=pooled,tail,prep L14PERF_REFERENCE_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/l14perf/identity-compact/legacy L14PERF_REAL_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/l14perf/identity-pooled pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m pytest tests/labelmaker/test_l14perf_real_identity.py -q -W error -p no:cacheprovider -s
CUDA_VISIBLE_DEVICES=1 L14PERF_TEST_DEVICE=cuda CUDA_CACHE_PATH=/tmp/l14perf/cuda-cache /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/envs/phase3/bin/python -m pytest tests/labelmaker/test_l14perf_identity.py::test_device_overlap_and_compaction_are_exact tests/labelmaker/test_l14perf_identity.py::test_copy_stream_pool_matches_full_probability_reference -q -W error -p no:cacheprovider
L14PERF_REAL_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/l14perf/identity-cuda-preserved L14PERF_REAL_DEVICE=cuda L14PERF_REAL_BATCH=96 L14PERF_REAL_AMP=1 CUDA_VISIBLE_DEVICES=1 CUDA_CACHE_PATH=/tmp/l14perf/cuda-cache /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/envs/phase3/bin/python -m pytest tests/labelmaker/test_l14perf_real_identity.py -q -W error -p no:cacheprovider -s
# The rejected unrestricted CUDA experiment used identity-cuda instead.
```

The corrected CUDA AMP real-shot test passed in **80.04 s, exit 0**, across
legacy, compact, pooled input transfers, tail and prep paths (10 blocks /
1,662 events). Every path has these same hashes:

| CUDA AMP product | SHA-256 |
|---|---|
| NPZ file | `8f3a11e12382d68d8e1098497759e5bc0032b1153c4770b7064984d7c1280370` |
| Mask arrays | `391b6a6ab2bb8384ad37c8998da2d250db7de575e7d1f3c88070e30a9b1362b3` |
| Normalized events | `d77b2be84d6ec300298cfdb8f44853f33d66c75cf3a2992af34036ca998b5c15` |
| Normalized sources | `81e3d8cf807915c98d91ed79dfde3a985fdc808fad026c8aba6a21bf28f52c89` |

CPU and CUDA AMP are independently compared against their own same-settings
oracle; cross-device/precision equality is not claimed. The CUDA validation
used the same pinned phase3 Python and physical V100S GPU 1 as the profile.

All three additional real CPU schedules passed in **1501.18 s, exit 0**:
pooled 495.56 s, tail 497.79 s, prep 503.23 s. Their mask-file, mask-array,
events and sources hashes exactly match the CPU table above. Evidence is in
`identity-compact/identity.json`, `identity-pooled/identity.json` and
`identity-cuda-preserved/identity.json` under the L14 runtime root.
The two final CUDA edge/stream tests passed together in 3.82 s, exit 0.
Additional regressions cover a timed-out shot followed by a successful shot,
OOM in transfer and forward, later-block recovery, and multiple ordered tail
workers. Multiple tail workers require `--no-index` to avoid shared-index races;
SBATCH CPU reservations account for prep + tail + parent explicitly.

Pre-profile implementation gate: labelmaker **1,621 passed, 3 skipped**, 214.74 s,
exit **0**; IDEATE **1,252 passed**, 132.69 s, exit **0** (the pre-profile
run, distinct from the final 126.47 s rerun below); required Ruff command
passed, exit **0**. The labelmaker atexit XRootD FutureWarning printed after the
successful summary and did not change its exit code. No commit occurred while
any suite or real identity test was running.

## Deliverable 3: AFTER profile and pilot sizing

Profile completed at `c707b26`, exit **0**. Same physical GPU 1 on
stellar-vis2 (V100S 32 GB), phase3 Python, shots 185786/185955/185956, both
passes, batch cap 96, AMP, 6 prep / 6 prefetch / 0 tail, readonly text, no index.
Both profiles completed 38 blocks / 1,170 tiles / 4,542 events. The original
profile predates C3fix; the AFTER profile includes its required current coverage
contract. Instrumentation synchronizes CUDA for attribution and disables much
of the overlap; these throughput figures are not utilization-pilot measurements.

| Phase | Before seconds | After seconds |
|---|---:|---:|
| Prepared-future wait | 6.635684 | 11.689123 |
| Host tiling / pooled buffer fill | 0.637067 | 0.433324 |
| H2D | 0.280518 | 0.224046 |
| Successful forward, synchronized wall | 6.976396 | 7.791690 |
| D2H | 1.717956 | 0.130861 |
| Stitch (host before, device after) | 7.611698 | 0.204567 |
| Device threshold/pack/statistics | — | 0.254318 |
| Describe | 5.263547 | 6.537106 |
| Mask write | 2.278039 | 3.655924 |
| Events write | 0.230823 | 0.478939 |
| Sources write | 0.063069 | 0.062074 |
| Inference excluding input/plan waits | 18.814418 | 9.370647 |

The AFTER inference total subtracts measured input/plan waits from the pooled
generator time (24.980185 s inclusive), matching the BEFORE scope. Plan wait
was 3.920415 s after; it was not separately captured before. The driver JSON
retains the inclusive pooled timer, so its `infer_s` must not be summed with
`prep_wait_s`. Tail-workers-zero finishing is synchronous even though the
future-result-only `tail_wait_s` is near zero in this schedule; use `finish_s`.
The driver wall was **43.27 -> 44.55 s**, **27.04 -> 26.26 tiles/s**: overall
instrumented throughput did not improve. Exposed prep waits and synchronous
CPU tails dominate after the GPU-side savings. No improvement is claimed for
that whole-run measurement.

D2H was **107,683,288 bytes** versus 2,453,667,840 bytes
for the old tile probabilities (**22.786x fewer**). Device peak allocation was
18.212 GiB after (17.647 GiB before); allocator cache is a separate measurement.
The AFTER run used the existing halving fallback for V100S OOM attempts.
Profiles are `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/l14perf/profile-before/` and `profile-after/`: JSON,
Chrome trace and operator table. The updated profiler records device stitching,
compaction, full-buffer H2D and total prep-worker service time.

Pilot sizing is fixed BEFORE submission, with no threshold changes:

- Measured AFTER prep work: **60.277594 worker-seconds** for 1,170 tiles;
  describe **6.537106 s** plus finishing **12.964 s**. Six prep workers leave
  11.689123 s of observed wait.
- As a conservative A100 service estimate, L12's sampled GPU-active time is
  `152.72 * 0.069 = 10.53768 s` for 7,478 tiles, or approximately 709.64
  tiles/active-second. This is a sizing estimate from sampled utilization,
  not a new A100 kernel benchmark. The 1,170-tile equivalent is 1.648714 s.
- `ceil(60.277594 / 1.648714) = 37` prep workers and
  `ceil((6.537106 + 12.964) / 1.648714) = 12` tail workers; **50 CPUs**
  including the parent, **prefetch 37**, unchanged **tile batch 96**. This
  deliberately supplies CPU capacity for the faster A100; the one pilot must
  determine whether IPC, bandwidth or startup limits it instead.
- **68G memory**: conservative reservation envelope
  `(3.846 parent + 49 * 0.951 worker) * 1.3 = 65.579 GiB`, rounded up.
  This sum is NOT a measured cgroup RSS: shared pages make worker VmHWM
  non-additive. Actual sampled memory and sacct MaxRSS will both be reported.
- **00:07:00**: `44.55 / 3 * 20 * 1.3 = 386.1 s`, rounded up to seven
  minutes. This uses the slower instrumented AFTER wall, with headroom; it
  stays below the 30-minute limit. No allocation was shrunk to pass a gate.

profile-before profile.json SHA-256: `e269f56276707e1598524de09f5a39dd22c0d954f57f96f76d27dd6ebc75754d`.

profile-after profile.json SHA-256: `0f7191167db0268b343cca1b54e5915061f18eb97e750938140c5e042a391569`.

Exact AFTER command (exit 0):

```bash
L14_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/l14perf/profile-after
printf 'Resolved root: %s\n' "$L14_ROOT"
mkdir -p "$L14_ROOT/text"
cp /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/text/logs_subset.jsonl "$L14_ROOT/text/"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 XDG_CACHE_HOME="$L14_ROOT/cache" CUDA_CACHE_PATH="$L14_ROOT/cache/cuda" /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/envs/phase3/bin/python -u .superpowers/sdd/profile_l14perf.py --shot-file /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/l12/pilot20.txt --limit 3 --passes wide zoom --root "$L14_ROOT" --corpus /scratch/gpfs/EKOLEMEN/foundation_model --unet /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/tokeye/big_tf_unet_251210.pt --device cuda --tile-batch 96 --amp --prep-workers 6 --prefetch 6 --tail-workers 0 --text-subset readonly --no-index --run-id profile-after > "$L14_ROOT/profile.log" 2>&1
```

## Deliverable 4: the single A100 pilot

**2931999_0**, source HEAD `5a8386e`, ran on **stellar-m01g5**, NVIDIA
A100-PCIE-40GB. It started 2026-09-14 18:49:56 EDT and finished 18:50:57:
**61 s SLURM wall, COMPLETED, exit 0:0**. The card reported 0 MiB used before
launch. Exactly one `sbatch` command was issued; no companion, second pilot or
production job was submitted. `squeue` emptied and every sacct step completed.

**20/20 shots completed, 244 stored blocks, 7,478 tiles, 26,913 events.** Driver
wall was **53.49 s**, **139.81 tiles/s**, versus L12's 48.97 tiles/s (**2.855x**).
The same tile count was processed; L12 lost five shot tails whereas this run
published all 20. `text_subset_missing` is empty. NPZ block counts and event
counts were checked against every run row, and every sources frame was read
and checked for intervals/min_gap_s. Product hashes and source-status counts
are in `pilot-l14perf/completion-evidence.json`.

| Measurement | L12 pilot 2931065_0 | L14 pilot 2931999_0 |
|---|---:|---:|
| CPU utilization, jobstats detailed | 27.1% | **12.8%** |
| CPU memory, sampled jobstats | 4.7GB / 24GB (19.6%) | **22.4GB / 68GB (32.9%)** |
| GPU utilization, jobstats | 6.9% | **100%** |
| GPU memory, jobstats maximum | 37.9GB / 40GB (94.7%) | **38.6GB / 40GB (96.6%)** |
| sacct MaxRSS | 6,750,340K (6.438 GiB) | **24,748,068K (23.601597 GiB)** |
| sacct MaxRSS / requested host memory | — | **34.7%** |
| Torch peak allocated | 17.647 GiB | **18.212 GiB** |

These are the recorded measurements, not recomputed percentages rounded to
make a gate pass. The jobstats summary bars show 12/33/100/97; the table uses
the detailed values selected by the gate. GPU memory includes allocator cache;
it is not the Torch live-allocation peak. Sampled CPU time is 390 s, whereas
sacct TotalCPU is 454.406 s; those are also distinct observations.

The **GPU >=70% target is met**. The full jobstats verdict remains
**FAIL (exempt)**: CPU 12.8% and CPU-memory 32.9% are below 70%; GPU and
GPU-memory pass. The prescribed `--pilot` gate returned **exit 0** because of
the exemption, not because every check passed. The site report covers a
61-second pilot and does not expose sample count here; 100% is reported as
measured, not extrapolated into a claim of continuous utilization on a long run.

L-A's five former error shots all completed with empty shot-level error fields:

| Shot | Blocks | Events | Status |
|---|---:|---:|---|
| 185962 | 14 | 1,537 | ok |
| 185980 | 14 | 1,421 | ok |
| 185982 | 14 | 1,800 | ok |
| 186090 | 10 | 588 | ok |
| 186196 | 10 | 1,241 | ok |

Their out-of-coverage tracks remain isolated per block in `skipped`; they do
not abort mask writes, independent detectors or source writes. This confirms
the L-A fix under the optimized driver without pretending those rejected
tracks became valid. Scratch roots intentionally lack production feature
files, so feature-dependent sources are honestly skipped, as in the identity
oracle and the profiles; production stores were not modified.

Observed scheduling: prepared-future wait **0.002 s**; pooled inference timer
**39.251 s** (includes planning/input-generator time); description **27.772
worker-seconds**; finishing **134.626 worker-seconds**; tail-result wait
**8.344 s**. Concurrent worker service totals cannot be added to driver wall.
Parent VmHWM was **9.498 GiB**, largest worker **2.576 GiB**. The pool report
contains 21 prep and eight tail PIDs, below the configured maxima of 37/12.
Thus the CPU capacity estimate was conservative: lazy worker creation, I/O,
queue/serialization costs and a short cold run leave much of the 50-core
reservation unused. More workers alone are not evidence of useful CPU demand.

### Original production sizing proposal — withdrawn by F2

The following historical proposal is superseded by the Fix loop. It placed
both pools below measured demand and let one timeout consume the entire
wall request; its memory measurement came from different pool sizes.

GPU performance permits recording the requested **8 x 60 shots**,
`--array 0-7%2`, `N_CHUNKS=8` proposal. It does **not** erase the two failed
production utilization checks. Using the pilot's measured resource use x 1.3:

| Resource | Calculation | Proposed request |
|---|---|---:|
| CPU | `ceil((454.406 TotalCPU seconds / 61 wall seconds) * 1.3)` | **10 CPUs** |
| Host memory | `ceil(23.601597 GiB MaxRSS * 1.3)` | **31G** |
| Time per 60-shot chunk | `ceil(61 s / 20 * 60 * 1.3)` | **00:04:00** |
| GPU | Same one A100, batch cap 96, AMP | **1 A100** |

A matching worker configuration would be **6 prep, prefetch 6, 3 tail, 1
parent**, with readonly text, no shared index, and the same compact/preserved
CUDA batch schedule. This is a **capacity proposal**, not a validated claim
that smaller pools sustain the observed GPU utilization: the AFTER profile
exposed waits at six prep workers, and mean CPU demand does not measure burst
capacity. The controller must review that concurrency risk and the failed
CPU/host-memory gates before treating it as production-ready. The submitted
pilot retained its original 50 CPUs/68G allocation; no hypothetical smaller
allocation was passed back through jobstats and no threshold was changed.
No production command was executed or placed in a submission script.

Submission and gate, exactly (common environment exports as above):

```bash
L14_PILOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/l14perf/pilot-l14perf
printf 'Resolved root: %s\n' "$L14_PILOT"
mkdir -p "$L14_PILOT/text" "$L14_PILOT/slurm"
cp /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/text/logs_subset.jsonl "$L14_PILOT/text/"
REPO=/scratch/gpfs/nc1514/FusionAIHub-L14perf ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/l14perf/pilot-l14perf SHOT_FILE=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/l12/pilot20.txt PREP_WORKERS=37 PREFETCH=37 TAIL_WORKERS=12 TILE_BATCH=96 N_CHUNKS=1 sbatch --parsable --job-name=tokeye-L14perf --cpus-per-task=50 --mem=68G --time=00:07:00 --array=0-0%1 --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/l14perf/pilot-l14perf/slurm/%A_%a.out scripts/labelmaker/tokeye_masks.sbatch
# Returned 2931999. Polled squeue and sacct to completion.
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 TMPDIR=/tmp/l14perf
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m labelmaker.jobstats --job-id 2931999_0 --pilot --preserve-dir /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm --out /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/jobstats.json
# Exit 0; FAIL (exempt), CPU 12.8 / CPU-mem 32.9 / GPU 100 / GPU-mem 96.6.
```

Raw captures are `runs/slurm/2931999_0.jobstats.txt` and
`runs/slurm/2931999_0.sacct.txt`; ledger entry is
`runs/slurm/jobstats.json -> jobs["2931999_0"]`. These prescribed gate captures
are the only task writes outside `runs/l14perf/` in the data root.
Scheduler output, caches and products are inside `pilot-l14perf/`.

Pilot evidence SHA-256:

- `completion-evidence.json`: `039c3762056a0c4df3bc056dad85952f28390c9126e3d898f04b77ec0201105b`
- `2931999_0.jobstats.txt`: `868eb9ae396077962efcabeeab16e8f0b0829d651d1953a34aade42fabda2c07`
- `2931999_0.sacct.txt`: `04bfb5c819900ec1d86dab492f736e641eb72cc9fc4e0c59717b7039c673be38`
- `tokeye-2931999_c0of1_r0.json`: `5ed8c186d4c68bba213eaa2ac6408eca647d756f3db166579d234a9fc1b7bed6`

## Deliverable 5: documentation, deviations and final validation

`docs/LABELMAKER.md` now documents compact device results, sparse descriptor
probabilities, pinned cross-shot input pooling, CUDA forward-boundary identity,
tail scheduling, explicit output roots/read-only runtime inputs, the frozen
worktree environment command, and this pilot's measured result and gate limits.

Material deviations and limits:

- Full CUDA cross-block **forward** pooling was rejected by an additional real
  AMP identity test. Full transfer pooling remains; CUDA forwards retain oracle
  boundaries. Both the failing evidence and the corrected passing evidence are
  retained. This protects the binding output-identity contract.
- Extra one-shot GPU identity smokes were used to catch that failure beyond the
  required CPU checks. They used physical V100S GPU 1 and isolated L14 roots;
  they were not SLURM pilots. Only job 2931999_0 was submitted.
- The initial real CPU oracle was reused read-only at unchanged HEAD for the
  later CPU schedules, with a git_sha check; every file/frame comparison stayed
  exact. Fresh-oracle execution remains the test default.
- The frozen BEFORE profile predates C3fix and was not repeated. AFTER uses
  current coverage semantics, the same data/flags/GPU, and adapted attribution
  for pooled input waits and device compaction. Its serial-tail wall did not
  improve; the actual A100 pilot did.
- Larger prep/tail limits were chosen from the AFTER service measurements.
  The A100 capacity estimate over-reserved CPU and host memory: those two
  production checks failed. Their measured failures remain in the ledger.
  The requested usage-x-1.3 production proposal is explicitly provisional;
  smaller-pool throughput has not been established by this one pilot.
- Scratch feature stores were not populated from production. Feature-dependent
  detectors report skips consistently in oracle, profiles and pilot; no
  production masks, events, labels, features, runs/events or IDEATE data were
  written. Gate captures use the explicitly prescribed runs/slurm exception.

Final required checks ran from this worktree, using the main checkout's existing
environments only. No install and no worktree `.pixi` were created. Commands:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 TMPDIR=/tmp/l14perf
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m pytest tests/labelmaker -q -W error -p no:cacheprovider
# 1621 passed, 3 skipped in 214.74s (0:03:34); EXIT 0.
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m pytest tests/ideate -q -W error -p no:cacheprovider
# 1252 passed in 126.47s (0:02:06); EXIT 0.
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker ruff check src/labelmaker src/ideate scripts/labelmaker tests/labelmaker tests/ideate
# All checks passed; EXIT 0.
```

All source changes were covered by the final labelmaker run; subsequent changes
are profiler/report/documentation only. IDEATE was also rerun on the final
source snapshot, with its exit code captured alongside the log. The
XRootD atexit FutureWarning after the labelmaker summary did not alter exit 0.
Raw suite/identity/gate logs are preserved under `runs/l14perf/evidence/` and
`pilot-l14perf/slurm/`. No commit was made while any suite was running.

`git log --oneline recommender..recommender-L14perf`, captured immediately before
the closing report/documentation commit (the final terminal log includes that
closing commit as well):

```text
17c0b06 labelmaker: record A100 pilot metrics and report-only sizing
5a8386e labelmaker: record L14perf after profile and size the single A100 pilot
c707b26 labelmaker: overlap compact TokEye inference while preserving output bytes
48a5ba4 labelmaker: isolate TokEye output root from runtime inputs
38a3c13 labelmaker: L14perf brief - resume addendum (base 67f47e3 with C3fix coverage contract; WIP tests unverified; mandatory pixi run form; pilot ROOT under runs/l14perf)
67f47e3 Merge branch 'recommender' into recommender-L14perf
5d7deee labelmaker: L14perf WIP snapshot at pause (user, 2026-09-14) - unverified
bba8e8f labelmaker: record L14perf pre-change CUDA profile
```

## Fix loop

### Outcome and scope

Applied F1–F13 in `/scratch/gpfs/nc1514/FusionAIHub-L14perf` on
`recommender-L14perf`. The review and binding fix brief were already committed
at `6978f6e` on entry. Blocking fixes were committed before the **one** new
submission, pilot **2932066_0**, at **56d4521**. It completed all 20 shots but
missed CPU, CPU-memory and GPU utilization: **FAIL (exempt)**. Production
remains unvalidated and was **not submitted**. No further pilot was submitted.

### Findings and changes

Paths below are relative to the worktree; line numbers describe the final source.

| Finding | Change and evidence |
|---|---|
| F1 | `src/labelmaker/events/masks.py:648` retries a failed reference transfer once after `empty_cache`; a second OOM marks only its spans. `masks.py:507` drains failed blocks and releases partial groups across buffers. Regression: `tests/labelmaker/test_l14perf_identity.py:374`, transient and repeated failures, including a split forward group and successful later shots. |
| F2 | Replaced the withdrawn 10 CPU / 6 prep / 3 tail / 31G / four-minute proposal with measurements from exactly the prescribed 12 CPU / 7 prep / 4 tail / prefetch 8 / 32G / 12-minute pilot. Derived capacity is below; the failed utilization gate remains a finding. |
| F3 | `masks.py:709` attributes shared-batch StageTimeout only to expired spans and retries healthy spans; `src/labelmaker/events/driver.py:1080` supplies each shot's remaining time. Regressions at `test_l14perf_identity.py:428` and `:502` cover two shots in a four-tile batch, both default and opt-in CPU schedules. |
| F4 | `masks.py:667` resets reduced forward size on entering each block and stops reduced slices at its boundary. `test_l14perf_identity.py:402` pins recovery inside a pooled buffer. |
| F5 | Conditional identity is explicit in this report, `docs/LABELMAKER.md:494`, and the driver docstring: **byte-identical when no OOM halving occurs or when it occurs identically**. Different OOM decisions can change AMP forward shapes. |
| F6 | Prep failure/death, unreadable corpus, wedged worker, unpicklable tail, task exception/timeout, worker crash and hung tail tests now cover both `pooled=False/True` in `tests/labelmaker/test_tokeye_masks.py` (notably `:654`, `:773`, `:1119`, `:1144`). This exposed an additional pooled task-TimeoutError restart: `driver.py:1022` now distinguishes it from an expired future wait. `driver.py:1591` rejects multiple index writers via argparse before model loading; regression at `test_tokeye_masks.py:1351`. |
| F7 | `driver.py:1032` resubmits tail futures in their existing deque positions. `test_tokeye_masks.py:1314` forces BrokenProcessPool with two workers and mixed completed/pending tails; only the unfinished payload is repeated and rows retain shot order. |
| F8 | `masks.py:568` defaults reference forward groups on **every device**, pinned by CPU-runnable CPU/CUDA-device tests at `test_l14perf_identity.py:366` and the existing transfer/forward-boundary probe. `driver.py:1469` exposes `--pool-cpu-forwards`; CUDA and legacy-schedule combinations are refused. CPU pooling exactness remains empirical for the pinned single-threaded setup. |
| F9 | `scripts/labelmaker/tokeye_masks.sbatch:2` records pilot 2, with measured capacity defaults at `:29` and pool settings at `:48`. `:65` opens the full log under resolved ROOT; Slurm's bootstrap `--output=/dev/null` avoids a hard-coded production write. An explicit submission `--output` captures bootstrap diagnostics. `tests/labelmaker/test_tokeye_sbatch.py:28` checks CPUs = prep + tail + 1; `:92` checks the actual ROOT log and `:142` pins pilot-derived defaults. |
| F10 | Restored the recovered shared-index/text lost-update rationale, CPU-per-channel note, memory arithmetic and SIGALRM semantics in `driver.py:34`. Restored the runnable pre-pass/array/afterok workflow and staged jobstats-client explanation at `docs/LABELMAKER.md:529`. CPU companions now use the mandatory frozen **no-install** invocation (`tokeye_text_subset.sh:23`, `tokeye_masks_afterok.sbatch:43`) and offline exports. |
| F11 | `driver.py:1215` passes an explicit options dictionary; the surface regression at `test_tokeye_masks.py:1364` rejects stray locals such as `bad` and `pooled`. |
| F12 | `masks.py:506` and `:557` raise RuntimeError for malformed reference groups, retaining checks under `python -O`; tests at `test_l14perf_identity.py:467`. |
| F13 | Clarified that IDEATE's 132.69 s and 126.47 s were different historical runs. Float32 versus float64 overlap accumulation is indistinguishable at TILE 512 / STRIDE 448: at most two tiles per column, with exact halving. Float64 is extra protection matching the oracle, not a test-proven requirement. CPU default preservation removes reliance on single-threaded cross-block exactness. |
| Compact activity coupling | `src/labelmaker/events/pipeline.py:825` asserts `transients.ACTIVITY_THR == masks.PROB_THRESHOLD` at substitution; regression at `test_l14perf_identity.py:485`. No threshold or tolerance changed. |

TDD evidence: inference checks first **13 failed**; after adding the timeout
callback seam alone, the shared-batch regression specifically reproduced the
healthy shot receiving StageTimeout (**1 failed, 1 passed**). Default/opt-in
checks first **3 failed**. After fixes, **34 identity tests passed**, plus
**2 driver deadline tests**. Scheduler red: **4 failed, 21 passed**; green:
**75 passed**. Launcher/workflow red: **4 failed, 5 passed**; green: **9 passed**.
The later measured-default test failed first; all **10 launcher tests** then
passed. CUDA compaction/copy-stream checks: **2 passed in 3.92 s**, using the
phase3 Python on stellar-vis2 with `CUDA_VISIBLE_DEVICES=1`.

### Second A100 pilot: measured result

**2932066_0**, source **56d4521**, stellar-m01g5, A100-PCIE-40GB, GPU 1,
initially **0 MiB used**. SLURM start/end **2026-09-14 20:01:08–20:02:30 EDT**;
**82 s**, all steps **COMPLETED**, exit **0:0**. `squeue` emptied.

| Measurement | Observed value |
|---|---:|
| Request | 12 CPUs; 7 prep, 4 tail, prefetch 8; 32G; 00:12:00; one A100; batch 96; AMP; readonly text; no index |
| Completion | **20/20 shots, 244 blocks, 7,478 tiles, 26,913 events** |
| Driver wall / throughput | **76.87 s / 97.29 tiles/s** |
| CPU, detailed jobstats | **31.3%** (sampled CPU time 00:05:07) |
| CPU memory, jobstats | **9.8GB / 32GB = 30.6%** |
| GPU, jobstats | **7.4%** |
| GPU memory, jobstats maximum | **38.6GB / 40GB = 96.6%** |
| sacct MaxRSS, numbered step | **13,559,492K = 12.931339263916016 GiB**; 40.4104% of 32G |
| sacct TotalCPU | **06:20.657 = 380.657 s** for the job; numbered step 06:20.619 |
| Torch peak allocated | **18.211864471435547 GiB** |
| Parent / largest worker VmHWM | **6.050 / 2.573 GiB**, 7 live prep and 4 live tail PIDs |
| Prep wait / pooled inference timer | **44.823 / 64.781 s** |
| Description / finishing worker service | **26.365 / 82.895 s** |
| Tail-result wait | **5.803 s** |

Jobstats summary bars are 31/31/7/97; the table uses detailed gate values.
Sampled CPU time and sacct TotalCPU differ; neither replaces the other.
Per-PID peaks are not additive because of shared pages. The pooled inference
timer includes input/planning waits; do not add it to prep wait or add
concurrent worker service to driver wall.

Exact submission and gate, from the worktree (directories and readonly text
copy were prepared under the printed pilot root first):

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/tmp/l14perf
REPO=/scratch/gpfs/nc1514/FusionAIHub-L14perf ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/l14perf/pilot2-l14perf SHOT_FILE=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/l12/pilot20.txt PREP_WORKERS=7 PREFETCH=8 TAIL_WORKERS=4 TILE_BATCH=96 N_CHUNKS=1 sbatch --parsable --job-name=tokeye-L14perf --cpus-per-task=12 --mem=32G --time=00:12:00 --array=0-0%1 --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/l14perf/pilot2-l14perf/slurm/%A_%a.out scripts/labelmaker/tokeye_masks.sbatch
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m labelmaker.jobstats --job-id 2932066_0 --pilot --preserve-dir /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm --out /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/jobstats.json
```

Gate output, **exit 0 due to pilot exemption**, with no threshold changes:

```text
2932066_0  CPU 31.3 %  CPU-mem 30.6 %  GPU 7.4 %  GPU-mem 96.6 %  FAIL (exempt)
    cpu 31.3 % < 70 % (jobstats)
    cpu_mem 30.6 % < 70 % (jobstats)
    gpu 7.4 % < 70 % (jobstats)
```

**Diagnosis:** throughput fell 30.4% from 139.81 tiles/s, while remaining
1.987× L12's 48.97. Prep wait rose from 0.002 to 44.823 s (58.3% of driver
wall), whereas tail wait fell from 8.344 to 5.803 s. This is direct evidence
of an exposed input-feeding bottleneck with the smaller pools. Low aggregate
CPU use does not establish that fewer workers would feed the GPU: startup,
I/O and IPC can leave allocated CPUs idle while preparation is awaited.
This pilot did not separately profile those causes, so their contributions
remain unresolved. GPU memory includes allocator cache and cannot establish
GPU activity. The short reports expose no sample count; the previous 100%
and current 7.4% are retained as measured, without extrapolating continuous
utilization. The GPU miss invokes the stop rule: no more jobs or production.

### Completion and exact hash finding

All per-shot block/event counts match run rows; sources were read and include
`intervals`/`min_gap_s`; `text_subset_missing` is empty. The same five L-A shots
completed: 185962 (14 blocks / 1,537 events), 185980 (14 / 1,421), 185982
(14 / 1,800), 186090 (10 / 588), 186196 (10 / 1,241).

Compared every product to pilot 2931999_0 and revalidated that pilot's recorded
hashes. **NPZ: 20/20 hashes equal. Events: 0/20 equal. Sources: 0/20 equal.**
The requested cross-pilot hash equality therefore **fails** for both parquet
products. Their only differing frame columns are `git_sha`, `run_id`, and
`written_at`. Dropping only the contract's `run_id`/`written_at` still fails
because git_sha changed from `5a8386e` to `56d4521`. As a separate diagnosis,
all other columns compare exactly, including coverage; that is **not** a
relaxed identity pass. No product, provenance or tolerance was changed to
make hashes match. Same-HEAD synthetic identity remains exact.

Per-shot hashes, strict frame hashes, changed columns, counts, source-status
counts and worker measurements are preserved in
`runs/l14perf/pilot2-l14perf/completion-evidence.json` and `comparison.log`.
The reproducible comparison script is `runs/l14perf/evidence-fix/check_pilot.py`.

### Derived production capacity — report only, utilization unvalidated

| Resource | Derivation / request |
|---|---|
| CPUs / pools | Keep the measured pools: **7 prep + 4 tail + 1 parent = 12 CPUs**, prefetch **8**. Changing pools again would invalidate the measured memory/throughput basis. |
| Memory | `ceil(12.931339263916016 × 1.3)` = **17G** (16.810741 GiB before rounding). |
| Time for 60 shots | Startup `82 − 76.87 = 5.13 s`, plus one **240 s timeout**, plus `60 × (76.87 / 20) × 1.3` = **544.923 s**, rounded up to **00:10:00**. |
| GPU / array | **One A100**, batch **96**, AMP; proposed **8 × 60 shots**, `--array=0-7%2`, `N_CHUNKS=8`. |

These capacity figures set the sbatch defaults, but **are not a validated
production request**: three gates failed at the measured pool sizes. The
pilot allocation remained 12 CPUs / 32G / 12 minutes; no smaller hypothetical
allocation was sent through the gate. Mean sacct demand was 4.642 cores, but
that aggregate is not evidence for reducing prep concurrency after observing
44.823 s of prep wait. Production was not submitted.

### Final verification and limits

Final source/scripts/tests snapshot: **02a8b16**. From the worktree, with the
exports above and the main checkout's existing environments:

```bash
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m pytest tests/labelmaker -q -W error -p no:cacheprovider
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m pytest tests/ideate -q -W error -p no:cacheprovider
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker ruff check --no-cache src/labelmaker src/ideate scripts/labelmaker tests/labelmaker tests/ideate
```

- Labelmaker: **1,655 passed, 3 skipped in 277.32 s; exit 0**.
- IDEATE: **1,252 passed in 133.03 s; exit 0**.
- Ruff: **All checks passed, exit 0** over all five required paths.
- All logs and exit files are under `runs/l14perf/evidence-fix/`. Labelmaker's
  XRootD atexit FutureWarning printed after its successful summary; exit stayed 0.
- The two live FDP tests (`test_live_fetch_of_the_reference_points_for_one_shot`
  and `test_live_ip_matches_the_archive_in_amps`) stayed skipped by the existing
  opt-in gate: neither `--run-live` nor `LABELMAKER_FDP=1` was requested.
- The opt-in fresh real-shot CPU identity test was not rerun in this loop;
  its earlier exact evidence is retained, and this loop ran synthetic identity,
  CUDA edge/stream checks and the full 20-shot real A100 comparison. Profiles
  were not repeated: the fix brief requested one second pilot, not a new profile.
- The restored afterok workflow is documentation; no companion or production
  was submitted. Feature-dependent skips remain consistent with the first
  pilot because scratch roots do not contain production feature stores.
- No production masks/events/labels/features or IDEATE data writes, no edits
  to `docs/superpowers/plans/**`, no installation, no worktree `.pixi`, and no
  commit while a suite ran. Scratch files use `/tmp/l14perf`; persistent data
  uses `runs/l14perf/` plus the three prescribed `runs/slurm/` gate captures.

Evidence SHA-256:

| Artifact | SHA-256 |
|---|---|
| `runs/slurm/2932066_0.jobstats.txt` | `8a4ea76069f62df7305d91cb53f7a29899ef3b9663cfbd44f08074b6307d4be7` |
| `runs/slurm/2932066_0.sacct.txt` | `3abce5420d66afb1c9576547123b0769fcde6f2531983c32882ac976a1038978` |
| `pilot2-l14perf/completion-evidence.json` | `f94d4aa5e17667e69db85625d46659347bffdeca7b18c4df77f4481eb4f9b6a1` |
| `pilot2-l14perf/runs/events/tokeye-2932066_c0of1_r0.json` | `c04096075d21134a466c42a9e855bd2d405c20645f5855123d1413cf4efa0f05` |
