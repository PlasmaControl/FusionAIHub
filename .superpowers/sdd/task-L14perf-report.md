# L14-perf report (in progress)

Worktree `/scratch/gpfs/nc1514/FusionAIHub-L14perf`, branch `recommender-L14perf`,
base `3cced76004183f1b3eef4a485b9e294a207ab9e5`. Read the binding brief and
L12 report before work. All runtime products use
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/l14perf/`.
No production job is authorized or will be submitted.

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
- Profile complete; implementation, identity verification, pilot, final suites,
  documentation and final report pending.

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
oracle's per-block forward groups on the device. CPU retains full cross-block
forward batches. This is a deliberate deviation from full CUDA forward pooling
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
exit **0**; IDEATE **1,252 passed**, 132.69 s, exit **0**; required Ruff command
passed, exit **0**. The labelmaker atexit XRootD FutureWarning printed after the
successful summary and did not change its exit code. No commit occurred while
any suite or real identity test was running.
