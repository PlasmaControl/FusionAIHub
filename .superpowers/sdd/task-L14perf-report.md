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
