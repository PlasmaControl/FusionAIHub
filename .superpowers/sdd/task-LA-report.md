# Task L-A report

Worktree: `/scratch/gpfs/nc1514/FusionAIHub-build`; branch: `recommender-LA`; base: `b265f40`.

The committed [assessment](../../docs/superpowers/specs/2026-09-13-labels-assessment-A.md) compares all four CSV Proposed Method / Notes entries, records the census and ten-shot ECE check, and defines nine bounded follow-up tasks with exit criteria. All four CSV rows remain **in_progress**. L-A publishes the requested observed-ELM software family; completing a physical label row additionally requires independent validation and production source/index coverage.

## Ledger-ready census and statuses

Snapshot: **2026-09-13, measured at commit `b265f40`**. Population: the 500 distinct shots in the read-only production `recommender_v1.txt` (185786–204925), SHA-256 `a58c9c8983bab1c71d5ab74eb4949f86b0a7d06df7b0f1ff5a39132aee6ac2e2`.

| Row | labels_wide rows / shots | Detector / heuristic / forecast / other event rows | Observed-event shots | Forecast-only event shots | Status | Remaining acceptance gap |
|---|---:|---:|---:|---:|---|---|
| Tearing Mode | 13,800 / 500 | 0 / 0 / 1,016 / 0 | 0 | 69 | in_progress | CNN and survival outputs are forecasts; observed onset and n=1/2 confirmation need implementation/validation |
| AE Mode | 1,000 / 500 | 0 / 0 / 0 / 0 | 0 | 0 | in_progress | Human AUROC 0.6209; publish and validate observed TokEye AE intervals |
| ELM | 2,000 / 500 | 0 / 0 / 21 / 0 | 0 | 7 | in_progress | D-alpha code and temporary real outputs exist; manual validation and production join remain |
| Sawtooth | 0 / 0 | 0 / 0 / 0 / 0 | 0 | 0 | in_progress | Ten-shot check exposed false candidates; no validated radius mapping or independent crash truth |
| Total | 16,800 / 500 unique | 0 / 0 / 1,037 / 0 | 0 | 76 unique | — | Production source table is empty |

`labels_wide` has no evidence_kind column; summary rows are not time samples or observations. Valid-shot counts: CNN tm_prob 163, DSM tm_risk_1s 151, continued DSM 139/140, AE activity 203, ELM risk 10. The 1,037 event rows all use `label_forecast/forecast`. At that snapshot there were exactly three production event files, on **185946, 185953, 198658**, all outside the 500; no production source files. No production products were written by this assessment.

**Current inventory as of the read-only review (2026-09-14):** production `events/` holds **18 event files and 15 source files**, including **15 `recommender_v1` shots** from L12's pilot products. The joined `events.parquet` is unchanged (SHA-256 `7a46f795e2e05ba6617a92e4b5c5c7767edd3a4fec6d919086904616fe46195d`) with **zero observed rows**. Per-shot products are distinct from the joined database; they do not revise the historical census. Evidence: [review check 2](task-LA-review.md).

The census command and table hashes are in the assessment; the complete output is `/tmp/task-LA/census/census.json`. The committed reproduction script is `scripts/labelmaker/assess_labels_a.py`. All 500 label and feature files were opened read-only.

## Sawtooth validation

CPU-only invocation:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub-build
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$PWD/src /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python scripts/labelmaker/assess_labels_a.py sawtooth --out /tmp/task-LA/sawtooth-labelmaker
```

| Shot | Crash candidates | Median period (ms) | Modal dropping-block boundary (zero based) | Radius |
|---|---:|---:|---|---|
| 185786 | 59 | 16.0 | 46/47 | unknown |
| 187168 | 55 | 41.5 | 39/40 | unknown |
| 189890 | 64 | 77.0 | 31/32 | unknown |
| 191213 | 163 | 21.0 | 39/40 | unknown |
| 195037 | 57 | 67.0 | 31/32 | unknown |
| 196488 | 61 | 77.0 | 39/40 | unknown |
| 200133 | 35 | 84.5 | 39/40 | unknown |
| 201718 | 65 | 45.5 | 13/14 | unknown |
| 203660 | 42 | 55.0 | 31/32 | unknown |
| 204925 | 71 | 78.0 | 46/47 | unknown |

All ten are real pool shots selected at evenly spaced sorted-pool indices. Total **672 candidates**; **67** are at |Ip| <100 kA, and four have unknown Ip. The counts/periods are unvalidated candidates, not a claim of sawtooth accuracy. Plots and full event/edge distributions are under `/tmp/task-LA/sawtooth-labelmaker`; the assessment documents the pre-pulse and low-signal tail failure and the ELM-triggered inversion-visibility limitation.

The features store has ECE signals but no channel-to-radius mapping. A reusable fdp/ECEGEOM plus equilibrium route exists in the peer omnimode code, inspected read-only; it was not fetched, ingested or assumed calibrated for these shots. Thus no claim about rho 0.3–0.6 is supported yet. Python 3.11.16 / NumPy 1.26.4 / SciPy 1.17.1 produced the reported counts. A preliminary NumPy 2.4.6 run had 670 candidates; this runtime sensitivity is recorded in the assessment.

## Observed-ELM implementation

- Independent filterscopes 0–7 clock; first channel with an adjacent finite pair, finite-range normalization, shared smooth-and-pick helper.
- `elm_clock/elm` heuristic POINT events with exactly prominence, width_ms, channel and rate_hz_local attrs, NaN confidence and filterscope finite-span coverage. `elm_free` derives from the same peak train; internal NaN gaps receive no peaks or quiet intervals.
- `tokeye_transient/transient` detector points retain their class-agnostic meaning, burst/checkpoint metadata and diagnostic coverage. `transient` is in the lexicon and registry.
- Windows and ideate select `elm_clock/elm`; unchanged numerical window reductions are guarded by tests. The brief's premise that these already used a D-alpha clock was false at the base commit, so the source/input change was necessary.
- Legacy stored TokEye `elm` rows cannot become ELM query hits; MCP exposes them as transient without rewriting the store. Legacy magnetics clock coverage is excluded from D-alpha coverage. Reruns replace obsolete per-shot index keys while preserving other shots, including empty results.

Real stage command:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub-build
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$PWD/src /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python -u -m labelmaker.run events --shots 185786 191213 204925 --root /tmp/task-LA/real-events --unet /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/tokeye/big_tf_unet_251210.pt --device cpu --passes wide --tile-batch 4 --timeout 3600 --run-id task-LA-cpu-final
```

The initial stage completed 185786 and 191213, then failed on 204925: an existing ECE track ended at 6.14391994539734 s beyond its 6.143147945404053 s coverage. A supplementary run on 187168 reproduced the same failure (0.516 ms overrun). Neither failed attempt counts as a successful demonstration. A test-first fix now isolates the rejected track block, recording skipped/unknown coverage while valid blocks and the independent D-alpha clock continue. The successful rerun uses the same command with `--shots 187168 --run-id task-LA-cpu-isolated`; its log is `/tmp/task-LA/real-events-isolated.log`. Finer-grained valid-track recovery remains follow-up A9.

| Shot | ELM points (`elm_clock`, heuristic) | ELM-free intervals | Transient points (`tokeye_transient`, detector) | Filterscope channel | Clock coverage (s) | Recorded skips |
|---|---:|---:|---:|---:|---|---:|
| 185786 | 1388 | 0 | 2 | 1 | [-0.049880, 7.949920] | 9 |
| 191213 | 1057 | 0 | 12 | 0 | [-0.049880, 6.949920] | 11 |
| 187168 | 488 | 3 | 548 | 1 | [-0.049880, 7.949920] | 11 |

Total: **2,933 ELM points**, **562 generic transient points**, and **3 ELM-free intervals** across three successful shots. All have five actual CPU mask blocks. On 187168, ECE track blocks 8 and 20 are explicitly skipped with unknown coverage after conversion errors; its D-alpha clock and remaining blocks complete.

A fresh CPU recomputation of each saved filterscope trace exactly matches the stored ELM times and verifies point semantics, attrs, evidence kind and coverage. Evidence: `/tmp/task-LA/verify_real.py`, `/tmp/task-LA/real-events-verification.json` and `/tmp/task-LA/real-events-verification.log`. This is a serialization/current-code check, not independent manual classification.


The actual network ran on CPU. Artifacts are `/tmp/task-LA/real-events/{masks,events,runs}`, `/tmp/task-LA/real-events/events_index.parquet`, and `/tmp/task-LA/real-events-final.log`. ELM counts are the initial rule's observations, not manual truth. Source records distinguish missing diagnostics and skipped rules from successful quiet results. The assessment lists the required threshold, channel, plasma-gating and manual-label validation follow-ups.

## Test-first evidence and verification

The new synthetic clock/registry tests were run and failed before implementation. The physical-amplitude regression then failed with 166 peaks instead of four before normalization was added. Review-driven regressions failed before fixes for isolated finite-channel fallback, legacy MCP ELM leakage, legacy clock coverage, obsolete index keys and empty index replacement. RED logs:

- `/tmp/task-LA/tdd-labelmaker-direct-red.log`, `tdd-ideate-red.log`, `tdd-scaling-red.log`.
- `/tmp/task-LA/tdd-fallback-red.log`, `tdd-legacy-red.log`, `tdd-coverage-red.log`, `tdd-index-red.log`, `tdd-empty-index-red.log`.

Focused verification after review fixes: **63 passed** in labelmaker and **130 passed** in ideate; the subsequent real-shot track-isolation regression and coverage suite passed **72 tests**. Its RED failure is `/tmp/task-LA/tdd-track-isolation-red.log`, and GREEN results are `/tmp/task-LA/track-isolation-green.log`. A second read-only review confirmed all four findings addressed and no new material defect. Full exact commands:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub-build && PYTHONPATH=$PWD/src pixi run --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m pytest tests/labelmaker -q -W error
```

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=$PWD/src IDEATE_DATA_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/ideate LABELMAKER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker IDEATE_CORPUS=/scratch/gpfs/EKOLEMEN/foundation_model /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/ideate-cpu/bin/python -m pytest tests/ideate -q -W error
```

| Suite | Result | Exit code | Log |
|---|---|---:|---|
| labelmaker | **1,410 passed, 2 skipped** in 135.94 s | 0 | `/tmp/task-LA/labelmaker-final-isolation.log` |
| ideate | **933 passed** in 215.73 s | 0 | `/tmp/task-LA/ideate-final-isolation.log` |

The labelmaker process printed an installed XRootD/torch deprecation warning during interpreter finalization after pytest completed; pytest itself passed with warnings treated as errors and the process exit code was zero.

The known `.mcp.json` worktree test failure was corrected without changing the production launch config: the test now accepts a configured primary checkout sharing this worktree's Git common directory, while retaining absolute-path and server-module assertions. No HEAD movement occurred during either final suite.

Lint with Ruff 0.16.5:

```bash
/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/ruff check src/labelmaker src/ideate tests/labelmaker tests/ideate scripts/labelmaker
```

**All checks passed** for these components. Two pre-existing golden-data helper lint issues were also corrected. The broader brief-requested `ruff check src tests scripts/labelmaker` is **not green**: 1,222 pre-existing findings remain in `src/tokamak_foundation_model`, `tests/ignite`, `tests/e2e`, and `tests/data`. A clean archive of base `b265f40` has 1,224 findings with the same Ruff/config. Comparison by path/rule/message confirms **zero new findings, two resolved**. Evidence: `/tmp/task-LA/ruff-baseline.json` and `/tmp/task-LA/ruff-full.json`. Broad unrelated model/test refactoring was not folded into this labels task. This is an outstanding literal acceptance condition, not a claim of a clean repository-wide lint run.

The real outputs record base Git SHA `b265f40` because implementation was tested before committing. The tested tracked-file patch is `/tmp/task-LA/implementation.patch`, SHA-256 `fba0a9b7467da859b4b02b7fd09ebc2996e4917dbe91e8cd806d7754e31cadef`; the implementation commit below captures that patch. Generated scientific outputs are reproducible from the committed helper and recorded runtime.

## Deliverable commits and boundaries

| Deliverable | Commit | Subject |
|---|---|---|
| Observed-ELM code, compatibility, tests and integration docs | `7c9c202` | `labelmaker: publish D-alpha ELM points and separate generic transients` |
| Assessment and read-only reproduction helper | `3927647` | `labelmaker: assess four label rows against the 500-shot census` |
| This final report | This commit | `recommender: record L-A evidence and remaining acceptance gaps` |

Every deliverable commit carries `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. No data/checkpoints/run artifacts were added to git, no SLURM jobs were submitted, and no controller ledger under `docs/superpowers/plans` was edited. Writes were confined to this worktree and `/tmp`. The pre-existing untracked briefs/reports for other tasks were left alone.
