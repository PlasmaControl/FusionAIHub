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

## Fix pass (2026-09-14)

**Stopped at full-suite verification, as the fix brief requires.** Items 1–6 were implemented in order with focused checks, but the final labelmaker and IDEATE suites are both red. This pass is not ready for acceptance. No implementation changes or test reruns were made after the first full-suite failure; the already-running IDEATE suite was allowed to finish, and this report records the remaining work.

### Findings 1 and 2

1. `test_slow_baseline_with_gap_contains_no_elm_spikes` failed first with the review's two false points at 0.25/1.25 s, widths 333.218/207.785 ms (`/tmp/la-fix/finding1-red.log`). The D-alpha path now rejects half-prominence widths above **`DALPHA_MAX_WIDTH_MS = 5.0` ms**, measured separately inside each contiguous finite run. This is the brief's width-only shape-policy option: millisecond bursts can form a 200 Hz train with 5 ms spacing; hundreds-of-ms humps are baseline excursions. No smoothing or width measurement crosses gaps or padding, and mask peak picking is unchanged. The constant and rationale are documented in `docs/LABELMAKER.md` and the function docstring.

   After the fix: **zero false ELMs**, with quiet intervals exactly `[0, 0.7999]` and `[1.1001, 2]` s. `test_narrow_dalpha_pulses_survive_slow_baseline_and_gap`, parameterized for 1 and 2 ms full widths, recovers all four pulses within one 0.1 ms sample. An additional scratch 200 Hz control recovered **40/40** pulses within one sample. All **190** tests in the transients, pipeline, windows, label-store and lexicon files passed, including the amplitude/noise, padding/gap, shared-train and review-check-3 pipeline guards (`/tmp/la-fix/finding1-green.log`).

2. Extended `test_outside_track_does_not_prevent_independent_dalpha_events` to keep a valid co-occurring descriptor in the block that also contains a wholly outside descriptor. It failed first because published partners named the skipped block (`/tmp/la-fix/finding2-red.log`). `finish_shot` now converts blocks first and computes co-occurrence from the accepted original runs only. Block-local track ordering and raw descriptor identities are preserved; rejected blocks remain skipped with unknown coverage. The regression now passes, retaining successful-block references, and **326** pipeline/track/coverage tests passed (`/tmp/la-fix/finding2-green.log`).

   The review's original parquets were read without modification and reproduced **13** references to skipped blocks (`/tmp/la-fix/finding2-original-parquet.json`). A new CPU inference run completed with run ID **`la-fix-185962`** in **438.4 s**, writing only under `/tmp/la-fix/root`:

   ```bash
   cd /scratch/gpfs/nc1514/FusionAIHub-build
   source /tmp/la-fix/env.sh
   "$LA_PYTHON" -u -m labelmaker.run events --shots 185962 --root /tmp/la-fix/root --device cpu --passes wide --tile-batch 4 --timeout 3600 --unet "$LABELMAKER_ROOT/models/tokeye/big_tf_unet_251210.pt" --run-id la-fix-185962
   ```

   It produced **7 mask blocks, 1,453 events, 8 skips**: **546 ELM points, 3 ELM-free intervals, 687 generic transients, 158 track rows, 54 sawtooth candidates and 5 actuator rows**. ECE8/wide remains the sole rejected conversion block, with zero events and NaN coverage; the other track-source skips describe unavailable planned inputs. The new parquets have **zero references to skipped or unpublished blocks**, retaining **20 valid references**. Fresh filterscope recomputation exactly matches all saved ELM timestamps and checks point semantics, NaN confidence and the width cap. Compared with the review's 548 ELMs, the two excluded peaks had widths **5.0169 ms and 20.2891 ms**; no parameter was selected using these production counts. This is software reproduction, not physical validation. Evidence: `/tmp/la-fix/events-185962.log`, `/tmp/la-fix/verify_185962.py`, `/tmp/la-fix/verify-185962.json` and `/tmp/la-fix/verify-185962.log`. The run records `1c62fd9` plus the then-uncommitted finding-2 patch subsequently committed as `55bbd40`.

### Housekeeping and legacy window verdict

3. The explicit two-shot CLI failed first at the minimum-ten assertion (`/tmp/la-fix/item3-red.log`). Smaller explicit lists now work while the ten-shot default remains. The actual command `"$LA_PYTHON" scripts/labelmaker/assess_labels_a.py sawtooth --shots 185786 191213 --out /tmp/la-fix/sawtooth` generated **59 / 163 candidates** and **16 / 21 ms median periods**, matching review check 6. An explicit shot outside the pool returns argparse exit **2**. Evidence: `/tmp/la-fix/item3-green.log`, `/tmp/la-fix/item3-invalid.log` and `/tmp/la-fix/sawtooth/sawtooth.json`.

4. Factored `_is_repository_checkout_root` in `tests/ideate/test_mcp.py`. `test_mcp_cwd_requires_a_root_of_this_repository[subdirectory-False]` failed first; the helper now requires both equality with the path's own Git top level and the same Git common directory. The worktree root passes; `docs/` and a temporary unrelated Git repository fail; the existing configured primary-checkout test passes. All **51 MCP tests** passed. `.mcp.json` was not edited. Evidence: `/tmp/la-fix/item4-red.log` and `/tmp/la-fix/item4-green.log`.

5. Preserved the census as the **2026-09-13 / `b265f40` snapshot** (the saved census contains that SHA), and added a separate review inventory to the assessment and this report: **2026-09-14, 18 event files / 15 source files / 15 pool shots**, from L12's pilot. These are per-shot products. The unchanged joined `events.parquet` hash and zero observed joined rows remain separate historical facts. This update cites the review's read-only inventory; it does not claim a new production census.

6. **Confirmed and fixed legacy-magnetics contamination in window selection.** Both new regressions failed first in all three ELM features:

   | Regression table | Before: rate Hz / quiet fraction / age s | Required and obtained after |
   |---|---|---|
   | Legacy-only `mhr` clock rows | 1 / 0.5 / 0.2 | 0 / 0 / 1 |
   | Mixed legacy `mhr` and current filterscopes | 2 / 0.7 / 0.2 | 1 / 0.2 / 0.4 |

   The tests are `test_legacy_magnetics_clock_rows_do_not_supply_dalpha_features` and `test_legacy_magnetics_clock_rows_do_not_change_mixed_dalpha_features`. The brief's `elm_age_s` is named **`time_since_last_elm_s`** in this code. `EventTable.of` calls `diagnostic_rows`/`diagnostic_mask` before clustering points or merging intervals; both ELM families now require `diag == "filterscopes"` in addition to their source and evidence rules. Updated `_elm`'s default diagnostic and the positive ELM-free fixtures in `test_events_windows.py`. All **51 window tests** passed (`/tmp/la-fix/item6-red.log`, `/tmp/la-fix/item6-green.log`). The full suite subsequently exposed another old positive fixture in `test_events_evidence.py`, described below; that remains unfixed under the stop instruction.

### Final verification and remaining work

Only the existing main-checkout Pixi interpreters were executed; no environment was created or modified. `/tmp/la-fix/env.sh` sets `LA_PYTHON=/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python`, `LA_IDEATE_PYTHON=/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/ideate-cpu/bin/python`, `PYTHONPATH` to this worktree's `src`, and `PYTHONDONTWRITEBYTECODE=1`. It sets `CUDA_VISIBLE_DEVICES=''`, `HF_HUB_OFFLINE=1`, four CPU threads, the required read-only production roots, and redirects temporary files and caches to `/tmp/la-fix`. It does **not** reproduce the manifest's `LD_LIBRARY_PATH` activation; that omission matters for the subprocess failures below. The labelmaker runtime was Python 3.11.16, NumPy 1.26.4, SciPy 1.17.1, Torch 2.14.0+cpu and pandas 3.0.5 (`/tmp/la-fix/runtime-labelmaker.json`).

Commands actually run at stable code HEAD **`e22c71a`**:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub-build
source /tmp/la-fix/env.sh
"$LA_PYTHON" -m pytest tests/labelmaker -q -W error -p no:cacheprovider --basetemp=/tmp/la-fix/pytest-labelmaker-final
"$LA_IDEATE_PYTHON" -m pytest tests/ideate -q -W error -p no:cacheprovider --basetemp=/tmp/la-fix/pytest-ideate-final
/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/ruff check --no-cache src/labelmaker src/ideate tests/labelmaker tests/ideate scripts/labelmaker
```

| Check | Result | Exit | Evidence |
|---|---|---:|---|
| Full labelmaker | **1,410 passed, 5 failed, 2 skipped**, 59.16 s | 1 | `/tmp/la-fix/labelmaker-final.log` |
| Full IDEATE | **935 passed, 1 failed**, 90.91 s | 1 | `/tmp/la-fix/ideate-final.log` |
| Component Ruff | **All checks passed** | 0 | `/tmp/la-fix/ruff-final.log` |

Failures left for the next pass:

- **One fixture regression caused by item 6:** `test_events_evidence.py::test_the_real_elm_tuning_note_is_not_an_elm`. Its `_detected_elm` and `_elm_free` helpers still use `diag="mhr"` while calling those rows genuine D-alpha evidence. The new filter correctly excludes them, so the positive-control assertion fails. Align those helpers with filterscopes and rerun the evidence and complete suites; do not weaken the new gate.
- **Four labelmaker subprocess/runtime failures:** `test_resolve_fdp.py::test_available_survives_a_fork_after_torch_is_already_loaded`, `test_tokeye_masks.py::test_the_driver_writes_exactly_what_process_shot_writes[1-4]`, the same test's `[2-2]` case, and `test_a_block_whose_prep_fails_in_a_worker_is_a_skip_and_not_a_hang`. The three driver cases report system `/lib64/libstdc++.so.6` missing `GLIBCXX_3.4.26`; the fork case reports unavailable toksearch after Torch loads. The read-only manifest explicitly documents this loader-order problem and supplies `LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"` in its FDP activation. The direct-interpreter test wrapper omitted that activation. Reproduce it using the existing environment's library directory before rechecking; no environment installation or mutation is needed. No successful rerun is claimed.
- **One IDEATE cache setup failure:** `test_integration_real.py::test_minilm_loads_offline_from_the_local_cache`. Its presence check sees the default home cache, while my `HF_HOME`/`HF_HUB_CACHE` redirection makes Transformers load from empty `/tmp/la-fix/huggingface/hub`. A next run needs the existing cached model made available within the scratch-cache constraint. No download, cache repair, or successful rerun was performed after the stop.

A read-only reviewer found no blocking issue in the six-fix diff and noted that the co-occurrence fixture could optionally pin a nonzero surviving track index too. That source review does not override the failed full suites. **Deviation from the requested outcome: both full suites are not green; acceptance is unfinished.** Production stores and both environments remained read-only. No SLURM, other-worktree source edits, `.mcp.json` edits, or `docs/superpowers/plans/**` edits were made. Pre-existing untracked task files were left alone. No commit was made while the labelmaker suite was running.

| Item | Commit | Subject |
|---|---|---|
| 1 | `1c62fd9` | `labelmaker: reject broad D-alpha baseline peaks within finite runs` |
| 2 | `55bbd40` | `labelmaker: compute co-occurrence only for published track blocks` |
| 3 | `a10ef43` | `labelmaker: allow explicit smaller assessment shot lists` |
| 4 | `8c312d3` | `ideate: require MCP cwd to be a repository checkout root` |
| 5 | `41e35d0` | `labelmaker: distinguish the dated census from newer pilot products` |
| 6 | `e22c71a` | `labelmaker: exclude legacy magnetics clocks from D-alpha window features` |
| Fix-pass report | This commit | `labelmaker: record fix-pass results and failed full-suite checks` |

Every fix-pass commit carries `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
