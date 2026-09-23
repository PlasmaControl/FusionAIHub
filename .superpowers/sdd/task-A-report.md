# Part A label review report

Completed in `/scratch/gpfs/nc1514/FusionAIHub-review`, branch `label-review`,
from base `7fcd175`. A1–A9 were applied in order, one commit per task, using
only the plan's extractor for all code, tests and documentation blocks.
A7's deletion used `git rm`. Every commit carries the required co-author line.

## Task results

Each A1–A8 test set was extracted and run before its implementation, then run
again afterward. Counts and failure reasons matched the plan. A9 prescribed
only documentation extraction and a commit.

| Task | Test files under `tests/labeler/` | Before implementation | After implementation | Commit |
|---|---|---|---|---|
| A1 | `test_config.py`, `test_events_raw.py` | 2 failed, 27 passed: raw cache still under `.cache`; missing `Paths.spectrograms` | 29 passed | `2850cb1` |
| A2 | `test_review_labels.py` | 1 collection error: no module `labeler.events.review` | 20 passed | `1d6c875` |
| A3 | `test_review_rows.py` | 1 collection error: cannot import `rows` | 11 passed | `bca5635` |
| A4 | `test_review_alfven.py` | 1 collection error: cannot import `alfven` | 4 passed | `249ea3b` |
| A5 | `test_review_panel_rows.py` | 1 collection error: cannot import `panel_rows` | 6 passed | `13afd23` |
| A6 | `test_review_build.py` | 1 collection error: cannot import `build` | 2 passed | `8a7eb36` |
| A7 | `test_events_ui.py`, `test_events_panels.py` | 25 failed, 33 passed: old server routes and AE panel registry | 58 passed | `7fde202` |
| A8 | `test_review_page.py`, `test_review_browser.py`, `test_events_ui.py` | 6 failed, 53 passed: five page rule tests and browser state timeout | 59 passed, no skips | `58cb230` |
| A9 | Documentation only | Not prescribed | Not prescribed | `12e77a1` |

All task test commands were run from the worktree root as:

```bash
PYTHONPATH=$PWD/src pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m pytest <files> -q -W error -rs
```

Ruff passed at each prescribed task check. A6's
`bash -n scripts/labeler/spectrograms.sbatch` exited 0 with no output.
A1's removal grep printed nothing.

## Full suites

```bash
PYTHONPATH=$PWD/src pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m pytest tests/labeler -q -W error -rs
```

Exit 0; exact summary:

```text
SKIPPED [1] tests/labeler/test_l14perf_real_identity.py:45: set L14PERF_REAL_ROOT to an l14perf scratch directory
SKIPPED [1] tests/labeler/test_resolve_fdp.py:454: live fdp fetch is opt-in: --run-live or LABELER_FDP=1
SKIPPED [1] tests/labeler/test_resolve_fdp.py:476: live fdp fetch is opt-in: --run-live or LABELER_FDP=1
2021 passed, 3 skipped in 666.78s (0:11:06)
```

The identity skip is `test_real_shot_cpu_output_identity`; the other two are
the opt-in live FDP tests named by the plan.

```bash
PYTHONPATH=$PWD/src pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e shot-design-cpu python -m pytest tests/shot_design -q -W error -rs
```

Exit 1 as expected; exact summary:

```text
SKIPPED [1] tests/shot_design/test_ignite.py:236: no production frame codes for 190090
SKIPPED [1] tests/shot_design/test_seed.py:347: no shipped cache at /scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/models/IGNITE_v4/frame_codes/202537.pt
SKIPPED [1] tests/shot_design/test_simulate_core.py:130: needs a CUDA device
2 failed, 1834 passed, 3 skipped in 256.44s (0:04:16)
```

The two failures match the documented base failures owned by Part 0:

- `test_ignite_v4.py::test_model_cfg_declares_fifteen_v4_modalities`:
  `cfg["t0_start_s"]` is `0.0`, expected `1.0`.
- `test_seed.py::test_encode_frame_codes_writes_the_shipped_dict_structure`:
  missing `IGNITE_v4/frame_codes/190090.pt` in the production bundle.

## Page, browser and acceptance checks

All seven page tests and the browser test ran, both in A8 and in the full
labeler suite. None skipped. The page rules checked 302 cases against the
server; the browser test passed all 11 checks and validated the saved labels
and history. Node was already on PATH at
`/home/nc1514/.nvm/versions/node/v25.6.1/bin/node`. The required shell existed
under `~/.cache/ms-playwright/chromium_headless_shell-1181/`.

Ruff reported `All checks passed!` for all 25 remaining touched Python files,
using the required binary:

```bash
git diff --name-only --diff-filter=ACMR -z 7fcd175..HEAD -- '*.py' | xargs -0 /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/ruff check
```

The acceptance grep printed nothing (exit 1 means no matches):

```bash
git grep -nE 'GUIDANCE|guidance\(|plotly|/vendor/|api/panels|api/save|api/progress' -- src/labeler/events/ui src/labeler/events/panels
```

A supplemental read-only comparison confirmed that all 21 complete-file
blocks match the plan byte for byte; all 11 extracted patches applied
cleanly. The implementation diff contains only the 33 Part A files in the
file map. This report is the only additional committed file.

No production data was written, no jobs were submitted, and no live FDP
fetch was run. No pixi install, lock or update command was run. Neither
`HF_HOME` nor `HF_HUB_CACHE` was set. Suites used the prescribed pixi command.
`pyproject.toml`, `pixi.lock` and `data/events/` match the base. All intentional
file edits were inside this worktree; test fixtures used pytest temporary
paths. The worktree was clean after both implementation and the labeler run.

## Differences and diagnostics

No task test results or acceptance counts differed from the plan, and no
implementation blocks were changed by hand.

- Pixi printed a network-filesystem cache warning on test invocations and
  automatically redirected repodata caching to `/tmp/pixi-cache-nc1514/repodata`.
- After the successful labeler summary, XRootD's shutdown finalizer emitted
  a `FutureWarning` about deprecated `torch.distributed.reduce_op`. The suite
  still exited 0; no tests failed or warned in the pytest summary.
- An extra read-only byte-comparison command initially failed because the
  system `python3` is Python 3.6 and does not support `subprocess`'s `text=True`
  argument. Repeating that check with `universal_newlines=True` succeeded.
  This affected neither the prescribed extractor nor any test or source file.
