# Label review rebuild, IGNITE v4 only, pixi tidy: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. A Codex agent without those skills: run the tasks in
> order, one commit per task, test first, and write the report named in your part's brief.

**Goal:** Make IGNITE v4 the only generation anywhere in shot_design (Part 0), rebuild the label
review surface around one editable label per shot and a prebuilt spectrogram store (Part A), and
make the pixi manifest's roots defaults with described tasks (Part E).

**Architecture:** Part 0 deletes the generation switch so every reader takes `model:` from
`ignite_modalities.yaml`. Part A adds `labeler.events.review` (label store, row store, AE and
generic builders, a build CLI) and replaces the review app and page: FastAPI serves binary rows
and one label per shot; the page draws canvases with no plotting library. Part E edits
`pyproject.toml`'s pixi sections only; `pixi.lock` does not change.

**Tech Stack:** Python 3.11, FastAPI 0.141.1, pydantic 2.13.5, Starlette 1.6.0, h5py, numpy,
scipy (`signal.resample_poly`, `signal.stft`, `ndimage.uniform_filter1d`), pandas, pytest,
plain browser JavaScript and canvas 2D, node 25 for the page's unit tests, pixi 0.76.1, SLURM.

Spec: `.claude/superpowers/specs/2026-09-22-label-review-v4-only-pixi-design.md`.

## Global Constraints

- Python `>=3.11,<3.12`; ruff `line-length = 88`. Run
  `/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/ruff check <files>` from the
  worktree root on every Python file you touch (ruff is not a suite; the bare binary is fine).
- Suites run only as
  `PYTHONPATH=$PWD/src pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m pytest tests/labeler -q -W error`
  and the same with `-e shot-design-cpu ... tests/shot_design`, from the checkout being tested
  (a worktree's `$PWD/src` wins over the editable install). Never the bare `.pixi` interpreter
  for a suite; never set or redirect `HF_HOME` / `HF_HUB_CACHE`.
- Never run `pixi install`, `pixi lock`, `pixi update`; Parts 0 and A never edit
  `pyproject.toml` or `pixi.lock`.
- No production writes: nothing under `/scratch/gpfs/EKOLEMEN`, no `sbatch`, no `srun`, no fdp
  fetch. Tests write only under pytest's `tmp_path`.
- Nothing new under `/scratch/gpfs/nc1514` except repository source.
- `data/events/` is not edited by Parts 0 and A (the server writes `review/` there at run time,
  never a test).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Label times are whole milliseconds; the longest label window is 20,000 ms.
- `LABELER_ROOT` production value: `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker`; the store is
  `$LABELER_ROOT/spectrograms/<event>/<shot>.h5`; the raw cache is `$LABELER_ROOT/raw`.
- IGNITE v4: bundle `/scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/models/IGNITE_v4` (Hub
  `nc1/IGNITE-v4` @ `d2f12b82`), frame 0 at 1.0 s, 50 ms frames, 219 frames per shot, 1209
  tokens per frame, `frame_embed` 100 rows, SEED_FRAMES 20 (earliest window start 2.0 s,
  `k0 + n_predict <= 100`, so `n_predict <= 80`).
- Words on the review page: row titles, axis units, state names, button labels, the key legend.
  No other prose.
- Code style: elegance over handling every error. One-line comments where the code is not
  obvious; no essays.

## Spec amendments folded in while planning

The spec already carries these (its line 4 says so):

1. Store datasets are named by pooling factor, `rows/<name>/{1,8,64}`; file and row attrs as
   listed in the spec's "Spectrogram store".
2. The wheel: a plain wheel scrolls the rows pane; Ctrl/⌘ + wheel, or any wheel over the axis or
   the two bottom tracks, zooms at the cursor; a horizontal wheel pans.
3. The source glob is `format/*_format_*.csv` (RWM's table has no event prefix).
4. The Stellar Simulate button submits a new `scripts/shot_design/simulate.sbatch`, not
   `simulate_batch.sbatch`.

Two facts found while planning that change nothing in the spec but explain the code:

- No format table has a shot wider than 6,000 ms (AE 2,000; ELM 6,000; H/L-mode 5,150; NTM
  4,989); RWM's 56 rows are all point events, so RWM shots open on an empty label.
- NTM and RWM tables carry no category-0 rows, so their source window is the extent of their
  events.

---

## File map

| Part | File | Change |
|---|---|---|
| 0 | `configs/shot_design/ignite_modalities.yaml` | delete `model_generations` and v2 comments |
| 0 | `src/shot_design/shotdb/ignite.py`, `cli.py`, `simulate/cli.py`, `shotdb/build.py`, `design/seed.py`, `design/program_reference.py`, `design/assistant.py` | v2 purge; readers require `frame_origin_s` / `t0_start_s` |
| 0 | `src/shot_design/config.py`, `configs/shot_design/paths.yaml`, `paths.frontier.yaml`, `ui.yaml`, `src/shot_design/ui/simulate_routes.py`, `src/shot_design/ui/static/design.js` | per-cluster simulate submit; v4 window defaults |
| 0 | `scripts/shot_design/simulate.sbatch` (new), `_stellar_common.sh`, `g_enc.py`, `batch_collect.py` | Stellar Simulate; no generation export |
| 0 | `src/shot_design/simulate/decode.py`, `simulate/report.py`, `dev/paper/figures/make_figures.py`, `dev/paper/figures/decode_all_v4.py` | B9 signed mean; `frac_static` both arms |
| 0 | `tests/shot_design/*` | v4 fixtures; delete the pin and the override test |
| 0 | `docs/clusters/{stellar,frontier}.md`, `docs/shot-design/simulation.md`, `docs/models/ignite.md`, `docs/data/corpus.md`, `dev/paper/README.md` | v2 instructions out |
| A | `src/labeler/config.py` | `raw_cache` follows `root`; `spectrograms`, `spectrogram_file` |
| A | `src/labeler/events/raw.py` | delete `groups_in`, `CORPUS_GROUPS`, `promote`, `clean`, the CLI |
| A | `src/labeler/events/review/__init__.py` (new) | package docstring |
| A | `src/labeler/events/review/labels.py` (new) | label store: normalise, read, save, states, queue, resume |
| A | `src/labeler/events/review/rows.py` (new) | row store: pyramid write, meta, window read |
| A | `src/labeler/events/review/alfven.py` (new) | Heidbrink AE rows |
| A | `src/labeler/events/review/panel_rows.py` (new) | any event's panels as rows |
| A | `src/labeler/events/review/build.py` (new) | `build()`, the CLI |
| A | `scripts/labeler/spectrograms.sbatch` (new) | the store build job |
| A | `src/labeler/events/ui/app.py` | rewritten around the five routes |
| A | `src/labeler/events/ui/static/{index.html,style.css,app.js}` | rewritten: canvas split screen |
| A | `src/labeler/events/panels/*` | GUIDANCE out; `alfven_eigenmode.py` deleted |
| A | `tests/labeler/test_review_{labels,rows,alfven,panel_rows,build,page,browser}.py`, `tests/labeler/review_browser.mjs` (new) | tests of the new code; the page's rules under node; the page in headless Chromium |
| A | `tests/labeler/test_events_ui.py` (rewritten), `test_events_panels.py`, `test_events_raw.py`, `test_config.py` | tests |
| A | `docs/labeler/review.md` (new), `docs/reference/environment-variables.md` | docs |
| E | `pyproject.toml` (pixi sections), `docs/reference/pixi-environments.md` (new), `tests/shot_design/test_docs_scratch_db.py` | roots as defaults, described tasks, the comments moved to a doc |
| E | `src/labeler/events/ui/serve.py`, `src/labeler/events/raw.py` | the fdp command is the `labeler-verify` task |
| E | `docs/shot-design/{overview,simulation,llm-providers}.md`, `docs/clusters/{stellar,frontier,adding-a-cluster}.md`, `docs/reference/environment-variables.md`, `scripts/shot_design/{_stellar_common.sh,batch_design.py}`, `AGENTS.md` | the sentences the defaults make false |

Parts 0 and A touch disjoint files. Part E comes after both: it edits `raw.py` and
`environment-variables.md` on top of A, and four files P0 also edits (`docs/clusters/stellar.md`,
`docs/clusters/frontier.md`, `docs/shot-design/simulation.md`, `_stellar_common.sh`). E owns
`pyproject.toml`.

## Execution map

| Who | Where | Branch | Report |
|---|---|---|---|
| Part 0: one Codex agent (gpt-6-astra, xhigh, write) | `/scratch/gpfs/nc1514/FusionAIHub-v4only` | `v4-only` | `.superpowers/sdd/task-P0-report.md` |
| Part A: one Codex agent (gpt-6-astra, xhigh, write), tasks A1-A9 in order | `/scratch/gpfs/nc1514/FusionAIHub-review` | `label-review` | `.superpowers/sdd/task-A-report.md` |
| Part E, merges, ops, scoring | the session, main checkout | `nathan_dev` | ledger |

Both worktrees branch from the commit that adds this plan and
`.claude/superpowers/plans/extract_blocks.py`, which Part A uses to put its code blocks in
place. `.superpowers/` is gitignored:
`git add -f` the report. The session merges `v4-only` and `label-review` into `nathan_dev`
after reviewing each diff and rerunning both suites.

---

## Part 0 -- IGNITE v4 is the only generation (brief for one Codex agent)

Work in `/scratch/gpfs/nc1514/FusionAIHub-v4only` on branch `v4-only`. Read the spec's Part 0
first. The inventory is this grep, which must be empty when you are done:

```bash
git grep -nE 'model_generations|IGNITE_GENERATION|"v2"' -- src/shot_design scripts/shot_design \
  configs/shot_design tests/shot_design docs dev/paper
```

It finds 27 lines today: `ignite_modalities.yaml` 301, 303, 307; `docs/clusters/stellar.md`
105; `docs/shot-design/simulation.md` 96; `_stellar_common.sh` 14, 29, 37; `cli.py` 433, 486;
`program_reference.py` 192; `shotdb/ignite.py` 51, 57, 62, 64, 66, 69, 70, 77, 79, 290, 304;
`simulate/cli.py` 147; `test_ignite_generation_override.py` 1, 30, 32, 51.

The wider sweep `git grep -niwE 'v2'` over the same paths plus `scripts/slurm_frontier` also
finds words that are NOT the IGNITE generation. Leave these alone: the CO2 V2 chord (`ne_line`,
`flags/rules.py`, `signals.yaml`, `flags.yaml`, `census.sbatch`, `test_corpus_signals.py`,
`test_build_store.py`), `all-MiniLM-L6-v2`, the LLM prompt version in `llm.yaml`, the evalsets
versioning in `configs/shot_design/evalsets/README.md`, the codec-v2 proposal in
`docs/models/ignite-rollout-quality-plan.md`, and the measured-vocabulary note in
`scripts/slurm_frontier/_submit_spectro_pair_arms.sh`. Every IGNITE-generation mention goes,
including prose ("v2's twelve", "v2 came from the Hub") in docstrings, comments and tests.

### P0.1 Config and the model block

- [ ] `configs/shot_design/ignite_modalities.yaml`: delete the comment block and
  `model_generations:` (from line 300 to the end of that block). Rewrite the comments at
  11-13, 43, 64, 69-71, 79 and 145 so they describe v4 alone (no "v2 had ...").
- [ ] `src/shot_design/shotdb/ignite.py`:

```python
def model_cfg() -> dict:
    """The IGNITE bundle: `model:` in configs/shot_design/ignite_modalities.yaml (v4).

    A local copy taken with `pin_bundle` from `codec_tmpl`/`dynamics_src`, or a snapshot of
    `repo_id` @ `revision` taken with `download_bundle`; either way verified by the sha256
    table in its own manifest.
    """
    return load_yaml("ignite_modalities.yaml")["model"]
```

  Delete `_select_model_cfg` and `_GENERATION_NOTED` (and the `os` import if nothing else uses
  it). `bundle_identity` reads `model_cfg()["generation"]`; `check_same_bundle` compares
  `old_model.get(key)` for every key (no v2 default). `download_bundle` keeps its body (it
  follows `repo_id`); its docstring and error lose the v2 wording ("14 codecs (453 MB)" becomes
  the v4 count read from the manifest, or is dropped). `load_codecs`' docstring loses "or
  `download_bundle` for v2" and "(v4: 15)" becomes "15".
- [ ] Every `.get("generation", "v2")` becomes `["generation"]`, every
  `.get("t0_start_s", 0.0)` becomes `["t0_start_s"]`: `cli.py:433` and `:486` (the `absent`
  message is "not installed (--download --full or --pin)"), the v2 manifest branch at
  `cli.py:492-494` (`channels` vs `n_tok`: keep only `n_tok`), `simulate/cli.py:147`,
  `design/program_reference.py:50-59` (`frame_origin_s` returns
  `float(ignite.model_cfg()["t0_start_s"])`, docstring says "1.0 for v4" only) and its comment
  at 192, `shotdb/build.py:1383`, `design/seed.py` docstring (15-19, 39-42, 74),
  `design/assistant.py`.
- [ ] Readers of `simulation.h5` require `frame_origin_s`: `simulate/report.py:133` stops
  defaulting `codec_generation` to `"v4"` and reads `frame_origin_s`;
  `scripts/shot_design/batch_collect.py:117` keeps recording `codec_generation` when present
  (old files) but nothing branches on it.
- [ ] `scripts/shot_design/g_enc.py`: keep the v4 gate; delete the v2 historical record (11-27,
  125-136, 198-199, 404-409, 518, 591 and whatever depends on them).

### P0.2 Design defaults follow v4

- [ ] `src/shot_design/ui/static/design.js:97-98`: the new-design window defaults become
  `start_s: 2, end_s: 6` (2.0 s is the earliest start: frame 0 at 1.0 s plus 20 seed frames).
- [ ] `src/shot_design/cli.py:1999`: `--n-predict` defaults to `None`, help "predicted frames
  (default: all the seed and the checkpoint allow)". In `simulate/cli.py:run`, resolve it after
  the dynamics model is loaded, reading `trained = cfg.max_frames` BEFORE any assignment (the
  existing comment explains why):

```python
        n_frames = design_seed["n_frames"]
        n_predict = (
            args.n_predict
            if args.n_predict is not None
            else min(n_frames, trained) - args.k0
        )
        total = args.k0 + n_predict
        if n_predict < 1 or total > n_frames or total > trained:
            raise ValueError(
                f"--k0 {args.k0} + --n-predict {n_predict} = {total} does not fit the "
                f"design seed's {n_frames} frames and the checkpoint's {trained}"
            )
        cfg.k0_seed, cfg.n_predict = args.k0, n_predict
```

  Use `n_predict` (not `args.n_predict`) everywhere below, including the value recorded in
  `simulation.h5`'s attrs. When `args.n_predict` is given, keep the cheap seed check before the
  model load so a bad flag fails fast.

### P0.3 Stellar Simulate

- [ ] `src/shot_design/config.py` `Paths` gains
  `simulate_submit_cmd: str = "sbatch scripts/shot_design/simulate.sbatch {ident}"`.
  `configs/shot_design/paths.yaml` gets
  `simulate_submit_cmd: sbatch scripts/shot_design/simulate.sbatch {ident}` and
  `paths.frontier.yaml` gets
  `simulate_submit_cmd: sbatch scripts/slurm_frontier/shot_design_simulate.sh {ident}`.
  `configs/shot_design/ui.yaml`'s `simulate:` block keeps only `poll_s: 15` (check who reads
  `poll_s` and keep it working).
- [ ] `src/shot_design/ui/simulate_routes.py:submit`:
  `cmd = request.app.state.paths.simulate_submit_cmd.format(ident=ident)` (check how the app
  holds its paths; use the same object the other routes use), and its docstring says the command
  comes from the cluster's paths file. Update the route's tests.
- [ ] New `scripts/shot_design/simulate.sbatch` (it does NOT source `_stellar_common.sh`, which
  refuses the production root the UI runs against):

```bash
#!/bin/bash
#SBATCH --job-name=sd-simulate
#SBATCH --partition=gpu
#SBATCH --qos=gpu-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/ideate/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/ideate/runs/slurm/%j.out
# One design's paired real/proposed IGNITE v4 rollout on one A100: what the UI's Simulate
# button submits on Stellar (paths.yaml: simulate_submit_cmd). The Stellar twin of
# scripts/slurm_frontier/shot_design_simulate.sh.
#   sbatch scripts/shot_design/simulate.sbatch <ident> [simulate args]
# Sizing is a pilot guess (v4, 80 frames x 10 passes, estimated 70-130 min on an A100):
# resize --mem and --time from the first run's MaxRSS and elapsed time.
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-/scratch/gpfs/nc1514/FusionAIHub}"
ENV="$REPO/.pixi/envs/shot-design"
export SHOT_DESIGN_DATA_ROOT="${SHOT_DESIGN_DATA_ROOT:-/scratch/gpfs/EKOLEMEN/nc1514/ideate}"
export LABELER_ROOT="${LABELER_ROOT:-/scratch/gpfs/EKOLEMEN/nc1514/labelmaker}"
export SHOT_DESIGN_CORPUS="${SHOT_DESIGN_CORPUS:-/scratch/gpfs/EKOLEMEN/foundation_model}"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
IDENT="${1:?design ident}"
cd "$REPO"
echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -Is); design $IDENT; root $SHOT_DESIGN_DATA_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun "$ENV/bin/python" -m shot_design simulate "$IDENT" --device cuda "${@:2}"
echo "finished at $(date -Is)"
```

- [ ] `scripts/shot_design/_stellar_common.sh`: delete lines 14-16 and 29, drop the generation
  from the echo at 37, and fix the header's list of scripts that source it: the new
  `simulate.sbatch` does not; `simulate_batch.sbatch` does (check each listed script with
  `grep -l _stellar_common scripts/shot_design/*`).

### P0.4 Decode fix (bug B9) and `frac_static`

- [ ] `src/shot_design/simulate/decode.py:88`: average the signed values,
  `np.mean(sub, axis=(2, 3))`, not `np.abs(...)`. The spectrogram channels are standardised
  log-power; `abs` folded below-mean values onto above-mean ones.
- [ ] `src/shot_design/simulate/report.py:34`: the quantity is "mean z", not "band-power";
  `report.py:178`: apply the `frac_static` filter to both arms, not only the proposed one.
- [ ] `dev/paper/figures/decode_all_v4.py:31`: `arr.mean(axis=(1, 3))`;
  `dev/paper/figures/make_figures.py:146-147`: `origin = float(f.attrs["frame_origin_s"])`;
  `make_figures.py:161`: `np.asarray(arr).mean(1)`.
- [ ] Tests: one test per fix, e.g. a decode of a spectrogram whose values are `-2` everywhere
  gives `-2`, not `2`; a report whose real arm is static is filtered like the proposed arm.

### P0.5 Tests

- [ ] Delete `tests/shot_design/test_ignite_generation_override.py`.
- [ ] `tests/shot_design/conftest.py:94-97`: delete the pin that forces `t0_start_s = 0.0`
  (keep dropping `frame_codes_cache`).
- [ ] Rewrite fixtures in the v4 frame (origin 1.0 s): `test_simulate_cli.py:238`,
  `test_frame_origin.py`, `test_seed.py:584`, and every test that the conftest pin was hiding
  (run the suite to find them). A design window that started at 1.0 s now starts at 2.0 s.
- [ ] `test_seed.py`'s shipped-cache test compares against `model.frame_codes_cache` (v4 ships
  no `frame_codes/`). Today the suite gives `1834 passed, 2 failed, 3 skipped`, and the two
  failures are that test (`test_seed.py::test_encode_frame_codes_writes_the_shipped_dict_structure`)
  and `test_ignite_v4.py::test_model_cfg_declares_fifteen_v4_modalities`, which the conftest pin
  breaks.
- [ ] Test comments that describe a still-needed mechanism by naming v2 are reworded to the
  mechanism ("vocabularies from another generation", "a cache from another bundle").

### P0.6 Docs and paper

- [ ] Remove the v2 instructions from `docs/clusters/stellar.md` (105 and around),
  `docs/clusters/frontier.md`, `docs/shot-design/simulation.md` (96 and the batch example at 134,
  which gains the Stellar single-design script), `docs/models/ignite.md`, `docs/data/corpus.md`,
  `dev/paper/README.md`. Say what v4 is, where it lives, and that Stellar runs it.

### Part 0 acceptance

- [ ] The inventory grep is empty; the wider sweep finds only the words listed above.
- [ ] Both suites pass under `-W error` (commands in Global Constraints); ruff is clean on
  every touched Python file.
- [ ] No production writes, no `sbatch`, no pixi install/lock/update, no `pyproject.toml` or
  `pixi.lock` change.
- [ ] Commits on `v4-only` (one per P0.x is fine); `.superpowers/sdd/task-P0-report.md`
  force-added: what changed, the grep output, both suite summaries, anything left undone and
  why. The Stellar Simulate run is the session's job after the merge, not yours.

---

## Part A -- label review (brief for one Codex agent)

Work in `/scratch/gpfs/nc1514/FusionAIHub-review` on branch `label-review`. Do A1-A9 in order,
one commit per task. Read the spec's Part A first.

**Part A's code is already written and tested.** It was built and exercised in a scratch copy of
this repository, including a manual drive of the page in headless Chromium. The tasks were then
replayed in order on a clean checkout of this plan's base commit: each task's tests failed
before its code went in and passed after, with the counts each step quotes, and the whole
`tests/labeler` suite passed at the end. The blocks below are those files. Your job is to land
them exactly, run what each step says, and report. When a result differs from what a step says
to expect, stop and report it rather than improvising a fix.

A block is introduced by a line of one of two forms:

- ``Write `path`:`` -- the block is the file's complete content.
- ``Patch `path`:`` -- the block is a git patch against the plan's base commit.

One command, run from the worktree root, puts named blocks in place: it writes each `Write` block
(creating directories, ending the file with one newline) and `git apply`s each `Patch` block.

```bash
python3 .claude/superpowers/plans/extract_blocks.py \
  .claude/superpowers/plans/2026-09-22-label-review-v4-only-pixi.md <path> [<path> ...]
```

Use it rather than retyping. `app.js` alone is over a thousand lines, and a hand copy that drops
one character is a bug no test names. ``Delete `path` `` lines are `git rm path`.

The new package is `src/labeler/events/review/`:

```
review/
  __init__.py     package docstring only
  labels.py       one label per shot: normalise, read, save, states, queue, resume
  rows.py         the row store: Grid, ImageRow, TraceRow, pool, write, meta, read_window
  alfven.py       Heidbrink AE rows from CO2
  panel_rows.py   any event's panels as rows
  build.py        build(event, shot, paths) -> Path, and the CLI
```

The app (`ui/app.py`) and the page (`ui/static/`) are rewritten on top of it in A7 and A8.
Nothing in Part A edits `pyproject.toml`. The `labeler-raw` task that A1 breaks is removed in
Part E.

Each step names the test files to run; the command is always

```bash
PYTHONPATH=$PWD/src pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker \
  python -m pytest <files> -q -W error -rs
```

The page tests (A8) need `node` on `PATH`; if `which node` prints nothing, run
`export PATH=$HOME/.nvm/versions/node/v25.6.1/bin:$PATH` first. The browser test also needs the
Playwright headless shell, which is under `~/.cache/ms-playwright/chromium_headless_shell-1181/`.
They skip themselves when either is missing; say in the report whether they ran.

### Task A1: The raw cache follows the root; promote and clean go

**Files:**
- Modify: `src/labeler/config.py`, `src/labeler/events/raw.py`
- Test: `tests/labeler/test_config.py`, `tests/labeler/test_events_raw.py`

**Interfaces:**
- Produces: `Paths.raw_cache` defaults to `root / "raw"` (`LABELER_RAW_CACHE` still overrides
  it); `Paths.spectrograms -> Path` (`root / "spectrograms"`);
  `Paths.spectrogram_file(event: str, shot: int) -> Path` (`spectrograms / event / f"{shot}.h5"`).
  `raw.py` keeps `raw_signal`, `write_group`, `cache_path`, `progress_for` and `FDP_UI_COMMAND`;
  it loses `groups_in`, `CORPUS_GROUPS`, `promote`, `clean` and its command line.

- [ ] **Step 1: The failing tests.** `test_config.py` stops importing `DEFAULT_RAW_CACHE` and says
  the cache follows the root; `test_events_raw.py` loses the `promote`/`clean` tests and reads a
  file's groups with its own `_groups`.

```bash
python3 .claude/superpowers/plans/extract_blocks.py \
  .claude/superpowers/plans/2026-09-22-label-review-v4-only-pixi.md \
  tests/labeler/test_config.py tests/labeler/test_events_raw.py
```

Patch `tests/labeler/test_config.py`:

```diff
diff --git a/tests/labeler/test_config.py b/tests/labeler/test_config.py
index 39290c2..c6793dd 100644
--- a/tests/labeler/test_config.py
+++ b/tests/labeler/test_config.py
@@ -1,7 +1,7 @@
 """Paths resolve from the environment and nothing else hard-codes a root."""
 from pathlib import Path
 
-from labeler.config import DEFAULT_RAW_CACHE, Paths, git_sha
+from labeler.config import Paths, git_sha
 
 
 def test_default_root_is_group_storage():
@@ -110,13 +110,18 @@ def test_the_label_tables_root_is_the_repo_and_is_overridable(
     assert not tables.exists()
 
 
-def test_raw_cache_defaults_under_the_repo():
-    # Label DATA lives under data/events for the same reason: a run from a
-    # SLURM scratch directory must find the same place a run from the repo
-    # does, so this resolves off this file, never off cwd.
-    assert Paths().raw_cache == DEFAULT_RAW_CACHE
-    assert DEFAULT_RAW_CACHE.name == "raw"
-    assert DEFAULT_RAW_CACHE.parent.name == ".cache"
+def test_raw_cache_follows_the_root(tmp_path):
+    # Beside the 181 prefetched AE shots, on group storage, not in the repository.
+    assert Paths().raw_cache == Paths().root / "raw"
+    assert Paths(root=tmp_path).raw_cache == tmp_path / "raw"
+
+
+def test_spectrograms_live_under_the_root(tmp_path):
+    paths = Paths(root=tmp_path)
+    assert paths.spectrograms == tmp_path / "spectrograms"
+    assert paths.spectrogram_file("alfven_eigenmode", 170790) == (
+        tmp_path / "spectrograms" / "alfven_eigenmode" / "170790.h5"
+    )
 
 
 def test_raw_cache_honours_the_environment(monkeypatch, tmp_path):
```

Patch `tests/labeler/test_events_raw.py`:

```diff
diff --git a/tests/labeler/test_events_raw.py b/tests/labeler/test_events_raw.py
index a7de91e..ef70c43 100644
--- a/tests/labeler/test_events_raw.py
+++ b/tests/labeler/test_events_raw.py
@@ -1,7 +1,6 @@
 """The three-tier raw read, and the cache file it writes."""
 from __future__ import annotations
 
-import os
 import shutil
 import threading
 
@@ -15,6 +14,11 @@ from labeler.events.verify import NoDataError
 from labeler.features.store import FeatureArray
 
 
+def _groups(path):
+    with h5py.File(path, "r") as f:
+        return set(f)
+
+
 def write_corpus_file(path, group, times_ms, values):
     """A corpus file by hand: seconds on xdata, no attributes anywhere."""
     path.parent.mkdir(parents=True, exist_ok=True)
@@ -93,7 +97,7 @@ def test_write_group_is_additive(roots, tmp_path):
     path = tmp_path / "6_processed.h5"
     raw.write_group(path, "co2", np.arange(4.0), np.zeros((1, 4)))
     raw.write_group(path, "ece", np.arange(4.0), np.ones((1, 4)))
-    assert raw.groups_in(path) == {"co2", "ece"}
+    assert _groups(path) == {"co2", "ece"}
 
 
 def test_write_group_leaves_no_partial_file_when_it_fails(roots, tmp_path):
@@ -143,7 +147,7 @@ def test_write_group_leaves_no_partial_file_when_the_h5py_write_fails(
 
     # The original file is exactly what it was before the failed write -
     # still holding co2, not holding a half-written ece.
-    assert raw.groups_in(path) == {"co2"}
+    assert _groups(path) == {"co2"}
     with h5py.File(path, "r") as f:
         assert np.allclose(f["co2"]["ydata"][:], 0.0)
     # And the scratch file the failed write created is gone, not left
@@ -202,7 +206,7 @@ def test_write_group_survives_concurrent_writers_to_the_same_path(
     t2.join(timeout=10)
 
     assert not errors
-    assert raw.groups_in(path) == {"base", "co2", "ece"}
+    assert _groups(path) == {"base", "co2", "ece"}
 
 
 def test_a_miss_fetches_and_fills_the_cache(roots, monkeypatch):
@@ -382,131 +386,3 @@ def test_ece_fetches_the_channels_a_corpus_row_would_index(roots, monkeypatch):
     assert seen["exprs"][-1].endswith("TECEF48")
     assert seen["exprs"] == sorted(seen["exprs"]), "ascending, like a corpus row"
     assert len(set(seen["exprs"])) == 48, "no channel fetched twice"
-
-
-def test_promote_refuses_a_partial_shot_and_names_what_is_missing(roots):
-    raw.write_group(raw.cache_path(12, paths=roots), "co2",
-                    np.arange(4.0), np.zeros((1, 4)))
-    with pytest.raises(ValueError) as caught:
-        raw.promote(12, paths=roots)
-    message = str(caught.value)
-    assert "1 of 32" in message
-    assert "ece" in message, "the refusal must name groups that are missing"
-    assert "--partial" in message
-    assert raw.cache_path(12, paths=roots).is_file(), "nothing moved"
-
-
-def test_promote_moves_a_partial_shot_when_told_to(roots):
-    raw.write_group(raw.cache_path(13, paths=roots), "co2",
-                    np.arange(4.0), np.ones((1, 4)))
-    landed = raw.promote(13, partial=True, paths=roots)
-    assert landed == roots.corpus / "13_processed.h5"
-    assert landed.is_file()
-    assert not raw.cache_path(13, paths=roots).exists(), "a move, not a copy"
-
-
-def test_a_promoted_shot_reads_identically_from_tier_one(roots):
-    times, values = np.arange(10.0), np.arange(20.0).reshape(2, 10)
-    raw.write_group(raw.cache_path(14, paths=roots), "co2", times, values)
-    before = raw.raw_signal(14, "co2", paths=roots)
-    assert before.attrs["tier"] == "cache"
-    raw.promote(14, partial=True, paths=roots)
-    after = raw.raw_signal(14, "co2", paths=roots)
-    assert after.attrs["tier"] == "corpus"
-    assert np.allclose(before.x, after.x)
-    assert np.allclose(before.y, after.y)
-
-
-def test_corpus_groups_matches_a_real_corpus_shot(roots):
-    """Pin CORPUS_GROUPS against reality so it cannot silently rot.
-
-    The 32 names below are hardcoded independently of `raw.CORPUS_GROUPS` -
-    measured by hand off a real corpus file,
-    `/scratch/gpfs/EKOLEMEN/foundation_model/185601_processed.h5` - so this
-    cannot become tautological by looping over `raw.CORPUS_GROUPS` itself.
-    A synthetic cache file holding exactly these must pass the completeness
-    gate without `--partial`. If `CORPUS_GROUPS` ever drifts from what a
-    real `*_processed.h5` actually holds - an invented name added, a real
-    one dropped - this either refuses a genuinely complete shot or, worse,
-    accepts a genuinely incomplete one.
-    """
-    real_groups = (
-        "beam_voltage", "bes", "bolo", "cer_rot", "cer_ti", "co2", "ece",
-        "ech_pol_angle", "ech_polarization", "ech_power", "ech_tor_angle",
-        "filterscopes", "gas_flow", "gas_raw", "i_coil", "ich", "irtv",
-        "langmuir", "mhr", "mirnov", "mse", "neutron_rate", "pinj", "rmp",
-        "sxr", "tangtv", "tinj", "ts_core_density", "ts_core_temp",
-        "ts_tangential_density", "ts_tangential_temp", "vib",
-    )
-    assert len(real_groups) == 32
-    path = raw.cache_path(17, paths=roots)
-    for group in real_groups:
-        raw.write_group(path, group, np.arange(4.0), np.zeros((1, 4)))
-    landed = raw.promote(17, paths=roots)
-    assert landed == roots.corpus / "17_processed.h5"
-    assert raw.groups_in(landed) == set(real_groups)
-
-
-def test_promote_survives_a_cross_filesystem_copy_failure(roots, monkeypatch):
-    """`os.link` failing on the direct source->target step must not corrupt.
-
-    Forces the fallback branch `promote` takes when the cache and the
-    corpus are on different filesystems (`os.link`, like `replace`, cannot
-    cross them). `os.link` is patched to raise `OSError` on its first call
-    only - the direct `source -> target` attempt - and to behave normally
-    after, so the second call (`scratch -> target`, inside the fallback)
-    succeeds. Against the unfixed function (a bare `shutil.copy2` straight
-    to `target`) this same failure mode would have to be simulated on
-    `copy2` instead, and would leave a truncated file sitting in the
-    corpus root; here the assertion is that the shot arrives intact and no
-    scratch file is left behind.
-    """
-    times, values = np.arange(4.0), np.ones((1, 4))
-    raw.write_group(raw.cache_path(18, paths=roots), "co2", times, values)
-
-    real_link = os.link
-    calls = {"n": 0}
-
-    def flaky_link(src, dst, *args, **kwargs):
-        calls["n"] += 1
-        if calls["n"] == 1:
-            raise OSError("simulated cross-filesystem failure")
-        return real_link(src, dst, *args, **kwargs)
-
-    monkeypatch.setattr(os, "link", flaky_link)
-
-    landed = raw.promote(18, partial=True, paths=roots)
-
-    assert calls["n"] == 2, "the fallback must retry via the scratch copy"
-    assert landed == roots.corpus / "18_processed.h5"
-    with h5py.File(landed, "r") as f:
-        assert np.allclose(f["co2"]["ydata"][:], 1.0)
-    assert not raw.cache_path(18, paths=roots).exists(), "a move, not a copy"
-    leftovers = [p for p in roots.corpus.glob(".*") if p.is_file()]
-    assert leftovers == [], "no scratch file left behind"
-
-
-def test_promote_refuses_to_clobber_a_corpus_shot(roots):
-    write_corpus_file(roots.corpus / "15_processed.h5", "co2",
-                      np.arange(4.0), np.zeros((1, 4)))
-    raw.write_group(raw.cache_path(15, paths=roots), "co2",
-                    np.arange(4.0), np.ones((1, 4)))
-    with pytest.raises(FileExistsError):
-        raw.promote(15, partial=True, paths=roots)
-
-
-def test_clean_empties_the_cache_and_a_read_refetches(roots, monkeypatch):
-    calls = []
-
-    def fake_fdp_signal(shot, exprs, *, tree, via, t_range=None, **kwargs):
-        calls.append(shot)
-        return FeatureArray(x=np.arange(10.0),
-                            y=np.zeros((4, 10), dtype="float32"),
-                            attrs={"units": "ms"})
-
-    monkeypatch.setattr(raw, "fdp_signal", fake_fdp_signal)
-    raw.raw_signal(16, "co2", paths=roots)
-    assert raw.clean(paths=roots) > 0
-    assert not raw.cache_path(16, paths=roots).exists()
-    raw.raw_signal(16, "co2", paths=roots)
-    assert len(calls) == 2
```

- [ ] **Step 2: Run** `tests/labeler/test_config.py tests/labeler/test_events_raw.py`. Expect `2 failed, 27 passed`:
  `test_raw_cache_follows_the_root` (the cache is still under `.cache`) and
  `test_spectrograms_live_under_the_root`
  (`AttributeError: 'Paths' object has no attribute 'spectrograms'`).

- [ ] **Step 3: The code.** `config.py`: the cache defaults to `<root>/raw` and the store's paths
  appear. `raw.py`: a new module docstring; `groups_in`, `CORPUS_GROUPS`, `promote`, `clean` and
  the command line go, and `write_group`'s docstring stops describing the promote workflow.

```bash
python3 .claude/superpowers/plans/extract_blocks.py \
  .claude/superpowers/plans/2026-09-22-label-review-v4-only-pixi.md \
  src/labeler/config.py src/labeler/events/raw.py
```

Patch `src/labeler/config.py`:

```diff
diff --git a/src/labeler/config.py b/src/labeler/config.py
index 02f08c6..fd7e320 100644
--- a/src/labeler/config.py
+++ b/src/labeler/config.py
@@ -42,13 +42,6 @@ DEFAULT_LOGS_JSONL = Path(
 #: package and `LABELER_LABEL_TABLES` is the answer. Overridable anyway,
 #: because a table too large or too restricted to commit lives on /scratch.
 DEFAULT_LABEL_TABLES = Path(__file__).resolve().parents[2] / "data" / "events"
-#: Where a live fetch parks a shot's raw record. Deliberately the PROJECT
-#: directory and not the corpus: EKOLEMEN is the long-term home for bulk raw
-#: signal data and has the capacity for it, while this directory is meant for
-#: temporary and smaller things. A fetch lands here as scratch and stays
-#: scratch until `raw.promote` moves it. `.cache` is gitignored, and deleting
-#: this directory at any time is safe - the next read refetches.
-DEFAULT_RAW_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "raw"
 
 
 @dataclass(frozen=True)
@@ -60,21 +53,26 @@ class Paths:
     text_root: Path = DEFAULT_TEXT
     logs_jsonl: Path = DEFAULT_LOGS_JSONL
     label_tables: Path = DEFAULT_LABEL_TABLES
-    raw_cache: Path = DEFAULT_RAW_CACHE
+    #: Where a live fetch parks a shot's raw record. None means `<root>/raw`, beside the
+    #: prefetched AE shots; deleting it is safe, the next read refetches.
+    raw_cache: Path | None = None
+
+    def __post_init__(self) -> None:
+        if self.raw_cache is None:
+            object.__setattr__(self, "raw_cache", self.root / "raw")
 
     @classmethod
     def from_env(cls) -> Paths:
+        raw_cache = getenv("LABELER_RAW_CACHE")
         return cls(
             root=Path(getenv("LABELER_ROOT", str(DEFAULT_ROOT))),
             corpus=Path(getenv("LABELER_CORPUS", str(DEFAULT_CORPUS))),
-            text_root=Path(getenv("LABELER_TEXT_ROOT",
-                                          str(DEFAULT_TEXT))),
-            logs_jsonl=Path(getenv("LABELER_LOGS_JSONL",
-                                           str(DEFAULT_LOGS_JSONL))),
-            label_tables=Path(getenv("LABELER_LABEL_TABLES",
-                                             str(DEFAULT_LABEL_TABLES))),
-            raw_cache=Path(getenv("LABELER_RAW_CACHE",
-                                          str(DEFAULT_RAW_CACHE))),
+            text_root=Path(getenv("LABELER_TEXT_ROOT", str(DEFAULT_TEXT))),
+            logs_jsonl=Path(getenv("LABELER_LOGS_JSONL", str(DEFAULT_LOGS_JSONL))),
+            label_tables=Path(
+                getenv("LABELER_LABEL_TABLES", str(DEFAULT_LABEL_TABLES))
+            ),
+            raw_cache=Path(raw_cache) if raw_cache else None,
         )
 
     @property
@@ -175,6 +173,14 @@ class Paths:
     def text_file(self, shot: int) -> Path:
         return self.text_root / f"shot_{shot}.txt"
 
+    @property
+    def spectrograms(self) -> Path:
+        """The review store: one rows file per shot, `<event>/<shot>.h5`."""
+        return self.root / "spectrograms"
+
+    def spectrogram_file(self, event: str, shot: int) -> Path:
+        return self.spectrograms / event / f"{int(shot)}.h5"
+
     def mkdirs(self) -> None:
         for d in (self.features, self.labels, self.models, self.runs,
                   self.validation, self.events, self.masks, self.annotate,
```

Patch `src/labeler/events/raw.py`:

```diff
diff --git a/src/labeler/events/raw.py b/src/labeler/events/raw.py
index 8f9f2f6..f99e95d 100644
--- a/src/labeler/events/raw.py
+++ b/src/labeler/events/raw.py
@@ -1,15 +1,9 @@
 """One way to get a raw signal, whatever tier it happens to live on.
 
-Three places are tried in order: the corpus, the project's fetch cache, and
-a live fetch that writes the cache. A caller cannot tell which one answered
-except by looking at `attrs["tier"]`, which exists for diagnostics and for
-the promote command, not for branching.
-
-The split between the two on-disk roots is about what the storage is FOR.
-EKOLEMEN holds the long-term bulk raw record and has the capacity for it.
-The project directory has room but is meant for temporary and smaller
-things, so a fetch lands there as scratch. Nothing here ever writes the
-corpus; `promote` does, deliberately and by hand.
+Three places are tried in order: the corpus, the fetch cache (`$LABELER_ROOT/raw`),
+and a live fetch that writes the cache. A caller cannot tell which one answered
+except by `attrs["tier"]`, which exists for diagnostics, not for branching.
+Nothing here writes the corpus.
 """
 
 from __future__ import annotations
@@ -107,17 +101,6 @@ def cache_path(shot: int, *, paths: Paths | None = None) -> Path:
     return paths.raw_cache / f"{int(shot)}_processed.h5"
 
 
-def groups_in(path) -> set[str]:
-    """The group names one corpus-layout file holds; empty if it is absent."""
-    import h5py
-
-    path = Path(path)
-    if not path.is_file():
-        return set()
-    with h5py.File(path, "r") as f:
-        return set(f.keys())
-
-
 def write_group(path, group: str, times_ms, values) -> None:
     """Add one group to a corpus-layout file, atomically and additively.
 
@@ -131,11 +114,11 @@ def write_group(path, group: str, times_ms, values) -> None:
     process, not separate processes - `co2` and `ece` for the same shot can
     land on different threads at once. This function serializes those
     threads against each other (see `_lock_for`), so the additivity promise
-    above actually holds. It does NOT serialize across separate OS
-    processes: this module's own CLI running alongside the server, or a
-    future multi-worker deployment, could still race. No file locking is
-    used to close that gap - this deployment is single-process, and adding
-    it now would guard against a case that doesn't exist yet.
+    above actually holds. It does NOT serialize separate OS processes: the
+    store build hands each shot to one worker process, so its workers never
+    share a file, but the server fetching a shot the build is fetching at
+    the same moment could race. That is rare enough to go without file
+    locking.
 
     `times_ms` arrives in milliseconds, the convention `corpus_signal` and
     `fdp_signal` both return, and is stored in SECONDS, the convention the
@@ -249,8 +232,8 @@ def _holds_record(shot: int, group: str, root: Path) -> bool:
     the corpus's record does not cover but the cache's does would raise
     WindowEmptyError here instead of falling through to the tier that could
     have answered it. Unreachable today - `write_group` always writes a
-    whole record and `promote` moves whole files, never partial ones - but
-    a future partial-record cache would need to guard against this.
+    whole record - but a future partial-record cache would need to guard
+    against this.
     """
     import h5py
 
@@ -366,153 +349,3 @@ def _fetch(shot, group, *, channels, t_range, paths) -> FeatureArray:
         shot, group, channels=channels, t_range=t_range, corpus=paths.raw_cache
     )
     return FeatureArray(x=array.x, y=array.y, attrs={**array.attrs, "tier": "fetch"})
-
-
-#: The 32 groups a complete corpus shot holds - measured by opening a real
-#: corpus file, `/scratch/gpfs/EKOLEMEN/foundation_model/185601_processed.h5`,
-#: and listing its groups, which match the `signals:` keys in
-#: `src/tokamak_foundation_model/data/config/modalities/modalities.yaml`.
-#: `promote` compares against this to decide whether a cache entry is a
-#: whole shot or a verification fetch of one diagnostic. No EFIT scalars
-#: (ip, betan, q95, ...) belong here: the corpus holds raw diagnostics and
-#: actuators only, never equilibrium or fitted-profile quantities - see
-#: `src/labeler/features/resolve_corpus.py`'s docstring.
-CORPUS_GROUPS: tuple[str, ...] = (
-    "beam_voltage", "bes", "bolo", "cer_rot", "cer_ti", "co2", "ece",
-    "ech_pol_angle", "ech_polarization", "ech_power", "ech_tor_angle",
-    "filterscopes", "gas_flow", "gas_raw", "i_coil", "ich", "irtv",
-    "langmuir", "mhr", "mirnov", "mse", "neutron_rate", "pinj", "rmp",
-    "sxr", "tangtv", "tinj", "ts_core_density", "ts_core_temp",
-    "ts_tangential_density", "ts_tangential_temp", "vib",
-)
-
-
-def promote(shot: int, *, partial: bool = False, paths: Paths | None = None) -> Path:
-    """Move a cached shot into the corpus, where it lives long-term.
-
-    Manual and separate from anything the reviewer clicks: this moves
-    hundreds of megabytes, and a Save that did it as a side effect would be
-    a Save that can half-fail.
-
-    The default refuses an incomplete shot. Training loaders glob
-    `*_processed.h5` in the corpus root, and a verification fetch
-    materialises the one group a panel asked for - so a partial file there
-    is one those globs hand to training with the rest of the groups
-    missing.
-    """
-    paths = Paths.from_env() if paths is None else paths
-    source = cache_path(shot, paths=paths)
-    if not source.is_file():
-        raise FileNotFoundError(f"shot {int(shot)} is not in {paths.raw_cache}")
-    target = paths.corpus / source.name
-    if target.exists():
-        raise FileExistsError(
-            f"{target} already exists; promote never overwrites a corpus "
-            f"shot. Inspect both and remove one by hand."
-        )
-    present = groups_in(source)
-    # WHICH groups are missing, computed before branching, not just how
-    # many. A shot can hold 32 groups that aren't the right 32 - e.g. a
-    # cache polluted by a differently-shaped fetch - and a count-only check
-    # would wave that through.
-    missing = sorted(set(CORPUS_GROUPS) - present)
-    if not partial and missing:
-        raise ValueError(
-            f"shot {int(shot)} holds {len(present)} of {len(CORPUS_GROUPS)} "
-            f"groups; missing {', '.join(missing)}. Training globs "
-            f"*_processed.h5 in the corpus root and would read this as a "
-            f"whole shot. Pass --partial if that is what you want."
-        )
-    target.parent.mkdir(parents=True, exist_ok=True)
-    import os
-    import shutil
-
-    # The `exists()` check above is only the friendly early message for the
-    # common case; it cannot BE the no-clobber guarantee because a second
-    # `promote` for the same shot can pass it before this one finishes. The
-    # guarantee actually lives here: `os.link` stakes the target name
-    # atomically and raises `FileExistsError` itself on collision, straight
-    # from the filesystem, so there is no gap between checking and acting.
-    try:
-        os.link(source, target)
-    except FileExistsError:
-        raise FileExistsError(
-            f"{target} already exists; promote never overwrites a corpus "
-            f"shot. Inspect both and remove one by hand."
-        ) from None
-    except OSError:
-        # `os.link`, like `replace`, cannot cross filesystems - which the
-        # cache and the corpus may well be. Copy to a scratch name IN
-        # `target.parent` first, so the commit step below lands on one
-        # filesystem: a `shutil.copy2` straight to `target` would leave a
-        # truncated `*_processed.h5` sitting exactly where the training
-        # glob looks for a whole shot if it died partway - full disk,
-        # killed process, flaky network filesystem are all realistic on
-        # this cross-filesystem path.
-        scratch = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
-        try:
-            shutil.copy2(source, scratch)
-            # Same guarantee as the direct-link branch above, just staked
-            # on the scratch copy instead of `source`.
-            try:
-                os.link(scratch, target)
-            except FileExistsError:
-                raise FileExistsError(
-                    f"{target} already exists; promote never overwrites a "
-                    f"corpus shot. Inspect both and remove one by hand."
-                ) from None
-        finally:
-            scratch.unlink(missing_ok=True)
-    source.unlink()
-    return target
-
-
-def clean(*, paths: Paths | None = None) -> int:
-    """Delete the whole fetch cache; return the bytes recovered."""
-    import shutil
-
-    paths = Paths.from_env() if paths is None else paths
-    root = paths.raw_cache
-    if not root.is_dir():
-        return 0
-    freed = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
-    shutil.rmtree(root)
-    return freed
-
-
-def main(argv: Sequence[str] | None = None) -> int:
-    """`python -m labeler.events.raw promote 178642 [--partial]` / `clean`."""
-    import argparse
-
-    parser = argparse.ArgumentParser(prog="labeler.events.raw")
-    sub = parser.add_subparsers(dest="command", required=True)
-    move = sub.add_parser("promote", help="move a cached shot into the corpus")
-    move.add_argument("shot", type=int)
-    move.add_argument(
-        "--partial", action="store_true",
-        help="promote a shot that does not hold all 32 groups",
-    )
-    sub.add_parser("clean", help="delete the whole fetch cache")
-    args = parser.parse_args(argv)
-
-    paths = Paths.from_env()
-    if args.command == "clean":
-        freed = clean(paths=paths)
-        print(f"removed {freed / 1e9:.2f} GB from {paths.raw_cache}")
-        return 0
-    try:
-        landed = promote(args.shot, partial=args.partial, paths=paths)
-    # OSError covers FileExistsError and FileNotFoundError already, plus
-    # the plain OSErrors the copy/unlink fallback and `clean`'s rmtree can
-    # raise on the destructive paths - disk full, permission denied, a
-    # stale NFS handle - which deserve `error: ...` and exit 1, not a
-    # traceback.
-    except (ValueError, OSError) as error:
-        print(f"error: {error}")
-        return 1
-    print(f"promoted {args.shot} -> {landed}")
-    return 0
-
-
-if __name__ == "__main__":
-    raise SystemExit(main())
```

- [ ] **Step 4: Run** the same two files: `29 passed`. Ruff on the four files. Then

```bash
git grep -nE 'DEFAULT_RAW_CACHE|groups_in|raw\.promote|raw\.clean|CORPUS_GROUPS' -- \
  src/labeler tests/labeler scripts/labeler docs
```

  prints nothing (`pyproject.toml`'s `labeler-raw` task is Part E's).

- [ ] **Step 5: Commit** `labeler: the raw cache follows LABELER_ROOT; promote and clean go`.

### Task A2: The label store

**Files:**
- Create: `src/labeler/events/review/__init__.py`, `src/labeler/events/review/labels.py`
- Test: `tests/labeler/test_review_labels.py`

**Interfaces:**
- Consumes: `labeler.config.atomic_path`; `interval_tables.INTERVAL_COLUMNS`,
  `category_labels(event) -> dict[str, str]`, `validate_intervals(frame) -> frame` (raises
  `DatabaseError`, a `ValueError`).
- Produces (all in `labeler.events.review.labels`):
  - `LONGEST_WINDOW_MS = 20_000`
  - `Label(window: tuple[int, int], intervals: tuple[tuple[int, int, int], ...] = ())`, frozen;
    `.as_json() -> {"window": [lo, hi], "intervals": [[a, b, c], ...]}`;
    `.rows(shot) -> list[list]` (format-table rows tiling the window).
  - `categories(event) -> dict[int, str]` (category 0 left out)
  - `normalise(window, intervals, known: set[int] | None = None) -> Label` (raises `ValueError`)
  - `labels_path(event_dir)`, `history_path(event_dir)`, `source_path(event_dir) -> Path | None`
  - `read_source(event_dir) -> dict[int, Label]`, `read_saved(event_dir) -> dict[int, Label]`
    (cached per file version; never mutate the returned dict)
  - `read_history(event_dir) -> list[dict]`
  - `save(event_dir, shot, label, *, source: str | None) -> dict` (the history entry)
  - `state(saved: Label | None, source: Label | None) -> "unreviewed" | "confirmed" | "changed"`
  - `queue(event_dir, roster: DataFrame[shot, tier, ...]) -> {"shots": [{shot, tier, state, saved_at}], "resume": int | None}`
  - `resume(shots: list[dict], history: list[dict]) -> int | None`
  - `shot_labels(event_dir, shot) -> {"source", "saved", "state", "last_save"}`

- [ ] **Step 1: The failing test.**

```bash
python3 .claude/superpowers/plans/extract_blocks.py \
  .claude/superpowers/plans/2026-09-22-label-review-v4-only-pixi.md \
  tests/labeler/test_review_labels.py
```

Write `tests/labeler/test_review_labels.py`:

```python
"""One label per shot: normalising, the source table, saving, states, the queue."""

from __future__ import annotations

import pandas as pd
import pytest

from labeler.events.interval_tables import validate_intervals
from labeler.events.review import labels
from labeler.events.review.labels import Label, normalise

SOURCE = (
    "shot,category,t_start,t_end,confidence\n"
    "170815,0,0,100,\n"
    "170815,1,100,300,\n"
    "170815,0,300,2000,\n"
    "178642,0,0,2000,\n"
    "178642,1,500,500.4,\n"
)


@pytest.fixture
def event_dir(tmp_path):
    directory = tmp_path / "alfven_eigenmode"
    (directory / "format").mkdir(parents=True)
    (directory / "format" / "alfven_eigenmode_format_2026_v1.csv").write_text(SOURCE)
    return directory


def test_edges_snap_to_whole_milliseconds_half_up():
    assert normalise((0, 100), [(10.4, 20.5, 1)]).intervals == ((10, 21, 1),)


def test_overlapping_and_touching_spans_merge():
    label = normalise((0, 100), [(10, 30, 1), (20, 40, 1), (50, 55, 1), (55, 60, 1)])
    assert label.intervals == ((10, 40, 1), (50, 60, 1))


def test_a_later_span_paints_over_an_earlier_one():
    label = normalise((0, 100), [(10, 50, 1), (30, 40, 2)])
    assert label.intervals == ((10, 30, 1), (30, 40, 2), (40, 50, 1))


def test_a_category_zero_span_erases():
    label = normalise((0, 100), [(10, 50, 1), (20, 30, 0)])
    assert label.intervals == ((10, 20, 1), (30, 50, 1))


def test_the_window_grows_to_cover_every_span():
    assert normalise((100, 200), [(50, 120, 1), (190, 260, 1)]).window == (50, 260)


def test_a_zero_length_span_is_dropped():
    assert normalise((0, 100), [(10, 10.2, 1)]).intervals == ()


@pytest.mark.parametrize(
    "window, intervals, known, match",
    [
        ((0, 100), [(30, 20, 1)], None, "runs backwards"),
        ((0, 100), [(10, 20, 7)], {1}, "category 7"),
        ((100, 100), [], None, "not 1 to 20000 ms"),
        ((0, 20001), [], None, "not 1 to 20000 ms"),
    ],
)
def test_normalise_refuses(window, intervals, known, match):
    with pytest.raises(ValueError, match=match):
        normalise(window, intervals, known=known)


def test_rows_tile_the_window_with_category_zero_gaps():
    label = normalise((0, 100), [(10, 20, 1), (50, 60, 1)])
    assert label.rows(170815) == [
        [170815, 0, 0, 10, ""],
        [170815, 1, 10, 20, ""],
        [170815, 0, 20, 50, ""],
        [170815, 1, 50, 60, ""],
        [170815, 0, 60, 100, ""],
    ]
    assert Label((0, 50)).rows(1) == [[1, 0, 0, 50, ""]]


def test_the_source_is_read_per_shot_and_point_events_are_ignored(event_dir):
    assert labels.read_source(event_dir) == {
        170815: Label((0, 2000), ((100, 300, 1),)),
        178642: Label((0, 2000)),
    }


def test_the_newest_format_table_by_name_is_the_source(event_dir):
    (event_dir / "format" / "alfven_eigenmode_format_2027_v1.csv").write_text(
        "shot,category,t_start,t_end,confidence\n170815,1,0,50,\n"
    )
    assert labels.source_path(event_dir).name == "alfven_eigenmode_format_2027_v1.csv"
    assert labels.read_source(event_dir) == {170815: Label((0, 50), ((0, 50, 1),))}


def test_an_event_without_a_format_table_has_no_source(tmp_path):
    assert labels.source_path(tmp_path) is None
    assert labels.read_source(tmp_path) == {}


def test_saving_replaces_one_shots_rows_and_appends_the_history(event_dir, monkeypatch):
    monkeypatch.setattr(labels.getpass, "getuser", lambda: "nc1514")
    table = "alfven_eigenmode_format_2026_v1.csv"
    labels.save(event_dir, 170815, normalise((0, 2000), [(100, 300, 1)]), source=table)
    labels.save(event_dir, 178642, normalise((0, 2000), []), source=None)
    second = normalise((0, 2000), [(150, 400, 1)])
    entry = labels.save(event_dir, 170815, second, source=table)

    written = validate_intervals(pd.read_csv(labels.labels_path(event_dir)))
    assert written[["shot", "category", "t_start", "t_end"]].values.tolist() == [
        [170815, 0, 0, 150],
        [170815, 1, 150, 400],
        [170815, 0, 400, 2000],
        [178642, 0, 0, 2000],
    ]
    assert labels.read_saved(event_dir) == {170815: second, 178642: Label((0, 2000))}
    history = labels.read_history(event_dir)
    assert [h["shot"] for h in history] == [170815, 178642, 170815]
    assert history[-1] == entry
    assert entry["reviewer"] == "nc1514"
    assert entry["window"] == [0, 2000] and entry["intervals"] == [[150, 400, 1]]
    assert entry["source"] == table
    assert sorted(p.name for p in (event_dir / "review").iterdir()) == [
        "history.jsonl",
        "labels.csv",
    ]


def test_states():
    source = normalise((0, 2000), [(100, 300, 1)])
    assert labels.state(None, source) == "unreviewed"
    assert labels.state(source, source) == "confirmed"
    assert labels.state(normalise((0, 2000), [(100, 310, 1)]), source) == "changed"
    assert labels.state(normalise((0, 2100), [(100, 300, 1)]), source) == "changed"
    assert labels.state(Label((0, 50)), None) == "confirmed"
    assert labels.state(normalise((0, 50), [(0, 10, 1)]), None) == "changed"


def _shots(*states):
    return [{"shot": shot, "state": state} for shot, state in zip([1, 2, 3, 4], states)]


def test_resume_is_the_first_unreviewed_shot_after_the_newest_save():
    shots = _shots("confirmed", "unreviewed", "changed", "unreviewed")
    assert labels.resume(shots, [{"shot": 3}]) == 4
    assert labels.resume(shots, [{"shot": 4}, {"shot": 1}]) == 2
    assert labels.resume(shots, []) == 2
    wraps = _shots("unreviewed", "confirmed", "unreviewed", "confirmed")
    assert labels.resume(wraps, [{"shot": 3}]) == 1
    done = _shots("confirmed", "changed", "confirmed", "changed")
    assert labels.resume(done, [{"shot": 2}]) == 2
    assert labels.resume([], []) is None


def test_the_queue_follows_the_roster_and_resumes(event_dir):
    roster = pd.DataFrame({"shot": [178642, 170815], "tier": ["unverified", "gold"]})
    assert labels.queue(event_dir, roster) == {
        "shots": [
            {"shot": 178642, "tier": "unverified", "state": "unreviewed",
             "saved_at": None},
            {"shot": 170815, "tier": "gold", "state": "unreviewed", "saved_at": None},
        ],
        "resume": 178642,
    }
    entry = labels.save(event_dir, 178642, normalise((0, 2000), []), source=None)
    after = labels.queue(event_dir, roster)
    assert after["shots"][0] == {
        "shot": 178642,
        "tier": "unverified",
        "state": "confirmed",
        "saved_at": entry["saved_at"],
    }
    assert after["resume"] == 170815


def test_shot_labels_carries_source_saved_state_and_last_save(event_dir):
    assert labels.shot_labels(event_dir, 170815) == {
        "source": {"window": [0, 2000], "intervals": [[100, 300, 1]]},
        "saved": None,
        "state": "unreviewed",
        "last_save": None,
    }
    label = normalise((0, 2000), [(100, 250, 1)])
    entry = labels.save(event_dir, 170815, label, source="x.csv")
    view = labels.shot_labels(event_dir, 170815)
    assert view["saved"] == {"window": [0, 2000], "intervals": [[100, 250, 1]]}
    assert view["state"] == "changed"
    assert view["last_save"] == entry


def test_categories_leave_out_absent():
    assert labels.categories("alfven_eigenmode") == {1: "present"}
    assert labels.categories("minimum_safety_factor") == {
        1: "low",
        2: "hybrid",
        3: "elevated",
        4: "high",
    }
```

- [ ] **Step 2: Run it.** Expect a collection error:
  `ModuleNotFoundError: No module named 'labeler.events.review'`.

- [ ] **Step 3: The code.**

```bash
python3 .claude/superpowers/plans/extract_blocks.py \
  .claude/superpowers/plans/2026-09-22-label-review-v4-only-pixi.md \
  src/labeler/events/review/__init__.py src/labeler/events/review/labels.py
```

Write `src/labeler/events/review/__init__.py`:

```python
"""What the review page reads and writes: one label per shot, and its rows."""
```

Write `src/labeler/events/review/labels.py`:

```python
"""One label per shot, kept as a format table.

`review/labels.csv` holds the current label of every reviewed shot in the format
schema (`shot, category, t_start, t_end, confidence`, ms): a shot's rows tile its
window, each span with its category and the gaps as category 0. `history.jsonl`
gets one line per save. A shot nobody has saved opens on its source label, the
newest `format/*_format_*.csv`.
"""

from __future__ import annotations

import getpass
import json
import math
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import atomic_path
from ..interval_tables import INTERVAL_COLUMNS, category_labels, validate_intervals

LONGEST_WINDOW_MS = 20_000
REVIEW = "review"
_write_lock = threading.Lock()


@dataclass(frozen=True)
class Label:
    """A reviewed window and the categorised spans inside it, in whole ms."""

    window: tuple[int, int]
    intervals: tuple[tuple[int, int, int], ...] = ()

    def as_json(self) -> dict:
        return {
            "window": list(self.window),
            "intervals": [list(span) for span in self.intervals],
        }

    def rows(self, shot: int) -> list[list]:
        """Format-table rows tiling the window; the gaps are category 0."""
        lo = self.window[0]
        cells = _paint(self.window, self.intervals)
        return [[int(shot), c, lo + a, lo + b, ""] for a, b, c in _runs(cells)]


def categories(event: str) -> dict[int, str]:
    """The categories a span can carry; 0 (absent) is the gaps."""
    return {int(k): v for k, v in category_labels(event).items() if k != "0"}


def _ms(t) -> int:
    """Whole ms, halves up (JavaScript's `Math.floor(t + 0.5)`, so the page agrees)."""
    return math.floor(float(t) + 0.5)


def normalise(window, intervals, known: set[int] | None = None) -> Label:
    """Snap to whole ms, grow the window over every span, merge, and clip.

    Later spans paint over earlier ones, so a category-0 span erases.
    """
    lo, hi = _ms(window[0]), _ms(window[1])
    spans = []
    for a, b, c in intervals:
        a, b, c = _ms(a), _ms(b), int(c)
        if b < a:
            raise ValueError(f"span {a}-{b} ms runs backwards")
        if known is not None and c and c not in known:
            raise ValueError(f"category {c} is not one of {sorted(known)}")
        if b > a:
            spans.append((a, b, c))
    painted = [span for span in spans if span[2]]
    if painted:
        lo = min(lo, *(a for a, _, _ in painted))
        hi = max(hi, *(b for _, b, _ in painted))
    if not 0 < hi - lo <= LONGEST_WINDOW_MS:
        raise ValueError(f"window {lo}-{hi} ms is not 1 to {LONGEST_WINDOW_MS} ms long")
    runs = _runs(_paint((lo, hi), spans))
    return Label((lo, hi), tuple((lo + a, lo + b, c) for a, b, c in runs if c))


def _paint(window, spans) -> np.ndarray:
    """One cell per ms of the window, holding the category painted there last."""
    lo, hi = window
    cells = np.zeros(hi - lo, dtype=np.int64)
    for a, b, c in spans:
        start, stop = max(a, lo) - lo, min(b, hi) - lo
        if stop > start:
            cells[start:stop] = c
    return cells


def _runs(cells) -> list[tuple[int, int, int]]:
    """(start, stop, category) of each run of equal cells, as offsets."""
    if not len(cells):
        return []
    edges = np.flatnonzero(np.diff(cells)) + 1
    starts = np.concatenate([[0], edges])
    stops = np.concatenate([edges, [len(cells)]])
    return [(int(a), int(b), int(cells[a])) for a, b in zip(starts, stops)]


def labels_path(event_dir) -> Path:
    return Path(event_dir) / REVIEW / "labels.csv"


def history_path(event_dir) -> Path:
    return Path(event_dir) / REVIEW / "history.jsonl"


def source_path(event_dir) -> Path | None:
    """The newest format table by name. `*_format_*`: RWM's has no event prefix."""
    tables = Path(event_dir).glob("format/*_format_*.csv")
    return max(tables, key=lambda path: path.name, default=None)


def read_source(event_dir) -> dict[int, Label]:
    return _table(source_path(event_dir))


def read_saved(event_dir) -> dict[int, Label]:
    return _table(labels_path(event_dir))


def _table(path) -> dict[int, Label]:
    if path is None or not path.is_file():
        return {}
    stat = path.stat()
    return _read_table(path, stat.st_mtime_ns, stat.st_ino, stat.st_size)


@lru_cache(maxsize=64)
def _read_table(path, _mtime_ns, _ino, _size) -> dict[int, Label]:
    """One label per shot of a format table; cached per file version, never mutate."""
    frame = validate_intervals(pd.read_csv(path))
    frame = frame[frame.t_end - frame.t_start >= 1]  # a point event has no span to edit
    found = {}
    for shot, rows in frame.groupby("shot", sort=False):
        spans = rows[rows.category != 0]
        found[int(shot)] = normalise(
            (rows.t_start.min(), rows.t_end.max()),
            zip(spans.t_start, spans.t_end, spans.category),
        )
    return found


def read_history(event_dir) -> list[dict]:
    path = history_path(event_dir)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(event_dir, shot: int, label: Label, *, source: str | None) -> dict:
    """Replace one shot's rows in `labels.csv` and append the save to the history."""
    with _write_lock:
        saved = dict(read_saved(event_dir))  # a copy: the cached dict is shared
        saved[int(shot)] = label
        frame = pd.DataFrame(
            [row for s in sorted(saved) for row in saved[s].rows(s)],
            columns=list(INTERVAL_COLUMNS),
        )
        validate_intervals(frame)
        with atomic_path(labels_path(event_dir)) as tmp:
            frame.to_csv(tmp, index=False)
        entry = {
            "shot": int(shot),
            "reviewer": getpass.getuser(),
            "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **label.as_json(),
            "source": source,
        }
        with history_path(event_dir).open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
    return entry


def state(saved: Label | None, source: Label | None) -> str:
    if saved is None:
        return "unreviewed"
    if source is None:
        return "confirmed" if not saved.intervals else "changed"
    return "confirmed" if saved == source else "changed"


def queue(event_dir, roster: pd.DataFrame) -> dict:
    """The roster in order, each shot's state and last save, and where to resume."""
    saved, source = read_saved(event_dir), read_source(event_dir)
    history = read_history(event_dir)
    saved_at = {entry["shot"]: entry["saved_at"] for entry in history}
    shots = [
        {
            "shot": int(row.shot),
            "tier": row.tier,
            "state": state(saved.get(int(row.shot)), source.get(int(row.shot))),
            "saved_at": saved_at.get(int(row.shot)),
        }
        for row in roster.itertuples()
    ]
    return {"shots": shots, "resume": resume(shots, history)}


def resume(shots: list[dict], history: list[dict]) -> int | None:
    """The first unreviewed shot after the newest save, wrapping; else that save."""
    if not shots:
        return None
    order = [row["shot"] for row in shots]
    last = history[-1]["shot"] if history else None
    start = order.index(last) + 1 if last in order else 0
    for row in shots[start:] + shots[:start]:
        if row["state"] == "unreviewed":
            return row["shot"]
    return last if last in order else order[0]


def shot_labels(event_dir, shot: int) -> dict:
    shot = int(shot)
    saved = read_saved(event_dir).get(shot)
    source = read_source(event_dir).get(shot)
    history = read_history(event_dir)
    last = next((entry for entry in reversed(history) if entry["shot"] == shot), None)
    return {
        "source": None if source is None else source.as_json(),
        "saved": None if saved is None else saved.as_json(),
        "state": state(saved, source),
        "last_save": last,
    }
```

- [ ] **Step 4: Run it:** `20 passed`. Ruff on the three files.

- [ ] **Step 5: Commit** `labeler review: one label per shot, stored as a format table`.

### Task A3: The row store

**Files:**
- Create: `src/labeler/events/review/rows.py`
- Test: `tests/labeler/test_review_rows.py`

**Interfaces:**
- Consumes: `labeler.config.atomic_path`.
- Produces (all in `labeler.events.review.rows`):
  - `LEVELS = (1, 8, 64)`, `CHUNK_COLUMNS = 512`
  - `Grid(t0_ms: float, dt_ms: float, n: int)`; column `i` spans
    `[t0_ms + i * dt_ms, t0_ms + (i + 1) * dt_ms)`
  - `ImageRow(name, title, values, y0, dy, y_units, z_lo, z_hi, z_units, band=None)`, values
    `uint8 (n_y, n)`, `.kind == "image"`, `.meta() -> dict`
  - `TraceRow(name, title, values, y_units="", legend=[], hlines=[])`, values
    `float32 (2, n_channels, n)` (min then max), `.kind == "trace"`, `.meta() -> dict`
  - `pool(values, level, kind) -> ndarray`
  - `write(path, grid, rows, **info)`: atomic; `info` becomes file attrs (JSON unless scalar)
  - `meta(path) -> {"grid": {"t0", "dt", "n"}, "t_range": [t0, t0 + n * dt], "rows": [{"name", "kind", "title", ...}]}`
  - `read_window(path, t0, t1, cols) -> (bytes, {"t0", "t1", "n"})`; `ValueError` for a window
    outside the record

- [ ] **Step 1: The failing test.**

```bash
python3 .claude/superpowers/plans/extract_blocks.py \
  .claude/superpowers/plans/2026-09-22-label-review-v4-only-pixi.md \
  tests/labeler/test_review_rows.py
```

Write `tests/labeler/test_review_rows.py`:

```python
"""The row store: pooling, the pyramid, and reading a window."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from labeler.events.review import rows
from labeler.events.review.rows import Grid, ImageRow, TraceRow


def _store(path, n=10_000):
    image = ImageRow(
        "R0",
        "R0",
        np.tile(np.arange(n) % 256, (3, 1)).astype(np.uint8),
        y0=0.0,
        dy=1.0,
        y_units="kHz",
        z_lo=-3.0,
        z_hi=27.0,
        z_units="dB",
        band=(80.0, 250.0),
    )
    trace = TraceRow(
        "p1",
        "Density",
        np.stack([np.zeros((2, n)), np.ones((2, n))]).astype(np.float32),
        y_units="1e19 m^-3",
        legend=["a", "b"],
        hlines=[0.5],
    )
    rows.write(path, Grid(0.0, 1.0, n), [image, trace], event="alfven_eigenmode",
               shot=170790, params={"hop": 128})
    return path


@pytest.fixture
def store(tmp_path):
    return _store(tmp_path / "170790.h5")


def test_images_pool_by_max_and_the_last_block_pads_with_its_own_last_column():
    assert rows.pool(np.array([[1, 5, 2, 7, 3]]), 2, "image").tolist() == [[5, 7, 3]]


def test_traces_pool_min_and_max_and_skip_nan_unless_a_block_is_all_nan():
    values = np.array([[[1, np.nan, np.nan, 0]], [[2, np.nan, 5, np.nan]]])
    pooled = rows.pool(values, 2, "trace")
    assert pooled[0].tolist() == [[1, 0]]
    assert pooled[1].tolist() == [[2, 5]]
    assert np.isnan(rows.pool(np.full((2, 1, 2), np.nan), 2, "trace")).all()


def test_the_store_holds_a_pyramid_per_row(store):
    with h5py.File(store, "r") as f:
        shapes = [f["rows/R0"][str(level)].shape for level in rows.LEVELS]
        assert shapes == [(3, 10000), (3, 1250), (3, 157)]
        assert f["rows/p1/64"].shape == (2, 2, 157)
        assert f["rows/R0/1"].chunks == (3, 512)
        assert f["rows/R0/1"].compression == "gzip"
        assert f.attrs["event"] == "alfven_eigenmode" and f.attrs["shot"] == 170790
        assert json.loads(f.attrs["params"]) == {"hop": 128}


def test_meta_describes_the_grid_and_the_rows(store):
    meta = rows.meta(store)
    assert meta["grid"] == {"t0": 0.0, "dt": 1.0, "n": 10000}
    assert meta["t_range"] == [0.0, 10000.0]
    assert meta["rows"] == [
        {"name": "R0", "kind": "image", "title": "R0", "n_y": 3, "y0": 0.0, "dy": 1.0,
         "y_units": "kHz", "z_lo": -3.0, "z_hi": 27.0, "z_units": "dB",
         "band": [80.0, 250.0]},
        {"name": "p1", "kind": "trace", "title": "Density", "n_channels": 2,
         "y_units": "1e19 m^-3", "legend": ["a", "b"], "hlines": [0.5]},
    ]


def test_a_whole_record_read_takes_the_coarsest_level_that_fills_the_columns(store):
    data, grid = rows.read_window(store, 0, 10_000, 1000)
    assert grid == {"t0": 0.0, "t1": 10000.0, "n": 625}
    assert len(data) == 3 * 625 + 2 * 2 * 625 * 4


def test_a_zoomed_read_comes_from_the_finest_level(store):
    data, grid = rows.read_window(store, 100, 150, 1000)
    assert grid == {"t0": 100.0, "t1": 150.0, "n": 50}
    image = np.frombuffer(data[: 3 * 50], dtype=np.uint8).reshape(3, 50)
    assert image[0].tolist() == list(range(100, 150))


def test_a_read_past_the_start_is_clamped_to_the_record(store):
    _, grid = rows.read_window(store, -500, 50, 1000)
    assert grid == {"t0": 0.0, "t1": 50.0, "n": 50}


def test_a_read_outside_the_record_is_refused(store):
    with pytest.raises(ValueError, match="outside the record"):
        rows.read_window(store, 20_000, 30_000, 1000)


def test_a_read_with_fewer_columns_pools_on_the_fly(store):
    data, grid = rows.read_window(store, 0, 1000, 300)
    assert grid == {"t0": 0.0, "t1": 1000.0, "n": 250}
    image = np.frombuffer(data[: 3 * 250], dtype=np.uint8).reshape(3, 250)
    assert image[0, :3].tolist() == [3, 7, 11]


def test_a_read_across_a_chunk_boundary_is_seamless(store):
    data, _ = rows.read_window(store, 500, 530, 1000)
    image = np.frombuffer(data[: 3 * 30], dtype=np.uint8).reshape(3, 30)
    assert image[0].tolist() == [v % 256 for v in range(500, 530)]
    trace = np.frombuffer(data[3 * 30 :], dtype="<f4").reshape(2, 2, 30)
    assert (trace[0] == 0).all() and (trace[1] == 1).all()


def test_a_row_that_does_not_fit_the_grid_is_refused_and_nothing_is_left(tmp_path):
    image = ImageRow("R0", "R0", np.zeros((2, 5), np.uint8), y0=0, dy=1,
                     y_units="kHz", z_lo=0, z_hi=1, z_units="")
    with pytest.raises(ValueError, match="5 columns, the grid 6"):
        rows.write(tmp_path / "s.h5", Grid(0.0, 1.0, 6), [image])
    assert list(tmp_path.iterdir()) == []
```

- [ ] **Step 2: Run it.** Expect a collection error:
  `ImportError: cannot import name 'rows' from 'labeler.events.review'`.

- [ ] **Step 3: The code.**

```bash
python3 .claude/superpowers/plans/extract_blocks.py \
  .claude/superpowers/plans/2026-09-22-label-review-v4-only-pixi.md \
  src/labeler/events/review/rows.py
```

Write `src/labeler/events/review/rows.py`:

```python
"""Review rows on one time grid, stored as a pyramid a window can be read from fast.

A store file holds every row of one shot. Column `i` of the grid spans
`[t0_ms + i * dt_ms, t0_ms + (i + 1) * dt_ms)`. An image row is `uint8`
`(n_y, n)`, row 0 the lowest y; a trace row is `float32` `(2, n_channels, n)`,
each column's minimum and then its maximum, so pooling never hides a spike.
Each row is kept at three pooling factors, `rows/<name>/{1,8,64}`, and a read
takes the coarsest one that still gives the page a column per pixel.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import h5py
import numpy as np

from ...config import atomic_path

LEVELS = (1, 8, 64)
CHUNK_COLUMNS = 512


@dataclass(frozen=True)
class Grid:
    t0_ms: float
    dt_ms: float
    n: int


@dataclass(frozen=True)
class ImageRow:
    name: str
    title: str
    values: np.ndarray
    y0: float
    dy: float
    y_units: str
    z_lo: float
    z_hi: float
    z_units: str
    band: tuple[float, float] | None = None
    kind = "image"

    def meta(self) -> dict:
        meta = {
            "kind": self.kind,
            "title": self.title,
            "n_y": int(self.values.shape[0]),
            "y0": float(self.y0),
            "dy": float(self.dy),
            "y_units": self.y_units,
            "z_lo": float(self.z_lo),
            "z_hi": float(self.z_hi),
            "z_units": self.z_units,
        }
        if self.band is not None:
            meta["band"] = [float(v) for v in self.band]
        return meta


@dataclass(frozen=True)
class TraceRow:
    name: str
    title: str
    values: np.ndarray
    y_units: str = ""
    legend: list[str] = field(default_factory=list)
    hlines: list[float] = field(default_factory=list)
    kind = "trace"

    def meta(self) -> dict:
        return {
            "kind": self.kind,
            "title": self.title,
            "n_channels": int(self.values.shape[1]),
            "y_units": self.y_units,
            "legend": list(self.legend),
            "hlines": [float(v) for v in self.hlines],
        }


def pool(values: np.ndarray, level: int, kind: str) -> np.ndarray:
    """Pool the last axis by `level`: max for images, (min, max) for traces.

    The last block pads with its own last column. NaN is skipped unless a
    whole block is NaN.
    """
    if level == 1:
        return values
    pad = -values.shape[-1] % level
    if pad:
        edge = np.repeat(values[..., -1:], pad, axis=-1)
        values = np.concatenate([values, edge], axis=-1)
    blocks = values.reshape(*values.shape[:-1], -1, level)
    if kind == "image":
        return blocks.max(axis=-1)
    low = np.fmin.reduce(blocks[0], axis=-1)
    high = np.fmax.reduce(blocks[1], axis=-1)
    return np.stack([low, high])


def write(path, grid: Grid, rows, **info) -> None:
    """Write one shot's rows; `info` becomes file attributes (JSON if not scalar)."""
    with atomic_path(path) as tmp, h5py.File(tmp, "w") as f:
        f.attrs.update(
            {
                "t0_ms": float(grid.t0_ms),
                "dt_ms": float(grid.dt_ms),
                "n": int(grid.n),
                "rows": json.dumps([row.name for row in rows]),
            }
        )
        for key, value in info.items():
            scalar = isinstance(value, (str, int, float))
            f.attrs[key] = value if scalar else json.dumps(value)
        for row in rows:
            dtype = "uint8" if row.kind == "image" else "float32"
            values = np.asarray(row.values, dtype=dtype)
            if values.shape[-1] != grid.n:
                raise ValueError(
                    f"row {row.name} has {values.shape[-1]} columns, the grid {grid.n}"
                )
            group = f.create_group(f"rows/{row.name}")
            group.attrs["meta"] = json.dumps(row.meta())
            for level in LEVELS:
                data = pool(values, level, row.kind)
                group.create_dataset(
                    str(level),
                    data=data,
                    chunks=(*data.shape[:-1], min(CHUNK_COLUMNS, data.shape[-1])),
                    compression="gzip",
                    compression_opts=1,
                )


def meta(path) -> dict:
    """The grid, the time range and every row's description."""
    with h5py.File(path, "r") as f:
        t0, dt, n = float(f.attrs["t0_ms"]), float(f.attrs["dt_ms"]), int(f.attrs["n"])
        rows = [
            {"name": name, **json.loads(f["rows"][name].attrs["meta"])}
            for name in json.loads(f.attrs["rows"])
        ]
    grid = {"t0": t0, "dt": dt, "n": n}
    return {"grid": grid, "t_range": [t0, t0 + n * dt], "rows": rows}


def read_window(path, t0: float, t1: float, cols: int) -> tuple[bytes, dict]:
    """Every row over `t0`-`t1` ms at about `cols` columns, as one byte string.

    Rows follow each other in store order: images as `uint8` `(n_y, k)`,
    traces as little-endian `float32` `(2, n_channels, k)`. The dict is the
    grid actually returned, `{"t0", "t1", "n": k}`.
    """
    with h5py.File(path, "r") as f:
        g0, dt, n = float(f.attrs["t0_ms"]), float(f.attrs["dt_ms"]), int(f.attrs["n"])
        i0 = max(0, math.floor((t0 - g0) / dt))
        i1 = min(n, math.ceil((t1 - g0) / dt))
        if i1 <= i0:
            end = g0 + n * dt
            raise ValueError(f"{t0}-{t1} ms is outside the record {g0}-{end} ms")
        level = max((lv for lv in LEVELS if (i1 - i0) // lv >= cols), default=1)
        j0, j1 = i0 // level, math.ceil(i1 / level)
        factor = math.ceil((j1 - j0) / cols)
        k = math.ceil((j1 - j0) / factor)
        parts = []
        for name in json.loads(f.attrs["rows"]):
            group = f["rows"][name]
            kind = json.loads(group.attrs["meta"])["kind"]
            block = pool(group[str(level)][..., j0:j1], factor, kind)
            parts.append(block.astype("uint8" if kind == "image" else "<f4").tobytes())
    start = g0 + j0 * level * dt
    return b"".join(parts), {"t0": start, "t1": start + k * factor * level * dt, "n": k}
```

- [ ] **Step 4: Run it:** `11 passed`. Ruff on both files.

- [ ] **Step 5: Commit** `labeler review: the row store, a pyramid read by window`.

### Task A4: The Heidbrink AE rows

**Files:**
- Create: `src/labeler/events/review/alfven.py`
- Test: `tests/labeler/test_review_alfven.py`

**Interfaces:**
- Consumes: `raw.raw_signal(shot, "co2", paths=paths) -> FeatureArray` (`.x` ms, `.y`
  `(4, T)` float32 for R0, V1, V2, V3, `.attrs["tier"]`), `raw.cache_path(shot, paths=paths)`,
  `Paths.corpus_file(shot)`; A3's `Grid`, `ImageRow`.
- Produces: `alfven.spectrogram_rows(time_ms, chords) -> (Grid, list[ImageRow])`, seven rows
  named `R0, V1, V2, V3, R0xV1, R0xV2, R0xV3`; `alfven.build(event, shot, paths) -> (Grid, rows,
  {"params": PARAMS, "source": {"tier", "path", "size", "mtime_ns"}})`.

- [ ] **Step 1: The failing test.**

```bash
python3 .claude/superpowers/plans/extract_blocks.py \
  .claude/superpowers/plans/2026-09-22-label-review-v4-only-pixi.md \
  tests/labeler/test_review_alfven.py
```

Write `tests/labeler/test_review_alfven.py`:

```python
"""The Heidbrink AE rows, on a synthetic CO2 record with a known mode and burst."""

from __future__ import annotations

import numpy as np
import pytest

from labeler.config import Paths
from labeler.events import raw
from labeler.events.review import alfven

RATE = 5e6 / 3  # the CO2 digitiser, about 1.6667 MHz


def _record(seconds=0.4):
    """Unit noise on four chords; a 120 kHz mode on all four over 150-250 ms,
    and a 180 kHz burst on V2 alone over 50-120 ms."""
    n = int(seconds * RATE)
    t_s = np.arange(n) / RATE
    chords = np.random.default_rng(0).normal(0.0, 1.0, (4, n))
    mode = (t_s > 0.15) & (t_s < 0.25)
    chords[:, mode] += 0.5 * np.sin(2 * np.pi * 120e3 * t_s[mode])
    burst = (t_s > 0.05) & (t_s < 0.12)
    chords[2, burst] += 0.5 * np.sin(2 * np.pi * 180e3 * t_s[burst])
    return t_s * 1000, chords.astype(np.float32)


def _db(grid, row, t_ms, khz):
    """Mean dB over the columns centred strictly inside `t_ms`, at one frequency."""
    centres = grid.t0_ms + (np.arange(grid.n) + 0.5) * grid.dt_ms
    inside = (centres > t_ms[0]) & (centres < t_ms[1])
    level = row.values[round((khz - row.y0) / row.dy), inside].mean()
    return level * (row.z_hi - row.z_lo) / 255 + row.z_lo


@pytest.fixture(scope="module")
def made():
    return alfven.spectrogram_rows(*_record())


def test_the_grid_is_the_heidbrink_stft(made):
    grid, rows = made
    assert grid.dt_ms == pytest.approx(0.256)
    assert [row.name for row in rows] == [
        "R0", "V1", "V2", "V3", "R0xV1", "R0xV2", "R0xV3"
    ]
    for row in rows:
        assert row.values.shape == (257, grid.n) and row.values.dtype == np.uint8
        assert row.y0 == 0 and row.dy == pytest.approx(500 / 512)
        assert row.band == (80.0, 250.0) and (row.z_lo, row.z_hi) == (-3.0, 27.0)


def test_a_mode_on_every_chord_shows_on_every_row(made):
    grid, rows = made
    for row in rows:
        assert _db(grid, row, (160, 240), 120) >= 12, row.name


def test_a_burst_on_one_chord_shows_there_and_nowhere_else(made):
    grid, rows = made
    by_name = {row.name: row for row in rows}
    assert _db(grid, by_name["V2"], (60, 110), 180) >= 12
    for name in ("R0", "V1", "V3", "R0xV1", "R0xV3"):
        assert _db(grid, by_name[name], (60, 110), 180) <= 3, name
    cross = by_name["R0xV2"]
    assert _db(grid, cross, (160, 240), 120) - _db(grid, cross, (60, 110), 180) >= 10


def test_build_reads_co2_through_the_raw_tiers_and_says_where_it_came_from(tmp_path):
    paths = Paths(root=tmp_path, corpus=tmp_path / "corpus")
    t_ms, chords = _record(0.05)
    raw.write_group(raw.cache_path(170790, paths=paths), "co2", t_ms, chords)
    grid, rows, info = alfven.build("alfven_eigenmode", 170790, paths)
    assert info["source"]["tier"] == "cache"
    assert info["source"]["path"] == str(tmp_path / "raw" / "170790_processed.h5")
    assert info["params"] == alfven.PARAMS
    assert len(rows) == 7 and rows[0].values.shape[1] == grid.n
```

- [ ] **Step 2: Run it.** Expect a collection error:
  `ImportError: cannot import name 'alfven' from 'labeler.events.review'`.

- [ ] **Step 3: The code.**

```bash
python3 .claude/superpowers/plans/extract_blocks.py \
  .claude/superpowers/plans/2026-09-22-label-review-v4-only-pixi.md \
  src/labeler/events/review/alfven.py
```

Write `src/labeler/events/review/alfven.py`:

```python
"""The Heidbrink AE rows: CO2 power per chord and cross-power against R0.

The CO2 interferometer's four chords (R0, V1, V2, V3) are resampled to
500 kHz and given a Hann STFT of 512 samples every 128: 0.256 ms columns,
257 bins 0.977 kHz apart up to 250 kHz. Each frequency bin is flattened by
its median over 0-6 s, so the colour is dB above that bin's own background
and one scale (-3 to 27 dB) serves every shot. The cross-power rows average
R0 x conj(chord) over 8 columns before the magnitude: a coherent mode adds
up, incoherent noise cancels.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

from ...config import Paths
from .. import raw
from .rows import Grid, ImageRow

RATE_HZ = 500_000
NPERSEG = 512
HOP = 128
CROSS_COLUMNS = 8
QUIET_MS = (0.0, 6000.0)
Z_DB = (-3.0, 27.0)
BAND_KHZ = (80.0, 250.0)
CHORDS = ("R0", "V1", "V2", "V3")
PARAMS = {
    "rate_hz": RATE_HZ,
    "nperseg": NPERSEG,
    "hop": HOP,
    "cross_columns": CROSS_COLUMNS,
    "quiet_ms": list(QUIET_MS),
    "z_db": list(Z_DB),
    "band_khz": list(BAND_KHZ),
    "chords": list(CHORDS),
}


def spectrogram_rows(time_ms, chords) -> tuple[Grid, list[ImageRow]]:
    """Seven image rows (four powers, three cross-powers) on one grid."""
    time_ms = np.asarray(time_ms, dtype=np.float64)
    # The rate from the span, not a median step: float32 time vectors quantise.
    rate = (len(time_ms) - 1) / ((time_ms[-1] - time_ms[0]) / 1000)
    ratio = Fraction(RATE_HZ / rate).limit_denominator(100)
    x = signal.resample_poly(
        np.asarray(chords, dtype=np.float32),
        ratio.numerator,
        ratio.denominator,
        axis=-1,
    )
    fs = rate * ratio.numerator / ratio.denominator
    freq, t_s, spec = signal.stft(
        x, fs=fs, window="hann", nperseg=NPERSEG, noverlap=NPERSEG - HOP,
        boundary=None, padded=False, axis=-1,
    )
    dt_ms = HOP / fs * 1000
    centres = time_ms[0] + t_s * 1000
    quiet = (centres >= QUIET_MS[0]) & (centres <= QUIET_MS[1])
    if not quiet.any():
        quiet[:] = True
    power = 10 * np.log10(np.abs(spec) ** 2 + 1e-30)
    cross = []
    for k in range(1, len(CHORDS)):
        product = spec[0] * np.conj(spec[k])
        mean = uniform_filter1d(product.real, CROSS_COLUMNS, axis=-1) + 1j * (
            uniform_filter1d(product.imag, CROSS_COLUMNS, axis=-1)
        )
        cross.append(10 * np.log10(np.abs(mean) + 1e-30))
    names = [*CHORDS, *(f"R0x{c}" for c in CHORDS[1:])]
    titles = [*CHORDS, *(f"R0 × {c}" for c in CHORDS[1:])]
    grid = Grid(float(centres[0] - dt_ms / 2), float(dt_ms), len(centres))
    rows = [
        ImageRow(
            name, title, _quantise(db, quiet),
            y0=float(freq[0] / 1000), dy=float((freq[1] - freq[0]) / 1000),
            y_units="kHz", z_lo=Z_DB[0], z_hi=Z_DB[1], z_units="dB", band=BAND_KHZ,
        )
        for name, title, db in zip(names, titles, [*power, *cross])
    ]
    return grid, rows


def _quantise(db: np.ndarray, quiet: np.ndarray) -> np.ndarray:
    """dB above each bin's quiet-time median, onto 0-255 over `Z_DB`."""
    flat = db - np.median(db[:, quiet], axis=1, keepdims=True)
    scaled = np.rint((flat - Z_DB[0]) * 255 / (Z_DB[1] - Z_DB[0]))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def build(event: str, shot: int, paths: Paths) -> tuple[Grid, list[ImageRow], dict]:
    co2 = raw.raw_signal(int(shot), "co2", paths=paths)
    tier = co2.attrs["tier"]
    if tier == "corpus":
        path = paths.corpus_file(shot)
    else:
        path = raw.cache_path(shot, paths=paths)
    stat = path.stat()
    grid, rows = spectrogram_rows(co2.x, co2.y)
    source = {"tier": tier, "path": str(path), "size": stat.st_size,
              "mtime_ns": stat.st_mtime_ns}
    return grid, rows, {"params": PARAMS, "source": source}
```

- [ ] **Step 4: Run it:** `4 passed`. Ruff on both files.

- [ ] **Step 5: Commit** `labeler review: the Heidbrink AE rows from CO2`.

### Task A5: Any event's panels as rows

**Files:**
- Create: `src/labeler/events/review/panel_rows.py`
- Test: `tests/labeler/test_review_panel_rows.py`

**Interfaces:**
- Consumes: `labeler.events.panels.build(event, shot, paths=paths) -> list[Panel]` (`verify.Panel`:
  `title, x, y, kind` of `"line"` or `"heatmap"`, `z, ylabel, legend, bands, hlines, zmin, zmax`);
  `verify.NoDataError`; A3's `Grid`, `ImageRow`, `TraceRow`.
- Produces: `panel_rows.build(event, shot, paths) -> (Grid, rows, {"params"})`; raises
  `NoDataError` when no panel has data.

- [ ] **Step 1: The failing test.**

```bash
python3 .claude/superpowers/plans/extract_blocks.py \
  .claude/superpowers/plans/2026-09-22-label-review-v4-only-pixi.md \
  tests/labeler/test_review_panel_rows.py
```

Write `tests/labeler/test_review_panel_rows.py`:

```python
"""Panels onto one grid: binning fine data, interpolating coarse data, scaling."""

from __future__ import annotations

import numpy as np
import pytest

from labeler.events.review import panel_rows
from labeler.events.review.rows import Grid
from labeler.events.verify import NoDataError, Panel


def _panels(monkeypatch, *built):
    monkeypatch.setattr(
        panel_rows.panels, "build", lambda event, shot, **kw: list(built)
    )


def test_fine_lines_are_min_max_binned_at_the_finest_grid(monkeypatch):
    x = np.array([0, 0.02, 0.04, 0.06, 0.08, 0.1])
    _panels(monkeypatch, Panel("ne", x=x, y=np.array([[1.0, 5, 2, 4, 9, 3]])))
    grid, rows, info = panel_rows.build("detachment", 1, None)
    assert grid == Grid(-0.025, 0.05, 3)
    assert rows[0].values[0].tolist() == [[1, 2, 3]]
    assert rows[0].values[1].tolist() == [[5, 4, 9]]
    assert rows[0].legend == ["ch 0"]
    assert info == {"params": {"finest_dt_ms": 0.05, "percentiles": [1.0, 99.5]}}


def test_coarse_lines_are_interpolated_onto_the_grid():
    panel = Panel("ip", x=np.array([0.0, 0.1]), y=np.array([[0.0, 10.0]]))
    row = panel_rows._trace("p0", panel, np.array([0.0, 0.1]), Grid(-0.025, 0.05, 3))
    assert row.values.tolist() == [[[0, 5, 10]], [[0, 5, 10]]]


def test_a_heatmap_is_scaled_and_placed_and_empty_columns_are_zero(monkeypatch):
    heatmap = Panel("S", x=np.array([0.0, 1, 2]), y=np.array([100.0]), kind="heatmap",
                    z=np.array([[0.0, 0.5, 1.0]]), zmin=0, zmax=1, bands=[(80, 250)])
    line = Panel("L", x=np.array([0.0, 3.0]), y=np.array([[0.0, 1.0]]))
    _panels(monkeypatch, heatmap, line)
    grid, rows, _ = panel_rows.build("fishbone", 1, None)
    assert grid == Grid(-0.5, 1.0, 4)
    assert rows[0].values.tolist() == [[0, 128, 255, 0]]
    assert (rows[0].z_lo, rows[0].z_hi, rows[0].band) == (0.0, 1.0, (80, 250))
    assert [row.name for row in rows] == ["p0", "p1"]


def test_heatmap_edges_become_centres():
    panel = Panel("S", x=np.array([0.0, 1, 2]), y=np.array([0.0, 0.5, 1.0]),
                  kind="heatmap", z=np.zeros((2, 2)))
    row = panel_rows._image("p0", panel, np.array([0.5, 1.5]), Grid(0.0, 1.0, 2))
    assert (row.y0, row.dy) == (0.25, 0.5)


def test_the_colour_limits_are_the_1st_and_99_5th_percentiles():
    assert panel_rows._limits(np.arange(1000.0)) == pytest.approx((9.99, 994.005))
    assert panel_rows._limits(np.full(4, 2.0)) == (2.0, 3.0)


def test_a_shot_with_no_panels_has_no_data(monkeypatch):
    _panels(monkeypatch)
    with pytest.raises(NoDataError, match="no panels for shot 7"):
        panel_rows.build("detachment", 7, None)
```

- [ ] **Step 2: Run it.** Expect a collection error:
  `ImportError: cannot import name 'panel_rows' from 'labeler.events.review'`.

- [ ] **Step 3: The code.**

```bash
python3 .claude/superpowers/plans/extract_blocks.py \
  .claude/superpowers/plans/2026-09-22-label-review-v4-only-pixi.md \
  src/labeler/events/review/panel_rows.py
```

Write `src/labeler/events/review/panel_rows.py`:

```python
"""Any event's panels as review rows.

Heatmaps become image rows, scaled over their 1st-99.5th percentile unless
the panel pins `zmin`/`zmax`; lines become trace rows. Every row goes onto
one grid at the finest panel spacing, but no finer than 0.05 ms: finer data
is min/max-binned, coarser data is interpolated (lines) or nearest-sampled
(heatmaps).
"""

from __future__ import annotations

import math

import numpy as np

from ...config import Paths
from .. import panels
from ..verify import NoDataError
from .rows import Grid, ImageRow, TraceRow

FINEST_DT_MS = 0.05
PERCENTILES = (1.0, 99.5)


def build(event: str, shot: int, paths: Paths) -> tuple[Grid, list, dict]:
    built = [p for p in panels.build(event, int(shot), paths=paths) if len(p.x)]
    if not built:
        raise NoDataError(f"no panels for shot {int(shot)}")
    xs = [
        _centred(p.x, p.z.shape[1]) if p.kind == "heatmap" else np.asarray(p.x, float)
        for p in built
    ]
    grid = _grid(xs)
    rows = [
        (_image if p.kind == "heatmap" else _trace)(f"p{i}", p, x, grid)
        for i, (p, x) in enumerate(zip(built, xs))
    ]
    params = {"finest_dt_ms": FINEST_DT_MS, "percentiles": list(PERCENTILES)}
    return grid, rows, {"params": params}


def _centred(x, n: int) -> np.ndarray:
    """Cell centres from `n + 1` edges; centres pass through."""
    x = np.asarray(x, dtype=float)
    return (x[:-1] + x[1:]) / 2 if len(x) == n + 1 else x


def _spacing(x) -> float:
    return float(np.median(np.diff(x))) if len(x) > 1 else math.inf


def _grid(xs) -> Grid:
    lo = min(float(x[0]) for x in xs)
    hi = max(float(x[-1]) for x in xs)
    finest = min(_spacing(x) for x in xs)
    dt = max(FINEST_DT_MS, finest) if math.isfinite(finest) else 1.0
    n = max(1, math.ceil((hi - lo) / dt) + 1)
    return Grid(lo - dt / 2, dt, n)


def _columns(x, grid: Grid) -> np.ndarray:
    cols = np.floor((x - grid.t0_ms) / grid.dt_ms).astype(np.int64)
    return np.clip(cols, 0, grid.n - 1)


def _bin(values, cols, n: int, reduce, fill) -> np.ndarray:
    """Reduce the samples falling in each column; empty columns get `fill`."""
    out = np.full((*values.shape[:-1], n), fill, dtype=values.dtype)
    starts = np.flatnonzero(np.r_[True, np.diff(cols) > 0])
    out[..., cols[starts]] = reduce.reduceat(values, starts, axis=-1)
    return out


def _fine(x, grid: Grid) -> bool:
    return len(x) < 2 or _spacing(x) <= grid.dt_ms


def _centres(grid: Grid) -> np.ndarray:
    return grid.t0_ms + (np.arange(grid.n) + 0.5) * grid.dt_ms


def _trace(name: str, panel, x, grid: Grid) -> TraceRow:
    y = np.atleast_2d(np.asarray(panel.y, dtype=np.float32))
    if _fine(x, grid):
        cols = _columns(x, grid)
        low = _bin(y, cols, grid.n, np.fmin, np.nan)
        high = _bin(y, cols, grid.n, np.fmax, np.nan)
        values = np.stack([low, high])
    else:
        centres = _centres(grid)
        line = np.stack([np.interp(centres, x, channel) for channel in y])
        outside = (centres < x[0] - grid.dt_ms / 2) | (centres > x[-1] + grid.dt_ms / 2)
        line[:, outside] = np.nan
        values = np.stack([line, line]).astype(np.float32)
    legend = list(panel.legend) if panel.legend else [f"ch {i}" for i in range(len(y))]
    return TraceRow(name, panel.title, values, y_units=panel.ylabel, legend=legend,
                    hlines=[float(v) for v in panel.hlines])


def _limits(z, zmin=None, zmax=None) -> tuple[float, float]:
    finite = z[np.isfinite(z)]
    lo, hi = np.percentile(finite, PERCENTILES) if finite.size else (0.0, 1.0)
    lo = float(lo if zmin is None else zmin)
    hi = float(hi if zmax is None else zmax)
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def _image(name: str, panel, x, grid: Grid) -> ImageRow:
    z = np.asarray(panel.z, dtype=float)
    lo, hi = _limits(z, panel.zmin, panel.zmax)
    scaled = np.rint((np.nan_to_num(z, nan=lo) - lo) * 255 / (hi - lo))
    q = np.clip(scaled, 0, 255).astype(np.uint8)
    if _fine(x, grid):
        values = _bin(q, _columns(x, grid), grid.n, np.maximum, 0)
    else:
        centres = _centres(grid)
        nearest = np.clip(np.searchsorted(x, centres), 1, len(x) - 1)
        nearest -= (centres - x[nearest - 1]) < (x[nearest] - centres)
        values = q[:, nearest]
        values[:, np.abs(x[nearest] - centres) > _spacing(x) / 2] = 0
    y = _centred(panel.y, z.shape[0])
    dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
    band = tuple(panel.bands[0]) if panel.bands else None
    return ImageRow(name, panel.title, values, y0=float(y[0]), dy=dy,
                    y_units=panel.ylabel, z_lo=lo, z_hi=hi, z_units="", band=band)
```

- [ ] **Step 4: Run it:** `6 passed`. Ruff on both files.

- [ ] **Step 5: Commit** `labeler review: any event's panels as rows`.

### Task A6: Building the store

**Files:**
- Create: `src/labeler/events/review/build.py`, `scripts/labeler/spectrograms.sbatch`
- Test: `tests/labeler/test_review_build.py`

**Interfaces:**
- Consumes: A3-A5; `rosters.roster_path(event, root=...)`, `rosters.read_roster(path)`,
  `labeler.config.git_sha()`.
- Produces: `build.BUILDERS = {"alfven_eigenmode": alfven.build}` (every other event uses
  `panel_rows.build`); `build.build(event, shot, paths=None) -> Path`, which writes
  `paths.spectrogram_file(event, shot)` with the attrs `event, shot, builder, params, source,
  made_at, git_sha`; `python -m labeler.events.review.build --event E [--shots ...]
  [--workers N] [--force]`, exit 1 when any shot failed.

- [ ] **Step 1: The failing test.**

```bash
python3 .claude/superpowers/plans/extract_blocks.py \
  .claude/superpowers/plans/2026-09-22-label-review-v4-only-pixi.md \
  tests/labeler/test_review_build.py
```

Write `tests/labeler/test_review_build.py`:

```python
"""Building the review store: provenance, the command line, failures."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np

from labeler.config import Paths
from labeler.events.review import build as review_build
from labeler.events.review.rows import Grid, TraceRow
from labeler.events.verify import NoDataError


def fake(event, shot, paths):
    if shot == 2:
        raise NoDataError("no co2")
    values = np.zeros((2, 1, 10), dtype=np.float32)
    return Grid(0.0, 1.0, 10), [TraceRow("p0", "Trace", values)], {"params": {"k": 1}}


def test_build_writes_the_rows_and_their_provenance(tmp_path, monkeypatch):
    monkeypatch.setitem(review_build.BUILDERS, "detachment", fake)
    path = review_build.build("detachment", 1, Paths(root=tmp_path))
    assert path == tmp_path / "spectrograms" / "detachment" / "1.h5"
    with h5py.File(path, "r") as f:
        assert f.attrs["event"] == "detachment" and f.attrs["shot"] == 1
        assert f.attrs["builder"] == "test_review_build"
        assert json.loads(f.attrs["params"]) == {"k": 1}
        assert f.attrs["git_sha"] and f.attrs["made_at"]


def test_the_command_builds_what_is_missing_and_reports_failures(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setitem(review_build.BUILDERS, "detachment", fake)
    monkeypatch.setattr(review_build, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setenv("LABELER_ROOT", str(tmp_path))
    assert review_build.main(["--event", "detachment", "--shots", "1", "2"]) == 1
    out = capsys.readouterr().out
    assert "detachment: 2 of 2 shots to build" in out
    assert "NoDataError: no co2" in out
    assert review_build.main(["--event", "detachment", "--shots", "1"]) == 0
    assert "detachment: 0 of 1 shots to build" in capsys.readouterr().out
```

- [ ] **Step 2: Run it.** Expect a collection error:
  `ImportError: cannot import name 'build' from 'labeler.events.review'`.

- [ ] **Step 3: The code**, and the job that builds the 180 AE shots. The job's sizing was
  measured on the login node (its header says how); do NOT submit it. The session runs its
  pilot after the merge.

```bash
python3 .claude/superpowers/plans/extract_blocks.py \
  .claude/superpowers/plans/2026-09-22-label-review-v4-only-pixi.md \
  src/labeler/events/review/build.py scripts/labeler/spectrograms.sbatch
```

Write `src/labeler/events/review/build.py`:

```python
"""Build the review store: one HDF5 file of rows per shot.

    python -m labeler.events.review.build --event alfven_eigenmode [--shots ...]
        [--workers 8] [--force]

AE rows come from the Heidbrink recipe (`alfven.py`); every other event's
from its panel builder (`panel_rows.py`). The review server calls `build`
itself for a shot that has no file yet.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from ...config import Paths, git_sha
from .. import rosters
from . import alfven, panel_rows, rows

BUILDERS = {"alfven_eigenmode": alfven.build}


def build(event: str, shot: int, paths: Paths | None = None) -> Path:
    """Build one shot's rows file and return its path."""
    paths = Paths.from_env() if paths is None else paths
    builder = BUILDERS.get(event, panel_rows.build)
    grid, built, info = builder(event, int(shot), paths)
    path = paths.spectrogram_file(event, shot)
    rows.write(
        path, grid, built, event=event, shot=int(shot),
        builder=builder.__module__.rsplit(".", 1)[-1], **info,
        made_at=datetime.now(UTC).isoformat(timespec="seconds"), git_sha=git_sha(),
    )
    return path


def _timed(event: str, shot: int) -> tuple[int, float, str | None]:
    started = time.monotonic()
    try:
        build(event, shot)
    except Exception as error:  # noqa: BLE001 - one bad shot must not stop the rest
        return shot, time.monotonic() - started, f"{type(error).__name__}: {error}"
    return shot, time.monotonic() - started, None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m labeler.events.review.build",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("--event", required=True)
    parser.add_argument("--shots", nargs="*", type=int, help="default: the roster")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="rebuild existing files")
    args = parser.parse_args(argv)
    paths = Paths.from_env()
    roster = rosters.roster_path(args.event, root=paths.label_tables)
    shots = args.shots or list(rosters.read_roster(roster).shot)
    todo = [
        int(shot) for shot in shots
        if args.force or not paths.spectrogram_file(args.event, shot).is_file()
    ]
    print(f"{args.event}: {len(todo)} of {len(shots)} shots to build", flush=True)
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_timed, args.event, shot) for shot in todo]
        for future in as_completed(futures):
            shot, seconds, error = future.result()
            failed += error is not None
            print(f"{shot} {seconds:.1f}s {error or 'ok'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Write `scripts/labeler/spectrograms.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=labeler-spectrograms
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out

# The review page's AE rows for every alfven_eigenmode roster shot, one worker per cpu:
# `python -m labeler.events.review.build --event alfven_eigenmode` writes
# $LABELER_ROOT/spectrograms/alfven_eigenmode/<shot>.h5 (38-52 MB each, about 8 GB for 180).
# Every roster shot's CO2 is already in $LABELER_ROOT/raw, so nothing is fetched and no fdp
# is needed. A shot that fails prints its error on its own line and the rest carry on.
# Shots already built are skipped; FORCE=1 rebuilds them.
#
#   pilot:      SHOTS="170659 170660 170661 170662 170663 170664 170665 170666" sbatch <this>
#   the rest:   sbatch <this>
#   gate each:  python -m labeler.jobstats --job-id <id> --cpu-only
#
# ------------------------------------------------ sizing (measured 2026-09-22, login node)
#   1 worker, 4 shots:  7.0-7.4 s/shot, 99 % of one core, max RSS 1.44 GB
#   4 workers, 8 shots: 16.7 s wall, 369 % of one core (92 % of 4), process-tree RSS peak
#                       5.63 GB, i.e. 1.41 GB per worker; the parent holds ~0.1 GB
#   The work is CPU-bound: a 312 MB raw file reads in well under a second.
#   cpus-per-task=8   8 workers; the idle start and the last round cost a few percent.
#   mem=15G           8 x 1.41 + 0.1 = 11.4 GB if every worker peaks at once; x 1.3 -> 15G.
#   time=00:15:00     180 x 7.5 s / 8 = 169 s; x 2 for cold reads and contention, plus start.
#   The pilot runs at this same shape, so its MaxRSS is the peak the full run will meet.
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
ENV=/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker
ROOT="$("$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LABELER_ROOT="$ROOT" PYTHONPATH="$REPO/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -Is), root $ROOT"
# SHOTS is left unquoted on purpose: it is a list of shot numbers.
# shellcheck disable=SC2086
srun "$ENV/bin/python" -m labeler.events.review.build --event alfven_eigenmode \
    --workers "${SLURM_CPUS_PER_TASK:-8}" ${SHOTS:+--shots $SHOTS} ${FORCE:+--force}
echo "finished at $(date -Is)"
```

- [ ] **Step 4: Run it:** `2 passed`. Ruff on both Python files;
  `bash -n scripts/labeler/spectrograms.sbatch` prints nothing.

- [ ] **Step 5: Commit** `labeler review: the store build and its SLURM job`.

### Task A7: The review server

**Files:**
- Modify (rewrite): `src/labeler/events/ui/app.py`
- Modify: `src/labeler/events/panels/{__init__,_generic,fishbone,minimum_safety_factor,sawtooth_oscillation}.py`
- Delete: `src/labeler/events/panels/alfven_eigenmode.py`
- Test: `tests/labeler/test_events_ui.py` (rewritten), `tests/labeler/test_events_panels.py`

**Interfaces:**
- Consumes: A1-A6; `rosters.ROSTER_NAME`, `rosters.read_roster`; `raw.progress_for(shot)`.
- Produces: `create_app(paths=None, token=None) -> FastAPI`, with `app.state.paths`, `.token` and
  `.builds`; module constants `STATIC`, `COOKIE`, `FRAMED_COOKIE`. The routes:

| route | returns |
|---|---|
| `GET /api/events` | `{"events": [{event, n_shots, n_reviewed, categories}]}`; a bad roster is `{event, error}` |
| `GET /api/queue?event` | `{shots: [{shot, tier, state, saved_at}], resume}` |
| `GET /api/shot?event&shot` | tier, grid, `t_range`, rows, source, saved, state, last save; `202 {building, progress}` while building; `502 {error}` when the build failed |
| `GET /api/rows?event&shot&t0&t1&cols` | binary rows in store order; header `X-Grid: {"t0","t1","n"}` |
| `POST /api/label` | body `{event, shot, window, intervals}` -> `{row, saved, last_save}` |

  `panels.guidance` and every `GUIDANCE` string are gone; `panels.BUILDERS` has no
  `alfven_eigenmode` (the review store builds AE shots itself).

- [ ] **Step 1: The failing tests.** `test_events_ui.py` is replaced whole;
  `test_events_panels.py` drops the guidance tests and the AE panel's.

```bash
python3 .claude/superpowers/plans/extract_blocks.py \
  .claude/superpowers/plans/2026-09-22-label-review-v4-only-pixi.md \
  tests/labeler/test_events_ui.py tests/labeler/test_events_panels.py
```

Write `tests/labeler/test_events_ui.py`:

```python
"""The label review server: its gate, what it serves, and the one write it makes."""

from __future__ import annotations

import json
import secrets

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from labeler.config import Paths
from labeler.events.interval_tables import validate_intervals
from labeler.events.review import build as review_build
from labeler.events.review import labels
from labeler.events.review import rows as review_rows
from labeler.events.review.rows import Grid, ImageRow, TraceRow
from labeler.events.ui.app import COOKIE, create_app
from labeler.events.verify import NoDataError

ROSTER = (
    "shot,tier,holdout,reviewers,verified_on,notes\n"
    "170815,gold,false,alice,2026-01-01,\n"
    "178642,unverified,true,,,\n"
)
OTHER_ROSTER = (
    "shot,tier,holdout,reviewers,verified_on,notes\n"
    "170815,gold,false,alice,2026-01-01,\n"
)

SOURCE = (
    "shot,category,t_start,t_end,confidence\n"
    "170815,0,0,100,\n"
    "170815,1,100,300,\n"
    "170815,0,300,2000,\n"
    "178642,0,0,2000,\n"
)

NO_TOKEN = "no token: reopen the link printed by the verify server"
BAD_TOKEN = "bad token"


def _tmp_paths(tmp_path, tables):
    """A `Paths` rooted entirely under `tmp_path` - never the real corpus."""
    return Paths(
        root=tmp_path / "root",
        corpus=tmp_path / "corpus",
        text_root=tmp_path / "text",
        logs_jsonl=tmp_path / "logs.jsonl",
        label_tables=tables,
        raw_cache=tmp_path / "raw",
    )


@pytest.fixture
def tables(tmp_path):
    """A label_tables root holding two events with rosters."""
    root = tmp_path / "events"
    event = root / "alfven_eigenmode"
    (event / "format").mkdir(parents=True)
    (event / "shots.csv").write_text(ROSTER)
    other = root / "detachment"
    other.mkdir(parents=True)
    (other / "shots.csv").write_text(OTHER_ROSTER)
    # A directory with no roster is not an event the page can offer.
    (root / "scratch").mkdir()
    return root


@pytest.fixture
def paths(tables, tmp_path):
    return _tmp_paths(tmp_path, tables)


@pytest.fixture
def app(paths):
    return create_app(paths=paths, token="secret")


@pytest.fixture
def client(app):
    transport = TestClient(app)
    transport.cookies.set(COOKIE, "secret")
    return transport


@pytest.fixture
def source(tables):
    """The event's format table: 170815 has one AE span, 178642 none."""
    path = tables / "alfven_eigenmode/format/alfven_eigenmode_format_2026_v1.csv"
    path.write_text(SOURCE)
    return path


def _built(paths, event="alfven_eigenmode", shot=170815):
    """A store file of two rows over 0-2000 ms at 1 ms: a 4-bin image and a trace."""
    image = ImageRow(
        "R0", "R0", (np.arange(8000) % 256).reshape(4, 2000).astype("uint8"),
        y0=0.0, dy=1.0, y_units="kHz", z_lo=-3.0, z_hi=27.0, z_units="dB",
        band=(80.0, 250.0),
    )
    trace = TraceRow(
        "p1", "Density", np.stack([np.zeros((1, 2000)), np.ones((1, 2000))])
    )
    path = paths.spectrogram_file(event, shot)
    review_rows.write(path, Grid(0.0, 1.0, 2000), [image, trace], event=event)
    return path


def _label(**change):
    """What the page posts for 170815 when the reviewer confirms its source label."""
    label = {
        "event": "alfven_eigenmode",
        "shot": 170815,
        "window": [0, 2000],
        "intervals": [[100, 300, 1]],
    }
    return {**label, **change}


def test_no_cookie_and_no_token_is_refused(app):
    response = TestClient(app).get("/api/events")
    assert response.status_code == 401
    assert response.json() == {"error": NO_TOKEN}


def test_a_wrong_token_is_refused(app):
    response = TestClient(app).get("/api/events?token=wrong")
    assert response.status_code == 401
    assert response.json() == {"error": BAD_TOKEN}
    assert COOKIE not in response.cookies


def test_an_empty_token_is_refused(app):
    response = TestClient(app).get("/api/events?token=")
    assert response.status_code == 401
    assert response.json() == {"error": BAD_TOKEN}


def test_a_wrong_cookie_is_refused(app):
    transport = TestClient(app)
    transport.cookies.set(COOKIE, "wrong")
    response = transport.get("/api/events")
    assert response.status_code == 401
    assert response.json() == {"error": NO_TOKEN}


def test_a_token_in_the_wrong_place_is_refused(app):
    """A header or a form field is not the gate; only query or cookie is."""
    transport = TestClient(app)
    header = transport.get("/api/events", headers={"Authorization": "secret"})
    assert header.status_code == 401
    assert header.json() == {"error": NO_TOKEN}
    posted = transport.post("/api/events", data={"token": "secret"})
    assert posted.status_code == 401
    assert posted.json() == {"error": NO_TOKEN}


def test_the_token_is_compared_in_constant_time(app, monkeypatch):
    seen = []
    real = secrets.compare_digest

    def spy(a, b):
        seen.append((a, b))
        return real(a, b)

    monkeypatch.setattr(secrets, "compare_digest", spy)
    TestClient(app).get("/api/events?token=wrong")
    assert seen == [(b"wrong", b"secret")]


def test_the_right_token_sets_the_cookie(app):
    response = TestClient(app, follow_redirects=False).get("/api/events?token=secret")
    assert response.status_code == 303
    assert response.headers["location"] == "/api/events"
    assert response.cookies[COOKIE] == "secret"
    cookie_header = response.headers["set-cookie"]
    assert "HttpOnly" in cookie_header
    assert "samesite=lax" in cookie_header.lower()


def test_the_right_token_also_sets_a_cookie_a_framed_page_can_keep(app):
    """VS Code's Simple Browser frames the page; a Lax cookie is dropped
    there. The second cookie is SameSite=None and Secure, which 127.0.0.1
    over http is allowed to keep, and the gate accepts it alone."""
    from labeler.events.ui.app import FRAMED_COOKIE

    response = TestClient(app).get("/?token=secret", follow_redirects=False)
    assert response.status_code == 303
    framed = [
        h for h in response.headers.get_list("set-cookie") if FRAMED_COOKIE in h
    ]
    assert len(framed) == 1
    assert "samesite=none" in framed[0].lower()
    assert "secure" in framed[0].lower()
    assert "HttpOnly" in framed[0]

    transport = TestClient(app)
    transport.cookies.set(FRAMED_COOKIE, "secret")
    assert transport.get("/api/events").status_code == 200
    wrong = TestClient(app)
    wrong.cookies.set(FRAMED_COOKIE, "wrong")
    assert wrong.get("/api/events").status_code == 401


def test_an_authorized_response_is_not_cached(client):
    response = client.get("/api/events")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"


def test_events_lists_each_roster_and_how_much_of_it_is_reviewed(client, source):
    # "scratch" has no roster, so it is not an event.
    present = {"1": "present"}
    assert client.get("/api/events").json() == {
        "events": [
            {"event": "alfven_eigenmode", "n_shots": 2, "n_reviewed": 0,
             "categories": present},
            {"event": "detachment", "n_shots": 1, "n_reviewed": 0,
             "categories": present},
        ]
    }


def test_the_queue_is_the_roster_in_order_with_where_to_resume(client, source):
    assert client.get("/api/queue?event=alfven_eigenmode").json() == {
        "shots": [
            {"shot": 170815, "tier": "gold", "state": "unreviewed",
             "saved_at": None},
            {"shot": 178642, "tier": "unverified", "state": "unreviewed",
             "saved_at": None},
        ],
        "resume": 170815,
    }


def test_a_built_shot_opens_with_its_grid_its_rows_and_its_labels(
    client, paths, source
):
    _built(paths)
    body = client.get("/api/shot?event=alfven_eigenmode&shot=170815").json()
    assert body["tier"] == "gold"
    assert body["grid"] == {"t0": 0.0, "dt": 1.0, "n": 2000}
    assert body["t_range"] == [0.0, 2000.0]
    assert [row["name"] for row in body["rows"]] == ["R0", "p1"]
    assert body["rows"][0]["band"] == [80.0, 250.0]
    assert body["source"] == {"window": [0, 2000], "intervals": [[100, 300, 1]]}
    assert (body["saved"], body["last_save"], body["state"]) == (
        None, None, "unreviewed"
    )


def _one_trace(event, shot, paths):
    """A builder standing in for the panel builder: one 10 ms trace."""
    trace = TraceRow("p0", "Trace", np.zeros((2, 1, 10), dtype="float32"))
    return Grid(0.0, 1.0, 10), [trace], {"params": {}}


def test_a_shot_with_no_rows_is_built_on_first_open(client, app, monkeypatch):
    monkeypatch.setitem(review_build.BUILDERS, "detachment", _one_trace)
    first = client.get("/api/shot?event=detachment&shot=170815")
    assert first.status_code == 202
    assert first.json() == {"building": True, "progress": None}
    app.state.builds.running[("detachment", 170815)].result()
    body = client.get("/api/shot?event=detachment&shot=170815").json()
    assert body["grid"] == {"t0": 0.0, "dt": 1.0, "n": 10}
    assert body["source"] is None and body["state"] == "unreviewed"


def test_a_shot_that_cannot_be_built_says_why_and_is_retried(
    client, app, monkeypatch
):
    def broken(event, shot, paths):
        raise NoDataError(f"no features for {shot}")

    monkeypatch.setitem(review_build.BUILDERS, "detachment", broken)
    url = "/api/shot?event=detachment&shot=170815"
    for _ in range(2):  # the second open retries rather than repeating the error
        assert client.get(url).status_code == 202
        with pytest.raises(NoDataError):
            app.state.builds.running[("detachment", 170815)].result()
        response = client.get(url)
        assert response.status_code == 502
        assert response.json() == {"error": "NoDataError: no features for 170815"}


def test_a_shot_off_the_roster_is_not_found_and_not_built(client, app):
    response = client.get("/api/shot?event=alfven_eigenmode&shot=1")
    assert response.status_code == 404
    assert response.json() == {"error": "shot 1 is not on this event's roster"}
    assert app.state.builds.running == {}


def test_rows_come_back_as_bytes_with_the_grid_they_cover(client, paths):
    _built(paths)
    response = client.get(
        "/api/rows?event=alfven_eigenmode&shot=170815&t0=0&t1=2000&cols=500"
    )
    assert response.headers["content-type"] == "application/octet-stream"
    grid = json.loads(response.headers["x-grid"])
    assert grid == {"t0": 0.0, "t1": 2000.0, "n": 500}
    # The rows in store order, every 4 columns pooled into one: the image as
    # uint8 (4, 500), then the trace as float32 (min/max, 1 channel, 500).
    image = np.frombuffer(response.content[:2000], dtype=np.uint8).reshape(4, 500)
    trace = np.frombuffer(response.content[2000:], dtype="<f4").reshape(2, 1, 500)
    assert image[0, :3].tolist() == [3, 7, 11]
    assert (trace[0] == 0).all() and (trace[1] == 1).all()


@pytest.mark.parametrize(
    ("query", "status", "reason"),
    [
        ("shot=178642&t0=0&t1=2000", 404, "has no rows yet"),
        ("shot=170815&t0=5000&t1=6000", 400, "outside the record"),
        ("shot=170815&t0=0&t1=2000&cols=8", 422, "cols:"),
    ],
)
def test_a_bad_rows_request_is_refused_with_its_reason(
    client, paths, query, status, reason
):
    _built(paths)
    response = client.get(f"/api/rows?event=alfven_eigenmode&{query}")
    assert response.status_code == status
    assert reason in response.json()["error"]


def test_confirming_the_source_label_saves_it_as_a_format_table(
    client, tables, source
):
    body = client.post("/api/label", json=_label()).json()
    assert body["row"]["state"] == "confirmed" and body["row"]["tier"] == "gold"
    assert body["saved"] == {"window": [0, 2000], "intervals": [[100, 300, 1]]}
    assert body["last_save"]["saved_at"] == body["row"]["saved_at"]
    table = pd.read_csv(tables / "alfven_eigenmode/review/labels.csv")
    written = validate_intervals(table)[["shot", "category", "t_start", "t_end"]]
    assert written.values.tolist() == [
        [170815, 0, 0, 100],
        [170815, 1, 100, 300],
        [170815, 0, 300, 2000],
    ]
    queue = client.get("/api/queue?event=alfven_eigenmode").json()
    assert queue["shots"][0]["state"] == "confirmed"
    assert queue["resume"] == 178642
    assert client.get("/api/events").json()["events"][0]["n_reviewed"] == 1


def test_a_changed_label_is_saved_merged_and_shown_beside_its_source(
    client, paths, tables, source
):
    _built(paths)
    client.post("/api/label", json=_label())
    client.post("/api/label", json=_label(shot=178642, intervals=[]))
    changed = [[150, 250.4, 1], [240, 400, 1]]
    body = client.post("/api/label", json=_label(intervals=changed)).json()
    assert body["saved"] == {"window": [0, 2000], "intervals": [[150, 400, 1]]}
    assert body["row"]["state"] == "changed"
    view = client.get("/api/shot?event=alfven_eigenmode&shot=170815").json()
    assert view["source"]["intervals"] == [[100, 300, 1]]
    assert view["saved"]["intervals"] == [[150, 400, 1]]
    assert view["state"] == "changed"
    history = labels.read_history(tables / "alfven_eigenmode")
    assert [entry["shot"] for entry in history] == [170815, 178642, 170815]


@pytest.mark.parametrize(
    ("change", "status"),
    [
        ({"shot": 1}, 404),
        ({"window": [0, 30000]}, 400),
        ({"window": [2000, 0], "intervals": []}, 400),
        ({"intervals": [[300, 100, 1]]}, 400),
        ({"intervals": [[100, 300, 7]]}, 400),
        ({"reviewer": "mallory"}, 422),
        ({"event": "../secret_area"}, 404),
    ],
)
def test_a_bad_label_is_refused_and_nothing_is_written(
    client, tables, foreign, source, change, status
):
    response = client.post("/api/label", json=_label(**change))
    assert response.status_code == status
    assert set(response.json()) == {"error"}
    assert not (tables / "alfven_eigenmode/review").exists()
    assert not (foreign / "review").exists()


def test_a_non_finite_time_is_refused(client, tables, source):
    content = json.dumps(_label()).replace("[0, 2000]", "[0, Infinity]")
    response = client.post(
        "/api/label", content=content, headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    assert not (tables / "alfven_eigenmode/review").exists()


def test_a_label_is_behind_the_token_gate(app, tables, source):
    response = TestClient(app).post("/api/label", json=_label())
    assert response.status_code == 401
    assert response.json() == {"error": NO_TOKEN}
    assert not (tables / "alfven_eigenmode/review").exists()


def test_the_roster_is_read_and_never_written(client, tables, source):
    roster = tables / "alfven_eigenmode" / "shots.csv"
    before = (roster.read_bytes(), roster.stat().st_mtime_ns)
    client.get("/api/events")
    client.get("/api/queue?event=alfven_eigenmode")
    assert client.post("/api/label", json=_label()).status_code == 200
    assert (roster.read_bytes(), roster.stat().st_mtime_ns) == before


@pytest.fixture
def foreign(tmp_path):
    """A real, valid roster OUTSIDE label_tables, for the traversal tests."""
    outside = tmp_path / "secret_area"
    outside.mkdir()
    (outside / "shots.csv").write_text(
        "shot,tier,holdout,reviewers,verified_on,notes\n999999,gold,false,,,\n"
    )
    return outside


def test_a_traversing_event_is_not_found(client, foreign):
    response = client.get("/api/queue", params={"event": "../secret_area"})
    assert response.status_code == 404
    assert "999999" not in response.text, "the foreign roster was read"


def test_an_absolute_event_is_not_found(client, foreign):
    # pathlib's `/` DISCARDS the root when the right operand is absolute, so
    # an unchecked event name reaches any directory the server's uid can read.
    response = client.get("/api/queue", params={"event": str(foreign)})
    assert response.status_code == 404
    assert "999999" not in response.text, "the foreign roster was read"


def test_an_unknown_event_is_not_found(client):
    response = client.get("/api/queue", params={"event": "no_such_event"})
    assert response.status_code == 404


def test_an_overlong_event_is_not_found_not_a_500(client):
    """`Path.is_file` re-raises ENAMETOOLONG rather than reporting False -

    it is not in pathlib's `_IGNORED_ERRNOS` - so an unchecked `is_file` call
    turns a client-supplied `event` this long into an unhandled 500 instead
    of the same 404 every other rejected `event` gets.
    """
    response = client.get("/api/queue", params={"event": "a" * 5000})
    assert response.status_code == 404
    assert set(response.json()) == {"error"}


def test_one_bad_roster_does_not_hide_the_others(client, tables):
    bad = tables / "broken_event"
    bad.mkdir()
    (bad / "shots.csv").write_text(
        "shot,tier,holdout,reviewers,verified_on,notes\n170815,gold,True,,,\n"
    )
    payload = client.get("/api/events").json()
    rows = {row["event"]: row for row in payload["events"]}
    assert "alfven_eigenmode" in rows and "detachment" in rows
    assert "holdout" in rows["broken_event"]["error"], "the reason is not reported"


def test_a_bad_roster_is_a_json_error_from_the_queue_too(client, tables):
    """The roster `/api/events` flagged answers here in the page's one shape.

    Starlette's own 500 is bare text, and the page reads every refusal as
    `{"error": ...}`; an event offered precisely so its broken roster can be
    found must not be the one that breaks that contract.
    """
    bad = tables / "broken_event"
    bad.mkdir()
    (bad / "shots.csv").write_text(
        "shot,tier,holdout,reviewers,verified_on,notes\n170815,gold,True,,,\n"
    )
    response = client.get("/api/queue", params={"event": "broken_event"})
    assert response.status_code == 500
    assert set(response.json()) == {"error"}
    assert "holdout" in response.json()["error"], "the reason is not reported"


def test_a_missing_label_tables_root_is_not_a_crash(tmp_path):
    app = create_app(paths=_tmp_paths(tmp_path, tmp_path / "absent"), token="secret")
    transport = TestClient(app)
    transport.cookies.set(COOKIE, "secret")
    response = transport.get("/api/events")
    assert response.status_code == 200
    assert response.json() == {"events": []}


def test_a_duplicated_token_takes_the_last_value(app):
    """Starlette's QueryParams.get returns the LAST value; pin that."""
    transport = TestClient(app, follow_redirects=False)
    assert transport.get("/api/events?token=wrong&token=secret").status_code == 303
    assert transport.get("/api/events?token=secret&token=wrong").status_code == 401


def test_every_request_is_logged_by_path_and_never_by_token(app, caplog):
    """The request log is how a reviewer tells an unreachable server from a
    slow one. It carries the path and the status, and never the query
    string, because the token rides there."""
    import logging

    caplog.set_level(logging.INFO, logger="labeler.events.ui.app")
    with TestClient(app) as client:
        client.get(f"/api/queue?event=alfven_eigenmode&token={app.state.token}")
        client.get("/api/queue?event=alfven_eigenmode")
        TestClient(app).get("/api/events")
    lines = [r.getMessage() for r in caplog.records if " -> " in r.getMessage()]
    assert any(line.startswith("GET /api/queue -> 303") for line in lines)
    assert any(line.startswith("GET /api/queue -> 200") for line in lines)
    assert any(line.startswith("GET /api/events -> 401") for line in lines)
    assert all(app.state.token not in line for line in lines)
    assert all("token=" not in line and "event=" not in line for line in lines)


def test_the_page_and_its_two_files_are_served(client):
    """The reviewer opens `/` and gets the page, the script and the styles."""
    page = client.get("/")
    assert page.status_code == 200
    assert "/app.js" in page.text and "/style.css" in page.text
    for path in ("/app.js", "/style.css"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.content, path


def test_label_data_is_never_cached(client, paths, source):
    """Every answer carries label data or the page that shows it; none is held."""
    _built(paths)
    answers = [
        client.get(path)
        for path in (
            "/",
            "/api/events",
            "/api/queue?event=alfven_eigenmode",
            "/api/shot?event=alfven_eigenmode&shot=170815",
            "/api/rows?event=alfven_eigenmode&shot=170815&t0=0&t1=100",
        )
    ]
    answers.append(client.post("/api/label", json=_label()))
    for response in answers:
        assert response.status_code == 200, response.url
        assert response.headers["cache-control"] == "no-store", response.url


def test_the_page_is_behind_the_token_gate(app):
    """The static mount is inside the gate, not beside it."""
    transport = TestClient(app)
    for path in ("/", "/app.js", "/style.css"):
        response = transport.get(path)
        assert response.status_code == 401, path
        assert response.json() == {"error": NO_TOKEN}, path


# --- the launcher -----------------------------------------------------------


def _off_the_real_tree(monkeypatch, tmp_path):
    """Point `Paths.from_env` at `tmp_path` so `main()` cannot open the corpus."""
    paths = _tmp_paths(tmp_path, tmp_path / "events")
    monkeypatch.setenv("LABELER_ROOT", str(paths.root))
    monkeypatch.setenv("LABELER_CORPUS", str(paths.corpus))
    monkeypatch.setenv("LABELER_TEXT_ROOT", str(paths.text_root))
    monkeypatch.setenv("LABELER_LOGS_JSONL", str(paths.logs_jsonl))
    monkeypatch.setenv("LABELER_LABEL_TABLES", str(paths.label_tables))
    monkeypatch.setenv("LABELER_RAW_CACHE", str(paths.raw_cache))


def _runner(monkeypatch, serve):
    """Replace the only layer that would bind a socket, recording its call."""
    calls = []

    def fake(app, host, port):
        calls.append({"app": app, "host": host, "port": port})
        return 0

    monkeypatch.setattr(serve, "_run", fake)
    return calls


def test_serve_refuses_any_host_but_the_loopback():
    from labeler.events.ui import serve

    with pytest.raises(ValueError, match="127.0.0.1"):
        serve.main(host="0.0.0.0")


def test_serve_prints_the_token_link_and_the_forward(monkeypatch, capsys, tmp_path):
    from labeler.events.ui import serve

    _off_the_real_tree(monkeypatch, tmp_path)
    monkeypatch.setattr(serve, "_run", lambda app, host, port: 0)
    serve.main(token="secret", port=9999)
    printed = capsys.readouterr().out
    assert "http://127.0.0.1:9999/?token=secret" in printed
    assert "ssh -L 9999:localhost:9999" in printed


def test_serve_binds_the_loopback_and_hands_over_the_gated_app(
    monkeypatch, capsys, tmp_path
):
    """The real `main` path: what the runner is given, value by value."""
    from labeler.events.ui import serve

    _off_the_real_tree(monkeypatch, tmp_path)
    calls = _runner(monkeypatch, serve)
    assert serve.main(token="secret") == 0
    capsys.readouterr()
    assert len(calls) == 1
    assert calls[0]["host"] == "127.0.0.1"
    assert calls[0]["port"] == serve.DEFAULT_PORT == 8811
    assert calls[0]["app"].state.token == "secret"


def test_serve_never_puts_the_token_bearing_url_in_uvicorn_s_access_log(monkeypatch):
    """The token rides in the query string, so an access log would print the
    credential into the terminal and into anything capturing it.
    """
    import uvicorn

    from labeler.events.ui import serve

    seen = {}

    def fake_run(app, **kwargs):
        seen["app"] = app
        seen.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    sentinel = object()
    assert serve._run(sentinel, "127.0.0.1", 8811) == 0
    assert seen["app"] is sentinel
    assert seen["access_log"] is False
    assert seen["host"] == "127.0.0.1"
    assert seen["port"] == 8811


def test_serve_mints_an_unguessable_token_that_differs_between_runs(
    monkeypatch, capsys, tmp_path
):
    from labeler.events.ui import serve

    _off_the_real_tree(monkeypatch, tmp_path)
    calls = _runner(monkeypatch, serve)
    serve.main(port=9999)
    first = capsys.readouterr().out
    serve.main(port=9999)
    second = capsys.readouterr().out

    minted = [call["app"].state.token for call in calls]
    assert minted[0] != minted[1]
    # 16 bytes of hex: too wide to guess from the loopback, and every
    # character accounted for so a truncated or non-hex source shows up.
    for token in minted:
        assert len(token) == 32
        assert set(token) <= set("0123456789abcdef")
    for token, printed in zip(minted, (first, second)):
        assert f"http://127.0.0.1:9999/?token={token}" in printed

    # The printed token is the only one that opens the page.
    app = calls[0]["app"]
    refused = TestClient(app).get(f"/api/events?token={minted[1]}")
    assert refused.status_code == 401
    assert refused.json() == {"error": BAD_TOKEN}


def test_serve_takes_its_token_from_secrets(monkeypatch, capsys, tmp_path):
    """A token from `random` would read the same but be predictable."""
    from labeler.events.ui import serve

    _off_the_real_tree(monkeypatch, tmp_path)
    _runner(monkeypatch, serve)
    seen = []
    real = secrets.token_hex

    def spy(n=None):
        seen.append(n)
        return real(n)

    monkeypatch.setattr(secrets, "token_hex", spy)
    serve.main(port=9999)
    capsys.readouterr()
    assert seen == [16]


def test_serve_warns_when_the_process_is_not_under_the_fdp_wrapper(
    monkeypatch, capsys, tmp_path
):
    """A shot outside the corpus needs a live fetch, and a live fetch needs
    the wrapper; saying so at startup costs a restart, saying so at the first
    fetch costs the review up to that point.
    """
    from labeler.events.ui import serve

    _off_the_real_tree(monkeypatch, tmp_path)
    _runner(monkeypatch, serve)
    for marker in serve.FDP_MARKERS:
        monkeypatch.delenv(marker, raising=False)
    serve.main(token="secret")
    assert serve.FDP_COMMAND in capsys.readouterr().out


def test_serve_stays_quiet_when_it_is_under_the_fdp_wrapper(
    monkeypatch, capsys, tmp_path
):
    """The markers are the ones `fdp run` really sets; a note on every single
    run is a note nobody reads.
    """
    from labeler.events.ui import serve

    _off_the_real_tree(monkeypatch, tmp_path)
    _runner(monkeypatch, serve)
    for marker in serve.FDP_MARKERS:
        monkeypatch.setenv(marker, "set-by-fdp-run")
    serve.main(token="secret")
    assert "fdp" not in capsys.readouterr().out
```

Patch `tests/labeler/test_events_panels.py`:

```diff
diff --git a/tests/labeler/test_events_panels.py b/tests/labeler/test_events_panels.py
index 411eb9f..e1937db 100644
--- a/tests/labeler/test_events_panels.py
+++ b/tests/labeler/test_events_panels.py
@@ -7,7 +7,6 @@ import pytest
 
 from labeler.config import Paths
 from labeler.events import panels as registry
-from labeler.events.panels import alfven_eigenmode as ae
 from labeler.events.panels import fishbone as fb
 from labeler.events.panels import minimum_safety_factor as msf
 from labeler.events.panels import sawtooth_oscillation as saw
@@ -98,137 +97,14 @@ def test_an_unregistered_event_falls_back_to_the_generic_builder(monkeypatch):
     assert calls == [42]
 
 
-def test_every_registered_event_has_guidance():
-    for event in registry.BUILDERS:
-        assert registry.guidance(event).strip(), event
-
-
-def test_guidance_for_an_unregistered_event_says_it_is_generic():
-    assert "generic" in registry.guidance("no_such_event").lower()
-
-
 def test_the_registry_covers_the_events_with_bespoke_panels():
     assert set(registry.BUILDERS) == {
-        "alfven_eigenmode",
         "fishbone",
         "minimum_safety_factor",
         "sawtooth_oscillation",
     }
 
 
-def test_crosspower_takes_its_rate_from_the_span_not_a_median_diff():
-    """A float32 time vector quantises its spacing at t ~ 3 s.
-
-    Successive differences of such a vector are wrong by percents and in a
-    biased direction; the span is exact. A 120 kHz tone must land at 120 kHz
-    on a time base that starts at 3000 ms, not merely on one starting at 0.
-    """
-    rate = 1_000_000.0
-    n = 200_000
-    t_ms = (3000.0 + np.arange(n) / rate * 1000.0).astype("float32")
-    tone = np.sin(2 * np.pi * 120_000.0 * np.arange(n) / rate).astype("float32")
-    freq_khz, _, power = ae.crosspower(t_ms.astype("float64"), tone, tone)
-    peak = freq_khz[np.argmax(power.mean(axis=1))]
-    assert abs(peak - 120.0) < 1.0
-
-
-def test_crosspower_caps_its_time_bins():
-    rate = 1_000_000.0
-    n = 400_000
-    t_ms = np.arange(n) / rate * 1000.0
-    noise = np.random.default_rng(0).normal(size=n).astype("float32")
-    _, t_out, power = ae.crosspower(t_ms, noise, noise, max_bins=250)
-    assert power.shape[1] <= 250
-    assert len(t_out) == power.shape[1]
-
-
-def test_crosspower_averages_power_before_taking_the_log():
-    """A geometric mean would be dragged down by the quiet bins in a block.
-
-    That is exactly what suppresses a short burst - the thing the panel
-    exists to show. One loud block among quiet ones must survive averaging.
-    """
-    rate = 1_000_000.0
-    n = 200_000
-    t_ms = np.arange(n) / rate * 1000.0
-    rng = np.random.default_rng(1)
-    signal = rng.normal(scale=1e-3, size=n)
-    signal[100_000:110_000] += 5.0 * np.sin(
-        2 * np.pi * 120_000.0 * np.arange(10_000) / rate
-    )
-    freq_khz, t_out, power = ae.crosspower(t_ms, signal, signal, max_bins=40)
-    band = (freq_khz > 110.0) & (freq_khz < 130.0)
-    profile = power[band].mean(axis=0)
-    burst = np.argmax(profile)
-    assert 95.0 < t_out[burst] < 115.0
-    assert profile[burst] > np.median(profile) + 1.0
-
-
-def test_alfven_panels_are_three_crosspower_heatmaps(monkeypatch):
-    """The whole unit seam, on a shot-like time base rather than one at zero.
-
-    `raw_signal` hands back MILLISECONDS and `t_range` is milliseconds, so a
-    time base starting at 0 hides both halves of the bug this plan has hit
-    twice: a dropped window offset and a second, spurious seconds-to-ms
-    conversion both leave the panels sitting at t ~ 0 on such a base.
-    """
-    rate = 1_000_000.0
-    n = 60_000
-    start_ms = 2000.0
-    fake = FeatureArray(
-        x=start_ms + np.arange(n) / rate * 1000.0,
-        y=np.random.default_rng(2).normal(size=(4, n)).astype("float32"),
-        attrs={"tier": "cache"},
-    )
-    seen = []
-
-    def fake_raw_signal(shot, group, **kwargs):
-        seen.append((shot, group, kwargs))
-        return fake
-
-    monkeypatch.setattr(ae, "raw_signal", fake_raw_signal)
-    window = (start_ms, start_ms + 60.0)
-    built = registry.build("alfven_eigenmode", 178642, t_range=window)
-    assert len(built) == 3
-    assert {p.kind for p in built} == {"heatmap"}
-    assert all(p.bands == [(80.0, 250.0)] for p in built)
-    assert all(p.ylabel == "kHz" for p in built)
-
-    # The pairing IS the diagnostic - the reference chord against each
-    # vertical one - so the titles pin channel identity, not just the count.
-    assert [panel.title for panel in built] == [
-        "CO2 crosspower DENR0UF x DENV1UF",
-        "CO2 crosspower DENR0UF x DENV2UF",
-        "CO2 crosspower DENR0UF x DENV3UF",
-    ]
-
-    # Every panel must be drawn inside the window that was fetched.
-    end_ms = start_ms + (n - 1) / rate * 1000.0
-    for panel in built:
-        assert panel.x.min() >= start_ms
-        assert panel.x.max() <= end_ms
-
-    # The window has to reach the fetch: without it a reviewer wanting a
-    # 60 ms look waits on a whole shot of PTDATA.
-    assert seen == [(178642, "co2", {"t_range": window, "paths": None})]
-
-
-def test_crosspower_rejects_a_degenerate_window():
-    """An out-of-range `t_range` slices the signal to nothing.
-
-    A zero span divides to nan and silently draws a blank panel; an empty one
-    used to die on an IndexError far from the cause.
-    """
-    one = np.array([1.0])
-    two = np.array([1.0, 2.0])
-    with pytest.raises(ValueError, match="more than one instant"):
-        ae.crosspower(np.array([]), np.array([]), np.array([]))
-    with pytest.raises(ValueError, match="more than one instant"):
-        ae.crosspower(np.array([2000.0]), one, one)
-    with pytest.raises(ValueError, match="more than one instant"):
-        ae.crosspower(np.array([2000.0, 2000.0]), two, two)
-
-
 def test_fishbone_draws_b1_power_and_the_b1_x_b5_cross_phase(monkeypatch):
     """The real builder body, on a shot-like time base.
 
@@ -293,8 +169,7 @@ def test_fishbone_rejects_a_degenerate_window(monkeypatch):
     `scipy.signal.spectrogram` then runs with `fs=nan` and returns
     `freq=[nan]`; `freq <= MAX_HZ` is all-False since nan comparisons are
     always False, so the panel would render as a silent 0-row heatmap
-    instead of failing loudly. Mirrors
-    `test_crosspower_rejects_a_degenerate_window`.
+    instead of failing loudly.
     """
 
     def fake_raw_signal(shot, group, *, channels=None, t_range=None, paths=None):
```

- [ ] **Step 2: Run** `tests/labeler/test_events_ui.py tests/labeler/test_events_panels.py`. Expect `25 failed, 33 passed`: the old app
  has none of the new routes, and the registry test still finds `alfven_eigenmode`.

- [ ] **Step 3: The code.**

```bash
python3 .claude/superpowers/plans/extract_blocks.py \
  .claude/superpowers/plans/2026-09-22-label-review-v4-only-pixi.md \
  src/labeler/events/ui/app.py src/labeler/events/panels/__init__.py src/labeler/events/panels/_generic.py src/labeler/events/panels/fishbone.py src/labeler/events/panels/minimum_safety_factor.py src/labeler/events/panels/sawtooth_oscillation.py
```

Write `src/labeler/events/ui/app.py`:

```python
"""The label review server: one label per shot, over rows prebuilt per shot.

Loopback only, behind a token: `?token=` sets two cookies and redirects to
the same URL without it. The server reads `shots.csv` and never writes it;
its one write is `POST /api/label`, into the event's `review/` directory.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from ...config import Paths
from .. import raw, rosters
from ..review import build as review_build
from ..review import labels, rows

STATIC = Path(__file__).parent / "static"
COOKIE = "labeler_verify_token"
#: The same token again for a FRAMED page: VS Code's Simple Browser frames
#: every page, and drops a Lax cookie there. SameSite=None needs Secure, which
#: browsers allow on 127.0.0.1 over plain http.
FRAMED_COOKIE = "labeler_verify_token_framed"
NO_TOKEN = "no token: reopen the link printed by the verify server"
BAD_TOKEN = "bad token"

log = logging.getLogger(__name__)


class LabelIn(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    event: str
    shot: int
    window: tuple[float, float]
    intervals: list[tuple[float, float, int]] = Field(max_length=1000)


class Builds:
    """Store files built on first open, two at a time, off the request thread."""

    def __init__(self, paths: Paths):
        self.paths = paths
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.running: dict = {}
        self.lock = threading.Lock()

    def status(self, event: str, shot: int) -> tuple[Path | None, str | None]:
        """`(path, None)` once built, `(None, error)` once failed, else building."""
        key = (event, shot)
        path = self.paths.spectrogram_file(event, shot)
        with self.lock:
            future = self.running.get(key)
            if future is not None and not future.done():
                return None, None
            self.running.pop(key, None)
            if future is not None and future.exception() is not None:
                error = future.exception()
                return None, f"{type(error).__name__}: {error}"
            if path.is_file():
                return path, None
            self.running[key] = self.pool.submit(
                review_build.build, event, shot, self.paths
            )
            return None, None


def require_event(event: str, paths: Paths) -> Path:
    """The event's directory, else 404; decided on the name before any join."""
    if event != Path(event).name or event in {"", ".", ".."} or "\0" in event:
        raise HTTPException(404, f"unknown event {event!r}")
    directory = paths.label_tables / event
    try:
        found = (directory / rosters.ROSTER_NAME).is_file()
    except OSError:  # ENAMETOOLONG is raised, not reported as False
        found = False
    if not found:
        raise HTTPException(404, f"unknown event {event!r}")
    return directory


def _roster(directory: Path):
    return rosters.read_roster(directory / rosters.ROSTER_NAME)


def roster_tier(roster, shot: int) -> str:
    match = roster[roster.shot == int(shot)]
    if match.empty:
        raise HTTPException(404, f"shot {int(shot)} is not on this event's roster")
    return str(match.tier.iloc[0])


def _admit(request: Request, token: str) -> Response | None:
    """None for a request carrying the token; else its 401, or the 303 that sets it."""
    expected = token.encode()
    given = request.query_params.get("token")
    if given is not None:
        if not secrets.compare_digest(given.encode(), expected):
            return JSONResponse({"error": BAD_TOKEN}, status_code=401)
        url = request.url.remove_query_params("token")
        path = url.path if not url.path.startswith("//") else "/"
        response = RedirectResponse(path + (f"?{url.query}" if url.query else ""), 303)
        response.set_cookie(COOKIE, token, httponly=True, samesite="lax")
        response.set_cookie(
            FRAMED_COOKIE, token, httponly=True, samesite="none", secure=True
        )
        return response
    cookie = request.cookies.get(COOKIE) or request.cookies.get(FRAMED_COOKIE, "")
    if not cookie or not secrets.compare_digest(cookie.encode(), expected):
        return JSONResponse({"error": NO_TOKEN}, status_code=401)
    return None


def create_app(paths: Paths | None = None, token: str | None = None) -> FastAPI:
    paths = Paths.from_env() if paths is None else paths
    app = FastAPI(
        title="labeler review", docs_url=None, redoc_url=None, openapi_url=None
    )
    app.state.paths = paths
    app.state.token = token or secrets.token_hex(16)
    app.state.builds = Builds(paths)
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=1)

    @app.middleware("http")
    async def gate(request: Request, call_next):
        # One log line per request, by PATH: the token rides in the query string.
        started = time.monotonic()
        response = _admit(request, app.state.token)
        if response is None:
            response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        log.info(
            "%s %s -> %d in %.1fs",
            request.method,
            request.url.path,
            response.status_code,
            time.monotonic() - started,
        )
        return response

    @app.exception_handler(StarletteHTTPException)
    async def refused(request: Request, exc: StarletteHTTPException):
        return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def malformed(request: Request, exc: RequestValidationError):
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"][1:]) or "query"
        return JSONResponse({"error": f"{where}: {first['msg']}"}, status_code=422)

    @app.exception_handler(ValueError)
    @app.exception_handler(OSError)
    async def failed(request: Request, exc: Exception):
        log.error("%s failed", request.url.path, exc_info=exc)
        return JSONResponse({"error": str(exc) or type(exc).__name__}, status_code=500)

    @app.get("/api/events")
    def events():
        try:
            directories = sorted(
                p for p in paths.label_tables.iterdir()
                if (p / rosters.ROSTER_NAME).is_file()
            )
        except OSError:
            return {"events": []}
        found = []
        for directory in directories:
            event = directory.name
            try:
                roster = _roster(directory)
                saved = labels.read_saved(directory)
                categories = labels.categories(event)
                found.append({
                    "event": event,
                    "n_shots": len(roster),
                    "n_reviewed": int(roster.shot.isin(list(saved)).sum()),
                    "categories": {str(k): v for k, v in categories.items()},
                })
            except Exception as error:  # noqa: BLE001 - one bad roster must not hide the rest
                message = str(error) or type(error).__name__
                found.append({"event": event, "error": message})
        return {"events": found}

    @app.get("/api/queue")
    def queue(event: str):
        directory = require_event(event, paths)
        return labels.queue(directory, _roster(directory))

    @app.get("/api/shot")
    def shot_view(event: str, shot: int):
        directory = require_event(event, paths)
        tier = roster_tier(_roster(directory), shot)
        path, error = app.state.builds.status(event, shot)
        if error is not None:
            return JSONResponse({"error": error}, status_code=502)
        if path is None:
            body = {"building": True, "progress": raw.progress_for(shot)}
            return JSONResponse(body, status_code=202)
        return {
            "event": event,
            "shot": shot,
            "tier": tier,
            **rows.meta(path),
            **labels.shot_labels(directory, shot),
        }

    @app.get("/api/rows")
    def rows_view(
        event: str,
        shot: int,
        t0: float,
        t1: float,
        cols: Annotated[int, Query(ge=16, le=8192)] = 1500,
    ):
        require_event(event, paths)
        path = paths.spectrogram_file(event, shot)
        if not path.is_file():
            raise HTTPException(404, f"shot {shot} has no rows yet: open it first")
        try:
            data, grid = rows.read_window(path, t0, t1, cols)
        except (ValueError, OverflowError) as error:
            raise HTTPException(400, str(error)) from None
        return Response(
            data,
            media_type="application/octet-stream",
            headers={"X-Grid": json.dumps(grid)},
        )

    @app.post("/api/label")
    def save_label(body: LabelIn):
        directory = require_event(body.event, paths)
        tier = roster_tier(_roster(directory), body.shot)
        known = set(labels.categories(body.event))
        try:
            label = labels.normalise(body.window, body.intervals, known=known)
        except ValueError as error:
            raise HTTPException(400, str(error)) from None
        table = labels.source_path(directory)
        entry = labels.save(
            directory, body.shot, label, source=table.name if table else None
        )
        source = labels.read_source(directory).get(body.shot)
        row = {
            "shot": body.shot,
            "tier": tier,
            "state": labels.state(label, source),
            "saved_at": entry["saved_at"],
        }
        return {"row": row, "saved": label.as_json(), "last_save": entry}

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app
```

Patch `src/labeler/events/panels/__init__.py`:

```diff
diff --git a/src/labeler/events/panels/__init__.py b/src/labeler/events/panels/__init__.py
index c22afeb..9802cb5 100644
--- a/src/labeler/events/panels/__init__.py
+++ b/src/labeler/events/panels/__init__.py
@@ -4,26 +4,19 @@ Deciding that is case-by-case work that does not generalise, which is why it
 used to live in sixteen notebook cells. It lives here instead so it can be
 imported, tested, and called by a server - and so there is one copy of it.
 
-A builder module exposes `panels(shot, *, t_range, paths) -> list[Panel]` and
-a `GUIDANCE` string, the prose a reviewer needs beside the figure. An event
-with no module gets `_generic`, which is what twelve of those notebooks
-already were.
+A builder module exposes `panels(shot, *, t_range, paths) -> list[Panel]`.
+An event with no module gets `_generic`, which is what twelve of those
+notebooks already were. Alfven eigenmodes are not here: the review store
+builds them with the Heidbrink recipe, `review/alfven.py`.
 """
 
 from __future__ import annotations
 
 from ...config import Paths
 from ..verify import Panel
-from . import (
-    _generic,
-    alfven_eigenmode,
-    fishbone,
-    minimum_safety_factor,
-    sawtooth_oscillation,
-)
+from . import _generic, fishbone, minimum_safety_factor, sawtooth_oscillation
 
 BUILDERS = {
-    "alfven_eigenmode": alfven_eigenmode,
     "fishbone": fishbone,
     "minimum_safety_factor": minimum_safety_factor,
     "sawtooth_oscillation": sawtooth_oscillation,
@@ -40,8 +33,3 @@ def build(
     """The panels for one shot of one event, over one window."""
     builder = BUILDERS.get(event, _generic)
     return list(builder.panels(int(shot), t_range=t_range, paths=paths))
-
-
-def guidance(event: str) -> str:
-    """What a reviewer of this event needs to be told, as HTML."""
-    return BUILDERS.get(event, _generic).GUIDANCE
```

Patch `src/labeler/events/panels/_generic.py`:

```diff
diff --git a/src/labeler/events/panels/_generic.py b/src/labeler/events/panels/_generic.py
index ceaeabe..80862a7 100644
--- a/src/labeler/events/panels/_generic.py
+++ b/src/labeler/events/panels/_generic.py
@@ -12,14 +12,6 @@ from ...config import Paths
 from ...features.store import read_feature
 from ..verify import Panel
 
-GUIDANCE = (
-    "<b>These are generic panels.</b> <code>ip</code>, <code>betan</code> and "
-    "<code>pinj_total</code> show that the shot exists and that its labels sit "
-    "inside the discharge. They do not show whether this phenomenon actually "
-    "happened. Until someone adds a builder for this event under "
-    "<code>labeler/events/panels/</code>, treat a review here as provisional."
-)
-
 TRACES = (("ip", "A"), ("betan", ""), ("pinj_total", "kW"))
 
 
```

Patch `src/labeler/events/panels/fishbone.py`:

```diff
diff --git a/src/labeler/events/panels/fishbone.py b/src/labeler/events/panels/fishbone.py
index edfd8cb..46ebc6a 100644
--- a/src/labeler/events/panels/fishbone.py
+++ b/src/labeler/events/panels/fishbone.py
@@ -14,13 +14,6 @@ NPERSEG = 4096
 MAX_HZ = 40_000.0
 FISHBONE_BAND = (2.0, 30.0)
 
-GUIDANCE = (
-    "<b>What you are looking for:</b> bursts in 2-30 kHz (the shaded band) "
-    "that chirp DOWNWARD over a few milliseconds, with a cross-phase that "
-    "stays flat across the burst - that flatness is what says the two probes "
-    "are seeing one coherent mode rather than two patches of turbulence."
-)
-
 
 def panels(shot, *, t_range=None, paths=None):
     mhr = raw_signal(
@@ -30,7 +23,7 @@ def panels(shot, *, t_range=None, paths=None):
     # than raising: `scipy.signal.spectrogram` then runs with `fs=nan`,
     # returns `freq=[nan]`, and `freq <= MAX_HZ` is all-False since nan
     # comparisons are always False - a 0-row heatmap that renders blank
-    # instead of erroring. Mirror alfven_eigenmode.crosspower's guard.
+    # instead of erroring.
     if mhr.x.shape[0] < 2 or mhr.x[-1] == mhr.x[0]:
         raise ValueError(
             "fishbone needs a time vector spanning more than one instant; "
```

Patch `src/labeler/events/panels/minimum_safety_factor.py`:

```diff
diff --git a/src/labeler/events/panels/minimum_safety_factor.py b/src/labeler/events/panels/minimum_safety_factor.py
index a91295b..b8e6051 100644
--- a/src/labeler/events/panels/minimum_safety_factor.py
+++ b/src/labeler/events/panels/minimum_safety_factor.py
@@ -13,12 +13,6 @@ from ..verify import Panel
 #: is a line, so `hlines` draws one.
 THRESHOLDS = [0.95, 1.5, 2.0]
 
-GUIDANCE = (
-    "<b>What you are looking for:</b> where the qmin trace crosses 0.95, 1.5 "
-    "and 2.0 (the dashed lines), and whether the label boundary sits on the "
-    "crossing. The q profile below is on normalized psi, <b>not rho</b>."
-)
-
 
 def panels(shot, *, t_range=None, paths=None):
     paths = Paths.from_env() if paths is None else paths
```

Patch `src/labeler/events/panels/sawtooth_oscillation.py`:

```diff
diff --git a/src/labeler/events/panels/sawtooth_oscillation.py b/src/labeler/events/panels/sawtooth_oscillation.py
index 1e5fa88..c953cf7 100644
--- a/src/labeler/events/panels/sawtooth_oscillation.py
+++ b/src/labeler/events/panels/sawtooth_oscillation.py
@@ -16,15 +16,6 @@ CHANNEL_ROWS = (
     (32, 33, 34, 35),
 )
 
-GUIDANCE = (
-    "<b>What you are looking for:</b> a sawtooth ramp on the inner channels "
-    "that collapses abruptly while the outer channels jump up at the same "
-    "instant - the inversion across the q = 1 surface. A rise or fall that "
-    "moves every channel the same way is not a sawtooth."
-    "<br><br>A shot outside the corpus is fetched live, 48 ECE channels over "
-    "MDSplus, which is slow the first time and cached afterwards."
-)
-
 
 def panels(shot, *, t_range=None, paths=None):
     built = []
```

Delete `src/labeler/events/panels/alfven_eigenmode.py` (`git rm src/labeler/events/panels/alfven_eigenmode.py`).

- [ ] **Step 4: Run** the same two files: `58 passed` (the old page is still in `static/`, which
  these tests do not read beyond its links to `/app.js` and `/style.css`). Ruff on every touched
  Python file.

- [ ] **Step 5: Commit** `labeler review: the server, five routes over the store and the labels`.

### Task A8: The page

**Files:**
- Modify (rewrite): `src/labeler/events/ui/static/index.html`, `style.css`, `app.js`
- Test: `tests/labeler/test_review_page.py`, `tests/labeler/test_review_browser.py`,
  `tests/labeler/review_browser.mjs`

**Interfaces:**
- Consumes: A7's routes and gate.
- Produces: the page. For node, `app.js` exports its pure rules as
  `module.exports = {ms, paint, runs, normalise, diffRuns, niceStep, hitTest, lut}` and boots
  only in a browser. `normalise` agrees with `labels.normalise` case for case (the first test
  checks 302 cases).

- [ ] **Step 1: The failing tests.** `test_review_page.py` runs the script's rules under node.
  `test_review_browser.py` serves two synthetic shots and drives the page in headless Chromium
  through `review_browser.mjs`. The drive opens a shot from a link, draws a span, undoes it and
  draws it again, saves, reloads, zooms, then saves and moves on. It then opens a shot off the
  roster and carries on past it. Finally it checks `labels.csv` and `history.jsonl`.

```bash
python3 .claude/superpowers/plans/extract_blocks.py \
  .claude/superpowers/plans/2026-09-22-label-review-v4-only-pixi.md \
  tests/labeler/test_review_page.py tests/labeler/test_review_browser.py tests/labeler/review_browser.mjs
```

Write `tests/labeler/test_review_page.py`:

```python
"""The review page's own rules, run under node: they must agree with the server's."""

from __future__ import annotations

import json
import random
import shutil
import subprocess

import pytest

from labeler.events.review.labels import normalise
from labeler.events.ui.app import STATIC

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node runs the page's script")
APP = STATIC / "app.js"


def _node(expression: str, payload=None):
    """`expression` over `m` (the page's exports) and `input` (the payload)."""
    script = (
        "const m = require(process.argv[1]);"
        "const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
        f"process.stdout.write(JSON.stringify({expression}));"
    )
    result = subprocess.run(
        [NODE, "-e", script, str(APP)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(result.stdout)


def test_the_page_loads_nothing_from_the_network():
    for name in ("index.html", "app.js", "style.css"):
        assert "http" not in (STATIC / name).read_text().lower(), name


@needs_node
def test_the_script_parses():
    subprocess.run([NODE, "--check", str(APP)], check=True, timeout=30)


def _cases():
    rng = random.Random(0)
    cases = []
    for _ in range(300):
        lo = rng.uniform(-50, 50)
        width = rng.choice([0.3, 0.5, 40.0, 900.0, 25000.0])
        spans = []
        for _ in range(rng.randint(0, 4)):
            a = rng.uniform(-100, 1000)
            spans.append([a, a + rng.uniform(-3, 200), rng.choice([0, 1, 1, 1, 2])])
        cases.append({"window": [lo, lo + width], "intervals": spans})
    # Halves round up on both sides, and a span may land on the window's edge.
    cases.append({"window": [0.5, 10.5], "intervals": [[1.5, 2.5, 1], [-0.5, 0.5, 1]]})
    cases.append({"window": [-2.5, 2.5], "intervals": [[-1.5, -0.5, 1], [0.5, 0.5, 1]]})
    return cases


def _server(case):
    try:
        return normalise(case["window"], case["intervals"], known={1}).as_json()
    except ValueError:
        return None


@needs_node
def test_the_page_normalises_a_label_exactly_as_the_server_does():
    """What the page shows while editing is what a save will store."""
    cases = _cases()
    page = _node("input.map((c) => m.normalise(c.window, c.intervals, [1]))", cases)
    server = [_server(case) for case in cases]
    assert page == server
    assert None in server and any(server)


@needs_node
def test_a_change_is_where_the_label_leaves_its_source():
    source = {"window": [0, 100], "intervals": [[10, 20, 1]]}
    labels = [
        source,
        {"window": [0, 100], "intervals": [[10, 30, 1]]},
        {"window": [0, 120], "intervals": [[10, 20, 1]]},
        {"window": [0, 100], "intervals": []},
    ]
    found = _node("input.labels.map((b) => m.diffRuns(input.source, b))",
                  {"source": source, "labels": labels})
    assert found == [[], [[20, 30]], [[100, 120]], [[10, 20]]]


@needs_node
def test_ticks_step_by_one_two_or_five():
    spans = [[1000, 10], [9000, 8], [50, 10], [3.7, 5]]
    assert _node("input.map(([s, n]) => m.niceStep(s, n))", spans) == [100, 2000, 5, 1]


@needs_node
def test_a_press_on_the_label_grabs_the_nearest_thing():
    """Window edges only in the foot strip; span edges before span bodies."""
    label = {"window": [0, 1000], "intervals": [[100, 300, 1], [500, 520, 1]]}
    points = [[100, 5], [200, 5], [700, 5], [0, 35], [1000, 35], [510, 5], [298, 35]]
    found = _node(
        "input.points.map(([x, y]) => m.hitTest(input.label, x, y, 40, (t) => t))",
        {"label": label, "points": points},
    )
    assert found == [
        {"kind": "edge", "index": 0, "edge": 0},
        {"kind": "move", "index": 0},
        {"kind": "new"},
        {"kind": "window", "edge": 0},
        {"kind": "window", "edge": 1},
        {"kind": "move", "index": 1},
        {"kind": "edge", "index": 0, "edge": 1},
    ]


@needs_node
def test_the_colour_map_runs_from_inferno_black_to_its_yellow():
    """Packed as RGBA bytes; values under the contrast floor take its colour."""
    found = _node("[m.lut(0)[0], m.lut(0)[255], m.lut(128)[100] === m.lut(128)[0]]")
    assert found == [0xFF040000, 0xFFA4FFFC, True]
```

Write `tests/labeler/test_review_browser.py`:

```python
"""The review page in a real browser: draw a span, save, reload, move on."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import uvicorn

from labeler.config import Paths
from labeler.events.review import labels
from labeler.events.review import rows as review_rows
from labeler.events.review.rows import Grid, ImageRow, TraceRow
from labeler.events.ui.app import create_app

NODE = shutil.which("node")
SHELLS = sorted(
    Path.home().glob(
        ".cache/ms-playwright/chromium_headless_shell-*/chrome-linux/headless_shell"
    )
)
DRIVER = Path(__file__).with_name("review_browser.mjs")
TOKEN = "0123456789abcdef" * 2
ROSTER = (
    "shot,tier,holdout,reviewers,verified_on,notes\n"
    "170815,gold,false,,,\n"
    "170816,unverified,false,,,\n"
)
SOURCE = (
    "shot,category,t_start,t_end,confidence\n"
    "170815,0,0,100,\n"
    "170815,1,100,300,\n"
    "170815,0,300,2000,\n"
)

pytestmark = pytest.mark.skipif(
    NODE is None or not SHELLS, reason="needs node and a headless Chromium"
)


def _store(paths: Paths, shot: int) -> None:
    """Two rows over 0-4000 ms at 0.5 ms: a chirping image and a trace."""
    t = np.arange(8000) * 0.5
    bins = np.arange(64)[:, None]
    chirp = np.exp(-((bins - 32 - 20 * np.sin(t / 300)) ** 2) / 6)
    image = ImageRow(
        "R0", "R0", (40 + 200 * chirp).astype("uint8"),
        y0=0.0, dy=4.0, y_units="kHz", z_lo=-3.0, z_hi=27.0, z_units="dB",
    )
    wave = np.sin(t / 120)[None]
    trace = TraceRow("p1", "Density", np.stack([wave, wave]), hlines=[0.0])
    review_rows.write(
        paths.spectrogram_file("alfven_eigenmode", shot),
        Grid(0.0, 0.5, 8000), [image, trace], event="alfven_eigenmode",
    )


@pytest.fixture
def served(tmp_path):
    """The review server on a free loopback port, over two built shots."""
    paths = Paths(
        root=tmp_path / "root",
        corpus=tmp_path / "corpus",
        text_root=tmp_path / "text",
        logs_jsonl=tmp_path / "logs.jsonl",
        label_tables=tmp_path / "tables",
        raw_cache=tmp_path / "raw",
    )
    event = paths.label_tables / "alfven_eigenmode"
    (event / "format").mkdir(parents=True)
    (event / "shots.csv").write_text(ROSTER)
    (event / "format/alfven_eigenmode_format_2026_v1.csv").write_text(SOURCE)
    for shot in (170815, 170816):
        _store(paths, shot)
    app = create_app(paths=paths, token=TOKEN)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}", event
    server.should_exit = True
    thread.join(10)


def test_a_drawn_label_is_saved_found_again_and_left_behind(served, tmp_path):
    base, event = served
    result = subprocess.run(
        [NODE, str(DRIVER), base, TOKEN, str(SHELLS[-1]), str(tmp_path / "profile")],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    checks = json.loads(result.stdout.splitlines()[-1])
    assert [c for c in checks if not c["ok"]] == []
    assert len(checks) == 11
    saved = labels.read_saved(event)
    assert list(saved) == [170815]
    kept, (a, b, c) = saved[170815].intervals
    assert kept == (100, 300, 1) and c == 1 and abs(a - 500) <= 2 and abs(b - 800) <= 2
    assert [entry["shot"] for entry in labels.read_history(event)] == [170815] * 2
```

Write `tests/labeler/review_browser.mjs`:

```javascript
// The review page in headless Chromium, driven over the DevTools protocol.
//   node review_browser.mjs <base-url> <token> <headless-shell> <profile-dir>
// Prints one JSON line: every check made, [{name, ok, detail}].
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";

const [BASE, TOKEN, SHELL, PROFILE] = process.argv.slice(2);
const browser = spawn(
  SHELL,
  ["--no-sandbox", "--disable-gpu", "--remote-debugging-port=0", `--user-data-dir=${PROFILE}`,
    "--window-size=1400,900", "about:blank"],
  { stdio: "ignore" }
);
process.on("exit", () => browser.kill());

let port;
for (let i = 0; i < 200 && !port; i++) {
  try {
    port = readFileSync(`${PROFILE}/DevToolsActivePort`, "utf8").split("\n")[0];
  } catch {
    await sleep(50);
  }
}
const page = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((open) => ws.addEventListener("open", open));

let id = 0;
const pending = new Map();
const errors = [];
ws.addEventListener("message", ({ data }) => {
  const msg = JSON.parse(data);
  if (pending.has(msg.id)) {
    const [ok, fail] = pending.get(msg.id);
    pending.delete(msg.id);
    if (msg.error) fail(new Error(msg.error.message));
    else ok(msg.result);
  } else if (msg.method === "Runtime.exceptionThrown") {
    errors.push(msg.params.exceptionDetails.exception?.description);
  }
});
const send = (method, params = {}) =>
  new Promise((ok, fail) => {
    pending.set(++id, [ok, fail]);
    ws.send(JSON.stringify({ id, method, params }));
  });

async function js(expression) {
  const r = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description);
  return r.result.value;
}

async function until(expression, timeout = 20000) {
  for (const start = Date.now(); Date.now() - start < timeout; await sleep(50)) {
    if (await js(expression)) return;
  }
  throw new Error(`timed out waiting for ${expression}`);
}

const opened = (shot) => until(`typeof S !== "undefined" && S.shot === ${shot} && S.data !== null`);
const mouse = (type, x, y, more = {}) =>
  send("Input.dispatchMouseEvent", { type, x, y, button: "left", clickCount: 1, ...more });

async function drag(x0, x1, y) {
  await mouse("mouseMoved", x0, y, { button: "none" });
  await mouse("mousePressed", x0, y, { buttons: 1 });
  for (let i = 1; i <= 8; i++) await mouse("mouseMoved", x0 + ((x1 - x0) * i) / 8, y, { buttons: 1 });
  await mouse("mouseReleased", x1, y, { buttons: 0 });
}

/** Draw a span over t0-t1 ms on the label track. */
async function draw(t0, t1) {
  const [x0, x1, y] = await js(`(() => {
    const r = $("label-track").getBoundingClientRect();
    return [r.left + px(${t0}), r.left + px(${t1}), r.top + 15];
  })()`);
  await drag(x0, x1, y);
}

async function press(key, modifiers = 0) {
  const named = { Enter: [13, "\r"] }[key];
  const [code, text] = named || [key.toUpperCase().charCodeAt(0), key];
  const event = { key, code: named ? key : `Key${key.toUpperCase()}`, windowsVirtualKeyCode: code, modifiers };
  await send("Input.dispatchKeyEvent", { type: "keyDown", ...event, ...(modifiers ? {} : { text }) });
  await send("Input.dispatchKeyEvent", { type: "keyUp", ...event });
}

const checks = [];
const check = (name, ok, detail) => checks.push({ name, ok: Boolean(ok), detail });
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

try {
  await send("Runtime.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: 1400, height: 900, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url: `${BASE}/?token=${TOKEN}#alfven_eigenmode/170815` });
  await opened(170815);
  const source = await js("S.meta.source");
  check("the link opens its shot on its source label", same(await js("S.label"), source), source);

  await draw(500, 800);
  check("a drag on the label adds a span", (await js("S.label.intervals.length")) === 2, await js("S.label"));
  check(
    "the edit shows as unsaved and is kept as a draft",
    await js(`!$("dirty").hidden && localStorage.getItem("labeler:alfven_eigenmode:170815") !== null`)
  );
  await press("z", 2);
  check("ctrl-z undoes it", same(await js("S.label"), source) && (await js(`$("dirty").hidden`)));

  await draw(500, 800);
  await press("s");
  await until("S.meta.saved !== null && !S.saving");
  const saved = await js("S.label");
  check(
    "s saves it and says so",
    await js(`$("state").textContent === "changed" && $("saved").textContent.startsWith("saved")`),
    await js(`[$("state").textContent, $("saved").textContent]`)
  );

  await send("Page.reload");
  await opened(170815);
  check(
    "a reload opens the saved label",
    same(await js("S.label"), saved) && (await js(`$("count").textContent`)) === "1/2",
    await js("S.label")
  );

  const [x, y, before] = await js(`(() => {
    const r = $("top").getBoundingClientRect();
    return [r.left + px(1000), r.top + 60, S.view[1] - S.view[0]];
  })()`);
  await send("Input.dispatchMouseEvent", { type: "mouseWheel", x, y, deltaX: 0, deltaY: -400, modifiers: 2 });
  await until(`S.view[1] - S.view[0] < ${before / 2} && S.data.t1 - S.data.t0 < ${before}`);
  check("ctrl-wheel zooms in and fetches the rows it shows", true, await js("[S.view, S.data.n]"));

  await press("Enter");
  await opened(170816);
  check("enter saves and opens the next unreviewed shot", true);

  await js(`$("shot").focus(); $("shot").value = "1"`);
  await press("Enter");
  await until(`$("rows").querySelector(".failure") !== null`);
  check("a shot off the roster says why", await js(`$("rows").textContent.includes("not on this event")`),
    await js(`$("rows").textContent`));
  await js("document.activeElement.blur()");
  await press("k");
  await opened(170815);
  check("k carries on from it", true);
} catch (error) {
  check("the page did what was asked", false, String(error));
}
check("no script error", errors.length === 0, errors);
console.log(JSON.stringify(checks));
process.exit(0);
```

- [ ] **Step 2: Run** `tests/labeler/test_review_page.py tests/labeler/test_review_browser.py tests/labeler/test_events_ui.py`. Expect `6 failed, 53 passed`. The old
  script exports nothing to node (five page tests), and the driver times out waiting for the new
  page's state (the browser test).

- [ ] **Step 3: The page.**

```bash
python3 .claude/superpowers/plans/extract_blocks.py \
  .claude/superpowers/plans/2026-09-22-label-review-v4-only-pixi.md \
  src/labeler/events/ui/static/index.html src/labeler/events/ui/static/style.css src/labeler/events/ui/static/app.js
```

Write `src/labeler/events/ui/static/index.html`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label review</title>
<link rel="stylesheet" href="/style.css">
</head>
<body>
<header id="bar">
  <select id="event" aria-label="Event"></select>
  <span id="count"></span>
  <label>Shot <input id="shot" inputmode="numeric" autocomplete="off"></label>
  <span id="dirty" title="Unsaved" hidden>●</span>
  <span id="tier"></span>
  <span id="state" class="pill"></span>
  <span id="saved"></span>
  <span id="status" role="status" aria-live="polite"></span>
  <button id="help" type="button" aria-label="Keys">?</button>
</header>
<main id="top">
  <div id="rows"></div>
  <canvas id="axis-row" aria-hidden="true"></canvas>
</main>
<footer id="bottom">
  <div class="track"><span class="name">Source</span><canvas id="source-track"></canvas></div>
  <div class="track"><span class="name">Label</span><canvas id="label-track"></canvas></div>
  <div id="controls">
    <div id="swatches"></div>
    <button id="save-next" class="primary" type="button">Save and next <kbd>Enter</kbd></button>
    <button id="revert" type="button">Revert <kbd>R</kbd></button>
  </div>
  <nav id="queue" aria-label="Shots"></nav>
</footer>
<div id="cursor" hidden><span id="cursor-time"></span></div>
<dialog id="keys">
  <table>
    <tr><td><kbd>Enter</kbd></td><td>Save and next</td></tr>
    <tr><td><kbd>S</kbd></td><td>Save</td></tr>
    <tr><td><kbd>R</kbd></td><td>Revert to source</td></tr>
    <tr><td><kbd>J</kbd> <kbd>K</kbd></td><td>Previous, next shot</td></tr>
    <tr><td><kbd>U</kbd></td><td>Next unreviewed</td></tr>
    <tr><td>Drag on the label</td><td>Add, move or resize a span; the foot moves the window</td></tr>
    <tr><td>Shift-drag on the rows</td><td>Add a span</td></tr>
    <tr><td>Drag on the rows</td><td>Pan</td></tr>
    <tr><td><kbd>Ctrl</kbd>/<kbd>⌘</kbd> wheel</td><td>Zoom</td></tr>
    <tr><td>Double-click</td><td>Fit the window</td></tr>
    <tr><td><kbd>←</kbd> <kbd>→</kbd></td><td>Pan</td></tr>
    <tr><td><kbd>-</kbd> <kbd>=</kbd> <kbd>0</kbd></td><td>Zoom out, in, fit</td></tr>
    <tr><td><kbd>1</kbd>-<kbd>9</kbd></td><td>Category</td></tr>
    <tr><td><kbd>Delete</kbd></td><td>Remove the span</td></tr>
    <tr><td><kbd>[</kbd> <kbd>]</kbd></td><td>Contrast</td></tr>
    <tr><td><kbd>Ctrl</kbd>/<kbd>⌘</kbd> <kbd>Z</kbd></td><td>Undo</td></tr>
  </table>
  <form method="dialog"><button>Close</button></form>
</dialog>
<script src="/app.js"></script>
</body>
</html>
```

Write `src/labeler/events/ui/static/style.css`:

```css
:root {
  --paper: #eceeec;
  --panel: #fff;
  --ink: #1b211e;
  --muted: #69736e;
  --rule: #cdd3cf;
  --label: #0e8a8c;
  --confirmed: #0e8a8c;
  --unsaved: #d69a00;
  --changed: #6b4fa0;
  --unreviewed: #d3d8d5;
  --error: #b3261e;
  --veil: rgba(236, 238, 236, 0.62);
  --gutter: 96px;
  color-scheme: light;
  font: 13px/1.35 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}

@media (prefers-color-scheme: dark) {
  :root {
    --paper: #151a18;
    --panel: #1d2321;
    --ink: #e4e8e6;
    --muted: #9aa4a0;
    --rule: #333b38;
    --label: #2bb3b1;
    --confirmed: #2bb3b1;
    --changed: #9b7fd0;
    --unreviewed: #3a423f;
    --error: #f2b8b5;
    --veil: rgba(21, 26, 24, 0.62);
    color-scheme: dark;
  }
}

* { box-sizing: border-box; }

body {
  margin: 0;
  height: 100vh;
  height: 100dvh;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto;
  background: var(--paper);
  color: var(--ink);
  font-variant-numeric: tabular-nums;
}

button, select, input {
  font: inherit;
  color: inherit;
  background: var(--paper);
  border: 1px solid var(--rule);
  border-radius: 4px;
  padding: 3px 8px;
}

button { cursor: pointer; }
button:disabled { cursor: default; opacity: 0.5; }
:focus-visible { outline: 2px solid var(--label); outline-offset: 1px; }
kbd { font: inherit; font-size: 11px; opacity: 0.7; }

#bar {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 6px 12px;
  background: var(--panel);
  border-bottom: 1px solid var(--rule);
  white-space: nowrap;
  overflow: hidden;
}

#shot { width: 7em; margin-left: 4px; }
#count, #tier, #saved { color: var(--muted); }
#dirty { color: var(--unsaved); font-size: 16px; line-height: 1; }
#status { margin-left: auto; color: var(--muted); overflow: hidden; text-overflow: ellipsis; }
#status.error { color: var(--error); }
#help { width: 28px; padding: 3px 0; }

.pill { padding: 1px 9px; border-radius: 999px; background: var(--unreviewed); }
.pill:empty { display: none; }
.pill.confirmed { background: var(--confirmed); color: #fff; }
.pill.changed { background: var(--changed); color: #fff; }

#top {
  overflow-x: hidden;
  overflow-y: auto;
  scrollbar-gutter: stable;
  background: var(--panel);
}

canvas { display: block; width: 100%; }
#rows canvas { cursor: grab; }
#rows canvas:active { cursor: grabbing; }

#rows .failure { margin: 0; padding: 40px 12px 40px var(--gutter); color: var(--muted); }
body.busy #rows, body.busy .track canvas { opacity: 0.35; pointer-events: none; }

#axis-row {
  position: sticky;
  bottom: 0;
  height: 24px;
  background: var(--panel);
  border-top: 1px solid var(--rule);
}

#bottom {
  max-height: 42vh;
  overflow-y: auto;
  scrollbar-gutter: stable;
  padding-bottom: 10px;
  border-top: 1px solid var(--rule);
}

.track { position: relative; margin-top: 6px; }
.track .name {
  position: absolute;
  left: 8px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--muted);
  pointer-events: none;
}
#source-track { height: 22px; }
#label-track { height: 40px; cursor: crosshair; touch-action: none; }

#controls, #queue { padding: 0 12px 0 var(--gutter); }
#controls { display: flex; align-items: center; gap: 8px; margin: 10px 0; }
#swatches { display: flex; gap: 6px; margin-right: auto; }
.swatch { display: inline-flex; align-items: center; gap: 6px; }
.swatch::before { content: ""; width: 11px; height: 11px; border-radius: 2px; background: var(--c); }
.swatch[aria-pressed="true"] { border-color: var(--ink); }
button.primary { background: var(--label); border-color: var(--label); color: #fff; }

#queue {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  max-height: 68px; /* three rows of chips */
  overflow-y: auto;
  scroll-snap-type: y mandatory;
}
.chip {
  position: relative;
  height: 20px;
  padding: 0 6px;
  scroll-snap-align: start;
  font-size: 12px;
  border-color: transparent;
  border-radius: 3px;
  background: var(--unreviewed);
}
.chip.confirmed { background: var(--confirmed); color: #fff; }
.chip.changed { background: var(--changed); color: #fff; }
.chip.current { border-color: var(--ink); }
.chip.dirty::after {
  content: "";
  position: absolute;
  top: -3px;
  right: -3px;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--unsaved);
}

#cursor {
  position: fixed;
  width: 1px;
  background: var(--ink);
  opacity: 0.5;
  pointer-events: none;
}
#cursor-time {
  position: absolute;
  top: 2px;
  left: 5px;
  padding: 0 4px;
  font-size: 11px;
  background: var(--panel);
  white-space: nowrap;
}

dialog {
  border: 1px solid var(--rule);
  border-radius: 6px;
  background: var(--panel);
  color: var(--ink);
}
dialog::backdrop { background: rgba(0, 0, 0, 0.3); }
dialog td { padding: 2px 16px 2px 0; }
dialog form { margin-top: 10px; text-align: right; }
```

Write `src/labeler/events/ui/static/app.js`:

```javascript
// The label review page: spectrogram rows on top, one editable label per shot below.
"use strict";

// ---- The label rules, as the server applies them; exported for the tests ----

const LONGEST_WINDOW_MS = 20000;
const HANDLE_BAND = 10; // px at the foot of the label track that hold the window's edges
const GRAB = 5; // px either side of an edge that grab it
const INFERNO = [
  "000004", "0b0724", "210c4a", "3d0965", "57106e", "71196e", "8a226a", "a32c61", "bc3754",
  "d24644", "e45a31", "f1731d", "f98e09", "fcac11", "f9cb35", "f2ea69", "fcffa4",
];

/** Whole ms, halves up, as the server's `_ms`. */
function ms(t) {
  return Math.floor(Number(t) + 0.5);
}

/** One cell per ms of the window, holding the category painted there last. */
function paint(window, spans) {
  const [lo, hi] = window;
  const cells = new Int32Array(hi - lo);
  for (const [a, b, c] of spans) {
    const start = Math.max(a, lo) - lo;
    const stop = Math.min(b, hi) - lo;
    if (stop > start) cells.fill(c, start, stop);
  }
  return cells;
}

/** [start, stop, value] of each run of equal cells, as offsets. */
function runs(cells) {
  const out = [];
  for (let start = 0, i = 1; i <= cells.length; i++) {
    if (i === cells.length || cells[i] !== cells[start]) {
      out.push([start, i, cells[start]]);
      start = i;
    }
  }
  return out;
}

/** The server's `normalise`: snap to whole ms, grow the window over every span, merge
 * and clip; later spans paint over earlier ones. Null where the server would refuse. */
function normalise(window, intervals, known) {
  let lo = ms(window[0]);
  let hi = ms(window[1]);
  const spans = [];
  for (const [a0, b0, c0] of intervals) {
    const [a, b, c] = [ms(a0), ms(b0), Math.trunc(c0)];
    if (b < a || (known && c && !known.includes(c))) return null;
    if (b > a) spans.push([a, b, c]);
  }
  for (const [a, b, c] of spans) {
    if (c) [lo, hi] = [Math.min(lo, a), Math.max(hi, b)];
  }
  if (!(hi - lo > 0 && hi - lo <= LONGEST_WINDOW_MS)) return null;
  const painted = runs(paint([lo, hi], spans)).filter(([, , c]) => c);
  return { window: [lo, hi], intervals: painted.map(([a, b, c]) => [lo + a, lo + b, c]) };
}

/** The ms ranges where two labels disagree; outside a window counts as unknown. */
function diffRuns(a, b) {
  const lo = Math.min(a.window[0], b.window[0]);
  const hi = Math.max(a.window[1], b.window[1]);
  const cells = (label) => {
    const out = new Int32Array(hi - lo).fill(-1);
    out.fill(0, label.window[0] - lo, label.window[1] - lo);
    for (const [s, e, c] of label.intervals) out.fill(c, s - lo, e - lo);
    return out;
  };
  const [x, y] = [cells(a), cells(b)];
  return runs(x.map((v, i) => (v === y[i] ? 0 : 1)))
    .filter(([, , differs]) => differs)
    .map(([s, e]) => [lo + s, lo + e]);
}

/** A tick step giving about `n` ticks over `span`: 1, 2 or 5 times a power of ten. */
function niceStep(span, n) {
  const raw = span / n;
  const power = 10 ** Math.floor(Math.log10(raw));
  const f = raw / power;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * power;
}

/** What a press at (x, y) on the label track grabs: a window edge (only in the foot
 * strip), a span edge, a span, or nothing. `px` maps ms to the track's px. */
function hitTest(label, x, y, height, px) {
  if (y >= height - HANDLE_BAND) {
    for (const edge of [0, 1]) {
      if (Math.abs(x - px(label.window[edge])) <= GRAB) return { kind: "window", edge };
    }
  }
  const spans = label.intervals;
  for (let index = spans.length - 1; index >= 0; index--) {
    for (const edge of [0, 1]) {
      if (Math.abs(x - px(spans[index][edge])) <= GRAB) return { kind: "edge", index, edge };
    }
  }
  for (let index = spans.length - 1; index >= 0; index--) {
    if (x > px(spans[index][0]) && x < px(spans[index][1])) return { kind: "move", index };
  }
  return { kind: "new" };
}

/** Colours for the byte values 0-255: inferno from `lo` up, its floor below `lo`. */
function lut(lo, hi = 255) {
  const anchors = INFERNO.map((hex) => [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16)));
  const out = new Uint32Array(256);
  for (let v = 0; v < 256; v++) {
    const t = Math.min(1, Math.max(0, (v - lo) / (hi - lo))) * (anchors.length - 1);
    const i = Math.min(anchors.length - 2, Math.floor(t));
    const [r, g, b] = anchors[i].map((c, k) => Math.round(c + (anchors[i + 1][k] - c) * (t - i)));
    out[v] = ((255 << 24) | (b << 16) | (g << 8) | r) >>> 0; // RGBA bytes, little-endian
  }
  return out;
}

// ---- The page ----

const GUTTER = 96; // px left of every plot: a row's title and units, then its y ticks
const RIGHT = 12;
const IMAGE_H = 150;
const TRACE_H = 110;
const CATEGORY_COLOURS = { 2: "#e69f00", 3: "#cc79a7", 4: "#56b4e9" }; // 1 is --label
const TRACE_COLOURS = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9"];
const FONT = "11px system-ui, sans-serif";

const S = {
  events: [],
  event: null,
  categories: {},
  category: 1,
  queue: [],
  shot: null,
  meta: null, // what /api/shot said: grid, t_range, rows, source, saved, state, last_save
  label: null, // the label being edited, always normalised
  selected: -1,
  view: [0, 1], // the ms on screen
  data: null, // the last /api/rows: {t0, t1, n, rows}
  asked: "", // the last /api/rows query sent
  bitmaps: new Map(),
  lo: 0,
  lut: lut(0),
  undo: [],
  drag: null,
  ticket: 0, // bumped per shot opened, so an answer for an older shot is dropped
  frame: 0,
  timer: 0,
  saving: false,
};
let T = {}; // colour tokens, read from the stylesheet
const $ = (id) => document.getElementById(id);
const enc = encodeURIComponent;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const sleep = (delay) => new Promise((done) => setTimeout(done, delay));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const known = () => Object.keys(S.categories).map(Number);
const draftKey = (shot) => `labeler:${S.event}:${shot}`;

function stored(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function store(key, value) {
  try {
    if (value == null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    // storage refused (a private window): a draft then lasts as long as the page
  }
}

async function api(path, options) {
  const response = await fetch(path, { credentials: "same-origin", ...options });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || `${response.status} ${response.statusText}`);
  }
  return response;
}

function say(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}

function readTokens() {
  const style = getComputedStyle(document.documentElement);
  const names = ["panel", "ink", "muted", "rule", "label", "changed", "veil"];
  T = Object.fromEntries(names.map((name) => [name, style.getPropertyValue(`--${name}`).trim()]));
}

// -- time and geometry

const plotWidth = () => Math.max(1, $("axis-row").clientWidth - GUTTER - RIGHT);
const px = (t) => GUTTER + ((t - S.view[0]) / (S.view[1] - S.view[0])) * plotWidth();
const middle = () => (S.view[0] + S.view[1]) / 2;

function timeAt(clientX) {
  const x = clientX - $("axis-row").getBoundingClientRect().left;
  return S.view[0] + ((x - GUTTER) / plotWidth()) * (S.view[1] - S.view[0]);
}

function setView(t0, t1) {
  const lo = Math.min(S.meta.t_range[0], S.label.window[0]);
  const hi = Math.max(S.meta.t_range[1], S.label.window[1]);
  const span = clamp(t1 - t0, 5, hi - lo);
  const start = clamp(t0, lo, hi - span);
  S.view = [start, start + span];
  render();
  fetchRows();
}

function zoomAt(t, factor) {
  const [v0, v1] = S.view;
  setView(t - (t - v0) * factor, t + (v1 - t) * factor);
}

function pan(fraction) {
  const shift = (S.view[1] - S.view[0]) * fraction;
  setView(S.view[0] + shift, S.view[1] + shift);
}

function fit() {
  const [lo, hi] = S.label.window;
  setView(lo - (hi - lo) * 0.05, hi + (hi - lo) * 0.05);
}

// -- the label

function emptyLabel() {
  const lo = ms(S.meta.t_range[0]);
  return { window: [lo, Math.min(ms(S.meta.t_range[1]), lo + LONGEST_WINDOW_MS)], intervals: [] };
}

const baseline = () => S.meta.saved || S.meta.source || emptyLabel();
const dirty = () => Boolean(S.meta) && !same(S.label, baseline());

function draft(shot) {
  try {
    const label = JSON.parse(stored(draftKey(shot)));
    return label && normalise(label.window, label.intervals, known());
  } catch {
    return null;
  }
}

function keepForUndo(label) {
  S.undo.push(label);
  if (S.undo.length > 100) S.undo.shift();
}

/** Take a new label if the server would accept it. */
function edit(window, intervals) {
  const next = normalise(window, intervals, known());
  if (!next) return;
  keepForUndo(S.label);
  S.label = next;
  touch();
}

/** Keep the draft until it is saved or reverted, and mark the shot unsaved. */
function touch() {
  store(draftKey(S.shot), dirty() ? JSON.stringify(S.label) : null);
  $("dirty").hidden = !dirty();
  renderQueue();
  render();
}

/** Select the span holding `t`, if any. */
function selectAt(t) {
  S.selected = S.label.intervals.findIndex(([a, b]) => a <= t && t <= b);
}

function undo() {
  if (!S.undo.length) return;
  S.label = S.undo.pop();
  S.selected = -1;
  touch();
}

function revert() {
  const source = S.meta.source || emptyLabel();
  S.selected = -1;
  edit(source.window, source.intervals);
}

function removeSelected() {
  if (S.selected < 0) return;
  const kept = S.label.intervals.filter((_, i) => i !== S.selected);
  S.selected = -1;
  edit(S.label.window, kept);
}

function setCategory(c) {
  if (!known().includes(c)) return;
  S.category = c;
  renderSwatches();
  if (S.selected < 0) return;
  const [a, b] = S.label.intervals[S.selected];
  edit(S.label.window, S.label.intervals.map((s, i) => (i === S.selected ? [a, b, c] : s)));
  selectAt((a + b) / 2);
}

function contrast(delta) {
  S.lo = clamp(S.lo + delta, 0, 192);
  S.lut = lut(S.lo);
  store("labeler:contrast", String(S.lo));
  S.bitmaps.clear();
  render();
}

// -- loading

async function boot() {
  readTokens();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    readTokens();
    renderSwatches();
    render();
  });
  S.lo = clamp(Number(stored("labeler:contrast")) || 0, 0, 192);
  S.lut = lut(S.lo);
  wire();
  try {
    S.events = (await (await api("/api/events")).json()).events;
    $("event").replaceChildren(
      ...S.events.map((row) => {
        const option = new Option(row.error ? `${row.event} (broken)` : row.event, row.event);
        option.disabled = Boolean(row.error);
        option.title = row.error || "";
        return option;
      })
    );
    const usable = S.events.filter((row) => !row.error);
    if (!usable.length) return say("no events", true);
    const [hashEvent, hashShot] = fromHash();
    const pick =
      [hashEvent, stored("labeler:event")].find((name) => usable.some((r) => r.event === name)) ||
      usable.reduce((a, b) => (b.n_shots > a.n_shots ? b : a)).event;
    await openEvent(pick, hashShot);
  } catch (error) {
    say(error.message, true);
  }
}

/** `#event/shot`, the address of what is open. */
function fromHash() {
  const [event, shot] = decodeURIComponent(location.hash.slice(1)).split("/");
  return [event, Number(shot) || null];
}

/** A link pasted into the address bar opens what it names. */
function followHash() {
  const [event, shot] = fromHash();
  if (event !== S.event && S.events.some((row) => row.event === event && !row.error)) {
    openEvent(event, shot).catch((error) => say(error.message, true));
  } else if (event === S.event && shot && shot !== S.shot) {
    openShot(shot);
  }
}

async function openEvent(event, shot) {
  S.event = event;
  S.categories = S.events.find((row) => row.event === event).categories;
  S.category = known()[0] || 1;
  $("event").value = event;
  store("labeler:event", event);
  renderSwatches();
  $("queue").replaceChildren();
  const queue = await (await api(`/api/queue?event=${enc(event)}`)).json();
  S.queue = queue.shots;
  renderQueue();
  await openShot(S.queue.some((row) => row.shot === shot) ? shot : queue.resume);
}

async function openShot(shot) {
  if (shot == null) return;
  const ticket = ++S.ticket;
  try {
    let response, meta;
    for (;;) {
      response = await api(`/api/shot?event=${enc(S.event)}&shot=${shot}`);
      meta = await response.json();
      if (ticket !== S.ticket) return;
      if (response.status !== 202) break;
      const p = meta.progress;
      busy(`${shot}: building${p ? ` ${p.done}/${p.total} ${p.stage}` : ""}`);
      await sleep(800);
      if (ticket !== S.ticket) return;
    }
    Object.assign(S, { shot, meta, data: null, asked: "", undo: [], selected: -1 });
    S.bitmaps.clear();
    S.label = draft(shot) || baseline();
    buildRows();
    arrive();
    fit();
    fetchRows(0);
    prefetch();
  } catch (error) {
    if (ticket !== S.ticket) return;
    // Stay on the shot, with nothing to edit, so J, K and U carry on from it.
    Object.assign(S, { shot, meta: null, data: null, label: null, undo: [], selected: -1 });
    const note = document.createElement("p");
    note.className = "failure";
    note.textContent = `${shot} has nothing to show: ${error.message}. K opens the next shot.`;
    $("rows").replaceChildren(note);
    for (const canvas of document.querySelectorAll("#top canvas, .track canvas")) context(canvas);
    arrive();
  }
}

/** The header, the address and the queue follow the shot just opened. */
function arrive() {
  busy(null);
  $("cursor").hidden ||= !S.meta;
  for (const id of ["save-next", "revert"]) $(id).disabled = !S.meta;
  history.replaceState(null, "", `#${S.event}/${S.shot}`);
  $("shot").value = S.shot;
  showHeader();
  renderQueue();
  $("queue").querySelector(".current")?.scrollIntoView({ block: "nearest" });
}

/** Dim the shot on screen while the next one builds. */
function busy(text) {
  document.body.classList.toggle("busy", Boolean(text));
  say(text || "");
}

/** Ask for the next shot now, so a build it needs is done when the reviewer gets there. */
function prefetch() {
  const next = nextUnreviewed() ?? neighbour(1);
  if (next != null && next !== S.shot) api(`/api/shot?event=${enc(S.event)}&shot=${next}`).catch(() => {});
}

function nextUnreviewed() {
  const i = S.queue.findIndex((row) => row.shot === S.shot);
  const after = [...S.queue.slice(i + 1), ...S.queue.slice(0, Math.max(i, 0))];
  return after.find((row) => row.state === "unreviewed")?.shot ?? null;
}

function neighbour(delta) {
  const i = S.queue.findIndex((row) => row.shot === S.shot);
  return S.queue[(i + delta + S.queue.length) % S.queue.length]?.shot ?? null;
}

function fetchRows(delay = 90) {
  clearTimeout(S.timer);
  S.timer = setTimeout(loadRows, delay);
}

async function loadRows() {
  if (!S.meta) return;
  const [v0, v1] = S.view;
  // A fifth wider than the view either side, so a pan has data while the next one loads.
  const t0 = Math.max(S.meta.t_range[0], v0 - (v1 - v0) * 0.2);
  const t1 = Math.min(S.meta.t_range[1], v1 + (v1 - v0) * 0.2);
  if (t1 <= t0) return;
  const cols = clamp(Math.round((plotWidth() * (t1 - t0)) / (v1 - v0)), 16, 8192);
  const query = `event=${enc(S.event)}&shot=${S.shot}&t0=${t0}&t1=${t1}&cols=${cols}`;
  if (query === S.asked) return;
  S.asked = query;
  const ticket = S.ticket;
  try {
    const response = await api(`/api/rows?${query}`);
    const grid = JSON.parse(response.headers.get("X-Grid"));
    const buffer = await response.arrayBuffer();
    if (ticket !== S.ticket || query !== S.asked) return;
    S.data = { ...grid, rows: unpack(buffer, grid.n) };
    S.bitmaps.clear();
    render();
  } catch (error) {
    if (ticket === S.ticket) say(error.message, true);
  }
}

/** Split /api/rows' bytes into rows: images uint8 (n_y, n), traces float32 (2, C, n). */
function unpack(buffer, n) {
  let offset = 0;
  return S.meta.rows.map((row) => {
    if (row.kind === "image") {
      const values = new Uint8Array(buffer, offset, row.n_y * n);
      offset += row.n_y * n;
      return values;
    }
    const bytes = 2 * row.n_channels * n * 4;
    const values = new Float32Array(buffer.slice(offset, offset + bytes)); // a copy: aligned
    offset += bytes;
    return values;
  });
}

async function save(next) {
  if (!S.meta || S.saving) return;
  S.saving = true;
  const [shot, key] = [S.shot, draftKey(S.shot)];
  try {
    const response = await api("/api/label", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ event: S.event, shot, ...S.label }),
    });
    const body = await response.json();
    store(key, null);
    S.queue = S.queue.map((row) => (row.shot === shot ? body.row : row));
    if (S.shot === shot) {
      Object.assign(S.meta, { saved: body.saved, last_save: body.last_save });
      S.label = body.saved;
      showHeader();
      touch();
    }
    if (next) await openShot(nextUnreviewed() ?? neighbour(1));
  } catch (error) {
    say(error.message, true);
  } finally {
    S.saving = false;
  }
}

// -- drawing

function buildRows() {
  $("rows").replaceChildren(...S.meta.rows.map(() => document.createElement("canvas")));
  sizeCanvases();
}

/** Each row at its own height, or taller when the rows would not fill the view. */
function sizeRows() {
  if (!S.meta) return;
  const base = S.meta.rows.map((row) => (row.kind === "image" ? IMAGE_H : TRACE_H));
  const room = $("top").clientHeight - $("axis-row").offsetHeight;
  const scale = Math.max(1, room / base.reduce((a, b) => a + b, 0));
  [...$("rows").children].forEach((canvas, i) => {
    canvas.style.height = `${Math.floor(base[i] * scale)}px`;
  });
}

function sizeCanvases() {
  sizeRows();
  const ratio = window.devicePixelRatio || 1;
  for (const canvas of document.querySelectorAll("canvas")) {
    const [w, h] = [canvas.clientWidth * ratio, canvas.clientHeight * ratio].map(Math.round);
    if (canvas.width !== w || canvas.height !== h) [canvas.width, canvas.height] = [w, h];
  }
}

/** A cleared 2D context drawing in CSS px. */
function context(canvas) {
  const g = canvas.getContext("2d");
  const ratio = canvas.width / Math.max(1, canvas.clientWidth);
  g.setTransform(ratio, 0, 0, ratio, 0, 0);
  g.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  return g;
}

function render() {
  if (S.frame || !S.meta) return;
  S.frame = requestAnimationFrame(() => {
    S.frame = 0;
    drawRows();
    drawAxis();
    drawTrack($("source-track"), S.meta.source, false);
    drawTrack($("label-track"), S.label, true);
  });
}

const categoryColour = (c) => (c === 1 ? T.label : CATEGORY_COLOURS[c] || T.muted);

function drawRows() {
  const canvases = $("rows").children;
  S.meta.rows.forEach((row, i) => {
    const canvas = canvases[i];
    const [w, h] = [canvas.clientWidth, canvas.clientHeight];
    const g = context(canvas);
    const values = S.data && S.data.rows[i];
    const range = row.kind === "image" ? imageRange(row) : values && traceRange(row, values);
    g.save();
    g.beginPath();
    g.rect(GUTTER, 0, w - GUTTER - RIGHT, h);
    g.clip();
    if (values && row.kind === "image") drawImage(g, row, values, i, h);
    if (values && row.kind === "trace") drawTrace(g, row, values, range, w, h);
    drawOverlay(g, w, h);
    g.restore();
    drawGutter(g, row, range, h);
    g.fillStyle = T.rule;
    g.fillRect(0, h - 1, w, 1);
  });
}

/** The bins an image row shows, [first, stop): its band if it has one, else all. */
function bandBins(row) {
  if (!row.band) return [0, row.n_y];
  const bin = (y) => clamp(Math.round((y - row.y0) / row.dy), 0, row.n_y - 1);
  return [bin(row.band[0]), bin(row.band[1]) + 1];
}

function imageRange(row) {
  const [first, stop] = bandBins(row);
  return [row.y0 + (first - 0.5) * row.dy, row.y0 + (stop - 0.5) * row.dy];
}

function drawImage(g, row, values, i, h) {
  const [first, stop] = bandBins(row);
  const [x0, x1] = [px(S.data.t0), px(S.data.t1)];
  g.imageSmoothingEnabled = false;
  g.drawImage(bitmap(row, values, i), 0, row.n_y - stop, S.data.n, stop - first, x0, 0, x1 - x0, h - 1);
}

/** A row's image at the current contrast: a pixel per column and bin, top bin first. */
function bitmap(row, values, i) {
  let canvas = S.bitmaps.get(i);
  if (canvas) return canvas;
  const n = S.data.n;
  canvas = document.createElement("canvas");
  [canvas.width, canvas.height] = [n, row.n_y];
  const g = canvas.getContext("2d");
  const image = g.createImageData(n, row.n_y);
  const pixels = new Uint32Array(image.data.buffer);
  for (let y = 0; y < row.n_y; y++) {
    const from = (row.n_y - 1 - y) * n; // the store's row 0 is the lowest bin
    for (let x = 0; x < n; x++) pixels[y * n + x] = S.lut[values[from + x]];
  }
  g.putImageData(image, 0, 0);
  S.bitmaps.set(i, canvas);
  return canvas;
}

/** A trace row's y range over the columns on screen, its dashed lines included. */
function traceRange(row, values) {
  const n = S.data.n;
  const step = (S.data.t1 - S.data.t0) / n;
  const j0 = clamp(Math.floor((S.view[0] - S.data.t0) / step), 0, n);
  const j1 = clamp(Math.ceil((S.view[1] - S.data.t0) / step), 0, n);
  let [lo, hi] = [Math.min(Infinity, ...row.hlines), Math.max(-Infinity, ...row.hlines)];
  for (let p = 0; p < 2 * row.n_channels; p++) {
    for (let j = j0; j < j1; j++) {
      const v = values[p * n + j];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!(hi > lo)) [lo, hi] = Number.isFinite(lo) ? [lo - 1, lo + 1] : [0, 1];
  const pad = (hi - lo) * 0.06;
  return [lo - pad, hi + pad];
}

function drawTrace(g, row, values, [lo, hi], w, h) {
  const [n, channels] = [S.data.n, row.n_channels];
  const step = (S.data.t1 - S.data.t0) / n;
  const y = (v) => ((hi - v) / (hi - lo)) * (h - 1);
  g.lineWidth = 1;
  g.strokeStyle = T.muted;
  g.setLineDash([4, 3]);
  for (const v of row.hlines) {
    g.beginPath();
    g.moveTo(0, y(v));
    g.lineTo(w, y(v));
    g.stroke();
  }
  g.setLineDash([]);
  for (let c = 0; c < channels; c++) {
    // Each column's minimum then its maximum: an envelope zoomed out, a line zoomed in.
    g.strokeStyle = TRACE_COLOURS[c % TRACE_COLOURS.length];
    g.beginPath();
    let pen = false;
    for (let j = 0; j < n; j++) {
      const [low, high] = [values[c * n + j], values[(channels + c) * n + j]];
      if (!Number.isFinite(low)) {
        pen = false;
        continue;
      }
      const x = px(S.data.t0 + (j + 0.5) * step);
      if (pen) g.lineTo(x, y(low));
      else g.moveTo(x, y(low));
      g.lineTo(x, y(high));
      pen = true;
    }
    g.stroke();
  }
  if (row.legend.length < 2) return;
  g.font = FONT;
  g.textAlign = "right";
  let x = w - RIGHT - 6;
  for (let c = row.legend.length - 1; c >= 0; c--) {
    g.fillStyle = TRACE_COLOURS[c % TRACE_COLOURS.length];
    g.fillText(row.legend[c], x, 14);
    x -= g.measureText(row.legend[c]).width + 10;
  }
}

/** The label on a row: outside its window veiled, each span's edges and a bar on top. */
function drawOverlay(g, w, h) {
  const [lo, hi] = S.label.window;
  g.fillStyle = T.veil;
  g.fillRect(0, 0, px(lo), h);
  g.fillRect(px(hi), 0, w - px(hi), h);
  S.label.intervals.forEach(([a, b, c], i) => {
    const width = i === S.selected ? 2 : 1;
    g.fillStyle = categoryColour(c);
    g.fillRect(px(a), 0, px(b) - px(a), 3);
    g.fillRect(px(a) - width / 2, 0, width, h);
    g.fillRect(px(b) - width / 2, 0, width, h);
  });
}

function ticks(lo, hi, n) {
  const step = niceStep(hi - lo, n);
  if (!(step > 0 && Number.isFinite(step))) return [];
  const digits = Math.max(0, -Math.floor(Math.log10(step)));
  const text = (t) => (step >= 1e4 || step < 1e-3 ? t.toExponential(1) : t.toFixed(digits));
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) out.push([t, text(t)]);
  return out;
}

function drawGutter(g, row, range, h) {
  g.textAlign = "left";
  g.fillStyle = T.ink;
  g.font = `600 ${FONT}`;
  g.fillText(row.title, 8, 16, GUTTER - 44);
  g.font = FONT;
  g.fillStyle = T.muted;
  g.fillText(row.y_units, 8, 30, GUTTER - 44);
  if (!range) return;
  const [lo, hi] = range;
  g.textAlign = "right";
  for (const [t, text] of ticks(lo, hi, h / 40)) {
    const y = ((hi - t) / (hi - lo)) * (h - 1);
    if (y > 6 && y < h - 4) g.fillText(text, GUTTER - 6, y + 4);
  }
}

function drawAxis() {
  const canvas = $("axis-row");
  const g = context(canvas);
  g.font = FONT;
  g.fillStyle = T.muted;
  g.fillText("ms", 8, 16);
  g.textAlign = "center";
  for (const [t, text] of ticks(S.view[0], S.view[1], plotWidth() / 90)) {
    const x = px(t);
    if (x < GUTTER || x > canvas.clientWidth - RIGHT) continue;
    g.fillRect(x, 0, 1, 4);
    g.fillText(text, x, 16);
  }
}

/** A label as a track: its window lit, its spans filled; the editable one also shows
 * the window's edges in its foot strip and, along its top, where it leaves the source. */
function drawTrack(canvas, label, editable) {
  const g = context(canvas);
  const [w, h] = [canvas.clientWidth, canvas.clientHeight];
  if (!label) return;
  g.save();
  g.beginPath();
  g.rect(GUTTER, 0, w - GUTTER - RIGHT, h);
  g.clip();
  const [lo, hi] = label.window;
  const [top, foot] = editable ? [5, HANDLE_BAND + 2] : [4, 4];
  g.fillStyle = T.panel;
  g.fillRect(px(lo), 0, px(hi) - px(lo), h);
  label.intervals.forEach(([a, b, c], i) => {
    g.globalAlpha = editable ? 1 : 0.55;
    g.fillStyle = categoryColour(c);
    g.fillRect(px(a), top, Math.max(1, px(b) - px(a)), h - top - foot);
    g.globalAlpha = 1;
    if (editable && i === S.selected) {
      g.strokeStyle = T.ink;
      g.lineWidth = 2;
      g.strokeRect(px(a), top, px(b) - px(a), h - top - foot);
    }
  });
  if (editable) {
    g.fillStyle = T.ink;
    g.fillRect(px(lo), h - HANDLE_BAND / 2 - 0.5, px(hi) - px(lo), 1);
    for (const t of label.window) g.fillRect(px(t) - 2, h - HANDLE_BAND, 4, HANDLE_BAND);
    const source = S.meta.source || { window: label.window, intervals: [] };
    g.fillStyle = T.changed;
    for (const [a, b] of diffRuns(source, label)) g.fillRect(px(a), 0, Math.max(1, px(b) - px(a)), 3);
  }
  g.restore();
}

function showHeader() {
  const row = S.queue.find((r) => r.shot === S.shot) || {};
  const last = S.meta?.last_save;
  $("tier").textContent = row.tier || "";
  $("state").textContent = row.state || "";
  $("state").className = `pill ${row.state || ""}`;
  $("saved").textContent = last
    ? `saved ${new Date(last.saved_at).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}`
    : "";
  $("dirty").hidden = !dirty();
  const reviewed = S.queue.filter((row) => row.state !== "unreviewed").length;
  $("count").textContent = `${reviewed}/${S.queue.length}`;
}

function renderQueue() {
  const nav = $("queue");
  if (nav.children.length !== S.queue.length) {
    nav.replaceChildren(
      ...S.queue.map((row) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.dataset.shot = row.shot;
        chip.textContent = row.shot;
        return chip;
      })
    );
  }
  S.queue.forEach((row, i) => {
    const chip = nav.children[i];
    const marks = [row.state, row.shot === S.shot && "current", stored(draftKey(row.shot)) && "dirty"];
    chip.className = ["chip", ...marks.filter(Boolean)].join(" ");
    chip.title = row.tier;
  });
}

function renderSwatches() {
  $("swatches").replaceChildren(
    ...Object.entries(S.categories).map(([c, name]) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "swatch";
      button.textContent = `${c} ${name}`;
      button.style.setProperty("--c", categoryColour(Number(c)));
      button.setAttribute("aria-pressed", String(Number(c) === S.category));
      button.addEventListener("click", () => setCategory(Number(c)));
      return button;
    })
  );
}

// -- input

function startDrag(event, drag) {
  S.drag = { ...drag, base: S.label };
  event.currentTarget.setPointerCapture(event.pointerId);
  event.preventDefault();
}

function onRowsDown(event) {
  if (event.button !== 0 || !S.meta || event.target.tagName !== "CANVAS") return;
  if (event.shiftKey) startDrag(event, { kind: "new", from: timeAt(event.clientX) });
  else startDrag(event, { kind: "pan", x: event.clientX, view: S.view });
}

function onLabelDown(event) {
  if (event.button !== 0 || !S.meta) return;
  const rect = event.currentTarget.getBoundingClientRect();
  const hit = hitTest(S.label, event.clientX - rect.left, event.clientY - rect.top, rect.height, px);
  S.selected = hit.kind === "edge" || hit.kind === "move" ? hit.index : -1;
  startDrag(event, { ...hit, from: timeAt(event.clientX) });
  render();
}

function onLabelHover(event) {
  if (S.drag || !S.meta) return;
  const rect = event.currentTarget.getBoundingClientRect();
  const hit = hitTest(S.label, event.clientX - rect.left, event.clientY - rect.top, rect.height, px);
  const cursors = { edge: "ew-resize", window: "ew-resize", move: "grab", new: "crosshair" };
  event.currentTarget.style.cursor = cursors[hit.kind];
}

function dragTo(clientX) {
  const d = S.drag;
  const t = timeAt(clientX);
  if (d.kind === "pan") {
    const shift = ((clientX - d.x) / plotWidth()) * (d.view[1] - d.view[0]);
    return setView(d.view[0] - shift, d.view[1] - shift);
  }
  const edges = [...d.base.window];
  const spans = d.base.intervals.map((span) => [...span]);
  let moved = null;
  if (d.kind === "new") moved = [Math.min(d.from, t), Math.max(d.from, t), S.category];
  if (d.kind === "edge") {
    const span = spans.splice(d.index, 1)[0];
    span[d.edge] = t;
    moved = [Math.min(span[0], span[1]), Math.max(span[0], span[1]), span[2]];
  }
  if (d.kind === "move") {
    const [a, b, c] = spans.splice(d.index, 1)[0];
    moved = [a + t - d.from, b + t - d.from, c];
  }
  if (d.kind === "window") {
    const [lo, hi] = edges;
    edges[d.edge] = d.edge
      ? clamp(t, lo + 1, lo + LONGEST_WINDOW_MS)
      : clamp(t, hi - LONGEST_WINDOW_MS, hi - 1);
    // The window is the edge of what was looked at: spans are cut back to it.
    for (const span of spans) [span[0], span[1]] = [0, 1].map((k) => clamp(span[k], ...edges));
  }
  if (moved) spans.push(moved); // last, so it paints over what it crosses
  const next = normalise(edges, spans, known());
  if (!next) return;
  S.label = next;
  if (moved) selectAt((moved[0] + moved[1]) / 2);
  render();
}

function endDrag() {
  const d = S.drag;
  S.drag = null;
  if (!d || d.kind === "pan" || same(d.base, S.label)) return;
  keepForUndo(d.base);
  touch();
}

function onWheel(event, zoomAlways) {
  if (!S.meta) return;
  const scale = event.deltaMode === 1 ? 16 : 1; // lines, not px
  const [dx, dy] = [event.deltaX * scale, event.deltaY * scale];
  if (Math.abs(dx) > Math.abs(dy)) {
    event.preventDefault();
    return pan(dx / plotWidth());
  }
  if (!zoomAlways && !event.ctrlKey && !event.metaKey) return; // a plain wheel scrolls
  event.preventDefault();
  zoomAt(timeAt(event.clientX), Math.exp(dy * 0.002));
}

function showCursor(clientX) {
  const cursor = $("cursor");
  const axis = $("axis-row").getBoundingClientRect();
  const x = clientX - axis.left;
  cursor.hidden = !S.meta || x < GUTTER || x > axis.width - RIGHT;
  if (cursor.hidden) return;
  const top = $("top").getBoundingClientRect().top;
  const bottom = $("label-track").getBoundingClientRect().bottom;
  Object.assign(cursor.style, { left: `${clientX}px`, top: `${top}px`, height: `${bottom - top}px` });
  $("cursor-time").textContent = `${Math.round(timeAt(clientX))} ms`;
}

function toggleKeys() {
  const dialog = $("keys");
  if (dialog.open) dialog.close();
  else dialog.showModal();
}

const KEYS = {
  Enter: () => save(true),
  s: () => save(false),
  r: revert,
  j: () => openShot(neighbour(-1)),
  k: () => openShot(neighbour(1)),
  u: () => (nextUnreviewed() == null ? say("all reviewed") : openShot(nextUnreviewed())),
  "[": () => contrast(-16),
  "]": () => contrast(16),
  Escape: () => {
    S.selected = -1;
    render();
  },
  Delete: removeSelected,
  Backspace: removeSelected,
  ArrowLeft: () => pan(-0.1),
  ArrowRight: () => pan(0.1),
  "-": () => zoomAt(middle(), 1.25),
  "=": () => zoomAt(middle(), 0.8),
  "+": () => zoomAt(middle(), 0.8),
  0: fit,
};

const MOVES = new Set(["j", "k", "u"]); // the keys that work on a shot with nothing to show

function onKey(event) {
  const target = event.target;
  const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
  if (target.closest("input, select, textarea")) return;
  if (target.closest("button") && (key === "Enter" || key === " ")) return;
  if (key === "?" || $("keys").open) return key === "?" && toggleKeys();
  const mod = event.ctrlKey || event.metaKey;
  if (event.altKey || (mod && key !== "z") || (!S.meta && !MOVES.has(key))) return;
  if (mod) {
    event.preventDefault();
    return undo();
  }
  if (/^[1-9]$/.test(key)) return setCategory(Number(key));
  if (KEYS[key]) {
    event.preventDefault();
    KEYS[key]();
  }
}

function wire() {
  const top = $("top");
  const tracks = [$("source-track"), $("label-track")];
  top.addEventListener("pointerdown", onRowsDown);
  top.addEventListener("wheel", (event) => onWheel(event, event.target.id === "axis-row"), {
    passive: false,
  });
  for (const track of tracks) {
    track.addEventListener("wheel", (event) => onWheel(event, true), { passive: false });
  }
  for (const target of [top, ...tracks]) target.addEventListener("dblclick", () => S.meta && fit());
  tracks[0].addEventListener("pointerdown", (event) => {
    if (event.button === 0 && S.meta) startDrag(event, { kind: "pan", x: event.clientX, view: S.view });
  });
  tracks[1].addEventListener("pointerdown", onLabelDown);
  tracks[1].addEventListener("pointermove", onLabelHover);
  window.addEventListener("pointermove", (event) => {
    if (S.drag) dragTo(event.clientX);
    if (S.drag || event.target.closest?.("#top, .track")) showCursor(event.clientX);
    else $("cursor").hidden = true;
  });
  window.addEventListener("hashchange", followHash);
  document.documentElement.addEventListener("pointerleave", () => ($("cursor").hidden = true));
  window.addEventListener("pointerup", endDrag);
  window.addEventListener("pointercancel", endDrag);
  document.addEventListener("keydown", onKey);
  $("event").addEventListener("change", () =>
    openEvent($("event").value, null).catch((error) => say(error.message, true))
  );
  $("shot").addEventListener("keydown", (event) => {
    const typed = $("shot").value.trim();
    if (event.key === "Enter" && /^\d+$/.test(typed)) openShot(Number(typed));
  });
  $("queue").addEventListener("click", (event) => {
    const chip = event.target.closest(".chip");
    if (chip) openShot(Number(chip.dataset.shot));
  });
  $("save-next").addEventListener("click", () => save(true));
  $("revert").addEventListener("click", () => S.meta && revert());
  $("help").addEventListener("click", toggleKeys);
  new ResizeObserver(() => {
    sizeCanvases();
    render();
    fetchRows();
  }).observe(top);
}

if (typeof module !== "undefined") {
  module.exports = { ms, paint, runs, normalise, diffRuns, niceStep, hitTest, lut };
} else {
  boot();
}
```

- [ ] **Step 4: Run** the same three files: `59 passed`, and `-rs` shows no skip (the browser
  test must have run; if it skipped, say why in the report).

- [ ] **Step 5: Commit** `labeler review: the split-screen canvas page`.

### Task A9: Docs

**Files:**
- Create: `docs/labeler/review.md`
- Modify: `docs/reference/environment-variables.md`

- [ ] **Step 1: Write them.**

```bash
python3 .claude/superpowers/plans/extract_blocks.py \
  .claude/superpowers/plans/2026-09-22-label-review-v4-only-pixi.md \
  docs/labeler/review.md docs/reference/environment-variables.md
```

Write `docs/labeler/review.md`:

````markdown
---
title: "Label review"
sidebar_position: 2
---

# Label review

A browser page for checking an event's labels one shot at a time. The
diagnostic rows fill the top of the page on one time axis. The shot's source
label and your label sit below them, with the whole roster as a row of chips.
Each shot has one label. You change it, save it, and it is still there when you
come back.

Code: `src/labeler/events/review/` (labels and rows) and
`src/labeler/events/ui/` (server and page). Tests: `tests/labeler/test_review_*.py`
and `tests/labeler/test_events_ui.py`.

## Opening it

```bash
pixi run -e labelmaker labeler-verify
```

This prints a link carrying a fresh token and an `ssh -L 8811:localhost:8811 <node>`
line. Run the `ssh` line on your laptop, then open the link there. The server
listens on the loopback only. The first visit trades the token for a cookie, so
reloads and `#<event>/<shot>` links work until the server restarts. `--port`
and `--token` change either one.

The page opens on the event you last had open, at the first unreviewed shot
after the newest save.

## The page

- **Rows.** Alfvén eigenmode shots show the CO2 interferometer: the power of
  chords R0, V1, V2 and V3, then the cross-power of R0 with each of the other
  three. A mode seen by several chords shows in all seven rows. Noise on one
  chord shows only in that chord's power row. Colour is dB above each
  frequency's own quiet level (its median over 0-6 s), from -3 to 27 dB. Other
  events show the rows of their panel builder.
- **Source** is the label the event's newest `format/*_format_*.csv` gives the
  shot. **Label** is yours; it starts as a copy of the source. Where the two
  differ, a strip along the top of the label track marks the difference.
- **Chips**, one per roster shot, coloured by state: *unreviewed* (never saved),
  *confirmed* (saved as the source had it) or *changed*. A dot marks a shot with
  unsaved edits. Unsaved edits live in the browser until you save or revert, so
  a reload keeps them.

| Key or gesture | Does |
|---|---|
| Drag on empty label track | add a span |
| Drag a span, or its edge | move it, or resize it |
| Drag the foot of the label track | move the window's edges |
| Shift-drag on the rows | add a span |
| Drag on the rows | pan |
| Wheel | scroll the rows |
| Ctrl/⌘ + wheel, or any wheel over the axis or tracks | zoom at the cursor |
| Double-click, `0` | fit the window |
| `←` `→`, `-` `=` | pan, zoom |
| `1`-`9` | category for new spans and the selected one |
| `Delete` | remove the selected span |
| Ctrl/⌘ + `Z` | undo |
| `Enter` | save and open the next unreviewed shot |
| `S` / `R` | save / revert to the source |
| `J` `K` / `U` | previous, next shot / next unreviewed |
| `[` `]` | contrast |
| `?` | this list |

## What a save writes

Saves go under the event's directory in the label tables
(`data/events/<event>/` unless `LABELER_LABEL_TABLES` says otherwise):

- `review/labels.csv` holds the current label of every reviewed shot, in the
  format-table schema (`shot, category, t_start, t_end, confidence`, whole ms).
  A shot's rows tile its window, and the gaps are category 0. Each save replaces
  that shot's rows and rewrites the file atomically. The result validates like
  any other format table.
- `review/history.jsonl` gets one line per save: shot, reviewer, time, the
  window and spans saved, and the source file they were compared with.

## The row store

The page reads rows from `$LABELER_ROOT/spectrograms/<event>/<shot>.h5`. Each
file holds one time grid. Every row is stored at full resolution and again
max-pooled by 8 and by 64, so a zoomed-out view reads a small slice and a
zoomed-in one reads the full columns. The Alfvén eigenmode rows are built ahead
of time, because each shot takes about 8 s:

```bash
sbatch scripts/labeler/spectrograms.sbatch          # every roster shot; skips built ones
python -m labeler.events.review.build --event alfven_eigenmode --shots 170659 170660
```

The AE recipe resamples the four chords to 500 kHz and applies a Hann STFT of
512 samples every 128. That gives 0.256 ms columns and 257 bins up to 250 kHz.
The cross-power rows average R0 × conj(chord) over 8 columns before taking the
magnitude. Other events build their file the first time a shot is opened, which
takes a few seconds. The page says so while it waits.

A shot's CO2 comes from the corpus if the corpus has it, else from the fetch
cache `$LABELER_ROOT/raw`. Failing both, it is fetched live, which needs the
`fdp` wrapper and a login node. All 180 AE roster shots are in the fetch cache.
````

Patch `docs/reference/environment-variables.md`:

```diff
diff --git a/docs/reference/environment-variables.md b/docs/reference/environment-variables.md
index ee40aac..40fe593 100644
--- a/docs/reference/environment-variables.md
+++ b/docs/reference/environment-variables.md
@@ -15,7 +15,7 @@ wires its own values in.
 | Variable | Purpose | Stellar value | Frontier value |
 |---|---|---|---|
 | `SHOT_DESIGN_DATA_ROOT` | the `shot_design` workstream root (database, frame codes, caches, outputs) | `/scratch/gpfs/EKOLEMEN/nc1514/ideate` | `/lustre/orion/fus187/proj-shared/nchen/shot_design` |
-| `LABELER_ROOT` | the `labeler` workstream root (detector features, masks, labels, models) | `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker` | `/lustre/orion/fus187/proj-shared/nchen/labeler` |
+| `LABELER_ROOT` | the `labeler` workstream root (detector features, masks, labels, models, the review page's rows in `spectrograms/`, the fetch cache in `raw/`) | `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker` | `/lustre/orion/fus187/proj-shared/nchen/labeler` |
 | `SHOT_DESIGN_CORPUS` | the read-only DIII-D per-shot HDF5 corpus | `/scratch/gpfs/EKOLEMEN/foundation_model` | `/lustre/orion/fus187/proj-shared/foundation_model` |
 
 An explicit `SHOT_DESIGN_DATA_ROOT` always wins over whatever a paths file
@@ -38,8 +38,9 @@ for the paths-file convention itself.
 
 | Variable | Fallback for |
 |---|---|
-| `LABELER_CORPUS`, `LABELER_LOGS_JSONL`, `LABELER_RAW_CACHE`, `LABELER_TEXT_ROOT` | narrower overrides of paths that would otherwise come from `LABELER_ROOT` / the paths file |
-| `LABELER_LABEL_TABLES` | overrides which label tables `labeler` reads |
+| `LABELER_CORPUS`, `LABELER_LOGS_JSONL`, `LABELER_TEXT_ROOT` | narrower overrides of paths that would otherwise come from `LABELER_ROOT` / the paths file |
+| `LABELER_RAW_CACHE` | where fetched raw signals park (default `$LABELER_ROOT/raw`) |
+| `LABELER_LABEL_TABLES` | the label-table directory (default `data/events` in the checkout): what `labeler` reads, and where the review page saves (`<event>/review/`) |
 | `SHOT_DESIGN_TEXT_ROOT` | overrides the text-corpus root independent of `SHOT_DESIGN_CORPUS` |
 
 Legacy `IDEATE_*`/`LABELMAKER_*` names (from before the 2026-09-15 package
```

- [ ] **Step 2: Commit** `docs: the label review page`.

### Part A acceptance

- [ ] The whole labeler suite, from the worktree root:
  `PYTHONPATH=$PWD/src pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m pytest tests/labeler -q -W error -rs`.
  Expect `2021 passed, 3 skipped`. The skips are
  `test_l14perf_real_identity.py::test_real_shot_cpu_output_identity` (it needs
  `L14PERF_REAL_ROOT`) and the two `live` tests in `test_resolve_fdp.py` (opt-in). And the
  shot_design suite, which imports `labeler.config`, with `-e shot-design-cpu ... tests/shot_design`:
  `1834 passed, 2 failed, 3 skipped`, as at the base commit. The two failures are Part 0's to fix:
  `test_ignite_v4.py::test_model_cfg_declares_fifteen_v4_modalities` (the conftest pin) and
  `test_seed.py::test_encode_frame_codes_writes_the_shipped_dict_structure` (the shipped cache).
- [ ] Ruff is clean on every touched Python file.
- [ ] `git grep -nE 'GUIDANCE|guidance\(|plotly|/vendor/|api/panels|api/save|api/progress' -- src/labeler/events/ui src/labeler/events/panels`
  prints nothing.
- [ ] No production writes, no `sbatch`, no pixi install/lock/update, no `pyproject.toml` or
  `pixi.lock` change, nothing under `data/events/`. `git diff --stat <base>..HEAD` lists only
  the Part A files in the file map, and the report.
- [ ] `.superpowers/sdd/task-A-report.md`, force-added. It holds the before and after counts you
  saw for each task, the full-suite summary with its skips, and whether the browser test ran. It
  also lists anything that differed from this plan.

---

## Part E -- pixi (the session, after Part A merges)

In the main checkout on `nathan_dev`, after `label-review` is merged: A1 deletes the module the
`labeler-raw` task runs, and E4 edits `raw.py` on top of A1. Every block below was checked on
2026-09-22 against a scratch copy of the post-A tree. The manifest's parsed content differs from
HEAD's only in the tasks and the seven activation values. `pixi shell-hook` on it prints the
defaults, `pixi task list` shows nine described tasks, and `pixi.lock` did not change.
`tests/shot_design/test_docs_scratch_db.py` failed 5 before the change and passed 5 after.

What it rests on, measured with pixi 0.76.1 on stellar-vis2 the same day:

- `${VAR:-default}` in an activation env is left to the shell, by `pixi run` and by
  `pixi shell-hook` alike: a value the caller exported wins, and an unset one takes the default.
- When two features set one variable, the feature listed first in the environment wins.
- A task's `env` beats the caller's environment, so `labeler-test` sets no `PYTHONPATH` (it would
  override a worktree's).
- `pixi run --locked` refuses every environment at HEAD: `torchvision`'s index (`[project]` lists
  it bare, the lock has it from the pytorch index). Every pixi command needs `--frozen`.
- `pixi run -e labelmaker fdp run printenv PTDATA_LIBRARY default_tree_path` prints both, with no
  prompt.

### E0: the incident class, before

- [ ] `SHOT_DESIGN_DATA_ROOT=/tmp/e0 pixi run --frozen --no-install -e shot-design-cpu printenv SHOT_DESIGN_DATA_ROOT`
  prints `/scratch/gpfs/EKOLEMEN/nc1514/ideate`, not `/tmp/e0`.

### E1-E3: roots as defaults, described tasks, the rationale in a doc

- [ ] **The failing test.** It pins every root in both shot_design activation blocks as a
  default, and the overview section that says so.

```bash
python3 .claude/superpowers/plans/extract_blocks.py .claude/superpowers/plans/2026-09-22-label-review-v4-only-pixi.md \
  tests/shot_design/test_docs_scratch_db.py
```

Write `tests/shot_design/test_docs_scratch_db.py`:

```python
"""The docs state the three things the 2026-09-14 incident turned into code.

A documented escape hatch that has drifted from the code is worse than none: it reads as
authority. So each assertion here is tied to the thing it documents -- the origin labels
`config.data_root_origin` actually returns, the roots `pyproject.toml` actually defaults, and
the flag the guard actually names.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from shot_design import config

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs" / "shot-design" / "overview.md"
HEADING = "## Scratch databases and the data root"
ROOTS = ("SHOT_DESIGN_DATA_ROOT", "LABELER_ROOT", "SHOT_DESIGN_CORPUS")


def section() -> str:
    body = DOCS.read_text(encoding="utf-8")
    assert HEADING in body, f"{DOCS} has no {HEADING!r} section"
    after = body.split(HEADING, 1)[1]
    return re.split(r"^## ", after, maxsplit=1, flags=re.MULTILINE)[0]


def activation(feature: str) -> dict[str, str]:
    manifest = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    return manifest["tool"]["pixi"]["feature"][feature]["target"]["unix"]["activation"]["env"]


def test_every_root_the_manifest_sets_is_a_default():
    """A plain value would replace a root the caller exported, which is what once replaced the
    production database with a one-shot one."""
    for feature in ("shot-design", "shot-design-frontier"):
        env = activation(feature)
        for name in (*ROOTS, "SHOT_DESIGN_PATHS"):
            if name in env:
                assert env[name].startswith("${" + name + ":-"), (feature, name, env[name])
    assert set(ROOTS) <= set(activation("shot-design"))


def test_the_section_names_every_root_the_shot_design_features_default():
    text = section()
    for name in ROOTS:
        assert name in text, f"the docs section does not name {name}"


def test_the_section_gives_the_ways_to_build_a_scratch_database():
    text = section()
    assert "SHOT_DESIGN_DATA_ROOT=/tmp/scratch-db pixi run --frozen -e shot-design-cpu" in text
    assert ".pixi/envs/shot-design-cpu/bin/python -m shot_design" in text
    assert "SHOT_DESIGN_PATHS=" in text


def test_the_section_quotes_the_origin_labels_the_code_prints(monkeypatch):
    text = section()
    # Not a paraphrase: each label is what `data_root_origin` returns for that precedence.
    monkeypatch.setenv("SHOT_DESIGN_DATA_ROOT", "/tmp/whatever")
    assert config.data_root_origin() in text
    monkeypatch.delenv("SHOT_DESIGN_DATA_ROOT")
    monkeypatch.delenv("SHOT_DESIGN_PATHS", raising=False)
    # The third label names the resolved paths file in full -- an absolute path, and so specific
    # to the checkout -- because `SHOT_DESIGN_CONFIG_DIR` can move it. The docs quote the repo-relative
    # path it is in a plain checkout, which is that label minus the repo root.
    label = config.data_root_origin()
    assert label.endswith(" default"), label
    resolved = Path(label.removesuffix(" default"))
    assert f"{resolved.relative_to(REPO)} default" in text


def test_the_section_says_what_the_publish_guard_refuses_and_how_to_override_it():
    text = section()
    assert "--force" in text
    assert "shot_source" in text and "n_shots" in text
```

- [ ] Run it (`-e shot-design-cpu`): `5 failed`, four on the missing
  `## Scratch databases and the data root` heading and one on the plain root values.

- [ ] **The manifest and its doc.** The manifest's pixi sections keep one comment line per block.
  Their measurements move to `docs/reference/pixi-environments.md` word for word, apart from the
  plotly and fastapi notes that described the old page.

```bash
python3 .claude/superpowers/plans/extract_blocks.py .claude/superpowers/plans/2026-09-22-label-review-v4-only-pixi.md \
  pyproject.toml docs/reference/pixi-environments.md
```

Write `pyproject.toml`:

```toml
[project]
name = "faith"
requires-python = ">= 3.11"
authors = [
  { name = "Azarakhsh Jalalvand", email = "azarakhsh.jalalvand@princeton.edu" },
  { name = "Kouroche Bouchiat", email = "bouchiat@princeton.edu" },
  { name = "Nathaniel Chen", email = "nathaniel@princeton.edu" },
  { name = "Peter Steiner", email = "peter.steiner@princeton.edu" },
]
dependencies = [
  "einops>=0.8.2,<0.9",
  "h5py>=3.15.1,<4",
  # imageio + imageio-ffmpeg used by scripts/training/eval_e2e_stage1_phase3_1_video.py
  # to write video stitched plots as mp4. imageio-ffmpeg ships a static
  # ffmpeg binary, so the encoder works on Frontier without an OS-level
  # ffmpeg package. Listed in [project] (shared across default / fdp /
  # frontier features) so every env carries the same encoder.
  "imageio>=2.30,<3",
  "imageio-ffmpeg>=0.4.9,<1",
  "ipykernel>=7.2.0,<8",
  "ipywidgets>=8.1.8,<9",
  "matplotlib>=3.10.8,<4",
  "numpy>=1.26.4,<3",
  # Image-processing libs for tokamak-animation cam ↔ PNG alignment
  # (cv2.findTransformECC + skimage.registration).
  "opencv-python-headless>=4.10,<5",
  "scikit-image>=0.24,<0.26",
  "pandas>=3.0.0,<4",
  "scipy",
  "tables>=3.10.2,<4",
  "torch",
  "torchmetrics>=1.9.0,<2",
  "torchinfo>=1.8.0,<2",
  "torchvision",
  "transformers>=5.1.0,<6",
  "pytest>=9.0.2,<10",
  "tensorboard>=2.20.0,<3",
  "wandb>=0.25.1,<0.26",
  "hydra-core", "vector-quantize-pytorch>=1.31.0,<2", "x-transformers>=2.23.5,<3",
]
dynamic = ["version"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.version]
path = "src/faith/__init__.py"

[tool.hatch.build.targets.wheel]
# Four top-level packages live under src/. Without this list hatchling
# autodiscovers only the one matching the project name (faith), so
# `import labeler` (and now `import shot_design`) fails in every pixi env.
packages = ["src/faith", "src/tokamak_foundation_model", "src/labeler", "src/shot_design"]

# Why each pixi block below is the way it is: docs/reference/pixi-environments.md.
[tool.pixi.workspace]
channels = ["conda-forge", "ga-fdp"]
platforms = ["linux-64", "win-64", "osx-arm64"]

[tool.pixi.pypi-dependencies]
faith = { path = ".", editable = true }

[tool.pixi.target.osx.pypi-dependencies]
# No CUDA support on macOS, use `cpu` wheels.
torch = { version = ">=2.5.1", index = "https://download.pytorch.org/whl/cpu" }
torchvision = { version = ">=0.20.1", index = "https://download.pytorch.org/whl/cpu" }

[tool.pixi.feature.cuda]
platforms = ["linux-64", "win-64"]

[tool.pixi.feature.cuda.pypi-dependencies]
torch = { version = ">=2.5.1,<2.11", index = "https://download.pytorch.org/whl/cu124" }
torchvision = { version = ">=0.20.1,<0.22", index = "https://download.pytorch.org/whl/cu124" }

[tool.ruff]
line-length = 88

[tool.pixi.tasks]

[tool.pixi.dependencies]
python = ">=3.11,<3.12"
hydra-core = ">=1.3.2,<2"
line_profiler = ">=5.0.2,<6"

[tool.pixi.feature.fdp]
platforms = ["linux-64"]

# torch binds the system libstdc++, which lacks GLIBCXX_3.4.29; the env's own copy goes first.
[tool.pixi.feature.fdp.target.unix.activation.env]
LD_LIBRARY_PATH = "$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

[tool.pixi.feature.fdp.dependencies]
# The working fetch env's versions; `fdp` is required, toksearch_d3d imports it.
fdp = { version = ">=0.5.1,<0.6", channel = "ga-fdp" }
toksearch = { version = ">=2.8.1,<3", channel = "ga-fdp" }
toksearch_d3d = { version = ">=0.10.1,<0.11", channel = "ga-fdp" }

[tool.pixi.feature.labelmaker]
platforms = ["linux-64"]

[tool.pixi.feature.labelmaker.tasks]
label = { cmd = "bash scripts/labeler/label_shot.sh", description = "Every labeler model on the given shots: labels, a summary and a figure per shot" }
labeler-verify = { cmd = "fdp run python -m labeler.events.ui", description = "The label review page, under the fdp wrapper so any shot can be fetched" }
labeler-spectrograms = { cmd = "python -m labeler.events.review.build", description = "Build the review page's row store: --event EVENT [--shots ...] [--force]" }
labeler-test = { cmd = "python -m pytest tests/labeler -q -W error", description = "The labeler suite, in the checkout that holds this manifest" }

[tool.pixi.feature.labelmaker.dependencies]
# pandas 3 writes parquet only through pyarrow.
pyarrow = ">=17,<22"
# Model cards' front matter; declared rather than left to arrive through hydra-core.
pyyaml = ">=6,<7"
# conda-forge's protobuf 7 cannot satisfy wandb's protobuf<7.
protobuf = "<7"
# The review page's server; the same ranges as feature.shot-design.
fastapi = ">=0.115,<1"
uvicorn = ">=0.30,<1"

[tool.pixi.feature.labelmaker.pypi-dependencies]
# CPU wheels: labeler never uses a GPU.
torch = { version = ">=2.5.1", index = "https://download.pytorch.org/whl/cpu" }
torchvision = { version = ">=0.20.1", index = "https://download.pytorch.org/whl/cpu" }
# The verification notebooks' figures (verify.py) and their widget.
plotly = ">=6.9,<7"
anywidget = ">=0.9,<1"

[tool.pixi.feature.shot-design]
platforms = ["linux-64"]

[tool.pixi.feature.shot-design.tasks]
shot_design = { cmd = "python -m shot_design", description = "The shot analysis and experiment-design CLI" }
shot_design-mcp = { cmd = "python -m shot_design.mcp", description = "The MCP server, stdio transport" }
shot_design-test = { cmd = "python -m pytest tests/shot_design -q -W error", description = "The shot_design suite, in the checkout that holds this manifest" }

[tool.pixi.feature.shot-design.dependencies]
# Ranges checked against shot-recommender-system's uv.lock; all conda but sentence-transformers.
pyarrow = ">=17,<22"
pyyaml = ">=6,<7"
# >=2.12 is conda-forge mcp's own floor.
pydantic = ">=2.12,<3"
scipy = "*"
scikit-learn = ">=1.5,<2"
h5py = ">=3.15,<4"
tqdm = ">=4.66,<5"
pypdf = ">=4,<7"
# The log bundles must stay byte-identical to d3dlogfetching's, so its parser.
beautifulsoup4 = ">=4.12,<5"
httpx = ">=0.27,<1"
fastapi = ">=0.115,<1"
uvicorn = ">=0.30,<1"
# 2.x only: FastMCP, the 1.x entry point, is gone.
mcp = ">=2.1,<3"
pytest = ">=9,<10"
# The same protobuf pin as feature.labelmaker.
protobuf = "<7"
duckdb = ">=1.1,<2"

[tool.pixi.feature.shot-design.pypi-dependencies]
# From pypi, so that torch stays on the pypi side.
sentence-transformers = ">=6,<7"

# Stellar roots as defaults: a root the caller exports wins. Offline, or the MiniLM load hangs.
[tool.pixi.feature.shot-design.target.unix.activation.env]
SHOT_DESIGN_DATA_ROOT = "${SHOT_DESIGN_DATA_ROOT:-/scratch/gpfs/EKOLEMEN/nc1514/ideate}"
LABELER_ROOT = "${LABELER_ROOT:-/scratch/gpfs/EKOLEMEN/nc1514/labelmaker}"
SHOT_DESIGN_CORPUS = "${SHOT_DESIGN_CORPUS:-/scratch/gpfs/EKOLEMEN/foundation_model}"
HF_HUB_OFFLINE = "1"
TOKENIZERS_PARALLELISM = "false"

[tool.pixi.feature.shot-design-cpu]
platforms = ["linux-64"]

[tool.pixi.feature.shot-design-cpu.pypi-dependencies]
# The CPU twin of `shot-design`: only design_rollout needs a GPU.
torch = { version = ">=2.5.1", index = "https://download.pytorch.org/whl/cpu" }
torchvision = { version = ">=0.20.1", index = "https://download.pytorch.org/whl/cpu" }

[tool.pixi.feature.frontier]
platforms = ["linux-64"]

[tool.pixi.feature.frontier.dependencies]
# pip for the setup-flash-attn task; ninja for aiter's build at the first `import flash_attn`.
pip = "*"
ninja = "*"

[tool.pixi.feature.frontier.pypi-dependencies]
# rocm7.1 ships torch 2.10 only; uv does not find triton-rocm unless it is listed.
torch       = { version = ">=2.10,<2.11",   index = "https://download.pytorch.org/whl/rocm7.1" }
torchvision = { version = ">=0.25,<0.27",   index = "https://download.pytorch.org/whl/rocm7.1" }
triton-rocm = { version = "*",              index = "https://download.pytorch.org/whl/rocm7.1" }
# Evaluation suite (scripts/evaluation/): latent-space analyses + probes.
scikit-learn = ">=1.5,<2"
umap-learn   = ">=0.5.7,<0.6"
# flash-attn comes from the setup-flash-attn task: its build needs `module load rocm/7.1.1`.

[tool.pixi.feature.frontier.tasks]
setup-flash-attn = { cmd = "bash scripts/slurm_frontier/setup_frontier_env.sh", description = "Build & install flash-attn 2 into the frontier pixi env on a Frontier compute node (gfx90a). Auto-salloc's if run from a login node." }
verify-flash-attn = { cmd = "python scripts/slurm_frontier/verify_flash_attn.py", description = "Smoke-test flash_attn on the local MI250X." }

# `shot-design` on Frontier: feature.shot-design's conda set plus ROCm torch.
[tool.pixi.feature.shot-design-frontier]
platforms = ["linux-64"]

[tool.pixi.feature.shot-design-frontier.pypi-dependencies]
# The same rocm7.1 pins as feature.frontier.
torch       = { version = ">=2.10,<2.11", index = "https://download.pytorch.org/whl/rocm7.1" }
torchvision = { version = ">=0.25,<0.27", index = "https://download.pytorch.org/whl/rocm7.1" }
triton-rocm = { version = "*",            index = "https://download.pytorch.org/whl/rocm7.1" }
x-transformers = "*"
vector-quantize-pytorch = "*"
einops = "*"
loguru = "*"
# tests/labeler imports plotly; Frontier has one environment for both packages.
plotly = ">=6.9,<7"

# Frontier roots and paths file as defaults; this feature is listed first, and the first wins.
[tool.pixi.feature.shot-design-frontier.target.unix.activation.env]
SHOT_DESIGN_PATHS     = "${SHOT_DESIGN_PATHS:-/lustre/orion/fus187/scratch/nchen/FusionAIHub/configs/shot_design/paths.frontier.yaml}"
SHOT_DESIGN_DATA_ROOT = "${SHOT_DESIGN_DATA_ROOT:-/lustre/orion/fus187/proj-shared/nchen/shot_design}"
LABELER_ROOT          = "${LABELER_ROOT:-/lustre/orion/fus187/proj-shared/nchen/labeler}"
SHOT_DESIGN_CORPUS    = "${SHOT_DESIGN_CORPUS:-/lustre/orion/fus187/proj-shared/foundation_model}"
HF_HUB_OFFLINE = "1"
TOKENIZERS_PARALLELISM = "false"
HDF5_USE_FILE_LOCKING = "FALSE"

# Dashes: pixi rejects underscores in an environment name.
[tool.pixi.environments]
default = ["cuda"]
fdp = ["fdp", "cuda"]
frontier = ["frontier"]
labelmaker = ["labelmaker", "fdp"]
shot-design = ["shot-design", "cuda"]
shot-design-cpu = ["shot-design", "shot-design-cpu"]
# First: the first feature listed wins the activation env.
shot-design-frontier = ["shot-design-frontier", "shot-design"]

[tool.pytest.ini_options]
markers = ["real_data: requires the read-only EKOLEMEN stores and/or local model weights"]
```

Write `docs/reference/pixi-environments.md`:

````markdown
---
title: Pixi environments
sidebar_position: 4
---

# Pixi environments

`pyproject.toml` defines seven environments. Each block there has a one-line
comment. This page keeps the measurements behind those lines.

| Environment | Features | For |
|---|---|---|
| `default` | `cuda` | the model code, cu124 torch |
| `fdp` | `fdp`, `cuda` | direct DIII-D data access (fdp, toksearch) |
| `frontier` | `frontier` | Frontier training, ROCm 7.1 torch |
| `labelmaker` | `labelmaker`, `fdp` | `labeler`, CPU torch |
| `shot-design` | `shot-design`, `cuda` | `shot_design` with a GPU (IGNITE rollouts) |
| `shot-design-cpu` | `shot-design`, `shot-design-cpu` | `shot_design` on a CPU |
| `shot-design-frontier` | `shot-design-frontier`, `shot-design` | `shot_design` and `labeler` on Frontier |

| Task | Environment | Runs |
|---|---|---|
| `label` | `labelmaker` | `scripts/labeler/label_shot.sh`: every labeler model on the given shots |
| `labeler-verify` | `labelmaker` | the [label review page](../labeler/review.md), under `fdp run` |
| `labeler-spectrograms` | `labelmaker` | `python -m labeler.events.review.build`, the review page's row store |
| `labeler-test` | `labelmaker` | `python -m pytest tests/labeler -q -W error` |
| `shot_design` | the `shot-design` three | `python -m shot_design` |
| `shot_design-mcp` | the `shot-design` three | the MCP server, stdio |
| `shot_design-test` | the `shot-design` three | `python -m pytest tests/shot_design -q -W error` |
| `setup-flash-attn`, `verify-flash-attn` | `frontier` | build flash-attn 2 on a compute node, then smoke-test it |

A task runs in the checkout that holds the manifest. To test a worktree, run the
suite in the worktree itself:
`PYTHONPATH=$PWD/src pixi run --frozen --no-install --manifest-path <main checkout>/pyproject.toml -e labelmaker python -m pytest tests/labeler -q -W error`.

## Always `--frozen`

Measured 2026-09-22 with pixi 0.76.1: `pixi run --locked` refuses every
environment, "lock file not up-to-date with the workspace". The reason is
`torchvision`'s index. `[project]` lists it bare, so pypi.org, while the lock
holds it from the pytorch index each feature pins. Without `--frozen`, pixi
re-solves before it runs anything. Re-locking would clear this, but it changes
every environment, so it waits for its own change.

## Roots are defaults

The activation blocks set the roots as `${VAR:-value}`: `SHOT_DESIGN_DATA_ROOT`,
`LABELER_ROOT` and `SHOT_DESIGN_CORPUS`, plus `SHOT_DESIGN_PATHS` on Frontier.
A root you export before `pixi run` is the one the command sees. Unset, the
cluster's production root applies. Measured 2026-09-22 with pixi 0.76.1: `pixi run`
and `pixi shell-hook` both leave the default to the shell (`shell-hook` prints
`export SHOT_DESIGN_DATA_ROOT="${SHOT_DESIGN_DATA_ROOT:-...}"`). So

```bash
SHOT_DESIGN_DATA_ROOT=/tmp/x pixi run --frozen -e shot-design-cpu printenv SHOT_DESIGN_DATA_ROOT
```

prints `/tmp/x`, and with the variable unset it prints the production root.

Until 2026-09-22 the blocks set plain values. Activation runs after your shell,
so a value you exported was replaced without a word. On 2026-09-14 a one-shot
scratch build, run through `pixi run` with `SHOT_DESIGN_DATA_ROOT` exported to a
`/tmp` directory, published itself over the 500-shot production database. Every
`shot_design` command that writes still prints the root it resolved before its
first write ([Shot design](../shot-design/overview.md#scratch-databases-and-the-data-root)).

The roots are absolute paths rather than repo-relative ones: the corpus and the
label store live outside this checkout, on /scratch/gpfs/EKOLEMEN, and the same
environment is used from a login node, a SLURM job and an MCP client launched by
an assistant, none of which share a working directory.

`HF_HUB_OFFLINE`: sentence-transformers reaches into huggingface_hub on every
model load even when the MiniLM checkpoint is already cached, and on a compute
node with no outbound route that call does not fail - it HANGS, for minutes,
inside httpx.connect_tcp (measured in shotrec, see the same note at the top of
its cli.py). The checkpoint is in ~/.cache/huggingface/hub, so offline is the
correct and much faster mode.

`TOKENIZERS_PARALLELISM`: the corpus build forks worker processes after a
tokenizer has been used; without this, every fork prints the HuggingFace
deadlock warning and the fast tokenizer disables its own threads anyway.

## Activation order on Frontier

feature.shot-design's own activation block sets the three roots to their STELLAR
values, so one of the two blocks has to win. MEASURED 2026-09-19 with pixi 0.73:
the FIRST feature listed for the environment wins, not the last -- with
`shot-design-frontier = ["shot-design", "shot-design-frontier"]` the environment
came up with `/scratch/gpfs/EKOLEMEN/nc1514/ideate`, the Stellar root. That is why
[tool.pixi.environments] lists `shot-design-frontier` FIRST; reordering it
silently sends every job at the Stellar paths, which do not exist there.
Measured again 2026-09-22 with pixi 0.76.1 on a two-feature manifest: still the
first, and with defaults the first feature's default is the one used.

SHOT_DESIGN_PATHS selects the Frontier paths file
(configs/shot_design/paths.frontier.yaml); feature.shot-design does not set it,
because on Stellar the default paths.yaml is already the right file.

HDF5_USE_FILE_LOCKING: the corpus lives on Lustre, where h5py's default locking
fails to open a read-only file the way it does on GPFS.

## feature.fdp

MEASURED 2026-09-04 (Task 16b): `import torch` binds the SYSTEM
`/lib64/libstdc++.so.6`, which lacks `GLIBCXX_3.4.29`. Every compiled extension
in this feature that needs that symbol then fails to import after torch does -
`toksearch_d3d` (via `fdp` -> `pyxrootd`) and `scipy` - with a bare
`ImportError`, which `resolve_fdp.available()` swallowed into an
indistinguishable `ToksearchUnavailable` and silently disabled the whole fdp
scaling path. The pixi env ships its OWN `lib/libstdc++.so.6.0.36`, which DOES
have the symbol; it is just not first on the loader's path. `LD_LIBRARY_PATH`
(rather than `LD_PRELOAD`) is sufficient: verified that with it set, `torch`
followed by `fdp`, `toksearch`, `toksearch_d3d` and `scipy` all import, which
means torch's extensions carry `DT_RUNPATH` (which `LD_LIBRARY_PATH` overrides),
not `DT_RPATH` (which it would not). Scoped to this feature - not the shared
workspace - because it is `fdp`'s own packages (toksearch/toksearch_d3d, via
pyxrootd) that need the symbol; both the `fdp` and `labelmaker` environments
include this feature, so both are covered. Prepended, never overwritten: an
inherited `LD_LIBRARY_PATH` (e.g. from a module-loaded library on a SLURM node)
must still take effect first for anything it already covers.

Versions match Nathan's working fetch env (/scratch/gpfs/nc1514/fdp/pixi.toml)
rather than floating. `fdp` itself is REQUIRED, not optional: `toksearch_d3d`
imports it at module scope, so leaving it out resolved an env where
`import toksearch_d3d` raised ModuleNotFoundError while `toksearch` imported fine
- the fdp path was unreachable and nothing said so.

## feature.labelmaker

`labeler-verify` runs the page under `fdp run`, which sets `PTDATA_LIBRARY` and
`default_tree_path` and nothing interactive (measured 2026-09-22 on
stellar-vis2). A shot outside the corpus and the fetch cache can then be fetched
live. Arguments after the task name reach the page: `pixi run -e labelmaker
labeler-verify --port 8812`.

`pyarrow`: labels_index.parquet. pandas 3 can write parquet only through
pyarrow, and the default env does not have it.

`pyyaml`: registry.py parses every model card's YAML front matter. `yaml`
currently reaches this environment only transitively, via hydra-core ->
omegaconf, and hydra-core is declared in [project] for IGNITE's benefit, not
labeler's - anyone tidying IGNITE's dependencies would silently break every card
read in this package. Declare it directly.

`protobuf`: solving `labelmaker` (labelmaker + fdp + the implicit default
feature) against live conda-forge picks up protobuf 7.35.1, published to
conda-forge sometime after pixi.lock's other environments were last solved
(2026-07-31, per `git log -- pixi.lock`). The shared [project.dependencies] pin
`wandb>=0.25.1,<0.26` needs protobuf>4.21.0,!=5.28.0,!=5.29.0,<7, so the two are
unsatisfiable together:

```
Because wandb==0.25.1 depends on one of:
    protobuf>4.21.0,<5.28.0 / >5.28.0,<5.29.0 / >5.29.0,<7
and protobuf==7.35.1, we can conclude that wandb==0.25.1 cannot be used.
help: protobuf==7.35.1 has been pinned by the conda solve.
```

This is channel drift, not a toksearch/torch conflict (the brief's anticipated
failure mode and documented CPU-wheel-drop fallback, tried first, did not change
this error). It reproduced identically with and without the two
pypi-dependencies lines below and with pyarrow removed entirely, isolating the
cause to protobuf's rolling conda-forge version rather than to either of those.
The already-locked `fdp`/`default` environments are unaffected since they stay
pinned to their 2026-07-31 solve (libprotobuf/protobuf 6.31.1) until someone
reruns `pixi update`. Pinning protobuf here, scoped to this feature only, keeps
the new environment on the same already-known-good major version, the same way
`feature.cuda` pins torch's index rather than editing [project].

`fastapi`, `uvicorn`: the review page is a FastAPI app on the loopback, reached
over an SSH forward. Same ranges as feature.shot-design, which runs shot_design's
UI, so an environment holding both cannot want two different majors.

CPU torch: the workspace installs `faith` editable into every environment, which
drags torch in. Labelmaker never uses a GPU (its heaviest model is 12k
parameters evaluated in torch on CPU, see models/runners/keras_h5.py), so take
the CPU wheels and save ~3 GB in an environment that would otherwise duplicate
cu124.

`plotly`, `anywidget`: the verification notebooks draw with plotly
(`labeler.events.verify`), and `tests/labeler/test_events_verify.py` imports it
when the suite is collected. The review page draws its own canvases and needs
neither. `FigureWidget` imports `anywidget` at construction time, so a bare
`import plotly` succeeding says nothing about whether the widget works. That
widget is the notebooks' surface only, so this pin lives and dies with them.

## feature.shot-design

shot_design is ported from the shot-recommender-system repo
(/scratch/gpfs/nc1514/shot-recommender-system), which pins its runtime through a
uv.lock. Every range here was checked on 2026-09-07 against the versions that
lock actually resolves - pydantic 2.13.5, fastapi 0.141.1, sentence-transformers
6.0.1, httpx 0.28.1, pypdf 6.16.2, scikit-learn 1.9.0 - so none of them had to be
widened; a range that excluded the running version would produce a port that
passes here and fails there. `mcp` and `duckdb` have no counterpart in that lock
(the MCP server and the DuckDB-backed store are new in this port).

Everything is a CONDA dependency: conda-forge carries current builds of the
whole set (`mcp` 2.1.1 was published there on 2026-08-26), and the conda solver
is the one that also chooses this environment's python. The single exception is
`sentence-transformers`, which has to come from pypi (below).

- `pyarrow`: shots.parquet and the per-segment tables. Same range as
  feature.labelmaker so the two packages cannot end up wanting different pyarrow
  majors in an environment that includes both; shotrec only asks for
  `pyarrow>=16`.
- `pyyaml`: phenomena.yaml, actuators.yaml and the model cards shot_design reads.
  Declared directly for the same reason feature.labelmaker declares it.
- `pydantic`: the schema layer (PhenomenonHit, DesignReport, CircumstanceReport,
  ...). >=2.12 is also conda-forge `mcp`'s own floor, so this is not a free
  choice.
- `scipy`: signal processing behind the segment finder and the circumstances
  stats. Unpinned to match [project], which also lists a bare `scipy`.
- `h5py`: corpus reads, the DIII-D shot HDF5 files under $SHOT_DESIGN_CORPUS.
- `pypdf`: miniproposal PDFs -> the text that phenomenon search quotes from.
- `beautifulsoup4`: summary.html -> the per-shot text bundles. `shot_design logs
  import` re-runs d3dlogfetching's own composition (shotdb/logs.py) over a synced
  runs/ tree, and that composition parses the session summary with
  BeautifulSoup; the output has to be byte-identical to the 22,950 bundles
  already in the corpus, so the parser is not a free choice. 4.x is what the
  tool pins.
- `fastapi`, `uvicorn`: `shot_design serve`, the browser UI. Kept in the base
  feature rather than a separate one because the MCP server and the UI share the
  same query layer.
- `mcp`: `from mcp.server import MCPServer`. 2.x only: FastMCP, the 1.x entry
  point, is gone, so a solve that fell back to mcp 1.x would import-error rather
  than merely behave differently.
- `protobuf`: the same channel-drift pin, and the same reason, as
  feature.labelmaker.
- `duckdb`: the shot/segment/label store shot_design queries. conda-forge ships
  the Python bindings as `duckdb`.

`sentence-transformers` (MiniLM text embeddings,
sentence-transformers/all-MiniLM-L6-v2, 384-d). The 6.x major is what shotrec
runs; 5.x renamed enough of the encode path that a lower floor would not be the
code that was ported. It is the ONE package of the set taken from pypi rather
than conda-forge. conda-forge does publish it (6.0.1), but its recipe depends on
`pytorch >=1.11.0`, so adding it to the conda dependencies makes the conda
solver choose torch - it picks the newest, 2.12.0 - and pixi then hands that as
a PINNED version to the pypi solve, where it collides head-on with feature.cuda:

```
x failed to solve the pypi requirements of environment 'shot-design'
`-> Because you require torch>=2.5.1,<2.11 and torch==2.12.0, we can
    conclude that your requirements are unsatisfiable.
help: torch==2.12.0 has been pinned by the conda solve.
```

The cu124 range in feature.cuda is not ours to widen (cu124 is what ships sm_70
kernels for the V100S cards these environments run on), and neither environment
wants a second copy of torch. Taking sentence-transformers from pypi keeps torch
entirely on the pypi side, where feature.cuda and feature.shot-design-cpu
already decide which wheel each environment gets.

## feature.shot-design-cpu

A CPU-only twin of the `shot-design` environment, exactly as feature.labelmaker
does it. The workspace installs `faith` editable everywhere, which drags torch
in, and the cu124 wheels are ~3 GB that a laptop-shaped run of shot_design -
phenomenon search, the events tables, the MiniLM encoder - never touches. Only
`design_rollout` needs a GPU, and that is what the `shot-design` environment is
for. Split into its own feature (rather than putting the cpu index on
`shot-design` itself) so the two environments can differ in the one line that
separates them.

## feature.frontier

pip is needed for the `setup-flash-attn` task to install flash-attn from a git
URL with --no-build-isolation. The PyTorch wheels pulled from the rocm7.1 index
don't drag pip in transitively. ninja: aiter (a transitive dep of flash_attn on
ROCm) JIT-compiles a small C++ extension at first `import flash_attn`. It calls
`ninja` from PATH.

The rocm7.1 index ships torch 2.10.0 + torchvision 0.25-0.26 only. torch 2.10
declares triton-rocm as a dep; uv won't auto-discover it through the per-package
`index = ...`, so it is listed explicitly.

Flash-Attention 2 (gfx90a / MI250X) is NOT listed intentionally: the build needs
`module load rocm/7.1.1` + `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, which
pixi/uv can't set. Install it with the `setup-flash-attn` task; it uses the AMD
Triton backend (not Composable Kernel) per the AMD docs at
rocm.docs.amd.com/.../model-acceleration-libraries.html - Triton skips the
multi-hour CK template/hipcc compile and builds in ~10-15 min.

## feature.shot-design-frontier

The `shot-design` environment on Frontier: feature.shot-design's conda set
(pyarrow, duckdb, fastapi, ...) plus ROCm torch instead of feature.cuda's cu124
wheels. It is a third torch source beside feature.cuda and
feature.shot-design-cpu for the same reason those two are separate: the index is
the one line that differs. Same rocm7.1 pins as feature.frontier, so the two
environments on that machine run the same torch and a checkpoint written by one
loads in the other.

`plotly`: on Stellar the labeler runs in its own `labelmaker` environment, which
pins it; on Frontier there is one environment for both packages, so the pin is
repeated here rather than the suite being split (`tests/labeler` imports plotly
at collection).

## Environment names

Dash form, not `shot_design`: pixi rejects underscores in an environment name
("please use only lowercase letters, numbers and dashes"). The python package
these environments install is still `shot_design`.
````

- [ ] **The section the test reads.**

In `docs/shot-design/overview.md`, replace

````text
## Scratch databases and the pixi activation env

`pixi run -e shot-design` and `-e shot-design-cpu` set `SHOT_DESIGN_DATA_ROOT`, `LABELER_ROOT` and
`SHOT_DESIGN_CORPUS` from `[tool.pixi.feature.shot-design.target.unix.activation.env]` in `pyproject.toml`.
Activation runs *after* your shell, so a value you exported is replaced without a word. On
2026-09-14 a one-shot scratch build, run through `pixi run` with `SHOT_DESIGN_DATA_ROOT` exported to a
`/tmp` directory, published itself over the 500-shot production database.

So a scratch build must not go through `pixi run`. Call the environment's interpreter directly,
with the variables exported:

```bash
export SHOT_DESIGN_DATA_ROOT=/tmp/scratch-db HF_HUB_OFFLINE=1
/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/shot-design-cpu/bin/python -m shot_design build --shots 190000
```

or give it a paths file of its own — `SHOT_DESIGN_PATHS=<file>` — remembering that `SHOT_DESIGN_DATA_ROOT`
still wins over it, so it has to be out of the environment:
````

with

````text
## Scratch databases and the data root

`pixi run -e shot-design` and `-e shot-design-cpu` give `SHOT_DESIGN_DATA_ROOT`, `LABELER_ROOT` and
`SHOT_DESIGN_CORPUS` their production values only as defaults
(`[tool.pixi.feature.shot-design.target.unix.activation.env]` in `pyproject.toml`). A root you
export is the one the command writes under:

```bash
SHOT_DESIGN_DATA_ROOT=/tmp/scratch-db pixi run --frozen -e shot-design-cpu python -m shot_design build --shots 190000
```

Until 2026-09-22 the activation replaced an exported root without a word. On 2026-09-14 a one-shot
scratch build, run through `pixi run` with `SHOT_DESIGN_DATA_ROOT` exported to a `/tmp` directory,
published itself over the 500-shot production database. The batch scripts call the environment's
interpreter directly (`.pixi/envs/shot-design-cpu/bin/python -m shot_design ...`) with the roots
exported, which does not depend on the activation at all.

A paths file of its own works too — `SHOT_DESIGN_PATHS=<file>` — remembering that `SHOT_DESIGN_DATA_ROOT`
still wins over it, so it has to be out of the environment:
````

- [ ] Run the test: `5 passed`.

### E4: the fdp command is the task

In `src/labeler/events/ui/serve.py`, replace

```text
#: The wrapper every live fdp fetch has to run under, spelled the way
#: `docs/LABELER.md` spells it. `fdp run` execs a program, so it takes the
#: module form rather than the `labeler-verify` pixi task, which is not an
#: executable on PATH.
FDP_COMMAND = "pixi run -e labelmaker fdp run python -m labeler.events.ui"
```

with

```text
#: How to start the page so a live fdp fetch works: the task runs it under
#: `fdp run`.
FDP_COMMAND = "pixi run -e labelmaker labeler-verify"
```

In `src/labeler/events/raw.py`, replace

```text
FDP_UI_COMMAND = "pixi run -e labelmaker fdp run python -m labeler.events.ui"
```

with

```text
FDP_UI_COMMAND = "pixi run -e labelmaker labeler-verify"
```

- [ ] `tests/labeler/test_events_ui.py tests/labeler/test_events_raw.py tests/labeler/test_resolve_fdp.py`
  pass: `99 passed, 2 skipped` (the skips are `test_resolve_fdp.py`'s two `live` tests).

### E5: the sentences the defaults make false

P0 also edited `docs/clusters/stellar.md`, `docs/clusters/frontier.md`,
`docs/shot-design/simulation.md` and `scripts/shot_design/_stellar_common.sh`. If an old text
below no longer matches after the P0 merge, make the same change to the merged text.

In `docs/reference/environment-variables.md`, replace

```text
wrappers read, gathered in one place. Most are set by a pixi activation
block (`shot-design`, `shot-design-cpu`, `shot-design-frontier` in
`pyproject.toml`) rather than by hand — see
[Adding a cluster](../clusters/adding-a-cluster.md) for how a new cluster
wires its own values in.
```

with

```text
wrappers read, gathered in one place. Most get their value from a pixi
activation block (`shot-design`, `shot-design-cpu`, `shot-design-frontier` in
`pyproject.toml`) rather than by hand. The roots there are defaults, so a value
you export wins ([Pixi environments](./pixi-environments.md)). See
[Adding a cluster](../clusters/adding-a-cluster.md) for how a new cluster
wires its own values in.
```

In `docs/reference/environment-variables.md`, replace

```text
(below) would resolve `data_root` to — this is what lets a scratch build
never touch production by accident, and also what makes it dangerous to run
`pixi run` (which re-asserts the pixi activation values) against a scratch
target: use the environment's interpreter directly instead.
```

with

```text
(below) would resolve `data_root` to — this is what lets a scratch build
never touch production by accident. `pixi run` keeps a root you exported: the
activation blocks set the roots only as defaults (`${SHOT_DESIGN_DATA_ROOT:-...}`).
```

In `docs/clusters/stellar.md`, replace

```text
The `shot-design`/`shot-design-cpu` envs pin these on activation, so `pixi run -e shot-design`
always sees production paths, even if you exported something else:
```

with

```text
The `shot-design`/`shot-design-cpu` envs default these on activation:
```

In `docs/clusters/stellar.md`, replace

```text
To point at another root, use the env's interpreter directly, not `pixi run`:
`.pixi/envs/shot-design-cpu/bin/python -m shot_design ...` with your own exports.
Never run a `shot_design` write command through `pixi run` against a scratch
target; it will write to production.
```

with

```text
A root you export first wins: `SHOT_DESIGN_DATA_ROOT=/tmp/scratch-db pixi run --frozen -e
shot-design-cpu python -m shot_design ...` writes under `/tmp/scratch-db`. A write command
prints the root it resolved before its first write.
```

In `docs/clusters/frontier.md`, replace

```text
`pixi run --frozen -e shot-design-frontier ...` sets all four (plus
`HF_HUB_OFFLINE=1`, `TOKENIZERS_PARALLELISM=false`,
`HDF5_USE_FILE_LOCKING=FALSE`) through
`[tool.pixi.feature.shot-design-frontier.target.unix.activation.env]` in
`pyproject.toml`. `configs/shot_design/paths.frontier.yaml` carries every key
```

with

```text
`pixi run --frozen -e shot-design-frontier ...` defaults all four (plus
`HF_HUB_OFFLINE=1`, `TOKENIZERS_PARALLELISM=false`,
`HDF5_USE_FILE_LOCKING=FALSE`) through
`[tool.pixi.feature.shot-design-frontier.target.unix.activation.env]` in
`pyproject.toml`; a value you export first wins.
`configs/shot_design/paths.frontier.yaml` carries every key
```

In `docs/clusters/adding-a-cluster.md`, replace

```text
   cluster has no pixi) whose activation environment sets
   `SHOT_DESIGN_PATHS`, `SHOT_DESIGN_DATA_ROOT`, `LABELER_ROOT` and
   `SHOT_DESIGN_CORPUS` to that cluster's values — see
```

with

```text
   cluster has no pixi) whose activation environment defaults
   `SHOT_DESIGN_PATHS`, `SHOT_DESIGN_DATA_ROOT`, `LABELER_ROOT` and
   `SHOT_DESIGN_CORPUS` to that cluster's values (`"${VAR:-value}"`) — see
```

In `docs/shot-design/simulation.md`, replace

```text
The Stellar batch scripts under `scripts/shot_design/` run the interpreter
directly (never `pixi run`, whose activation would re-point the data root at
production) and REQUIRE `SHOT_DESIGN_DATA_ROOT` to name a batch root such as
```

with

```text
The Stellar batch scripts under `scripts/shot_design/` run the interpreter
directly and REQUIRE `SHOT_DESIGN_DATA_ROOT` to name a batch root such as
```

In `docs/shot-design/llm-providers.md`, replace

```text
create its `llm` directory first; account for Pixi activation overriding
`SHOT_DESIGN_DATA_ROOT` as described above. Endpoint JSON contains `url`, `models`,
```

with

```text
create its `llm` directory first. Endpoint JSON contains `url`, `models`,
```

In `scripts/shot_design/_stellar_common.sh`, replace

```text
# scripts/slurm_frontier/_shot_design_common.sh: the interpreter is called directly ($PY),
# never through `pixi run`, because pixi's activation would silently replace an exported
# SHOT_DESIGN_DATA_ROOT with the production root (docs/shot-design/overview.md, "Scratch
# databases and the pixi activation env").
```

with

```text
# scripts/slurm_frontier/_shot_design_common.sh: the interpreter is called directly ($PY),
# with the roots exported here, so a job does not depend on pixi's activation.
```

In `scripts/shot_design/batch_design.py`, replace

```text
the input row and add design_id / error / elapsed_s / trace path. Runs the interpreter it
was started with (never `pixi run`, whose activation would re-point the data root).
```

with

```text
the input row and add design_id / error / elapsed_s / trace path. Runs the interpreter it
was started with.
```

In `AGENTS.md`, replace

```text
- `pixi install`: install the default CUDA environment.
- `pixi install -e frontier`: install the Frontier environment; follow `README.md` for FlashAttention setup.
- `pixi run pytest tests/e2e/test_lora.py -q`: run a focused model test.
- `pixi run -e labelmaker pytest tests/labelmaker -q`: run labeling tests with their environment dependencies.
- `pixi run -e shot-design shot_design-test`: run the Shot Designer suite.
```

with

```text
- `pixi install --frozen`: install the default CUDA environment. Always `--frozen`: pixi 0.76 calls the lock out of date, and without it re-solves every environment.
- `pixi install --frozen -e frontier`: install the Frontier environment; follow `README.md` for FlashAttention setup.
- `pixi run --frozen pytest tests/e2e/test_lora.py -q`: run a focused model test.
- `pixi run --frozen -e labelmaker labeler-test`: run the labeler suite.
- `pixi run --frozen -e shot-design-cpu shot_design-test`: run the Shot Designer suite.
```

### Part E checks

- [ ] `SHOT_DESIGN_DATA_ROOT=/tmp/e0 pixi run --frozen --no-install -e shot-design-cpu printenv SHOT_DESIGN_DATA_ROOT`
  prints `/tmp/e0`; `env -u SHOT_DESIGN_DATA_ROOT pixi run --frozen --no-install -e shot-design-cpu printenv SHOT_DESIGN_DATA_ROOT`
  prints `/scratch/gpfs/EKOLEMEN/nc1514/ideate`.
- [ ] `pixi shell-hook --frozen --no-install -e shot-design-frontier | grep SHOT_DESIGN_PATHS`
  prints `export SHOT_DESIGN_PATHS="${SHOT_DESIGN_PATHS:-/lustre/orion/fus187/scratch/nchen/FusionAIHub/configs/shot_design/paths.frontier.yaml}"`.
- [ ] `pixi task list` shows `label`, `labeler-spectrograms`, `labeler-test`, `labeler-verify`,
  `setup-flash-attn`, `shot_design`, `shot_design-mcp`, `shot_design-test`, `verify-flash-attn`,
  each with its description.
- [ ] `pixi run --frozen --no-install -e labelmaker labeler-verify --port 8899` prints the link
  and the `ssh -L` line and no "not under the fdp wrapper" note; stop it with Ctrl-C.
- [ ] `git diff --quiet pixi.lock`.
- [ ] Both suites on the main checkout, with P0 and A in: no failures. (On the scratch post-A
  tree with E and without P0, shot_design gave `1835 passed, 2 failed, 3 skipped`: the base's two failures, which
  are P0's, and one more pass from the rewritten doc test.)
- [ ] Ruff on `serve.py`, `raw.py`, `batch_design.py`, `test_docs_scratch_db.py`;
  `bash -n scripts/shot_design/_stellar_common.sh`.
- [ ] Memory: `pixi-run-pins-ideate-data-root.md` says the roots are defaults since this commit,
  and how to check (`printenv` above); its index line in `MEMORY.md` changes to match.
- [ ] Commit `pixi: the roots are defaults, the tasks are described, the rationale is a doc`.

---

## After the merges (the session)

1. **Part 0.** Read `task-P0-report.md` and `git diff nathan_dev...v4-only`. Rerun the inventory
   grep and both suites in the worktree, then `git merge --no-ff v4-only`.
2. **Part A.** Read `task-A-report.md`. Check the branch against the plan: in a scratch
   directory, `git archive` the plan's commit, run the extractor there for every Part A path,
   apply the A7 delete, and `diff -r` the result against the branch's files. The only
   differences allowed are ones the report names. Rerun both suites in the worktree, then
   `git merge --no-ff label-review`.
3. **Part E**, as above; then both suites on `nathan_dev` with all three parts in.
4. **Stellar Simulate** (spec Part 0, Done when). Submit one design from the UI, or
   `sbatch scripts/shot_design/simulate.sbatch <ident>`. Gate it with
   `python -m labeler.jobstats --job-id <id>`; one design is a pilot, so it is reported rather
   than held to 70 %. Its `simulation.h5` must carry `frame_origin_s = 1.0`. Resize the script's
   `--mem` and `--time` from the run's MaxRSS and elapsed time.
5. **The re-encode.** Put array 2938824's jobstats line in `scripts/shot_design/encode_cpu.sbatch`'s
   sizing notes. Then refresh the spec's item 9 data, now that the codes are v4. That covers
   `db/shots.parquet`'s `has_frame_codes` (a `shot_design build` over `list:recommender_v1`
   with the production root, whose publish guard must pass without `--force`), and
   `ignite_inputs/reference_cache/199597.*` from the v4 bundle. The v2 seeds go to a
   `_v2_retired/` sibling. Remove the `FusionAIHub-v4enc` worktree once its branch is merged or
   empty.
6. **The AE store.** Run the pilot:
   `SHOTS="170659 170660 170661 170662 170663 170664 170665 170666" sbatch scripts/labeler/spectrograms.sbatch`.
   Gate it with `python -m labeler.jobstats --job-id <id> --cpu-only`, which is exempt as a
   pilot but reported. Then `sbatch scripts/labeler/spectrograms.sbatch` builds the other 172,
   gated at 70 % CPU and CPU memory. A miss is diagnosed and the script resized, never gamed.
   Confirm with a count of `$LABELER_ROOT/spectrograms/alfven_eigenmode/*.h5`: 180.
7. **The page on production data, read only.** Copy `data/events/` to a scratch directory and
   point `LABELER_LABEL_TABLES` at it. Start the server with the real `LABELER_ROOT`, and drive it
   headless: open an AE shot, time the first paint, and check that a reload restores the state.
   Nothing may save into the checkout's `data/events/alfven_eigenmode/review/`. Then the
   owner's own check through the SSH forward is the spec's Done when.
8. **Scoring.** The `iteration-scorer` agent (Opus 5.5, extra-high) and GPT-6-sol (extra-high,
   through `codex:codex-rescue`, read-only) each score the iteration 1-10. Average the two. Below
   8, fix what they name and score again.
9. **Records.** Append to the ledger, and update the memories this changes: the resume point, and
   pixi's roots (Part E).

Left for the owner, not done here: removing the sixteen `verification.ipynb` notebooks (and
with them `plotly`/`anywidget` in `labelmaker`); committing the 180-shot
`data/events/alfven_eigenmode/shots.csv` that sits uncommitted in the main checkout; any
`data/events/README.md` edit; renaming the v2 experiment roots and `models/IGNITE` (spec item
11), which waits until the paper figures read v4 roots.

---

## Self-review

**Spec coverage.**

| Spec | Plan |
|---|---|
| Part 0, changes 1-3 | P0.1 |
| Part 0, change 4 | P0.2 |
| Part 0, changes 5-6 | P0.3 (`g_enc.py` in P0.1's purge) |
| Part 0, change 7 | P0.5 |
| Part 0, change 8 | P0.6 |
| Part 0, change 9 | After the merges, 5 |
| Part 0, change 10 | P0.4 |
| Part 0, change 11 | left for the owner (waits for the paper figures) |
| Part 0, Done when | Part 0 acceptance; After the merges, 4 |
| A, What the reviewer gets | A7, A8 |
| A, Label store | A2 |
| A, Spectrogram store | A3, A4, A5, A6 |
| A, Server | A7 |
| A, Page | A8 |
| A, Tests | A2-A8 (each task's tests) |
| A, Done when | A8's browser test; After the merges, 6-7 |
| A, Left as it is | left for the owner |
| E, items 1-4 | E1-E5 |

**Placeholders.** None. Every Part A and Part E file is a block, generated from the files the
replay ran. `<ident>` and `<id>` in the session's steps are the job and design the steps
themselves create.

**Consistency.** Each Interfaces block was written against the code in its task's blocks: the
names, signatures and return shapes are the ones those files define.
