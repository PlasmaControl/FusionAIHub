# IGNITE on MAST, stage 1a: the headroom table for the other seven tasks. Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> Approved by the owner on 2026-11-22 (16:17 UTC, "yes i approve the plan. please carry it out"). The open decisions for the owner are taken as the author recommends until the owner says otherwise. The stage-0 numbers quoted in this text are those of the day it was written; `docs/models/ignite-mast-stage0.md` holds the current ones. Written 2026-11-22 by the plan's author. Since it was written, the controller started job J4's trainings on 2026-11-22 from 13:56 UTC, before Tasks 1 to 3 (ruling R68 of the stage-0 ledger); nothing else of this plan had run at approval. It is the first of three stage-1 plans; the order is 1a (this one), 1b (`2026-11-22-ignite-mast-stage1b-fixed-tokenizers.md`), 1c (`2026-11-22-ignite-mast-stage1c-learned-tokenizers.md`). 1a comes first because its trainings are the longest jobs of stage 1 and need nothing from 1b or 1c.

**Goal:** The headroom table of stage 0 has a measured row for each of the 14 TokaMark tasks, so that the projected 14-task mean is made of 14 measured scores and each of the other seven tasks has a trained direct model that 1b and 1c use as an instrument.

**Architecture:** No new predictor. The stage-0 commands already accept all 14 tasks; only the table restricts itself to seven (`tasks.STAGE0_TASKS` in `headroom/table.py`). This plan gives the table a task list, adds a second table command for the 14 tasks, checks on the CPU that the three predictors build for the seven tasks, and names the controller's scoring and training jobs.

**Tech Stack:** Python 3, numpy, pandas, torch, pytest; TokaMark's loader and evaluator through the stage-0 harness.

**Spec:** `docs/models/ignite-mast-design.md`, row "1. Tokenizers" of "Order of work" ("the headroom table for the other seven tasks"), and the paragraph "The headroom table" below it.

## Open decisions for the owner

These change the goal or the bar, so the plan does not decide them. Each has the author's recommendation.

1. **What the 14-task projection does to the bar.** The bar is unchanged in this plan: 0.183. Stage 0's table (2026-11-22, 09:55 UTC) projects 0.1999 for the 14-task mean, made of the direct model's `val_report` score on the seven stage-0 tasks and the best published score on the other seven; the best published score everywhere gives 0.194. By the design's own text the headroom check did not pass with this direct model, and the bar is with the owner, with that table. This plan replaces the seven borrowed numbers by measured ones: after job J5 the table prints the mean over the 14 tasks of the direct model's own `val_report` scores (P14). P14 can come out above or below 0.1999, and no task of this plan, of 1b or of 1c assumes either a changed bar or a passed check; stage 1 is not gated by the check, stage 2 is. Recommendation: do not move the bar on P14 alone. The direct model is one architecture at one budget with no pretraining, and stage 0 showed that its budget is not exhausted (task 1-1 still falling at 90,000 steps; batch 256 on task 4-4), so P14 is a lower bound on what is reachable and not the ceiling. Take the 14-row table to the owner with the per-task verdicts on the day J5 runs, and decide the bar at the token-path check early in stage 2, when a second kind of model has been scored.
2. **Whether a task where nothing beats the best published score keeps its place in the 14-task mean.** Task 4-5 is the known case: the best published score (0.6510 on the test split) is 0.002 under the published mean baseline (0.653), and the direct model ties it (0.6475 on `val_report`). On the validation shots the direct model is below the mean baseline by a paired difference of 0.0377 (95% interval 0.0298 to 0.0461, 351 shots, carried by a minority of the shots): something is predicted there, and it is small. Stage 0 also has ties on tasks 3-1, 4-1 and 4-4 (base run). The seven new rows may add others. Recommendation: every task stays in the mean, as the benchmark defines it; a task whose row is a tie is reported as a tie and gets no further direct-model training time in this plan.

## Global Constraints

- Machine: node13, aarch64, 4 GPUs of 286 GiB, 352 cores, 1.4 TB RAM. Data and runs go under `/models/intermediate/mast/`, never into the repository. Nothing is ever written under `/models/tokamind/`. Nothing is uploaded anywhere. Temporary files do not go to `/tmp`; use a directory under `/models/intermediate/mast/trials/`.
- Work in the main checkout `/home/nvidia/FusionAIHub` on branch `nvidia` (a git worktree run on this machine imports the main checkout's code). Commit locally. Never `git push`, `reset --hard`, `clean`, `stash`, `checkout -- <file>` or `commit --amend`. `git add` only this task's files, by path; never `-A` or `.`.
- Commit subjects start with `ignite.mast: `. The body says what changed and the measured facts behind it. The message ends with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.
- Two environments. `scripts/mast/py.sh` is `/models/envs/ignite-mast/bin/python -I` and has `tokamark`. `pixi run --frozen -e nvidia ...` (always `--frozen`) has no `tokamark`, so tests that need it skip there. `pyproject.toml` and `pixi.lock` do not change. No new dependency. No scikit-learn.
- No module under `src/tokamak_foundation_model/ignite/mast/` imports `tokamark`, `MAST_tools`, `mmt` or `zarr` at import time. A test that needs `tokamark` starts with `pytest.importorskip("tokamark")`.
- Hashed files do not change in this plan: `ignite/mast/harness.py`, `tasks.py`, `baselines.py`, `headroom/floor.py`, `headroom/cheap.py`, `headroom/direct.py`. Every stored score carries their sha256, and one changed byte marks all stored scores stale. `headroom/table.py` and `scripts/mast/run_headroom.py` are not hashed.
- Files under `src/tokamak_foundation_model/ignite/` outside `ignite/mast/` do not change.
- Tests run with `-W error` on the CPU: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/<file> -q -W error`, and the whole directory with `CUDA_VISIBLE_DEVICES="" pixi run --frozen -e nvidia python -m pytest tests/ignite/mast -q -W error`. An implementer never runs the whole `tests/ignite` suite (some of its tests use a GPU).
- An implementer runs no training, no scoring pass and no cache build, and uses no GPU. Long jobs are the controller's (section "Controller jobs"). A GPU process is started only with `--gpu N` for a GPU the controller has checked with `nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader`.
- NRMSE always means TokaMark's evaluator (`tokamark.evaluator.compute_metrics`) through `harness.score`. Fitted predictors are scored on `val_report` only; the test split is read by nothing fitted.
- Read files in parts (at most 150 lines per read; grep first). Never print arrays or long logs.
- Docs are part of a change. A number is written with its date, its split, its limits and whether it is measured or inferred. No data is not a negative; a missing value is never printed as zero.

## Review Focus

- A task of the 14-task table that has no stored score at all: its row is there, every cell is NaN and its verdict is `missing`, not a crash and not zeros. [Task 1, `test_the_table_can_cover_all_fourteen_tasks`]
- A 14-task table in which some tasks have no direct score: the projected mean is not computed, and the line says how many are missing; it does not fall back to the best published score for them without saying so. [Task 1, `test_the_projection_of_a_table_is_over_its_own_rows`; Task 2, `test_the_all_tasks_table_says_how_many_direct_scores_are_missing`]
- `--table-all` given together with another option: refused with exit code 2, as `--table` is. [Task 2, `test_the_all_tasks_table_refuses_every_other_option`]
- `--table` after this change: the same seven rows, the same file name and the same wording as before. [Task 2, `test_the_stage0_table_is_unchanged_by_the_all_tasks_table`]
- An output signal with two value axes (the flux map, `(65, 65)`) or with 2,500 samples per window (task 3-2) in the direct model: the model builds and its output has the target's shape. [Task 3, `test_the_direct_model_builds_and_runs_for_each_of_the_seven_tasks`]

## Measured facts this plan relies on (2026-11-22)

- The other seven tasks and what they predict (read from `tasks.load_task` on 2026-11-22): 1-3 flux map `equilibrium-psi` (65, 65), one frame; 2-2 boundary `equilibrium-lcfs_r` and `lcfs_z` (170 each), 5 frames; 2-3 flux map, 5 frames; 3-2 D-alpha (3 channels) and the two soft X-ray cameras (18 each), 2,500 samples at 0.02 ms; 3-3 Thomson `t_e` and `n_e` (120 each) and three betas, one frame, 150 ms of history; 4-2 the two soft X-ray cameras, 5,000 samples, 150 ms of history, 16 inputs; 4-3 `equilibrium-magnetic_axis_z`, 20 frames, 150 ms of history.
- Best published scores (`tasks.best_published`): 1-3 0.1156, 2-2 0.0370, 2-3 0.0828, 3-2 0.2522, 3-3 0.2147, 4-2 0.2140, 4-3 0.0691. TokaMind replication (`table.TOKAMIND_REPLICATION`): 0.1198, 0.0439, 0.0917, 0.2671, 0.2197, 0.2155, 0.0810.
- `scripts/mast/run_headroom.py --task` and `scripts/mast/check_harness.py --tasks` already accept all 14 tasks. Under `/models/intermediate/mast/stage0/scores/` each of the seven has only `mean/test` and `mean/val_report` (looked at `task_2-2` on 2026-11-22).
- TokaMind's replication runs `/models/tokamind/repo/runs/ft-task_<t>-ws-tokamind-base-v2-repl/embeddings/dct3d.yaml` exist for the seven tasks; every output signal is in `selection_mode: rank` with `nan_imputation: interpolate`, which is what `floor.tokamind_output_indices` reads. Not verified: that the floor runs end to end on them (Task 3 checks what the CPU can check).
- Durations measured in stage 0: a direct training of 30,000 steps took 6,336 s (task 1-2 alone), 11,680 s (3-1), 8,780 s (4-1 with 50 ms history) and about 12,600 s (4-1 base, extrapolated from 27,500 steps in 11,568 s). Scoring the floor of task 4-1 on the test split took 4,131 s; ridge on task 1-1 took 1,147 s; shrunk persistence on 3-1 took 635 s.
- What stage 0 measured about the direct model's settings (`docs/models/ignite-mast-stage0.md`, 2026-11-22; `val_report`, paired differences to the base run). The base configuration (30,000 steps, learning rate 3e-4, batch 64, the task's own history) was fixed before any score existed and is not tuned. Learning rate 1e-4 (`lr1e4`) helps on task 1-2 only (0.0362 against 0.0493 for the base run, which was unstable: best checkpoint at its first validation); on tasks 1-1 and 2-1 it is worse than base by 0.0056 each. Three times the steps (`s90k`) helps on task 1-1 (0.1235 against 0.1361, paired -0.0127, still falling at step 90,000). On task 4-4, 50 ms of history with batch 256 (`h050b256`) gives 0.2859 against 0.3195 (paired -0.034, interval -0.043 to -0.025), the largest gain of any variant; 50 ms of history alone (`h050`, batch 64) changes tasks 4-1 and 4-4 by less than about 0.005, so on group 4 the base batch is a limit of the budget, not a choice. Two more variants (`h050b256` on 4-1, `b256` on 4-5) were still training when this was written and are not results.
- What limits a training's speed (stage 0, 12 trainings sharing the four GPUs; the machine has 352 cores): on tasks 1-1, 1-2 and 2-1 TokaMark's loader (median 668 to 1,334 windows per second), on the group-4 tasks the GPU (median 200 to 205 windows per second at batch 64, 323 and 383 with 50 ms of history). Not measured for task 3-1. (Figures corrected 2026-11-22 16:10 UTC by the controller to the medians of `docs/models/ignite-mast-stage0.md` at `c4489281`; the draft first carried the upper-middle values 690 to 1,343, 202 to 213, 324 to 384.)

## File structure

```
src/tokamak_foundation_model/ignite/mast/headroom/table.py   (modify: a task list)
scripts/mast/run_headroom.py                                 (modify: --table-all)
tests/ignite/mast/test_table.py                              (modify: new tests)
tests/ignite/mast/test_headroom_seven.py                     (create)
docs/models/ignite-mast-stage0.md                            (modify, Task 4: the 14-task table)
```

Results stay in the stage-0 tree, because the headroom table is one instrument: scores under `/models/intermediate/mast/stage0/scores/<task>/<predictor>/<split>/`, direct runs under `/models/intermediate/mast/stage0/direct/<task>/<variant>/`, the new table at `/models/intermediate/mast/stage0/headroom_table_all.csv`. Decided by the plan's author: the directory keeps its name `stage0`.

### Task 1: the table takes a task list

**Files:**
- Modify: `src/tokamak_foundation_model/ignite/mast/headroom/table.py` (`build_table` near line 254, `projected_mean` near 310, `variants_markdown` near 391)
- Test: `tests/ignite/mast/test_table.py`

**Interfaces:**
- Consumes (existing, unchanged): `tasks.STAGE0_TASKS`, `tasks.ALL_TASKS` (tuples of task names), `tasks.best_published(task: str) -> float`, `table.TOKAMIND_REPLICATION: dict[str, float | None]` (all 14 tasks). The test helpers `write_score(scores, task, predictor, split, per_shot, *, code=None, **extra)` and `write_log(stage0, task, variant, val_nrmse, config=None)` at the top of `tests/ignite/mast/test_table.py`.
- Produces:
  - `build_table(scores_dir: Path, tokamind_repl: dict[str, float | None], stale_reason: Callable[[str, str, str], str] | None = None, task_names: Sequence[str] = tasks.STAGE0_TASKS) -> pd.DataFrame` with one row per name of `task_names`, in that order.
  - `projected_mean(table: pd.DataFrame) -> float`: the 14-task mean with the direct score on the table's own rows and the best published score on the tasks that are not rows; NaN while any row has no direct score.
  - `variants_markdown(scores_dir: Path, stale_reason: Callable[[str, str, str], str] | None = None, task_names: Sequence[str] = tasks.STAGE0_TASKS) -> str`.

- [ ] **Step 1: Write the failing tests** (append to `tests/ignite/mast/test_table.py`)

```python
def test_the_table_can_cover_all_fourteen_tasks(tmp_path):
    stage0 = tmp_path / "stage0"
    write_score(stage0 / "scores", "task_2-3", "mean", "val_report", [0.4, 0.6])
    write_score(stage0 / "scores", "task_2-3", "direct", "val_report", [0.07, 0.09], trained_steps=100, max_steps=100)
    write_log(stage0, "task_2-3", "base", [0.09, 0.08])
    t = table.build_table(stage0 / "scores", table.TOKAMIND_REPLICATION, task_names=tasks.ALL_TASKS)
    assert list(t.index) == list(tasks.ALL_TASKS) and t.index.name == "task"
    assert t.loc["task_2-3", "mean_val"] == pytest.approx(0.5)
    assert t.loc["task_2-3", "direct_val"] == pytest.approx(0.08)
    assert t.loc["task_2-3", "tokamind_repl"] == 0.0917
    assert t.loc["task_2-3", "best_published"] == tasks.best_published("task_2-3")
    for task in ("task_1-3", "task_4-2"):        # nothing stored: NaN cells and a verdict that says so
        assert math.isnan(t.loc[task, "direct_val"]) and math.isnan(t.loc[task, "mean_val"])
        assert t.loc[task, "verdict"] == "missing"


def test_the_default_table_is_still_the_seven_stage0_tasks(tmp_path):
    t = table.build_table(tmp_path / "scores", REPL)
    assert list(t.index) == list(tasks.STAGE0_TASKS)


def test_the_projection_of_a_table_is_over_its_own_rows():
    t = pd.DataFrame({"direct_val": [0.1] * 14}, index=pd.Index(tasks.ALL_TASKS, name="task"))
    assert table.projected_mean(t) == pytest.approx(0.1)          # 14 measured rows, no published score used
    t.loc["task_3-2", "direct_val"] = np.nan
    assert math.isnan(table.projected_mean(t))                    # not a silent fall-back to the published score


def test_the_variants_listing_takes_the_task_list(tmp_path):
    stage0 = tmp_path / "stage0"
    write_score(stage0 / "scores", "task_2-3", "direct", "val_report", [0.08], trained_steps=100, max_steps=100)
    write_log(stage0, "task_2-3", "base", [0.09, 0.08])
    assert table.variants_markdown(stage0 / "scores") == ""
    assert "Task 2-3:" in table.variants_markdown(stage0 / "scores", task_names=tasks.ALL_TASKS)
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_table.py -q -W error -k "fourteen or still_the_seven or own_rows or takes_the_task_list"`
Expected: 3 failed (`TypeError: build_table() got an unexpected keyword argument 'task_names'`, the same for `variants_markdown`, and `test_the_projection_of_a_table_is_over_its_own_rows`), 1 passed (`test_the_default_table_is_still_the_seven_stage0_tasks`).

- [ ] **Step 3: Implement the three signatures of "Produces" in `headroom/table.py`**

Replace each use of `tasks.STAGE0_TASKS` in `build_table` and `variants_markdown` by the new parameter. In `projected_mean` the rows are `table.index`: NaN if `table["direct_val"]` has a NaN, else the mean over `tasks.ALL_TASKS` of the row's value where the task is a row and `tasks.best_published(task)` where it is not. Update the three docstrings: they say "the stage-0 tasks"; they now say "the tasks of `task_names` (default: the seven stage-0 tasks)".

- [ ] **Step 4: Run the file**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_table.py -q -W error`
Expected: all pass (the tests that were there before this task included).

- [ ] **Step 5: Commit**

```bash
git add src/tokamak_foundation_model/ignite/mast/headroom/table.py tests/ignite/mast/test_table.py
git commit -m "ignite.mast: the headroom table takes a task list; its projection is over its own rows"
```

### Task 2: `--table-all`, the 14-task table

**Files:**
- Modify: `scripts/mast/run_headroom.py` (`print_table` near line 544, the `--table` argument and its refusal in `main` near 612 to 640)
- Test: `tests/ignite/mast/test_table.py`

**Interfaces:**
- Consumes: Task 1's `build_table(..., task_names=...)`, `projected_mean(table)`, `variants_markdown(..., task_names=...)`. The test helper `_run_headroom()` (loads the script as a module) and `write_score` in `tests/ignite/mast/test_table.py`.
- Produces:
  - `print_table(task_names: Sequence[str] = tasks.STAGE0_TASKS, csv_name: str = "headroom_table.csv") -> int` in `scripts/mast/run_headroom.py`.
  - The command line option `--table-all`: `print_table(tasks.ALL_TASKS, "headroom_table_all.csv")`. Like `--table` it scores nothing, takes no other option and returns 0.
  - Output lines the controller reads: `projected mean over the 14 tasks (direct model on all 14 tasks): <value>; the bar is 0.183, matching the best published score everywhere would give 0.194` or `projected mean: not computed, <n> of 14 tasks have no direct score`.

- [ ] **Step 1: Write the failing tests** (append to `tests/ignite/mast/test_table.py`)

```python
def _all_tasks_stage0(tmp_path, monkeypatch, direct_for):
    rh = _run_headroom()
    stage0 = tmp_path / "stage0"
    for task in direct_for:
        write_score(stage0 / "scores", task, "direct", "val_report", [0.1, 0.1], trained_steps=10, max_steps=10)
        write_log(stage0, task, "base", [0.2, 0.1], config={"depth": 8, "max_steps": 10})
    monkeypatch.setattr(rh.paths, "stage0_dir", lambda: stage0)
    monkeypatch.setattr(rh.paths, "prepare_write_root", lambda p: Path(p))
    monkeypatch.setattr(rh, "stale_reason", lambda root, task, name, split: "")
    return rh, stage0


def test_the_all_tasks_table_has_fourteen_rows_and_its_own_file(tmp_path, monkeypatch, capsys):
    rh, stage0 = _all_tasks_stage0(tmp_path, monkeypatch, tasks.ALL_TASKS)
    assert rh.main(["--table-all"]) == 0
    out = capsys.readouterr().out
    line = next(line for line in out.splitlines() if line.startswith("projected mean"))
    assert "direct model on all 14 tasks" in line and "0.1000" in line and "0.183" in line
    csv = pd.read_csv(stage0 / "headroom_table_all.csv", index_col="task")
    assert list(csv.index) == list(tasks.ALL_TASKS)
    assert not (stage0 / "headroom_table.csv").exists()


def test_the_all_tasks_table_says_how_many_direct_scores_are_missing(tmp_path, monkeypatch, capsys):
    rh, _ = _all_tasks_stage0(tmp_path, monkeypatch, tasks.STAGE0_TASKS)
    assert rh.main(["--table-all"]) == 0
    assert "projected mean: not computed, 7 of 14 tasks have no direct score" in capsys.readouterr().out


@pytest.mark.parametrize("option", [["--task", "task_1-3"], ["--split", "test"], ["--force"], ["--table"]])
def test_the_all_tasks_table_refuses_every_other_option(tmp_path, monkeypatch, capsys, option):
    rh, _ = _all_tasks_stage0(tmp_path, monkeypatch, ())
    with pytest.raises(SystemExit) as e:
        rh.main(["--table-all", *option])
    assert e.value.code == 2 and "takes no other option" in capsys.readouterr().err


def test_the_stage0_table_is_unchanged_by_the_all_tasks_table(tmp_path, monkeypatch, capsys):
    rh, stage0 = _all_tasks_stage0(tmp_path, monkeypatch, tasks.ALL_TASKS)
    assert rh.main(["--table"]) == 0
    out = capsys.readouterr().out
    assert "direct model on the 7 stage-0 tasks, best published score on the other 7" in out
    assert list(pd.read_csv(stage0 / "headroom_table.csv", index_col="task").index) == list(tasks.STAGE0_TASKS)
    assert not (stage0 / "headroom_table_all.csv").exists()
```

If `stale_reason` is not a module-level name of `run_headroom.py` taking `(root, task, name, split)` (read `print_table` to see how it is called), patch the name that `print_table` calls instead and say so in the report.

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_table.py -q -W error -k "all_tasks_table"`
Expected: FAIL; the first with `SystemExit: 2` (`unrecognized arguments: --table-all`).

- [ ] **Step 3: Implement `print_table(task_names, csv_name)` and `--table-all`**

`--table` and `--table-all` share the refusal of every other option (each also refuses the other). The count line says "stage-0 tasks" when `task_names` is `tasks.STAGE0_TASKS` and "tasks" otherwise; the projection line for the seven-task table keeps its present wording byte for byte.

- [ ] **Step 4: Run the file**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_table.py -q -W error`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/mast/run_headroom.py tests/ignite/mast/test_table.py
git commit -m "ignite.mast: run_headroom --table-all prints the headroom table of all 14 tasks"
```

### Task 3: the three predictors build for the seven tasks (CPU check)

Nothing in stage 0 ran the floor, the cheap fits or the direct model on these tasks. This task finds on the CPU, in minutes, what would otherwise stop a job of hours. It changes no source file. If a test of this task fails because of a defect in a hashed file (`floor.py`, `cheap.py`, `direct.py`), do not edit that file: report the failure with the traceback's last ten lines; the controller decides (a change to a hashed file re-scores every stored score).

**Files:**
- Create: `tests/ignite/mast/test_headroom_seven.py`

**Interfaces:**
- Consumes (existing; quoted from the code on 2026-11-22):
  - `tasks.load_task(name: str) -> TaskSpec`; `TaskSpec.inputs`, `.actuators`, `.outputs` are tuples of `SignalSpec(name, dt, values_shape, mean, std)`; `TaskSpec.n_in(s)`, `.n_act(s)`, `.n_out(s)` give the number of samples of a window.
  - `harness.WindowBatch(shot_ids, window_indices, inputs, actuators, targets, last_valid)`; arrays are `(B, *values_shape, T)` float32 in physical units, `last_valid[name]` is `(B, *values_shape)`.
  - `floor.tokamind_output_indices(task_name: str) -> dict[str, tuple[np.ndarray, tuple[int, int, int]]]`, `floor.tokamind_nan_imputation(task_name: str) -> str`, `floor.floor_predictor(task, indices, nan_imputation) -> PredictFn`.
  - `direct.DirectConfig` (dataclass; fields `d_model`, `depth`, `heads` among others), `direct.default_config(task) -> DirectConfig`, `direct.DirectModel(task, cfg)` with `forward(inputs: dict[str, Tensor], actuators: dict[str, Tensor]) -> dict[str, Tensor]`, `direct.to_tensors(batch, task, device) -> (inputs, actuators, targets)`, `direct.masked_mse(pred, target) -> Tensor`.
  - `cheap.RidgeFrames`, `cheap.ShrunkPersistence`: read their constructors and `fit`/`predict` methods in `headroom/cheap.py` (in parts) before writing the cheap test; the plan's author did not read them.
- Produces: nothing for other tasks. A report line per task and predictor: builds or does not, with the error.

- [ ] **Step 1: Write the tests**

```python
"""The stage-0 predictors on the seven tasks stage 0 did not run. CPU only; no scoring, no training."""

import dataclasses

import numpy as np
import pytest
import torch

pytest.importorskip("tokamark")
from tokamak_foundation_model.ignite.mast import tasks                      # noqa: E402
from tokamak_foundation_model.ignite.mast.harness import WindowBatch        # noqa: E402
from tokamak_foundation_model.ignite.mast.headroom import direct, floor     # noqa: E402

SEVEN = ("task_1-3", "task_2-2", "task_2-3", "task_3-2", "task_3-3", "task_4-2", "task_4-3")
pytestmark = pytest.mark.real_data


def synthetic_batch(task, n=2, seed=0):
    rng = np.random.default_rng(seed)

    def draw(specs, length):
        return {s.name: (s.mean + s.std * rng.standard_normal((n, *s.values_shape, length(s)))).astype(np.float32)
                for s in specs}

    inputs = draw(task.inputs, task.n_in)
    return WindowBatch(shot_ids=np.full(n, 30000, np.int64), window_indices=np.arange(n, dtype=np.int64),
                       inputs=inputs, actuators=draw(task.actuators, task.n_act),
                       targets=draw(task.outputs, task.n_out),
                       last_valid={k: a[..., -1].copy() for k, a in inputs.items()})


def test_the_seven_are_the_tasks_stage0_left():
    assert set(SEVEN) == set(tasks.ALL_TASKS) - set(tasks.STAGE0_TASKS)


@pytest.mark.parametrize("name", SEVEN)
def test_the_floor_builds_and_keeps_the_shape_for_each_of_the_seven_tasks(name):
    task = tasks.load_task(name)
    predict = floor.floor_predictor(task, floor.tokamind_output_indices(name), floor.tokamind_nan_imputation(name))
    batch = synthetic_batch(task)
    out = predict(batch)
    for s in task.outputs:
        assert out[s.name].shape == batch.targets[s.name].shape and np.isfinite(out[s.name]).all()


@pytest.mark.parametrize("name", SEVEN)
def test_the_direct_model_builds_and_runs_for_each_of_the_seven_tasks(name):
    task = tasks.load_task(name)
    cfg = dataclasses.replace(direct.default_config(task), d_model=32, depth=1, heads=2)
    model = direct.DirectModel(task, cfg)
    inputs, actuators, targets = direct.to_tensors(synthetic_batch(task), task, "cpu")
    pred = model(inputs, actuators)
    for s in task.outputs:
        assert pred[s.name].shape == targets[s.name].shape
    loss = direct.masked_mse(pred, targets)
    loss.backward()
    assert torch.isfinite(loss)
```

Add a third parametrized test, `test_the_cheap_predictors_fit_and_predict_for_each_of_the_seven_tasks(name)`, after reading `cheap.py`: fit `RidgeFrames` and `ShrunkPersistence` on three synthetic batches of `synthetic_batch(task, n=8, seed=i)` in the way `cheap.obtain` fits them, predict on a fourth, and assert for every output signal that the prediction has the target's shape and is finite.

- [ ] **Step 2: Run them**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_headroom_seven.py -q -W error`
Expected: 22 passed (1 + 3 × 7) in a few minutes. A failure is a finding, not a test to weaken: report it (see the note under the task title). If the 4-2 or 3-2 direct case needs more than 10 minutes on the CPU, lower `n` to 1 and say so.

- [ ] **Step 3: Commit**

```bash
git add tests/ignite/mast/test_headroom_seven.py
git commit -m "ignite.mast: the stage-0 predictors build on the CPU for the other seven tasks"
```

### Task 4: the 14-task table in the docs

Runs after the controller's jobs J1 to J5 have finished. The controller hands the implementer the Markdown that `--table-all` printed (file `/models/intermediate/mast/stage0/headroom_table_all.md`, see J5).

**Files:**
- Modify: `docs/models/ignite-mast-stage0.md` (a new section at the end: "The other seven tasks (stage 1a)")

**Interfaces:**
- Consumes: `/models/intermediate/mast/stage0/headroom_table_all.md` and `headroom_table_all.csv`, written by J5.
- Produces: the section, which 1b and 1c cite for the direct model's score per task.

- [ ] **Step 1: Write the section.** It holds, in this order: the date of the table and the git revision (`git rev-parse --short HEAD`); the seven new rows copied from the Markdown table with their marks († stale, ‡ unfinished) and the legend; for each of the seven tasks the variant that the table chose and its learning rate; the projected 14-task mean as printed, with the sentence "on `val_report` shots, direct model with no pretraining; the published scores it is compared with are on test shots"; and one paragraph "What it does not say": the direct model is one architecture at one budget, a score above the best published one is not evidence that the task has no headroom. A cell that is missing is written "not scored", never 0.
- [ ] **Step 2: Build the docs.** Run: `uvx --with-requirements requirements-docs.txt --from 'mkdocs<2' mkdocs build --strict`. Expected: exit code 0, no warning. If `uvx` has no network, say so in the report and check the page's links by hand.
- [ ] **Step 3: Commit**

```bash
git add docs/models/ignite-mast-stage0.md
git commit -m "ignite.mast: docs, the headroom table of the other seven tasks"
```

## Controller jobs

Run from `/home/nvidia/FusionAIHub` after Tasks 1 to 3 are merged. Each command keeps a stored score that is current and says so. Logs go to `/models/intermediate/mast/stage0/<job>_<task>.log` (`> file 2>&1`).

| Job | Command (once per task T of the seven) | GPU | Expected duration | What is read from its output |
|---|---|---|---|---|
| J1 | `scripts/mast/check_harness.py --tasks task_1-3 task_2-2 task_2-3 task_3-2 task_3-3 task_4-2 task_4-3 --split val_report`, then the same with `--split test` | none | 20 to 60 min per split (inferred from stage 0's loader rates of 1,400 to 5,500 windows/s) | per task the mean and persistence NRMSE; the mean baseline on test must agree with `tasks.PUBLISHED[T]["mean"]` within the tolerance the command prints, or the harness is wrong for that task and everything below waits |
| J2 | `scripts/mast/run_headroom.py --task T --predictor floor --split val_report --num-workers 32`, then `--split test` | none | 5 to 20 min for 1-3, 2-2, 2-3, 3-3, 4-3; 60 to 90 min for 3-2 and 4-2 (4-1 took 4,131 s on test). The 14 runs can go in parallel | the last line `T floor <split>: nrmse <v>, <n> windows, <s> s`. This is TokaMind's representation floor for T |
| J3 | `scripts/mast/run_headroom.py --task T --predictor shrunk --split val_report --num-workers 32` and the same with `--predictor ridge` | none | 10 to 40 min each (1-1 ridge took 1,147 s); 3-2 and 4-2 longer, not measured | the last line, as J2 |
| J4 | `scripts/mast/run_headroom.py --task T --predictor direct --split val_report --train --gpu N --share-gpu --num-workers 24`, with the options of the table "The configurations of J4" below: 11 runs | 4, shared by the 11 runs (stage 0 ran 12 at once) | base and `lr1e4` runs of groups 1 to 3: 2 to 4 h (stage 0: 6,339 s to 14,355 s for 30,000 steps); `h050b256`: about 4 h (4-4: 13,367 s on a shared GPU); base runs of 4-2 and 4-3: 4 to 6 h (150 ms of history, GPU-bound). One wave: 5 to 7 h | `train_log.csv` in the run directory (`step,train_loss,val_nrmse,seconds,...`) and the last line `T direct[.variant] val_report: nrmse <v>` |
| J5 | `scripts/mast/run_headroom.py --table-all > /models/intermediate/mast/stage0/headroom_table_all.md` | none | seconds | the seven new rows, the marks, and the line `projected mean ...` |

The configurations of J4, decided by the plan's author before any number exists. None of them is tuned: the base configuration was fixed in stage 0 before any score existed, and each departure below is carried over from one measured stage-0 task to a task where it was not measured.

| Task | Runs | Options | Why |
|---|---|---|---|
| 2-2 (boundary, 5 frames) | `lr1e4` | `--variant lr1e4 --lr 1e-4` | The same output signals as task 1-2, the one task where 3e-4 was unstable and 1e-4 was measured to be better. |
| 1-3, 2-3 (flux map) | base and `lr1e4` | none; `--variant lr1e4 --lr 1e-4` | No stage-0 task predicts the flux map. The lower rate helped one task and hurt two, so neither rate has a measured claim here; both run in the same wave, which is cheaper than a second wave after a failed first. |
| 3-2, 3-3 | base | none | Nothing measured in stage 0 argues for another setting on these (task 3-1 has a base run only). |
| 4-2, 4-3 | base and `h050b256` | none; `--variant h050b256 --history-s 0.05 --train-batch-size 256` | On task 4-4 batch 256 at 50 ms of history was the largest gain of any variant, and the shorter history cost nothing measurable on 4-1 and 4-4. Whether either holds for 4-2 (5,000 output samples) or 4-3 (20 output frames) is not measured, so the base run is trained beside it. |

Rules for J4:

- The table takes, per task, the variant with the lowest best `val_nrmse` on `val_select` (its existing rule); `val_report` is not used to choose.
- A run whose lowest `val_nrmse` is at its last or second-last validation is reported as "still falling" beside its row: its score is then an upper bound on what this configuration reaches, as on task 1-1 in stage 0. It is extended to 90,000 steps (`--variant s90k --max-steps 90000`) only if its verdict is "behind" and a GPU is free; J5 does not wait for it.
- One rerun is allowed for a task with a single run (2-2, 3-2, 3-3), and only by this rule: if the lowest `val_nrmse` of the run is at its first or second validation and the last validation is more than 20% above it, the task is trained once more at the other learning rate. No other variant is trained in this plan.
- A run that stops early is resumed or repeated; an unfinished run is marked ‡ by the table and is not a result.

What decides what:

- `verdict` per row (the table's existing rule on the bootstrap interval of the direct score against the best published score) says for each task whether the direct model is ahead of, level with or behind the published best. It decides nothing by itself; it is what open decision 2 is about.
- The line `projected mean over the 14 tasks` is the number of open decision 1. The controller sends it to the owner with the table on the day J5 runs, and does not wait for stage 1b or 1c.
- For 1b and 1c: a task's direct checkpoint is used as an instrument (prediction pass-through and input round-trip) only if its row is not marked † or ‡ and is not `not_competitive`. Otherwise the instrument is not available for that task, and 1b's rules fall back as they say.

## What this plan leaves to later plans

- Recording TokaMark's revision and the data root in every score, the `continue` in `harness._last_finite`, and hashing a normalized form of the hashed files: all three change hashed files and wait for the next batch of such changes.
- Any further variant of the direct model (other widths, batch 256 at the full history, more steps beyond the one extension rule). The direct model is an instrument here: its rows are lower bounds at a stated budget, not tuned results, and the page says so beside each row.
