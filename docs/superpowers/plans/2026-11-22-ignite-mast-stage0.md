# IGNITE on MAST, stage 0: harness, data and headroom. Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A scoring harness that reproduces TokaMark's published baselines, a standardized MAST cache with a channel registry, and a headroom table that says, for seven tasks, how far the published models are from what the data allows.

**Architecture:** A new package `tokamak_foundation_model.ignite.mast` beside the existing IGNITE modules. Test windows always come from TokaMark's own dataset class and are scored by TokaMark's own evaluator; our code only supplies a `predict_fn`. Stage 0 also trains from TokaMark's loader, so every window it sees is TokaMark's by construction. The cache is built through TokaMark's shot-level loader and is for stages 1 and 2.

**Tech Stack:** Python 3.12 in a dedicated uv environment (`/models/envs/ignite-mast`), torch 2.15.0+cu132, numpy, pandas, h5py, scipy, `tokamark` and `MAST_tools` (from `/models/tokamind/tokamark`), `mmt` (TokaMind, from `/models/tokamind/repo`), pytest.

**Spec:** [`docs/models/ignite-mast-design.md`](../../models/ignite-mast-design.md). Read "The bar", "Data path", "Splits and leakage", "TokaMark adaptation and scoring" and "Order of work" before starting. This plan covers the row "0. Harness, data and headroom" of its order-of-work table and nothing later.

## Global Constraints

- Work in place in `/home/nvidia/FusionAIHub` on branch `nvidia`. Commit locally after each task. Never push.
- Data, caches, checkpoints and results go under `/models/intermediate/mast/`. Nothing of that kind goes into the repository. `.gitignore` already excludes `*.h5`, `*.pt` and `*.npy`.
- Every Python process that reads MAST data runs through `scripts/mast/py.sh` (Task 1), which passes `-I` and sets the NVRTC preload. Do not use `PYTHONPATH`; `-I` ignores it.
- `pixi run --frozen -e nvidia ignite-test` must stay green. The pixi environment has no `tokamark`, `zarr` or `scikit-learn`, so a test that needs `tokamark` starts with `pytest.importorskip("tokamark")`, and no module under `ignite/mast/` imports `tokamark`, `MAST_tools`, `mmt` or `zarr` at import time (import them inside the functions that use them). Do not use scikit-learn.
- Do not change `pyproject.toml` or `pixi.lock`. Do not edit any existing file under `src/tokamak_foundation_model/ignite/`.
- Do not write anything under `/models/tokamind/`. The TokaMind replication queue is running there (`/models/tokamind/run_all_tasks.sh`, log `/models/tokamind/logs/run_all.log`).
- A GPU may be used only if `nvidia-smi` shows no compute process on it and the queue log has a line `gpuN done` for it. Check before every launch.
- Test shots: only predictors with nothing fitted by us (mean, persistence, TokaMind's representation floor) are scored on the test split. Everything fitted is scored on validation shots. The spec says the test shots are read once, for the final table.
- Shots are taken from TokaMark's split file only. The 385 shots that are in none of its three splits (negative plasma current) are never loaded.
- NRMSE always means the task score written by `tokamark.evaluator.compute_metrics`. Do not reimplement the metric.
- Docs are part of the change. Record measurements with their date and say why a setting has its value.
- Commit subjects start with `ignite.mast:` (or `docs:`), bodies state what changed and the measured facts. End each commit message with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.

## Review Focus

Each line names a condition the spec implies and the behaviour a user would expect. The test that pins it is in the task named in brackets.

1. A window in which an input channel has no finite sample at all. Expected: every predictor still returns a finite prediction for every finite target, and the harness refuses a prediction that does not. [Task 3, `test_score_rejects_nonfinite_prediction`; Task 9, `test_direct_model_is_finite_with_all_nan_inputs`]
2. A shot that lacks a signal entirely (11 shots hold only magnetics and soft X-ray; TokaMark returns empty arrays). Expected: the cache writes no dataset for it and the reader reports the signal as absent. It must not write zeros. [Task 6, `test_missing_signal_is_absent_not_zero`]
3. A signal delivered at a different sampling interval than the registry says (soft X-ray is 500 kHz in a minority of raw stores). Expected: the cache writer refuses the signal for that shot and counts it. It must not store it under the wrong `dt`. [Task 6, `test_writer_refuses_dt_mismatch`]
4. Running a build or a scoring command twice, or with a root that points into the repository or at an existing cache built with different statistics. Expected: a clear refusal unless `--force`, and the resolved root printed before the first write. [Task 1, `test_refuses_root_inside_repo`; Task 6, `test_build_refuses_other_stats_without_force`]
5. A fitted predictor reaching test shots, or a shot outside the three splits reaching training. Expected: an error, not a number. [Task 2, `test_split_shots_are_disjoint_and_from_the_split_file`; Task 3, `test_train_windows_refuse_non_training_shots`; Task 10, `test_cli_refuses_test_split_for_fitted_predictors`]

## Measured facts this plan relies on (2026-11-22)

- TokaMark's shot-level loader returns, per shot, `{"<source>-<signal>": {"time": (T,), "values": (*values_shape, T)}}` in float64, time last, on a regular grid per signal. An absent signal is `{"time": array([]), "values": array([])}`. One shot with all 39 signals loads in about 0.2 s.
- The union of signals over the 14 task configs is 39. The statistics file has 42 keys (three equilibrium signals no task uses).
- Sampling intervals: 5 ms (EFIT, Thomson), 0.25 ms (coils, Ip, NBI, gas, line density, requested waveforms), 0.2 ms (magnetic probes, flux loops), 0.02 ms (saddle coils, soft X-ray, D-alpha), 1.024 ms (the two Mirnov signals, which TokaMark delivers as spectrograms of shape `(3, 256, T)`).
- A window's `window_index` is its position in `np.arange(global_start + input_length, global_end - delta - output_length, stride)`, where the start and end are found per shot and per task from the first and last valid samples. `stride` is 1 ms.
- Non-markovian tasks (3-3 and all of group 4) give each window the whole history from the shot's start to the cut. `input_length` (0.15 s) only sets the earliest cut.
- The evaluator ignores NaN targets (`nanmean`), averages squared error over a shot's windows, divides the RMSE by one standard deviation per signal from the statistics file, and averages over signals and then shots.
- TokaMind encodes each output over the whole output window with a rank-mode 3-D cosine transform, after standardizing with TokaMark's statistics and replacing non-finite values by 0. The kept coefficient indices of each fine-tuning run are in `/models/tokamind/repo/runs/ft-<task>-ws-tokamind-base-v2-repl/embeddings/dct3d.yaml` under `embeddings.per_signal_overrides.output`. Task 1-2 keeps 10 of 170 coefficients for `equilibrium-lcfs_r`.
- Published baselines (TokaMind paper, Table 2) are in `/models/tokamind/compare_to_paper.py` as `PAPER[task] = (Mean, Persistence, CNN, CNN+LSTM, Scratch, FT-Tiny, FT-Base)`. GenToka's scores are in the spec's table under "The bar".
- Split sizes: 8,963 training, 1,115 validation, 1,110 test shots.

## File structure

```
scripts/mast/
  make_env.sh               builds /models/envs/ignite-mast
  py.sh                     the stage's Python: NVRTC preload, -I
  requirements-node13.txt   frozen package list of the environment (record, not input)
  check_harness.py          Task 4 command
  build_cache.py            Task 6 command
  run_headroom.py           Tasks 7-10 command
src/tokamak_foundation_model/ignite/mast/
  __init__.py               empty
  paths.py                  roots, environment overrides, write guard
  tasks.py                  TaskSpec, SignalSpec, splits, published scores
  harness.py                WindowBatch, window iterators, scoring through TokaMark
  baselines.py              mean and persistence predictors
  registry.py               channel table
  transform.py              channel statistics, model units <-> physical units
  cache.py                  per-shot HDF5 writer and reader
  headroom/
    __init__.py             empty
    floor.py                TokaMind's output-representation floor
    cheap.py                shrunk persistence, ridge on frame means
    direct.py               direct supervised model and its trainer
    table.py                bootstrap, table, projected mean
tests/ignite/mast/
  __init__.py, conftest.py, test_paths.py, test_tasks.py, test_harness.py,
  test_baselines.py, test_registry.py, test_transform.py, test_cache.py,
  test_floor.py, test_cheap.py, test_direct.py, test_table.py
docs/models/ignite-mast-stage0.md   results page (added to the nav)
```

Result layout under `/models/intermediate/mast/`:

```
cache/v1/{registry.csv, stats.npz, manifest.json, shots/<shot>.h5}
stage0/scores/<task>/<predictor>/<split>/{score.json, task_metrics.csv, shot metrics}
stage0/direct/<task>/{config.json, train_log.csv, best.pt}
stage0/headroom_table.csv
```

---

### Task 1: Environment, wrapper and paths

**Files:**
- Create: `scripts/mast/make_env.sh`, `scripts/mast/py.sh`, `scripts/mast/requirements-node13.txt`
- Create: `src/tokamak_foundation_model/ignite/mast/__init__.py`, `src/tokamak_foundation_model/ignite/mast/paths.py`
- Create: `tests/ignite/mast/__init__.py`, `tests/ignite/mast/conftest.py`, `tests/ignite/mast/test_paths.py`
- Modify: `docs/clusters/nvidia.md` (the tools table under "Environments")

**Interfaces:**
- Produces: `scripts/mast/py.sh <python args>`; and in `paths.py`:
  - `tokamark_zarr_root() -> Path`: `$IGNITE_MAST_TOKAMARK`, default `/models/mast/tokamark/v1`
  - `mast_root() -> Path`: `$IGNITE_MAST_ROOT`, default `/models/intermediate/mast`
  - `cache_dir(version: str = "v1") -> Path`: `mast_root() / "cache" / version`
  - `stage0_dir() -> Path`: `mast_root() / "stage0"`
  - `tokamind_runs_dir() -> Path`: `$IGNITE_MAST_TOKAMIND_RUNS`, default `/models/tokamind/repo/runs`
  - `prepare_write_root(path: Path) -> Path`: prints `"[ignite.mast] writing under <path> (<IGNITE_MAST_ROOT set | default>)"` once per process, raises `ValueError` if `path` resolves inside the git checkout that holds `paths.py`, creates the directory and returns it. Every command calls it before its first write.

Why a uv environment and not the pixi one: scoring needs `tokamark` and the model code in one interpreter. Adding `tokamark` and `zarr` to `feature.nvidia` means re-locking, and the lock is knowingly out of sync with the manifest (CLAUDE.md). The alternative, if it is ever wanted, is the additive procedure in `docs/reference/pixi-environments.md#featurenvidia`.

- [ ] **Step 1: Write `make_env.sh` and `py.sh`.**

`make_env.sh` (arguments: none; `REPO` is the checkout that holds the script):

```bash
ENV=/models/envs/ignite-mast
uv venv "$ENV" --python 3.12
uv pip freeze --python /models/tokamind/.venv/bin/python | grep -v -e '^-e ' -e ' @ file://' > "$TMP/base.txt"
uv pip install --python "$ENV/bin/python" -r "$TMP/base.txt"
uv pip install --python "$ENV/bin/python" einops==0.8.2 x-transformers==2.23.5 \
    vector-quantize-pytorch==1.31.0 "flash-attn-4[cu13]==4.0.0b40" pytest
SITE=$("$ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
printf '%s\n' /models/tokamind/tokamark/src /models/tokamind/repo/src "$REPO/src" > "$SITE/ignite_mast.pth"
uv pip freeze --python "$ENV/bin/python" > "$REPO/scripts/mast/requirements-node13.txt"
```

`$TMP` is a `mktemp -d` directory. The base list copies the TokaMind environment's exact versions so that TokaMark behaves as it does in the replication. The three source trees enter through a `.pth` file, which writes nothing into `/models/tokamind/` (an editable install would write `*.egg-info` there).

`py.sh`:

```bash
#!/usr/bin/env bash
export LD_PRELOAD=/usr/local/cuda-13.5/lib64/libnvrtc.so.13:/usr/local/cuda-13.5/lib64/libnvrtc-builtins.so.13.5${LD_PRELOAD:+:$LD_PRELOAD}
exec "${IGNITE_MAST_ENV:-/models/envs/ignite-mast}/bin/python" -I "$@"
```

- [ ] **Step 2: Build the environment and verify it.**

Run: `bash scripts/mast/make_env.sh && scripts/mast/py.sh -c "import torch, tokamark, MAST_tools, mmt, einops, h5py, tokamak_foundation_model.ignite.quantizer as q; print(torch.__version__, torch.cuda.is_available())"`
Expected: `2.15.0+cu132 True`.

Run: `scripts/mast/py.sh -c "import torch; x=torch.randn(64,64,device='cuda'); print(torch.linalg.eigvalsh(x@x.T).shape)"`
Expected: `torch.Size([64])` and no `nvrtc` error.

- [ ] **Step 3: Write the failing tests** in `tests/ignite/mast/test_paths.py`:

```python
def test_defaults(monkeypatch):
    monkeypatch.delenv("IGNITE_MAST_ROOT", raising=False)
    monkeypatch.delenv("IGNITE_MAST_TOKAMARK", raising=False)
    assert paths.mast_root() == Path("/models/intermediate/mast")
    assert paths.tokamark_zarr_root() == Path("/models/mast/tokamark/v1")
    assert paths.cache_dir() == Path("/models/intermediate/mast/cache/v1")

def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path))
    assert paths.stage0_dir() == tmp_path / "stage0"

def test_refuses_root_inside_repo():
    repo = Path(paths.__file__).resolve().parents[4]
    with pytest.raises(ValueError, match="inside the repository"):
        paths.prepare_write_root(repo / "scratch_out")

def test_prepare_write_root_announces_and_creates(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path))
    out = paths.prepare_write_root(paths.stage0_dir())
    assert out.is_dir() and "IGNITE_MAST_ROOT set" in capsys.readouterr().out
```

- [ ] **Step 4: Run them.** `scripts/mast/py.sh -m pytest tests/ignite/mast/test_paths.py -q`. Expected: FAIL, `paths` has no attribute `mast_root`.
- [ ] **Step 5: Implement `paths.py`** with the six functions above.
- [ ] **Step 6: Run in both environments.** `scripts/mast/py.sh -m pytest tests/ignite/mast -q` and `pixi run --frozen -e nvidia python -m pytest tests/ignite/mast -q`. Expected: 4 passed in each.
- [ ] **Step 7: Docs.** Add a row to the tools table in `docs/clusters/nvidia.md`: the environment's path, what it is for (TokaMark scoring together with IGNITE code), that `scripts/mast/make_env.sh` builds it and `scripts/mast/py.sh` runs it, and the date.
- [ ] **Step 8: Commit.** `git add scripts/mast src/tokamak_foundation_model/ignite/mast tests/ignite/mast docs/clusters/nvidia.md && git commit -m "ignite.mast: the ignite-mast environment, its wrapper and the data roots"`

---

### Task 2: Task specs, splits and published scores

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/tasks.py`
- Test: `tests/ignite/mast/test_tasks.py`
- Modify: `tests/ignite/mast/conftest.py`

**Interfaces:**
- Produces:

```python
ALL_TASKS: tuple[str, ...]        # "task_1-1" ... "task_4-5", 14 names in benchmark order
PROBE_TASKS = ("task_1-1", "task_1-2", "task_2-1", "task_3-1", "task_4-1", "task_4-4")
STAGE0_TASKS = PROBE_TASKS + ("task_4-5",)
SPECTROGRAM_SIGNALS = ("magnetics-b_field_tor_probe_cc_field", "magnetics-b_field_pol_probe_omv_voltage")
PUBLISHED: dict[str, dict[str, float | None]]
# per task: "mean", "persistence", "tokamind_paper", "tokamind_rerun", "gentoka"

@dataclass(frozen=True)
class SignalSpec:
    name: str                      # "<source>-<signal>"
    dt: float                      # seconds
    values_shape: tuple[int, ...]  # without time
    mean: float                    # TokaMark's statistics file
    std: float

@dataclass(frozen=True)
class TaskSpec:
    name: str
    markovian: bool
    input_length: float; output_length: float; delta: float; stride: float   # seconds
    inputs: tuple[SignalSpec, ...]; actuators: tuple[SignalSpec, ...]; outputs: tuple[SignalSpec, ...]
    steps: Mapping[str, Mapping[str, int]]     # role ("input" | "actuator" | "output") -> signal name -> time stamps,
                                               # TokaMark's `length_in_time_stamps`; a hand-built spec passes them in
    def n_out(self, s: SignalSpec) -> int      # steps["output"][s.name]
    def n_act(self, s: SignalSpec) -> int      # steps["actuator"][s.name]
    def n_in(self, s: SignalSpec, history_s: float | None = None) -> int
        # markovian: steps["input"][s.name]; non-markovian: round((history_s or input_length) / s.dt)

def load_task(name: str) -> TaskSpec          # from tokamark.tasks; imports tokamark inside
def split_shots(split: str) -> list[int]      # "train" | "val" | "test" | "val_select" | "val_report", sorted
def best_published(task: str) -> float        # min of tokamind_paper, tokamind_rerun, gentoka
```

`val_select` is the validation shots at even positions of the sorted list and `val_report` those at odd positions. Models are selected on the first and reported on the second, so no reported number was selected on its own shots.

`conftest.py` gains two fixtures that need no TokaMark: `toy_task` (a hand-built `TaskSpec` named `"task_toy"` with inputs `summary-ip` `(1,)` at 0.25 ms and `magnetics-flux_loop_flux` `(15,)` at 0.2 ms, actuator `pf_active-coil_voltage` `(4,)` at 0.25 ms, output `summary-ip`; 5 ms in, 25 ms out, `delta` 0, markovian; means and stds copied from the statistics file) and `toy_batch(n, seed)` returning a `WindowBatch` (Task 3) of random finite values.

- [ ] **Step 1: Write the failing tests:**

```python
tokamark = pytest.importorskip("tokamark")

def test_task_1_1_lengths():
    t = tasks.load_task("task_1-1")
    assert (t.input_length, t.output_length, t.delta, t.stride) == (0.005, 0.005, -0.005, 0.001)
    assert t.markovian

def test_task_4_5_outputs_are_spectrograms():
    t = tasks.load_task("task_4-5")
    assert {s.name for s in t.outputs} == set(tasks.SPECTROGRAM_SIGNALS)
    assert all(s.values_shape == (3, 256) and t.n_out(s) == 98 for s in t.outputs)
    assert not t.markovian and t.input_length == 0.15

def test_union_of_signals_is_39():
    names = {s.name for n in tasks.ALL_TASKS for t in [tasks.load_task(n)]
             for s in t.inputs + t.actuators + t.outputs}
    assert len(names) == 39

def test_split_shots_are_disjoint_and_from_the_split_file():
    tr, va, te = (tasks.split_shots(s) for s in ("train", "val", "test"))
    assert (len(tr), len(va), len(te)) == (8963, 1115, 1110)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    sel, rep = tasks.split_shots("val_select"), tasks.split_shots("val_report")
    assert sorted(sel + rep) == va and (len(sel), len(rep)) == (558, 557)
    with pytest.raises(ValueError):
        tasks.split_shots("all")

def test_published_scores():
    assert tasks.PUBLISHED["task_4-5"]["mean"] == 0.653
    assert tasks.PUBLISHED["task_4-4"]["persistence"] == 0.718
    assert tasks.PUBLISHED["task_1-1"]["persistence"] is None
    gentoka = [tasks.PUBLISHED[t]["gentoka"] for t in tasks.ALL_TASKS]
    assert abs(sum(gentoka) / 14 - 0.2033) < 5e-4
    assert tasks.best_published("task_4-1") == 0.2122 and tasks.best_published("task_1-1") == 0.0941
```

- [ ] **Step 2: Run.** `scripts/mast/py.sh -m pytest tests/ignite/mast/test_tasks.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement `tasks.py`.** `load_task` uses `tokamark.tasks.get_task_config`, `get_task_metadata` and `get_signals_metadata`. `split_shots` uses `tokamark.data_split.get_train_test_val_shots()` with its default (random-split) file; the size test above fixes which returned list is which. Copy `PUBLISHED` from `compare_to_paper.py` (`mean` = element 0, `persistence` = element 1, `tokamind_paper` = element 6) and from the spec's table (`tokamind_rerun`, `gentoka`), with a comment naming both sources and the date.
- [ ] **Step 4: Run.** Expected: 5 passed. In the pixi environment the file is skipped.
- [ ] **Step 5: Commit.** `ignite.mast: task specs, split lists and the published scores`

---

### Task 3: The scoring harness

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/harness.py`
- Test: `tests/ignite/mast/test_harness.py`

**Interfaces:**
- Consumes: `TaskSpec`, `split_shots` (Task 2); `tokamark_zarr_root`, `prepare_write_root` (Task 1).
- Produces:

```python
@dataclass
class WindowBatch:
    shot_ids: np.ndarray                 # (B,) int64
    window_indices: np.ndarray           # (B,) int64
    inputs: dict[str, np.ndarray]        # name -> (B, *values_shape, T_in)  float32, physical units, NaN = missing
    actuators: dict[str, np.ndarray]     # name -> (B, *values_shape, T_act)
    targets: dict[str, np.ndarray]       # name -> (B, *values_shape, T_out)
    last_valid: dict[str, np.ndarray]    # name -> (B, *values_shape): last finite input sample over the WHOLE history, NaN if none

PredictFn = Callable[[WindowBatch], dict[str, np.ndarray]]   # physical units, same shapes as targets

@dataclass
class ScoreResult:
    task_nrmse: float
    per_signal: pd.DataFrame     # feature_name, NRMSE
    per_shot: pd.DataFrame       # shot_id, NRMSE (the task score of each shot)
    n_windows: int
    out_dir: Path

def iter_windows(task, shots, *, test_mode: bool, history_s: float | None = None, batch_size: int = 64,
                 num_workers: int = 32, shuffle: bool = False, stride_s: float | None = None) -> Iterator[WindowBatch]
def train_windows(task, shots=None, **kw) -> Iterator[WindowBatch]   # shots default: split_shots("train")
def score_batches(task_name: str, batches: Iterable[WindowBatch], predict_fn: PredictFn, out_dir: Path) -> ScoreResult
def score(task, predict_fn, out_dir, *, split: str, shots=None, history_s=None, **kw) -> ScoreResult
```

Decisions the code must follow:

- `iter_windows` builds the dataset exactly as TokaMark's `scripts/persistence.py` does: `initialize_MAST_dataset(config_task=..., shots_list=shots, local_flag=True, use_std_scaling=False, remove_outliers=True, outlier_metadata_file=RANDOM_SPLIT_OUTLIER_METADATA_FILE, store_manager_settings={"base_local_zarr_path": str(tokamark_zarr_root())})`, then `initialize_TokaMark_dataset(dataset, task_metadata, config_task, custom_transform=<ours>, test_mode=test_mode, shuffle_windows=shuffle)`, then a `DataLoader` with our collate function. The custom transform turns one TokaMark window into the arrays of a one-window `WindowBatch`.
- **History.** For a markovian task `T_in` is the task's input length. For a non-markovian task TokaMark hands over the whole history; the transform computes `last_valid` on the whole history first and then keeps the last `history_s` seconds (default `task.input_length`), so arrays have a fixed length and batch. Using less history than the protocol allows is within the protocol. Using more is a later variant.
- **`stride_s`** (training only) replaces the window stride in a copy of the task config. TokaMind trains with windows 5 ms apart; stage 0 does the same.
- `train_windows` raises `ValueError` if any requested shot is not in `split_shots("train")`.
- `score_batches` feeds `WindowMetricsAccumulator(task=task_name).add_batch(y_target, y_pred, shot_ids, window_indices, feature_name)` with both arrays reshaped to `(B, -1)`, then `compute_metrics(task=..., output_dir=..., window_metrics_accumulator=..., save_shot_metrics=True, save_task_metrics=True)`. Before adding, it raises `ValueError` when a prediction is missing for an output, has another shape than the target, or is non-finite where the target is finite.
- `score(split=...)` accepts `"test"`, `"val"`, `"val_select"`, `"val_report"`; it calls `prepare_write_root(out_dir)` and uses `test_mode=True`.

- [ ] **Step 1: Write the failing tests** (the first four use `toy_batch` and real signal names, so they need `tokamark` for the evaluator but no data):

```python
def test_known_answer_nrmse_is_one(toy_task, toy_batch, tmp_path):
    b = toy_batch(n=8, seed=0)
    s = toy_task.outputs[0]
    b.targets[s.name][:] = s.mean + s.std
    r = harness.score_batches("task_4-4", [b], lambda bb: {s.name: np.full_like(bb.targets[s.name], s.mean)}, tmp_path)
    assert abs(r.task_nrmse - 1.0) < 1e-6 and r.n_windows == 8

def test_nan_targets_are_ignored(toy_task, toy_batch, tmp_path):
    b = toy_batch(n=8, seed=0); s = toy_task.outputs[0]
    b.targets[s.name][:] = s.mean; b.targets[s.name][:, ..., ::2] = np.nan
    r = harness.score_batches("task_4-4", [b], lambda bb: {s.name: np.full_like(bb.targets[s.name], s.mean)}, tmp_path)
    assert r.task_nrmse < 1e-9

def test_score_rejects_nonfinite_prediction(toy_task, toy_batch, tmp_path):
    b = toy_batch(n=4, seed=0); s = toy_task.outputs[0]
    with pytest.raises(ValueError, match="non-finite"):
        harness.score_batches("task_4-4", [b], lambda bb: {s.name: np.full_like(bb.targets[s.name], np.nan)}, tmp_path)

def test_score_rejects_missing_output_and_wrong_shape(toy_task, toy_batch, tmp_path): ...
    # {} -> ValueError("missing"); prediction with one time stamp fewer -> ValueError("shape")

def test_train_windows_refuse_non_training_shots():
    t = tasks.load_task("task_1-1")
    with pytest.raises(ValueError, match="not in the training split"):
        next(iter(harness.train_windows(t, shots=[tasks.split_shots("test")[0]])))

@pytest.mark.real_data
def test_real_windows_shapes_and_history_crop():
    t = tasks.load_task("task_4-4"); shot = tasks.split_shots("val")[0]
    b = next(iter(harness.iter_windows(t, [shot], test_mode=True, batch_size=4, num_workers=0)))
    for s in t.inputs:
        assert b.inputs[s.name].shape == (4, *s.values_shape, t.n_in(s, 0.15))
        assert b.last_valid[s.name].shape == (4, *s.values_shape)
    for s in t.outputs:
        assert b.targets[s.name].shape == (4, *s.values_shape, t.n_out(s))

@pytest.mark.real_data
def test_training_stride_thins_windows():
    t = tasks.load_task("task_1-1"); shot = tasks.split_shots("train")[0]
    n1 = sum(len(b.shot_ids) for b in harness.train_windows(t, [shot], num_workers=0))
    n5 = sum(len(b.shot_ids) for b in harness.train_windows(t, [shot], num_workers=0, stride_s=0.005))
    assert 4.0 < n1 / n5 < 6.0
```

A `real_data` test skips when `paths.tokamark_zarr_root()` does not exist (put the skip in `conftest.py`).

- [ ] **Step 2: Run.** Expected: FAIL (module missing).
- [ ] **Step 3: Implement `harness.py`** following the decisions above.
- [ ] **Step 4: Run.** `scripts/mast/py.sh -m pytest tests/ignite/mast/test_harness.py -q`. Expected: 7 passed.
- [ ] **Step 5: Measure throughput and record it.** Time `train_windows` for 200 training shots with `num_workers=64, stride_s=0.005` on `task_1-1` and on `task_4-1`, and `iter_windows(test_mode=True)` on 100 test shots of each. Write windows per second and the date into the module docstring of `harness.py`. If training throughput on `task_4-1` is under 500 windows per second, say so in the task report; Task 9 then needs a smaller step budget for group 4.
- [ ] **Step 6: Commit.** `ignite.mast: scoring harness over TokaMark's windows and evaluator`

---

### Task 4: Harness check against the published baselines (a gate)

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/baselines.py`, `scripts/mast/check_harness.py`
- Test: `tests/ignite/mast/test_baselines.py`

**Interfaces:**
- Consumes: `score`, `WindowBatch`, `PredictFn` (Task 3); `PUBLISHED`, `ALL_TASKS` (Task 2).
- Produces: `mean_predictor(task: TaskSpec) -> PredictFn`; `persistence_predictor(task: TaskSpec) -> PredictFn`; `write_score_json(result: ScoreResult, predictor: str, split: str) -> Path` in `baselines.py` (writes `score.json` beside the evaluator's files: task, predictor, split, nrmse, n_shots, n_windows, git revision, date).

The persistence rule is TokaMark's: the last finite input value of each channel (forward fill over the history), repeated over the output window; where a channel has no finite input, TokaMark's signal mean. It is defined only when every output signal is also an input. `persistence_predictor` raises `ValueError` otherwise.

- [ ] **Step 1: Write the failing tests:**

```python
def test_mean_predictor_fills_with_the_signal_mean(toy_task, toy_batch):
    p = baselines.mean_predictor(toy_task)(toy_batch(n=3, seed=0)); s = toy_task.outputs[0]
    assert p[s.name].shape == (3, *s.values_shape, toy_task.n_out(s)) and np.all(p[s.name] == np.float32(s.mean))

def test_persistence_repeats_last_valid_and_falls_back_to_mean(toy_task, toy_batch):
    b = toy_batch(n=2, seed=0); s = toy_task.outputs[0]
    b.last_valid[s.name][0] = 7.0; b.last_valid[s.name][1] = np.nan
    p = baselines.persistence_predictor(toy_task)(b)[s.name]
    assert np.all(p[0] == 7.0) and np.all(p[1] == np.float32(s.mean))

def test_persistence_needs_outputs_among_inputs():
    pytest.importorskip("tokamark")
    with pytest.raises(ValueError):
        baselines.persistence_predictor(tasks.load_task("task_1-1"))
```

- [ ] **Step 2: Run, see them fail, implement, run again.** Expected: 3 passed.
- [ ] **Step 3: Write `check_harness.py`.** For each task in `--tasks` (default all 14) it scores `mean` on `--split` (default `test`), and `persistence` where `PUBLISHED[task]["persistence"]` is not None, into `stage0/scores/<task>/<predictor>/<split>/`, skipping a score whose `score.json` exists unless `--force`. It prints a table `task, predictor, ours, published, diff` and exits 1 if any `|diff| > 0.002`. The published values have three decimals, so rounding alone allows 0.0005; 0.002 leaves room for a small difference in benchmark version and no more.
- [ ] **Step 4: Run it.** `scripts/mast/py.sh scripts/mast/check_harness.py --num-workers 64 2>&1 | tee /models/intermediate/mast/stage0/check_harness.log`
Expected: 20 rows (14 mean, 6 persistence), exit 0.
- [ ] **Step 5: If a row is outside 0.002, stop.** No later number is believed until this passes. Use `superpowers:systematic-debugging`. First things to compare with `scripts/persistence.py`: the outlier file, `remove_bad_efit_rating`, sparse against filled test mode, and the number of windows per shot. If the cause is a benchmark-version difference that cannot be removed, report the row, the cause and the evidence, and wait for a decision.
- [ ] **Step 6: Score both baselines on `val_report` too** (`--split val_report`). The table of Task 10 shows them beside the test values so the reader sees how much the split alone moves a score.
- [ ] **Step 7: Commit.** `ignite.mast: mean and persistence baselines reproduce the published columns`, with the 20 rows in the body.

---

### Task 5: Channel registry, statistics and the unit transform

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/registry.py`, `src/tokamak_foundation_model/ignite/mast/transform.py`
- Test: `tests/ignite/mast/test_registry.py`, `tests/ignite/mast/test_transform.py`

**Interfaces:**
- Consumes: `load_task`, `ALL_TASKS`, `split_shots` (Task 2).
- Produces:

```python
# registry.py
REGISTRY_COLUMNS = ("device", "signal", "channel", "channel_name", "family", "dt", "units", "imas_path",
                    "r", "z", "tm_mean", "tm_std", "roles", "sat_abs", "valid_fraction", "dead_fraction",
                    "compress", "scale", "mean", "std")
def all_signals() -> list[SignalSpec]                      # the 39, in a fixed order
def family_of(s: SignalSpec) -> str                        # "slow" | "high_rate" | "profile" | "map"
def build_registry(zarr_root: Path, sample_shots: list[int]) -> pd.DataFrame      # the last four columns empty
def attach_stats(df: pd.DataFrame, stats: dict[str, "ChannelStats"]) -> pd.DataFrame   # fills the last four columns
def save_registry(df, path) -> None; def load_registry(path) -> pd.DataFrame      # CSV

# transform.py
@dataclass
class ChannelStats:            # arrays have the signal's values_shape
    compress: str              # "none" | "asinh"
    scale: np.ndarray          # asinh scale (1 where compress == "none")
    mean: np.ndarray; std: np.ndarray       # of the compressed values
def load_shot(shot: int, signals: list[SignalSpec]) -> dict[str, dict[str, np.ndarray]]   # TokaMark's shot-level loader
def fit_stats(shots: list[int], signals, *, samples_per_channel: int = 64, seed: int = 0, workers: int = 64) -> dict[str, ChannelStats]
def save_stats(stats, path) -> None; def load_stats(path) -> dict[str, ChannelStats]        # one .npz
def to_model(x: np.ndarray, st: ChannelStats) -> np.ndarray       # x: (*values_shape, T) physical -> standardized
def to_physical(z: np.ndarray, st: ChannelStats) -> np.ndarray
def stats_sha256(path: Path) -> str
```

Decisions:

- **One registry row per channel** of the 39 signals: 1 row for a scalar, `C` rows for `(C,)`, and `H*W` rows for a map or a spectrogram (`channel` is the flat index, `channel_name` is `"i,j"`). `device` is `"MAST"`. `roles` is the sorted set of `inp`/`act`/`out` over the 14 tasks, joined by `+`. `units`, `imas_path`, `channel_name`, `r` and `z` come from the Zarr attributes and coordinates where they exist and are empty otherwise; an empty cell means not found, not zero. `sat_abs` is 4.99 for `spectrometer_visible-filter_spectrometer_dalpha_voltage` (ADC range of plus or minus 5 V, data profile of 2026-11-21) and empty elsewhere. `valid_fraction` and `dead_fraction` are measured over `sample_shots` (a channel is dead in a shot when it has no finite sample or its finite samples are all equal).
- **Families**, by rule: two non-time dimensions and `dt` of 5 ms is `map` (ψ); Thomson profiles and the boundary (`lcfs_r`, `lcfs_z`) are `profile`; `dt` of 0.02 ms and the two spectrogram signals are `high_rate`; everything else is `slow`.
- **`load_shot`** calls `initialize_MAST_dataset({"sources_and_signals": {"input_name": [[source, signal], ...]}}, [shot], ...)` with the same keyword arguments as the harness, so outlier removal, the x-point clip, the boundary resampling and the Mirnov spectrograms are TokaMark's by construction.
- **Statistics** come from 2,000 training shots (`split_shots("train")`, `numpy.random.default_rng(0).choice`, sorted) with 64 finite samples drawn per channel per shot. From that sample, per channel: the median and the scaled median absolute deviation `s = 1.4826 * MAD`; the excess kurtosis of `(x - median) / s`. A signal whose median channel has excess kurtosis above 10 gets `compress = "asinh"` with `scale = s` per channel (`to_model` computes `asinh(x / scale)`, then standardizes; a channel whose `s` is 0 uses its sample standard deviation as the scale); the others get `"none"`. Then `mean` and `std` of the compressed sample. The code reads `stats.npz`; the registry's last four columns are the readable record of the same numbers, and a test holds the two together. A channel with fewer than 1,000 finite samples or zero spread gets `mean = 0`, `std = 1`, and is listed in the task report. Why a sample and not a full pass: the constants only have to be fixed and reasonable, and 128,000 samples put the error of a standard deviation under 1%.
- The sign of the plasma current is kept as recorded: nothing in the transform takes an absolute value.

- [ ] **Step 1: Write the failing tests:**

```python
def test_roundtrip_none_and_asinh():
    rng = np.random.default_rng(0)
    for compress in ("none", "asinh"):
        st = transform.ChannelStats(compress, scale=np.full((3,), 0.5), mean=np.array([0.1, -2.0, 3.0]), std=np.array([1.0, 0.2, 5.0]))
        x = rng.standard_normal((3, 100)) * 10
        np.testing.assert_allclose(transform.to_physical(transform.to_model(x, st), st), x, rtol=1e-9, atol=1e-9)

def test_nan_stays_nan_and_sign_is_kept():
    st = transform.ChannelStats("none", np.ones((1,)), np.zeros((1,)), np.ones((1,)))
    z = transform.to_model(np.array([[np.nan, -3.0, 3.0]]), st)
    assert np.isnan(z[0, 0]) and z[0, 1] == -z[0, 2]

def test_fit_stats_picks_asinh_for_heavy_tails(monkeypatch):
    # monkeypatch transform.load_shot to return one gaussian signal and one with 1% of samples x1000
    # assert stats["gauss"].compress == "none" and stats["spiky"].compress == "asinh"
    # assert abs(stats["gauss"].std - true_std) / true_std < 0.05

def test_save_load_stats_roundtrip_and_hash(tmp_path): ...   # arrays equal, sha256 stable across two saves

def test_attach_stats_matches_arrays():
    # a two-signal registry frame ((1,) and (3,)) and matching ChannelStats: after attach_stats, the "mean", "std" and
    # "scale" cells of channel c equal stats[signal].mean.ravel()[c] etc., and "compress" equals stats[signal].compress

def test_family_rules():
    pytest.importorskip("tokamark")
    fam = {s.name: registry.family_of(s) for s in registry.all_signals()}
    assert fam["equilibrium-psi"] == "map" and fam["equilibrium-lcfs_r"] == "profile"
    assert fam["thomson_scattering-t_e"] == "profile" and fam["soft_x_rays-horizontal_cam_upper"] == "high_rate"
    assert fam["magnetics-b_field_tor_probe_cc_field"] == "high_rate" and fam["summary-ip"] == "slow"
    assert fam["equilibrium-x_point_r"] == "slow" and len(fam) == 39

@pytest.mark.real_data
def test_registry_rows_and_columns():
    df = registry.build_registry(paths.tokamark_zarr_root(), tasks.split_shots("train")[:5])
    assert tuple(df.columns) == registry.REGISTRY_COLUMNS
    n = {s.name: int(np.prod(s.values_shape)) for s in registry.all_signals()}
    assert len(df) == sum(n.values()) and (df.groupby("signal").size() == pd.Series(n)).all()
    assert df.loc[df.signal == "summary-ip", "roles"].iloc[0] == "inp+out"
```

- [ ] **Step 2: Run, see them fail, implement both modules, run again.** Expected: 7 passed (5 in the pixi environment, the other two skipped).
- [ ] **Step 3: Commit.** `ignite.mast: channel registry, per-channel statistics and the unit transform`

---

### Task 6: The standardized cache

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/cache.py`, `scripts/mast/build_cache.py`
- Test: `tests/ignite/mast/test_cache.py`
- Modify: `/models/README.md` (one paragraph on `/models/intermediate/mast/` and `/models/envs/ignite-mast`)

**Interfaces:**
- Consumes: `load_shot`, `fit_stats`, `to_model`, `to_physical`, `ChannelStats`, `stats_sha256` (Task 5); `build_registry` (Task 5); `cache_dir`, `prepare_write_root` (Task 1).
- Produces:

```python
CACHE_VERSION = "v1"
@dataclass
class CachedSignal:
    values: np.ndarray        # float16, (*values_shape, T), model units, NaN = not valid
    t0: float; dt: float      # seconds
    saturated: np.ndarray | None    # bool, same shape; only for signals with sat_abs
    @property
    def valid(self) -> np.ndarray   # np.isfinite(values)
def write_shot(path: Path, shot: int, split: str, raw: dict, signals, stats, registry: pd.DataFrame) -> dict   # per-signal status
def read_shot(path: Path, names: Sequence[str] | None = None) -> dict[str, CachedSignal | None]   # None = signal absent in this shot
def build(root: Path, shots: dict[str, list[int]], *, workers: int, force: bool = False) -> dict   # the manifest
```

Decisions:

- **Scope of v1:** the 39 benchmark signals as TokaMark's loader delivers them, for the 11,188 shots of the three splits. The two Mirnov signals are therefore TokaMark's spectrograms, not 500 kHz waveforms. Raw waveforms belong to the stage-1 plan, with the tokenizer that needs them.
- **File:** `shots/<shot>.h5`, one dataset per signal named `<source>-<signal>`, float16, contiguous, no compression (93 GB in total, from array shapes; `/models` has 27 TB free). Dataset attributes `t0` and `dt`. File attributes `shot`, `split`, `cache_version`, `stats_sha256`. The file is written to `<shot>.h5.tmp` and renamed, so a killed build leaves no half file.
- **Validity is the NaN pattern.** A sample is not valid when it is non-finite from the loader, non-finite after the cast to float16, or belongs to a channel that is dead in this shot. Missing chunk files arrive as NaN from Zarr and need no rule.
- **Deviation from the spec, to be confirmed by Nathaniel:** the spec says saturated samples become "not valid". Here they stay valid and are flagged in a boolean dataset `<name>__saturated`, because TokaMark scores them as targets (task 3-2 predicts D-alpha) and a model that never saw them would miss every clipped peak.
- **`write_shot` refuses a signal** (status `"dt_mismatch"`, no dataset written) when the loader's median time step differs from `SignalSpec.dt` by more than 1e-6 relative, and writes no dataset for an absent signal (status `"absent"`). Other statuses: `"ok"`, `"all_invalid"` (dataset written, all NaN).
- **`build`** fits or loads `stats.npz`, writes `registry.csv`, then writes the shots with a process pool, skipping shots whose file exists. `manifest.json` records the date, the git revisions of this repository and of `/models/tokamind/tokamark`, `stats_sha256`, counts per split, counts per status per signal, the wall time and the size on disk. With an existing `manifest.json` whose `stats_sha256` differs from the statistics about to be used, `build` raises unless `force=True`.

- [ ] **Step 1: Write the failing tests** (all synthetic, with a tiny `ChannelStats` and hand-made `raw` dictionaries):

```python
def test_roundtrip_within_float16(tmp_path): ...
    # write one (15, 400) signal, read back, to_physical; max abs error / std < 4e-3 (float16 spacing is 0.002 for |z| in [2, 4));
    # t0 and dt equal

def test_missing_signal_is_absent_not_zero(tmp_path):
    raw = {"summary-ip": {"time": np.array([]), "values": np.array([])}, "magnetics-flux_loop_flux": good}
    status = cache.write_shot(p, 1, "train", raw, signals, stats, registry)
    assert status["summary-ip"] == "absent" and cache.read_shot(p)["summary-ip"] is None

def test_writer_refuses_dt_mismatch(tmp_path): ...
    # time grid with dt = spec.dt / 10 -> status "dt_mismatch", read_shot(...)[name] is None

def test_dead_channel_is_invalid(tmp_path): ...
    # one constant channel, one all-NaN channel -> both entirely not valid; live channels untouched

def test_saturated_samples_are_flagged_and_kept(tmp_path): ...
    # sat_abs = 4.99: samples at +-5.0 have saturated == True and valid == True

def test_overflow_becomes_invalid(tmp_path): ...     # a physical value that maps beyond float16 range -> not valid

def test_build_refuses_other_stats_without_force(tmp_path, monkeypatch): ...
    # manifest.json with another stats_sha256 -> build(...) raises ValueError("different statistics"); force=True passes

def test_build_skips_existing_and_leaves_no_tmp(tmp_path, monkeypatch): ...
```

- [ ] **Step 2: Run, see them fail, implement `cache.py`, run again.** Expected: 8 passed in both environments.
- [ ] **Step 3: Write `build_cache.py`** (`--splits train val test`, `--limit N` for a trial, `--workers`, `--force`), which calls `prepare_write_root(cache_dir())` first.
- [ ] **Step 4: Trial build.** `IGNITE_MAST_ROOT=/models/intermediate/mast_trial scripts/mast/py.sh scripts/mast/build_cache.py --limit 50 --workers 32`
Expected: 150 shot files, a manifest, no `*.tmp`.
- [ ] **Step 5: Add and run the fidelity test on the trial cache:**

```python
@pytest.mark.real_data
def test_cache_fidelity_against_the_loader():
    # for 5 validation shots of the trial cache and every signal present:
    # err = nan-RMS(to_physical(cached.values) - loader values) / SignalSpec.std   (TokaMark's std)
    # assert err <= 1e-3, and the NaN pattern of the loader is a subset of the cache's
```

1e-3 is a seventh of the tightest tokenizer fidelity target of the spec (0.007 for the boundary), so the cache cannot be what limits a tokenizer. If a signal fails, the fix is its compression choice, not the threshold.
- [ ] **Step 6: Full build.** `scripts/mast/py.sh scripts/mast/build_cache.py --workers 96`, then delete `/models/intermediate/mast_trial`. Expected: 11,188 shot files; record wall time, size and the per-signal status counts.
- [ ] **Step 7: Commit.** `ignite.mast: standardized float16 cache of the 39 benchmark signals`, with the manifest's numbers in the body.

---

### Task 7: Headroom column A, TokaMind's representation floor

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/headroom/__init__.py`, `src/tokamak_foundation_model/ignite/mast/headroom/floor.py`, `scripts/mast/run_headroom.py`
- Test: `tests/ignite/mast/test_floor.py`

**Interfaces:**
- Consumes: `score`, `WindowBatch`, `PredictFn` (Task 3); `write_score_json`, `mean_predictor`, `persistence_predictor` (Task 4); `tokamind_runs_dir` (Task 1).
- Produces:

```python
def tokamind_output_indices(task_name: str) -> dict[str, tuple[np.ndarray, tuple[int, int, int]]]
    # per output signal: (flat coefficient indices, coeff_shape (H, W, T)) from the replication run's dct3d.yaml
def truncate_dct(x: np.ndarray, indices: np.ndarray, coeff_shape: tuple[int, int, int]) -> np.ndarray
    # x: (B, *values_shape, T), finite. Orthonormal 3-D DCT-II, keep `indices`, inverse. Same shape out.
def floor_predictor(task: TaskSpec, indices) -> PredictFn
    # reads batch.targets: standardize with TokaMark's mean/std, non-finite -> 0, truncate_dct, unstandardize
```

and the command `scripts/mast/run_headroom.py --task T --predictor {mean,persistence,floor,shrunk,ridge,direct} --split {val_report,test} [--gpu N] [--force]`, which writes `stage0/scores/<task>/<predictor>/<split>/` through `score` and `write_score_json`. Later tasks add their predictors to it.

The floor is the score TokaMind would get if its network predicted every kept coefficient exactly. It is the only "predictor" that reads the targets. `tokamind_output_indices` raises `ValueError` when a run's `coeff_shape` does not equal the 3-D view of the task's output (`(H, W, T_out)`), so a changed chunking cannot pass silently.

- [ ] **Step 1: Write the failing tests:**

```python
def test_all_coefficients_reproduce_the_input():
    x = np.random.default_rng(0).standard_normal((4, 18, 50))
    np.testing.assert_allclose(floor.truncate_dct(x, np.arange(18 * 1 * 50), (18, 1, 50)), x, atol=1e-10)

def test_dc_only_gives_the_window_mean():
    x = np.random.default_rng(0).standard_normal((4, 18, 50))
    y = floor.truncate_dct(x, np.array([0]), (18, 1, 50))
    np.testing.assert_allclose(y, np.broadcast_to(x.mean(axis=(1, 2), keepdims=True), x.shape), atol=1e-10)

def test_matches_tokaminds_codec():
    pytest.importorskip("mmt")
    from mmt.data.embeddings.dct3d import DCT3DCodec, DCT3DTorchDecoder
    # random indices (40 of 18*50), one window: DCT3DTorchDecoder(codec, (18, 50))(codec.encode(x)) equals truncate_dct(x[None], ...)[0] to 1e-5

@pytest.mark.real_data
def test_task_1_2_indices():
    idx = floor.tokamind_output_indices("task_1-2")
    assert set(idx) == {"equilibrium-lcfs_r", "equilibrium-lcfs_z"}
    assert len(idx["equilibrium-lcfs_r"][0]) == 10 and idx["equilibrium-lcfs_r"][1] == (170, 1, 1)
```

- [ ] **Step 2: Run, see them fail, implement, run again.** Expected: 4 passed.
- [ ] **Step 3: Score the floor** on `test` and on `val_report` for the seven `STAGE0_TASKS`: `for t in task_1-1 task_1-2 task_2-1 task_3-1 task_4-1 task_4-4 task_4-5; do for s in test val_report; do scripts/mast/py.sh scripts/mast/run_headroom.py --task $t --predictor floor --split $s; done; done`
Expected: 14 `score.json` files. Each floor is below our TokaMind replication score for the task (`/models/tokamind/.venv/bin/python -I /models/tokamind/compare_to_paper.py`, where finished). A floor above the replication score means the floor is computed differently from what TokaMind does; stop and find the difference.
- [ ] **Step 4: Commit.** `ignite.mast: TokaMind's output-representation floor on seven tasks`, with the 14 numbers in the body.

---

### Task 8: Headroom column B, cheap predictors

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/headroom/cheap.py`
- Modify: `scripts/mast/run_headroom.py`
- Test: `tests/ignite/mast/test_cheap.py`

**Interfaces:**
- Consumes: `train_windows`, `iter_windows`, `score`, `WindowBatch` (Task 3); `SPECTROGRAM_SIGNALS`, `split_shots` (Task 2).
- Produces:

```python
FRAME_S = 0.005
def frame_len(dt: float) -> int                                   # max(1, round(FRAME_S / dt)) samples; Task 9's patches use it too
def frame_means(x: np.ndarray, dt: float) -> np.ndarray           # (..., T) -> (..., ceil(T / frame_len)), NaN-mean per frame; a last partial frame counts
def hold_frames(m: np.ndarray, dt: float, n: int) -> np.ndarray   # (..., n_frames) -> (..., n), each frame's value repeated
def window_features(batch: WindowBatch, task: TaskSpec) -> np.ndarray     # (B, F) float64

class ShrunkPersistence:
    def fit(self, batches: Iterable[WindowBatch]) -> "ShrunkPersistence"
    def predict(self, batch: WindowBatch) -> dict[str, np.ndarray]
class RidgeFrames:
    lambdas = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    def fit(self, batches) -> "RidgeFrames"                       # accumulates X'X and X'Y in float64, one pass
    def select(self, val_batches) -> float                        # picks lambda by mean squared error in standardized units
    def predict(self, batch) -> dict[str, np.ndarray]
```

Decisions:

- All fitting is in TokaMark-standardized units (`(x - mean) / std` per signal), predictions are returned in physical units.
- **`ShrunkPersistence`**: for each output signal that is also an input, the anchor `p` is `last_valid` (for the two spectrogram signals: the NaN-mean of the input over the kept history, per bin). With `m_c` the training mean of the target per channel, the prediction for output frame `k` is `m_c + a[s, k] * (p_c - m_c)`, where `a[s, k]` is the least-squares slope pooled over channels and windows. An output that is not an input is predicted by `m_c`. A non-finite anchor falls back to `m_c`.
- **`window_features`**: for each input signal, the NaN-mean per channel over the last 5 ms of the input (spectrogram signals: per bin over the kept history), standardized, non-finite set to 0, followed by one 0/1 validity flag per channel (per signal for a map or spectrogram). For each actuator, the standardized frame means per channel over its whole window. A final constant 1.
- **`RidgeFrames`** predicts the frame means of each output channel (for spectrogram signals: one mean over the whole output window), centred by the training mean, and holds each value over its frame. A non-finite target frame counts as 0 after centring. That biases the fit toward the mean where targets are missing, which is acceptable for a cheap predictor and is stated on the results page. The penalty is `lambda * n_windows` on all weights but the constant.
- Both are fitted on `train_windows(task, stride_s=0.005)` over a fixed random 3,000 training shots (seed 0), `RidgeFrames.select` uses 200 shots of `val_select`, and both are scored on `val_report` only.

- [ ] **Step 1: Write the failing tests:**

```python
def test_frame_means_and_hold_roundtrip():
    x = np.repeat(np.arange(4.0), 25)[None]            # 4 frames of 25 samples at dt = 0.2 ms
    m = cheap.frame_means(x, 0.0002); assert m.shape == (1, 4) and np.allclose(m, [[0, 1, 2, 3]])
    assert np.array_equal(cheap.hold_frames(m, 0.0002, 100), x)

def test_frame_means_partial_last_frame_and_nans():
    x = np.ones((1, 98)); x[0, :5] = np.nan            # 98 stamps at 1.024 ms: frames of 5 stamps -> 20 frames, the last has 3
    assert cheap.frame_len(0.001024) == 5
    m = cheap.frame_means(x, 0.001024); assert m.shape[-1] == 20 and np.isnan(m[0, 0]) and m[0, -1] == 1.0

def test_shrunk_persistence_recovers_the_slope(toy_task, toy_batch):
    # targets = mean + 0.6 * (last_valid - mean) + small noise -> fitted a within 0.02 of 0.6 for every frame;
    # a window with NaN last_valid is predicted by the training mean

def test_ridge_recovers_a_linear_map(toy_task, toy_batch):
    # targets = linear function of window_features + noise -> after fit and select, standardized RMSE < 0.05

def test_window_features_shape_and_finiteness(toy_task, toy_batch):
    b = toy_batch(n=5, seed=0); b.inputs["summary-ip"][0] = np.nan
    f = cheap.window_features(b, toy_task)
    assert f.shape == (5, 2 * (1 + 15) + 4 * 6 + 1) and np.isfinite(f).all() and np.all(f[:, -1] == 1.0)
```

(The toy task has 16 input channels and 4 actuator channels over 30 ms, which is 6 frames.)

- [ ] **Step 2: Run, see them fail, implement, run again.** Expected: 5 passed in both environments.
- [ ] **Step 3: Add `shrunk` and `ridge` to `run_headroom.py`.** The command fits, saves the fitted arrays under `stage0/cheap/<task>/`, and scores on `val_report`.
- [ ] **Step 4: Run both on the seven tasks.** Expected: 14 `score.json` files. On `task_4-5`, write down both scores beside the mean baseline on `val_report` from Task 4: that comparison is the first evidence on whether the task has anything predictable in it.
- [ ] **Step 5: Commit.** `ignite.mast: cheap predictors (shrunk persistence, ridge on frame means)`, with the numbers.

---

### Task 9: Headroom column C, the direct supervised model

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/headroom/direct.py`
- Modify: `scripts/mast/run_headroom.py`
- Test: `tests/ignite/mast/test_direct.py`

**Interfaces:**
- Consumes: `train_windows`, `iter_windows`, `score`, `WindowBatch` (Task 3); `TaskSpec`, `SPECTROGRAM_SIGNALS`, `split_shots` (Task 2).
- Produces:

```python
@dataclass
class DirectConfig:
    d_model: int = 384; depth: int = 8; heads: int = 8
    patch_s: float = 0.005; history_s: float | None = None        # None -> task.input_length
    batch_size: int = 256; max_steps: int = 30_000; lr: float = 3e-4; weight_decay: float = 0.05
    warmup: int = 1_000; ema: float = 0.999; val_every: int = 2_500; val_shots: int = 200; seed: int = 0
class DirectModel(torch.nn.Module):
    def __init__(self, task: TaskSpec, cfg: DirectConfig)
    def forward(self, inputs: dict[str, Tensor], actuators: dict[str, Tensor]) -> dict[str, Tensor]
        # standardized tensors in (NaN allowed), standardized (B, *values_shape, T_out) out, always finite
def to_tensors(batch: WindowBatch, task: TaskSpec, device) -> tuple[dict, dict, dict]   # inputs, actuators, targets, TokaMark-standardized
def masked_mse(pred: dict[str, Tensor], target: dict[str, Tensor]) -> Tensor          # NaN targets ignored, signals weighted equally
def train_direct(task: TaskSpec, cfg: DirectConfig, out_dir: Path, device: str) -> Path   # path of best.pt
def direct_predictor(task: TaskSpec, checkpoint: Path, device: str) -> PredictFn
```

What the model is, and why. It is a plain transformer that reads the raw windows and regresses the targets. It is not a simulator and never becomes a headline number. Its job is to show what a specialized model reaches when nothing is quantized or truncated. Its tokens are deliberately the simulator's tokens (one channel over one 5 ms frame), so a gap between this model and the later simulator points at the token path and not at how the data are cut.

- **Tokens.** Each signal is cut along time into patches of `p = max(1, round(patch_s / dt))` samples, the same rule as `cheap.frame_len` (a last partial patch is padded with NaN). One token is one channel over one patch; for a spectrogram signal one token is one probe over one patch with all 256 bins. A token's embedding is a per-signal `Linear` over the concatenation of the values (NaN replaced by 0) and their 0/1 validity, plus a learned vector per (signal, channel), plus a learned vector per role (input, actuator, query), plus a sinusoidal encoding of the patch's centre time in seconds relative to the cut, passed through a `Linear`. Time enters in seconds, never as an index, which is what the spec's time-base section asks of the simulator.
- **Queries.** One query token per output channel and output patch, with the same identity and time embeddings and no values. All tokens go through one stack of pre-norm self-attention blocks (`torch.nn.TransformerEncoderLayer(norm_first=True, batch_first=True, activation="gelu")`, feed-forward width `4 * d_model`, no dropout). A per-signal `Linear` maps each query to its patch of output samples; patches are concatenated and cut to `T_out`.
- **Loss and selection.** `masked_mse` in standardized units, AdamW (betas 0.9 and 0.95), linear warm-up then cosine to zero, `torch.autocast("cuda", dtype=torch.bfloat16)`, gradient clipping at 1.0, an exponential moving average of the weights. Every `val_every` steps the averaged weights are scored through `score` on the first `val_shots` shots of `val_select`; the best is kept as `best.pt`. `train_log.csv` records step, training loss, validation NRMSE and seconds.
- **Data.** `train_windows(task, stride_s=0.005, shuffle=True, history_s=cfg.history_s)`, restarted when exhausted. Group-4 tasks use `batch_size=64` (about 6,000 tokens per window).
- **A weak model is not a headroom finding.** If the reported score is worse than our TokaMind replication on the same task, the table marks the cell "not competitive" and the task report says so. Up to two further configurations per task may then be tried (wider or deeper, or a longer `history_s` for group 4), each recorded with its validation curve. No more than two: the point of the stage is the table, not this model.

- [ ] **Step 1: Write the failing tests** (CPU, tiny configurations, the toy task):

```python
TINY = direct.DirectConfig(d_model=32, depth=2, heads=4, batch_size=16, max_steps=300, warmup=10)

def test_output_shapes(toy_task, toy_batch):
    m = direct.DirectModel(toy_task, TINY); x, a, y = direct.to_tensors(toy_batch(n=3, seed=0), toy_task, "cpu")
    out = m(x, a); s = toy_task.outputs[0]
    assert out[s.name].shape == (3, *s.values_shape, toy_task.n_out(s))

def test_partial_last_patch_for_spectrogram_outputs():
    # a hand-built task whose output is a (3, 256) signal at dt = 1.024 ms with 98 output stamps
    # -> out[name].shape == (2, 3, 256, 98)

def test_direct_model_is_finite_with_all_nan_inputs(toy_task, toy_batch):
    b = toy_batch(n=2, seed=0)
    for v in b.inputs.values(): v[:] = np.nan
    out = direct.DirectModel(toy_task, TINY)(*direct.to_tensors(b, toy_task, "cpu")[:2])
    assert all(torch.isfinite(o).all() for o in out.values())

def test_masked_mse_ignores_nan_targets_and_has_finite_gradients(toy_task, toy_batch): ...
    # half the targets NaN: loss finite, equals the MSE over the finite half; backward gives finite grads

def test_overfits_a_linear_rule(toy_task, toy_batch):
    # targets (standardized) = 2 * last input sample of summary-ip + 1, held over the output window
    # 300 Adam steps on 64 fixed windows -> masked_mse < 1e-2

def test_predictor_returns_physical_units(toy_task, toy_batch, tmp_path): ...
    # save a checkpoint of an untrained TINY model, direct_predictor(...)(batch) has target shapes, float32, finite
```

- [ ] **Step 2: Run, see them fail, implement, run again.** Expected: 6 passed in both environments.
- [ ] **Step 3: Add `direct` to `run_headroom.py`**: `--train` trains into `stage0/direct/<task>/` on `cuda:<gpu>`; without it the command scores `best.pt` on `val_report`. It refuses a GPU that has a compute process.
- [ ] **Step 4: Smoke run.** `scripts/mast/py.sh scripts/mast/run_headroom.py --task task_1-2 --predictor direct --train --gpu <free> --max-steps 2000`
Expected: validation NRMSE falls between the first and last evaluation and is below the mean baseline (0.998 for 1-2). Record steps per second.
- [ ] **Step 5: Train and score the seven tasks**, one process per free GPU, the probe tasks of groups 1 and 2 first (they decide the most: the spec's arithmetic says the margin has to come from there and from 4-5), then 4-5, 3-1, 4-4, 4-1. If Task 3 measured under 500 windows per second for group 4, use `--max-steps 15000` there and say so.
Expected: seven `best.pt`, seven `score.json` on `val_report`, each training log showing a validation curve that has flattened. A curve still falling at the last step is reported as such; the number is then an upper bound on what the model reaches.
- [ ] **Step 6: Commit.** `ignite.mast: direct supervised model for the headroom table`, with the seven scores, the step counts and the wall times.

---

### Task 10: The headroom table, the check and the results page

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/headroom/table.py`, `docs/models/ignite-mast-stage0.md`
- Modify: `scripts/mast/run_headroom.py` (sub-command `--table`), `mkdocs.yml` (nav, after `models/ignite-mast-design.md`), `docs/models/ignite-mast-design.md` (status line and a link to the results page)
- Test: `tests/ignite/mast/test_table.py`

**Interfaces:**
- Consumes: the `score.json` and per-shot files written by Tasks 4, 7, 8 and 9; `PUBLISHED`, `best_published`, `ALL_TASKS`, `STAGE0_TASKS` (Task 2).
- Produces:

```python
FITTED = ("shrunk", "ridge", "direct")
def bootstrap_ci(per_shot_nrmse: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]   # 95% interval of the mean over shots
def verdict(score: float, lo: float, hi: float, reference: float) -> str     # "ahead" if hi < reference, "behind" if lo > reference, else "tie"
def build_table(scores_dir: Path, tokamind_repl: dict[str, float | None]) -> pd.DataFrame
def projected_mean(table: pd.DataFrame) -> float
```

`build_table` gives one row per stage-0 task with: mean and persistence on `test` and on `val_report`, the floor on both, `shrunk`, `ridge` and `direct` on `val_report` with the direct model's interval, our TokaMind replication, the best published score and whose it is, `direct / best_published`, and the verdict of the direct model against the best published score. `projected_mean` is the 14-task mean with the direct model's score on the seven measured tasks and the best published score on the other seven. The spec's bar is 0.183; a model that only matched the best published score everywhere would have 0.194.

`run_headroom.py` exits with an error for `--split test` with a predictor in `FITTED`.

- [ ] **Step 1: Write the failing tests:**

```python
def test_bootstrap_ci_brackets_the_mean():
    x = np.random.default_rng(0).normal(0.2, 0.05, 500); lo, hi = table.bootstrap_ci(x)
    assert lo < x.mean() < hi and 0.003 < (hi - lo) / 2 < 0.006 and table.bootstrap_ci(x) == (lo, hi)

def test_verdict():
    assert table.verdict(0.08, 0.075, 0.085, 0.0941) == "ahead"
    assert table.verdict(0.09, 0.085, 0.095, 0.0941) == "tie"
    assert table.verdict(0.11, 0.10, 0.12, 0.0941) == "behind"

def test_projected_mean_uses_best_published_elsewhere(tmp_path):
    pytest.importorskip("tokamark")
    # a table whose seven direct scores equal best_published -> projected_mean within 1e-4 of the mean of best_published over 14 (0.194)

def test_build_table_reads_score_files(tmp_path): ...
    # fabricate score.json + per-shot csv for two tasks; missing cells are NaN, not 0

def test_cli_refuses_test_split_for_fitted_predictors():
    r = subprocess.run([str(REPO / "scripts/mast/py.sh"), str(REPO / "scripts/mast/run_headroom.py"),
                        "--task", "task_1-1", "--predictor", "ridge", "--split", "test"], capture_output=True, text=True)
    assert r.returncode != 0 and "test split" in r.stderr
```

- [ ] **Step 2: Run, see them fail, implement, run again.** Expected: 5 passed (the CLI test and the projection test skip in the pixi environment).
- [ ] **Step 3: Build the table.** `scripts/mast/py.sh scripts/mast/run_headroom.py --table` writes `stage0/headroom_table.csv` and prints it with the projected mean. Fill `tokamind_repl` from `compare_to_paper.py`'s output for the tasks that have finished; an unfinished task is empty.
- [ ] **Step 4: Write `docs/models/ignite-mast-stage0.md`.** Sections: what was run and when; the harness check (the 20 rows of Task 4); the cache (counts, size, time, status counts, the saturation deviation); the headroom table as measured; and "What it says", which separates what is measured from what is inferred and answers three questions from the spec in order: (1) per task, is the direct model ahead of, tied with or behind the best published score; (2) how much of each published score is TokaMind's representation floor; (3) is 0.183 within reach according to `projected_mean`, and which tasks carry it. State the limits: validation shots against published test scores, one seed, the direct model's training budget, and anything Task 9 reported as not converged or not competitive. If question (3) is answered no, the page says so in its first paragraph: the spec sends the bar back to Nathaniel with this table before stage 2 is committed.
- [ ] **Step 5: Check the docs build.** `uvx --with-requirements requirements-docs.txt --from 'mkdocs<2' mkdocs build -d /home/nvidia/.claude/jobs/ac52297d/tmp/site 2>&1 | grep -i "ignite-mast"`
Expected: no warning that names `ignite-mast-stage0.md` or `ignite-mast-design.md`. (`--strict` already fails on this branch from labeler pages; that is not part of this plan.)
- [ ] **Step 6: Run the whole suite in both environments.** `scripts/mast/py.sh -m pytest tests/ignite/mast -q` and `pixi run --frozen -e nvidia ignite-test`. Expected: all pass, with skips only for `tokamark` and `real_data` in the pixi environment.
- [ ] **Step 7: Commit.** `ignite.mast: headroom table for seven TokaMark tasks; docs: stage-0 results`, with the table in the body.

---

## What this plan leaves to later plans

- Our own window sampler over the cache, with a parity test against TokaMark's windows. Stage 0 trains from TokaMark's loader, which is fast enough for seven supervised models. Pretraining on whole shots needs the cache; per-task adaptation needs the sampler. Both belong to the stage-2 plan.
- Raw 500 kHz Mirnov waveforms in the cache, the position columns of the registry beyond what the Zarr attributes give, and the headroom rows of the other seven tasks: the stage-1 plan.
- Seeds. The table is one run per task. Run-to-run noise on the probe tasks is measured in stage 2, as the spec says.
