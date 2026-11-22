"""The scoring harness: TokaMark's windows in, TokaMark's task score out.

Every number this project reports goes through this module. It does not re-implement
anything of the benchmark: windows come from TokaMark's own dataset class
(`tokamark.data.initialize_TokaMark_dataset`), built the way TokaMark's reference script
`scripts/persistence.py` builds it, and predictions are scored by TokaMark's own
`WindowMetricsAccumulator` and `compute_metrics`. `tokamark`, `MAST_tools`, `mmt` and `zarr`
are imported inside the functions that need them, so the module imports without them.

Windows
-------
A ``predict_fn`` receives a `WindowBatch` in physical units (NaN = missing) and returns the
targets' physical values for the same windows. Test mode is TokaMark's *sparse* test mode
(the default of the dataset class, and what the reference script uses): a window is kept when
not all inputs and not all actuators are entirely NaN and no output is entirely NaN (an
output may still hold some NaN samples; the evaluator ignores them). The filled variant
(`filled_test_mode`) is not reachable through `initialize_TokaMark_dataset` and is not used.

History. A markovian task delivers exactly round(input_length / dt) samples of each input
(computed from seconds, not from `TaskSpec.steps`, see `_WindowTransform`). A
non-markovian task delivers the whole history since the shot's first valid sample; this module
computes `last_valid` on that whole history and then keeps the last `history_s` seconds
(default: the task's `input_length`), left-padding with NaN if less was delivered. Actuators
of a non-markovian task keep the same history plus the unchanged future part (delta + output
length). Using less history than the protocol allows is within the protocol; more is a later
variant. `history_s` is ignored by markovian tasks.

Workers. `TokaMarkDataset.__iter__` shares the shots among DataLoader workers as contiguous,
disjoint chunks of `ceil(n_shots / num_workers)` shots, and each worker batches its own
stream (the last, short batch of every worker is emitted). So with `num_workers > 0` no window
is delivered twice and none is dropped (`test_no_duplicate_windows_with_workers`), but a batch
holds windows of the shots of one worker only, and at most `ceil(n / ceil(n / num_workers))`
workers are busy.

Throughput
----------
Measured 2026-11-22 on node13 (352 cores, shared with the TokaMind replication queue, load
average 40 to 50), `num_workers=64`, batch size 64, shots drawn with `random.Random(0).sample`,
each timing capped at 10 minutes (never reached), windows per second over the whole run
including worker start-up (in brackets: after the first batch arrived):

    train_windows, stride_s=0.005, 200 train shots   task_1-1  18180 windows   5329/s (7626/s)
                                                      task_4-1  12296 windows   2059/s (4104/s)
    iter_windows(test_mode=True), 100 test shots      task_1-1  26384 windows   5538/s (7046/s)
                                                      task_4-1  28843 windows   1395/s (1528/s)
    train_windows, 2000 train shots (steady state)    task_1-1 178374 windows   2530/s
                                                      task_4-1 123245 windows   1434/s

Why the longer runs: with 200 shots a worker holds about 4 shots, fewer windows than the
shuffle buffer (512), so nothing is emitted before its shards are done and the short runs
are mostly start-up. The first read of a shot (cold page cache) is the slow part: the
2000-shot task_4-1 run read cold, with an earlier version of this module, gave 658/s
(about 700/s in its second half). The numbers above are the warm repeat; the harness
transform costs about 0.7 ms of the 3.7 ms a task_4-1 window takes in one process.
No GPU was involved.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .paths import prepare_write_root, tokamark_zarr_root
from .tasks import TaskSpec, split_shots

TEST_SPLITS = ("test", "val", "val_select", "val_report")


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
    task: str                    # the task name given to `score_batches`
    task_nrmse: float            # TokaMark's task score: mean over shots of the shot's mean over signals
    per_signal: pd.DataFrame     # feature_name, NRMSE (mean over shots)
    per_shot: pd.DataFrame       # shot_id, NRMSE (the task score of each shot)
    n_windows: int
    out_dir: Path                # as given; TokaMark's files are in out_dir / task


# ---------------------------------------------------------------------------- building the windows

def open_mast_dataset(config_task: dict, shots: list[int]):
    """TokaMark's shot-level dataset, with the loader settings of its reference script.

    Physical units (no standard scaling), outliers replaced per TokaMark's outlier file, bad
    EFIT ratings removed (the default, as in the reference script), read from the local Zarr
    store. The one place where these settings live: whole-shot readers use this too.
    """
    from MAST_tools.utils.path_utils import RANDOM_SPLIT_OUTLIER_METADATA_FILE
    from tokamark.data import initialize_MAST_dataset

    return initialize_MAST_dataset(
        config_task=config_task,
        shots_list=list(shots),
        local_flag=True,
        use_std_scaling=False,
        remove_outliers=True,
        outlier_metadata_file=RANDOM_SPLIT_OUTLIER_METADATA_FILE,
        store_manager_settings={"base_local_data_path": str(tokamark_zarr_root())},
    )


def _as_array(values, spec) -> np.ndarray:
    """Shape (*values_shape, T), dtype as delivered. TokaMark squeezes a leading 1 of a (1, T) array to (T,)."""
    return np.asarray(values).reshape(*spec.values_shape, -1)


def _last_finite(a: np.ndarray, tail: int) -> np.ndarray:
    """(*shape, T) -> (*shape) float32: the last finite sample along time, NaN where there is none.

    The last `tail` samples are searched first; the rest of the history is only searched for the
    channels that have no finite sample there (cheap in the common case, same result).
    """
    out = np.full(a.shape[:-1], np.nan, dtype=np.float32)
    found = np.zeros(a.shape[:-1], dtype=bool)
    stop = a.shape[-1]
    for start in (max(0, stop - tail), 0):
        if start >= stop or found.all():
            break
        seg = a[..., start:stop]
        finite = np.isfinite(seg)
        idx = seg.shape[-1] - 1 - np.argmax(finite[..., ::-1], axis=-1)
        value = np.take_along_axis(seg, idx[..., None], axis=-1)[..., 0]
        new = finite.any(axis=-1) & ~found
        out[new] = value[new]
        found |= new
        stop = start
    return out


def _fit_time(a: np.ndarray, n: int, name: str, *, flexible: bool) -> np.ndarray:
    """float32, exactly `n` time samples. Flexible: keep the last n, left-pad with NaN. Otherwise a mismatch is an error."""
    t = a.shape[-1]
    if t != n and not flexible:
        raise ValueError(f"{name}: TokaMark delivered {t} time samples, expected {n}")
    a = a[..., max(0, t - n):].astype(np.float32, copy=False)  # crop first: only the kept samples are converted
    if t < n:
        a = np.concatenate([np.full((*a.shape[:-1], n - t), np.nan, dtype=np.float32), a], axis=-1)
    return a


class _WindowTransform:
    """TokaMark's custom transform: one TokaMark window -> the arrays of a one-window `WindowBatch`.

    A class, not a closure, so that it pickles for spawn-based workers.
    """

    def __init__(self, task: TaskSpec, history_s: float | None):
        self.task = task
        self.history_s = history_s

    # Lengths in samples, from the task's seconds and the signal's dt with TokaMark's own
    # formulas (`TokaMarkDataset._build_window`). `TaskSpec.steps` is not used: TokaMark's
    # `get_task_metadata` shares one dict between the roles of a signal that is both an input
    # and an output, so `length_in_time_stamps` of such an input holds the output's length
    # (15 inputs in 7 tasks on 2026-11-22, e.g. task_3-1 thomson_scattering-t_e: 10, not 1).
    def _n_input(self, s) -> int:
        task = self.task
        return round((task.input_length if task.markovian else (self.history_s or task.input_length)) / s.dt)

    def _n_actuator(self, s) -> int:
        """Actuator samples kept: the input history in use plus the future part (delta + output length)."""
        task = self.task
        return self._n_input(s) + round(task.delta / s.dt) + round(task.output_length / s.dt)

    def __call__(self, w: dict) -> dict:
        task, flexible = self.task, not self.task.markovian
        inputs, last_valid = {}, {}
        for s in task.inputs:
            a = _as_array(w["input"][s.name]["values"], s)
            n = self._n_input(s)
            last_valid[s.name] = _last_finite(a, tail=n)  # over the whole delivered history, before the crop
            inputs[s.name] = _fit_time(a, n, s.name, flexible=flexible)
        actuators = {
            s.name: _fit_time(_as_array(w["actuator"][s.name]["values"], s), self._n_actuator(s),
                              s.name, flexible=flexible)
            for s in task.actuators
        }
        targets = {
            s.name: _fit_time(_as_array(w["output"][s.name]["values"], s),
                              round(task.output_length / s.dt), s.name, flexible=False)
            for s in task.outputs
        }
        return {"inputs": inputs, "actuators": actuators, "targets": targets, "last_valid": last_valid}


def _collate(items: list[dict]) -> dict:
    """Stack transformed windows into tensors (tensors cross the worker boundary through shared memory)."""
    out: dict = {
        "shot_ids": torch.tensor([it["shot_id"] for it in items], dtype=torch.int64),
        "window_indices": torch.tensor([it["window_index"] for it in items], dtype=torch.int64),
    }
    for role in ("inputs", "actuators", "targets", "last_valid"):
        out[role] = {
            name: torch.from_numpy(np.stack([it[role][name] for it in items]))
            for name in items[0][role]
        }
    return out


def _to_window_batch(c: dict) -> WindowBatch:
    return WindowBatch(
        shot_ids=c["shot_ids"].numpy(),
        window_indices=c["window_indices"].numpy(),
        **{role: {n: t.numpy() for n, t in c[role].items()}
           for role in ("inputs", "actuators", "targets", "last_valid")},
    )


def iter_windows(task: TaskSpec, shots: list[int], *, test_mode: bool, history_s: float | None = None,
                 batch_size: int = 64, num_workers: int = 32, shuffle: bool = False,
                 stride_s: float | None = None) -> Iterator[WindowBatch]:
    """Batches of TokaMark's windows of `shots`.

    `test_mode=True` is TokaMark's sparse test mode: the evaluation windows. `stride_s`
    replaces the window stride (training only: it changes which windows exist, so it is refused
    in test mode). `shuffle` shuffles the windows inside each worker's stream.
    """
    if stride_s is not None and test_mode:
        raise ValueError("stride_s changes the evaluation windows; it is only for training (test_mode=False)")
    if len(set(shots)) != len(shots):
        raise ValueError("shots contains duplicates, which would deliver windows twice")
    if history_s is not None and history_s <= 0:
        raise ValueError(f"history_s must be positive, got {history_s}")

    from tokamark.data import initialize_TokaMark_dataset
    from tokamark.tasks import get_task_config, get_task_metadata
    from torch.utils.data import DataLoader

    config = copy.deepcopy(get_task_config(task.name))
    if stride_s is not None:
        config["stride_window"] = float(stride_s)
    metadata = get_task_metadata(config)
    dataset = initialize_TokaMark_dataset(
        open_mast_dataset(config, list(shots)),
        metadata,
        config,
        custom_transform=_WindowTransform(task, history_s),
        test_mode=test_mode,
        shuffle_windows=shuffle,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=min(num_workers, len(shots)),  # a worker without a shot only costs a fork
        collate_fn=_collate,
    )
    return (_to_window_batch(c) for c in loader)


def train_windows(task: TaskSpec, shots: list[int] | None = None, **kw) -> Iterator[WindowBatch]:
    """Training windows (not test mode, shuffled unless `shuffle=False`); shots default to the training split."""
    train = split_shots("train")
    if shots is None:
        shots = train
    else:
        outside = sorted(set(shots) - set(train))
        if outside:
            raise ValueError(f"shots {outside[:5]} are not in the training split")
    kw.setdefault("shuffle", True)
    return iter_windows(task, shots, test_mode=False, **kw)


# ---------------------------------------------------------------------------- scoring

def _check_prediction(name: str, pred, target: np.ndarray) -> np.ndarray:
    if pred is None:
        raise ValueError(f"prediction for {name} is missing")
    pred = np.asarray(pred)
    if pred.shape != target.shape:
        raise ValueError(f"prediction for {name} has shape {pred.shape}, the target has shape {target.shape}")
    bad = np.isfinite(target) & ~np.isfinite(pred)
    if bad.any():
        raise ValueError(f"prediction for {name} is non-finite at {int(bad.sum())} samples where the target is finite")
    return pred


def score_batches(task_name: str, batches: Iterable[WindowBatch], predict_fn: PredictFn,
                  out_dir: Path) -> ScoreResult:
    """Score `predict_fn` on `batches` with TokaMark's evaluator; its files go to `out_dir / task_name`.

    The outputs to predict are the keys of each batch's `targets`. A prediction that is
    missing, has another shape than its target, or is non-finite where the target is finite is
    an error, and so is a (shot, window) pair delivered twice.
    """
    out_dir = prepare_write_root(Path(out_dir))
    accumulator = None
    seen: set[tuple[int, int]] = set()
    for batch in batches:
        pairs = list(zip(batch.shot_ids.tolist(), batch.window_indices.tolist()))
        if len(set(pairs)) != len(pairs) or seen.intersection(pairs):
            raise ValueError(f"task {task_name}: a (shot, window) pair was delivered twice; it would count twice")
        predictions = predict_fn(batch)
        checked = {name: _check_prediction(name, predictions.get(name), target)
                   for name, target in batch.targets.items()}
        if accumulator is None:
            from tokamark.evaluator import WindowMetricsAccumulator
            accumulator = WindowMetricsAccumulator(task=task_name)
        n = len(batch.shot_ids)
        for name, target in batch.targets.items():
            accumulator.add_batch(
                y_target=target.astype(np.float64).reshape(n, -1),
                y_pred=checked[name].astype(np.float64).reshape(n, -1),
                shot_ids=batch.shot_ids,
                window_indices=batch.window_indices,
                feature_name=name,
            )
        seen.update(pairs)
    if accumulator is None:
        raise ValueError(f"task {task_name}: no windows were scored")

    from tokamark.evaluator import aggregate_windows_metrics, compute_metrics

    table = compute_metrics(
        task=task_name,
        output_dir=out_dir,
        window_metrics_accumulator=accumulator,
        save_shot_metrics=True,
        save_task_metrics=True,
    )
    task_nrmse = float(table.loc[task_name, "NRMSE_mean"])
    signals = table.drop(index=task_name)["NRMSE_mean"]
    per_signal = pd.DataFrame({"feature_name": signals.index.astype(str), "NRMSE": signals.to_numpy()})
    # compute_metrics returns no per-shot table; TokaMark's own aggregation gives it, from the same windows.
    _, shots_table, _ = aggregate_windows_metrics(accumulator.to_dataframe())
    per_shot = pd.DataFrame({"shot_id": shots_table.index.to_numpy(), "NRMSE": shots_table["NRMSE"].to_numpy()})
    if abs(per_shot["NRMSE"].mean() - task_nrmse) > 1e-9 * max(1.0, abs(task_nrmse)):
        raise RuntimeError(
            f"task {task_name}: the mean of the per-shot scores ({per_shot['NRMSE'].mean()}) differs from "
            f"TokaMark's task score ({task_nrmse})"
        )
    return ScoreResult(task=task_name, task_nrmse=task_nrmse, per_signal=per_signal, per_shot=per_shot,
                       n_windows=len(seen), out_dir=out_dir)


def score(task: TaskSpec, predict_fn: PredictFn, out_dir: Path, *, split: str,
          shots: list[int] | None = None, history_s: float | None = None, **kw) -> ScoreResult:
    """Score `predict_fn` on a split in TokaMark's test mode (the benchmark's evaluation windows).

    `split` is "test", "val", "val_select" or "val_report"; `shots` restricts it to a subset.
    """
    if split not in TEST_SPLITS:
        raise ValueError(f"split must be one of {TEST_SPLITS}, got {split!r}")
    split_list = split_shots(split)
    if shots is None:
        shots = split_list
    else:
        outside = sorted(set(shots) - set(split_list))
        if outside:
            raise ValueError(f"shots {outside[:5]} are not in the {split!r} split")
    out_dir = prepare_write_root(Path(out_dir))
    batches = iter_windows(task, shots, test_mode=True, history_s=history_s, **kw)
    return score_batches(task.name, batches, predict_fn, out_dir)
