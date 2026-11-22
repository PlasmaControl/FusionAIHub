"""TokaMind's output-representation floor.

TokaMind's network does not predict an output signal sample by sample. It predicts the
coefficients of an orthonormal 3-D DCT-II of the standardized signal, only the K coefficients
that its `dct3d` embedding keeps for that signal (`selection_mode: rank`), and a fixed decoder
turns them back into a signal. Even a network that predicted every kept coefficient exactly
returns the target with the discarded coefficients set to zero. The task score of that signal
is the floor of TokaMind's output representation: no training of TokaMind's kind can go below it.

Missing target samples are filled before the transform, as the replication runs do. All seven
configs say `preprocess.embed_chunks.nan_imputation: interpolate` (read 2026-11-22), not the
default `zero`: on the (H, W, T) view of the standardized signal, linear interpolation along T per
(h, w) with the ends held, then along H per (w, t), then 0 for what is left
(`mmt/data/nan_imputation.py`). For a (C, T) signal H is the channel axis, so a dead probe is
filled from its neighbouring channels, which changes the kept coefficients at the valid channels.
`fill_non_finite` reproduces that without importing `mmt`, on batches.

The floor is the only headroom "predictor" that reads the targets, and it fits nothing, so it may
be scored on the test split. Which coefficients are kept is read from TokaMind's replication run
of the task (read-only): `<runs>/ft-<task>-ws-tokamind-base-v2-repl/embeddings/dct3d.yaml` and the
`.npy` index files it names. TokaMind's source (`mmt`) is not imported here; its codec
(`mmt/data/embeddings/dct3d.py`) is matched by `tests/ignite/mast/test_floor.py` when `mmt` is
importable. Read 2026-11-22: every output signal of the seven stage-0 tasks is in rank mode, and
the `.npy` files hold flat indices into the C-order flattening of the (H, W, T) DCT array.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml
from scipy.fft import dctn, idctn

from tokamak_foundation_model.ignite.mast import paths, tasks
from tokamak_foundation_model.ignite.mast.harness import PredictFn, WindowBatch

# sha256 of this file as loaded: `run_headroom.py` stores it in `score.json` as `predictor_sha256`,
# because the scoring-code hash of `baselines.write_score_json` does not cover the predictor.
PREDICTOR_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

Indices = dict[str, tuple[np.ndarray, tuple[int, int, int]]]

# The strategies of `mmt.data.nan_imputation` that a replication run may use here. A run with any
# other value (None included) is refused, not given a default.
NAN_IMPUTATIONS = ("interpolate", "zero")


def _view3d(values_shape: tuple[int, ...], n_time: int) -> tuple[int, int, int]:
    """TokaMind's (H, W, T) view of a signal: `(T,)` -> (1, 1, T), `(C, T)` -> (C, 1, T), `(H, W, T)` as is."""
    if len(values_shape) == 0:
        return 1, 1, n_time
    if len(values_shape) == 1:
        return values_shape[0], 1, n_time
    if len(values_shape) == 2:
        return values_shape[0], values_shape[1], n_time
    raise ValueError(f"only 1-D, 2-D and 3-D signals have a DCT view, got values_shape {values_shape}")


def _interpolate_across_h(view: np.ndarray) -> None:
    """Fill, in place, each (b, w, t) column of `view` (B, H, W, T) that is partly non-finite by
    linear interpolation along H (ends held), as `np.interp` does on that column.

    Columns are grouped by which of the H entries are valid: a pattern's fill is a fixed linear map
    of its valid entries (the weights are `np.interp` of unit vectors), so one matrix product serves
    every column of the pattern instead of one `np.interp` call per column.
    """
    n_h = view.shape[1]
    if n_h < 2:
        return
    columns = np.moveaxis(view, 1, -1).reshape(-1, n_h)  # a copy: (B*W*T, H)
    valid = np.isfinite(columns)
    n_valid = valid.sum(axis=1)
    partial = np.flatnonzero((n_valid > 0) & (n_valid < n_h))
    if partial.size == 0:
        return
    patterns, inverse = np.unique(valid[partial], axis=0, return_inverse=True)
    positions = np.arange(n_h)
    for k, pattern in enumerate(patterns):
        keep = np.flatnonzero(pattern)
        weights = np.stack([np.interp(positions, keep, np.eye(len(keep))[j]) for j in range(len(keep))], axis=1)
        rows = partial[inverse.reshape(-1) == k]
        columns[rows] = columns[rows][:, keep] @ weights.T
    view[...] = np.moveaxis(columns.reshape(view.shape[0], *view.shape[2:], n_h), -1, 1)


def fill_non_finite(x: np.ndarray, strategy: str) -> np.ndarray:
    """A copy of `x` (B, *values_shape, T) with non-finite samples (NaN, inf, -inf) filled as `mmt` does.

    `zero`: 0. `interpolate`: per window, on the (H, W, T) view, along T for each (h, w) and then
    along H for each (w, t), linear with the ends held; whatever is still non-finite (a window or a
    channel with nothing finite) becomes 0. Matches `mmt.data.nan_imputation.impute_non_finite`
    (checked in `tests/ignite/mast/test_floor.py` when `mmt` is importable). Only the rows and
    columns that are partly non-finite cost anything.
    """
    if strategy not in NAN_IMPUTATIONS:
        raise ValueError(f"nan_imputation must be one of {NAN_IMPUTATIONS}, got {strategy!r}")
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if finite.all():
        return x.copy()
    if strategy == "zero":
        return np.where(finite, x, 0.0)
    if not (1 <= x.ndim - 1 <= 3):
        raise ValueError(f"expected (B, *values_shape, T) with 1 to 3 axes after the batch, got {x.shape}")
    view = np.where(finite, x, np.nan).reshape(x.shape[0], *_view3d(x.shape[1:-1], x.shape[-1]))
    n_t = view.shape[-1]
    t_idx = np.arange(n_t)
    bad = ~np.isfinite(view)
    n_bad = bad.sum(axis=-1)
    for b, h, w in np.argwhere((n_bad > 0) & (n_bad < n_t)):  # along T per (b, h, w)
        row = view[b, h, w]
        ok = np.flatnonzero(~bad[b, h, w])
        view[b, h, w] = np.interp(t_idx, ok, row[ok])
    _interpolate_across_h(view)
    return np.where(np.isfinite(view), view, 0.0).reshape(x.shape)


def truncate_dct(x: np.ndarray, indices: np.ndarray, coeff_shape: tuple[int, int, int]) -> np.ndarray:
    """Keep only the DCT coefficients `indices` of each window of `x`.

    x: (B, *values_shape, T), finite; its (H, W, T) view must be `coeff_shape`. The transform is the
    orthonormal 3-D DCT-II over the view, `indices` are flat positions in the C-order flattening of
    the (H, W, T) coefficient array (as in TokaMind's `coeff_indices`), and the output is the
    inverse transform of the kept coefficients, in the shape and dtype of `x`.
    """
    x = np.asarray(x)
    if x.ndim < 2 or x.ndim > 4:
        raise ValueError(f"expected (B, *values_shape, T) with 1 to 3 axes after the batch, got {x.shape}")
    view = _view3d(x.shape[1:-1], x.shape[-1])
    if view != tuple(coeff_shape):
        raise ValueError(f"the (H, W, T) view of {x.shape[1:]} is {view}, not coeff_shape {tuple(coeff_shape)}")
    indices = np.asarray(indices, dtype=np.int64)
    n = int(np.prod(view))
    if indices.ndim != 1 or (indices.size and (indices.min() < 0 or indices.max() >= n)):
        raise ValueError(f"indices out of range for {n} coefficients (or not 1-D)")
    coeffs = dctn(x.reshape(x.shape[0], *view), axes=(1, 2, 3), norm="ortho").reshape(x.shape[0], n)
    kept = np.zeros_like(coeffs)
    kept[:, indices] = coeffs[:, indices]
    out = idctn(kept.reshape(x.shape[0], *view), axes=(1, 2, 3), norm="ortho")
    return out.reshape(x.shape)


def output_indices_from_run(embeddings_dir: Path, task: tasks.TaskSpec) -> Indices:
    """Per output signal of `task`: (flat coefficient indices, coeff_shape) from a run's `dct3d.yaml`.

    Raises `ValueError` if a signal is missing, is not in rank mode, or its `coeff_shape` is not
    the (H, W, T_out) view of the task's output, so that a changed chunking cannot pass silently.
    """
    embeddings_dir = Path(embeddings_dir)
    config = yaml.safe_load((embeddings_dir / "dct3d.yaml").read_text())
    entries = config["embeddings"]["per_signal_overrides"]["output"]
    result: Indices = {}
    for spec in task.outputs:
        if spec.name not in entries:
            raise ValueError(f"{embeddings_dir / 'dct3d.yaml'} has no output signal {spec.name}")
        kwargs = entries[spec.name]["encoder_kwargs"]
        if kwargs["selection_mode"] != "rank":
            raise ValueError(f"{spec.name}: selection_mode {kwargs['selection_mode']!r}, only 'rank' is read")
        coeff_shape = tuple(int(v) for v in kwargs["coeff_shape"])
        expected = _view3d(spec.values_shape, task.n_out(spec))
        if coeff_shape != expected:
            raise ValueError(f"{spec.name}: coeff_shape {coeff_shape} is not the (H, W, T) view "
                             f"{expected} of the task's output")
        indices = np.load(embeddings_dir / kwargs["coeff_indices_path"], allow_pickle=False)
        if indices.ndim != 1 or len(indices) != int(kwargs["num_coeffs"]):
            raise ValueError(f"{spec.name}: {indices.shape} indices for num_coeffs {kwargs['num_coeffs']}")
        result[spec.name] = (indices.astype(np.int64), coeff_shape)
    return result


def tokamind_output_indices(task_name: str) -> Indices:
    """Per output signal of the task: (flat coefficient indices, coeff_shape (H, W, T)) of TokaMind's run."""
    embeddings = paths.tokamind_runs_dir() / f"ft-{task_name}-ws-tokamind-base-v2-repl" / "embeddings"
    return output_indices_from_run(embeddings, tasks.load_task(task_name))


def nan_imputation_from_run(run_dir: Path, task_name: str) -> str:
    """`preprocess.embed_chunks.nan_imputation` of the run's config, `interpolate` or `zero`.

    Raises `ValueError` naming the file if the key is missing or has any other value; there is no default.
    """
    path = Path(run_dir) / f"ft-{task_name}-ws-tokamind-base-v2-repl.yaml"
    config = yaml.safe_load(path.read_text())
    try:
        strategy = config["preprocess"]["embed_chunks"]["nan_imputation"]
    except (KeyError, TypeError):
        raise ValueError(f"{path} has no preprocess.embed_chunks.nan_imputation") from None
    if strategy not in NAN_IMPUTATIONS:
        raise ValueError(f"{path}: nan_imputation {strategy!r}, expected one of {NAN_IMPUTATIONS}")
    return strategy


def tokamind_nan_imputation(task_name: str) -> str:
    """How TokaMind's replication run of the task fills missing samples before the DCT."""
    return nan_imputation_from_run(paths.tokamind_runs_dir() / f"ft-{task_name}-ws-tokamind-base-v2-repl",
                                   task_name)


def representation_sha256(indices: Indices, nan_imputation: str) -> str:
    """Digest of TokaMind's output representation: for each signal in name order its name,
    coeff_shape and index array (int64, little endian), then the fill strategy."""
    digest = hashlib.sha256()
    for name in sorted(indices):
        kept, coeff_shape = indices[name]
        digest.update(name.encode() + b"\0")
        digest.update(np.asarray(coeff_shape, dtype="<i8").tobytes())
        digest.update(np.asarray(kept, dtype="<i8").tobytes())
    digest.update(nan_imputation.encode())
    return digest.hexdigest()


def floor_predictor(task: tasks.TaskSpec, indices: Indices, nan_imputation: str) -> PredictFn:
    """The prediction of a network that gets every kept coefficient exactly right.

    Reads `batch.targets`: standardize with TokaMark's mean and std, fill missing samples with
    `nan_imputation` as TokaMind's run does (`fill_non_finite`), truncate the DCT, unstandardize.
    At a sample whose target is missing the prediction is some finite value; the score ignores it.
    """
    def predict(batch: WindowBatch) -> dict[str, np.ndarray]:
        out = {}
        for spec in task.outputs:
            z = fill_non_finite(spec.standardize(batch.targets[spec.name].astype(np.float64)), nan_imputation)
            kept, coeff_shape = indices[spec.name]
            out[spec.name] = spec.unstandardize(truncate_dct(z, kept, coeff_shape)).astype(np.float32)
        return out

    return predict
