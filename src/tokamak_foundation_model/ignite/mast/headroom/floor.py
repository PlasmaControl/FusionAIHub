"""TokaMind's output-representation floor.

TokaMind's network does not predict an output signal sample by sample. It predicts the
coefficients of an orthonormal 3-D DCT-II of the standardized signal, only the K coefficients
that its `dct3d` embedding keeps for that signal (`selection_mode: rank`), and a fixed decoder
turns them back into a signal. Even a network that predicted every kept coefficient exactly
returns the target with the discarded coefficients set to zero. The task score of that signal
is the floor of TokaMind's output representation: no training of TokaMind's kind can go below it.

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


def _view3d(values_shape: tuple[int, ...], n_time: int) -> tuple[int, int, int]:
    """TokaMind's (H, W, T) view of a signal: `(T,)` -> (1, 1, T), `(C, T)` -> (C, 1, T), `(H, W, T)` as is."""
    if len(values_shape) == 0:
        return 1, 1, n_time
    if len(values_shape) == 1:
        return values_shape[0], 1, n_time
    if len(values_shape) == 2:
        return values_shape[0], values_shape[1], n_time
    raise ValueError(f"only 1-D, 2-D and 3-D signals have a DCT view, got values_shape {values_shape}")


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


def floor_predictor(task: tasks.TaskSpec, indices: Indices) -> PredictFn:
    """The prediction of a network that gets every kept coefficient exactly right.

    Reads `batch.targets`: standardize with TokaMark's mean and std, set non-finite samples to 0
    (the standardized mean; the score ignores those samples), truncate the DCT, unstandardize.
    """
    def predict(batch: WindowBatch) -> dict[str, np.ndarray]:
        out = {}
        for spec in task.outputs:
            z = spec.standardize(batch.targets[spec.name].astype(np.float64))
            z = np.where(np.isfinite(z), z, 0.0)
            kept, coeff_shape = indices[spec.name]
            out[spec.name] = spec.unstandardize(truncate_dct(z, kept, coeff_shape)).astype(np.float32)
        return out

    return predict
