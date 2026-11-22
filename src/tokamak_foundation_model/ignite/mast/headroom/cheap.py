"""Two cheap fitted predictors: shrunk persistence and ridge regression on frame means.

They say how much of a task is predictable by trivial means, the second column of the headroom
table (the first is TokaMind's representation floor, `floor.py`). Neither uses a network or a
spectrogram codec. Ridge looks at the last 5 ms of each non-spectrogram input and at the whole kept
history of a spectrogram input (per bin); the anchor of `ShrunkPersistence` is `batch.last_valid`, the
last finite sample over the whole delivered history (`harness.py`), and for a spectrogram the mean over
the kept history. Both are fitted on the same 3,000
training shots (`train_shots`), in TokaMark-standardized units (`(x - mean) / std` per signal),
and return physical units. They are scored on `val_report` only, by `scripts/mast/run_headroom.py`.

Frames. Everything is done on frames of `FRAME_S` = 5 ms (`frame_len(dt)` samples; a last partial
frame counts). The two spectrogram signals are the exception on the output side: their output
window is one frame, because a spectrogram's STFT stamps are 1.024 ms apart and its content is
the band structure per bin, not the time course within 20 ms.

Training mean. The "training mean" of a channel is the mean of its finite standardized target
samples over all training windows. A channel that never has a finite target gets TokaMark's mean
(`SignalSpec.mean`, which is 0 standardized); `n_channels_without_target` counts them.

`ShrunkPersistence`. For an output signal that is also an input the prediction of output frame k is
`m_c + a[s, k] * (p_c - m_c)`: `m_c` the training mean, `p_c` the anchor (the last finite input
sample; for the spectrogram signals the mean of the input over the kept history, per bin), `a` the
least-squares slope of the frame means of the target on `p - m`, pooled over channels and windows.
a = 0 is the mean predictor and a = 1 persistence, so on the windows it was fitted on it cannot be
worse than either (in the frame-mean sense). A non-finite anchor falls back to `m_c`; an output that is not
an input is predicted by `m_c`.

`RidgeFrames`. Features (`window_features`): per input signal the standardized NaN-mean over its
last 5 ms (spectrogram signals: per bin over the kept history), non-finite set to 0, then a 0/1
validity flag per channel (per signal for a spectrogram or a map); per actuator the standardized
frame means over its whole window (the future part included, because the future actuator program is
what the task is conditioned on); a constant 1. Targets: the frame means of each output channel,
centred by the training mean and held over their frame in the prediction. A non-finite target frame
counts as 0 after centring. That pulls the fit toward the mean wherever targets are missing, which is
acceptable for a cheap predictor and stated on the results page. The fit accumulates `X'X` and the two
target moments in float64 in one pass, solves `(X'X + lambda * n * D) W = X'Y` by Cholesky for each
`lambdas` (D is the identity with a zero at the constant, so it is not penalised), and `select`
keeps the lambda with the smallest mean squared error in standardized units on the finite
validation frames of `select_shots()`: 200 shots evenly spaced over the sorted `val_select` list, the rule of
`direct.validation_shots` (the first 200 shots are one old campaign). The candidates are the ten decades
1e-8 ... 10. A penalty whose solve does not give finite weights is not a candidate and is printed. The fit
prints and stores the chosen penalty, the candidates that solved and `at_edge` ("lower" or "upper" when the
choice is the smallest or the largest candidate, so that the optimum is not bracketed; else none), and
`run_headroom.py` copies the penalty, the edge flag and the two counts into `score.json`.
Centring by the training mean cannot be done while streaming (the mean is known
at the end), so the accumulator holds `X'(valid * y)` and `X'valid` and forms
`X'Y_centred = X'(valid * y) - m * X'valid` at the end, which is exact.

Measured 2026-11-22 on node13 with 48 loader workers (task_1-1, the task with the smallest F x Q:
F = 231, Q = 17): 3,000 training shots give 266,854 windows at stride 5 ms; the one-pass fit takes about 90 to
100 s (the data pass: the solve was six Cholesky factorizations of a 231 x 231 matrix, ten with today's
grid); scoring the 144,569 `val_report` windows takes about 32 s. No stage-0 task needs more than 25 MB for `X'Y`
(task_4-5: F = 2049, Q = 1536). task_1-1 has no output that is also an input, so `shrunk` is there only a
per-channel training mean. The training mean differs from TokaMark's mean baseline because several
outputs hold a pair of channels with opposite signs (the two x-points), for which TokaMark's one mean
per signal is far from each channel's mean.

Fit files. `save_fit` writes one `.npz` per task and predictor (atomically: temporary name, then
rename) holding the fitted arrays and a metadata record: the code hash (`PREDICTOR_SHA256`, this
file as loaded), the digest of the training shot list, the stride, the scoring-code hash, the number of windows seen,
the chosen lambda (ridge: also the grid, the candidates and `at_edge`), and returns the sha256 of the bytes it
wrote. `obtain` reuses a file only if every one of them matches the present code, shot
list, stride and scoring code, otherwise it fits again: a score is never computed from a fit made by other
code. A file that cannot be read by this code (a fit from before the penalty record, a missing array) counts
as no current fit, never as current and never as an error. `obtain` returns the predictor with the sha256
of the bytes it was made from (the file is read once, or the digest is that of what `save_fit` wrote), which
`run_headroom.py` stores in `score.json` as `fit_sha256`. `current_fit_digest` is the same digest, or
`NO_CURRENT_FIT`, for the check before a stored score is kept.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from tokamak_foundation_model.ignite.mast import baselines, harness, tasks
from tokamak_foundation_model.ignite.mast.harness import WindowBatch
from tokamak_foundation_model.ignite.mast.tasks import SPECTROGRAM_SIGNALS, SignalSpec, TaskSpec

# sha256 of this file as loaded: `run_headroom.py` stores it as `predictor_sha256`, and a fit file
# made by other code is not reused.
PREDICTOR_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

FRAME_S = 0.005
N_TRAIN_SHOTS = 3000          # fixed random sample of the training split (seed 0), the same for every task
TRAIN_STRIDE_S = 0.005        # window stride of the training windows (the benchmark's own is task-dependent)
N_SELECT_SHOTS = 200          # shots of `val_select`, evenly spaced over the sorted list, that choose ridge's lambda
NO_CURRENT_FIT = "no current fit"


# ------------------------------------------------------------------------------------ frames

def frame_len(dt: float) -> int:
    """Samples of one 5 ms frame at sampling interval `dt` (at least 1)."""
    return max(1, round(FRAME_S / dt))


def _nanmean_last(x: np.ndarray) -> np.ndarray:
    """float64 mean over the finite entries of the last axis; NaN where there is none (and no warning)."""
    finite = np.isfinite(x)
    total = np.where(finite, x, 0.0).sum(axis=-1, dtype=np.float64)
    count = finite.sum(axis=-1)
    out = np.full(total.shape, np.nan)
    np.divide(total, count, out=out, where=count > 0)
    return out


def frame_means(x: np.ndarray, dt: float) -> np.ndarray:
    """(..., T) -> (..., ceil(T / frame_len)): NaN-mean per frame; a last partial frame counts."""
    x = np.asarray(x)
    if x.dtype.kind != "f":
        x = x.astype(np.float64)
    n = frame_len(dt)
    frames = -(-x.shape[-1] // n)
    if frames * n != x.shape[-1]:
        x = np.concatenate([x, np.full((*x.shape[:-1], frames * n - x.shape[-1]), np.nan, dtype=x.dtype)], axis=-1)
    return _nanmean_last(x.reshape(*x.shape[:-1], frames, n))


def hold_frames(m: np.ndarray, dt: float, n: int) -> np.ndarray:
    """(..., n_frames) -> (..., n): each frame's value repeated over its frame, cut to `n` samples."""
    m = np.asarray(m)
    length = frame_len(dt)
    if m.shape[-1] * length < n:
        raise ValueError(f"{m.shape[-1]} frames of {length} samples cannot cover {n} samples")
    return np.repeat(m, length, axis=-1)[..., :n]


def _is_spectrogram(s: SignalSpec) -> bool:
    return s.name in SPECTROGRAM_SIGNALS


def _n_channels(s: SignalSpec) -> int:
    return int(np.prod(s.values_shape, dtype=np.int64))


def _n_frames(task: TaskSpec, s: SignalSpec) -> int:
    """Frames of an output signal's window: one for a spectrogram, else ceil(n_out / frame_len)."""
    return 1 if _is_spectrogram(s) else -(-task.n_out(s) // frame_len(s.dt))


def _standardized(s: SignalSpec, a: np.ndarray) -> np.ndarray:
    """(B, *values_shape, T) physical -> (B, C, T) standardized float64."""
    return s.standardize(np.asarray(a, dtype=np.float64)).reshape(a.shape[0], _n_channels(s), a.shape[-1])


def _target_frames(z: np.ndarray, s: SignalSpec) -> np.ndarray:
    """(B, C, T_out) standardized -> (B, C, K) frame means (NaN where a frame has nothing finite)."""
    return _nanmean_last(z)[..., None] if _is_spectrogram(s) else frame_means(z, s.dt)


def _physical(frames_z: np.ndarray, s: SignalSpec, n_out: int) -> np.ndarray:
    """(B, C, K) standardized frame values -> (B, *values_shape, n_out) float32 physical, each held over its frame."""
    held = (np.broadcast_to(frames_z, (*frames_z.shape[:-1], n_out)) if _is_spectrogram(s)
            else hold_frames(frames_z, s.dt, n_out))
    return s.unstandardize(held).reshape(frames_z.shape[0], *s.values_shape, n_out).astype(np.float32)


# ------------------------------------------------------------------------------------ features

def _flag_per_signal(s: SignalSpec) -> bool:
    return _is_spectrogram(s) or len(s.values_shape) >= 2


def feature_count(task: TaskSpec) -> int:
    """F: the columns of `window_features`, from the spec alone."""
    n = 1
    for s in task.inputs:
        n += _n_channels(s) + (1 if _flag_per_signal(s) else _n_channels(s))
    for s in task.actuators:
        n += _n_channels(s) * -(-task.n_act(s) // frame_len(s.dt))
    return n


def target_count(task: TaskSpec) -> int:
    """Q: the regression targets of `RidgeFrames` (output frame means over all channels)."""
    return sum(_n_channels(s) * _n_frames(task, s) for s in task.outputs)


def window_features(batch: WindowBatch, task: TaskSpec) -> np.ndarray:
    """(B, F) float64. See the module docstring for the layout; every entry is finite."""
    n = len(batch.shot_ids)
    cols: list[np.ndarray] = []
    # Finiteness is tested after `standardize`: a signal whose std is 0 would otherwise pass an inf into the solve.
    with np.errstate(divide="ignore", invalid="ignore"):
        for s in task.inputs:
            x = batch.inputs[s.name]
            z = s.standardize(_nanmean_last(x if _is_spectrogram(s) else x[..., -frame_len(s.dt):]).reshape(n, -1))
            valid = np.isfinite(z)
            cols.append(np.where(valid, z, 0.0))
            cols.append((valid.any(axis=1, keepdims=True) if _flag_per_signal(s) else valid).astype(np.float64))
        for s in task.actuators:
            z = s.standardize(frame_means(batch.actuators[s.name], s.dt).reshape(n, -1))
            cols.append(np.where(np.isfinite(z), z, 0.0))
    cols.append(np.ones((n, 1)))
    return np.concatenate(cols, axis=1)


# ------------------------------------------------------------------------------------ shared pieces

class _ChannelMeans:
    """Sums and counts of the finite standardized target samples per output channel."""

    def __init__(self):
        self.total: dict[str, np.ndarray] = {}
        self.count: dict[str, np.ndarray] = {}

    def add(self, name: str, z: np.ndarray) -> None:
        finite = np.isfinite(z)
        total = np.where(finite, z, 0.0).sum(axis=(0, 2))
        count = finite.sum(axis=(0, 2))
        if name in self.total:
            self.total[name] += total
            self.count[name] += count
        else:
            self.total[name], self.count[name] = total, count.astype(np.int64)

    def means(self, task: TaskSpec) -> tuple[dict[str, np.ndarray], int]:
        """(name -> (C,) standardized training mean, number of channels with no finite target at all).
        Such a channel gets TokaMark's mean, which is 0 standardized."""
        out, missing = {}, 0
        for s in task.outputs:
            if s.name not in self.total:
                raise ValueError(f"no windows were seen for output {s.name}")
            count = self.count[s.name]
            m = np.zeros(count.shape)
            np.divide(self.total[s.name], count, out=m, where=count > 0)
            out[s.name] = m
            missing += int((count == 0).sum())
        return out, missing


class _Predictor:
    NAME = ""

    def __init__(self, task: TaskSpec):
        self.task = task
        self.mean: dict[str, np.ndarray] = {}      # output name -> (C,) standardized training mean
        self.n_windows = 0
        self.n_channels_without_target = 0
        self.lam: float | None = None

    def arrays(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    @classmethod
    def from_arrays(cls, task: TaskSpec, arrays: dict[str, np.ndarray], meta: dict | None = None):
        raise NotImplementedError

    def _restore_counts(self, meta: dict | None) -> None:
        if meta:
            self.n_windows = int(meta["n_windows"])
            self.n_channels_without_target = int(meta["n_channels_without_target"])

    def fit_record(self) -> dict:
        """What the fit knows that the score should carry: `run_headroom.py` copies it into `score.json`."""
        return {"fit_n_windows": int(self.n_windows),
                "fit_n_channels_without_target": int(self.n_channels_without_target)}

    def fit_meta(self) -> dict:
        """Extra keys of the metadata record in the fit file (`save_fit`); `RidgeFrames` adds its penalty record."""
        return {}

    def _mean_arrays(self) -> dict[str, np.ndarray]:
        if not self.mean:
            raise ValueError(f"{self.NAME} is not fitted")
        return {f"mean/{name}": m for name, m in self.mean.items()}

    def _mean_from(self, arrays: dict[str, np.ndarray]) -> None:
        self.mean = {s.name: np.asarray(arrays[f"mean/{s.name}"], dtype=np.float64) for s in self.task.outputs}


# ------------------------------------------------------------------------------------ shrunk persistence

def _anchor(batch: WindowBatch, s: SignalSpec) -> np.ndarray:
    """(B, C) standardized anchor, NaN where there is none: the last finite input sample, or for a
    spectrogram the mean over the kept history, per bin."""
    n = len(batch.shot_ids)
    if _is_spectrogram(s):
        return _nanmean_last(_standardized(s, batch.inputs[s.name]))
    return s.standardize(np.asarray(batch.last_valid[s.name], dtype=np.float64)).reshape(n, -1)


class ShrunkPersistence(_Predictor):
    NAME = "shrunk"

    def __init__(self, task: TaskSpec):
        super().__init__(task)
        self.slope: dict[str, np.ndarray] = {}     # anchored output name -> (K,)
        self._anchored = tuple(s for s in task.outputs if any(i.name == s.name for i in task.inputs))

    def fit(self, batches: Iterable[WindowBatch]) -> "ShrunkPersistence":
        means = _ChannelMeans()
        # Per (channel, frame) sums over the windows where both the anchor and the target frame are finite.
        sums: dict[str, dict[str, np.ndarray]] = {}
        self.n_windows = 0
        for batch in batches:
            self.n_windows += len(batch.shot_ids)
            for s in self.task.outputs:
                z = _standardized(s, batch.targets[s.name])
                means.add(s.name, z)
                if s not in self._anchored:
                    continue
                p = _anchor(batch, s)[..., None]                       # (B, C, 1)
                y = _target_frames(z, s)                               # (B, C, K)
                ok = np.isfinite(p) & np.isfinite(y)
                pp, yy = np.where(ok, p, 0.0), np.where(ok, y, 0.0)
                new = {"n": ok.sum(axis=0), "p": pp.sum(axis=0), "y": yy.sum(axis=0),
                       "pp": (pp * pp).sum(axis=0), "py": (pp * yy).sum(axis=0)}
                if s.name in sums:
                    for k, v in new.items():
                        sums[s.name][k] = sums[s.name][k] + v
                else:
                    sums[s.name] = {k: v.astype(np.float64) for k, v in new.items()}
        if self.n_windows == 0:
            raise ValueError("fit received no windows")
        self.mean, self.n_channels_without_target = means.means(self.task)
        self.slope = {}
        for s in self._anchored:
            t, m = sums[s.name], self.mean[s.name][:, None]
            sxx = t["pp"] - 2 * m * t["p"] + t["n"] * m * m
            sxy = t["py"] - m * t["p"] - m * t["y"] + t["n"] * m * m
            den = sxx.sum(axis=0)
            self.slope[s.name] = np.divide(sxy.sum(axis=0), den, out=np.zeros_like(den), where=den > 0)
        return self

    def predict(self, batch: WindowBatch) -> dict[str, np.ndarray]:
        if not self.mean:
            raise ValueError("ShrunkPersistence is not fitted")
        n = len(batch.shot_ids)
        out = {}
        for s in self.task.outputs:
            m = self.mean[s.name][None, :, None]                       # (1, C, 1)
            if s in self._anchored:
                p = _anchor(batch, s)[..., None]
                p = np.where(np.isfinite(p), p, m)                     # no anchor: the training mean
                frames = m + self.slope[s.name][None, None, :] * (p - m)
            else:
                frames = np.broadcast_to(m, (n, m.shape[1], _n_frames(self.task, s)))
            out[s.name] = _physical(frames, s, self.task.n_out(s))
        return out

    def arrays(self) -> dict[str, np.ndarray]:
        return {**self._mean_arrays(), **{f"slope/{n}": a for n, a in self.slope.items()}}

    @classmethod
    def from_arrays(cls, task, arrays, meta=None):
        p = cls(task)
        p._mean_from(arrays)
        p.slope = {s.name: np.asarray(arrays[f"slope/{s.name}"], dtype=np.float64) for s in p._anchored}
        p._restore_counts(meta)
        return p


# ------------------------------------------------------------------------------------ ridge

class RidgeFrames(_Predictor):
    NAME = "ridge"
    # The ten decades. The first grid (1e-4 ... 10) had its optimum at the lower edge on five of seven tasks, so
    # the number was "ridge at the smallest penalty offered"; `select` now reports when the choice is at an edge.
    lambdas = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)

    def __init__(self, task: TaskSpec):
        super().__init__(task)
        self.W: np.ndarray | None = None
        self.candidates: tuple[float, ...] = ()       # the penalties whose solve gave finite weights
        self.at_edge: str | None = None               # "lower" / "upper" if the choice is the first / last candidate
        self._moments: tuple[np.ndarray, np.ndarray, int] | None = None   # X'X, X'Y centred, n

    # ---- layout of the Q regression targets: for each output signal, channel-major (C, K)
    def _centres(self) -> np.ndarray:
        return np.concatenate([np.repeat(self.mean[s.name], _n_frames(self.task, s)) for s in self.task.outputs])

    def _design(self, batch: WindowBatch) -> tuple[np.ndarray, list[np.ndarray]]:
        return window_features(batch, self.task), [
            _target_frames(_standardized(s, batch.targets[s.name]), s).reshape(len(batch.shot_ids), -1)
            for s in self.task.outputs]

    def fit(self, batches: Iterable[WindowBatch]) -> "RidgeFrames":
        n_f, n_q = feature_count(self.task), target_count(self.task)
        xtx, a, b = np.zeros((n_f, n_f)), np.zeros((n_f, n_q)), np.zeros((n_f, n_q))
        means, n = _ChannelMeans(), 0
        for batch in batches:
            x, ys = self._design(batch)
            if x.shape[1] != n_f:
                raise ValueError(f"window_features has {x.shape[1]} columns, feature_count says {n_f}")
            for s in self.task.outputs:
                means.add(s.name, _standardized(s, batch.targets[s.name]))
            y = np.concatenate(ys, axis=1)
            valid = np.isfinite(y)
            xtx += x.T @ x
            a += x.T @ np.where(valid, y, 0.0)
            b += x.T @ valid.astype(np.float64)
            n += len(x)
        if n == 0:
            raise ValueError("fit received no windows")
        self.n_windows = n
        self.mean, self.n_channels_without_target = means.means(self.task)
        self._moments = (xtx, a - b * self._centres()[None, :], n)
        self.W, self.lam, self.candidates, self.at_edge = None, None, (), None
        return self

    def _solve(self, lam: float) -> np.ndarray:
        if self._moments is None:
            raise ValueError("RidgeFrames is not fitted")
        xtx, xty, n = self._moments
        m = xtx.copy()
        idx = np.arange(m.shape[0] - 1)                                 # the constant (last column) is not penalised
        m[idx, idx] += lam * n
        return cho_solve(cho_factor(m, lower=True, check_finite=False), xty, check_finite=False)

    def _solved(self) -> dict[float, np.ndarray]:
        """penalty -> weights for the penalties whose solve gives finite weights. The others (a Cholesky
        failure, or a non-finite solution: the smallest penalties of a rank-deficient `X'X`) are printed and
        are not candidates."""
        weights, dropped = {}, []
        for lam in self.lambdas:
            try:
                w = self._solve(lam)
            except np.linalg.LinAlgError as e:
                dropped.append(f"{lam:g} ({e})")
                continue
            if np.isfinite(w).all():
                weights[lam] = w
            else:
                dropped.append(f"{lam:g} (non-finite weights)")
        if dropped:
            print(f"[cheap] ridge {self.task.name}: penalties that are not a candidate: {', '.join(dropped)}",
                  flush=True)
        return weights

    def select(self, val_batches: Iterable[WindowBatch]) -> float:
        """Choose lambda by mean squared error in standardized units over the finite validation frames.

        Among the penalties that solve (`candidates`); `at_edge` says whether the choice is the smallest
        ("lower") or the largest ("upper") of them, in which case the optimum is not bracketed."""
        weights = self._solved()
        if not weights:
            raise ValueError(f"no penalty of {self.lambdas} gives finite ridge weights")
        sse, count, centres = dict.fromkeys(weights, 0.0), 0, self._centres()
        for batch in val_batches:
            x, ys = self._design(batch)
            y = np.concatenate(ys, axis=1)
            valid = np.isfinite(y)
            count += int(valid.sum())
            for lam, w in weights.items():
                err = np.where(valid, x @ w + centres - np.where(valid, y, 0.0), 0.0)
                sse[lam] += float((err * err).sum())
        if count == 0:
            raise ValueError("select received no finite validation targets")
        best = min(weights, key=lambda lam: sse[lam])
        self.lam, self.W, self.candidates = best, weights[best], tuple(weights)
        self.at_edge = "lower" if best == self.candidates[0] else "upper" if best == self.candidates[-1] else None
        return best

    def fit_record(self) -> dict:
        return {**super().fit_record(), "fit_lambda": None if self.lam is None else float(self.lam),
                "fit_lambda_at_edge": self.at_edge}

    def fit_meta(self) -> dict:
        return {"lambda_grid": list(self.lambdas), "lambda_candidates": list(self.candidates),
                "lambda_at_edge": self.at_edge}

    def predict(self, batch: WindowBatch) -> dict[str, np.ndarray]:
        if self.W is None:
            raise ValueError("RidgeFrames has no weights: call select() first")
        z = window_features(batch, self.task) @ self.W + self._centres()[None, :]
        out, start, n = {}, 0, len(batch.shot_ids)
        for s in self.task.outputs:
            c, k = _n_channels(s), _n_frames(self.task, s)
            out[s.name] = _physical(z[:, start:start + c * k].reshape(n, c, k), s, self.task.n_out(s))
            start += c * k
        return out

    def arrays(self) -> dict[str, np.ndarray]:
        if self.W is None:
            raise ValueError("RidgeFrames has no weights to save: call select() first")
        return {**self._mean_arrays(), "W": self.W, "lam": np.array(self.lam)}

    @classmethod
    def from_arrays(cls, task, arrays, meta=None):
        p = cls(task)
        p._mean_from(arrays)
        p.W, p.lam = np.asarray(arrays["W"], dtype=np.float64), float(arrays["lam"])
        p._restore_counts(meta)
        if meta:
            p.candidates, p.at_edge = tuple(float(v) for v in meta["lambda_candidates"]), meta["lambda_at_edge"]
        return p


PREDICTOR_CLASSES = {c.NAME: c for c in (ShrunkPersistence, RidgeFrames)}


# ------------------------------------------------------------------------------------ fit files

def train_shots() -> list[int]:
    """The fixed random 3,000 shots of the training split (seed 0), sorted: the same for every task and predictor."""
    train = tasks.split_shots("train")
    return sorted(int(s) for s in np.random.default_rng(0).choice(train, N_TRAIN_SHOTS, replace=False))


def select_shots(n: int = N_SELECT_SHOTS) -> list[int]:
    """`n` shots evenly spaced over the sorted `val_select` list, first and last included: the shots that
    choose ridge's penalty.

    This is the rule of `direct.validation_shots`, positions `linspace(0, len - 1, min(n, len)).round()`;
    it is written twice because `direct.py` imports this module (and must not change, ten trainings run from
    it). `test_select_shots_are_the_direct_models_validation_shots` pins the two equal. The list is sorted by
    shot number, so its first 200 shots are one old campaign (measured 2026-11-22: only 65 of them have
    windows for tasks 4-x, and `cc_field` is nearly constant there), which is why the selection is spread.
    """
    selection = tasks.split_shots("val_select")
    positions = np.linspace(0, len(selection) - 1, num=min(n, len(selection))).round().astype(int)
    return [selection[i] for i in positions]


def training_meta() -> dict:
    """What, besides this file, defines a fit: the training shot list, the window stride and the scoring
    code (harness.py, tasks.py, baselines.py: the training windows and the standardization come from
    them), as the hash `score.json` stores as `code_sha256`."""
    return {"shots_sha256": hashlib.sha256(",".join(map(str, train_shots())).encode()).hexdigest(),
            "stride_s": TRAIN_STRIDE_S, "scoring_code_sha256": baselines.scoring_code_sha256()}


def fit_path(root: Path, task_name: str, name: str) -> Path:
    return Path(root) / "cheap" / task_name / f"{name}.npz"


def save_fit(path: Path, predictor: _Predictor, extra: dict) -> str:
    """Write the fitted arrays and the metadata record atomically (temporary name, then rename); return the
    sha256 of the bytes written."""
    path = Path(path)
    meta = {"predictor": predictor.NAME, "task": predictor.task.name, "code_sha256": PREDICTOR_SHA256,
            "n_windows": int(predictor.n_windows),
            "n_channels_without_target": int(predictor.n_channels_without_target),
            "lambda": None if predictor.lam is None else float(predictor.lam), **predictor.fit_meta(), **extra}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **predictor.arrays(), meta=np.array(json.dumps(meta)))
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        os.replace(tmp, path)
        return digest
    finally:
        tmp.unlink(missing_ok=True)


def _parse_fit(data: bytes) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: z[k] for k in z.files if k != "meta"}, json.loads(str(z["meta"]))


def load_fit(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    return _parse_fit(Path(path).read_bytes())


# What a ridge fit file must hold to be read by this code (a file from before the penalty record is fitted again).
_RIDGE_META = ("lambda_grid", "lambda_candidates", "lambda_at_edge")


def _read_current_fit(path: Path, name: str, task_name: str):
    """((arrays, meta, digest), "") for a fit file that matches the present code, shot list, stride and scoring
    code, else (None, why not). The file is read once: the arrays, the metadata and the digest are of the same
    bytes."""
    path = Path(path)
    if not path.exists():
        return None, "no fit file"
    try:
        data = path.read_bytes()
        arrays, meta = _parse_fit(data)
    except Exception as e:  # a truncated or foreign file is fitted again, not a crash
        return None, f"unreadable fit file ({type(e).__name__})"
    if not isinstance(meta, dict):
        return None, "unreadable fit file (no metadata record)"
    expected = {"predictor": name, "task": task_name, "code_sha256": PREDICTOR_SHA256, **training_meta()}
    for key, value in expected.items():
        if meta.get(key) != value:
            return None, f"{'code' if key == 'code_sha256' else key} changed"
    if not any(k.startswith("mean/") for k in arrays):
        return None, "no fitted arrays"
    if name == "ridge":
        if meta.get("lambda") is None:
            return None, "no lambda chosen"
        if any(k not in meta for k in _RIDGE_META) or "W" not in arrays or "lam" not in arrays:
            return None, "fit record incomplete (a fit from before the penalty record)"
    return (arrays, meta, hashlib.sha256(data).hexdigest()), ""


def current_fit_digest(path: Path, name: str, task_name: str) -> str:
    """sha256 of the fit file's bytes if it is reusable now, else `NO_CURRENT_FIT` (which no stored score has)."""
    found, _ = _read_current_fit(path, name, task_name)
    return found[2] if found else NO_CURRENT_FIT


def obtain(name: str, task: TaskSpec, root: Path, *, num_workers: int, batch_size: int,
           force: bool) -> tuple[_Predictor, str]:
    """(the fitted predictor, sha256 of the fit file's bytes): from its fit file under `root` if that is
    current, else fitted now and saved. The digest is of the bytes the predictor was made from (read once, or
    written by `save_fit`), never of a second read of the path."""
    cls = PREDICTOR_CLASSES[name]
    path = fit_path(root, task.name, name)
    if force:
        why = "--force"
    else:
        found, why = _read_current_fit(path, name, task.name)
        if found:
            try:
                predictor = cls.from_arrays(task, found[0], found[1])
            except (KeyError, ValueError, TypeError) as e:   # the record passed, the arrays do not fit this task
                why = f"unreadable fit file ({type(e).__name__}: {e})"
            else:
                print(f"[cheap] reusing {path} ({found[1]['n_windows']} windows)", flush=True)
                return predictor, found[2]
    shots = train_shots()
    print(f"[cheap] fitting {name} on {task.name} ({why}): {len(shots)} training shots, stride {TRAIN_STRIDE_S} s",
          flush=True)
    started = time.monotonic()
    predictor = cls(task)
    predictor.fit(harness.train_windows(task, shots, stride_s=TRAIN_STRIDE_S, shuffle=False,
                                        batch_size=batch_size, num_workers=num_workers))
    fitted = time.monotonic()
    print(f"[cheap] {name} {task.name}: {predictor.n_windows} training windows, {fitted - started:.0f} s; "
          f"{predictor.n_channels_without_target} output channels never have a finite target", flush=True)
    if isinstance(predictor, RidgeFrames):
        chosen_on = select_shots()
        lam = predictor.select(harness.iter_windows(task, chosen_on, test_mode=True,
                                                    batch_size=batch_size, num_workers=num_workers))
        print(f"[cheap] ridge {task.name}: lambda {lam:g} chosen on {len(chosen_on)} val_select shots spread over "
              f"the sorted list, {time.monotonic() - fitted:.0f} s; candidates "
              f"{', '.join(f'{c:g}' for c in predictor.candidates)}", flush=True)
        if predictor.at_edge:
            print(f"[cheap] ridge {task.name}: the choice is at the {predictor.at_edge} edge of the candidates, so the "
                  f"optimum is not bracketed; the score is ridge at the "
                  f"{'smallest' if predictor.at_edge == 'lower' else 'largest'} penalty that solved", flush=True)
    return predictor, save_fit(path, predictor, training_meta())
