"""Per-channel statistics and the unit transform between physical and model units.

The model sees every channel standardized: ``z = (c(x) - mean) / std`` where ``c`` is the
identity (``compress == "none"``) or ``asinh(x / scale)`` (``"asinh"``, chosen per signal for
heavy-tailed signals). The constants are fitted once on a sample of training shots
(`fit_stats`), saved in one ``.npz`` (`save_stats`, `stats_sha256`) and never refitted.

Why a sample and not a full pass: the constants only have to be fixed and reasonable, and
128,000 samples (2,000 shots x 64) put the error of a standard deviation under 1%.

`tokamark` (through `harness`) is imported inside `load_shot`: this module imports without it.
Nothing here takes an absolute value: the sign of the plasma current is kept as recorded, and
NaN (a missing sample) stays NaN through both transforms.
"""

from __future__ import annotations

import hashlib
import io
import multiprocessing
import os
import zipfile
import zlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import prepare_write_root
from .tasks import SignalSpec

MIN_SAMPLES = 1000              # fewer finite samples than this: the channel is not fitted
KURTOSIS_LIMIT = 10.0           # a signal whose median channel is above this gets asinh
MAD_TO_STD = 1.4826             # scaled median absolute deviation = std of a gaussian
_CHANNEL_CHUNK = 256            # channels fitted at a time (bounds the float64 working set)
_LISTED = 20                    # degenerate channels named in the printed summary


@dataclass
class ChannelStats:
    """Constants of one signal. The arrays have the signal's ``values_shape``."""

    compress: str               # "none" | "asinh"
    scale: np.ndarray           # asinh scale (1 where compress == "none")
    mean: np.ndarray            # of the compressed values
    std: np.ndarray


# ------------------------------------------------------------------------------ the transform

def _per_channel(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """A values_shape-shaped constant, broadcast against ``x`` of shape (*values_shape, T)."""
    a = np.asarray(a, dtype=np.float64)
    if x.ndim != a.ndim + 1 or x.shape[: a.ndim] != a.shape:
        raise ValueError(f"data of shape {x.shape} does not match constants of shape {a.shape} "
                         "(expected (*values_shape, T))")
    return a[..., None]


def to_model(x: np.ndarray, st: ChannelStats) -> np.ndarray:
    """Physical units (*values_shape, T) to standardized units. Float64; NaN stays NaN."""
    x = np.asarray(x)
    scale, mean, std = (_per_channel(a, x) for a in (st.scale, st.mean, st.std))
    y = np.arcsinh(x / scale) if st.compress == "asinh" else x.astype(np.float64)
    return (y - mean) / std


def to_physical(z: np.ndarray, st: ChannelStats) -> np.ndarray:
    """Inverse of `to_model`."""
    z = np.asarray(z)
    scale, mean, std = (_per_channel(a, z) for a in (st.scale, st.mean, st.std))
    y = z * std + mean
    return np.sinh(y) * scale if st.compress == "asinh" else y


# ---------------------------------------------------------------------------------- the loader

def load_shot(shot: int, signals: Sequence[SignalSpec]) -> dict[str, dict[str, np.ndarray]]:
    """One shot through TokaMark's shot-level loader, in physical units.

    The one place that opens the dataset is `harness.open_mast_dataset`, so outlier removal,
    the x-point clip, the boundary resampling and the Mirnov spectrograms are TokaMark's by
    construction. Returns ``{"<source>-<signal>": {"time": (T,), "values": (*values_shape, T)}}``;
    a scalar signal is (1, T). A signal the shot does not have comes back with ``time`` and
    ``values`` of shape (0,).
    """
    from . import harness

    pairs = [list(s.name.split("-", 1)) for s in signals]
    ds = harness.open_mast_dataset({"sources_and_signals": {"input_name": pairs}}, [shot])
    item = ds[0]
    return {s.name: item[s.name] for s in signals if s.name in item}


# ------------------------------------------------------------------------------- the sampling

def _channels(item: dict, spec: SignalSpec) -> np.ndarray | None:
    """The signal as (C, T), or None if the shot does not have it (or has another shape)."""
    d = item.get(spec.name)
    if d is None:
        return None
    v = np.asarray(d["values"])
    shape = tuple(spec.values_shape)
    if v.ndim != len(shape) + 1 or v.shape[:-1] != shape or v.shape[-1] == 0:
        return None
    return v.reshape(-1, v.shape[-1])


def _sample_shot(args: tuple) -> tuple[int, dict | None, str | None]:
    """Up to ``n`` finite samples per channel of one shot, NaN-padded: ``{name: (C, n) float32}``.

    The generator is seeded from the seed, the shot and the signal name, never from the order
    in which shots or signals arrive, so the sample does not depend on the number of workers.
    """
    shot, signals, n, seed = args
    try:
        item = load_shot(shot, signals)
    except Exception as e:  # a shot that cannot be read is counted by the caller, not hidden
        return shot, None, f"{type(e).__name__}: {e}"
    out: dict[str, np.ndarray | None] = {}
    for spec in signals:
        v = _channels(item, spec)
        if v is None:
            out[spec.name] = None
            continue
        rng = np.random.default_rng([seed, shot, zlib.crc32(spec.name.encode())])
        s = np.full((v.shape[0], n), np.nan, dtype=np.float32)
        for c in range(v.shape[0]):
            x = v[c][np.isfinite(v[c])]
            if x.size > n:
                x = x[rng.choice(x.size, n, replace=False)]
            s[c, : x.size] = x
        out[spec.name] = s
    return shot, out, None


def map_shots(fn, shots: Sequence[int], args: tuple, workers: int) -> Iterator:
    """``fn((shot, *args))`` for each shot, in the order of ``shots``.

    ``workers <= 1`` runs in the calling process. More starts a process pool with the *spawn*
    start method: zarr runs an event-loop thread, and a forked child of a process that has used
    zarr can deadlock.
    """
    jobs = [(int(s), *args) for s in shots]
    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            yield fn(job)
        return
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(min(workers, len(jobs))) as pool:
        yield from pool.imap(fn, jobs, chunksize=1)


# ------------------------------------------------------------------------------- the statistics

def _excess_kurtosis(x: np.ndarray) -> np.ndarray:
    """Excess kurtosis of each row of ``x`` (C, n), ignoring NaN; NaN for a row of zero spread.

    Shift and scale invariant, so it is also the kurtosis of ``(x - median) / s``.
    """
    fin = np.isfinite(x)
    cnt = fin.sum(axis=1)
    xz = np.where(fin, x, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(fin, xz - (xz.sum(axis=1) / cnt)[:, None], 0.0)
        m2 = (d ** 2).sum(axis=1) / cnt
        m4 = (d ** 4).sum(axis=1) / cnt
        k = m4 / m2 ** 2 - 3.0
    k[~(m2 > 0)] = np.nan
    return k


def _row_moments(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population std of each row of ``x`` (C, n), ignoring NaN."""
    fin = np.isfinite(x)
    cnt = fin.sum(axis=1)
    xz = np.where(fin, x, 0.0)
    mean = xz.sum(axis=1) / cnt
    var = np.where(fin, xz - mean[:, None], 0.0)
    return mean, np.sqrt((var ** 2).sum(axis=1) / cnt)


def _fit_signal(pooled: np.ndarray) -> tuple[ChannelStats, np.ndarray]:
    """Statistics of one signal from its sample ``pooled`` (C, m), NaN where there is none.

    Returns the constants (flat arrays of length C, in a `ChannelStats` the caller reshapes)
    and the boolean mask of degenerate channels (fewer than `MIN_SAMPLES` finite samples, or
    zero spread), which get ``mean = 0, std = 1``.
    """
    n_ch = pooled.shape[0]
    n_fin = np.isfinite(pooled).sum(axis=1)
    median = np.zeros(n_ch)
    scale = np.ones(n_ch)
    kurt = np.full(n_ch, np.nan)
    degenerate = n_fin < MIN_SAMPLES
    spread = np.zeros(n_ch)
    for lo in range(0, n_ch, _CHANNEL_CHUNK):
        sl = slice(lo, min(lo + _CHANNEL_CHUNK, n_ch))
        ok = np.flatnonzero(~degenerate[sl]) + lo
        if ok.size == 0:
            continue
        x = pooled[ok].astype(np.float64)               # every row here has >= MIN_SAMPLES finite values
        med = np.nanmedian(x, axis=1)
        mad = np.nanmedian(np.abs(x - med[:, None]), axis=1)
        _, std = _row_moments(x)
        median[ok], spread[ok] = med, std
        scale[ok] = np.where(mad > 0, MAD_TO_STD * mad, std)    # MAD 0: the sample std is the scale
        kurt[ok] = _excess_kurtosis(x)
    degenerate |= ~(spread > 0)
    scale[degenerate] = 1.0
    kurt[degenerate] = np.nan
    asinh = bool(np.any(~degenerate) and np.median(kurt[~degenerate]) > KURTOSIS_LIMIT)

    mean, std = np.zeros(n_ch), np.ones(n_ch)
    for lo in range(0, n_ch, _CHANNEL_CHUNK):
        ok = np.flatnonzero(~degenerate[lo:lo + _CHANNEL_CHUNK]) + lo
        if ok.size == 0:
            continue
        x = pooled[ok].astype(np.float64)
        if asinh:
            x = np.arcsinh(x / scale[ok, None])
        mean[ok], std[ok] = _row_moments(x)
    if asinh:
        compress, out_scale = "asinh", scale
    else:
        compress, out_scale = "none", np.ones(n_ch)
    return ChannelStats(compress, out_scale, mean, std), degenerate


def fit_stats(shots: Sequence[int], signals: Sequence[SignalSpec], *, samples_per_channel: int = 64,
              seed: int = 0, workers: int = 64) -> dict[str, ChannelStats]:
    """Fit the constants of ``signals`` on ``shots`` (the caller passes training shots only).

    Per shot and channel, ``samples_per_channel`` finite samples are drawn without replacement
    (all of them if there are fewer). Per channel: the median, the scaled MAD ``s`` (1.4826 x
    MAD), and the excess kurtosis. A signal whose median channel has excess kurtosis above 10
    gets ``compress = "asinh"`` with ``scale = s`` per channel (the sample standard deviation
    where ``s`` is 0); the others get ``"none"``. Then the mean and standard deviation of the
    compressed sample. A channel with fewer than 1,000 finite samples or zero spread gets
    ``mean = 0, std = 1``.

    The same ``shots``, ``signals``, ``samples_per_channel`` and ``seed`` give identical
    statistics whatever the number of ``workers``, the order of ``shots`` and of ``signals``.
    ``workers <= 1`` runs in the calling process. A shot that cannot be loaded, and a shot that
    lacks a signal, contribute nothing for it; both are counted, as are the degenerate channels,
    in one summary printed at the end. A shot outside TokaMark's splits raises ``ValueError``.
    """
    shots = sorted({int(s) for s in shots})
    signals = list(signals)
    if len({s.name for s in signals}) != len(signals):
        raise ValueError("duplicate signal names")
    n = int(samples_per_channel)
    n_ch = {s.name: int(np.prod(s.values_shape)) for s in signals}
    pools = {name: np.full((len(shots), c, n), np.nan, dtype=np.float32) for name, c in n_ch.items()}
    failed: list[tuple[int, str]] = []
    without = dict.fromkeys(n_ch, 0)
    results = map_shots(_sample_shot, shots, (signals, n, seed), workers)
    for i, (shot, out, err) in enumerate(results):
        if out is None:
            if "ValueError" in err and "never loaded" in err:
                raise ValueError(err.split(": ", 1)[1])     # the split rule is a bug in the caller
            failed.append((shot, err))
            continue
        for name, s in out.items():
            if s is None:
                without[name] += 1
            else:
                pools[name][i] = s

    stats: dict[str, ChannelStats] = {}
    degenerate: dict[str, list[int]] = {}
    for spec in signals:
        flat = pools.pop(spec.name).transpose(1, 0, 2).reshape(n_ch[spec.name], -1)
        st, dead = _fit_signal(flat)
        shape = tuple(spec.values_shape)
        stats[spec.name] = ChannelStats(st.compress, st.scale.reshape(shape), st.mean.reshape(shape),
                                        st.std.reshape(shape))
        degenerate[spec.name] = [int(c) for c in np.flatnonzero(dead)]

    print(f"fit_stats: {len(shots) - len(failed)} of {len(shots)} shots loaded, "
          f"{len(failed)} shots failed to load, seed {seed}, {n} samples per channel and shot")
    for shot, err in failed[:3]:
        print(f"  shot {shot}: {err[:200]}")
    for name in n_ch:
        if without[name]:
            print(f"  {name}: {without[name]} shots without the signal")
        d = degenerate[name]
        if len(d) <= _LISTED and d:
            print(f"  {name}: degenerate channels {d}")
        elif d:
            print(f"  {name}: {len(d)} of {n_ch[name]} channels degenerate, first {_LISTED}: {d[:_LISTED]}")
    return stats


# --------------------------------------------------------------------------------- the file

_FIELDS = ("compress", "scale", "mean", "std")
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def save_stats(stats: dict[str, ChannelStats], path: Path) -> None:
    """One ``.npz`` (readable by ``np.load``), keys ``"<signal>::<field>"``.

    Written entry by entry in sorted order with a fixed time stamp, so the same statistics
    always give the same bytes and `stats_sha256` is a version of the numbers.
    """
    path = Path(path)
    prepare_write_root(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
        for name in sorted(stats):
            st = stats[name]
            arrays = {"compress": np.array(st.compress), "scale": np.asarray(st.scale, dtype=np.float64),
                      "mean": np.asarray(st.mean, dtype=np.float64), "std": np.asarray(st.std, dtype=np.float64)}
            for field in _FIELDS:
                buf = io.BytesIO()
                np.lib.format.write_array(buf, arrays[field], allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}::{field}.npy", date_time=_FIXED_TIME)
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o600 << 16
                zf.writestr(info, buf.getvalue())
    os.replace(tmp, path)


def load_stats(path: Path) -> dict[str, ChannelStats]:
    out: dict[str, ChannelStats] = {}
    with np.load(Path(path), allow_pickle=False) as z:
        names = sorted({k.rsplit("::", 1)[0] for k in z.files})
        for name in names:
            out[name] = ChannelStats(str(z[f"{name}::compress"]), z[f"{name}::scale"],
                                     z[f"{name}::mean"], z[f"{name}::std"])
    return out


def stats_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
