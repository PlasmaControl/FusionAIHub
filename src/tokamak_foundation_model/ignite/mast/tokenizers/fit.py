"""The fit of the fixed tokenizers: the principal-component bases and the ranges of every value.

Everything here is fitted on the cached values of training shots (model units, float32, NaN for
missing) and nothing else. A `frame_dct` or `band_dct` signal has no weights: only the range of each
value, which turns a value into one of `N_BINS` bins. A `linear` signal (a profile or a map) has a
basis too: the principal components of its samples, fitted from every pair of valid points
(`codecs.fit_pca`), with `MAX_COMPONENTS` components at most. Both are fitted once, at the signal's
highest rung, and serve every lower rung: the first columns of a range array are the ranges of the
lower rung (the cosine values and the components are nested), so `build_codec` only slices them.

A signal is fitted in each of its domains (`assign.domains`), because the floors are compared across
domains. `fit_all` reads one signal at a time over all the shots, so it holds one signal's values and
no more. A signal no shot has is reported with `n_shots == 0`, never as a fit of nothing.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tokamak_foundation_model.ignite.mast import cache, registry, transform
from tokamak_foundation_model.ignite.mast.paths import prepare_write_root
from tokamak_foundation_model.ignite.mast.tasks import SignalSpec
from tokamak_foundation_model.ignite.mast.tokenizers import assign, codecs, frames
from tokamak_foundation_model.ignite.mast.tokenizers.grid import GridSet, fit_ranges
from tokamak_foundation_model.ignite.mast.transform import ChannelStats

MAX_COMPONENTS = 96             # principal components fitted (the top of `assign.PCA_RUNGS`)
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)      # zip time stamp: the same fit always gives the same bytes

Ranges = tuple[np.ndarray, np.ndarray, np.ndarray]                    # lo, hi, width: (n_tok, n_val)
Pca = tuple[np.ndarray, np.ndarray, np.ndarray, float]                # mean, basis, eigenvalues, total


@dataclass
class SignalFit:
    """What was fitted for one signal.

    `n_shots` shots had the signal and `n_skipped` did not (None). `n_examples` counts the frames with
    at least one value (`frame_dct`, `band_dct`) or the samples with at least one valid point
    (`linear`), over all shots. `partly_valid` is True if some example had missing points and valid
    ones. `ranges` maps a domain to ``(lo, hi, width)``, each ``(n_tok, n_val)`` float64 in the
    domain's unit, at the signal's highest rung (``(1, min(MAX_COMPONENTS, D))`` for `linear`).
    `pca` maps a domain to ``(mean (D,), basis (n, D), eigenvalues (n,), total_variance)`` in that
    domain's unit; it is empty unless `kind == "linear"`.
    """

    name: str
    kind: str
    n_shots: int
    n_skipped: int
    n_examples: int
    partly_valid: bool
    ranges: dict[str, Ranges] = field(default_factory=dict)
    pca: dict[str, Pca] = field(default_factory=dict)


# ------------------------------------------------------------------------------------- the fit

def _partly_valid_frames(z: np.ndarray, dt: float) -> bool:
    """True if some channel's frame of `z` holds both valid and missing samples.

    `z` is ``(*values_shape, T)``, any unit, NaN or inf for missing.
    """
    t = z.shape[-1]
    ok = np.isfinite(z).reshape(-1, t)
    f = frames.frame_index(t, dt)
    starts = np.flatnonzero(np.diff(f, prepend=-1))
    size = np.diff(starts, append=t)
    count = np.add.reduceat(ok, starts, axis=1, dtype=np.int64)
    return bool(((count > 0) & (count < size)).any())


def _top_codec(spec: SignalSpec, stats: ChannelStats, kind: str, domain: str) -> codecs.Codec:
    """The `frame_dct` or `band_dct` codec at the signal's highest rung, without grids."""
    top = assign.rungs(spec, domain)[-1]
    if kind == "band_dct":
        return codecs.BandDCT(spec.name, spec.dt, spec.values_shape, stats, top.k)
    return codecs.FrameDCT(spec.name, spec.dt, spec.values_shape, stats, top.k[0], domain,
                           signal_mean=spec.mean, signal_std=spec.std)


def _check_shape(spec: SignalSpec, z: np.ndarray) -> np.ndarray:
    z = np.asarray(z)
    if z.ndim != len(spec.values_shape) + 1 or z.shape[:-1] != tuple(spec.values_shape):
        raise ValueError(f"{spec.name}: a shot's values must be {tuple(spec.values_shape)} + (T,), "
                         f"got {z.shape}")
    return z


def _fit_series(spec: SignalSpec, stats: ChannelStats, shots_values: Iterable[np.ndarray | None],
                kind: str) -> SignalFit:
    """`frame_dct` and `band_dct`: encode each shot in every domain and fit the ranges of the values."""
    doms = assign.domains_of_kind(kind, stats.compress)
    top = {d: _top_codec(spec, stats, kind, d) for d in doms}
    enc: dict[str, list[np.ndarray]] = {d: [] for d in doms}
    n_shots = n_skipped = n_examples = 0
    partly = False
    for z in shots_values:
        if z is None:
            n_skipped += 1
            continue
        z = _check_shape(spec, z)
        n_shots += 1
        partly = partly or _partly_valid_frames(z, spec.dt)
        for d, c in top.items():
            v = c.encode_domain(c.domain_from_model(z[None]))[0]                # (F, n_tok, n_val)
            enc[d].append(v)
        n_examples += int(np.isfinite(enc[doms[0]][-1]).any(axis=(1, 2)).sum())
    out = SignalFit(spec.name, kind, n_shots, n_skipped, n_examples, partly)
    if n_shots:
        for d in doms:
            out.ranges[d] = fit_ranges(np.concatenate(enc[d], axis=0))
    return out


def _linear_rows(base: codecs.Codec, z: np.ndarray) -> np.ndarray:
    """Samples of a shot in the codec's domain: `z` ``(*values_shape, T)`` model units -> ``(T, D)`` float64."""
    u = base.domain_from_model(z[None])[0]                                       # (*values_shape, T)
    return np.moveaxis(u.reshape(-1, u.shape[-1]), 1, 0)


def _fit_linear(spec: SignalSpec, stats: ChannelStats, shots_values: Iterable[np.ndarray | None],
                kind: str) -> SignalFit:
    """`linear`: two passes over the shots, the basis first and then the ranges of the scores."""
    doms = assign.domains_of_kind(kind, stats.compress)
    d_pts = math.prod(spec.values_shape)
    n_comp = min(MAX_COMPONENTS, d_pts)
    held: list[np.ndarray] = []
    n_skipped = n_examples = 0
    partly = False
    for z in shots_values:
        if z is None:
            n_skipped += 1
            continue
        z = _check_shape(spec, z)
        ok = np.isfinite(z).reshape(d_pts, -1)                                  # (D, T)
        has = ok.any(axis=0)
        partly = partly or bool((has & ~ok.all(axis=0)).any())
        n_examples += int(has.sum())
        held.append(z)
    out = SignalFit(spec.name, kind, len(held), n_skipped, n_examples, partly)
    if not held:
        return out
    for d in doms:
        base = codecs.Codec(spec.name, spec.dt, spec.values_shape, stats, 1, 1, domain=d,
                            signal_mean=spec.mean, signal_std=spec.std)
        x = np.empty((sum(z.shape[-1] for z in held), d_pts))
        n_rows = 0
        for z in held:
            rows = _linear_rows(base, z)
            rows = rows[np.isfinite(rows).any(axis=1)]
            x[n_rows:n_rows + len(rows)] = rows
            n_rows += len(rows)
        mean, basis, eig, total = out.pca[d] = codecs.fit_pca(x[:n_rows], n_comp)
        del x
        n = len(eig)
        codec = codecs.LinearCodec(spec.name, spec.dt, spec.values_shape, stats, mean, basis, eig, n, n,
                                   d, _ridge(total, eig, n, d_pts, partly),
                                   signal_mean=spec.mean, signal_std=spec.std)
        scores = [codec.encode_domain(codec.domain_from_model(z[None]))[0] for z in held]   # (F, 1, n)
        out.ranges[d] = fit_ranges(np.concatenate(scores, axis=0))
    return out


def fit_signal(spec: SignalSpec, stats: ChannelStats, shots_values: Iterable[np.ndarray | None],
               kind: str | None = None) -> SignalFit:
    """Fit the ranges (and for `linear` the basis) of one signal, in each of its domains.

    `shots_values` yields, per shot, the cached values ``(*values_shape, T)`` float32 in model units
    (NaN for missing) or None for a shot without the signal. `kind` is ``"frame_dct"``, ``"linear"`` or
    ``"band_dct"``; None means `assign.codec_kind(spec)`. The codec is built at the highest rung; for
    `linear` after `codecs.fit_pca` on every sample with a valid point (`MAX_COMPONENTS` components, in one
    token). Every shot is encoded from model units and `grid.fit_ranges` runs over all the values. A signal
    no shot has gives ``n_shots == 0`` and empty dicts.
    """
    kind = kind or assign.codec_kind(spec)
    if kind not in assign.CODECS:
        raise ValueError(f"unknown codec {kind!r}; one of {assign.CODECS}")
    if kind == "linear":
        return _fit_linear(spec, stats, shots_values, kind)
    return _fit_series(spec, stats, shots_values, kind)


def _read_cached(job: tuple) -> np.ndarray | None:
    """``(shot, root, name, stats_sha256)`` to the signal of one cache file, float32 model units or None."""
    shot, root, name, sha = job
    got = cache.read_shot(Path(root) / "shots" / f"{shot}.h5", [name], stats_sha256=sha)[name]
    return None if got is None else got.values


def fit_all(cache_root: Path, shots: Sequence[int], *, workers: int,
            names: Sequence[str] | None = None) -> dict[str, SignalFit]:
    """Fit each signal of the cache over `shots`; needs `tokamark` (for the signal specs).

    Reads ``stats.npz`` and ``shots/<shot>.h5`` of `cache_root`, and refuses a shot file made from
    other statistics than ``manifest.json``'s `stats_sha256`. One signal is read over all the shots
    at a time (so memory holds one signal), with `workers` processes. `names` limits the signals
    (default: all 39); a name that is not a signal raises `KeyError`. Returns ``{name: SignalFit}``.
    """
    root = Path(cache_root)
    stats = transform.load_stats(root / "stats.npz")
    sha = json.loads((root / "manifest.json").read_text())["stats_sha256"]
    specs = {s.name: s for s in registry.all_signals()}
    if names is not None:
        unknown = [n for n in names if n not in specs]
        if unknown:
            raise KeyError(f"not signals of the 14 tasks: {unknown}")
        specs = {n: specs[n] for n in names}
    out: dict[str, SignalFit] = {}
    for name, spec in specs.items():
        values = transform.map_shots(_read_cached, shots, (str(root), name, sha), workers)
        out[name] = fit_signal(spec, stats[name], values)
    return out


# ------------------------------------------------------------------------------------- the file

_FIT_NAMES = ("lo", "hi", "width")
_PCA_NAMES = ("mean", "basis", "eig", "total")


def save_fit(fits: Mapping[str, SignalFit], out_dir: Path, meta: dict) -> None:
    """Write ``fit.npz`` (the arrays) and ``fit.json`` (`meta` and the counts) under `out_dir`.

    The ``.npz`` is written entry by entry in sorted order with a fixed time stamp, so the same fit
    always gives the same bytes and `fit_sha256` is a version of the numbers. `meta` must be JSON.
    """
    out_dir = prepare_write_root(Path(out_dir))
    entries: dict[str, np.ndarray] = {}
    info: dict[str, dict] = {}
    for name, f in fits.items():
        for d, rng in f.ranges.items():
            for part, a in zip(_FIT_NAMES, rng):
                entries[f"{name}::ranges::{d}::{part}"] = np.asarray(a, dtype=np.float64)
        for d, p in f.pca.items():
            for part, a in zip(_PCA_NAMES, p):
                entries[f"{name}::pca::{d}::{part}"] = np.asarray(a, dtype=np.float64)
        info[name] = {"kind": f.kind, "n_shots": f.n_shots, "n_skipped": f.n_skipped,
                      "n_examples": f.n_examples, "partly_valid": f.partly_valid,
                      "range_domains": list(f.ranges), "pca_domains": list(f.pca)}
    tmp = out_dir / "fit.npz.tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
        for key in sorted(entries):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, entries[key], allow_pickle=False)
            zi = zipfile.ZipInfo(f"{key}.npy", date_time=_FIXED_TIME)
            zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = 0o600 << 16
            zf.writestr(zi, buf.getvalue())
    os.replace(tmp, out_dir / "fit.npz")
    text = json.dumps({"meta": meta, "signals": info}, indent=2, sort_keys=True) + "\n"
    (out_dir / "fit.json.tmp").write_text(text)
    os.replace(out_dir / "fit.json.tmp", out_dir / "fit.json")


def load_fit(out_dir: Path) -> tuple[dict[str, SignalFit], dict]:
    """Read what `save_fit` wrote: ``({name: SignalFit}, meta)``."""
    out_dir = Path(out_dir)
    doc = json.loads((out_dir / "fit.json").read_text())
    fits: dict[str, SignalFit] = {}
    with np.load(out_dir / "fit.npz", allow_pickle=False) as z:
        for name, i in doc["signals"].items():
            f = SignalFit(name, i["kind"], i["n_shots"], i["n_skipped"], i["n_examples"], i["partly_valid"])
            for d in i["range_domains"]:
                f.ranges[d] = tuple(z[f"{name}::ranges::{d}::{p}"] for p in _FIT_NAMES)
            for d in i["pca_domains"]:
                mean, basis, eig, total = (z[f"{name}::pca::{d}::{p}"] for p in _PCA_NAMES)
                f.pca[d] = (mean, basis, eig, float(total))
            fits[name] = f
    return fits, doc["meta"]


def fit_sha256(out_dir: Path) -> str:
    """sha256 of ``fit.npz`` under `out_dir`."""
    h = hashlib.sha256()
    with open(Path(out_dir) / "fit.npz", "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------------- the codecs

def _ridge(total: float, eig: np.ndarray, n_values: int, d_pts: int, partly_valid: bool) -> float:
    """The ridge of a `LinearCodec` with `n_values` components: the variance they leave, per point.

    `total` and `eig` are in the codec's domain (variance); 0 if no example had missing points.
    """
    if not partly_valid:
        return 0.0
    return max(0.0, float(total - np.asarray(eig)[:n_values].sum()) / d_pts)


def _grids(rng: Ranges, n_tok: int, n_val: int, n_bins: int, what: str) -> GridSet:
    """The first `n_val` columns of fitted ``(lo, hi, width)`` as grids of `n_tok` rows; `ValueError` if too few."""
    if n_val > rng[0].shape[1]:
        raise ValueError(f"{what} needs {n_val} fitted values per token; the fit has {rng[0].shape[1]}")
    return GridSet(*(np.array(a[:n_tok, :n_val], dtype=np.float64) for a in rng), n_bins=n_bins)


def build_codec(spec: SignalSpec, stats: ChannelStats, fit: SignalFit, setting: assign.Setting) -> codecs.Codec:
    """The codec of `setting`, with grids from the fitted ranges (the first columns of the highest rung's).

    `frame_dct`: grids of the first ``K`` values of each channel. `linear`: the first `n_values`
    fitted components regrouped into tokens of `values_per_token`; the ridge is
    ``(total_variance - eigenvalues[:n_values].sum()) / D`` if ``fit.partly_valid``, else 0.
    `band_dct`: grids of the first ``max(k)`` values of each token. `signal_mean` and `signal_std` are
    `spec.mean` and `spec.std` for the first two (`band_dct` has no signal domain). `ValueError` names a
    domain the signal was not fitted in, and a `setting.codec` that is not one of `assign.CODECS`.
    """
    if setting.codec not in assign.CODECS:
        raise ValueError(f"unknown codec {setting.codec!r} for {spec.name}; this module builds {assign.CODECS}")
    d = setting.domain
    if d not in fit.ranges:
        raise ValueError(f"{spec.name} was not fitted in domain {d!r}; fitted: {sorted(fit.ranges)}")
    rng = fit.ranges[d]
    what = f"{spec.name} {setting.codec} {setting.k}"
    if setting.codec == "frame_dct":
        k = setting.k[0]
        grids = _grids(rng, rng[0].shape[0], k, setting.n_bins, what)
        return codecs.FrameDCT(spec.name, spec.dt, spec.values_shape, stats, k, d, grids,
                               signal_mean=spec.mean, signal_std=spec.std)
    if setting.codec == "band_dct":
        grids = _grids(rng, rng[0].shape[0], max(setting.k), setting.n_bins, what)
        return codecs.BandDCT(spec.name, spec.dt, spec.values_shape, stats, setting.k, grids)
    n_values, per_token = setting.k
    if d not in fit.pca:
        raise ValueError(f"{spec.name} has no basis in domain {d!r}; fitted: {sorted(fit.pca)}")
    mean, basis, eig, total = fit.pca[d]
    if per_token < 1 or n_values % per_token:
        raise ValueError(f"{what}: n_values must be a multiple of values_per_token")
    if n_values > rng[0].shape[1]:
        raise ValueError(f"{what} needs {n_values} fitted components; the fit has {rng[0].shape[1]}")
    lo, hi, width = (np.array(a[0, :n_values], dtype=np.float64).reshape(n_values // per_token, per_token)
                     for a in rng)
    return codecs.LinearCodec(spec.name, spec.dt, spec.values_shape, stats, mean, basis, eig, n_values,
                              per_token, d, _ridge(total, eig, n_values, math.prod(spec.values_shape),
                                                   fit.partly_valid),
                              GridSet(lo, hi, width, n_bins=setting.n_bins),
                              signal_mean=spec.mean, signal_std=spec.std)


def build_codecs(signals: Sequence[SignalSpec], stats: Mapping[str, ChannelStats],
                 fits: Mapping[str, SignalFit], assignment: Mapping[str, assign.Setting]) -> dict[str, codecs.Codec]:
    """`build_codec` for each of `signals`, with its stats, fit and setting looked up by name."""
    return {s.name: build_codec(s, stats[s.name], fits[s.name], assignment[s.name]) for s in signals}
