"""The standardized float16 cache of the 39 benchmark signals, one HDF5 file per shot.

The tokenizers of the next stage train from this cache and never touch TokaMark's loader: every
signal is already in model units (`transform.to_model`), in float16, with the validity of each
sample encoded as the NaN pattern. Scope of v1: the signals exactly as TokaMark's loader delivers
them (the two Mirnov signals are therefore its spectrograms, not 500 kHz waveforms), for the
shots of TokaMark's three splits.

File ``shots/<shot>.h5``: one contiguous, uncompressed float16 dataset per signal, named
``<source>-<signal>``, shape ``(*values_shape, T)``, attributes ``t0`` and ``dt`` (seconds) and
``dead_nan`` / ``dead_const`` (see below); a boolean ``<name>__saturated`` beside a signal that has
a saturation limit; file attributes ``shot``, ``split``, ``cache_version``, ``stats_sha256`` and
``status`` (JSON, the status of every signal asked for, so that a signal the shot lacks reads back
as None and not as a missing key). The file is written as ``<shot>.h5.tmp`` and renamed.

A sample is not valid (NaN) when the loader gave a non-finite value, when the value is non-finite
after the cast to float16 (overflow), or when it belongs to a channel that is dead in this shot:
all NaN (any channel) or all finite samples equal (a constant sensor). An *actuator* held at one
value (a beam at 0 W all shot) is information, so the constant rule skips channels whose registry
roles contain ``act``. Saturated samples (``|x| >= sat_abs``, physical units) stay valid and are
flagged: TokaMark scores them as targets, so a model that never saw them would miss every clipped
peak. (The spec says "not valid"; this is a deviation the project owner confirms.)

`write_shot` and `read_shot` need only numpy, pandas and h5py. `tokamark` and `zarr` are reached
through `transform.load_shot` and `registry.build_registry`, which `build` calls.
"""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import os
import re
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from . import registry as registry_mod
from . import transform
from .paths import prepare_write_root, tokamark_zarr_root
from .tasks import SignalSpec
from .transform import ChannelStats

CACHE_VERSION = "v1"
DT_TOLERANCE = 1e-6             # relative difference between the loader's time step and the spec's
STATS_SHOTS = 2000              # training shots the statistics are drawn from
REGISTRY_SAMPLE_STEP = 10       # the registry's validity sample is every tenth of those
MIN_FIT_FRACTION = 0.99         # share of the statistics shots that must load
FAR_FROM_ZERO = 3.0             # |mean| of an asinh channel above this: ten scales from zero and more
TOKAMARK_CHECKOUT = Path("/models/tokamind/tokamark")
_REPO = Path(__file__).resolve().parents[4]
_MESSAGE = 300                  # characters of a shot's error message kept in the manifest
_SAT_SUFFIX = "__saturated"

STATUSES = ("ok", "all_invalid", "absent", "dt_mismatch", "shape_mismatch")


@dataclass
class CachedSignal:
    values: np.ndarray                      # float16, (*values_shape, T), model units, NaN = not valid
    t0: float                               # seconds
    dt: float
    saturated: np.ndarray | None = None     # bool, same shape; only for signals with a saturation limit

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.values)


# ------------------------------------------------------------------------------------ writing

def _tables(registry: pd.DataFrame, signals: Sequence[SignalSpec]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per signal ``(actuator (C,) bool, sat_abs (C,) float)`` in channel order, from the registry."""
    out = {}
    groups = {k: g for k, g in registry.groupby("signal", sort=False)}
    for spec in signals:
        n = int(np.prod(spec.values_shape))
        g = groups.get(spec.name)
        if g is None or len(g) != n or sorted(g["channel"]) != list(range(n)):
            raise ValueError(f"the registry does not describe the {n} channels of {spec.name}")
        g = g.sort_values("channel")
        act = np.array(["act" in str(r).split("+") for r in g["roles"]], dtype=bool)
        out[spec.name] = (act, g["sat_abs"].to_numpy(dtype=np.float64))
    return out


def _signal_status(spec: SignalSpec, d: dict) -> tuple[str, np.ndarray | None, np.ndarray | None]:
    """The status of one signal in one shot, with its values (*values_shape, T) and time if usable."""
    time_, values = np.asarray(d["time"], dtype=np.float64), np.asarray(d["values"])
    if values.size == 0 or time_.size == 0:
        return "absent", None, None
    shape = tuple(spec.values_shape)
    if values.shape != (*shape, time_.size):
        return "shape_mismatch", None, None
    if time_.size < 2:
        return "dt_mismatch", None, None      # a time step cannot be checked on one sample
    dt = float(np.median(np.diff(time_)))
    if not abs(dt - spec.dt) <= DT_TOLERANCE * spec.dt:
        return "dt_mismatch", None, None
    return "ok", values.astype(np.float64), time_


def _convert(spec: SignalSpec, x: np.ndarray, st: ChannelStats, act: np.ndarray, sat: np.ndarray):
    """``(float16 values, saturated or None, dead_nan, dead_const)`` of one signal, shape (*values_shape, T)."""
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    finite = np.isfinite(flat)
    count = finite.sum(axis=1)
    hi = np.where(finite, flat, -np.inf).max(axis=1)
    lo = np.where(finite, flat, np.inf).min(axis=1)
    dead_nan = count == 0
    dead_const = (count > 0) & (hi == lo) & ~act
    z = transform.to_model(x, st).reshape(flat.shape)
    z[dead_nan | dead_const] = np.nan
    with np.errstate(over="ignore", invalid="ignore"):
        half = z.astype(np.float16)
    half[~np.isfinite(half)] = np.nan             # overflow (inf) and anything non-finite: not valid
    saturated = None
    if np.isfinite(sat).any():
        with np.errstate(invalid="ignore"):
            saturated = np.isfinite(half) & (np.abs(flat) >= sat[:, None])
        saturated = saturated.reshape(shape)
    return half.reshape(shape), saturated, int(dead_nan.sum()), int(dead_const.sum())


def _write_shot(path: Path, shot: int, split: str, raw: dict, signals: Sequence[SignalSpec],
                stats: dict[str, ChannelStats], tables: dict, stats_sha: str) -> dict[str, str]:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["shot"] = int(shot)
            f.attrs["split"] = str(split)
            f.attrs["cache_version"] = CACHE_VERSION
            f.attrs["stats_sha256"] = str(stats_sha)
            for spec in signals:
                d = raw.get(spec.name)
                if d is None:
                    status[spec.name] = "absent"
                    continue
                state, x, time_ = _signal_status(spec, d)
                if x is None:
                    status[spec.name] = state
                    continue
                act, sat = tables[spec.name]
                half, saturated, dead_nan, dead_const = _convert(spec, x, stats[spec.name], act, sat)
                ds = f.create_dataset(spec.name, data=half, dtype=np.float16)
                ds.attrs["t0"] = float(time_[0])
                ds.attrs["dt"] = float(spec.dt)
                ds.attrs["dead_nan"] = dead_nan
                ds.attrs["dead_const"] = dead_const
                if saturated is not None:
                    f.create_dataset(spec.name + _SAT_SUFFIX, data=saturated)
                status[spec.name] = "ok" if np.isfinite(half).any() else "all_invalid"
            f.attrs["status"] = json.dumps(status)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return status


def write_shot(path: Path, shot: int, split: str, raw: dict, signals: Sequence[SignalSpec],
               stats: dict[str, ChannelStats], registry: pd.DataFrame, *, stats_sha: str = "") -> dict[str, str]:
    """Write one shot's cache file and return ``{signal: status}``.

    ``raw`` is `transform.load_shot`'s output (physical units); ``registry`` supplies, per
    channel, the roles (``act`` exempts a channel from the constant-channel rule) and ``sat_abs``.
    Status: ``"ok"``; ``"all_invalid"`` (dataset written, all NaN); ``"absent"`` (no data, no
    dataset); ``"dt_mismatch"`` (the median time step differs from the spec's by more than 1e-6
    relative, or fewer than 2 samples: no dataset); ``"shape_mismatch"`` (the values are not
    ``(*values_shape, T)`` for the length of the time grid: no dataset). ``stats_sha`` is stored as
    a file attribute so that a build can tell which statistics made the file.
    """
    return _write_shot(path, shot, split, raw, signals, stats, _tables(registry, signals), stats_sha)


# ------------------------------------------------------------------------------------ reading

def _file_status(f: h5py.File) -> dict[str, str]:
    return json.loads(f.attrs["status"])


def _read_signal(f: h5py.File, name: str) -> CachedSignal | None:
    if name not in f:
        return None
    ds = f[name]
    sat = f[name + _SAT_SUFFIX][...] if name + _SAT_SUFFIX in f else None
    return CachedSignal(ds[...], float(ds.attrs["t0"]), float(ds.attrs["dt"]), sat)


def read_shot(path: Path, names: Sequence[str] | None = None) -> dict[str, CachedSignal | None]:
    """The signals of one cache file; None for a signal the shot does not have.

    ``names=None`` returns every signal the file was written for (absent and refused ones too, as
    None); a name the file knows nothing about is None as well.
    """
    with h5py.File(path, "r") as f:
        if names is None:
            names = list(_file_status(f))
        return {n: _read_signal(f, n) for n in names}


def summarize_shot(path: Path) -> dict:
    """``{"status": {signal: status}, "dead": {signal: (dead_nan, dead_const)}}`` from the file's attributes."""
    with h5py.File(path, "r") as f:
        status = _file_status(f)
        dead = {n: (int(f[n].attrs["dead_nan"]), int(f[n].attrs["dead_const"])) for n in status if n in f}
    return {"status": status, "dead": dead}


def _is_current(path: Path, stats_sha: str) -> bool:
    """The file exists, is readable, and was made by this cache version from these statistics."""
    try:
        with h5py.File(path, "r") as f:
            return (f.attrs["cache_version"] == CACHE_VERSION and f.attrs["stats_sha256"] == stats_sha
                    and "status" in f.attrs)
    except Exception:       # missing, half-written by an older crash, not HDF5: write it again
        return False


# ------------------------------------------------------------------------------------ the build

def pick_evenly(shots: Sequence[int], n: int) -> list[int]:
    """``n`` shots evenly spaced over the sorted list (a trial that spans the campaigns); all if fewer."""
    if n < 1:
        raise ValueError("n must be at least 1")
    shots = sorted(int(s) for s in shots)
    if n >= len(shots):
        return shots
    idx = np.unique(np.round(np.linspace(0, len(shots) - 1, n)).astype(int))
    return [shots[i] for i in idx]


def _read_git_head(checkout: Path) -> str:
    """The commit a checkout is at, read as text (no ``git`` run inside downloaded material); "unknown" if not found."""
    try:
        git = Path(checkout) / ".git"
        head = (git / "HEAD").read_text().strip()
        if re.fullmatch(r"[0-9a-f]{40}", head):
            return head
        ref = head.removeprefix("ref: ").strip()
        ref_file = git / ref
        if ref_file.is_file():
            return ref_file.read_text().strip()
        for line in (git / "packed-refs").read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref and re.fullmatch(r"[0-9a-f]{40}", parts[0]):
                return parts[0]
    except OSError:
        pass
    return "unknown"


def _tokamark_revision() -> str:
    return _read_git_head(TOKAMARK_CHECKOUT)


def _repo_revision() -> str:
    try:
        out = subprocess.run(["git", "-C", str(_REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=30, check=True)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _stats_shots(pool: Sequence[int]) -> list[int]:
    """The statistics shots: 2,000 of the pool without replacement (all of them if fewer), sorted."""
    pool = np.array(sorted({int(s) for s in pool}))
    if len(pool) <= STATS_SHOTS:
        return [int(s) for s in pool]
    rng = np.random.default_rng(0)
    return sorted(int(s) for s in rng.choice(pool, STATS_SHOTS, replace=False))


def _write_one(job: tuple, state: tuple):
    """Load and write (or skip) one shot. Returns ``(shot, split, summary, wrote, error)``."""
    shot, split = job
    signals, stats, tables, sha, shots_dir = state
    path = shots_dir / f"{shot}.h5"
    if _is_current(path, sha):
        return shot, split, summarize_shot(path), False, None
    try:
        raw = transform.load_shot(shot, signals)
        _write_shot(path, shot, split, raw, signals, stats, tables, sha)
        return shot, split, summarize_shot(path), True, None
    except ImportError:         # a broken environment (no tokamark), not a bad shot
        raise
    except Exception as e:
        path.unlink(missing_ok=True)
        return shot, split, None, False, f"{type(e).__name__}: {str(e)[:_MESSAGE]}"


_STATE: tuple | None = None


def _init_worker(state: tuple) -> None:
    global _STATE
    _STATE = state


def _pool_job(job: tuple):
    return _write_one(job, _STATE)


def _run_jobs(jobs: list, state: tuple, workers: int):
    """Yield `_write_one`'s result for each job: in process for one worker, else a spawn pool.

    Spawn, because zarr starts an event-loop thread and a forked child can deadlock; a
    `ProcessPoolExecutor` because a worker that dies raises (`BrokenProcessPool`) instead of hanging.
    """
    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            yield _write_one(job, state)
        return
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(min(workers, len(jobs)), mp_context=ctx,
                                                initializer=_init_worker, initargs=(state,)) as ex:
        try:
            futures = [ex.submit(_pool_job, job) for job in jobs]
            for fut in concurrent.futures.as_completed(futures):
                yield fut.result()
        finally:
            for fut in futures:
                fut.cancel()


def _statistics(root: Path, shots: dict, stats_pool: Sequence[int] | None, workers: int, force: bool,
                old_manifest: dict | None) -> tuple[dict, Path, str, dict | None, list[int] | None, bool]:
    """Load ``stats.npz`` or fit it. Returns ``(stats, path, sha256, fit summary, stats shots, fitted)``."""
    path = root / "stats.npz"
    pool = stats_pool if stats_pool is not None else shots.get("train")
    stats_shots = _stats_shots(pool) if pool else None
    if path.exists():
        stats, fit_summary, fitted = transform.load_stats(path), None, False
        sha = transform.stats_sha256(path)
    else:
        if not stats_shots:
            raise ValueError("no statistics at stats.npz and no training shots to fit them on "
                             "(pass shots['train'] or stats_pool)")
        stats, fit_summary = transform.fit_stats_with_summary(stats_shots, registry_mod.all_signals(),
                                                              workers=workers)
        if fit_summary["n_loaded"] < MIN_FIT_FRACTION * fit_summary["n_shots"]:
            raise RuntimeError(
                f"the statistics fit loaded {fit_summary['n_loaded']} of {fit_summary['n_shots']} shots "
                f"(under {MIN_FIT_FRACTION:.0%}); stats.npz not written. First failures: "
                f"{dict(list(fit_summary['failed'].items())[:3])}")
        part = root / "stats.fit.npz"
        transform.save_stats(stats, part)
        sha = transform.stats_sha256(part)
        fitted = True
    if old_manifest is not None and old_manifest.get("stats_sha256") != sha and not force:
        if fitted:
            part.unlink(missing_ok=True)
        raise ValueError(
            f"{root} was built with different statistics (manifest {old_manifest.get('stats_sha256')}, "
            f"now {sha}); the existing shot files would not match. Use a new root or force=True.")
    if fitted:
        os.replace(part, path)
    return stats, path, sha, fit_summary, stats_shots, fitted


def build(root: Path, shots: dict[str, list[int]], *, workers: int, force: bool = False,
          stats_pool: Sequence[int] | None = None) -> dict:
    """Build (or resume) the cache under ``root`` and return the manifest (also ``manifest.json``).

    Order: the statistics (``stats.npz`` is loaded if present, else fitted on 2,000 shots drawn
    with ``default_rng(0)`` from ``stats_pool``, default ``shots["train"]``; with fewer, all of
    them), the registry (``registry.csv``, validity measured on every tenth of those shots),
    then ``shots/<shot>.h5`` for every shot of ``shots`` ({split: [shot, ...]}) with ``workers``
    processes. A shot whose file exists and was made from the same statistics is skipped; a shot
    whose write raises is listed in the manifest's ``errors`` and leaves no file; the others go
    on. The manifest is the summary of the files of the requested shots, written or skipped.

    Raises ``ValueError`` when ``manifest.json`` was made from other statistics (unless
    ``force``), and when there are no statistics and no training shots; ``RuntimeError`` when a
    fresh fit loaded less than 99% of its shots (nothing is written then).
    """
    start = time.time()
    root = prepare_write_root(Path(root))
    manifest_path = root / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    signals = registry_mod.all_signals()

    stats, stats_path, sha, fit_summary, stats_shots, fitted = _statistics(
        root, shots, stats_pool, workers, force, old)
    if fit_summary is None and old is not None and old.get("stats_sha256") == sha:
        fit_summary = old.get("stats_fit")

    reg_path = root / "registry.csv"
    if reg_path.exists() and not fitted:
        reg = registry_mod.load_registry(reg_path)
    else:
        if not stats_shots:
            raise ValueError("no registry.csv and no training shots to measure it on")
        reg = registry_mod.build_registry(tokamark_zarr_root(), stats_shots[::REGISTRY_SAMPLE_STEP],
                                          workers=workers)
        reg = registry_mod.attach_stats(reg, stats)
        registry_mod.save_registry(reg, reg_path)

    shots_dir = root / "shots"
    shots_dir.mkdir(exist_ok=True)
    state = (signals, stats, _tables(reg, signals), sha, shots_dir)
    jobs = [(int(s), split) for split, lst in shots.items() for s in sorted({int(x) for x in lst})]
    counts = {split: {"requested": len({int(x) for x in lst}), "files": 0, "written": 0, "skipped": 0, "failed": 0}
              for split, lst in shots.items()}
    sig = {s.name: {"status": dict.fromkeys(STATUSES, 0), "dead_nan": 0, "dead_const": 0} for s in signals}
    errors: dict[str, str] = {}
    for i, (shot, split, summary, wrote, err) in enumerate(_run_jobs(jobs, state, workers), 1):
        if err is not None:
            errors[str(shot)] = err
            counts[split]["failed"] += 1
        else:
            counts[split]["files"] += 1
            counts[split]["written" if wrote else "skipped"] += 1
            for name, status in summary["status"].items():
                sig[name]["status"][status] += 1
            for name, (n_nan, n_const) in summary["dead"].items():
                sig[name]["dead_nan"] += n_nan
                sig[name]["dead_const"] += n_const
        if i % 500 == 0 or i == len(jobs):
            print(f"[ignite.mast] cache: {i} of {len(jobs)} shots done, {len(errors)} errors, "
                  f"{time.time() - start:.0f} s", flush=True)
    for v in sig.values():
        v["status"] = {k: n for k, n in v["status"].items() if n}

    asinh = [n for n in sorted(stats) if stats[n].compress == "asinh"]
    manifest = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cache_version": CACHE_VERSION,
        "repo_revision": _repo_revision(),
        "tokamark_revision": _tokamark_revision(),
        "stats_sha256": sha,
        "stats_fit": fit_summary,
        "asinh_signals": asinh,
        "asinh_far_from_zero": {n: int((np.abs(np.asarray(stats[n].mean)) > FAR_FROM_ZERO).sum()) for n in asinh},
        "counts": counts,
        "signals": sig,
        "errors": errors,
        "wall_seconds": round(time.time() - start, 1),
        "size_bytes": sum(p.stat().st_size for p in shots_dir.glob("*.h5")),
    }
    tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=False))
    os.replace(tmp, manifest_path)
    return manifest

