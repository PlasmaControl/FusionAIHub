"""Centred log10-power band-DCT codec and fitter for the two MAST Mirnov spectrograms."""

from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import multiprocessing
import os
import subprocess
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .. import cache, paths, registry, tasks, transform
from ..tokenizers import assign, codecs as tok_codecs
from ..tokenizers.grid import GridSet, fit_ranges
from . import framecache

DEFAULT_LOGSPEC_OUT = Path("/models/intermediate/mast/stage2r/logspec_fit_v1")
LOGSPEC_N_BINS: int = 64
LOGSPEC_MARGIN: float = 0.05
LOGSPEC_TARGET_SHOTS: int = 400
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)
_NPZ_FIELDS = ("floor", "log_mean", "lo", "hi", "width", "k")


class LogBandDCT(tok_codecs.BandDCT):
    """BandDCT in centred log10 power: ``u = log10(max(x, floor)) - log_mean``."""

    def __init__(
        self,
        name: str,
        dt: float,
        values_shape: tuple[int, int],
        stats: transform.ChannelStats,
        k: Sequence[int],
        floor: np.ndarray,
        log_mean: np.ndarray,
        grids: GridSet | None = None,
    ) -> None:
        super().__init__(name, dt, values_shape, stats, k, grids=grids)
        floor_arr = np.asarray(floor, dtype=np.float64)
        log_mean_arr = np.asarray(log_mean, dtype=np.float64)
        if floor_arr.shape != self.values_shape:
            raise ValueError(
                f"{name}: floor must have shape {self.values_shape}, got {floor_arr.shape}"
            )
        if log_mean_arr.shape != self.values_shape:
            raise ValueError(
                f"{name}: log_mean must have shape {self.values_shape}, got {log_mean_arr.shape}"
            )
        if not (np.isfinite(floor_arr).all() and (floor_arr > 0.0).all()):
            raise ValueError(f"{name}: floor must be finite and strictly positive")
        if not np.isfinite(log_mean_arr).all():
            raise ValueError(f"{name}: log_mean must be finite")
        self.floor = floor_arr
        self.log_mean = log_mean_arr

    def to_domain(self, x: np.ndarray) -> np.ndarray:
        """Physical power ``(..., P, n_bins, T)`` to centred log10 power (NaN kept)."""
        arr = np.asarray(x, dtype=np.float64)
        return np.log10(np.maximum(arr, self.floor[..., None])) - self.log_mean[..., None]

    def from_domain(self, u: np.ndarray) -> np.ndarray:
        """Centred log10 power ``(..., P, n_bins, T)`` to physical power (NaN kept)."""
        return np.power(10.0, np.asarray(u, dtype=np.float64) + self.log_mean[..., None])

    def domain_from_model(self, z: np.ndarray) -> np.ndarray:
        """Model units (the cache's) ``(..., P, n_bins, T)`` to centred log10 power."""
        phys = self._per_signal(lambda a: transform.to_physical(a, self.stats), z)
        return self.to_domain(phys)


def _limit_worker_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def _read_physical_worker(
    job: tuple[str, str, str, transform.ChannelStats, str | None],
) -> tuple[str, np.ndarray | None, float | None]:
    """Read one signal from cache v1 and convert to physical power ``(P, n_bins, T)``."""
    shot_str, shot_path_str, sig_name, st, stats_sha = job
    sd = cache.read_shot(Path(shot_path_str), [sig_name], stats_sha256=stats_sha)
    cs = sd.get(sig_name)
    if cs is None or cs.values.shape[-1] == 0:
        return shot_str, None, None
    phys = transform.to_physical(cs.values, st)
    if not np.isfinite(phys).any():
        return shot_str, None, None
    return shot_str, phys, float(cs.dt)


def _load_train_shots(splits_path: Path) -> list[str]:
    """Load the ``train`` split from ``splits.json``, sorted numerically."""
    path = Path(splits_path)
    if path.is_dir():
        path = path / "splits.json"
    if not path.exists():
        raise FileNotFoundError(f"splits.json not found at {path}")
    splits = json.loads(path.read_text())
    return sorted((str(s) for s in splits["train"]), key=lambda s: int(s))


def _estimate_stride(
    n_train: int,
    target_shots: int,
    manifest_v1: Mapping,
    sig_name: str,
) -> int:
    """Deterministic stride ``m`` so ~``target_shots`` shots with ``sig_name`` are sampled."""
    if target_shots <= 0 or n_train <= target_shots:
        return 1
    present_frac = 1.0
    sig_info = (manifest_v1.get("signals") or {}).get(sig_name)
    if isinstance(sig_info, Mapping):
        statuses = sig_info.get("status") or {}
        n_ok = int(statuses.get("ok", 0))
        n_tot = sum(int(v) for v in statuses.values())
        if n_ok > 0 and n_tot > 0:
            present_frac = n_ok / n_tot
    expected_present = n_train * present_frac
    return max(1, int(round(expected_present / float(target_shots))))


def _compute_floor_and_log_mean(
    arrays: Sequence[np.ndarray],
    values_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-(probe, bin) 1st-percentile positive floor and log10 mean."""
    p_count, n_bins = values_shape
    x_all = np.concatenate(arrays, axis=-1)  # (P, n_bins, T_total) float64
    floor = np.full((p_count, n_bins), np.nan, dtype=np.float64)
    for p in range(p_count):
        for b in range(n_bins):
            col = x_all[p, b]
            pos = col[np.isfinite(col) & (col > 0.0)]
            if pos.size > 0:
                floor[p, b] = float(np.percentile(pos, 1.0))
        missing_b = np.isnan(floor[p])
        if missing_b.any():
            valid_p = floor[p][~missing_b]
            if valid_p.size > 0:
                floor[p, missing_b] = float(np.median(valid_p))
    if np.isnan(floor).any():
        valid_all = floor[np.isfinite(floor)]
        fallback = float(np.median(valid_all)) if valid_all.size > 0 else 1e-30
        floor = np.where(np.isfinite(floor), floor, fallback)

    ok = np.isfinite(x_all)
    clipped = np.where(ok, np.maximum(x_all, floor[..., None]), 1.0)
    log_x = np.where(ok, np.log10(clipped), 0.0)
    cnt = ok.sum(axis=-1)
    log_mean = np.divide(
        log_x.sum(axis=-1),
        cnt,
        out=np.log10(floor).copy(),
        where=cnt > 0,
    )
    return floor, log_mean


def _utc_now_from_date() -> str:
    """UTC timestamp from ``date -u``."""
    return subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip()


def _save_logspec_fit(
    out_dir: Path,
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    meta_doc: Mapping,
) -> None:
    """Atomically write deterministic ``logspec_fit.npz`` and ``logspec_fit.json``."""
    out_dir = paths.prepare_write_root(Path(out_dir))
    flat_entries: dict[str, np.ndarray] = {}
    for sig_name, parts in arrays.items():
        for field in _NPZ_FIELDS:
            dtype = np.int64 if field == "k" else np.float64
            flat_entries[f"{sig_name}::{field}"] = np.asarray(parts[field], dtype=dtype)

    npz_tmp = out_dir / f"logspec_fit.npz.{os.getpid()}.tmp"
    try:
        with zipfile.ZipFile(npz_tmp, "w", zipfile.ZIP_STORED) as zf:
            for key in sorted(flat_entries):
                buf = io.BytesIO()
                np.lib.format.write_array(buf, flat_entries[key], allow_pickle=False)
                zi = zipfile.ZipInfo(f"{key}.npy", date_time=_FIXED_TIME)
                zi.compress_type = zipfile.ZIP_STORED
                zi.external_attr = 0o600 << 16
                zf.writestr(zi, buf.getvalue())
        os.replace(npz_tmp, out_dir / "logspec_fit.npz")
    finally:
        npz_tmp.unlink(missing_ok=True)

    framecache.save_json_atomic(dict(meta_doc), out_dir / "logspec_fit.json")


def fit_logspec(
    cache_root: Path,
    splits_path: Path,
    assignment_path: Path,
    out_dir: Path,
    *,
    signal_names: Sequence[str] = tasks.SPECTROGRAM_SIGNALS,
    target_shots: int = LOGSPEC_TARGET_SHOTS,
    n_bins: int = LOGSPEC_N_BINS,
    margin: float = LOGSPEC_MARGIN,
    workers: int = 32,
    signal_specs: Mapping[str, tasks.SignalSpec] | None = None,
) -> dict:
    """Fit per-bin floors, log10 means, and 64-bin grids for the Mirnov spectrograms."""
    cache_root = Path(cache_root)
    out_dir = paths.prepare_write_root(Path(out_dir))

    manifest_v1 = json.loads((cache_root / "manifest.json").read_text())
    stats_sha = str(manifest_v1["stats_sha256"])
    if transform.stats_sha256(cache_root / "stats.npz") != stats_sha:
        raise ValueError(f"{cache_root}/stats.npz sha256 disagrees with manifest.json")
    stats = transform.load_stats(cache_root / "stats.npz")
    assignment = assign.load_assignment(Path(assignment_path))
    train_shots = _load_train_shots(Path(splits_path))
    if not train_shots:
        raise ValueError(f"no training shots in {splits_path}")

    specs = (
        dict(signal_specs)
        if signal_specs is not None
        else {s.name: s for s in registry.all_signals()}
    )

    npz_payload: dict[str, dict[str, np.ndarray]] = {}
    sig_meta: dict[str, dict] = {}

    for sig_name in signal_names:
        if sig_name not in assignment:
            raise KeyError(f"{sig_name!r} not in assignment {assignment_path}")
        if sig_name not in stats:
            raise KeyError(f"{sig_name!r} not in stats {cache_root / 'stats.npz'}")

        setting = assignment[sig_name]
        k = [int(v) for v in setting.k]
        stride = _estimate_stride(len(train_shots), target_shots, manifest_v1, sig_name)
        candidate_shots = train_shots[::stride]

        jobs = [
            (s, str(cache_root / "shots" / f"{s}.h5"), sig_name, stats[sig_name], stats_sha)
            for s in candidate_shots
            if (cache_root / "shots" / f"{s}.h5").exists()
        ]
        if workers <= 1 or len(jobs) <= 1:
            results = [_read_physical_worker(j) for j in jobs]
        else:
            ctx = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                min(workers, len(jobs)),
                mp_context=ctx,
                initializer=_limit_worker_threads,
            ) as ex:
                results = list(
                    ex.map(
                        _read_physical_worker,
                        jobs,
                        chunksize=max(1, len(jobs) // (workers * 4)),
                    )
                )

        used_shots: list[str] = []
        phys_arrays: list[np.ndarray] = []
        dt_seen: float | None = None
        for shot_str, phys, dt_val in results:
            if phys is None:
                continue
            used_shots.append(shot_str)
            phys_arrays.append(phys)
            if dt_seen is None and dt_val is not None:
                dt_seen = dt_val

        if not phys_arrays:
            raise ValueError(f"{sig_name}: no finite samples found in {len(jobs)} candidate shots")

        spec = specs.get(sig_name)
        dt = float(spec.dt) if spec is not None else float(dt_seen)
        values_shape = tuple(int(d) for d in phys_arrays[0].shape[:-1])
        if len(values_shape) != 2:
            raise ValueError(f"{sig_name}: expected 2D values_shape (P, n_bins), got {values_shape}")

        floor, log_mean = _compute_floor_and_log_mean(phys_arrays, values_shape)
        bare_codec = LogBandDCT(
            sig_name, dt, values_shape, stats[sig_name], k, floor, log_mean, grids=None
        )

        frame_chunks = [
            bare_codec.encode_domain(bare_codec.to_domain(x[None]))[0] for x in phys_arrays
        ]
        all_frames = np.concatenate(frame_chunks, axis=0)  # (N_frames, n_tok, n_val)
        n_frames_total = int(all_frames.shape[0])
        n_frames_valid = int(np.isfinite(all_frames).any(axis=(1, 2)).sum())
        lo, hi, width = fit_ranges(all_frames, margin=float(margin))
        # Validate immediately by constructing the GridSet and codec.
        grids = GridSet(lo, hi, width, n_bins=int(n_bins))
        _ = LogBandDCT(
            sig_name, dt, values_shape, stats[sig_name], k, floor, log_mean, grids=grids
        )

        npz_payload[sig_name] = {
            "floor": floor,
            "log_mean": log_mean,
            "lo": lo,
            "hi": hi,
            "width": width,
            "k": np.asarray(k, dtype=np.int64),
        }
        sig_meta[sig_name] = {
            "shots": used_shots,
            "n_shots": len(used_shots),
            "n_candidates": len(jobs),
            "stride": int(stride),
            "n_frames": n_frames_total,
            "n_valid_frames": n_frames_valid,
            "dt": dt,
            "values_shape": list(values_shape),
            "k": k,
            "floor_min": float(np.min(floor)),
            "floor_median": float(np.median(floor)),
            "floor_max": float(np.max(floor)),
        }

    meta_doc = {
        "n_bins": int(n_bins),
        "margin": float(margin),
        "stats_sha256": stats_sha,
        "cache_v1_root": str(cache_root.resolve()),
        "splits_path": str(Path(splits_path).resolve()),
        "assignment_path": str(Path(assignment_path).resolve()),
        "created_utc": _utc_now_from_date(),
        "signals": sig_meta,
    }
    _save_logspec_fit(out_dir, npz_payload, meta_doc)
    meta_doc["logspec_fit_sha256"] = logspec_fit_sha256(out_dir)
    return meta_doc


def logspec_fit_sha256(out_dir: Path | str) -> str:
    """sha256 hex digest of ``<out_dir>/logspec_fit.npz``."""
    h = hashlib.sha256()
    with open(Path(out_dir) / "logspec_fit.npz", "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_logspec_codecs(
    fit_dir: Path | str,
    stats: Mapping[str, transform.ChannelStats],
) -> dict[str, LogBandDCT]:
    """Load ``<fit_dir>/logspec_fit.{json,npz}`` and construct ``{name: LogBandDCT}``."""
    fit_path = Path(fit_dir)
    meta = json.loads((fit_path / "logspec_fit.json").read_text())
    n_bins = int(meta["n_bins"])
    codecs: dict[str, LogBandDCT] = {}
    with np.load(fit_path / "logspec_fit.npz", allow_pickle=False) as z:
        for name, info in meta["signals"].items():
            if name not in stats:
                raise KeyError(f"signal {name!r} missing from stats")
            floor = np.asarray(z[f"{name}::floor"], dtype=np.float64)
            log_mean = np.asarray(z[f"{name}::log_mean"], dtype=np.float64)
            lo = np.asarray(z[f"{name}::lo"], dtype=np.float64)
            hi = np.asarray(z[f"{name}::hi"], dtype=np.float64)
            width = np.asarray(z[f"{name}::width"], dtype=np.float64)
            k = [int(v) for v in z[f"{name}::k"].tolist()]
            values_shape = (int(floor.shape[0]), int(floor.shape[1]))
            dt = float(info["dt"])
            grids = GridSet(lo, hi, width, n_bins=n_bins)
            codec = LogBandDCT(
                name, dt, values_shape, stats[name], k, floor, log_mean, grids=grids
            )
            assert codec.n_tok == 24 and codec.n_val == 32 and codec.grids.n_bins == 64, (
                f"{name}: unexpected LogBandDCT layout "
                f"(n_tok={codec.n_tok}, n_val={codec.n_val}, n_bins={codec.grids.n_bins}); "
                "expected (24, 32, 64)"
            )
            codecs[name] = codec
    return codecs


def resolve_manifest_logspec_fit(manifest: Mapping) -> Path | None:
    """Return the manifest's ``logspec_fit_path`` after verifying ``logspec_fit_sha256``, or ``None``."""
    raw_path = manifest.get("logspec_fit_path")
    expected_sha = manifest.get("logspec_fit_sha256")
    if raw_path is None:
        if expected_sha is not None:
            raise ValueError(
                f"manifest has logspec_fit_sha256={expected_sha!r} but logspec_fit_path is null"
            )
        return None
    fit_path = Path(str(raw_path))
    actual_sha = logspec_fit_sha256(fit_path)
    if expected_sha is not None and actual_sha != str(expected_sha):
        raise ValueError(
            f"logspec_fit_sha256 mismatch for {fit_path}: "
            f"manifest has {expected_sha}, disk has {actual_sha}"
        )
    return fit_path
