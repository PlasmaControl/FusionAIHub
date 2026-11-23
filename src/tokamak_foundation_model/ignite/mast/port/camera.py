"""MAST 2D visible camera (``rgb``) preprocessing, timing, codec bridge, and encoding."""

from __future__ import annotations

import bisect
import contextlib
import hashlib
import io
import itertools
import json
import math
import os
import subprocess
import warnings
import zipfile
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import zarr

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r".*torch\.jit\.script.*is deprecated.*",
        category=FutureWarning,
    )
    from ... import eval_dynamics as ev
    from ... import train_codec as tc
    from ... import train_dynamics as td

from ...config import VideoCodecConfig
from .. import cache, paths, transform
from ..tokenizers.grid import GridSet, fit_ranges
from . import framecache

CAMERA_MODALITY: str = "camera-rgb"
CAMERA_STATS_MODALITY: str = "camera-rgb-stats"
CAMERA_MODALITIES: tuple[str, str] = (CAMERA_MODALITY, CAMERA_STATS_MODALITY)

CAMERA_RAW_HEIGHT: int = 480
CAMERA_RAW_WIDTH: int = 640
CAMERA_POOL: int = 4
CAMERA_HEIGHT: int = CAMERA_RAW_HEIGHT // CAMERA_POOL  # 120
CAMERA_WIDTH: int = CAMERA_RAW_WIDTH // CAMERA_POOL    # 160
CAMERA_SATURATION_COUNT: int = 4095

CAMERA_CODEBOOK_SIZE: int = 1000
CAMERA_MISSING_CODE: int = 1000
CAMERA_STATS_N_BINS: int = 64
CAMERA_STATS_MISSING_CODE: int = 64
CAMERA_STATS_MARGIN: float = 0.05
CAMERA_MIN_OVERLAP_S: float = 0.0025

CAMERA_DALPHA_SIGNAL: str = "spectrometer_visible-filter_spectrometer_dalpha_voltage"

DEFAULT_LEVEL1_ROOT: Path = Path("/models/mast/fairmast/level1")
DEFAULT_CAMERA_ROOT: Path = Path("/models/intermediate/mast/camera")
DEFAULT_FRAMES_DIR: Path = DEFAULT_CAMERA_ROOT / "frames_rgb_v1"
DEFAULT_CAMERA_FIT_DIR: Path = DEFAULT_CAMERA_ROOT / "camera_fit_v1"
DEFAULT_SPLITS_PATH: Path = Path(
    "/models/intermediate/mast/stage2r/frame_codes_v1b_logspec/splits.json"
)

_FIXED_TIME = (1980, 1, 1, 0, 0, 0)


# ----------------------------------------------------------------------- codec config & hashing

def default_video_codec_config(
    *,
    height: int = CAMERA_HEIGHT,
    width: int = CAMERA_WIDTH,
    patch_h: int = 20,
    patch_w: int = 20,
    d_model: int = 256,
    enc_depth: int = 6,
    dec_depth: int = 6,
    heads: int = 8,
) -> VideoCodecConfig:
    """MAST ``rgb`` :class:`VideoCodecConfig`: 1×1×120×160, 20×20 patches -> 48 tokens.

    ``divertor`` stays at its default ``"lower"`` because ``VideoCodecConfig.__post_init__``
    asserts ``divertor in ("lower", "upper")``.
    """
    return VideoCodecConfig(
        channels=1,
        frames=1,
        height=int(height),
        width=int(width),
        patch_t=1,
        patch_h=int(patch_h),
        patch_w=int(patch_w),
        d_model=int(d_model),
        enc_depth=int(enc_depth),
        dec_depth=int(dec_depth),
        heads=int(heads),
    )


def file_sha256(path: Path | str) -> str:
    """sha256 hex digest of a file."""
    h = hashlib.sha256()
    with open(Path(path), "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _utc_now_from_date() -> str:
    return subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip()


# ----------------------------------------------------------------------- zarr completeness & reading

def _read_json_dict(path: Path) -> dict[str, Any] | None:
    """Read a non-empty JSON object file, returning ``None`` if absent, empty, or invalid."""
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            return None
        doc = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return None
    return doc if isinstance(doc, dict) else None


def _check_zarray_chunks(arr_dir: Path) -> tuple[bool, list[int] | None]:
    """Verify ``.zarray`` exists and every expected chunk file exists with non-zero size."""
    meta = _read_json_dict(arr_dir / ".zarray")
    if meta is None:
        return False, None
    try:
        shape = [int(v) for v in meta["shape"]]
        chunks = [int(v) for v in meta["chunks"]]
        sep = str(meta.get("dimension_separator") or ".")
    except (ValueError, KeyError, TypeError):
        return False, None
    if not shape or len(shape) != len(chunks):
        return False, None
    if any(s <= 0 for s in shape) or any(c <= 0 for c in chunks):
        return False, None
    counts = [math.ceil(s / c) for s, c in zip(shape, chunks)]
    for idx in itertools.product(*(range(n) for n in counts)):
        rel = sep.join(str(i) for i in idx)
        chunk_file = arr_dir / rel
        try:
            if not chunk_file.is_file() or chunk_file.stat().st_size <= 0:
                return False, None
        except OSError:
            return False, None
    return True, shape


def check_rgb_zarr_complete(shot_zarr_dir: Path | str) -> tuple[bool, str]:
    """Check whether ``<shot>.zarr/rgb`` is completely downloaded before reading.

    Requires non-empty ``rgb/.zgroup`` and ``rgb/.zattrs``, complete ``data`` and ``time``
    arrays with matching frame counts, and a valid positive ``exposure`` attribute in
    ``rgb/.zattrs``. Returns ``(True, "ok")``, ``(False, "no_rgb")``,
    ``(False, "incomplete")``, or ``(False, "no_exposure_attr")``.
    """
    shot_dir = Path(shot_zarr_dir)
    rgb_dir = shot_dir / "rgb"
    if not shot_dir.is_dir() or not rgb_dir.is_dir():
        return False, "no_rgb"
    if _read_json_dict(rgb_dir / ".zgroup") is None:
        return False, "incomplete"
    zattrs = _read_json_dict(rgb_dir / ".zattrs")
    if zattrs is None:
        return False, "incomplete"
    data_ok, data_shape = _check_zarray_chunks(rgb_dir / "data")
    if not data_ok or data_shape is None or len(data_shape) != 3:
        return False, "incomplete"
    time_ok, time_shape = _check_zarray_chunks(rgb_dir / "time")
    if not time_ok or time_shape is None or len(time_shape) != 1:
        return False, "incomplete"
    if time_shape[0] != data_shape[0]:
        return False, "incomplete"
    try:
        extract_exposure_s(zattrs)
    except ValueError:
        return False, "no_exposure_attr"
    return True, "ok"


def extract_exposure_s(attrs: Mapping[str, Any], time: np.ndarray | None = None) -> float:
    """Exposure duration in seconds from zarr ``exposure`` attr (microseconds on FAIR-MAST).

    Raises :class:`ValueError` if ``exposure`` is absent, non-finite, or non-positive.
    """
    del time  # kept optional for signature compatibility; never used as a fallback
    raw = attrs.get("exposure")
    if raw is not None:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            val = float("nan")
        if math.isfinite(val) and val > 0.0:
            # FAIR-MAST IPX headers store exposure in microseconds (e.g. 9090.0 us).
            return val * 1e-6 if val > 1.0 else val
    raise ValueError(f"missing or invalid 'exposure' attribute in rgb/.zattrs: {raw!r}")


def read_rgb_zarr(
    shot_zarr_dir: Path | str,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    """Read-only zarr load of ``<shot>.zarr/rgb`` after verifying chunk completeness."""
    ok, status = check_rgb_zarr_complete(shot_zarr_dir)
    if not ok:
        raise RuntimeError(f"rgb zarr not complete ({status}): {shot_zarr_dir}")
    rgb_dir = Path(shot_zarr_dir) / "rgb"
    grp = zarr.open_group(str(rgb_dir), mode="r")
    data = np.asarray(grp["data"][:])
    time = np.asarray(grp["time"][:], dtype=np.float64)
    if data.ndim != 3 or time.ndim != 1 or len(time) != data.shape[0]:
        raise RuntimeError(
            f"rgb shape mismatch in {rgb_dir}: data={data.shape}, time={time.shape}"
        )
    attrs = dict(grp.attrs)
    exposure_s = extract_exposure_s(attrs)
    return data, time, exposure_s, attrs


# ----------------------------------------------------------------------- preprocessing

def downsample_frames(counts: np.ndarray, pool: int = CAMERA_POOL) -> np.ndarray:
    """Mean-pool ``(N, H, W)`` counts by ``pool x pool`` in float32."""
    arr = np.asarray(counts, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected (N, H, W) counts, got {arr.shape}")
    n, h, w = arr.shape
    if h % pool != 0 or w % pool != 0:
        raise ValueError(f"frame shape {(h, w)} not divisible by pool={pool}")
    return arr.reshape(n, h // pool, pool, w // pool, pool).mean(
        axis=(2, 4), dtype=np.float32
    )


def preprocess_frames(
    counts: np.ndarray,
    *,
    pool: int = CAMERA_POOL,
    saturation_count: int = CAMERA_SATURATION_COUNT,
) -> dict[str, Any]:
    """Downsample raw counts to 120x160, apply ``u = log1p(counts)``, and compute per-frame ``m, s``."""
    raw = np.asarray(counts)
    if raw.ndim != 3:
        raise ValueError(f"expected (N, H, W) counts, got {raw.shape}")
    sat_frac = float(np.mean(raw >= saturation_count)) if raw.size > 0 else 0.0
    raw_mean_counts = raw.mean(axis=(1, 2), dtype=np.float64).astype(np.float32)
    pooled = downsample_frames(raw, pool=pool)
    u_f32 = np.log1p(np.maximum(pooled, np.float32(0.0))).astype(np.float32)
    m = u_f32.mean(axis=(1, 2), dtype=np.float64).astype(np.float32)
    s = u_f32.std(axis=(1, 2), ddof=1, dtype=np.float64).astype(np.float32)
    return {
        "u": u_f32.astype(np.float16),
        "u_f32": u_f32,
        "m": m,
        "s": s,
        "raw_mean_counts": raw_mean_counts,
        "saturation_frac": sat_frac,
    }


def normalize_u_frames(
    u: np.ndarray,
    m: np.ndarray | float | None,
    s: np.ndarray | float | None,
    s_floor: float,
) -> torch.Tensor:
    """Standardize ``u`` -> ``z = (u - m) / max(s, s_floor)`` shaped ``(N, 1, 1, H, W)`` or ``(1, 1, H, W)``.

    When ``u`` is float16 (from the frames cache) or ``m``/``s`` are ``None``, computes
    ``m`` and unbiased ``s`` directly on ``u.astype(float32)`` so ``z`` always has exact
    zero mean and ``std <= 1`` (making ``VideoCodec.standardize_input(z)`` the identity).
    """
    u_arr = np.asarray(u)
    u_f32 = u_arr.astype(np.float32, copy=False)
    sf = np.float32(s_floor)
    recompute = (m is None) or (s is None) or (u_arr.dtype == np.float16)
    if u_f32.ndim == 2:
        if recompute:
            m_val = np.float32(u_f32.mean(dtype=np.float64))
            s_val = np.float32(u_f32.std(ddof=1, dtype=np.float64))
        else:
            m_val = np.float32(m)
            s_val = np.float32(s)
        z = (u_f32 - m_val) / max(s_val, sf)
        return torch.from_numpy(z[None, None, :, :].copy())
    if u_f32.ndim == 3:
        if recompute:
            m_arr = u_f32.mean(axis=(1, 2), dtype=np.float64).astype(np.float32)[:, None, None]
            s_raw = u_f32.std(axis=(1, 2), ddof=1, dtype=np.float64).astype(np.float32)
        else:
            m_arr = np.asarray(m, dtype=np.float32)[:, None, None]
            s_raw = np.asarray(s, dtype=np.float32)
        s_arr = np.maximum(s_raw, sf)[:, None, None]
        z = (u_f32 - m_arr) / s_arr
        return torch.from_numpy(z[:, None, None, :, :].copy())
    raise ValueError(f"expected (H, W) or (N, H, W) u array, got {u_f32.shape}")


def save_frames_npz_atomic(
    path: Path,
    *,
    u: np.ndarray,
    time: np.ndarray,
    m: np.ndarray,
    s: np.ndarray,
    raw_mean_counts: np.ndarray,
    exposure_s: float,
) -> None:
    """Atomically write one shot's preprocessed camera frames ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez(
            tmp,
            u=np.asarray(u, dtype=np.float16),
            time=np.asarray(time, dtype=np.float64),
            m=np.asarray(m, dtype=np.float32),
            s=np.asarray(s, dtype=np.float32),
            raw_mean_counts=np.asarray(raw_mean_counts, dtype=np.float32),
            exposure_s=np.asarray(float(exposure_s), dtype=np.float64),
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_frames_npz(path: Path | str, *, load_u: bool = True) -> dict[str, Any]:
    """Load one shot's cached camera ``.npz``."""
    with np.load(Path(path), allow_pickle=False) as z:
        out: dict[str, Any] = {
            "time": np.asarray(z["time"], dtype=np.float64),
            "m": np.asarray(z["m"], dtype=np.float32),
            "s": np.asarray(z["s"], dtype=np.float32),
            "raw_mean_counts": np.asarray(z["raw_mean_counts"], dtype=np.float32),
            "exposure_s": float(np.asarray(z["exposure_s"]).reshape(())),
        }
        if load_u:
            out["u"] = np.asarray(z["u"], dtype=np.float16)
    return out


# ----------------------------------------------------------------------- timing & exposure mapping

def exposure_intervals(
    times: np.ndarray,
    exposure_s: float,
    convention: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lo, hi)`` exposure intervals in seconds for ``start``, ``centre``, or ``end``."""
    t = np.asarray(times, dtype=np.float64)
    e = float(exposure_s)
    if convention == "start":
        return t, t + e
    if convention == "centre":
        return t - 0.5 * e, t + 0.5 * e
    if convention == "end":
        return t - e, t
    raise ValueError(
        f"unknown time convention {convention!r}; expected 'start', 'centre', or 'end'"
    )


def valid_exposure_indices(times: np.ndarray) -> np.ndarray:
    """Indices of the longest strictly increasing subsequence of finite timestamps.

    Non-finite timestamps, duplicates, and glitched jumps are excluded; duplicate ties
    keep the earliest index.
    """
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    n = t.size
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if np.isfinite(t).all() and (n == 1 or bool((np.diff(t) > 0.0).all())):
        return np.arange(n, dtype=np.int64)

    tails_val: list[float] = []
    tails_idx: list[int] = []
    prev = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        v = float(t[i])
        if not math.isfinite(v):
            continue
        pos = bisect.bisect_left(tails_val, v)
        if pos == len(tails_val):
            tails_val.append(v)
            tails_idx.append(i)
            if pos > 0:
                prev[i] = tails_idx[pos - 1]
        elif v < tails_val[pos]:
            tails_val[pos] = v
            tails_idx[pos] = i
            if pos > 0:
                prev[i] = tails_idx[pos - 1]
    if not tails_idx:
        return np.empty(0, dtype=np.int64)
    out: list[int] = []
    cur = tails_idx[-1]
    while cur != -1:
        out.append(cur)
        cur = int(prev[cur])
    out.reverse()
    return np.asarray(out, dtype=np.int64)


def map_frames_to_exposures(
    times: np.ndarray,
    exposure_s: float,
    t0: float,
    n_frames: int,
    *,
    dt_frame: float = framecache.DT_FRAME,
    convention: str = "centre",
    min_overlap_s: float = CAMERA_MIN_OVERLAP_S,
) -> np.ndarray:
    """Map each 5 ms frame ``f`` in ``[0, n_frames)`` to its best-overlapping valid exposure, or ``-1``."""
    n_f = int(n_frames)
    out = np.full(n_f, -1, dtype=np.int64)
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    exp_s = float(exposure_s)
    t0_f = float(t0)
    dt_f = float(dt_frame)
    if (
        n_f <= 0
        or t.size == 0
        or not (math.isfinite(exp_s) and exp_s > 0.0)
        or not (math.isfinite(t0_f) and math.isfinite(dt_f) and dt_f > 0.0)
    ):
        return out
    valid_idx = valid_exposure_indices(t)
    if valid_idx.size == 0:
        return out
    exp_lo, exp_hi = exposure_intervals(t[valid_idx], exp_s, convention)
    f_lo = t0_f + np.arange(n_f, dtype=np.float64) * dt_f
    f_hi = f_lo + dt_f
    ov = np.minimum(f_hi[:, None], exp_hi[None, :]) - np.maximum(f_lo[:, None], exp_lo[None, :])
    ov = np.where(np.isfinite(ov), ov, -np.inf)
    best_pos = np.argmax(ov, axis=1)
    best_ov = ov[np.arange(n_f), best_pos]
    ok = np.isfinite(best_ov) & (best_ov >= float(min_overlap_s) - 1e-12)
    out[ok] = valid_idx[best_pos[ok]]
    return out


def _pearson_r_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r of first differences over mutually finite entries."""
    da = np.diff(np.asarray(a, dtype=np.float64))
    db = np.diff(np.asarray(b, dtype=np.float64))
    ok = np.isfinite(da) & np.isfinite(db)
    if int(ok.sum()) < 3:
        return float("nan")
    x = da[ok]
    y = db[ok]
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def _window_mean_1d(
    sig_t: np.ndarray, sig_y: np.ndarray, win_lo: np.ndarray, win_hi: np.ndarray
) -> np.ndarray:
    """Mean of finite ``sig_y`` samples whose centre time ``sig_t`` lies in ``[lo, hi]``."""
    lo_idx = np.searchsorted(sig_t, win_lo, side="left")
    hi_idx = np.searchsorted(sig_t, win_hi, side="right")
    out = np.full(len(win_lo), np.nan, dtype=np.float64)
    for k, (i0, i1) in enumerate(zip(lo_idx, hi_idx)):
        if i1 <= i0:
            continue
        seg = sig_y[i0:i1]
        fin = seg[np.isfinite(seg)]
        if fin.size > 0:
            out[k] = float(fin.mean())
    return out


def measure_timing_convention(
    frames_dir: Path | str,
    cache_v1_root: Path | str,
    frame_cache_dir: Path | str,
    *,
    n_shots: int = 30,
    dalpha_signal: str = CAMERA_DALPHA_SIGNAL,
    min_plasma_exposures: int = 10,
) -> dict[str, Any]:
    """Measure the rgb timestamp convention (start / centre / end) against physical D-alpha."""
    frames_path = Path(frames_dir)
    v1_root = Path(cache_v1_root)
    fc_dir = Path(frame_cache_dir)

    index_doc = json.loads((frames_path / "index.json").read_text())
    shots_index = index_doc["shots"]
    splits = json.loads((fc_dir / "splits.json").read_text())
    dt_frame = float(
        json.loads((fc_dir / "_codec_manifest.json").read_text()).get(
            "dt_frame", framecache.DT_FRAME
        )
    )

    manifest_v1 = json.loads((v1_root / "manifest.json").read_text())
    stats_sha = str(manifest_v1["stats_sha256"])
    stats = transform.load_stats(v1_root / "stats.npz")
    da_stats = stats[dalpha_signal]

    train_ok = sorted(
        (
            str(s)
            for s in splits["train"]
            if (shots_index.get(str(s)) or {}).get("status") == "ok"
            and (frames_path / f"{s}.npz").exists()
            and (fc_dir / f"{s}.pt").exists()
            and (v1_root / "shots" / f"{s}.h5").exists()
        ),
        key=lambda s: int(s),
    )

    # Deterministic stride across sorted eligible train shots, scanning until n_shots qualify.
    step = max(1, len(train_ok) // max(1, n_shots * 2))
    ordered_candidates = train_ok[::step] + [
        s for i, s in enumerate(train_ok) if i % step != 0
    ]

    conventions = ("start", "centre", "end")
    per_shot: dict[str, dict[str, float]] = {}
    used_shots: list[str] = []

    for shot in ordered_candidates:
        if len(used_shots) >= int(n_shots):
            break
        entry = torch.load(fc_dir / f"{shot}.pt", map_location="cpu", weights_only=False)
        t0 = float(entry["t0"])
        n_frames = int(entry["n_frames"])
        t_lo = t0
        t_hi = t0 + n_frames * dt_frame

        fr = load_frames_npz(frames_path / f"{shot}.npz", load_u=False)
        valid_idx = valid_exposure_indices(fr["time"])
        cam_t = fr["time"][valid_idx]
        cam_mean = fr["raw_mean_counts"].astype(np.float64)[valid_idx]
        exp_s = float(fr["exposure_s"])

        in_plasma = (cam_t >= t_lo) & (cam_t <= t_hi)
        if int(in_plasma.sum()) < int(min_plasma_exposures):
            continue

        sd = cache.read_shot(v1_root / "shots" / f"{shot}.h5", [dalpha_signal], stats_sha256=stats_sha)
        cs_da = sd.get(dalpha_signal)
        if cs_da is None or cs_da.values.shape[-1] == 0:
            continue
        da_phys = transform.to_physical(cs_da.values, da_stats)
        da_2d = da_phys.reshape(-1, da_phys.shape[-1])
        ok_da = np.isfinite(da_2d)
        cnt_da = ok_da.sum(axis=0)
        da_1d = np.divide(
            np.where(ok_da, da_2d, 0.0).sum(axis=0),
            cnt_da,
            out=np.full(da_2d.shape[1], np.nan, dtype=np.float64),
            where=cnt_da > 0,
        )
        if not np.isfinite(da_1d).any():
            continue
        t_da = framecache.sample_times(da_1d.shape[0], cs_da.t0, cs_da.dt)

        t_sub = cam_t[in_plasma]
        c_sub = cam_mean[in_plasma]
        shot_rs: dict[str, float] = {}
        for conv in conventions:
            w_lo, w_hi = exposure_intervals(t_sub, exp_s, conv)
            da_win = _window_mean_1d(t_da, da_1d, w_lo, w_hi)
            shot_rs[conv] = _pearson_r_diff(c_sub, da_win)
        if all(math.isfinite(shot_rs[c]) for c in conventions):
            per_shot[shot] = shot_rs
            used_shots.append(shot)

    if not used_shots:
        raise ValueError("measure_timing_convention: no eligible training shots with rgb and D-alpha")

    medians = {
        conv: float(np.median([per_shot[s][conv] for s in used_shots]))
        for conv in conventions
    }
    best_conv = max(conventions, key=lambda c: medians[c])
    chosen = best_conv if (medians[best_conv] - medians["centre"] >= 0.02) else "centre"
    return {
        "medians": medians,
        "best": best_conv,
        "chosen": chosen,
        "margin_over_centre": float(medians[best_conv] - medians["centre"]),
        "n_shots_used": len(used_shots),
        "shots_used": used_shots,
        "per_shot": per_shot,
    }


# ----------------------------------------------------------------------- stats codec & camera fit

@dataclass(frozen=True)
class CameraStatsCodec:
    """64-bin grid codec for per-frame ``(m, log10(max(s, s_floor)))``."""

    grids: GridSet
    s_floor: float

    @property
    def n_bins(self) -> int:
        return int(self.grids.n_bins)

    @property
    def missing_code(self) -> int:
        return int(self.grids.n_bins)

    @property
    def vocab_size(self) -> int:
        return int(self.grids.n_bins) + 1

    def encode(self, m: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Encode ``(F,)`` arrays ``m, s`` -> ``(F, 2)`` int64 codes in ``0..n_bins``."""
        m_arr = np.asarray(m, dtype=np.float64).reshape(-1)
        s_arr = np.asarray(s, dtype=np.float64).reshape(-1)
        if m_arr.shape != s_arr.shape:
            raise ValueError(f"m and s shapes differ: {m_arr.shape} vs {s_arr.shape}")
        ok_s = np.isfinite(s_arr)
        log10_s = np.full_like(s_arr, np.nan)
        log10_s[ok_s] = np.log10(np.maximum(s_arr[ok_s], float(self.s_floor)))
        vals = np.stack([m_arr, log10_s], axis=-1)[..., None]  # (F, 2, 1)
        idx = self.grids.index(vals)  # (F, 2, 1), -1 for NaN
        return np.where(idx < 0, self.missing_code, idx).reshape(-1, 2).astype(np.int64)

    def decode(self, codes: torch.Tensor | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode ``(F, 2)`` codes -> ``(m_dec, s_dec)`` float64 ``(F,)`` (NaN on missing)."""
        arr = (
            codes.detach().cpu().numpy()
            if isinstance(codes, torch.Tensor)
            else np.asarray(codes)
        )
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"expected (F, 2) stats codes, got {arr.shape}")
        c_idx = arr.astype(np.int64)
        n_bins = self.n_bins
        valid = (c_idx >= 0) & (c_idx < n_bins)
        safe = np.where(valid, c_idx, 0)
        centres = self.grids.centres()[:, 0, :]  # (2, n_bins)
        m_dec = np.where(valid[:, 0], centres[0, safe[:, 0]], np.nan)
        log10_s_dec = np.where(valid[:, 1], centres[1, safe[:, 1]], np.nan)
        s_dec = np.where(valid[:, 1], np.power(10.0, log10_s_dec), np.nan)
        return m_dec, s_dec


@dataclass(frozen=True)
class CameraFit:
    """Fitted camera normalization parameters, timing convention, and stats grids."""

    s_floor: float
    time_convention: str
    stats_codec: CameraStatsCodec
    u_std: float
    meta: dict[str, Any]


def fit_camera_stats(
    m_all: np.ndarray,
    s_all: np.ndarray,
    *,
    n_bins: int = CAMERA_STATS_N_BINS,
    margin: float = CAMERA_STATS_MARGIN,
) -> tuple[float, CameraStatsCodec, float]:
    """Fit ``s_floor`` (1st percentile of ``s``), 64-bin stats grids, and global ``u_std``."""
    m_arr = np.asarray(m_all, dtype=np.float64).reshape(-1)
    s_arr = np.asarray(s_all, dtype=np.float64).reshape(-1)
    ok = np.isfinite(m_arr) & np.isfinite(s_arr) & (s_arr > 0.0)
    if not ok.any():
        raise ValueError("fit_camera_stats: no finite positive (m, s) frames")
    m_f = m_arr[ok]
    s_f = s_arr[ok]
    s_floor = float(np.percentile(s_f, 1.0))
    if not (math.isfinite(s_floor) and s_floor > 0.0):
        s_floor = float(np.min(s_f))
    log10_s = np.log10(np.maximum(s_f, s_floor))
    vals = np.stack([m_f, log10_s], axis=-1)[..., None]  # (N, 2, 1)
    lo, hi, width = fit_ranges(vals, margin=float(margin))
    grids = GridSet(lo, hi, width, n_bins=int(n_bins))
    # Law of total variance over equal-sized frames: Var(u) = E[s^2] + Var(m).
    u_var = float(np.mean(s_f * s_f) + np.var(m_f))
    u_std = float(math.sqrt(max(u_var, 1e-12)))
    return s_floor, CameraStatsCodec(grids=grids, s_floor=s_floor), u_std


def save_camera_fit(
    out_dir: Path | str,
    s_floor: float,
    stats_codec: CameraStatsCodec,
    meta_doc: Mapping[str, Any],
) -> str:
    """Atomically write deterministic ``camera_fit.npz`` and ``camera_fit.json``; return sha256."""
    out_path = paths.prepare_write_root(Path(out_dir))
    flat_entries = {
        "s_floor": np.asarray([float(s_floor)], dtype=np.float64),
        "lo": np.asarray(stats_codec.grids.lo, dtype=np.float64),
        "hi": np.asarray(stats_codec.grids.hi, dtype=np.float64),
        "width": np.asarray(stats_codec.grids.width, dtype=np.float64),
    }
    npz_tmp = out_path / f"camera_fit.npz.{os.getpid()}.tmp"
    try:
        with zipfile.ZipFile(npz_tmp, "w", zipfile.ZIP_STORED) as zf:
            for key in sorted(flat_entries):
                buf = io.BytesIO()
                np.lib.format.write_array(buf, flat_entries[key], allow_pickle=False)
                zi = zipfile.ZipInfo(f"{key}.npy", date_time=_FIXED_TIME)
                zi.compress_type = zipfile.ZIP_STORED
                zi.external_attr = 0o600 << 16
                zf.writestr(zi, buf.getvalue())
        os.replace(npz_tmp, out_path / "camera_fit.npz")
    finally:
        npz_tmp.unlink(missing_ok=True)

    framecache.save_json_atomic(dict(meta_doc), out_path / "camera_fit.json")
    return camera_fit_sha256(out_path)


def camera_fit_sha256(fit_dir: Path | str) -> str:
    """sha256 hex digest of ``camera_fit.npz`` + ``camera_fit.json``."""
    fit_path = Path(fit_dir)
    h = hashlib.sha256()
    for fname in ("camera_fit.npz", "camera_fit.json"):
        with open(fit_path / fname, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def load_camera_fit(fit_dir: Path | str) -> CameraFit:
    """Load ``<fit_dir>/camera_fit.{json,npz}`` into a :class:`CameraFit`."""
    fit_path = Path(fit_dir)
    meta = json.loads((fit_path / "camera_fit.json").read_text())
    n_bins = int(meta.get("n_bins", CAMERA_STATS_N_BINS))
    with np.load(fit_path / "camera_fit.npz", allow_pickle=False) as z:
        s_floor = float(np.asarray(z["s_floor"], dtype=np.float64).reshape(()))
        lo = np.asarray(z["lo"], dtype=np.float64)
        hi = np.asarray(z["hi"], dtype=np.float64)
        width = np.asarray(z["width"], dtype=np.float64)
    grids = GridSet(lo, hi, width, n_bins=n_bins)
    conv = str(meta["time_convention"])
    if conv not in ("start", "centre", "end"):
        raise ValueError(f"invalid time_convention in {fit_path / 'camera_fit.json'}: {conv!r}")
    u_std = float(meta.get("u_std", float("nan")))
    return CameraFit(
        s_floor=s_floor,
        time_convention=conv,
        stats_codec=CameraStatsCodec(grids=grids, s_floor=s_floor),
        u_std=u_std,
        meta=meta,
    )


def resolve_manifest_camera(manifest: Mapping[str, Any]) -> dict[str, Path] | None:
    """Return verified camera paths from ``_codec_manifest.json``, or ``None`` if absent."""
    path_keys = ("camera_codec_path", "camera_fit_path", "camera_frames_dir")
    sha_keys = (
        "camera_codec_sha256",
        "camera_fit_sha256",
        "camera_frames_index_sha256",
    )
    raw_paths = [manifest.get(k) for k in path_keys]
    raw_shas = [manifest.get(k) for k in sha_keys]
    if all(v is None for v in raw_paths) and all(v is None for v in raw_shas):
        return None
    if any(v is None for v in raw_paths):
        raise ValueError(
            "manifest has incomplete camera path fields: "
            + ", ".join(f"{k}={manifest.get(k)!r}" for k in path_keys)
        )
    if any(not v for v in raw_shas):
        raise ValueError(
            "manifest sets camera paths without all camera sha256 digests: "
            + ", ".join(f"{k}={manifest.get(k)!r}" for k in sha_keys)
        )
    codec_path = Path(str(raw_paths[0]))
    fit_path = Path(str(raw_paths[1]))
    frames_dir = Path(str(raw_paths[2]))

    checks = (
        ("camera_codec_sha256", file_sha256(codec_path), str(raw_shas[0])),
        ("camera_fit_sha256", camera_fit_sha256(fit_path), str(raw_shas[1])),
        (
            "camera_frames_index_sha256",
            file_sha256(frames_dir / "index.json"),
            str(raw_shas[2]),
        ),
    )
    for key, actual, expected in checks:
        if actual != expected:
            raise ValueError(
                f"{key} mismatch: manifest has {expected}, disk has {actual}"
            )
    return {
        "codec_path": codec_path,
        "fit_path": fit_path,
        "frames_dir": frames_dir,
    }


# ----------------------------------------------------------------------- plasma exposures & dataset

def load_plasma_ranges(
    cache_dir: Path | str,
    shots: Sequence[str],
) -> tuple[dict[str, tuple[float, int]], float]:
    """Load per-shot ``(t0, n_frames)`` plasma ranges from ``<cache_dir>/<shot>.pt``."""
    cdir = Path(cache_dir)
    manifest = json.loads((cdir / "_codec_manifest.json").read_text())
    dt_frame = float(manifest.get("dt_frame", framecache.DT_FRAME))
    ranges: dict[str, tuple[float, int]] = {}
    for s in shots:
        pt = cdir / f"{s}.pt"
        if not pt.exists():
            continue
        entry = torch.load(pt, map_location="cpu", weights_only=False)
        ranges[str(s)] = (float(entry["t0"]), int(entry["n_frames"]))
    return ranges, dt_frame


def compute_plasma_exposures(
    frames_dir: Path | str,
    plasma_ranges: Mapping[str, tuple[float, int]],
    dt_frame: float,
    convention: str,
) -> dict[str, list[int]]:
    """Return sorted unique exposure indices mapped to plasma frames for each shot."""
    fdir = Path(frames_dir)
    index_doc = json.loads((fdir / "index.json").read_text())
    shots_index = index_doc["shots"]
    out: dict[str, list[int]] = {}
    for shot, (t0, n_frames) in plasma_ranges.items():
        if (shots_index.get(shot) or {}).get("status") != "ok":
            continue
        npz_path = fdir / f"{shot}.npz"
        if not npz_path.exists():
            continue
        fr = load_frames_npz(npz_path, load_u=False)
        mapping = map_frames_to_exposures(
            fr["time"],
            fr["exposure_s"],
            t0,
            n_frames,
            dt_frame=dt_frame,
            convention=convention,
        )
        uniq = np.unique(mapping[mapping >= 0])
        if uniq.size > 0:
            out[shot] = [int(i) for i in uniq.tolist()]
    return out


def _camera_worker_init(_worker_id: int) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


class MastCameraDataset(Dataset):
    """Stream preprocessed MAST ``rgb`` exposures in ``(shot, exposure_idx)`` order.

    Top-level and picklable so ``DataLoader(num_workers > 0)`` works under both
    ``fork`` and ``spawn`` as well as ``torchrun`` DDP.
    """

    def __init__(
        self,
        frames_dir: Path | str,
        shots: Sequence[str | int],
        cfg: VideoCodecConfig,
        *,
        s_floor: float,
        plasma_exposures: Mapping[str, Sequence[int]],
        max_cached_shots: int = 32,
    ) -> None:
        self.frames_dir = str(Path(frames_dir))
        self.codec_cfg = cfg
        self.s_floor = float(s_floor)
        self.max_cached_shots = max(1, int(max_cached_shots))

        shot_list: list[str] = []
        shot_idx_chunks: list[np.ndarray] = []
        exp_idx_chunks: list[np.ndarray] = []
        for s in shots:
            s_str = str(s)
            exps = plasma_exposures.get(s_str)
            if not exps:
                continue
            npz_path = Path(self.frames_dir) / f"{s_str}.npz"
            if not npz_path.exists():
                continue
            sid = len(shot_list)
            shot_list.append(s_str)
            arr = np.asarray(exps, dtype=np.int32)
            shot_idx_chunks.append(np.full(arr.shape[0], sid, dtype=np.int32))
            exp_idx_chunks.append(arr)

        if not shot_list:
            raise ValueError(
                f"MastCameraDataset: no plasma exposures found under {self.frames_dir} "
                f"for {len(list(shots))} requested shots"
            )
        self.shots = tuple(shot_list)
        self._shot_ids = np.concatenate(shot_idx_chunks)
        self._exp_ids = np.concatenate(exp_idx_chunks)
        self._cache_pid: int = -1
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache_pid"] = -1
        state["_cache"] = OrderedDict()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._cache_pid = -1
        self._cache = OrderedDict()

    def __len__(self) -> int:
        return int(self._exp_ids.shape[0])

    def _get_shot_arrays(self, shot: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pid = os.getpid()
        if self._cache_pid != pid:
            self._cache = OrderedDict()
            self._cache_pid = pid
        hit = self._cache.get(shot)
        if hit is not None:
            self._cache.move_to_end(shot)
            return hit
        fr = load_frames_npz(Path(self.frames_dir) / f"{shot}.npz", load_u=True)
        entry = (fr["u"], fr["m"], fr["s"])
        self._cache[shot] = entry
        if len(self._cache) > self.max_cached_shots:
            self._cache.popitem(last=False)
        return entry

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        shot = self.shots[int(self._shot_ids[idx])]
        exp_idx = int(self._exp_ids[idx])
        u, m, s = self._get_shot_arrays(shot)
        z = normalize_u_frames(u[exp_idx], float(m[exp_idx]), float(s[exp_idx]), self.s_floor)
        cfg = self.codec_cfg
        if (z.shape[-2], z.shape[-1]) != (cfg.height, cfg.width):
            z = F.interpolate(
                z,
                size=(cfg.height, cfg.width),
                mode="bilinear",
                align_corners=False,
            )
        frame_mask = torch.ones(cfg.frames, dtype=torch.float32)
        return z, frame_mask


def make_mast_video_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Plain ``DataLoader`` (with ``DistributedSampler`` when ``world_size > 1``)."""
    if world_size > 1:
        sampler: torch.utils.data.Sampler | None = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )
        dl_shuffle = False
    else:
        sampler = None
        dl_shuffle = shuffle
    per_rank = len(sampler) if sampler is not None else len(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=dl_shuffle,
        num_workers=num_workers,
        collate_fn=tc._video_collate,
        worker_init_fn=_camera_worker_init if num_workers > 0 else None,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=per_rank >= batch_size,
    )


@contextlib.contextmanager
def patched_video_codec(
    *,
    frames_dir: Path | str,
    s_floor: float,
    plasma_exposures: Mapping[str, Sequence[int]],
) -> Iterator[None]:
    """Patch ``train_codec.VideoCodecPairDataset`` and ``make_video_loader`` for MAST rgb."""
    fdir = Path(frames_dir)
    sf = float(s_floor)
    pe = plasma_exposures

    def _dataset_factory(
        _modality: str,
        shots: Sequence[str | int],
        cfg: VideoCodecConfig,
        data_dir: Path | str = "",
        *,
        seed: int = 0,
        lengths_cache_path: Path | str | None = None,
        **_kwargs: Any,
    ) -> MastCameraDataset:
        return MastCameraDataset(
            fdir,
            shots,
            cfg,
            s_floor=sf,
            plasma_exposures=pe,
        )

    prev_ds = tc.VideoCodecPairDataset
    prev_loader = tc.make_video_loader
    tc.VideoCodecPairDataset = _dataset_factory  # type: ignore[assignment]
    tc.make_video_loader = make_mast_video_loader  # type: ignore[assignment]
    try:
        yield
    finally:
        tc.VideoCodecPairDataset = prev_ds
        tc.make_video_loader = prev_loader


# ----------------------------------------------------------------------- encode & decode

@torch.no_grad()
def encode_camera_codes(
    codec,
    camera_fit: CameraFit,
    frame_npz_path: Path | str | None,
    *,
    t0: float,
    n_frames: int,
    dt_frame: float = framecache.DT_FRAME,
    batch_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Encode one shot's camera frames onto the ``n_frames`` 5 ms grid.

    Returns ``(codes (F, n_tok) int64, stats_codes (F, 2) int64, present: bool)``.
    Every overlapping exposure is encoded once and indexed onto all 5 ms frames
    mapped to it. Uncovered frames receive ``CAMERA_MISSING_CODE`` (1000) and
    ``CAMERA_STATS_MISSING_CODE`` (64). Absent shots receive ``PLACEHOLDER_CODE`` (0)
    and ``present = False``.
    """
    n_f = int(n_frames)
    n_tok = int(codec.cfg.n_tok)
    if frame_npz_path is None or not Path(frame_npz_path).exists():
        return (
            torch.full((n_f, n_tok), framecache.PLACEHOLDER_CODE, dtype=torch.int64),
            torch.full((n_f, 2), framecache.PLACEHOLDER_CODE, dtype=torch.int64),
            False,
        )

    fr = load_frames_npz(frame_npz_path, load_u=True)
    mapping = map_frames_to_exposures(
        fr["time"],
        fr["exposure_s"],
        t0,
        n_f,
        dt_frame=dt_frame,
        convention=camera_fit.time_convention,
    )
    valid = mapping >= 0
    if not valid.any():
        return (
            torch.full((n_f, n_tok), framecache.PLACEHOLDER_CODE, dtype=torch.int64),
            torch.full((n_f, 2), framecache.PLACEHOLDER_CODE, dtype=torch.int64),
            False,
        )

    uniq_exp, inv = np.unique(mapping[valid], return_inverse=True)
    z_all = normalize_u_frames(
        fr["u"][uniq_exp],
        fr["m"][uniq_exp],
        fr["s"][uniq_exp],
        camera_fit.s_floor,
    )
    cfg = codec.cfg
    if (z_all.shape[-2], z_all.shape[-1]) != (cfg.height, cfg.width):
        b_cnt = z_all.shape[0]
        z_all = F.interpolate(
            z_all.reshape(b_cnt, cfg.channels, z_all.shape[-2], z_all.shape[-1]),
            size=(cfg.height, cfg.width),
            mode="bilinear",
            align_corners=False,
        ).reshape(b_cnt, cfg.channels, cfg.frames, cfg.height, cfg.width)

    code_chunks: list[torch.Tensor] = []
    bs = max(1, int(batch_size))
    for start in range(0, z_all.shape[0], bs):
        code_chunks.append(td.encode_flat(codec, z_all[start : start + bs]))
    uniq_codes = torch.cat(code_chunks, dim=0).to(torch.int64)  # (K, n_tok)
    uniq_stats = camera_fit.stats_codec.encode(fr["m"][uniq_exp], fr["s"][uniq_exp])  # (K, 2)

    cam_missing_code = int(codec.codebook_size)
    stats_missing_code = int(camera_fit.stats_codec.missing_code)
    codes = torch.full((n_f, n_tok), cam_missing_code, dtype=torch.int64)
    stats_codes = torch.full((n_f, 2), stats_missing_code, dtype=torch.int64)
    inv_t = torch.from_numpy(inv.astype(np.int64))
    valid_idx = torch.from_numpy(np.flatnonzero(valid).astype(np.int64))
    codes[valid_idx] = uniq_codes[inv_t]
    stats_codes[valid_idx] = torch.from_numpy(uniq_stats[inv])
    return codes, stats_codes, True


@torch.no_grad()
def decode_camera(
    codec,
    camera_fit: CameraFit,
    codes: torch.Tensor | np.ndarray,
    stats_codes: torch.Tensor | np.ndarray,
    *,
    device: torch.device | None = None,
    chunk_size: int = 32,
    return_u: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Decode ``(F, n_tok)`` camera codes and ``(F, 2)`` stats codes back to counts ``(F, H, W)``."""
    c_t = (
        codes.detach().cpu().long()
        if isinstance(codes, torch.Tensor)
        else torch.as_tensor(codes, dtype=torch.long)
    )
    if c_t.ndim != 2 or c_t.shape[1] != int(codec.cfg.n_tok):
        raise ValueError(
            f"expected (F, {codec.cfg.n_tok}) camera codes, got {tuple(c_t.shape)}"
        )
    n_f = int(c_t.shape[0])
    m_dec, s_dec = camera_fit.stats_codec.decode(stats_codes)
    if m_dec.shape[0] != n_f:
        raise ValueError(
            f"camera codes ({n_f}) and stats codes ({m_dec.shape[0]}) frame count mismatch"
        )
    codebook = int(codec.codebook_size)
    cam_valid = ((c_t >= 0) & (c_t < codebook)).all(dim=1).numpy()
    frame_valid = cam_valid & np.isfinite(m_dec) & np.isfinite(s_dec)

    h, w = int(codec.cfg.height), int(codec.cfg.width)
    u_out = np.full((n_f, h, w), np.nan, dtype=np.float32)
    counts_out = np.full((n_f, h, w), np.nan, dtype=np.float32)

    valid_idx = np.flatnonzero(frame_valid)
    dev = device if device is not None else next(codec.parameters()).device
    cs = max(1, int(chunk_size))
    for start in range(0, valid_idx.size, cs):
        idx_chunk = valid_idx[start : start + cs]
        flat_chunk = c_t[torch.from_numpy(idx_chunk)].to(dev)
        z_rec = ev.decode_flat(codec, flat_chunk).float().cpu().numpy()  # (B, 1, 1, H, W)
        z_2d = z_rec[:, 0, 0, :, :]
        scale = np.maximum(s_dec[idx_chunk], float(camera_fit.s_floor)).astype(np.float32)
        mean = m_dec[idx_chunk].astype(np.float32)
        u_chunk = z_2d * scale[:, None, None] + mean[:, None, None]
        u_out[idx_chunk] = u_chunk
        counts_out[idx_chunk] = np.expm1(u_chunk).astype(np.float32)

    if return_u:
        return counts_out, u_out
    return counts_out
