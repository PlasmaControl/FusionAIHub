"""Read-only EFIT geometry for lower-outer-leg detachment review.

The shelf gate reproduces the producer's physical bounds. Magnetic configuration
is reported separately: two real opposite X-points do not establish double null
without a separatrix-balance measurement, and missing X-points do not establish a
limited plasma. Such cases remain unknown.
"""

from __future__ import annotations

import os
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

MAX_GAP_MS = 40.0
MIN_PLASMA_IP_A = 300000.0  # Same fixed threshold as detach_bins.MIN_IP_A.
SENTINELS = (-0.89, -9.99, 0.0)
THRESHOLDS = {
    "max_efit_gap_ms": MAX_GAP_MS,
    "lsn_zx_max_m": -0.5,
    "shelf_r_min_m": 1.37,
    "shelf_z_m": -1.25,
    "shelf_z_tolerance_m": 0.05,
    # A display convention, never a fitted label or eligibility threshold.
    "double_null_drsep_tolerance_m": 0.01,
}
NODES = ("rvsod", "zvsod", "rxpt1", "zxpt1", "rxpt2", "zxpt2", "drsep")


def source_path(shot, paths) -> Path:
    """A portable local cache root; reading this module never fetches."""
    root = os.environ.get("LABELER_DETACHMENT_GEOMETRY_ROOT")
    root = Path(root) if root else paths.root / "indicators" / "detachment"
    return root / f"{int(shot)}.npz"


def current_window(shot, paths):
    """First/last cached |Ip| >= 300 kA samples, in ms, as in the producer.

    This only supplies external producer shots without a catalogued window. A
    missing, unfinished, or nonphysical current record cannot establish one.
    """
    path = source_path(shot, paths)
    if not path.is_file():
        return None
    try:
        with path.open("rb") as stream, np.load(stream, allow_pickle=False) as data:
            times = np.asarray(data["ipmeas__t"], dtype=float)
            current = np.asarray(data["ipmeas__y"], dtype=float)
    except (OSError, ValueError, KeyError, BadZipFile, EOFError):
        return None
    if times.ndim != 1 or current.shape != times.shape:
        return None
    live = np.isfinite(times) & np.isfinite(current) & (abs(current) >= MIN_PLASMA_IP_A)
    if live.sum() < 2:
        return None
    start, end = float(times[live].min()), float(times[live].max())
    return (start, end) if end > start else None


def real(values, lower, upper):
    """Finite physical positions, excluding the producer's EFIT sentinels."""
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(values) & (values > lower) & (values < upper)
    for sentinel in SENTINELS:
        keep &= ~np.isclose(values, sentinel)
    return keep


def shelf_gate(rvsod, zvsod, rxpt1, zxpt1):
    """Producer-compatible shelf validity and first failing reason, per sample."""
    rv, zv, rx, zx = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (rvsod, zvsod, rxpt1, zxpt1))
    )
    have = (
        real(rv, 0.8, 2.5)
        & real(zv, -1.6, -0.9)
        & real(rx, 0.8, 2.5)
        & real(zx, -1.6, 1.6)
    )
    lower = have & (zx < THRESHOLDS["lsn_zx_max_m"])
    shelf = (rv >= THRESHOLDS["shelf_r_min_m"]) & (
        np.abs(zv - THRESHOLDS["shelf_z_m"]) <= THRESHOLDS["shelf_z_tolerance_m"]
    )
    floor = (rv < THRESHOLDS["shelf_r_min_m"]) & (
        zv < THRESHOLDS["shelf_z_m"] - THRESHOLDS["shelf_z_tolerance_m"]
    )
    valid = lower & shelf
    reason = np.full(rv.shape, "", dtype=object)
    reason[have & ~lower] = "not_lower_null"
    reason[lower & ~shelf] = "strike_not_on_shelf"
    reason[lower & floor] = "strike_on_floor"
    reason[~have] = "efit_missing"
    return valid, reason


def nearest(times_ms, values, at_ms, max_gap_ms=MAX_GAP_MS):
    """Nearest EFIT sample within the producer's 40 ms limit; no extrapolation."""
    times = np.asarray(times_ms, dtype=float)
    values = np.asarray(values, dtype=float)
    at = np.asarray(at_ms, dtype=float)
    out = np.full(at.shape, np.nan)
    if times.ndim != 1 or values.shape != times.shape or not len(times):
        return out
    good = np.isfinite(times)
    order = np.argsort(times[good], kind="stable")
    times, values = times[good][order], values[good][order]
    if not len(times):
        return out
    right = np.clip(np.searchsorted(times, at), 0, len(times) - 1)
    left = np.clip(right - 1, 0, len(times) - 1)
    pick = np.where(abs(times[left] - at) <= abs(times[right] - at), left, right)
    close = np.isfinite(at) & (abs(times[pick] - at) <= max_gap_ms)
    out[close] = values[pick[close]]
    return out


def configurations(rx1, zx1, rx2, zx2, drsep):
    """Conservative topology from X-points and, when available, DRSEP.

    A sole primary X-point identifies LSN/USN. Two opposite X-points need DRSEP
    to distinguish a balanced DN from the primary single null. The cache does not
    carry a verified limiter-status field, so missing points stay unknown.
    """
    rx1, zx1, rx2, zx2, drsep = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (rx1, zx1, rx2, zx2, drsep))
    )
    primary = real(rx1, 0.8, 2.5) & real(zx1, -1.6, 1.6)
    secondary = real(rx2, 0.8, 2.5) & real(zx2, -1.6, 1.6)
    lower, upper = primary & (zx1 < -0.5), primary & (zx1 > 0.5)
    opposite = (lower & secondary & (zx2 > 0.5)) | (upper & secondary & (zx2 < -0.5))
    # +-0.4 m is EFIT's saturated DRSEP search, not a balance measurement.
    balance = np.isfinite(drsep) & (abs(drsep) < 0.399)
    unambiguous = ~opposite | balance
    result = np.full(rx1.shape, "unknown", dtype=object)
    result[lower & unambiguous] = "LSN"
    result[upper & unambiguous] = "USN"
    result[
        opposite & balance & (abs(drsep) <= THRESHOLDS["double_null_drsep_tolerance_m"])
    ] = "DN"
    return result


def load(shot, paths, window_ms=None):
    """JSON-ready EFIT samples, configuration counts, and shelf-gate evidence."""
    path = source_path(shot, paths)
    result = {
        "source": str(path),
        "window_ms": list(window_ms) if window_ms else None,
        "thresholds": THRESHOLDS,
        "note": (
            "EFIT primary/secondary X-points; DN needs DRSEP balance. Missing "
            "X-points do not prove limited geometry. Shelf gate is the producer's "
            "primary lower-null/outer-strike-point test, separate from topology."
        ),
        "counts": dict.fromkeys(("LSN", "USN", "DN", "limited", "unknown"), 0),
        "shelf_gate_samples": 0,
        "total_samples": 0,
        "samples": [],
    }
    if not path.is_file():
        result["reason"] = "efit_missing"
        return result
    try:
        with path.open("rb") as stream, np.load(stream, allow_pickle=False) as stored:
            if "zxpt1__t" not in stored:
                result["reason"] = "efit_missing"
                return result
            times = np.asarray(stored["zxpt1__t"], dtype=float)
            times = np.unique(times[np.isfinite(times)])
            if window_ms:
                times = times[(times >= window_ms[0]) & (times <= window_ms[1])]
            values = {}
            for node in NODES:
                tk, yk = f"{node}__t", f"{node}__y"
                values[node] = (
                    nearest(stored[tk], stored[yk], times)
                    if tk in stored and yk in stored
                    else np.full(len(times), np.nan)
                )
    except (OSError, ValueError, KeyError, BadZipFile, EOFError) as error:
        result["reason"] = f"efit_unreadable: {type(error).__name__}"
        return result
    valid, reasons = shelf_gate(*(values[k] for k in NODES[:4]))
    config = configurations(*(values[k] for k in NODES[2:]))
    result["counts"] = {k: int(np.sum(config == k)) for k in result["counts"]}
    result["shelf_gate_samples"] = int(valid.sum())
    result["total_samples"] = len(times)
    for i, time in enumerate(times):
        sample = {
            "time_ms": float(time),
            "configuration": str(config[i]),
            "shelf_gate": bool(valid[i]),
            "reason": str(reasons[i]),
        }
        for key, array in values.items():
            sample[f"{key}_m"] = float(array[i]) if np.isfinite(array[i]) else None
        result["samples"].append(sample)
    return result


def at_times(metadata, times_ms):
    """Configuration and shelf gate at camera/cursor times; gaps remain unknown."""
    at = np.asarray(times_ms, dtype=float)
    samples = metadata["samples"]
    source = np.asarray([s["time_ms"] for s in samples])
    index = nearest(source, np.arange(len(source), dtype=float), at)
    configuration = np.full(at.shape, "unknown", dtype=object)
    gate = np.zeros(at.shape, dtype=bool)
    for i in np.flatnonzero(np.isfinite(index)):
        sample = samples[int(index[i])]
        configuration[i] = sample["configuration"]
        gate[i] = sample["shelf_gate"]
    return configuration, gate
