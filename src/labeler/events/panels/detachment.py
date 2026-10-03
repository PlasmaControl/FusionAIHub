"""Local detachment context and validity-gated producer indicators.

Raw TPLANG sweeps and medians of uncalibrated bolometer voltages are omitted:
neither measures Isat nor radiated power. This view never fetches or votes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ...config import Paths
from ..verify import Panel
from ._shared import plasma_window

log = logging.getLogger(__name__)
# Channel order from modalities.yaml. Calibrated spectroscopy units are recorded
# by the raw source and configs/shot_design/signals.yaml (dalpha). Corpus HDF5
# strips chord locations, so do not assign upper/lower/midplane from an index.
TRACES = (
    (
        "filterscopes",
        "D-alpha filterscopes",
        [f"FS{i:02d}" for i in range(1, 9)],
        "ph/(sr cm2 s)",
        1.0,
    ),
    (
        "co2",
        "CO2 R0 density proxy (corpus DENUF)",
        ["R0 DENUF"],
        "native units (unverified)",
        1.0,
    ),
    (
        "gas_flow",
        "Gas flow",
        [
            "GASA",
            "GASB",
            "GASC",
            "GASD",
            "GASE",
            "LOB1",
            "LOB2",
            "PFX1",
            "PFX2",
            "PFX3",
            "UOB",
        ],
        "Torr L/s",
        1.0,
    ),
)
CSV_INDICATORS = (
    ("afrac", "Afrac", "dimensionless"),
    ("prad_div", "f_div = Prad,div / P_in", "dimensionless"),
    ("tangtv_front_height", "TangTV normalized front DZ", "dimensionless"),
)
BIN_INDICATORS = (
    ("afrac", "Afrac", "dimensionless"),
    ("prad", "f_div = Prad,div / P_in", "dimensionless"),
    ("tangtv", "TangTV normalized front DZ", "dimensionless"),
)
AFRAC_CAVEAT = (
    "Uncalibrated local proxies may use a within-shot reference. Check the "
    "producer recipe and per-bin method before interpreting Afrac; "
    "do not use this indicator alone."
)
CO2_CAVEAT = (
    "CO2 UF density input is a line integral without a chord-length division; "
    "native UF units are unverified. It is upstream density context, not a "
    "calibrated line-average or local Thomson measurement."
)


def indicator_path(shot, paths):
    """Prefer producer bins, retaining the original indicator interchange."""
    override = os.environ.get("LABELER_DETACHMENT_INDICATORS")
    roots = (
        [Path(override)]
        if override
        else [
            paths.root / "round4/detach/bins",
            paths.root / "round4/detach/indicators",
        ]
    )
    for root in roots:
        for suffix in (".npz", ".csv"):
            path = root / f"{int(shot)}{suffix}"
            if path.is_file():
                return path
    return roots[0] / f"{int(shot)}.npz"


def density_cache_path(shot, paths):
    """Local producer CO2 cache; this adapter never fetches missing signals."""
    root = Path(
        os.environ.get(
            "LABELER_DETACHMENT_CACHE_ROOT", str(paths.root / "round4/detach/cache")
        )
    )
    return root / f"{int(shot)}.npz"


def indicator_panels(shot, paths, t_range=None):
    """Read detach_bins' NPZ/CSV, or the original CSV interchange.

    Producer values are Afrac, Prad_divL/P_in, and normalized DZ, NOT MW and
    metres. Preserve its validity gates, reasons and votes; never recompute them.
    Configure LABELER_DETACHMENT_INDICATORS, otherwise prefer producer bins and
    fall back to its original indicator CSVs.
    """
    from ..review import recipe

    labels = Path(
        os.environ.get(
            "LABELER_DETACHMENT_LABELS",
            str(paths.root / "round4/detach/labels_bins.csv.gz"),
        )
    )
    interpretation = recipe.load(labels)
    path = indicator_path(shot, paths)
    if not path.is_file():
        return []
    try:
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as source:
                data = {name: source[name] for name in source.files}
        else:
            data = {
                name: series.to_numpy() for name, series in pd.read_csv(path).items()
            }
        bins = "start_ms" in data
        x = np.asarray(data["start_ms" if bins else "t_ms"], dtype=float)
        if x.ndim != 1 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
            raise ValueError("indicator clock must be finite and increasing")
        keep = np.ones(len(x), dtype=bool)
        bin_metadata = {}
        if bins:
            if len(x) < 2:
                raise ValueError("at least two bin starts are needed to infer width")
            width = float(np.median(np.diff(x)))
            starts, ends = x.copy(), x + width
            if t_range is not None:
                keep = (starts < t_range[1]) & (ends > t_range[0])
                starts = np.maximum(starts, t_range[0])
                ends = np.minimum(ends, t_range[1])
            x = (starts + ends) / 2
            bin_metadata = {
                "trace_style": "step",
                "bin_width_ms": width,
                "bin_start_ms": starts[keep].tolist(),
                "bin_end_ms": ends[keep].tolist(),
            }
        elif t_range is not None:
            keep = (x >= t_range[0]) & (x <= t_range[1])
        built = []
        schema = "detach_bins" if bins else "indicator_csv"
        for name, title, unit in BIN_INDICATORS if bins else CSV_INDICATORS:
            value = f"{name}_value" if bins else name
            if value not in data or f"{name}_valid" not in data:
                continue
            y = np.asarray(data[value], dtype=float).copy()
            valid = np.asarray(data[f"{name}_valid"], dtype=float) == 1
            if y.shape != x.shape or valid.shape != x.shape:
                raise ValueError(f"{name}: value/valid shape disagrees with clock")
            y[~valid] = np.nan
            indicator = {"prad_div": "prad", "tangtv_front_height": "tangtv"}.get(
                name, name
            )
            metadata = {
                "source": str(path),
                "schema": schema,
                "indicator": indicator,
                "valid": valid[keep].tolist(),
                **bin_metadata,
            }
            for suffix in ("reason", "vote"):
                key = f"{name}_{suffix}"
                if key in data:
                    values = np.asarray(data[key])
                    if values.shape != x.shape:
                        raise ValueError(f"{key}: shape disagrees with clock")
                    metadata[suffix] = values[keep].tolist()
            hlines = recipe.guides(interpretation["record"], indicator)
            if indicator == "prad":
                hlines = sorted(set(hlines) | {1.0})
                metadata["caveat"] = (
                    "f_div > 1: check the heating-power denominator and "
                    "radiated-power estimate before interpreting this ratio."
                )
            metadata["recipe_sources"] = interpretation["sources"]
            if indicator == "afrac":
                if "afrac_method" in data:
                    metadata["afrac_method"] = data["afrac_method"][keep].tolist()
                metadata.update(
                    {
                        "caveat": AFRAC_CAVEAT,
                        "interpretation": "See the stored producer recipe and bin votes.",
                    }
                )
            if indicator == "tangtv":
                sources = np.asarray(data.get("tangtv_source", ["unknown"] * len(x)))
                if sources.shape != x.shape:
                    raise ValueError("tangtv_source: shape disagrees with clock")
                metadata["tangtv_source"] = sources[keep].astype(str).tolist()
                # Legacy CSVs omit provenance, so never infer an inversion from
                # their height column. Bin titles visibly identify every source.
                title = _tangtv_title(title, metadata["tangtv_source"])
            if keep.any() and (np.isfinite(y[keep]).any() or "vote" in metadata):
                built.append(
                    Panel(
                        title=title,
                        x=x[keep],
                        y=y[None, keep],
                        ylabel=unit,
                        legend=[f"{title} ({unit})"],
                        hlines=hlines,
                        metadata=metadata,
                    )
                )
        if bins:
            for name, title, unit in (
                (
                    "aux_ne",
                    "CO2 density proxy (producer aux_ne)",
                    "native units (unverified)",
                ),
                (
                    "aux_te_div",
                    "Divertor Thomson Te (independent check; per-bin peak)",
                    "eV",
                ),
            ):
                if name not in data:
                    continue
                y = np.asarray(data[name], dtype=float).copy()
                if y.shape != x.shape:
                    raise ValueError(f"{name}: shape disagrees with clock")
                y[~np.isfinite(y) | (y <= 0)] = np.nan
                if not np.isfinite(y[keep]).any():
                    continue
                metadata = {
                    "source": str(path),
                    "schema": schema,
                    "quantity": name,
                    "missing_policy": "nonfinite and nonpositive values are missing",
                    **bin_metadata,
                }
                if name == "aux_ne":
                    metadata.update(
                        {
                            "context_quantity": "density",
                            "measurement": "CO2 line-integrated density proxy",
                            "source_selection": (
                                "Producer selects V2 (DENV2UF), then R0 (DENR0UF), "
                                "then V3 (DENV3UF); chosen chord not recorded in bins"
                            ),
                            "caveat": CO2_CAVEAT,
                        }
                    )
                else:
                    metadata.update(
                        {
                            "independent_check": True,
                            "measurement": (
                                "Peak over divertor Thomson channel bin medians"
                            ),
                            "caveat": (
                                "Independent temperature check; not an indicator. "
                                "Zero is a failed Thomson fit and is missing."
                            ),
                        }
                    )
                built.append(
                    Panel(
                        title=title,
                        x=x[keep],
                        y=y[None, keep],
                        ylabel=unit,
                        legend=[f"{title} ({unit})"],
                        metadata=metadata,
                    )
                )
        return built
    except (ValueError, KeyError, OSError) as error:
        log.warning("shot %s: cannot read detachment indicators: %s", shot, error)
        return []


def _tangtv_title(title, sources):
    """Identify model readings, including traces whose source changes by bin."""
    have = set(sources)
    descriptions = []
    if "inversion" in have:
        descriptions.append("tomographic inversion")
    if "surrogate" in have:
        descriptions.append("surrogate regression (model estimate)")
    if have - {"inversion", "surrogate", "none"}:
        descriptions.append("source not recorded")
    return f"{title} ({' + '.join(descriptions) or 'no front-height source'})"


def _block_means(data, x, ids, start, stop, step, *, positive=False):
    """Average contiguous native samples, including a final partial block.

    HDF5 reads are bounded to ~262k samples per channel. Float32 clocks use the
    span for spacing, avoiding quantized median-diff aliases at late shot times.
    """
    xs, ys = [], []
    valid_counts = np.zeros(len(ids), dtype=np.int64)
    chunk = max(1, 262144 // step) * step
    for lo in range(start, stop, chunk):
        hi = min(stop, lo + chunk)
        starts = np.arange(0, hi - lo, step)
        counts = np.minimum(step, hi - lo - starts)
        native = np.asarray(data[ids, lo:hi], dtype=np.float64)
        finite = np.isfinite(native)
        if positive:
            finite &= native > 0
        sums = np.add.reduceat(np.where(finite, native, 0), starts, axis=1)
        ns = np.add.reduceat(finite.astype(np.int32), starts, axis=1)
        valid_counts += ns.sum(axis=1)
        mean = np.full(sums.shape, np.nan)
        np.divide(sums, ns, out=mean, where=ns > 0)
        xs.append(np.add.reduceat(x[lo:hi], starts) / counts)
        ys.append(mean)
    return (
        np.concatenate(xs),
        np.concatenate(ys, axis=1),
        valid_counts / (stop - start),
    )


def _cached_density(shot, paths, t_range):
    """Use the producer's cached R0 input before corpus/local-density fallbacks."""
    path = density_cache_path(shot, paths)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if not {"denr0uf__t", "denr0uf__y"}.issubset(data.files):
                return None
            x = np.asarray(data["denr0uf__t"], dtype=float)
            y = np.asarray(data["denr0uf__y"], dtype=float).copy()
        if (
            x.ndim != 1
            or len(x) < 2
            or y.shape != x.shape
            or not np.isfinite(x).all()
            or np.any(np.diff(x) <= 0)
        ):
            raise ValueError("DENR0UF clock/value shapes must agree and increase")
        keep = np.ones(len(x), dtype=bool)
        if t_range is not None:
            keep = (x >= t_range[0]) & (x <= t_range[1])
        y[~np.isfinite(y) | (y <= 0)] = np.nan
        if not np.isfinite(y[keep]).any():
            return None
        title = "CO2 R0 density proxy (cached DENR0UF)"
        unit = "native units (unverified)"
        return Panel(
            title=title,
            x=x[keep],
            y=y[None, keep],
            ylabel=unit,
            legend=[f"DENR0UF ({unit})"],
            metadata={
                "source": str(path),
                "node": "DENR0UF",
                "context_quantity": "density",
                "measurement": "CO2 line-integrated density proxy",
                "source_selection": "Cached R0; producer aux_ne unavailable",
                "missing_policy": "nonfinite and nonpositive values are missing",
                "caveat": CO2_CAVEAT,
            },
        )
    except (ValueError, KeyError, OSError) as error:
        log.warning("shot %s: cannot read cached CO2 density: %s", shot, error)
        return None


def panels(shot, *, t_range=None, paths=None):
    """Block-mean context over the catalog plasma window, when available."""
    paths = Paths.from_env() if paths is None else paths
    if t_range is None:
        t_range = plasma_window(int(shot), paths)
    built = indicator_panels(shot, paths, t_range)
    have_density = any(p.metadata.get("context_quantity") == "density" for p in built)
    if not have_density:
        panel = _cached_density(shot, paths, t_range)
        if panel is not None:
            built.append(panel)
            have_density = True
    path = paths.corpus_file(int(shot))
    if not path.is_file():
        return built
    with h5py.File(path, "r") as source:
        for name, title, names, unit, spacing in TRACES:
            if name not in source or (name == "co2" and have_density):
                continue
            panel = _context_panel(
                source[name],
                title,
                names,
                unit,
                spacing,
                t_range,
                positive=name == "co2",
                positive_chords=name == "filterscopes",
                preplasma_baseline=name == "gas_flow",
            )
            if panel is not None:
                panel.metadata.update({"source": str(path), "corpus_group": name})
                if name == "filterscopes":
                    for c, (y, legend) in enumerate(
                        zip(panel.y, panel.legend, strict=True)
                    ):
                        chord = legend.split()[0]
                        built.append(
                            Panel(
                                title=f"D-alpha {chord} (location not recorded)",
                                x=panel.x,
                                y=y[None, :],
                                legend=[legend],
                                ylabel=unit,
                                metadata={
                                    **panel.metadata,
                                    "valid_fraction": [
                                        panel.metadata["valid_fraction"][c]
                                    ],
                                    "node": f"\\SPECTROSCOPY::{chord}",
                                    "location": "not recorded in corpus",
                                    "units_source": (
                                        "SPECTROSCOPY FS units; "
                                        "configs/shot_design/signals.yaml (dalpha)"
                                    ),
                                    "layout": "one calibrated chord per row",
                                },
                            )
                        )
                else:
                    if name == "co2":
                        panel.metadata.update(
                            {
                                "context_quantity": "density",
                                "measurement": "CO2 line-integrated density proxy",
                                "source_selection": "Corpus R0 chord",
                                "caveat": CO2_CAVEAT,
                            }
                        )
                    built.append(panel)
                have_density |= name == "co2"
        if not have_density and "ts_core_density" in source:
            panel = _context_panel(
                source["ts_core_density"],
                "Thomson core local density (not line-averaged)",
                [f"core channel {i}" for i in range(44)],
                "m^-3",
                1.0,
                t_range,
                positive=True,
                max_channels=8,
            )
            if panel is not None:
                panel.metadata.update(
                    {
                        "source": str(path),
                        "corpus_group": "ts_core_density",
                        "fallback_for": "CO2 density unavailable in plasma window",
                        "measurement": "local Thomson channels; not line-averaged",
                        "channel_selection": (
                            "up to eight channels ranked by positive native-sample "
                            "valid fraction; ties retain channel order"
                        ),
                    }
                )
                built.append(panel)
    return built


def _context_panel(
    group,
    title,
    names,
    unit,
    spacing,
    t_range,
    *,
    positive=False,
    positive_chords=False,
    max_channels=None,
    preplasma_baseline=False,
):
    """Reduce a local context trace, rejecting stubs and empty plasma windows."""
    if "xdata" not in group or "ydata" not in group:
        return None
    x = np.asarray(group["xdata"], dtype=float) * 1000
    data = group["ydata"]
    if (
        x.ndim != 1
        or len(x) < 2
        or data.ndim != 2
        or data.shape[1] != len(x)
        or not np.isfinite(x).all()
        or np.any(np.diff(x) <= 0)
    ):
        return None
    dt = float((x[-1] - x[0]) / (len(x) - 1))
    step = max(1, int(np.ceil(spacing / dt - 1e-6)))
    lo, hi = t_range if t_range is not None else (0.0, x[-1])
    start = int(np.searchsorted(x, lo))
    stop = int(np.searchsorted(x, hi, side="right"))
    ids = list(range(min(len(names), data.shape[0])))
    if stop <= start or not ids:
        return None
    bx, y, fraction = _block_means(data, x, ids, start, stop, step, positive=positive)
    offsets = np.full(len(ids), np.nan)
    if preplasma_baseline:
        before = int(np.searchsorted(x, 0.0))
        if before:
            # One channel at a time bounds memory even for long native clocks.
            for c, channel in enumerate(ids):
                native = np.asarray(data[channel, :before], dtype=float)
                finite = native[np.isfinite(native)]
                if finite.size:
                    offsets[c] = np.median(finite)
        y -= np.where(np.isfinite(offsets), offsets, 0)[:, None]
    live = np.isfinite(y).any(axis=1)
    if positive_chords:
        # Suppress empty/negative-offset chords, without asserting a calibrated
        # noise floor. The explicit policy describes this availability screen.
        live[live] &= np.nanmedian(y[live], axis=1) > 0
    if not live.any():
        return None
    selected = np.flatnonzero(live)
    if max_channels is not None:
        selected = selected[
            np.argsort(-fraction[selected], kind="stable")[:max_channels]
        ]
    metadata = {
        "reduction": "block mean",
        "samples_per_block": step,
        "block_ms": step * dt,
        "valid_fraction": fraction[selected].tolist(),
    }
    if positive:
        metadata["missing_policy"] = (
            "nonfinite and nonpositive native samples masked before block means"
        )
    if preplasma_baseline:
        metadata["preplasma_baseline"] = [
            float(offsets[c]) if np.isfinite(offsets[c]) else None for c in selected
        ]
        metadata["baseline_policy"] = (
            "Subtract each channel's finite native-sample median over t < 0; "
            "without pre-plasma samples the offset is uncorrected."
        )
    if positive_chords:
        metadata["channel_policy"] = (
            "retain finite chords with positive median block mean in the displayed "
            "window; availability screen, not a calibration or noise-floor test"
        )
    return Panel(
        title=title,
        x=bx,
        y=y[selected],
        legend=[
            f"{names[ids[c]]} ({unit})"
            + (
                " · offset uncorrected"
                if preplasma_baseline and not np.isfinite(offsets[c])
                else ""
            )
            for c in selected
        ],
        ylabel=unit,
        metadata=metadata,
    )
