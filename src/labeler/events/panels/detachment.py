"""Detachment's default rows (plan D.B) and the producer's indicators.

The default is Afrac lower, Prad,div lower and upper (\\BOLOM::PRAD_DIVL/U, in
MW), the gas total, the divertor Thomson Te the plasma control system receives
(PTDATA tssdivte00-05) and the lower outer divertor infrared heat flux map
(\\IRTVQ2DR2), in that order. Afrac upper is left out (owner, 2026-10-16); the
bolometer and D-alpha chords left the default (the picker reaches them).
`indicator_panels` still reads every producer indicator. Raw TPLANG sweeps are
omitted. This view never fetches or votes: scripts/labeler/detach_default_fetch.py
fetches.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ...config import Paths
from ..verify import Panel
from ._shared import plasma_window, with_caveat

#: The corpus gas_flow valves (modalities.yaml), each
#: \D3D::TOP.NEUTRALS.GASFLOW.<valve>:FLOW. DIII-D holds no calibrated total:
#: the NEUTRALS tree has no total leaf, and PTDATA GASTOTAL sums the raw valve
#: signals (volts), so the row sums these calibrated flows instead.
GAS_VALVES = (
    "GASA", "GASB", "GASC", "GASD", "GASE", "LOB1", "LOB2", "PFX1", "PFX2", "PFX3",
    "UOB",
)
GAS_TITLE = "Gas total (valve sum)"
#: The divertor radiated powers, (cache key, node, title). The producer cache
#: holds them for its roster; detach_default_fetch.py fetches the other shots.
PRAD_ROWS = (
    ("prad_divl", r"\BOLOM::PRAD_DIVL", "Prad,div lower"),
    ("prad_divu", r"\BOLOM::PRAD_DIVU", "Prad,div upper"),
)
PRAD_UNITS_NOTE = (
    "BOLOM labels these nodes MW but stores W (3.7e6 at the peak on 189061), as "
    "the producer reads them; the row divides by 1e6."
)
DTS_TITLE = "Divertor Te (DTS)"
DTS_POINTNAMES = tuple(f"tssdivte{k:02d}" for k in range(6))
#: Update instants closer than this merge (the later one stands): the channels of
#: one laser pulse can reach the control system a sample or two apart.
DTS_MERGE_MS = 1.0
#: A held value older than this many median update periods is a gap, not data.
DTS_HOLD_PERIODS = 2.0
DTS_CAVEAT = (
    "The plasma control system's real-time divertor Thomson Te (PTDATA "
    "tssdivte00-05; eV, though PTDATA labels them 'vo'; 0 is no measurement, "
    "drawn as a gap). Channel k is closest to blessed divertor chord k on 84 % "
    "of the 371 channel-shots checked (median difference 9 %); on the rest no "
    "chord comes within 19 %. A real-time estimate, not the blessed fit: compare with "
    "the blessed Te before reading a value. The chords move with the Thomson "
    "configuration (upper, lower or split), so read each line's Z."
)
IRTV_TITLE = "IRTV heat flux (W/cm²)"
#: The fetch's time blocks (detach_default_fetch.IRTV_BLOCK_MS), aligned to
#: multiples of it; the row puts each block at its centre.
IRTV_BLOCK_MS = 1.0
#: Radial bin of the heat-flux map; the pixels are about 0.2 cm apart.
IRTV_R_BIN_CM = 0.5
#: A radial bin whose pixels span more than this in Z lies on a vertical wall.
IRTV_WALL_SPAN_CM = 2.0
IRTV_CAVEAT = (
    "Infrared surface heat flux of the lower divertor seen by the 60-degree "
    "camera (IRTV analysis, W/cm²): 1 ms means of each pixel, then the pixel "
    "mean in 0.5 cm radial bins; the centre post's vertical wall (one radius, "
    "many heights) is left out. Colour from 0 to the window's 99.5th percentile."
)
CSV_INDICATORS = (
    ("afrac", "Afrac lower", "dimensionless"),
    ("prad_div", "f_div = Prad,div / P_in", "dimensionless"),
    ("tangtv_front_height", "TangTV front DZ", "dimensionless"),
)
BIN_INDICATORS = (
    ("afrac", "Afrac lower", "dimensionless"),
    ("prad", "f_div = Prad,div / P_in", "dimensionless"),
    ("tangtv", "TangTV front DZ", "dimensionless"),
)
AFRAC_CAVEAT = (
    "Lower outer strike point: an uncalibrated local Jsat ratio, not the "
    "published Eldon Afrac. Uncalibrated local proxies may use a within-shot "
    "reference. Check the producer recipe and per-bin method before "
    "interpreting Afrac; do not use this indicator alone."
)
AFRAC_UNGATED_CAVEAT = (
    "Bins the producer marks invalid show the UNGATED ratio as the second line: "
    "its Jsat / (ne^2 P_SOL^-3/7) with the bin gates and the minimum reference "
    "length removed, read on the probe nearest the separatrix at any distance; "
    "the method readout gives each bin's invalid reason. Context only: an "
    "ungated bin never votes."
)
CO2_CAVEAT = (
    "CO2 UF density input is a line integral without a chord-length division; "
    "BCI units are m·cm⁻³ (configs/shot_design/signals.yaml). "
    "It is upstream density context, not a "
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
    if not any(root.is_dir() for root in roots):
        raise FileNotFoundError(
            f"Detachment producer indicator directory missing: {roots[0]}. "
            "Set LABELER_DETACHMENT_INDICATORS to the producer bins directory."
        )
    return roots[0] / f"{int(shot)}.npz"


def density_cache_path(shot, paths):
    """Local producer cache (CO2, Prad,div); this adapter never fetches."""
    root = Path(
        os.environ.get(
            "LABELER_DETACHMENT_CACHE_ROOT", str(paths.root / "round4/detach/cache")
        )
    )
    if not root.is_dir():
        raise FileNotFoundError(
            f"Detachment producer density cache missing: {root}. "
            "Set LABELER_DETACHMENT_CACHE_ROOT to the producer cache."
        )
    return root / f"{int(shot)}.npz"


def default_inputs_path(shot, paths):
    """detach_default_fetch.py's Prad,div for shots the producer cache lacks."""
    return paths.root / "detach_default" / f"{int(shot)}.npz"


def dts_path(shot, paths):
    """detach_default_fetch.py's tssdivte00-05 and blessed chord positions."""
    return paths.root / "detach_default" / "dts" / f"{int(shot)}.npz"


def irtv_path(shot, paths):
    """detach_default_fetch.py's 1 ms block means of the IRTV heat flux map."""
    return paths.root / "detach_default" / "irtv" / f"{int(shot)}.npz"


def probes_path(shot, paths):
    """The producer's positioned Langmuir probes, which the ungated Afrac reads."""
    return paths.root / "round4/detach/processed_probes" / f"{int(shot)}.npz"


def flux_map(shot, paths):
    """`(maps, source)` for the probes' psiN: the producer's close EFIT map, else the
    default fetch's equilibrium grid; `(None, None)` without either."""
    path = paths.root / "round4/detach/efit" / f"{int(shot)}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as f:
            source = str(f["source"]) if "source" in f.files else "EFIT01"
            if len(f["gtime_ms"]) >= 2 and source in ("EFIT01", "EFIT02"):
                return {k: f[k] for k in f.files}, f"producer flux map ({source})"
    path = paths.root / "equilibria" / f"{int(shot)}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as f:
            need = {"t_ms", "r", "z", "psirz", "ssimag", "ssibry"}
            if need <= set(f.files) and len(f["t_ms"]) >= 2:
                maps = {k: f[k] for k in need - {"t_ms"}}
                maps["gtime_ms"] = np.asarray(f["t_ms"], dtype=float)
                tree = str(f["tree"]) if "tree" in f.files else "EFIT"
                return maps, f"default-fetch equilibrium ({tree})"
    return None, None


def _load_cache(path):
    """The producer cache as `{name: (t_ms, y)}`, as `signals.load_cache` reads it."""
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {
            key[:-3]: (data[key[:-3] + "__t"], data[key])
            for key in data.files
            if key.endswith("__y") and key[:-3] + "__t" in data.files
        }


def ungated_afrac(shot, paths, edges, data):
    """The producer's local Jsat ratio on its own bins with every gate removed.

    Returns `(value, method, note)`: the ratio per bin (NaN where none), a method
    readout per bin (None when nothing could be computed) and, when no bin has a
    value, the reason. As `afrac.afrac_indicator`: each probe's inter-ELM bin
    median Jsat over `<ne>^2 P_SOL^(-3/7)`, divided by that probe's attached
    reference, read on the probe nearest the separatrix. Removed: the power,
    ramp, ELM-share, L-mode and ELM-coverage gates, the flux window on the read
    probe and the minimum reference length. A probe's reference is the
    producer's own where one of its valid bins reports it (so the two lines
    agree), else the `AFRAC_REFERENCE_QUANTILE` of its ratio over its bins
    within `AFRAC_PSI_WINDOW` of the separatrix, else over all its bins. Without
    a D-alpha ELM mask no sample is dropped. psiN comes from the producer's flux
    map, else from the default fetch's equilibrium.
    """
    from ..detachment import core, signals
    from ..detachment import thresholds as th

    n = len(edges) - 1
    nothing = np.full(n, np.nan)
    path = probes_path(shot, paths)
    if not path.is_file():
        return nothing, None, "no positioned Langmuir probes (processed_probes)"
    cache = _load_cache(density_cache_path(shot, paths))
    density = signals.line_density(cache)
    if density is None:
        return nothing, None, "no CO2 line density in the producer cache"
    power = signals.heating_power(shot, cache)
    if power is None:
        return nothing, None, "no heating power (EFIT poh/wmhd, beams) in the cache"
    maps, flux_source = flux_map(shot, paths)
    if maps is None:
        return nothing, None, "no EFIT flux map for the probe positions"
    elm = signals.elm_mask(shot, cache)
    with np.load(path, allow_pickle=False) as f:
        keys = sorted(
            (k[:-5] for k in f.files if k.endswith("_jsat")), key=lambda k: int(k[1:])
        )
        if not keys:
            return nothing, None, "no positioned Langmuir probes (processed_probes)"
        jsat = np.vstack(
            [
                core.bin_median(
                    f[k + "_t_ms"],
                    f[k + "_jsat"],
                    edges,
                    keep=None if elm is None else ~core.elm_at(f[k + "_t_ms"], *elm),
                )[0]
                for k in keys
            ]
        )
        positions = np.array([f[k + "_rz"] for k in keys], dtype=float)
    numbers = np.array([int(k[1:]) for k in keys])
    psi = signals.flux_at_positions(maps, core.bin_centres(edges), positions)
    ne = core.bin_median(*density, edges)[0]
    p_sol = core.interp_at_centres(power[0], power[2], edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = jsat / (ne**2 * np.power(p_sol, -3.0 / 7.0))[None, :]
    raw[~np.isfinite(raw) | ~(jsat > 0)] = np.nan
    produced = {}
    if {"aux_jsat_selected_probe", "aux_jsat_reference", "afrac_valid"} <= set(data):
        for probe, reference, valid in zip(
            data["aux_jsat_selected_probe"],
            data["aux_jsat_reference"],
            data["afrac_valid"],
            strict=True,
        ):
            if valid and probe >= 0 and np.isfinite(reference) and reference > 0:
                produced[int(probe)] = float(reference)
    near = np.isfinite(psi) & (np.abs(psi - 1.0) <= th.AFRAC_PSI_WINDOW)
    references = np.full(len(keys), np.nan)
    kinds = ["no reference"] * len(keys)
    for i, number in enumerate(numbers):
        finite = np.isfinite(raw[i])
        if number in produced:
            references[i], kinds[i] = produced[number], "producer reference"
        elif (finite & near[i]).any():
            references[i] = np.quantile(
                raw[i, finite & near[i]], th.AFRAC_REFERENCE_QUANTILE
            )
            kinds[i] = "ungated near-separatrix reference"
        elif finite.any():
            references[i] = np.quantile(raw[i, finite], th.AFRAC_REFERENCE_QUANTILE)
            kinds[i] = "whole-shot reference (never near the separatrix)"
    with np.errstate(invalid="ignore", divide="ignore"):
        level = np.where(references[:, None] > 0, raw / references[:, None], np.nan)
    distance = np.where(np.isfinite(level) & np.isfinite(psi), np.abs(psi - 1.0), np.inf)
    which = distance.argmin(axis=0)
    ok = np.isfinite(distance.min(axis=0))
    value = np.where(ok, level[which, np.arange(n)], np.nan)
    method = [
        f"P{numbers[w]}, |psiN-1| {distance[w, i]:.3f}, {kinds[w]}, {flux_source}"
        + ("" if elm is not None else ", no ELM mask")
        if ok[i]
        else (
            "no probe flux at this bin"
            if not np.isfinite(psi[:, i]).any()
            else "no probe current, density or power in this bin"
        )
        for i, w in enumerate(which)
    ]
    if not ok.any():
        return value, method, "no bin has a probe current with a known flux"
    return value, method, None


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
            rows = y[None, keep]
            legend = [f"{title} ({unit})"]
            if indicator == "afrac":
                methods = np.asarray(
                    data.get("afrac_method", np.full(len(x), "")), dtype=str
                )
                if methods.shape != x.shape:
                    raise ValueError("afrac_method: shape disagrees with clock")
                caveat = AFRAC_CAVEAT
                if bins:
                    edges = np.r_[data["start_ms"], data["start_ms"][-1] + width]
                    try:
                        loose, readout, note = ungated_afrac(shot, paths, edges, data)
                    except (ValueError, KeyError, OSError) as error:
                        # The producer's own row still stands; say why the
                        # ungated line is missing instead of failing the store.
                        loose, readout = np.full(len(x), np.nan), None
                        note = f"ungated ratio unreadable ({error})"
                    loose = np.where(valid, np.nan, loose)
                    reasons = np.asarray(
                        data.get("afrac_reason", np.full(len(x), "")), dtype=str
                    )
                    if readout is None:
                        readout = [f"no value ({note})" if note else "no value"] * len(x)
                    methods = np.array(
                        [
                            m if v else f"invalid ({r or 'no reason'}); ungated: {u}"
                            for m, v, r, u in zip(
                                methods, valid, reasons, readout, strict=True
                            )
                        ]
                    )
                    rows = np.vstack([y[keep], loose[keep]])
                    legend.append("ungated, invalid bins")
                    metadata["ungated"] = np.isfinite(loose[keep]).tolist()
                    caveat = f"{AFRAC_CAVEAT} {AFRAC_UNGATED_CAVEAT}"
                    if not np.isfinite(rows).any():
                        why = note or "no valid or ungated bin in the plasma window"
                        caveat += f" No value on this shot: {why}."
                    elif note:
                        caveat += f" No ungated values: {note}."
                metadata["afrac_method"] = methods[keep].tolist()
                metadata.update(
                    {
                        "caveat": caveat,
                        "interpretation": "See the stored producer recipe and bin votes.",
                    }
                )
            if indicator == "tangtv":
                sources = np.asarray(data.get("tangtv_source", ["unknown"] * len(x)))
                if sources.shape != x.shape:
                    raise ValueError("tangtv_source: shape disagrees with clock")
                metadata["tangtv_source"] = sources[keep].astype(str).tolist()
                # Legacy CSVs omit provenance, so never infer an inversion from
                # their height column. The row's caveat names every source.
                metadata["caveat"] = (
                    "Normalized front height from "
                    + _tangtv_sources(metadata["tangtv_source"])
                )
            if keep.any() and (np.isfinite(rows).any() or "vote" in metadata):
                built.append(
                    Panel(
                        title=title,
                        x=x[keep],
                        y=rows,
                        ylabel=unit,
                        legend=legend,
                        hlines=hlines,
                        metadata=metadata,
                    )
                )
        if bins:
            for name, title, unit in (
                (
                    "aux_ne",
                    "CO2 density (aux_ne)",
                    "m·cm⁻³",
                ),
                (
                    "aux_te_div",
                    "Divertor Thomson Te",
                    "eV",
                ),
                (
                    "aux_jsat_peak",
                    "Langmuir Jsat peak",
                    "producer units (uncalibrated)",
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
                elif name == "aux_jsat_peak":
                    probes = np.asarray(
                        data.get("aux_jsat_probe", np.full(len(x), -1)), dtype=float
                    )
                    if probes.shape != x.shape:
                        raise ValueError("aux_jsat_probe: shape disagrees with clock")
                    metadata.update(
                        independent_check=True,
                        probe_id=[
                            int(p)
                            if np.isfinite(v) and np.isfinite(p) and p >= 0
                            else None
                            for v, p in zip(y[keep], probes[keep], strict=True)
                        ],
                        caveat=(
                            "Per-bin peak over available Langmuir probes; probe "
                            "identity can change. Uncalibrated producer units; "
                            "not necessarily the target/SOL-selected probe. "
                            "Probe switching alone cannot establish rollover. "
                            "Confirm target location and compare the same probe."
                        ),
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
        raise ValueError(
            f"shot {shot}: detachment indicators unreadable at {path}: {error}"
        ) from error


def _tangtv_sources(sources):
    """Name the front height's sources, including traces whose source changes by
    bin (a model reading is said to be one)."""
    have = set(sources)
    descriptions = []
    if "inversion" in have:
        descriptions.append("tomographic inversion")
    if "surrogate" in have:
        descriptions.append("surrogate regression (model estimate)")
    if have - {"inversion", "surrogate", "none"}:
        descriptions.append("source not recorded")
    return " + ".join(descriptions) or "no front-height source"


def _block_means(data, x, ids, start, stop, step):
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


def prad_panels(shot, paths, t_range=None):
    """Prad,div lower and upper in MW: the producer cache, else the default fetch.

    The radiated power itself, not the producer's f_div ratio (the owner named
    the nodes). A node a shot lacks leaves its row out. Without a plasma window
    the row starts at t = 0, as the gas total does: BOLOM's clock starts 5 s
    before the shot.
    """
    lo, hi = t_range if t_range is not None else (0.0, np.inf)
    built = []
    for key, node, title in PRAD_ROWS:
        for path in (density_cache_path(shot, paths), default_inputs_path(shot, paths)):
            if not path.is_file():
                continue
            with np.load(path, allow_pickle=False) as data:
                if not {f"{key}__t", f"{key}__y"} <= set(data.files):
                    continue
                x = np.asarray(data[f"{key}__t"], dtype=float)
                y = np.asarray(data[f"{key}__y"], dtype=float) / 1e6
            break
        else:
            continue
        if (
            x.ndim != 1
            or len(x) < 2
            or y.shape != x.shape
            or not np.isfinite(x).all()
            or np.any(np.diff(x) <= 0)
        ):
            raise ValueError(
                f"shot {shot}: {key} clock/value shapes must agree and increase "
                f"at {path}"
            )
        keep = (x >= lo) & (x <= hi)
        if not np.isfinite(y[keep]).any():
            continue
        built.append(
            Panel(
                title=title,
                x=x[keep],
                y=np.where(np.isfinite(y), y, np.nan)[None, keep],
                ylabel="MW",
                legend=[f"{node.split('::')[1]} (MW)"],
                metadata={
                    "source": str(path),
                    "node": node,
                    "quantity": key,
                    "units": "MW",
                    "units_note": PRAD_UNITS_NOTE,
                    "caveat": (
                        "Radiated power the bolometer analysis assigns to the "
                        f"{title.split()[-1]} divertor; not the f_div ratio."
                    ),
                },
            )
        )
    return built


def gas_total(source, path, t_range):
    """The calibrated valve flows summed, each less its own pre-plasma median."""
    if "gas_flow" not in source:
        return None
    panel = _context_panel(
        source["gas_flow"], GAS_TITLE, GAS_VALVES, "Torr L/s", 1.0, t_range
    )
    if panel is None:
        return None
    valves = [GAS_VALVES[c] for c in panel.metadata["channel_indices"]]
    # Every listed valve must be known in a block: a missing valve is not zero flow.
    total = np.where(np.isfinite(panel.y).all(axis=0), panel.y.sum(axis=0), np.nan)
    uncorrected = [
        valve
        for valve, offset in zip(valves, panel.metadata["preplasma_baseline"], strict=True)
        if offset is None
    ]
    note = (
        f"Sum of the calibrated valve flows {', '.join(valves)} (corpus gas_flow, "
        "NEUTRALS GASFLOW <valve>:FLOW); DIII-D records no calibrated total, and "
        "PTDATA GASTOTAL sums the raw valve volts. Each valve's pre-plasma median "
        "is subtracted before the sum"
    )
    if uncorrected:
        note += f"; {', '.join(uncorrected)} had no pre-plasma samples (uncorrected)"
    return with_caveat(
        Panel(
            title=GAS_TITLE,
            x=panel.x,
            y=total[None, :],
            legend=[
                "+".join(valves)
                + " (Torr L/s)"
                + (" · offset uncorrected" if uncorrected else "")
            ],
            ylabel="Torr L/s",
            metadata={
                **panel.metadata,
                "source": str(path),
                "corpus_group": "gas_flow",
                "valves": valves,
                "nodes": [rf"\D3D::TOP.NEUTRALS.GASFLOW.{v}:FLOW" for v in valves],
                "sum_policy": (
                    "per block, the sum over the listed valves; missing where "
                    "any of them has no finite block mean"
                ),
            },
        ),
        note,
    )


def dts_panel(shot, paths, t_range=None):
    """Divertor Te from tssdivte00-05 as steps between the control system's updates.

    The pointnames are sample-and-hold at 0.25 ms; the row keeps only the instants
    a value changes (one per laser pulse, about 20 ms apart), so it never refines
    the store grid. Each line is labelled with its blessed chord's Z when the
    fetch found the chords. None without the fetched file or a live channel.
    """
    path = dts_path(shot, paths)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        channels = {
            name: (
                np.asarray(data[f"{name}__t"], dtype=float),
                np.asarray(data[f"{name}__y"], dtype=float),
            )
            for name in DTS_POINTNAMES
            if {f"{name}__t", f"{name}__y"} <= set(data.files)
        }
        chords = {
            key: np.asarray(data[key], dtype=float)
            for key in ("chord_r_m", "chord_z_m")
            if key in data.files
        }
    for name, (t, y) in channels.items():
        if t.ndim != 1 or t.shape != y.shape or not np.isfinite(t).all():
            raise ValueError(f"shot {shot}: {name} clock/value shapes disagree at {path}")
        if np.any(np.diff(t) < 0):
            raise ValueError(f"shot {shot}: {name} clock decreases at {path}")
    if not channels:
        return None
    lo, hi = t_range if t_range is not None else (0.0, np.inf)
    x = np.unique(
        np.concatenate(
            [t[np.r_[True, np.diff(y) != 0]] for t, y in channels.values() if len(t)]
        )
    )
    x = x[(x >= lo) & (x <= hi)]
    x = x[np.r_[np.diff(x) >= DTS_MERGE_MS, True]] if len(x) else x
    if len(x) < 2:
        return None
    rows = []
    for t, y in channels.values():
        j = np.searchsorted(t, x, side="right") - 1
        value = np.where(j >= 0, y[np.maximum(j, 0)], np.nan)
        rows.append(np.where(np.isfinite(value) & (value > 0), value, np.nan))
    rows = np.asarray(rows)
    live = np.flatnonzero(np.isfinite(rows).any(axis=1))
    if not len(live):
        return None
    period = float(np.median(np.diff(x)))
    ends = np.minimum(np.r_[x[1:], x[-1] + period], x + DTS_HOLD_PERIODS * period)
    names = list(channels)
    k = [int(names[i][-2:]) for i in live]
    position = {
        key: [
            float(values[c]) if c < len(values) and np.isfinite(values[c]) else None
            for c in k
        ]
        for key, values in chords.items()
    }
    z = position.get("chord_z_m", [None] * len(k))
    return Panel(
        title=DTS_TITLE,
        x=(x + ends) / 2,
        y=rows[live],
        ylabel="eV",
        legend=[
            f"{c:02d} Z {zc:+.3f} m (eV)" if zc is not None else f"{c:02d} (eV)"
            for c, zc in zip(k, z, strict=True)
        ],
        metadata={
            "source": str(path),
            "pointnames": [names[i] for i in live],
            "units": "eV",
            "units_note": "PTDATA labels tssdivte 'vo' (the control system's "
            "default); the values are Te in eV",
            "chord": "blessed divertor chord k (\\ELECTRONS::TOP.TS.BLESSED."
            "DIVERTOR:R/Z) for channel k",
            "chord_r_m": position.get("chord_r_m"),
            "chord_z_m": position.get("chord_z_m"),
            "update_policy": (
                "steps from each update instant (a value changes; instants under "
                f"{DTS_MERGE_MS:g} ms apart merge) to the next, at most "
                f"{DTS_HOLD_PERIODS:g} median periods"
            ),
            "trace_style": "step",
            "bin_width_ms": period,
            "bin_start_ms": x.tolist(),
            "bin_end_ms": ends.tolist(),
            "caveat": DTS_CAVEAT,
        },
    )


def irtv_panel(shot, paths, t_range=None):
    """The IRTV heat flux as an R x time map on uniform radial bins.

    The fetch keeps 1 ms means of every pixel; here each `IRTV_R_BIN_CM` bin is
    the mean of its pixels, and a bin whose pixels span more than
    `IRTV_WALL_SPAN_CM` in Z (the centre post, seen edge-on in R) is left out.
    None without the fetched file or two blocks in the window.
    """
    path = irtv_path(shot, paths)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        flux = np.asarray(data["heat_flux"], dtype=np.float32)
        t = np.asarray(data["t_ms"], dtype=float)
        r = np.asarray(data["r_cm"], dtype=float)
        z = np.asarray(data["z_cm"], dtype=float)
        about = {k: str(data[k]) for k in ("node", "system", "units") if k in data.files}
    if (
        t.ndim != 1
        or flux.shape != (len(t), len(r))
        or z.shape != r.shape
        or not np.isfinite(t).all()
        or np.any(np.diff(t) <= 0)
    ):
        raise ValueError(f"shot {shot}: IRTV map, clock and pixels disagree at {path}")
    lo, hi = t_range if t_range is not None else (0.0, np.inf)
    keep = (t >= lo) & (t <= hi)
    placed = np.isfinite(r) & np.isfinite(z)
    if keep.sum() < 2 or not placed.any():
        return None
    t, flux = t[keep], flux[keep][:, placed]
    # The fetch stores each block's mean sample time; its centre keeps the map on
    # the same 1 ms lattice as the other rows, so it never refines the grid.
    t = (np.floor(t / IRTV_BLOCK_MS) + 0.5) * IRTV_BLOCK_MS
    if np.any(np.diff(t) <= 0):
        raise ValueError(f"shot {shot}: two IRTV samples share a block at {path}")
    r, z = r[placed], z[placed]
    r0 = np.floor(r.min() / IRTV_R_BIN_CM) * IRTV_R_BIN_CM
    n = max(1, int(np.ceil((r.max() - r0) / IRTV_R_BIN_CM + 1e-9)))
    which = np.clip(((r - r0) / IRTV_R_BIN_CM).astype(int), 0, n - 1)
    onehot = np.zeros((len(r), n), dtype=np.float32)
    onehot[np.arange(len(r)), which] = 1
    finite = np.isfinite(flux)
    sums = np.where(finite, flux, 0) @ onehot
    counts = finite.astype(np.float32) @ onehot
    mean = np.full(sums.shape, np.nan, dtype=np.float32)
    np.divide(sums, counts, out=mean, where=counts > 0)
    wall = []
    for b in np.unique(which):
        heights = z[which == b]
        if heights.max() - heights.min() > IRTV_WALL_SPAN_CM:
            mean[:, b] = np.nan
            wall.append(
                {
                    "r_cm": [round(r0 + b * IRTV_R_BIN_CM, 2),
                             round(r0 + (b + 1) * IRTV_R_BIN_CM, 2)],
                    "z_cm": [round(float(heights.min()), 1),
                             round(float(heights.max()), 1)],
                    "pixels": int((which == b).sum()),
                }
            )
    if not np.isfinite(mean).any():
        return None
    return Panel(
        title=IRTV_TITLE,
        kind="heatmap",
        x=t,
        y=r0 + (np.arange(n) + 0.5) * IRTV_R_BIN_CM,
        z=mean.T,
        ylabel="R (cm)",
        zmin=0.0,
        metadata={
            "source": str(path),
            **about,
            "view": "LODIV_60RP2 (lower divertor, 60-degree camera)",
            "quantity": "surface heat flux",
            "units": "W/cm²",
            "time_block_ms": IRTV_BLOCK_MS,
            "r_bin_cm": IRTV_R_BIN_CM,
            "wall_left_out": wall,
            "caveat": IRTV_CAVEAT,
        },
    )


def afrac_placeholder(shot, paths, span, window=True):
    """An empty "Afrac lower" row over `span` whose note says why the shot has no
    value; [] without a span."""
    if span is None:
        return []
    path = indicator_path(shot, paths)
    why = (
        "the producer has no bins for this shot (not in its roster)"
        if not path.is_file()
        else "the producer bins hold no Afrac"
        + (" in the plasma window" if window else "")
    )
    return [
        Panel(
            title="Afrac lower",
            x=np.asarray(span, dtype=float),
            y=np.full((1, 2), np.nan),
            ylabel="dimensionless",
            legend=["Afrac lower (dimensionless)"],
            metadata={
                "source": str(path),
                "indicator": "afrac",
                "caveat": f"{AFRAC_CAVEAT} No value on this shot: {why}.",
            },
        )
    ]


def panels(shot, *, t_range=None, paths=None):
    """Plan D.B's default rows over the catalog plasma window, when available."""
    paths = Paths.from_env() if paths is None else paths
    if t_range is None:
        t_range = plasma_window(int(shot), paths)
    afrac = [
        p
        for p in indicator_panels(shot, paths, t_range)
        if p.metadata.get("indicator") == "afrac"
    ]
    built = prad_panels(shot, paths, t_range)
    path = paths.corpus_file(int(shot))
    if path.is_file():
        with h5py.File(path, "r") as source:
            panel = gas_total(source, path, t_range)
        if panel is not None:
            built.append(panel)
    for panel in (dts_panel(shot, paths, t_range), irtv_panel(shot, paths, t_range)):
        if panel is not None:
            built.append(panel)
    if not afrac:
        # Without a plasma window the empty row spans the other rows.
        span = t_range
        if span is None and built:
            span = (min(float(p.x[0]) for p in built), max(float(p.x[-1]) for p in built))
        afrac = afrac_placeholder(shot, paths, span, window=t_range is not None)
    return afrac + built


def _context_panel(group, title, names, unit, spacing, t_range):
    """Block means of a corpus group, each channel less its pre-plasma median.

    Rejects stubs and empty plasma windows.
    """
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
    bx, y, fraction = _block_means(data, x, ids, start, stop, step)
    offsets = np.full(len(ids), np.nan)
    before = int(np.searchsorted(x, 0.0))
    if before:
        # One channel at a time bounds memory even for long native clocks.
        for c, channel in enumerate(ids):
            native = np.asarray(data[channel, :before], dtype=float)
            finite = native[np.isfinite(native)]
            if finite.size:
                offsets[c] = np.median(finite)
    y -= np.where(np.isfinite(offsets), offsets, 0)[:, None]
    selected = np.flatnonzero(np.isfinite(y).any(axis=1))
    if not len(selected):
        return None
    return Panel(
        title=title,
        x=bx,
        y=y[selected],
        legend=[
            f"{names[ids[c]]} ({unit})"
            + (" · offset uncorrected" if not np.isfinite(offsets[c]) else "")
            for c in selected
        ],
        ylabel=unit,
        metadata={
            "reduction": "block mean",
            "channel_indices": [ids[c] for c in selected],
            "samples_per_block": step,
            "block_ms": step * dt,
            "valid_fraction": fraction[selected].tolist(),
            "preplasma_baseline": [
                float(offsets[c]) if np.isfinite(offsets[c]) else None
                for c in selected
            ],
            "baseline_policy": (
                "Subtract each channel's finite native-sample median over t < 0; "
                "without pre-plasma samples the offset is uncorrected."
            ),
        },
    )
