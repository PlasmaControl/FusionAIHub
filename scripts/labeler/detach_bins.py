#!/usr/bin/env python
"""Run the three detachment indicators over shots and park their 50 ms bins.

    python scripts/labeler/detach_bins.py --shots-file shots.txt --workers 6

Reads the corpus, the fetch cache (`detach_fetch.py`), the packed TangTV
inversions (`detach_inversions.py`) and, where a shot has no inversion, the front
heights regressed from its raw frames (`detach_tv_surrogate.py`); never fetches.
One file per shot, `$LABELER_ROOT/round4/detach/bins/<shot>.npz`, holds on the
shot's 50 ms grid (`start_ms`, the bin starts): for each indicator `<name>_value`,
`<name>_valid`, `<name>_reason`, `<name>_vote`, and `tangtv_source` (`inversion`,
`surrogate` or `none`); plus the quantities behind them (`aux_*`) and the
geometry and evidence provenance. Unfiltered divertor Thomson is not exported.

The grid is the discharge only: bins where EFIT's plasma current is under
`MIN_IP_A` are dropped. A shot with no fetch cache is skipped and logged.
"""

from __future__ import annotations

import argparse
import functools
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

from labeler.events.detachment import afrac, core, langmuir, prad, signals, tangtv
from labeler.events.detachment import thresholds as th

#: Bins with less plasma current than this are not part of the discharge.
MIN_IP_A = 0.3e6
INDICATORS = ("afrac", "prad", "tangtv")


def root() -> Path:
    return Path(os.environ["LABELER_ROOT"]) / "round4" / "detach"


def geometry_aux(geo, edges):
    """Finite physical EFIT bin medians; sentinels never describe a camera leg."""
    out = {}
    bounds = {
        "rvsod": (0.8, 2.5),
        "rxpt1": (0.8, 2.5),
        "zvsod": (-1.6, -0.9),
        "zxpt1": (-1.6, 1.6),
    }
    for key, (lo, hi) in bounds.items():
        if key not in geo:
            out[f"aux_{key}"] = np.full(len(edges) - 1, np.nan, np.float32)
            continue
        t, y = geo[key]
        keep = tangtv._is_real(y, lo, hi)
        out[f"aux_{key}"] = core.bin_median(t, y, edges, keep=keep)[0].astype(
            np.float32
        )
    return out


def add_envelope(out):
    record = (
        Path(__file__).resolve().parents[2]
        / "docs/labeler/results/detachment_tangtv_surrogate.json"
    )
    domain = json.loads(record.read_text())["domain"]
    envelope = np.ones(len(out["start_ms"]), bool)
    for key, values in (
        ("leg", out["aux_zxpt1"] - out["aux_zvsod"]),
        ("rv", out["aux_rvsod"]),
        ("rx", out["aux_rxpt1"]),
    ):
        lo, hi = domain[key]
        envelope &= (values >= lo) & (values <= hi)
    out["tangtv_in_envelope"] = envelope


def frame_geometry_aux(shot, edges, geo, accepted):
    """Medians of the exact EFIT slices belonging to accepted camera frames."""
    inv = load_inversion(shot)
    rec = inv if inv is not None else load_surrogate(shot)
    if rec is None or accepted is None or not accepted.any():
        return {}
    ft = rec["times_ms"]
    et = geo["rxpt1"][0]
    near = np.abs(et[None, :] - ft[:, None]).argmin(axis=1)
    out = {}
    for key in ("rvsod", "zvsod", "rxpt1", "zxpt1"):
        value, count = core.bin_median(ft, geo[key][1][near], edges, keep=accepted)
        out[f"aux_{key}"] = (value.astype(np.float32), count > 0)
    return out


def refresh_geometry(directory, width, shots):
    """Refresh auxiliary medians/envelope from cached EFIT, leaving votes intact."""
    for path in sorted(directory.glob("*.npz")):
        if int(path.stem) not in shots:
            continue
        with np.load(path) as f:
            out = {k: f[k] for k in f.files}
        edges = np.r_[out["start_ms"], out["start_ms"][-1] + width]
        shot = int(path.stem)
        cache = signals.load_cache(shot)
        geo, _ = signals.tangtv_geometry(shot, cache)
        out.update(geometry_aux(geo, edges))
        if out["tangtv_valid"].any():
            ind, _, accepted = tangtv_for(
                shot, edges, cache, signals.elm_mask(shot), with_frame_mask=True
            )
            assert np.array_equal(ind.valid, out["tangtv_valid"]), shot
            for key, (value, keep) in frame_geometry_aux(
                shot, edges, geo, accepted
            ).items():
                out[key][keep] = value[keep]
        add_envelope(out)
        tmp = path.with_name(f".{path.name}.tmp.npz")
        np.savez_compressed(tmp, **out)
        tmp.replace(path)


def load_inversion(shot: int):
    path = root() / "inversions" / f"{shot}.npz"
    if not path.is_file():
        return None
    with np.load(path) as npz:
        return {k: npz[k] for k in ("frames", "times_ms", "radii", "elevation")}


def load_surrogate(shot: int):
    """Front heights regressed from the raw frames (`detach_tv_surrogate.py`), or None."""
    path = root() / "tv_surrogate" / f"{shot}.npz"
    if not path.is_file():
        return None
    with np.load(path) as npz:
        return {k: npz[k] for k in npz.files}


def tangtv_for(shot, edges, cache, elm=None, *, with_frame_mask=False):
    """The TangTV indicator and where its front height came from.

    The inversion when the shot has one, else the regression from the raw frames,
    else an all-invalid indicator saying why. The source is `inversion`, `surrogate`
    or `none`.
    """
    n = len(edges) - 1
    cache, _efit_source = signals.tangtv_geometry(shot, cache)

    def invalid(why):
        return core.assemble(
            "tangtv",
            np.full(n, np.nan),
            np.zeros(n, bool),
            np.full(n, why, dtype=object),
            np.zeros(n),
        )

    inversion = load_inversion(shot)
    surrogate = None if inversion is not None else load_surrogate(shot)
    if inversion is None and surrogate is None:
        result = (invalid("no_inversion"), "none")
        return (*result, None) if with_frame_mask else result
    need = ("rvsod", "zvsod", "rxpt1", "zxpt1")
    if any(k not in cache for k in need):
        result = (invalid("efit_missing"), "none")
        return (*result, None) if with_frame_mask else result
    et = cache["rxpt1"][0]
    if inversion is not None:
        ft = inversion["times_ms"].astype(float)
        near = np.abs(et[None, :] - ft[:, None]).argmin(axis=1)
        rv = np.asarray(cache["rvsod"][1])[near]
        zv = np.asarray(cache["zvsod"][1])[near]
        lower = (rv < th.SHELF_WALL_R) & (zv < -1.30)
        ze = tangtv.outer_leg_ze(
            inversion["frames"].astype(np.float32),
            inversion["radii"],
            inversion["elevation"],
            np.asarray(cache["rxpt1"][1])[near],
        )
        if lower.any():
            lower_ze = tangtv.outer_leg_ze(
                inversion["frames"],
                inversion["radii"],
                inversion["elevation"],
                np.asarray(cache["rxpt1"][1])[near],
                r_max=th.SHELF_WALL_R,
            )
            ze = np.where(lower, lower_ze, ze)
        source = "inversion"
    else:
        ft = surrogate["times_ms"].astype(float)
        ze = surrogate["ze"].astype(float)
        source = "surrogate"
        lower = False
    indicator, accepted = tangtv.tangtv_indicator(
        edges,
        ft,
        ze,
        et,
        cache["rvsod"][1],
        cache["zvsod"][1],
        cache["rxpt1"][1],
        cache["zxpt1"][1],
        lower_shelf=lower,
        frame_valid=None
        if source == "inversion"
        else surrogate.get("valid", np.zeros(len(ft), bool)),
        elm_t_ms=None if elm is None else elm[0],
        elm_flag=None if elm is None else elm[1],
        return_frame_mask=True,
    )
    indicator = tangtv.void_lower_shelf(
        indicator, shelf_tier_of(edges, ft, cache, accepted)
    )
    result = (indicator, source)
    return (*result, accepted) if with_frame_mask else result


def shelf_tier_of(edges, frame_t_ms, geo, accepted):
    """The per-bin shelf geometry tier of the accepted frames (`tangtv.shelf_tier`)."""
    et = geo["rxpt1"][0]
    near = np.abs(et[None, :] - frame_t_ms[:, None]).argmin(axis=1)
    return tangtv.shelf_tier(
        edges,
        frame_t_ms,
        geo["rvsod"][1][near],
        geo["zvsod"][1][near],
        accepted,
    )


def greenwald_cue(edges, cache):
    """Conservative fG>=0.8 cue from unit-confirmed V2 and elliptical path."""
    fraction = signals.greenwald_fraction(edges, cache)
    return np.isfinite(fraction) & (fraction >= th.GREENWALD_CUE_MIN), fraction


def spatial_evidence(shot, edges, geo, frame_quality=None):
    """Global inversion peak psiN<1, within 30 cm of X and near/above ZX."""
    from scipy.interpolate import RegularGridInterpolator

    n = len(edges) - 1
    result = np.zeros(n, bool)
    inv = load_inversion(shot)
    maps = signals.load_flux_map(shot)
    if inv is None or maps is None or frame_quality is None:
        return result
    if maps is not None:
        f = maps
        ft = inv["times_ms"]
        flags = np.zeros(len(ft), bool)
        for j, t in enumerate(ft):
            if not frame_quality[j]:
                continue
            k = np.argmin(np.abs(f["gtime_ms"] - t))
            if abs(f["gtime_ms"][k] - t) > 40:
                continue
            if not np.isfinite(inv["frames"][j]).any():
                continue
            iz, ir = np.unravel_index(
                np.nanargmax(inv["frames"][j]), inv["frames"][j].shape
            )
            r, z = inv["radii"][ir], inv["elevation"][iz]
            psin = (f["psirz"][k] - f["ssimag"][k]) / (f["ssibry"][k] - f["ssimag"][k])
            psi = RegularGridInterpolator(
                (f["z"], f["r"]), psin, bounds_error=False, fill_value=np.nan
            )((z, r))
            rx = f["rxpt1"][k] if "rxpt1" in f else np.interp(t, *geo["rxpt1"])
            zx = f["zxpt1"][k] if "zxpt1" in f else np.interp(t, *geo["zxpt1"])
            flags[j] = bool(
                0 < psi < 1 and zx - 0.02 <= z <= zx + 0.30 and abs(r - rx) <= 0.30
            )
        value, count = core.bin_median(
            ft, flags.astype(float), edges, keep=frame_quality
        )
        result = (count > 0) & (value >= 0.5)
    return result


def confinement(shot, edges):
    """Project's explicit Jalal Butt confinement labels; unknown stays 0."""
    import pandas as pd

    from labeler.config import STORE_EVENTS, Paths

    path = (
        Paths.from_env().label_tables
        / STORE_EVENTS["confinement"]
        / "raw/Jalal_28042024_confinement_regime_shotlist.csv"
    )
    times = core.bin_centres(edges)
    mode = np.zeros(len(times), int)
    back = np.zeros(len(times), bool)
    if not path.is_file():
        return mode, back
    rows = pd.read_csv(path).query("Shot == @shot")
    for row in rows.to_dict("records"):
        hit = (times >= row["Confinement Start Time (ms)"]) & (
            times < row["Confinement Stop Time (ms)"]
        )
        if row.get("L") == 1:
            mode[hit] = 2
        elif any(row.get(k) == 1 for k in ("H", "QH", "WP")):
            mode[hit] = 1
        transition = row.get("HL back-transition", np.nan)
        if np.isfinite(transition):
            back |= (times >= transition) & (times < transition + 200)
    return mode, back


def processed_ratio(
    shot, edges, cache, elm, lmode=None, afrac_window=th.AFRAC_PSI_WINDOW
):
    """Local Jsat proxy from the positioned processed probes; no camera-dependent fit.

    Each probe is referenced to its own near-separatrix attached level and the
    probe nearest the separatrix in flux is read (`afrac.afrac_indicator`; the
    window is `AFRAC_PSI_WINDOW`). The probes and the outer strike point use the
    same close EFIT map; the strike point is only provenance, and an EFIT sentinel
    is never exported as one (`tangtv.real_strike`). `lmode` (True where the shot is
    known to be in L-mode) abstains the bin and keeps it out of every reference.
    Provenance (the reported probe's position, flux, distance and margin) is
    exported for every bin, valid or not: an invalid bin reports the probe nearest
    the separatrix and the reason it could not vote.
    """
    n = len(edges) - 1
    modes = np.full(n, "per_probe_reference", dtype="U32")
    provenance = {
        "aux_jsat_selected_probe": np.full(n, -1, np.int16),
        "afrac_efit_source": np.full(n, "none", dtype="U16"),
        "afrac_probe_position_valid": np.zeros(n, bool),
        "afrac_probe_n_eligible": np.zeros(n, np.int16),
        **{
            f"aux_jsat_{name}": np.full(n, np.nan, np.float32)
            for name in (
                "selected_r_m",
                "selected_z_m",
                "selected_psin",
                "strike_r_m",
                "strike_z_m",
                "selected_distance_m",
                "radial_margin_m",
                "reference",
            )
        },
    }

    def invalid(reason):
        why = np.full(n, reason, object)
        why[~core.elm_bin_known(edges, *(elm or (None, None)))] = "elm_unknown"
        return (
            core.assemble(
                "afrac",
                np.full(n, np.nan),
                np.zeros(n, bool),
                why,
                np.full(n, core.ABSTAIN),
            ),
            modes,
            provenance,
        )

    path = root() / "processed_probes" / f"{shot}.npz"
    if not path.exists():
        return invalid("no_positioned_probes")
    with np.load(path) as f:
        keys = sorted(
            [k[:-5] for k in f.files if k.endswith("_jsat")],
            key=lambda k: int(k[1:]),
        )
        if not keys:
            return invalid("no_positioned_probes")
        per_probe = []
        for key in keys:
            t, y = f[key + "_t_ms"], f[key + "_jsat"]
            flags = core.elm_at(t, *(elm or (None, None)))
            per_probe.append(core.bin_median(t, y, edges, keep=~flags)[0])
        jsat = np.vstack(per_probe)
        positions = np.array([f[k + "_rz"] for k in keys])
    maps = signals.load_flux_map(shot)
    if maps is None or any(k not in maps for k in ("rvsod", "zvsod")):
        return invalid("probe_flux_unknown")
    centres = core.bin_centres(edges)
    near = np.abs(maps["gtime_ms"][:, None] - centres[None, :]).argmin(axis=0)
    close = np.abs(maps["gtime_ms"][near] - centres) <= 40
    strike = np.stack([maps[k][near] for k in ("rvsod", "zvsod")], axis=1)
    strike[~close | ~tangtv.real_strike(strike[:, 0], strike[:, 1])] = np.nan
    psi_n = signals.flux_at_positions(maps, centres, positions)
    density = signals.line_density(cache)
    power = signals.heating_power(shot, cache)
    if density is None or power is None:
        return invalid("no_density" if density is None else "no_power")
    indicator, which, references = afrac.afrac_indicator(
        edges,
        jsat,
        psi_n,
        *density,
        power[0],
        power[2],
        *cache["ipmeas"],
        *(elm or (None, None)),
        lmode,
        window=afrac_window,
    )
    usable = indicator.valid
    column = np.arange(n)
    reported = afrac.reported_probe(psi_n, which, usable)
    index = np.maximum(reported, 0)
    numbers = np.array([int(k[1:]) for k in keys])
    provenance["afrac_efit_source"][:] = str(maps.get("source", "EFIT01"))
    provenance["afrac_probe_position_valid"] = usable
    provenance["afrac_probe_n_eligible"] = (
        (np.isfinite(psi_n) & (np.abs(psi_n - 1.0) <= afrac_window))
        .sum(axis=0)
        .astype(np.int16)
    )
    provenance["aux_jsat_strike_r_m"] = strike[:, 0].astype(np.float32)
    provenance["aux_jsat_strike_z_m"] = strike[:, 1].astype(np.float32)
    have = reported >= 0
    provenance["aux_jsat_selected_probe"][have] = numbers[reported[have]]
    reported_positions = np.where(have[:, None], positions[index], np.nan)
    provenance["aux_jsat_selected_r_m"] = reported_positions[:, 0].astype(np.float32)
    provenance["aux_jsat_selected_z_m"] = reported_positions[:, 1].astype(np.float32)
    provenance["aux_jsat_selected_psin"] = np.where(
        have, psi_n[index, column], np.nan
    ).astype(np.float32)
    provenance["aux_jsat_selected_distance_m"] = np.linalg.norm(
        reported_positions - strike, axis=1
    ).astype(np.float32)
    provenance["aux_jsat_radial_margin_m"] = (
        reported_positions[:, 0] - strike[:, 0]
    ).astype(np.float32)
    provenance["aux_jsat_reference"] = np.where(
        usable, references[np.maximum(which, 0)], np.nan
    ).astype(np.float32)
    return indicator, modes, provenance


def process(
    shot: int,
    width_ms: float = core.BIN_MS,
    out_dir: Path | None = None,
    *,
    raw_probe_diagnostics: bool = True,
    afrac_window: float = th.AFRAC_PSI_WINDOW,
) -> dict:
    """Compute and save one shot's bins; return a one-line status.

    `afrac_window` is the Afrac flux window (`AFRAC_PSI_WINDOW`); other values are
    for the window sensitivity record (`detach_afrac_check.py`) only.
    """
    cache = signals.load_cache(shot)
    if "ipmeas" not in cache:
        return {"shot": shot, "status": "no_cache"}
    t_ip, ip = cache["ipmeas"]
    live = np.abs(ip) >= MIN_IP_A
    if not live.any():
        return {"shot": shot, "status": "no_plasma"}
    edges = core.bin_edges(t_ip[live][0], t_ip[live][-1], width_ms)
    starts = edges[:-1]
    n = len(starts)

    power = signals.heating_power(shot, cache)
    p_t, p_in, _ = (None, None, None) if power is None else power
    elm = signals.elm_mask(shot, cache)
    elm_t, elm_flag = (None, None) if elm is None else elm
    density = signals.line_density(cache)
    n_t, n_y = (None, None) if density is None else density
    # Unpositioned raw sweeps supply diagnostics only (`aux_jsat_peak`); the Afrac
    # vote always comes from the positioned processed probes (`processed_ratio`).
    probes = langmuir.read_shot(shot) if raw_probe_diagnostics else None

    prad_ind = prad.prad_indicator(
        edges,
        *cache.get("prad_divl", (None, None)),
        p_t,
        p_in,
        elm_t,
        elm_flag,
    )
    # The exported f_div vote is the per-shot relative one; the absolute vote (the
    # shot-201081-anchored global cutoffs) stays beside it as a sensitivity.
    centres = core.bin_centres(edges)
    p_in_window = (
        None
        if p_t is None
        else signals.window_mean(p_t, p_in, centres, th.PRAD_AVERAGING_MS)
    )
    ratio = prad.relative_fdiv(prad_ind.value, prad_ind.valid, p_in_window)
    prad_abs_vote, prad_abs_valid = prad_ind.vote.copy(), prad_ind.valid.copy()
    prad_ind = prad.with_relative_vote(prad_ind, ratio)
    regime, regime_source = signals.regime(shot, centres)
    tangtv_ind, tangtv_source, frame_quality = tangtv_for(
        shot, edges, cache, elm, with_frame_mask=True
    )
    geo, geo_source = signals.tangtv_geometry(shot, cache)
    # Spatial evidence exists only for true inversions with a close flux map.
    spatial = spatial_evidence(shot, edges, geo, frame_quality)
    # The MARFE density cue is the Greenwald fraction alone. The confinement
    # table's H-L back-transition is recorded but is not a cue: it can also follow
    # a detachment that is not a MARFE.
    second, fg = greenwald_cue(edges, cache)
    _, back_transition = confinement(shot, edges)
    afrac_ind, afrac_mode, probe_provenance = processed_ratio(
        shot, edges, cache, elm, regime == "L", afrac_window
    )
    vote, candidate = tangtv.evidence_votes(
        tangtv_ind.value, tangtv_ind.valid, spatial, second
    )
    tangtv_ind = core.assemble(
        "tangtv", tangtv_ind.value, tangtv_ind.valid, tangtv_ind.reason, vote
    )

    out: dict[str, np.ndarray] = {"start_ms": starts}
    for ind in (afrac_ind, prad_ind, tangtv_ind):
        out[f"{ind.name}_value"] = ind.value.astype(np.float32)
        out[f"{ind.name}_valid"] = ind.valid
        out[f"{ind.name}_reason"] = ind.reason.astype(str)
        out[f"{ind.name}_vote"] = ind.vote
    out["afrac_method"] = afrac_mode
    out["regime"] = regime
    out["regime_source"] = np.full(n, regime_source)
    out["prad_abs_valid"] = prad_abs_valid
    out["prad_abs_vote"] = prad_abs_vote.astype(np.int8)
    out.update(probe_provenance)
    out["tangtv_source"] = np.full(n, tangtv_source)
    out["tangtv_efit_source"] = np.full(n, geo_source)
    out["tangtv_marfe_candidate"] = candidate
    out["tangtv_marfe_spatial"] = spatial
    out["tangtv_marfe_second_cue"] = second
    out["tangtv_marfe_back_transition"] = back_transition
    maps = signals.load_flux_map(shot)
    out["tangtv_marfe_efit_source"] = np.full(
        n, "none" if maps is None else str(maps.get("source", "EFIT01"))
    )
    out["aux_greenwald_fraction"] = fg
    out["greenwald_source"] = np.full(
        n,
        "BCI_DENV2_unit_confirmed_ellipse" if "density_v2_si" in cache else "none",
    )
    # the quantities behind the indicators, for the figure and the failure analysis
    out["aux_ip_a"] = core.bin_median(t_ip, ip, edges)[0].astype(np.float32)
    if p_in_window is not None:
        out["aux_p_in_w"] = p_in_window.astype(np.float32)
    out["prad_rel_value"] = ratio.astype(np.float32)
    for name in ("prad_divl", "prad_tot"):
        out[f"aux_{name}_w"] = np.full(n, np.nan, np.float32)
        if name in cache:
            rt, ry = cache[name]
            out[f"aux_{name}_w"] = signals.window_mean(
                rt,
                ry,
                core.bin_centres(edges),
                th.PRAD_AVERAGING_MS,
                keep=~core.elm_at(rt, elm_t, elm_flag),
            ).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        div, total = out["aux_prad_divl_w"], out["aux_prad_tot_w"]
        out["aux_prad_div_fraction_total"] = np.where(
            (total > 0) & (div >= -th.RADIATION_NEGATIVE_TOL_W),
            np.maximum(div, 0.0) / total,
            np.nan,
        )
    out["aux_prad_divl_native_w"] = np.full(n, np.nan, np.float32)
    if "prad_divl" in cache:
        rt, ry = cache["prad_divl"]
        out["aux_prad_divl_native_w"] = core.bin_mean(
            rt, ry, edges, keep=~core.elm_at(rt, elm_t, elm_flag)
        )[0].astype(np.float32)
    out["prad_averaging_ms"] = np.full(n, th.PRAD_AVERAGING_MS, np.float32)
    out["aux_prad_elm_window_known"] = prad.elm_window_known(edges, elm_t, elm_flag)
    if n_t is not None:
        out["aux_ne"] = core.bin_median(n_t, n_y, edges)[0].astype(np.float32)
    if probes is not None:
        peak, which = afrac.peak_jsat(edges, probes["t_ms"], probes["jsat"])
        out["aux_jsat_peak"] = peak.astype(np.float32)
        out["aux_jsat_probe"] = probes["probe"][which].astype(np.int16)
    out["aux_elm_known"] = core.elm_bin_known(edges, elm_t, elm_flag)
    out["aux_elm_share"] = np.full(n, np.nan, np.float32)
    if elm_t is not None:
        share = core.bin_fraction(elm_t, elm_flag, edges)
        out["aux_elm_share"][out["aux_elm_known"]] = share[out["aux_elm_known"]]
    # Unlocalised real-time DTS is deliberately excluded from all claims.
    out.update(geometry_aux(geo, edges))
    for key, (value, keep) in frame_geometry_aux(
        shot, edges, geo, frame_quality
    ).items():
        out[key][keep] = value[keep]
    rec = load_inversion(shot) if tangtv_source == "inversion" else load_surrogate(shot)
    out["tangtv_tier"] = np.full(n, "none", dtype="U24")
    if rec is not None and frame_quality is not None:
        out["tangtv_tier"] = shelf_tier_of(edges, rec["times_ms"], geo, frame_quality)
    add_envelope(out)
    target = (out_dir or root() / "bins") / f"{shot}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(target)
    return {
        "shot": shot,
        "status": "ok",
        "bins": n,
        "n_probes": 0 if probes is None else len(probes["probe"]),
        **{
            f"valid_{i.name}": int(i.valid.sum())
            for i in (afrac_ind, prad_ind, tangtv_ind)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shots-file", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--redo", action="store_true", help="recompute existing bins")
    parser.add_argument(
        "--refresh-geometry",
        action="store_true",
        help="only refresh cached auxiliary geometry and envelope; keep votes",
    )
    parser.add_argument("--width-ms", type=float, default=core.BIN_MS)
    parser.add_argument(
        "--afrac-window",
        type=float,
        default=th.AFRAC_PSI_WINDOW,
        help="Afrac flux window; other than the default only for the sensitivity",
    )
    parser.add_argument(
        "--out-dir", default=None, help="default: $LABELER_ROOT/round4/detach/bins"
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else root() / "bins"
    shots = [int(s) for s in Path(args.shots_file).read_text().split()]
    if args.refresh_geometry:
        refresh_geometry(out_dir, args.width_ms, set(shots))
        return 0
    if not args.redo:
        shots = [s for s in shots if not (out_dir / f"{s}.npz").is_file()]
    log = (
        out_dir.parent / f"{out_dir.name}_log.jsonl"
        if args.out_dir
        else root() / "bins_log.jsonl"
    )
    done = 0
    with mp.Pool(args.workers) as pool, open(log, "a") as handle:
        for result in pool.imap_unordered(
            functools.partial(
                process_safe,
                width_ms=args.width_ms,
                out_dir=out_dir,
                afrac_window=args.afrac_window,
            ),
            shots,
            chunksize=1,
        ):
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            done += 1
            print(done, len(shots), result, flush=True)
    return 0


def process_safe(
    shot: int, width_ms: float, out_dir: Path, afrac_window: float = th.AFRAC_PSI_WINDOW
) -> dict:
    try:
        return process(shot, width_ms, out_dir, afrac_window=afrac_window)
    except Exception as error:  # noqa: BLE001  one bad shot must not stop the run
        return {"shot": shot, "status": f"error {type(error).__name__}: {error}"[:200]}


if __name__ == "__main__":
    sys.exit(main())
