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
independent divertor Thomson Te (`aux_te_div`), which is NOT an indicator.

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

#: Bins with less plasma current than this are not part of the discharge.
MIN_IP_A = 0.3e6
INDICATORS = ("afrac", "prad", "tangtv")


def root() -> Path:
    return Path(os.environ["LABELER_ROOT"]) / "round4" / "detach"


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


def tangtv_for(shot, edges, cache, elm=None):
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
        return invalid("no_inversion"), "none"
    need = ("rvsod", "zvsod", "rxpt1", "zxpt1")
    if any(k not in cache for k in need):
        return invalid("efit_missing"), "none"
    et = cache["rxpt1"][0]
    if inversion is not None:
        ft = inversion["times_ms"].astype(float)
        near = np.abs(et[None, :] - ft[:, None]).argmin(axis=1)
        lower = float(np.nanmedian(cache["zvsod"][1])) < -1.30
        ze = tangtv.outer_leg_ze(
            inversion["frames"].astype(np.float32),
            inversion["radii"],
            inversion["elevation"],
            np.asarray(cache["rxpt1"][1])[near],
            r_max=1.37 if lower else None,
        )
        source = "inversion"
    else:
        ft = surrogate["times_ms"].astype(float)
        ze = surrogate["ze"].astype(float)
        source = "surrogate"
        lower = False
    indicator = tangtv.tangtv_indicator(
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
    )
    return indicator, source


def greenwald_cue(edges, cache):
    """Conservative fG>=0.8 cue, only with explicitly confirmed density units.

    UF PTDATA records on disk have no units. Do not infer Greenwald fraction from
    their magnitude. A future fetch may supply density_v2_si with m^-2 units;
    use an elliptical V2 chord at R=1.94 m (shot_design geometry), flag its
    approximation. Missing unit-confirmed density produces no cue.
    """
    n = len(edges) - 1
    if "density_v2_si" not in cache:
        return np.zeros(n, bool), np.full(n, np.nan)
    density = core.bin_median(*cache["density_v2_si"], edges)[0]
    a, kappa, axis, ip = [
        core.bin_median(*cache[k], edges)[0]
        for k in ("aminor", "kappa", "rout", "ipmeas")
    ]
    with np.errstate(invalid="ignore", divide="ignore"):
        length = 2 * kappa * np.sqrt(a * a - (1.94 - axis) ** 2)
        fraction = density / length / 1e20 / (np.abs(ip) / 1e6 / (np.pi * a * a))
    return np.isfinite(fraction) & (fraction >= 0.8), fraction


def spatial_evidence(shot, edges, geo):
    """Global inversion peak psiN<1, within 30 cm of X and near/above ZX."""
    from scipy.interpolate import RegularGridInterpolator

    n = len(edges) - 1
    result = np.zeros(n, bool)
    inv = load_inversion(shot)
    path = root() / "efit" / f"{shot}.npz"
    if inv is None or not path.exists():
        return result
    with np.load(path) as f:
        if str(f.get("source", "EFIT01")) != "EFIT02":
            return result
        ft = inv["times_ms"]
        flags = np.zeros(len(ft), bool)
        for j, t in enumerate(ft):
            k = np.argmin(np.abs(f["gtime_ms"] - t))
            if abs(f["gtime_ms"][k] - t) > 40:
                continue
            iz, ir = np.unravel_index(
                np.nanargmax(inv["frames"][j]), inv["frames"][j].shape
            )
            r, z = inv["radii"][ir], inv["elevation"][iz]
            psin = (f["psirz"][k] - f["ssimag"][k]) / (f["ssibry"][k] - f["ssimag"][k])
            psi = RegularGridInterpolator(
                (f["z"], f["r"]), psin, bounds_error=False, fill_value=np.nan
            )((z, r))
            rx = np.interp(t, *geo["rxpt1"])
            zx = np.interp(t, *geo["zxpt1"])
            flags[j] = bool(
                0 < psi < 1 and zx - 0.02 <= z <= zx + 0.30 and abs(r - rx) <= 0.30
            )
        result = core.bin_fraction(ft, flags, edges) >= 0.5
    return result


def confinement(shot, edges):
    """Project's explicit Jalal Butt confinement labels; unknown stays 0."""
    import pandas as pd

    path = (
        Path(os.environ["LABELER_LABEL_TABLES"])
        / "confinement/raw/Jalal_28042024_confinement_regime_shotlist.csv"
    )
    rows = pd.read_csv(path).query("Shot == @shot")
    times = core.bin_centres(edges)
    mode = np.zeros(len(times), int)
    back = np.zeros(len(times), bool)
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


def processed_ratio(shot, edges, cache, base, tv, elm):
    """Use positioned processed Jsat; calibrate only attached pre-puff L/H bins.

    If the attached/regime reference is absent, export a local ratio with a
    whole-shot 90th percentile reference, explicitly marked local_proxy.
    """
    n = len(edges) - 1
    modes = np.full(n, "local_proxy")
    path = root() / "processed_probes" / f"{shot}.npz"
    if not path.exists():
        return base, modes
    with np.load(path) as f:
        keys = [k[:-5] for k in f.files if k.endswith("_jsat")]
        if not keys:
            return base, modes
        per_probe = []
        for key in keys:
            t, y = f[key + "_t_ms"], f[key + "_jsat"]
            flags = core.elm_at(t, *(elm or (None, None)))
            per_probe.append(core.bin_median(t, y, edges, keep=~flags)[0])
        jsat = np.vstack(per_probe)
        positions = np.array([f[k + "_rz"] for k in keys])
    geo, _ = signals.tangtv_geometry(shot, cache)
    strike = np.stack(
        [core.bin_median(*geo[k], edges)[0] for k in ("rvsod", "zvsod")], axis=1
    )
    density = signals.line_density(cache)
    power = signals.heating_power(shot, cache)
    if density is None or power is None:
        return base, modes
    base = afrac.afrac_indicator(
        edges,
        core.bin_centres(edges),
        jsat,
        *density,
        power[0],
        power[2],
        *cache["ipmeas"],
        *(elm or (None, None)),
    )
    ne = core.bin_median(*density, edges)[0]
    # Eldon 2021 DOD uses C*n^2 (no power scaling); attached C is fitted by regime.
    scaling = ne**2
    regime, _ = confinement(shot, edges)
    gas = signals.corpus_group(shot, "gas_flow")
    pre = np.zeros(n, bool)
    if gas is not None:
        t, y = gas
        g = core.bin_mean(t, np.nansum(np.maximum(y, 0), axis=0), edges)[0]
        finite = np.isfinite(g)
        if finite.any():
            floor = np.nanpercentile(g, 10)
            onset = np.flatnonzero(finite & (g > floor + 0.1 * (np.nanmax(g) - floor)))
            if len(onset):
                pre = np.arange(n) < onset[0]
    value, valid, which = afrac.calibrated_ratio(
        jsat, positions, strike, scaling, pre & (tv.vote == core.ATTACHED), regime
    )
    valid &= np.isin(base.reason, ("", "short_reference"))
    value[~valid] = np.nan
    modes[valid] = "eldon_pre_puff_LH"
    # A fallback is a local proxy, even with calibrated current: its attached
    # reference has not been identified. Preserve geometry and quality gates.
    nearest = np.linalg.norm(positions[which] - strike, axis=1) <= 0.02
    raw = jsat[which, np.arange(n)] / (ne**2)
    usable = nearest & np.isfinite(raw) & (raw > 0) & base.valid
    if usable.sum() * float(edges[1] - edges[0]) >= 3000:
        fallback = usable & ~valid
        value[fallback] = raw[fallback] / np.quantile(raw[usable], 0.9)
        valid |= fallback
    reason = np.where(valid, "", "no_attached_regime_reference")
    reason[~nearest] = "probe_far_from_strike"
    return core.assemble("afrac", value, valid, reason, afrac.afrac_vote(value)), modes


def process(
    shot: int, width_ms: float = core.BIN_MS, out_dir: Path | None = None
) -> dict:
    """Compute and save one shot's bins; return a one-line status."""
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
    p_t, p_in, p_sol = (None, None, None) if power is None else power
    elm = signals.elm_mask(shot)
    elm_t, elm_flag = (None, None) if elm is None else elm
    density = signals.line_density(cache)
    n_t, n_y = (None, None) if density is None else density
    probes = langmuir.read_shot(shot)

    prad_ind = prad.prad_indicator(
        edges,
        *cache.get("prad_divl", (None, None)),
        p_t,
        p_in,
        elm_t,
        elm_flag,
    )
    afrac_ind = afrac.afrac_indicator(
        edges,
        None if probes is None else probes["t_ms"],
        None if probes is None else probes["jsat"],
        n_t,
        n_y,
        p_t,
        p_sol,
        t_ip,
        ip,
        elm_t,
        elm_flag,
    )
    tangtv_ind, tangtv_source = tangtv_for(shot, edges, cache, elm)
    geo, geo_source = signals.tangtv_geometry(shot, cache)
    # Spatial evidence exists only for true inversions with a close flux map.
    spatial = spatial_evidence(shot, edges, geo)
    second, fg = greenwald_cue(edges, cache)
    _, back_transition = confinement(shot, edges)
    second |= back_transition
    afrac_ind, afrac_mode = processed_ratio(
        shot, edges, cache, afrac_ind, tangtv_ind, elm
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
    out["tangtv_source"] = np.full(n, tangtv_source)
    out["tangtv_efit_source"] = np.full(n, geo_source)
    out["tangtv_marfe_candidate"] = candidate
    out["tangtv_marfe_spatial"] = spatial
    out["tangtv_marfe_second_cue"] = second
    out["aux_greenwald_fraction"] = fg
    # the quantities behind the indicators, for the figure and the failure analysis
    out["aux_ip_a"] = core.bin_median(t_ip, ip, edges)[0].astype(np.float32)
    if p_t is not None:
        out["aux_p_in_w"] = core.bin_mean(p_t, p_in, edges)[0].astype(np.float32)
    if n_t is not None:
        out["aux_ne"] = core.bin_median(n_t, n_y, edges)[0].astype(np.float32)
    if probes is not None:
        peak, which = afrac.peak_jsat(edges, probes["t_ms"], probes["jsat"])
        out["aux_jsat_peak"] = peak.astype(np.float32)
        out["aux_jsat_probe"] = probes["probe"][which].astype(np.int16)
    if elm_t is not None:
        out["aux_elm_share"] = core.bin_fraction(elm_t, elm_flag, edges).astype(
            np.float32
        )
    # Unlocalised real-time DTS is deliberately excluded from all claims.
    for key in ("rvsod", "zvsod", "rxpt1", "zxpt1"):
        if key in geo:
            out[f"aux_{key}"] = core.bin_median(*geo[key], edges)[0].astype(np.float32)
    domain_record = (
        Path(__file__).resolve().parents[2]
        / "docs/labeler/results/detachment_tangtv_surrogate.json"
    )
    domain = json.loads(domain_record.read_text())["domain"]
    envelope = np.ones(n, bool)
    for key, values in (
        ("leg", out["aux_zxpt1"] - out["aux_zvsod"]),
        ("rv", out["aux_rvsod"]),
        ("rx", out["aux_rxpt1"]),
    ):
        lo, hi = domain[key]
        envelope &= (values >= lo) & (values <= hi)
    out["tangtv_in_envelope"] = envelope
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
    parser.add_argument("--width-ms", type=float, default=core.BIN_MS)
    parser.add_argument(
        "--out-dir", default=None, help="default: $LABELER_ROOT/round4/detach/bins"
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else root() / "bins"
    shots = [int(s) for s in Path(args.shots_file).read_text().split()]
    if not args.redo:
        shots = [s for s in shots if not (out_dir / f"{s}.npz").is_file()]
    log = (
        out_dir.parent / "bins_log.jsonl" if args.out_dir else root() / "bins_log.jsonl"
    )
    done = 0
    with mp.Pool(args.workers) as pool, open(log, "a") as handle:
        for result in pool.imap_unordered(
            functools.partial(process_safe, width_ms=args.width_ms, out_dir=out_dir),
            shots,
            chunksize=1,
        ):
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            done += 1
            print(done, len(shots), result, flush=True)
    return 0


def process_safe(shot: int, width_ms: float, out_dir: Path) -> dict:
    try:
        return process(shot, width_ms, out_dir)
    except Exception as error:  # noqa: BLE001  one bad shot must not stop the run
        return {"shot": shot, "status": f"error {type(error).__name__}: {error}"[:200]}


if __name__ == "__main__":
    sys.exit(main())
