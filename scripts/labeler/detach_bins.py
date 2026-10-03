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
        return {"times_ms": npz["times_ms"], "ze": npz["ze"]}


def tangtv_for(shot, edges, cache):
    """The TangTV indicator and where its front height came from.

    The inversion when the shot has one, else the regression from the raw frames,
    else an all-invalid indicator saying why. The source is `inversion`, `surrogate`
    or `none`.
    """
    n = len(edges) - 1

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
        ze = tangtv.outer_leg_ze(
            inversion["frames"].astype(np.float32),
            inversion["radii"],
            inversion["elevation"],
            np.asarray(cache["rxpt1"][1])[near],
        )
        source = "inversion"
    else:
        ft = surrogate["times_ms"].astype(float)
        ze = surrogate["ze"].astype(float)
        source = "surrogate"
    indicator = tangtv.tangtv_indicator(
        edges,
        ft,
        ze,
        et,
        cache["rvsod"][1],
        cache["zvsod"][1],
        cache["rxpt1"][1],
        cache["zxpt1"][1],
    )
    return indicator, source


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
    tangtv_ind, tangtv_source = tangtv_for(shot, edges, cache)

    out: dict[str, np.ndarray] = {"start_ms": starts}
    for ind in (afrac_ind, prad_ind, tangtv_ind):
        out[f"{ind.name}_value"] = ind.value.astype(np.float32)
        out[f"{ind.name}_valid"] = ind.valid
        out[f"{ind.name}_reason"] = ind.reason.astype(str)
        out[f"{ind.name}_vote"] = ind.vote
    out["tangtv_source"] = np.full(n, tangtv_source)
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
    divte = [cache[k] for k in cache if k.startswith("tssdivte")]
    if divte:
        stack = np.vstack([core.bin_median(t, y, edges)[0] for t, y in divte])
        with np.errstate(all="ignore"):
            out["aux_te_div"] = np.nanmax(stack, axis=0).astype(np.float32)
    for key in ("rvsod", "zvsod", "rxpt1", "zxpt1"):
        if key in cache:
            out[f"aux_{key}"] = core.bin_median(*cache[key], edges)[0].astype(
                np.float32
            )
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
