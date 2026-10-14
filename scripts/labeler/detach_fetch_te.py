#!/usr/bin/env python
"""Fetch divertor Thomson Te and EFIT01 flux maps for the detachment Te source.

Run on the login node under the fdp wrapper, at most three workers:

    TMPDIR=$LABELER_ROOT/scratch/tmp-fetch pixi run --frozen -e labeler fdp run \\
        python scripts/labeler/detach_fetch_te.py --select --workers 3

`--select` first writes the shot list, `$LABELER_ROOT/round5/detach/shelf_shots.txt`
(with the reason per shot in `shelf_shots.json`): the silver and gold shots whose
outer strike sits on the shelf the divertor Thomson chords look at, by the geometry
already on disk, for at least `MIN_SHELF_MS`:

* a flux map (`source.load_maps`: the round-5 EFIT02 park, the round-4 maps) puts one
  of the Te-check chords (`CHORDS`, the nominal R, Z of the two chords 1.5-5 cm
  above the shelf) at 1.000 < psiN <= 1.05, the Te check's own gate; or
* the round-4 EFIT01 outer strike point (`signals.load_cache`: RVSOD, ZVSOD) lies on
  the shelf, |Z + 1.25 m| <= 0.05 m and 1.37 m <= R <= 1.485 m (inboard of the
  chords, else they look into the private flux region).

The shelf is the round-4 code's "upper shelf" (Z = -1.25 m, R >= 1.37 m), the one
the chords view; the floor (Z = -1.363 m) is out of their view. A shot with neither
geometry is listed as `no_geometry` and not fetched.

Per listed shot: `<root>/dts/<shot>.npz`, the round-4 divertor Thomson nodes
(`detach_fetch_round4.DTS_NODES`; an npz with no `te` parks an absent node), and
`<root>/efit01/<shot>.npz`, the EFIT01 map of `detach_fetch_efit.fetch` (the Te
source reads EFIT02 first and EFIT01 where EFIT02 has no slice). `<root>` is
`$LABELER_ROOT/round5/detach`; a record parked there or under `round4/detach` is
not fetched again. An authentication error stops every worker (exit 3).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from detach_fetch import is_auth_error
from detach_fetch_efit import fetch as fetch_efit
from detach_fetch_round4 import DTS_NODES

from labeler.config import Paths
from labeler.events.detachment import signals, source
from labeler.events.detachment import thresholds as th

ROOT = source.data_root()
STOP = ROOT / "fetch_auth_stop"
LOG = ROOT / "fetch_te_log.jsonl"
ROUND4 = ROOT.parent.parent / "round4" / "detach"
#: The two chords the Te check reads (TSR_DIV, TSZ_DIV on 2018-2023 shots: R 1.485 m,
#: Z -1.223 and -1.205 m, 2.7 and 4.5 cm above the shelf).
CHORDS = np.array([[1.485, -1.223], [1.485, -1.205]])
MIN_SHELF_MS = 100.0
STRIKE_R_MAX = 1.485


def on_shelf_ms(shot: int) -> dict:
    """How long each geometry on disk puts the strike on the chords' shelf."""
    out = {}
    maps = source.load_maps(shot)
    if maps:
        t = np.unique(np.concatenate([m["gtime_ms"] for _, m in maps]))
        psi, _ = source.chord_flux(maps, t, CHORDS)
        sol = ((psi > source.PSI_N_SOL[0]) & (psi <= source.PSI_N_SOL[1])).any(axis=0)
        step = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
        out["efit_chords_ms"] = float(sol.sum() * step)
    cache = signals.load_cache(shot)
    if "rvsod" in cache and "zvsod" in cache:
        t, r = cache["rvsod"]
        tz, z = cache["zvsod"]
        if len(t) == len(tz) and len(t) > 1:
            on = (
                (np.abs(z - th.SHELF_Z) <= th.SHELF_Z_TOL)
                & (r >= th.SHELF_WALL_R)
                & (r <= STRIKE_R_MAX)
            )
            out["efit01_strike_ms"] = float(on.sum() * np.median(np.diff(t)))
    return out


def select(shots) -> tuple[list[int], dict]:
    reasons = {}
    for shot in shots:
        found = on_shelf_ms(shot)
        reasons[str(shot)] = found or "no_geometry"
    chosen = [
        int(s)
        for s, found in reasons.items()
        if isinstance(found, dict) and max(found.values()) >= MIN_SHELF_MS
    ]
    return sorted(chosen), reasons


def parked(kind: str, shot: int) -> bool:
    names = {"dts": ("dts", "dts"), "efit01": ("efit01", "efit")}[kind]
    return (ROOT / names[0] / f"{shot}.npz").is_file() or (
        ROUND4 / names[1] / f"{shot}.npz"
    ).is_file()


def save(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def fetch_dts(shot: int) -> dict:
    from labeler.features import resolve_fdp

    arrays, status = {}, {}
    for key, node in DTS_NODES.items():
        if STOP.exists():
            raise RuntimeError("authentication stop flag")
        try:
            rec = resolve_fdp._fetch_mds(rf"\ELECTRONS::{node}", "ELECTRONS", shot)
            arrays[key] = np.asarray(rec["data"], dtype=np.float32)
            status[key] = f"ok {arrays[key].shape}"
        except Exception as error:
            text = f"{type(error).__name__}: {error}"[:200]
            if is_auth_error(text):
                raise
            status[key] = "error " + text
    if "te" not in arrays or "t_ms" not in arrays:
        arrays = {}  # park the absence so a rerun does not refetch
    save(ROOT / "dts" / f"{shot}.npz", {"status": json.dumps(status), **arrays})
    return {"dts": "ok" if arrays else "absent"}


def worker(shot: int) -> dict:
    row = {"shot": shot}
    try:
        if not parked("dts", shot):
            row.update(fetch_dts(shot))
        if not parked("efit01", shot):
            try:
                save(ROOT / "efit01" / f"{shot}.npz", fetch_efit(shot, "efit01"))
                row["efit01"] = "ok"
            except Exception as error:
                text = f"{type(error).__name__}: {error}"[:200]
                if is_auth_error(text):
                    raise
                row["efit01"] = "error " + text
    except Exception as error:  # noqa: BLE001
        text = f"{type(error).__name__}: {error}"[:200]
        row["error"] = text
        if is_auth_error(text) or STOP.exists():
            STOP.touch()
            row["auth"] = True
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--select", action="store_true", help="rewrite the shot list")
    parser.add_argument("--select-only", action="store_true", help="fetch nothing")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--pace", type=float, default=1.0)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise SystemExit("one to three workers")
    if STOP.exists():
        raise SystemExit(f"AUTH ERROR: {STOP} exists; fetching stopped")
    ROOT.mkdir(parents=True, exist_ok=True)
    listed = ROOT / "shelf_shots.txt"
    if args.select or args.select_only or not listed.is_file():
        shots = source.silver_gold_shots(Paths.from_env())
        chosen, reasons = select(shots)
        listed.write_text("\n".join(map(str, chosen)) + "\n")
        (ROOT / "shelf_shots.json").write_text(
            json.dumps(
                {
                    "shots_considered": len(shots),
                    "chosen": len(chosen),
                    "no_geometry": sum(v == "no_geometry" for v in reasons.values()),
                    "min_shelf_ms": MIN_SHELF_MS,
                    "reasons": reasons,
                },
                indent=1,
            )
            + "\n"
        )
        print(f"{len(chosen)} of {len(shots)} shots on the shelf", flush=True)
    if args.select_only:
        return 0
    todo = [
        s
        for s in map(int, listed.read_text().split())
        if not (parked("dts", s) and parked("efit01", s))
    ]
    print(f"{len(todo)} shots to fetch", flush=True)
    with mp.Pool(args.workers) as pool, LOG.open("a") as log:
        for row in pool.imap_unordered(worker, todo):
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
            if row.get("auth"):
                pool.terminate()
                print("AUTH ERROR: stopping, the login has lapsed", file=sys.stderr)
                return 3
            time.sleep(args.pace / args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
