#!/usr/bin/env python
"""Fetch the DIII-D signals the detachment label needs and park them per shot.

Run on the login node under the fdp wrapper, never in a SLURM job:

    pixi run --frozen -e labelmaker fdp run python scripts/labeler/detach_fetch.py \
        --shots-file shots.txt --workers 3 --pace 1

What is fetched, per shot (everything else comes from the corpus):

* divertor and total bolometry, ``\\BOLOM::PRAD_{DIVL,DIVU,TOT,CORE}``
  (calibrated, post-shot; Eldon 2019's standard Prad,div,L);
* the EFIT01 scalars that gate and normalise the indicators: X-point and the four
  strike points, stored energy, beam and ohmic power, the plasma current, field and
  minor radius; the CO2 line densities come from PTDATA;
* the divertor Thomson Te real-time points ``TSSDIVTE00-05``, which are NOT a
  voter: they are the independent check of the combined label.

One ``<shot>.npz`` per shot under ``$LABELER_ROOT/round4/detach/cache`` holds
``<name>__t`` (ms) and ``<name>__y`` per node plus a JSON ``status`` string.
A shot whose file already holds every node is skipped (a file missing a node is
topped up), so the script resumes. An authentication
or login failure stops the whole run (exit status 3): the owner's login has
lapsed and nothing more may be fetched until it is renewed.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

#: (name, tree, expression). Names are the keys in the cache file.
MDS_NODES: tuple[tuple[str, str, str], ...] = (
    ("prad_divl", "BOLOM", r"\BOLOM::PRAD_DIVL"),
    ("prad_divu", "BOLOM", r"\BOLOM::PRAD_DIVU"),
    ("prad_tot", "BOLOM", r"\BOLOM::PRAD_TOT"),
    ("prad_core", "BOLOM", r"\BOLOM::PRAD_CORE"),
    *(
        (name, "efit01", rf"\efit01::top.results.aeqdsk:{name}")
        for name in (
            "rvsod",
            "zvsod",
            "rvsid",
            "zvsid",
            "rxpt1",
            "zxpt1",
            "rxpt2",
            "zxpt2",
            "wmhd",
            "aminor",
            "rout",
            "kappa",
            "betan",
            "ipmeas",
            "bcentr",
            "poh",
            "q95",
            "rsurf",
        )
    ),
)
#: Divertor Thomson Te (an independent check) and the three CO2 vertical chords plus
#: the radial one (the line-averaged density the Afrac model scales with; the corpus
#: `co2` group is a stub on many shots, and EFIT's aeqdsk has no density node).
#: The total ECH power point joins them: EFIT's own `pbinj` is zero on these shots
#: and the corpus `ech_power` group is a stub on many.
PTDATA_NODES: tuple[str, ...] = (
    *(f"TSSDIVTE{i:02d}" for i in range(6)),
    "DENV2UF",
    "DENV3UF",
    "DENR0UF",
    "ECHPWR",
)

#: Injected power per neutral beam (W): fetched only for a shot whose corpus `pinj`
#: group is a stub. The PTDATA total `PINJ` is not in the archive fdp reaches.
BEAMS = ("15L", "15R", "21L", "21R", "30L", "30R", "33L", "33R")
BEAM_NODES: tuple[tuple[str, str, str], ...] = tuple(
    (f"pinj_{b.lower()}", "D3D", rf"\D3D::TOP.NB.NB{b}:PINJ_{b}") for b in BEAMS
)

#: Nodes sampled far faster than a label bin needs (the CO2 chords run at 2 MHz) are
#: stored as block means of this many milliseconds.
DECIMATE_MS = {"DENV2UF": 10.0, "DENV3UF": 10.0, "DENR0UF": 10.0}

AUTH_WORDS = ("auth", "token", "login", "credential", "401", "403", "permission")


def cache_dir() -> Path:
    root = Path(os.environ["LABELER_ROOT"])
    return root / "round4" / "detach" / "cache"


def is_auth_error(text: str) -> bool:
    text = text.lower()
    return any(word in text for word in AUTH_WORDS)


def corpus_is_stub(shot: int, group: str) -> bool:
    """True when the corpus has no real record of `group` for this shot."""
    import h5py

    path = Path("/scratch/gpfs/EKOLEMEN/foundation_model") / f"{shot}_processed.h5"
    if not path.is_file():
        return True
    with h5py.File(path, "r") as f:
        return group not in f or f[group]["ydata"].shape[-1] <= 1


def mds_nodes(shot: int) -> tuple[tuple[str, str, str], ...]:
    """The MDSplus nodes to fetch for this shot (beams only without a corpus record)."""
    if corpus_is_stub(shot, "pinj"):
        return MDS_NODES + BEAM_NODES
    return MDS_NODES


def block_mean(t_ms: np.ndarray, y: np.ndarray, width_ms: float):
    """Means of `y` over consecutive `width_ms` blocks, and the block centres."""
    block = np.floor((t_ms - t_ms[0]) / width_ms).astype(np.int64)
    count = np.bincount(block).astype(float)
    ok = np.isfinite(y)
    total = np.bincount(block[ok], weights=y[ok], minlength=count.size)
    n_ok = np.bincount(block[ok], minlength=count.size).astype(float)
    centre = t_ms[0] + (np.arange(count.size) + 0.5) * width_ms
    with np.errstate(invalid="ignore", divide="ignore"):
        return centre, np.where(n_ok > 0, total / n_ok, np.nan)


def fetch_shot(shot: int, out: Path) -> dict:
    """Fetch every missing node for one shot; return a status dict, write the npz.

    An existing file is topped up: only the nodes it lacks are fetched, so adding
    a node to the lists above does not refetch what is already parked.
    """
    from labeler.features import resolve_fdp

    arrays: dict[str, np.ndarray] = {}
    status: dict[str, str] = {}
    if out.is_file():
        with np.load(out) as old:
            arrays = {k: old[k] for k in old.files if k != "status"}
            status = json.loads(str(old["status"]))
    auth = False
    for name, tree, expr in mds_nodes(shot):
        if f"{name}__y" in arrays:
            continue
        try:
            rec = resolve_fdp._fetch_mds(expr, tree, shot)
            y = np.asarray(rec["data"])
            t = rec.get("times", rec.get("dim0"))
            arrays[f"{name}__y"] = y.astype("float64")
            arrays[f"{name}__t"] = np.asarray(t, dtype="float64")
            units = rec.get("units", {})
            status[name] = f"ok {y.size} {units.get('data', '')}"
        except Exception as error:  # noqa: BLE001  a node absent on a shot is data
            text = f"{type(error).__name__}: {str(error)[:160]}"
            status[name] = "error " + text
            auth = auth or is_auth_error(text)
    for name in PTDATA_NODES:
        key = name.lower()
        if f"{key}__y" in arrays:
            continue
        try:
            rec = resolve_fdp._fetch_ptdata(name, shot)
            y = np.asarray(rec["data"], dtype="float64")
            t = np.asarray(rec["times"], dtype="float64")
            if name in DECIMATE_MS:
                t, y = block_mean(t, y, DECIMATE_MS[name])
            arrays[f"{key}__y"] = y.astype("float32")
            arrays[f"{key}__t"] = t
            status[key] = f"ok {y.size}"
        except Exception as error:  # noqa: BLE001
            text = f"{type(error).__name__}: {str(error)[:160]}"
            status[key] = "error " + text
            auth = auth or is_auth_error(text)
    if auth:
        return {"shot": shot, "auth": True, "status": status}
    tmp = out.with_name(f".{out.name}.tmp.npz")
    np.savez(tmp, status=json.dumps(status), **arrays)
    tmp.replace(out)
    return {"shot": shot, "auth": False, "status": status}


def is_complete(shot: int, path: Path) -> bool:
    """True when the parked file holds an attempt (ok or error) at every node."""
    if not path.is_file():
        return False
    with np.load(path) as old:
        done = json.loads(str(old["status"]))
    wanted = [n for n, _, _ in mds_nodes(shot)] + [n.lower() for n in PTDATA_NODES]
    return all(name in done for name in wanted)


def worker(args: tuple[int, float]) -> dict:
    shot, pace = args
    out = cache_dir() / f"{shot}.npz"
    if is_complete(shot, out):
        return {"shot": shot, "skipped": True, "auth": False}
    started = time.monotonic()
    try:
        result = fetch_shot(shot, out)
    except Exception as error:  # noqa: BLE001  one bad shot must not stop the run
        result = {"shot": shot, "auth": False, "fatal": str(error)[:200]}
        if is_auth_error(str(error)):
            result["auth"] = True
    result["seconds"] = round(time.monotonic() - started, 1)
    time.sleep(pace)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shots-file", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--pace", type=float, default=1.0)
    parser.add_argument("--log", default=None)
    args = parser.parse_args()
    if args.workers > 3:
        raise SystemExit("at most 3 workers (rules file)")
    shots = [int(s) for s in Path(args.shots_file).read_text().split()]
    cache_dir().mkdir(parents=True, exist_ok=True)
    log = Path(args.log) if args.log else cache_dir().parent / "fetch_log.jsonl"
    done = 0
    with mp.Pool(args.workers) as pool, open(log, "a") as handle:
        for result in pool.imap_unordered(
            worker, [(s, args.pace) for s in shots], chunksize=1
        ):
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            done += 1
            if result.get("auth"):
                print("AUTH ERROR: stopping, the login has lapsed", file=sys.stderr)
                pool.terminate()
                return 3
            print(done, len(shots), result["shot"], result.get("seconds"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
