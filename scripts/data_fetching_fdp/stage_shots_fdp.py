#!/usr/bin/env python
"""Download each shot's native DIII-D files from FDP's per-shot view, for local extraction.

FDP publishes every shot as a flat "view" under
`/fdp-d3d/views/shots/<shot // 100>/<shot>/v1/`: the shot's MDSplus tree files
(`mdsplus/{shots,codes,usershots}/<tree>_<shot>.{tree,characteristics,datafile}`), its
native PTDATA files (`ptdata/<shot>.<EXT>`, one per digitizer group) and
`meta/index.json` (pointname -> EXT). Reading signals node by node over Pelican is
latency-bound (17-40 s per ECE channel through TDI); downloading the files whole runs at
full bandwidth (185955's 1.95 GB of PTDATA in 73 s, one stream), and reading them from
local disk is 3-4x faster again than `PtDataSignal` over the index (ECE 1.3 s against
5.0 s per channel). So this script only downloads, and `fetch_raw_fdp.py --stage-dir`
extracts from the local copy, on a compute node, with no network.

A whole view is ~41 GB, nearly all of it PTDATA groups nothing here reads. Staged per
shot (~3 GB):

  * every `mdsplus/shots` `.tree`/`.characteristics` file (24 MB: subtree structure),
    and its `.datafile`s except SKIP_DATAFILES (large, unread);
  * the PTDATA files holding the pointnames the data-generation config reads, found
    through the shot's own index.json, plus ALWAYS_EXTS (the groups that the MDSplus
    wrappers read: ECE `ecevsNN`, SXR, bolometers).

Layout: `<stage>/<shot>/{mdsplus,ptdata}/...`, written into `<shot>.part/` and renamed
when every file is in, so an existing `<stage>/<shot>/` is complete and is skipped.
Runs on the LOGIN node under `fdp run` (it needs the bearer token and the network):

    pixi run --frozen -e fdp fdp -D d3d run python scripts/data_fetching_fdp/stage_shots_fdp.py \\
        --shot-file /scratch/gpfs/EKOLEMEN/D3D/shot_lists/recommender_v1.txt \\
        --stage-dir /scratch/gpfs/EKOLEMEN/D3D/fdp_stage --shots-parallel 4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO / "src/tokamak_foundation_model/data/config/modalities/modalities.yaml"
)
PELICAN = "pelican://osg-htc.org:443"
VIEW = "/fdp-d3d/views/shots/{prefix}/{shot}/v1"
# Unread and large on 185955: rpi 456 MB, slp 395 MB. irtv (~480 MB) is read, but every
# node in it is TreeNODATA over fdp, with or without its datafile.
SKIP_DATAFILES = {"irtv", "rpi", "slp"}
# PTDATA read through MDSplus expressions rather than named in the config: ECE TECEF wraps
# PTDATA2(ecevsNN), SXR and bolometer leaves wrap ptdata("sx...")/("bol_..."), and NB
# VOLTAGE_CAL reads \NBxx:PTDATA_RAW = PTDATA("NBVACxxT") from .NBD (16 MB on 185955) and
# its FIRED/NBVAC_SCALAR from the PTHEAD_* header of NBxxLEFT etc. in .NBI (421 MB).
ALWAYS_EXTS = {".ECF", ".SXR", ".BOL", ".NBD", ".NBI"}
# OSDF cache reachability from stellarai fluctuates: on 2026-10-14 the director's
# mghpcc, internet2 newy32aoa and cinc caches refused port 8443 for a while, then
# answered, while internet2 chic dropped to half its connects. A transfer assigned only
# unreachable caches fails after ~31 s, and one failed file aborts the rest of a multi-file
# `pelican object get`. So files go in small batches, and retries alternate between the
# director's own choice and this pinned list (the caches that answered best; "+" falls
# back to discovered ones last).
CACHES = (
    "https://osdf-uw-cache.svc.osg-htc.org:8443,"
    "https://osdf-cache-01.gwave.ics.psu.edu:8443,"
    "https://osdf1.chic.nrp.internet2.edu:8443,+"
)
BATCH = 8      # files per `pelican object get`
ATTEMPTS = 6   # per batch, alternating default/pinned caches, backing off 10 s, 20 s, ...


def read_shots(path: Path) -> list[int]:
    shots = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            shots.append(int(line))
    return shots


def ptdata_pointnames(config: dict) -> set[str]:
    """Every PTDATA pointname the config reads directly, fallbacks included."""
    names = set()
    for cfg in config["signals"].values():
        if cfg["tree"].upper() == "PTDATA":
            names.update(cfg["input_key"])
        fb = cfg.get("fallback")
        if fb and fb["tree"].upper() == "PTDATA":
            names.update(fb["input_key"])
    names.update(f"GAS{c}" for c in "ABCDEFGHIJK")  # gas_raw's PTDATA("GASx") wrappers
    return {n.upper() for n in names}


def pelican(*args: str, timeout: int = 3600, pinned: bool = False) -> str:
    if args[0] == "get" and pinned:
        args = ("get", "-c", CACHES, *args[1:])
    out = subprocess.run(
        ["pelican", "object", *args], capture_output=True, text=True, timeout=timeout
    )
    if out.returncode:
        raise RuntimeError(f"pelican object {' '.join(args)}: {out.stderr.strip()[-300:]}")
    return out.stdout


def listing(base: str) -> dict[str, int]:
    """Relative path -> size for every object under a view."""
    files = {}
    for line in pelican("ls", "-l", "-r", f"{PELICAN}{base}").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith(base) and parts[1].isdigit():
            files[parts[0][len(base) + 1:]] = int(parts[1])
    return files


def wanted(files: dict[str, int], index: dict, shot: int, names: set[str]) -> list[str]:
    """The subset of a view this pipeline reads."""
    pt_ext = {k.upper(): v.upper() for k, v in index["pointname_ext"].items()}
    exts = ALWAYS_EXTS | {pt_ext[n] for n in names if n in pt_ext}
    keep = []
    for rel in files:
        if rel.startswith("mdsplus/shots/"):
            leaf = rel.rsplit("/", 1)[1]
            if leaf.endswith(".datafile") and leaf.split("_")[0] in SKIP_DATAFILES:
                continue
            keep.append(rel)
        elif rel.startswith("ptdata/"):
            if "." + rel.rsplit(".", 1)[1].upper() in exts:
                keep.append(rel)
    return sorted(keep)


def stage_shot(shot: int, stage: Path, names: set[str], streams: int) -> dict:
    final = stage / str(shot)
    if final.exists():
        return {"shot": shot, "status": "exists"}
    part = stage / f"{shot}.part"
    started = time.monotonic()
    base = VIEW.format(prefix=shot // 100, shot=shot)
    files = listing(base)
    if not files:
        return {"shot": shot, "status": "no view"}
    part.mkdir(parents=True, exist_ok=True)
    (part / "meta").mkdir(exist_ok=True)
    for attempt in range(ATTEMPTS):
        try:
            pelican("get", f"{PELICAN}{base}/meta/index.json", str(part / "meta/index.json"),
                    pinned=bool(attempt % 2))
            break
        except RuntimeError:
            if attempt == ATTEMPTS - 1:
                raise
            time.sleep(10 * (attempt + 1))
    index = json.loads((part / "meta/index.json").read_text())
    todo = wanted(files, index, shot, names)

    def complete(rel: str) -> bool:
        dest = part / rel
        return dest.exists() and dest.stat().st_size == files[rel]

    def get(batch: list[str]) -> None:
        rel_dir = batch[0].rsplit("/", 1)[0]
        (part / rel_dir).mkdir(parents=True, exist_ok=True)
        for attempt in range(ATTEMPTS):
            batch = [r for r in batch if not complete(r)]
            if not batch:
                return
            try:
                pelican("get", *[f"{PELICAN}{base}/{r}" for r in batch], str(part / rel_dir),
                        pinned=bool(attempt % 2))
            except RuntimeError:
                time.sleep(10 * (attempt + 1))

    pending = [r for r in todo if not complete(r)]
    batches: list[list[str]] = []
    for rel_dir in sorted({r.rsplit("/", 1)[0] for r in pending}):
        rels = [r for r in pending if r.rsplit("/", 1)[0] == rel_dir]
        # big PTDATA files alone, so one slow or failed file holds back nothing else
        size = 1 if rel_dir == "ptdata" else BATCH
        batches += [rels[i:i + size] for i in range(0, len(rels), size)]
    with ThreadPoolExecutor(streams) as pool:
        list(pool.map(get, batches))
    short = [r for r in todo if not complete(r)]
    if short:
        raise RuntimeError(f"{len(short)} files incomplete after retries, e.g. {short[:3]}")
    nbytes = sum(files[r] for r in todo)
    (part / "meta/stage.json").write_text(json.dumps({
        "shot": shot, "view": base, "files": {r: files[r] for r in todo},
        "bytes": nbytes, "view_bytes": sum(files.values()),
    }, indent=1))
    part.rename(final)
    return {"shot": shot, "status": "staged", "files": len(todo),
            "gb": round(nbytes / 1e9, 2), "seconds": round(time.monotonic() - started, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shot-file", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--shots-parallel", type=int, default=4)
    parser.add_argument("--streams", type=int, default=4, help="pelican calls in flight per shot")
    parser.add_argument("--rounds", type=int, default=3, help="passes over shots that failed")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not os.environ.get("BEARER_TOKEN"):
        sys.exit("BEARER_TOKEN is not set: run under `fdp -D d3d run`")
    names = ptdata_pointnames(yaml.safe_load(args.config.read_text()))
    args.stage_dir.mkdir(parents=True, exist_ok=True)
    shots = [s for s in read_shots(args.shot_file) if not (args.stage_dir / str(s)).exists()]
    if args.limit:
        shots = shots[: args.limit]
    print(f"{len(shots)} shots to stage -> {args.stage_dir}", flush=True)

    def one(shot: int) -> dict:
        try:
            return stage_shot(shot, args.stage_dir, names, args.streams)
        except Exception as exc:  # noqa: BLE001 - report and move on; rerun resumes
            return {"shot": shot, "status": f"failed: {type(exc).__name__}: {exc}"[:400]}

    for rnd in range(1, args.rounds + 1):
        done, failed = 0, []
        with ThreadPoolExecutor(args.shots_parallel) as pool:
            futures = [pool.submit(one, s) for s in shots]
            for res in (f.result() for f in as_completed(futures)):
                done += 1
                if res["status"].startswith("failed"):
                    failed.append(res["shot"])
                print(f"[round {rnd}: {done}/{len(shots)}] {json.dumps(res)}", flush=True)
        if not failed:
            break
        print(f"round {rnd}: {len(failed)} failed; retrying them", flush=True)
        shots = sorted(failed)
        time.sleep(60)
    free = shutil.disk_usage(args.stage_dir).free / 1e12
    print(f"done; {free:.1f} TB free on the stage filesystem", flush=True)


if __name__ == "__main__":
    main()
