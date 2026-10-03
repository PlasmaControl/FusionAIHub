#!/usr/bin/env python
"""Supplement detachment caches with ELM coverage, confirmed density and maps.

Run via the login-node fdp wrapper. Defaults to inversion/surrogate shots;
`--shots-file` can extend that set. One worker, one-second pacing. Every
authentication error sets the shared stop flag and stops before another fetch.
Existing cache arrays are preserved. FS01-FS04 are fetched only without live
corpus D-alpha. Sparse EFIT02 maps are replaced only by multi-slice EFIT01.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from detach_fetch import block_mean, is_auth_error
from detach_fetch_efit import fetch as fetch_maps

from labeler.events.detachment import signals
from labeler.features import resolve_fdp

ROOT = Path(os.environ["LABELER_ROOT"]) / "round4/detach"
STOP = ROOT / "fetch_auth_stop"


def fetch_record(expr, tree, shot):
    if STOP.exists():
        raise RuntimeError("authentication stop flag")
    try:
        return resolve_fdp._fetch_mds(expr, tree, shot)
    except Exception as error:
        if is_auth_error(str(error)):
            STOP.touch()
        raise


def supplement(shot, ensure_flux=False):
    path = ROOT / "cache" / f"{shot}.npz"
    row = {"shot": shot}
    if not path.exists():
        return {**row, "status": "no_cache"}
    with np.load(path) as f:
        arrays = {k: f[k] for k in f.files if k != "status"}
        status = json.loads(str(f["status"]))
    cache = signals.load_cache(shot)
    # Availability only: running the wide median detector is unnecessary
    # when deciding whether a fetch is needed, especially on cached reruns.
    corpus = signals.corpus_group(shot, "filterscopes", channels=range(8))
    corpus_live = corpus is not None and any(
        np.isfinite(row).mean() > 0.9 and np.nanmedian(row) > 0 for row in corpus[1]
    )
    cached_live = any(
        name in cache
        and len(cache[name][0]) > 1
        and np.isfinite(cache[name][1]).mean() > 0.9
        and np.nanmedian(cache[name][1]) > 0
        for name in ("fs01", "fs02", "fs03", "fs04")
    )
    need_filters = not (corpus_live or cached_live)
    requests = [
        ("density_v2", "bci", r"\BCI::DENV2"),
        ("r0", "efit01", r"\efit01::top.results.aeqdsk:r0"),
    ]
    if need_filters:
        requests.extend(
            (f"fs{i:02d}", "SPECTROSCOPY", rf"\SPECTROSCOPY::FS{i:02d}")
            for i in range(1, 5)
        )
    for name, tree, expr in requests:
        if f"{name}__y" in arrays:
            continue
        try:
            rec = fetch_record(expr, tree, shot)
            t = np.asarray(rec.get("times", rec.get("dim0")), float)
            y = np.asarray(rec["data"], float)
            if len(t) < 2 or len(t) != len(y):
                raise ValueError("not a usable time series")
            units = rec.get("units", {})
            units = units.get("data", "") if isinstance(units, dict) else str(units)
            if name == "density_v2":
                # Read the leaf's units explicitly: a tag has historically
                # reported V despite the line integral's m/cm3 convention.
                import MDSplus

                if STOP.exists():
                    raise RuntimeError("authentication stop flag")
                node = MDSplus.Tree("bci", shot, "readonly").getNode(r"\BCI::DENV2")
                units = str(node.units)
                arrays["density_v2__node"] = np.array(str(node.fullpath))
            # 0.1 ms retains 2 ms ELM excursions; density needs only 10 ms.
            if name != "r0":
                t, y = block_mean(t, y, 0.1 if name.startswith("fs") else 10.0)
            arrays[f"{name}__t"], arrays[f"{name}__y"] = t, y.astype(np.float32)
            arrays[f"{name}__units"] = np.array(units)
            status[name] = f"ok {len(y)} {units}"
            if name == "density_v2":
                si = signals.density_line_si(y, units)
                if si is not None:
                    arrays["density_v2_si__t"] = t
                    arrays["density_v2_si__y"] = si
                    arrays["density_v2_si__units"] = np.array("m^-2")
                    status["density_v2_si"] = f"unit-confirmed conversion from {units}"
            row[name] = status[name]
        except Exception as error:  # noqa: BLE001  absent diagnostics are recorded
            message = f"{type(error).__name__}: {error}"[:200]
            status[name] = "error " + message
            row[name] = status[name]
            if STOP.exists() or is_auth_error(message):
                STOP.touch()
                return {**row, "auth": True}
    temporary = path.with_name(f".{path.name}.physics.npz")
    np.savez_compressed(temporary, status=json.dumps(status), **arrays)
    temporary.replace(path)
    maps_path = ROOT / "efit" / f"{shot}.npz"
    sparse = False
    if maps_path.exists():
        with np.load(maps_path) as f:
            sparse = len(f["gtime_ms"]) < 2
    if sparse or (ensure_flux and not maps_path.exists()):
        try:
            maps = fetch_maps(shot, "efit01")
            if len(maps["gtime_ms"]) < 2:
                raise ValueError("EFIT01 also has fewer than two slices")
            temporary = maps_path.with_name(f".{maps_path.name}.physics.npz")
            np.savez_compressed(temporary, **maps)
            temporary.replace(maps_path)
            row["flux_map"] = f"EFIT01 {len(maps['gtime_ms'])} slices"
        except Exception as error:  # noqa: BLE001  leave previous map intact
            message = f"{type(error).__name__}: {error}"[:200]
            row["flux_map"] = "error " + message
            if STOP.exists() or is_auth_error(message):
                STOP.touch()
                return {**row, "auth": True}
    row["status"] = "ok"
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shots-file", type=Path)
    parser.add_argument("--pace", type=float, default=1.0)
    parser.add_argument("--ensure-flux", action="store_true")
    args = parser.parse_args()
    if args.pace < 1:
        raise SystemExit("pace must be at least one second")
    if STOP.exists():
        raise SystemExit("AUTH ERROR: shared stop flag exists; fetching stopped")
    shots = (
        sorted(set(map(int, args.shots_file.read_text().split())))
        if args.shots_file
        else sorted(
            {
                int(p.stem)
                for name in ("inversions", "tv_surrogate")
                for p in (ROOT / name).glob("*.npz")
            }
        )
    )
    with (ROOT / "physics_fetch_log.jsonl").open("a") as log:
        for shot in shots:
            row = supplement(shot, args.ensure_flux)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
            if row.get("auth"):
                print("AUTH ERROR: stopping all fetching", flush=True)
                return 3
            time.sleep(args.pace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
