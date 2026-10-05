#!/usr/bin/env python
"""Fetch EFIT02 geometry: two workers, one-second pacing, shared auth stop."""

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from labeler.features import resolve_fdp

ROOT = Path(os.environ["LABELER_ROOT"]) / "round4/detach"
STOP = ROOT / "fetch_auth_stop"


def fetch(shot):
    dest = ROOT / "geometry02" / f"{shot}.npz"
    if dest.exists():
        return {"shot": shot, "status": "cached"}
    row = {"shot": shot}
    try:
        arrays = {}
        for name in ("rvsod", "zvsod", "rxpt1", "zxpt1"):
            if STOP.exists():
                return {"shot": shot, "auth": True}
            rec = resolve_fdp._fetch_mds(
                rf"\efit02::top.results.aeqdsk:{name}", "efit02", shot
            )
            arrays[name] = np.asarray(rec["data"])
            arrays["t_ms"] = np.asarray(rec.get("times", rec.get("dim0")))
        np.savez_compressed(dest, **arrays)
        row["status"] = "ok"
    except Exception as error:  # noqa: BLE001  missing tree is recorded
        msg = f"{type(error).__name__}: {error}"[:200]
        row["status"] = msg
        if any(
            k in msg.lower()
            for k in (
                "auth",
                "login",
                "credential",
                "token",
                "401",
                "403",
                "permission",
            )
        ):
            STOP.touch()
            row["auth"] = True
    time.sleep(1)
    return row


def main():
    (ROOT / "geometry02").mkdir(exist_ok=True)
    attempted = set()
    logpath = ROOT / "geometry02_log.jsonl"
    if logpath.exists():
        attempted = {
            json.loads(line)["shot"] for line in logpath.read_text().splitlines()
        }
    shots = [
        s
        for s in map(int, (ROOT / "shots_fetch.txt").read_text().split())
        if s not in attempted
    ]
    with mp.Pool(2) as pool, logpath.open("a") as log:
        for row in pool.imap_unordered(fetch, shots):
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(row, flush=True)
            if row.get("auth"):
                pool.terminate()
                raise SystemExit(3)


if __name__ == "__main__":
    main()
