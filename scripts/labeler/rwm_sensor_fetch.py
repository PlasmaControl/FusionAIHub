"""Fetch PTDATA ``ONSBRADIAL`` (the n = 1 radial-field signal) for the RWM study shots.

``ONSBRADIAL`` is the radial-field point disruption-py reads where ``DUSBRADIAL`` is
unavailable; ``DUSBRADIAL`` (already cached by ``rwm_fetch.py``) is zero on 79 of the
2014 study shots and reported corrupted for 176030-176912, which holds every 2018
Hanson shot. The shots are the usable rows of ``$LABELER_ROOT/round4/rwm/shots.csv``
(33 Hanson and 174 comparison shots). Each shot's whole record goes to
``<shot>.npz`` (``t_ms`` float64, ``onsbradial`` float32, native units, native rate)
under ``$LABELER_ROOT/round5/rwm/sensor/onsbradial``. It stops at the first
authentication error, and every worker stops on the shared stop file.

A connection or service-lookup error (``getservbyname failed for task 'PTSERVER'``)
does not say the point is absent: the PTDATA client falls back to a live PTSERVER,
which no stellar node reaches, both when the archive's pointname index lacks the
point and when the archive lookup merely failed. Such a shot is checked against
that index (``PTDATA_JSON_INDEX_DIR``, one ``<shot>.json`` per shot listing every
point it holds, newest index version first). Only a shot whose index lists no such
point gets ``<shot>.missing.json``, with the index's evidence, and is not retried.
Otherwise it is retried once after a pause and, failing again, left unmarked for
the next run; six such shots in a row stop the worker. A marker without index
evidence (written before this rule) is retried. Any other fetch error is
permanent, as before.

Needs fdp, so run it on the login node while logged in, at most three parts::

    TMPDIR=$LABELER_ROOT/scratch/tmp-fetch pixi run --frozen -e labeler fdp run \\
        python scripts/labeler/rwm_sensor_fetch.py --part 0/3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.events.verify import NoDataError, fdp_signal

LABELER = Path(os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labeler"))
SHOTS = LABELER / "round4/rwm/shots.csv"
OUT = LABELER / "round5/rwm/sensor/onsbradial"
POINT = "ONSBRADIAL"
AUTH_WORDS = (
    "auth",
    "credential",
    "kerberos",
    "permission denied",
    "expired",
    "kinit",
    "login failed",
    "login required",
    "not logged in",
)
#: Errors that say the PTDATA client could not reach a server, not that the
#: point is absent (`labeler.events.raw.RETRY_DELAY_S`'s note).
CONNECTION_WORDS = (
    "getservbyname",
    "ptserver",
    "network connection",
    "connection refused",
    "connection reset",
    "timed out",
    "timeout",
)
RETRY_DELAY_S = 2.0
MAX_TRANSIENT_IN_A_ROW = 6


def study_shots(path: Path = SHOTS) -> list[int]:
    """The usable RWM study shots (Hanson and comparison)."""
    frame = pd.read_csv(path)
    return sorted(int(shot) for shot in frame[frame.usable.astype(bool)].shot)


def _pelican(*args: str) -> str:
    return subprocess.run(
        ["pelican", "object", *args],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    ).stdout


def index_evidence(shot: int, point: str = POINT) -> dict | None:
    """What the PTDATA pointname index (the one fdp points the client at) lists
    for `shot`: the index version, its point count, whether it lists `point` and
    ``DUSBRADIAL``, and their file extensions; None when no index version holds
    the shot."""
    root = os.environ["PTDATA_JSON_INDEX_DIR"].rstrip("/")
    pattern = os.environ.get("PTDATA_JSON_INDEX_PATTERN", "json_indexes_*")
    prefix = pattern.rstrip("*")
    try:
        listing = _pelican("ls", f"{root}/")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    versions = sorted((v.strip().rstrip("/") for v in listing.split()), reverse=True)
    for version in (v for v in versions if v.startswith(prefix)):
        url = f"{root}/{version}/{shot // 100:04d}/{shot}.json"
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / f"{shot}.json"
            try:
                _pelican("get", url, str(local))
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                continue
            index = json.loads(local.read_text())
        points = {str(k).upper(): v for k, v in index["pointname_ext"].items()}
        return {
            "index": f"{root}/{version}",
            "points_in_index": len(points),
            "point_in_index": point in points,
            "point_file": points.get(point),
            "dusbradial_in_index": "DUSBRADIAL" in points,
            "dusbradial_file": points.get("DUSBRADIAL"),
        }
    return None


def _settled(marker: Path) -> bool:
    """A marker counts only when it says why the point is absent (`reason`);
    markers from before the connection rule do not."""
    try:
        return "reason" in json.loads(marker.read_text())
    except (OSError, ValueError):
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shots", type=int, nargs="+", help="these shots instead")
    ap.add_argument("--part", default="0/1", help="k/n: every n-th shot, offset k")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--pace", type=float, default=1.0, help="seconds between shots")
    args = ap.parse_args(argv)
    shots = sorted(set(args.shots)) if args.shots else study_shots()
    k, n = (int(v) for v in args.part.split("/"))
    if n > 3:
        ap.error("at most 3 parts")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stop = args.out_dir / "fetch_auth_stop.json"
    transient = 0
    for shot in shots[k::n]:
        if stop.exists():
            print(json.dumps({"stopped": "authentication in another worker"}))
            return 2
        done = args.out_dir / f"{shot}.npz"
        gone = args.out_dir / f"{shot}.missing.json"
        if done.exists() or (gone.exists() and _settled(gone)):
            continue
        started = time.time()
        record, message, evidence = None, "", None
        for attempt in (1, 2):
            try:
                record = fdp_signal(shot, [POINT], via="ptdata")
                break
            except NoDataError as error:
                message = str(error)
            if any(word in message.lower() for word in AUTH_WORDS):
                stop.write_text(json.dumps({"shot": shot, "error": message[:300]}))
                print(json.dumps({"shot": shot, "stopped": "authentication"}))
                return 2
            if not any(word in message.lower() for word in CONNECTION_WORDS):
                break
            evidence = index_evidence(shot)
            if evidence is not None and not evidence["point_in_index"]:
                break
            if attempt == 1:
                time.sleep(RETRY_DELAY_S)
        if record is None:
            connection = any(word in message.lower() for word in CONNECTION_WORDS)
            if connection and (evidence is None or evidence["point_in_index"]):
                transient += 1
                print(
                    json.dumps(
                        {
                            "shot": shot,
                            "status": "transient, not marked",
                            "index": evidence,
                            "error": message[:200],
                        }
                    ),
                    flush=True,
                )
                if transient >= MAX_TRANSIENT_IN_A_ROW:
                    print(json.dumps({"stopped": f"{transient} transient in a row"}))
                    return 3
                time.sleep(args.pace)
                continue
            gone.write_text(
                json.dumps(
                    {
                        "shot": shot,
                        "point": POINT,
                        "reason": (
                            "not in the PTDATA pointname index"
                            if connection
                            else "fetch error"
                        ),
                        "error": message[:400],
                        **(evidence if connection else {}),
                    }
                )
                + "\n"
            )
            print(json.dumps({"shot": shot, "status": "missing"}), flush=True)
            transient = 0
            time.sleep(args.pace)
            continue
        transient = 0
        y = np.asarray(record.y[0], dtype=np.float32)
        np.savez(done, t_ms=np.asarray(record.x, dtype=np.float64), onsbradial=y)
        print(
            json.dumps(
                {
                    "shot": shot,
                    "status": "ok",
                    "samples": int(y.size),
                    "nonzero": int((np.isfinite(y) & (y != 0)).sum()),
                    "seconds": round(time.time() - started, 1),
                }
            ),
            flush=True,
        )
        time.sleep(args.pace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
