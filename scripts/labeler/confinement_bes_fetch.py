#!/usr/bin/env python
"""Fetch the native-rate BES of the curated confinement shots and write window features.

Gill et al. (2024) train on BES at 1 MHz; the corpus holds it at 500 kHz, and only for
the 119 curated shots that lie inside the corpus' shot range. This fetches PTDATA
``BESFU01``-``BESFU64`` (1 MS/s, 64 channels) and the 15L and 15R beams' injected power
(MDSplus ``\\D3D::TOP.NB.NB15L:PINJ_15L``, watts) for the shots of the curated
confinement intervals, and writes ONE FILE PER SHOT, ``<shot>.npz``, holding only the
windows inside labelled intervals: the log-spectral features of
``labeler.confinement.bes_features`` (float16, ``(n, 2, 64, 128)``), each window's mean
squared band-passed signal per channel, its start time, the interval it lies in and the
beam powers over it. Raw samples are not kept (about 1.5 GB a shot).

A shot whose BES fdp does not hold gets ``<shot>.missing.json`` and is not retried; a
failed fetch is tried again on the next run. It stops at the first authentication error,
so run it on the login node under fdp while logged in::

    pixi run --frozen -e labelmaker fdp run python \\
        scripts/labeler/confinement_bes_fetch.py --part 0/3 --pace 1

The cohort's blind ``test`` shots are never fetched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.confinement import bes_features as bf
from labeler.confinement import bes_windows as bw

LABELER = Path(
    os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
)
DEFAULT_OUT = LABELER / "round4/conf/bes1mhz"
CLASSES = ("L", "H", "QH", "WP")
CHANNELS = 64
FS = 1e6
STRIDE = 2048
BEAMS = {"15L": r"\D3D::TOP.NB.NB15L:PINJ_15L", "15R": r"\D3D::TOP.NB.NB15R:PINJ_15R"}
#: How PTDATA reports a point the shot does not hold (no BES): the same message as a
#: missing server entry, so a canary fetch at the start of a run rules out a lapsed
#: login.
ABSENT_WORDS = ("getservbyname", "NODATA", "NNF", "No data")
AUTH_WORDS = ("auth", "credential", "kerberos", "permission denied", "expired", "kinit")
CANARY = (149992, "BESFU02")
MAX_CONSECUTIVE_FAILURES = 8


def beam_trace(shot: int, expr: str) -> tuple[np.ndarray, np.ndarray] | None:
    """One beam's injected power: time in ms and watts, or None if the shot lacks it."""
    from labeler.features.resolve_fdp import _fetch_mds

    try:
        rec = _fetch_mds(expr, "D3D", int(shot), dims=["dim0"])
    except Exception as error:
        if type(error).__name__ in ("TreeNODATA", "TreeNNF"):
            return None
        raise
    return np.asarray(rec["dim0"], dtype=np.float64), np.asarray(
        rec["data"], dtype=np.float32
    )


def fetch_bes(shot: int) -> tuple[np.ndarray, float, float, int]:
    """All 64 channels of one shot as float32 (64, T); absent channels are NaN rows.

    Returns the array, the first sample's time in ms, the sample step in ms and the
    number of channels that came back empty.
    """
    from labeler.features.resolve_fdp import _fetch_ptdata

    rows: list[np.ndarray | None] = []
    t0 = dt = None
    for c in range(CHANNELS):
        name = f"BESFU{c + 1:02d}"
        record = None
        for attempt in (1, 2):
            try:
                record = _fetch_ptdata(name, int(shot))
                break
            except Exception as error:
                message = str(error)
                if any(w in message.lower() for w in AUTH_WORDS):
                    raise
                if attempt == 2 or any(w in message for w in ABSENT_WORDS):
                    break
                time.sleep(3.0)
        if record is None:
            rows.append(None)
            continue
        times = np.asarray(record["times"], dtype=np.float64)
        if record["units"].get("times") != "ms":
            raise ValueError(f"{shot} {name}: time unit {record['units']}")
        if t0 is None:
            t0, dt = float(times[0]), float((times[-1] - times[0]) / (times.size - 1))
            if abs(dt * FS / 1000.0 - 1.0) > 0.01:
                raise ValueError(f"{shot}: BES step {dt * 1000:.4f} us, not 1 us")
            length = times.size
        data = np.asarray(record["data"], dtype=np.float32)
        if data.size != length:
            rows.append(None)
            continue
        rows.append(data)
    if t0 is None:
        raise LookupError(f"{shot}: no BES channel came back")
    empty = sum(r is None for r in rows)
    nan = np.full(length, np.nan, dtype=np.float32)
    return np.stack([nan if r is None else r for r in rows]), t0, dt, empty


def process_shot(shot: int, intervals: pd.DataFrame) -> dict:
    """Fetch one shot and return the arrays to write."""
    bes, t0, dt, empty = fetch_bes(shot)
    out = bw.shot_windows(intervals, bes, t0, dt, FS, STRIDE)
    if not out:
        raise LookupError(
            f"{shot}: no window of {len(intervals)} intervals lies in the record"
        )
    span = bf.WINDOW * dt
    for name, expr in BEAMS.items():
        trace = beam_trace(shot, expr)
        t_ms, y = trace if trace else (None, None)
        for stat, values in bw.window_beam_power(
            t_ms, y, out["start_ms"], span
        ).items():
            out[f"p{name}_{stat}"] = values
    out["record"] = np.array([t0, dt, bes.shape[1], empty])
    out["channel_std"] = np.nanstd(bes[:, :: max(1, bes.shape[1] // 200_000)], axis=1)
    return out


def shot_order(shots: list[int], order_file: Path | None) -> list[int]:
    """The shots in the order of ``order_file`` (one per line), the rest after,
    ascending."""
    if order_file is None:
        return sorted(shots)
    listed = [int(v) for v in order_file.read_text().split()]
    wanted = set(shots)
    first = [s for s in dict.fromkeys(listed) if s in wanted]
    return first + sorted(wanted - set(first))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--shots", type=int, nargs="+", help="these shots only")
    ap.add_argument("--part", default="0/1", help="k/n: every n-th shot of the order")
    ap.add_argument("--pace", type=float, default=1.0, help="seconds between shots")
    ap.add_argument(
        "--order",
        type=Path,
        default=None,
        help="file of shots, one per line: fetch these first",
    )
    ap.add_argument("--no-canary", action="store_true")
    args = ap.parse_args(argv)

    table = bw.curated_intervals()
    shots = sorted(int(s) for s in table.shot.unique())
    if args.shots:
        shots = [s for s in args.shots if s in set(shots)]
    else:
        shots = shot_order(shots, args.order)
    k, n = (int(v) for v in args.part.split("/"))
    shots = shots[k::n]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_canary:
        from labeler.features.resolve_fdp import _fetch_ptdata

        try:
            _fetch_ptdata(CANARY[1], CANARY[0])
        except Exception as error:  # noqa: BLE001
            print(json.dumps({"stopped": "canary", "error": str(error)[:300]}))
            return 2
    failures = 0
    for shot in shots:
        done = args.out_dir / f"{shot}.npz"
        gone = args.out_dir / f"{shot}.missing.json"
        if done.exists() or gone.exists():
            continue
        started = time.time()
        try:
            arrays = process_shot(shot, table[table.shot == shot])
        except (LookupError, ValueError) as error:
            gone.write_text(json.dumps({"shot": shot, "error": str(error)[:500]}))
            print(
                json.dumps(
                    {"shot": shot, "status": "missing", "error": str(error)[:160]}
                ),
                flush=True,
            )
            continue
        except Exception as error:  # noqa: BLE001
            message = str(error)
            if any(w in message.lower() for w in AUTH_WORDS):
                print(
                    json.dumps(
                        {
                            "shot": shot,
                            "stopped": "authentication",
                            "error": message[:300],
                        }
                    )
                )
                return 2
            failures += 1
            print(
                json.dumps({"shot": shot, "status": "failed", "error": message[:160]}),
                flush=True,
            )
            if failures >= MAX_CONSECUTIVE_FAILURES:
                print(json.dumps({"stopped": f"{failures} failures in a row"}))
                return 3
            continue
        failures = 0
        tmp = done.with_suffix(".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.rename(done)
        print(
            json.dumps(
                {
                    "shot": shot,
                    "status": "fetched",
                    "windows": int(arrays["feats"].shape[0]),
                    "empty_channels": int(arrays["record"][3]),
                    "seconds": round(time.time() - started, 1),
                }
            ),
            flush=True,
        )
        time.sleep(args.pace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
