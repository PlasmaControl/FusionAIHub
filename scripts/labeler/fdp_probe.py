#!/usr/bin/env python
"""Probe explicit points serially through resolve_fdp; write metadata, not labels.

Run on a login node under ``pixi run -e labelmaker fdp run python ...``.
The candidates JSON is a list of objects with ``row``, ``kind`` (ptdata/mds),
``node``, ``found_by``, and (for MDSplus) ``tree``. The shots JSON is a list
of integers or objects containing ``shot``. Output is one JSON record per
point and shot, flushed immediately. No feature store is opened or written.
Use an explicit output path in /tmp for the actuation probe.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np

from labelmaker.features import resolve_fdp


def describe(values) -> dict:
    """Describe the actual record without retaining its waveform."""
    values = np.asarray(values)
    result = {"shape": list(values.shape), "dtype": str(values.dtype)}
    if np.issubdtype(values.dtype, np.number):
        finite = values[np.isfinite(values)]
        result["finite"] = int(finite.size)
        if finite.size:
            result.update(min=float(finite.min()), max=float(finite.max()))
    elif values.size <= 20:
        result["values"] = values.astype(str).tolist()
    return result


def probe(candidate: dict, shot: int) -> dict:
    """One raw resolver fetch; preserve errors without the store's truncation."""
    result = {**candidate, "shot": shot}
    started = time.monotonic()
    try:
        if candidate["kind"] == "ptdata":
            record = resolve_fdp._fetch_ptdata(candidate["node"], shot)
        else:
            record = resolve_fdp._fetch_mds(
                candidate["node"],
                candidate["tree"],
                shot,
            )
        result.update(
            status="returned",
            data=describe(record["data"]),
            units=record.get("units", {}),
            dimensions={
                key: describe(value)
                for key, value in record.items()
                if key not in ("data", "units", "n_over", "n_under")
            },
        )
        for key in ("n_over", "n_under"):
            if key in record:
                result[key] = int(record[key])
        try:
            data, times = resolve_fdp._scalar_axes(record, candidate["node"])
            steps = np.diff(times)
            result["scalar_timebase"] = {
                "n": int(times.size),
                "start_s": float(times[0]),
                "end_s": float(times[-1]),
                "dt_median_s": float(np.median(steps)),
                "dt_min_s": float(steps.min()),
                "dt_max_s": float(steps.max()),
                "finite_data": int(np.isfinite(data).sum()),
            }
        except (ValueError, TypeError) as exc:
            result["scalar_axis_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001 - retain arbitrary backend errors
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    result["elapsed_s"] = round(time.monotonic() - started, 4)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    shots = [
        int(s["shot"] if isinstance(s, dict) else s)
        for s in json.loads(args.shots.read_text())
    ]
    candidates = json.loads(args.candidates.read_text())
    if not shots or len(shots) > 20 or len(set(shots)) != len(shots):
        parser.error("provide 1–20 distinct shots for this bounded probe")
    for candidate in candidates:
        if candidate.get("kind") not in ("ptdata", "mds"):
            parser.error("candidate kind must be ptdata or mds")
        for key in ("row", "node", "found_by"):
            if not candidate.get(key):
                parser.error(f"candidate needs {key}")
        if candidate["kind"] == "mds" and not candidate.get("tree"):
            parser.error("MDSplus candidate needs tree")
    print(f"host={socket.gethostname()} resolver={resolve_fdp.__file__}", flush=True)
    # No pool or fork: toksearch/PTDATA state stays in this one process.
    with args.output.open("x", encoding="utf-8") as output:
        for shot in shots:
            for candidate in candidates:
                result = probe(candidate, shot)
                output.write(json.dumps(result) + "\n")
                output.flush()
                print(shot, candidate["node"], result["status"], flush=True)


if __name__ == "__main__":
    main()
