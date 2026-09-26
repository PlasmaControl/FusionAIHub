"""The assessed window and the Ip flat-top (inclusion rule 4), from high-rate Ip.

Ip is the PTDATA point `ip`, in amperes, read through `raw.raw_signal(shot, "ip")`:
the first read fetches it live and parks the whole record in the raw cache. Its
native step is 0.5 ms on the 2021 shots and 0.05 ms from about 189,000 on.

The window is the plasma's stretch of |Ip| >= 50 kA. It is the longest one, with
gaps under `BRIDGE_MS` bridged, not simply the first to the last sample over 50 kA:
the ohmic pre-magnetisation puts a pickup pulse of 30-47 kA on Ip about 0.9 s
before breakdown (measured on 5 shots), and one sample over 50 kA there would open
the window a second early.

Measuring a shot list is a login-node job, under the fdp wrapper:

    pixi run -e labelmaker fdp run python -m labeler.events.catalog.window \\
        --shot-file <one shot per line> [--workers 4] [--limit 20]

Each shot appends one line to `$LABELER_ROOT/catalog/ip.jsonl`; a rerun skips the
shots already settled and retries the errors.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from shot_design.shotdb.select import MIN_FLATTOP_S, flattop_from_ip

from ...catalog import read_shot_file
from ...config import Paths
from ..raw import raw_signal
from ..verify import NoDataError

IP_GROUP = "ip"
WINDOW_IP_A = 50e3
BRIDGE_MS = 10.0
#: A shot with one of these is measured for good; an "error" is tried again.
SETTLED = ("ok", "no_plasma")
LOG_COLUMNS = (
    "shot",
    "status",
    "window_start_ms",
    "window_end_ms",
    "flattop_s",
    "ip_peak_ma",
    "dt_ms",
    "n",
    "error",
)


def assessed_window(
    t_ms, ip_a, *, threshold_a: float = WINDOW_IP_A, bridge_ms: float = BRIDGE_MS
) -> tuple[int, int] | None:
    """The plasma's stretch of |Ip| >= threshold, rounded inward to whole ms.

    None when Ip never gets there. NaN samples count as below the threshold. The
    edges are taken to the us before rounding, so a sample on a whole ms counts as
    that ms: time bases carry float rounding (the raw cache's float32 seconds put
    shot 200111's last sample over 50 kA at 5348.00005 ms, fdp's at 5347.99999).
    """
    t = np.asarray(t_ms, dtype=float).ravel()
    y = np.abs(np.asarray(ip_a, dtype=float).ravel())
    times = t[y >= threshold_a]
    if not times.size:
        return None
    breaks = np.flatnonzero(np.diff(times) > bridge_ms)
    firsts, lasts = np.r_[0, breaks + 1], np.r_[breaks, times.size - 1]
    k = int(np.argmax(times[lasts] - times[firsts]))
    first, last = (round(float(times[i]), 3) for i in (firsts[k], lasts[k]))
    start, end = math.ceil(first), math.floor(last)
    return (start, end) if start < end else None


def flattop_s(t_ms, ip_a, window: tuple[int, int]) -> float:
    """`select.flattop_from_ip` inside the window, so the record's length can't move it.

    Its plateau is the 95th percentile of |Ip|; over a whole record, a long enough
    run of pre- and post-shot zeros drags that percentile down onto the ramps.
    """
    t = np.asarray(t_ms, dtype=float).ravel()
    inside = (t >= window[0]) & (t <= window[1])
    ip = np.asarray(ip_a, dtype=float).ravel()
    return flattop_from_ip(t[inside] / 1000.0, ip[inside])


def _finite(x: float, digits: int) -> float | None:
    return round(float(x), digits) if math.isfinite(x) else None


def summarise(shot: int, t_ms, ip_a) -> dict:
    """One log line for a shot's Ip record."""
    t = np.asarray(t_ms, dtype=float).ravel()
    line = {
        "shot": int(shot),
        "n": int(t.size),
        # To 1 us: the raw cache keeps times as float32 seconds, so its spacings
        # carry about a microsecond of rounding (0.05 ms steps read 0.050008).
        "dt_ms": _finite(np.median(np.diff(t)), 3) if t.size > 1 else None,
    }
    window = assessed_window(t, ip_a)
    if window is None:
        return line | {"status": "no_plasma"}
    ip = np.abs(np.asarray(ip_a, dtype=float).ravel())
    peak = np.nanmax(ip[(t >= window[0]) & (t <= window[1])])
    return line | {
        "status": "ok",
        "window_start_ms": window[0],
        "window_end_ms": window[1],
        "flattop_s": _finite(flattop_s(t, ip_a, window), 4),
        "ip_peak_ma": _finite(peak / 1e6, 4),
    }


def measure(shot: int, paths: Paths) -> dict:
    """One shot's log line, fetching its Ip if the cache lacks it.

    A failure is the shot's `error` line, whatever raised it, so one bad shot never
    stops a run over thousands; a rerun tries it again.
    """
    try:
        record = raw_signal(shot, IP_GROUP, paths=paths)
    except NoDataError as error:
        return _error_line(shot, str(error))
    except Exception as error:  # noqa: BLE001 - logged for the shot, then retried
        return _error_line(shot, f"{type(error).__name__}: {error}")
    return summarise(shot, record.x, record.y[0]) | {"tier": record.attrs["tier"]}


def _error_line(shot: int, message: str) -> dict:
    return {"shot": int(shot), "status": "error", "error": message[:300]}


def read_log(path) -> pd.DataFrame:
    """The last line per shot, in shot order; empty when there is no log."""
    path = Path(path)
    last = {}
    if path.is_file():
        for text in path.read_text(encoding="utf-8").splitlines():
            if text.strip():
                line = json.loads(text)
                last[int(line["shot"])] = line
    frame = pd.DataFrame([last[s] for s in sorted(last)], columns=list(LOG_COLUMNS))
    for column in ("flattop_s", "ip_peak_ma", "dt_ms"):
        frame[column] = pd.to_numeric(frame[column]).astype(float)
    for column in ("shot", "window_start_ms", "window_end_ms", "n"):
        frame[column] = pd.to_numeric(frame[column]).astype("Int64")
    return frame


def _lines(shots: list[int], paths: Paths, workers: int) -> Iterator[dict]:
    if workers <= 1:
        yield from (measure(shot, paths) for shot in shots)
        return
    # Forked before anything imports toksearch: its PTDATA reader is not fork-safe.
    with Pool(workers) as pool:
        yield from pool.imap_unordered(partial(measure, paths=paths), shots)


def fetch(
    shots: Iterable[int], log: Path, paths: Paths, *, workers: int = 1
) -> Counter:
    """Measure every shot not yet settled in `log`, appending a line for each."""
    table = read_log(log)
    settled = set(table.loc[table["status"].isin(SETTLED), "shot"])
    todo = [shot for shot in sorted(set(shots)) if shot not in settled]
    log.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    with log.open("a", encoding="utf-8") as out:
        for i, line in enumerate(_lines(todo, paths, workers), 1):
            line["written_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            out.write(json.dumps(line) + "\n")
            out.flush()
            counts[line["status"]] += 1
            if i % 100 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} {dict(counts)}", flush=True)
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m labeler.events.catalog.window",
        description="Measure each listed shot's assessed window and Ip flat-top.",
    )
    parser.add_argument("--shot-file", type=Path, required=True)
    parser.add_argument(
        "--log", type=Path, help="default $LABELER_ROOT/catalog/ip.jsonl"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="the first N shots only (a pilot)")
    args = parser.parse_args(argv)
    paths = Paths.from_env()
    log = args.log or paths.catalog / "ip.jsonl"
    shots = read_shot_file(args.shot_file)[: args.limit]
    counts = fetch(shots, log, paths, workers=args.workers)
    table = read_log(log)
    table = table[table["shot"].isin(shots)]
    summary = {
        "log": str(log),
        "shots": len(shots),
        "this_run": dict(counts),
        "status": table["status"].value_counts().to_dict(),
        "flattop_at_least_1s": int((table["flattop_s"] >= MIN_FLATTOP_S).sum()),
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
