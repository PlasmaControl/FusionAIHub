"""The assessed window and the Ip flat-top (inclusion rule 4), from high-rate Ip.

Ip is the PTDATA point `ip`, in amperes, read through `raw.raw_signal(shot, "ip")`:
the first read fetches it live and parks the whole record in the raw cache. Its
native step is 0.5 ms through shot 187,328, and 0.05 ms from 188,349 on; no
pool shot lies between.

The window is the longest stretch of Ip >= 50 kA in the plasma's own direction,
with gaps of at most `BRIDGE_MS` bridged. Post-shot pickup of 51-107 kA about
0.9 s after the end (185779-185788, 189138, 189154), and a one-sample 54 kA spike
4.1 s after 188351, are why it is the longest stretch. The opposite-sign quench
tail of 189013, about -318 kA decaying over 1.1 s, is why the current is signed.

D2b ends that window at a restrike: after the longest flat-top stretch, a mean
below 30% of the plateau followed by one above 60% marks a second plasma. The
end is the lowest mean between them, rounded down to whole ms. Shot 204238 is
such a restrike; 200811's early dip precedes its flat-top and stays in the window.
Same-sign current above 50 kA after a quench stays in the window: 50 reviewed
shots extended more than 25 ms past t20, and 198958 by 362 ms. D2b ends the
window only at a restrike; it does not trim these post-quench tails.

The flat-top reads centred 25 ms means of |Ip| inside the window, matching the
feature grid on which its rule was calibrated. This keeps single noisy samples
and short dips from splitting the plateau and reduces sampling-rate dependence.

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
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from datetime import UTC, datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from shot_design.shotdb.select import FLATTOP_FRACTION, MIN_FLATTOP_S, flattop_from_ip

from ...catalog import read_shot_file
from ...config import Paths, atomic_path, git_sha, sha256_of
from ..raw import raw_signal
from ..verify import NoDataError
from .check import CatalogError

IP_GROUP = "ip"
WINDOW_IP_A = 50e3
#: Samples more than BRIDGE_MS apart split a stretch; exactly 10 ms is bridged.
BRIDGE_MS = 10.0
FLATTOP_MEAN_MS = 25.0
RESTRIKE_DIP_FRACTION = 0.3
RESTRIKE_RISE_FRACTION = 0.6
LOG_VERSION = 3
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
    "version",
)


def assessed_window(
    t_ms, ip_a, *, threshold_a: float = WINDOW_IP_A, bridge_ms: float = BRIDGE_MS
) -> tuple[int, int] | None:
    """The plasma's signed stretch above threshold, rounded inward to whole ms.

    Gaps of at most BRIDGE_MS are bridged. None when Ip never gets there.
    Non-finite samples count as below the threshold. The
    edges are taken to the us before rounding, so a sample on a whole ms counts as
    that ms: time bases carry float rounding (the raw cache's float32 seconds put
    shot 200111's last sample over 50 kA at 5348.00005 ms, fdp's at 5347.99999).
    """
    t = np.asarray(t_ms, dtype=float).ravel()
    y = np.asarray(ip_a, dtype=float).ravel()
    finite = np.isfinite(y)
    if not finite.any():
        return None
    samples = y[finite]
    sign = 1 if samples[np.argmax(np.abs(samples))] >= 0 else -1
    times = t[finite & (sign * y >= threshold_a)]
    if not times.size:
        return None
    breaks = np.flatnonzero(np.diff(times) > bridge_ms)
    firsts, lasts = np.r_[0, breaks + 1], np.r_[breaks, times.size - 1]
    k = int(np.argmax(times[lasts] - times[firsts]))
    first, last = (round(float(times[i]), 3) for i in (firsts[k], lasts[k]))
    start, end = math.ceil(first), math.floor(last)
    return (start, end) if start < end else None


def _centred_mean(values: np.ndarray, half_width: int) -> np.ndarray:
    """Finite-sample means; NaN where the full centred span cannot be measured."""
    width = 2 * half_width + 1
    result = np.full(values.size, np.nan)
    if width > values.size:
        return result
    finite = np.isfinite(values)
    total = np.r_[0.0, np.cumsum(np.where(finite, values, 0.0))]
    count = np.r_[0, np.cumsum(finite)]
    totals, counts = total[width:] - total[:-width], count[width:] - count[:-width]
    np.divide(
        totals,
        counts,
        out=result[half_width : values.size - half_width],
        where=counts > 0,
    )
    return result


def flattop_s(
    t_ms, ip_a, window: tuple[int, int], *, mean_ms: float = FLATTOP_MEAN_MS
) -> float:
    """`select.flattop_from_ip` on centred means of |Ip| inside the window.

    Its plateau is the 95th percentile of |Ip|; over a whole record, a long enough
    run of pre- and post-shot zeros drags that percentile down onto the ramps.
    """
    t = np.asarray(t_ms, dtype=float).ravel()
    inside = (t >= window[0]) & (t <= window[1])
    ip = np.asarray(ip_a, dtype=float).ravel()
    t, ip = t[inside], np.abs(ip[inside])
    if t.size < 2:
        return float("nan")
    half_width = round(mean_ms / 2 / float(np.median(np.diff(t))))
    return flattop_from_ip(t / 1000.0, _centred_mean(ip, half_width))


def _finite(x: float, digits: int) -> float | None:
    return round(float(x), digits) if math.isfinite(x) else None


def restrike_end(t_ms, ip_a, window: tuple[int, int]) -> int | None:
    """D2b end at the dip before a restrike after the longest flat-top, else None."""
    t = np.asarray(t_ms, dtype=float).ravel()
    inside = (t >= window[0]) & (t <= window[1])
    ip = np.asarray(ip_a, dtype=float).ravel()
    t, ip = t[inside], np.abs(ip[inside])
    if t.size < 2:
        return None
    half_width = round(FLATTOP_MEAN_MS / 2 / float(np.median(np.diff(t))))
    means = _centred_mean(ip, half_width)
    keep = np.isfinite(t) & np.isfinite(means)
    t, means = t[keep], means[keep]
    if t.size < 3:
        return None
    plateau = float(np.percentile(means, 95))
    if plateau <= 0:
        return None
    above = means >= FLATTOP_FRACTION * plateau
    edges = np.diff(np.r_[False, above, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    # Like flattop_from_ip, compare durations in seconds; argmax keeps the first tie.
    stretch_end = ends[np.argmax(t[ends] / 1000 - t[starts] / 1000)]
    dips = np.flatnonzero(means[stretch_end + 1 :] < RESTRIKE_DIP_FRACTION * plateau)
    if not dips.size:
        return None
    dip = stretch_end + 1 + dips[0]
    rises = np.flatnonzero(means[dip + 1 :] > RESTRIKE_RISE_FRACTION * plateau)
    if not rises.size:
        return None
    rise = dip + 1 + rises[0]
    lowest = dip + np.argmin(means[dip : rise + 1])
    return math.floor(float(t[lowest]))


def summarise(shot: int, t_ms, ip_a) -> dict:
    """One log line for a shot's Ip record; ValueError if it cannot be measured."""
    t = np.asarray(t_ms, dtype=float).ravel()
    ip = np.asarray(ip_a, dtype=float).ravel()
    if t.size != ip.size:
        raise ValueError("times and current have different lengths")
    if t.size < 2:
        raise ValueError("fewer than two samples")
    if not np.isfinite(t).all():
        raise ValueError("non-finite time")
    if not (np.diff(t) > 0).all():
        raise ValueError("times are not strictly increasing")
    if not np.isfinite(ip).any():
        raise ValueError("no finite current")
    line = {
        "shot": int(shot),
        "version": LOG_VERSION,
        "n": int(t.size),
        # To 1 us: the raw cache keeps times as float32 seconds, so its spacings
        # carry about a microsecond of rounding (0.05 ms steps read 0.050008).
        "dt_ms": _finite(np.median(np.diff(t)), 3) if t.size > 1 else None,
    }
    window = assessed_window(t, ip_a)
    if window is None:
        return line | {"status": "no_plasma"}
    end = restrike_end(t, ip_a, window)
    if end is not None:
        window = (window[0], end)
    ip = np.abs(ip)
    peak = np.max(ip[(t >= window[0]) & (t <= window[1]) & np.isfinite(ip)])
    flat = flattop_s(t, ip_a, window)
    return line | {
        "status": "ok",
        "window_start_ms": window[0],
        "window_end_ms": window[1],
        "flattop_s": float(flat) if math.isfinite(flat) else None,
        "ip_peak_ma": _finite(peak / 1e6, 4),
    }


def measure(shot: int, paths: Paths) -> dict:
    """One shot's log line, fetching its Ip if the cache lacks it.

    A failure is the shot's `error` line, whatever raised it, so one bad shot never
    stops a run over thousands; a rerun tries it again.
    """
    try:
        record = raw_signal(shot, IP_GROUP, paths=paths)
        return summarise(shot, record.x, record.y[0]) | {"tier": record.attrs["tier"]}
    except NoDataError as error:
        return _error_line(shot, str(error))
    except Exception as error:  # noqa: BLE001 - logged for the shot, then retried
        return _error_line(shot, f"{type(error).__name__}: {error}")


def _error_line(shot: int, message: str) -> dict:
    return {
        "shot": int(shot),
        "status": "error",
        "error": message[:300],
        "version": LOG_VERSION,
    }


def _finite_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _validate_line(line: dict) -> None:
    """Validate one complete decoded line before it can supersede a measurement."""
    if not isinstance(line, dict) or type(line.get("shot")) is not int:
        raise ValueError("shot must be an int")
    if line.get("status") not in (*SETTLED, "error"):
        raise ValueError("status must be ok, no_plasma or error")
    version = line.get("version", 1)
    if type(version) is not int or version < 1:
        raise ValueError("version must be an int >= 1")
    if line["status"] != "ok":
        return
    start, end = line.get("window_start_ms"), line.get("window_end_ms")
    if type(start) is not int or type(end) is not int or start >= end:
        raise ValueError("window ends must be ints with start < end")
    flat = line.get("flattop_s")
    if flat is not None and (
        not _finite_number(flat) or not 0 <= flat <= (end - start) / 1000 + 1e-6
    ):
        raise ValueError("flattop_s must be null or finite and within the window")
    if not _finite_number(line.get("ip_peak_ma")):
        raise ValueError("ip_peak_ma must be finite")
    dt = line.get("dt_ms")
    if dt is not None and (not _finite_number(dt) or dt <= 0):
        raise ValueError("dt_ms must be null or finite and > 0")


def read_log(path) -> pd.DataFrame:
    """Last complete line per shot; skip torn tails, refuse corrupt complete lines."""
    stream = hasattr(path, "read")
    if not stream:
        path = Path(path)
    last = {}
    if stream or path.is_file():
        with nullcontext(path) if stream else path.open("rb") as source:
            for number, text in enumerate(source, 1):
                if not text.endswith(b"\n"):
                    break
                if not text.strip():
                    continue
                try:
                    line = json.loads(text)
                    _validate_line(line)
                except (ValueError, UnicodeDecodeError) as error:
                    raise CatalogError(f"{path}:{number}: {error}") from error
                line.setdefault("version", 1)
                last[line["shot"]] = line
    frame = pd.DataFrame([last[s] for s in sorted(last)], columns=list(LOG_COLUMNS))
    for column in ("flattop_s", "ip_peak_ma", "dt_ms"):
        frame[column] = pd.to_numeric(frame[column]).astype(float)
    for column in ("shot", "window_start_ms", "window_end_ms", "n", "version"):
        frame[column] = pd.to_numeric(frame[column]).astype("Int64")
    return frame


def _lines(shots: list[int], paths: Paths, workers: int) -> Iterator[dict]:
    if workers <= 1:
        yield from (measure(shot, paths) for shot in shots)
        return
    if "toksearch" in sys.modules or "toksearch_d3d" in sys.modules:
        raise RuntimeError("cannot fork workers after importing toksearch")
    with Pool(workers) as pool:
        yield from pool.imap_unordered(partial(measure, paths=paths), shots)


def _set_aside_torn(path: Path) -> None:
    """Preserve a partial append separately before the next complete line is added."""
    if not path.is_file():
        return
    data = path.read_bytes()
    end = data.rfind(b"\n") + 1
    if end == len(data):
        return
    torn = path.with_name(path.name + ".torn")
    with torn.open("ab") as out:
        out.write(data[end:] + b"\n")
    with path.open("r+b") as out:
        out.truncate(end)
    print(f"Set aside torn segment from {path} in {torn}", flush=True)


def fetch(
    shots: Iterable[int], log: Path, paths: Paths, *, workers: int = 1
) -> Counter:
    """Measure every shot not yet settled in `log`, appending a line for each."""
    table = read_log(log)
    settled = set(
        table.loc[
            table["status"].isin(SETTLED) & table["version"].eq(LOG_VERSION), "shot"
        ]
    )
    todo = [shot for shot in sorted(set(shots)) if shot not in settled]
    log.parent.mkdir(parents=True, exist_ok=True)
    _set_aside_torn(log)
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


def definition() -> dict:
    """The measurement recorded by each current-version Ip log line."""
    return {
        "log_version": LOG_VERSION,
        "ip": "PTDATA ip, native rate",
        "window": (
            "Longest stretch of s * Ip >= window_ip_a, where s is the sign of the "
            "finite sample of largest |Ip| (zero is positive), bridging gaps of "
            "at most bridge_ms and rounding edges to us then inward to whole ms."
        ),
        "window_ip_a": WINDOW_IP_A,
        "bridge_ms": BRIDGE_MS,
        "flattop": (
            "select.flattop_from_ip on centred flattop_mean_ms means of finite "
            "|Ip| inside the window, with h = round(flattop_mean_ms / 2 / median "
            "dt), over i-h through i+h, and NaN for incomplete or empty spans."
        ),
        "flattop_mean_ms": FLATTOP_MEAN_MS,
        "flattop_fraction": FLATTOP_FRACTION,
        "min_flattop_s": MIN_FLATTOP_S,
        "restrike": (
            "Inside the D2a window use the same centred means as flattop; drop "
            "non-finite means and take their 95th percentile P. After the longest "
            "run >= flattop_fraction * P (first on a tie), find the first mean "
            "< restrike_dip_fraction * P and the first later mean "
            "> restrike_rise_fraction * P. End at the lowest mean between them, "
            "rounded down to whole ms; remeasure flat-top and peak inside it."
        ),
        "restrike_dip_fraction": RESTRIKE_DIP_FRACTION,
        "restrike_rise_fraction": RESTRIKE_RISE_FRACTION,
    }


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
    try:
        return _run(args, Paths.from_env())
    except CatalogError as error:
        parser.error(str(error))


def _run(args, paths) -> int:
    log = args.log or paths.catalog / "ip.jsonl"
    shots = read_shot_file(args.shot_file)[: args.limit]
    counts = fetch(shots, log, paths, workers=args.workers)
    meta = {
        "version": LOG_VERSION,
        "git_sha": git_sha(full=True),
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "log": str(log),
        "shot_file": str(args.shot_file),
        "shot_file_sha256": sha256_of(args.shot_file),
        "shots": len(shots),
        "this_run": dict(counts),
        "definition": definition(),
    }
    with atomic_path(log.with_suffix(".meta.json")) as tmp:
        Path(tmp).write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    table = read_log(log)
    table = table[table["shot"].isin(shots)]
    summary = {
        "log": str(log),
        "shots": len(shots),
        "this_run": dict(counts),
        "status": table["status"].value_counts().to_dict(),
        "versions": {
            int(version): int(count)
            for version, count in table["version"].value_counts().items()
        },
        "flattop_at_least_1s": int((table["flattop_s"] >= MIN_FLATTOP_S).sum()),
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
