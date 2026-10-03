"""Recover clock peak times without treating ELMy-period edges as events.

The original clock table stores present *periods*. Its metadata records the
channel and producing revision. Prefer point events saved from that revision;
otherwise run its unchanged point picker on read-only on-disk filterscopes.
Both routes retain only points inside the ORIGINAL saved present periods.
No current span, plasma-start or H-mode gate is recomputed.

These are heuristic D-alpha peak timestamps. The reviewed non-crowd span starts
are not independently verified ELM onset times, and the review began from the
clock. Matching those starts measures timing agreement with a dependent review.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Paths
from ..events import raw, spans, transients

CLOCK_CSV = Path(
    "suggestions/elm_clock/v1/edge_localized_mode_suggest_elm_clock_v1.csv"
)


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _picker_provenance(revision: str) -> dict:
    """Refuse to substitute a changed picker for the original clock's picker."""
    path = Path(transients.__file__).resolve()
    current = path.read_bytes()
    record = {
        "original_git": revision,
        "module": "src/labeler/events/transients.py",
        "current_sha256": _hash_bytes(current),
        "verified_identical": False,
    }
    result = subprocess.run(
        ["git", "show", f"{revision}:src/labeler/events/transients.py"],
        cwd=path.parents[3],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        record["reason"] = "original picker revision is unavailable locally"
    else:
        record["original_sha256"] = _hash_bytes(result.stdout)
        record["verified_identical"] = current == result.stdout
        if not record["verified_identical"]:
            record["reason"] = "current picker differs from original clock revision"
    return record


def _saved_points(paths: Paths, shot: int, channel: int, revision: str):
    """Only accept saved point events with the original source/run provenance."""
    event_file, source_file = paths.events_file(shot), paths.sources_file(shot)
    if not event_file.is_file() or not source_file.is_file():
        return None
    sources = pd.read_parquet(source_file)
    sources = sources[sources.source.eq(transients.ELM_SOURCE)]
    if (
        len(sources) != 1
        or not sources.status.eq("ran").all()
        or not sources.git_sha.eq(revision).all()
        or not sources.channel.eq(channel).all()
        or not sources.diag.eq("filterscopes").all()
    ):
        return None
    events = pd.read_parquet(event_file)
    events = events[events.source.eq(transients.ELM_SOURCE)]
    source = sources.iloc[0]
    if (
        len(events) != int(source.n_events)
        or not events.git_sha.eq(revision).all()
        or not events.run_id.eq(source.run_id).all()
        or not events.channel.eq(channel).all()
        or not events.diag.eq("filterscopes").all()
    ):
        return None
    points = events[events.phenomenon.eq(transients.ELM_PHENOMENON)]
    if not np.array_equal(points.t0_s.to_numpy(), points.t1_s.to_numpy()):
        return None
    return points.t0_s.to_numpy(float) * 1000.0, {
        "route": "saved_original_revision_points",
        "events": str(event_file),
        "events_sha256": _hash_bytes(event_file.read_bytes()),
        "sources": str(source_file),
        "sources_sha256": _hash_bytes(source_file.read_bytes()),
        "run_id": source.run_id,
    }


def _recomputed_points(paths: Paths, shot: int, expected_channel: int):
    """Read the same input tiers/channel policy; never fetch or write inputs."""
    tier = raw.record_tier(shot, "filterscopes", paths=paths)
    if tier is None:
        raise ValueError("no filterscopes on disk; fetching is disabled")
    t_s, y = spans.read(shot, "filterscopes", paths, range(8))
    channel = spans.dalpha_channel(y, shot)
    if channel != expected_channel:
        raise ValueError(
            f"current first finite channel FS{channel + 1:02d} differs from "
            f"original FS{expected_channel + 1:02d}"
        )
    found = transients.elm_clock_events(
        y[channel],
        t_s,
        shot=shot,
        channel=channel,
        smooth_ms=transients.SMOOTH_MS,
        prominence=transients.PROMINENCE,
        min_distance_ms=transients.MIN_DISTANCE_MS,
        max_rate_hz=transients.ELM_FREE_MAX_RATE_HZ,
        min_duration_s=transients.ELM_FREE_MIN_S,
        window_s=transients.RATE_WINDOW_S,
    )
    root = paths.corpus if tier == "corpus" else paths.raw_cache
    return np.asarray(
        [e.t0_s * 1000.0 for e in found if e.phenomenon == transients.ELM_PHENOMENON]
    ), {
        "route": "recomputed_original_picker_points",
        "input": str(root / f"{shot}_processed.h5"),
        "tier": tier,
        "time_seconds_sha256": _hash_bytes(np.ascontiguousarray(t_s).tobytes()),
        "channel_values_sha256": _hash_bytes(
            np.ascontiguousarray(y[channel]).tobytes()
        ),
        "time_dtype": str(t_s.dtype),
        "channel_dtype": str(y.dtype),
        "samples": len(t_s),
        "coverage_ms": [float(t_s[0] * 1000), float(t_s[-1] * 1000)],
        "native_median_step_ms": float(np.median(np.diff(t_s)) * 1000),
    }


def load_clock_onsets(
    paths: Paths,
    shots: Sequence[int],
    clock_spans: dict[int, pd.DataFrame],
) -> tuple[dict[int, np.ndarray], dict]:
    """Genuine clock point times (ms), gated by the saved clock's present spans.

    Missing/unverifiable shots are omitted and listed in ``metadata.unavailable``;
    callers must omit their timing comparator instead of treating them as empty
    detections. Supplied spans must match the original table exactly. The input
    hashes identify current snapshots; the original span metadata had no input
    hashes, so unchanged historical signals cannot be established from it.
    """
    csv_file = paths.root / CLOCK_CSV
    meta_file = csv_file.with_suffix(".meta.json")
    original = json.loads(meta_file.read_text())
    table = pd.read_csv(csv_file)
    table = table[table.category.eq(1)]
    revision = original["git_sha"]
    picker = _picker_provenance(revision)
    metadata = {
        "definition": "D-alpha heuristic peak timestamps, never period starts",
        "span_source": str(csv_file),
        "span_source_sha256": _hash_bytes(csv_file.read_bytes()),
        "span_metadata": str(meta_file),
        "span_metadata_sha256": _hash_bytes(meta_file.read_bytes()),
        "original_span_rule": original["rule"],
        "picker_provenance": picker,
        "picker_constants": {
            name: getattr(transients, name)
            for name in (
                "SMOOTH_MS",
                "PROMINENCE",
                "MIN_DISTANCE_MS",
                "DALPHA_MAX_WIDTH_MS",
                "RATE_WINDOW_S",
                "ELM_FREE_MAX_RATE_HZ",
                "ELM_FREE_MIN_S",
            )
        },
        "restriction": "points inside original saved category-1 present spans; "
        "half-open [start, end), no recomputation of span or H-mode gates",
        "review_independent": False,
        "timing_limit": "Heuristic smoothed D-alpha peak times; reviewed "
        "non-crowd present-span starts are not independently verified ELM "
        "onsets. A 5 ms match measures agreement with reviewed boundaries, "
        "not validated physical onset accuracy.",
        "historical_input_limit": "Original span metadata has no input hashes. "
        "The manifest identifies current on-disk signals and cannot prove "
        "they are unchanged since the original span generation.",
        "per_shot": {},
        "unavailable": {},
    }
    out = {}
    for shot in sorted({int(s) for s in shots}):
        saved = table[table.shot.eq(shot)][["t_start", "t_end"]].to_numpy(float)
        supplied = clock_spans.get(shot)
        intervals = (
            supplied[["t_start_ms", "t_end_ms"]].to_numpy(float)
            if supplied is not None
            else np.empty((0, 2))
        )
        if not np.array_equal(
            saved[np.lexsort((saved[:, 1], saved[:, 0]))],
            intervals[np.lexsort((intervals[:, 1], intervals[:, 0]))],
        ):
            raise ValueError(f"shot {shot}: supplied spans differ from original clock")
        if not picker["verified_identical"]:
            metadata["unavailable"][str(shot)] = picker["reason"]
            continue
        try:
            channel_name = original["per_shot"][str(shot)]["channel"]
            channel = int(channel_name.removeprefix("FS")) - 1
            points = _saved_points(paths, shot, channel, revision)
            if points is None:
                points = _recomputed_points(paths, shot, channel)
            times, record = points
            keep = np.zeros(len(times), dtype=bool)
            for lo, hi in saved:
                keep |= (times >= lo) & (times < hi)
            out[shot] = np.unique(times[keep])
            metadata["per_shot"][str(shot)] = {
                **record,
                "channel": channel_name,
                "points_before_original_span_gate": len(times),
                "points_retained": len(out[shot]),
                "original_present_spans": len(saved),
            }
        except (KeyError, ValueError, OSError) as exc:
            metadata["unavailable"][str(shot)] = str(exc)
    metadata["shots_requested"] = len(set(shots))
    metadata["shots_available"] = len(out)
    metadata["points_retained"] = sum(len(times) for times in out.values())
    return out, metadata
