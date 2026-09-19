"""Small public label tables, separate from internal event provenance records."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .databases import DatabaseError, write_csv, write_meta

LEGACY_INTERVAL_COLUMNS = ("shot", "t_start", "t_end", "confidence")
INTERVAL_COLUMNS = ("shot", "category", "t_start", "t_end", "confidence")
INTERVAL_SCHEMA_VERSION = 5
CATEGORY_NAMES = {
    "qmin_low": "low",
    "qmin_hybrid": "hybrid",
    "qmin_elevated": "elevated",
    "qmin_high": "high",
}


QMIN_CATEGORY_IDS = {
    "qmin_low": 1,
    "qmin_hybrid": 2,
    "qmin_elevated": 3,
    "qmin_high": 4,
}


def category_labels(category: str) -> dict[str, str]:
    """Names shared by CSV sidecars, sampled grids, and event documentation."""
    if category == "minimum_safety_factor":
        return {"0": "absent", "1": "low", "2": "hybrid", "3": "elevated", "4": "high"}
    return {"0": "absent", "1": "present"}


def validate_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate millisecond intervals; blank confidence means unknown."""
    if tuple(frame.columns) != INTERVAL_COLUMNS:
        raise DatabaseError(f"Expected columns {INTERVAL_COLUMNS}")
    result = frame.copy()
    values = pd.to_numeric(result["category"], errors="coerce")
    if not (
        np.isfinite(values) & (values >= 0) & (values < 2**63) & (values % 1 == 0)
    ).all():
        raise DatabaseError("category must be a nonnegative integer")
    result["category"] = values.astype("int64")
    for column in LEGACY_INTERVAL_COLUMNS:
        raw = result[column]
        values = pd.to_numeric(raw, errors="coerce")
        valid = np.isfinite(values)
        if column == "shot":
            valid &= (values % 1 == 0) & (values >= 0) & (values < 2**63)
        elif column == "confidence":
            valid = (valid & values.between(0, 1)) | raw.isna() | (raw == "")
        if not valid.all():
            raise DatabaseError(f"Invalid {column}")
        result[column] = values.astype("int64" if column == "shot" else "float64")
    if (result.t_end < result.t_start).any():
        raise DatabaseError("t_end must not precede t_start")
    return result


def project_intervals(events: pd.DataFrame) -> pd.DataFrame:
    """Select only shot, interval bounds and confidence from second-based events."""
    events = events.rename(columns={"t_start_ms": "t_start", "t_end_ms": "t_end"})
    if tuple(events.columns) == INTERVAL_COLUMNS:
        return validate_intervals(events)
    if tuple(events.columns) == LEGACY_INTERVAL_COLUMNS:
        return validate_intervals(events.assign(category=1)[list(INTERVAL_COLUMNS)])
    frame = (
        events[["shot", "t0_s", "t1_s", "confidence"]]
        .rename(
            columns={
                "t0_s": "t_start",
                "t1_s": "t_end",
            }
        )
        .copy()
    )
    for column in ("t_start", "t_end"):
        frame[column] = pd.to_numeric(frame[column], errors="raise") * 1000
    names = []
    for row in events.to_dict("records"):
        attrs = row.get("attrs", "{}")
        attrs = json.loads(attrs) if isinstance(attrs, str) else attrs
        # Explicit numeric labels include zero; do not treat it as missing.
        explicit = row.get("category", attrs.get("category"))
        if explicit is not None:
            names.append(explicit)
        else:
            names.append(QMIN_CATEGORY_IDS.get(row.get("phenomenon"), 1))
    frame["category"] = names
    return validate_intervals(frame[list(INTERVAL_COLUMNS)])


def write_interval_table(frame: pd.DataFrame, path: Path, meta: dict) -> None:
    """Write the labelled interval CSV, with provenance kept in its JSON sidecar."""
    frame = validate_intervals(frame).sort_values(
        ["shot", "t_start", "t_end", "category", "confidence"], kind="stable"
    )
    metadata = {
        **meta,
        "schema_version": INTERVAL_SCHEMA_VERSION,
        "time_units": "ms",
        "columns": list(INTERVAL_COLUMNS),
        "categories": meta.get("categories", category_labels(meta.get("category", ""))),
        "n_rows": len(frame),
        "n_shots": int(frame.shot.nunique()),
    }
    json.dumps(metadata, allow_nan=False)
    write_csv(frame, Path(path))
    write_meta(Path(path), metadata)


SAMPLE_MS = 50.0
RHO_EDGES = np.linspace(0.0, 1.0, 21)


def write_label_grid(
    path: Path, time_ms, labels, *, rho_edges=None, event_count=None, categories=None
) -> None:
    """Save actual integer values as sparse coordinates on a time × 20-rho grid.

    One-dimensional labels are broadcast across rho. Zero cells are implicit;
    unknown cells have separate coordinates, so missing is not confused with 0.
    No transitions are encoded. Binary and mutually exclusive class IDs use the
    same representation. NPZ arrays can be loaded with allow_pickle=False.
    """
    from labeler.config import atomic_path

    times = np.asarray(time_ms, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = RHO_EDGES if rho_edges is None else np.asarray(rho_edges, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError("time_ms must be finite and strictly increasing")
    if edges.shape != (21,) or not np.allclose(edges, RHO_EDGES):
        raise ValueError("Expected 20 equal rho bins over 0 to 1")
    if labels.ndim == 1:
        labels = np.repeat(labels[:, None], 20, axis=1)
    if labels.shape != (len(times), 20):
        raise ValueError("Labels must have shape (n_times, 20), or (n_times,)")
    known = ~np.isnan(labels)
    if (
        not np.isfinite(labels[known]).all()
        or (labels[known] < 0).any()
        or (labels[known] % 1 != 0).any()
        or (labels[known] >= 2**63).any()
    ):
        raise ValueError("Known labels must be nonnegative integer class IDs")
    extras = {}
    if event_count is not None:
        counts = np.asarray(event_count)
        if (
            counts.shape != times.shape
            or not np.isfinite(counts).all()
            or (counts < 0).any()
            or (counts % 1 != 0).any()
        ):
            raise ValueError(
                "event_count must contain nonnegative integers per time bin"
            )
        extras["event_count"] = counts.astype(np.int64)
    if categories is not None:
        ids = sorted(int(k) for k in categories)
        if not set(labels[known].astype(np.int64).tolist()) <= set(ids):
            raise ValueError("Category mapping does not cover all stored labels")
        extras["category_ids"] = np.asarray(ids, dtype=np.int64)
        extras["category_names"] = np.asarray([categories[str(k)] for k in ids])
        extras["time_units"] = np.array("ms")
    positive = known & (labels != 0)
    with atomic_path(path) as temporary, temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.array(2),
            **extras,
            time_ms=times,
            rho_edges=edges,
            shape=np.array(labels.shape, dtype=np.int64),
            indices=np.argwhere(positive).astype(np.int32),
            values=labels[positive].astype(np.int64),
            unknown_indices=np.argwhere(~known).astype(np.int32),
        )


def read_label_grid(path: Path) -> dict:
    """Reconstruct actual sampled values; NaN marks missing/unknown cells."""
    with np.load(path, allow_pickle=False) as data:
        shape = tuple(data["shape"])
        labels = np.zeros(shape, dtype=float)
        indices = data["indices"]
        unknown = data["unknown_indices"]
        if len(indices):
            labels[tuple(indices.T)] = data["values"]
        if len(unknown):
            labels[tuple(unknown.T)] = np.nan
        return {
            "time_ms": data["time_ms"].copy(),
            "rho_edges": data["rho_edges"].copy(),
            "label": labels,
            **(
                {
                    "categories": dict(
                        zip(
                            data["category_ids"].astype(str),
                            data["category_names"].tolist(),
                            strict=True,
                        )
                    )
                }
                if "category_ids" in data
                else {}
            ),
            **(
                {"event_count": data["event_count"].copy()}
                if "event_count" in data
                else {}
            ),
        }


def sample_event_labels(events: pd.DataFrame, time_ms, *, class_ids=None) -> dict:
    """Aggregate half-open 50 ms bins, preserving short and point events.

    Binary bins use any event. Categorical bins select the class with greatest
    total interval overlap; equal durations select the larger class ID. Unknown
    categorical bins remain NaN. Coordinates are bin starts, with no extra times.
    """
    times = np.asarray(time_ms, dtype=float)
    if (
        times.ndim != 1
        or not np.isfinite(times).all()
        or (np.diff(times) != SAMPLE_MS).any()
        or (times % SAMPLE_MS != 0).any()
    ):
        raise ValueError("Sampling times must be consecutive 50 ms bin starts")
    labels = np.full(len(times), np.nan if class_ids is not None else 0.0)
    scores = {}
    for event in events.itertuples(index=False):
        start, stop = float(event.t0_s) * 1000, float(event.t1_s) * 1000
        if not np.isfinite([start, stop]).all() or stop < start:
            raise ValueError("Invalid event interval")
        value = 1 if class_ids is None else class_ids[event.phenomenon]
        overlap = np.maximum(
            0, np.minimum(times + SAMPLE_MS, stop) - np.maximum(times, start)
        )
        if start == stop:
            overlap[(times <= start) & (start < times + SAMPLE_MS)] = np.finfo(
                float
            ).eps
        scores[value] = scores.get(value, np.zeros(len(times))) + overlap
    best = np.zeros(len(times))
    for value, score in sorted(scores.items()):
        chosen = (score > 0) & (score >= best)
        labels[chosen], best[chosen] = value, score[chosen]
    return {"time_ms": times, "label": labels}


def bin_binary_samples(time_ms, labels) -> dict:
    """Any-positive 50 ms bins; missing bins stay unknown and counts retain ones."""
    times, values = np.asarray(time_ms, float), np.asarray(labels)
    if (
        times.ndim != 1
        or values.shape != times.shape
        or not np.isfinite(times).all()
        or (np.diff(times) <= 0).any()
        or not np.isin(values, [0, 1]).all()
    ):
        raise ValueError("Invalid binary samples")
    if not len(times):
        return {
            "time_ms": times,
            "label": np.array([]),
            "event_count": np.array([], dtype=int),
        }
    bins = np.floor(times / SAMPLE_MS).astype(np.int64)
    axis = np.arange(bins[0], bins[-1] + 1)
    index = bins - axis[0]
    observed = np.bincount(index, minlength=len(axis))
    counts = np.bincount(index, weights=values, minlength=len(axis)).astype(np.int64)
    result = np.where(observed, (counts > 0).astype(float), np.nan)
    return {"time_ms": axis * SAMPLE_MS, "label": result, "event_count": counts}


def grid_intervals(shot: int, grid: dict) -> pd.DataFrame:
    """Compress constant known 50 ms bins to half-open CSV intervals."""
    times, values = np.asarray(grid["time_ms"]), np.asarray(grid["label"])
    rows = []
    begin = 0
    for end in range(1, len(times) + 1):
        if (
            end < len(times)
            and times[end] == times[end - 1] + SAMPLE_MS
            and values[end] == values[begin]
        ):
            continue
        if np.isfinite(values[begin]):
            rows.append(
                [
                    shot,
                    int(values[begin]),
                    times[begin],
                    times[end - 1] + SAMPLE_MS,
                    None,
                ]
            )
        begin = end
    return pd.DataFrame(rows, columns=INTERVAL_COLUMNS)


def formatted_binary_grids(frame: pd.DataFrame, *, minimum_stop_ms=6000):
    """Sample explicit 0/1 intervals; leave unannotated regions unknown.

    Point events mark their containing bin. Other bins are unknown unless an
    explicit zero interval covers them; a positive annotation wins within a bin.
    """
    frame = validate_intervals(frame)
    if not frame.category.isin([0, 1]).all():
        raise ValueError("Expected binary interval categories")
    for shot, rows in frame.groupby("shot", sort=True):
        last = float(rows.t_end.max())
        point_at_last = ((rows.t_start == last) & (rows.t_end == last)).any()
        stop = max(minimum_stop_ms, np.ceil(last / SAMPLE_MS) * SAMPLE_MS)
        if point_at_last and last >= stop:
            stop = last + SAMPLE_MS
        times = np.arange(
            min(0, np.floor(rows.t_start.min() / SAMPLE_MS) * SAMPLE_MS),
            stop,
            SAMPLE_MS,
        )
        events = pd.DataFrame({"t0_s": rows.t_start / 1000, "t1_s": rows.t_end / 1000})
        observed = sample_event_labels(events, times)["label"] > 0
        grid = sample_event_labels(events.loc[rows.category == 1], times)
        grid["label"][~observed] = np.nan
        yield int(shot), grid
