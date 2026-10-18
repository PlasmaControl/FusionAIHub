"""Read original AE, ELM, RWM and tearing labels and write the common event schema.

One command formats any category's original labels:

    python -m labeler.events.source_formatters <category> [--dataset STEM]

It reads the dataset list from `configs/labeler/events.yaml` (`Paths.events_manifest`)
and the raw files under the label tables (`Paths.label_tables`), and writes
`<category>/format/<output>.csv` there. No converter modifies raw files or extracts
archives onto disk.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import re
import tarfile
from itertools import pairwise
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from ..config import Paths
from .databases import FORMAT_COLUMNS, FORMAT_SCHEMA_VERSION, validate_format
from .interval_tables import (
    SAMPLE_MS,
    bin_binary_samples,
    category_labels,
    formatted_binary_grids,
    grid_intervals,
    project_intervals,
    sample_event_labels,
    write_interval_table,
    write_label_grid,
)

AE_CLASSES = ("lfm", "bae", "eae", "rsae", "tae")
CONVERSION_DATE = "2026-09-14T00:00:00Z"


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=FORMAT_COLUMNS)


def _row(shot, start, stop, phenomenon, stem, attrs):
    return {
        "shot": int(shot),
        "t0_s": float(start),
        "t1_s": float(stop),
        "phenomenon": phenomenon,
        "evidence_kind": "database",
        "source": f"database:{stem}",
        "confidence": float("nan"),
        "attrs": json.dumps({"table": stem, **attrs}, sort_keys=True, allow_nan=False),
    }


def _runs(mask: np.ndarray):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def format_rwm(path: Path) -> pd.DataFrame:
    """Convert millisecond point onsets; retain duplicate rows and annotations."""
    path = Path(path)
    raw = pd.read_csv(path)
    required = {"SHOT", "ONSET_TIME", "NTOR", "MODE_TYPE"}
    if not required <= set(raw):
        raise ValueError(f"{path}: missing columns {sorted(required - set(raw))}")
    rows = []
    for record in raw.to_dict("records"):
        shot, onset, ntor = [float(record[k]) for k in ("SHOT", "ONSET_TIME", "NTOR")]
        if not all(np.isfinite([shot, onset, ntor])) or shot % 1 or ntor % 1:
            raise ValueError(f"{path}: invalid shot, onset or mode number")
        rows.append(
            _row(
                shot,
                onset / 1000,
                onset / 1000,
                "rwm",
                path.stem,
                {
                    "NTOR": int(ntor),
                    "MODE_TYPE": str(record["MODE_TYPE"]),
                    "raw_row": len(rows),
                    "raw_time_units": "ms",
                },
            )
        )
    return _frame(rows)


def read_ae(path: Path) -> list[tuple[int, str, np.ndarray]]:
    """Read the trusted six-part pickle, retaining labels but releasing spectra.

    Layout: train_shots, X_train, y_train, valid_shots, X_valid, y_valid.
    Loading the original pickle temporarily requires several GB of memory.
    """
    with Path(path).open("rb") as stream:
        dataset = pickle.load(stream)
    if not isinstance(dataset, (list, tuple)) or len(dataset) != 6:
        raise ValueError(f"{path}: expected the six-part original AE pickle")
    records = []
    for split, offset in (("train", 0), ("valid", 3)):
        shots, spectra, labels = dataset[offset : offset + 3]
        if not len(shots) == len(spectra) == len(labels):
            raise ValueError(f"{path}: inconsistent {split} lengths")
        for shot, label in zip(shots, labels):
            if not np.isfinite(float(shot)) or float(shot) % 1:
                raise ValueError(f"{path}: invalid shot {shot}")
            records.append((int(shot), split, np.asarray(label)))
    return records


def format_ae_records(records, *, start_s=0.0, stop_s=2.0) -> pd.DataFrame:
    """Map AE classes to uniform bins over an explicitly reconstructed window.

    The pickle has no time vector. The upstream make_ts_dataset.py uses 0–2000
    ms and rescales labels across that window. This is a bin mapping, not a
    recovered STFT timestamp. LFM is excluded. The union of the four AE classes
    is category 1; annotated absence is category 0. Values are not confidences.
    """
    if not np.isfinite([start_s, stop_s]).all() or stop_s <= start_s:
        raise ValueError("AE time window must be finite and increasing")
    rows = []
    for shot, split, labels in records:
        if labels.ndim != 2 or labels.shape[1] != 5 or not len(labels):
            raise ValueError(f"AE shot {shot}: expected nonempty (time, 5) labels")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f"AE shot {shot}: expected original binary labels")
        edges = np.linspace(start_s, stop_s, len(labels) + 1)
        active = np.any(labels[:, 1:] == 1, axis=1)
        intervals = pd.DataFrame(
            [(edges[begin], edges[end]) for begin, end in _runs(active)],
            columns=["t0_s", "t1_s"],
        )
        grid = sample_event_labels(
            intervals,
            np.arange(
                np.floor(start_s * 1000 / SAMPLE_MS) * SAMPLE_MS,
                np.ceil(stop_s * 1000 / SAMPLE_MS) * SAMPLE_MS,
                SAMPLE_MS,
            ),
        )
        for row in grid_intervals(shot, grid).itertuples(index=False):
            rows.append(
                _row(
                    shot,
                    row.t_start / 1000,
                    row.t_end / 1000,
                    "ae",
                    "co2_detector_2021",
                    {
                        "category": row.category,
                        "split": split,
                        "time_mapping": "uniform_bins_then_any_positive_50ms_bins",
                        "window_start_s": float(start_s),
                        "window_stop_s": float(stop_s),
                        "n_samples": len(labels),
                    },
                )
            )
    return _frame(rows)


def format_ae(path: Path, *, start_s=0.0, stop_s=2.0) -> pd.DataFrame:
    return format_ae_records(read_ae(path), start_s=start_s, stop_s=stop_s)


AE_ECE_STEM = "ece_labels_2022"
AE_ECE_COLUMNS = ["t1", "t2", "f1", "f2", "channel", "type"]
AE_ECE_TYPES = ("TAE", "RSAE", "BAE", "EAE", "LFM")


def read_ae_ece(folder: Path) -> list[tuple[int, pd.DataFrame]]:
    """Read the per-shot ECE box files `labels_<shot>.txt`, one box per line.

    A box is a time-frequency rectangle drawn on one ECE channel's spectrogram:
    `t1, t2` in seconds, `f1, f2` in kHz, the radiometer `channel` and the mode
    `type`. The files carry no author and no record of which stretch of the shot
    was inspected.
    """
    records = []
    for path in sorted(Path(folder).glob("labels_*.txt")):
        match = re.fullmatch(r"labels_(\d+)\.txt", path.name)
        boxes = pd.read_csv(path, skipinitialspace=True)
        if match is None or list(boxes.columns) != AE_ECE_COLUMNS:
            raise ValueError(f"{path}: expected columns {AE_ECE_COLUMNS}")
        boxes["type"] = boxes["type"].astype(str).str.strip().str.upper()
        unknown = sorted(set(boxes["type"]) - set(AE_ECE_TYPES))
        if unknown:
            raise ValueError(f"{path}: unknown mode types {unknown}")
        numbers = boxes.drop(columns="type").apply(pd.to_numeric, errors="coerce")
        valid = (
            np.isfinite(numbers).all().all()
            and (numbers.t2 > numbers.t1).all()
            and (numbers.f2 > numbers.f1).all()
            and (numbers.channel % 1 == 0).all()
            and (numbers.channel >= 1).all()
        )
        if not valid:
            raise ValueError(f"{path}: boxes need finite t1 < t2, f1 < f2, channel >= 1")
        boxes = pd.concat([numbers, boxes["type"]], axis=1)
        boxes["channel"] = boxes["channel"].astype(int)
        records.append((int(match.group(1)), boxes))
    if not records:
        raise ValueError(f"{folder}: no labels_<shot>.txt files")
    return records


def format_ae_ece_records(records) -> pd.DataFrame:
    """AE time spans of the ECE boxes: category 1 where any box but LFM covers.

    The union runs over every ECE channel and every frequency; overlapping or
    touching spans merge. LFM boxes are ignored, as in the CO2 labels. The
    files do not say which stretch of the shot was inspected, so no stretch is
    marked absent: everything outside a span stays unknown.
    """
    rows = []
    for shot, boxes in records:
        spans: list[list] = []
        for box in boxes[boxes["type"] != "LFM"].sort_values(["t1", "t2"]).itertuples():
            if spans and box.t1 <= spans[-1][1]:
                span = spans[-1]
                span[1] = max(span[1], box.t2)
                span[2].add(box.type)
                span[3] += 1
            else:
                spans.append([box.t1, box.t2, {box.type}, 1])
        for start, stop, types, count in spans:
            attrs = {"category": 1, "types": sorted(types), "n_boxes": count}
            rows.append(_row(shot, start, stop, "ae", AE_ECE_STEM, attrs))
    return _frame(rows)


def format_ae_ece(folder: Path) -> pd.DataFrame:
    return format_ae_ece_records(read_ae_ece(folder))


def ae_ece_audit(records) -> dict:
    """Counts that bound what the boxes can say, for the sidecar and the README."""
    boxes = pd.concat([b.assign(shot=s) for s, b in records], ignore_index=True)
    ae = boxes[boxes["type"] != "LFM"]
    return {
        "n_shots": len(records),
        "n_boxes": len(boxes),
        "boxes_by_type": {t: int((boxes["type"] == t).sum()) for t in AE_ECE_TYPES},
        "duplicate_boxes": int(boxes.duplicated().sum()),
        "t_range_s": [float(boxes.t1.min()), float(boxes.t2.max())],
        "f_range_khz": [float(boxes.f1.min()), float(boxes.f2.max())],
        "channel_range": [int(boxes.channel.min()), int(boxes.channel.max())],
        "ae_boxes": len(ae),
        "ae_boxes_starting_at_0.30_s": int((ae.t1 == 0.3).sum()),
        "shots_without_ae_boxes": sorted(
            {int(s) for s in boxes.shot} - {int(s) for s in ae.shot}
        ),
    }


def read_elm(*paths: Path):
    """Merge original millisecond label traces, checking overlapping WPQH slices.

    WPQH keys are shot_phase; their times are absolute, not phase-relative.
    Duplicate samples must agree and are stored once. Survival targets are
    deliberately not read: their event flags describe censoring, not ELM times.
    """
    samples = {}
    for path in paths:
        with Path(path).open("rb") as stream:
            data = pickle.load(stream)
        if not isinstance(data, dict):
            raise TypeError(f"{path}: expected a shot-to-trace dictionary")
        for key, trace in data.items():
            match = re.fullmatch(r"(\d+)(?:_\d+)?", str(key))
            if (
                not match
                or not isinstance(trace, dict)
                or not {"times", "labels"} <= trace.keys()
            ):
                raise ValueError(f"{path}: invalid ELM record {key}")
            shot = int(match[1])
            times = np.asarray(trace["times"], dtype=float)
            labels = np.asarray(trace["labels"])
            if (
                times.ndim != 1
                or labels.shape != times.shape
                or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0)
                or not np.isin(labels, [0, 1]).all()
            ):
                raise ValueError(f"{path}: invalid time or label trace for {key}")
            samples.setdefault(shot, []).append((times, labels))
    for shot, traces in sorted(samples.items()):
        times = np.concatenate([t for t, _ in traces])
        labels = np.concatenate([v for _, v in traces]).astype(np.int64)
        order = np.argsort(times, kind="stable")
        times, labels = times[order], labels[order]
        duplicate = np.diff(times) == 0
        if np.any(duplicate & (np.diff(labels) != 0)):
            raise ValueError(f"ELM shot {shot}: conflicting overlapping labels")
        keep = np.r_[True, ~duplicate] if len(times) else np.array([], dtype=bool)
        yield shot, times[keep], labels[keep]


def format_elm(*paths: Path) -> pd.DataFrame:
    """Count 1 ms onset labels in 50 ms bins; emit presence/absence intervals."""
    rows = []
    for shot, times, labels in read_elm(*paths):
        grid = bin_binary_samples(times, labels)
        for row in grid_intervals(shot, grid).itertuples(index=False):
            rows.append(
                _row(
                    shot,
                    row.t_start / 1000,
                    row.t_end / 1000,
                    "elm",
                    "elm_labels",
                    {
                        "category": row.category,
                        "raw_time_units": "ms",
                        "interval_bounds": "half_open_50ms_bins",
                    },
                )
            )
    return _frame(rows)


ELM_CHALLENGE_STEM = "elm_challenge_key"
#: The challenge key's event types as catalog states (0 absent, 1 present, 2 uncertain).
#: Types 1, 2 and 6 are ELMs (large, small, pellet-triggered). Types 3 and 4 are
#: "other" (back-transition transients, disruptions, unidentified blips): neither
#: called ELM nor not-ELM, so uncertain. Types 5 and 7 are pellets with no ELM,
#: and 8 is an unspecified event.
ELM_CHALLENGE_STATES = {1: 1, 2: 1, 6: 1, 3: 2, 4: 2, 5: 0, 7: 0, 8: 2}
ELM_CHALLENGE_DEVICE = "DIII-D"


def _challenge_pair(where, event, name):
    value = event.get(name)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in value
        )
        or not np.isfinite(value).all()
        or value[1] < 0
    ):
        raise ValueError(f"{where}: {name} must be [nominal, uncertainty]: {value!r}")
    return float(value[0]), float(value[1])


def read_elm_challenge(path: Path) -> list[dict]:
    """The DIII-D cases of the ELM detection challenge's answer key.

    Each case is one hand-timed window (`tstart`..`tend`, ms) of one shot with its
    events, `(start, end, type)` at the nominal edge times. The edge uncertainties
    are checked but not carried; the raw key keeps them. Other devices are skipped.
    """
    data = json.loads(Path(path).read_text())
    cases, seen = [], set()
    for key, case in data.items():
        if (
            key.startswith("__")
            or not isinstance(case, dict)
            or case.get("device") != ELM_CHALLENGE_DEVICE
        ):
            continue
        where = f"ELM challenge case {key}"
        shot = case.get("shot")
        if not isinstance(shot, int) or isinstance(shot, bool) or shot < 0:
            raise ValueError(f"{where}: invalid shot {shot!r}")
        if shot in seen:
            raise ValueError(f"{where}: shot {shot} appears twice")
        seen.add(shot)
        if case.get("time_units") != "ms":
            raise ValueError(f"{where}: times must be in ms")
        tstart, tend = case.get("tstart"), case.get("tend")
        if not all(isinstance(v, (int, float)) for v in (tstart, tend)) or not (
            np.isfinite([tstart, tend]).all() and tend > tstart
        ):
            raise ValueError(f"{where}: invalid window {tstart!r}..{tend!r}")
        events = []
        for name, event in case.items():
            if not isinstance(event, dict) or "start" not in event:
                continue
            start, _ = _challenge_pair(f"{where} event {name}", event, "start")
            end, _ = _challenge_pair(f"{where} event {name}", event, "end")
            if event.get("type") not in ELM_CHALLENGE_STATES:
                raise ValueError(
                    f"{where} event {name}: unknown type {event.get('type')!r}"
                )
            if not tstart <= start < end <= tend:
                raise ValueError(
                    f"{where} event {name}: {start}..{end} is not inside the window"
                )
            events.append((start, end, event["type"]))
        cases.append(
            {
                "shot": shot,
                "tstart": float(tstart),
                "tend": float(tend),
                "events": events,
            }
        )
    if not cases:
        raise ValueError("no DIII-D cases in the ELM challenge key")
    return cases


def elm_challenge_segments(case: dict) -> list[tuple[int, float, float]]:
    """The case's window cut at every event edge as `(state, start, end)` in ms.

    Present beats uncertain beats absent where events overlap, so the segments do
    not overlap; abutting segments of one state are joined. A window with no
    events is absent throughout, and time outside the window is not assessed.
    """
    cuts = sorted(
        {case["tstart"], case["tend"]}
        | {edge for start, end, _ in case["events"] for edge in (start, end)}
    )
    segments = []
    for begin, stop in pairwise(cuts):
        states = {
            ELM_CHALLENGE_STATES[kind]
            for start, end, kind in case["events"]
            if start < stop and end > begin
        }
        state = 1 if 1 in states else 2 if 2 in states else 0
        if segments and segments[-1][0] == state:
            segments[-1] = (state, segments[-1][1], stop)
        else:
            segments.append((state, begin, stop))
    return segments


def format_elm_challenge(path: Path) -> pd.DataFrame:
    """Present, uncertain and absent segments of each challenge window, as events."""
    rows = []
    for case in read_elm_challenge(path):
        for state, begin, stop in elm_challenge_segments(case):
            rows.append(
                _row(
                    case["shot"],
                    begin / 1000,
                    stop / 1000,
                    "elm",
                    ELM_CHALLENGE_STEM,
                    {"category": state, "raw_time_units": "ms"},
                )
            )
    return _frame(rows)


def elm_challenge_audit(path: Path) -> dict:
    """What the key holds and what formatting leaves out, counted from the raw file."""
    cases = read_elm_challenge(path)
    data = json.loads(Path(path).read_text())
    others = sorted(
        {
            str(case["device"])
            for key, case in data.items()
            if not key.startswith("__")
            and isinstance(case, dict)
            and case.get("device") not in (None, ELM_CHALLENGE_DEVICE)
        }
    )
    repeats = {}
    for case in cases:
        counts = {}
        for event in case["events"]:
            counts[event] = counts.get(event, 0) + 1
        if any(n > 1 for n in counts.values()):
            repeats[str(case["shot"])] = sum(n - 1 for n in counts.values())
    kinds = [kind for case in cases for _, _, kind in case["events"]]
    return {
        "n_cases": len(cases),
        "n_events": len(kinds),
        "events_by_type": {str(k): kinds.count(k) for k in sorted(set(kinds))},
        "cases_without_events": sorted(c["shot"] for c in cases if not c["events"]),
        "window_ms_total": sum(c["tend"] - c["tstart"] for c in cases),
        "skipped_devices": others,
        "identical_repeated_events": repeats,
    }


def elm_challenge_grids(intervals: pd.DataFrame):
    """50 ms views of the challenge segments: 1 where any ELM touches the bin.

    A bin touched by an uncertain segment and no ELM, or by no segment, is
    unknown; 0 needs an absent segment and nothing else.
    """
    for shot, rows in intervals.groupby("shot", sort=True):
        stop = np.ceil(rows.t_end.max() / SAMPLE_MS) * SAMPLE_MS
        times = np.arange(0, stop, SAMPLE_MS)
        touched = {}
        for state in (0, 1, 2):
            part = rows[rows.category == state]
            events = pd.DataFrame(
                {"t0_s": part.t_start / 1000, "t1_s": part.t_end / 1000}
            )
            touched[state] = sample_event_labels(events, times)["label"] > 0
        label = np.full(len(times), np.nan)
        label[touched[0]] = 0
        label[touched[2]] = np.nan
        label[touched[1]] = 1
        yield int(shot), {"time_ms": times, "label": label}


def read_tm_h5(path: Path, shots=None):
    """Yield (shot, time_ms, binary label, metadata), one HDF5 group at a time."""
    with h5py.File(path, "r") as store:
        keys = sorted(store, key=int) if shots is None else map(str, shots)
        for key in keys:
            if key not in store:
                continue
            group = store[key]
            if not {"time", "label"} <= set(group):
                raise ValueError(f"{path}:{key}: missing time or label")
            attrs = {"raw_group": key}
            if "has_tm" in group:
                attrs["raw_has_tm"] = bool(group["has_tm"][()])
            yield int(key), group["time"][:], group["label"][:], attrs


def read_tm_tar(path: Path, shots=None):
    """Stream original 2×N NumPy traces from the tar without extracting files."""
    selected = None if shots is None else set(map(int, shots))
    with tarfile.open(path, "r|*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            match = re.fullmatch(r"(?:.*/)?(\d+)_ntm\.npy", member.name)
            if not match:
                raise ValueError(f"{path}: unexpected archive member {member.name}")
            shot = int(match[1])
            if selected is not None and shot not in selected:
                continue
            with archive.extractfile(member) as stream:
                values = np.load(io.BytesIO(stream.read()), allow_pickle=False)
            if values.ndim != 2 or values.shape[0] != 2:
                raise ValueError(f"{member.name}: expected a 2 x N array")
            yield shot, values[0], values[1], {"raw_member": member.name}


def format_tm_traces(traces, stem: str) -> pd.DataFrame:
    """Positive runs span first-to-last labelled samples, without extrapolation.

    A single positive sample is a point. Zeros are not emitted as events.
    Gaps over 1.5 times the median cadence break runs. HDF5 and tar evidence
    remain independent, even where their shots overlap.
    """
    rows = []
    for shot, times, labels, attrs in traces:
        times, labels = np.asarray(times, float), np.asarray(labels)
        if (
            times.ndim != 1
            or labels.shape != times.shape
            or not np.isfinite(times).all()
            or not np.isin(labels, [0, 1]).all()
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError(f"{stem}: shot {shot}: invalid time or binary label trace")
        if not len(times):
            continue
        positive = labels == 1
        cadence = float(np.median(np.diff(times))) if len(times) > 1 else None
        breaks = (
            np.flatnonzero(np.diff(times) > 1.5 * cadence) + 1
            if cadence is not None
            else []
        )
        bounds = [0, *breaks, len(times)]
        for left, right in pairwise(bounds):
            for begin, end in _runs(positive[left:right]):
                begin, end = int(begin + left), int(end + left)
                rows.append(
                    _row(
                        shot,
                        times[begin] / 1000,
                        times[end - 1] / 1000,
                        "tearing",
                        stem,
                        {
                            **attrs,
                            "raw_time_units": "ms",
                            "sample_start": begin,
                            "sample_stop_exclusive": end,
                            "sample_count": end - begin,
                            "median_cadence_ms": cadence,
                            "interval_bounds": "first_and_last_positive_samples",
                        },
                    )
                )
    return _frame(rows)


def format_tm(h5_path: Path | None = None, tar_path: Path | None = None):
    frames = []
    if h5_path is not None:
        frames.append(format_tm_traces(read_tm_h5(h5_path), "tm_labels_h5"))
    if tar_path is not None:
        frames.append(format_tm_traces(read_tm_tar(tar_path), "tm_labels_archive"))
    if not frames:
        raise ValueError("Provide at least one TM source")
    return pd.concat(frames, ignore_index=True)


CONFINEMENT_MODES = ("L", "H", "QH", "WP")
CONFINEMENT_TIMES = ("Confinement Start Time (ms)", "Confinement Stop Time (ms)")


def read_confinement(path: Path):
    """Read explicitly labelled regime intervals; report unlabelled source rows.

    Confinement times are milliseconds. BES acquisition bounds and transition
    notes are not interval bounds. Blank regime flags are not negative labels.
    """
    raw = pd.read_csv(path)
    required = {"Shot", *CONFINEMENT_TIMES, *CONFINEMENT_MODES}
    if not required <= set(raw):
        raise ValueError(f"{path}: missing columns {sorted(required - set(raw))}")
    shots = pd.to_numeric(raw.Shot, errors="coerce")
    if not (
        np.isfinite(shots) & (shots >= 0) & (shots < 2**63) & (shots % 1 == 0)
    ).all():
        raise ValueError("Confinement shot IDs must be nonnegative integers")
    flags = raw[list(CONFINEMENT_MODES)].fillna(0).apply(pd.to_numeric, errors="raise")
    if not flags.isin([0, 1]).all().all() or (flags.sum(axis=1) > 1).any():
        raise ValueError("Expected at most one binary confinement flag per row")
    labelled = flags.sum(axis=1) == 1
    times = raw[list(CONFINEMENT_TIMES)].apply(pd.to_numeric, errors="coerce")
    valid = (
        np.isfinite(times).all(axis=1)
        & (times.iloc[:, 0] >= 0)
        & (times.iloc[:, 1] > times.iloc[:, 0])
    )
    if (labelled & ~valid).any():
        raise ValueError("Labelled confinement rows require finite increasing bounds")
    intervals = (
        pd.DataFrame(
            {
                "shot": shots.astype("int64"),
                "t0_s": times.iloc[:, 0] / 1000,
                "t1_s": times.iloc[:, 1] / 1000,
                "regime": flags.idxmax(axis=1),
            }
        )
        .loc[labelled]
        .copy()
    )
    duplicates = int(intervals.duplicated().sum())
    intervals = intervals.drop_duplicates().sort_values(["shot", "t0_s", "t1_s"])
    # Reject contradictory annotations, but allow adjacent regimes and repeated spans.
    for shot, group in intervals.groupby("shot"):
        records = list(group.itertuples(index=False))
        for i, left in enumerate(records):
            for right in records[i + 1 :]:
                if right.t0_s >= left.t1_s:
                    break
                if left.regime != right.regime:
                    raise ValueError(
                        f"Confinement shot {shot}: conflicting regime intervals"
                    )
    report = {
        "raw_rows": len(raw),
        "raw_shots": int(shots.nunique()),
        "labelled_rows": int(labelled.sum()),
        "duplicate_intervals_removed": duplicates,
        "unlabelled_rows": int((~labelled).sum()),
        "unlabelled_rows_without_valid_bounds": int((~labelled & ~valid).sum()),
        "unlabelled_csv_rows": (np.flatnonzero(~labelled) + 2).tolist(),
        "test_only": "Retain explicitly labelled rows; no train/test split is generated",
    }
    return intervals, sorted(shots.astype("int64").unique().tolist()), report


def confinement_grids(intervals, shots, positive_regimes):
    """Any-positive aggregation in 50 ms bins, unknown outside explicit labels."""
    if not positive_regimes or not set(positive_regimes) <= set(CONFINEMENT_MODES):
        raise ValueError("Invalid positive confinement regimes")
    for shot in shots:
        rows = intervals.loc[intervals.shot == shot]
        stop_ms = max(6000, float(rows.t1_s.max()) * 1000) if len(rows) else 6000
        times = np.arange(0, np.ceil(stop_ms / SAMPLE_MS) * SAMPLE_MS, SAMPLE_MS)
        coverage = sample_event_labels(rows, times)["label"] > 0
        positive = rows.loc[rows.regime.isin(positive_regimes)]
        grid = sample_event_labels(positive, times)
        grid["label"][~coverage] = np.nan
        yield shot, grid


def format_confinement(path: Path, category: str, *, high_regimes=("H", "QH", "WP")):
    """Return internal intervals, per-shot grids, and audit metadata for H or L."""
    if category not in {"high_confinement_mode", "low_confinement_mode"}:
        raise ValueError("Expected high_confinement_mode or low_confinement_mode")
    original, shots, report = read_confinement(path)
    positive = high_regimes if category == "high_confinement_mode" else ("L",)
    grids = dict(confinement_grids(original, shots, positive))
    rows = []
    for shot, grid in grids.items():
        for row in grid_intervals(shot, grid).itertuples(index=False):
            rows.append(
                _row(
                    shot,
                    row.t_start / 1000,
                    row.t_end / 1000,
                    "hmode" if category == "high_confinement_mode" else "lmode",
                    "jalal_confinement_2024",
                    {"category": row.category},
                )
            )
    report["positive_raw_regimes"] = list(positive)
    return _frame(rows), grids, report


def tm_label_grids(h5_path=None, tar_path=None):
    """Merge original binary samples into 50 ms bins; any positive source wins.

    Bins containing only supplied zeros remain zero, and unsampled bins remain
    unknown. Preserve all source shots, including those with no positive labels.
    """
    grids = {}
    sources = []
    if h5_path is not None:
        sources.append(read_tm_h5(h5_path))
    if tar_path is not None:
        sources.append(read_tm_tar(tar_path))
    for traces in sources:
        for shot, times, labels, _ in traces:
            sampled = bin_binary_samples(times, labels)
            if not len(sampled["time_ms"]):
                continue
            old = grids.get(shot)
            start = min(0, sampled["time_ms"][0], old["time_ms"][0] if old else 0)
            stop = max(
                6000,
                sampled["time_ms"][-1] + SAMPLE_MS,
                old["time_ms"][-1] + SAMPLE_MS if old else 0,
            )
            axis = np.arange(start, stop, SAMPLE_MS)
            values = np.full(len(axis), np.nan)
            for incoming in [old, sampled] if old else [sampled]:
                index = np.searchsorted(axis, incoming["time_ms"])
                values[index] = np.fmax(values[index], incoming["label"])
            grids[shot] = {"time_ms": axis, "label": values}
    return grids


def sha256(path: Path) -> str:
    """A file's digest; a folder's is the digest of its `<digest>  <name>` lines."""
    path = Path(path)
    if path.is_dir():
        listing = "".join(
            f"{sha256(file)}  {file.name}\n"
            for file in sorted(path.iterdir())
            if file.is_file()
        )
        return hashlib.sha256(listing.encode()).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _made_from(stem: str, path: Path, entry: dict) -> dict:
    record = {
        "raw_file": entry["path"],
        "sha256": sha256(path),
        "source": stem,
        "provenance": entry["provenance"],
    }
    if path.is_dir():
        record["files"] = {
            file.name: sha256(file) for file in sorted(path.iterdir()) if file.is_file()
        }
    return record


def convert_category(
    category: str,
    root: Path,
    *,
    manifest=None,
    output=None,
    ae_start_s=0.0,
    ae_stop_s=2.0,
    dataset=None,
) -> Path:
    """Use events.yaml names and source paths; emit CSV and checksum metadata.

    `manifest` is the events.yaml to read (default `Paths.events_manifest`); its
    raw paths are relative to the label tables `root`. A category may list several
    format datasets; `dataset` (an output stem or file name) picks one, and None
    picks the first.
    """
    root = Path(root).resolve()
    manifest_file = Path(
        Paths.from_env().events_manifest if manifest is None else manifest
    )
    manifest = yaml.safe_load(manifest_file.read_text())
    spec = next(
        (
            row
            for row in manifest["format_datasets"]
            if row["name"] == category
            and dataset in (None, Path(row["raw_path"]).stem, row["raw_path"])
        ),
        None,
    )
    if spec is None:
        raise ValueError(f"No format dataset {dataset!r} listed for {category}")
    raw = {row["stem"]: row for row in manifest["raw_datasets"]}
    paths = {stem: root / raw[stem]["path"] for stem in spec["sources"]}
    ae_ece = category == "alfven_eigenmode" and AE_ECE_STEM in paths
    elm_challenge = category == "edge_localized_mode" and ELM_CHALLENGE_STEM in paths
    confinement = None
    if category in {"high_confinement_mode", "low_confinement_mode"}:
        frame, grids, report = format_confinement(
            next(iter(paths.values())),
            category,
            high_regimes=spec.get("positive_raw_regimes", ["H", "QH", "WP"]),
        )
        confinement = (grids, report)
    elif category == "resistive_wall_mode":
        frame = pd.concat(
            [format_rwm(path) for path in paths.values()], ignore_index=True
        )
    elif category == "alfven_eigenmode" and ae_ece:
        if len(paths) != 1:
            raise ValueError("The ECE boxes are formatted on their own")
        ece_records = read_ae_ece(paths[AE_ECE_STEM])
        frame = format_ae_ece_records(ece_records)
    elif category == "alfven_eigenmode":
        frame = format_ae(
            paths["co2_detector_2021"], start_s=ae_start_s, stop_s=ae_stop_s
        )
    elif elm_challenge:
        if len(paths) != 1:
            raise ValueError("The ELM challenge key is formatted on its own")
        frame = format_elm_challenge(paths[ELM_CHALLENGE_STEM])
    elif category == "edge_localized_mode":
        frame = format_elm(*paths.values())
    elif category == "neoclassical_tearing_mode":
        frame = format_tm(paths.get("tm_labels_h5"), paths.get("tm_labels_archive"))
    else:
        raise ValueError(f"No formatter for {category}")
    out = (
        Path(output).resolve()
        if output
        else root / category / "format" / spec["raw_path"]
    )
    if any(
        out == path.resolve() or path.parent.resolve() in out.parents
        for path in paths.values()
    ):
        raise ValueError("Output must not overwrite raw data or be inside raw/")
    meta = {
        "schema_version": FORMAT_SCHEMA_VERSION,
        "made_from": [
            _made_from(stem, path, raw[stem]) for stem, path in paths.items()
        ],
        "category": category,
        "categories": category_labels(category),
        "made_by": f"python -m labeler.events.source_formatters {category}",
        "made_at": spec.get("date") or CONVERSION_DATE,
    }
    if elm_challenge:
        meta["label_mapping"] = {
            "time_units": "ms",
            "scope": "DIII-D cases only; other devices in the key are skipped",
            "window": "each case's tstart..tend is assessed; time outside it has no "
            "row and is not assessed",
            "states": "ELM types 1, 2 and 6 are present; types 3 and 4 (other "
            "transients) and 8 are uncertain; types 5 and 7 (pellet, no ELM) and "
            "event-free window time are absent",
            "type_to_state": {str(k): v for k, v in ELM_CHALLENGE_STATES.items()},
            "overlap": "present beats uncertain beats absent; segments do not "
            "overlap and abutting segments of one state are joined",
            "edges": "nominal start and end; the key's edge uncertainties are not "
            "carried (the raw key has them) and event types are not carried either",
            "confidence": "unknown",
            "limitation": "Segments join abutting ELMs, so counts and types come "
            "from the raw key, not this table",
            "reserved": "evaluation set: the shots are holdout rows of the ELM roster "
            "and are not used in fitting or tuning",
        }
        meta["source_audit"] = elm_challenge_audit(paths[ELM_CHALLENGE_STEM])
    elif category == "edge_localized_mode":
        meta["label_mapping"] = {
            "time_units": "ms",
            "cadence_ms": 1,
            "interval_bounds": "half_open_50ms_bins",
            "sample_interval_ms": SAMPLE_MS,
            "aggregation": "category = 1 when the bin contains any labelled onset",
            "overlap": "matching WPQH samples deduplicated; conflicts rejected",
            "source_reference": "wpqh_elm_hiro/hiro_scripts/data_processing.ipynb",
            "limitation": "Original labels simplify burst width to onset samples",
        }
    if ae_ece:
        meta["label_mapping"] = (
            "1 where any TAE/RSAE/BAE/EAE box of any ECE channel covers the time; "
            "LFM boxes ignored; nothing is marked 0"
        )
        meta["time_mapping"] = {
            "method": "union_of_box_time_spans",
            "raw_time_units": "s",
            "interval_bounds": "the spans as drawn, converted to ms",
            "grid": "50 ms bins that a span overlaps are 1; all other bins unknown",
            "unknown": "every time outside a span: the files do not say which "
            "stretch of the shot was inspected",
            "ignored_columns": ["f1", "f2", "channel"],
            "excluded_class": "lfm",
            "limitation": "Boxes are drawn on spectrograms plotted from 0 to about "
            "2 s and most start at 0.30 s, so a mode already present before 0.30 s "
            "is cut at 0.30 s; ECE channel and frequency are not carried over",
        }
        meta["source_audit"] = ae_ece_audit(ece_records)
    elif category == "alfven_eigenmode":
        meta["label_mapping"] = (
            "1 if any BAE/EAE/RSAE/TAE is active; otherwise 0; LFM excluded"
        )
        meta["time_mapping"] = {
            "method": "uniform_bins_over_declared_window_then_50ms_bins",
            "sample_interval_ms": SAMPLE_MS,
            "interval_bounds": "half_open_50ms_bins",
            "start_s": ae_start_s,
            "stop_s": ae_stop_s,
            "reference": "aemodes/scripts/make_ts_dataset.py: time_range=(0, 2000)",
            "limitation": "Original pickle has no timestamps; STFT times not recovered",
            "excluded_class": "lfm",
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    if category == "edge_localized_mode" and not elm_challenge:
        folder = out.parent / "shots"
        folder.mkdir(parents=True, exist_ok=True)
        for shot, times, labels in read_elm(*paths.values()):
            grid = bin_binary_samples(times, labels)
            write_label_grid(
                folder / f"{shot}.npz",
                grid["time_ms"],
                grid["label"],
                event_count=grid["event_count"],
                categories=category_labels(category),
            )
        meta["per_shot_files"] = {
            "path": "shots/<shot>.npz",
            "sample_interval_ms": SAMPLE_MS,
            "time_coordinate": "left edge of half-open bin",
            "event_count": "number of original positive onset samples in each bin",
            "classes": category_labels(category),
            "rho_bins": 20,
            "radial_mapping": "broadcast scalar label across all rho bins",
        }
    if elm_challenge:
        # Beside, never over, the pinned grids in `shots/`: 163518 is in both.
        folder = out.parent / "challenge_shots"
        folder.mkdir(parents=True, exist_ok=True)
        grids = dict(elm_challenge_grids(project_intervals(validate_format(frame))))
        for shot, grid in grids.items():
            write_label_grid(
                folder / f"{shot}.npz",
                grid["time_ms"],
                grid["label"],
                categories=category_labels(category),
            )
        meta["per_shot_files"] = {
            "path": "challenge_shots/<shot>.npz",
            "n_shots": len(grids),
            "sample_interval_ms": SAMPLE_MS,
            "time_coordinate": "left edge of half-open bin",
            "classes": category_labels(category),
            "rho_bins": 20,
            "radial_mapping": "broadcast scalar label across all rho bins",
            "label_origin": "1 where any ELM touches the bin; 0 where only absent "
            "segments do; uncertain-only and unassessed bins unknown",
        }
    if confinement is not None:
        grids, report = confinement
        folder = out.parent / "shots"
        folder.mkdir(parents=True, exist_ok=True)
        for shot, grid in grids.items():
            write_label_grid(
                folder / f"{shot}.npz",
                grid["time_ms"],
                grid["label"],
                categories=category_labels(category),
            )
        meta["source_audit"] = report
        meta["label_mapping"] = {
            "positive_raw_regimes": report["positive_raw_regimes"],
            "zero": "Other explicitly labelled regimes; never unannotated gaps",
            "aggregation": "Any overlap with a positive interval in each half-open 50 ms bin",
            "unknown": "No overlap with any explicitly labelled regime",
            "time_columns": list(CONFINEMENT_TIMES),
            "raw_time_units": "ms",
            "interval_bounds": "half_open_50ms_bins",
            "confidence": "unknown",
        }
        meta["per_shot_files"] = {
            "path": "shots/<shot>.npz",
            "n_shots": len(grids),
            "sample_interval_ms": SAMPLE_MS,
            "time_coordinate": "left edge of half-open bin",
            "time_range": "0..6000 ms, extended if a labelled interval ends later",
            "classes": category_labels(category),
            "rho_bins": 20,
            "radial_mapping": "broadcast scalar label across all rho bins",
        }
    if category in {
        "alfven_eigenmode",
        "resistive_wall_mode",
        "neoclassical_tearing_mode",
    }:
        public = project_intervals(validate_format(frame))
        if category == "neoclassical_tearing_mode":
            grids = tm_label_grids(
                paths.get("tm_labels_h5"), paths.get("tm_labels_archive")
            )
            origin = "Original sampled binary labels; any positive source sample wins; supplied zeros retained"
        else:
            grids = dict(
                formatted_binary_grids(
                    public,
                    minimum_stop_ms=(0 if category == "alfven_eigenmode" else 6000),
                )
            )
            origin = "Formatted binary intervals; positive overlap wins; unannotated bins unknown"
        # The ECE boxes' grids stay out of `shots/`, which holds the pinned CO2 grids.
        shots_dir = "ece_shots" if ae_ece else "shots"
        folder = out.parent / shots_dir
        folder.mkdir(parents=True, exist_ok=True)
        for shot, grid in grids.items():
            write_label_grid(
                folder / f"{shot}.npz",
                grid["time_ms"],
                grid["label"],
                categories=category_labels(category),
            )
        meta["per_shot_files"] = {
            "path": f"{shots_dir}/<shot>.npz",
            "n_shots": len(grids),
            "sample_interval_ms": SAMPLE_MS,
            "time_coordinate": "left edge of half-open bin",
            "classes": category_labels(category),
            "rho_bins": 20,
            "radial_mapping": "broadcast scalar label across all rho bins",
            "label_origin": origin,
        }
    write_interval_table(project_intervals(validate_format(frame)), out, meta)
    return out


def main(argv=None) -> int:
    paths = Paths.from_env()
    parser = argparse.ArgumentParser(
        prog="python -m labeler.events.source_formatters",
        description="Format one category's original labels into the interval schema",
    )
    parser.add_argument("category", help="the event directory, e.g. alfven_eigenmode")
    parser.add_argument(
        "--root",
        type=Path,
        default=paths.label_tables,
        help="the label tables the raw paths are relative to "
        "(default: LABELER_LABEL_TABLES)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=paths.events_manifest,
        help="the events.yaml to read (default: configs/labeler/events.yaml)",
    )
    parser.add_argument("--output", type=Path, help="Override the output CSV path")
    parser.add_argument(
        "--dataset",
        help="Format dataset (output stem or file name) when events.yaml lists "
        "several for the category; default: the first",
    )
    parser.add_argument(
        "--start-s", type=float, default=0.0, help="alfven_eigenmode only"
    )
    parser.add_argument("--stop-s", type=float, default=2.0, help="alfven_eigenmode only")
    args = parser.parse_args(argv)
    print(
        convert_category(
            args.category,
            args.root,
            manifest=args.manifest,
            output=args.output,
            dataset=args.dataset,
            ae_start_s=args.start_s,
            ae_stop_s=args.stop_s,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
