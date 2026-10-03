"""One label per shot, kept as a format table.

`review/labels.csv` holds the current label of every reviewed shot in the format
schema (`shot, category, t_start, t_end, confidence`, ms): individual and crowd
spans may overlap; their shared gaps are category 0. `history.jsonl`
gets one line per save. A shot nobody has saved opens on its source label: the
table `review/source.json` points at (a suggestion table), else the newest
`format/*_format_*.csv`.
"""

from __future__ import annotations

import getpass
import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import atomic_path
from ..catalog.points import validate_csv_fields
from ..catalog.states import NOT_OBSERVABLE, STATE_NAMES
from ..interval_tables import (
    INTERVAL_COLUMNS,
    WITH_ATTRS,
    attrs_text,
    category_labels,
    crowd_flag,
    parse_attrs,
    validate_intervals,
)
from ..times import whole_ms

LONGEST_WINDOW_MS = 20_000
REVIEW = "review"
_write_lock = threading.Lock()


@dataclass(frozen=True)
class Label:
    """A reviewed window and the categorised spans inside it, in whole ms."""

    window: tuple[int, int]
    intervals: tuple[tuple[int, int, int], ...] = ()
    #: Aligned to intervals: 0 individual, 1 group, None unspecified. Empty
    #: means a legacy category-only label; it does not assert instance counts.
    iscrowd: tuple[int | None, ...] = ()

    def as_json(self) -> dict:
        return {
            "window": list(self.window),
            "intervals": [list(span) for span in self.intervals],
            **({"iscrowd": list(self.iscrowd)} if self.iscrowd else {}),
        }

    def rows(self, shot: int) -> list[list]:
        """Annotation rows and the category-0 gaps outside their combined coverage."""
        lo = self.window[0]
        if self.iscrowd:
            result = []
            for (a, b, c), flag in zip(self.intervals, self.iscrowd, strict=True):
                if a > lo:
                    result.append([int(shot), 0, lo, a, "", ""])
                attrs = "" if flag is None else attrs_text({"iscrowd": flag})
                result.append([int(shot), c, a, b, "", attrs])
                lo = max(lo, b)
            if lo < self.window[1]:
                result.append([int(shot), 0, lo, self.window[1], "", ""])
            return result
        cells = _paint(self.window, self.intervals)
        return [[int(shot), c, lo + a, lo + b, ""] for a, b, c in _runs(cells)]


#: Events the page lists as one, each with the event it is reviewed as. The
#: confinement regimes were four events; `confinement` carries them all, on the
#: H-mode event's cohort and rows.
FOLDED = {
    "high_confinement_mode": "confinement",
    "low_confinement_mode": "confinement",
    "quiescent_high_confinement_mode": "confinement",
    "wide_pedestal_quiescent_high_confinement_mode": "confinement",
}


def categories(event: str) -> dict[int, str]:
    """The categories a span can carry; 0 (absent) is the gaps.

    A catalog phenomenon's spans carry present and uncertain; q_min and
    confinement add their regimes. Not observable is not offered: whatever the
    reader marks neither present nor uncertain is a gap.
    """
    hidden = STATE_NAMES[NOT_OBSERVABLE]
    return {
        int(k): v for k, v in category_labels(event).items() if int(k) and v != hidden
    }


def offered(event: str, label: Label | None) -> Label | None:
    """`label` as the page shows it: a span in a category the page does not offer
    (a table's not-observable stretch) is a gap, like any other time left unmarked."""
    if label is None:
        return None
    shown = categories(event)
    keep = [i for i, (_, _, c) in enumerate(label.intervals) if c in shown]
    if len(keep) == len(label.intervals):
        return label
    return Label(
        label.window,
        tuple(label.intervals[i] for i in keep),
        tuple(label.iscrowd[i] for i in keep) if label.iscrowd else (),
    )


def _ms(t) -> int:
    """Whole ms, halves up (JavaScript's `Math.floor(t + 0.5)`, so the page agrees)."""
    return whole_ms(t)


def normalise(
    window, intervals, known: set[int] | None = None, *, iscrowd=None
) -> Label:
    """Snap to whole ms, grow the window over every span, merge, and clip.

    Later spans paint over earlier ones within the same annotation lane.
    Crowds overlap individual/unspecified annotations. An unscoped category-0
    span erases both lanes; an explicit flag limits erasure to that lane.
    Explicit annotation resolution retains each input annotation's boundaries.
    Legacy labels still merge equal categories; unspecified is not individual.
    """
    lo, hi = _ms(window[0]), _ms(window[1])
    intervals = list(intervals)
    if iscrowd is not None and len(iscrowd) != len(intervals):
        raise ValueError("iscrowd must have one flag per interval")
    flags = [None] * len(intervals) if iscrowd is None else [
        None if flag is None else crowd_flag(flag) for flag in iscrowd
    ]
    spans, resolution = [], []
    for (a, b, c), flag in zip(intervals, flags, strict=True):
        a, b, c = _ms(a), _ms(b), int(c)
        if b < a:
            raise ValueError(f"span {a}-{b} ms runs backwards")
        if known is not None and c and c not in known:
            raise ValueError(f"category {c} is not one of {sorted(known)}")
        if b > a:
            spans.append((a, b, c))
            resolution.append(flag)
    painted = [span for span in spans if span[2]]
    if painted:
        lo = min(lo, *(a for a, _, _ in painted))
        hi = max(hi, *(b for _, b, _ in painted))
    if not 0 < hi - lo <= LONGEST_WINDOW_MS:
        raise ValueError(f"window {lo}-{hi} ms is not 1 to {LONGEST_WINDOW_MS} ms long")
    if any(flag is not None for flag in resolution):
        runs = []
        for crowd in (False, True):
            ids = []
            for i, (a, b, c) in enumerate(spans):
                flag = resolution[i]
                if (c or flag is not None) and (flag == 1) != crowd:
                    continue
                ids.append((a, b, i + 1 if c else 0))
            runs.extend((a, b, i) for a, b, i in _runs(_paint((lo, hi), ids)) if i)
        runs.sort(key=lambda run: (
            run[0], run[1], resolution[run[2] - 1] == 1,
        ))
        kept_flags = tuple(resolution[i - 1] for _, _, i in runs)
        kept_spans = tuple((lo + a, lo + b, spans[i - 1][2]) for a, b, i in runs)
        if any(flag is not None for flag in kept_flags):
            return Label((lo, hi), kept_spans, kept_flags)
        spans = kept_spans
    runs = _runs(_paint((lo, hi), spans))
    return Label((lo, hi), tuple((lo + a, lo + b, c) for a, b, c in runs if c))


def has_overlaps(label: Label | None) -> bool:
    """Whether a single-lane client would flatten this label's annotations."""
    if label is None:
        return False
    end = label.window[0]
    for a, b, _ in sorted(label.intervals):
        if a < end:
            return True
        end = max(end, b)
    return False


def _paint(window, spans) -> np.ndarray:
    """One cell per ms of the window, holding the category painted there last."""
    lo, hi = window
    cells = np.zeros(hi - lo, dtype=np.int64)
    for a, b, c in spans:
        start, stop = max(a, lo) - lo, min(b, hi) - lo
        if stop > start:
            cells[start:stop] = c
    return cells


def _runs(cells) -> list[tuple[int, int, int]]:
    """(start, stop, category) of each run of equal cells, as offsets."""
    if not len(cells):
        return []
    edges = np.flatnonzero(np.diff(cells)) + 1
    starts = np.concatenate([[0], edges])
    stops = np.concatenate([edges, [len(cells)]])
    return [(int(a), int(b), int(cells[a])) for a, b in zip(starts, stops)]


def labels_path(event_dir) -> Path:
    return Path(event_dir) / REVIEW / "labels.csv"


def history_path(event_dir) -> Path:
    return Path(event_dir) / REVIEW / "history.jsonl"


def pointer_path(event_dir) -> Path:
    return Path(event_dir) / REVIEW / "source.json"


def source_path(event_dir) -> Path | None:
    """The table a shot nobody has saved opens on.

    The one `review/source.json` names, when the event has one; else the newest
    format table by name (`*_format_*`: RWM's has no event prefix). A pointer
    to a table that is gone raises: falling back would open every shot on
    another method's labels.
    """
    pointer = pointer_path(event_dir)
    if pointer.is_file():
        table = Path(json.loads(pointer.read_text())["table"])
        if not table.is_file():
            raise FileNotFoundError(f"{pointer} names {table}, which does not exist")
        return table
    tables = Path(event_dir).glob("format/*_format_*.csv")
    return max(tables, key=lambda path: path.name, default=None)


def write_pointer(event_dir, table, *, method: str, version: str) -> dict:
    """Point the event's review page at `table`, a suggestion table."""
    table = Path(table).resolve()
    if not table.is_file():
        raise FileNotFoundError(f"no suggestion table at {table}")
    entry = {
        "table": str(table),
        "method": method,
        "version": version,
        "set_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "set_by": getpass.getuser(),
    }
    with atomic_path(pointer_path(event_dir)) as tmp:
        tmp.write_text(json.dumps(entry, indent=1) + "\n")
    return entry


def read_source(event_dir) -> dict[int, Label]:
    return _table(source_path(event_dir))


def read_saved(event_dir) -> dict[int, Label]:
    return _table(labels_path(event_dir))


def read_labels(path) -> dict[int, Label]:
    """One label per shot of any format table at `path`; none if it is absent."""
    return _table(Path(path))


def _table(path) -> dict[int, Label]:
    if path is None or not path.is_file():
        return {}
    stat = path.stat()
    return _read_table(path, stat.st_mtime_ns, stat.st_ino, stat.st_size)


@lru_cache(maxsize=64)
def _read_table(path, _mtime_ns, _ino, _size) -> dict[int, Label]:
    """One label per shot of a format table; cached per file version, never mutate."""
    validate_csv_fields(path)
    frame = validate_intervals(pd.read_csv(path, index_col=False))
    frame = frame[frame.t_end - frame.t_start >= 1]  # a point event has no span to edit
    found = {}
    for shot, rows in frame.groupby("shot", sort=False):
        spans = rows[rows.category != 0]
        found[int(shot)] = normalise(
            (rows.t_start.min(), rows.t_end.max()),
            zip(spans.t_start, spans.t_end, spans.category),
            iscrowd=([parse_attrs(cell).get("iscrowd") for cell in spans["attrs"]]
                     if "attrs" in spans else None),
        )
    return found


def read_history(event_dir) -> list[dict]:
    path = history_path(event_dir)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class SaveRefused(ValueError):
    """The page cannot save safely over the existing table."""


def save(
    event_dir,
    shot: int,
    label: Label,
    *,
    source: str | None,
    name: str | None = None,
    source_sha256: str | None = None,
    crowd_edit: bool = False,
) -> dict:
    """Replace one shot's rows in `labels.csv` and append the save to history.

    Refuse malformed CSV records, ragged rows or duplicate headers, and any
    unsupported attrs on the selected shot (the page can edit `iscrowd`). Validate
    both the existing and replacement-combined interval tables: exact columns,
    nonnegative integral int64 shot/category values, finite time bounds with
    end >= start, confidence missing or in [0, 1], and plain JSON object attrs
    without duplicate keys or nonfinite values. These checks precede writing.
    Other shots retain their original precision and attrs text. This validates
    interval schema; catalog tiling, observability, points and allowed windows
    are checked separately by the catalog checker.

    `reviewer` is the server's login; `name` is the reviewer's name, or None
    (see `versions`). `source` names the table the save was made against and
    `source_sha256` is that table's sha256 when it was made, kept in the history
    line when given (see `agreement`: a save without one is matched to the table
    by name).
    """
    with _write_lock:
        path = labels_path(event_dir)
        if path.is_file():
            try:
                validate_csv_fields(path)
            except pd.errors.ParserError as error:
                raise SaveRefused(str(error)) from error
        # Read cells as written: other shots keep their precision and attrs text.
        current = (
            pd.read_csv(path, dtype=str, keep_default_na=False, index_col=False)
            if path.is_file()
            else pd.DataFrame(columns=list(INTERVAL_COLUMNS))
        )
        selected = pd.to_numeric(current.shot, errors="coerce") == int(shot)
        validate_intervals(current)
        if "attrs" in current:
            for cell in current.loc[selected, "attrs"]:
                if not cell.strip():
                    continue
                if set(parse_attrs(cell)) != {"iscrowd"}:
                    raise SaveRefused(
                        f"shot {shot} has attrs that the review page cannot edit; "
                        "save refused"
                    )
                if not label.iscrowd and not crowd_edit:
                    raise SaveRefused(
                        f"shot {shot} has iscrowd metadata; restart the review "
                        "server before editing its annotation resolution"
                    )
        columns = WITH_ATTRS if label.iscrowd else INTERVAL_COLUMNS
        replacement = pd.DataFrame(label.rows(shot), columns=list(columns))
        if "attrs" in current and "attrs" not in replacement:
            replacement["attrs"] = ""
        elif "attrs" in replacement and "attrs" not in current:
            current["attrs"] = ""
        frame = pd.concat([current.loc[~selected], replacement], ignore_index=True)
        frame = frame.sort_values("shot", key=pd.to_numeric, kind="stable")
        validate_intervals(frame)
        with atomic_path(path) as tmp:
            frame.to_csv(tmp, index=False)
        entry = {
            "shot": int(shot),
            "reviewer": getpass.getuser(),
            "name": name,
            "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **label.as_json(),
            "source": source,
        }
        if source_sha256 is not None:
            entry["source_sha256"] = source_sha256
        with history_path(event_dir).open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
    return entry


def state(saved: Label | None, source: Label | None) -> str:
    if saved is None:
        return "unreviewed"
    if source is None:
        return "confirmed" if not saved.intervals else "changed"
    return "confirmed" if saved == source else "changed"


def queue(event_dir, roster: pd.DataFrame) -> dict:
    """The roster in order, each shot's state and last save, and where to resume."""
    event = Path(event_dir).name
    saved, source = read_saved(event_dir), read_source(event_dir)
    history = read_history(event_dir)
    saved_at = {entry["shot"]: entry["saved_at"] for entry in history}
    shots = [
        {
            "shot": int(row.shot),
            "tier": row.tier,
            "state": state(
                offered(event, saved.get(int(row.shot))),
                offered(event, source.get(int(row.shot))),
            ),
            "saved_at": saved_at.get(int(row.shot)),
        }
        for row in roster.itertuples()
    ]
    return {"shots": shots, "resume": resume(shots, history)}


def resume(shots: list[dict], history: list[dict]) -> int | None:
    """The first unreviewed shot after the newest save, wrapping; else that save."""
    if not shots:
        return None
    order = [row["shot"] for row in shots]
    last = history[-1]["shot"] if history else None
    start = order.index(last) + 1 if last in order else 0
    for row in shots[start:] + shots[:start]:
        if row["state"] == "unreviewed":
            return row["shot"]
    return last if last in order else order[0]


def shot_labels(event_dir, shot: int, *, suppress_source=False) -> dict:
    shot = int(shot)
    event = Path(event_dir).name
    saved = offered(event, read_saved(event_dir).get(shot))
    source = None if suppress_source else offered(event, read_source(event_dir).get(shot))
    history = read_history(event_dir)
    last = next((entry for entry in reversed(history) if entry["shot"] == shot), None)
    return {
        "source": None if source is None else source.as_json(),
        "saved": None if saved is None else saved.as_json(),
        "state": state(saved, source),
        "last_save": last,
    }
