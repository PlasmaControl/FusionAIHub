"""One label per shot, kept as a format table.

`review/labels.csv` holds the current label of every reviewed shot in the format
schema (`shot, category, t_start, t_end, confidence`, ms): a shot's rows tile its
window, each span with its category and the gaps as category 0. `history.jsonl`
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
from ..catalog.states import NOT_OBSERVABLE, PHENOMENA
from ..interval_tables import INTERVAL_COLUMNS, category_labels, validate_intervals
from ..times import whole_ms

LONGEST_WINDOW_MS = 20_000
REVIEW = "review"
_write_lock = threading.Lock()


@dataclass(frozen=True)
class Label:
    """A reviewed window and the categorised spans inside it, in whole ms."""

    window: tuple[int, int]
    intervals: tuple[tuple[int, int, int], ...] = ()

    def as_json(self) -> dict:
        return {
            "window": list(self.window),
            "intervals": [list(span) for span in self.intervals],
        }

    def rows(self, shot: int) -> list[list]:
        """Format-table rows tiling the window; the gaps are category 0."""
        lo = self.window[0]
        cells = _paint(self.window, self.intervals)
        return [[int(shot), c, lo + a, lo + b, ""] for a, b, c in _runs(cells)]


def categories(event: str) -> dict[int, str]:
    """The categories a span can carry; 0 (absent) is the gaps.

    A catalog phenomenon's spans carry its states: present, uncertain and not
    observable, except phenomena that are always observable.
    """
    excluded = {0}
    if event in PHENOMENA and PHENOMENA[event].observable_always:
        excluded.add(NOT_OBSERVABLE)
    return {
        int(k): v for k, v in category_labels(event).items() if int(k) not in excluded
    }


def _ms(t) -> int:
    """Whole ms, halves up (JavaScript's `Math.floor(t + 0.5)`, so the page agrees)."""
    return whole_ms(t)


def normalise(window, intervals, known: set[int] | None = None) -> Label:
    """Snap to whole ms, grow the window over every span, merge, and clip.

    Later spans paint over earlier ones, so a category-0 span erases.
    """
    lo, hi = _ms(window[0]), _ms(window[1])
    spans = []
    for a, b, c in intervals:
        a, b, c = _ms(a), _ms(b), int(c)
        if b < a:
            raise ValueError(f"span {a}-{b} ms runs backwards")
        if known is not None and c and c not in known:
            raise ValueError(f"category {c} is not one of {sorted(known)}")
        if b > a:
            spans.append((a, b, c))
    painted = [span for span in spans if span[2]]
    if painted:
        lo = min(lo, *(a for a, _, _ in painted))
        hi = max(hi, *(b for _, b, _ in painted))
    if not 0 < hi - lo <= LONGEST_WINDOW_MS:
        raise ValueError(f"window {lo}-{hi} ms is not 1 to {LONGEST_WINDOW_MS} ms long")
    runs = _runs(_paint((lo, hi), spans))
    return Label((lo, hi), tuple((lo + a, lo + b, c) for a, b, c in runs if c))


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
    event_dir, shot: int, label: Label, *, source: str | None, name: str | None = None
) -> dict:
    """Replace one shot's rows in `labels.csv` and append the save to history.

    Refuse malformed CSV records, ragged rows or duplicate headers, and any
    nonblank attrs on the selected shot (the page cannot edit them). Validate
    both the existing and replacement-combined interval tables: exact columns,
    nonnegative integral int64 shot/category values, finite time bounds with
    end >= start, confidence missing or in [0, 1], and plain JSON object attrs
    without duplicate keys or nonfinite values. These checks precede writing.
    Other shots retain their original precision and attrs text. This validates
    interval schema; catalog tiling, observability, points and allowed windows
    are checked separately by the catalog checker.

    `reviewer` is the server's login; `name` is what the reviewer typed, or None
    (see `versions`).
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
        if (
            "attrs" in current
            and current.loc[selected, "attrs"].str.strip().ne("").any()
        ):
            raise SaveRefused(
                f"shot {shot} has attrs that the review page cannot edit; save refused"
            )
        validate_intervals(current)
        replacement = pd.DataFrame(label.rows(shot), columns=list(INTERVAL_COLUMNS))
        if "attrs" in current:
            replacement["attrs"] = ""
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
    saved, source = read_saved(event_dir), read_source(event_dir)
    history = read_history(event_dir)
    saved_at = {entry["shot"]: entry["saved_at"] for entry in history}
    shots = [
        {
            "shot": int(row.shot),
            "tier": row.tier,
            "state": state(saved.get(int(row.shot)), source.get(int(row.shot))),
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


def shot_labels(event_dir, shot: int) -> dict:
    shot = int(shot)
    saved = read_saved(event_dir).get(shot)
    source = read_source(event_dir).get(shot)
    history = read_history(event_dir)
    last = next((entry for entry in reversed(history) if entry["shot"] == shot), None)
    return {
        "source": None if source is None else source.as_json(),
        "saved": None if saved is None else saved.as_json(),
        "state": state(saved, source),
        "last_save": last,
    }
