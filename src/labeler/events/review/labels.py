"""One label per shot, kept as a format table.

`review/labels.csv` holds the current label of every reviewed shot in the format
schema (`shot, category, t_start, t_end, confidence`, ms): individual and crowd
spans may overlap; their shared gaps are category 0. `history.jsonl`
gets one line per save. A save is first written to its writer's own file under
`review/saves/` and then merged into the two files (see `saves`), so several
people may save into one folder: the last save of a shot, by time, is its label.
A shot nobody has saved opens on its source label: the
table `review/source.json` points at (a suggestion table), else the newest
`format/*_format_*.csv`.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import threading
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import atomic_path
from ...env import getenv
from ..catalog.points import validate_csv_fields
from ..catalog.states import NOT_OBSERVABLE, STATE_NAMES
from ..grid import GRID_MS, snap, snap_span, snap_window
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
from . import saves
from .saves import REVIEW

LONGEST_WINDOW_MS = 20_000
_write_lock = threading.Lock()
EVIDENCE_ATTRS = {"evidence_tier", "coverage_verified"}


@dataclass(frozen=True)
class Label:
    """A reviewed window and the categorised spans inside it, in whole ms."""

    window: tuple[int, int]
    intervals: tuple[tuple[int, int, int], ...] = ()
    #: Aligned to intervals: 0 individual, 1 group, None unspecified. Empty
    #: means a legacy category-only label; it does not assert instance counts.
    iscrowd: tuple[int | None, ...] = ()
    #: Evidence-bearing tables retain every explicit span, including category 0,
    #: and its canonical JSON attrs. They require explicit unassessed gaps and
    #: cannot be edited by the category-only review page.
    attrs: tuple[str, ...] = ()

    @classmethod
    def from_entry(cls, entry: dict) -> Label:
        """The label a history line saved."""
        return cls(
            tuple(int(v) for v in entry["window"]),
            tuple((int(a), int(b), int(c)) for a, b, c in entry["intervals"]),
            tuple(entry["iscrowd"]) if entry.get("iscrowd") else (),
        )

    def as_json(self) -> dict:
        return {
            "window": list(self.window),
            "intervals": [list(span) for span in self.intervals],
            **({"iscrowd": list(self.iscrowd)} if self.iscrowd else {}),
            **({"attrs": [parse_attrs(cell) for cell in self.attrs]}
               if self.attrs else {}),
        }

    def rows(self, shot: int) -> list[list]:
        """Annotation rows and the category-0 gaps outside their combined coverage."""
        if self.attrs:
            return [
                [int(shot), c, a, b, "", cell]
                for (a, b, c), cell in zip(self.intervals, self.attrs, strict=True)
            ]
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
#: confinement regimes (and I-mode, now its category 6) were separate events;
#: `confinement` carries them all, on the H-mode event's cohort and rows. The
#: sawtooth, fishbone and tearing-mode events are MHD n=1, whose subcategories
#: (see `taxonomy`) say which (m, n) it is. Their saved data is untouched.
FOLDED = {
    "high_confinement_mode": "confinement",
    "low_confinement_mode": "confinement",
    "quiescent_high_confinement_mode": "confinement",
    "wide_pedestal_quiescent_high_confinement_mode": "confinement",
    "improved_energy_confinement_mode": "confinement",
    "sawtooth_oscillation": "mhd_n1",
    "fishbone": "mhd_n1",
    "neoclassical_tearing_mode": "mhd_n1",
}


def categories(event: str) -> dict[int, str]:
    """The categories a span can carry; 0 (absent) is the gaps.

    A catalog phenomenon's spans carry present and uncertain, as do the MHD n=1..3
    events and every subcategory; q_min and confinement add their regimes. Not
    observable is not offered: whatever the reader marks neither present nor
    uncertain is a gap.
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
    keep = [
        i for i, (_, _, c) in enumerate(label.intervals)
        if c in shown or (label.attrs and c == 0)
    ]
    if len(keep) == len(label.intervals):
        return label
    if label.attrs:
        raise ValueError("the review page cannot discard evidence-bearing spans")
    return Label(
        label.window,
        tuple(label.intervals[i] for i in keep),
        tuple(label.iscrowd[i] for i in keep) if label.iscrowd else (),
    )


def on_grid(label: Label | None, grid_ms=GRID_MS) -> Label | None:
    """`label` on the grid. Spans snap as `grid.snap_span` (a vanishing span keeps
    its midpoint frame); an evidence-bearing label snaps each boundary and drops
    the rows that collapse, so it still tiles its window."""
    if label is None:
        return None
    window = snap_window(*label.window, grid_ms)
    if label.attrs:
        keep = [
            (i, (snap(a, grid_ms), snap(b, grid_ms), c))
            for i, (a, b, c) in enumerate(label.intervals)
            if snap(b, grid_ms) > snap(a, grid_ms)
        ]
        return Label(
            window, tuple(span for _, span in keep),
            tuple(label.iscrowd[i] for i, _ in keep) if label.iscrowd else (),
            tuple(label.attrs[i] for i, _ in keep),
        )
    spans = [(*snap_span(a, b, grid_ms), c) for a, b, c in label.intervals]
    return normalise(window, spans, iscrowd=label.iscrowd or None, grid_ms=grid_ms)


#: Events whose annotations are spans (`iscrowd` 1) wherever a table says nothing
#: about resolution (owner 2026-10-11): an AE or a confinement regime is a stretch of
#: activity, not one countable event. Their subcategories follow them.
SPAN_EVENTS = frozenset({"alfven_eigenmode", "confinement"})


def span_default(event: str, label: Label | None) -> Label | None:
    """`label` with every annotation a span when it carries no resolution at all.
    A table that sets any flag keeps its own; so do evidence-bearing tables."""
    if (label is None or label.iscrowd or label.attrs or not label.intervals
            or event.split(".")[0] not in SPAN_EVENTS):
        return label
    return replace(label, iscrowd=(1,) * len(label.intervals))


def page_label(event: str, label: Label | None) -> Label | None:
    """A whole-ms label as the page shows it: offered categories, on the grid,
    spans where the event's default says so.
    The page, its queues and the silver view all read through this one rule."""
    return span_default(event, on_grid(offered(event, label)))


#: Events whose rows file marks candidate onsets, with the marker label that holds them.
#: The page opens a shot nobody has saved with each onset as an Individual annotation,
#: so the candidates are edited like any source label rather than only drawn.
ONSET_MARKERS = {"edge_localized_mode": "ELM onset"}
#: Events whose curated onsets each start one mode that lasts (RWM: `rwm.onsets`, not
#: a rows file's markers). Their draft makes each onset one Individual annotation over
#: the source's present span from it (`with_onsets`'s `extend`).
ONSET_EXTENTS = frozenset({"resistive_wall_mode"})
#: What the page's save accepts per label (`LabelIn.intervals`); a draft above it is
#: not offered.
MAX_INTERVALS = 1000


def onset_times(event: str, markers) -> list[float]:
    """The event's candidate onsets, in ms, from a rows file's marker sets."""
    name = ONSET_MARKERS.get(event)
    return sorted(
        float(t)
        for item in markers or ()
        if name is not None and item.get("label") == name
        for t in item["t_ms"]
    )


def with_onsets(
    label: Label | None, times, t_range, *, category: int = 1, grid_ms=GRID_MS,
    extend: bool = False,
) -> Label | None:
    """`label` (or, with none, an empty label over `t_range`) as a draft: its spans
    that carry no resolution become Span annotations, and each onset in `times` is an
    Individual annotation of one frame. Onsets in the same frame make one; onsets
    outside the window are left out. `label` itself when it carries evidence attrs,
    when there are no onsets to add, or when the draft would be more than
    `MAX_INTERVALS` annotations (the ticks are still drawn; a draft the page could not
    save is not offered).

    With `extend` (`ONSET_EXTENTS`) an onset starts at its nearest frame edge, as the
    source's spans are snapped, and when a `category` Span holds that edge its
    annotation runs to the Span's end; the Span keeps only the time no onset covers:
    one mode, one annotation. An onset no such Span holds stays one frame."""
    original = label
    if label is not None and label.attrs:
        return label
    if label is None:
        lo = grid_ms * int(np.floor(t_range[0] / grid_ms))
        hi = grid_ms * int(np.ceil(t_range[1] / grid_ms))
        label = Label((lo, min(hi, lo + LONGEST_WINDOW_MS)))
    lo, hi = label.window
    if extend:
        frames = [f for f in sorted({snap(t, grid_ms) for t in times}) if lo <= f < hi]
    else:
        frames = sorted({
            grid_ms * int(np.floor(t / grid_ms)) for t in times if lo <= t < hi
        })
    if not frames or len(label.intervals) + len(frames) > MAX_INTERVALS:
        return original
    flags = label.iscrowd or (None,) * len(label.intervals)
    flags = [1 if flag is None else flag for flag in flags]
    spans = list(label.intervals)
    onsets = [(f, f + grid_ms) for f in frames]
    if extend:
        held = [(a, b) for (a, b, c), flag in zip(spans, flags)
                if c == category and flag]
        onsets = [(f, next((b for a, b in held if a <= f < b), f + grid_ms))
                  for f in frames]
        kept = [((x, y, c), flag)
                for (a, b, c), flag in zip(spans, flags)
                for x, y in (_less(a, b, onsets) if c == category and flag
                             else [(a, b)])]
        spans, flags = [span for span, _ in kept], [flag for _, flag in kept]
    spans += [(a, b, category) for a, b in onsets]
    flags += [0] * len(onsets)
    return normalise(label.window, spans, iscrowd=flags, grid_ms=grid_ms)


def _less(a, b, holes) -> list[tuple]:
    """`[a, b)` without the time in `holes`, as pieces."""
    pieces, cursor = [], a
    for x, y in sorted(holes):
        if y <= cursor or x >= b:
            continue
        if x > cursor:
            pieces.append((cursor, x))
        cursor = max(cursor, y)
    if cursor < b:
        pieces.append((cursor, b))
    return pieces


def _ms(t) -> int:
    """Whole ms, halves up (JavaScript's `Math.floor(t + 0.5)`, so the page agrees)."""
    return whole_ms(t)


def normalise(
    window, intervals, known: set[int] | None = None, *, iscrowd=None, grid_ms=GRID_MS
) -> Label:
    """Snap to the grid, grow the window over every span, merge, and clip.

    With `grid_ms=1`, keep the legacy whole-ms rounding and drop spans that
    round to nothing. On the review grid, short spans keep their midpoint frame.

    Later spans paint over earlier ones within the same annotation lane.
    Crowds overlap individual/unspecified annotations. An unscoped category-0
    span erases both lanes; an explicit flag limits erasure to that lane.
    Explicit annotation resolution retains each input annotation's boundaries.
    Legacy labels still merge equal categories; unspecified is not individual.
    """
    lo, hi = (
        (_ms(window[0]), _ms(window[1])) if grid_ms == 1
        else snap_window(*window, grid_ms)
    )
    intervals = list(intervals)
    if iscrowd is not None and len(iscrowd) != len(intervals):
        raise ValueError("iscrowd must have one flag per interval")
    flags = [None] * len(intervals) if iscrowd is None else [
        None if flag is None else crowd_flag(flag) for flag in iscrowd
    ]
    spans, resolution = [], []
    for (a, b, c), flag in zip(intervals, flags, strict=True):
        a, b = (_ms(a), _ms(b)) if grid_ms == 1 else snap_span(a, b, grid_ms)
        c = int(c)
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


def read_source(event_dir, *, grid_ms=1) -> dict[int, Label]:
    """The pointer's table, whole ms by default: evaluations read it as written.
    The page reads through `page_label`."""
    return _table(source_path(event_dir), grid_ms=grid_ms)


def read_saved(event_dir, *, grid_ms=1) -> dict[int, Label]:
    """Saved labels, whole ms by default: a model's frozen copy stays as trained on."""
    return _table(labels_path(event_dir), grid_ms=grid_ms)


def read_labels(path, *, grid_ms=1) -> dict[int, Label]:
    """Read whole-ms labels; tiered tables retain attrs and explicit absent spans.

    Evidence-bearing tables must tile their window with explicit unassessed rows;
    gaps, overlaps, or missing evidence attributes are refused. Their `Label.attrs`
    is aligned to `Label.intervals` and survives `rows()` and `as_json()`.
    """
    return _table(Path(path), grid_ms=grid_ms)


def _table(path, *, grid_ms) -> dict[int, Label]:
    if path is None or not path.is_file():
        return {}
    stat = path.stat()
    return _read_table(path, stat.st_mtime_ns, stat.st_ino, stat.st_size, grid_ms)


@lru_cache(maxsize=512)  # the silver view reads ~200 tables a shot
def _read_table(path, _mtime_ns, _ino, _size, grid_ms) -> dict[int, Label]:
    """One label per shot of a format table; cached per file version, never mutate."""
    validate_csv_fields(path)
    frame = validate_intervals(pd.read_csv(path, index_col=False))
    found = {}
    for shot, rows in frame.groupby("shot", sort=False):
        if "attrs" in rows and any(
            EVIDENCE_ATTRS.intersection(parse_attrs(cell)) for cell in rows["attrs"]
        ):
            # Shared boundaries snap together so evidence still tiles its window.
            boundary = _ms if grid_ms == 1 else lambda t: snap(t, grid_ms)
            rows = rows[rows.t_end.map(boundary) > rows.t_start.map(boundary)]
            if len(rows):
                found[int(shot)] = _evidence_label(int(shot), rows, grid_ms=grid_ms)
            continue
        rows = rows[
            rows.t_end - rows.t_start >= 1 if grid_ms == 1
            else rows.t_end > rows.t_start
        ]  # point events are not editable
        if not len(rows):
            continue
        spans = rows[rows.category != 0]
        found[int(shot)] = normalise(
            (rows.t_start.min(), rows.t_end.max()),
            zip(spans.t_start, spans.t_end, spans.category),
            iscrowd=([parse_attrs(cell).get("iscrowd") for cell in spans["attrs"]]
                     if "attrs" in spans else None),
            grid_ms=grid_ms,
        )
    return found


def _evidence_label(shot: int, rows: pd.DataFrame, *, grid_ms=1) -> Label:
    """Keep tiered rows intact; never infer absent coverage across their gaps."""
    rows = rows.sort_values(["t_start", "t_end"], kind="stable")
    spans, attrs, flags = [], [], []
    boundary = _ms if grid_ms == 1 else lambda t: snap(t, grid_ms)
    end = boundary(rows.t_start.iloc[0])
    for row in rows.itertuples():
        evidence = parse_attrs(row.attrs)
        if (
            not isinstance(evidence.get("evidence_tier"), str)
            or not evidence["evidence_tier"].strip()
            or type(evidence.get("coverage_verified")) is not bool
        ):
            raise ValueError(
                f"shot {shot} needs evidence_tier and boolean coverage_verified "
                "on every evidence-bearing interval"
            )
        a, b = boundary(row.t_start), boundary(row.t_end)
        if a != end:
            raise ValueError(
                f"shot {shot} evidence intervals must tile without overlaps; "
                "write explicit category-4 unassessed rows for gaps"
            )
        spans.append((a, b, int(row.category)))
        attrs.append(attrs_text(evidence))
        flags.append(evidence.get("iscrowd"))
        end = b
    window = normalise((spans[0][0], end), [], grid_ms=grid_ms).window
    return Label(
        window, tuple(spans),
        tuple(flags) if any(flag is not None for flag in flags) else (),
        tuple(attrs),
    )


def read_history(event_dir) -> list[dict]:
    path = history_path(event_dir)
    if not path.is_file():
        return []
    text = saves.retrying(lambda: path.read_text(), path.parent)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class SaveRefused(ValueError):
    """The page cannot save safely over the existing table."""


def _current(path) -> pd.DataFrame:
    """The table as written: other shots keep their precision and attrs text."""
    if not path.is_file():
        return pd.DataFrame(columns=list(INTERVAL_COLUMNS))
    return saves.retrying(
        lambda: pd.read_csv(path, dtype=str, keep_default_na=False, index_col=False),
        path.parent,
    )


def _check_editable(current: pd.DataFrame, shot: int, label: Label, crowd_edit: bool):
    """Refuse a table, or a shot in it, that the page cannot edit without losing data."""
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


def _replaced(current: pd.DataFrame, shot: int, label: Label) -> pd.DataFrame:
    """`current` with the shot's rows replaced by `label`'s, sorted by shot."""
    current = current.copy()
    selected = pd.to_numeric(current.shot, errors="coerce") == int(shot)
    columns = WITH_ATTRS if label.iscrowd else INTERVAL_COLUMNS
    replacement = pd.DataFrame(label.rows(shot), columns=list(columns))
    if "attrs" in current and "attrs" not in replacement:
        replacement["attrs"] = ""
    elif "attrs" in replacement and "attrs" not in current:
        current["attrs"] = ""
    frame = pd.concat([current.loc[~selected], replacement], ignore_index=True)
    frame = frame.sort_values("shot", key=pd.to_numeric, kind="stable")
    validate_intervals(frame)
    return frame


def apply_labels(event_dir) -> saves.Apply:
    """The merge's step for `labels.csv`: the winning lines replace their shots'
    rows, one rewrite for the batch; a line the table refuses comes back."""

    def apply(winners: list[dict]) -> list:
        path = labels_path(event_dir)
        if path.is_file():
            validate_csv_fields(path)
        frame, refused = _current(path), []
        for entry in winners:
            try:
                label = Label.from_entry(entry)
                _check_editable(frame, entry["shot"], label, crowd_edit=True)
                frame = _replaced(frame, entry["shot"], label)
            except (ValueError, KeyError, TypeError) as error:
                refused.append((entry, str(error)))
        if len(refused) < len(winners):
            with saves.replacing(path) as tmp:
                frame.to_csv(tmp, index=False)
        return refused

    return apply


def diverged(
    event_dir, history: list[dict] | None = None, *, stale_only: bool = False,
    legacy: bool = False,
) -> list[dict]:
    """The last save of each shot whose label in `labels.csv` is not that save's.

    Both are read on the grid, so a table regridded after the save agrees with it.
    Only a line with a `save_id` is held to its label unless `legacy`: older lines
    were edited around by scripts. With `stale_only` only a shot the table has
    lost is returned (no label, or an earlier save's): an update another client
    overwrote, not a label somebody edited on purpose.
    """
    history = read_history(event_dir) if history is None else history
    saved = read_saved(event_dir)
    last, earlier = {}, {}
    for entry in history:
        shot = entry.get("shot")
        if shot in last:
            earlier.setdefault(shot, []).append(last[shot])
        last[shot] = entry
    found = []
    for shot, entry in last.items():
        if "save_id" not in entry and not legacy:
            continue
        try:
            want = on_grid(Label.from_entry(entry))
            have = on_grid(saved.get(int(shot)))
            if have == want:
                continue
            if stale_only and have is not None and not any(
                on_grid(Label.from_entry(old)) == have for old in earlier.get(shot, ())
            ):
                continue
        except (ValueError, KeyError, TypeError):
            continue
        found.append(entry)
    return found


def stale_labels(event_dir) -> saves.Stale:
    return lambda history: diverged(event_dir, history, stale_only=True)


def _merge_history(event_dir, save_id: str) -> dict:
    """Merge the writers' saves into `labels.csv` and `history.jsonl`, and say what
    became of `save_id`. `LABELER_AUTO_MERGE=0` leaves the merge to `labeler-sync`."""
    if getenv("LABELER_AUTO_MERGE", "1") == "0":
        return {"pending": True, "why": "LABELER_AUTO_MERGE=0"}
    try:
        merged = saves.merge(
            event_dir, "history", apply_labels(event_dir), stale_labels(event_dir)
        )
    except (saves.StoreBusy, SaveRefused, OSError, ValueError) as error:
        return {"pending": True, "why": str(error)}
    return merged.outcome(save_id)


def save(
    event_dir,
    shot: int,
    label: Label,
    *,
    source: str | None,
    name: str | None = None,
    source_sha256: str | None = None,
    crowd_edit: bool = False,
    suggestions_shown: bool | None = None,
    prefilled: bool | None = None,
    extra: dict | None = None,
    outcome: dict | None = None,
) -> dict:
    """Save one shot's label: write it to the caller's own file under `review/saves/`,
    then merge the writers' files into `labels.csv` and `history.jsonl`.

    Refuse malformed CSV records, ragged rows or duplicate headers, and any
    unsupported attrs on the selected shot (the page can edit `iscrowd`), or an
    evidence-bearing source which this page cannot edit without losing tiers. Validate
    both the existing and replacement-combined interval tables: exact columns,
    nonnegative integral int64 shot/category values, finite time bounds with
    end >= start, confidence missing or in [0, 1], and plain JSON object attrs
    without duplicate keys or nonfinite values. These checks precede writing.
    Other shots retain their original precision and attrs text. This validates
    interval schema; catalog tiling, observability, points and allowed windows
    are checked separately by the catalog checker.

    `reviewer` is the server's login; `name` is the reviewer's name, or None
    (see `versions`). The line also carries `save_id` and `writer` (the login and
    node; see `saves`). `source` names the table the save was made against and
    `source_sha256` is that table's sha256 when it was made, kept in the history
    line when given (see `agreement`: a save without one is matched to the table
    by name).

    `suggestions_shown` and `prefilled` record producer exposure and anchoring
    for each saved version. None preserves unknown provenance from old clients;
    it must never be interpreted as an independent blind review.

    `extra` is more provenance for the history line (a subcategory save names the
    parent version it was clipped to; see `parent_stamp`).

    Once the line is in the writer's file the save is durable. If the merge cannot
    run (the folder's lock stays busy, `LABELER_AUTO_MERGE=0`), the save is still
    returned and the next merge completes it; `outcome`, when given, is filled with
    `pending` (and `why`) and `superseded` (a later save of the shot already stands).
    """
    with _write_lock:
        try:
            source_label = read_source(event_dir).get(int(shot))
        except (ValueError, OSError):  # a broken pointer table has nothing to protect
            source_label = None
        if label.attrs or (source_label is not None and source_label.attrs):
            raise SaveRefused(
                f"shot {shot} has evidence attributes that the review page cannot "
                "edit; save refused"
            )
        path = labels_path(event_dir)
        if path.is_file():
            try:
                validate_csv_fields(path)
            except pd.errors.ParserError as error:
                raise SaveRefused(str(error)) from error
        current = _current(path)
        _check_editable(current, shot, label, crowd_edit)
        _replaced(current, shot, label)  # the table this save makes must be valid
        reviewer = getpass.getuser()
        saved_at, save_id = saves.new_save()
        entry = {
            "shot": int(shot),
            "reviewer": reviewer,
            "name": name,
            "saved_at": saved_at,
            **label.as_json(),
            "source": source,
            "suggestions_shown": suggestions_shown,
            "prefilled": prefilled,
        }
        if source_sha256 is not None:
            entry["source_sha256"] = source_sha256
        if extra:
            entry.update(extra)
        entry.update(save_id=save_id, writer=saves.writer_id(reviewer))
        saves.append(event_dir, "history", entry)
        result = _merge_history(event_dir, save_id)
    if outcome is not None:
        outcome.update(result)
    return entry


def state(saved: Label | None, source: Label | None) -> str:
    if saved is None:
        return "unreviewed"
    if source is None:
        return "confirmed" if not saved.intervals else "changed"
    return "confirmed" if saved == source else "changed"


def queue(event_dir, roster: pd.DataFrame, *, event: str | None = None) -> dict:
    """The roster in order, each shot's state and last save, and where to resume."""
    event = event or Path(event_dir).name
    saved, source = read_saved(event_dir), read_source(event_dir)
    history = read_history(event_dir)
    saved_at = {entry["shot"]: entry["saved_at"] for entry in history}
    shots = [
        {
            "shot": int(row.shot),
            "tier": row.tier,
            "state": state(
                page_label(event, saved.get(int(row.shot))),
                page_label(event, source.get(int(row.shot))),
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


def shot_labels(
    event_dir, shot: int, *, suppress_source=False, event: str | None = None,
    region: Region | None = None,
) -> dict:
    """A shot's source and saved labels, as the page opens them.

    `event` names the categories on offer when the directory's name is not the
    event's id (a subcategory's is). With `region`, both labels are clipped to it
    and given its window: whatever the parent's label no longer marks is gone.
    """
    shot = int(shot)
    event = event or Path(event_dir).name
    saved = page_label(event, read_saved(event_dir).get(shot))
    source = None if suppress_source else page_label(
        event, read_source(event_dir).get(shot)
    )
    if region is not None:
        saved, source = region.clip(saved), region.clip(source)
    history = read_history(event_dir)
    last = next((entry for entry in reversed(history) if entry["shot"] == shot), None)
    return {
        "source": None if source is None else source.as_json(),
        "saved": None if saved is None else saved.as_json(),
        "state": state(saved, source),
        "last_save": last,
    }


# ---- subcategories: a label inside its parent's label ----


@dataclass(frozen=True)
class Region:
    """Where a subcategory may be labelled on one shot.

    `window` is the parent's saved window and `spans` the merged union of its
    saved spans with a category on offer (present or uncertain, both lanes);
    the parent's absent gaps and anything it does not offer are outside it.
    """

    window: tuple[int, int]
    spans: tuple[tuple[int, int], ...]

    def clip(self, label: Label | None) -> Label | None:
        """`label` cut to the region, on the region's window; a span that crosses a
        gap in the region comes back as the pieces inside it, with its lane."""
        if label is None:
            return None
        flags = label.iscrowd or (None,) * len(label.intervals)
        pieces, kept = [], []
        for (a, b, c), flag in zip(label.intervals, flags, strict=True):
            for lo, hi in self.spans:
                start, stop = max(a, lo), min(b, hi)
                if stop > start:
                    pieces.append((start, stop, c))
                    kept.append(flag)
        return normalise(
            self.window, pieces,
            iscrowd=kept if any(flag is not None for flag in kept) else None,
        )

    def as_json(self) -> list[list[int]]:
        return [list(span) for span in self.spans]


def region_of(parent_event: str, parent: Label | None) -> Region | None:
    """The region a parent's saved label opens to its subcategories; None while the
    parent is unsaved on the shot (an all-absent saved label is a region of nothing)."""
    if parent is None:
        return None
    shown = categories(parent_event)
    spans = sorted((a, b) for a, b, c in parent.intervals if c in shown)
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return Region(parent.window, tuple((a, b) for a, b in merged))


def parent_stamp(parent_event: str, parent_dir, shot: int, parent: Label) -> dict:
    """What a subcategory save records of the parent version it was clipped to: the
    parent's event, when its label for the shot was last saved and a digest of it."""
    last = next(
        (e for e in reversed(read_history(parent_dir)) if e.get("shot") == int(shot)),
        None,
    )
    digest = hashlib.sha256(
        json.dumps(parent.as_json(), sort_keys=True).encode()
    ).hexdigest()
    return {
        "parent": parent_event,
        "parent_saved_at": None if last is None else last.get("saved_at"),
        "parent_sha256": digest,
    }


def child_queue(
    event: str, event_dir, parent_event: str, parent_dir, roster: pd.DataFrame
) -> dict:
    """The subcategory's queue: the parent roster's shots whose saved parent label
    marks something, each with the subcategory's own state and last save."""
    parents = read_saved(parent_dir)
    saved, source = read_saved(event_dir), read_source(event_dir)
    history = read_history(event_dir)
    saved_at = {entry["shot"]: entry["saved_at"] for entry in history}
    shots = []
    for row in roster.itertuples():
        shot = int(row.shot)
        region = region_of(parent_event, page_label(parent_event, parents.get(shot)))
        if region is None or not region.spans:
            continue
        shots.append({
            "shot": shot,
            "tier": row.tier,
            "state": state(
                region.clip(page_label(event, saved.get(shot))),
                region.clip(page_label(event, source.get(shot))),
            ),
            "saved_at": saved_at.get(shot),
        })
    return {"shots": shots, "resume": resume(shots, history)}
