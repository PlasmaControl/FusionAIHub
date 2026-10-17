#!/usr/bin/env python
"""Jalal Butt's WPQH phases as verified confinement labels, and his saved name.

Three steps, each a dry run until `--apply`:

    rename   every saved reviewer name exactly "Jalal" becomes "Jalal Butt": the
             review logs of every event (`review/history.jsonl`, `masks.jsonl`, a
             subcategory's too) and the list `reviewers.txt` behind `/api/names`.
    list     the shot list `confinement_jalal_wpqh`: every shot of the archive, so
             each opens on the review page (written as `shotlists make` writes it).
    import   the archive's phases as confinement labels, saved as the page saves one.

The archive is `WPQHphasesMay262026.tar` (Jalal Butt, 2026-05-26), unpacked read-only
under `$LABELER_ROOT/raw_external/jalal_wpqh_2026-05-26/`. Per shot, `<shot>-wpqh1.dat`
lists the WPQH phases, `ON OFF` in ms (the shot has WPQH); `<shot>-wpqh0.dat` is a shot
with none (its lines, when it has any, are blocks too short to count: not WPQH). The
`.pkl` files are the detector's diagnostics (inputs, conditions, `auto_spans`: the same
phases), not a regime timeline, so only class 4 comes from the archive. Both files for
one shot (four shots): `wpqh1` is the later run and wins. `postprocessed_2/` (the same
phases with close blocks merged) is not read.

A shot's label starts from what the page shows: its saved label, else its draft. Its
class-4 (WPQH) spans are replaced by the archive's; every other class stays. It is
saved under the name "Jalal Butt" with `labels.save`, the call the page's
`POST /api/label` ends in, and its history line also carries `verified: true` and
`note: "WPQHphasesMay262026"` (`extra`). The archive says nothing about other classes,
so a wpqh0 shot with no label to start from is not saved: an empty saved label would
assert that no regime at all was present (`--save-empty` saves it anyway). A shot whose
history has a save under any other name is a conflict and is left as it is.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.config import Paths, atomic_path, sha256_of
from labeler.events import grid, rosters, shotlists
from labeler.events.review import labels, reviewers

EVENT = "confinement"
WPQH = 4
OLD_NAME, NEW_NAME = "Jalal", "Jalal Butt"
SOURCE_NOTE = "WPQHphasesMay262026"
LIST_NAME = "confinement_jalal_wpqh"
#: The detector looked at 1000-5000 ms of every shot (`t0_ms`, `tf_ms` in the pkl inputs).
ANALYSIS_WINDOW = (1000, 5000)
ARCHIVE = Path(
    "/scratch/gpfs/EKOLEMEN/nc1514/labeler/raw_external/jalal_wpqh_2026-05-26"
)
BACKUP_ROOT = Path("/scratch/gpfs/EKOLEMEN/nc1514/labeler/backups/jalal_2026-10-17")
_DAT = re.compile(r"(\d+)-wpqh([01])\.dat\Z")


# ---- the saved name ----


def _logs(tables: Path) -> list[Path]:
    """The review logs that carry a `name` (as `reviewers.saved_names` finds them)."""
    found = []
    for log in reviewers.LOGS:
        found += [
            *tables.glob(f"*/{labels.REVIEW}/{log}"),
            *tables.glob(f"*/subcategories/*/{labels.REVIEW}/{log}"),
        ]
    return sorted(found)


def rename_saved_names(
    tables, *, apply: bool, old: str = OLD_NAME, new: str = NEW_NAME
) -> dict:
    """Rename every saved reviewer name exactly `old` to `new`; with `apply` false,
    only count. Returns `{log or list path (relative to the tables): lines changed}`.
    A line that is not `old` is kept byte for byte; a changed one is written again as
    JSON. The names list keeps its other lines and drops a duplicate of `new`."""
    tables = Path(tables)
    changed: dict[str, int] = {}
    for path in _logs(tables):
        out, count = [], 0
        for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                entry = None
            if isinstance(entry, dict) and entry.get("name") == old:
                tail = line[len(line.rstrip("\r\n")) :]
                line = json.dumps({**entry, "name": new}) + tail
                count += 1
            out.append(line)
        if count:
            changed[path.relative_to(tables).as_posix()] = count
            if apply:
                with atomic_path(path) as tmp:
                    tmp.write_text("".join(out), encoding="utf-8")
    names = reviewers.names_path(tables)
    if names.is_file():
        text = names.read_text(encoding="utf-8-sig")
        lines, count = [], 0
        for line in text.splitlines(keepends=True):
            if line.strip() == old:
                line = line.replace(old, new)
                count += 1
            lines.append(line)
        if count:
            seen, kept = set(), []
            for (
                line
            ) in lines:  # a name already listed twice (case aside) is listed once
                key = line.strip().casefold()
                if key and key in seen:
                    continue
                seen.add(key)
                kept.append(line)
            changed[reviewers.FILE] = count
            if apply:
                with atomic_path(names) as tmp:
                    tmp.write_text("".join(kept), encoding="utf-8")
    return changed


def roster_mentions(tables, name: str = OLD_NAME) -> dict[str, int]:
    """Roster rows whose `reviewers` cell names `name`: read only, never changed
    (a roster is not a saved label)."""
    found = {}
    for path in sorted(Path(tables).glob("*/" + rosters.ROSTER_NAME)):
        cells = _roster_cells(path)
        count = sum(name in rosters.split_reviewers(cell) for cell in cells)
        if count:
            found[path.relative_to(tables).as_posix()] = count
    return found


def _roster_cells(path) -> list:
    """A roster's `reviewers` cells; none for a roster that does not read."""
    try:
        return list(rosters.read_roster(path).reviewers)
    except (OSError, ValueError, KeyError):
        return []


# ---- the archive ----


@dataclass(frozen=True)
class Phases:
    """One shot of the archive: `flag` 1 when it has WPQH, `spans` in ms as its file
    lists them (a flag-0 file's are blocks too short to count, not WPQH)."""

    shot: int
    flag: int
    spans: tuple[tuple[float, float], ...]
    both: bool = False
    file: str = ""

    @property
    def wpqh(self) -> tuple[tuple[float, float], ...]:
        return self.spans if self.flag else ()


def read_dat(path) -> tuple[tuple[float, float], ...]:
    """`ON OFF` rows of one file. The header is skipped; a `0.0 0.0` placeholder row
    (an older table's no-WPQH shot) is no phase."""
    lines = Path(path).read_text().splitlines()
    spans = []
    for number, line in enumerate(lines[1:], 2):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 2:
            raise ValueError(f"{path}:{number}: expected `ON OFF`, got {line!r}")
        on, off = float(parts[0]), float(parts[1])
        if off < on:
            raise ValueError(f"{path}:{number}: phase {on}-{off} ms runs backwards")
        if off > on:
            spans.append((on, off))
    return tuple(spans)


def read_archive(directory=ARCHIVE) -> dict[int, Phases]:
    """Every shot with a `<shot>-wpqh0.dat` or `<shot>-wpqh1.dat` directly under
    `directory`; `wpqh1` wins when a shot has both."""
    found: dict[int, dict[int, Path]] = {}
    for path in sorted(Path(directory).iterdir()):
        match = _DAT.match(path.name)
        if match:
            found.setdefault(int(match[1]), {})[int(match[2])] = path
    result = {}
    for shot, files in sorted(found.items()):
        flag = 1 if 1 in files else 0
        result[shot] = Phases(
            shot, flag, read_dat(files[flag]), len(files) == 2, files[flag].name
        )
    return result


# ---- the label ----


def spans_ms(label: labels.Label | None, category: int = WPQH) -> int:
    return (
        0
        if label is None
        else sum(b - a for a, b, c in label.intervals if c == category)
    )


def with_wpqh(
    base: labels.Label | None, spans, *, window=ANALYSIS_WINDOW, known=None
) -> labels.Label | None:
    """`base` with its class-4 spans replaced by `spans` (ms, Spans), every other
    annotation kept as it is; with no base the label is the spans over `window`. None
    when there is nothing to say: no base and no spans."""
    if base is None and not spans:
        return None
    if base is None:
        kept, flags, lo_hi = [], [], window
    else:
        base = labels.span_default(EVENT, base)
        flags = base.iscrowd or (None,) * len(base.intervals)
        pairs = [
            (span, flag)
            for span, flag in zip(base.intervals, flags, strict=True)
            if span[2] != WPQH
        ]
        kept, flags = [span for span, _ in pairs], [flag for _, flag in pairs]
        lo_hi = base.window
    new = [(*grid.snap_span(a, b), WPQH) for a, b in spans]
    return labels.normalise(
        grid.snap_window(*lo_hi),
        [*kept, *new],
        known=known,
        iscrowd=[*flags, *([1] * len(new))] if (flags or new) else None,
    )


def label_of(body: dict | None) -> labels.Label | None:
    """A `Label` from the JSON the page serves (`Label.as_json`)."""
    if body is None:
        return None
    return labels.Label(
        tuple(body["window"]),
        tuple(tuple(span) for span in body["intervals"]),
        tuple(body.get("iscrowd", ())),
        tuple(json.dumps(a, sort_keys=True) for a in body["attrs"])
        if body.get("attrs")
        else (),
    )


def spans_text(spans) -> str:
    return ";".join(f"{a:g}-{b:g}" for a, b in spans)


def class4(label: labels.Label | None) -> tuple[tuple[int, int], ...]:
    return (
        ()
        if label is None
        else tuple((a, b) for a, b, c in label.intervals if c == WPQH)
    )


# ---- the plan and the save ----


@dataclass
class Row:
    """What one shot of the archive came to."""

    shot: int
    flag: int
    n_phases: int
    wpqh_ms: float
    base: str = "none"
    action: str = ""
    reason: str = ""
    before_ms: int = 0
    after_ms: int = 0
    jalal: str = ""
    existing: str = ""
    saved_at: str = ""
    extra: dict = field(default_factory=dict)


COLUMNS = (
    "shot",
    "flag",
    "n_phases",
    "wpqh_ms",
    "base",
    "action",
    "reason",
    "before_ms",
    "after_ms",
    "jalal",
    "existing",
    "saved_at",
)


def save_names(history: list[dict]) -> dict[int, list]:
    """Each shot's saves, oldest first, as the `name` each carried (None for none)."""
    found: dict[int, list] = {}
    for entry in history:
        found.setdefault(int(entry["shot"]), []).append(entry.get("name"))
    return found


def import_shot(
    phases: Phases,
    reviewed: dict,
    names: list,
    directory,
    *,
    reviewable: bool = True,
    table: Path | None = None,
    origin: str | None = None,
    known=None,
    save_empty: bool = False,
    last_save: dict | None = None,
    apply: bool,
) -> Row:
    """Plan, and with `apply` save, one shot. `reviewed` is the page's
    `{saved, source}` for the shot (`ui.app.shot_labels`), `names` the `name` of each
    of its saves, `table` and `origin` what the page would record as the save's source."""
    row = Row(
        phases.shot,
        phases.flag,
        len(phases.spans),
        sum(b - a for a, b in phases.wpqh),
        jalal=spans_text(phases.wpqh),
    )
    saved, source = label_of(reviewed.get("saved")), label_of(reviewed.get("source"))
    if not reviewable:
        row.action, row.reason = (
            "not_reviewable",
            "on no shot list, or a blind test shot",
        )
        return row
    others = sorted({str(n) for n in names if n not in (OLD_NAME, NEW_NAME)})
    base = saved if saved is not None else source
    row.base = (
        "saved"
        if saved is not None
        else f"draft:{origin or 'source'}"
        if source is not None
        else "none"
    )
    row.before_ms = spans_ms(base)
    if others:
        row.action = "conflict"
        row.reason = f"saved by {', '.join(others)}"
        row.existing = spans_text(class4(saved))
        return row
    if base is not None and base.attrs:
        row.action, row.reason = "refused", "evidence attributes the page cannot edit"
        return row
    try:
        label = with_wpqh(base, phases.wpqh, known=known)
    except ValueError as error:  # e.g. a window the page would not save
        row.action, row.reason = "refused", str(error)
        return row
    if label is None and not save_empty:
        row.action = "skip_no_label"
        row.reason = (
            "wpqh0 and no label to start from: an empty save would assert no regime"
        )
        return row
    if label is None:
        label = labels.normalise(grid.snap_window(*ANALYSIS_WINDOW), [], known=known)
    row.after_ms = spans_ms(label)
    if (
        saved == label
        and last_save
        and last_save.get("note") == SOURCE_NOTE
        and last_save.get("name") == NEW_NAME
    ):
        row.action, row.reason = "already", "this label is already the last save"
        return row
    row.action = "save"
    if apply:
        try:
            entry = labels.save(
                directory,
                phases.shot,
                label,
                source=origin or (table.name if table else None),
                name=NEW_NAME,
                source_sha256=sha256_of(table) if table else None,
                crowd_edit=True,
                suggestions_shown=None,
                prefilled=None,
                extra={
                    "verified": True,
                    "note": SOURCE_NOTE,
                    "wpqh_flag": phases.flag,
                    "base": row.base,
                },
            )
        except labels.SaveRefused as error:
            row.action, row.reason = "refused", str(error)
            return row
        row.saved_at = entry["saved_at"]
    return row


def write_table(rows: list[Row], path) -> None:
    with open(path, "w", newline="") as stream:
        out = csv.writer(stream)
        out.writerow(COLUMNS)
        for row in rows:
            out.writerow([getattr(row, name) for name in COLUMNS])


# ---- the list ----


def make_list(archive: dict[int, Phases], *, apply: bool, root=None) -> Path:
    """`confinement_jalal_wpqh`: every archive shot, as `shotlists make` writes a list
    (the blind test shots are left out and counted)."""
    paths = Paths.from_env()
    root = Path(paths.root if root is None else root)
    source = root / "raw_external" / "jalal_wpqh_2026-05-26_shots.txt"
    text = "".join(f"{shot}\n" for shot in sorted(archive))
    if apply:
        source.parent.mkdir(parents=True, exist_ok=True)
        with atomic_path(source) as tmp:
            tmp.write_text(text)
        code = shotlists.main(
            [
                "make",
                LIST_NAME,
                "--from-txt",
                str(source),
                "--kind",
                "dataset",
                "--title",
                "Jalal Butt's WPQH phases (26 May 2026)",
                "--drop-blind",
                "--provenance",
                (
                    "WPQHphasesMay262026.tar (Jalal Butt, /projects/EKOLEMEN/jalal/), "
                    "unpacked to $LABELER_ROOT/raw_external/jalal_wpqh_2026-05-26: every "
                    "shot with a -wpqh0.dat or -wpqh1.dat file"
                ),
                "--made-by",
                "nc1514 (scripts/labeler/jalal_wpqh_import.py list)",
            ]
        )
        if code:
            raise SystemExit(code)
    return source


def backup(paths: Paths, files) -> list[Path]:
    """Copy each existing file under the label tables into `BACKUP_ROOT`, keeping
    the first copy of a file that is already there."""
    kept = []
    for path in map(Path, files):
        if not path.is_file():
            continue
        target = BACKUP_ROOT / "events" / path.relative_to(paths.label_tables)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        kept.append(target)
    return kept


# ---- command line ----


def _import(args, paths: Paths) -> int:
    from fastapi import HTTPException

    from labeler.events.ui import app  # the page's own reading of a shot

    archive = read_archive(args.archive)
    shots = args.shots or sorted(archive)
    directory = app.require_event(EVENT, paths)
    if args.apply:
        backup(paths, [labels.labels_path(directory), labels.history_path(directory)])
    history = labels.read_history(directory)
    names = save_names(history)
    last = {int(e["shot"]): e for e in history}
    known = set(labels.categories(EVENT))
    rows = []
    for k, shot in enumerate(shots, 1):
        reviewable = True
        try:
            app.shot_tier(EVENT, shot, paths)
        except HTTPException:
            reviewable = False
        reviewed, table, origin = ({}, None, None)
        if reviewable:
            reviewed, table, origin = app.shot_labels(EVENT, shot, paths)
        row = import_shot(
            archive[shot],
            reviewed,
            names.get(shot, []),
            directory,
            reviewable=reviewable,
            table=table,
            origin=origin,
            known=known,
            save_empty=args.save_empty,
            last_save=last.get(shot),
            apply=args.apply,
        )
        rows.append(row)
        if k % 100 == 0:
            print(f"{k}/{len(shots)}", flush=True)
    if args.table:
        write_table(rows, args.table)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.action] = counts.get(row.action, 0) + 1
    print(
        ("applied: " if args.apply else "dry run: ")
        + json.dumps(counts, sort_keys=True)
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("rename", "list", "import"):
        sub = commands.add_parser(name)
        sub.add_argument(
            "--apply", action="store_true", help="write (default: dry run)"
        )
        if name != "rename":
            sub.add_argument("--archive", type=Path, default=ARCHIVE)
    imported = commands.choices["import"]
    imported.add_argument("--shots", nargs="+", type=int, help="only these shots")
    imported.add_argument(
        "--table", type=Path, help="write one row per shot here (CSV)"
    )
    imported.add_argument(
        "--save-empty",
        action="store_true",
        help="also save a wpqh0 shot that has no label to start from",
    )
    args = parser.parse_args(argv)
    paths = Paths.from_env()
    if args.command == "rename":
        if args.apply:
            backup(
                paths,
                [*_logs(paths.label_tables), reviewers.names_path(paths.label_tables)],
            )
        changed = rename_saved_names(paths.label_tables, apply=args.apply)
        print(
            ("renamed: " if args.apply else "would rename: ")
            + json.dumps(changed, indent=1)
        )
        print(
            "rosters naming Jalal (not changed): "
            + json.dumps(roster_mentions(paths.label_tables))
        )
    elif args.command == "list":
        archive = read_archive(args.archive)
        if args.apply:
            backup(
                paths,
                [
                    paths.shotlists / f"{LIST_NAME}.csv",
                    paths.shotlists / f"{LIST_NAME}.meta.json",
                ],
            )
        source = make_list(archive, apply=args.apply)
        print(f"{len(archive)} archive shots -> {LIST_NAME} (from {source})")
    else:
        return _import(args, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
