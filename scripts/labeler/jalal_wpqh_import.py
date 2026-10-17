#!/usr/bin/env python
"""Jalal Butt's WPQH phases as verified confinement labels, and his saved name.

Three steps, each a dry run until `--apply`:

    rename   every saved reviewer name exactly "Jalal" becomes "Jalal Butt": the
             review logs of every event (`review/history.jsonl`, `masks.jsonl`, a
             subcategory's too) and the list `reviewers.txt` behind `/api/names`.
    list     the shot list `confine_butt`: every shot of the archive and of his
             2024 regime table, so each opens on the review page, and `confine_butt_gill`
             over it (written as `shotlists make` writes them).
    import   the archive's phases as confinement labels, saved as the page saves one.

The archive is `WPQHphasesMay262026.tar` (Jalal Butt, 2026-05-26), unpacked read-only
under `$LABELER_ROOT/raw_external/jalal_wpqh_2026-05-26/`. His final version is
`postprocessed_2/<shot>-wpqh1-pp.dat` (the shot has WPQH, one `ON OFF` phase per line,
in ms) or `<shot>-wpqh0-pp.dat` (it has none; its lines, when it has any, are blocks
too short to count: not WPQH). The detector's own output (`<shot>-wpqh[01].dat`
directly under the archive, close blocks not yet merged) fills a shot that
`postprocessed_2` lacks (one shot, 184971) and is read alone with `final=False`. The
`.pkl` files are the detector's diagnostics (inputs, conditions, `auto_spans`: the raw
phases), not a regime timeline, so only class 4 comes from the archive.

A shot's label starts from a human label only: its saved label, else its draft when
the draft is a person's (the event's source table for a shot it draws from a curated
table, or a format table). A model's suggestion is never a start: such a shot starts
empty, with only his WPQH spans over 1000-5000 ms, grown to hold them. A saved label
made by this import is no start either (it is derived): the shot starts again from
its human draft, so a rerun replaces it. The start's class-4 (WPQH) spans are replaced
by the archive's; every other class stays. It is saved under the name "Jalal Butt" with
`labels.save`, the call the page's `POST /api/label` ends in, and its history line also
carries `verified: true` and `note: "WPQHphasesMay262026"` (`extra`). The archive says
nothing about other classes, so a wpqh0 shot with no human start is not saved (an empty
saved label would assert that no regime at all was present; `--save-empty` saves it
anyway), and an earlier save of this import for such a shot is withdrawn. A shot whose
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

import pandas as pd

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
LIST_NAME = "confine_butt"
#: The union over `LIST_NAME` and Kevin Gill's list; it is made again with it.
UNION_NAME = "confine_butt_gill"
#: Butt's 2024 regime table, the other source of `LIST_NAME` (under the label tables).
REGIME_TABLE = "high_confinement_mode/raw/Jalal_28042024_confinement_regime_shotlist.csv"
#: The detector looked at 1000-5000 ms of every shot (`t0_ms`, `tf_ms` in the pkl inputs).
ANALYSIS_WINDOW = (1000, 5000)
ARCHIVE = Path(
    "/scratch/gpfs/EKOLEMEN/nc1514/labeler/raw_external/jalal_wpqh_2026-05-26"
)
BACKUP_ROOT = Path("/scratch/gpfs/EKOLEMEN/nc1514/labeler/backups/jalal_2026-10-17")
FINAL = "postprocessed_2"
_DAT = re.compile(r"(\d+)-wpqh([01])(?:-pp)?\.dat\Z")


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


def _scan(folder: Path) -> dict[int, dict[int, Path]]:
    """`{shot: {flag: file}}` of the `<shot>-wpqh[01][-pp].dat` files in `folder`."""
    found: dict[int, dict[int, Path]] = {}
    if folder.is_dir():
        for path in sorted(folder.iterdir()):
            match = _DAT.match(path.name)
            if match:
                found.setdefault(int(match[1]), {})[int(match[2])] = path
    return found


def read_archive(directory=ARCHIVE, *, final: bool = True) -> dict[int, Phases]:
    """Every shot of the archive. With `final` (the default) a shot's phases are its
    `postprocessed_2/<shot>-wpqh[01]-pp.dat`, and a shot that folder lacks keeps its
    detector file; with `final=False` only the detector's `<shot>-wpqh[01].dat` files
    directly under `directory` are read. `wpqh1` wins when a shot has both files."""
    directory = Path(directory)
    raw = _scan(directory)
    merged = {**raw, **_scan(directory / FINAL)} if final else raw
    result = {}
    for shot, files in sorted(merged.items()):
        flag = 1 if 1 in files else 0
        result[shot] = Phases(
            shot,
            flag,
            read_dat(files[flag]),
            len(files) == 2,
            files[flag].relative_to(directory).as_posix(),
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
    dropped: str = ""
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
    "dropped",
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


def _own(entry: dict | None) -> bool:
    """True for a save made by this import (it carries `SOURCE_NOTE`)."""
    return (
        entry is not None
        and entry.get("note") == SOURCE_NOTE
        and entry.get("name") == NEW_NAME
    )


def curated_shots(directory) -> set[int] | None:
    """The shots whose rows in the event's source table come from a person's table.

    The page's source table is a machine suggestion that carries a person's curated
    intervals where there are any (its sidecar names that table and its sha256: a
    mismatch raises). None when the source is no suggestion table (no pointer): a
    format table, every row a person's."""
    pointer = labels.pointer_path(directory)
    if not pointer.is_file():
        return None
    table = Path(json.loads(pointer.read_text())["table"])
    sidecar = table.with_suffix(".meta.json")
    meta = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    curated = (meta.get("sources") or {}).get("curated")
    if not curated:
        return set()
    path = Path(curated["table"])
    if sha256_of(path) != curated["sha256"]:
        raise ValueError(f"{path} is not the table {sidecar} was made from")
    with path.open(newline="") as stream:
        return {int(row["shot"]) for row in csv.DictReader(stream)}


def human_draft(origin: str | None, shot: int, curated: set[int] | None) -> bool:
    """True when the page's draft for `shot` is a person's label. `origin` is what
    `ui.app.shot_labels` returns: None for the event's source table, else the silver
    layer's name (`format:...` is a person's table; `suggest:`, `extend:` ... are not)."""
    if origin is None:
        return curated is None or shot in curated
    return origin.startswith("format:")


def withdraw(directory, shot: int) -> None:
    """Take this import's saves of `shot` back: its rows leave `labels.csv` and its
    lines leave `history.jsonl`, as if it had not been saved. Refuses a shot with any
    other save in its history."""
    shot = int(shot)
    with labels._write_lock:  # the page's lock: its saves rewrite the same files
        history = labels.history_path(directory)
        lines = history.read_text().splitlines(keepends=True) if history.is_file() else []
        kept, own = [], 0
        for line in lines:
            entry = json.loads(line) if line.strip() else {}
            if entry.get("shot") != shot:
                kept.append(line)
            elif _own(entry):
                own += 1
            else:
                raise ValueError(f"shot {shot} has a save that is not this import's")
        path = labels.labels_path(directory)
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, index_col=False)
        rows = pd.to_numeric(frame.shot, errors="coerce") == shot
        with atomic_path(path) as tmp:
            frame.loc[~rows].to_csv(tmp, index=False)
        with atomic_path(history) as tmp:
            tmp.write_text("".join(kept))


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
    human: bool = True,
    apply: bool,
) -> Row:
    """Plan, and with `apply` save, one shot. `reviewed` is the page's
    `{saved, source}` for the shot (`ui.app.shot_labels`), `names` the `name` of each
    of its saves, `table` and `origin` what the page would record as the save's source.
    `human` says whether the draft `source` is a person's (`human_draft`); a model's
    draft, and a saved label this import made (`last_save`), are no start."""
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
    own = _own(last_save)
    start = None if own else saved
    draft = source if human else None
    base = start if start is not None else draft
    row.base = (
        "saved"
        if start is not None
        else f"draft:{origin or 'source'}"
        if draft is not None
        else "none"
    )
    if source is not None and not human and start is None:
        row.dropped = origin or "source"
    row.before_ms = spans_ms(saved if saved is not None else source)
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
        if own:
            row.action = "withdraw"
            row.reason = (
                "wpqh0 and no human start: this import's earlier save is taken back"
            )
            if apply:
                try:
                    withdraw(directory, phases.shot)
                except ValueError as error:
                    row.action, row.reason = "refused", str(error)
            return row
        row.action = "skip_no_label"
        row.reason = (
            "wpqh0 and no human label to start from: an empty save would assert "
            "no regime"
        )
        return row
    if label is None:
        label = labels.normalise(grid.snap_window(*ANALYSIS_WINDOW), [], known=known)
    row.after_ms = spans_ms(label)
    if saved == label and own:
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
                    "phases_file": phases.file,
                    "base": row.base,
                    **({"dropped_draft": row.dropped} if row.dropped else {}),
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
    """`confine_butt`: every archive shot and every shot of his 2024 regime table, then
    `confine_butt_gill` over it, as `shotlists make` writes lists (the blind test shots
    are left out and counted). The archive's shots are kept in `raw_external/` as the
    list's input."""
    paths = Paths.from_env()
    root = Path(paths.root if root is None else root)
    source = root / "raw_external" / "jalal_wpqh_2026-05-26_shots.txt"
    text = "".join(f"{shot}\n" for shot in sorted(archive))
    if apply:
        source.parent.mkdir(parents=True, exist_ok=True)
        with atomic_path(source) as tmp:
            tmp.write_text(text)
        for command in (
            [
                "make",
                LIST_NAME,
                "--union",
                f"table({Path(paths.label_tables) / REGIME_TABLE}),txt({source})",
                "--drop-blind",
                "--provenance",
                (
                    "Jalal Butt's confinement regime table of 28 April 2024 "
                    "(events.yaml jalal_hmode_confinement_2024) and WPQHphasesMay262026"
                    ".tar (/projects/EKOLEMEN/jalal/, unpacked to $LABELER_ROOT/"
                    "raw_external/jalal_wpqh_2026-05-26: every shot with a -wpqh0.dat "
                    "or -wpqh1.dat file)"
                ),
                "--made-by",
                "nc1514 (scripts/labeler/jalal_wpqh_import.py list)",
            ],
            ["make", UNION_NAME, "--union", f"{LIST_NAME},confine_gill",
             "--made-by", "nc1514 (scripts/labeler/jalal_wpqh_import.py list)"],
        ):
            code = shotlists.main(command)
            if code:
                break
        if code:
            raise SystemExit(code)
    return source


def backup(paths: Paths, files, root: Path = BACKUP_ROOT) -> list[Path]:
    """Copy each existing file under the label tables into `root`, keeping the first
    copy of a file that is already there."""
    kept = []
    for path in map(Path, files):
        if not path.is_file():
            continue
        target = Path(root) / "events" / path.relative_to(paths.label_tables)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        kept.append(target)
    return kept


# ---- command line ----


def _import(args, paths: Paths) -> int:
    from fastapi import HTTPException

    from labeler.events.ui import app  # the page's own reading of a shot

    archive = read_archive(args.archive, final=not args.raw)
    shots = args.shots or sorted(archive)
    directory = app.require_event(EVENT, paths)
    if args.apply:
        backup(
            paths,
            [labels.labels_path(directory), labels.history_path(directory)],
            args.backup_root,
        )
    curated = curated_shots(directory)
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
            human=human_draft(origin, shot, curated),
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
        "--backup-root",
        type=Path,
        default=BACKUP_ROOT,
        help="where --apply first copies labels.csv and history.jsonl",
    )
    imported.add_argument(
        "--raw",
        action="store_true",
        help="read the detector's own .dat files, not postprocessed_2",
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
                    paths.shotlists / f"{name}{suffix}"
                    for name in (LIST_NAME, UNION_NAME)
                    for suffix in (".csv", ".meta.json")
                ],
            )
        source = make_list(archive, apply=args.apply)
        print(f"{len(archive)} archive shots -> {LIST_NAME}, {UNION_NAME} (from {source})")
    else:
        return _import(args, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
