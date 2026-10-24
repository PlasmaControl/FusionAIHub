"""`labeler-sync`: the merge, import and checks of a shared label folder.

Every save already merges itself (see `saves`); this is for the owner and for
whoever looks after the folder. Run as `python -m labeler.events.review.sync
COMMAND` or `pixi run -e labeler labeler-sync COMMAND`; every command takes
`--events DIR` (default: the label tables of this environment) and `--event REL`
(repeatable: only that event directory, e.g. `edge_localized_mode` or
`alfven_eigenmode/subcategories/tae`).

    status       what each event holds: lines, the writers' files, what is unmerged
    merge        fold the writers' files into labels.csv, history.jsonl, masks.jsonl
    import-tree  fold another label folder's review logs into this one
    init         create `saves/` and set the shared directory modes (the owner, once)
    check        read-only invariants; exit 1 when one is broken
    repair       re-apply the last save of the shots `check` reports

`status` and `check` write nothing. `merge`, `import-tree` and `repair` take the
event's lock; `import-tree` can be run again at any time, it adds only what is new.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
from pathlib import Path

from ...config import Paths
from . import labels, saves


def event_dirs(events, *, missing: bool = False) -> list[Path]:
    """Directories of the folder with a `review/`: events and their subcategories.
    With `missing`, those that could have one (a roster or a subcategory) too."""
    root = Path(events)
    found = {p.parent for pat in ("*/review", "*/subcategories/*/review")
             for p in root.glob(pat) if p.is_dir()}
    if missing:
        found |= {p.parent for p in root.glob("*/shots.csv")}
        found |= {p for p in root.glob("*/subcategories/*") if p.is_dir()}
    return sorted(found)


def select(events, rels, *, missing: bool = False) -> list[Path]:
    """The event directories asked for by `--event`, or all."""
    found = event_dirs(events, missing=missing)
    if not rels:
        return found
    wanted = {Path(rel) for rel in rels}
    unknown = wanted - {p.relative_to(events) for p in found}
    if unknown:
        raise SystemExit(f"no such event directory: {', '.join(map(str, sorted(unknown)))}")
    return [p for p in found if p.relative_to(events) in wanted]


def writer_files(event_dir, kind: str) -> dict[str, Path]:
    saves_dir = Path(event_dir) / saves.REVIEW / saves.SAVES
    return {
        p.name[len(kind) + 1:-len(".jsonl")]: p
        for p in sorted(saves_dir.glob(f"{kind}.*.jsonl"))
    }


def steps(event_dir, kind: str):
    """The merge's callbacks: the label table follows the history, nothing else does."""
    if kind != "history":
        return None, None
    return labels.apply_labels(event_dir), labels.stale_labels(event_dir)


def _rel(events, event_dir) -> str:
    return str(Path(event_dir).relative_to(events))


# ---- status ----


def status(events, rels) -> int:
    print(f"label folder {events}")
    total = 0
    for event_dir in select(events, rels):
        review = event_dir / saves.REVIEW
        out = []
        for kind in saves.KINDS:
            files = writer_files(event_dir, kind)
            canonical = review / f"{kind}.jsonl"
            if not files and not canonical.is_file():
                continue
            bad: list = []
            held = len(saves.read_lines(canonical, bad=bad))
            who = ", ".join(
                f"{name} {len(saves.read_lines(path, bad=bad))}" for name, path in files.items()
            )
            try:
                unmerged = saves.pending(event_dir, kind)
            except ValueError as error:
                unmerged, who = 0, f"{who}; canonical unreadable: {error}"
            total += unmerged
            note = f", {len(bad)} torn lines" if bad else ""
            out.append(f"  {kind}: {held} lines, {unmerged} unmerged{note}"
                       + (f"; writers: {who}" if who else ""))
        refused = len(saves.read_lines(review / saves.SAVES / saves.REJECTED))
        if refused:
            out.append(f"  {refused} saves the table refused (saves/{saves.REJECTED})")
        held_by = saves.holder(review)
        if held_by:
            out.append(f"  lock held by {held_by}")
        if out:
            print(_rel(events, event_dir))
            print("\n".join(out))
    print(f"unmerged saves: {total}")
    return 0


# ---- merge ----


def merge_all(events, rels, *, dry_run: bool) -> int:
    code = 0
    for event_dir in select(events, rels):
        for kind in saves.KINDS:
            if not writer_files(event_dir, kind):
                continue
            try:
                merged = saves.merge(event_dir, kind, *steps(event_dir, kind), dry_run=dry_run)
            except (saves.StoreBusy, saves.StoreMissing, OSError, ValueError) as error:
                print(f"{_rel(events, event_dir)} {kind}: not merged: {error}")
                code = 2
                continue
            print(f"{_rel(events, event_dir)} {kind}: {_counts(merged)}")
            if merged.skipped:
                code = 2
    return code


def _counts(merged: saves.Merged) -> str:
    text = (f"+{len(merged.added)} added, {len(merged.applied)} applied, "
            f"{len(merged.superseded)} superseded")
    if merged.repaired:
        text += f", {len(merged.repaired)} repaired"
    if merged.skipped:
        text += f", {len(merged.skipped)} refused (see saves/{saves.REJECTED})"
    if merged.bad:
        text += f", {len(merged.bad)} torn lines skipped"
    return text


# ---- import-tree ----


def _lines_of(event_dir, kind: str) -> dict[str, tuple[dict, str]]:
    """Every line of the event's canonical log and writer files, by id."""
    review = Path(event_dir) / saves.REVIEW
    found: dict[str, tuple[dict, str]] = {}
    for path in [review / f"{kind}.jsonl", *writer_files(event_dir, kind).values()]:
        pairs = saves.read_lines(path)
        for i, pair in zip(saves.ids_of(e for e, _ in pairs), pairs, strict=True):
            found.setdefault(i, pair)
    return found


def _stamped(save_id: str, pair: tuple[dict, str]) -> tuple[dict, str]:
    """The line with its id written into it when the id is not the one it computes
    (a later one of several old lines that shared an id), so it stays one line here."""
    entry = pair[0]
    if saves.line_id(entry) == save_id:
        return pair
    entry = {**entry, "save_id": save_id}
    return entry, json.dumps(entry)


def import_tree(source, events, rels, *, tag: str | None, dry_run: bool) -> int:
    """Add the lines of `source`'s review logs that this folder does not know to
    a writer file, `saves/<kind>.import.<tag>.jsonl`, then merge them."""
    source, events = Path(source).resolve(), Path(events).resolve()
    if not source.is_dir():
        raise SystemExit(f"{source} is not a folder")
    if source == events:
        raise SystemExit("the source is this folder")
    tag = re.sub(r"[^A-Za-z0-9_-]+", "_", tag or "_".join(source.parts[-2:]))
    code = 0
    for there in select(source, rels):
        rel = there.relative_to(source)
        here = events / rel
        if not here.is_dir():
            print(f"{rel}: not in this folder, skipped")
            continue
        for kind in saves.KINDS:
            theirs = _lines_of(there, kind)
            if not theirs:
                continue
            known = set(_lines_of(here, kind)) | saves.rejected_ids(here / saves.REVIEW)
            new = [_stamped(i, pair) for i, pair in theirs.items() if i not in known]
            name = f"{rel} {kind}: {len(theirs)} lines there, {len(new)} new"
            try:
                if dry_run:
                    merged = saves.merge(
                        here, kind, *steps(here, kind), extra=[e for e, _ in new], dry_run=True
                    )
                    print(f"{name}; would add {len(merged.added)}, apply {len(merged.applied)}")
                    continue
                if new:
                    review = saves.review_dir(here)
                    saves.shared_mkdir(review / saves.SAVES, under=review)
                    writer = re.sub(r"[^A-Za-z0-9_.-]", "_", f"import.{tag}")
                    saves.append_text(
                        review / saves.SAVES / f"{kind}.{writer}.jsonl",
                        "\n".join(text for _, text in new),
                    )
                merged = saves.merge(here, kind, *steps(here, kind))
            except (saves.StoreBusy, saves.StoreMissing, OSError, ValueError) as error:
                print(f"{name}; not merged: {error}")
                code = 2
                continue
            print(f"{name}; {_counts(merged)}")
            if merged.skipped:
                code = 2
    return code


# ---- init ----


def init(events, rels, *, dry_run: bool) -> int:
    """Make `review/` and `review/saves/` (and keep `review/legacy/`) group-writable
    and setgid, so accounts in the folder's group can save; nothing else changes."""
    code = 0
    for event_dir in select(events, rels, missing=True):
        review = event_dir / saves.REVIEW
        for path in (review, review / saves.SAVES, review / "legacy"):
            if not path.exists() and path.name == "legacy":
                continue
            try:
                if not path.exists():
                    print(f"{'would make' if dry_run else 'made'} {path}")
                    if not dry_run:
                        saves.shared_mkdir(path, under=path.parent)
                elif stat.S_IMODE(path.stat().st_mode) != saves.DIR_MODE:
                    print(f"{'would set' if dry_run else 'set'} {path} to {saves.DIR_MODE:o}")
                    if not dry_run:
                        os.chmod(path, saves.DIR_MODE)
            except (OSError, saves.StoreMissing) as error:
                print(f"cannot change {path}: {error}")
                code = 2
    return code


# ---- check ----


def check(events, rels) -> int:
    problems, notes = [], []
    for event_dir in select(events, rels):
        rel, review = _rel(events, event_dir), event_dir / saves.REVIEW
        for kind in saves.KINDS:
            try:
                saves.read_lines(review / f"{kind}.jsonl", strict=True)
            except ValueError as error:
                problems.append(f"{rel}: {error}")
            bad: list = []
            for path in writer_files(event_dir, kind).values():
                saves.read_lines(path, bad=bad)
            if bad:
                notes.append(f"{rel}: {len(bad)} torn or non-JSON lines in {kind} writer files "
                             "(skipped by the merge)")
        age = saves.lock_age(review)
        if age is not None:
            (problems if age > saves.STALE_LOCK_S else notes).append(
                f"{rel}: lock held by {saves.holder(review)}")
        for path in [*review.glob("*"), *(review / saves.SAVES).glob("*")]:
            if path.is_file() and stat.S_IMODE(path.stat().st_mode) & 0o044 != 0o044:
                problems.append(f"{rel}: {path.name} is mode "
                                f"{stat.S_IMODE(path.stat().st_mode):o}: not readable by all")
        for path in (review, review / saves.SAVES):
            if path.is_dir() and not path.stat().st_mode & stat.S_IWGRP:
                notes.append(f"{rel}: {path.relative_to(event_dir)} is not group-writable "
                             "(labeler-sync init, for a shared folder)")
        if (review / "history.jsonl").is_file() and (review / "labels.csv").is_file():
            try:
                wrong = labels.diverged(event_dir)
                older = labels.diverged(event_dir, legacy=True)
            except ValueError as error:
                problems.append(f"{rel}: cannot compare labels.csv with the history: {error}")
                continue
            for entry in wrong:
                problems.append(f"{rel}: shot {entry.get('shot')} is not what its last save "
                                f"({saves.line_id(entry)}) made; labeler-sync repair --shot "
                                f"{entry.get('shot')}")
            if len(older) > len(wrong):
                notes.append(f"{rel}: {len(older) - len(wrong)} shots differ from a last save "
                             "that has no id (older lines, edited around by scripts: not checked)")
    for line in notes:
        print(f"note: {line}")
    for line in problems:
        print(f"PROBLEM: {line}")
    print(f"{len(problems)} problems" if problems else "ok")
    return 1 if problems else 0


# ---- repair ----


def repair(events, rels, shots) -> int:
    code = 0
    for event_dir in select(events, rels):
        review = event_dir / saves.REVIEW
        if not (review / "history.jsonl").is_file():
            continue
        try:
            with saves.lock(review):
                again = [e for e in labels.diverged(event_dir, legacy=bool(shots))
                         if not shots or e.get("shot") in shots]
                refused = labels.apply_labels(event_dir)(again) if again else []
        except (saves.StoreBusy, saves.StoreMissing, OSError, ValueError) as error:
            print(f"{_rel(events, event_dir)}: not repaired: {error}")
            code = 2
            continue
        if again:
            done = sorted(int(e["shot"]) for e in again if all(e is not r for r, _ in refused))
            print(f"{_rel(events, event_dir)}: re-applied the last save of shots {done}")
            for entry, why in refused:
                print(f"{_rel(events, event_dir)}: shot {entry.get('shot')} refused: {why}")
                code = 2
    return code


# ---- command line ----


def main(argv=None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--events", type=Path, help="the label folder (default: this "
                        "environment's label tables)")
    common.add_argument("--event", action="append", default=[], metavar="REL",
                        help="only this event directory (repeatable)")
    parser = argparse.ArgumentParser(prog="labeler-sync", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", parents=[common], help="lines, writers, unmerged saves")
    merged = commands.add_parser("merge", parents=[common], help="merge the writers' files")
    merged.add_argument("--dry-run", action="store_true")
    tree = commands.add_parser("import-tree", parents=[common],
                               help="fold another label folder's review logs into this one")
    tree.add_argument("source", type=Path, help="the other folder (same layout)")
    tree.add_argument("--tag", help="names the writer file (default: from the path)")
    tree.add_argument("--dry-run", action="store_true")
    made = commands.add_parser("init", parents=[common], help="create saves/, set directory modes")
    made.add_argument("--dry-run", action="store_true")
    commands.add_parser("check", parents=[common], help="read-only invariants")
    fixed = commands.add_parser("repair", parents=[common],
                                help="re-apply the last save of the shots check reports")
    fixed.add_argument("--shot", type=int, action="append", default=[],
                       help="only this shot (also one whose last line has no id)")
    args = parser.parse_args(argv)
    events = (args.events or Paths.from_env().label_tables).resolve()
    if not events.is_dir():
        print(f"{events} is not a folder", file=sys.stderr)
        return 2
    match args.command:
        case "status":
            return status(events, args.event)
        case "merge":
            return merge_all(events, args.event, dry_run=args.dry_run)
        case "import-tree":
            return import_tree(args.source, events, args.event, tag=args.tag, dry_run=args.dry_run)
        case "init":
            return init(events, args.event, dry_run=args.dry_run)
        case "check":
            return check(events, args.event)
        case _:
            return repair(events, args.event, set(args.shot))


if __name__ == "__main__":
    sys.exit(main())
