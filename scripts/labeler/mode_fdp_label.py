#!/usr/bin/env python
"""MHD mode labels (toroidal n, with m/n) from omnimode's port-v6 labels, in the labeler's formats.

Source: omnimode's port-v6 track tables, one set of files a shot (``<shot>.json``,
``.tracks.parquet``, ``.frames.parquet``; ``--labels``). Nothing is computed here and nothing is
fetched: this converts tables, so it runs anywhere the files are readable, with no GPU, no fdp and
no network. Rules (owner, 2026-11-07; ``labeler.modes.tracks``): harmonics are not labelled (a
track with ``harm`` true is dropped, an independent 4/2 stays); scope is |n| = 1, 2, 3 with their
m (a later redesign adds the "n>3" and "n=0" classes); only labelled modes are written (status
"m/n"); amplitudes are uncalibrated and carry none. The events are ``mhd_n1``, ``mhd_n2`` and
``mhd_n3``, each with an ``m<m>`` subcategory per class (a 2/1 mode is ``mhd_n1`` / ``m2``).

Every command takes ``--shots FILE``, one shot a line (``#`` starts a comment): only those shots
are read, so the list is the seal. Four subcommands:

``status``: how many listed shots have port-v6 labels and how many do not (counts only: no shot
    number is printed, as the list may hold held-out shots).
``suggest``: the ``mhd_modes`` suggestion tables (v5's format and semantics: ``shot, category,
    t_start, t_end, confidence``, 0 absent, 1 present, 2 uncertain, 3 not observable) into
    ``<root>/suggestions/mhd_modes/<version>/``; a version is never written into twice, and a
    class with no entry in ``--manifest``'s taxonomy is refused before anything is written. With
    ``--events`` it also points each event's ``review/source.json`` at its table
    (``labeler.events.review.labels.write_pointer``) and makes the ``m<m>`` folders it needs.
``roster``: replace the ``shots.csv`` of ``mhd_n1``, ``mhd_n2`` and ``mhd_n3`` under ``--events``
    with the listed shots (``labeler.modes.roster``), after copying each file to ``--backup``.
    Prints counts only; ``--dry-run`` writes nothing.
``publish``: the 1 ms databases of ``docs/labeler/fdp_labels.md`` (spans, shots, valid, grid_1ms),
    one a parent event, at ``<out-root>/<event>/<run>/``. ``--out-root`` has no default: name
    ``.../tokamaksi/diiid/dev`` for a development run (``release/`` and the Hugging Face export
    are never written by this script).

The labels are PROPOSED, not checked: the table rows are model output with the labeller's own
confidence, to be superseded by the labeller redesign. An audited shot (``--audited-shot``) takes
the tier ``silver`` with its reviewer and date, every other shot ``unverified``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

DEFAULT_ROOT = "/scratch/gpfs/EKOLEMEN/nc1514/labeler"
EVENTS = ("mhd_n1", "mhd_n2", "mhd_n3")
FAILED_NOTE = "no label (labeller failed)"
NO_DATA_NOTE = "no label (no magnetics data on disk)"
NO_WINDOW_NOTE = "no label (no assessed window)"
NO_MODES_NOTE = "no n=1-3 modes"


def read_shots(path) -> list[int]:
    """The listed shots: one a line, ``#`` starts a comment, blank lines skipped, no repeats."""
    shots = []
    for line in Path(path).read_text().splitlines():
        text = line.split("#", 1)[0].strip()
        if text:
            shots.append(int(text))
    if len(shots) != len(set(shots)):
        raise SystemExit(f"{path} lists a shot twice")
    return sorted(shots)


def audited_args(args) -> dict[int, dict]:
    from labeler.modes import roster

    if not args.audited_shot:
        return {}
    return {
        int(shot): roster.audited_row(
            shot, args.audited_tier, args.reviewer, args.verified_on, args.note
        )
        for shot in args.audited_shot
    }


def cmd_status(args) -> None:
    """Counts only: no shot number is printed (the list may hold held-out shots)."""
    from collections import Counter

    from labeler.modes import suggest

    shots = read_shots(args.shots)
    ready, skipped, _ = suggest.load(args.labels, shots)
    why = Counter(reason.split(":")[0] for reason in skipped.values())
    print(
        f"{len(shots)} listed; {len(ready)} labeled; {len(skipped)} without a label "
        f"{dict(why)}"
    )


def cmd_suggest(args) -> None:
    from collections import Counter

    from labeler.events import taxonomy
    from labeler.modes import suggest

    shots = read_shots(args.shots)
    summary = suggest.write(
        args.labels,
        shots,
        args.root,
        version=args.version,
        repo=REPO,
        taxonomy=taxonomy.load(args.manifest),
    )
    print(
        f"{summary['shots']} of {len(shots)} listed shots written; "
        f"{len(summary['skipped'])} skipped "
        f"{dict(Counter(v.split(':')[0] for v in summary['skipped'].values()))}"
    )
    for event, row in summary["events"].items():
        print(
            f"{event}: {row['rows']} rows, {row['present_runs']} present runs -> {row['path']}"
        )
    print("subcategories:", json.dumps(summary["classes"]))
    if args.events:
        pointers = suggest.point(
            args.events, args.root, summary["classes"], version=args.version
        )
        for event, entry in pointers.items():
            print(
                f"pointer {event}: {entry['method']} {entry['version']} {entry['set_at']}"
            )


def note_for(got) -> str | None:
    """The roster note of a shot read by `tracks.read_shot` (None when it is labelled with modes).

    A shot with no label says why (``no label (<port status>)``); a labelled shot with no
    |n| 1-3 mode says so. The notes hold no shot number."""
    from labeler.modes import suggest

    if got is None:
        return NO_DATA_NOTE
    status = str(got.info.get("status", ""))
    if got.info.get("model_error") or status.startswith("read error"):
        return FAILED_NOTE
    if not got.info.get("window"):
        return (
            f"no label ({status})" if status.startswith("no window") else NO_WINDOW_NOTE
        )
    if status == "no n=1-3 tracks":
        return NO_MODES_NOTE
    if status not in suggest.LABELLED:
        return f"no label ({status})"
    return None


def cmd_roster(args) -> None:
    from collections import Counter

    from labeler.modes import roster, tracks

    shots = read_shots(args.shots)
    notes = {}
    for shot in shots:
        note = note_for(tracks.read_shot(args.labels, shot))
        if note:
            notes[shot] = note
    audited = audited_args(args)
    missing = sorted(set(audited) - set(shots))
    if missing:
        raise SystemExit("an audited shot is not in the listed shots")
    print(
        f"{len(shots)} listed; {len(notes)} with a note: "
        f"{dict(Counter(notes.values()))}"
    )
    old = {
        event: roster.read(Path(args.events) / event / "shots.csv") for event in EVENTS
    }
    for event, frame in old.items():
        new, counts = roster.replace(frame, shots, audited, notes)
        print(f"{event}/shots.csv: {json.dumps(counts)}")
        if args.dry_run:
            continue
        path = Path(args.events) / event / "shots.csv"
        roster.backup(path, args.base, args.backup)
        roster.write(path, new)
        again = roster.read(path)
        assert len(again) == counts["new"], event


def cmd_publish(args) -> None:
    from labeler.modes import fdp, suggest

    shots = read_shots(args.shots)
    ready, skipped, dropped = suggest.load(args.labels, shots)
    kids = suggest.subcategories(ready)
    tiers = {
        s: {k: v for k, v in row.items() if k != "shot" and k != "holdout"}
        for s, row in audited_args(args).items()
    }
    freeze = sorted({str(got.info.get("freeze")) for _, got in ready})
    trees = sorted(
        {str((got.info.get("inputs") or {}).get("tree")) for _, got in ready}
    )
    source = {
        "labels": "omnimode port-v6 (<shot>.json, .tracks.parquet, .frames.parquet)",
        "equilibrium": "each shot's own tree in omnimode's info record "
        f"(EFIT02ER first, else EFIT02 or EFIT01): {trees}",
        "freeze": freeze,
        "status": "proposed, not checked; to be superseded by the labeller redesign",
    }
    rules = [
        "harmonics are not labelled: a track with harm true is dropped, an independent m/n stays",
        "scope |n| = 1, 2, 3 with their m; a later redesign adds the n>3 and n=0 classes",
        "only labelled modes are written (status m/n, MAP class at P >= 0.9 inside the gate)",
        "amplitudes are uncalibrated and not carried",
    ]
    note = [
        (
            "Proposed, not checked: model output of omnimode port-v6, to be superseded by "
            "the labeller redesign. Only a shot with a tier above unverified was audited."
        ),
        "Unlabelled tracks and abstentions (uncertain, no rotation, q_min above) are not written.",
    ]
    for event in EVENTS:
        out = Path(args.out_root) / event / args.run
        meta = fdp.write(
            out,
            event,
            ready,
            kids,
            shots,
            skipped,
            dropped,
            tiers,
            run=args.run,
            source=source,
            rules=rules,
            repo=REPO,
            note=note,
        )
        c = meta["counts"]
        print(
            f"{event}: {c['labeled']} of {c['listed']} shots labeled, {c['parent_spans']} parent "
            f"spans, {c['subcategory_spans']} subcategory spans {c['by_subcategory']} -> {out}"
        )


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="omnimode's port-v6 folder (<shot>.json, .tracks.parquet, .frames.parquet)",
    )
    p.add_argument(
        "--shots",
        type=Path,
        required=True,
        help="the shots to read, one a line; no other shot is read",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")

    def audited(q):
        q.add_argument("--audited-shot", type=int, nargs="*", default=[])
        q.add_argument("--audited-tier", default="silver", choices=("silver", "gold"))
        q.add_argument("--reviewer", default="")
        q.add_argument("--verified-on", default="")
        q.add_argument("--note", default="")

    s = sub.add_parser("suggest")
    s.add_argument(
        "--root",
        type=Path,
        default=Path(DEFAULT_ROOT),
        help="the labeler root holding suggestions/ (default %(default)s)",
    )
    s.add_argument("--version", default="v6port")
    s.add_argument(
        "--manifest",
        type=Path,
        default=REPO / "configs" / "labeler" / "events.yaml",
        help="the events.yaml whose taxonomy must name every subcategory (default %(default)s)",
    )
    s.add_argument(
        "--events",
        type=Path,
        default=None,
        help="also point <events>/<event>/review/source.json at the tables",
    )
    r = sub.add_parser("roster")
    r.add_argument(
        "--events",
        type=Path,
        required=True,
        help="the events folder holding mhd_n*/shots.csv",
    )
    r.add_argument(
        "--base",
        type=Path,
        required=True,
        help="the backup mirrors paths relative to this",
    )
    r.add_argument(
        "--backup",
        type=Path,
        required=True,
        help="where each file is copied before it is changed",
    )
    r.add_argument("--dry-run", action="store_true")
    audited(r)
    u = sub.add_parser("publish")
    u.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="<out-root>/<event>/<run>/ is written",
    )
    u.add_argument("--run", required=True)
    audited(u)
    args = p.parse_args(argv)
    {
        "status": cmd_status,
        "suggest": cmd_suggest,
        "roster": cmd_roster,
        "publish": cmd_publish,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
