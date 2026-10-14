#!/usr/bin/env python
"""Sawtooth review source and 1/1 crash table, gated by the q = 1 surface.

No q = 1 surface, no 1/1 mode: time the gate (`labeler.sawtooth.qgate`: EFIT02
q_min >= 1.2, else EFIT02ER >= 1.2, else EFIT01 >= 2.0, sustained 100 ms)
calls `no_q1` is a hard absent. Nothing is re-detected.

`source` writes `saw_physics` (the fix5 physics labels, the span rows of
`extend_saw_physics/*.csv`) with every non-present span inside no-q = 1 time
made absent: uncertain, both q-prior states and unassessed alike. Present spans
are kept and the time of theirs the gate would cover is reported. A shot with
no physics record gets absent in no-q = 1 time and not observable elsewhere in
its window (`frames.targets.window_for`). The table covers the named sets (by
default the sawtooth roster, silver, gold and every shot of the event's
`review/labels.csv`), never a blind cohort shot; `--all-physics` adds every
shot of the physics labels (saw_physics v2's 13,666-shot scope, which makes
the review page's first read take 10 s). `--point` moves the event's review
pointer to it.

`crash` writes `crash` for `mhd_n1.m1` from an earlier version's table: present
inside no-q = 1 time becomes absent, everything else is copied, blind cohort
shots are dropped; its per-shot meta is recomputed from the new rows. It never
moves a pointer.

Both metas record, per shot, every equilibrium file the gate read (tree, path
under `$LABELER_ROOT`, bytes, sha256) and one digest over all of them.

    LABELER_ROOT=... LABELER_NO_FETCH=1 PYTHONPATH=src pixi run --frozen -e labeler \\
        python scripts/labeler/sawtooth_source.py source --version v3 --point
    ... python scripts/labeler/sawtooth_source.py crash --from-version v1 --version v3
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from labeler.config import Paths, git_sha
from labeler.events import suggestions
from labeler.events.catalog.states import ABSENT, NOT_OBSERVABLE, PRESENT
from labeler.events.review import labels
from labeler.events.review.silver import blind_shots
from labeler.frames.targets import window_for
from labeler.sawtooth import qgate

EVENT = "sawtooth_oscillation"
METHOD = "saw_physics"
CRASH_EVENT, CRASH_METHOD = "mhd_n1.m1", "crash"
SETS = {
    "roster": f"{EVENT}/shots.csv",
    "silver": "silver/silver_shots.csv",
    "gold": "gold/gold_shots.csv",
    "reviewed": f"{EVENT}/review/labels.csv",
}
#: The shots each stage covers (source) or reports apart (crash) by default.
DEFAULT_SETS = {"source": list(SETS), "crash": ["silver", "gold"]}
CATEGORIES = {"0": "absent", "1": "present", "2": "uncertain", "3": "not_observable"}
#: Rounding gaps between physics spans (0.05-0.1 ms) are closed; wider ones,
#: which the fix5 export never leaves, would be not observable.
CLOSE_GAP_MS = 1.0
#: Shots whose rows the run prints in full (the gold tearing-mode experiment).
SHOW = (199597, 199603, 199607)


def set_shots(paths: Paths, names) -> dict[str, set[int]]:
    return {
        name: set(pd.read_csv(paths.label_tables / SETS[name]).shot.astype(int))
        for name in names
    }


def physics_rows(directory) -> dict[int, list[tuple]]:
    """Each shot's span rows `(category, start, stop, confidence, state)`.

    The crash points (`crowd` False) are dropped: the table has no attrs to keep
    them apart. A shot in a cohort and a population shard appears once; a shot
    split across two population shards is joined. State is the q-prior reason
    where there is one, else the record's state.
    """
    frames = []
    for path in sorted(Path(directory).glob("*.csv")):
        frame = pd.read_csv(path, keep_default_na=False)
        frames.append(frame[frame["crowd"].astype(str).str.lower() == "true"])
    frame = pd.concat(frames, ignore_index=True)
    attrs = frame["attrs"].map(json.loads)
    frame["state"] = [a.get("reason") or a.get("state", "") for a in attrs]
    frame = frame.drop_duplicates(
        ["shot", "category", "t_start", "t_end", "confidence", "state"]
    )
    found = {}
    for shot, rows in frame.groupby("shot", sort=True):
        rows = rows.sort_values(["t_start", "t_end"], kind="stable")
        tiled = []
        for r in rows.itertuples(index=False):
            row = [
                int(r.category),
                float(r.t_start),
                float(r.t_end),
                r.confidence,
                r.state,
            ]
            if tiled:
                gap = row[1] - tiled[-1][2]
                if gap < -1e-6:
                    raise ValueError(f"shot {shot}: physics spans overlap at {row[1]}")
                if 0 < gap < CLOSE_GAP_MS:
                    tiled[-1][2] = row[1]
                elif gap >= CLOSE_GAP_MS:
                    tiled.append([NOT_OBSERVABLE, tiled[-1][2], row[1], "", "gap"])
            tiled.append(row)
        found[int(shot)] = [tuple(row) for row in tiled]
    return found


def gate_one(shot):
    return shot, qgate.shot_gate(shot, Paths.from_env())


def gate_spans(shots, paths: Paths, workers: int) -> tuple[dict, dict]:
    """Each shot's gate spans and the equilibrium files read
    (`qgate.shot_gate`), in `workers` processes."""
    shots = [int(s) for s in sorted(shots)]
    if workers <= 1:
        found = {shot: qgate.shot_gate(shot, paths) for shot in shots}
    else:
        with ProcessPoolExecutor(workers) as pool:
            found = dict(pool.map(gate_one, shots, chunksize=16))
    spans = {shot: got[0] for shot, got in found.items()}
    files = {shot: list(got[1]) for shot, got in found.items()}
    return spans, files


def equilibria_meta(files: dict) -> dict:
    """Counts by tree and one digest over every shot's equilibrium files."""
    trees = Counter(record["tree"] for shot in files for record in files[shot])
    return {
        "files_by_tree": dict(sorted(trees.items())),
        "shots_without_equilibrium": sorted(
            s for s, found in files.items() if not found
        ),
        "digest": qgate.files_digest(
            (shot, record["tree"], record["sha256"])
            for shot in files
            for record in files[shot]
        ),
        "series_digest": qgate.files_digest(
            (shot, record["tree"], record["series_sha256"])
            for shot in files
            for record in files[shot]
        ),
        "digest_of": "sha256 over sorted 'shot tree sha256' lines of the files "
        "(digest) and of the (time ms, q_min) float64 series read (series_digest)",
    }


def no_q1(spans) -> list[tuple[float, float]]:
    return [(a, b) for a, b, state, _ in spans if state == qgate.NO_Q1]


def gated(rows, absent, *, keep):
    """Rows with the parts of each row inside `absent` made absent (state
    `no_q1:<old state>`, blank confidence); rows of a `keep` category are kept."""
    out = []
    for category, a, b, confidence, state in rows:
        if category in keep:
            out.append((category, a, b, confidence, state))
            continue
        cursor = a
        for lo, hi in absent:
            if hi <= cursor or lo >= b:
                continue
            if lo > cursor:
                out.append((category, cursor, lo, confidence, state))
            stop = min(hi, b)
            out.append((ABSENT, max(lo, cursor), stop, "", f"no_q1:{state}"))
            cursor = stop
        if cursor < b:
            out.append((category, cursor, b, confidence, state))
    return out


def merged(shot, rows) -> list[list]:
    """Table rows: touching rows of one category and confidence joined."""
    out = []
    for category, a, b, confidence, _ in rows:
        if b <= a:
            continue
        if (
            out
            and out[-1][1] == category
            and out[-1][4] == confidence
            and abs(out[-1][3] - a) < 1e-6
        ):
            out[-1][3] = b
            continue
        out.append([int(shot), int(category), a, b, confidence])
    for row in out:
        for k in (2, 3):
            value = round(row[k], 4)
            row[k] = int(value) if value == int(value) else value
    return out


def overlap_ms(rows, spans, category) -> float:
    total = 0.0
    for c, a, b, *_ in rows:
        if c == category:
            total += sum(max(0.0, min(b, hi) - max(a, lo)) for lo, hi in spans)
    return total


def seconds(rows, key) -> Counter:
    found = Counter()
    for row in rows:
        found[key(row)] += (row[2] - row[1]) / 1000.0
    return found


def rounded(counter) -> dict:
    return {str(k): round(v, 1) for k, v in sorted(counter.items(), key=str)}


def build_source(args, paths: Paths) -> dict:
    sets = set_shots(paths, args.set)
    physics = physics_rows(args.physics)
    blind = blind_shots(paths.label_tables)
    wanted = set().union(*sets.values()) if sets else set()
    shots = sorted(((set(physics) if args.all_physics else set()) | wanted) - blind)
    spans, files = gate_spans(shots, paths, args.workers)
    rows, per_shot, skipped = [], {}, {}
    before, after, moved, kept_present = {}, {}, Counter(), Counter()
    sources = Counter()
    for shot in shots:
        absent = no_q1(spans[shot])
        if shot in physics:
            old = physics[shot]
        else:
            try:
                (lo, hi), origin = window_for(shot, paths, None)
            except ValueError as error:
                skipped[str(shot)] = str(error)
                continue
            old = [(NOT_OBSERVABLE, float(lo), float(hi), "", f"no_record:{origin}")]
        new = gated(old, absent, keep=(PRESENT, ABSENT))
        rows += merged(shot, new)
        present_in = overlap_ms(old, absent, PRESENT)
        before[shot], after[shot] = old, new
        for category, a, b, _, state in new:
            if state.startswith("no_q1:"):
                moved[state.split(":", 1)[1].split(":")[0]] += (b - a) / 1000.0
        kept_present[shot] = present_in
        for a, b, state, source in spans[shot]:
            sources[(source or "none", state)] += (b - a) / 1000.0
        per_shot[str(shot)] = {
            "physics": shot in physics,
            "no_q1_ms": round(sum(b - a for a, b in absent), 1),
            "gate_ms": {
                source or "none": round(
                    sum(b - a for a, b, _, s in spans[shot] if s == source), 1
                )
                for source in sorted({s for *_, s in spans[shot]})
            },
            **({"present_in_no_q1_ms": round(present_in, 1)} if present_in else {}),
            "equilibria": files[shot],
        }
    reporting = set().union(*set_shots(paths, ["silver", "gold"]).values())
    report = {
        "shots": len(after),
        "physics_shots": sum(s in physics for s in after),
        "gate_only_shots": sum(s not in physics for s in after),
        "skipped": skipped,
        "gate_seconds_by_source_state": {
            f"{source}/{state}": round(v, 1)
            for (source, state), v in sorted(sources.items())
        },
        "absent_from_state_s": rounded(moved),
        "present_in_no_q1_s": round(sum(kept_present.values()) / 1000.0, 3),
        "present_in_no_q1_shots": sorted(s for s, v in kept_present.items() if v > 0),
        "equilibria": equilibria_meta({s: files[s] for s in after}),
        "all": summary(sorted(after), before, after, physics, sets),
        "silver_gold": summary(
            sorted(s for s in after if s in reporting), before, after, physics, sets
        ),
        "show": {
            str(s): {
                "gate": spans.get(s, []),
                "rows": [r for r in rows if r[0] == s],
            }
            for s in SHOW
            if s in after
        },
    }
    meta = {
        "event": EVENT,
        "method": METHOD,
        "version": args.version,
        "git_sha": git_sha(),
        "categories": CATEGORIES,
        "inputs": {
            "physics": str(Path(args.physics).resolve()),
            "physics_files": sorted(p.name for p in Path(args.physics).glob("*.csv")),
            "sets": {name: SETS[name] for name in args.set},
            "all_physics": bool(args.all_physics),
        },
        "rule": {
            "gate": qgate.RULE,
            "absent": "every non-present physics span (uncertain, q_prior_untested, "
            "q_prior_ece_contradicted, unassessed) inside no_q1 time",
            "present": "kept as exported (gaps under 1 ms between physics spans are "
            "closed onto the earlier span); present_in_no_q1_ms reports any overlap",
            "no_record": "shots without a physics record: absent in no_q1 time, not "
            "observable elsewhere in frames.targets.window_for's window",
            "dropped": "crash points (crowd False); blind cohort shots",
            "confidence": "the physics span's; blank where the gate made it absent",
        },
        "report": {k: v for k, v in report.items() if k != "show"},
        "skipped": skipped,
        "per_shot": per_shot,
    }
    return {"rows": rows, "meta": meta, "report": report}


def summary(shots, before, after, physics, sets) -> dict:
    """Seconds per category and per physics state on `shots`, before and after
    the gate; every count is over the same `shots`."""
    by_category = {
        "before": Counter(),
        "after": Counter(),
    }
    by_state = Counter()
    for shot in shots:
        for row in before[shot]:
            by_category["before"][CATEGORIES[str(row[0])]] += (row[2] - row[1]) / 1e3
            by_state[row[4].split(":")[0]] += (row[2] - row[1]) / 1e3
        for row in after[shot]:
            by_category["after"][CATEGORIES[str(row[0])]] += (row[2] - row[1]) / 1e3
    per_set = {
        name: {
            "shots": len(members & set(shots)),
            "physics_shots": len(members & set(shots) & set(physics)),
        }
        for name, members in sets.items()
    }
    zero_absent = {
        when: sum(not any(r[0] == ABSENT for r in rows[s]) for s in shots)
        for when, rows in (("before", before), ("after", after))
    }
    return {
        "shots": len(shots),
        "physics_shots": sum(s in physics for s in shots),
        "sets": per_set,
        "seconds_before": rounded(by_category["before"]),
        "seconds_after": rounded(by_category["after"]),
        "physics_state_seconds_before": rounded(by_state),
        "shots_without_absent": {**zero_absent, "of": len(shots)},
    }


def build_crash(args, paths: Paths) -> dict:
    source = suggestions.table_path(paths, CRASH_EVENT, CRASH_METHOD, args.from_version)
    frame = pd.read_csv(source, keep_default_na=False)
    old_meta = json.loads(source.with_suffix(".meta.json").read_text())
    blind = blind_shots(paths.label_tables)
    dropped = sorted(int(s) for s in set(frame.shot.astype(int)) & blind)
    frame = frame[~frame.shot.astype(int).isin(blind)]
    shots = sorted(frame.shot.unique())
    spans, files = gate_spans(shots, paths, args.workers)
    sets = set_shots(paths, args.set)
    wanted = set().union(*sets.values()) if sets else set()
    rows, removed, totals, per_shot = [], {}, defaultdict(Counter), {}
    for shot, group in frame.groupby("shot", sort=True):
        old = [
            (int(r.category), float(r.t_start), float(r.t_end), r.confidence, "")
            for r in group.sort_values("t_start").itertuples(index=False)
        ]
        absent = no_q1(spans[shot])
        new = gated(old, absent, keep=(ABSENT, NOT_OBSERVABLE, 2))
        table = merged(shot, new)
        rows += table
        gone = overlap_ms(old, absent, PRESENT)
        if gone:
            removed[str(shot)] = round(gone, 1)
        present = [r for r in table if r[1] == PRESENT]
        per_shot[str(int(shot))] = {
            "runs": len(present),
            "present_ms": round(sum(r[3] - r[2] for r in present), 1),
            "no_q1_ms": round(sum(b - a for a, b in absent), 1),
            **({"present_removed_ms": round(gone, 1)} if gone else {}),
            "equilibria": files[shot],
        }
        for scope in ("all", "silver_gold") if shot in wanted else ("all",):
            totals[scope]["present_before"] += seconds(old, lambda r: r[0])[PRESENT]
            totals[scope]["present_after"] += seconds(new, lambda r: r[0])[PRESENT]
            totals[scope]["window"] += sum(r[2] - r[1] for r in old) / 1e3
            totals[scope]["shots"] += 1
            totals[scope]["shots_losing_present"] += bool(gone)
    meta = {
        **{
            k: v
            for k, v in old_meta.items()
            if k not in ("rows", "shots", "per_shot", "from", "gate")
        },
        "version": args.version,
        "git_sha": git_sha(),
        "from": {"table": str(source), "version": args.from_version},
        "gate": {
            "rule": qgate.RULE,
            "applied": "present inside no_q1 time becomes absent; other rows copied",
            "present_removed_ms": removed,
            "equilibria": equilibria_meta(files),
        },
        "dropped": "blind cohort shots",
        "blind_shots_dropped": len(dropped),
        "per_shot_note": "recomputed from this version's rows: runs = present "
        "rows, present_ms, the shot's no_q1_ms, present_removed_ms where the gate "
        "removed present time, and the equilibrium files read; the source "
        "version's crash counts are in its own meta",
        "per_shot": per_shot,
    }
    report = {
        scope: {k: round(v, 1) for k, v in counter.items()}
        for scope, counter in totals.items()
    }
    return {"rows": rows, "meta": meta, "report": report}


def write(path: Path, built: dict) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists: a changed rule or input is a new version")
    path.parent.mkdir(parents=True, exist_ok=True)
    suggestions.write_table(path, built["rows"], built["meta"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("source", "crash"):
        stage = sub.add_parser(name)
        stage.add_argument("--version", required=True)
        stage.add_argument(
            "--set",
            type=lambda s: s.split(","),
            default=DEFAULT_SETS[name],
            help=f"comma-separated, of {', '.join(SETS)} "
            f"(default {','.join(DEFAULT_SETS[name])})",
        )
        stage.add_argument("--workers", type=int, default=8)
        stage.add_argument(
            "--out", type=Path, help="write here instead of the suggestions tree"
        )
        stage.add_argument("--report", type=Path, help="also save the report JSON")
    sub.choices["source"].add_argument(
        "--physics",
        type=Path,
        default=None,
        help="the extend_saw_physics directory (default: the label tables')",
    )
    sub.choices["source"].add_argument(
        "--all-physics",
        action="store_true",
        help="also cover every shot of the physics labels (the v2 scope)",
    )
    sub.choices["source"].add_argument("--point", action="store_true")
    sub.choices["crash"].add_argument("--from-version", default="v1")
    args = parser.parse_args()
    unknown = set(args.set) - set(SETS)
    if unknown:
        parser.error(f"unknown sets {sorted(unknown)}; choose from {sorted(SETS)}")
    if args.stage == "source" and args.point and args.out is not None:
        parser.error("--point needs the suggestions-tree table, not --out")
    paths = Paths.from_env()
    if args.stage == "source":
        if args.physics is None:
            args.physics = paths.label_tables / EVENT / "extend_saw_physics"
        built = build_source(args, paths)
        path = args.out or suggestions.table_path(paths, EVENT, METHOD, args.version)
    else:
        built = build_crash(args, paths)
        path = args.out or suggestions.table_path(
            paths, CRASH_EVENT, CRASH_METHOD, args.version
        )
    write(path, built)
    report = {"table": str(path), "rows": len(built["rows"]), **built["report"]}
    if args.stage == "source" and args.point:
        report["pointer"] = labels.write_pointer(
            paths.label_tables / EVENT, path, method=METHOD, version=args.version
        )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
