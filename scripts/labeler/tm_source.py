#!/usr/bin/env python
"""Tearing-mode review source `tm_mhd`: the RMS rule's labels joined with the MHD tracks.

Decisions S5 (as amended on 2026-10-14) and EQ4 (round 5); the rule is
`labeler.tearing.source`. Nothing is re-detected: the rule labels are read from
`extend_tm_interval/tm_interval.csv` (the 450 development shots; never rewritten, the
benchmark pins its sha256) and, for other shots, from
`$LABELER_ROOT/round4/tm/labels/tm_interval_population.csv`; the tracks from the
`mhd_modes` pipeline cache (`labeler.mhd.pipeline.load`, never rerun; `--mhd-cache` names
the cache version, by default the pipeline's `CACHE_VERSION`). The table covers the
event's roster, silver, gold and every shot of the event's `review/labels.csv`, never a
blind cohort shot. A shot with neither rule labels nor a cached MHD run is skipped. The
meta records the inputs' sha256, each shot's equilibrium tree and its time per state,
against the rule-only flattening of the same shots. A version that exists is refused;
`--point` moves the event's review pointer (`labels.write_pointer`).

    LABELER_ROOT=... LABELER_NO_FETCH=1 PYTHONPATH=src pixi run --frozen -e labeler \\
        python scripts/labeler/tm_source.py --version v2 [--mhd-cache v5] [--point]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from labeler.config import Paths, git_sha
from labeler.events import suggestions
from labeler.events.review import labels
from labeler.events.review.silver import blind_shots
from labeler.mhd import equilibrium as EQ
from labeler.mhd import freeze, pipeline
from labeler.tearing import source as S

EVENT = "neoclassical_tearing_mode"
METHOD = "tm_mhd"
SETS = {
    "roster": f"{EVENT}/shots.csv",
    "silver": "silver/silver_shots.csv",
    "gold": "gold/gold_shots.csv",
    "reviewed": f"{EVENT}/review/labels.csv",
}
COHORT_RULE = f"{EVENT}/extend_tm_interval/tm_interval.csv"
POPULATION_RULE = "round4/tm/labels/tm_interval_population.csv"
NAMES = {0: "absent", 1: "present", 2: "uncertain", 3: "not_observable"}
#: The gold tearing-mode experiment, printed in full.
SHOW = (199597, 199603, 199607)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mhd_pointer(paths: Paths) -> dict | None:
    """The mhd_n1 review pointer (the `mhd_modes` tables made from the same cache)."""
    path = labels.pointer_path(paths.label_tables / "mhd_n1")
    return json.loads(path.read_text()) if path.is_file() else None


def set_shots(paths: Paths) -> dict[str, set[int]]:
    out = {}
    for name, rel in SETS.items():
        path = paths.label_tables / rel
        out[name] = set(pd.read_csv(path).shot.astype(int)) if path.is_file() else set()
    return out


def rule_tables(paths: Paths) -> tuple[dict, dict]:
    """Each shot's rule rows and which table they came from (the cohort table first)."""
    tables = {
        "cohort": paths.label_tables / COHORT_RULE,
        "population": paths.root / POPULATION_RULE,
    }
    rows, origin = {}, {}
    for name, path in tables.items():
        for shot, group in pd.read_csv(path).groupby("shot"):
            if int(shot) not in rows:
                rows[int(shot)], origin[int(shot)] = group, name
    inputs = {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in tables.items()
    }
    return rows, {"tables": inputs, "origin": origin}


def cache_digest(shot: int, paths: Paths) -> str:
    """sha256 over the shot's cache files (info JSON, tracks, frames) in that order."""
    h = hashlib.sha256()
    for f in ("info", "tracks", "frames"):
        path = pipeline._files(paths, shot)[f]
        h.update(path.read_bytes() if path.is_file() else b"")
    return h.hexdigest()


def rot_times(paths: Paths, shot: int) -> tuple[np.ndarray, str | None]:
    """ZIPFIT times (ms) `EQ.rotation` keeps, and the file's sha256 (None: no file)."""
    got, _ = EQ.rotation(paths, shot)
    times = np.concatenate([tt for _, _, tt in got]) if got else np.empty(0)
    path = EQ.rot_path(paths, shot)
    return times, sha256(path) if path.is_file() else None


def seconds(state) -> dict[str, float]:
    state = np.asarray(state)
    return {
        NAMES[k]: round(float((state == k).sum()) / 1000.0, 3)
        for k in NAMES
        if (state == k).any()
    }


def ms(mask) -> float:
    """Seconds of a per-ms mask."""
    return round(float(np.sum(mask)) / 1000.0, 3)


def one_shot(shot: int, paths: Paths, rule_rows) -> dict | None:
    """The shot's rows and its record; None when it has neither input."""
    got = pipeline.load(shot, paths)
    if rule_rows is None and got is None:
        return None
    if rule_rows is not None:
        spans = rule_rows[rule_rows.t_end > rule_rows.t_start]
        lo, hi = int(np.floor(spans.t_start.min())), int(np.ceil(spans.t_end.max()))
        rule = S.rule_cells(rule_rows, lo, hi)
        locked = S.locked_cells(rule_rows, lo, hi)
    else:
        lo, hi = (
            int(np.floor(got.info["window"][0])),
            int(np.ceil(got.info["window"][1])),
        )
        rule = np.full(hi - lo, S.NO_RULE, np.int64)
        locked = np.zeros(hi - lo, bool)
    n = hi - lo
    record = {"window": [lo, hi]}
    if got is None:
        masks = {
            "present": np.zeros(n, bool),
            "conf": np.full(n, np.nan),
            "seen": np.zeros(n, bool),
            "unsure": np.zeros(n, bool),
            "cls": {c: np.zeros(n, bool) for c in S.CLASSES + S.DETAILS},
        }
        observed = read = np.zeros(n, bool)
        record.update(
            mhd=f"not in the mhd_modes {pipeline.CACHE_VERSION} cache", tree=None
        )
    else:
        tree = pipeline.tree(got.info)
        times, rot_sha = rot_times(paths, shot)
        masks = S.track_masks(got.tracks, got.frames, lo, hi)
        observed = S.observed_cells(got.info, got.mirnov, tree, times, lo, hi)
        read = S.read_cells(got.info, got.mirnov, lo, hi)
        mlo, mhi = got.info["window"]
        record.update(
            mhd=got.info.get("status", ""),
            tree=tree,
            equilibrium=got.info.get("inputs", {}).get("equilibrium"),
            rotation=bool(len(times)),
            mhd_window=[mlo, mhi],
            window_differs_ms=round(abs(mlo - lo) + abs(mhi - hi), 1),
            cache_sha256=cache_digest(shot, paths),
            rot_sha256=rot_sha,
        )
    state = S.merge(
        rule,
        observed,
        masks["present"],
        masks["seen"],
        masks["unsure"],
        locked=locked,
        read=read,
    )
    conf = np.where(state == S.PRESENT, masks["conf"], np.nan)
    has_rule = rule != S.NO_RULE
    moved = Counter()
    for old, new in zip(rule[has_rule], state[has_rule]):
        if old != new:
            moved[f"{NAMES[int(old)]}->{NAMES[int(new)]}"] += 1
    kept = (rule == S.UNCERTAIN) & (state == S.UNCERTAIN)
    sure = kept & ~masks["unsure"]
    why_kept = {  # a partition of the rule-uncertain time kept uncertain
        "mhd_unsure_track": kept & masks["unsure"],
        "not_observed": sure & ~observed,
        "no_track": sure & observed & ~masks["seen"],
        "locked_phase": sure & observed & masks["seen"],  # absent but for decision M2
    }
    unsure_kinds = {  # inside mhd_unsure_track; they overlap
        "out_of_set_label": kept & masks["cls"]["unsure:label"],
        "q1_harmonic": kept & masks["cls"]["unsure:harmonic"],
    }
    record.update(
        rule="yes" if has_rule.all() else ("no" if not has_rule.any() else "partial"),
        observed_s=ms(observed),
        unread_no_rule_s=ms(~has_rule & ~read),
        seconds=seconds(state),
        rule_seconds=seconds(rule[has_rule]),
        moved_s={k: round(v / 1000.0, 3) for k, v in sorted(moved.items())},
        uncertain_kept_s={k: ms(v) for k, v in why_kept.items() if v.any()},
        uncertain_kept_unsure_kinds_s={
            k: ms(v) for k, v in unsure_kinds.items() if v.any()
        },
        resonant_s=ms(masks["present"]),
    )
    if got is not None:
        detail = S.track_classes(got.tracks, got.frames, detail=True)
        if masks["present"].any():
            record["present_by_label_s"] = label_seconds(
                got, detail == "resonant", lo, hi, masks["present"]
            )
        for kind in S.DETAILS:
            if (detail == kind).any():
                record[f"{kind.replace(':', '_')}_by_label_s"] = label_seconds(
                    got, detail == kind, lo, hi
                )
    return {"rows": S.table_rows(shot, lo, state, conf), "record": record}


def label_seconds(got, sel, lo, hi, inside=None) -> dict[str, float]:
    """Seconds each m/n's tracks of `sel` cover over [lo, hi), inside `inside` when
    given (the present spans); labels overlap."""
    out = {}
    for name, group in got.tracks[sel].groupby("label"):
        fr = got.frames[got.frames.family_id.isin(set(group.family_id))]
        mask = S._paint(lo, hi, S.spans(fr.t_ms, np.zeros(len(fr))))
        if inside is not None:
            mask &= inside
        out[str(name)] = round(float(mask.sum()) / 1000.0, 3)
    return dict(sorted(out.items()))


def totals(records: dict, shots) -> dict:
    out = {"shots": 0, "seconds": Counter(), "rule_seconds": Counter()}
    moved, kept, kinds, labels_s = Counter(), Counter(), Counter(), Counter()
    out_of_set, harmonic = Counter(), Counter()
    trees = Counter()
    unread = 0.0
    for s in shots:
        r = records[s]
        out["shots"] += 1
        out["seconds"].update(r["seconds"])
        out["rule_seconds"].update(r["rule_seconds"])
        moved.update(r["moved_s"])
        kept.update(r["uncertain_kept_s"])
        kinds.update(r.get("uncertain_kept_unsure_kinds_s", {}))
        labels_s.update(r.get("present_by_label_s", {}))
        out_of_set.update(r.get("unsure_label_by_label_s", {}))
        harmonic.update(r.get("unsure_harmonic_by_label_s", {}))
        trees[str(r.get("tree"))] += 1
        unread += r.get("unread_no_rule_s", 0.0)
    rnd = lambda c: {k: round(v, 1) for k, v in sorted(c.items())}
    return {
        "shots": out["shots"],
        "by_rule": dict(Counter(records[s]["rule"] for s in shots)),
        "trees": dict(sorted(trees.items())),
        "seconds": rnd(out["seconds"]),
        "rule_only_seconds": rnd(out["rule_seconds"]),
        "moved_s": rnd(moved),
        "rule_uncertain_kept_s": rnd(kept),
        "rule_uncertain_kept_unsure_kinds_s": rnd(kinds),
        "no_rule_unread_s": round(unread, 1),
        "present_by_label_s": rnd(labels_s),
        "out_of_set_track_cover_by_label_s": rnd(out_of_set),
        "q1_harmonic_track_cover_by_label_s": rnd(harmonic),
        "shots_with_present": sum(
            records[s]["seconds"].get("present", 0) > 0 for s in shots
        ),
    }


def build(args, paths: Paths) -> dict:
    sets = set_shots(paths)
    blind = blind_shots(paths.label_tables)
    wanted = set().union(*sets.values())
    shots = sorted(wanted - blind)
    rules, rule_meta = rule_tables(paths)
    rows, records, skipped = [], {}, {}
    for shot in shots:
        got = one_shot(shot, paths, rules.get(shot))
        if got is None:
            skipped[str(shot)] = (
                f"no rule labels and no mhd_modes {pipeline.CACHE_VERSION} cache entry"
            )
            continue
        got["record"]["rule_table"] = rule_meta["origin"].get(shot)
        rows += got["rows"]
        records[shot] = got["record"]
    done = sorted(records)
    with_rule = [s for s in done if records[s]["rule"] != "no"]
    report = {
        "all": totals(records, done),
        "with_rule_labels": totals(records, with_rule),
        "without_rule_labels": totals(records, [s for s in done if s not in with_rule]),
        "sets": {
            name: totals(records, [s for s in done if s in members])
            for name, members in sets.items()
            if members
        },
        "blind_excluded": len(wanted & blind),
        "skipped": skipped,
        "window_differs": {
            str(s): records[s]["window_differs_ms"]
            for s in done
            if records[s].get("window_differs_ms")
        },
    }
    cache_all = hashlib.sha256(
        "".join(f"{s} {records[s].get('cache_sha256')}\n" for s in done).encode()
    ).hexdigest()
    meta = {
        "event": EVENT,
        "method": METHOD,
        "version": args.version,
        "git_sha": git_sha(),
        "categories": {str(k): v for k, v in NAMES.items()},
        "inputs": {
            "rule": rule_meta["tables"],
            "rule_note": "tm_interval.csv rows for its 450 shots, the population table's "
            "for the rest (identical on the shared shots); read, never rewritten",
            "mhd": {
                "cache": str(pipeline.cache_dir(paths)),
                "cache_version": pipeline.CACHE_VERSION,
                "mhd_modes_pointer": mhd_pointer(paths),
                "cache_digest": cache_all,
                "cache_digest_of": "sha256 over sorted 'shot sha256' lines, each the "
                "sha256 of the shot's info JSON, tracks and frames files",
                "freeze_sha": freeze.sha(),
            },
            "sets": {k: v for k, v in SETS.items() if sets[k]},
            "blind": "data/events/catalog/cohort.csv blind shots, excluded",
        },
        "rule": {
            "decision": "S5 as amended 2026-10-14 05:45 (tm review 8/10), EQ4",
            "tm_labels": list(S.TM_LABELS),
            "locked_reasons": list(S.LOCKED_REASONS),
            "present": "the rule's present time UNION co-rotating resonant tracks "
            f"labelled {', '.join(S.TM_LABELS)} (status m/n; 4/2 marks an island at "
            "q = 2) that are not 1/1 harmonics (below): frames "
            f"+-{S.PAD_MS:g} ms, gaps <= {S.MERGE_MS:g} ms joined, spans >= "
            f"{S.MIN_MS:g} ms; confidence the span's largest P(resonant), blank on "
            "rule-only present time",
            "unsure_tracks": "uncertain, never absent: status uncertain / no rotation "
            "/ q_min above; unjudged (no equilibrium, no slice within 30 ms, no views, "
            "model failed); status m/n with any other m != n label (4/1, 5/2, 5/3, "
            "7/3: outside the named classes, the resonance law was not fit on them); "
            "and an n >= 2 track that would be resonant with at least "
            f"{S.HARM_SHARE:.0%} of its frames at n times a simultaneous resonant 1/1 "
            f"track frame's frequency (|n f_1 - f| <= max({S.HARM_KHZ:g} kHz, "
            f"{S.HARM_REL:g} f), labeler.mhd.rules.harmonic_lock's tolerance: an "
            "unlocked harmonic of the 1/1)",
            "q1_tracks": "status m/n with m == n (1/1 and the k/k harmonics)",
            "rule_uncertain": "absent where the MHD observed the time and only 1/1 "
            "(k/k) or not-resonant tracks cover it, except inside a rule row whose "
            f"reason is {' or '.join(S.LOCKED_REASONS)} (a locked island is invisible "
            "to rotating-mode tracks); uncertain where an unsure track or a resonant "
            "track shorter than a span covers it, where the MHD did not observe it, or "
            "where no track does",
            "rule_absent_not_observable": "kept, unless a resonant span covers it",
            "no_rule_labels": "present from the resonant spans; not observable where "
            "the Mirnov was never read (no readable record, outside its finite "
            "stretch, its NaN gaps; decision M6), also under a resonant span; absent "
            "where the MHD observed and found no unsure or resonant track; uncertain "
            "elsewhere",
            "observed": "decision EQ4: Mirnov record read and covering the ms (outside "
            "its NaN gaps), an equilibrium tree (EFIT02, else EFIT02ER or EFIT01; "
            f"per-shot `tree`) with a slice within {S.EQ_MAX_DT:g} ms, and a ZIPFIT "
            f"rotation profile within {S.ROT_MAX_DT:g} ms",
            "rule_overlaps": "overlapping rule rows: not observable > present > "
            "uncertain > absent at each ms centre; onset points dropped",
            "track_cover": f"frames +-{S.PAD_MS:g} ms, gaps <= {S.MERGE_MS:g} ms joined",
        },
        "report": {k: v for k, v in report.items()},
        "skipped": skipped,
        "per_shot": {str(s): records[s] for s in done},
    }
    show = {str(s): [r[1:] for r in rows if r[0] == s] for s in SHOW if s in records}
    return {"rows": rows, "meta": meta, "report": report, "show": show}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--version", required=True)
    parser.add_argument("--out", type=Path, help="write here, not the suggestions tree")
    parser.add_argument("--point", action="store_true")
    parser.add_argument("--report", type=Path, help="also save the report JSON")
    parser.add_argument(
        "--mhd-cache",
        default=pipeline.CACHE_VERSION,
        help="the mhd_modes pipeline cache version to read (default %(default)s)",
    )
    args = parser.parse_args(argv)
    if args.point and args.out is not None:
        parser.error("--point needs the suggestions-tree table, not --out")
    pipeline.CACHE_VERSION = args.mhd_cache  # `pipeline.load` reads this cache
    paths = Paths.from_env()
    path = args.out or suggestions.table_path(paths, EVENT, METHOD, args.version)
    if path.exists():
        raise SystemExit(f"{path} exists: a changed rule or input is a new version")
    built = build(args, paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = suggestions.write_table(path, built["rows"], built["meta"])
    out = {
        "table": str(path),
        "rows": len(frame),
        "shots": int(frame.shot.nunique()),
        **built["report"],
        "show": built["show"],
    }
    if args.point:
        out["pointer"] = labels.write_pointer(
            paths.label_tables / EVENT, path, method=METHOD, version=args.version
        )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
