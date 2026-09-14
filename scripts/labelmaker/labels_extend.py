#!/usr/bin/env python
"""Project a producer's per-shot events onto a shot-list table or bounded summary."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from labelmaker.config import Paths
from labelmaker.events.databases import (
    FORMAT_COLUMNS,
    FORMAT_SCHEMA_VERSION,
    load_manifest,
    validate_format,
    write_csv,
    write_format_table,
    write_meta,
)
from labelmaker.events.schema import read_events, read_sources

MAX_EVENT_ROWS = 50_000
SUMMARY_COLUMNS = ("shot", "n_events", "t_first_s", "t_last_s", "t_cov0_s", "t_cov1_s")
# Only categories with registered phenomenon IDs can yet be exported. Producer
# tasks add the missing regime/q-min IDs here when they add them to the lexicon.
CATEGORY_PHENOMENA = {
    "alfven_eigenmode": ("ae",),
    "detachment": ("detachment",),
    "edge_localized_mode": ("elm",),
    "resistive_wall_mode": ("rwm",),
    "sawtooth_oscillation": ("sawtooth",),
    "tearing_mode": ("tearing",),
}
# Source records lack a phenomenon column. These unambiguous family producers
# must be known even when every shot has zero events (or the source failed).
PHENOMENON_SOURCES = {
    "elm": ("elm_clock", "tokeye_transient"),
    "sawtooth": ("ece_sawtooth",),
}


def producer_slug(producer: str) -> str:
    """Source IDs retain their spelling in rows; colons become directory underscores."""
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.:-]*", producer):
        raise ValueError("producer must be a source ID or phenomenon ID")
    return producer.replace(":", "_")


def read_shots(path: Path) -> list[int]:
    """Read the recommender YAML, refusing duplicate, fractional or missing shots."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("shots"), list):
        raise TypeError(f"{path}: expected a YAML mapping with a shots list")
    shots = []
    for entry in raw["shots"]:
        value = entry.get("shot") if isinstance(entry, dict) else entry
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{path}: every shot must be an integer")
        shots.append(value)
    if len(set(shots)) != len(shots) or not shots:
        raise ValueError(f"{path}: shots must be non-empty and unique")
    if "n" in raw and raw["n"] != len(shots):
        raise ValueError(f"{path}: n disagrees with the shot list")
    return sorted(shots)


def _bound(values, *, last=False) -> float:
    finite = [float(v) for v in values if math.isfinite(v)]
    return (max(finite) if last else min(finite)) if finite else float("nan")


def export(args) -> Path:
    """Read products without modifying them; retain at most 50,000 projected rows."""
    shots = read_shots(args.shot_list)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    frames, summaries, source_frames = [], [], []
    missing = []
    n_events = 0
    provenance = set()
    actual_sources = {args.producer}
    if args.producer in CATEGORY_PHENOMENA[args.category]:
        actual_sources.update(PHENOMENON_SOURCES.get(args.producer, ()))
        actual_sources.update(s.source for s in load_manifest()
                              if s.phenomenon == args.producer and s.dir == args.category)

    def remember(frame):
        provenance.update((str(r.source), str(r.run_id), str(r.git_sha))
                          for r in frame[["source", "run_id", "git_sha"]]
                          .drop_duplicates().itertuples(index=False))

    for shot in shots:
        path = args.events_root / f"{shot}_events.parquet"
        if not path.exists():
            missing.append(shot)
        frame = read_events(path)
        if not (frame["shot"] == shot).all():
            raise ValueError(f"{path}: contains another shot's events")
        frame = frame[
            ((frame["source"] == args.producer) | (frame["phenomenon"] == args.producer))
            & frame["phenomenon"].isin(CATEGORY_PHENOMENA[args.category])
        ]
        projected = validate_format(frame[list(FORMAT_COLUMNS)], where=str(path))
        n_events += len(frame)
        actual_sources.update(frame["source"])
        remember(frame)
        if n_events <= MAX_EVENT_ROWS:
            if not projected.empty:
                frames.append(projected)
        else:
            frames.clear()
        summaries.append({
            "shot": shot, "n_events": len(frame),
            "t_first_s": _bound(frame["t0_s"]),
            "t_last_s": _bound(frame["t1_s"], last=True),
            "t_cov0_s": _bound(frame["t_cov0_s"]),
            "t_cov1_s": _bound(frame["t_cov1_s"], last=True),
        })
        sources_path = args.events_root / f"{shot}_sources.parquet"
        sources = read_sources(sources_path)
        if not (sources["shot"] == shot).all():
            raise ValueError(f"{sources_path}: contains another shot's sources")
        source_frames.append(sources)

    statuses = Counter()
    for row, sources in zip(summaries, source_frames, strict=True):
        # A phenomenon selector learns its actual producers across the whole list,
        # so their successful zero-event shots can still carry source coverage.
        selected = sources[sources["source"].isin(actual_sources)]
        remember(selected)
        statuses.update(selected["status"])
        ok = selected[selected["status"] == "ok"]
        if not ok.empty:
            row["t_cov0_s"] = _bound(ok["t_cov0_s"])
            row["t_cov1_s"] = _bound(ok["t_cov1_s"], last=True)

    made_from = [{"producer": source, "run_id": run_id, "git_sha": sha}
                 for source, run_id, sha in sorted(provenance)]
    producing_sources = sorted({source for source, _, _ in provenance})
    if args.producer in CATEGORY_PHENOMENA[args.category]:
        if len(producing_sources) > 1:
            raise ValueError(
                f"phenomenon selector {args.producer!r} spans sources "
                f"{producing_sources}; select --producer <source> and write each "
                "to its own extend_<source>/ directory"
            )
        if producing_sources:
            directory = f"extend_{producer_slug(producing_sources[0])}"
            if args.out.parent.name != directory:
                raise ValueError(f"--out must use the producing source's {directory}/")
        elif args.out.parent.name != f"extend_{producer_slug(args.producer)}":
            raise ValueError("no producing source found; use extend_<phenomenon>/ "
                             "for an empty scan")
    # A databases-only run legitimately writes no per-shot products on an absent
    # shot. Its run JSON is the evidence for a complete zero-result scan.
    if args.run_id:
        run_path = args.root / "runs/events" / f"{args.run_id}.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if (not run.get("settings", {}).get("databases_only")
                or run.get("run_id") != args.run_id
                or args.events_root.resolve() != (args.root / "events").resolve()):
            raise ValueError(f"{run_path}: expected the matching databases-only run")
        tables = load_manifest(run["settings"]["label_tables"])
        relevant = {s.source for s in tables if s.dir == args.category
                    and args.producer in (s.source, s.phenomenon)}
        totals = run["totals"]
        if (not relevant or not relevant.issubset(totals["tables"])
                or totals["n_events"] != 0 or totals["n_source_records"] != 0):
            raise ValueError(f"{run_path}: not a zero-event scan of the selected tables")
        if {int(r["shot"]) for r in run["shots"]} != set(shots):
            raise ValueError(f"{run_path}: run shots disagree with the export list")
        if any(r["status"] != "ok" for r in run["shots"]):
            raise ValueError(f"{run_path}: the producer run did not complete successfully")
        if not made_from:
            made_from = [{"producer": args.producer, "run_id": args.run_id,
                          "git_sha": run["git_sha"]}]
    try:
        events_root = str(args.events_root.absolute().relative_to(args.root.absolute()))
    except ValueError:
        events_root = str(args.events_root.absolute())
    missing_path = args.out.with_suffix(".missing_event_shots.json")
    meta = {
        "schema_version": FORMAT_SCHEMA_VERSION,
        "made_from": made_from,
        "made_by": "scripts/labelmaker/labels_extend.py",
        "made_at": now,
        "category": args.category, "producer": args.producer,
        "shot_list": args.shot_list.name,
        "shot_list_sha256": hashlib.sha256(args.shot_list.read_bytes()).hexdigest(),
        "n_requested_shots": len(shots), "n_events": n_events,
        "n_shots_with_events": sum(r["n_events"] > 0 for r in summaries),
        "missing_event_shots": {"count": len(missing), "first_20": missing[:20],
                                "full_list": missing_path.name},
        "source_status_counts": dict(statuses),
        "full_events_root": events_root,
        "full_events": "<full_events_root>/<shot>_events.parquet; "
                       "a relative full_events_root is relative to --root "
                       "($LABELMAKER_ROOT)",
        "coverage": "Bounds of successful source coverage; not a claim of continuous "
                    "coverage. Missing products and absent curated shots are not negatives.",
    }
    if args.run_id:
        meta["run_metadata"] = f"runs/events/{args.run_id}.json"
    summary_path = args.out.with_suffix(".summary.csv")
    if n_events > MAX_EVENT_ROWS:
        out, stale = summary_path, args.out
        frame = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)
        meta.update(table_kind="per_shot_summary", n_rows=len(frame), n_shots=len(shots))
        write_csv(frame, out)
        write_meta(out, meta)
    else:
        out, stale = args.out, summary_path
        frame = pd.concat(frames, ignore_index=True) if frames \
            else pd.DataFrame(columns=FORMAT_COLUMNS)
        meta["table_kind"] = "events"
        write_format_table(frame, out, meta)
    missing_path.write_text(json.dumps(missing) + "\n", encoding="utf-8")
    stale.unlink(missing_ok=True)
    stale.with_suffix(".meta.json").unlink(missing_ok=True)
    print(f"{out}: {n_events} events on {len(shots)} requested shots")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True, choices=sorted(CATEGORY_PHENOMENA))
    parser.add_argument("--producer", required=True)
    parser.add_argument("--shot-list", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Paths.from_env().root,
                        help="Labelmaker root (read-only); also locates --run-id metadata")
    parser.add_argument("--events-root", type=Path)
    parser.add_argument("--run-id", help="Completed run JSON for a scan with no shot products")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.events_root is None:
        args.events_root = args.root / "events"
    try:
        slug = producer_slug(args.producer)
    except ValueError as exc:
        parser.error(str(exc))
    phenomenon_selector = args.producer in CATEGORY_PHENOMENA[args.category]
    if ((not phenomenon_selector and args.out.parent.name != f"extend_{slug}")
            or not args.out.parent.name.startswith("extend_")
            or args.out.parent.parent.name != args.category
            or args.out.name != f"{args.shot_list.stem}.csv"):
        parser.error("--out must be <category>/extend_<producer>/<shot-list-stem>.csv")
    export(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
