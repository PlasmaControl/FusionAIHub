"""Uncertain gaps inside otherwise clean sawtooth trains (fix round 4, minor).

A block is a run of present spans joined across gaps that are entirely uncertain
and at most ``--max-gap`` seconds long. The share is the uncertain gap time over
the block time. Each gap is attributed to what is in it:

* an accepted crash point left uncertain, with its uncertainty reasons;
* no accepted crash point (a crash missed by the profile or edge tests, so the
  period ratio test splits the train).

Reads the per-shot records of the current label round and writes
``uncertain_gaps.json``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sawtooth_physics import OUTPUT, REPO, WORK, save_json


def blocks(record, max_gap):
    """Present blocks and the uncertain gaps inside them."""
    states = record.get("states", [])
    found, current = [], None
    for index, span in enumerate(states):
        if span["state"] != "present":
            continue
        if current is not None:
            between = states[current["last_index"] + 1 : index]
            gap = sum(s["end_s"] - s["start_s"] for s in between)
            if all(s["state"] == "uncertain" for s in between) and gap <= max_gap:
                if between:
                    current["gaps"].append(
                        (between[0]["start_s"], between[-1]["end_s"])
                    )
                current["end_s"] = span["end_s"]
                current["last_index"] = index
                continue
            found.append(current)
        current = {
            "start_s": span["start_s"],
            "end_s": span["end_s"],
            "last_index": index,
            "gaps": [],
        }
    if current is not None:
        found.append(current)
    return found


def attribute(record, start, end):
    """What an uncertain gap contains: crash reasons, or nothing accepted."""
    inside = [p for p in record["crashes"] if start - 1e-6 <= p["time_s"] <= end + 1e-6]
    reasons = Counter()
    for point in inside:
        if point["attrs"]["state"] == "present":
            continue
        for reason in point["attrs"].get("uncertainty_reasons") or ["unspecified"]:
            reasons[reason] += 1
    if reasons:
        return "uncertain_crash", dict(reasons)
    return "no_accepted_crash", {}


def summarize(records, max_gap):
    totals = Counter()
    reasons = Counter()
    kinds = Counter()
    gap_seconds = Counter()
    listing = []
    for record in records:
        if "error" in record:
            continue
        for block in blocks(record, max_gap):
            if not block["gaps"]:
                continue
            totals["blocks_with_gaps"] += 1
            totals["block_seconds"] += block["end_s"] - block["start_s"]
            for start, end in block["gaps"]:
                kind, why = attribute(record, start, end)
                kinds[kind] += 1
                gap_seconds[kind] += end - start
                totals["gap_seconds"] += end - start
                reasons.update(why)
                listing.append(
                    {
                        "shot": record["shot"],
                        "start_s": start,
                        "end_s": end,
                        "kind": kind,
                        "reasons": why,
                    }
                )
    return totals, reasons, kinds, gap_seconds, listing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--max-gap", type=float, default=0.5)
    parser.add_argument("--population", action="store_true")
    parser.add_argument("--example", type=int, default=192148)
    args = parser.parse_args()
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    if args.population:
        paths = sorted((args.work / "shots").glob("*.json"))
    else:
        paths = [args.work / "shots" / f"{int(s)}.json" for s in cohort.shot]
    records = [json.loads(p.read_text()) for p in paths if p.exists()]
    for record in records:
        record.pop("candidates", None)
    totals, reasons, kinds, gap_seconds, listing = summarize(records, args.max_gap)
    # Share of total observable time and of block time.
    observable = sum(
        r["state_seconds"][s]
        for r in records
        if "error" not in r
        for s in (
            "present",
            "absent",
            "q_prior_ece_contradicted",
            "q_prior_untested",
            "uncertain",
        )
    )
    example = [item for item in listing if item["shot"] == args.example]
    example_record = next((r for r in records if r["shot"] == args.example), None)
    result = {
        "definition": __doc__.strip(),
        "max_gap_s": args.max_gap,
        "scope": "population records" if args.population else "cohort records",
        "records": len(records),
        "blocks_with_uncertain_gaps": totals["blocks_with_gaps"],
        "block_seconds": totals["block_seconds"],
        "uncertain_gap_seconds": totals["gap_seconds"],
        "gap_share_of_block_time": (
            totals["gap_seconds"] / totals["block_seconds"]
            if totals["block_seconds"]
            else None
        ),
        "gap_share_of_observable_time": totals["gap_seconds"] / observable
        if observable
        else None,
        "gaps": sum(kinds.values()),
        "gap_kinds": dict(kinds),
        "gap_seconds_by_kind": dict(gap_seconds),
        "uncertain_crash_reasons": dict(reasons),
        "example": {
            "shot": args.example,
            "gaps": example,
            "uncertain_s": sum(i["end_s"] - i["start_s"] for i in example),
            "state_seconds": example_record["state_seconds"]
            if example_record
            else None,
        },
    }
    save_json(OUTPUT / "uncertain_gaps.json", result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("definition", "example")}
        )
    )


if __name__ == "__main__":
    main()
