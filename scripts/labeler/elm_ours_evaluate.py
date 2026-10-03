#!/usr/bin/env python
"""Score `elm-ours` against the reviewed ELM spans, beside ELM-O and the ELM clock.

    python scripts/labeler/elm_ours_evaluate.py --run cv2 [--out-dir DIR]

`elm-ours` (`labeler.elm`) is a 1D U-Net over filterscope D-alpha and line density.
Its predictions here are the out-of-fold ones of `python -m labeler.elm.train`: every
reviewed shot is predicted by the model of the fold it sat in, trained on other
shots, with the decision threshold chosen on that fold's
inner-validation shots. No cohort test shot is read.

**Bins.** The benchmark's own: 50 ms bins wholly inside one absent, non-crowd or
crowd span and inside analysed time (`labeler.elm.labels.scored_bins`, the rule of
`scripts/labeler/elmo_benchmark.bin_table`). Two sets:

* `bes73`: the 73 reviewed shots ELM-O runs on (it needs BES), analysed time = ELM-O's
  chunks (`review_coverage.csv`); `elm-ours`, ELM-O (the published setting) and the
  ELM clock are scored on identical bins. ELM-O's counts are checked against the
  benchmark's published ones.
* `all119`: every reviewed shot, analysed time = where the fetched D-alpha and
  density records have samples; `elm-ours` and the ELM clock (ELM-O cannot run).

**Scores.** Per bin: precision, recall, F1, false-alarm bin rate (hard calls: an
`elm-ours` bin is called present when its mean event probability reaches the fold's
threshold; ELM-O and the clock call a bin when a detected span touches it), AUROC and
AUPRC of the continuous score (ELM-O's from its threshold sweep). Per kind: the
share of crowd bins called, and the share of non-crowd present spans a method
touches (for `elm-ours` the stretches where the same 50 ms mean reaches the same
threshold are its detected spans), plus raw and 25 ms edge-guarded absent-span
touch rates. The unsupported auxiliary onset head is excluded from paper outputs.
95 % intervals are percentile intervals over 1000 shot
resamples, the same resamples for every method on a set, so differences are paired.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from labeler.config import Paths, git_sha
from labeler.elm import compare, labels, methods, score, swap, train

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "labeler" / "elm" / "ours"
PUBLISHED_ELMO = {"tp": 2322, "fp": 443, "fn": 428, "tn": 3650}
NAMES = compare.NAME


def evaluate_set(
    name,
    shots,
    data,
    bins_of,
    cover_of,
    oof,
    elmo_spans,
    clock_spans,
    boot,
    elmo_sweep=None,
    clock_points=None,
) -> dict:
    parts: dict[str, list[score.ShotScore]] = {"ours": [], "clock": []}
    if elmo_spans is not None:
        parts["elmo"] = []
    for shot in shots:
        d, bins, cover = data[shot], bins_of[shot], cover_of[shot]
        parts["ours"].append(
            methods.trace_part(
                d.spans, shot, bins, cover, oof.trace(shot)[0], oof.threshold[shot]
            )
        )
        c = clock_spans.get(shot, methods.span_frame([], []))
        parts["clock"].append(methods.span_part(d.spans, shot, bins, cover, c))
        if elmo_spans is not None:
            e = elmo_spans.get(shot, methods.span_frame([], []))
            part = methods.span_part(d.spans, shot, bins, cover, e)
            if elmo_sweep is not None:
                part.score = swap.sweep_bin_scores(
                    elmo_sweep[elmo_sweep.shot == shot], bins
                )
            parts["elmo"].append(part)
    out: dict = {
        "shots": [int(s) for s in shots],
        "n_shots": len(shots),
        "bins": int(sum(len(parts["ours"][i].truth) for i in range(len(shots)))),
    }
    named = {NAMES[m]: plist for m, plist in parts.items()}
    out.update(methods.summarise_methods(named, boot, NAMES["ours"]))
    if elmo_spans is not None:
        counts = out["methods"][NAMES["elmo"]]["counts"]
        out["elmo_counts_match_published"] = all(
            counts[k] == v for k, v in PUBLISHED_ELMO.items()
        )
        elmo_res = out["methods"][NAMES["elmo"]]
        elmo_res["score_source"] = "largest eta whose saved detections touch the bin"
    starts = contributing = non_crowd_bins = 0
    for shot in shots:
        spans = data[shot].spans
        bins = bins_of[shot]
        rows = spans[spans.kind == "non_crowd"]
        non_crowd_bins += int((bins.kind == "non_crowd").sum())
        for r in rows.itertuples():
            starts += int(
                ((r.t_start >= bins.t0) & (r.t_start < bins.t0 + labels.BIN_MS)).any()
            )
            contributing += int(
                ((bins.t0 >= r.t_start) & (bins.t0 + labels.BIN_MS <= r.t_end)).any()
            )
    out["non_crowd_bin_audit"] = {
        "scored_bins": non_crowd_bins,
        "spans_with_scored_bins": contributing,
        "span_starts_inside_scored_bins": starts,
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="a `labeler.elm.train` run name")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args(argv)
    paths = Paths.from_env()
    data = train.load(paths)
    oof = methods.Oof(paths.root / "round4" / "elm" / "cv" / args.run)
    shots_all = sorted(data)
    missing = [s for s in shots_all if s not in oof.fold_of]
    if missing:
        raise SystemExit(f"shots without out-of-fold predictions: {missing}")

    sets = compare.load_sets(paths, data)
    elmo_spans, clock_spans = compare.load_detected(paths)
    elmo_sweep = compare.load_elmo_sweep(paths)
    bes, full = sets["bes73"], sets["all119"]

    record = {
        "git": git_sha(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": args.run,
        "train_record": str(oof.dir / "run.json"),
        "reference": "expert-reviewed spans (review/labels.csv)",
        "cohort_test_shots_used": 0,
        "folds": oof.record["folds"],
        "event_thresholds": {
            r["fold"]: r["threshold"] for r in oof.record["fold_records"]
        },
        "config": oof.record["config"],
        "label_audit": labels.span_length_summary(
            pd.concat([data[s].spans for s in shots_all], ignore_index=True)
        ),
        "onset_head": "Dropped from paper outputs: reviewed span starts are not "
        "verified physical ELM onsets and the auxiliary head has weak agreement. "
        "Existing checkpoints retain it; no onset retraining was performed.",
        "span_alarm_definition": {
            "raw": "Any detected-span touch on a sufficiently analysed absent span.",
            "guard25": "Touch in [start+25ms,end-25ms), with the same raw denominator; "
            "empty interiors counted separately. A conditional nonempty-interior "
            "rate is also reported.",
            "mechanism": "The centered 50ms elm-ours moving mean can extend detected "
            "runs across annotation edges; short inter-ELM gaps are often touched.",
        },
        "sets": {},
    }
    reviewed = pd.concat([data[s].spans for s in shots_all], ignore_index=True)
    record["annotation_modes"] = {
        "non_crowd_present_shots": int(
            reviewed[reviewed.kind == "non_crowd"].shot.nunique()
        ),
        "crowd_present_shots": int(reviewed[reviewed.kind == "crowd"].shot.nunique()),
        "note": "Per-ELM annotations and whole ELMing-period crowds define different "
        "occupancy targets across shots; non-crowd spans are not verified onsets.",
    }
    boot_bes = score.draws(len(bes.shots))
    record["sets"]["bes73"] = evaluate_set(
        "bes73",
        bes.shots,
        data,
        bes.bins,
        bes.cover,
        oof,
        elmo_spans,
        clock_spans,
        boot_bes,
        elmo_sweep=elmo_sweep,
    )
    boot_all = score.draws(len(full.shots))
    record["sets"]["all119"] = evaluate_set(
        "all119",
        full.shots,
        data,
        full.bins,
        full.cover,
        oof,
        None,
        clock_spans,
        boot_all,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "evaluation.json").write_text(json.dumps(record, indent=1))
    for name, s in record["sets"].items():
        print(name, s["n_shots"], "shots", s["bins"], "bins")
        for m, res in s["methods"].items():
            pt = res["point"]
            print(
                f"  {m:9s}",
                {k: round(v, 3) for k, v in pt.items() if k != "prevalence"},
            )
        if "elmo_counts_match_published" in s:
            print(
                "  elm-elmo counts match the published ones:",
                s["elmo_counts_match_published"],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
