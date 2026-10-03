"""Read-only old-rule rerun and four-state, per-shot independent validation."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sawtooth_physics import (
    OUTPUT,
    REPO,
    REVIEW,
    WORK,
    frozen_rule,
    records_at,
    save_json,
)

from labeler.config import Paths
from labeler.events.heuristics import sawtooth_events
from labeler.sawtooth.metrics import (
    aggregate,
    bin_times,
    event_cells,
    masked_interval_cells,
    point_metrics,
    presence_metrics,
    score_histogram,
    spans_at,
)
from labeler.sawtooth.physics import trains
from labeler.sawtooth.preprocessing import mask_spans


def old_shot(job):
    shot, work, window = job
    out = Path(work) / "old_rule" / f"{shot}.json"
    if out.exists():
        return json.loads(out.read_text())
    record = {"shot": int(shot), "crashes": [], "intervals": []}
    try:
        with h5py.File(Paths.from_env().corpus_file(shot), "r", locking=False) as h:
            g = h["ece"]
            # Original rule receives unmodified native ECE, not the new FIR reader.
            t = np.asarray(g["xdata"], dtype=float)
            keep = (t >= window[0]) & (t <= window[1])
            index = np.flatnonzero(keep)
            if len(index) < 32:
                raise ValueError("insufficient ECE native samples")
            y = np.asarray(g["ydata"][:, index[0] : index[-1] + 1])
            t = t[keep]
            found = sawtooth_events(y, t, shot=shot, t_cov=window)
        times = [e.t0_s for e in found if window[0] <= e.t0_s <= window[1]]
        record.update(
            crashes=[{"time_s": float(t)} for t in times],
            intervals=[
                {"start_s": times[a], "end_s": times[b - 1]}
                for a, b in trains(times, frozen_rule(work))
            ],
            window_s=list(window),
        )
    except (OSError, ValueError, KeyError, IndexError) as e:
        record["error"] = f"{type(e).__name__}: {e}"
    save_json(out, record)
    return record


def points(record):
    return np.array([r["time_s"] for r in record["crashes"]])


def intervals(record):
    return [(r["start_s"], r["end_s"]) for r in record["intervals"]]


def expert_score(record, old, rows):
    bins = bin_times(record["window_s"])
    positive = [
        (r.t_start / 1000, r.t_end / 1000) for r in rows.itertuples() if r.category == 1
    ]
    known = [
        (r.t_start / 1000, r.t_end / 1000)
        for r in rows.itertuples()
        if r.category in (0, 1)
    ]
    observable = spans_at(bins, record["observable_spans"]) & spans_at(bins, known)
    assessed = observable & spans_at(bins, record["assessed_spans"])
    truth = spans_at(bins, positive)
    answer = {
        "shot": record["shot"],
        "observable_known_bins": int(observable.sum()),
        "assessed_bins": int(assessed.sum()),
        "uncertain_bins": int((observable & ~assessed).sum()),
        "expert_positive_observable_bins": int((truth & observable).sum()),
        "expert_positive_uncertain_bins": int((truth & observable & ~assessed).sum()),
        "assessment_fraction": float(assessed.sum() / max(1, observable.sum())),
    }
    for name, rec in [("new", record), ("old", old)]:
        prediction = spans_at(bins, intervals(rec))
        hist = score_histogram(truth[assessed], prediction[assessed].astype(float))
        all_hist = score_histogram(
            truth[observable], prediction[observable].astype(float)
        )
        # bin_times returns centers; construct half-open 2ms cells from left edges.
        supports = mask_spans(bins - 0.001, assessed) if len(bins) else []
        observable_supports = mask_spans(bins - 0.001, observable) if len(bins) else []
        cells = masked_interval_cells(positive, intervals(rec), supports, 0.1)
        observable_cells = masked_interval_cells(
            positive, intervals(rec), observable_supports, 0.1
        )
        pts = points(rec)
        covered = spans_at(pts, record["observable_spans"]) & spans_at(pts, known)
        supported = covered & spans_at(pts, positive)
        conditional_covered = covered & spans_at(pts, record["assessed_spans"])
        conditional_supported = conditional_covered & spans_at(pts, positive)
        answer[name] = {
            "presence": presence_metrics(all_hist),
            "conditional_assessed_presence": presence_metrics(hist),
            "observable_presence_including_abstentions": presence_metrics(all_hist),
            "span_matching": {
                "cells": observable_cells.astype(int).tolist(),
                "metrics": point_metrics(observable_cells),
                "minimum_iou": 0.1,
                "support": "expert-known & observable 2ms cells",
                "annotation_identity": "original spans; zero-support spans excluded",
            },
            "conditional_assessed_span_matching": {
                "cells": cells.astype(int).tolist(),
                "metrics": point_metrics(cells),
                "minimum_iou": 0.1,
                "support": "expert-known & observable & algorithm-assessed 2ms cells",
                "annotation_identity": "original spans; zero-support spans excluded",
            },
            "observable_span_matching_including_abstentions": {
                "cells": observable_cells.astype(int).tolist(),
                "metrics": point_metrics(observable_cells),
                "minimum_iou": 0.1,
                "support": "expert-known & observable 2ms cells",
                "annotation_identity": "original spans; zero-support spans excluded",
            },
            "observable_positive_supported_picks": int(supported.sum()),
            "observable_known_picks": int(covered.sum()),
            "conditional_assessed_positive_supported_picks": int(
                conditional_supported.sum()
            ),
            "conditional_assessed_known_picks": int(conditional_covered.sum()),
            "true_crash_f1": None,
            "histogram": hist.tolist(),
        }
    # New four-state predictions retain possible positives instead of calling them absent.
    uncertain = spans_at(
        bins,
        [(r["start_s"], r["end_s"]) for r in record.get("uncertain_intervals", [])],
    )
    answer["physical_train_or_uncertain_positive_support"] = {
        "bins": int(
            (truth & observable & (spans_at(bins, intervals(record)) | uncertain)).sum()
        ),
        "denominator": int((truth & observable).sum()),
    }
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["old", "validate"])
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    review = pd.read_csv(REVIEW)
    shots = sorted(set(cohort.shot) | set(review.shot))
    windows = {
        int(r.shot): (r.window_start_ms / 1000, r.window_end_ms / 1000)
        for r in cohort.itertuples()
    }
    for s in set(review.shot) - set(windows):
        rows = review[review.shot == s]
        windows[s] = (rows.t_start.min() / 1000, rows.t_end.max() / 1000)
    if args.stage == "old":
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            old = []
            for i, rec in enumerate(
                pool.map(old_shot, [(s, str(args.work), windows[s]) for s in shots])
            ):
                old.append(rec)
                if i % 50 == 0:
                    print(
                        f"old rule {i + 1}/{len(shots)} shot{rec['shot']}", flush=True
                    )
        save_json(
            OUTPUT / "old_rule_run.json",
            {
                "requested_shots": shots,
                "requested_count": len(shots),
                "completed_shots": [r["shot"] for r in old if "error" not in r],
                "errors": {r["shot"]: r["error"] for r in old if "error" in r},
                "crashes": sum(len(r["crashes"]) for r in old),
                "rule": "labeler.events.heuristics.sawtooth_events, unchanged defaults",
                "input": "native read-only corpus ECE; original 1ms envelope",
                "span_adapter": "same frozen period/train grouping, no new amplitude/q guards",
            },
        )
        return
    new = {r["shot"]: r for r in records_at(args.work, shots)}
    old = {
        s: json.loads((args.work / "old_rule" / f"{s}.json").read_text()) for s in shots
    }
    agreement = []
    by_shot = []
    failures = {}
    for shot in shots:
        rec, legacy = new[shot], old[shot]
        if "error" in rec or "error" in legacy:
            failures[shot] = {"new": rec.get("error"), "old": legacy.get("error")}
            continue
        t = bin_times(rec["window_s"])
        mask = spans_at(t, rec["assessed_spans"])
        a, b = points(legacy), points(rec)
        a = a[spans_at(a, rec["assessed_spans"])]
        b = b[spans_at(b, rec["assessed_spans"])]
        cells = event_cells(a, b, 2)
        hist = score_histogram(
            spans_at(t, intervals(legacy))[mask],
            spans_at(t, intervals(rec))[mask].astype(float),
        )
        agreement.append({"shot": shot, "cells": cells, "histogram": hist})
        by_shot.append(
            {
                "shot": shot,
                "old_crashes": len(a),
                "new_crashes": len(b),
                "observable_bins": int(spans_at(t, rec["observable_spans"]).sum()),
                "assessed_bins": int(mask.sum()),
                "cells_2ms": cells.astype(int).tolist(),
                "crash": point_metrics(cells),
                "presence": presence_metrics(hist),
            }
        )
    experts = [
        expert_score(new[s], old[s], review[review.shot == s])
        for s in sorted(review.shot.unique())
        if "error" not in new[s] and "error" not in old[s]
    ]
    for r in experts:
        for name in ["old", "new"]:
            r[name].pop("histogram")
    summary = {
        "expert": {
            "by_shot": experts,
            "bootstrap": False,
            "mask": "primary: identical expert-known & core-observable for both rules and models; conditional: also exclude new algorithm uncertainty for both rules",
            "observable_abstention_policy": "Only explicit present spans are positive decisions; uncertainty remains a four-state label and counts as no positive decision in unconditional expert sensitivity. Coverage and conditional assessed metrics are reported separately.",
            "limitation": "Only span annotations supplied; true crash scores unavailable",
        },
        "legacy_agreement": aggregate(agreement),
        "agreement_by_shot": by_shot,
        "agreement_shot_count": len(agreement),
        "failed_shots": failures,
        "bin_ms": 2,
        "crash_tolerance_ms": 2,
        "rule_frozen_before_validation": True,
        "review_csv": str(REVIEW),
    }
    save_json(OUTPUT / "validation.json", summary)
    references = []
    for shot in [141182, 141195]:
        exists = Paths.from_env().corpus_file(shot).exists()
        result = {
            "shot": shot,
            "corpus_present": exists,
            "detected_period_ms": None,
            "detected_amplitude": None,
            "detected_crash_time_ms": None,
            "published_period_ms": [85, 5],
            "published_amplitude": [0.35, 0.02],
            "published_crash_ms": 2837.1 if shot == 141182 else None,
        }
        if exists:
            raise RuntimeError(
                "reference shot became available; run detector and extract published windows before reporting"
            )
        result["status"] = "unavailable; no fetching authorized"
        references.append(result)
    save_json(
        OUTPUT / "muscatello_reference.json",
        {"by_shot": references, "digest": "Muscatello_ST.md"},
    )
    print({"agreement_shots": len(agreement), "experts": experts}, flush=True)


if __name__ == "__main__":
    main()
