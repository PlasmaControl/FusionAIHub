#!/usr/bin/env python
"""Measure reviewed non-crowd starts minus nearest saved ELM-O BES onset.

The population is every non-crowd span on the canonical 73-shot BES subset,
including spans too short to supply an occupancy bin. Each reviewed start is
matched independently within +/-50 ms; an ELM-O onset may match several starts.
This describes annotation timing on a matched subset, not physical onset error.
The measurement reads reviewed starts and cached BES detections directly.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from labeler.config import Paths, git_sha
from labeler.elm import labels, prepare

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/labeler/elm/review_start_offsets.json"
TOLERANCE_MS = 50.0


def source_record(path: Path) -> dict:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> int:
    paths = Paths.from_env()
    canonical = REPO / "outputs/labeler/elm/dsm/evaluation.json"
    shots = json.loads(canonical.read_text())["sets"]["bes73"]["shots"]
    assert len(shots) == 73 and len(set(shots)) == 73
    cohort_path = paths.catalog / "cohort.csv"
    cohort = pd.read_csv(cohort_path).set_index("shot")
    assert not any(cohort.loc[shot, "split"] == "test" for shot in shots)
    review_path = prepare.review_csv(paths)
    review = labels.review_table(review_path)
    review = review[(review.shot.isin(shots)) & (review.kind == "non_crowd")]
    onset_path = paths.root / "benchmarks/elm/elmo/review_elms.csv"
    detections = pd.read_csv(onset_path)
    detections = detections[
        (detections.variant == "paper") & detections.shot.isin(shots)
    ]
    rows = []
    for shot in sorted(shots):
        onsets = np.sort(
            detections.loc[detections.shot == shot, "t_start_ms"].to_numpy(float)
        )
        for span in review[review.shot == shot].itertuples():
            start = float(span.t_start)
            nearest = (
                float(onsets[np.argmin(np.abs(onsets - start))])
                if len(onsets)
                else None
            )
            delta = start - nearest if nearest is not None else None
            matched = delta is not None and abs(delta) <= TOLERANCE_MS
            rows.append(
                {
                    "shot": int(shot),
                    "review_start_ms": start,
                    "review_end_ms": float(span.t_end),
                    "nearest_elmo_onset_ms": nearest,
                    "matched_within_50ms": matched,
                    "review_minus_elmo_ms": delta if matched else None,
                }
            )
    matched = [row for row in rows if row["matched_within_50ms"]]
    delta = np.array([row["review_minus_elmo_ms"] for row in matched])
    assert len(delta) and np.isfinite(delta).all()
    p25, median, p75 = np.quantile(delta, [0.25, 0.5, 0.75])
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_sha(),
        "script": source_record(Path(__file__)),
        "sources": {
            "canonical_bes73": source_record(canonical),
            "reviewed_spans": source_record(review_path),
            "elmo_onsets": source_record(onset_path),
            "cohort": source_record(cohort_path),
        },
        "protocol": {
            "population": "All category-1 iscrowd-0 reviewed spans on bes73; "
            "not restricted to spans contributing whole 50 ms occupancy bins.",
            "elmo_setting": "paper",
            "elmo_onset_column": "t_start_ms",
            "difference": "review_start_ms minus nearest_elmo_onset_ms",
            "tolerance_ms": TOLERANCE_MS,
            "matching": "Independent nearest onset per reviewed start in the same "
            "shot; inclusive tolerance; earlier onset wins exact-distance ties; "
            "many reviewed starts may match the same ELM-O onset.",
            "quantiles": "Unweighted matched-span quantiles, linear interpolation.",
            "limitation": "Describes the matched annotation subset relative to "
            "ELM-O BES detections; unmatched starts omitted from timing quantiles; "
            "neither reference establishes independently verified physical onsets.",
        },
        "shots": [int(shot) for shot in sorted(shots)],
        "n_shots": len(shots),
        "n_shots_with_non_crowd_spans": int(review.shot.nunique()),
        "n_starts": len(rows),
        "n_matched": len(matched),
        "n_unmatched": len(rows) - len(matched),
        "n_matched_shots": len({row["shot"] for row in matched}),
        "n_unique_matched_elmo_onsets": len(
            {(row["shot"], row["nearest_elmo_onset_ms"]) for row in matched}
        ),
        "offset_ms": {
            "median": float(median),
            "p25": float(p25),
            "p75": float(p75),
            "iqr": float(p75 - p25),
            "min": float(delta.min()),
            "max": float(delta.max()),
        },
        "per_start": rows,
    }
    OUT.write_text(json.dumps(record, indent=1) + "\n")
    print(
        f"Matched {len(matched)}/{len(rows)} starts on "
        f"{record['n_matched_shots']} shots; median {median:.3f} ms, "
        f"IQR [{p25:.3f}, {p75:.3f}] ms (width {p75 - p25:.3f} ms)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
