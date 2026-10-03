"""Audit the six Figure 1 records and emit a train/val-only ranked shortlist.

Descriptive counts are reproducible from committed render records. Ranking
uses the controller/reviewers' visual criteria, not a fitted quality score.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from labeler.config import sha256_of

SHOTS = (201978, 201973, 203187, 186636, 191376, 191782)
RANKING = {
    201978: "Clearest AE cascade and persistent n=1/n=2 low-frequency mode; "
    "local corpus CO2 supports paper-model inference, D-alpha and NBI. "
    "Retain for AE plus low-frequency structure; no verified sawtooth survives.",
    201973: "Strongest visual alternate: clear AE cascade, low-frequency mode, "
    "D-alpha and NBI with paper-model CO2 inference; current physics export "
    "does not supply verified sawtooth crashes in the window.",
    203187: "AE and low-frequency structure with local CO2/D-alpha/NBI; "
    "cascade is less distinct than the first two, and no verified crash survives.",
    186636: "Reviewed ELM intervals and imported NTM archive improve provenance; "
    "weaker AE structure, earlier interferometer frame detector, no definite "
    "confinement shading or verified sawtooth crash in this window.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("outputs/labeler/paper/fig_interpreter_tokeye"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", default="e5a56995")
    args = parser.parse_args()
    audited = []
    for shot in SHOTS:
        file = args.records / f"{shot}.json"
        record = json.loads(file.read_text())
        baseline_file = (
            Path("outputs/labeler/paper/fig_interpreter_tokeye") / f"{shot}.json"
        )
        previous = json.loads(
            subprocess.check_output(
                ["git", "show", f"{args.baseline}:{baseline_file}"], text=True
            )
        )
        assert record["split"] in ("train", "val"), f"blind shot {shot}"
        drawn = record["drawn"]
        for band in drawn["projection_audit"].values():
            for event in band.values():
                assert event["outside_present"] == event["outside_band"] == 0
        crashes = drawn["sawtooth_crashes"]
        shown = crashes["drawn_times_ms"]
        assert not set(shown) & set(crashes["rejected_elm_times_ms"])
        assert set(shown) <= set(crashes["ece_times_ms"])
        assert drawn["sawtooth_strip_shown"] == bool(shown)
        for t in drawn["elm_peak_times_ms"]:
            assert not any(a <= t < b for a, b in drawn["elm_uncertain_spans_ms"])
        caption_file = args.records / f"{shot}.caption.tex"
        caption = caption_file.read_text()
        words = len(caption.split())
        assert words <= 150
        assert not any(
            s in caption
            for s in (
                "ntm_frames",
                "sawtooth_frames",
                "dalpha_lh",
                "MPI66M",
                "PRESENT",
            )
        )
        assert sha256_of(caption_file) == record["caption"]["sha256"]
        layout = record["print_layout"]
        assert layout["width_in"] == 6.75 and layout["minimum_font_pt"] >= 7
        assert record["decision_thresholds"] == {
            "ae": 0.7,
            "ntm": 0.63,
            "sawtooth": 0.6,
            "tokeye": 0.2,
            "sawtooth_elm_veto_ms": 5.0,
            "ece_crash_match_ms": 3.0,
        }
        audited.append(
            {
                "shot": shot,
                "split": record["split"],
                "window_ms": record["window_ms"],
                "record": str(file),
                "record_sha256": sha256_of(file),
                "ae_source": record["tracks"]["alfven_eigenmode"]["what"],
                "tag_counts": drawn["blobs"]["tagged"],
                "verified_crashes": len(shown),
                "elm_peak_count": drawn["elm_peaks_in_label"],
                "caption_words": words,
                "projection_violations": 0,
                "regimes_shown": drawn["regimes_shown"],
                "baseline_commit": args.baseline,
                "baseline_crash_ticks": len(
                    previous["drawn"]["sawtooth_crashes"]["drawn_times_ms"]
                ),
            }
        )
    shortlist = [
        {
            "rank": i + 1,
            **next(r for r in audited if r["shot"] == shot),
            "reason": reason,
        }
        for i, (shot, reason) in enumerate(RANKING.items())
    ]
    args.out.write_text(
        json.dumps(
            {
                "primary_shot": 201978,
                "selection_rule": "retain primary if AE and low-frequency mode persist; "
                "verified sawtooth preferred, never required by fabrication",
                "ranked_shortlist": shortlist,
                "renders": audited,
                "blind_test_shots_used": 0,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"Audited {len(audited)} non-blind renders; zero projection violations")


if __name__ == "__main__":
    main()
