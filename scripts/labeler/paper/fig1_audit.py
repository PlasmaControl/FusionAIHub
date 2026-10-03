"""Audit Figure 1 records, source hashes, print layout and optional rebuild.

Counts describe the displayed train/val shots, not classification performance.
All checks use recorded sources and pixel audits; no training or fetching.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from labeler.config import sha256_of
from labeler.paper.figure_sources import AE_THRESHOLD

SHOTS = (201978, 201973, 203187, 186636, 191376, 191782)


def rebuild_primary(record):
    """Rebuild with the recorded sources and verify identical PDF/PNG bytes."""
    files = [Path(p) for p in record["drawn"]["figure"]]
    before = {str(p): sha256_of(p) for p in files}
    cmd = [
        "pixi", "run", "--frozen", "--no-install", "--manifest-path",
        "/scratch/gpfs/nc1514/FusionAIHub/pyproject.toml", "-e", "labelmaker",
        "python", "scripts/labeler/paper/fig_interpreter_tokeye.py",
        "--shot", "201978", "--tmin", str(record["window_ms"][0]),
        "--tmax", str(record["window_ms"][1]), "--out", str(files[0].parent),
    ]  # fmt: skip
    ae = record["ae_ours_lookup"]["supplied_predictions"]
    if ae:
        cmd.extend(["--ae-labels", ae])
    source = record["drawn"]["sawtooth_crashes"]["source"]
    if Path(source).exists():
        cmd.extend(["--sawtooth-source", source])
    subprocess.run(cmd, check=True)
    after = {str(p): sha256_of(p) for p in files}
    assert before == after, "primary rebuild changed PDF/PNG bytes"
    return {"command": cmd, "before": before, "after": after, "identical": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("outputs/labeler/paper/fig_interpreter_tokeye"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rebuild-primary", action="store_true")
    args = parser.parse_args()
    audited, checked_sources, labels = [], {}, set()
    for shot in SHOTS:
        file = args.records / f"{shot}.json"
        record = json.loads(file.read_text())
        assert record["split"] in ("train", "val"), f"blind shot {shot}"
        drawn = record["drawn"]
        for band in drawn["projection_audit"].values():
            for event in band.values():
                assert event["outside_present"] == event["outside_band"] == 0
        for band in drawn["ntm_measured_pixel_audit"].values():
            assert band["outside_measured_n"] == band["above_n_view_band"] == 0
        assert drawn["ae_boxes_ms_khz"] == []
        crashes = drawn["sawtooth_crashes"]
        shown = crashes["drawn_times_ms"]
        assert not set(shown) & set(crashes["rejected_elm_times_ms"])
        assert set(shown) <= set(crashes["ece_times_ms"])
        assert drawn["sawtooth_strip_shown"] == bool(shown)
        if shot == 201978:
            assert record["window_ms"] == [1500, 3300]
            assert not drawn["sawtooth_track_shown"]
        for t in drawn["elm_peak_times_ms"]:
            assert not any(a <= t < b for a, b in drawn["elm_uncertain_spans_ms"])
        caption_file = args.records / f"{shot}.caption.tex"
        caption = caption_file.read_text()
        # Count prose, excluding TeX wrapper and standalone math delimiters.
        prose = caption.split("\\label")[0].removeprefix("\\caption{").rstrip("}\n")
        words = len(prose.replace(r"$\geq$", "≥").replace("$", "").split())
        assert words <= 150
        assert not any(
            s in caption
            for s in (
                "ntm_frames",
                "sawtooth_frames",
                "dalpha_lh",
                "MPI66M",
                "PRESENT",
                "cand.",
            )
        )
        label = re.search(r"\\label\{([^}]+)\}", caption)[1]
        assert label not in labels
        assert (label == "fig:interpreter") == (shot == 201978)
        labels.add(label)
        assert sha256_of(caption_file) == record["caption"]["sha256"]
        layout = record["print_layout"]
        assert layout["width_in"] == 6.75 and layout["minimum_font_pt"] >= 7
        assert layout["height_in"] <= 5.5
        assert record["decision_thresholds"]["ae"] == AE_THRESHOLD
        external = Path(record["caption"]["path"]).parent / "fig_interpreter.json"
        assert external.read_bytes() == file.read_bytes()
        assert Path(record["caption"]["path"]).read_bytes() == caption_file.read_bytes()
        sources = [*crashes["files"], *drawn["stores"].values()]
        for track in record["tracks"].values():
            if track:
                sources.append({"path": track["path"], "sha256": track["sha256"]})
                if track.get("metadata"):
                    sources.append(
                        {
                            "path": track["metadata"],
                            "sha256": track["metadata_sha256"],
                        }
                    )
        sources.append(record["tokeye"])
        # The checkpoint key differs from ordinary source records.
        sources[-1] = {
            "path": sources[-1]["checkpoint"],
            "sha256": sources[-1]["sha256"],
        }
        for source in sources:
            if not source or not source.get("sha256"):
                continue
            path, digest = source["path"], source["sha256"]
            if path not in checked_sources:
                assert sha256_of(Path(path)) == digest, f"changed source {path}"
                checked_sources[path] = digest
            assert checked_sources[path] == digest
        pdf = Path(drawn["figure"][0])
        info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
        size = re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", info)
        width, height = (float(v) / 72 for v in size.groups())
        assert width == 6.75 and height <= 5.5
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
                "unmeasured_ntm_pixels": 0,
                "regimes_shown": drawn["regimes_shown"],
                "harmonic_support": drawn["harmonic_support"],
                "pdf_size_in": [width, height],
                "figure_sha256": {p: sha256_of(Path(p)) for p in drawn["figure"]},
            }
        )
    primary = json.loads((args.records / "201978.json").read_text())
    reproducibility = rebuild_primary(primary) if args.rebuild_primary else None
    args.out.write_text(
        json.dumps(
            {
                "primary_shot": 201978,
                "ae_threshold": AE_THRESHOLD,
                "ae_threshold_source": "scripts/labeler/ae_baselines_evaluate.py, SELDnet",
                "renders": audited,
                "checked_sources": checked_sources,
                "blind_test_shots_used": 0,
                "reproducibility": reproducibility,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"Audited {len(audited)} non-blind renders; zero projection violations")
    if reproducibility:
        print("Primary PDF and PNG rebuild identically byte-for-byte")


if __name__ == "__main__":
    main()
