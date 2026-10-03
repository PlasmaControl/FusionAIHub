"""Audit Figure 1 records, source hashes, print layout and optional rebuild.

Counts describe the displayed train/val shots, not classification performance.
All checks use recorded sources and pixel audits; no training or fetching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import fig_interpreter_tokeye as renderer
import numpy as np
from PIL import Image

from labeler.config import Paths, sha256_of
from labeler.paper import figure_sources as fs
from labeler.paper import label_figure as lf
from labeler.paper import mode_tags as mt
from labeler.paper.figure_sources import AE_THRESHOLD

SHOTS = (201978, 201973, 203187, 186636, 191376, 191782)


def rebuild_primary(record):
    """Rebuild with the recorded sources and verify identical PDF/PNG bytes."""
    files = [Path(p) for p in record["drawn"]["figure"]]
    before = {str(p): sha256_of(p) for p in files}
    rebuild_dir = Path(os.environ["TMPDIR"]) / "audit7-primary-rebuild"
    rebuild_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "pixi", "run", "--frozen", "--no-install", "--manifest-path",
        "/scratch/gpfs/nc1514/FusionAIHub/pyproject.toml", "-e", "labelmaker",
        "python", "scripts/labeler/paper/fig_interpreter_tokeye.py",
        "--shot", "201978", "--tmin", str(record["window_ms"][0]),
        "--tmax", str(record["window_ms"][1]), "--out", str(rebuild_dir),
    ]  # fmt: skip
    ae = record["ae_ours_lookup"]["supplied_predictions"]
    if ae:
        cmd.extend(["--ae-labels", ae])
    source = record["drawn"]["sawtooth_crashes"]["source"]
    if Path(source).exists():
        cmd.extend(["--sawtooth-source", source])
    evidence = record["drawn"]["sawtooth_crashes"].get("evidence_source")
    if evidence:
        cmd.extend(["--sawtooth-evidence", evidence])
    subprocess.run(cmd, check=True)
    rebuilt = [rebuild_dir / p.name for p in files]
    after = {str(p): sha256_of(q) for p, q in zip(files, rebuilt, strict=True)}
    assert before == after, "primary rebuild changed PDF/PNG bytes"
    return {
        "command": cmd,
        "before": before,
        "after": after,
        "identical": True,
        "rebuild_files": [str(p) for p in rebuilt],
    }


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
    manifest_file = args.records / "sawtooth_source_manifest.json"
    manifest = json.loads(manifest_file.read_text())
    assert manifest["source_commit"] == "ad0ca40f1a7e85c22a88ffcf71ba7ee7456c13e2"
    complete = Path(manifest["completion_snapshot_path"])
    assert sha256_of(complete) == manifest["completion_sha256"]
    completion = json.loads(complete.read_text())
    assert set(completion["requested_shots"]) == set(completion["processed_shots"])
    assert not completion["errors"]
    snapshot_hashes = {}
    for source in manifest["files"]:
        path = source["snapshot_path"]
        assert sha256_of(Path(path)) == source["sha256"]
        snapshot_hashes[path] = source["sha256"]
    catalog_comparison = manifest["catalog_comparison"]
    for source in catalog_comparison:
        assert sha256_of(Path(source["path"])) == source["sha256"]
        spec = next(s for s in lf.TRACKS if s.key == mt.SAWTOOTH)
        raw = lf.read_rows(Path(source["path"]))
        for shot in SHOTS:
            record = json.loads((args.records / f"{shot}.json").read_text())
            assert source["shots"][str(shot)] == fs.state_intervals(
                lf.Track(spec, rows=raw.get(shot, ())), record["window_ms"]
            )
    audited, checked_sources, labels = [], {}, set()
    render_commits = set()
    for shot in SHOTS:
        file = args.records / f"{shot}.json"
        record = json.loads(file.read_text())
        assert record["render_started_from_clean_head"]
        assert record["split"] in ("train", "val"), f"blind shot {shot}"
        drawn = record["drawn"]
        render_commits.add(record["git"])
        for path, digest in record["render_code_sha256"].items():
            committed = subprocess.check_output(
                ["git", "show", f"{record['git']}:{path}"]
            )
            assert hashlib.sha256(committed).hexdigest() == digest
            assert sha256_of(Path(path)) == digest
        for path, digest in drawn["figure_sha256"].items():
            assert sha256_of(Path(path)) == digest, f"changed figure {path}"
        for band in drawn["projection_audit"].values():
            for event in band.values():
                assert event["outside_present"] == event["outside_band"] == 0
        for band in drawn["ntm_measured_pixel_audit"].values():
            assert band["outside_measured_n"] == band["above_n_view_band"] == 0
            assert band["measured_n3_outline_pixels"] == 0
        assert drawn["ae_boxes_ms_khz"] == []
        crashes = drawn["sawtooth_crashes"]
        shown = crashes["drawn_times_ms"]
        assert not set(shown) & set(crashes["rejected_elm_times_ms"])
        assert set(shown) <= set(crashes["ece_times_ms"])
        assert drawn["sawtooth_strip_shown"] == bool(shown)
        if shot == 201978:
            assert record["window_ms"] == [1500, 3300]
        assert drawn["catalog_sawtooth_frame_model_shown"] is False
        saw = record["tracks"]["sawtooth_oscillation"]
        assert saw["state_intervals_ms"]
        for source in crashes["files"]:
            assert snapshot_hashes[source["path"]] == source["sha256"]
        spec = next(s for s in lf.TRACKS if s.key == mt.SAWTOOTH)
        review = spec.sources[0].locate(Paths.from_env())
        reviewed = lf.read_rows(review).get(shot)
        if reviewed:
            assert saw["path"] == str(review) and saw["tier"] == lf.SILVER
            expected = fs.state_intervals(
                lf.Track(spec, rows=reviewed), record["window_ms"]
            )
            assert saw["state_intervals_ms"] == expected
        else:
            assert snapshot_hashes[saw["path"]] == saw["sha256"]
        if Path(saw["path"]).suffix == ".json":
            physics = json.loads(Path(saw["path"]).read_text())
            expected = []
            for row in physics["states"]:
                a = max(row["start_s"] * 1000, record["window_ms"][0])
                b = min(row["end_s"] * 1000, record["window_ms"][1])
                if b > a:
                    expected.append(
                        {
                            "start_ms": a,
                            "end_ms": b,
                            "state": row["state"],
                            "category": {
                                "absent": 0,
                                "present": 1,
                                "uncertain": 2,
                                "unassessed": 3,
                            }[row["state"]],
                        }
                    )
            assert saw["state_intervals_ms"] == expected
            assert saw["density_guard"] == physics.get("density_guard")
        present_saw = any(r["category"] == 1 for r in saw["state_intervals_ms"])
        assert drawn["sawtooth_track_shown"] == present_saw
        spec = next(s for s in lf.TRACKS if s.key == mt.SAWTOOTH)
        raw = lf.Track(
            spec,
            rows=tuple(
                lf.Row(r["start_ms"], r["end_ms"], r["category"])
                for r in saw["state_intervals_ms"]
            ),
        )
        display, changes = fs.sawtooth_display(raw, record["window_ms"])
        assert saw["display_intervals_ms"] == (
            fs.state_intervals(display, record["window_ms"]) if present_saw else []
        )
        assert saw["display_merge"]["changes"] == (changes if present_saw else [])
        assert saw["display_merge"]["minimum_duration_ms"] == 10
        vermillion = "#D55E00"
        saw_present_drawn = any(r["category"] == 1 for r in saw["display_intervals_ms"])
        assert (
            vermillion in drawn["layout"]["present_chip_colours"]
        ) == saw_present_drawn
        confine = record["tracks"]["confinement"]
        mapping = {int(k): v for k, v in confine["states"].items()}
        for row in confine["state_intervals_ms"]:
            category = row["category"]
            state = mapping.get(category, "absent" if category == 0 else str(category))
            if confine["title"] == "H-mode" and category == 3:
                state = "unassessed"
            assert row["state"] == state
        geometry = drawn["layout"]
        assert geometry["n_panel_height_units"] >= 0.8
        assert geometry["n_panel_height_in"] >= 0.8
        assert geometry["n_panel_band_khz"] == [0, 30]
        assert geometry["processed_omitted_band_khz"] == []
        assert geometry["processed_restored_strip_khz"] == [30, 55]
        panels = geometry["frequency_panels"]
        for part, limits in (("hi", [55, 250]), ("mid", [30, 55]), ("lo", [0, 30])):
            raw_panel, processed = panels[f"raw_{part}"], panels[f"pr_{part}"]
            assert raw_panel["band_khz"] == processed["band_khz"] == limits
            assert abs(raw_panel["height_in"] - processed["height_in"]) < 1e-9
            assert raw_panel["ticks_khz"] == processed["ticks_khz"]
        assert 55 in panels["pr_mid"]["ticks_khz"]
        assert not geometry["ae_fixed_callout_shown"]
        for text in geometry["heading_and_legend_text_bounds"]:
            assert text["font_pt"] >= 7
            x0, y0, x1, y1 = text["bounds"]
            assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, text["text"]
        legend = [label.replace("\n", " ") for label in geometry["legend_labels"]]
        tags = drawn["blobs"]["tagged"]
        assert ("AE (detector band ≥80 kHz)" in legend) == bool(tags[mt.AE])
        ntm_key = (
            "NTM suggestions"
            if record["tracks"][mt.NTM]["tier"] == lf.GENERATED
            else "NTM labels"
        )
        assert (f"{ntm_key} (n=1 or 2, ≤30 kHz)" in legend) == bool(tags[mt.NTM])
        assert geometry["ntm_key_black_swatch"] == bool(tags[mt.NTM])
        png = Path(drawn["figure"][1])
        with Image.open(png) as native:
            assert all(abs(dpi - 150) < 0.1 for dpi in native.info["dpi"])
            rgb = np.asarray(native.convert("RGB"))
        raster_ae = fs.raster_ae_audit(
            rgb,
            panels["pr_hi"]["bounds"],
            record["window_ms"],
            panels["pr_hi"]["band_khz"],
            record["tracks"][mt.AE]["present_spans_ms"],
        )
        assert raster_ae["outside_present"] == raster_ae["below_detector_band"] == 0
        assert bool(raster_ae["pink_pixels"]) == bool(tags[mt.AE])
        if confine["title"] == "regime":
            assert set(drawn["regimes_shown"]) <= set(legend)
        elm_expert = record["tracks"]["edge_localized_mode"]["tier"] == lf.SILVER
        crowd = bool(drawn["elm_crowd_spans_ms"])
        assert any("expert ELM" in name for name in legend) == (elm_expert and crowd)
        late = drawn["late_untagged_high_frequency"]
        if late:
            assert late["pixels"] > 0
            assert "no duration cutoff" in late["rule"]
        harmonic = drawn["harmonic_support"]
        if harmonic["joint_columns"]:
            assert harmonic["passing_fraction"] == (
                harmonic["passing_columns"] / harmonic["joint_columns"]
            )
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
                "n=1/2",
                "second harmonic",
                "ECE-verified",
            )
        )
        assert "no present time" not in caption.lower()
        expected_saw = fs.sawtooth_caption(saw).removesuffix(".")
        assert expected_saw in caption
        if not present_saw:
            assert "; row omitted" in caption
        if shown:
            assert "ECE-supported crash candidates" in caption
            assert "channel-order geometry" in caption
        if drawn["first_large_peak_before_expert_ms"] is not None:
            peak = int(np.floor(drawn["largest_dalpha_peak_ms"] + 0.5))
            start = drawn["expert_elm_start_ms"]
            assert (
                f"The largest D-alpha spike ({peak} ms) precedes the expert ELM "
                f"interval (from {start:.0f} ms)" in caption
            )
        if drawn["lmode_inferred"]:
            assert "L-mode (inferred)" in caption
        if drawn["elm_hmode_conflicts_ms"]:
            assert "Expert ELM intervals overlap H-mode-detector absent time" in caption
        ntm = record["tracks"][mt.NTM]
        if ntm["tier"] == lf.GENERATED and tags[mt.NTM]:
            assert fs.ntm_description(ntm) in caption
            evaluation = json.loads(Path(ntm["performance"]["evaluation"]).read_text())
            assert ntm["performance"]["f1"] == evaluation["scores"]["ntm_frames"]["f1"]
        if shot == 201978:
            assert "neural detector on CO2 interferometer data" in caption
            assert (
                "Evenly spaced magnetics-only lines after 2.8 s remain unlabelled"
                in caption
            )
        if drawn.get("ae_physical_review_caveat"):
            assert drawn["ae_physical_review_caveat"] in caption
        if shot in (191376, 191782):
            assert record["publication_suitability"]["suitable_alternate"] is False
        label = re.search(r"\\label\{([^}]+)\}", caption)[1]
        assert label not in labels
        assert label == f"fig:interpreter-{shot}"
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
                if track.get("performance"):
                    sources.append(
                        {
                            "path": track["performance"]["evaluation"],
                            "sha256": track["performance"]["evaluation_sha256"],
                        }
                    )
        sources.append(record["tokeye"])
        # The checkpoint key differs from ordinary source records.
        sources[-1] = {
            "path": sources[-1]["checkpoint"],
            "sha256": sources[-1]["sha256"],
        }
        tokeye = record["tokeye"]
        sources.append({"path": tokeye["cache"], "sha256": tokeye["cache_sha256"]})
        fingerprints = fs.tokeye_fingerprints(
            Paths.from_env(),
            shot,
            renderer.roster.GATE_GROUP,
            renderer.roster.GATE_ROW,
            renderer.inspect.getsource(renderer.run_tokeye),
        )
        assert tokeye["fingerprints"] == fingerprints
        with np.load(tokeye["cache"]) as cache:
            assert json.loads(str(cache["fingerprints"])) == fingerprints
            assert str(cache["checkpoint_sha256"]) == tokeye["sha256"]
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
                "raster_ae_audit": raster_ae,
                "unmeasured_ntm_pixels": 0,
                "measured_n3_outline_pixels": 0,
                "layout": geometry,
                "regimes_shown": drawn["regimes_shown"],
                "harmonic_support": drawn["harmonic_support"],
                "tokeye_cache": {
                    "path": tokeye["cache"],
                    "sha256": tokeye["cache_sha256"],
                    "fingerprints": fingerprints,
                },
                "first_large_peak_before_expert_ms": drawn[
                    "first_large_peak_before_expert_ms"
                ],
                "largest_dalpha_peak_ms": drawn["largest_dalpha_peak_ms"],
                "expert_elm_start_ms": drawn["expert_elm_start_ms"],
                "lmode_inferred": drawn["lmode_inferred"],
                "ae_physical_review_caveat": drawn.get("ae_physical_review_caveat"),
                "ntm_performance": ntm["performance"],
                "publication_suitability": record["publication_suitability"],
                "elm_hmode_conflicts_ms": drawn["elm_hmode_conflicts_ms"],
                "render_source_commit": record["git"],
                "sawtooth_source": crashes["files"],
                "sawtooth_interval_source": {
                    "path": saw["path"],
                    "sha256": saw["sha256"],
                    "tier": saw["tier"],
                },
                "sawtooth_states": saw["state_intervals_ms"],
                "sawtooth_track_shown": present_saw,
                "sawtooth_display_intervals_ms": saw["display_intervals_ms"],
                "sawtooth_display_merge": saw["display_merge"],
                "confinement_intervals_ms": confine["state_intervals_ms"],
                "sawtooth_state_duration_ms": {
                    state: sum(
                        r["end_ms"] - r["start_ms"]
                        for r in saw["state_intervals_ms"]
                        if r["state"] == state
                    )
                    for state in ("present", "absent", "uncertain", "unassessed")
                },
                "sawtooth_density_proxy": saw["density_guard"],
                "late_untagged_high_frequency": drawn["late_untagged_high_frequency"],
                "pdf_size_in": [width, height],
                "figure_sha256": {p: sha256_of(Path(p)) for p in drawn["figure"]},
            }
        )
    primary = json.loads((args.records / "201978.json").read_text())
    assert len(render_commits) == 1, "all six renders must use the same source commit"
    reproducibility = rebuild_primary(primary) if args.rebuild_primary else None
    for shot in SHOTS:
        file = args.records / f"{shot}.json"
        record = json.loads(file.read_text())
        external = Path(record["caption"]["path"]).parent / "fig_interpreter.json"
        assert external.read_bytes() == file.read_bytes()
    args.out.write_text(
        json.dumps(
            {
                "primary_shot": 201978,
                "ae_threshold": AE_THRESHOLD,
                "ae_threshold_source": "scripts/labeler/ae_baselines_evaluate.py, SELDnet",
                "renders": audited,
                "checked_sources": checked_sources,
                "blind_test_shots_used": 0,
                "sawtooth_source_manifest": {
                    "path": str(manifest_file),
                    "sha256": sha256_of(manifest_file),
                    "original_source": manifest["original_source"],
                    "snapshot_source": manifest["snapshot_source"],
                    "completion_sha256": manifest["completion_sha256"],
                    "source_commit": manifest["source_commit"],
                    "validation_status": manifest["validation_status"],
                    "catalog_comparison": catalog_comparison,
                },
                "reproducibility": reproducibility,
            },
            indent=1,
        )
        + "\n"
    )
    print(
        f"Audited {len(audited)} non-blind renders; zero array/raster AE clipping violations"
    )
    if reproducibility:
        print("Primary PDF and PNG rebuild identically byte-for-byte")


if __name__ == "__main__":
    main()
