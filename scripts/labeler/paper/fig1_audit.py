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
from itertools import pairwise
from pathlib import Path

import fig_interpreter_tokeye as renderer
import numpy as np
from PIL import Image

from labeler.ae.xpower.train import read_split
from labeler.config import Paths, sha256_of
from labeler.paper import figure_sources as fs
from labeler.paper import label_figure as lf
from labeler.paper import mode_tags as mt
from labeler.paper.figure_sources import AE_THRESHOLD

#: Figure 1 is one shot on one linear 0-250 kHz frequency axis. The earlier
#: alternates are not rendered on this axis; `--shots` audits any rendered set.
PRIMARY = 199563
SHOTS = (PRIMARY,)
AXIS_TICKS_KHZ = [0, 50, 100, 150, 200, 250]
#: Words that would describe a broken, stretched or compressed frequency axis.
BROKEN_AXIS_WORDS = ("stretched", "compressed", "scale break", "three frequency")


def rebuild_primary(record):
    """Rebuild with the recorded sources and verify identical PDF/PNG bytes."""
    files = [Path(p) for p in record["drawn"]["figure"]]
    before = {str(p): sha256_of(p) for p in files}
    rebuild_dir = Path(os.environ["TMPDIR"]) / "fig1b-primary-rebuild"
    rebuild_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "pixi", "run", "--frozen", "--no-install", "--manifest-path",
        "/scratch/gpfs/nc1514/FusionAIHub/pyproject.toml", "-e", "labelmaker",
        "python", "scripts/labeler/paper/fig_interpreter_tokeye.py",
        "--shot", str(record["shot"]), "--tmin", str(record["window_ms"][0]),
        "--tmax", str(record["window_ms"][1]), "--out", str(rebuild_dir),
    ]  # fmt: skip
    cmd.extend(["--annotations", record["annotations"]["path"]])
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
        default=Path("outputs/labeler/paper/fig_interpreter_tokeye/fig1b"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--shots", type=int, nargs="+", default=list(SHOTS), help="rendered shots"
    )
    parser.add_argument("--rebuild-primary", action="store_true")
    parser.add_argument(
        "--baseline-ref",
        help="git ref whose records must have identical scientific data",
    )
    args = parser.parse_args()
    rendered = tuple(args.shots)
    manifest_file = args.records / "sawtooth_source_manifest.json"
    manifest = json.loads(manifest_file.read_text())
    # The source is the saw stream's final export, pinned by the digest of its
    # SHA256SUMS (no stream worktree or commit is needed to check it).
    assert len(manifest["export_sums_sha256"]) == 64
    sums_file = Path(manifest["export_sums_path"])
    assert sha256_of(sums_file) == manifest["export_sums_sha256"]
    sums = dict(line.split()[::-1] for line in sums_file.read_text().splitlines())
    complete = Path(manifest["completion_snapshot_path"])
    assert sha256_of(complete) == manifest["completion_sha256"]
    completion = json.loads(complete.read_text())
    assert not set(map(str, rendered)) & set(completion["errors"])
    assert set(rendered) <= set(completion["processed_shots"])
    snapshot_hashes = {}
    for source in manifest["files"]:
        path = source["snapshot_path"]
        assert sha256_of(Path(path)) == source["sha256"]
        snapshot_hashes[path] = source["sha256"]
        # The drawn physics record equals the shot's rows in the hashed shard.
        shard = Path(source["population_shard_snapshot_path"])
        assert sha256_of(shard) == source["population_shard_sha256"]
        assert sums[shard.name] == source["population_shard_sha256"]
        assert source["states_equal_population_shard"] is True
    catalog_comparison = manifest["catalog_comparison"]
    for source in catalog_comparison:
        if source.get("missing"):
            continue
        assert sha256_of(Path(source["path"])) == source["sha256"]
        spec = next(s for s in lf.TRACKS if s.key == mt.SAWTOOTH)
        raw = lf.read_rows(Path(source["path"]))
        for shot in rendered:
            record = json.loads((args.records / f"{shot}.json").read_text())
            assert source["shots"][str(shot)] == fs.state_intervals(
                lf.Track(spec, rows=raw.get(shot, ())), record["window_ms"]
            )
    audited, checked_sources, labels = [], {}, set()
    render_commits = set()
    for shot in rendered:
        file = args.records / f"{shot}.json"
        record = json.loads(file.read_text())
        if args.baseline_ref:
            record_path = file.resolve().relative_to(Path.cwd())
            baseline = json.loads(
                subprocess.check_output(
                    ["git", "show", f"{args.baseline_ref}:{record_path}"]
                )
            )
            presentation = {
                "annotations",
                "appendix",
                "caption",
                "drawn",
                "git",
                "render_code_sha256",
                "render_started_from_clean_head",
                "tracks",
            }
            for key in record.keys() - presentation:
                new, old = record[key], baseline[key]
                if key == "filter":
                    # The display rule text gained the minimum outline size and
                    # the owner moved the AE display floor to 60 kHz; the frequency
                    # axis became linear (the old record names a fold, the new one
                    # the split and the axis); every other filtering parameter
                    # must match.
                    assert new["split_khz"] == 60.0, shot
                    assert new["frequency_axis"].startswith("linear 0-250 kHz"), shot
                    assert new["bands_khz"][mt.AE] == [60.0, None], shot
                    moved = {
                        "outline_display_rule",
                        "fold_khz",
                        "split_khz",
                        "frequency_axis",
                        "raw_normalisation",
                    }
                    new = {k: v for k, v in new.items() if k not in moved}
                    old = {k: v for k, v in old.items() if k not in moved}
                    for side in (new, old):
                        side["bands_khz"] = {
                            k: v for k, v in side["bands_khz"].items() if k != mt.AE
                        }
                if key == "detector_training":
                    # Round 11 records the frame AE detector's membership too:
                    # earlier entries are unchanged and only AE may be added.
                    assert set(new) - set(old) <= {mt.AE}, shot
                    new = {k: v for k, v in new.items() if k in old}
                assert new == old, (shot, "changed input", key)
            for key, track in record["tracks"].items():
                display = {"display_intervals_ms", "display_merge"}
                source = {k: v for k, v in track.items() if k not in display}
                old = {
                    k: v for k, v in baseline["tracks"][key].items() if k not in display
                }
                assert source == old, (shot, "changed source track", key)
            presentation = {
                "figure",
                "figure_sha256",
                "layout",
                "display_state_keys",
                "ntm_outline_display",
                "ntm_measured_pixel_audit",
            }
            # Records added after the baseline; each is checked below.
            added = {"harmonic3_support", "persistent_line_rows"}
            # The 60 kHz AE floor and split change what counts as AE: the tag
            # counts, the late untagged band and the projection audit.  The
            # harmonic gate is new.  Everything else must be unchanged.
            moved = {
                "blobs",
                "late_untagged_high_frequency",
                "projection_audit",
                "n2_harmonic_consistent",
                "n3_harmonic_consistent",
            }
            new_tags = record["drawn"]["blobs"]["tagged"]
            old_tags = baseline["drawn"]["blobs"]["tagged"]
            assert new_tags[mt.NTM] == old_tags[mt.NTM], (shot, "NTM tags changed")
            assert new_tags[mt.SAWTOOTH] == old_tags[mt.SAWTOOTH], shot
            assert new_tags[mt.AE] >= old_tags[mt.AE], (shot, "AE tags shrank")
            late = record["drawn"]["late_untagged_high_frequency"]
            if late:
                old_late = baseline["drawn"]["late_untagged_high_frequency"]
                assert 60 <= late["band_khz"][0] <= old_late["band_khz"][0], shot
                assert late["pixels"] >= old_late["pixels"], shot
            for key in record["drawn"].keys() - presentation - added - moved:
                assert record["drawn"][key] == baseline["drawn"][key], (
                    shot,
                    "changed scientific drawing record",
                    key,
                )
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
        if shot == PRIMARY:
            assert record["window_ms"] == [700, 5800]
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
            # The two q-prior states export as uncertain, with the state as reason.
            exported = {
                "absent": "absent",
                "present": "present",
                "uncertain": "uncertain",
                "q_prior_ece_contradicted": "uncertain",
                "q_prior_untested": "uncertain",
                "unassessed": "unassessed",
            }
            for row in physics["states"]:
                a = max(row["start_s"] * 1000, record["window_ms"][0])
                b = min(row["end_s"] * 1000, record["window_ms"][1])
                if b > a:
                    expected.append(
                        {
                            "start_ms": a,
                            "end_ms": b,
                            "state": exported[row["state"]],
                            "category": {
                                "absent": 0,
                                "present": 1,
                                "uncertain": 2,
                                "unassessed": 3,
                            }[exported[row["state"]]],
                        }
                    )
            assert saw["state_intervals_ms"] == expected
            assert saw["density_guard"] == physics.get("density_guard")
        assert drawn["sawtooth_track_shown"] is True
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
            fs.state_intervals(display, record["window_ms"])
        )
        assert saw["display_merge"]["changes"] == changes
        assert saw["display_merge"]["minimum_duration_ms"] == 0
        assert changes == []
        assert saw["display_intervals_ms"] == saw["state_intervals_ms"]
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
        # One linear 0-250 kHz axis in both spectrograms, with no scale break.
        axis = geometry["frequency_axis"]
        assert axis["scale"] == "linear" and axis["scale_breaks_khz"] == []
        assert axis["band_khz"] == [0, 250]
        assert axis["ticks_khz"] == AXIS_TICKS_KHZ
        assert axis["mask_zoom_pass_khz"] == [0, 50]
        assert axis["mask_wide_pass_khz"] == [50, 250]
        assert axis["ae_ntm_split_khz"] == 60 and axis["n_view_top_khz"] == 30
        # The n view is the bottom 30 of 250 kHz of the processed panel.
        panels = geometry["frequency_panels"]
        assert geometry["n_view_height_in"] >= 0.15
        assert (
            abs(
                geometry["n_view_height_in"]
                - panels["pr"]["height_in"]
                * axis["n_view_top_khz"]
                / axis["band_khz"][1]
            )
            < 1e-6
        )
        raw_panel, processed = panels["raw"], panels["pr"]
        assert raw_panel["band_khz"] == processed["band_khz"] == [0, 250]
        assert abs(raw_panel["height_in"] - processed["height_in"]) < 1e-9
        assert raw_panel["ticks_khz"] == processed["ticks_khz"] == AXIS_TICKS_KHZ
        assert raw_panel["bounds"][0] == processed["bounds"][0]
        assert raw_panel["bounds"][2] == processed["bounds"][2]
        for prefix in ("raw", "pr"):
            ticks = sorted(
                geometry["frequency_tick_bounds"][prefix],
                key=lambda tick: tick["bounds"][1],
            )
            assert len(ticks) == len(AXIS_TICKS_KHZ)
            for lower, upper in pairwise(ticks):
                a, b = lower["bounds"], upper["bounds"]
                assert b[1] - a[3] >= 1 / (72 * record["print_layout"]["height_in"]), (
                    shot,
                    prefix,
                    lower["text"],
                    upper["text"],
                )
        for label in geometry["regime_text_bounds"]:
            a, b = label["bounds"], label["region_x_bounds"]
            padding = 2 / (72 * record["print_layout"]["width_in"])
            assert b[0] + padding <= a[0] < a[2] <= b[1] - padding, (
                shot,
                label["text"],
            )
        assert "frequency_scale_breaks_khz" not in geometry
        for label in geometry["regime_text_bounds"]:
            for patch in geometry["elm_box_bounds"]:
                a, b = label["bounds"], patch
                assert min(a[2], b[2]) <= max(a[0], b[0]) or min(a[3], b[3]) <= max(
                    a[1], b[1]
                ), (shot, label["text"], "ELM box")
        if geometry["dalpha_peak_box_gap_pt"] is not None:
            assert geometry["dalpha_peak_box_gap_pt"] >= 1, (
                shot,
                "peak triangles touch box",
            )
        anchor = record["annotations"]["shot"].get("ae_label")
        if anchor:
            # A small "AE" chip inside the upper processed panel, no leader line.
            chip = geometry["ae_in_panel_label"]
            assert chip["text"] == "AE"
            assert chip["anchor_ms_khz"] == [anchor["time_ms"], anchor["frequency_khz"]]
            hi = panels["pr"]["bounds"]
            assert hi[0] < chip["bounds"][0] < chip["bounds"][2] < hi[2]
            assert hi[1] < chip["bounds"][1] < chip["bounds"][3] < hi[3]
            # The chip is in the AE band, above the n view and the 60 kHz split.
            assert anchor["frequency_khz"] >= mt.BANDS[mt.AE][0]
            assert "leader_anchor_ms_khz" not in geometry["ae_margin_label"]
            assert geometry["ae_margin_label"]["bounds"][0] > hi[2]
            spans = record["tracks"][mt.AE]["present_spans_ms"]
            assert any(a <= anchor["time_ms"] < b for a, b in spans)
        else:
            assert geometry["ae_in_panel_label"] is None
        n_key = geometry["n_key_bounds"]
        if n_key is not None:
            panel = panels["pr"]["bounds"]
            assert n_key[0] > panel[2]
            assert panel[1] <= n_key[1] < n_key[3] <= panel[3]
        sources_text = {t["track"]: t for t in geometry["track_source_text_bounds"]}
        panel_right = panels["pr"]["bounds"][2]
        for t in sources_text.values():
            x0, y0, x1, y1 = t["bounds"]
            assert panel_right < x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, t["text"]
        ordered = sorted(sources_text.values(), key=lambda t: t["bounds"][1])
        for lower, upper in pairwise(ordered):
            # Text boxes of neighbouring rows touch by under 2.5 px (descenders).
            assert lower["bounds"][3] <= upper["bounds"][1] + 0.003, (
                shot,
                lower["text"],
                upper["text"],
            )
        ntm_source = record["tracks"][mt.NTM]
        if ntm_source["tier"] == lf.GENERATED:
            assert sources_text[mt.NTM]["text"] == "detector (suggestions)"
        assert "\n" not in sources_text[mt.SAWTOOTH]["text"], (
            "sawtooth source on one line"
        )
        saw_row = sources_text[mt.SAWTOOTH]["text"]
        if saw["tier"] == lf.GENERATED:
            expected = fs.sawtooth_row_source(
                saw["display_intervals_ms"], saw["density_guard"]
            )
            assert saw_row == expected
        # No stretch/compression note: the axis needs none.
        assert "scale_note" not in geometry
        for text in geometry["heading_and_legend_text_bounds"]:
            flat = text["text"].replace("\n", " ").lower()
            assert not any(word in flat for word in BROKEN_AXIS_WORDS), text["text"]
        n_labels = [
            x for x in geometry["legend_labels"] if x.startswith("n=") or x == "other n"
        ]
        assert n_labels == [
            x for x in ("n=1", "n=2", "n=3", "other n") if x in n_labels
        ]
        for text in geometry["heading_and_legend_text_bounds"]:
            assert text["font_pt"] >= 7
            x0, y0, x1, y1 = text["bounds"]
            assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, text["text"]
        text_bounds = geometry["heading_and_legend_text_bounds"]
        for i, left in enumerate(text_bounds):
            for right in text_bounds[i + 1 :]:
                a, b = left["bounds"], right["bounds"]
                width = min(a[2], b[2]) - max(a[0], b[0])
                height = min(a[3], b[3]) - max(a[1], b[1])
                assert width <= 0 or height <= 0, (left["text"], right["text"])
        legend = [label.replace("\n", " ") for label in geometry["legend_labels"]]
        tags = drawn["blobs"]["tagged"]
        assert ("AE (detector-positive time; mask ≥60 kHz)" in legend) == bool(
            tags[mt.AE]
        )
        ntm_key = (
            "NTM candidate suggestions"
            if record["tracks"][mt.NTM]["tier"] == lf.GENERATED
            else "NTM labels"
        )
        assert (f"{ntm_key} (n=1 or 2, ≤30 kHz)" in legend) == bool(tags[mt.NTM])
        assert geometry["ntm_key_black_swatch"] == bool(tags[mt.NTM])
        png = Path(drawn["figure"][1])
        with Image.open(png) as native:
            assert all(abs(dpi - 150) < 0.1 for dpi in native.info["dpi"])
            rgb = np.asarray(native.convert("RGB"))
        # Read the processed panel above the n view (30-250 kHz): n hues are
        # excluded, but pink below the 60 kHz AE floor would still be caught.
        px0, py0, px1, py1 = panels["pr"]["bounds"]
        n_top, band_top = axis["n_view_top_khz"], axis["band_khz"][1]
        above_n_view = [px0, py0 + (py1 - py0) * n_top / band_top, px1, py1]
        raster_ae = fs.raster_ae_audit(
            rgb,
            above_n_view,
            record["window_ms"],
            [n_top, band_top],
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
        assert words <= fs.CAPTION_MAX_WORDS
        assert not any(
            s in caption
            for s in (
                "ntm_frames",
                "sawtooth_frames",
                "dalpha_lh",
                "MPI66M",
                "PRESENT",
                "cand.",
                "second harmonic",
                "ECE-verified",
                "below bar",
                "circularity",
                "four-state",
                "first ELM",
            )
        )
        assert "no present time" not in caption.lower()
        assert caption.startswith(f"\\caption{{DIII-D shot {shot}. Top: raw Mirnov ")
        assert "(linear frequency axis, 0--250 kHz), D-alpha, NBI power." in caption
        assert not any(w in caption.lower() for w in BROKEN_AXIS_WORDS)
        assert "normalised" not in caption
        assert "Middle: TokEye coherent-mode mask after small-object removal" in caption
        assert "toroidal mode number $n$ (Mirnov array)" in caption
        assert "Bottom: label tracks with sources." in caption
        assert "row omitted" not in caption and "shared inputs" not in caption
        assert "Circles:" not in caption and "Triangles:" not in caption
        appendix_file = args.records / f"{shot}.appendix.txt"
        appendix = appendix_file.read_text()
        assert appendix.strip() == fs.appendix_notes(
            shot, record["tracks"], drawn, record["detector_training"]
        )
        assert fs.sawtooth_caption(saw) in appendix
        thresholds = ["TokEye 0.2"]
        for key, name in ((mt.AE, "AE"), (mt.NTM, "NTM")):
            if record["tracks"][key]["tier"] == lf.GENERATED:
                value = record["tracks"][key]["decision_threshold"]
                thresholds.append(f"{name} {(value or AE_THRESHOLD):g}")
        assert ("Operating probability thresholds: " + "; ".join(thresholds)) in (
            appendix
        )
        # Detector-only thresholds: an imported NTM table has none to list.
        assert ("NTM 0.63" in appendix) == (
            record["tracks"][mt.NTM]["tier"] == lf.GENERATED
        )
        ae_ours = record["tracks"][mt.AE]["what"].startswith("ae-ours")
        ae_text = record["tracks"][mt.AE]["tier"] == lf.GENERATED and bool(tags[mt.AE])
        assert ("AE targets used TokEye's mask" in appendix) == (ae_ours and ae_text)
        if ae_text and not ae_ours:
            assert "trained on the owner's reviewed AE labels" in appendix
        assert ("input band is 80–250 kHz" in appendix) == ae_text
        assert ("mask pixels ≥60 kHz" in appendix) == ae_text
        assert ("trained on TokEye-mask-derived targets" in caption) == (
            ae_ours and bool(tags[mt.AE])
        )
        assert "not separate islands" not in appendix
        assert "not separate islands" not in caption
        for name, key in (("n2", "harmonic_support"), ("n3", "harmonic3_support")):
            assert drawn[f"{name}_harmonic_consistent"] == fs.harmonic_consistent(
                drawn[key]
            )
        # The caption names a harmonic ridge only above the passing-share gate.
        assert ("harmonic" in caption) == (
            bool(tags[mt.NTM]) and bool(fs.harmonic_clause(drawn))
        )
        assert (
            "The frequency axis is linear, 0–250 kHz, in both spectrograms, with no "
            "scale break." in appendix
        )
        assert "The raw spectrogram uses one colour scale." in appendix
        assert not any(w in appendix.lower() for w in BROKEN_AXIS_WORDS[:2])
        assert "persistent-row step" in appendix
        assert "not an identified pickup line" in appendix
        with np.load(record["tokeye"]["cache"]) as cache:
            for name in ("wide", "zoom"):
                share = np.asarray(cache[f"{name}_row_lit"])
                assert drawn["persistent_line_rows"][name] == int(
                    (share > mt.PERSISTENT_ROW_SHARE).sum()
                )
        display = drawn["ntm_outline_display"]
        assert display["min_px"] == renderer.NTM_OUTLINE_MIN_PX
        omitted = [
            size
            for band in display["bands"].values()
            for size in band.get("omitted_sizes_px", [])
        ]
        assert display["omitted_fragments"] == len(omitted)
        assert all(size < display["min_px"] for size in omitted)
        assert (
            f"fewer than {display['min_px']} print pixels are not drawn" in appendix
        ) == (record["tracks"][mt.NTM] is not None)
        assert "shorter than 10 ms, with no state smoothing" in appendix
        assert "magnetics-only" not in appendix
        assert "Bt unavailable" not in appendix
        if shown:
            assert "ECE-supported crash candidates" in appendix
        if drawn["first_large_peak_before_expert_ms"] is not None:
            assert "precedes the expert span" in appendix
        if drawn["elm_hmode_conflicts_ms"]:
            assert "sources disagree" in appendix
        ntm = record["tracks"][mt.NTM]
        if ntm["tier"] == lf.GENERATED and tags[mt.NTM]:
            assert fs.ntm_qualification(ntm) in caption
            assert "held-out F1" in caption
            evaluation = json.loads(Path(ntm["performance"]["evaluation"]).read_text())
            assert ntm["performance"]["f1"] == evaluation["scores"]["ntm_frames"]["f1"]
        # Every generated AE/NTM track records whether the shot was in the
        # detector's training set, from the training list itself.
        for key in (mt.AE, mt.NTM):
            if record["tracks"][key]["tier"] != lf.GENERATED:
                assert key not in record["detector_training"]
                continue
            training = record["detector_training"][key]
            source = Path(training["training_source"])
            assert sha256_of(source) == training["training_source_sha256"]
            if source.suffix == ".txt":
                shots = [int(x) for x in source.read_text().split()]
            elif source.suffix == ".csv":
                shots = sorted(s for s, v in read_split(source).items() if v == "train")
            else:
                shots = json.loads(source.read_text())["shots"]["train"]
            assert training["training_shots"] == shots
            assert training["figure_shot_in_training"] is (shot in shots)
            assert (
                sha256_of(Path(training["checkpoint"])) == training["checkpoint_sha256"]
            )
        if record["detector_training"]:
            assert fs.training_note(record["detector_training"]) in appendix
        if shot == PRIMARY:
            assert "CO2 neural detector" in appendix
            assert (
                "Pink: mask pixels $\\geq$60 kHz while the CO2 AE detector "
                "(80--250 kHz input band; trained on TokEye-mask-derived targets, "
                "so not independent of TokEye) is positive (25 ms bins)." in caption
            )
            assert "(held-out F1 0.46, below our 0.7 bar) is positive." in caption
            assert "Highlights mark time/band coincidence only." in caption
            # 19 % and 0 % of the jointly measured time pass the 5 % ratio test:
            # below the 0.6 gate, so the caption makes no harmonic claim.
            assert "harmonics" not in caption
            assert drawn["n2_harmonic_consistent"] is False
            assert drawn["n3_harmonic_consistent"] is False
            assert "in 38 of 198 ms where both are measured (19%" in appendix
            assert "in 0 of 139 ms where both are measured (0%" in appendix
            assert "cannot separate harmonics of one island from phase-locked" in (
                appendix
            )
            assert "linear frequency axis, 0--250 kHz" in caption
            # No row of this shot is an expert review, so no D-alpha spike
            # "precedes the expert span" and the ELM source is named a detector.
            assert drawn["first_large_peak_before_expert_ms"] is None
            assert "precedes the expert span" not in appendix
            assert "Detected ELM intervals and the H-mode detector disagree" in caption
            assert "Detected ELM intervals overlap H-mode-detector absent time" in (
                appendix
            )
            assert not any(
                t["tier"] == lf.SILVER for t in record["tracks"].values() if t
            )
            assert "Sawtooth: present 366 ms, uncertain 4195 ms, unassessed 539 ms" in (
                appendix
            )
            assert len(shown) == 12 and min(shown) > 5000
            assert "detector F1" not in " ".join(legend)
            for key in (mt.AE, mt.NTM):
                assert record["detector_training"][key]["figure_shot_in_training"] is (
                    False
                )
        assert (
            sha256_of(Path(record["annotations"]["path"]))
            == record["annotations"]["sha256"]
        )
        if drawn.get("ae_physical_review_caveat"):
            assert drawn["ae_physical_review_caveat"] in appendix
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
        assert (
            Path(record["appendix"]["path"]).read_bytes() == appendix_file.read_bytes()
        )
        assert sha256_of(appendix_file) == record["appendix"]["sha256"]
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
                "sawtooth_track_shown": drawn["sawtooth_track_shown"],
                "detector_training": record["detector_training"],
                "appendix": record["appendix"],
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
    primary = json.loads((args.records / f"{PRIMARY}.json").read_text())
    assert len(render_commits) == 1, "all renders must use the same source commit"
    reproducibility = rebuild_primary(primary) if args.rebuild_primary else None
    for shot in rendered:
        file = args.records / f"{shot}.json"
        record = json.loads(file.read_text())
        external = Path(record["caption"]["path"]).parent / "fig_interpreter.json"
        assert external.read_bytes() == file.read_bytes()
    args.out.write_text(
        json.dumps(
            {
                "primary_shot": PRIMARY,
                "ae_threshold": AE_THRESHOLD,
                "ae_threshold_source": "scripts/labeler/ae_baselines_evaluate.py, "
                "SELDnet",
                "renders": audited,
                "checked_sources": checked_sources,
                "blind_test_shots_used": 0,
                "sawtooth_source_manifest": {
                    "path": str(manifest_file),
                    "sha256": sha256_of(manifest_file),
                    "export_root": manifest["export_root"],
                    "export_sums_sha256": manifest["export_sums_sha256"],
                    "snapshot_source": manifest["snapshot_source"],
                    "completion_sha256": manifest["completion_sha256"],
                    "population_shards": {
                        str(f["shot"]): f["population_shard_sha256"]
                        for f in manifest["files"]
                    },
                    "validation_status": manifest["validation_status"],
                    "catalog_comparison": catalog_comparison,
                },
                "reproducibility": reproducibility,
                "scientific_data_unchanged_from": args.baseline_ref,
            },
            indent=1,
        )
        + "\n"
    )
    print(
        f"Audited {len(audited)} non-blind renders; "
        "zero array/raster AE clipping violations"
    )
    if reproducibility:
        print("Primary PDF and PNG rebuild identically byte-for-byte")


if __name__ == "__main__":
    main()
