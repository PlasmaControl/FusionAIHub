#!/usr/bin/env python
"""Write the stable detach-ui contract and per-shot diagnostic availability."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import detach_label as dl
import numpy as np
import pandas as pd
from detach_json import dumps

ROOT = Path(os.environ["LABELER_ROOT"]) / "round4/detach"
RESULT = dl.REPO / "docs/labeler/results/detachment_availability.json"


def availability(shots):
    from scipy.io import readsav

    survey_path = ROOT / "survey/corpus_survey.csv"
    survey = pd.read_csv(survey_path).set_index("shot")
    rows = []
    for shot in sorted(shots):
        row = survey.loc[shot] if shot in survey.index else pd.Series(dtype=float)
        rec = {"shot": shot}
        for name in ("tangtv", "bolo"):
            count = row.get(f"n_{name}", 0)
            count = int(count) if pd.notna(count) else 0
            rec[name] = {
                "path": f"/scratch/gpfs/EKOLEMEN/foundation_model/{shot}_processed.h5",
                "group": name,
                "time_samples": count,
                "usable_time_record": count > 1,
                "xdata_units": "seconds",
                "time_range_s": [
                    float(row.get(f"{name}_t{i}", np.nan)) for i in (0, 1)
                ],
            }
        live = row.get("bolo_live_channels", 0)
        rec["bolo"]["live_channels"] = int(live) if pd.notna(live) else 0
        inv = ROOT / "inversions" / f"{shot}.npz"
        rec["inversion"] = {"path": str(inv), "usable": inv.exists()}
        rec["raw_sav"] = {"usable": False, "candidates": []}
        if inv.exists():
            with np.load(inv) as f:
                times = f["times_ms"]
                source = Path(str(f["source"]))
                rec["inversion"].update(
                    frames=len(times),
                    time_range_ms=[float(times[0]), float(times[-1])],
                    frame_axes=["time", "Z", "R"],
                )
            candidates = [source, source.with_name(source.stem + "_raw.sav")]
            rec["raw_sav"]["candidates"] = [str(p) for p in candidates]
            for source in candidates:
                if not source.exists():
                    continue
                data = readsav(str(source))["emission_structure"][0]
                if "VID" not in data.dtype.names:
                    continue
                vt = np.asarray(data["VID_TIMES"], float)
                rec["raw_sav"].update(
                    usable=len(vt) > 1,
                    path=str(source),
                    frames=len(vt),
                    time_range_ms=[float(vt[0]), float(vt[-1])],
                    frame_axes=["time", "y", "x"],
                )
                break
        maps = ROOT / "efit" / f"{shot}.npz"
        rec["flux_map"] = {"path": str(maps), "usable": False}
        if maps.exists():
            with np.load(maps) as f:
                rec["flux_map"].update(
                    usable=len(f["gtime_ms"]) > 1,
                    source=str(f.get("source", "EFIT01")),
                    slices=len(f["gtime_ms"]),
                )
        rows.append(rec)
    return {"survey_source": str(survey_path), "shots": rows}


def main():
    labels_path = ROOT / "labels_bins.csv.gz"
    labels = pd.read_csv(labels_path)
    shots = set(labels.shot.astype(int)) | {
        int(p.stem) for p in (ROOT / "inversions").glob("*.npz")
    }
    record = availability(shots)
    record["labels_sha256"] = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    record["producer"] = "scripts/labeler/detach_handoff.py"
    RESULT.write_text(dumps(record, indent=1) + "\n")
    certain = labels.state_rule.isin((1, 2, 3))
    states = {
        dl.core.STATE_NAMES[s]: int(labels.state_rule.eq(s).sum()) for s in (1, 2, 3, 4)
    }
    lines = [
        "# Detachment data interface for detach-ui",
        "",
        (
            "Exploratory labels; **no independent benchmark**. "
            f"{len(labels):,} assessed bins/{labels.shot.nunique()} shots; "
            f"{int(certain.sum()):,} certain bins/"
            f"{labels.loc[certain, 'shot'].nunique()} shots/"
            f"{certain.sum() * 0.05:.2f} s. State bins: {states}. "
            "Certainty is upper-shelf TangTV with f_div corroboration, not "
            "independently validated physical truth. MARFE is a single-shot "
            "candidate class within threshold uncertainty."
        ),
        "",
        "## Stable label interface",
        "",
        (
            "Codes: absent=0 internally, attached=1, detached=2, MARFE=3, "
            "uncertain=4. Missing rows mean unassessed. Confidence is null. "
            "state_lm aliases state_rule; state_model_diagnostic is vestigial. "
            "Lower-shelf votes remain provisional: primary state=4, "
            "tier=lower_shelf_window. Temporal suggestions do not promote certainty."
        ),
        "",
        (
            "- Worktree intervals: data/events/detachment/extend_detach_vote/"
            "detach_shots.csv; columns shot,category,t_start,t_end,confidence,attrs "
            "(JSON tier). Grids: extend_detach_vote/detach_shots/<shot>.npz."
        ),
        (
            "- Full bins: round4/detach/labels_bins.csv.gz. labels_rule.csv stays "
            "stable; labels_label_model.csv is diagnostic."
        ),
        (
            "- indicators/<shot>.csv retains t_ms, state, tier, afrac, prad_div "
            "(MW), prad_fraction, tangtv_dz, tangtv_front_height and validity. "
            "Times are centers; NPZ start_ms and interval boundaries are bin starts."
        ),
        (
            "- bins/<shot>.npz retains all existing keys. aux_p_in_w now means "
            "centered 250 ms heating power; aux_prad_divl_w and aux_prad_tot_w "
            "are centered 250 ms inter-ELM radiation means. "
            "aux_prad_div_fraction_total is a diagnostic ratio, not another vote. "
            "aux_prad_divl_native_w retains the native label-bin radiation mean; "
            "aux_prad_elm_window_known records D-alpha availability across "
            "the full radiation averaging window; "
            "prad_averaging_ms=250 documents the window. "
            "tangtv_marfe_back_transition separates the H-L cue from fG."
        ),
        (
            "- D-alpha NaNs remain unavailable. An uncovered 50 ms window is "
            "elm_unknown, as is an uncovered 250 ms radiation averaging window; "
            "uncovered heating windows are no_input_power. "
            "Averaging windows include both endpoints in means and availability; "
            "native label bins are left-closed/right-open. "
            "Native-bin or 250 ms radiation means below -0.05 MW are "
            "negative_radiation; this is a "
            "local offset tolerance, not a calibrated uncertainty."
        ),
        (
            "- MIN_VALID_BINS=20 is an explicit eligibility deviation from "
            "exporting every two-measurement shot: require >=20 assessed bins and "
            ">=20 valid bins per contributing indicator. Narrow valid snippets "
            "remain in bins/<shot>.npz, but have no exported label row."
        ),
        "",
        "## Diagnostic sources and time bases",
        "",
        (
            "Corpus HDF5 xdata is seconds. tangtv ydata is "
            "(channel,time,240,720); channel 2 is lower perpendicular. "
            "The table below checks time length >1; camera exposure/brightness "
            "gates still apply. Bolo contains 48 chord voltages, not a spatial "
            "image; dead lower-fan chord 11 is masked in figures. "
            "inversions/<shot>.npz: frames(time,Z,R), times_ms, radii/elevation "
            "in metres. cache/<shot>.npz and geometry02/<shot>.npz use ms. "
            "efit/<shot>.npz uses gtime_ms and psirz(time,Z,R), with named EFIT "
            "source and wall/boundary. Raw SAV VID/VID_TIMES remains available "
            "through detach_figure.read_video when the owner's SAV contains it."
        ),
        "",
        f"Availability JSON: {RESULT}; every assessed or inversion shot is listed.",
        "",
        (
            "| Shot | TangTV corpus frames (s) | Bolo samples/live chords (s) | "
            "Inversion frames (ms) | Raw SAV frames (ms) | Flux source/slices |"
        ),
        "|---:|---|---|---|---|---|",
    ]
    for rec in record["shots"]:
        cells = []
        for name in ("tangtv", "bolo"):
            item = rec[name]
            if not item["usable_time_record"]:
                cells.append("absent/stub")
                continue
            a, b = item["time_range_s"]
            live = f"/{item['live_channels']}" if name == "bolo" else ""
            cells.append(f"{item['time_samples']}{live} ({a:.3f}–{b:.3f})")
        inv, maps = rec["inversion"], rec["flux_map"]
        iv = "absent"
        if inv["usable"]:
            a, b = inv["time_range_ms"]
            iv = f"{inv['frames']} ({a:.1f}–{b:.1f})"
        fv = f"{maps['source']}/{maps['slices']}" if maps["usable"] else "absent"
        raw = rec["raw_sav"]
        rv = "absent"
        if raw["usable"]:
            a, b = raw["time_range_ms"]
            rv = f"{raw['frames']} ({a:.1f}–{b:.1f})"
        lines.append(
            f"| {rec['shot']} | {cells[0]} | {cells[1]} | {iv} | {rv} | {fv} |"
        )
    panel = dl.REPO / "docs/labeler/figure2_detach.json"
    schema = json.loads(panel.read_text())
    lines += [
        "",
        "## Figure 2 schema change",
        "",
        (
            "docs/labeler/figure2_detach.json now provides coverage and upper-shelf "
            "Prad–TangTV agreement. F1-vs-consensus rows were removed. "
            "Do not pass it to the former detector-benchmark row renderer. "
            "Detachment has no independent benchmark; the JSON says so in "
            "`independent_benchmark` and `scope`. "
            f"Schema name: `{schema.get('schema')}`. "
            f"Top-level keys: {', '.join(schema)}."
        ),
        (
            "coverage.{assessed,certain}, coverage.by_state.<state>.{assessed,"
            "certain}, coverage.by_tangtv_tier.<tier> and coverage.by_split "
            "contain populations with bins, shots, seconds and shot_ids. "
            "indicator_agreement.by_tangtv_tier.<tier> gives both_valid and "
            "both_vote populations; confusion_counts has Prad rows/TangTV "
            "columns in attached/detached/MARFE order. metrics contains "
            "kappa_3class, kappa_binary, agreement_3class and agreement_binary "
            "with point, CI and valid_replicates. Binary agreement maps MARFE "
            "to detached. Conflicts retain attached-TangTV/detached-Prad counts. "
            "rows has kind=coverage (bins) or kind=agreement (kappa); draw "
            "these on separate axes. sources.labels_sha256 links the generation."
        ),
        "",
        (
            "Agreement record paths stay stable. records/agreement.json "
            "all_eligible_bins and fit_bins_train_val_outside now contain "
            "by_tangtv_tier.<tier>.<pair>, with top-level by_tangtv_tier and "
            "fit_by_tangtv_tier aliases. The benchmark JSON's corresponding "
            "keys contain <tier>.<pair> directly, with named by_tangtv_tier "
            "aliases. Cross-tier agreement scalars and circular benchmark "
            "tables were removed."
        ),
        (
            "Figure paths remain round4/detach/figure/fig_detachment_{views,"
            "timeline,figure2}.{pdf,png}; the MARFE witness adds "
            "fig_detachment_marfe_witness.{pdf,png}. Appendix panels use full "
            "6.75-inch width. Rebuild detach-ui from these outputs."
        ),
    ]
    (ROOT / "HANDOFF.md").write_text("\n".join(lines) + "\n")
    print(f"{len(record['shots'])} per-shot diagnostic availability rows")


if __name__ == "__main__":
    main()
