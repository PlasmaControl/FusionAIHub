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
    tiers = labels.tier.value_counts().to_dict()
    sha = record["labels_sha256"]
    lines = [
        "# Detachment data interface for detach-ui",
        "",
        (
            "Exploratory labels; **no independent benchmark**. "
            f"{len(labels):,} assessed bins/{labels.shot.nunique()} shots; "
            f"{int(certain.sum()):,} certain bins/"
            f"{labels.loc[certain, 'shot'].nunique()} shots/"
            f"{certain.sum() * 0.05:.2f} s. State bins: {states}. "
            f"Tier counts: {tiers}. Labels sha256 {sha}. "
            "A certain attached or detached bin has a TangTV vote plus a compatible "
            "vote from Afrac or Prad,div and no conflicting vote; a certain MARFE "
            "is the TangTV MARFE vote (spatial cue, density cue, persistence). "
            "Agreement with the divertor Thomson temperature, an independent "
            "measurement, is in docs/labeler/results/detachment_te_check.json."
        ),
        "",
        "## Stable label interface",
        "",
        (
            "Codes: absent=0 internally, attached=1, detached=2, MARFE=3, "
            "uncertain=4. Missing rows mean unassessed. Confidence is null. "
            "state_lm aliases state_rule; state_model_diagnostic is vestigial. "
            "`tier` says why a bin is not certain: conflict, insufficient_support, "
            "low_confidence_pair, no_vote, candidate_marfe (a TangTV MARFE "
            "candidate without the full evidence), lower_shelf_window, "
            "geometry_unknown, elm_unknown. Temporal suggestions do not promote "
            "certainty."
        ),
        "",
        (
            "- Lower-shelf TangTV window: the owner's restricted lower-shelf "
            "extraction is NOT a label column. It is absent from the interval "
            "table, the per-shot grids, labels_bins.csv.gz and indicators/*.csv "
            "(the earlier state_lower_shelf_window column was removed), and "
            "its bins carry state=4 and tier=lower_shelf_window. Only the "
            "per-indicator vote and value columns of those bins remain, for "
            "reference."
        ),
        (
            "- Worktree intervals: data/events/detachment/extend_detach_vote/"
            "detach_shots.csv; columns shot,category,t_start,t_end,confidence,attrs "
            "(JSON tier). Grids: extend_detach_vote/detach_shots/<shot>.npz."
        ),
        (
            "- Full bins: round4/detach/labels_bins.csv.gz. labels_rule.csv stays "
            "stable; labels_label_model.csv is diagnostic. `state_rule` uses the "
            "absolute Prad,div cutoffs anchored on 201081; "
            "`state_rule_relative_prad` and `tier_relative_prad` are the same rule "
            "with the per-shot relative f_div votes (`prad_rel_value`, "
            "`prad_rel_vote`), kept as a sensitivity column."
        ),
        (
            "- indicators/<shot>.csv retains t_ms, state, tier, afrac, prad_div "
            "(MW), prad_fraction, tangtv_dz, tangtv_front_height, validity, and the "
            "Afrac probe provenance (aux_jsat_selected_probe/_r_m/_z_m/_psin, "
            "strike position, distance to the outer strike, radial margin, "
            "afrac_probe_n_eligible). Times are centers; NPZ start_ms and interval "
            "boundaries are bin starts."
        ),
        (
            "- Afrac probe: the peak-Jsat probe among those on the scrape-off side "
            "(psiN > 1.000, at least 5 mm outboard of the outer strike, psiN <= "
            "1.05). The provenance columns are exported for invalid bins too; "
            "afrac_probe_n_eligible=0 means no probe passed the flux gate."
        ),
        (
            "- bins/<shot>.npz keys: afrac_*, prad_*, tangtv_* (value, valid, "
            "reason, vote), aux_p_in_w (centered 250 ms input power: neutral "
            "beams + EFIT ohmic + ECH; the beams come from PTDATA BMSPINJ where "
            "the corpus pinj group is a stub), aux_prad_divl_w and aux_prad_tot_w "
            "(centered 250 ms inter-ELM radiation means), "
            "aux_prad_div_fraction_total (diagnostic ratio, not a vote), "
            "aux_prad_divl_native_w, aux_prad_elm_window_known, prad_averaging_ms, "
            "prad_rel_value/prad_rel_vote, afrac_probe_n_eligible, "
            "tangtv_marfe_candidate/_spatial/_second_cue/_back_transition."
        ),
        (
            "- D-alpha NaNs remain unavailable. An uncovered 50 ms window is "
            "elm_unknown, as is an uncovered 250 ms radiation averaging window; "
            "uncovered heating windows are no_input_power. "
            "Native-bin or 250 ms radiation means below -0.05 MW are "
            "negative_radiation; this is a local offset tolerance, not a "
            "calibrated uncertainty."
        ),
        (
            "- MIN_VALID_BINS=20 is an explicit eligibility deviation from "
            "exporting every two-measurement shot: require >=20 assessed bins and "
            ">=20 valid bins per contributing indicator. Narrow valid snippets "
            "remain in bins/<shot>.npz, but have no exported label row."
        ),
        (
            "- dts/<shot>.npz: processed divertor Thomson Te (te in eV, te_err, "
            "ne as stored, chord r/z in m, t_ms in ms); an npz with "
            "no `te` key records an absent node. The independent check selects "
            "the chords 1.5-5 cm above the shelf with psiN in (1.000, 1.05]."
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
        "## Figure 2 record",
        "",
        (
            "docs/labeler/figure2_detach.json is the one canonical Figure 2 record "
            "(the duplicate detachment_figure2.json was removed). "
            "Detachment has no independent benchmark; the JSON says so in "
            "`independent_benchmark` and `scope`. "
            f"Schema name: `{schema.get('schema')}`. "
            f"Top-level keys: {', '.join(schema)}."
        ),
        (
            "`agreement` is a list of rows {key, label, kind, chance, value, ci95, "
            "n_bins, n_shots, drawn, undefined_reason}: kind=threshold_free rows "
            "(AUROC and Spearman of f_div against the TangTV vote, pooled and "
            "within shot, shot-bootstrap 95% intervals) and kind=kappa rows. A "
            "kappa on a table where either rater used one class is null with "
            "drawn=false and must not be plotted. `coverage` holds the "
            "populations (bins, shots, seconds, shot_ids) for assessed and certain "
            "bins, by state, by TangTV tier and by split. `coverage_table` gives, "
            "per indicator, how many bins have a measurement, a valid "
            "measurement, a vote and a certain label. `definitions` states each "
            "term. sources.labels_sha256 links the generation."
        ),
        "",
        (
            "Agreement record paths: records/agreement.json and the benchmark "
            "JSON hold by_tangtv_tier.<tier>.<pair> (bins, agreement, kappa); "
            "threshold_free_agreement.<tier>.<pair> holds the AUROC/Spearman "
            "entries; paper_agreement is the upper-shelf Prad-TangTV block."
        ),
        (
            "Figure paths: round4/detach/figure/fig_detachment_{views,timeline,"
            "marfe_witness,figure2}.{pdf,png}. The views and timeline figures "
            "draw shot 201081 (two Te cliffs); detach_figure.py picks time-ordered "
            "columns from sustained intervals of at least five bins. The "
            "bolometer row shows PRAD_DIVL/PRAD_TOT traces because no 2D "
            "bolometer emissivity exists "
            "(docs/labeler/results/detachment_bolometer_availability.json). "
            "Rebuild detach-ui from these outputs."
        ),
    ]
    (ROOT / "HANDOFF.md").write_text("\n".join(lines) + "\n")
    print(f"{len(record['shots'])} per-shot diagnostic availability rows")


if __name__ == "__main__":
    main()
