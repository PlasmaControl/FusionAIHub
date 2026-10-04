#!/usr/bin/env python
"""Check regenerated detachment exports against their rule and source records."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import detach_label as dl
import numpy as np
import pandas as pd
from detach_json import dumps

from labeler.events.detachment import core, label_model, thresholds
from labeler.events.interval_tables import read_label_grid

ROOT = Path(os.environ["LABELER_ROOT"]) / "round4/detach"
RESULTS = dl.REPO / "docs/labeler/results"


def main():
    frame = pd.read_csv(ROOT / "labels_bins.csv.gz")
    votes, valid = dl.matrices(frame)
    rule, _ = label_model.compatibility_decide(
        votes,
        valid,
        tangtv_tier=frame.tangtv_tier.to_numpy(),
        elm_known=np.isfinite(frame.aux_elm_share),
    )
    assert np.array_equal(frame.state_rule, rule)
    assert np.array_equal(frame.state_lm, frame.state_rule)
    # the sensitivity column is the same rule with the per-shot relative f_div votes
    relative_votes = votes.copy()
    relative_votes[:, 1] = np.where(valid[:, 1], frame.prad_rel_vote, core.ABSTAIN)
    relative, _ = label_model.compatibility_decide(
        relative_votes,
        valid,
        tangtv_tier=frame.tangtv_tier.to_numpy(),
        elm_known=np.isfinite(frame.aux_elm_share),
    )
    assert np.array_equal(frame.state_rule_relative_prad, relative)
    # the lower-shelf column is not a label column and is not in any handoff file
    assert not [c for c in frame.columns if "lower_shelf" in c and c != "tangtv_tier"]
    cutoffs = thresholds.prad_cutoffs()
    prad_cast = frame.prad_valid.to_numpy()
    expected_prad = np.where(
        frame.prad_value <= cutoffs[0],
        core.ATTACHED,
        np.where(frame.prad_value >= cutoffs[1], core.DETACHED, core.ABSTAIN),
    )
    assert np.array_equal(frame.prad_vote[prad_cast], expected_prad[prad_cast])
    assert frame.confidence.isna().all()
    assert (valid.sum(axis=1) >= 2).all()
    assert (votes[~valid] == core.ABSTAIN).all()
    assert (
        frame.loc[frame.prad_valid, "aux_prad_divl_w"]
        .ge(-thresholds.RADIATION_NEGATIVE_TOL_W)
        .all()
    )
    assert (
        frame.loc[frame.prad_valid, "aux_prad_divl_native_w"]
        .ge(-thresholds.RADIATION_NEGATIVE_TOL_W)
        .all()
    )
    assert frame.loc[frame.prad_valid, "aux_p_in_w"].notna().all()
    assert frame.loc[frame.prad_valid, "aux_prad_elm_window_known"].all()
    assert frame.prad_averaging_ms.eq(250.0).all()
    certain = frame.state_rule.isin(core.VOTE_STATES)
    assert frame.loc[certain, "tangtv_tier"].eq("upper_shelf").all()
    # certain attached/detached: TangTV votes the state, one other valid indicator
    # votes it too, and no valid indicator votes against it
    for state in (core.ATTACHED, core.DETACHED):
        rows = frame.state_rule.eq(state)
        assert votes[rows, 2].tolist() == [state] * int(rows.sum())
        others = votes[rows][:, :2]
        assert ((others == state).any(axis=1)).all()
        assert not ((others > 0) & (others != state)).any()
    # certain MARFE: the TangTV MARFE vote, no attached vote elsewhere; Prad,div and
    # Afrac never corroborate it
    marfe = frame.state_rule.eq(core.MARFE)
    assert (frame.loc[marfe, "tangtv_vote"] == core.MARFE).all()
    assert not (votes[marfe][:, :2] == core.ATTACHED).any()
    assert frame.loc[marfe, ["tangtv_marfe_spatial"]].all().all()
    candidate = frame.tier.eq("candidate_marfe")
    assert frame.loc[candidate, "state_rule"].eq(core.UNCERTAIN).all()
    assert frame.loc[candidate, "tangtv_marfe_candidate"].all()
    assert np.isfinite(frame.loc[certain, "aux_elm_share"]).all()
    certain_by_split = {
        split: int((certain & frame.split.eq(split)).sum())
        for split in ("train", "val", "test", "outside")
    }
    assert (
        not frame.loc[frame.tier.eq("lower_shelf_window"), "state_rule"]
        .isin(core.VOTE_STATES)
        .any()
    )
    jsat = frame.afrac_vote > 0
    assert frame.loc[jsat, "afrac_probe_position_valid"].all()
    assert (
        frame.loc[jsat, "aux_jsat_selected_psin"] <= thresholds.PROBE_SOL_PSI_N_MAX
    ).all()
    assert (frame.loc[jsat, "afrac_probe_n_eligible"] >= 1).all()
    assert frame.loc[jsat, "afrac_method"].notna().all()
    assert (
        frame.loc[jsat, "aux_jsat_selected_psin"]
        >= thresholds.PROBE_SOL_PSI_N_MIN - 1e-6
    ).all()
    assert (
        frame.loc[jsat, "aux_jsat_radial_margin_m"]
        >= thresholds.PROBE_STRIKE_MARGIN_M - 1e-6
    ).all()
    table = pd.read_csv(dl.OUT / "detach_shots.csv")
    assert table.confidence.isna().all()
    primary_meta = json.loads((dl.OUT / "detach_shots.meta.json").read_text())
    diagnostic_meta = json.loads((ROOT / "labels_label_model.meta.json").read_text())
    assert primary_meta["primary_method"] == "compatibility rule with TangTV required"
    assert primary_meta["diagnostic_only"] is False
    assert diagnostic_meta["diagnostic_only"] is True
    assert "per_shot_files" not in diagnostic_meta
    reconstructed = []
    for row in table.itertuples():
        tier = json.loads(row.attrs)["tier"]
        reconstructed += [
            (row.shot, float(start), row.category, tier)
            for start in np.arange(row.t_start, row.t_end, core.BIN_MS)
        ]
    expected = list(
        frame[["shot", "start_ms", "state_rule", "tier"]].itertuples(
            index=False, name=None
        )
    )
    assert sorted(expected) == sorted(reconstructed)
    assert not frame[["shot", "start_ms"]].duplicated().any()
    shot_ids = set(frame.shot.astype(int))
    grid_dir = dl.OUT / "detach_shots"
    assert {int(p.stem) for p in grid_dir.glob("*.npz")} == shot_ids
    assert {int(p.stem) for p in (ROOT / "indicators").glob("*.csv")} == shot_ids
    trace_rows = 0
    for shot, rows in frame.groupby("shot"):
        grid = read_label_grid(grid_dir / f"{int(shot)}.npz")
        expected_grid = np.full(len(grid["time_ms"]), np.nan)
        expected_grid[np.rint(rows.start_ms / core.BIN_MS).astype(int)] = (
            rows.state_rule
        )
        expected_grid = np.broadcast_to(expected_grid[:, None], grid["label"].shape)
        assert np.array_equal(grid["label"], expected_grid, equal_nan=True)
        traces = pd.read_csv(ROOT / "indicators" / f"{int(shot)}.csv")
        assert np.array_equal(traces.state, rows.state_rule)
        assert np.array_equal(traces.t_ms, rows.start_ms + core.BIN_MS / 2)
        assert "aux_jsat_selected_probe" in traces
        assert "afrac_probe_n_eligible" in traces
        assert not [c for c in traces.columns if "lower_shelf" in c]
        trace_rows += len(traces)
    sha = hashlib.sha256((ROOT / "labels_bins.csv.gz").read_bytes()).hexdigest()
    reference = json.loads((RESULTS / "detachment_reference.json").read_text())
    assert reference["labels_record"]["sha256"] == sha
    figure = json.loads((ROOT / "figure/figure.json").read_text())
    assert figure["labels_sha256"] == sha
    panel = json.loads((dl.REPO / "docs/labeler/figure2_detach.json").read_text())
    assert panel["sources"]["labels_sha256"] == sha
    # one canonical Figure 2 record; the duplicate is gone
    assert not (RESULTS / "detachment_figure2.json").exists()
    for row in panel["agreement"]:
        if row["kind"] == "kappa" and row["value"] is None:
            assert not row["drawn"]
    current = json.loads((RESULTS / "detachment_current.json").read_text())
    assert current["labels_sha256"] == sha
    d9 = current["paper_criterion_d9"]
    states = frame.state_rule
    cohort = frame.split.isin(("train", "val", "test"))
    both = set(frame.loc[states.eq(core.ATTACHED), "shot"]) & set(
        frame.loc[states.eq(core.DETACHED), "shot"]
    )
    cohort_both = both & set(frame.loc[cohort, "shot"])
    assert sorted(both) == sorted(d9["shots_with_both"])
    assert sorted(cohort_both) == sorted(d9["cohort_shots_with_both"])
    assert d9["met"] == (len(both) >= 3 and len(cohort_both) >= 1)
    assert d9["presentation"] == (
        "three_state_label_set_exploratory_with_te_check"
        if d9["met"]
        else "indicator_agreement_appendix"
    )
    model = json.loads((dl.OUT / "records/label_model.json").read_text())
    assert "used_anchor" in model["fit"]
    assert model["fit"]["anchor_bins"] == int(valid.all(axis=1).sum()) or (
        not model["fit"]["used_anchor"]
    )
    readme = (dl.REPO / "data/events/detachment/README.md").read_text()
    models = readme.split("<!-- MODELS -->")[1].split("<!-- /MODELS -->")[0]
    assert "detach-ours" not in models and "detach-victor" not in models
    handoff = (ROOT / "HANDOFF.md").read_text()
    assert sha[:12] in handoff and "lower_shelf" in handoff
    json_paths = sorted(
        p
        for p in RESULTS.glob("detachment_*.json")
        if p.name != "detachment_validation.json"
    )
    json_paths += sorted((dl.OUT / "records").glob("*.json"))
    json_paths.append(dl.REPO / "docs/labeler/figure2_detach.json")
    replicate_fields = []

    def check_replicates(item):
        if isinstance(item, dict):
            count = item.get("valid_replicates")
            if isinstance(count, int):
                assert 0 <= count <= 1000
                replicate_fields.append(count)
            elif isinstance(count, dict):
                for value in count.values():
                    assert isinstance(value, int) and 0 <= value <= 1000
                    replicate_fields.append(value)
            for value in item.values():
                check_replicates(value)
        elif isinstance(item, list):
            for value in item:
                check_replicates(value)

    def reject_nonfinite(value):
        raise ValueError(f"Nonfinite JSON literal {value}")

    for path in json_paths:
        record = json.loads(path.read_text(), parse_constant=reject_nonfinite)
        check_replicates(record)
    log = ROOT / "logs/round4-covering-tests.log"
    tests = re.search(r"(\d+) passed", log.read_text())
    assert tests, "Covering test success absent"
    lint_log = ROOT / "logs/round4-ruff.log"
    assert "All checks passed!" in lint_log.read_text()
    assert "would reformat" not in lint_log.read_text()
    record = {
        "status": "passed",
        "producer": "scripts/labeler/detach_validate_outputs.py",
        "labels_sha256": sha,
        "assessed_bins": len(frame),
        "intervals": len(table),
        "trace_rows": trace_rows,
        "grids_checked": len(shot_ids),
        "certain_bins": int(certain.sum()),
        "certain_bins_by_split": certain_by_split,
        "sol_current_voting_bins": int(jsat.sum()),
        "reference_points": reference["n_primary_reference_points"],
        "finite_json_files_checked": len(json_paths),
        "bootstrap_valid_count_fields_checked": len(replicate_fields),
        "covering_tests_passed": int(tests.group(1)),
        "covering_test_log": str(log),
        "lint_log": str(lint_log),
        "checks": [
            "rule reproduction and legacy alias",
            "valid assessed bins and abstention encoding",
            "native and 250 ms radiation offset gates and covered heating windows",
            "upper-shelf/known-ELM certainty and explicit cohort counts",
            "SOL position and flux margins",
            "exact interval, sparse-grid and trace reconstruction; no stale shots",
            "reference, figure, Figure 2 and current-state checksums match the labels",
            "certain MARFE needs the TangTV vote; candidate_marfe never certain",
            "relative-Prad sensitivity column and absolute cutoffs reproduce",
            "no lower-shelf column in labels, traces or handoff; one Figure 2 record",
            "paper D9 decision recomputed from the labels",
            "diagnostic metadata does not claim primary-rule grids",
            "strict finite JSON and bootstrap replicate counts",
        ],
    }
    (RESULTS / "detachment_validation.json").write_text(dumps(record, indent=1))
    print(dumps(record, indent=1))


if __name__ == "__main__":
    main()
