#!/usr/bin/env python
"""Check regenerated detachment exports against their rule and source records."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import detach_label as dl
import numpy as np
import pandas as pd
from detach_json import dumps

from labeler.events.detachment import core, label_model, prad, signals, thresholds
from labeler.events.interval_tables import read_label_grid

ROOT = Path(os.environ["LABELER_ROOT"]) / "round4/detach"
RESULTS = dl.REPO / "docs/labeler/results"
STRIKE_REAL_R_M = (0.8, 2.5)


def check_provenance(meta: dict) -> dict:
    """The label tables were made from a clean tree whose producers are still HEAD's."""
    made = meta["made_from"][0]
    sha = made["git_sha"]
    assert made["code_dirty"] is False, "labels made from a dirty tree"
    changed = subprocess.run(
        [
            "git",
            "-C",
            str(dl.REPO),
            "diff",
            "--name-only",
            sha,
            "HEAD",
            "--",
            *dl.PRODUCER_PATHS,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert not changed, f"label producers changed since {sha}: {changed}"
    return {
        "made_from_sha": sha,
        "code_dirty": False,
        "producer_paths": list(dl.PRODUCER_PATHS),
    }


def main():
    frame = pd.read_csv(ROOT / "labels_bins.csv.gz")
    assert frame.regime.notna().all()
    regime = frame.regime.to_numpy()
    votes, valid = dl.matrices(frame)
    rule, _ = label_model.compatibility_decide(
        votes,
        valid,
        tangtv_tier=frame.tangtv_tier.to_numpy(),
        elm_known=np.isfinite(frame.aux_elm_share),
        regime=regime,
    )
    assert np.array_equal(frame.state_rule, rule)
    assert np.array_equal(frame.state_lm, frame.state_rule)
    # the sensitivity columns are the same rule with f_div added as a second voter,
    # on the relative votes and on the absolute ones
    extra = ("afrac", "prad")
    relative, _ = label_model.compatibility_decide(
        votes,
        valid,
        tangtv_tier=frame.tangtv_tier.to_numpy(),
        elm_known=np.isfinite(frame.aux_elm_share),
        second=extra,
        regime=regime,
    )
    assert np.array_equal(frame.state_rule_relative_prad, relative)
    absolute_votes, absolute_valid = votes.copy(), valid.copy()
    absolute_votes[:, 1] = frame.prad_abs_vote.to_numpy()
    absolute_valid[:, 1] = frame.prad_abs_valid.to_numpy(bool)
    absolute, _ = label_model.compatibility_decide(
        absolute_votes,
        absolute_valid,
        tangtv_tier=frame.tangtv_tier.to_numpy(),
        elm_known=np.isfinite(frame.aux_elm_share),
        second=extra,
        regime=regime,
    )
    assert np.array_equal(frame.state_rule_absolute_prad, absolute)
    # the lower-shelf column is not a label column and is not in any handoff file
    assert not [c for c in frame.columns if "lower_shelf" in c and c != "tangtv_tier"]
    # f_div is not a vote of the label; its relative vote (per-shot baseline) and its
    # absolute one (global cutoffs) are sensitivities
    relative_cut = thresholds.prad_relative_cutoffs()
    prad_cast = frame.prad_valid.to_numpy(bool)
    expected_prad = prad.fdiv_vote(frame.prad_rel_value.to_numpy(float), *relative_cut)
    assert np.array_equal(frame.prad_vote[prad_cast], expected_prad[prad_cast])
    absolute_cut = thresholds.prad_cutoffs()
    abs_cast = frame.prad_abs_valid.to_numpy(bool)
    expected_abs = prad.fdiv_vote(frame.prad_value.to_numpy(float), *absolute_cut)
    assert np.array_equal(frame.prad_abs_vote[abs_cast], expected_abs[abs_cast])
    assert not (prad_cast & ~abs_cast).any(), "relative vote valid without absolute"
    assert frame.confidence.isna().all()
    assert (valid.sum(axis=1) >= 1).all()
    assert ((valid.sum(axis=1) >= 2) | (votes[:, 2] > 0)).all()
    assert (votes[~valid] == core.ABSTAIN).all()
    assert (
        frame.loc[frame.prad_abs_valid, "aux_prad_divl_w"]
        .ge(-thresholds.RADIATION_NEGATIVE_TOL_W)
        .all()
    )
    assert (
        frame.loc[frame.prad_abs_valid, "aux_prad_divl_native_w"]
        .ge(-thresholds.RADIATION_NEGATIVE_TOL_W)
        .all()
    )
    assert frame.loc[frame.prad_abs_valid, "aux_p_in_w"].notna().all()
    assert frame.loc[frame.prad_abs_valid, "aux_prad_elm_window_known"].all()
    assert frame.prad_averaging_ms.eq(250.0).all()
    # tiers: certain / tangtv_only are the only tiers with an attached or detached
    # state, MARFE is never a state, and both need an upper-shelf TangTV vote
    labelled = frame.state_rule.isin((core.ATTACHED, core.DETACHED))
    certain = frame.tier.eq("certain")
    silver = frame.tier.eq("tangtv_only")
    assert (labelled == (certain | silver)).all()
    assert not frame.state_rule.eq(core.MARFE).any(), "no certain MARFE is exported"
    # the a-priori L-mode gate: a DETACHED TangTV vote on a known L-mode bin or in a
    # probable-L window is its own uncertain tier and no detached state is exported
    # there
    gated = frame.tier.eq("tangtv_only_lmode")
    assert frame.loc[gated, "state_rule"].eq(core.UNCERTAIN).all()
    assert frame.loc[gated, "regime"].isin(signals.GATED_REGIMES).all()
    assert frame.loc[gated, "tangtv_vote"].eq(core.DETACHED).all()
    in_gate = frame.regime.isin(signals.GATED_REGIMES)
    assert not (in_gate & frame.state_rule.eq(core.DETACHED)).any()
    # the probable-regime proxy is a pure function of the stored columns: typing every
    # bin of the unknown-regime windows again from their ELM share and input power
    # reproduces the regime and its source
    for shot, rows in frame.groupby("shot"):
        base = rows.regime.where(~rows.regime.str.startswith("probable_"), "unknown")
        source = rows.regime_source.where(
            ~rows.regime.str.startswith("probable_"), "unknown"
        )
        again, again_source = signals.probable_regimes(
            base.to_numpy(),
            source.to_numpy(),
            rows.aux_elm_known.to_numpy(bool),
            rows.aux_elm_share.to_numpy(float),
            rows.aux_p_in_w.to_numpy(float),
            core.BIN_MS,
        )
        assert np.array_equal(again, rows.regime.to_numpy()), shot
        assert np.array_equal(again_source, rows.regime_source.to_numpy()), shot
    assert frame.loc[labelled, "tangtv_tier"].eq("upper_shelf").all()
    # certain: TangTV votes the state and a valid Afrac vote is compatible with it;
    # tangtv_only: Afrac casts no vote; f_div is not consulted
    compatible = label_model.COMPATIBLE
    for state in (core.ATTACHED, core.DETACHED):
        rows = frame.state_rule.eq(state)
        assert votes[rows, 2].tolist() == [state] * int(rows.sum())
        for tier_rows, need_second in ((rows & certain, True), (rows & silver, False)):
            afrac_votes = votes[tier_rows.to_numpy()][:, 0]
            if need_second:
                assert (afrac_votes > 0).all()
                for vote in (core.ATTACHED, core.DETACHED):
                    if (afrac_votes == vote).any():
                        assert state in compatible["afrac"][vote]
            else:
                assert not (afrac_votes > 0).any()
    conflict = frame.tier.eq("conflict")
    assert frame.loc[conflict, "state_rule"].eq(core.UNCERTAIN).all()
    assert (votes[conflict.to_numpy(), 0] > 0).all()
    candidate = frame.tier.eq("candidate_marfe")
    assert frame.loc[candidate, "state_rule"].eq(core.UNCERTAIN).all()
    assert (
        frame.loc[candidate, "tangtv_vote"].eq(core.MARFE)
        | frame.loc[candidate, "tangtv_marfe_candidate"].astype(bool)
    ).all()
    assert np.isfinite(frame.loc[labelled, "aux_elm_share"]).all()
    certain_by_split = {
        split: int((certain & frame.split.eq(split)).sum())
        for split in ("train", "val", "test", "outside")
    }
    # M4: the lower-shelf extraction is never a valid vote and never a state
    lower = frame.tangtv_tier.eq("lower_shelf_window")
    assert not frame.loc[lower, "tangtv_valid"].astype(bool).any()
    assert frame.loc[lower, "tangtv_reason"].eq("lower_shelf_window").all()
    assert (
        not frame.loc[frame.tier.eq("lower_shelf_window"), "state_rule"]
        .isin((core.ATTACHED, core.DETACHED))
        .any()
    )
    # Afrac: per-probe reference inside the flux window, never in a known L-mode
    jsat = frame.afrac_valid.astype(bool)
    assert frame.loc[jsat, "afrac_probe_position_valid"].all()
    assert (
        (frame.loc[jsat, "aux_jsat_selected_psin"] - 1.0).abs()
        <= thresholds.AFRAC_PSI_WINDOW + 1e-6
    ).all()
    assert (frame.loc[jsat, "afrac_probe_n_eligible"] >= 1).all()
    assert frame.loc[jsat, "afrac_method"].eq("per_probe_reference").all()
    assert (frame.loc[jsat, "aux_jsat_reference"] > 0).all()
    assert not frame.loc[jsat, "regime"].eq("L").any()
    assert frame.loc[frame.afrac_reason.eq("l_mode"), "regime"].eq("L").all()
    # Afrac's own gate is known L only: it is never valid on a probable-L bin
    assert not frame.loc[jsat, "regime"].eq("probable_L").any()
    # M1: an EFIT sentinel is never exported as the outer strike point
    strike = frame.aux_jsat_strike_r_m
    assert (
        strike.isna()
        | ((strike >= STRIKE_REAL_R_M[0]) & (strike <= STRIKE_REAL_R_M[1]))
    ).all()
    table = pd.read_csv(dl.OUT / "detach_shots.csv")
    assert table.confidence.isna().all()
    primary_meta = json.loads((dl.OUT / "detach_shots.meta.json").read_text())
    diagnostic_meta = json.loads((ROOT / "labels_label_model.meta.json").read_text())
    provenance = check_provenance(primary_meta)
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
    cohort = frame.split.isin(("train", "val", "test"))
    for name, tiers in (
        ("certain", ("certain",)),
        ("certain_or_tangtv_only", ("certain", "tangtv_only")),
    ):
        rows = frame[frame.tier.isin(tiers)]
        both = set(rows.loc[rows.state_rule.eq(core.ATTACHED), "shot"]) & set(
            rows.loc[rows.state_rule.eq(core.DETACHED), "shot"]
        )
        cohort_both = both & set(frame.loc[cohort, "shot"])
        assert sorted(both) == sorted(d9[name]["shots_with_both"])
        assert sorted(cohort_both) == sorted(d9[name]["cohort_shots_with_both"])
        assert d9[name]["met"] == (len(both) >= 3 and len(cohort_both) >= 1)
    assert d9["met"] == d9["certain"]["met"]
    assert d9["presentation"] == (
        "three_state_label_set_exploratory_with_te_check"
        if d9["met"]
        else "indicator_agreement_appendix"
    )
    # the Afrac check decided to keep or drop the proxy; the vote agrees with it
    afrac_check = json.loads((RESULTS / "detachment_afrac_check.json").read_text())
    keep = afrac_check["decision"]["keep_afrac_in_vote"]
    assert keep == bool((frame.afrac_vote.gt(0) & frame.afrac_valid.astype(bool)).any())
    # the te check and the prad sensitivity score the same labels
    te = json.loads((RESULTS / "detachment_te_check.json").read_text())
    assert te["bins_assessed"] == len(frame)
    sensitivity = json.loads((RESULTS / "detachment_prad_sensitivity.json").read_text())
    assert sensitivity["variants"]["primary"]["reproduces_exported_labels"]
    assert sensitivity["second_voters"] == list(label_model.SECOND_VOTERS)
    assert sensitivity["n_assessed_bins"] == len(frame)
    # f_div is reported per shot as a corroborator, and Figure 2 draws that record
    fdiv_check = json.loads((RESULTS / "detachment_fdiv_check.json").read_text())
    assert fdiv_check["role"].endswith("not a vote")
    assert len(panel["fdiv_corroborator"]) == 4
    for row in panel["fdiv_corroborator"]:
        family, against = row["key"].split("_vs_")
        within = fdiv_check["summary"][family.removeprefix("fdiv_")][against][
            "within_shot"
        ]
        assert row["within_shot"]["n_shots"] == within["n_shots"]
    # the learned baselines are scored on these labels, or the docs call them stale
    docs = (dl.REPO / "docs/labeler/detachment.md").read_text()
    baseline_block = docs.split("<!-- BASELINES -->")[1].split("<!-- /BASELINES -->")[0]
    for name in ("ours", "victor"):
        record = json.loads((RESULTS / f"detachment_{name}.json").read_text())
        current = record["label_source"]["sha256"] == sha
        assert current != ("stale" in baseline_block), f"detach-{name} baseline status"
    model = json.loads((dl.OUT / "records/label_model.json").read_text())
    assert "used_anchor" in model["fit"]
    assert model["fit"]["anchor_bins"] == int(valid.all(axis=1).sum()) or (
        not model["fit"]["used_anchor"]
    )
    readme = (dl.REPO / "data/events/detachment/README.md").read_text()
    models = readme.split("<!-- MODELS -->")[1].split("<!-- /MODELS -->")[0]
    assert "detach-ours" not in models and "detach-victor" not in models
    handoff = (ROOT / "HANDOFF.md").read_text()
    assert sha[:12] in handoff and "lower_shelf" in handoff and "tangtv_only" in handoff
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
        "tangtv_only_bins": int(silver.sum()),
        "certain_bins_by_split": certain_by_split,
        "provenance": provenance,
        "afrac_valid_bins": int(jsat.sum()),
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
            (
                "tiers: certain needs TangTV plus a compatible valid Afrac vote; "
                "tangtv_only has no Afrac vote; conflict is TangTV against Afrac; "
                "f_div is not consulted; no MARFE state; upper shelf only"
            ),
            (
                "Afrac inside the flux window with a per-probe reference; no valid "
                "bin in a known L-mode; no sentinel strike point"
            ),
            "lower-shelf TangTV invalid, never a state",
            "label tables made from a clean tree whose producers are unchanged at HEAD",
            "the Afrac check's keep decision matches the vote",
            "exact interval, sparse-grid and trace reconstruction; no stale shots",
            "reference, figure, Figure 2 and current-state checksums match the labels",
            "candidate_marfe never carries a state",
            (
                "tangtv_only_lmode: a detached TangTV vote on a known L-mode bin is "
                "uncertain, and no detached state sits on a known L-mode bin"
            ),
            (
                "f_div relative and absolute sensitivity columns (f_div added as a "
                "second voter) reproduce; the f_div check feeds Figure 2"
            ),
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
