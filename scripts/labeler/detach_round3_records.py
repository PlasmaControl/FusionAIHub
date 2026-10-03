#!/usr/bin/env python
"""Record current exploratory coverage, threshold margins and sensitivity.

Sensitivity changes one threshold family at a time on complete eligible non-test
shot timelines. Votes and the primary compatibility rule are recomputed, including
the adjacent-bin spatial/cue MARFE gate. These descriptive counts never select a
threshold and are not an independent physical benchmark.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import detach_label as dl
import numpy as np
import pandas as pd
from detach_json import dumps
from detach_round2_records import summarize

from labeler.events.detachment import core, label_model, tangtv
from labeler.events.detachment import thresholds as th

ROOT = Path(os.environ["LABELER_ROOT"]) / "round4/detach"
RESULT = dl.REPO / "docs/labeler/results/detachment_round3.json"
STATE_NAMES = {1: "attached", 2: "detached", 3: "marfe", 4: "uncertain"}


def distribution(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    q = np.percentile(values, (0, 5, 25, 50, 75, 95, 100))
    return {
        "n": len(values),
        **dict(zip(("min", "q05", "q25", "median", "q75", "q95", "max"), q)),
    }


def population(frame):
    return {
        "bins": len(frame),
        "shots": int(frame.shot.nunique()),
        "shot_ids": sorted(int(s) for s in frame.shot.unique()),
    }


def state_support(frame, state):
    return {
        name: population(frame.loc[np.asarray(state) == code])
        for code, name in STATE_NAMES.items()
    }


def composition(labels, intervals):
    by_state = state_support(labels, labels.state_rule)
    for code, name in STATE_NAMES.items():
        spans = intervals[intervals.category == code]
        by_state[name].update(
            seconds=by_state[name]["bins"] * core.BIN_MS / 1000,
            intervals=len(spans),
            single_50ms_intervals=int(((spans.t_end - spans.t_start) == 50).sum()),
        )
        by_state[name]["time_ranges_by_shot_ms"] = {
            str(int(shot)): [
                float(rows.start_ms.min()),
                float(rows.start_ms.max() + core.BIN_MS),
            ]
            for shot, rows in labels[labels.state_rule == code].groupby("shot")
        }
        by_state[name]["intervals_by_shot_ms"] = {
            str(int(shot)): [
                [float(row.t_start), float(row.t_end)]
                for row in rows.itertuples(index=False)
            ]
            for shot, rows in spans.groupby("shot")
        }
    per_shot = {
        str(int(shot)): {
            name: int((rows.state_rule == code).sum())
            for code, name in STATE_NAMES.items()
        }
        for shot, rows in labels.groupby("shot")
    }
    transitions = [
        int(shot)
        for shot, counts in per_shot.items()
        if counts["attached"] and counts["detached"]
    ]
    return {
        "by_state": by_state,
        "per_shot": per_shot,
        "shots_with_certain_attached_and_detached": transitions,
        "transition_interpretation": (
            "No shot has both certain attached and detached bins; the export "
            "cannot support a study of attached-to-detached transitions."
            if not transitions
            else "Both states occur on the listed shots; this is coverage only."
        ),
    }


def threshold_margins(bins, labels):
    upper = bins[
        bins.tangtv_tier.eq("upper_shelf") & bins.prad_valid & bins.tangtv_valid
    ]
    out = {
        "upper_shelf_both_valid": {
            **population(upper),
            "f_div": distribution(upper.prad_value),
        },
        "by_certain_state": {},
    }
    for code, name in STATE_NAMES.items():
        if code == core.UNCERTAIN:
            continue
        rows = labels[labels.state_rule == code]
        margin = (
            th.PRAD_ATTACHED_MAX - rows.prad_value
            if code == core.ATTACHED
            else rows.prad_value - th.PRAD_DETACHED_MIN
        )
        out["by_certain_state"][name] = {
            **population(rows),
            "f_div": distribution(rows.prad_value),
            "f_div_signed_distance_to_voting_threshold": distribution(margin),
            "f_div_bins_within_0_05": int((margin.abs() <= 0.05).sum()),
            "f_div_bins_within_0_1": int((margin.abs() <= 0.1).sum()),
            "greenwald_fraction": distribution(rows.aux_greenwald_fraction),
            "greenwald_distance_to_cue": distribution(
                rows.aux_greenwald_fraction - th.GREENWALD_CUE_MIN
            ),
        }
    return out


def recompute(frame, *, prad_shift=0.0, greenwald_shift=0.0):
    votes, valid = dl.matrices(frame)
    f_div = frame.prad_value.to_numpy(float)
    votes[:, 1] = core.ABSTAIN
    votes[valid[:, 1] & (f_div <= th.PRAD_ATTACHED_MAX + prad_shift), 1] = core.ATTACHED
    votes[valid[:, 1] & (f_div >= th.PRAD_DETACHED_MIN + prad_shift), 1] = core.DETACHED
    for idx in frame.groupby("shot").indices.values():
        rows = frame.iloc[idx]
        # Complete per-shot bins preserve adjacency before export selection.
        if len(rows) > 1 and not np.allclose(np.diff(rows.start_ms), core.BIN_MS):
            raise ValueError("Sensitivity needs consecutive full-shot bins")
        fg = rows.aux_greenwald_fraction.to_numpy(float)
        cue = rows.tangtv_marfe_back_transition.to_numpy(bool) | (
            np.isfinite(fg) & (fg >= th.GREENWALD_CUE_MIN + greenwald_shift)
        )
        votes[idx, 2], _ = tangtv.evidence_votes(
            rows.tangtv_value.to_numpy(float),
            valid[idx, 2],
            rows.tangtv_marfe_spatial.to_numpy(bool),
            cue,
        )
    known = np.isfinite(frame.aux_elm_share.to_numpy(float))
    return label_model.compatibility_decide(
        votes, valid, tangtv_tier=frame.tangtv_tier.to_numpy(), elm_known=known
    )[0]


def threshold_sensitivity(bins, labels):
    eligible = bins[bins.shot.isin(labels.shot.unique())].copy()
    eligible["split"] = eligible.shot.map(dl.cohort_split(eligible.shot.unique()))
    non_test = eligible[eligible.split != "test"].reset_index(drop=True)
    required = "tangtv_marfe_back_transition"
    if required not in non_test:
        raise ValueError(f"Regenerated bins must export {required}")
    state = recompute(non_test)
    observed = labels[labels.split != "test"][["shot", "start_ms", "state_rule"]]
    check = non_test[["shot", "start_ms"]].assign(recomputed=state).merge(observed)
    mismatch = int((check.recomputed != check.state_rule).sum())
    if mismatch:
        raise ValueError(f"Threshold baseline differs on {mismatch} exported bins")
    out = {
        "population": population(non_test),
        "note": (
            "Eligible full-shot bins on train/validation/outside shots; fixed test "
            "excluded. Both Prad cutoffs shift together. Greenwald changes only "
            "the cue threshold; spatial evidence, H-L cue, adjacent-bin "
            "persistence and all measurement gates are recomputed unchanged. "
            "Descriptive sensitivity; no threshold is chosen from these counts."
        ),
        "baseline_exported_bins_checked": len(check),
        "baseline_mismatch_bins": mismatch,
        "prad": [],
        "greenwald": [],
    }
    for shift in (-0.1, -0.05, 0.0, 0.05, 0.1):
        state = recompute(non_test, prad_shift=shift)
        certain = np.isin(state, (1, 2, 3))
        out["prad"].append(
            {
                "shift": shift,
                "attached_max": th.PRAD_ATTACHED_MAX + shift,
                "detached_min": th.PRAD_DETACHED_MIN + shift,
                "certain": population(non_test.loc[certain]),
                "by_state": state_support(non_test, state),
            }
        )
    for shift in (-0.1, 0.0, 0.1):
        state = recompute(non_test, greenwald_shift=shift)
        certain = np.isin(state, (1, 2, 3))
        out["greenwald"].append(
            {
                "shift": shift,
                "cue_min": th.GREENWALD_CUE_MIN + shift,
                "certain": population(non_test.loc[certain]),
                "by_state": state_support(non_test, state),
            }
        )
    return out


def prad_votes_by_tier(labels):
    """Cast Prad votes by TangTV tier, to show where attached votes can occur."""
    names = {1: "attached", 2: "detached", 3: "marfe", core.ABSTAIN: "abstain"}
    out = {}
    for tier, rows in labels.groupby("tangtv_tier"):
        counts = rows.prad_vote.value_counts()
        out[str(tier)] = {
            name: int(counts.get(code, 0)) for code, name in names.items()
        }
    return out


def afrac_absence(bins):
    probes = {int(p.stem) for p in (ROOT / "processed_probes").glob("*.npz")}
    maps = set()
    for path in (ROOT / "efit").glob("*.npz"):
        with np.load(path) as f:
            if len(f["gtime_ms"]) >= 2:
                maps.add(int(path.stem))
    positioned = bins.afrac_probe_position_valid.fillna(False).to_numpy(bool)
    return {
        "processed_probe_files": len(probes),
        "flux_map_files": len(maps),
        "processed_probe_shots_with_flux_maps": len(probes & maps),
        "processed_probe_shots_without_flux_maps": len(probes - maps),
        "probe_flux_unknown_bins": int(
            bins.afrac_reason.eq("probe_flux_unknown").sum()
        ),
        "upper_shelf_sol_probe_valid_bins": int(
            (positioned & bins.tangtv_tier.eq("upper_shelf").to_numpy()).sum()
        ),
        "lower_shelf_sol_probe_valid_bins": int(
            (positioned & bins.tangtv_tier.eq("lower_shelf_window").to_numpy()).sum()
        ),
        "valid_bins_by_tangtv_tier": {
            str(tier): int(rows.afrac_valid.sum())
            for tier, rows in bins.groupby("tangtv_tier")
        },
        "invalid_reason_bins": {
            str(k): int(v)
            for k, v in bins.loc[~bins.afrac_valid, "afrac_reason"]
            .value_counts()
            .items()
        },
        "min_reference_duration_ms": None,
        "interpretation": (
            "No upper-shelf SOL probe passes the position/flux gate. Sparse EFIT "
            "maps prevent assessing most positioned-probe shots. The local Jsat "
            "proxy lacks an independently calibrated attached-current reference; "
            "it does not reproduce published Afrac. No arbitrary minimum "
            "reference duration is applied."
        ),
    }


def normalization_change(labels):
    path = ROOT / "fix_round3_before/labels_bins.csv.gz"
    if not path.exists():
        return {"status": "no saved baseline"}
    before = pd.read_csv(path)
    paired = before[["shot", "start_ms", "state_rule"]].merge(
        labels[["shot", "start_ms", "state_rule"]],
        on=["shot", "start_ms"],
        how="outer",
        suffixes=("_before", "_after"),
    )
    old = paired.state_rule_before.isin((1, 2, 3))
    new = paired.state_rule_after.isin((1, 2, 3))
    return {
        "source": str(path),
        "note": "All measurement-gate and normalization repairs, not an isolated ablation.",
        "before_certain": int(old.sum()),
        "after_certain": int(new.sum()),
        "retained_certain": int((old & new).sum()),
        "lost_certain": int((old & ~new).sum()),
        "added_certain": int((~old & new).sum()),
        "lost_certain_by_state": {
            name: int((old & ~new & paired.state_rule_before.eq(code)).sum())
            for code, name in STATE_NAMES.items()
            if code != core.UNCERTAIN
        },
        "added_certain_by_state": {
            name: int((~old & new & paired.state_rule_after.eq(code)).sum())
            for code, name in STATE_NAMES.items()
            if code != core.UNCERTAIN
        },
        "before_by_state": state_support(before, before.state_rule),
        "after_by_state": state_support(labels, labels.state_rule),
    }


def measurement_gate_audit(bins):
    before_dir = ROOT / "fix_round3_before/bins"
    if not before_dir.exists():
        return {"status": "no saved baseline"}
    before = dl.load_all(before_dir)
    old_negative = before[before.prad_valid & (before.prad_value < 0)]
    paired = old_negative[["shot", "start_ms"]].merge(
        bins[["shot", "start_ms", "prad_valid", "prad_reason", "prad_value"]],
        on=["shot", "start_ms"],
        how="left",
    )
    out = {
        "before_bins_source": str(before_dir),
        "after_bins_source": str(ROOT / "bins"),
        "radiation_negative_tolerance_w": th.RADIATION_NEGATIVE_TOL_W,
        "averaging_ms": th.PRAD_AVERAGING_MS,
        "availability_interval_convention": {
            "averaging_window": "closed at both endpoints, matching window_mean",
            "native_label_bin": "left-closed, right-open",
        },
        "tolerance_note": (
            "Operational 0.05 MW allowance on both the native 50 ms label-bin "
            "mean and centered 250 ms inter-ELM radiation mean, not a calibrated "
            "uncertainty. A mean below -0.05 MW on either scale invalidates the "
            "measurement; a smaller negative 250 ms mean is clipped to zero "
            "in the ratio. Changes "
            "combine averaging and all repaired gates, not an isolated ablation."
        ),
        "old_valid_negative_prad_bins": len(old_negative),
        "new_negative_radiation_reason_bins": int(
            bins.prad_reason.eq("negative_radiation").sum()
        ),
        "native_mean_below_negative_tolerance_bins": int(
            (bins.aux_prad_divl_native_w < -th.RADIATION_NEGATIVE_TOL_W).sum()
        ),
        "averaged_mean_below_negative_tolerance_bins": int(
            (bins.aux_prad_divl_w < -th.RADIATION_NEGATIVE_TOL_W).sum()
        ),
        "negative_native_mean_hidden_by_averaging_bins": int(
            (
                (bins.aux_prad_divl_native_w < -th.RADIATION_NEGATIVE_TOL_W)
                & (bins.aux_prad_divl_w >= -th.RADIATION_NEGATIVE_TOL_W)
            ).sum()
        ),
        "full_250ms_elm_window_known_bins": int(bins.aux_prad_elm_window_known.sum()),
        "full_250ms_elm_window_unknown_bins": int(
            (~bins.aux_prad_elm_window_known).sum()
        ),
        "native_elm_known_but_full_window_unknown_bins": int(
            (
                np.isfinite(bins.aux_elm_share.to_numpy(float))
                & ~bins.aux_prad_elm_window_known.to_numpy(bool)
            ).sum()
        ),
        "full_window_elm_note": (
            "D-alpha availability must cover the full centered 250 ms radiation "
            "window, not only the native 50 ms label bin. Unknown availability "
            "invalidates radiation rather than dropping unknown samples silently."
        ),
        "old_negative_retained_valid_bins": int(paired.prad_valid.eq(True).sum()),
        "old_negative_invalid_bins": int(paired.prad_valid.eq(False).sum()),
        "old_negative_missing_after_bins": int(paired.prad_valid.isna().sum()),
        "old_negative_after_invalid_reasons": {
            str(k): int(v)
            for k, v in paired.loc[paired.prad_valid.eq(False), "prad_reason"]
            .value_counts()
            .items()
        },
        "reason_counts": {},
        "witness_190788_2750ms": {},
    }
    for key, column, reason in (
        ("prad_no_input_power", "prad_reason", "no_input_power"),
        ("prad_elm_unknown", "prad_reason", "elm_unknown"),
        ("tangtv_elm_unknown", "tangtv_reason", "elm_unknown"),
        ("afrac_elm_unknown", "afrac_reason", "elm_unknown"),
    ):
        old, new = [int(frame[column].eq(reason).sum()) for frame in (before, bins)]
        out["reason_counts"][key] = {"before": old, "after": new, "delta": new - old}
    old, new = [
        int((~np.isfinite(frame.aux_elm_share.to_numpy(float))).sum())
        for frame in (before, bins)
    ]
    out["elm_unknown_bins"] = {"before": old, "after": new, "delta": new - old}
    for name, frame in (("before", before), ("after", bins)):
        rows = frame[(frame.shot == 190788) & (frame.start_ms == 2750)]
        if len(rows) != 1:
            raise ValueError(f"Expected one 190788/2750 witness in {name}")
        row = rows.iloc[0]
        values = {}
        for key in (
            "prad_value",
            "prad_valid",
            "prad_reason",
            "prad_vote",
            "aux_p_in_w",
            "aux_prad_divl_w",
            "aux_prad_divl_native_w",
            "aux_prad_tot_w",
            "aux_elm_share",
            "aux_prad_elm_window_known",
            "prad_averaging_ms",
        ):
            value = row.get(key)
            if isinstance(value, (bool, np.bool_)):
                values[key] = bool(value)
            elif isinstance(value, (float, int, np.number)):
                values[key] = float(value) if np.isfinite(value) else None
            else:
                values[key] = None if value is None else str(value)
        out["witness_190788_2750ms"][name] = values
    return out


def main():
    bins = dl.load_all(ROOT / "bins").sort_values(["shot", "start_ms"])
    bins = bins.reset_index(drop=True)
    labels = pd.read_csv(ROOT / "labels_bins.csv.gz")
    intervals = pd.read_csv(dl.OUT / "detach_shots.csv")
    result = {
        "scope": "exploratory coverage and indicator agreement; no independent benchmark",
        "sources": {
            "bins": str(ROOT / "bins"),
            "labels": str(ROOT / "labels_bins.csv.gz"),
            "labels_sha256": hashlib.sha256(
                (ROOT / "labels_bins.csv.gz").read_bytes()
            ).hexdigest(),
            "intervals": str(dl.OUT / "detach_shots.csv"),
            "script": "scripts/labeler/detach_round3_records.py",
        },
        "current": summarize(),
        "composition": composition(labels, intervals),
        "threshold_margins": threshold_margins(bins, labels),
        "threshold_sensitivity": threshold_sensitivity(bins, labels),
        "afrac_absence": afrac_absence(bins),
        "prad_votes_by_tangtv_tier": prad_votes_by_tier(labels),
        "normalization_change": normalization_change(labels),
        "measurement_gate_audit": measurement_gate_audit(bins),
        "threshold_derivation": {
            "source": "Chen 2026, Nuclear Fusion 66 036014, shot 201081 worked example",
            "reported_nbi_mw": 4.0,
            "local_total_heating_assumption_mw": 4.4,
            "local_total_heating_assumption_status": (
                "Local assumed total, not a published measured P_in. No independent "
                "source establishes the extra 0.4 MW."
            ),
            "attached_prad_mw": 1.6,
            "detached_prad_mw": 2.2,
            "attached_ratio_before_rounding": 1.6 / 4.4,
            "attached_max": th.PRAD_ATTACHED_MAX,
            "detached_min": th.PRAD_DETACHED_MIN,
            "pinj_fetch_failure": (
                "PTDATA client configuration: PTSERVER/ptserver not in /etc/services; "
                "this failure does not establish missing experimental data."
            ),
        },
        "eligibility": {
            "minimum_valid_bins_per_indicator": dl.MIN_VALID_BINS,
            "minimum_jointly_assessed_bins": dl.MIN_VALID_BINS,
            "note": (
                "At least two indicators must each be valid in 20 bins and at least "
                "20 bins must have two valid indicators. This one-second shot "
                "eligibility condition narrows the brief's every-shot criterion."
            ),
        },
    }
    RESULT.write_text(dumps(result, indent=1))
    print(
        dumps(
            {
                "certain": result["current"]["certain"],
                "sensitivity": result["threshold_sensitivity"],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
