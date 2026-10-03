#!/usr/bin/env python
"""Reproducible fix audit, source sensitivity, development validation and Figure 2."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import detach_benchmark as bench
import detach_label as dl
import numpy as np
import pandas as pd

from labeler.events.detachment import core, label_model, thresholds

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ["LABELER_ROOT"]) / "round4/detach"
RESULTS = REPO / "docs/labeler/results"


def population(frame, mask):
    rows = frame.loc[mask]
    return {
        "bins": len(rows),
        "shots": int(rows.shot.nunique()),
        "shot_ids": sorted(int(s) for s in rows.shot.unique()),
    }


def snapshot_legacy_populations():
    path = ROOT / "fix_previous/baseline_populations.json"
    if path.exists():
        return json.loads(path.read_text())
    populations = {}
    for name in ("ours", "victor"):
        with np.load(ROOT / name / "dataset.npz") as f:
            cv = f["split"] != "test"
            populations[name] = {
                "total_bins": len(cv),
                "total_shots": len(np.unique(f["shot"])),
                "cv_bins": int(cv.sum()),
                "cv_shots": len(np.unique(f["shot"][cv])),
            }
    frame = pd.read_csv(ROOT / "fix_previous/labels_bins.csv.gz")
    votes, valid = dl.matrices(frame)
    record = json.loads((ROOT / "fix_previous/label_model.json").read_text())["model"]
    model = label_model.LabelModel(corr=tuple(map(tuple, record["corr"])))
    model.theta = np.asarray(record["theta"])
    populations["loo_contributors"] = {}
    for j, name in enumerate(label_model.LF_NAMES):
        v = votes.copy()
        v[:, j] = core.ABSTAIN
        both = np.delete(valid, j, axis=1).all(axis=1)
        post = label_model.pool_marfe(model.posterior(v), v[:, 2] > 0)
        reference = label_model.decide(post, both, (v > 0).any(axis=1), 0.7)
        contributes = np.isin(reference, (1, 2, 3)) & valid[:, j] & (votes[:, j] > 0)
        populations["loo_contributors"][name] = {
            **population(frame, contributes),
            "test": population(frame, contributes & (frame.split.to_numpy() == "test")),
        }
    path.write_text(json.dumps(populations, indent=1))
    return populations


def legacy_sensitivity():
    old = pd.read_csv(ROOT / "fix_previous/labels_bins.csv.gz")
    record = json.loads((ROOT / "fix_previous/label_model.json").read_text())
    votes, valid = dl.matrices(old)
    fit = old.split.to_numpy() != "test"
    source = old.tangtv_source.to_numpy()
    invert = source == "inversion"
    subset = label_model.LabelModel(
        corr=tuple(map(tuple, record["model"]["corr"]))
    ).fit_anchored(votes[fit & invert], valid[fit & invert])
    post = label_model.pool_marfe(subset.posterior(votes), votes[:, 2] > 0)
    state = label_model.decide(
        post, np.ones(len(old), bool), (votes > 0).any(axis=1), 0.7
    )
    for idx in old.groupby("shot").indices.values():
        state[idx] = dl.smooth_segments(state[idx], old.start_ms.to_numpy()[idx])
    changed = state != old.state_lm.to_numpy()
    return {
        "population": population(old, np.ones(len(old), bool)),
        "original_anchor": {
            s: population(old, fit & valid.all(axis=1) & (source == s))
            for s in ("inversion", "surrogate")
        },
        "inversion_only_refit_changed": int(changed.sum()),
        "changed_by_source": {
            s: population(old, changed & (source == s))
            for s in ("inversion", "surrogate", "none")
        },
        "refit_implied_accuracies": subset.accuracies(),
        "original_implied_accuracies": record["labelling_functions"],
        "interpretation": "Model-implied weights are not physical accuracy. Original TangTV weight was a bound solution; posteriors uncalibrated.",
    }


def current_sensitivity(frame):
    votes, valid = dl.matrices(frame)
    fit = (frame.split != "test").to_numpy()
    source = frame.tangtv_source.to_numpy()
    models = {}
    for name, mask in (
        ("mixed", np.ones(fit.sum(), bool)),
        ("inversion_only", source[fit] == "inversion"),
    ):
        model = label_model.LabelModel().fit_anchored(
            votes[fit], valid[fit], anchor_mask=mask
        )
        posterior = label_model.pool_marfe(model.posterior(votes), votes[:, 2] > 0)
        states, _ = label_model.redundant_decide(posterior, votes, valid, 0.7)
        models[name] = (model, states)
    changed = models["mixed"][1] != models["inversion_only"][1]
    return {
        "changed_bins": int(changed.sum()),
        "anchor_by_source": {
            s: population(frame, fit & valid.all(axis=1) & (source == s))
            for s in ("inversion", "surrogate")
        },
        "changed_by_source": {
            s: population(frame, changed & (source == s))
            for s in ("inversion", "surrogate", "none")
        },
        "models": {
            k: {"anchor_bins": m.anchor_bins, "accuracies": m.accuracies()}
            for k, (m, _) in models.items()
        },
    }


def development_validation(bins):
    train = (bins.split == "train").to_numpy()
    tv = bins.tangtv_vote.to_numpy()
    pv = bins.prad_vote.to_numpy()
    common = (
        train
        & bins.prad_valid.to_numpy()
        & bins.tangtv_valid.to_numpy()
        & (bins.tangtv_source == "inversion").to_numpy()
        & np.isin(tv, (1, 2))
    )
    voting = common & (pv > 0)
    table = np.array(
        [[np.sum(voting & (tv == a) & (pv == b)) for b in (1, 2)] for a in (1, 2)]
    )
    rng = np.random.default_rng(0)
    by_shot = [
        np.array(
            [
                [
                    np.sum(voting & (bins.shot.to_numpy() == s) & (tv == a) & (pv == b))
                    for b in (1, 2)
                ]
                for a in (1, 2)
            ]
        )
        for s in bins.loc[voting, "shot"].unique()
    ]
    draws = []
    if by_shot:
        for _ in range(1000):
            t = np.sum(
                [by_shot[i] for i in rng.integers(0, len(by_shot), len(by_shot))],
                axis=0,
            )
            draws.append(bench.kappa_from(t))
    return {
        "thresholds": {
            "attached_max": thresholds.PRAD_ATTACHED_MAX,
            "detached_min": thresholds.PRAD_DETACHED_MIN,
        },
        "source": "Local Chen 201081 worked example; Eldon 2019 supplies sensor definition, not these thresholds.",
        "selection": "Fixed thresholds, no optimisation; validate only fixed cohort train inversion bins, not test/val/outside.",
        "valid_reference": population(bins, common),
        "voting_reference": population(bins, voting),
        "confusion_ref_by_vote": table.tolist(),
        "binary_kappa": bench.kappa_from(table),
        "kappa_ci95": np.nanpercentile(draws, [2.5, 97.5]).tolist() if draws else None,
        "agreement": float(np.trace(table) / table.sum()) if table.sum() else None,
        "power_range_mw": np.nanpercentile(
            bins.loc[common, "aux_p_in_w"], [0, 100]
        ).tolist()
        if common.any()
        else None,
        "temperature_check": "withdrawn; unlocalised DTS cannot validate detachment",
    }


def candidate_selection():
    survey = pd.read_csv(ROOT / "survey/corpus_survey.csv")
    cohort = set(pd.read_csv(REPO / "data/events/catalog/cohort.csv").shot.astype(int))
    selected = (
        set(
            survey.loc[
                (survey.n_bolo > 1) & ((survey.n_tangtv > 1) | (survey.n_langmuir > 1)),
                "shot",
            ].astype(int)
        )
        & cohort
    )
    inversion = set(map(int, (ROOT / "shots_inversion.txt").read_text().split()))
    examples = {180257, 180264, 189648, 189651}
    actual = set(map(int, (ROOT / "shots_fetch.txt").read_text().split()))
    assert selected | inversion | examples == actual
    return {
        "count": len(actual),
        "cohort_selected": len(selected),
        "inversions": len(inversion),
        "cohort_inversion_overlap": len(selected & inversion),
        "additional_examples": sorted(examples),
        "recipe": "Union of fixed cohort with real bolo (xdata>1) and real TangTV OR Langmuir (xdata>1), all local inversions, plus four explicit examples. Group presence/stubs do not count. No state labels/scores used.",
        "survey": str(ROOT / "survey/corpus_survey.csv"),
        "shot_list": str(ROOT / "shots_fetch.txt"),
    }


def fetch_audit():
    probes = [
        json.loads(line)
        for line in (ROOT / "processed_probes_log.jsonl").read_text().splitlines()
    ]
    geometry = [
        json.loads(line)
        for line in (ROOT / "geometry02_log.jsonl").read_text().splitlines()
    ]
    flux = {}
    for path in (ROOT / "efit").glob("*.npz"):
        with np.load(path) as f:
            flux[path.stem] = {
                "source": str(f.get("source", "EFIT01")),
                "slices": len(f["gtime_ms"]),
                "lim_available": "lim" in f,
            }
    return {
        "processed_probe_attempts": len(probes),
        "processed_probe_shots_with_positioned_current": sum(
            row.get("n_probes", 0) > 0 for row in probes
        ),
        "geometry_attempts": len(geometry),
        "geometry_status": dict(Counter(row["status"] for row in geometry)),
        "flux_maps": flux,
        "auth_stop": (ROOT / "fetch_auth_stop").exists(),
        "tree_and_pointname_attempts": str(ROOT / "processed_probe.json"),
        "node_read_attempts": json.loads((ROOT / "node_read.json").read_text()),
        "bolometer_geometry": "No chord geometry in surveyed BOLOM tree/local resources; row omitted, no invented chord rays.",
    }


def main():
    frame = pd.read_csv(ROOT / "labels_bins.csv.gz")
    bins = dl.load_all(ROOT / "bins")
    split = dl.cohort_split(bins.shot.unique())
    bins["split"] = bins.shot.map(split)
    votes, valid = dl.matrices(frame)
    state = frame.state_lm.to_numpy()
    certain = np.isin(state, (1, 2, 3))
    conflict = label_model.rule(votes, valid) == core.UNCERTAIN
    audit = {
        "certain_below_threshold": int(
            np.sum(certain & (frame.confidence.to_numpy() < 0.7))
        ),
        "certain_conflict": int(np.sum(certain & conflict)),
        "certain_without_valid_tangtv_vote": int(
            np.sum(certain & (~valid[:, 2] | (votes[:, 2] <= 0)))
        ),
        "marfe_without_tangtv_marfe_vote": int(
            np.sum((state == 3) & (votes[:, 2] != 3))
        ),
        "marfe_without_spatial_or_second_cue": int(
            np.sum(
                (state == 3)
                & (
                    ~frame.tangtv_marfe_spatial.to_numpy()
                    | ~frame.tangtv_marfe_second_cue.to_numpy()
                )
            )
        ),
        "tangtv_valid_elm_majority": int(
            np.sum(valid[:, 2] & (frame.aux_elm_share.to_numpy() > 0.5))
        ),
        "temporal_imputations": int(
            np.sum(frame.state_temporal_imputation.to_numpy() != state)
        ),
    }
    assert all(v == 0 for k, v in audit.items() if k != "temporal_imputations"), audit
    by_source = {}
    for source in ("inversion", "surrogate", "none"):
        for envelope in (True, False):
            selection = (bins.tangtv_source.to_numpy() == source) & (
                bins.tangtv_in_envelope.to_numpy() == envelope
            )
            by_source[f"{source}_{'inside' if envelope else 'outside'}"] = {
                **population(bins, selection),
                "valid_bins": int(np.sum(selection & bins.tangtv_valid.to_numpy())),
                "candidate_bins": int(
                    np.sum(selection & bins.tangtv_marfe_candidate.to_numpy())
                ),
                "marfe_votes": int(
                    np.sum(selection & (bins.tangtv_vote.to_numpy() == 3))
                ),
                "marfe_labels": int(
                    np.sum(
                        (frame.tangtv_source.to_numpy() == source)
                        & (frame.tangtv_in_envelope.to_numpy() == envelope)
                        & (state == 3)
                    )
                ),
            }
    lower = bins.tangtv_source.eq("inversion") & (bins.aux_zvsod < -1.30)
    lower_rows = bins.loc[lower].copy()
    lower_rows["state_lm"] = 4
    model = bench.load_model()
    references = {
        n: bench.loo_state(lower_rows, n, model, 0.7) for n in label_model.LF_NAMES
    }
    lower_checks = {}
    for name in label_model.LF_NAMES:
        tables, _, counts = bench.per_shot_tables(lower_rows, name, references[name])
        lower_checks[name] = bench.bootstrap(
            tables, counts, name, np.random.default_rng(0)
        )
    records = {
        "audit": audit,
        "population": population(frame, np.ones(len(frame), bool)),
        "certain": population(frame, certain),
        "tiers": {str(k): int(v) for k, v in frame.tier.value_counts().items()},
        "marfe_by_source_envelope": by_source,
        "geometry_source_bins": bins.tangtv_efit_source.value_counts().to_dict(),
        "afrac_methods": bins.afrac_method.value_counts().to_dict(),
        "legacy_anchor_sensitivity": legacy_sensitivity(),
        "legacy_populations": snapshot_legacy_populations(),
        "current_anchor_sensitivity": current_sensitivity(frame),
        "prad_development_validation": development_validation(bins),
        "lower_shelf": {
            **population(bins, lower),
            "all_inversion_shots_in_lower_geometry": sorted(
                int(s) for s in bins.loc[lower, "shot"].unique()
            ),
            "valid": population(bins, lower & bins.tangtv_valid),
            "votes": bins.loc[lower, "tangtv_vote"].value_counts().to_dict(),
            "method": "SSA rx<=R<1.37; lower-shelf strike Z; >=10 cm leg; same ELM/MARFE gates",
            "loo_other_indicators": lower_checks,
            "te_check": "withdrawn, no geometrically localised processed DTS",
        },
        "candidate_selection": candidate_selection(),
        "fetch": fetch_audit(),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "detachment_fix.json").write_text(json.dumps(records, indent=1))
    paper = {
        "task": "detachment",
        "bin_ms": 50,
        "reference": "unverified redundant diagnostic consensus; no expert truth",
        "legacy": [
            {
                "indicator": "Afrac (Eldon 2021/2022)",
                "setting": "nearest outer-target processed Jsat; attached pre-puff C, separate L/H; DOD=1/Afrac",
                "reproduced": bool((bins.afrac_method == "eldon_pre_puff_LH").any()),
                "coverage": population(
                    bins, bins.afrac_valid & bins.afrac_method.eq("eldon_pre_puff_LH")
                ),
            },
            {
                "indicator": "Prad,div (Eldon 2019)",
                "setting": "published calibrated multi-chord lower-divertor sensor; no universal published state threshold",
                "reproduced": False,
                "coverage": {"bins": 0, "shots": 0},
            },
            {
                "indicator": "TangTV (Chen 2026)",
                "setting": "C-III SSA height; DZ cliff about 0.5; upper shelf; height>1 is candidate MARFE",
                "reproduced": True,
                "coverage": population(
                    bins,
                    bins.tangtv_valid & bins.tangtv_source.eq("inversion") & ~lower,
                ),
            },
        ],
        "Tokamak-SI": {
            "setting": "compatible redundant votes plus posterior>=0.7, TangTV support, geometry/ELM gates, MARFE spatial+persistent+second cue; proxy pair uncertain",
            "assessed": population(frame, np.ones(len(frame), bool)),
            "certain": population(frame, certain),
            "state_counts": {
                core.STATE_NAMES[k]: int(np.sum(state == k)) for k in (1, 2, 3, 4)
            },
            "tier_counts": records["tiers"],
        },
        "local_proxies": {
            "jsat": population(
                bins, bins.afrac_valid & bins.afrac_method.eq("local_proxy")
            ),
            "prad": population(bins, bins.prad_valid),
        },
        "sources": [
            "detachment_fix.json",
            "detachment_benchmark.json",
            "../data/events/detachment/extend_detach_vote/records/label_model.json",
        ],
    }

    # Keep Figure 2 input small; detailed shot IDs are in the full fix audit.
    def trim(x):
        if isinstance(x, dict):
            return {k: trim(v) for k, v in x.items() if k != "shot_ids"}
        if isinstance(x, list):
            return [trim(v) for v in x]
        return x

    (RESULTS / "detachment_figure2.json").write_text(json.dumps(trim(paper), indent=1))
    print(
        json.dumps(
            {
                "audit": audit,
                "population": records["population"],
                "certain": records["certain"],
                "tiers": records["tiers"],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
