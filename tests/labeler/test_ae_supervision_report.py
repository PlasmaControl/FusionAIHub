"""The swap report reads paired differences in the direction its prose states."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def report(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts/labeler"
    monkeypatch.syspath_prepend(str(scripts))
    return importlib.import_module("ae_supervision_swap_report")


def score(mean, shot_ci):
    return {"mean": mean, "ci95": [mean - 1, mean + 1], "ci95_shot": shot_ci}


def test_pair_flips_a_stored_difference_and_its_interval(report):
    summary = {
        "paired_differences": {
            "a minus b": {"auroc": score(0.2, [0.1, 0.4])},
        }
    }
    same = report.pair(summary, "a", "b")["auroc"]
    flipped = report.pair(summary, "b", "a")["auroc"]
    assert same["mean"] == 0.2
    assert flipped["mean"] == -0.2
    assert flipped["ci95_shot"] == [-0.4, -0.1]
    assert flipped["ci95"] == [-1.2, 0.8]


def test_verdict_uses_the_shot_interval_only(report):
    assert report.verdict(score(0.3, [0.1, 0.5])) == "leads"
    assert report.verdict(score(-0.3, [-0.5, -0.1])) == "trails"
    assert report.verdict(score(0.0, [-0.1, 0.1])) == "is not resolved from"


def test_clock_profile_names_the_peak_and_the_edges(report):
    prior = [0.0] * 200
    prior[50] = 0.8
    text = report.profile(prior)
    assert "peaks at 505 ms (0.80)" in text
    assert "0.00 over its first and last 100 ms" in text


def test_table_cells_never_hold_both_an_sd_and_an_interval(report):
    seed = {"value": 0.5, "ci95": [0.4, 0.6]}
    mean = {"mean": 0.5, "sd": 0.02, "ci95": [0.4, 0.6], "ci95_shot": [0.45, 0.55]}
    assert "[" in report.point_ci(seed) and report.PM not in report.point_ci(seed)
    assert report.PM in report.mean_sd(mean) and "[" not in report.mean_sd(mean)


def metric(auroc, auprc):
    return {"auroc": {"value": auroc}, "auprc": {"value": auprc}}


def seeded(mean, sd=0.01):
    return {"mean": mean, "sd": sd, "ci95_shot": [mean - 0.1, mean + 0.1]}


def reference(offset, lstm=(0.05, [0.01, 0.09])):
    others = ("ae-lstm", "clock-annotation", "clock-dense")
    arms = ("ae-ours-legacy", "ae-ours-dense", "ae-ours-threeway")
    return {
        "methods": {
            "ae-rcn": metric(0.8 + offset, 0.7),
            "ae-lstm": metric(0.81 + offset, 0.71),
            "clock-annotation": metric(0.75 + offset, 0.6),
            "clock-dense": metric(0.76 + offset, 0.61),
        },
        "seed_summary": {
            "methods": {
                name: {"auroc": seeded(0.85 + offset), "auprc": seeded(0.8)}
                for name in arms
            },
            "paired_differences": {
                **{
                    f"{name} minus ae-rcn": {"auroc": score(0.05, [0.01, 0.09])}
                    for name in (*arms, *others)
                },
                "ae-ours-legacy minus ae-lstm": {"auroc": score(*lstm)},
            },
        },
    }


def synthetic(
    fair_threeway=(0.980, 0.01), all60_threeway=(0.975, 0.0044), auprc=(0.992, 0.989)
):
    record = {
        "dense_reference_coarseness": {
            "shots": 180,
            "single_present_span_in_table": 93,
            "single_run_of_present_frames": 102,
        },
        "results": {
            "fair_19": {
                "references": {"dense": reference(0), "legacy": reference(-0.1)}
            },
            "all_60": {
                "references": {"dense": reference(0), "legacy": reference(-0.1)}
            },
        },
        "precision_check": {
            "max_abs_seed_mean_change": {"threeway": {"auroc": 0.002, "auprc": 0.001}}
        },
    }
    for group, (mean, sd), pr in (
        ("fair_19", fair_threeway, auprc[0]),
        ("all_60", all60_threeway, auprc[1]),
    ):
        methods = record["results"][group]["references"]["dense"]["seed_summary"][
            "methods"
        ]
        methods["ae-ours-threeway"] = {
            "auroc": seeded(mean, sd),
            "auprc": seeded(pr, 0.0014),
        }
    return record


def test_one_name_for_the_legacy_annotation(report):
    assert dict(report.REFS)["legacy"] == "legacy annotation"
    assert report.pretty("clock-annotation") == "clock-legacy-annotation"
    assert report.CLOCKS["clock-annotation"] == "clock from legacy annotation"
    assert "(" not in report.CLOCK_TRAINING and "input-free" in report.CLOCK_TRAINING


def test_main_table_is_compact_and_carries_both_references(report):
    text = report.main_table(synthetic())
    body = text.split(r"\midrule")[1].split(r"\bottomrule")[0].strip().splitlines()
    assert len(body) == 7
    assert r"\texttt{ae-ours}" in body[0] and body[0].count("&") == 6
    assert "±" not in text and r"$\pm$" in body[0]
    assert "Legacy annotation" in text and "Dense reference" in text
    assert "93 of 180" in text and r"\label{tab:ae_supervision_swap_main}" in text
    rcn = next(row for row in body if row.startswith(r"\texttt{ae-rcn}"))
    assert rcn.count("n/a") == 2


def test_selection_text_is_derived_from_the_gaps_and_the_seed_sd(report):
    text = report.selection_text(synthetic())
    assert "exceeds the seed SD for AUROC on 60 shots and AUPRC on 60 shots" in text
    assert "at most about 0.005 AUROC and 0.003 AUPRC on the 60 shots" in text
    assert "only the epoch choice is clean" in text
    assert "100 training shots, the published model 120" in text


def test_selection_text_says_when_no_gap_exceeds_the_seed_sd(report):
    text = report.selection_text(
        synthetic(
            fair_threeway=(0.990, 0.01),
            all60_threeway=(0.990, 0.01),
            auprc=(0.995, 0.995),
        )
    )
    assert "No gap exceeds the seed SD." in text
    assert "at most about 0.000 AUROC" in text
    assert "only the epoch choice is clean" in text


def test_garcia_text_counts_shots_by_group_and_person(report):
    record = {
        "dense_reconciliation": {
            "differing_shots_by_name": {
                "Alvin Garcia": {"train": 22, "selection": 6, "evaluation": 0},
                "Nathaniel Chen": {"train": 5, "selection": 2, "evaluation": 1},
            },
            "difference": {
                "train_selection_120": {"differing_shots": list(range(29))},
            },
        }
    }
    text = report.garcia_text(record)
    assert "28 of the 29 training and selection shots" in text
    assert "22 training, 6 selection" in text
    assert "and on 0 evaluation shots;" in text
    assert "N. Chen on 7 of them and 1 evaluation shot;" in text
    assert "unnamed saves on 0 of them" in text


def test_float32_screen_note_reports_a_failing_record(report):
    rule = {
        "screen": {"within_shot_sd_min": 0.05, "selection_auroc_min_exclusive": 0.6}
    }
    good = {"mean_within_shot_sd": 0.1, "selection_auroc": 0.9}
    bad = {"mean_within_shot_sd": 0.01, "selection_auroc": 0.9}
    record = {
        "convergence": {"rule": rule},
        "runs": {"a": {"fp32_screen": good}, "b": {"fp32_screen": good}},
    }
    assert "2 of 2 accepted records pass rule 2" in report.float32_screen_note(record)
    record["runs"]["b"] = {"fp32_screen": bad}
    assert "1 accepted records fail rule 2" in report.float32_screen_note(record)


def test_findings_flag_a_marginal_resolution_against_ae_lstm(report):
    def record(low):
        dense = reference(0, lstm=(0.07, [low, 0.14]))
        legacy = reference(-0.1, lstm=(-0.15, [-0.25, -0.05]))
        return {
            "results": {"fair_19": {"references": {"dense": dense, "legacy": legacy}}}
        }

    marginal = report.findings(record(0.001))
    clear = report.findings(record(0.03))
    assert marginal["lstm_resolved"] and marginal["lstm_marginal"]
    assert clear["lstm_resolved"] and not clear["lstm_marginal"]
