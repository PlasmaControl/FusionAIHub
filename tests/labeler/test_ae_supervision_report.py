"""The swap report reads paired differences in the direction its prose states."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def report(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts/labeler"
    monkeypatch.syspath_prepend(str(scripts))
    return importlib.import_module("ae_supervision_swap_report")


@pytest.fixture
def swap(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts/labeler"
    monkeypatch.syspath_prepend(str(scripts))
    return importlib.import_module("ae_supervision_swap")


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
    assert "about 0.005 AUROC (within roughly 1 seed SD)" in text
    assert "0.003 AUPRC (about 2 seed SD) on the 60 shots" in text
    assert "point estimates, not bounds" in text and "at most about" not in text
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
    assert "about 0.000 AUROC (within roughly 1 seed SD)" in text
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
    assert "1 accepted record fails rule 2" in report.float32_screen_note(record)


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


def test_tex_minus_signs_only_replace_hyphens_before_numbers(report):
    text = r"\texttt{ae-ours} -0.055 [-0.149, 0.036] 1e-06 2026-10-03 seed-1"
    got = report.tex_minus(text)
    assert got == (
        r"\texttt{ae-ours} $-$0.055 [$-$0.149, 0.036] 1e-06 2026-10-03 seed-1"
    )
    main = report.main_table(synthetic())
    assert "-0." not in main.split(r"\midrule")[1]


def test_saved_detector_label_has_no_nested_brackets(report):
    label, training = report.display("ae-rcn")
    row = f"{label} ({training})"
    assert "801 shots" in row
    assert row.count("(") == 1 and row.count(")") == 1


def test_where_text_names_cohort_and_reference_in_prose(report):
    got = report.where_text("ae-ours-legacy-seed2, fair_19, legacy reference")
    assert got == (
        "ae-ours-legacy-seed2, 19 shared held-out shots, legacy annotation reference"
    )
    assert "_" not in report.where_text("ae-ours-legacy-seed1, all_60, dense reference")


def test_paired_rows_are_sorted_by_name(report):
    diffs = {
        name: {m: score(0.1, [0.0, 0.2]) for m in ("auroc", "auprc", "f1")}
        for name in ("b minus c", "a minus c", "a minus b")
    }
    record = {
        "results": {
            "all_60": {
                "references": {"dense": {"seed_summary": {"paired_differences": diffs}}}
            }
        }
    }
    rows = report.paired_rows(record, "all_60", "dense")
    assert [row.split(" | ")[0] for row in rows] == [
        "| a minus b",
        "| a minus c",
        "| b minus c",
    ]


def test_lstm_seed_note_counts_resolved_seeds_and_the_seed_interval(report):
    def record(seed_low):
        dense = reference(0, lstm=(0.07, [0.001, 0.14]))
        dense["seed_summary"]["paired_differences"]["ae-ours-legacy minus ae-lstm"][
            "auroc"
        ]["ci95"] = [seed_low, 0.15]
        dense["paired_differences"] = {
            f"ae-ours-legacy-seed{seed} minus ae-lstm": {"auroc": {"ci95": [low, 0.2]}}
            for seed, low in enumerate((0.02, -0.04, 0.01))
        }
        return {
            "convergence": {"accepted_seeds": {"legacy": [0, 1, 2]}},
            "results": {"fair_19": {"references": {"dense": dense}}},
        }

    assert report.lstm_seed_note(record(-0.008)) == (
        "2 of 3 seeds; not resolved once seed variance is included"
    )
    assert report.lstm_seed_note(record(0.01)) == "2 of 3 seeds"
    assert report.lstm_seed_note({"results": record(0)["results"]}) == ""


def test_band_source_cites_the_file_time_and_the_commit(report):
    band = {
        "source": "BAND_KHZ comment in alfven.py",
        "file_modified_utc": "2026-09-30T14:24:35+00:00",
        "commit": "43279081",
        "commit_utc": "2026-10-02T03:14:23+00:00",
    }
    record = {
        "dense_reconciliation": {"display_band_change": band},
        "dense_history": {
            "by_name": {"Alvin Garcia": {"last_change": "2026-10-01T21:39:39+00:00"}}
        },
    }
    text = report.band_source(record)
    assert "last modified 2026-09-30 14:24:35 UTC" in text
    assert "committed in 43279081 on 2026-10-02 03:14:23 UTC, after A. Garcia" in text
    band.update(file_modified_utc=None, commit=None)
    assert report.band_source(record) == "BAND_KHZ comment in alfven.py"


def test_provenance_separates_own_saves_from_the_batch_confirmation(report):
    def person(saves, shots, confirmed, **extra):
        return {
            "entries": saves + confirmed,
            "saves": saves,
            "shots_saved": shots,
            "confirmation_entries": confirmed,
            "shots_confirmed": confirmed,
            "first_saved": "2026-10-01T17:00:00+00:00",
            "last_saved": "2026-10-01T18:00:00+00:00",
            "interval_changing_saves": 29,
            "first_change": "2026-10-01T21:22:07+00:00",
            "last_change": "2026-10-01T21:39:39+00:00",
            "changed_shots": {"train": 22, "selection": 6, "evaluation": 0},
            **extra,
        }

    record = {
        "dense_history": {
            "entries": 400,
            "shots": 180,
            "logins": ["server"],
            "by_name": {
                "(unnamed)": person(208, 180, 0),
                "Alvin Garcia": person(30, 28, 180),
            },
            "notes": {"confirmed by Garcia": 180},
            "confirmation_times": ["2026-10-01T03:55:52+00:00"],
            "sources": ["table.csv"],
        },
        "dense_reconciliation": {
            "display_band_change": {
                "before": "80-250 kHz",
                "after": "60-250 kHz",
                "date": "2026-09-30",
            },
            "differing_shots_by_name": {
                "Alvin Garcia": {"train": 22, "selection": 6, "evaluation": 0}
            },
            "difference": {"train_selection_120": {"differing_shots": list(range(29))}},
        },
    }
    text = report.provenance_text(record)
    assert "a confirmation in his name was recorded for all 180 shots at the" in text
    assert "his own 30 saves cover 28 shots" in text
    assert "he saved all" not in text
    assert "Alvin Garcia 30 saves on 28 shots and 180 confirmation entries" in text
    assert "unnamed 208 saves on 180 shots" in text and "(unnamed)" not in text
    assert "all written at 2026-10-01 03:55:52 UTC" in text


def test_frame_sensitivity_names_the_arm_with_the_most_disagreeing_shots(report):
    def run(n):
        return {
            "fp32": {"frame_probability_difference": {"evaluation_shots_over_0.1": n}}
        }

    counts = {"legacy": (12, 37, 43), "dense": (8, 2, 1), "threeway": (0, 0, 3)}
    record = {
        "convergence": {"accepted_seeds": {arm: [0, 1, 2] for arm in counts}},
        "runs": {
            f"ae-ours-{arm}-seed{seed}": run(n)
            for arm, values in counts.items()
            for seed, n in enumerate(values)
        },
    }
    text = report.frame_sensitivity_text(record)
    assert (
        "sits in the legacy arm: its three seeds have 12, 37 and 43 of the 60" in text
    )
    assert "at most 8 in any other arm" in text
    assert "legacy epochs were selected under bfloat16" in text
    counts["dense"] = (50, 2, 1)
    record["runs"]["ae-ours-dense-seed0"] = run(50)
    assert "legacy epochs" not in report.frame_sensitivity_text(record)


def test_interrupted_launch_is_named_but_never_scored(report):
    item = {
        "attempt_status": "running",
        "has_run_json": False,
        "epochs_recorded": 3,
        "completed_run": "models/threeway/seed-0",
        "first_epochs_match_completed_run": True,
    }
    record = {"interrupted_runs": {"ae-ours-threeway-seed0-interrupted": item}}
    text = " ".join(report.interrupted_lines(record))
    assert "never scored" in text and "stopped after 3 recorded epochs" in text
    assert "no `run.json`" in text and "reproduce its validation losses exactly" in text
    assert report.interrupted_lines({}) == []


def test_dense_history_counts_a_persons_saves_apart_from_the_confirmation(
    swap, tmp_path
):
    def entry(shot, when, name=None, intervals=((0, 10, 1),), note=None):
        row = {
            "shot": shot,
            "reviewer": "server",
            "saved_at": when,
            "window": [0, 2000],
            "intervals": [list(i) for i in intervals],
            "source": "table.csv",
        }
        if name:
            row["name"] = name
        if note:
            row["note"] = note
        return row

    confirm = "2026-10-01T03:55:52+00:00"
    rows = [entry(s, f"2026-09-23T10:00:0{s}+00:00") for s in (1, 2)]
    rows += [entry(s, confirm, "Alvin Garcia", note="checked") for s in (1, 2)]
    rows += [entry(1, "2026-10-01T21:30:00+00:00", "Alvin Garcia", ((0, 20, 1),))]
    path = tmp_path / "history.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    manifest = {
        "inputs": {"dense": {"path": str(tmp_path / "labels.csv")}},
        "split": {"train": [1], "selection": [2], "evaluation": []},
    }
    history = swap.dense_history(manifest)
    garcia = history["by_name"]["Alvin Garcia"]
    assert garcia["entries"] == 3 and garcia["saves"] == 1
    assert garcia["shots_saved"] == 1 and garcia["shots_confirmed"] == 2
    assert garcia["confirmation_entries"] == 2
    assert garcia["first_saved"] == garcia["last_saved"] == "2026-10-01T21:30:00+00:00"
    assert garcia["interval_changing_saves"] == 1
    assert history["by_name"]["(unnamed)"]["confirmation_entries"] == 0
    assert history["confirmation_times"] == [confirm]


def test_interrupted_runs_are_found_beside_the_completed_run(swap, tmp_path):
    def training(*losses):
        return json.dumps({"history": [{"valid_val_loss": v} for v in losses]})

    base = tmp_path / "models/threeway"
    (base / "seed-0-interrupted").mkdir(parents=True)
    (base / "seed-0").mkdir()
    attempt = {
        "supervision": "threeway",
        "seed": 0,
        "status": "running",
        "generated": "2026-10-03T13:24:23+00:00",
    }
    (base / "seed-0-interrupted/attempt.json").write_text(json.dumps(attempt))
    (base / "seed-0-interrupted/training_x.json").write_text(training(2.7, 2.0))
    (base / "seed-0/training_x.json").write_text(training(2.7, 2.0, 1.3))
    found = swap.interrupted_runs(tmp_path)["ae-ours-threeway-seed0-interrupted"]
    assert found["epochs_recorded"] == 2 and found["has_run_json"] is False
    assert found["first_epochs_match_completed_run"] is True
    assert found["completed_run"] == "models/threeway/seed-0"
    (base / "seed-0/training_x.json").write_text(training(2.7, 2.1, 1.3))
    again = swap.interrupted_runs(tmp_path)["ae-ours-threeway-seed0-interrupted"]
    assert again["first_epochs_match_completed_run"] is False
