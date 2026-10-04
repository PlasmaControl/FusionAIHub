"""Summary readers retain metrics and can verify external per-shot records."""

import json

import pytest

from labeler.rwm.records import (
    evaluation_details_path,
    load_evaluation,
    write_evaluation,
)


def test_summary_keeps_metrics_and_round_trips_nested_shot_details(tmp_path):
    record = {
        "configs": {
            "forest": {
                "metrics": {"auroc": 0.54},
                "counts": {"shots": 33},
                "per_shot": [{"shot": 7, "warning_ms": [10]}],
                "within_shot": {"mean": 0.78, "per_shot": [{"shot": 7}]},
                "comparisons": {"n_shots": 2, "shots": [8, 9]},
            }
        }
    }
    summary = tmp_path / "repo" / "evaluation.json"
    details = tmp_path / "large" / "evaluation_details.json"
    write_evaluation(record, summary, details)
    small = load_evaluation(summary)
    forest = small["configs"]["forest"]
    assert forest["metrics"] == record["configs"]["forest"]["metrics"]
    assert forest["counts"] == {"shots": 33}
    assert "per_shot" not in forest and "per_shot" not in forest["within_shot"]
    assert forest["comparisons"] == {"n_shots": 2}
    assert small["external_details"]["path"] == str(details.resolve())
    assert load_evaluation(summary, details=True) == record


def test_external_detail_reader_rejects_changed_payload(tmp_path):
    summary, details = tmp_path / "summary.json", tmp_path / "details.json"
    write_evaluation({"per_shot": [{"shot": 7}]}, summary, details)
    details.write_text("{}\n")
    with pytest.raises(ValueError, match="SHA-256"):
        load_evaluation(summary, details=True)


def test_reader_accepts_older_full_record_without_pointer(tmp_path):
    path = tmp_path / "evaluation.json"
    record = {"per_shot": [{"shot": 7}]}
    path.write_text(json.dumps(record))
    assert load_evaluation(path, details=True) == record


def test_writer_rejects_oversize_summary_before_overwriting_records(tmp_path):
    summary, details = tmp_path / "summary.json", tmp_path / "details.json"
    write_evaluation({"per_shot": [{"shot": 7}]}, summary, details)
    before = summary.read_bytes(), details.read_bytes()
    with pytest.raises(ValueError, match="below 0.6 MB"):
        write_evaluation({"protocol": "x" * 600_000}, summary, details)
    assert (summary.read_bytes(), details.read_bytes()) == before


def test_custom_outputs_with_same_basename_preserve_baseline_details(tmp_path):
    canonical = tmp_path / "repo" / "evaluation.json"
    artifacts = tmp_path / "artifacts"
    paths = [canonical, tmp_path / "a/evaluation.json", tmp_path / "b/evaluation.json"]
    details = [
        evaluation_details_path(path, artifacts, canonical_summary=canonical)
        for path in paths
    ]
    assert details[0] == artifacts / "evaluation_details.json"
    assert len(set(details)) == 3
    records = [{"per_shot": [{"shot": shot}]} for shot in (7, 8, 9)]
    for path, detail, record in zip(paths, details, records, strict=True):
        write_evaluation(record, path, detail)
    assert [load_evaluation(path, details=True) for path in paths] == records


def test_reader_accepts_a_plain_string_path(tmp_path):
    summary, details = tmp_path / "summary.json", tmp_path / "details.json"
    record = {"per_shot": [{"shot": 7}]}
    write_evaluation(record, summary, details)
    assert load_evaluation(str(summary)) == load_evaluation(summary)
    assert load_evaluation(str(summary), details=True) == record
