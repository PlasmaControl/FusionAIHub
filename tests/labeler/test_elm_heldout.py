"""The frozen-model evaluation on a later review: pins, provenance, onsets and bins."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from labeler.config import Paths, sha256_of
from labeler.elm import heldout, inputs, onset, prepare, train

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "labeler"
HEADER = "shot,t_start,t_end,category,attrs\n"


@pytest.fixture
def paths(tmp_path):
    tables = tmp_path / "events"
    tables.mkdir()
    (tmp_path / "config").mkdir()
    return Paths(
        root=tmp_path / "root",
        corpus=tmp_path / "corpus",
        text_root=tmp_path / "text",
        logs_jsonl=tmp_path / "logs.jsonl",
        label_tables=tables,
        config_dir=tmp_path / "config",
        raw_cache=tmp_path / "raw",
    )


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _pin(paths, name, labels="shot,category\n1,0\n"):
    directory = prepare.pins_dir(paths) / name
    _write(directory / "labels.csv", labels)
    _write(directory / "history.jsonl", "")
    sums = "".join(
        f"{sha256_of(directory / f)}  {f}\n" for f in ("labels.csv", "history.jsonl")
    )
    _write(directory / "SHA256SUMS", sums)
    return directory


def test_the_pin_is_the_default_the_live_table_is_asked_for_and_a_missing_pin_is_an_error(
    paths,
):
    live = _write(prepare.live_review_csv(paths), "shot,category\n9,0\n")
    with pytest.raises(FileNotFoundError, match="no pinned ELM review labels"):
        prepare.review_csv(paths)
    pinned = _pin(paths, prepare.REVIEW_PIN) / "labels.csv"
    assert prepare.review_csv(paths) == pinned
    assert prepare.review_csv(paths, None) == live
    with pytest.raises(FileNotFoundError):
        prepare.review_csv(paths, "2099-01-01")


def _prepared(paths, shots):
    """A prepared input (`round4/elm/inputs/<shot>.npy`) of 150 ms for each shot."""
    folder = prepare.inputs_dir(paths)
    folder.mkdir(parents=True, exist_ok=True)
    for shot in shots:
        x = np.zeros((inputs.N_CHANNELS, 1500), dtype=np.float32)
        x[inputs.VALID] = 1
        np.save(folder / f"{shot}.npy", x)


def test_load_names_every_reviewed_shot_without_an_input_and_a_shot_filter_leaves_it_out(
    paths,
):
    _pin(paths, prepare.REVIEW_PIN, HEADER + "".join(f"{s},0,100,0,\n" for s in (1, 2, 3)))
    paths.catalog.mkdir(parents=True)
    pd.DataFrame({"shot": [1, 2, 3], "split": ["train"] * 3}).to_csv(
        paths.catalog / "cohort.csv", index=False
    )
    _prepared(paths, [1, 2])  # the table holds a shot no one has prepared an input for
    with pytest.raises(FileNotFoundError, match=r"reviewed shots \[3\]"):
        train.load(paths)
    assert sorted(train.load(paths, shots=[1, 2])) == [1, 2]
    other = _write(paths.root / "other.csv", HEADER + "1,0,100,0,\n")
    assert sorted(train.load(paths, review=other)) == [1]


def test_elm_inst_prepare_refuses_a_holdout_shot_before_it_reads_a_signal(paths, monkeypatch):
    _pin(paths, prepare.REVIEW_PIN, HEADER + "1,0,100,0,\n7,0,100,0,\n")
    _write(paths.shotlists_manifest, "lists:\n  key: {protect: holdout}\n")
    _write(paths.shotlists / "key.txt", "7\n")
    spec = importlib.util.spec_from_file_location("elm_inst", SCRIPTS / "elm_inst.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    def unread(*args, **kwargs):
        raise AssertionError("a shot was read before the holdout check")

    monkeypatch.setattr(script.challenge, "clock_spans", unread)
    with pytest.raises(ValueError, match=r"held-out challenge shots may not be used: \[7\]"):
        script.run_prepare(paths, None)


def test_a_pin_is_checked_against_its_sums_and_a_changed_file_is_refused(paths):
    directory = _pin(paths, "2026-10-23")
    assert set(heldout.verify_pin(directory)) == {"labels.csv", "history.jsonl"}
    (directory / "labels.csv").chmod(0o644)
    with (directory / "labels.csv").open("a") as fh:
        fh.write("2,0\n")
    with pytest.raises(ValueError, match="SHA256SUMS"):
        heldout.verify_pin(directory)


def test_training_refuses_a_holdout_shot_and_any_shot_when_the_registry_is_unreadable(
    paths,
):
    _write(paths.shotlists_manifest, "lists:\n  key: {protect: holdout}\n")
    _write(paths.shotlists / "key.txt", "7\n")
    train.check_no_holdout([1, 2], paths)
    with pytest.raises(ValueError, match=r"\[7\]"):
        train.check_no_holdout([1, 7], paths)
    _write(paths.shotlists / "key.txt", "7\n7\n")  # unreadable: fails closed
    with pytest.raises(ValueError, match="held-out"):
        train.check_no_holdout([1], paths)


def test_last_saves_keep_the_last_of_one_reviewer_and_count_the_saves(tmp_path):
    lines = [
        {"shot": 1, "name": "Ricardo Shousha", "window": [0, 10], "n": 1},
        {"shot": 1, "name": "someone else", "window": [0, 10], "n": 2},
        {"shot": 1, "name": "Ricardo Shousha", "window": [0, 20], "n": 3},
        {"shot": 2, "name": None, "window": [0, 10], "n": 4},
    ]
    path = tmp_path / "history.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    saves = heldout.last_saves(path)
    assert list(saves) == [1] and saves[1]["window"] == [0, 20] and saves[1]["saves"] == 2


def test_an_individual_row_is_the_accepted_prefill_when_it_is_a_prefilled_onsets_cell():
    onsets = [103.2, 250.0, 301.9, 999.0]  # the last is outside the save's window
    rows = pd.DataFrame(
        {"t_start": [100.0, 250.0, 300.0, 410.0], "t_end": [110.0, 260.0, 310.0, 420.0]}
    )
    out, removed = heldout.individual_provenance(rows, onsets, (0.0, 500.0))
    assert out.provenance.tolist() == [
        "prefill_accepted",
        "prefill_accepted",
        "prefill_accepted",
        "edited",
    ]
    assert removed == 0
    out, removed = heldout.individual_provenance(rows.iloc[[0]], onsets, (0.0, 500.0))
    assert out.provenance.tolist() == ["prefill_accepted"] and removed == 2
    assert rows.columns.tolist() == ["t_start", "t_end"]  # the input is not changed


def test_span_agreement_is_what_two_sets_of_spans_share_and_nan_when_one_is_empty():
    draft = pd.DataFrame({"t_start": [0.0, 100.0], "t_end": [50.0, 150.0]})
    his = pd.DataFrame({"t_start": [25.0], "t_end": [125.0]})
    both = heldout.span_agreement(draft, his)
    assert both["source_ms"] == 100.0 and both["his_present_ms"] == 100.0
    assert both["shared_ms"] == 50.0 and both["precision"] == both["recall"] == 0.5
    empty = heldout.span_agreement(draft.iloc[:0], his)
    assert empty["source_ms"] == 0.0 and np.isnan(empty["precision"])
    assert empty["recall"] == 0.0


def test_match_onsets_is_the_benchmarks_greedy_rule_with_flags_in_input_order():
    rng = np.random.default_rng(3)
    defined = np.ones(2000, bool)
    for _ in range(20):
        truth = rng.uniform(0, 900, rng.integers(0, 12))
        found = rng.uniform(0, 900, rng.integers(0, 12))
        for tol in (2.0, 10.0, 25.0):
            tp, fp, fn = onset.match(found, truth, defined, inputs.GRID0_MS, tol)
            hit, matched = heldout.match_onsets(found, truth, tol)
            assert (int(hit.sum()), int((~matched).sum()), int((~hit).sum())) == (tp, fp, fn)
    hit, matched = heldout.match_onsets([60.0, 10.0], [11.0, 500.0], 5.0)
    assert hit.tolist() == [True, False] and matched.tolist() == [False, True]


def _spans(rows):
    return pd.DataFrame(rows, columns=["t_start", "t_end", "kind"])


def test_onsets_are_judged_where_a_row_can_answer_and_false_in_absent_time_is_counted():
    spans = _spans(
        [
            (0.0, 100.0, "absent"),
            (100.0, 200.0, "non_crowd"),
            (200.0, 300.0, "uncertain"),
            (300.0, 400.0, "non_crowd"),
        ]
    )
    defined, absent = heldout.label_masks(spans, 600)
    cell = lambda t: int(t - inputs.GRID0_MS)
    assert defined[cell(50)] and absent[cell(50)] and not defined[cell(250)]
    assert defined[cell(350)] and not absent[cell(350)] and not defined[cell(450)]
    truth = heldout.OnsetTruth(
        np.array([105.0, 305.0, 350.0]), np.array([True, True, False]), defined, absent
    )
    cover = pd.DataFrame({"t_start_ms": [-50.0], "t_end_ms": [500.0]})
    # 106 hits row 105; 20 is false in absent time; 250 is in uncertain time; 330 is false
    # in judged present time; 450 is outside the labelled window
    counts = heldout.onset_counts([106.0, 20.0, 250.0, 330.0, 450.0], truth, cover, 5.0)
    assert dict(zip(heldout.ONSET_COLUMNS, counts.tolist())) == {
        "tp": 1,
        "fp": 2,
        "fn": 2,
        "fp_in_absent": 1,
        "prefill_accepted": 2,
        "prefill_accepted_found": 1,
        "edited": 1,
        "edited_found": 0,
    }
    clipped = pd.DataFrame({"t_start_ms": [-50.0], "t_end_ms": [200.0]})
    counts = heldout.onset_counts([106.0, 330.0], truth, clipped, 5.0)
    assert counts[:3].tolist() == [1, 0, 0]  # rows and detections outside the cover are out


def _shot(n_ms=400):
    valid = np.ones(n_ms * inputs.CELLS_PER_MS, np.float32)
    spans = _spans(
        [
            (0.0, 100.0, "absent"),
            (100.0, 200.0, "crowd"),
            (200.0, 250.0, "uncertain"),
        ]
    )
    return heldout.panel_of(1, spans, valid), n_ms


def test_bins_come_from_crowd_and_absent_rows_only_and_a_method_is_scored_on_them():
    panel, n_ms = _shot()
    assert panel.bins.kind.tolist() == ["absent", "absent", "crowd", "crowd"]
    trace = np.zeros(n_ms)
    trace[int(100 - inputs.GRID0_MS) : int(150 - inputs.GRID0_MS)] = 1.0
    trace[int(50 - inputs.GRID0_MS) : int(100 - inputs.GRID0_MS)] = 1.0
    called = {"ours": {1: heldout.Called(trace, 0.5)}}
    out = heldout.occupancy_set({1: panel}, called)
    assert out["bins"] == 4 and out["bins_by_kind"] == {"absent": 2, "crowd": 2}
    point = out["methods"]["ours"]["point"]
    assert point["crowd_bin_recall"] == 0.5 and point["false_alarm_bin_rate"] == 0.5
    assert out["methods"]["always present"]["point"]["false_alarm_bin_rate"] == 1.0
    spans = pd.DataFrame({"t_start_ms": [120.0], "t_end_ms": [130.0]})
    out = heldout.occupancy_set({1: panel}, {"table": {1: heldout.Called(spans=spans)}})
    point = out["methods"]["table"]["point"]
    assert point["crowd_bin_recall"] == 0.5 and point["false_alarm_bin_rate"] == 0.0


def test_the_bes_panel_keeps_the_bins_inside_the_chunks_elmo_analysed():
    panel, _ = _shot()
    bes = heldout.bes_panel(panel, [(80.0, 160.0), (-50.0, 10.0)])
    assert bes.bins.kind.tolist() == ["crowd"]  # only [100, 150) lies in a chunk
    assert panel.bins.kind.size == 4  # the full panel is not changed


def test_onset_sets_report_recall_of_accepted_and_edited_rows_apart():
    panel, _ = _shot()
    defined, absent = heldout.label_masks(panel.spans, 400)
    truth = {
        1: heldout.OnsetTruth(
            np.array([110.0, 130.0, 170.0]),
            np.array([True, True, False]),
            defined,
            absent,
        )
    }
    found = {"m": {1: np.array([111.0, 169.0, 20.0])}}
    out = heldout.onset_set([1], {1: panel.cover}, truth, found, tolerances=(5.0,))
    row = out["m"]["tolerances_ms"]["5.0"]
    assert row["counts"] == {"tp": 2, "fp": 1, "fn": 1}
    assert row["false_alarms_in_absent"] == 1
    assert row["recall_prefill_accepted"]["point"] == 0.5
    assert row["recall_edited"]["point"] == 1.0
    assert out["m"]["shots_scored"] == [1]


def test_detections_collapse_to_the_10_ms_cells_his_rows_are_on():
    cells = heldout.to_grid([103.2, 101.0, 109.9, 250.0, 119.99])
    assert cells.tolist() == [105.0, 115.0, 255.0]
    assert heldout.to_grid([]).size == 0
    # a method that fires three times in one cell no longer has two false alarms there
    truth = heldout.OnsetTruth(
        np.array([105.0]), np.array([True]), np.ones(400, bool), np.zeros(400, bool)
    )
    cover = pd.DataFrame({"t_start_ms": [-50.0], "t_end_ms": [300.0]})
    raw = heldout.onset_counts([101.0, 103.0, 107.0], truth, cover, 5.0)
    grid = heldout.onset_counts(heldout.to_grid([101.0, 103.0, 107.0]), truth, cover, 5.0)
    assert raw[:3].tolist() == [1, 2, 0] and grid[:3].tolist() == [1, 0, 0]


def test_finite_turns_nan_into_null_and_numpy_into_plain_values():
    value = {1: np.float32(0.5), "a": [np.nan, np.int64(3)], "b": np.array([1.0, np.nan])}
    assert heldout.finite(value) == {"1": 0.5, "a": [None, 3], "b": [1.0, None]}
    json.dumps(heldout.finite(value), allow_nan=False)
