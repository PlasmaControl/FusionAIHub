"""`scripts/labeler/elm_inst2.py`: his occupancy spans, the threshold rules, the key's
decoding, onset offsets and the lock that keeps the key scored once."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from labeler.config import Paths
from labeler.elm import inputs, inst

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "labeler"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("elm_inst2", SCRIPTS / "elm_inst2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_occupancy_spans_drop_only_the_short_absent_rows(script):
    spans = pd.DataFrame(
        [
            (0, 100, "absent"),
            (100, 110, "non_crowd"),
            (110, 130, "absent"),
            (130, 140, "non_crowd"),
            (140, 300, "crowd"),
            (300, 320, "uncertain"),
        ],
        columns=["t_start", "t_end", "kind"],
    )
    assert script.occupancy_spans(spans).t_start.tolist() == [0, 100, 130, 140, 300]


def test_rule_gains_price_crowd_elms_small_and_count_empty_cells_false(script):
    n = dict.fromkeys(script.TRADEOFF_COUNTS, 0)
    n.update(
        smith_tp=10,
        smith_fp=1,
        review_alarms=2,
        his_tp_outside_crowds=3,
        his_tp_in_crowds=20,
        his_fp=4,
        his_empty=5,
    )
    gains = script.rule_gains(n, 1.0)
    # as trained: 33 ELMs found, all priced large; 7 detections false
    assert gains["gain_as_trained"] == pytest.approx(33 * 6.0 - 7 * 10.5)
    # conservative: 13 large, the 20 in crowds small; the 5 empty ones false too
    assert gains["gain_conservative"] == pytest.approx(13 * 6.0 + 20 * 1.5 - 12 * 10.5)
    # reported as small ELMs (type 2): found ELMs save 7.0 or 4.0, a false one costs 3.0
    t2 = script.rule_gains(n, 1.0, reported=2)
    assert t2["gain_as_trained"] == pytest.approx(33 * 8.0 - 7 * 4.0)
    assert t2["gain_conservative"] == pytest.approx(13 * 8.0 + 20 * 5.0 - 12 * 4.0)


def test_decode_keeps_peaks_in_the_valid_input_and_the_window(script):
    n = 100 * inputs.CELLS_PER_MS
    heat = np.zeros(n, np.float32)
    valid = np.zeros(n, bool)
    valid[: 60 * inputs.CELLS_PER_MS] = True  # valid over the first 60 ms only
    t0 = inputs.GRID0_MS
    for t, h in (
        (t0 + 2.05, 0.9),  # before the window
        (t0 + 20.05, 0.9),  # kept
        (t0 + 21.05, 0.8),  # within 3 ms of a higher peak
        (t0 + 30.05, 0.4),  # below the threshold
        (t0 + 70.05, 0.9),  # outside the valid input
    ):
        heat[inst.cell_of(t)] = h
    served = {"onset_threshold": 0.5, "min_sep_ms": 3.0, "event_duration_ms": 1.1}
    spans = script.decode({"inst": heat, "valid": valid}, served, (t0 + 5, t0 + 90))
    np.testing.assert_allclose(spans, [[t0 + 20.05, t0 + 21.15]])


def test_onset_summary_keeps_signed_and_absolute_offsets(script):
    case = {
        "tstart": 0.0,
        "tend": 100.0,
        "events": [(10.0, 11.0, 1), (20.0, 21.0, 2), (40.0, 45.0, 3)],
    }
    found = np.array([[11.0, 12.0], [19.5, 20.5], [41.0, 42.0], [60.0, 61.0]])
    r = script.onset_summary({7: case}, {7: found}, [7], 2.0)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 1, 0)
    assert r["excluded_beside_uncertain_events"] == 1
    assert r["median_signed_offset_ms"] == pytest.approx(0.25)
    assert r["median_abs_offset_ms"] == pytest.approx(0.75)


def test_the_lock_is_claimed_once_and_stops_evaluate_and_key(
    script, paths, tmp_path, monkeypatch
):
    lock = script.work(paths) / script.LOCK
    lock.parent.mkdir(parents=True)
    script.claim(lock, {"job": "1"})
    with pytest.raises(SystemExit, match="scored once"):
        script.claim(lock, {"job": "2"})
    assert json.loads(lock.read_text()) == {"job": "1"}
    monkeypatch.setattr(script, "KEY_OUT", tmp_path / "out" / "key.json")
    with pytest.raises(SystemExit, match=r"key\.lock exists"):
        script.run_key(paths, None)
    with pytest.raises(SystemExit, match=r"key\.lock exists"):
        script.run_evaluate(paths, None)
