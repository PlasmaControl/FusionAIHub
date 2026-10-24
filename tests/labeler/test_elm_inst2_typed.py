"""`scripts/labeler/elm_inst2_typed.py`: the development labels of each detection, typed
submissions and the second lock."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from labeler.config import Paths
from labeler.elm import challenge, inputs

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "labeler"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location(
        "elm_inst2_typed", SCRIPTS / "elm_inst2_typed.py"
    )
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


def spans(rows):
    return pd.DataFrame(rows, columns=["t_start", "t_end", "kind"])


def test_detections_label_each_source(script):
    x = np.zeros((inputs.N_CHANNELS, 500 * inputs.CELLS_PER_MS), np.float32)
    x[inputs.VALID] = 1
    data = {
        1: SimpleNamespace(
            source="smith",
            x=x,
            truth=np.array([100.0, 300.0]),
            window_t0=np.array([50.0]),
            window_t1=np.array([250.0]),
        ),
        2: SimpleNamespace(
            source="review", x=x, spans=spans([(0, 200, "absent"), (200, 400, "crowd")])
        ),
        3: SimpleNamespace(
            source="shousha",
            x=x,
            onsets=np.array([151.0, 121.0, 251.0]),
            spans=spans(
                [
                    (0, 100, "absent"),
                    (100, 200, "crowd"),
                    (120, 130, "non_crowd"),
                    (150, 160, "non_crowd"),
                    (250, 260, "non_crowd"),
                ]
            ),
        ),
    }
    candidates = {
        # 101: Smith's onset found; 200: unmatched in his window; 290: outside it;
        # 40: below the threshold
        1: (np.array([40.0, 101.0, 200.0, 290.0]), np.array([0.2, 0.9, 0.9, 0.9])),
        # 50: guarded absent time; 300: present, unjudged
        2: (np.array([50.0, 300.0]), np.array([0.9, 0.9])),
        # 151.5 and 121.5 in his crowd, 251 outside it, 50 in absent time, 180 an
        # empty present cell; given out of order
        3: (
            np.array([251.0, 151.5, 50.0, 121.5, 180.0]),
            np.array([0.9, 0.9, 0.9, 0.9, 0.9]),
        ),
    }
    det = script.detections(data, candidates, {1: 0, 2: 1, 3: 2}, 0.5)
    got = {
        (int(s), float(t)): lab
        for s, t, lab in zip(det["shot"], det["t"], det["label"], strict=True)
    }
    assert got == {
        (1, 101.0): "large",
        (1, 200.0): "false",
        (1, 290.0): "",
        (2, 50.0): "false",
        (2, 300.0): "",
        (3, 50.0): "false",
        (3, 121.5): "small",
        (3, 151.5): "small",
        (3, 180.0): "false",
        (3, 251.0): "elm",
    }
    assert det["X"].shape == (10, 3)
    assert det["fold"].tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]


def test_typed_submission_reports_each_event_as_its_type(script):
    record = script.typed_submission(
        7, [[30.0, 31.1], [10.0, 11.1]], [1, 2], (0.0, 50.0), method="m", comment=""
    )
    starts = challenge.flag_spans(record)[:, 0]
    t = np.asarray(record["time"])
    ty = np.asarray(record["event_type"])
    assert starts.tolist() == [10.0, 30.0]
    assert [int(ty[t == s][0]) for s in starts] == [2, 1]
    assert ty[0] == 0 and ty[-1] == 0
    assert script.read_types(record) == [2, 1]
    with pytest.raises(ValueError, match="types"):
        script.typed_submission(7, [[1.0, 2.0]], [1, 2], (0, 5), method="m", comment="")
    with pytest.raises(ValueError, match="length"):
        script.read_types({**record, "event_type": record["event_type"][:-1]})


def test_frozen_sees_a_changed_graded_file(script, paths):
    signals = paths.root / "benchmarks/elm/elmo/signals"
    as2 = paths.root / script.r1.SUBMISSIONS / script.AS2
    signals.mkdir(parents=True)
    as2.mkdir(parents=True)
    (signals / "7.npz").write_bytes(b"input")
    (as2 / "DIII-D_7.json").write_text("{}")
    before = script.frozen(paths, [7])
    assert set(before) == {"code", "key_signals", script.AS2}
    assert script.frozen(paths, [7]) == before
    (as2 / "DIII-D_7.json").write_text('{"changed": 1}')
    assert script.frozen(paths, [7])[script.AS2] != before[script.AS2]


def test_the_second_lock_stops_round_two(script, paths, tmp_path, monkeypatch):
    lock = script.r1.work(paths) / script.LOCK
    lock.parent.mkdir(parents=True)
    script.r1.claim(lock, {"job": "1"})
    assert json.loads(lock.read_text()) == {"job": "1"}
    monkeypatch.setattr(script, "KEY_OUT", tmp_path / "out" / "key_typed.json")
    with pytest.raises(SystemExit, match=r"key2\.lock exists"):
        script.run_key(paths, None)
    with pytest.raises(SystemExit, match=r"key2\.lock exists"):
        script.run_evaluate(paths, None)
