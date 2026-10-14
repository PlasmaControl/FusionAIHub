"""Hermetic checks of `labeler.elm.serve`: the ensemble, the 10 ms table, the input check.

Synthetic records and randomly initialised networks only; no data root, no GPU.
"""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from labeler.elm import inputs, net, serve, train
from labeler.events.interval_tables import validate_intervals


def _record(n_ms=300, gain=1.0, density=3e14):
    t_fs = np.arange(n_ms * 50) / 50.0
    rng = np.random.default_rng(0)
    fs = gain * 1e15 * (1 + 0.05 * rng.random((3, t_fs.size)))
    t_int = np.arange(n_ms * 100) / 100.0
    ne = np.full((2, t_int.size), density)
    return t_fs, fs, t_int, ne


def _models(n=2):
    out = []
    for seed in range(n):
        torch.manual_seed(seed)
        out.append(net.ElmUNet().eval())
    return out


def test_ensemble_is_the_mean_of_the_fold_models():
    x = inputs.channels(*_record())
    models = _models()
    mean = serve.ensemble(models, x)
    each = [train.predict(m, x, "cpu")[0] for m in models]
    assert mean.shape == (x.shape[1] // inputs.CELLS_PER_MS,)
    assert np.allclose(mean, (each[0] + each[1]) / 2, atol=1e-6)
    assert not np.allclose(each[0], each[1])


def test_rows_tile_the_valid_window_on_the_grid():
    x = inputs.channels(*_record())
    trace = np.zeros(x.shape[1] // inputs.CELLS_PER_MS, dtype=np.float32)
    trace[int(50 - inputs.GRID0_MS) : int(150 - inputs.GRID0_MS)] = 1.0
    rows = serve.shot_rows(7, trace, x, 0.4)
    assert list(rows.category) == [0, 1, 0]
    assert rows.t_start.iloc[0] == 0.0
    assert (rows.t_start.iloc[1:].to_numpy() == rows.t_end.iloc[:-1].to_numpy()).all()
    assert (rows.t_start % 10 == 0).all() and (rows.t_end % 10 == 0).all()
    assert rows.t_start.iloc[1] == pytest.approx(50, abs=10)
    assert rows.t_end.iloc[1] == pytest.approx(150, abs=10)
    assert rows.confidence.iloc[1] > 0.9 and rows.confidence.iloc[0] > 0.9
    assert 0.2 < serve.present_fraction(rows) < 0.5
    assert validate_intervals(rows).shape == rows.shape


def test_a_score_below_the_threshold_is_one_absent_row():
    x = inputs.channels(*_record())
    trace = np.full(x.shape[1] // inputs.CELLS_PER_MS, 0.1, dtype=np.float32)
    rows = serve.shot_rows(7, trace, x, 0.4)
    assert list(rows.category) == [0] and rows.confidence.iloc[0] == pytest.approx(0.9)
    assert serve.present_fraction(rows) == 0.0


def test_input_problems_names_a_missing_chord_and_an_untrained_level():
    reference = np.array(
        [serve.level_medians(inputs.channels(*_record(gain=g))) for g in (0.3, 1, 3)]
    )
    assert serve.input_problems(inputs.channels(*_record()), reference) == []
    t_fs, fs, t_int, ne = _record()
    ne[1] = 1e18  # the failed-digitiser screen zeroes it
    assert any(
        "DENV3F" in p
        for p in serve.input_problems(inputs.channels(t_fs, fs, t_int, ne), reference)
    )
    dimmer = inputs.channels(*_record(gain=0.2))
    assert serve.input_problems(dimmer, reference)
    assert serve.input_problems(dimmer, reference, margin=0.6) == []
    dim = serve.input_problems(inputs.channels(*_record(gain=1e-3)), reference, 0.6)
    assert [p.split()[0] for p in dim] == ["FS02", "FS03", "FS04"]
    assert all("outside the training range" in p for p in dim)


def test_main_publishes_the_servable_shot_and_reports_the_others(tmp_path, monkeypatch):
    run = tmp_path / "round4/elm/cv/cv2"
    for k in range(serve.FOLDS):
        torch.manual_seed(k)
        (run / f"fold{k}").mkdir(parents=True)
        torch.save(net.ElmUNet().state_dict(), run / f"fold{k}/model.pt")
        (run / f"fold{k}/fold.json").write_text(json.dumps({"threshold": 0.5}))
    (run / "run.json").write_text(
        json.dumps({"config": {"dropout": 0.1}, "folds": [[1], [2], [3], [4], [5]]})
    )
    (tmp_path / "round4/elm/inputs").mkdir(parents=True)
    for shot, gain in zip(range(1, 6), (0.3, 1, 3, 1, 1)):
        np.save(
            tmp_path / f"round4/elm/inputs/{shot}.npy",
            inputs.channels(*_record(gain=gain)),
        )
    signals = tmp_path / "benchmarks/elm/elmo/signals"
    signals.mkdir(parents=True)
    for shot, gain in ((100, 1.0), (101, 1e-3)):
        t_fs, fs, t_int, ne = _record(gain=gain)
        np.savez(
            signals / f"{shot}.npz",
            t_fs_ms=t_fs,
            filterscopes=fs,
            t_int_ms=t_int,
            interferometer=ne,
        )
    (signals / "103.missing.json").write_text("{}")
    monkeypatch.setenv("LABELER_ROOT", str(tmp_path))
    out = tmp_path / "out"
    assert (
        serve.main(
            [
                "--name",
                "t",
                "--shots",
                "100",
                "101",
                "102",
                "103",
                "--out-dir",
                str(out),
            ]
        )
        == 0
    )
    table = validate_intervals(pd.read_csv(out / "t.csv"))
    assert set(table.shot) == {100}
    meta = json.loads((out / "t.meta.json").read_text())
    assert meta["shots_labelled"] == [100]
    assert set(meta["shots_skipped"]) == {"101", "102", "103"}
    assert "outside the training range" in meta["shots_skipped"]["101"]
    assert meta["shots_skipped"]["102"] == "no fetched record"
    assert "fdp has no" in meta["shots_skipped"]["103"]
    assert meta["threshold"] == 0.5 and len(meta["checkpoints"]) == serve.FOLDS
    assert all(len(c["sha256"]) == 64 for c in meta["checkpoints"])
    record = meta["input_records"]["100"]
    assert len(record["sha256"]) == 64
    assert record["filterscope_khz"] == 50.0 and record["interferometer_khz"] == 100.0
    assert (tmp_path / "round4/elm/serve/t/100.npz").is_file()


def test_a_fold_shot_and_a_near_empty_shot_are_not_published(tmp_path, monkeypatch):
    run = tmp_path / "round4/elm/cv/cv2"
    folds = [[1], [2], [3], [4], [100]]
    for k in range(serve.FOLDS):
        torch.manual_seed(k)
        (run / f"fold{k}").mkdir(parents=True)
        torch.save(net.ElmUNet().state_dict(), run / f"fold{k}/model.pt")
        (run / f"fold{k}/fold.json").write_text(json.dumps({"threshold": 0.5}))
    (run / "run.json").write_text(
        json.dumps({"config": {"dropout": 0.1}, "folds": folds})
    )
    (tmp_path / "round4/elm/inputs").mkdir(parents=True)
    signals = tmp_path / "benchmarks/elm/elmo/signals"
    signals.mkdir(parents=True)
    for shot in (1, 2, 3, 4, 100):
        np.save(tmp_path / f"round4/elm/inputs/{shot}.npy", inputs.channels(*_record()))
    for shot in (100, 104, 105):
        t_fs, fs, t_int, ne = _record()
        np.savez(
            signals / f"{shot}.npz",
            t_fs_ms=t_fs,
            filterscopes=fs,
            t_int_ms=t_int,
            interferometer=ne,
        )
    monkeypatch.setenv("LABELER_ROOT", str(tmp_path))
    monkeypatch.setattr(
        serve, "present_fraction", lambda rows: 0.5 if rows.shot.iloc[0] == 104 else 0.0
    )
    out = tmp_path / "out"
    argv = ["--name", "t", "--shots", "100", "104", "105", "--out-dir", str(out)]
    assert serve.main([*argv, "--min-present", "0.05"]) == 0
    meta = json.loads((out / "t.meta.json").read_text())
    assert meta["shots_labelled"] == [104]
    assert "trained on" in meta["shots_skipped"]["100"]
    assert meta["scored_not_published"]["shots"] == [105]
    assert set(pd.read_csv(out / "t.csv").shot) == {104}


def test_an_existing_table_is_not_overwritten_without_replace(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    (out / "t.csv").write_text("published\n")
    loaded = []
    monkeypatch.setattr(serve, "load_models", lambda *a: loaded.append(a))
    with pytest.raises(SystemExit):
        serve.main(["--name", "t", "--shots", "1", "--out-dir", str(out)])
    assert (out / "t.csv").read_text() == "published\n" and not loaded
