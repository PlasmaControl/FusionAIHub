"""Validation-only threshold evidence, using saved tiny FrameCNN checkpoints."""

import csv
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from labeler.ae.xpower import evaluate, event_dir, model_dir, train
from labeler.ae.xpower.model import FrameCNN, FrameCNNConfig
from labeler.config import Paths, atomic_path
from labeler.events.review import labels

from . import ae_tree


def _save(paths, name="band80-mhd3", *, bias=0.0, split=None):
    model = FrameCNN(FrameCNNConfig(width=4, dilations=(1,)))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head.bias.fill_(bias)
    train.save(
        model_dir(paths) / name,
        model,
        threshold=0.5,
        split=split or {101: "val", 102: "test", 103: "train"},
        history=[],
        config=train.TrainConfig(),
        band_khz=train.CANDIDATES[name]["band"],
        labels_file=labels.labels_path(event_dir(paths)),
        candidate=name,
    )


@pytest.mark.parametrize("poison", [False, True])
def test_command_sweeps_saved_candidates_without_reading_test_or_train_shots(
    tmp_path, monkeypatch, poison
):
    from labeler.ae.xpower import frontier

    paths = ae_tree.build(tmp_path, {101: "train"})
    _save(paths)
    _save(paths, "band0-mhd3", bias=-1.0)
    models = model_dir(paths)
    chosen = b'{"candidate": "band80-mhd3"}\n'
    (models / "chosen.json").write_bytes(chosen)
    if poison:
        for shot, split in ((102, "valid"), (103, "train")):
            paths.spectrogram_file("alfven_eigenmode", shot).write_bytes(b"poison")
            (paths.root / "ae/masks" / f"{shot}_{split}_clean.npz").write_bytes(
                b"poison"
            )
    # Validation uses each checkpoint's labels, not mutable live/source tables.
    labels.labels_path(event_dir(paths)).write_bytes(b"poison")
    labels.source_path(event_dir(paths)).write_bytes(b"poison")
    ae_tree.env(monkeypatch, paths)
    files = [models / f"validation_frontier.{suffix}" for suffix in ("csv", "md")]
    for file in files:
        file.write_text("previous complete report\n")
    replaced = set()

    @contextmanager
    def checked_atomic(path):
        with atomic_path(path) as temporary:
            yield temporary
            assert path.read_text() == "previous complete report\n"
            assert temporary.read_text() != "previous complete report\n"
            replaced.add(path)

    monkeypatch.setattr(frontier, "atomic_path", checked_atomic)
    assert frontier.main([] if poison else ["--models", str(models)]) == 0
    assert replaced == set(files)
    with files[0].open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 38
    for name, cutoff in (("band80-mhd3", 0.5), ("band0-mhd3", 0.25)):
        candidate = [row for row in rows if row["candidate"] == name]
        assert [float(row["threshold"]) for row in candidate] == [
            k / 20 for k in range(1, 20)
        ]
        assert {row["chosen"] for row in candidate} == {str(name == "band80-mhd3")}
        for row in candidate:
            on = float(row["threshold"]) <= cutoff
            assert float(row["f1"]) == pytest.approx(6 / 13 if on else 0)
            if on:
                assert float(row["precision"]) == 0.3
            else:
                assert row["precision"] == ""
            assert float(row["recall"]) == float(on)
            assert float(row["fp_rate_mhd"]) == float(on)
            assert float(row["fp_rate_other"]) == float(on)
        selected = [row for row in candidate if row["operating_point"] == "True"]
        assert len(selected) == 1 and float(selected[0]["threshold"]) == 0.05
    report = files[1].read_text()
    assert "band80-mhd3 (chosen)" in report and "band0-mhd3" in report
    assert "within 0.02" in report and "higher F1" in report
    assert "F1 >= 0.90 and MHD FP <= 0.05: no" in report
    assert (models / "chosen.json").read_bytes() == chosen
    assert not (models / "evaluation.json").exists()


def test_frontier_uses_scored_frames_and_mhd_rates_and_finds_a_feasible_point(
    tmp_path, monkeypatch
):
    from labeler.ae.xpower import frontier

    paths = ae_tree.build(tmp_path, {101: "train"})
    labels.save(
        event_dir(paths),
        101,
        labels.normalise((0, 2000), [(0, 100, 2), (100, 200, 3), (300, 900, 1)]),
        source="four-state validation fixture",
    )
    _save(paths)
    prob = np.full(200, 0.1)
    prob[:20] = 0.99  # uncertain and not observable: never false positives
    prob[30:90] = 0.9
    prob[88:90] = 0.4
    prob[120:122] = 0.6  # MHD-absent
    prob[100] = 0.7  # other absent
    observed = np.ones(200, bool)
    observed[120] = False  # excluded from numerator and denominator
    monkeypatch.setattr(evaluate, "probabilities", lambda *a, **kw: (prob, observed))
    rows = frontier.run(paths, model_dir(paths))
    middle = next(row for row in rows if row["threshold"] == 0.5)
    assert middle["f1"] == pytest.approx(29 / 30)
    assert middle["precision"] == middle["recall"] == pytest.approx(29 / 30)
    assert middle["fp_rate_mhd"] == pytest.approx(1 / 29)
    assert middle["fp_rate_other"] == pytest.approx(1 / 90)
    selected = [row for row in rows if row["operating_point"]]
    assert len(selected) == 1
    assert selected[0]["threshold"] == 0.75
    assert selected[0]["f1"] == pytest.approx(58 / 59)
    assert selected[0]["fp_rate_mhd"] == selected[0]["fp_rate_other"] == 0
    report = (model_dir(paths) / "validation_frontier.md").read_text()
    assert "F1 >= 0.90 and MHD FP <= 0.05: yes" in report


def test_operating_point_uses_the_f1_margin_then_mhd_then_higher_f1():
    from labeler.ae.xpower import frontier

    rows = [
        {"threshold": 0.3, "f1": 0.92, "fp_rate_mhd": 0.08},
        {"threshold": 0.4, "f1": 0.90, "fp_rate_mhd": 0.01},
        {"threshold": 0.5, "f1": 0.91, "fp_rate_mhd": 0.01},
        {"threshold": 0.6, "f1": 0.89, "fp_rate_mhd": 0.00},
    ]
    assert frontier.operating_point(rows)["threshold"] == 0.5
    assert frontier.operating_point(rows[:2])["threshold"] == 0.4


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"threshold": 0.5, "f1": None, "fp_rate_mhd": None}],
        [{"threshold": 0.5, "f1": 0.95, "fp_rate_mhd": None}],
    ],
)
def test_undefined_scores_do_not_claim_an_operating_point(rows):
    from labeler.ae.xpower import frontier

    assert frontier.operating_point(rows) is None


@pytest.mark.parametrize("empty_split", [False, True])
def test_command_refuses_missing_candidates_or_validation_without_outputs(
    tmp_path, monkeypatch, capsys, empty_split
):
    from labeler.ae.xpower import frontier

    paths = Paths(
        root=tmp_path / "root",
        label_tables=tmp_path / "events",
        corpus=tmp_path / "corpus",
    )
    if empty_split:
        paths = ae_tree.build(tmp_path, {101: "train"})
        _save(paths, split={101: "test"})
    ae_tree.env(monkeypatch, paths)
    with pytest.raises(SystemExit) as error:
        frontier.main([])
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert ("validation shots" if empty_split else "no saved candidate") in stderr
    assert "Traceback" not in stderr
    assert not (model_dir(paths) / "validation_frontier.csv").exists()
    assert not (model_dir(paths) / "validation_frontier.md").exists()


def test_frontier_intervals_and_counts_use_shots_at_both_reported_thresholds(
    tmp_path, monkeypatch
):
    from labeler.ae.xpower import frontier
    from labeler.scoring import stats

    paths = ae_tree.build(tmp_path, {101: "train", 104: "train"})
    _save(paths, split={101: "val", 104: "val"})
    frames = [
        evaluate.ShotFrames(
            101,
            np.array([1, 1, 0, 0]),
            np.array([0, 0, 1, 1], bool),
            np.ones(4, bool),
            {},
            np.array([0.9, 0.8, 0.7, 0.1]),
        ),
        evaluate.ShotFrames(
            104,
            np.array([1, 0, 0, 0]),
            np.array([0, 1, 1, 0], bool),
            np.array([1, 1, 0, 1], bool),
            {},
            np.array([0.9, 0.6, 0.95, 0.1]),
        ),
    ]
    monkeypatch.setattr(
        evaluate,
        "shot_frames",
        lambda shot, **kw: next(f for f in frames if f.shot == shot),
    )
    rows = frontier.run(paths, model_dir(paths))
    selected = [r for r in rows if r["operating_point"] or r["chosen_rule_threshold"]]
    assert len(selected) == 2
    for row in rows:
        assert row["validation_shots"] == 2
        assert row["scored_frames"] == 7
        assert row["mhd_absent_frames"] == 3
    for row in selected:
        for f in frames:
            f.said["ae_xpower"] = f.prob >= row["threshold"]
        expected_f1 = evaluate._estimate(evaluate.cells(frames, "ae_xpower"), stats.f1)
        expected_mhd = evaluate._estimate(
            evaluate.cells(frames, "ae_xpower", evaluate.mhd_absent), evaluate.fp_rate
        )
        for key, expected in (("f1", expected_f1), ("fp_rate_mhd", expected_mhd)):
            assert row[key] == expected["value"]
            assert row[f"{key}_low"] == expected["low"]
            assert row[f"{key}_high"] == expected["high"]
    report = (model_dir(paths) / "validation_frontier.md").read_text()
    assert "point estimates" in report
    assert "95 % shot-bootstrap" in report and "2000" in report
    assert "20260923" in report
    assert "chosen-rule" in report
