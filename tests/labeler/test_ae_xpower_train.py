"""Training archives the exact review snapshot used to build its targets."""

import hashlib
import json

import pytest

from labeler.ae.xpower import event_dir, train
from labeler.ae.xpower.model import FrameCNN, FrameCNNConfig
from labeler.config import Paths
from labeler.events.review import labels

from . import ae_tree


def test_owner_save_during_fit_does_not_change_the_training_snapshot(
    tmp_path,
    monkeypatch,
):
    paths = ae_tree.build(tmp_path, {101: "train", 102: "train"})
    ae_tree.env(monkeypatch, paths)
    live = labels.labels_path(event_dir(paths))
    original = live.read_bytes()
    out = tmp_path / "model"

    def fit(train_shots, val_shots, config, log):
        for shot in [*train_shots, *val_shots]:
            assert (shot.states == 1).sum() == 60
        labels.save(
            event_dir(paths),
            101,
            labels.normalise((0, 2000), []),
            source="owner edit during fit",
        )
        return FrameCNN(FrameCNNConfig(width=4)), [], 0.5

    monkeypatch.setattr(train, "fit", fit)
    assert train.main(["--candidate", "band80-mhd3", "--out", str(out)]) == 0
    assert live.read_bytes() != original
    assert labels.labels_path(out).read_bytes() == original
    _, blob = train.load(out / "model.pt")
    digest = hashlib.sha256(original).hexdigest()
    assert blob["labels_sha256"] == digest
    assert json.loads((out / "training.json").read_text())["labels_sha256"] == digest


def test_training_refuses_a_checkpoint_before_loading_data(
    tmp_path, monkeypatch, capsys
):
    ae_tree.env(
        monkeypatch,
        Paths(
            root=tmp_path / "root",
            label_tables=tmp_path / "events",
            corpus=tmp_path / "corpus",
        ),
    )
    file = tmp_path / "model.pt"
    file.write_bytes(b"original")
    with pytest.raises(SystemExit) as error:
        train.main(["--candidate", "band80-mhd3", "--out", str(tmp_path)])
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert str(file) in stderr and "Traceback" not in stderr
    assert file.read_bytes() == b"original"


def test_save_refuses_a_checkpoint(tmp_path):
    file = tmp_path / "model.pt"
    file.write_bytes(b"original")
    with pytest.raises(FileExistsError, match="trained candidate is not replaced"):
        train.save(
            tmp_path,
            FrameCNN(),
            threshold=0.5,
            split={},
            history=[],
            config=train.TrainConfig(),
            band_khz=(80, 250),
            labels_file=tmp_path / "missing.csv",
        )
    assert file.read_bytes() == b"original"


def test_each_version_has_its_own_candidates():
    v1 = {
        "band80-mhd3": {"band": (80.0, 250.0), "mhd_weight": 3.0},
        "band0-mhd3": {"band": (0.0, 250.0), "mhd_weight": 3.0},
        "band0-mhd10": {"band": (0.0, 250.0), "mhd_weight": 10.0},
    }
    assert train.candidates("v1") == v1 and list(train.candidates("v1")) == list(v1)
    assert train.CANDIDATES == v1  # the default version's, as before
    v2 = train.candidates("v2")
    assert list(v2) == ["band80-mhd3", "band80-mhd10", "band80-mhd30"]
    assert {c["band"] for c in v2.values()} == {(80.0, 250.0)}
    assert [c["mhd_weight"] for c in v2.values()] == [3.0, 10.0, 30.0]
    with pytest.raises(ValueError, match="v9"):
        train.candidates("v9")


@pytest.mark.parametrize(
    "version, candidate", [("v2", "band0-mhd3"), ("v1", "band80-mhd30")]
)
def test_training_refuses_another_versions_candidate(
    tmp_path, monkeypatch, capsys, version, candidate
):
    paths = Paths(
        root=tmp_path / "root",
        label_tables=tmp_path / "events",
        corpus=tmp_path / "corpus",
    )
    ae_tree.env(monkeypatch, paths)
    out = tmp_path / version / candidate
    with pytest.raises(SystemExit) as error:
        train.main(["--version", version, "--candidate", candidate, "--out", str(out)])
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert candidate in stderr and version in stderr and "Traceback" not in stderr
    assert not out.exists()


def test_the_validation_choice_reads_only_the_versions_candidates():
    from labeler.ae.xpower import evaluate

    def result(f1, mhd):
        return {"f1": {"value": f1}, "fp_rate_mhd": {"value": mhd}}

    results = {
        "band0-mhd3": result(0.95, 0.0),  # v1's; never v2's choice
        "band80-mhd10": result(0.93, 0.04),
        "band80-mhd30": result(0.93, 0.04),
    }
    assert evaluate.choose(results, "v2")[0] == "band80-mhd10"
    assert evaluate.choose(results, "v1")[0] == "band0-mhd3"
    with pytest.raises(ValueError, match="no trained candidate"):
        evaluate.choose({"band80-mhd30": result(0.9, 0.0)}, "v1")
