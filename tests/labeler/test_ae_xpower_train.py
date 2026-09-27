"""Training archives the exact review snapshot used to build its targets."""

import hashlib
import json

import pytest

from labeler.ae.xpower import event_dir, train
from labeler.ae.xpower.model import FrameCNN, FrameCNNConfig
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


def test_training_refuses_a_checkpoint_before_loading_data(tmp_path, capsys):
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
