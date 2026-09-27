"""Training archives the exact review snapshot used to build its targets."""

import hashlib
import json

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
