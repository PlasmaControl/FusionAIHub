"""The training bridge refuses altered inputs and never loads evaluation shots."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from labeler.ae.supervision import clean_split, sha256


@pytest.fixture
def trainer(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts/labeler"
    monkeypatch.syspath_prepend(str(scripts))
    return importlib.import_module("ae_train")


@pytest.fixture
def manifest(tmp_path):
    split = clean_split(list(range(100000, 100120)), list(range(100120, 100180)), set())
    labels = tmp_path / "dense.csv"
    labels.write_text(
        "shot,category,t_start,t_end,confidence\n"
        + "".join(f"{s},1,0,2000,\n" for s in range(100000, 100180))
    )
    cohort = tmp_path / "cohort.csv"
    cohort.write_text("shot,split\n999999,test\n")
    dataset = {}
    for shot in range(100000, 100120):
        path = tmp_path / f"{shot}_train.npz"
        np.savez(
            path,
            active=np.ones(800),
            annotated=np.tile([0, 1], 400),
            freq_khz=np.full(800, 100.0),
        )
        dataset[str(shot)] = {"path": str(path), "sha256": sha256(path)}
    # No files for the 60 evaluation shots: loading any of them would fail.
    value = {
        "split": split,
        "blind_gold_shots": [999999],
        "inputs": {
            "dense": {"snapshot": str(labels), "sha256": sha256(labels)},
            "cohort": {"snapshot": str(cohort), "sha256": sha256(cohort)},
        },
        "dataset": dataset,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    return path, value


@pytest.mark.parametrize("target", ["legacy", "dense", "threeway"])
def test_swap_training_bridge_loads_only_train_and_selection(trainer, manifest, target):
    path, source = manifest
    records, saved = trainer.load_swap_labels(path, target)
    assert saved == source
    train = [r for r in records if r.split == "train"]
    selection = [r for r in records if r.split == "valid"]
    assert [int(r.shot) for r in train] == source["split"]["train"]
    assert [int(r.shot) for r in selection] == source["split"]["selection"]
    if target == "dense":
        assert all(r.y.all() and r.w.all() for r in records)


def test_training_bridge_refuses_mutated_snapshot(trainer, manifest):
    path, source = manifest
    Path(source["inputs"]["dense"]["snapshot"]).write_text("altered\n")
    with pytest.raises(ValueError, match="snapshot changed"):
        trainer.load_swap_labels(path, "dense")


def test_training_bridge_refuses_mutated_dataset(trainer, manifest):
    path, source = manifest
    shot = source["split"]["train"][0]
    Path(source["dataset"][str(shot)]["path"]).write_bytes(b"altered")
    with pytest.raises(ValueError, match="dataset changed"):
        trainer.load_swap_labels(path, "legacy")


def test_new_targets_cannot_use_old_selection_protocol(trainer):
    with pytest.raises(SystemExit):
        trainer.main(["--target", "legacy", "--device", "cpu"])
