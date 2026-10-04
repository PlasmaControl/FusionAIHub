"""Run records retain completed folds and identify the bytes they describe."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from labeler.config import Paths
from labeler.elm import inputs, net, provenance, train


@pytest.fixture
def training_run(tmp_path, monkeypatch):
    tables = tmp_path / "tables"
    review = tables / "edge_localized_mode" / "review"
    review.mkdir(parents=True)
    pd.DataFrame(
        [(s, 0, 100, 0, None) for s in range(1, 5)],
        columns=["shot", "t_start", "t_end", "category", "attrs"],
    ).to_csv(review / "labels.csv", index=False)
    (tmp_path / "catalog").mkdir()
    pd.DataFrame({"shot": range(1, 5), "split": ["train"] * 4}).to_csv(
        tmp_path / "catalog" / "cohort.csv", index=False
    )
    prepared = tmp_path / "round4" / "elm" / "inputs"
    prepared.mkdir(parents=True)
    for shot in range(1, 5):
        x = np.zeros((inputs.N_CHANNELS, 1500), dtype=np.float32)
        x[inputs.VALID] = 1
        np.save(prepared / f"{shot}.npy", x)
    monkeypatch.setenv("LABELER_ROOT", str(tmp_path))
    monkeypatch.setenv("LABELER_LABEL_TABLES", str(tables))
    # Replace costly fitting and inference; keep real inputs, splits and file writes.
    state = net.ElmUNet().state_dict()
    best = {"val_threshold": 0.5, "epoch": 0, "val_auprc": 0.8}
    monkeypatch.setattr(train, "train_fold", lambda *a, **kw: (state, [best], best))
    monkeypatch.setattr(train, "onset_threshold", lambda *a: (0.4, 0.0))
    monkeypatch.setattr(
        train, "predict", lambda model, x, device: np.zeros((2, x.shape[1] // 10))
    )
    return SimpleNamespace(
        run="test",
        epochs=1,
        iters=1,
        batch=1,
        lr=0.002,
        seed=1,
        device="cpu",
        folds=2,
        inner_val=1,
        only=[1],
    ), tmp_path / "round4" / "elm" / "cv" / "test"


def test_only_retains_other_completed_folds(training_run):
    args, out = training_run
    train.run(args)
    before = json.loads((out / "run.json").read_text())
    args.only = [0]
    train.run(args)
    after = json.loads((out / "run.json").read_text())
    assert [r["fold"] for r in after["fold_records"]] == [0, 1]
    assert after["fold_records"][1] == before["fold_records"][0]


def test_only_refuses_changed_configuration(training_run):
    args, out = training_run
    train.run(args)
    before = (out / "run.json").read_bytes()
    args.only, args.lr = [0], 0.01
    with pytest.raises(ValueError, match="incompatible.*config"):
        train.run(args)
    assert (out / "run.json").read_bytes() == before
    assert not (out / "fold0").exists()


def test_new_training_hashes_labels_inputs_and_artifacts(training_run):
    args, out = training_run
    train.run(args)
    record = json.loads((out / "run.json").read_text())
    provenance = record["provenance"]
    assert provenance["mode"] == "training_time"
    for entry in [
        provenance["data"]["reviewed_labels"],
        *provenance["data"]["inputs"],
        *record["fold_records"][0]["artifacts"]["checkpoints"],
        *record["fold_records"][0]["artifacts"]["predictions"],
    ]:
        assert (
            entry["sha256"]
            == hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest()
        )


@pytest.mark.parametrize("changed", ["inner_val", "input", "labels"])
def test_only_refuses_changed_splits_or_data(training_run, changed):
    args, out = training_run
    train.run(args)
    before = (out / "run.json").read_bytes()
    args.only = [0]
    if changed == "inner_val":
        args.inner_val = 2
    else:
        paths = Paths.from_env()
        path = (
            paths.label_tables / "edge_localized_mode" / "review" / "labels.csv"
            if changed == "labels"
            else paths.root / "round4" / "elm" / "inputs" / "1.npy"
        )
        # A trailing newline remains a readable CSV / NPY but changes its bytes.
        with path.open("ab") as fh:
            fh.write(b"\n")
    with pytest.raises(ValueError, match="incompatible"):
        train.run(args)
    assert (out / "run.json").read_bytes() == before


def test_backfill_marks_retrospective_observation_and_preserves_folds(training_run):
    args, out = training_run
    train.run(args)
    before = json.loads((out / "run.json").read_text())
    result = provenance.backfill(out, Paths.from_env(), "0d16c19", "audit evidence")
    assert result["config"] == before["config"]
    assert result["folds"] == before["folds"]
    assert result["fold_records"] == before["fold_records"]
    assert result["provenance"]["mode"] == "retrospective"
    assert result["provenance"]["code"]["revision_is_reconstructed"] is True
    assert "not training-time hashes" in result["provenance"]["caveat"]
    assert result["git_recorded_at_training"] == before["git"]
    assert result["git"].startswith("0d16c19")
    train_source = next(
        r
        for r in result["provenance"]["code"]["sources"]
        if r["path"] == "src/labeler/elm/train.py"
    )
    assert train_source["sha256"] == (
        "e397b78af753adb7ef7cbd58ece628cd1773ab1171259dae4107e88ec8c36308"
    )


def test_repeated_backfill_preserves_original_record_identity(training_run):
    args, out = training_run
    train.run(args)
    first = provenance.backfill(out, Paths.from_env(), "0d16c19", "audit evidence")
    second = provenance.backfill(out, Paths.from_env(), "0d16c19", "audit evidence")
    assert (
        second["provenance"]["original_run_record"]
        == (first["provenance"]["original_run_record"])
    )
