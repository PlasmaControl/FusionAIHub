"""SegNet on the pseudo-masks: crops, the masked loss, the threshold, a training run."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

import numpy as np
import pytest
import torch

from labeler.ae.seg import model_dir, pseudo, regions, train, whole
from labeler.ae.seg.pseudo import IGNORE
from labeler.config import Paths, sha256_of
from labeler.events.review import labels

from . import ae_tree


def _example(n=64, n_y=32, seed=0) -> train.Example:
    """A bright line at bin 20 over columns 10-49; bins under 8 are not scored."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 60, (3, n_y, n)).astype(np.uint8)
    x[:, 20, 10:50] = 250
    y = np.zeros((n_y, n), dtype=np.uint8)
    y[:8] = IGNORE
    y[20, 10:50] = 1
    return train.Example(1, x, y, 0.0, 2.048, 0.0, 500 / 512)


@pytest.fixture
def two_threads():
    """Two torch threads: the default, one per core, crawls on a shared node."""
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def test_a_crop_past_the_ends_is_zero_and_ignored():
    ex = _example(n=10)
    x, y = train.crop(ex, -3, 8)
    assert (x[:, :, :3] == 0).all() and (y[:, :3] == IGNORE).all()
    assert (x[:, :, 3:] == ex.x[:, :, :5]).all() and (y[:, 3:] == ex.y[:, :5]).all()
    x, y = train.crop(ex, 7, 8)
    assert (y[:, 3:] == IGNORE).all() and (x[:, :, :3] == ex.x[:, :, 7:]).all()


def test_half_the_crops_hold_ae():
    ex = _example(n=2000)
    ex.y[20] = 0
    ex.y[20, 500:510] = 1
    _, ys = train.sample_crops([ex], np.random.default_rng(0), 8, 64)
    assert ys.shape == (8, 32, 64)
    assert all((ys[i] == 1).any() for i in range(0, 8, 2))


def test_the_loss_ignores_ignored_pixels_and_prefers_the_truth():
    target = torch.from_numpy(_example().y[None])
    logits = torch.randn(1, 32, 64)
    changed = logits.clone()
    changed[:, :8] = 50.0
    same = (
        train.masked_loss(logits, target, 5.0),
        train.masked_loss(changed, target, 5.0),
    )
    assert torch.isclose(*same)
    truth = torch.where(target == 1, 8.0, -8.0)
    assert train.masked_loss(truth, target, 5.0) < train.masked_loss(
        -truth, target, 5.0
    )


def test_the_background_weight_is_capped():
    ex = _example()
    assert train.pos_weight([ex], 20.0) == 20.0  # 40 AE pixels, 1496 background
    assert train.pos_weight([ex], 100.0) == pytest.approx((24 * 64 - 40) / 40)


def test_pixel_cells_count_only_scored_pixels_and_the_threshold_is_picked_on_them():
    ex = _example()
    prob = np.where(ex.y == 1, 0.7, 0.2).astype(np.float32)
    prob[:8] = 0.99  # not scored
    assert train.pixel_cells(prob, ex.y, 0.5).tolist() == [40, 0, 0, 24 * 64 - 40]
    assert train.dice_of(train.pixel_cells(prob, ex.y, 0.8)) == 0.0
    assert train.pick_threshold([prob], [ex.y]) == 0.5


def test_fit_learns_a_bright_line(two_threads):
    ex = _example()
    config = train.TrainConfig(
        epochs=30,
        batch=4,
        crop_cols=32,
        crops_per_shot=4,
        lr=1e-2,
        patience=30,
        width=4,
    )
    model, history, threshold = train.fit([ex, _example(seed=1)], [ex], config, log=str)
    prob = train.predict(model, ex.x)
    assert train.dice_of(train.pixel_cells(prob, ex.y, threshold)) > 0.8
    assert train.best_epoch(history) > 0


@dataclass(frozen=True)
class Small(train.TrainConfig):
    batch: int = 2
    crop_cols: int = 64
    crops_per_shot: int = 2
    width: int = 4


def test_the_command_trains_on_the_ae_split_and_saves_the_model(tmp_path, monkeypatch):
    splits = {s: "train" for s in (101, 102, 103, 104)}
    paths = ae_tree.build(tmp_path, splits, tokeye_dt=0.256)
    ae_tree.chosen(paths, {101: "train", 102: "train", 103: "val", 104: "test"})
    ae_tree.env(monkeypatch, paths)
    assert pseudo.main([]) == 0
    regions.save_decision(
        paths.label_tables / "alfven_eigenmode",
        102,
        [1],
        pseudo_sha256=regions.file_sha256(regions.pseudo_file(paths, 102)),
    )
    monkeypatch.setattr(train, "TrainConfig", Small)
    assert train.main(["--epochs", "2"]) == 0
    out = model_dir(paths)
    loaded, blob = train.load(out / "model.pt")
    assert not loaded.training
    assert blob["train"]["width"] == 4 and blob["threshold"] in train.THRESHOLDS
    assert blob["inputs"]["masks_sha256"] is not None
    directory = paths.label_tables / "alfven_eigenmode"
    assert labels.read_saved(out) == labels.read_saved(directory)
    assert regions.read_decisions(out) == regions.read_decisions(directory)
    manifest = json.loads((out / "pseudo_masks.json").read_text())
    assert set(manifest) == {"index.csv", "101.npz", "102.npz", "103.npz", "104.npz"}
    for name, digest in manifest.items():
        assert digest == sha256_of(train.pseudo_dir(paths) / name)
    assert blob["inputs"]["pseudo_masks_sha256"] == sha256_of(out / "pseudo_masks.json")
    ae_file = train.chosen_model(train.ae_model_dir(paths))
    assert blob["inputs"]["ae_model_sha256"] == sha256_of(ae_file)
    assert (out / "split.csv").read_text().split() == [
        "shot,split",
        "101,train",
        "102,train",
        "103,val",
        "104,test",
    ]
    record = json.loads((out / "training.json").read_text())
    assert blob["inputs"]["split_sha256"] == sha256_of(out / "split.csv")
    assert record["inputs"]["split_sha256"] == sha256_of(out / "split.csv")
    assert len(record["history"]) == 2 and record["peak_rss_mb"] > 0
    ex = train.load_example(
        paths, 102, regions.read_decisions(paths.label_tables / "alfven_eigenmode")
    )
    assert not (ex.y == 1).any(), "the rejected region is background"
    assert ex.y.shape[1] < 1123, "cut to the scored columns and a margin"


def page_mask(paths, shot: int, rejected, *, sha256: str | None = None):
    """The review page's mask for `shot` (pseudo-v1's regions, as `whole` would
    make them on 0-2 s, in another file), and a decision saved on it; the file."""
    file = whole.review_file(paths, shot)
    pm = pseudo.PseudoMask.load(regions.pseudo_file(paths, shot))
    replace(pm, present_unlit=pm.present_unlit + 1).save(file)  # other bytes
    regions.save_decision(
        paths.label_tables / "alfven_eigenmode",
        shot,
        rejected,
        pseudo_sha256=sha256 or regions.file_sha256(file),
        pseudo=whole.REVIEW,
    )
    return file


def test_a_decision_on_the_review_page_s_mask_reaches_training(
    tmp_path, monkeypatch, capsys
):
    splits = {s: "train" for s in (101, 102, 103, 104)}
    paths = ae_tree.build(tmp_path, splits, tokeye_dt=0.256)
    ae_tree.chosen(paths, {101: "train", 102: "train", 103: "val", 104: "test"})
    ae_tree.env(monkeypatch, paths)
    assert pseudo.main([]) == 0
    _, count = regions.label_regions(
        pseudo.PseudoMask.load(regions.pseudo_file(paths, 102)).mask
    )
    page = page_mask(paths, 102, range(1, count + 1))
    page_mask(paths, 103, [1], sha256="0" * 64)
    monkeypatch.setattr(
        train,
        "fit",
        lambda *a, **kw: (train.SegNet(train.SegNetConfig(width=4)), [], 0.5),
    )
    assert train.main([]) == 0
    output = capsys.readouterr()
    assert "1 shots' mask decisions made on other masks reach the targets" in output.out
    assert "103: made on an earlier" in output.err
    manifest = json.loads((model_dir(paths) / "pseudo_masks.json").read_text())
    assert manifest["pseudo-v1-full/102.npz"] == sha256_of(page)
    assert "pseudo-v1-full/103.npz" not in manifest
    decisions = regions.read_decisions(paths.label_tables / "alfven_eigenmode")
    before = train.load_example(paths, 102, decisions)
    after = train.load_example(paths, 102, decisions, clicked=page.read_bytes())
    assert (before.y == 1).any() and not (after.y == 1).any()
    with pytest.raises(ValueError, match="not the mask the decision was made on"):
        train.load_example(paths, 102, decisions, clicked=b"other bytes")


def test_a_mask_off_the_store_grid_is_refused(tmp_path):
    paths = ae_tree.build(tmp_path, {101: "train"}, tokeye_dt=0.256)
    file = regions.pseudo_file(paths, 101)
    file.parent.mkdir(parents=True)
    pseudo.PseudoMask(101, 0.0, 2.048, 0.0, 1.0, np.zeros((257, 10), np.uint8)).save(
        file
    )
    with pytest.raises(ValueError, match="level 8"):
        train.load_example(paths, 101, {})


@pytest.mark.parametrize(
    "changed", ["labels", "masks", "index", "npz", "ae", "clicked"]
)
@pytest.mark.parametrize("when", ["load", "fit"])
def test_changed_training_inputs_refuse_to_save(
    tmp_path, monkeypatch, capsys, changed, when
):
    paths = ae_tree.build(tmp_path, {101: "train", 102: "train"})
    ae_tree.chosen(paths, {101: "train", 102: "val"})
    ae_tree.env(monkeypatch, paths)
    assert pseudo.main([]) == 0
    directory = paths.label_tables / "alfven_eigenmode"
    clicked = page_mask(paths, 101, [1]) if changed == "clicked" else None
    files = {
        "labels": directory / "review" / "labels.csv",
        "masks": regions.log_path(directory),
        "index": train.pseudo_dir(paths) / "index.csv",
        "npz": regions.pseudo_file(paths, 101),
        "ae": train.chosen_model(train.ae_model_dir(paths)),
        "clicked": clicked,
    }
    changed_file = files[changed]
    out = tmp_path / "seg-model"
    load_example = train.load_example

    def change_file():
        original = changed_file.read_bytes() if changed_file.exists() else b""
        changed_file.write_bytes(original + b"\n")

    def load(paths, shot, decisions, **kw):
        example = load_example(paths, shot, decisions, **kw)
        if when == "load":
            change_file()
        return example

    def fit(train_examples, val_examples, config, log):
        if when == "fit":
            change_file()
        return train.SegNet(train.SegNetConfig(width=4)), [], 0.5

    monkeypatch.setattr(train, "load_example", load)
    monkeypatch.setattr(train, "fit", fit)
    assert train.main(["--out", str(out)]) != 0
    output = capsys.readouterr()
    assert str(changed_file) in output.err
    assert "changed" in output.err.lower()
    assert "Traceback" not in output.err
    assert not out.exists()


def test_train_refuses_an_existing_model_before_inputs(tmp_path, monkeypatch, capsys):
    ae_tree.env(
        monkeypatch,
        Paths(
            root=tmp_path / "root",
            label_tables=tmp_path / "events",
            corpus=tmp_path / "corpus",
        ),
    )
    path = tmp_path / "model.pt"
    path.write_bytes(b"existing")
    with pytest.raises(SystemExit) as error:
        train.main(["--out", str(tmp_path)])
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert str(path) in stderr and "Traceback" not in stderr
    assert path.read_bytes() == b"existing"
