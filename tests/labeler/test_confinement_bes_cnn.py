"""The BES network: the paper's parameter count, row blocks, the training length."""

from __future__ import annotations

import numpy as np
import torch

from labeler.confinement import bes_cnn as cnn


def _parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def test_paper_architecture_has_227644_parameters_and_the_padded_one_465244():
    assert _parameters(cnn.BesNet(6, padding=cnn.PAPER_PADDING)) == 227_644
    assert _parameters(cnn.BesNet(6)) == 227_644  # the paper's is the default
    assert _parameters(cnn.BesNet(6, padding=cnn.PADDED)) == 465_244
    assert cnn.PAPER_FULL.padding == cnn.PAPER_PADDING and not cnn.PAPER_FULL.early_stop
    assert cnn.PAPER_FULL.steps == 60_000 and cnn.PAPER_FULL.optimiser == "adam"
    out = cnn.BesNet(6)(torch.zeros(3, 2, 6, 8, 128))
    assert out.shape == (3, 4)


def test_take_rows_follows_each_windows_own_start_row():
    x = np.arange(4 * 2 * 64 * 5, dtype=np.float32).reshape(4, 2, 64, 5)
    got = cnn.take_rows(x, np.array([0, 2, 1, 2]), 6, axis=2)
    assert got.shape == (4, 2, 48, 5)
    for i, s in enumerate([0, 2, 1, 2]):
        np.testing.assert_array_equal(got[i], x[i, :, s * 8 : (s + 6) * 8])
    power = np.arange(3 * 64, dtype=np.float32).reshape(3, 64)
    got = cnn.take_rows(power, np.array([1, 0, 2]), 6, axis=1)
    assert got[0, 0] == 8 and got[1, 0] == 64 and got[2, 0] == 128 + 16


def _features(n=300):
    rng = np.random.default_rng(0)
    array = rng.normal(size=(n, 2, 64, 128)).astype(np.float16)
    return array, np.arange(n)


def test_features_block_uses_per_window_starts():
    array, index = _features(40)
    starts = np.arange(40) % 3
    feats = cnn.Features(
        array, index, (0, 6), np.zeros(48), torch.device("cpu"), starts=starts
    )
    got = feats.get(np.array([5, 7])).numpy()  # starts 2 and 1
    want = [array[5, :, 16:64], array[7, :, 8:56]]
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(g.reshape(2, 48, 128), w.astype(np.float32))


def test_training_without_early_stopping_runs_every_step():
    array, index = _features(300)
    labels = (np.arange(300) % 4).astype(np.int64)
    feats = cnn.Features(array, index, (0, 6), np.zeros(48), torch.device("cpu"))
    train, val = np.arange(200), np.arange(200, 300)
    base = cnn.TrainConfig(
        steps=60, eval_every=10, patience=1, batch=16, padding=cnn.PAPER_PADDING
    )
    _, stopped = cnn.train(
        feats, labels, train, val, base, torch.device("cpu"), log=lambda m: None
    )
    full = cnn.TrainConfig(
        steps=60,
        eval_every=10,
        patience=1,
        batch=16,
        padding=cnn.PAPER_PADDING,
        early_stop=False,
    )
    _, ran = cnn.train(
        feats, labels, train, val, full, torch.device("cpu"), log=lambda m: None
    )
    assert ran["steps_run"] == 60 and ran["parameters"] == 227_644
    assert stopped["steps_run"] <= ran["steps_run"]
