"""The U-Net, its losses and sampler, and the BES network's feature path."""

from __future__ import annotations

import numpy as np
import torch

from labeler.confinement import bes_cnn as cnn
from labeler.confinement import unet, zerod

torch.set_num_threads(2)  # the head node is shared; more threads only spin


def test_unet_keeps_the_time_axis_and_emits_four_logits():
    model = unet.UNet1d()
    x = torch.randn(2, zerod.N_INPUT, 2 * unet.STRIDE)
    assert model(x).shape == (2, unet.N_CLASSES, 2 * unet.STRIDE)


def test_dice_is_zero_for_a_perfect_prediction_and_ignores_unlabelled_bins():
    target = torch.tensor([[0, 1, 2, 3, -1, -1]])
    logits = torch.full((1, 4, 6), -20.0)
    for i, c in enumerate(target[0].tolist()):
        logits[0, max(c, 0), i] = 20.0
    assert unet.generalized_dice(logits, target).item() < 1e-3
    wrong = logits.clone()
    wrong[0, :, 4:] = torch.tensor([20.0, -20.0, -20.0, -20.0])[
        :, None
    ]  # unlabelled: no effect
    assert (
        abs(
            unet.generalized_dice(wrong, target).item()
            - unet.generalized_dice(logits, target).item()
        )
        < 1e-6
    )


def test_smoothing_penalises_a_flickering_prediction():
    target = torch.zeros(1, 40, dtype=torch.long)
    steady = torch.zeros(1, 4, 40)
    steady[0, 0] = 5.0
    flicker = steady.clone()
    flicker[0, 0, ::2] = -5.0
    flicker[0, 1, ::2] = 5.0
    assert (
        unet.smoothing(flicker, target).item()
        > unet.smoothing(steady, target).item() + 1.0
    )


def test_sampler_draws_every_class_and_windows_hold_their_centre_bin():
    n = 10 * unet.STRIDE
    y = np.full(n, -1, dtype=np.int8)
    y[100:300], y[400:600], y[700:900], y[1000:1200] = 0, 1, 2, 3
    x = np.zeros((zerod.N_INPUT, n), dtype=np.float32)
    x[0] = np.arange(n)
    corpus = unet.Corpus([x], [y], torch.device("cpu"), window=unet.STRIDE)
    pools = corpus.pools(np.array([0]))
    gen = torch.Generator()
    gen.manual_seed(0)
    bx, by = corpus.batch(pools, 64, gen)
    assert bx.shape == (64, zerod.N_INPUT, unet.STRIDE) and by.shape == (
        64,
        unet.STRIDE,
    )
    assert {int(c) for c in by.unique() if c >= 0} == {0, 1, 2, 3}
    classes = [{int(c) for c in row.unique() if c >= 0} for row in by]
    assert all(classes)  # every window holds a labelled bin


def test_a_few_training_steps_lower_the_loss_on_separable_data():
    n = 2 * unet.STRIDE
    rng = np.random.default_rng(0)
    xs, ys = [], []
    for _ in range(6):
        y = np.where(np.arange(n) < n // 2, 0, 1).astype(np.int8)
        x = np.zeros((zerod.N_INPUT, n), dtype=np.float32)
        x[0] = np.where(y == 1, 1.0, -1.0) + rng.normal(0, 0.1, n)
        xs.append(x)
        ys.append(y)
    cfg = unet.UNetConfig(
        steps=60, batch=4, window=unet.STRIDE, eval_every=30, patience=5
    )
    corpus = unet.Corpus(xs, ys, torch.device("cpu"), cfg.window)
    model, record = unet.train(
        corpus,
        np.arange(4),
        xs[4:],
        ys[4:],
        cfg,
        torch.device("cpu"),
        log=lambda m: None,
    )
    f1, conf = unet.evaluate(model, xs[4:], ys[4:], torch.device("cpu"))
    assert f1 > 0.9 and conf.sum() == 2 * n
    assert record["steps_run"] >= 30


def test_bes_net_shape_and_feature_standardisation_path():
    net = cnn.BesNet(rows=2, padding=cnn.PADDED)  # two rows leave none unpadded
    assert net(torch.randn(3, 2, 2, 8, cnn.FREQS)).shape == (3, 4)
    array = (
        (np.arange(5 * 2 * 64 * cnn.FREQS) % 97)
        .astype(np.float32)
        .reshape(5, 2, 64, cnn.FREQS)
        .astype(np.float16)
    )
    index = np.array([1, 3, 4])
    feats = cnn.Features(
        array, index, (1, 3), np.zeros(16, dtype=np.float32), torch.device("cpu")
    )
    got = feats.get(np.array([3, 1]))
    assert got.shape == (2, 2, 2, 8, cnn.FREQS)
    expect = torch.from_numpy(array[[3, 1]][:, :, 8:24, :].astype(np.float32)).view(
        2, 2, 2, 8, cnn.FREQS
    )
    assert torch.equal(got, expect)
    # the same rows addressed by the positions of a restricted table
    by_position = cnn.Features(
        array,
        index,
        (1, 3),
        np.zeros(16, dtype=np.float32),
        torch.device("cpu"),
        ids=np.arange(3),
    )
    assert torch.equal(by_position.get(np.array([1, 0])), expect)


def test_bes_training_loop_runs_and_keeps_a_checkpoint():
    rng = np.random.default_rng(0)
    n = 120
    labels = rng.integers(0, 4, n)
    array = rng.normal(0, 0.1, (n, 2, 64, cnn.FREQS)).astype(np.float16)
    array[np.arange(n), :, 0, :] += labels[:, None, None].astype(
        np.float16
    )  # class sits in channel 0
    feats = cnn.Features(
        array, np.arange(n), (0, 1), np.zeros(8, dtype=np.float32), torch.device("cpu")
    )
    cfg = cnn.TrainConfig(steps=40, batch=16, eval_every=20, patience=5)
    model, record = cnn.train(
        feats,
        labels.astype(np.int64),
        np.arange(80),
        np.arange(80, n),
        cfg,
        torch.device("cpu"),
        log=lambda m: None,
    )
    probs = cnn.predict(model, feats, np.arange(80, n))
    assert probs.shape == (40, 4) and np.allclose(probs.sum(1), 1, atol=1e-5)
    assert record["best_step"] in (20, 40)
