"""The retrainable copies of the two tearing models, and the targets they are given."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from labeler.models.runners import keras_h5
from labeler.tearing import detectors


def test_published_dsm_survival_alarm_is_converted_to_risk():
    assert detectors.DSM_SURVIVAL_THRESHOLD == pytest.approx(0.7)
    assert detectors.DSM_THRESHOLD == pytest.approx(0.3)


def test_magnetic_batch_inputs_do_not_depend_on_targets_or_uncertainty():
    script = Path(__file__).resolve().parents[2] / "scripts/labeler/tm_ours.py"
    spec = importlib.util.spec_from_file_location("tm_ours_batch_regression", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    centres = np.arange(3.0)
    features = np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, 5.0]])
    first = {1: (centres, features, np.array([1, 0, 1]), np.array([1, 1, 0]))}
    second = {1: (centres, features, np.array([0, 1, 0]), np.array([0, 1, 0]))}
    mean, std = np.array([1.0, 1.0]), np.array([2.0, 1.0])
    x1, y1, mask1 = module.batch(first, [1], mean, std, "cpu")
    x2, y2, mask2 = module.batch(second, [1], mean, std, "cpu")
    torch.testing.assert_close(x1, x2)
    # Finite features survive target uncertainty; missing feature bins are zero.
    torch.testing.assert_close(x1[0], torch.tensor([[0.0, 1.0, 0.0], [1.0, 3.0, 0.0]]))
    assert not torch.equal(y1, y2)
    assert not torch.equal(mask1, mask2)
    assert mask1.tolist() == [[1.0, 1.0, 0.0]]
    assert mask2.tolist() == [[0.0, 1.0, 0.0]]


def _layer(cls, name, inbound, **config):
    nodes = [[[d, 0, 0, {}] for d in inbound]] if inbound else []
    return {
        "class_name": cls,
        "config": {"name": name, **config},
        "inbound_nodes": nodes,
    }


def _tiny_graph() -> keras_h5.KerasGraph:
    """The onset CNN's graph at toy sizes: a profile branch, scalars joined in."""
    layers = [
        _layer("InputLayer", "input_2", [], batch_input_shape=[None, 8, 2]),
        _layer(
            "BatchNormalization",
            "bn0",
            ["input_2"],
            axis=[2],
            momentum=0.9,
            epsilon=1e-3,
        ),
        _layer(
            "Conv1D",
            "conv",
            ["bn0"],
            filters=3,
            kernel_size=[3],
            strides=[1],
            padding="valid",
            activation="sigmoid",
            use_bias=True,
            dilation_rate=[1],
        ),
        _layer(
            "MaxPooling1D",
            "pool",
            ["conv"],
            pool_size=[2],
            strides=[2],
            padding="valid",
        ),
        _layer("Flatten", "flat", ["pool"]),
        _layer("Dense", "d0", ["flat"], units=4, activation="sigmoid", use_bias=True),
        _layer("InputLayer", "input_1", [], batch_input_shape=[None, 3]),
        _layer("Concatenate", "cat", ["d0", "input_1"], axis=-1),
        _layer(
            "BatchNormalization", "bn1", ["cat"], axis=[1], momentum=0.9, epsilon=1e-3
        ),
        _layer("Dropout", "drop", ["bn1"], rate=0.2),
        _layer("Dense", "d1", ["drop"], units=2, activation="linear", use_bias=True),
    ]
    config = {
        "config": {
            "layers": layers,
            "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]],
            "output_layers": [["d1", 0, 0]],
        }
    }
    shapes = {
        "bn0": {
            "gamma": (2,),
            "beta": (2,),
            "moving_mean": (2,),
            "moving_variance": (2,),
        },
        "conv": {"kernel": (3, 2, 3), "bias": (3,)},
        "d0": {"kernel": (9, 4), "bias": (4,)},
        "bn1": {
            "gamma": (7,),
            "beta": (7,),
            "moving_mean": (7,),
            "moving_variance": (7,),
        },
        "d1": {"kernel": (7, 2), "bias": (2,)},
    }
    weights = {
        name: {short: torch.zeros(shape) for short, shape in group.items()}
        for name, group in shapes.items()
    }
    weights["top_level_model_weights"] = {}  # a group the files carry, with no layer
    return keras_h5.KerasGraph(
        config=config,
        weights=weights,
        input_names=("input_1", "input_2"),
        output_names=("d1",),
        input_shapes={"input_1": (None, 3), "input_2": (None, 8, 2)},
        dtype=torch.float32,
    )


def _toy(n=600, seed=0):
    """Scalars and profiles whose first scalar and profile mean carry the label."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(np.float32)
    scalars = rng.normal(size=(n, 3)).astype(np.float32)
    profiles = rng.normal(size=(n, 8, 2)).astype(np.float32)
    scalars[:, 0] += 2.5 * y
    profiles[:, :, 1] += 1.5 * y[:, None]
    return (scalars, profiles), y


def test_row_labels_are_positive_in_a_present_span_and_ignored_where_unsure():
    rows = pd.DataFrame(
        {
            "category": [1, 2, 3, 1],
            "t_start": [1000.0, 2000.0, 3000.0, 4000.0],
            "t_end": [1500.0, 2200.0, 3300.0, 4000.0],  # the last is a point: no span
        }
    )
    t = np.array([500.0, 1000.0, 1400.0, 2100.0, 3100.0, 4000.0, 6000.0])
    y, valid = detectors.row_labels(rows, t, (600.0, 5000.0))
    assert y.tolist() == [0, 1, 1, 0, 0, 0, 0]
    # 500 is before the window and 6000 after; 2100 and 3100 are in unsure spans
    assert valid.tolist() == [False, True, True, False, False, True, False]


def test_rows_to_bins_interpolates_valid_rows_and_leaves_a_gap_across_an_invalid_one():
    t = np.arange(0.0, 200.0, 25.0)
    score = np.linspace(0.0, 0.7, len(t))
    valid = np.ones(len(t), dtype=bool)
    valid[4] = False
    centres = np.array([5.0, 37.5, 85.0, 105.0, 130.0, 165.0])
    got = detectors.rows_to_bins(t, valid, score, centres)
    assert got[0] == pytest.approx(0.1 * 5.0 / 25.0)
    assert got[1] == pytest.approx(0.1 * 1.5)
    assert np.isnan(got[3])  # between the valid rows at 75 and 125
    shifted = detectors.rows_to_bins(t, valid, score, centres, shift_ms=25.0)
    # a score stamped t describes t + 25 ms: the record now starts at 25 ms
    assert np.isnan(shifted[0])
    assert shifted[1] == pytest.approx(0.05)


def test_onset_within_keeps_only_rows_before_the_onset():
    t = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    truth, keep = detectors.onset_within(t, 1.5, 0.5)
    assert keep.tolist() == [True, True, True, False, False]
    assert truth[:3].tolist() == [False, False, True]  # 1.5 - t <= 0.5 from t = 1.0
    truth, keep = detectors.onset_within(t, None, 0.5)
    assert keep.all()
    assert not truth.any()


def test_keras_detector_matches_the_keras_evaluator_in_eval_mode():
    graph = _tiny_graph()
    net = detectors.KerasDetector(graph, column=1, seed=3).double()
    # give the batch-norm statistics something other than identity
    with torch.no_grad():
        net.bn0__moving_mean.add_(0.3)
        net.bn1__moving_variance.mul_(2.0)
    net.eval()
    (scalars, profiles), _ = _toy(40)
    weights = {}
    for name, keys in net.keys.items():
        weights[name] = {k: getattr(net, f"{name}__{k}").detach().clone() for k in keys}
    twin = keras_h5.KerasGraph(
        config=graph.config,
        weights=weights,
        input_names=graph.input_names,
        output_names=graph.output_names,
        input_shapes=graph.input_shapes,
        dtype=torch.float64,
    )
    want = twin([scalars.astype(np.float64), profiles.astype(np.float64)])[0][:, 1]
    with torch.no_grad():
        got = net(
            torch.as_tensor(scalars, dtype=torch.float64),
            torch.as_tensor(profiles, dtype=torch.float64),
        ).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


def test_keras_detector_starts_from_fresh_weights_and_updates_its_batch_statistics():
    net = detectors.KerasDetector(_tiny_graph(), seed=1)
    other = detectors.KerasDetector(_tiny_graph(), seed=2)
    assert not torch.equal(net.d0__kernel, other.d0__kernel)
    assert torch.equal(net.d0__bias, torch.zeros(4))
    assert torch.equal(net.bn1__gamma, torch.ones(7))
    (scalars, profiles), _ = _toy(64)
    net.train()
    net(torch.as_tensor(scalars), torch.as_tensor(profiles))
    assert not torch.equal(net.bn1__moving_mean, torch.zeros(7))
    before = net.bn1__moving_mean.clone()
    net.eval()
    net(torch.as_tensor(scalars), torch.as_tensor(profiles))
    assert torch.equal(net.bn1__moving_mean, before)  # evaluation leaves them alone


def test_fit_learns_a_separable_toy_target_and_restores_the_best_epoch():
    inputs, y = _toy(800, seed=0)
    val_inputs, val_y = _toy(300, seed=1)
    net = detectors.KerasDetector(_tiny_graph(), seed=0)
    start = detectors.predict(net, val_inputs)
    record = detectors.fit(
        net,
        (inputs, y),
        (val_inputs, val_y),
        lr=5e-3,
        batch_size=64,
        max_epochs=60,
        patience=6,
        seed=0,
    )
    prob = detectors.predict(net, val_inputs)
    order = np.argsort(prob)
    ranks = np.empty(len(prob))
    ranks[order] = np.arange(len(prob))
    pos = val_y > 0.5
    auroc = (ranks[pos].sum() - pos.sum() * (pos.sum() - 1) / 2) / (
        pos.sum() * (~pos).sum()
    )
    assert auroc > 0.9
    assert not np.allclose(prob, start)
    assert record["best_epoch"] <= record["epochs"] - 1
    assert record["best_val_loss"] < 0.69


def test_dsm_detector_has_the_embedding_without_biases_and_one_logit():
    net = detectors.DsmDetector((6, 5, 4), seed=0)
    linears = [m for m in net.trunk if isinstance(m, torch.nn.Linear)]
    assert [tuple(m.weight.shape) for m in linears] == [(5, 6), (4, 5)]
    assert all(m.bias is None for m in linears)
    assert sum(isinstance(m, torch.nn.ReLU6) for m in net.trunk) == 2
    out = net(torch.zeros(7, 6))
    assert out.shape == (7,)
    inputs, y = _toy(500, seed=2)
    x = np.hstack([inputs[0], inputs[1].mean(axis=1)]).astype(np.float32)  # (n, 5)
    net = detectors.DsmDetector((5, 8), seed=0)
    detectors.fit(
        net, ((x,), y), ((x,), y), lr=1e-2, batch_size=64, max_epochs=30, seed=0
    )
    assert (
        detectors.predict(net, (x,))[y > 0.5].mean()
        > detectors.predict(net, (x,))[y < 0.5].mean()
    )
