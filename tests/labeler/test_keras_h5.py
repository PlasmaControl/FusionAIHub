"""The torch evaluator reproduces Keras semantics layer by layer.

Every expectation here is hand-computed from the layer definition, so the
tests are an independent oracle rather than a recording of our own output.
Equality with TensorFlow itself is Task 14.
"""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from labeler.models.runners.keras_h5 import (
    UnsupportedLayer,
    load_ensemble,
    load_graph,
    predict_members,
)

TM_UPSTREAM = Path(
    "/projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w"
)


def _layer(cls, name, inbound, **cfg):
    node = [[[i, 0, 0, {}] for i in inbound]] if inbound else []
    return {"class_name": cls, "config": {"name": name, **cfg}, "inbound_nodes": node}


def _write_legacy_h5(path, layers, input_layers, output_layers, weights):
    """A Keras-2.8-shaped legacy HDF5 file, written without TensorFlow."""
    cfg = {
        "class_name": "Functional",
        "config": {
            "name": "m",
            "layers": layers,
            "input_layers": input_layers,
            "output_layers": output_layers,
        },
    }
    with h5py.File(path, "w") as f:
        f.attrs["keras_version"] = "2.8.0"
        f.attrs["backend"] = "tensorflow"
        f.attrs["model_config"] = json.dumps(cfg)
        mw = f.create_group("model_weights")
        for lname in [lay["config"]["name"] for lay in layers]:
            g = mw.create_group(lname)
            names = []
            for wname, arr in weights.get(lname, {}).items():
                full = f"{lname}/{wname}:0"
                g.create_dataset(full, data=np.asarray(arr, dtype=np.float32))
                names.append(full.encode())
            g.attrs["weight_names"] = names


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def test_dense_with_sigmoid_matches_hand_computation(tmp_path):
    p = tmp_path / "m.h5"
    # Every value here is an exact binary fraction, so the float32 round trip
    # through the HDF5 is lossless and rtol=1e-12 tests the arithmetic rather
    # than the storage. 0.1 would not be: float32(0.1) differs from
    # float64(0.1) by ~1.5e-9 relative.
    kernel = np.array([[1.0, -2.0], [0.5, 0.25], [0.0, 3.0]])
    bias = np.array([0.25, -0.5])
    _write_legacy_h5(
        p,
        [
            _layer("InputLayer", "in", [], batch_input_shape=[None, 3], dtype="float32"),
            _layer("Dense", "d", ["in"], units=2, activation="sigmoid", use_bias=True),
        ],
        [["in", 0, 0]],
        [["d", 0, 0]],
        {"d": {"kernel": kernel, "bias": bias}},
    )
    g = load_graph(p)
    assert g.input_names == ("in",) and g.output_names == ("d",)
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    out = g([x])[0]
    np.testing.assert_allclose(out, _sigmoid(x @ kernel + bias), rtol=1e-12)


def test_batchnorm_uses_moving_statistics(tmp_path):
    p = tmp_path / "m.h5"
    stats = {
        "gamma": np.array([2.0, 1.0]),
        "beta": np.array([0.5, -0.5]),
        "moving_mean": np.array([1.0, 2.0]),
        "moving_variance": np.array([4.0, 9.0]),
    }
    _write_legacy_h5(
        p,
        [
            _layer("InputLayer", "in", [], batch_input_shape=[None, 2], dtype="float32"),
            _layer("BatchNormalization", "bn", ["in"], axis=[1], epsilon=1e-3),
        ],
        [["in", 0, 0]],
        [["bn", 0, 0]],
        {"bn": stats},
    )
    x = np.array([[3.0, 5.0]])
    out = load_graph(p)([x])[0]
    want = stats["gamma"] * (x - stats["moving_mean"]) / np.sqrt(
        stats["moving_variance"] + 1e-3
    ) + stats["beta"]
    np.testing.assert_allclose(out, want, rtol=1e-12)


def test_conv1d_valid_then_maxpool_matches_hand_computation(tmp_path):
    p = tmp_path / "m.h5"
    kernel = np.array([[[1.0]], [[1.0]]])          # (k=2, cin=1, cout=1), sum of pairs
    _write_legacy_h5(
        p,
        [
            _layer(
                "InputLayer", "in", [], batch_input_shape=[None, 5, 1], dtype="float32"
            ),
            _layer(
                "Conv1D", "c", ["in"], filters=1, kernel_size=[2], strides=[1],
                padding="valid", dilation_rate=[1], activation="linear", use_bias=True,
            ),
            _layer("MaxPooling1D", "mp", ["c"], pool_size=[2], strides=[2],
                   padding="valid"),
        ],
        [["in", 0, 0]],
        [["mp", 0, 0]],
        {"c": {"kernel": kernel, "bias": np.array([0.0])}},
    )
    x = np.arange(5.0).reshape(1, 5, 1)            # 0 1 2 3 4
    out = load_graph(p)([x])[0]
    # conv -> [1, 3, 5, 7]; maxpool/2 -> [3, 7]
    np.testing.assert_allclose(out[0, :, 0], [3.0, 7.0], rtol=1e-12)


def test_conv1d_same_padding_even_kernel_stride2_pads_right_not_left(tmp_path):
    """Regression coverage for the port's own docstring warning.

    `torch.nn.functional`'s native `padding='same'` puts the extra pad on
    the *left* for an odd amount and rejects `stride > 1` outright, where
    TensorFlow (and `_conv1d`'s manual `F.pad`) puts it on the *right*. This
    was differentially tested against the pre-port numpy evaluator over
    1,152 Conv1D configurations before the port, but nothing in the repo
    exercised it as a standalone case - so a regression to torch's native
    `'same'` would previously have shipped unnoticed.

    kernel_size=2 (even), stride=2, input length 5: TF-style 'same' needs 1
    padding sample total, and `_conv1d`'s `need // 2` puts it (`left=0`)
    entirely on the right. With kernel [1, 0] (i.e. `out[t]` reads only the
    first tap of each stride-2 window) and input [21, 21, 43, 43, 5], the
    right-padded convolution reads [21, 21, 43, 43, 5, pad] -> [21, 43, 5].
    A left-padded ('same' done torch's native way, `left=1`) convolution
    would instead read [pad, 21, 21, 43, 43, 5] -> [0, 21, 43], dropping the
    true last output entirely.
    """
    p = tmp_path / "m.h5"
    kernel = np.array([[[1.0]], [[0.0]]])           # (k=2, cin=1, cout=1)
    _write_legacy_h5(
        p,
        [
            _layer(
                "InputLayer", "in", [], batch_input_shape=[None, 5, 1], dtype="float32"
            ),
            _layer(
                "Conv1D", "c", ["in"], filters=1, kernel_size=[2], strides=[2],
                padding="same", dilation_rate=[1], activation="linear", use_bias=True,
            ),
        ],
        [["in", 0, 0]],
        [["c", 0, 0]],
        {"c": {"kernel": kernel, "bias": np.array([0.0])}},
    )
    x = np.array([21.0, 21.0, 43.0, 43.0, 5.0]).reshape(1, 5, 1)
    out = load_graph(p)([x])[0]
    np.testing.assert_allclose(out[0, :, 0], [21.0, 43.0, 5.0], rtol=1e-12)


def test_conv1d_dilated_valid_matches_hand_computation(tmp_path):
    """dilation_rate=2, padding='valid': `out[t] = x[t] + 100*x[t+2]`."""
    p = tmp_path / "m.h5"
    kernel = np.array([[[1.0]], [[100.0]]])         # (k=2, cin=1, cout=1)
    _write_legacy_h5(
        p,
        [
            _layer(
                "InputLayer", "in", [], batch_input_shape=[None, 6, 1], dtype="float32"
            ),
            _layer(
                "Conv1D", "c", ["in"], filters=1, kernel_size=[2], strides=[1],
                padding="valid", dilation_rate=[2], activation="linear", use_bias=True,
            ),
        ],
        [["in", 0, 0]],
        [["c", 0, 0]],
        {"c": {"kernel": kernel, "bias": np.array([0.0])}},
    )
    x = np.arange(6.0).reshape(1, 6, 1)              # 0 1 2 3 4 5
    out = load_graph(p)([x])[0]
    np.testing.assert_allclose(out[0, :, 0], [200.0, 301.0, 402.0, 503.0], rtol=1e-12)


def test_conv1d_same_output_length_is_ceil_of_length_over_stride(tmp_path):
    """`'same'` padding always outputs `ceil(L / stride)`, any kernel/stride."""
    p = tmp_path / "m.h5"
    kernel = np.ones((3, 1, 1))                      # (k=3, cin=1, cout=1)
    length, stride = 10, 3
    _write_legacy_h5(
        p,
        [
            _layer(
                "InputLayer", "in", [], batch_input_shape=[None, length, 1],
                dtype="float32",
            ),
            _layer(
                "Conv1D", "c", ["in"], filters=1, kernel_size=[3], strides=[stride],
                padding="same", dilation_rate=[1], activation="linear", use_bias=False,
            ),
        ],
        [["in", 0, 0]],
        [["c", 0, 0]],
        {"c": {"kernel": kernel}},
    )
    x = np.arange(float(length)).reshape(1, length, 1)
    out = load_graph(p)([x])[0]
    assert out.shape[1] == -(-length // stride)      # ceil division


def test_flatten_is_channels_last(tmp_path):
    p = tmp_path / "m.h5"
    _write_legacy_h5(
        p,
        [
            _layer(
                "InputLayer", "in", [], batch_input_shape=[None, 2, 3], dtype="float32"
            ),
            _layer("Flatten", "fl", ["in"]),
        ],
        [["in", 0, 0]],
        [["fl", 0, 0]],
        {},
    )
    x = np.arange(6.0).reshape(1, 2, 3)
    np.testing.assert_allclose(load_graph(p)([x])[0][0], np.arange(6.0))


def test_dropout_is_identity_and_graph_order_may_be_arbitrary(tmp_path):
    # `input_1` is listed *after* the layer that consumes it, exactly as in the
    # real artifact, so the evaluator cannot assume the list is topological.
    p = tmp_path / "m.h5"
    _write_legacy_h5(
        p,
        [
            _layer("InputLayer", "a", [], batch_input_shape=[None, 2], dtype="float32"),
            _layer("Concatenate", "cat", ["a", "b"], axis=-1),
            _layer("Dropout", "dr", ["cat"], rate=0.2),
            _layer("InputLayer", "b", [], batch_input_shape=[None, 3], dtype="float32"),
        ],
        [["a", 0, 0], ["b", 0, 0]],
        [["dr", 0, 0]],
        {},
    )
    g = load_graph(p)
    a = np.array([[1.0, 2.0]])
    b = np.array([[3.0, 4.0, 5.0]])
    np.testing.assert_allclose(g({"a": a, "b": b})[0], [[1, 2, 3, 4, 5]])
    np.testing.assert_allclose(g([a, b])[0], [[1, 2, 3, 4, 5]])


def test_a_dict_feed_missing_an_input_is_named_in_the_error(tmp_path):
    p = tmp_path / "m.h5"
    _write_legacy_h5(
        p,
        [
            _layer("InputLayer", "a", [], batch_input_shape=[None, 2], dtype="float32"),
            _layer("Concatenate", "cat", ["a", "b"], axis=-1),
            _layer("InputLayer", "b", [], batch_input_shape=[None, 3], dtype="float32"),
        ],
        [["a", 0, 0], ["b", 0, 0]],
        [["cat", 0, 0]],
        {},
    )
    with pytest.raises(ValueError, match="'b'"):
        load_graph(p)({"a": np.zeros((1, 2))})


def test_unsupported_layer_names_the_class(tmp_path):
    p = tmp_path / "m.h5"
    _write_legacy_h5(
        p,
        [
            _layer("InputLayer", "in", [], batch_input_shape=[None, 2], dtype="float32"),
            _layer("LSTM", "l", ["in"], units=4),
        ],
        [["in", 0, 0]],
        [["l", 0, 0]],
        {},
    )
    with pytest.raises(UnsupportedLayer, match="LSTM"):
        load_graph(p)([np.zeros((1, 2))])


def test_wrong_number_of_inputs_is_an_error(tmp_path):
    p = tmp_path / "m.h5"
    _write_legacy_h5(
        p,
        [
            _layer("InputLayer", "in", [], batch_input_shape=[None, 2], dtype="float32"),
            _layer("Dense", "d", ["in"], units=1, activation="linear", use_bias=False),
        ],
        [["in", 0, 0]],
        [["d", 0, 0]],
        {"d": {"kernel": np.ones((2, 1))}},
    )
    g = load_graph(p)
    with pytest.raises(ValueError):
        g([np.zeros((1, 2)), np.zeros((1, 2))])
    with pytest.raises(ValueError):
        g([np.zeros((1, 7))])        # wrong feature width


def test_predict_members_refuses_a_multi_output_graph(tmp_path):
    p = tmp_path / "m.h5"
    _write_legacy_h5(
        p,
        [
            _layer("InputLayer", "in", [], batch_input_shape=[None, 2], dtype="float32"),
            _layer("Dense", "a", ["in"], units=1, activation="linear", use_bias=False),
            _layer("Dense", "b", ["in"], units=1, activation="linear", use_bias=False),
        ],
        [["in", 0, 0]],
        [["a", 0, 0], ["b", 0, 0]],
        {"a": {"kernel": np.ones((2, 1))}, "b": {"kernel": np.ones((2, 1))}},
    )
    graphs = load_ensemble([p])
    with pytest.raises(ValueError, match="single-output"):
        predict_members(graphs, [np.zeros((1, 2))])


pytestmark_upstream = pytest.mark.skipif(
    not TM_UPSTREAM.exists(), reason=f"upstream weights not available: {TM_UPSTREAM}"
)


@pytestmark_upstream
def test_real_ensemble_members_have_different_layer_names():
    # MEASURED: all ten members were built in one Keras session, so the
    # global name counter ran on - inputs are input_1/input_2 for member 0,
    # input_3/input_4 for member 1, through input_19/input_20 for member 9,
    # and the outputs are dense_4, dense_9, ... dense_49. Only the
    # POSITIONAL order is common, which is why an ensemble is fed a
    # sequence and never a dict keyed on one member's names.
    graphs = load_ensemble(sorted(TM_UPSTREAM.glob("best_model_?_4c.h5")))
    assert len({g.input_names for g in graphs}) == 10
    assert len({g.output_names for g in graphs}) == 10
    for g in graphs:
        assert len(g.input_names) == 2 and len(g.output_names) == 1
        assert [g.input_shapes[n] for n in g.input_names] == [
            (None, 11), (None, 33, 5)
        ]


@pytestmark_upstream
def test_real_tearing_ensemble_loads_and_predicts():
    paths = sorted(TM_UPSTREAM.glob("best_model_?_4c.h5"))
    assert len(paths) == 10
    graphs = load_ensemble(paths)
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(7, 11))
    x1 = rng.normal(size=(7, 33, 5))
    members = predict_members(graphs, [x0, x1])       # positional, per above
    assert members.shape == (10, 7, 2)
    assert np.isfinite(members).all()
    # the members are different models, not ten copies
    assert members.std(axis=0).max() > 1e-6
    np.testing.assert_allclose(members, predict_members(graphs, [x0, x1]))


@pytestmark_upstream
def test_feeding_an_ensemble_by_name_fails_loudly():
    # The trap this guards: a dict keyed on member 0's names fits member 0
    # and raises for the other nine, rather than quietly mispredicting.
    graphs = load_ensemble(sorted(TM_UPSTREAM.glob("best_model_?_4c.h5")))
    rng = np.random.default_rng(0)
    feed = {"input_1": rng.normal(size=(3, 11)),
            "input_2": rng.normal(size=(3, 33, 5))}
    graphs[0](feed)                                    # member 0 is fine
    with pytest.raises(ValueError, match="missing input"):
        graphs[1](feed)
