"""Evaluate a Keras-2 legacy HDF5 model in numpy.

The group's Keras artifacts were saved by Keras 2.8, and there is no
TensorFlow build for this environment's Python (conda-forge ships
tensorflow-cpu 2.21 for py312 only, against this repo's `python <3.12`
pin). The graphs in the roster are small feed-forward networks whose layers
have closed-form inference semantics, so instead of a framework we read the
serialized config and the moving statistics out of the file and evaluate the
graph directly.

Supported layers: InputLayer, BatchNormalization, Conv1D, MaxPooling1D,
Flatten, Dense, Concatenate, Dropout (identity at inference). Anything else
raises UnsupportedLayer, naming the class, rather than silently skipping it.

Equality with TensorFlow is not assumed: `labelmaker.validate adapter`
compares this evaluator against a real Keras load of the same file to 1e-5
and stores the golden outputs under tests/labelmaker/data/.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


class UnsupportedLayer(RuntimeError):
    """A layer class or option this evaluator does not implement."""


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


_ACTIVATIONS = {
    None: lambda x: x,
    "linear": lambda x: x,
    "sigmoid": _sigmoid,
    "relu": lambda x: np.maximum(x, 0.0),
    "tanh": np.tanh,
    "softmax": _softmax,
}


def _activation(name):
    if name not in _ACTIVATIONS:
        raise UnsupportedLayer(f"activation {name!r}")
    return _ACTIVATIONS[name]


def _inbound_names(layer: dict) -> list[str]:
    """Names this layer consumes, in order, from either nesting Keras uses."""
    names: list[str] = []

    def walk(node):
        if isinstance(node, list):
            if node and isinstance(node[0], str):
                names.append(node[0])
            else:
                for item in node:
                    walk(item)

    walk(layer.get("inbound_nodes") or [])
    return names


def _conv1d(x, kernel, bias, cfg):
    k, stride = kernel.shape[0], int(cfg["strides"][0])
    dil = int(cfg.get("dilation_rate", [1])[0])
    eff = (k - 1) * dil + 1
    padding = cfg.get("padding", "valid")
    if padding == "same":
        need = max(0, (int(np.ceil(x.shape[1] / stride)) - 1) * stride + eff - x.shape[1])
        left = need // 2
        x = np.pad(x, ((0, 0), (left, need - left), (0, 0)))
    elif padding != "valid":
        raise UnsupportedLayer(f"Conv1D padding {padding!r}")
    out_len = (x.shape[1] - eff) // stride + 1
    if out_len <= 0:
        raise ValueError(f"Conv1D input too short: {x.shape} for kernel {k}")
    acc = np.zeros((x.shape[0], out_len, kernel.shape[2]), dtype=np.float64)
    for i in range(k):
        stop = i * dil + (out_len - 1) * stride + 1
        acc += x[:, i * dil : stop : stride, :] @ kernel[i]
    if bias is not None:
        acc += bias
    return acc


def _maxpool1d(x, cfg):
    pool = int(cfg["pool_size"][0])
    stride = int((cfg.get("strides") or [pool])[0])
    if cfg.get("padding", "valid") != "valid":
        raise UnsupportedLayer(f"MaxPooling1D padding {cfg['padding']!r}")
    out_len = (x.shape[1] - pool) // stride + 1
    idx = np.arange(out_len)[:, None] * stride + np.arange(pool)[None, :]
    return x[:, idx, :].max(axis=2)


def _batchnorm(x, w, cfg):
    axis = cfg.get("axis", -1)
    axis = int(axis[0] if isinstance(axis, (list, tuple)) else axis)
    n = x.shape[axis]
    shape = [1] * x.ndim
    shape[axis] = n
    gamma = w.get("gamma", np.ones(n))
    beta = w.get("beta", np.zeros(n))
    mean = w.get("moving_mean", np.zeros(n))
    var = w.get("moving_variance", np.ones(n))
    eps = float(cfg.get("epsilon", 1e-3))
    scaled = (x - mean.reshape(shape)) / np.sqrt(var.reshape(shape) + eps)
    return gamma.reshape(shape) * scaled + beta.reshape(shape)


def _apply(layer: dict, ins: list[np.ndarray], w: dict[str, np.ndarray]):
    cls, cfg = layer["class_name"], layer["config"]
    if cls == "Dense":
        out = ins[0] @ w["kernel"]
        if cfg.get("use_bias", True):
            out = out + w["bias"]
        return _activation(cfg.get("activation"))(out)
    if cls == "BatchNormalization":
        return _batchnorm(ins[0], w, cfg)
    if cls == "Conv1D":
        out = _conv1d(ins[0], w["kernel"], w.get("bias"), cfg)
        return _activation(cfg.get("activation"))(out)
    if cls == "MaxPooling1D":
        return _maxpool1d(ins[0], cfg)
    if cls == "Flatten":
        return ins[0].reshape(ins[0].shape[0], -1)
    if cls == "Concatenate":
        return np.concatenate(ins, axis=int(cfg.get("axis", -1)))
    if cls == "Dropout":
        return ins[0]
    raise UnsupportedLayer(f"layer class {cls} ({cfg.get('name')})")


@dataclass(frozen=True)
class KerasGraph:
    """A loaded graph: config, weights, and the order of its inputs."""

    config: dict
    weights: dict[str, dict[str, np.ndarray]]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    input_shapes: dict[str, tuple]

    def __call__(self, inputs) -> list[np.ndarray]:
        feed = self._as_dict(inputs)
        tensors: dict[str, np.ndarray] = {}
        layers = {lay["config"]["name"]: lay for lay in self.config["config"]["layers"]}
        for name, lay in layers.items():
            if lay["class_name"] == "InputLayer":
                tensors[name] = np.asarray(feed[name], dtype=np.float64)
        pending = [lay for lay in layers.values() if lay["class_name"] != "InputLayer"]
        while pending:
            ready = [
                lay for lay in pending
                if all(dep in tensors for dep in _inbound_names(lay))
            ]
            if not ready:
                stuck = [lay["config"]["name"] for lay in pending]
                raise UnsupportedLayer(f"cannot resolve graph at {stuck}")
            for lay in ready:
                name = lay["config"]["name"]
                ins = [tensors[dep] for dep in _inbound_names(lay)]
                tensors[name] = _apply(lay, ins, self.weights.get(name, {}))
                pending.remove(lay)
        return [tensors[name] for name in self.output_names]

    def _as_dict(self, inputs) -> dict[str, np.ndarray]:
        if isinstance(inputs, dict):
            feed = dict(inputs)
        else:
            seq = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
            if len(seq) != len(self.input_names):
                raise ValueError(
                    f"expected {len(self.input_names)} inputs "
                    f"{self.input_names}, got {len(seq)}"
                )
            feed = dict(zip(self.input_names, seq))
        for name in self.input_names:
            if name not in feed:
                raise ValueError(
                    f"missing input {name!r}; this graph expects {self.input_names}"
                )
        for name, want in self.input_shapes.items():
            got = np.asarray(feed[name]).shape
            if len(got) != len(want) or got[1:] != tuple(want[1:]):
                raise ValueError(f"input {name}: expected (N, *{want[1:]}), got {got}")
        return feed


def load_graph(path) -> KerasGraph:
    """Read one legacy `.h5` model file."""
    with h5py.File(path, "r") as f:
        raw = f.attrs["model_config"]
        config = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        weights: dict[str, dict[str, np.ndarray]] = {}
        group = f["model_weights"]
        for lname in group:
            g = group[lname]
            per_layer: dict[str, np.ndarray] = {}
            for wname in g.attrs.get("weight_names", []):
                key = (wname.decode() if isinstance(wname, bytes) else wname)
                short = key.split("/")[-1].split(":")[0]
                per_layer[short] = np.asarray(g[key], dtype=np.float64)
            weights[lname] = per_layer
    inner = config["config"]
    shapes = {
        lay["config"]["name"]: tuple(lay["config"]["batch_input_shape"])
        for lay in inner["layers"]
        if lay["class_name"] == "InputLayer"
    }
    return KerasGraph(
        config=config,
        weights=weights,
        input_names=tuple(entry[0] for entry in inner["input_layers"]),
        output_names=tuple(entry[0] for entry in inner["output_layers"]),
        input_shapes=shapes,
    )


def load_ensemble(paths) -> tuple[KerasGraph, ...]:
    """Load ensemble members in the order given."""
    return tuple(load_graph(Path(p)) for p in paths)


def predict_members(graphs, inputs) -> np.ndarray:
    """`(n_members, n_rows, n_out)` for single-output graphs.

    Pass `inputs` as a SEQUENCE, not a dict. Members trained in one Keras
    session carry different layer names - this project's tearing ensemble
    runs input_1/input_2 through input_19/input_20, with outputs dense_4
    through dense_49 - so only the positional order is common across
    members. A dict keyed on one member's names fits that member and raises
    for the rest.

    Statistics across members are the caller's business: the label store
    keeps the mean as the label and the min/max as its spread.
    """
    outs = []
    for g in graphs:
        got = g(inputs)
        if len(got) != 1:
            raise UnsupportedLayer(
                f"expected a single output, got {len(got)}: {g.output_names}"
            )
        outs.append(np.atleast_2d(got[0]))
    return np.stack(outs, axis=0)
