"""Evaluate a Keras-2 legacy HDF5 model in torch.

The group's Keras artifacts were saved by Keras 2.8, and there is no
TensorFlow build for this environment's Python (conda-forge ships
tensorflow-cpu 2.21 for py312 only, against this repo's `python <3.12`
pin). The graphs in the roster are small feed-forward networks whose layers
have closed-form inference semantics, so instead of a framework we read the
serialized config and the moving statistics out of the file and evaluate the
graph directly. The arithmetic is torch, not numpy: the lab standardizes on
torch (IGNITE is torch), and a second numerical framework living only here
was a maintenance liability nobody asked for. Torch is an implementation
detail of this evaluator alone - `__call__` and `predict_members` still take
and return plain numpy arrays, so nothing downstream has to know torch is
there.

Supported layers: InputLayer, BatchNormalization, Conv1D, MaxPooling1D,
Flatten, Dense, Concatenate, Dropout (identity at inference). Anything else
raises UnsupportedLayer, naming the class, rather than silently skipping it.

Equality with TensorFlow is not assumed: `labelmaker.validate.adapter_fidelity`
compares this evaluator against a real Keras load of the same file and
stores the golden outputs under tests/labelmaker/data/. Measured against
that golden file: float64 max_abs_diff = 5.600518791837317e-05, float32
max_abs_diff = 8.20159912109375e-05 - the same order of magnitude, not two
orders apart, so switching this evaluator's own arithmetic to float32 does
NOT make the residual collapse toward float32 eps (~1e-6). That falsifies
"TensorFlow's own float32 rounding against our float64 arithmetic", the
hypothesis Task 14 guessed but could not test.

Tracing every layer's output for one member (float64 vs float32, same
inputs) localizes the sensitivity: every one of the ten members' single
worst-diff row lands on the tearing-logit column, never `betan` - the
graph's only unbounded, non-saturating (`linear`) output, at the end of a
~15-layer BatchNorm/Conv1D/Dense chain whose other six activations are all
sigmoids that absorb small perturbations once saturated. One golden-file
profile row does carry a physically implausible `cer_rot_csaps_1d` value
(6,437,599, five orders of magnitude past the model's trained range and the
downstream `DomainRule(\"rot_zipfit\", \"absmax\", hi=150.0)`) that blows the
first BatchNormalization's output to ~2.3e5 and produces a large float64-
vs-float32 divergence right at that layer (~2.9e-2) - but that row is not
among the actual worst rows against the golden reference, so sigmoid
saturation elsewhere absorbs it; it is a real oddity in the reference data,
not the explanation for the measured residual. The residual itself is
ordinary float32-scale rounding, compounded across the chain and expressed
on the one linear output - see `validate.adapter_fidelity`'s docstring for
the full reasoning and why the tolerance is unchanged at 1e-4.

`load_graph`/`load_ensemble` take a `dtype` (default `torch.float64`,
matching this evaluator's historical numpy precision) and convert every
weight to a torch tensor of that dtype once, at load time - not per call,
since `predict_members` runs a graph over the whole corpus. `predict_members`
itself has no dtype of its own: it evaluates whatever dtype the graphs
passed to it were loaded with, so comparing dtypes means loading the
ensemble twice (see `validate.adapter_fidelity`).
"""
from __future__ import annotations

import json
import multiprocessing
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

# torch.sigmoid/tanh are numerically stable at any magnitude - unlike the
# numpy evaluator this replaces, which needed a hand-written sign-split
# _sigmoid to avoid `RuntimeWarning: overflow encountered in exp` under this
# suite's `-W error` (a Conv1D sigmoid gate reaches a pre-activation of
# ~1.4e5 on the real reference inputs). That overflow class of bug does not
# exist in torch, so there is nothing to port here.
_ACTIVATIONS = {
    None: lambda x: x,
    "linear": lambda x: x,
    "sigmoid": torch.sigmoid,
    "relu": torch.relu,
    "tanh": torch.tanh,
    "softmax": lambda x: F.softmax(x, dim=-1),
}


class UnsupportedLayer(RuntimeError):
    """A layer class or option this evaluator does not implement."""


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


def _conv1d(x: torch.Tensor, kernel: torch.Tensor, bias: torch.Tensor | None, cfg):
    """`x` is Keras-layout `(N, L, C_in)`; only this function sees `(N, C, L)`.

    `kernel` is Keras layout `(kernel_size, in_ch, out_ch)`; `permute(2, 1, 0)`
    is a view (no copy) to torch's `(out_ch, in_ch, kernel_size)`.

    `'same'` padding is computed exactly as the numpy evaluator this replaces
    computed it, not via `F.conv1d(padding='same')`: torch's `'same'` puts
    the extra pad on the *left* for an odd amount where TensorFlow puts it on
    the *right*, and torch's `'same'` also rejects stride > 1. Padding
    manually with `F.pad` and passing `padding=0` to the convolution keeps
    the existing, TF-matching semantics.
    """
    k, stride = kernel.shape[0], int(cfg["strides"][0])
    dil = int(cfg.get("dilation_rate", [1])[0])
    eff = (k - 1) * dil + 1
    padding = cfg.get("padding", "valid")
    xt = x.transpose(1, 2)  # (N, C_in, L)
    if padding == "same":
        length = xt.shape[2]
        need = max(0, (int(np.ceil(length / stride)) - 1) * stride + eff - length)
        left = need // 2
        xt = F.pad(xt, (left, need - left))
    elif padding != "valid":
        raise UnsupportedLayer(f"Conv1D padding {padding!r}")
    out_len = (xt.shape[2] - eff) // stride + 1
    if out_len <= 0:
        raise ValueError(f"Conv1D input too short: {tuple(x.shape)} for kernel {k}")
    weight = kernel.permute(2, 1, 0)  # (out_ch, in_ch, kernel_size)
    out = F.conv1d(xt, weight, bias=bias, stride=stride, dilation=dil)
    return out.transpose(1, 2)  # back to (N, L_out, C_out)


def _maxpool1d(x: torch.Tensor, cfg):
    """`x` is Keras-layout `(N, L, C)`; transpose in, `F.max_pool1d`, transpose out."""
    pool = int(cfg["pool_size"][0])
    stride = int((cfg.get("strides") or [pool])[0])
    if cfg.get("padding", "valid") != "valid":
        raise UnsupportedLayer(f"MaxPooling1D padding {cfg['padding']!r}")
    xt = x.transpose(1, 2)  # (N, C, L)
    out = F.max_pool1d(xt, kernel_size=pool, stride=stride)
    return out.transpose(1, 2)


def _batchnorm(x: torch.Tensor, w: dict[str, torch.Tensor], cfg):
    """Explicit affine form, not `F.batch_norm`.

    `F.batch_norm` normalizes dim 1 of an `(N, C, ...)` tensor, which would
    force a transpose whenever `axis != 1` - and every graph in the roster
    keeps Keras' `axis == -1` on tensors already in Keras' `(N, L, C)` (or
    `(N, C)`) layout. The explicit form honours whatever `axis` the config
    names directly and is exactly equivalent to `F.batch_norm` at inference.
    """
    axis = cfg.get("axis", -1)
    axis = int(axis[0] if isinstance(axis, (list, tuple)) else axis)
    n = x.shape[axis]
    shape = [1] * x.ndim
    shape[axis] = n
    dtype = x.dtype
    gamma = w.get("gamma", torch.ones(n, dtype=dtype))
    beta = w.get("beta", torch.zeros(n, dtype=dtype))
    mean = w.get("moving_mean", torch.zeros(n, dtype=dtype))
    var = w.get("moving_variance", torch.ones(n, dtype=dtype))
    eps = float(cfg.get("epsilon", 1e-3))
    scaled = (x - mean.reshape(shape)) / torch.sqrt(var.reshape(shape) + eps)
    return gamma.reshape(shape) * scaled + beta.reshape(shape)


def _apply(layer: dict, ins: list[torch.Tensor], w: dict[str, torch.Tensor]):
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
        return torch.cat(ins, dim=int(cfg.get("axis", -1)))
    if cls == "Dropout":
        return ins[0]
    raise UnsupportedLayer(f"layer class {cls} ({cfg.get('name')})")


@dataclass(frozen=True)
class KerasGraph:
    """A loaded graph: config, weights, and the order of its inputs."""

    config: dict
    weights: dict[str, dict[str, torch.Tensor]]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    input_shapes: dict[str, tuple]
    dtype: torch.dtype = torch.float64

    def __call__(self, inputs) -> list[np.ndarray]:
        feed = self._as_dict(inputs)
        tensors: dict[str, torch.Tensor] = {}
        layers = {lay["config"]["name"]: lay for lay in self.config["config"]["layers"]}
        with torch.no_grad():
            for name, lay in layers.items():
                if lay["class_name"] == "InputLayer":
                    tensors[name] = torch.as_tensor(
                        np.asarray(feed[name]), dtype=self.dtype
                    )
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
            return [tensors[name].numpy() for name in self.output_names]

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


def load_graph(path, dtype: torch.dtype = torch.float64) -> KerasGraph:
    """Read one legacy `.h5` model file, converting its weights to torch once."""
    with h5py.File(path, "r") as f:
        raw = f.attrs["model_config"]
        config = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        weights: dict[str, dict[str, torch.Tensor]] = {}
        group = f["model_weights"]
        for lname in group:
            g = group[lname]
            per_layer: dict[str, torch.Tensor] = {}
            for wname in g.attrs.get("weight_names", []):
                key = (wname.decode() if isinstance(wname, bytes) else wname)
                short = key.split("/")[-1].split(":")[0]
                per_layer[short] = torch.as_tensor(np.asarray(g[key]), dtype=dtype)
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
        dtype=dtype,
    )


def load_ensemble(paths, dtype: torch.dtype = torch.float64) -> tuple[KerasGraph, ...]:
    """Load ensemble members in the order given.

    Also pins torch to a single intra-op thread, but only inside a forked
    worker. `run.py` forks a worker pool (default `--workers 8`, one process
    per shot batch) and each worker calls this once per model via
    `_predictor`; torch's default intra-op thread count is the machine's
    core count, so N workers x that many threads oversubscribes a shared
    cluster node - a well-known multiprocessing-plus-torch pathology. Set
    here, in the only place a forked worker touches torch, rather than in
    `run.py`: `run.py` is otherwise framework-agnostic (it dispatches to
    `adapter.load`/`predict` without knowing or caring what runs underneath)
    and has no other reason to import torch.

    Conditioned on `multiprocessing.parent_process() is not None` - true
    only inside a process that multiprocessing itself started, i.e. inside
    one of `run.py`'s forked pool workers (including the `--workers 1`
    in-process path, which never forks and so never hits this branch: see
    `_run_pool`) - because this used to fire unconditionally from a library
    loader, mutating global torch state in whatever process called it. IGNITE
    is also torch, in this same repo and environment: a notebook, a combined
    script, or a single pytest session that touches this runner and then
    runs IGNITE was silently pinned to one thread with no way to see why.
    """
    if multiprocessing.parent_process() is not None:
        torch.set_num_threads(1)
    return tuple(load_graph(Path(p), dtype=dtype) for p in paths)


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
            raise ValueError(
                f"expected a single-output graph, got {len(got)} outputs: "
                f"{g.output_names}"
            )
        outs.append(np.atleast_2d(got[0]))
    return np.stack(outs, axis=0)
