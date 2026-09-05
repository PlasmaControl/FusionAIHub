"""Evaluate an auton-survival Deep Survival Machines checkpoint from its pickle.

The upstream tearing-survival model is a pickle of the group's auton-survival
fork: `SurvivalModel` wrapping `DeepSurvivalMachines` wrapping a torch module.
Only those three classes are foreign; everything under them is plain torch
(`Sequential`, `Linear`, `ReLU6`, `Tanh`, `ModuleDict`, `ParameterDict`) and
plain numbers. So the pickle is read with an unpickler that stands in a shell
class for each of the three and refuses any other class it has not been told
about - the same "upstream bytes, our evaluator" posture as `keras_h5`, with
the weights verified by the card's sha256 before the file is opened.

Evaluation follows the fork's `dsm_torch._init_dsm_layers` (LogNormal: the
heads pass through tanh and add the learned `shape`/`scale` parameters, the
gate is divided by `temp`) and `losses._lognormal_cdf`:

    mu_k    = tanh(shapeg_k(h)) + shape_k
    sigma_k = tanh(scaleg_k(h)) + scale_k
    w       = softmax(gate(h) / temp)
    S(t)    = sum_k w_k * (0.5 - 0.5 * erf((ln t - mu_k) / (exp(sigma_k) sqrt 2)))

with `h` the ReLU6 embedding and `t` in the units the model was trained on
(milliseconds, for the tearing model). Only LogNormal is implemented: it is
what the checkpoint uses, and the Weibull/Normal branches would be code no
golden file exercises.
"""
from __future__ import annotations

import collections
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


class UnsupportedModel(ValueError):
    """The pickle holds a DSM variant this evaluator does not reproduce."""


class _Plain:
    """Stands in for a fork class whose instance is only a bag of attributes."""


class _TorchShell(torch.nn.Module):
    """Stands in for the fork's `DeepSurvivalMachinesTorch`.

    `Module.__setstate__` restores `_modules`, `_parameters` and the plain
    attributes (`k`, `dist`, `temp`, `risks`) from the pickled state, which is
    everything the evaluator reads. No forward is defined: the pickle's own
    forward is the fork's code, and this module reproduces it in numpy.
    """


_STAND_INS = {
    ("auton_survival.estimators", "SurvivalModel"): _Plain,
    ("auton_survival.models.dsm", "DeepSurvivalMachines"): _Plain,
    ("auton_survival.models.dsm.dsm_torch", "DeepSurvivalMachinesTorch"): _TorchShell,
}
_ALLOWED_ROOTS = ("torch", "numpy")


class RestrictedUnpickler(pickle.Unpickler):
    """Loads torch, numpy, `OrderedDict`, and the three fork classes as shells.

    Anything else is refused: a pickle is executable, and this one is trusted
    for exactly the classes a DSM checkpoint is known to contain.
    """

    def find_class(self, module, name):
        stand_in = _STAND_INS.get((module, name))
        if stand_in is not None:
            return stand_in
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module.split(".", 1)[0] in _ALLOWED_ROOTS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"{module}.{name} is not a class a Deep Survival Machines pickle needs"
        )


@dataclass(frozen=True)
class DsmGraph:
    """The numbers the evaluator needs, as float64 numpy arrays."""

    k: int
    dist: str
    temp: float
    embedding: tuple[np.ndarray, ...]        # (out, in) weights; ReLU6 after each
    gate: np.ndarray                          # (k, hidden), no bias
    scaleg: tuple[np.ndarray, np.ndarray]     # (k, hidden), (k,)
    shapeg: tuple[np.ndarray, np.ndarray]     # (k, hidden), (k,)
    shape: np.ndarray                         # (k,)
    scale: np.ndarray                         # (k,)


def _find_torch_model(obj):
    """Depth-first for the one `_TorchShell` in the pickle's nesting."""
    if isinstance(obj, _TorchShell):
        return obj
    if isinstance(obj, (list, tuple)):
        children = obj
    elif isinstance(obj, dict):
        children = obj.values()
    elif isinstance(obj, _Plain):
        children = vars(obj).values()
    else:
        return None
    for child in children:
        found = _find_torch_model(child)
        if found is not None:
            return found
    return None


def _f64(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().to(torch.float64).numpy()


def load_dsm(path) -> DsmGraph:
    """Read one DSM checkpoint pickle into a `DsmGraph`, or raise."""
    with open(Path(path), "rb") as fh:
        payload = RestrictedUnpickler(fh).load()
    tm = _find_torch_model(payload)
    if tm is None:
        raise UnsupportedModel(f"{path}: no DeepSurvivalMachinesTorch inside")
    if tm.dist != "LogNormal":
        raise UnsupportedModel(
            f"{path}: distribution {tm.dist!r}; only LogNormal is implemented"
        )
    if not isinstance(tm.act, torch.nn.Tanh):
        raise UnsupportedModel(f"{path}: head activation {type(tm.act).__name__}")
    if int(tm.risks) != 1:
        raise UnsupportedModel(f"{path}: {tm.risks} risks; only one is implemented")
    weights = []
    for layer in tm.embedding:
        if isinstance(layer, torch.nn.Linear):
            if layer.bias is not None:
                raise UnsupportedModel(f"{path}: embedding Linear with a bias")
            weights.append(_f64(layer.weight))
        elif not isinstance(layer, torch.nn.ReLU6):
            raise UnsupportedModel(f"{path}: embedding layer {type(layer).__name__}")
    risk = "1"
    gate = tm.gate[risk][0]
    if gate.bias is not None:
        raise UnsupportedModel(f"{path}: gate with a bias")
    scaleg, shapeg = tm.scaleg[risk][0], tm.shapeg[risk][0]
    return DsmGraph(
        k=int(tm.k), dist=str(tm.dist), temp=float(tm.temp),
        embedding=tuple(weights),
        gate=_f64(gate.weight),
        scaleg=(_f64(scaleg.weight), _f64(scaleg.bias)),
        shapeg=(_f64(shapeg.weight), _f64(shapeg.bias)),
        shape=_f64(tm.shape[risk]), scale=_f64(tm.scale[risk]),
    )


def mixture(graph: DsmGraph, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(log_w, mu, sigma)`, each `(n, k)`, in checkpoint component order."""
    h = torch.as_tensor(np.asarray(x, dtype=np.float64))
    for w in graph.embedding:
        h = torch.clamp(h @ torch.as_tensor(w).T, 0.0, 6.0)
    mu = torch.tanh(h @ torch.as_tensor(graph.shapeg[0]).T + torch.as_tensor(graph.shapeg[1]))
    mu = mu + torch.as_tensor(graph.shape)
    sigma = torch.tanh(h @ torch.as_tensor(graph.scaleg[0]).T + torch.as_tensor(graph.scaleg[1]))
    sigma = sigma + torch.as_tensor(graph.scale)
    log_w = torch.log_softmax(h @ torch.as_tensor(graph.gate).T / graph.temp, dim=1)
    return log_w.numpy(), mu.numpy(), sigma.numpy()


def survival(graph: DsmGraph, x, horizons_ms: Sequence[float]) -> np.ndarray:
    """`S(t | x)` for every row of `x` at every horizon: `(n, len(horizons))`."""
    log_w, mu, sigma = map(torch.as_tensor, mixture(graph, x))
    out = torch.empty((mu.shape[0], len(horizons_ms)), dtype=torch.float64)
    for j, t in enumerate(horizons_ms):
        z = (np.log(float(t)) - mu) / (torch.exp(sigma) * np.sqrt(2.0))
        s_k = 0.5 - 0.5 * torch.erf(z)
        out[:, j] = torch.exp(torch.logsumexp(torch.log(s_k) + log_w, dim=1))
    return out.numpy()


def quantiles(graph: DsmGraph, x: np.ndarray, q: Sequence[float]) -> np.ndarray:
    """Invert mixture survival for event-time quantiles in milliseconds."""
    log_w, mu, sigma = map(torch.as_tensor, mixture(graph, x))
    target = 1.0 - torch.as_tensor(np.asarray(q, dtype=np.float64))
    lo = torch.full((mu.shape[0], target.numel()), np.log(1e-3), dtype=torch.float64)
    hi = torch.full_like(lo, np.log(1e7))

    def at(log_t):
        z = (log_t[:, :, None] - mu[:, None, :]) / (torch.exp(sigma[:, None, :]) * np.sqrt(2.0))
        return ((0.5 - 0.5 * torch.erf(z)) * torch.exp(log_w[:, None, :])).sum(dim=2)

    bracketed = (at(lo) >= target) & (at(hi) <= target)
    bad_rows = int((~bracketed.all(dim=1)).sum())
    if bad_rows:
        raise ValueError(f"quantiles outside [1e-3, 1e7] ms bracket for {bad_rows} rows")
    # Bisection in log time resolves both short and long event times uniformly.
    for _ in range(80):
        mid = (lo + hi) * 0.5
        below = at(mid) > target
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    return torch.exp((lo + hi) * 0.5).numpy()


def gate_entropy(log_w: np.ndarray) -> np.ndarray:
    """Gate entropy in nats per row, with the limiting convention 0 ln 0 = 0."""
    log_w = np.asarray(log_w, dtype=np.float64)
    terms = np.zeros_like(log_w)
    np.multiply(np.exp(log_w), log_w, out=terms, where=np.isfinite(log_w))
    return -terms.sum(axis=1)
