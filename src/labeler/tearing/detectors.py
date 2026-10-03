"""The two published tearing models retrained as detectors, and the labels they meet.

`tm-onsetcnn` (the onset CNN) forecasts the mode's presence 25 ms ahead and `tm-dsm`
(the Deep Survival Machines model) forecasts an onset within 250 ms to 1 s. The
whole-interval labels ask a different question: is a tearing mode present *now*?
Retraining the same architecture on the same inputs with that target is the fair
version of the comparison; this module holds what the retraining needs.

* `row_labels` / `onset_within`: the target at a model's own rows (present at the
  row's time stamp; or an onset within a horizon, on rows before the onset).
* `KerasDetector`: a trainable copy of a Keras functional graph (the onset CNN's), built
  from the same config with fresh weights. Layer arithmetic is `runners.keras_h5`'s, so
  an untrained copy and a loaded graph evaluate alike; batch normalisation uses batch
  statistics while training and the moving statistics after, as Keras does.
* `DsmDetector`: the DSM's embedding (bias-free linear layers with ReLU6) and a
  single-logit head in place of the log-normal mixture, which a horizon-0 target does
  not have.
* `fit` / `predict`: a plain minibatch loop with early stopping on a validation loss.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..models.runners import keras_h5
from . import scoring

#: The onset CNN's score at `t` describes the mode's presence at `t + 25 ms`.
CNN_SHIFT_MS = 25.0
#: Published alarm levels: the CNN's `tm_prob` at 0.5 (its card's F1 convention) and the
#: DSM's default per-shot alarm at 0.7 (its card, "Bias, risks and limitations").
CNN_THRESHOLD = 0.5
DSM_THRESHOLD = 0.7
DSM_HORIZONS_S = (0.25, 0.5, 1.0)


def row_labels(rows, t_ms, window) -> tuple[np.ndarray, np.ndarray]:
    """`(y, valid)` of an interval table's rows at time stamps `t_ms` (ms).

    `y` is 1 where the stamp is in a present span. `valid` is False where the label
    cannot say (an uncertain or not-observable span) and outside the catalog `window`
    `(start_ms, end_ms)`.
    """
    t = np.asarray(t_ms, dtype=float)
    y, valid = scoring.label_bins(rows, t)
    return y, valid & (t >= window[0]) & (t <= window[1])


def rows_to_bins(t_ms, valid, score, centres, *, shift_ms: float = 0.0) -> np.ndarray:
    """A model's row scores on bin `centres`; rows it calls invalid are dropped first.

    A bin is interpolated only between rows of the model's own spacing, so one that sits
    across an invalid row has no score (`scoring.align_scores`).
    """
    kept = np.where(
        np.asarray(valid, dtype=bool), np.asarray(score, dtype=float), np.nan
    )
    return scoring.align_scores(t_ms, kept, centres, shift_ms=shift_ms)


def onset_within(t_s, onset_s, horizon_s) -> tuple[np.ndarray, np.ndarray]:
    """`(truth, keep)`: whether an onset comes within `horizon_s`, on rows before it.

    Once the mode is there "will one appear within h" is no longer the question being
    asked, so only rows before `onset_s` are kept (all rows when the shot has none).
    """
    t = np.asarray(t_s, dtype=float)
    if onset_s is None or not np.isfinite(onset_s):
        return np.zeros(t.shape, dtype=bool), np.ones(t.shape, dtype=bool)
    return (onset_s - t) <= horizon_s + 1e-9, t < onset_s - 1e-9


def _glorot(shape, fan_in, fan_out, gen) -> torch.Tensor:
    limit = float(np.sqrt(6.0 / (fan_in + fan_out)))
    return (torch.rand(shape, generator=gen) * 2.0 - 1.0) * limit


class KerasDetector(nn.Module):
    """A trainable Keras functional graph with fresh weights.

    `graph` is a loaded `keras_h5.KerasGraph`: only its config and weight shapes are
    used. Dense and Conv1D kernels start Glorot-uniform with zero biases, batch
    normalisation at identity (Keras' defaults). `forward(scalars, profiles)` returns
    the logit in output column `column`.
    """

    def __init__(self, graph: keras_h5.KerasGraph, *, column: int = 1, seed: int = 0):
        super().__init__()
        inner = graph.config["config"]
        self.layers = {lay["config"]["name"]: lay for lay in inner["layers"]}
        self.output_name = graph.output_names[0]
        self.column = column
        # Inputs are told apart by rank: scalars (N, F), profiles (N, rho, channels).
        self.slots = {
            name: 0 if len(shape) == 2 else 1
            for name, shape in graph.input_shapes.items()
        }
        self.order = self._order()
        gen = torch.Generator().manual_seed(seed)
        self.keys: dict[str, list[str]] = {}
        for name, shapes in graph.weights.items():
            self.keys[name] = sorted(shapes)
            cls = self.layers[name]["class_name"]
            for short, tensor in shapes.items():
                key = f"{name}__{short}"
                shape = tuple(tensor.shape)
                if short in ("moving_mean", "moving_variance"):
                    init = (
                        torch.zeros(shape)
                        if short == "moving_mean"
                        else torch.ones(shape)
                    )
                    self.register_buffer(key, init)
                    continue
                if short == "kernel":
                    fan_in = int(np.prod(shape[:-1]))
                    fan_out = shape[-1] * (shape[0] if cls == "Conv1D" else 1)
                    init = _glorot(shape, fan_in, fan_out, gen)
                elif short == "gamma":
                    init = torch.ones(shape)
                else:  # bias, beta
                    init = torch.zeros(shape)
                self.register_parameter(key, nn.Parameter(init))

    def _order(self) -> list[str]:
        done = {
            name
            for name, lay in self.layers.items()
            if lay["class_name"] == "InputLayer"
        }
        pending = [n for n in self.layers if n not in done]
        order: list[str] = []
        while pending:
            ready = [
                n
                for n in pending
                if all(d in done for d in keras_h5._inbound_names(self.layers[n]))
            ]
            if not ready:
                raise keras_h5.UnsupportedLayer(f"cannot resolve graph at {pending}")
            for n in ready:
                order.append(n)
                done.add(n)
                pending.remove(n)
        return order

    def _batchnorm(self, name, x, cfg):
        axis = cfg.get("axis", -1)
        axis = int(axis[0] if isinstance(axis, (list, tuple)) else axis) % x.ndim
        w = {short: getattr(self, f"{name}__{short}") for short in self.keys[name]}
        if not self.training:
            return keras_h5._batchnorm(x, w, cfg)
        reduce = [d for d in range(x.ndim) if d != axis]
        mean, var = x.mean(reduce), x.var(reduce, unbiased=False)
        momentum, eps = (
            float(cfg.get("momentum", 0.99)),
            float(cfg.get("epsilon", 1e-3)),
        )
        with torch.no_grad():
            w["moving_mean"].mul_(momentum).add_(mean.detach() * (1.0 - momentum))
            w["moving_variance"].mul_(momentum).add_(var.detach() * (1.0 - momentum))
        shape = [1] * x.ndim
        shape[axis] = -1
        scaled = (x - mean.reshape(shape)) / torch.sqrt(var.reshape(shape) + eps)
        return w["gamma"].reshape(shape) * scaled + w["beta"].reshape(shape)

    def forward(self, scalars, profiles):
        tensors = {name: (scalars, profiles)[slot] for name, slot in self.slots.items()}
        for name in self.order:
            lay = self.layers[name]
            cfg = lay["config"]
            ins = [tensors[d] for d in keras_h5._inbound_names(lay)]
            if lay["class_name"] == "BatchNormalization":
                tensors[name] = self._batchnorm(name, ins[0], cfg)
            elif lay["class_name"] == "Dropout":
                tensors[name] = F.dropout(ins[0], float(cfg["rate"]), self.training)
            else:
                w = {
                    short: getattr(self, f"{name}__{short}")
                    for short in self.keys.get(name, [])
                }
                tensors[name] = keras_h5._apply(lay, ins, w)
        return tensors[self.output_name][:, self.column]


class DsmDetector(nn.Module):
    """The DSM's embedding (`sizes`: bias-free linear layers, ReLU6) and one logit."""

    def __init__(self, sizes=(38, 100, 1000), *, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.trunk = nn.Sequential(
            *[
                layer
                for a, b in pairwise(sizes)
                for layer in (nn.Linear(a, b, bias=False), nn.ReLU6())
            ]
        )
        self.head = nn.Linear(sizes[-1], 1)

    def forward(self, x):
        return self.head(self.trunk(x)).squeeze(-1)


def _tensors(arrays, device):
    return [
        torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
        for a in arrays
    ]


def predict(module, inputs, *, device="cpu", batch_size: int = 8192) -> np.ndarray:
    """Probabilities for `inputs` (a tuple of arrays, rows first), batched."""
    module.eval()
    n = len(inputs[0])
    out = np.empty(n, dtype=np.float64)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            chunk = _tensors([a[i : i + batch_size] for a in inputs], device)
            out[i : i + batch_size] = (
                torch.sigmoid(module(*chunk)).double().cpu().numpy()
            )
    return out


def fit(
    module,
    train,
    val,
    *,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 512,
    max_epochs: int = 40,
    patience: int = 5,
    seed: int = 0,
    device="cpu",
) -> dict:
    """Train `module` on `train = (inputs, y)`, keep the epoch with the best `val` loss.

    `inputs` is a tuple of arrays with rows first, `y` the 0/1 targets. The loss is the
    mean binary cross-entropy on the logit. Training stops after `patience` epochs
    without improvement; the best weights (and batch-norm statistics) are restored.
    Returns `{"epochs", "best_epoch", "best_val_loss", "train_loss"}`.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    module.to(device)
    opt = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=weight_decay)
    x_tr = _tensors(train[0], device)
    y_tr = torch.as_tensor(np.asarray(train[1]), dtype=torch.float32, device=device)
    x_va = _tensors(val[0], device)
    y_va = torch.as_tensor(np.asarray(val[1]), dtype=torch.float32, device=device)

    def val_loss() -> float:
        module.eval()
        with torch.no_grad():
            total = 0.0
            for i in range(0, len(y_va), 8192):
                logit = module(*[a[i : i + 8192] for a in x_va])
                total += F.binary_cross_entropy_with_logits(
                    logit, y_va[i : i + 8192], reduction="sum"
                ).item()
        return total / max(len(y_va), 1)

    best, best_state, best_epoch, stale = np.inf, None, -1, 0
    history = []
    for epoch in range(max_epochs):
        module.train()
        order = rng.permutation(len(y_tr))
        running, seen = 0.0, 0
        for i in range(0, len(order), batch_size):
            idx = torch.as_tensor(order[i : i + batch_size], device=device)
            if len(idx) < 8:  # batch norm needs a few rows
                continue
            loss = F.binary_cross_entropy_with_logits(
                module(*[a[idx] for a in x_tr]), y_tr[idx]
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
            seen += len(idx)
        history.append(running / max(seen, 1))
        current = val_loss()
        if current < best - 1e-6:
            best, best_epoch, stale = current, epoch, 0
            best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        module.load_state_dict(best_state)
    module.eval()
    return {
        "epochs": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": float(best),
        "train_loss": [float(h) for h in history],
    }
