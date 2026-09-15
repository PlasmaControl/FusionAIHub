"""SELDNet for frame-level AE activity and frequency (spec section 5.6).

The body follows ``aemodes/src/aemodes/models/detection/seldnet.py``: three
2-D convolution blocks that pool along **frequency only**, two bidirectional
GRUs whose two directions are multiplied together (that repo's variant of the
SELD-net trick), and a small feed-forward head. Time is never pooled, so the
output has one row per input frame and the network runs on a record of any
length.

Two things are ours rather than the reference's:

* **Pool sizes.** The band is 348 bins (80.57-250.00 kHz), and the reference's
  ``[9, 8, 2]`` neither divides 348 nor is meant to: it pools with ``stride=1``
  and ``ceil_mode=True``, which shrinks the axis by ``pool - 1`` instead of
  dividing it. We pool with ``stride = pool_size`` - the usual SELDNet
  arrangement - so the sizes must divide the axis exactly. ``348 = 6 x 2 x 29``
  and :data:`DEFAULT_POOL_SIZES` is ``(6, 2, 29)``, taking 348 -> 58 -> 29 -> 1.
  The GRU therefore sees ``conv_channels x 1 = 64`` features per frame. Any
  triple whose product divides the frequency axis works; the constructor
  checks and raises otherwise.
* **Two outputs per frame**, not a class vector: index 0 is the activity
  logit and index 1 is the frequency in the normalised band coordinate
  ``(f_khz - 80) / 170``, so 80 kHz -> 0.0 and 250 kHz -> 1.0.

The losses live here too. :func:`binary_sce_loss` is the binary case of
symmetric cross entropy (Wang et al. 2019): ``alpha * CE + beta * RCE`` with
the reverse term computed against the label clamped away from 0 and 1, which
is what makes it tolerant of flipped labels. Our activity label is noisy by
construction - the mask over-calls and the annotation under-calls - so the SCE
and BCE runs are the experiment that measures how much that matters.

Torch is imported at module import time; this module is only reachable from
the phase-3 venv, never from the pixi test env.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

#: lower edge of the normalised frequency coordinate, in kHz.
FREQ_LO_KHZ = 80.0
#: span of the normalised frequency coordinate, in kHz (80 -> 250 kHz).
FREQ_SPAN_KHZ = 170.0
#: 348 = 6 * 2 * 29; pools the band to a single row before the GRU.
DEFAULT_POOL_SIZES = (6, 2, 29)


def normalise_freq(f_khz: Tensor) -> Tensor:
    """kHz -> the band coordinate the model regresses."""
    return (f_khz - FREQ_LO_KHZ) / FREQ_SPAN_KHZ


def denormalise_freq(f_norm: Tensor) -> Tensor:
    """The band coordinate -> kHz."""
    return f_norm * FREQ_SPAN_KHZ + FREQ_LO_KHZ


@dataclass(frozen=True)
class AeSeldNetConfig:
    """Everything needed to rebuild the network from a checkpoint."""

    in_channels: int = 4
    n_freq: int = 348
    pool_sizes: tuple[int, ...] = DEFAULT_POOL_SIZES
    conv_channels: int = 64
    rnn_sizes: tuple[int, ...] = (128, 128)
    fnn_size: int = 128
    dropout: float = 0.0
    n_outputs: int = 2

    def as_dict(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "n_freq": self.n_freq,
            "pool_sizes": list(self.pool_sizes),
            "conv_channels": self.conv_channels,
            "rnn_sizes": list(self.rnn_sizes),
            "fnn_size": self.fnn_size,
            "dropout": self.dropout,
            "n_outputs": self.n_outputs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AeSeldNetConfig:
        return cls(
            in_channels=int(data["in_channels"]),
            n_freq=int(data["n_freq"]),
            pool_sizes=tuple(int(p) for p in data["pool_sizes"]),
            conv_channels=int(data["conv_channels"]),
            rnn_sizes=tuple(int(r) for r in data["rnn_sizes"]),
            fnn_size=int(data["fnn_size"]),
            dropout=float(data["dropout"]),
            n_outputs=int(data["n_outputs"]),
        )

    @property
    def freq_after_pooling(self) -> int:
        n = self.n_freq
        for pool in self.pool_sizes:
            n //= pool
        return n


class AeSeldNet(nn.Module):
    """``(B, C, T, F) -> (B, T, 2)``: activity logit and normalised frequency."""

    def __init__(self, config: AeSeldNetConfig | None = None) -> None:
        super().__init__()
        cfg = config or AeSeldNetConfig()
        product = 1
        for pool in cfg.pool_sizes:
            product *= pool
        if product <= 0 or cfg.n_freq % product != 0:
            raise ValueError(
                f"pool sizes {cfg.pool_sizes} (product {product}) do not divide "
                f"the {cfg.n_freq}-bin frequency axis exactly"
            )
        self.config = cfg

        blocks: list[nn.Module] = []
        in_ch = cfg.in_channels
        for pool in cfg.pool_sizes:
            blocks.append(
                nn.Conv2d(in_ch, cfg.conv_channels, kernel_size=(3, 3), stride=1, padding=1)
            )
            blocks.append(nn.BatchNorm2d(cfg.conv_channels))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.MaxPool2d(kernel_size=(1, pool), stride=(1, pool)))
            if cfg.dropout > 0:
                blocks.append(nn.Dropout(p=cfg.dropout))
            in_ch = cfg.conv_channels
        self.conv = nn.Sequential(*blocks)

        rnn_in = cfg.conv_channels * cfg.freq_after_pooling
        sizes = [rnn_in, *cfg.rnn_sizes]
        self.rnn = nn.ModuleList(
            [
                nn.GRU(
                    input_size=sizes[i],
                    hidden_size=sizes[i + 1],
                    batch_first=True,
                    bidirectional=True,
                )
                for i in range(len(sizes) - 1)
            ]
        )
        self.tanh = nn.Tanh()
        self.fnn = nn.Sequential(
            nn.Linear(cfg.rnn_sizes[-1], cfg.fnn_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(cfg.fnn_size, cfg.n_outputs),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, T, F), got shape {tuple(x.shape)}")
        x = self.conv(x)
        b, c, t, f = x.shape
        # (B, C, T, F) -> (B, T, C*F); permute first so the feature axis is
        # channel-major, which is what the reference's view achieves for its
        # own layout.
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        for rnn in self.rnn:
            x, _ = rnn(x)
            x = self.tanh(x)
            half = x.shape[-1] // 2
            x = x[:, :, half:] * x[:, :, :half]
        return self.fnn(x)


def _weighted_mean(values: Tensor, weight: Tensor) -> Tensor:
    """Mean of ``values`` over ``weight``; 0 (and no NaN) when nothing counts."""
    total = weight.sum()
    if total.item() == 0.0:
        return values.sum() * 0.0
    return (values * weight).sum() / total


def binary_bce_loss(logits: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    """Weighted binary cross entropy on logits."""
    ce = nn.functional.binary_cross_entropy_with_logits(
        logits, target.to(logits.dtype), reduction="none"
    )
    return _weighted_mean(ce, weight.to(logits.dtype))


def binary_sce_loss(
    logits: Tensor,
    target: Tensor,
    weight: Tensor,
    alpha: float = 1.0,
    beta: float = 0.5,
    label_eps: float = 1e-4,
) -> Tensor:
    """Symmetric cross entropy, binary case.

    ``alpha * CE(y, p) + beta * CE(p, y)``. The reverse term takes the log of
    the *label*, so the label is clamped to ``[label_eps, 1 - label_eps]``;
    with the default that caps the reverse penalty at ``-log(1e-4) = 9.21``,
    which is what bounds the gradient a wrong label can produce.
    """
    dtype = logits.dtype
    target = target.to(dtype)
    ce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    y = target.clamp(min=label_eps, max=1.0 - label_eps)
    rce = -(p * torch.log(y) + (1.0 - p) * torch.log1p(-y))
    return _weighted_mean(alpha * ce + beta * rce, weight.to(dtype))


def masked_huber_loss(pred: Tensor, target: Tensor, mask: Tensor, delta: float = 1.0) -> Tensor:
    """Huber on the frames ``mask`` selects; 0 when the mask is empty."""
    dtype = pred.dtype
    target = torch.nan_to_num(target.to(dtype), nan=0.0)
    per = nn.functional.huber_loss(pred, target, reduction="none", delta=delta)
    return _weighted_mean(per, mask.to(dtype))


@dataclass(frozen=True)
class AeLossConfig:
    """Which activity loss, and how hard the frequency head pulls."""

    loss: str = "sce"
    alpha: float = 1.0
    beta: float = 0.5
    lambda_f: float = 1.0
    huber_delta: float = 1.0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "loss": self.loss,
            "alpha": self.alpha,
            "beta": self.beta,
            "lambda_f": self.lambda_f,
            "huber_delta": self.huber_delta,
        }


def ae_loss(
    output: Tensor,
    activity_target: Tensor,
    activity_weight: Tensor,
    freq_target: Tensor,
    freq_weight: Tensor,
    config: AeLossConfig | None = None,
) -> dict[str, Tensor]:
    """Activity + frequency loss over one batch of ``(B, T, 2)`` outputs.

    ``activity_weight`` is the three-way target's 0/1 weight: frames the label
    calls neither confirmed-active nor confirmed-quiet contribute nothing.
    ``freq_weight`` selects the frames the frequency head is trained on
    (``annotated & active`` for the three-way target). ``freq_target`` is in
    the normalised band coordinate and may be NaN where the weight is 0.
    """
    cfg = config or AeLossConfig()
    logits = output[..., 0]
    freq = output[..., 1]
    if cfg.loss == "sce":
        activity = binary_sce_loss(
            logits, activity_target, activity_weight, alpha=cfg.alpha, beta=cfg.beta
        )
    elif cfg.loss == "bce":
        activity = binary_bce_loss(logits, activity_target, activity_weight)
    else:
        raise ValueError(f"unknown activity loss {cfg.loss!r}; expected 'sce' or 'bce'")
    freq_loss = masked_huber_loss(freq, freq_target, freq_weight, delta=cfg.huber_delta)
    return {
        "loss": activity + cfg.lambda_f * freq_loss,
        "activity": activity,
        "freq": freq_loss,
    }
