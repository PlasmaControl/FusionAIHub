"""A U-Time-style 1D U-Net that labels every millisecond of a shot L / H / QH / WPQH.

Perslev et al. (2019, U-Time): a fully convolutional encoder-decoder over a whole record
that outputs a label per time step, trained with a (generalized) Dice loss and
class-balanced sampling of windows. Here the input is the seven 0D channels of
``labeler.confinement.zerod`` plus a mask for each sparse one, at 1 kHz; the encoder
halves the time axis four times (strides 4, 4, 4, 2, so a bottleneck cell sees about
1.8 s) and the decoder restores it, with skip connections. Unlabelled bins carry -1 and
count in no loss. Farha and Gall (2019, MS-TCN): a truncated mean-squared penalty on the
change of the log-probabilities between neighbouring bins keeps the segmentation from
flickering.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import bes_protocol as bp
from . import zerod

N_CLASSES = len(zerod.CLASSES)
POOLS = (4, 4, 4, 2)
STRIDE = int(np.prod(POOLS))
WIDTHS = (16, 32, 64, 128, 128)


def _block(c_in: int, c_out: int, kernel: int = 5) -> nn.Sequential:
    pad = kernel // 2
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel, padding=pad),
        nn.BatchNorm1d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv1d(c_out, c_out, kernel, padding=pad),
        nn.BatchNorm1d(c_out),
        nn.ReLU(inplace=True),
    )


class UNet1d(nn.Module):
    """Encoder of ``len(WIDTHS) - 1`` pooled stages, decoder with skip connections."""

    def __init__(
        self, c_in: int = zerod.N_INPUT, widths=WIDTHS, pools=POOLS, dropout=0.1
    ):
        super().__init__()
        self.pools = pools
        self.enc = nn.ModuleList()
        prev = c_in
        for w in widths:
            self.enc.append(_block(prev, w))
            prev = w
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(widths) - 2, -1, -1):
            self.up.append(nn.Conv1d(widths[i + 1], widths[i], 3, padding=1))
            self.dec.append(_block(2 * widths[i], widths[i]))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Conv1d(widths[0], N_CLASSES, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits ``(B, 4, T)`` for input ``(B, C, T)``; ``T`` must be a multiple of
        ``STRIDE``."""
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x)
            if i < len(self.pools):
                skips.append(x)
                x = F.max_pool1d(x, self.pools[i])
        for j, (up, dec) in enumerate(zip(self.up, self.dec, strict=True)):
            skip = skips[-1 - j]
            x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
            x = dec(torch.cat([skip, up(x)], dim=1))
        return self.head(self.drop(x))


def generalized_dice(
    logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """1 - generalized Dice over the labelled bins (class weight 1 / volume squared)."""
    valid = target >= 0
    onehot = (
        F.one_hot(target.clamp(min=0), N_CLASSES).permute(0, 2, 1).float()
        * valid[:, None]
    )
    prob = torch.softmax(logits, 1) * valid[:, None]
    weight = 1.0 / (onehot.sum((0, 2)) ** 2 + 1.0)
    inter = (weight * (prob * onehot).sum((0, 2))).sum()
    union = (weight * (prob + onehot).sum((0, 2))).sum()
    return 1.0 - 2.0 * inter / (union + eps)


def smoothing(
    logits: torch.Tensor, target: torch.Tensor, tau: float = 4.0
) -> torch.Tensor:
    """MS-TCN's truncated T-MSE between neighbouring bins that are both labelled."""
    logp = F.log_softmax(logits, 1)
    step = (logp[..., 1:] - logp[..., :-1].detach()).pow(2).clamp(max=tau**2)
    both = ((target[..., 1:] >= 0) & (target[..., :-1] >= 0))[:, None]
    return (step * both).sum() / (both.sum() * N_CLASSES + 1.0)


def loss_fn(logits, target, *, smooth: float, class_weight: torch.Tensor | None = None):
    ce = F.cross_entropy(logits, target.long(), weight=class_weight, ignore_index=-1)
    loss = ce + generalized_dice(logits, target.long())
    if smooth > 0:
        loss = loss + smooth * smoothing(logits, target.long())
    return loss


@dataclass(frozen=True)
class UNetConfig:
    steps: int = 5000
    batch: int = 16
    window: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-2
    dropout: float = 0.1
    smooth: float = 0.15
    eval_every: int = 250
    patience: int = 16
    seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


class Corpus:
    """Padded shot tensors on the device and the class-balanced window sampler."""

    def __init__(
        self, inputs: list[np.ndarray], labels: list[np.ndarray], device, window: int
    ):
        self.n = np.array([x.shape[1] for x in inputs])
        self.length = int(max(self.n.max(), window))
        self.length += (-self.length) % STRIDE
        s = len(inputs)
        self.x = torch.zeros(
            (s, zerod.N_INPUT, self.length), dtype=torch.float32, device=device
        )
        self.y = torch.full((s, self.length), -1, dtype=torch.int8, device=device)
        for i, (x, y) in enumerate(zip(inputs, labels, strict=True)):
            self.x[i, :, : x.shape[1]] = torch.from_numpy(x)
            self.y[i, : y.shape[0]] = torch.from_numpy(y)
        self.device = device
        self.window = window

    def pools(self, shots: np.ndarray) -> list[torch.Tensor]:
        """Per class, the flat ``shot * length + bin`` index of every labelled bin of
        ``shots``."""
        out = []
        for c in range(N_CLASSES):
            hit = (self.y[torch.as_tensor(shots, device=self.device)] == c).nonzero()
            out.append(
                torch.as_tensor(shots, device=self.device)[hit[:, 0]] * self.length
                + hit[:, 1]
            )
        return out

    def batch(self, pools: list[torch.Tensor], size: int, gen: torch.Generator):
        """``size`` windows, each holding a bin of a uniformly drawn class."""
        live = [i for i, p in enumerate(pools) if len(p)]
        cls = torch.as_tensor(live, device=self.device)[
            torch.randint(len(live), (size,), device=self.device, generator=gen)
        ]
        flat = torch.empty(size, dtype=torch.long, device=self.device)
        for c in cls.unique().tolist():
            at = (cls == c).nonzero().squeeze(1)
            flat[at] = pools[c][
                torch.randint(
                    len(pools[c]), (len(at),), device=self.device, generator=gen
                )
            ]
        shot, centre = flat // self.length, flat % self.length
        offset = torch.randint(self.window, (size,), device=self.device, generator=gen)
        start = (centre - offset).clamp(0, self.length - self.window)
        idx = start[:, None] + torch.arange(self.window, device=self.device)[None, :]
        x = self.x[
            shot[:, None, None],
            torch.arange(self.x.shape[1], device=self.device)[None, :, None],
            idx[:, None, :],
        ]
        return x, self.y[shot[:, None], idx]


@torch.no_grad()
def predict_shot(model: nn.Module, x: np.ndarray, device) -> np.ndarray:
    """Class probabilities ``(4, n)`` for one shot's input ``(C, n)``."""
    n = x.shape[1]
    pad = (-n) % STRIDE
    t = torch.from_numpy(np.pad(x, ((0, 0), (0, pad)))).to(device)[None]
    model.eval()
    return torch.softmax(model(t), 1)[0, :, :n].cpu().numpy()


def evaluate(model: nn.Module, inputs, labels, device) -> tuple[float, np.ndarray]:
    """Per-bin macro-F1 over the shots' labelled bins, and their confusion matrix."""
    conf = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for x, y in zip(inputs, labels, strict=True):
        guess = predict_shot(model, x, device).argmax(0)
        keep = y >= 0
        conf += np.bincount(
            y[keep].astype(np.int64) * N_CLASSES + guess[keep], minlength=N_CLASSES**2
        ).reshape(N_CLASSES, N_CLASSES)
    return bp.macro_f1(conf), conf


def train(
    corpus: Corpus,
    train_shots: np.ndarray,
    val_inputs,
    val_labels,
    cfg: UNetConfig,
    device,
    log=print,
) -> tuple[nn.Module, dict]:
    """Train on ``train_shots`` (indices into ``corpus``); keep the best validation
    macro-F1."""
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)
    model = UNet1d(dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.steps, pct_start=0.1
    )
    pools = corpus.pools(train_shots)
    best, best_state, since, history, started = -1.0, None, 0, [], time.time()
    running = []
    step = 0
    for step in range(1, cfg.steps + 1):
        model.train()
        x, y = corpus.batch(pools, cfg.batch, gen)
        loss = loss_fn(model(x), y, smooth=cfg.smooth)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        running.append(loss.detach())
        if step % cfg.eval_every:
            continue
        f1, _ = evaluate(model, val_inputs, val_labels, device)
        history.append(
            {
                "step": step,
                "train_loss": float(torch.stack(running).mean()),
                "val_macro_f1": f1,
            }
        )
        running = []
        if f1 > best:
            best, best_state, since = f1, copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
        if step % (4 * cfg.eval_every) == 0:
            log(
                f"step {step} loss {history[-1]['train_loss']:.3f} "
                f"val macro-F1 {f1:.4f} best {best:.4f} ({time.time() - started:.0f} s)"
            )
        if since >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    record = {
        "config": cfg.as_dict(),
        "steps_run": step,
        "best_val_macro_f1": best,
        "train_shots": len(train_shots),
        "parameters": sum(p.numel() for p in model.parameters()),
        "torch": torch.__version__,
        "seconds": round(time.time() - started, 1),
        "history": history,
    }
    return model, record
