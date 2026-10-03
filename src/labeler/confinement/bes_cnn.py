"""The small 3D-convolutional BES classifier of Gill et al. (2024), and its training.

Dropout, one 3D convolution (10 kernels (3, 3, 5) over rows, columns and frequency, zero
padding, groups = 2, one per sub-window), batch norm, LeakyReLU, 3D max-pool (1, 2, 4),
flatten, an MLP of two 60-unit layers and 4 logits. Input ``(B, 2, rows, 8, 128)``
log-spectral features (``labeler.confinement.bes_features``). Training follows the
paper: cross-entropy, one learning rate for the convolution and another for the MLP,
60,000 steps, early stopping after 30 evaluations without a better validation loss, and
the checkpoint with the best validation macro-F1 is kept.

``Features`` reads batches from a memory-mapped ``(n, 2, 64, 128)`` float16 array, so
many training processes share one page cache.
"""

from __future__ import annotations

import copy
import queue
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from . import bes_protocol as bp

COLUMNS = 8
FREQS = 128
CLASSES = bp.CLASSES


class BesNet(nn.Module):
    """The paper's network for a ``rows`` x 8 block of BES channels."""

    def __init__(self, rows: int = 6, dropout: float = 0.2):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.conv = nn.Conv3d(2, 10, (3, 3, 5), padding=(1, 1, 2), groups=2)
        self.norm = nn.BatchNorm3d(10)
        self.act = nn.LeakyReLU()
        self.pool = nn.MaxPool3d((1, 2, 4))
        flat = 10 * rows * (COLUMNS // 2) * (FREQS // 4)
        self.mlp = nn.Sequential(
            nn.Linear(flat, 60),
            nn.LeakyReLU(),
            nn.Linear(60, 60),
            nn.LeakyReLU(),
            nn.Linear(60, len(CLASSES)),
        )

    def forward(self, x):
        x = self.pool(self.act(self.norm(self.conv(self.drop(x)))))
        return self.mlp(x.flatten(1))


@dataclass(frozen=True)
class TrainConfig:
    """Optimiser and schedule. ``adam``: coupled L2 decay; ``adamw``: decoupled."""

    optimiser: str = "adamw"
    lr_conv: float = 1e-3
    lr_mlp: float = 1e-4
    weight_decay: float = 1e-2
    dropout: float = 0.2
    batch: int = 256
    steps: int = 60000
    eval_every: int = 500
    patience: int = 30
    val_cap: int = 20000
    seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


#: What this repository assumed for the first retrain (not in the paper's excerpt).
OURS = TrainConfig()
#: The paper's: Adam, weight decay 1e-3, 1e-3 for the convolution and 1e-5 for the MLP.
PAPER = TrainConfig(optimiser="adam", lr_conv=1e-3, lr_mlp=1e-5, weight_decay=1e-3)


class Features:
    """Standardised batches of a channel block of the BES features.

    ``array`` is the ``(n, 2, 64, 128)`` float16 features (a memory map is fine); only
    the rows of channels ``rows`` are kept, in memory, for the windows in ``index``. A
    small set lives on the GPU, a large one in RAM and is gathered by a background
    thread. Callers address a kept row by its ``ids`` entry (by default its row in
    ``array``). The per-channel ``offset`` (``bes_features.standardising_offset``) is added
    on the GPU.
    """

    def __init__(
        self,
        array: np.ndarray,
        index: np.ndarray,
        rows: tuple[int, int],
        offset: np.ndarray,
        device,
        gpu_gb: float = 4.0,
        ids: np.ndarray | None = None,
    ):
        self.rows = rows[1] - rows[0]
        self.device = device
        # ``ids`` name the kept rows to the caller (default: their rows in ``array``)
        self.slot = {int(i): k for k, i in enumerate(index if ids is None else ids)}
        block = slice(rows[0] * COLUMNS, rows[1] * COLUMNS)
        out = np.empty((len(index), 2, self.rows * COLUMNS, FREQS), dtype=np.float16)
        for lo in range(0, len(index), 8192):
            out[lo : lo + 8192] = np.asarray(array[index[lo : lo + 8192]])[
                :, :, block, :
            ]
        self.on_gpu = out.nbytes <= gpu_gb * 2**30
        self.data = (
            torch.from_numpy(out).to(device) if self.on_gpu else torch.from_numpy(out)
        )
        self.offset = torch.tensor(offset, dtype=torch.float32, device=device).view(
            1, 1, -1, 1
        )

    def local(self, idx: np.ndarray) -> np.ndarray:
        """Positions in the kept block of dataset indices ``idx``."""
        return np.fromiter(
            (self.slot[int(i)] for i in idx), dtype=np.int64, count=len(idx)
        )

    def fetch(self, pos: np.ndarray):
        """Raw fp16 rows at block positions ``pos`` (on the GPU, or pinned in RAM)."""
        if self.on_gpu:
            return self.data[torch.from_numpy(pos).to(self.device)]
        rows = self.data[torch.from_numpy(pos)]
        return rows.pin_memory() if self.device.type == "cuda" else rows

    def standardise(self, raw):
        x = raw.to(self.device, non_blocking=True).float() + self.offset
        return x.view(len(x), 2, self.rows, COLUMNS, FREQS)

    def get(self, idx: np.ndarray):
        """``(len(idx), 2, rows, 8, 128)`` float32 on the device for dataset indices."""
        return self.standardise(self.fetch(self.local(idx)))


def _prefetch(fn, batches: int, depth: int = 6):
    """Yield ``fn(i)`` for ``i < batches``, computed ahead in one background thread."""
    q: queue.Queue = queue.Queue(depth)
    stop = threading.Event()

    def work():
        for i in range(batches):
            if stop.is_set():
                return
            item = fn(i)
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.5)
                    break
                except queue.Full:
                    continue
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    try:
        while (item := q.get()) is not None:
            yield item
    finally:
        stop.set()


def predict(model: nn.Module, feats: Features, idx: np.ndarray, chunk: int = 1024):
    """Class probabilities ``(len(idx), 4)`` as a float32 array."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), chunk):
            out.append(torch.softmax(model(feats.get(idx[i : i + chunk])), 1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, len(CLASSES)), np.float32)


def train(
    feats: Features,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: TrainConfig,
    device,
    log=print,
) -> tuple[nn.Module, dict]:
    """Train on ``train_idx``, keep the best validation macro-F1 checkpoint.

    Returns the model with that checkpoint loaded and a record of the run.
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    torch.backends.cudnn.benchmark = False
    model = BesNet(feats.rows, cfg.dropout).to(device)
    groups = [
        {
            "params": [*model.conv.parameters(), *model.norm.parameters()],
            "lr": cfg.lr_conv,
        },
        {"params": model.mlp.parameters(), "lr": cfg.lr_mlp},
    ]
    opt_cls = torch.optim.Adam if cfg.optimiser == "adam" else torch.optim.AdamW
    opt = opt_cls(groups, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    if len(val_idx) > cfg.val_cap:
        val_idx = np.sort(rng.choice(val_idx, cfg.val_cap, replace=False))
    val_pos = feats.local(val_idx)
    val_y = torch.from_numpy(labels[val_idx]).long().to(device)

    draws = rng.integers(len(train_idx), size=(cfg.steps, cfg.batch))
    train_pos = feats.local(train_idx)

    def batch(i):
        pos = train_pos[draws[i]]
        return feats.fetch(pos), torch.from_numpy(labels[train_idx[draws[i]]]).long()

    best_f1, best_loss, best_state, since = -1.0, float("inf"), None, 0
    history, running = [], []
    started = time.time()
    step = 0
    for step, (x, y) in enumerate(_prefetch(batch, cfg.steps), start=1):
        model.train()
        loss = loss_fn(model(feats.standardise(x)), y.to(device, non_blocking=True))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running.append(loss.detach())
        if step % cfg.eval_every:
            continue
        model.eval()
        with torch.no_grad():
            logits = torch.cat(
                [
                    model(feats.standardise(feats.fetch(val_pos[i : i + 1024])))
                    for i in range(0, len(val_pos), 1024)
                ]
            )
        val_loss = float(loss_fn(logits, val_y))
        conf = torch.bincount(
            val_y * len(CLASSES) + logits.argmax(1), minlength=len(CLASSES) ** 2
        ).view(len(CLASSES), len(CLASSES))
        f1 = bp.macro_f1(conf.cpu().numpy())
        history.append(
            {
                "step": step,
                "train_loss": float(torch.stack(running).mean()),
                "val_loss": val_loss,
                "val_macro_f1": f1,
            }
        )
        running = []
        if f1 > best_f1:
            best_f1, best_state = f1, copy.deepcopy(model.state_dict())
        since = 0 if val_loss < best_loss else since + 1
        best_loss = min(best_loss, val_loss)
        if step % (10 * cfg.eval_every) == 0:
            log(
                f"step {step} val loss {val_loss:.4f} macro-F1 {f1:.4f} "
                f"best {best_f1:.4f} ({time.time() - started:.0f} s)"
            )
        if since >= cfg.patience:
            break
    if best_state is None:  # fewer steps than one evaluation
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    best_step = (
        history[int(np.argmax([h["val_macro_f1"] for h in history]))]["step"]
        if history
        else step
    )
    record = {
        "config": cfg.as_dict(),
        "steps_run": step,
        "best_step": best_step,
        "best_val_macro_f1": best_f1,
        "train_windows": len(train_idx),
        "val_windows": len(val_idx),
        "parameters": sum(p.numel() for p in model.parameters()),
        "torch": torch.__version__,
        "seconds": round(time.time() - started, 1),
        "gpu_gb_reserved": round(torch.cuda.max_memory_reserved(device) / 2**30, 2)
        if device.type == "cuda"
        else None,
        "history": history,
    }
    return model, record
