"""The examples of a network recoder and its training (plan 1c).

`examples` turns the stored fronts of a signal's streams (`store.Store`) into the arrays a `nets.TokenNet`
is trained on, in example units: one row per (frame, stream) for a per-stream network, one row per frame
with all streams side by side for a concatenated one. A stream without finite values is never turned into
data: a per-stream network leaves it out, a concatenated one sees 0 there and the loss leaves the entries
out (`mask` False). `train` fits the network by AdamW on the masked squared error with the rounding and the
random prefix cut of `quantizer` in the loop, and keeps the weights of the best validation. The validation
error is the one the tokenizer is judged by: the mean, over the prefix lengths of `prefix_lengths`, of the
error of a reconstruction from that many values and zeros after them. This module imports torch.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd
import torch

from .identity import N_FEAT, StreamTable
from .nets import TokenNet, save_net
from .store import Norm, Store

LOG_CSV = "train_log.csv"
LOG_COLUMNS = ("step", "train_loss", "val_mse", "val_full")
_VAL_CHUNK = 1 << 16                    # validation examples per pass


@dataclass(frozen=True)
class TrainConfig:
    """The training settings. `d_model`, `depth` and `p_drop_ident` are the net's (the sub-command copies them
    into the net's config); `steps` optimizer steps of `batch` examples drawn with replacement; AdamW at `lr`
    and `weight_decay`, linear warm-up over `warmup` steps and constant after, gradient norm clipped at `clip`;
    a validation every `val_every` steps (and after the last one); `seed` for the net's initial weights, the
    batches and every random draw of the net."""

    d_model: int = 512
    depth: int = 4
    steps: int = 20000
    batch: int = 4096
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup: int = 500
    clip: float = 1.0
    val_every: int = 1000
    seed: int = 0
    p_drop_ident: float = 0.1


# ------------------------------------------------------------------------------------- the examples

def examples(stores: Mapping[str, Store], norms: Mapping[str, Norm], table: StreamTable, *,
             concat: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(e, mask, ident)``: the training examples of `stores` (each ``(M, S, D)`` front values, NaN for no token)
    in example units (`norms[name].apply`), signals in the order of `stores`.

    Per stream: one row per (frame, stream) whose D values are all finite, frame after frame; ``e (R, D)``
    float32, `mask` all True, `ident` int64 the stream's row in `table` (-1 if the table does not hold the
    signal). Concatenated (`ValueError` unless there is exactly one signal): one row per frame with at least
    one finite stream; ``e (R, S * D)`` float32, the streams one after the other, 0 where a stream is not
    finite and `mask` False there; `ident` all -1.
    """
    if concat and len(stores) != 1:
        raise ValueError(f"a concatenated network takes one signal, got {len(stores)}: {sorted(stores)}")
    es, masks, idents = [], [], []
    for name, st in stores.items():
        x = norms[name].apply(st.values).astype(np.float32)          # (M, S, D), NaN kept
        m, s, d = x.shape
        whole = np.isfinite(x).all(axis=-1)                          # (M, S): a stream with all D values finite
        if concat:
            frame = whole.any(axis=1)
            es.append(np.where(whole[..., None], x, 0.0)[frame].reshape(-1, s * d).astype(np.float32))
            masks.append(np.repeat(whole[frame], d, axis=1))
            idents.append(np.full(int(frame.sum()), -1, dtype=np.int64))
        else:
            es.append(x[whole])
            masks.append(np.ones((int(whole.sum()), d), dtype=bool))
            idents.append(np.broadcast_to(table.index_of(name, s), (m, s))[whole].astype(np.int64))
    if not es:
        raise ValueError("no signal")
    return np.concatenate(es), np.concatenate(masks), np.concatenate(idents)


def prefix_lengths(n_values: int) -> list[int]:
    """The prefixes a recoder is judged at: the powers of two below `n_values`, then `n_values`."""
    out, k = [], 1
    while k < n_values:
        out.append(k)
        k *= 2
    return [*out, int(n_values)]


# --------------------------------------------------------------------------------------- the loop

def _feats(table_feats: torch.Tensor, ident: torch.Tensor) -> torch.Tensor:
    """The table's feature rows of `ident` (int64 ``(B,)``), zeros where it is -1: ``(B, N_FEAT)``."""
    return table_feats[ident.clamp(min=0)] * (ident >= 0)[:, None].to(table_feats.dtype)


def _validate(net: TokenNet, val: tuple[torch.Tensor, torch.Tensor, torch.Tensor], table_feats: torch.Tensor,
              prefixes: list[int]) -> tuple[float, float]:
    """``(val_mse, val_full)`` of `net` over all validation examples, in eval mode and without gradients.

    The error at a prefix of k values is the sum of the squared error over the entries with `mask` True,
    after the values from k on are set to 0, over the number of such entries (float64 over chunks).
    `val_full` is the error at all values, and `val_mse` the mean over `prefixes`. Leaves the net in eval mode.
    """
    e, mask, ident = val
    net.eval()
    sums = np.zeros(len(prefixes))
    with torch.no_grad():
        for i in range(0, e.shape[0], _VAL_CHUNK):
            x, m, ids = e[i:i + _VAL_CHUNK], mask[i:i + _VAL_CHUNK], ident[i:i + _VAL_CHUNK]
            feats = _feats(table_feats, ids)
            v = net.encode(x, feats, ids)
            for j, k in enumerate(prefixes):
                cut = v.clone()
                cut[:, k:] = 0.0
                err = (net.decode(cut, feats, ids) - x) ** 2
                sums[j] += float((err * m).sum(dtype=torch.float64))
    errors = sums / float(mask.sum().item())
    return float(errors.mean()), float(errors[-1])


def _write_log(path: Path, rows: list[dict]) -> None:
    """The whole log as CSV, written through a temporary name and renamed: a reader sees whole rows only."""
    tmp = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows, columns=list(LOG_COLUMNS)).to_csv(tmp, index=False)
    os.replace(tmp, path)


def train(net_config: dict, cfg: TrainConfig, train_ex: tuple, val_ex: tuple, table: StreamTable, out_dir: Path,
          device: str = "cpu", log: Callable[[str], None] = print) -> dict:
    """Train a `TokenNet` built from `net_config` (the `config` of `nets.save_net`) on `train_ex`
    ``(e, mask, ident)`` and validate on `val_ex`, both as `examples` returns them; write into `out_dir`
    (which exists) ``train_log.csv`` and ``best.pt``.

    The net starts from weights drawn with `cfg.seed`. Each step draws `cfg.batch` examples with replacement
    with a `torch.Generator` seeded with `cfg.seed`, which also feeds the net's `forward` (identity drop,
    rounding, prefix cut). The loss is the mean of ``(e_hat - e) ** 2`` over the entries with `mask` True. The
    net is in training mode for every step (a validation puts it in eval mode and the loop puts it back).
    Every `cfg.val_every` steps and after the last: one row of ``train_log.csv`` (``step, train_loss,
    val_mse, val_full``, `train_loss` the mean since the row before; the file is rewritten whole each time),
    and ``best.pt`` through `nets.save_net` if `val_mse` is the lowest so far (`FloatingPointError` naming the
    step if the loss or a validation error is not finite, `ValueError` if there are no examples). Returns
    ``best_step``, ``best_val_mse``, ``best_val_full`` (at the best step), ``steps`` and ``seconds``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for key, (e, mask, ident) in (("training", train_ex), ("validation", val_ex)):
        if len(e) == 0:
            raise ValueError(f"there are no {key} examples")
        if e.shape[1] != net_config["d_in"] or mask.shape != e.shape or len(ident) != len(e):
            raise ValueError(f"the {key} examples are {e.shape}, the network takes d_in {net_config['d_in']}")
    with torch.random.fork_rng(devices=[]):                          # the weights come from the seed, nothing global changes
        torch.manual_seed(cfg.seed)
        net = TokenNet(net_config["d_in"], net_config["n_values"], net_config["n_streams"],
                       d_model=net_config["d_model"], depth=net_config["depth"],
                       p_drop_ident=net_config["p_drop_ident"])
    net = net.to(device)

    def hold(ex: tuple) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (torch.from_numpy(np.ascontiguousarray(ex[0], dtype=np.float32)).to(device),
                torch.from_numpy(np.ascontiguousarray(ex[1], dtype=bool)).to(device),
                torch.from_numpy(np.ascontiguousarray(ex[2], dtype=np.int64)).to(device))

    tr, va = hold(train_ex), hold(val_ex)
    features = np.asarray(table.features, dtype=np.float32).reshape(-1, N_FEAT)
    table_feats = torch.from_numpy(features if len(features) else np.zeros((1, N_FEAT), np.float32)).to(device)
    prefixes = prefix_lengths(net_config["n_values"])
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    rows: list[dict] = []
    best_mse, best_full, best_step = float("inf"), float("nan"), 0
    total, count = 0.0, 0
    net.train()
    for step in range(1, cfg.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = cfg.lr * min(1.0, step / cfg.warmup) if cfg.warmup > 0 else cfg.lr
        idx = torch.randint(0, tr[0].shape[0], (cfg.batch,), generator=generator, device=device)
        e, mask, ident = tr[0][idx], tr[1][idx], tr[2][idx]
        e_hat, _ = net(e, _feats(table_feats, ident), ident, generator)
        loss = (((e_hat - e) ** 2) * mask).sum() / mask.sum().clamp(min=1)
        value = float(loss.item())
        if not np.isfinite(value):
            raise FloatingPointError(f"the training loss is {value} at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.clip)
        optimizer.step()
        total, count = total + value, count + 1
        if step % cfg.val_every == 0 or step == cfg.steps:
            val_mse, val_full = _validate(net, va, table_feats, prefixes)
            net.train()                                              # validation ran in eval mode
            if not (np.isfinite(val_mse) and np.isfinite(val_full)):
                raise FloatingPointError(f"the validation error is {val_mse} (full {val_full}) at step {step}")
            rows.append({"step": step, "train_loss": total / count, "val_mse": val_mse, "val_full": val_full})
            total, count = 0.0, 0
            if val_mse < best_mse:
                best_mse, best_full, best_step = val_mse, val_full, step
                save_net(out_dir / "best.pt", net, net_config)
            _write_log(out_dir / LOG_CSV, rows)
            log(f"step {step}/{cfg.steps}: train_loss {rows[-1]['train_loss']:.5f}, val_mse {val_mse:.5f}, "
                f"val_full {val_full:.5f}, best {best_mse:.5f} at {best_step}")
    return {"best_step": best_step, "best_val_mse": best_mse, "best_val_full": best_full, "steps": cfg.steps,
            "seconds": time.monotonic() - started}


def plateau(log: pd.DataFrame, rel: float = 0.01) -> bool:
    """True if the lowest ``val_mse`` of the last quarter of the rows (``max(1, rows // 4)`` of them) is not more
    than `rel`, as a share, under the lowest of the rows before it; False with fewer than four rows."""
    n = len(log)
    if n < 4:
        return False
    v = log["val_mse"].to_numpy(dtype=np.float64)
    last = n // 4
    return bool(v[-last:].min() >= v[:-last].min() * (1.0 - rel))
