"""Train `elm-ours` in shot-grouped cross-validation on the reviewed spans.

    python -m labeler.elm.train --run NAME [--folds 5] [--epochs 30] [--device cuda]

The reviewed shots (119, all cohort train or val shots; no test shot) are dealt
into `FOLDS` folds by shot (`deal_folds`, stratified by what each shot's spans
hold). For each fold the model is trained on the other shots less
`INNER_VAL` held out for early stopping and for the decision threshold, then
predicts the fold's shots: every shot has one out-of-fold prediction, made by a
model that never saw it. Nothing here reads a cohort test shot; `check_no_test`
refuses one.

**Targets and loss.** Per 1 ms, ELMy time (present spans, crowds included) against
absent time, uncertain and unlabelled time ignored; and the start of a
non-crowd present span as a Gaussian, defined only in absent and non-crowd
present spans. Masked binary cross-entropy on both heads,
the onset head's positives weighted `ONSET_POS_WEIGHT`, its loss `ONSET_LOSS_WEIGHT`.

**Sampling and augmentation.** Random crops of `CROP_MS` of random training
shots, positioned to overlap the reviewed window. The D-alpha level is shifted by
a random amount (the shots' gains differ by decades), the contrast and density
channels are rescaled, a filterscope channel or both interferometer chords may be
dropped, and noise is added. Nothing is flipped.

**Selection.** Each epoch the model scores the inner-validation shots' 50 ms bins;
the best epoch by AUPRC is kept, and the threshold is the F1-maximising one on
those bins. The out-of-fold predictions are saved at 1 ms; the scoring
(`scripts/labeler/elm_ours_evaluate.py`) pools them.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from ..config import Paths, git_sha
from . import inputs, labels, net, onset, prepare, provenance, score

FOLDS = 5
INNER_VAL = 14
CROP_MS = 4096
SEED = 20261003
ONSET_POS_WEIGHT = 30.0
ONSET_LOSS_WEIGHT = 0.5


@dataclass(frozen=True)
class Config:
    epochs: int = 25
    iters: int = 40
    batch: int = 16
    lr: float = 2e-3
    weight_decay: float = 1e-2
    dropout: float = 0.1
    crop_ms: int = CROP_MS
    level_shift: float = 0.4  # sd of the D-alpha level offset (units of FS_SCALE)
    seed: int = SEED


@dataclass
class ShotData:
    shot: int
    x: np.ndarray  # (N_CHANNELS, n_cells) float32
    dense: labels.Dense
    n_ms: int
    spans: pd.DataFrame
    cov0: np.ndarray  # analysed intervals, ms
    cov1: np.ndarray
    bins: labels.Bins


def check_no_test(shots, cohort: pd.DataFrame) -> None:
    """Refuse a shot of the cohort's blind test split."""
    split = cohort.set_index("shot").split
    bad = [int(s) for s in shots if split.get(int(s)) == "test"]
    if bad:
        raise ValueError(f"cohort test shots may not be used: {bad}")


def load(paths: Paths, shots=None) -> dict[int, ShotData]:
    """Every reviewed shot's inputs, dense targets and scored bins."""
    table = labels.review_table(prepare.review_csv(paths))
    cohort = pd.read_csv(paths.catalog / "cohort.csv")
    wanted = sorted(int(s) for s in table.shot.unique())
    if shots is not None:
        wanted = [s for s in wanted if s in set(shots)]
    check_no_test(wanted, cohort)
    out = {}
    for shot in wanted:
        x = np.load(prepare.inputs_dir(paths) / f"{shot}.npy")
        n_ms = x.shape[1] // inputs.CELLS_PER_MS
        spans = table[table.shot == shot].reset_index(drop=True)
        c0, c1 = inputs.valid_intervals(x[inputs.VALID])
        cov0, cov1 = labels.merge_intervals(c0, c1)
        out[shot] = ShotData(
            shot,
            x,
            labels.dense(spans, n_ms),
            n_ms,
            spans,
            cov0,
            cov1,
            labels.scored_bins(spans, cov0, cov1),
        )
    return out


def deal_folds(data: dict[int, ShotData], k: int = FOLDS, seed: int = SEED):
    """Shots dealt into `k` folds, stratified by the kinds of present span they hold."""
    rng = np.random.default_rng(seed)
    strata: dict[tuple[bool, bool], list[int]] = {}
    for shot, d in data.items():
        kinds = set(d.spans.kind)
        strata.setdefault(("crowd" in kinds, "non_crowd" in kinds), []).append(shot)
    folds: list[list[int]] = [[] for _ in range(k)]
    turn = 0
    for key in sorted(strata):
        shots = sorted(strata[key])
        rng.shuffle(shots)
        for shot in shots:
            folds[turn % k].append(shot)
            turn += 1
    return [sorted(f) for f in folds]


def split_inner(train: list[int], n_val: int, seed: int):
    """`train` less `n_val` random shots, and those shots."""
    rng = np.random.default_rng(seed)
    pick = set(rng.choice(sorted(train), size=n_val, replace=False).tolist())
    return [s for s in sorted(train) if s not in pick], sorted(pick)


# ------------------------------------------------------------------ batches


def crop(
    rng: np.random.Generator, d: ShotData, crop_ms: int, cfg: Config
) -> tuple[np.ndarray, ...]:
    """One augmented crop: inputs `(C, cells)`, state, onset, onset mask `(ms,)`."""
    lo, hi = d.dense.window
    lo_ms, hi_ms = lo - inputs.GRID0_MS, hi - inputs.GRID0_MS
    first = rng.uniform(lo_ms - crop_ms / 2, max(lo_ms, hi_ms - crop_ms / 2))
    a = int(np.clip(first, 0, max(0, d.n_ms - crop_ms)))
    b = a + crop_ms
    cells = crop_ms * inputs.CELLS_PER_MS
    x = np.zeros((inputs.N_CHANNELS, cells), dtype=np.float32)
    state = np.full(crop_ms, labels.IGNORE, dtype=np.int8)
    onset = np.zeros(crop_ms, dtype=np.float32)
    mask = np.zeros(crop_ms, dtype=bool)
    m = min(b, d.n_ms) - a
    x[:, : m * inputs.CELLS_PER_MS] = d.x[
        :, a * inputs.CELLS_PER_MS : (a + m) * inputs.CELLS_PER_MS
    ]
    state[:m] = d.dense.state[a : a + m]
    onset[:m] = d.dense.onset[a : a + m]
    mask[:m] = d.dense.onset_mask[a : a + m]
    augment(rng, x, cfg)
    return x, state, onset, mask


def augment(rng: np.random.Generator, x: np.ndarray, cfg: Config) -> None:
    """In place. Levels shift, contrast and density rescale, channels drop, noise."""
    valid = x[inputs.VALID] > 0
    shift = rng.normal(0.0, cfg.level_shift) + rng.normal(0.0, 0.05, size=3)
    x[list(inputs.FS_LEVEL)] += (shift[:, None] * valid).astype(np.float32)
    x[list(inputs.FS_CONTRAST)] *= rng.uniform(0.75, 1.3)
    gain = rng.uniform(0.6, 1.5)
    x[list(inputs.DENSITY)] *= gain
    x[list(inputs.DENSITY_HP)] *= rng.uniform(0.6, 1.5)
    if rng.random() < 0.3:
        j = int(rng.integers(3))
        x[inputs.FS_LEVEL[j]] = 0.0
        x[inputs.FS_CONTRAST[j]] = 0.0
    if rng.random() < 0.15:
        for c in (*inputs.DENSITY, *inputs.DENSITY_HP):
            x[c] = 0.0
    x[:10] += rng.normal(0.0, 0.01, size=(10, x.shape[1])).astype(np.float32)
    x[:10] *= valid


def batch_of(rng, data, shots, cfg: Config, device) -> dict[str, torch.Tensor]:
    items = [
        crop(rng, data[int(s)], cfg.crop_ms, cfg) for s in rng.choice(shots, cfg.batch)
    ]
    x, state, onset, mask = (np.stack(v) for v in zip(*items))
    return {
        "x": torch.from_numpy(x).to(device),
        "state": torch.from_numpy(state.astype(np.int64)).to(device),
        "onset": torch.from_numpy(onset).to(device),
        "onset_mask": torch.from_numpy(mask).to(device),
    }


def ms_valid(x: torch.Tensor) -> torch.Tensor:
    """A 1 ms cell is usable when all ten of its input cells are valid: `(B, n_ms)`."""
    v = x[:, inputs.VALID]
    return v.reshape(v.shape[0], -1, inputs.CELLS_PER_MS).min(dim=2).values > 0


def loss_of(logits: torch.Tensor, b: dict[str, torch.Tensor]):
    """Masked cross-entropy of the two heads; returns the total and its parts."""
    usable = ms_valid(b["x"])
    ev_mask = usable & (b["state"] >= 0)
    ev = F.binary_cross_entropy_with_logits(
        logits[:, 0], (b["state"] > 0).float(), reduction="none"
    )
    ev = (ev * ev_mask).sum() / ev_mask.sum().clamp(min=1)
    on_mask = usable & b["onset_mask"]
    weight = 1.0 + ONSET_POS_WEIGHT * b["onset"]
    on = F.binary_cross_entropy_with_logits(
        logits[:, 1], b["onset"], weight=weight, reduction="none"
    )
    on = (on * on_mask).sum() / on_mask.sum().clamp(min=1)
    return ev + ONSET_LOSS_WEIGHT * on, ev.detach(), on.detach()


# ---------------------------------------------------------------- inference


@torch.no_grad()
def predict(model: net.ElmUNet, x: np.ndarray, device) -> np.ndarray:
    """`(2, n_ms)` probabilities (event, onset) of one shot's `(C, cells)` input."""
    model.eval()
    n_cells = x.shape[1]
    padded = net.pad_to(n_cells)
    buf = np.zeros((x.shape[0], padded), dtype=np.float32)
    buf[:, :n_cells] = x
    out = torch.sigmoid(model(torch.from_numpy(buf)[None].to(device)))[0]
    return out[:, : n_cells // inputs.CELLS_PER_MS].float().cpu().numpy()


def onset_threshold(model, data, shots, device, tol: float = 5.0):
    """Threshold maximising agreement with non-crowd span starts on `shots`."""
    traces, truths, defined = [], [], []
    for s in shots:
        d = data[s]
        traces.append(predict(model, d.x, device)[1])
        ind = d.spans[d.spans.kind == "non_crowd"]
        truths.append(ind.t_start.to_numpy(float))
        defined.append(d.dense.onset_mask)
    return onset.best_threshold(traces, truths, defined, inputs.GRID0_MS, tol)


def val_scores(model, data, shots, device):
    """Scored-bin truth and event score of `shots`, pooled; and per-shot parts."""
    truth, sc = [], []
    for s in shots:
        d = data[s]
        p = predict(model, d.x, device)[0]
        truth.append(d.bins.truth)
        sc.append(labels.bin_scores(p, d.bins))
    return np.concatenate(truth), np.concatenate(sc)


# ----------------------------------------------------------------- training


def train_fold(data, train, val, cfg: Config, device, log=print):
    """Train on `train`; keep the epoch with the best inner-validation AUPRC.

    Returns the best state dict, the history and the F1-maximising threshold on the
    inner-validation bins at that epoch.
    """
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    model = net.ElmUNet(dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    total = cfg.epochs * cfg.iters
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=total, pct_start=0.15
    )
    best, best_state, history = -1.0, None, []
    for epoch in range(cfg.epochs):
        model.train()
        t0, parts = time.time(), []
        for _ in range(cfg.iters):
            b = batch_of(rng, data, train, cfg, device)
            loss, ev, on = loss_of(model(b["x"]), b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            parts.append((float(loss.detach()), float(ev), float(on)))
        truth, sc = val_scores(model, data, val, device)
        ap = score.average_precision(truth, sc)
        thr, f1 = score.best_threshold(truth, sc)
        row = {
            "epoch": epoch,
            "loss": float(np.mean([p[0] for p in parts])),
            "event_loss": float(np.mean([p[1] for p in parts])),
            "onset_loss": float(np.mean([p[2] for p in parts])),
            "val_auprc": ap,
            "val_auroc": score.roc_auc(truth, sc),
            "val_f1": f1,
            "val_threshold": thr,
            "seconds": time.time() - t0,
        }
        history.append(row)
        log(
            json.dumps(
                {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}
            )
        )
        if ap > best:
            best = ap
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_row = row
    return best_state, history, best_row


def run(args: argparse.Namespace) -> int:
    paths = Paths.from_env()
    cfg = Config(
        epochs=args.epochs,
        iters=args.iters,
        batch=args.batch,
        lr=args.lr,
        seed=args.seed,
    )
    device = torch.device(args.device)
    data = load(paths)
    folds = deal_folds(data, args.folds, args.seed)
    out = paths.root / "round4" / "elm" / "cv" / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "pred").mkdir(exist_ok=True)
    record = {
        "config": asdict(cfg),
        "git": git_sha(),
        "folds": folds,
        "shots": len(data),
        "parameters": net.n_parameters(net.ElmUNet(dropout=cfg.dropout)),
        "inner_val": args.inner_val,
        "fold_records": [],
        "provenance": {
            "mode": "training_time",
            "observed_at": provenance.observed_at(),
            "code": provenance.code_record(),
            "data": provenance.data_record(paths, sorted(data)),
        },
    }
    record = provenance.continue_record(out, record)
    only = set(args.only) if args.only else None
    if only is not None and not only <= set(range(args.folds)):
        raise ValueError("--only contains a fold outside --folds")
    for k, test in enumerate(folds):
        if only is not None and k not in only:
            continue
        rest = [s for s in data if s not in set(test)]
        train, val = split_inner(rest, args.inner_val, args.seed + k)
        print(
            f"fold {k}: {len(train)} train, {len(val)} inner-val, {len(test)} test",
            flush=True,
        )
        fold_cfg = Config(**{**asdict(cfg), "seed": args.seed + 100 * k})
        state, history, best = train_fold(data, train, val, fold_cfg, device)
        fold_dir = out / f"fold{k}"
        fold_dir.mkdir(exist_ok=True)
        torch.save(state, fold_dir / "model.pt")
        model = net.ElmUNet(dropout=cfg.dropout).to(device)
        model.load_state_dict(state)
        on_thr, on_f1 = onset_threshold(model, data, val, device)
        for s in test:
            p = predict(model, data[s].x, device)
            np.savez_compressed(out / "pred" / f"{s}.npz", p=p.astype(np.float16))
        info = {
            "fold": k,
            "train": train,
            "inner_val": val,
            "test": test,
            "best": best,
            "threshold": best["val_threshold"],
            "onset_threshold": on_thr,
            "onset_val_f1": on_f1,
            "history": history,
        }
        (fold_dir / "fold.json").write_text(json.dumps(info, indent=1))
        provenance.add_fold(
            record,
            {
                k2: info[k2]
                for k2 in ("fold", "test", "threshold", "onset_threshold", "best")
            } | {"artifacts": provenance.fold_artifacts(out, k, test)},
        )
        (out / "run.json").write_text(json.dumps(record, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--folds", type=int, default=FOLDS)
    ap.add_argument("--only", type=int, nargs="+", help="train only these folds")
    ap.add_argument("--inner-val", type=int, default=INNER_VAL)
    ap.add_argument("--epochs", type=int, default=Config.epochs)
    ap.add_argument("--iters", type=int, default=Config.iters)
    ap.add_argument("--batch", type=int, default=Config.batch)
    ap.add_argument("--lr", type=float, default=Config.lr)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default="cuda")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
