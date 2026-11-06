#!/usr/bin/env python
"""AE label tiers: gold -> silver -> bronze -> the labeled database.

- **gold**: the 50 evaluation shots, owner-reviewed. Only scored, never trained on.
- **silver**: the 450 silver shots. Labels are the owner's review where one exists
  (``review/labels.csv``); otherwise the silver extension (``extend_ae_seldnet/silver_shots.csv``).
- **bronze**: the paper population (catalog/population.csv) minus silver and gold, 4,491 shots.
  A teacher ensemble trained on silver labels them, and
  the labels are refined automatically, not by hand (``refine``).
- **database**: every other FDP plasma shot, labeled by the student trained on silver + bronze.
  This runs through ``ae_fdp_label.py infer --checkpoint``.

The refinement recipe follows the literature check of 2026-11-01:
- pseudo-labels from a teacher ensemble, kept soft (the DCASE 2024 task 4 winner's ensemble
  distillation);
- test-time views averaged;
- temperature scaling and confidence thresholds set on out-of-fold silver predictions
  (FixMatch-style filtering, with thresholds measured rather than fixed);
- frames where the members disagree are masked;
- a 10 ms median filter, as in DCASE post-processing;
- frames near a call boundary are down-weighted, where temporal label noise concentrates
  (Nagaraj et al., ICLR 2025);
- a no-beam physics gate;
- a Noisy Student: augmentation and dropout on the student only, with batches split half
  silver, half bronze.

Confident learning (cleanlab's method) is used only to rank silver shots for review
(``audit``), because on bronze the labels are the teacher's own predictions.

Subcommands, all resumable:
``lists``     (login node, labeler env) writes ``<root>/lists/{silver,gold,bronze,to_stage}.csv``.
``train``     (GPU job) trains one AeSeldNet. ``--fold k``: a teacher trained on silver with fold
              k held out; it writes out-of-fold predictions to ``<root>/oof/<name>/``.
              ``--pseudo DIR``: the student, on silver plus refined bronze.
``predict``   (GPU job) runs checkpoints x test-time views over shots with specs. It writes
              ``<root>/preds/<name>/<shot>.npz`` (column mean and std of the smoothed
              probability).
``refine``    (CPU job) turns teacher predictions on bronze into ``<root>/pseudo/<name>/<shot>.npz``
              (soft target and weight per column), plus ``summary.json``.
``audit``     (CPU job) runs confident learning on silver out-of-fold frames and writes
              ``<root>/audit/silver_suspects.csv`` (shots ranked for review) and the matching spans.
``evaluate``  (CPU job) scores each prediction set on gold, and on silver out-of-fold where present.
              10 ms frames use the silver rule.

Inputs: ``<root>/specs/<shot>.npy`` holds the model input, ``(columns, 4, 348)`` float16 from
``labeler.ae.gpu.model_input``, with ``<shot>.meta.npz`` (column times). They are written by
``ae_fdp_label.py --root <root> infer --spec-dir <root>/specs``. That command also writes
``<root>/work/<shot>.npz``, the stable model's probabilities, which ``evaluate`` scores as the
baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LABELER_ROOT = Path(os.environ.get("LABELER_ROOT", REPO / ".cache/labeler"))
ROOT = Path(os.environ.get("AE_TIERS_ROOT", LABELER_ROOT / "ae_tiers"))
EVENTS = Path(os.environ.get("LABELER_LABEL_TABLES", "/scratch/gpfs/EKOLEMEN/D3D/events"))
AE = EVENTS / "alfven_eigenmode"
SILVER_LIST = EVENTS / "silver/silver_shots.csv"
GOLD_LIST = EVENTS / "gold/gold_shots.csv"
SILVER_EXT = AE / "extend_ae_seldnet/silver_shots.csv"
REVIEW = AE / "review/labels.csv"
POPULATION = EVENTS / "catalog" / "population.csv"
CANDIDATES = LABELER_ROOT / "ae_fdp_1ms/candidates.csv"
SLURM_LOGS = LABELER_ROOT / "runs/slurm"

FRAME_MS = 10            # the silver / gold grid
THRESHOLD, MIN_SHARE = 0.5, 0.5
SMOOTH = 5               # the stable chain's centred moving average, in columns
WINDOW, CONTEXT = 1024, 128
NO_BEAM_W = 1e5          # d3drdb pbeam (W) below this, and not NaN: no neutral beams
MEDIAN_COLS = 39         # 10 ms of 0.256 ms columns
EDGE_COLS = 20           # +-5 ms around a call boundary
EDGE_WEIGHT = 0.5
MAX_STD = 0.2            # ensemble disagreement above this masks the frame
TARGET_PRECISION = 0.95  # for both the kept positives and the kept negatives
# owner's AE states (configs/labeler/events.yaml): 0 absent, 1 present, 2 uncertain, 3 not observable


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def stamp() -> str:
    return datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]


def fold_of(shot: int, folds: int) -> int:
    return int(hashlib.sha1(str(int(shot)).encode()).hexdigest(), 16) % folds


def read_shots(path: Path) -> list[int]:
    import csv

    with open(path) as fh:
        return [int(r["shot"]) for r in csv.DictReader(fh)]


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str) + "\n")
    os.replace(tmp, path)


# ------------------------------------------------------------------ labels


def intervals(tier: str) -> dict[int, list[tuple[int, float, float]]]:
    """shot -> [(state, t_start_ms, t_end_ms)]. Gold: the owner's review only. Silver: the
    review where the shot has one, else the silver extension."""
    import pandas as pd

    review = pd.read_csv(REVIEW)
    out: dict[int, list] = {}
    for shot, g in review.groupby("shot"):
        out[int(shot)] = [(int(r.category), float(r.t_start), float(r.t_end)) for r in g.itertuples()]
    want = set(read_shots(GOLD_LIST if tier == "gold" else SILVER_LIST))
    if tier == "silver":
        ext = pd.read_csv(SILVER_EXT)
        for shot, g in ext.groupby("shot"):
            if int(shot) not in out:
                out[int(shot)] = [(int(r.category), float(r.t_start), float(r.t_end)) for r in g.itertuples()]
    return {s: v for s, v in out.items() if s in want}


def column_targets(rows, t_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column target and weight: 0/1 states set the target with weight 1 (a present span
    wins over an absent one), states 2-3 (uncertain, not observable) and unlabeled time weigh 0."""
    y = np.zeros(t_ms.size, np.float32)
    w = np.zeros(t_ms.size, np.float32)
    for want in (0, 1, 2):
        for state, a, b in rows:
            if (state if state in (0, 1) else 2) != want:
                continue
            m = (t_ms >= a) & (t_ms < b)
            if want == 2:
                w[m] = 0.0
            else:
                y[m], w[m] = float(want), 1.0
    return y, w


def frame_labels(rows, n: int) -> np.ndarray:
    """10 ms frame states: 0/1, -1 where unlabeled, uncertain or not observable."""
    lab = np.full(n, -1, np.int8)
    for want in (0, 1, 2):
        for state, a, b in rows:
            if (state if state in (0, 1) else 2) != want:
                continue
            k0, k1 = max(0, math.ceil(a / FRAME_MS)), min(n, math.ceil(b / FRAME_MS))
            if k1 > k0:
                lab[k0:k1] = -1 if want == 2 else want
    return lab


def frames(t_ms: np.ndarray, prob: np.ndarray, n: int):
    """10 ms frames: (call by the silver rule, mean probability, column count)."""
    k = np.floor(t_ms / FRAME_MS).astype(np.int64)
    ok = (k >= 0) & (k < n)
    cnt = np.bincount(k[ok], minlength=n)
    hit = np.bincount(k[ok], weights=(prob[ok] >= THRESHOLD).astype(np.float64), minlength=n)
    mean = np.bincount(k[ok], weights=prob[ok], minlength=n)
    c = np.maximum(cnt, 1)
    return (hit / c >= MIN_SHARE) & (cnt > 0), mean / c, cnt


def moving_average(p: np.ndarray, k: int = SMOOTH) -> np.ndarray:
    """labeler.models.d3d_ae_activity_seldnet.spec.moving_average: centred boxcar, ends over what exists."""
    p = np.asarray(p, np.float64)
    if k <= 1 or p.size == 0:
        return p
    ker = np.ones(k)
    return np.convolve(p, ker, "same") / np.convolve(np.ones_like(p), ker, "same")


# ------------------------------------------------------------------ specs


def spec_shots() -> set[int]:
    d = ROOT / "specs"
    return {int(p.name.split(".")[0]) for p in d.glob("*.meta.npz")} if d.exists() else set()


def spec_times(shot: int) -> np.ndarray:
    with np.load(ROOT / "specs" / f"{shot}.meta.npz") as z:
        return z["t_ms"].astype(np.float64)


def spec_array(shot: int):
    return np.load(ROOT / "specs" / f"{shot}.npy", mmap_mode="r")  # (T, C, F) float16


# ------------------------------------------------------------------ metrics


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata

    pos, neg = int(y.sum()), int(y.size - y.sum())
    if not pos or not neg:
        return float("nan")
    r = rankdata(score)
    return float((r[y.astype(bool)].sum() - pos * (pos + 1) / 2) / (pos * neg))


def auprc(score: np.ndarray, y: np.ndarray) -> float:
    """Average precision (sklearn's step definition), ties broken by order."""
    pos = int(y.sum())
    if not pos:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    hits = y[order].astype(np.float64)
    prec = np.cumsum(hits) / np.arange(1, hits.size + 1)
    return float((prec * hits).sum() / pos)


def scores(call: np.ndarray, mean: np.ndarray, lab: np.ndarray) -> dict:
    m = lab >= 0
    y, c, s = lab[m].astype(np.float64), call[m], mean[m]
    tp, fp, fn = int((c & (y == 1)).sum()), int((c & (y == 0)).sum()), int((~c & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if tp else 0.0
    return {"frames": int(m.sum()), "positive_frac": float(y.mean()) if y.size else float("nan"),
            "auroc": auroc(s, y), "auprc": auprc(s, y), "precision": p, "recall": r, "f1": f1}


# ------------------------------------------------------------------ lists (login node)


def cmd_lists(args) -> None:
    import pandas as pd

    cand = pd.read_csv(CANDIDATES)
    end = dict(zip(cand.shot.astype(int), cand.end_ms.astype(int)))
    silver, gold = read_shots(SILVER_LIST), read_shots(GOLD_LIST)
    ext_end = pd.read_csv(SILVER_EXT).groupby("shot").t_end.max().astype(int).to_dict()
    (ROOT / "lists").mkdir(parents=True, exist_ok=True)

    def write(name, shots):
        rows = [(s, end.get(s, ext_end.get(s))) for s in shots]
        miss = [s for s, e in rows if e is None]
        pd.DataFrame([(s, e) for s, e in rows if e is not None], columns=["shot", "end_ms"]).to_csv(
            ROOT / "lists" / f"{name}.csv", index=False)
        print(f"{name}: {len(rows) - len(miss)} shots" + (f" ({len(miss)} with no end time: {miss[:5]})" if miss else ""))

    write("silver", silver)
    write("gold", gold)
    have = spec_shots()
    # Bronze: the paper's population (catalog/population.csv: 4,872 shots, 185601-204999, five
    # rules on current, duration, heating and data) minus silver and gold, i.e. the population
    # shots no person checked; their model labels are refined by the teachers, not by hand.
    pop = read_shots(POPULATION)
    bronze = sorted(set(pop) - set(silver) - set(gold), reverse=True)
    local = [s for s in bronze if s in have]
    write("bronze", bronze)
    todo = [s for s in [*gold, *silver, *bronze] if s not in have]
    write("to_stage", todo)
    print(f"bronze: population {len(pop)} minus silver/gold = {len(bronze)}, {len(local)} with specs already; "
          f"to stage: {len(todo)}")


# ------------------------------------------------------------------ train (GPU job)


def make_model(dropout: float):
    sys.path.insert(0, str(REPO / "src"))
    from labeler.ae.model import AeSeldNet, AeSeldNetConfig

    return AeSeldNet(AeSeldNetConfig(dropout=dropout))


def load_model(path: Path, device):
    import torch

    sys.path.insert(0, str(REPO / "src"))
    from labeler.ae.model import AeSeldNet, AeSeldNetConfig

    blob = torch.load(path, map_location="cpu", weights_only=False)
    m = AeSeldNet(AeSeldNetConfig.from_dict(blob["config"]))
    m.load_state_dict(blob["state_dict"])
    return m.to(device).eval()


def predict_shot(models, shot: int, device, views=("id",), x=None) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed column probability of every (model, view): mean and std over them. x: the
    shot's (T, C, F) input already on the device, else read from specs/.

    The windows are labeler.ae.gpu.window_plan's (as run_windows), but every view goes through
    each model together: all full-size windows of all views in one batch (split at max_batch),
    each edge window with its views stacked. Sigmoid, the 5-column boxcar and the mean / std
    stay on the GPU; one copy back per shot."""
    import torch
    import torch.nn.functional as tf

    sys.path.insert(0, str(REPO / "src"))
    from labeler.ae import gpu

    if x is None:
        x = torch.from_numpy(np.array(spec_array(shot))).to(device)
    x = x.to(device, non_blocking=True).float().permute(1, 0, 2).contiguous()  # (C, T, F)
    vs = []
    for view in views:
        v = x
        if view.startswith("f"):          # frequency shift by n bins, edge bins zeroed
            n = int(view[1:])
            v = torch.roll(x, n, dims=2)
            if n > 0:
                v[:, :, :n] = 0
            elif n < 0:
                v[:, :, n:] = 0
        elif view.startswith("c"):        # one chord dropped (a dead chord)
            v = x.clone()
            v[int(view[1:])] = 0
        vs.append(v)
    X = torch.stack(vs)  # (V, C, T, F)
    V, C, T, F = X.shape
    size = WINDOW + 2 * CONTEXT
    plan = gpu.window_plan(T, WINDOW, CONTEXT)
    whole = [q for q in plan if q[3] - q[2] == size]
    edges = [q for q in plan if q[3] - q[2] != size]
    probs = []
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                enabled=device.type == "cuda"):
        for m in models:
            out = torch.empty((V, T), device=device)
            if whole:
                bs = gpu.max_batch(m, size, F, device)
                flat = torch.stack([X[:, :, lo:hi] for _, _, lo, hi in whole], 1).reshape(-1, C, size, F)
                got = torch.cat([m(flat[i:i + bs])[..., 0].float() for i in range(0, len(flat), bs)])
                got = got.reshape(V, len(whole), size)
                for j, (start, stop, lo, _) in enumerate(whole):
                    out[:, start:stop] = got[:, j, start - lo:stop - lo]
            for start, stop, lo, hi in edges:
                out[:, start:stop] = m(X[:, :, lo:hi])[..., 0].float()[:, start - lo:stop - lo]
            probs.append(torch.sigmoid(out))
    P = torch.cat(probs).double()  # (models * views, T)
    if SMOOTH > 1:  # moving_average on the GPU: centred boxcar, ends over what exists
        ker = torch.ones(1, 1, SMOOTH, dtype=P.dtype, device=device)
        num = tf.conv1d(P[:, None], ker, padding=SMOOTH // 2)[:, 0]
        den = tf.conv1d(torch.ones(1, 1, T, dtype=P.dtype, device=device), ker, padding=SMOOTH // 2)[0, 0]
        P = num / den
    return P.mean(0).cpu().numpy(), P.std(0, unbiased=False).cpu().numpy()


def cmd_train(args) -> None:
    import torch

    sys.path.insert(0, str(REPO / "src"))
    from labeler.ae.model import AeLossConfig, ae_loss

    teacher = args.pseudo is None
    fold = int(os.environ.get("SLURM_ARRAY_TASK_ID", args.fold)) if teacher else args.holdout
    name = args.name.format(fold=fold)
    out_dir = ROOT / "models" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed + fold)
    have = spec_shots()
    labels = intervals("silver")
    silver = sorted(s for s in labels if s in have)
    train_s = [s for s in silver if fold_of(s, args.folds) != fold]
    valid_s = [s for s in silver if fold_of(s, args.folds) == fold]
    bronze = []
    if not teacher:
        pdir = Path(args.pseudo)
        bronze = sorted(int(p.stem) for p in pdir.glob("*.npz") if int(p.stem) in have)
    print(f"{name}: {'teacher' if teacher else 'student'} fold {fold}/{args.folds}: silver {len(train_s)} train, "
          f"{len(valid_s)} valid; bronze {len(bronze)} at {now()}", flush=True)
    if not train_s or not valid_s:
        raise SystemExit("empty silver split")

    targets = {s: column_targets(labels[s], spec_times(s)) for s in silver}

    device = torch.device(args.device)
    W = args.window

    def pseudo_targets(s):
        with np.load(Path(args.pseudo) / f"{s}.npz") as z:
            return z["y"].astype(np.float32), z["w"].astype(np.float32)

    shard_rng = np.random.default_rng(args.seed + 1000 + fold)

    def load_shot(s, tw):
        """One shot's columns (pinned host memory) and its column targets."""
        y, w = tw(s)
        return s, torch.from_numpy(np.load(ROOT / "specs" / f"{s}.npy")).pin_memory(), y, w

    class Pool:
        """The resident shots of one pool on the GPU end to end, each followed by W empty
        columns, so a batch of windows is one gather. A pool larger than --shard-gb keeps a
        rotating shard: at each epoch's end --swap-frac of the resident shots (the oldest) make
        way for the next shots of a shuffled cycle through the whole pool, read from disk in the
        background during the epoch. Silver shots are drawn in proportion to their labelled
        columns, bronze shots uniformly; of 5 candidate starts the first whose window is > 10 %
        labelled is kept."""

        def __init__(self, name, shots, tw, by_labels):
            self.name, self.tw, self.by_labels = name, tw, by_labels
            self.cols = {s: spec_array(s).shape[0] for s in shots}
            self.row = int(np.prod(spec_array(shots[0]).shape[1:])) * 2  # bytes a column
            self.budget = args.shard_gb * 1e9
            self.rotating = sum(self.cols.values()) * self.row > self.budget
            self.order = [int(v) for v in shard_rng.permutation(shots)] if self.rotating else list(shots)
            self.shots, self.seen, self.x, self.pending = [], set(), None, None
            self.io = ThreadPoolExecutor(max_workers=16)
            self.bg = ThreadPoolExecutor(max_workers=1)
            self.pack([], self.load(self.take(self.budget)))
            if self.rotating:
                self.prefetch()

        def take(self, nbytes):
            """The next shots of the cycle up to nbytes, never one already resident; a new
            shuffled cycle starts when the order runs out."""
            out, used = [], 0
            while True:
                if not self.order:
                    if not self.rotating:
                        break
                    resident = set(self.shots) | set(out)
                    self.order = [int(v) for v in shard_rng.permutation(list(self.cols)) if int(v) not in resident]
                    if not self.order:
                        break
                s = self.order[0]
                if out and used + self.cols[s] * self.row > nbytes:
                    break
                out.append(self.order.pop(0))
                used += self.cols[s] * self.row
            return out

        def load(self, shots):
            return list(self.io.map(lambda s: load_shot(s, self.tw), shots))

        def prefetch(self):
            drop = self.shots[:max(1, round(args.swap_frac * len(self.shots)))]
            nbytes = sum(self.cols[s] for s in drop) * self.row
            self.pending = (drop, self.bg.submit(lambda: self.load(self.take(nbytes))))

        def rotate(self):
            if not self.rotating:
                return
            drop, fut = self.pending
            keep = [s for s in self.shots if s not in set(drop)]
            self.pack(keep, fut.result())
            self.prefetch()

        def pack(self, keep, new):
            """New end-to-end buffers: the kept shots' slices of the old ones, then the new
            shots copied up from pinned memory."""
            old = dict(zip(self.shots, zip(self.start.tolist(), self.length.tolist()))) if self.x is not None else {}
            shots = keep + [s for s, *_ in new]
            n = sum(self.cols[s] + W for s in shots)
            x = torch.zeros((n, *spec_array(shots[0]).shape[1:]), dtype=torch.float16, device=device)
            y = torch.zeros(n, device=device)
            w = torch.zeros(n, device=device)
            starts, a = [], 0
            for s in keep:
                b, t = old[s]
                x[a:a + t], y[a:a + t], w[a:a + t] = self.x[b:b + t], self.y[b:b + t], self.w[b:b + t]
                starts.append(a)
                a += t + W
            for s, xs, ys, ws in new:
                t = len(xs)
                x[a:a + t] = xs.to(device, non_blocking=True)
                y[a:a + t], w[a:a + t] = torch.from_numpy(ys).to(device), torch.from_numpy(ws).to(device)
                starts.append(a)
                a += t + W
            self.x, self.y, self.w = x, y, w
            self.shots = shots
            self.seen |= set(shots)
            self.start = torch.tensor(starts, device=device)
            self.length = torch.tensor([self.cols[s] for s in shots], device=device)
            self.cw = torch.cat([torch.zeros(1, dtype=torch.float64, device=device), w.double().cumsum(0)])
            p = torch.tensor([float((self.tw(s)[1] > 0).sum()) if self.by_labels else 1.0 for s in shots])
            self.p = (p / p.sum()).to(device)

        def status(self):
            return (f"{self.name} {len(self.shots)} shots {self.x.numel() * 2 / 1e9:.1f} GB" +
                    (f" (rotating; seen {len(self.seen)} of {len(self.cols)})" if self.rotating else ""))

        def sample(self, k):
            s = torch.multinomial(self.p, k, replacement=True)
            hi = (self.length[s] - W).clamp(min=0) + 1
            g = self.start[s][:, None] + (torch.rand(k, 5, device=device) * hi[:, None]).long()
            ok = (self.cw[g + W] - self.cw[g]) / W > 0.1
            g = g.gather(1, torch.where(ok.any(1), ok.int().argmax(1), 4)[:, None])
            idx = g + torch.arange(W, device=device)
            return self.x[idx], self.y[idx], self.w[idx]  # (k, W, C, F), (k, W), (k, W)

    t_load = time.time()
    pools = [(Pool("silver", train_s, lambda s: targets[s], True),
              args.batch if not bronze else round(args.batch * args.silver_frac))]
    if bronze:
        cache = {s: pseudo_targets(s) for s in bronze}
        pools.append((Pool("bronze", bronze, cache.__getitem__, False), args.batch - pools[0][1]))
    valid_x = {s: torch.from_numpy(np.load(ROOT / "specs" / f"{s}.npy")).to(device) for s in valid_s}
    print(f"loaded in {time.time() - t_load:.0f} s: " + "; ".join(p.status() for p, _ in pools) +
          f"; batch {args.batch} = " + " + ".join(str(k) for _, k in pools), flush=True)

    def batch():
        parts = [p.sample(k) for p, k in pools if k]
        return tuple(torch.cat(t) for t in zip(*parts))

    ahead = ThreadPoolExecutor(max_workers=1)  # the next batch is gathered while this one trains

    model = make_model(args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * args.steps,
                                                pct_start=0.1)
    loss_cfg = AeLossConfig(loss="sce", alpha=1.0, beta=0.5, lambda_f=0.0)
    student = not teacher

    def augment(x, y, w):
        """x (B, C, T, F). Label-preserving noise for every model: gain and offset per chord,
        a +-4 bin frequency roll, one chord dropped (p 0.2), two time masks (<= 40 columns)
        and two frequency masks (<= 30 bins) set to 0, the standardised mean. With --mixup
        the student adds mixup: half the batch mixed with a partner, target = probabilistic OR."""
        B, C, T, F = x.shape
        x = x * (1 + 0.1 * torch.randn(B, C, 1, 1, device=x.device)) + 0.1 * torch.randn(B, C, 1, 1, device=x.device)
        shift = int(torch.randint(-4, 5, ()).item())
        if shift:
            x = torch.roll(x, shift, dims=3)
        drop = torch.rand(B, device=x.device) < 0.2
        if drop.any():
            ch = torch.randint(0, C, (B,), device=x.device)
            mask = torch.zeros(B, C, 1, 1, device=x.device, dtype=torch.bool)
            mask[torch.arange(B, device=x.device), ch] = True
            x = x.masked_fill(mask & drop[:, None, None, None], 0.0)
        tt = torch.arange(T, device=x.device)[None, :]
        ff = torch.arange(F, device=x.device)[None, :]
        for _ in range(2):
            t0 = torch.randint(0, T, (B, 1), device=x.device)
            tw = torch.randint(0, 41, (B, 1), device=x.device)
            x = x.masked_fill(((tt >= t0) & (tt < t0 + tw))[:, None, :, None], 0.0)
            f0 = torch.randint(0, F, (B, 1), device=x.device)
            fw = torch.randint(0, 31, (B, 1), device=x.device)
            x = x.masked_fill(((ff >= f0) & (ff < f0 + fw))[:, None, None, :], 0.0)
        if student and args.mixup:
            half = B // 2
            perm = torch.randperm(B, device=x.device)[:half]
            lam = (0.3 + 0.4 * torch.rand(half, 1, 1, 1, device=x.device))
            x = torch.cat([lam * x[:half] + (1 - lam) * x[perm], x[half:]])
            y = torch.cat([1 - (1 - y[:half]) * (1 - y[perm]), y[half:]])
            w = torch.cat([w[:half] * w[perm], w[half:]])
        return x, y, w

    def validate():
        model.eval()
        calls, means, labs = [], [], []
        for s in valid_s:
            t = spec_times(s)
            p, _ = predict_shot([model], s, device, x=valid_x[s])
            n = int(math.ceil(t[-1] / FRAME_MS)) + 1
            c, m, _ = frames(t, p, n)
            calls.append(c), means.append(m), labs.append(frame_labels(labels[s], n))
        model.train()
        return scores(np.concatenate(calls), np.concatenate(means), np.concatenate(labs))

    hist, best, bad = [], -1.0, 0
    ckpt = out_dir / "model.pt"
    print("LOOP_START", stamp(), flush=True)
    t_all = time.time()
    model.train()
    for epoch in range(args.epochs):
        t0, tot = time.time(), 0.0
        nxt = ahead.submit(batch)
        for step in range(args.steps):
            x, y, w = nxt.result()
            if step + 1 < args.steps:
                nxt = ahead.submit(batch)
            x = x.float().permute(0, 2, 1, 3)  # (B, C, T, F)
            x, y, w = augment(x, y, w)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(x)
            zeros = torch.zeros_like(y)
            loss = ae_loss(out.float(), y, w, zeros, zeros, loss_cfg)["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
        if epoch == 0:
            print("LOOP_WARM", stamp(), flush=True)
        v = validate()
        row = {"epoch": epoch, "loss": tot / args.steps, "s": round(time.time() - t0, 1), **v}
        hist.append(row)
        print(f"epoch {epoch:2d} loss {row['loss']:.4f} val auprc {v['auprc']:.4f} auroc {v['auroc']:.4f} "
              f"f1 {v['f1']:.4f} [{row['s']}s]", flush=True)
        if v["auprc"] > best + 1e-4:
            best, bad = v["auprc"], 0
            tmp = ckpt.with_suffix(".pt.tmp")
            torch.save({"state_dict": model.state_dict(), "config": model.config.as_dict(), "epoch": epoch,
                        "valid": v, "role": "teacher" if teacher else "student", "fold": fold,
                        "folds": args.folds, "train_silver": train_s, "valid_silver": valid_s,
                        "bronze": len(bronze), "pseudo": args.pseudo, "made_at": now()}, tmp)
            os.replace(tmp, ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break
        atomic_json(out_dir / "training.json", {"name": name, "status": "running", "history": hist})
        t_rot = time.time()
        for pool, _ in pools:
            pool.rotate()
        if any(pool.rotating for pool, _ in pools):
            print(f"  rotated in {time.time() - t_rot:.1f} s: " + "; ".join(p.status() for p, _ in pools), flush=True)
    print("LOOP_END", stamp(), flush=True)
    model = load_model(ckpt, device)
    oof = ROOT / "oof" / name
    for s in valid_s:  # out-of-fold predictions: calibration, thresholds and the audit read these
        p, _ = predict_shot([model], s, device)
        atomic_npz(oof / f"{s}.npz", prob=p.astype(np.float32))
    atomic_json(out_dir / "training.json", {
        "name": name, "status": "finished", "role": "teacher" if teacher else "student", "fold": fold,
        "folds": args.folds, "n_train_silver": len(train_s), "n_valid_silver": len(valid_s),
        "n_bronze": len(bronze), "pseudo": args.pseudo, "args": vars(args), "best_auprc": best,
        "history": hist, "wall_s": round(time.time() - t_all, 1), "job": os.environ.get("SLURM_JOB_ID"),
        "labels": {"silver": f"{REVIEW} where reviewed, else {SILVER_EXT}"}, "made_at": now()})
    print(f"best val auprc {best:.4f}; wrote {ckpt} and oof/{name}", flush=True)


# ------------------------------------------------------------------ predict (GPU job)


def cmd_predict(args) -> None:
    import torch

    torch.backends.cudnn.benchmark = False  # every shot brings new window shapes; autotuning each stalls the GPU
    device = torch.device(args.device)
    paths = sorted(Path(p) for g in args.models for p in Path(ROOT).glob(g))
    if not paths:
        raise SystemExit(f"no checkpoints match {args.models}")
    models = [load_model(p, device) for p in paths]
    have = spec_shots()
    shots = [s for s in (read_shots(Path(args.list)) if args.list else sorted(have)) if s in have]
    task, ntask = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)), int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    shots = shots[task::ntask]
    out = ROOT / "preds" / args.name
    views = tuple(args.views.split(","))
    todo = [s for s in shots if not (out / f"{s}.npz").exists()]
    print(f"predict {len(todo)} of {len(shots)} shots with {len(models)} models x {len(views)} views "
          f"({', '.join(p.parent.name for p in paths)}) at {now()}", flush=True)
    atomic_json(out / "meta.json", {"models": [str(p) for p in paths], "views": views, "notch": bool(args.notch),
                                    "made_at": now()})
    print("LOOP_START", stamp(), flush=True)
    t0 = time.time()

    if args.notch:
        sys.path.insert(0, str(REPO / "src"))
        from labeler.ae import lines

    def load(s):  # read the next shots while the GPU works on this one
        x = np.load(ROOT / "specs" / f"{s}.npy")
        if args.notch:  # fixed-frequency lines notched (labeler.ae.lines)
            x = lines.notch_shot(x)[0]
        return torch.from_numpy(x).pin_memory()

    pool = ThreadPoolExecutor(max_workers=4)
    ahead = [pool.submit(load, s) for s in todo[:4]]
    for i, s in enumerate(todo, 1):
        x = ahead.pop(0).result()
        if i + 3 < len(todo):
            ahead.append(pool.submit(load, todo[i + 3]))
        mean, std = predict_shot(models, s, device, views, x=x)
        atomic_npz(out / f"{s}.npz", mean=mean.astype(np.float16), std=std.astype(np.float16),
                   members=len(models) * len(views))
        if i == 1:
            print("LOOP_WARM", stamp(), flush=True)
        if i % 50 == 0 or i <= 3:
            print(f"{i}/{len(todo)} {s} {(time.time() - t0) / i:.2f} s a shot", flush=True)
    print("LOOP_END", stamp(), flush=True)
    print(f"{len(todo)} shots in {time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------ refine (CPU job)


def oof_frames(globs: list[str]):
    """Silver out-of-fold frames: mean probability and label, over every matching oof dir."""
    labels = intervals("silver")
    means, labs, shots = [], [], []
    for d in sorted(p for g in globs for p in (ROOT / "oof").glob(g)):
        for f in sorted(d.glob("*.npz")):
            s = int(f.stem)
            if s not in labels:
                continue
            t = spec_times(s)
            with np.load(f) as z:
                p = z["prob"].astype(np.float64)
            n = int(math.ceil(t[-1] / FRAME_MS)) + 1
            _, m, cnt = frames(t, p, n)
            lab = frame_labels(labels[s], n)
            keep = (lab >= 0) & (cnt > 0)
            means.append(m[keep]), labs.append(lab[keep]), shots.append(np.full(keep.sum(), s))
    if not means:
        raise SystemExit(f"no out-of-fold predictions under {ROOT / 'oof'} for {globs}")
    return np.concatenate(means), np.concatenate(labs), np.concatenate(shots)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def cmd_refine(args) -> None:
    import pandas as pd
    from scipy.ndimage import median_filter

    p_oof, y_oof, _ = oof_frames(args.oof)
    # temperature scaling on silver out-of-fold frames
    grid = np.linspace(0.25, 4.0, 76)
    nll = [float(-np.mean(y_oof * np.log(np.clip(1 / (1 + np.exp(-logit(p_oof) / T)), 1e-7, 1))
                          + (1 - y_oof) * np.log(np.clip(1 - 1 / (1 + np.exp(-logit(p_oof) / T)), 1e-7, 1))))
           for T in grid]
    T = float(grid[int(np.argmin(nll))])
    pc = 1 / (1 + np.exp(-logit(p_oof) / T))
    # thresholds: the lowest t_hi whose kept positives reach TARGET_PRECISION, the highest t_lo
    # whose kept negatives reach it; never inside (t_lo, t_hi) = (0.5, 0.5)
    ts = np.linspace(0.5, 0.99, 50)
    prec = [(y_oof[pc >= t].mean() if (pc >= t).any() else 1.0) for t in ts]
    t_hi = float(next((t for t, p in zip(ts, prec) if p >= TARGET_PRECISION), 0.99))
    ts_lo = np.linspace(0.5, 0.01, 50)
    npv = [((1 - y_oof[pc <= t]).mean() if (pc <= t).any() else 1.0) for t in ts_lo]
    t_lo = float(next((t for t, p in zip(ts_lo, npv) if p >= TARGET_PRECISION), 0.01))
    kept = (pc >= t_hi) | (pc <= t_lo)
    calib = {"temperature": T, "t_hi": t_hi, "t_lo": t_lo, "oof_frames": int(y_oof.size),
             "oof_positive_frac": float(y_oof.mean()), "oof_kept_frac": float(kept.mean()),
             "oof_kept_error": float(((pc >= 0.5) != (y_oof == 1))[kept].mean()) if kept.any() else None}
    print("calibration on silver out-of-fold frames:", calib, flush=True)

    cand = pd.read_csv(CANDIDATES)
    pbeam = dict(zip(cand.shot.astype(int), cand.pbeam))
    # the no-beam gate's own check: how often silver has AE in a shot the gate would zero
    sil = intervals("silver")
    gated_silver = [s for s in sil if pd.notna(pbeam.get(s)) and pbeam[s] < NO_BEAM_W]
    gated_with_ae = [s for s in gated_silver if any(st == 1 for st, _, _ in sil[s])]
    pred = ROOT / "preds" / args.pred
    out = ROOT / "pseudo" / args.name
    bronze = set(read_shots(ROOT / "lists/bronze.csv")) if (ROOT / "lists/bronze.csv").exists() else None
    rows = []
    for f in sorted(pred.glob("*.npz")):
        s = int(f.stem)
        if bronze is not None and s not in bronze:
            continue
        with np.load(f) as z:
            mean, std = z["mean"].astype(np.float64), z["std"].astype(np.float64)
        p = 1 / (1 + np.exp(-logit(mean) / T))
        p = median_filter(p, size=MEDIAN_COLS, mode="nearest")
        w = (((p >= t_hi) | (p <= t_lo)) & (std <= MAX_STD)).astype(np.float32)
        on = p >= 0.5
        edge = np.flatnonzero(np.diff(on.astype(np.int8)) != 0)
        near = np.zeros(p.size, bool)
        for e in edge:
            near[max(0, e - EDGE_COLS):e + EDGE_COLS + 1] = True
        w[near] *= EDGE_WEIGHT
        gate = pd.notna(pbeam.get(s)) and pbeam[s] < NO_BEAM_W
        y = p.astype(np.float32)
        if gate:
            y[:], w[:] = 0.0, 1.0
        atomic_npz(out / f"{s}.npz", y=y.astype(np.float16), w=w.astype(np.float16))
        rows.append({"shot": s, "columns": p.size, "kept_frac": float((w > 0).mean()),
                     "positive_frac_kept": float((y[w > 0] >= 0.5).mean()) if (w > 0).any() else float("nan"),
                     "mean_std": float(std.mean()), "masked_by_std": float((std > MAX_STD).mean()),
                     "boundary_frac": float(near.mean()), "no_beam_gate": bool(gate)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    summary = {"pred": args.pred, "oof": args.oof, "calibration": calib, "shots": len(df),
               "kept_frac_mean": float(df.kept_frac.mean()) if len(df) else None,
               "positive_frac_kept_mean": float(df.positive_frac_kept.mean()) if len(df) else None,
               "no_beam_gated": int(df.no_beam_gate.sum()) if len(df) else 0,
               "no_beam_check": {"silver_shots_gated": len(gated_silver), "of_which_silver_has_ae": gated_with_ae},
               "rules": {"median_cols": MEDIAN_COLS, "max_std": MAX_STD, "edge_cols": EDGE_COLS,
                         "edge_weight": EDGE_WEIGHT, "no_beam_w": NO_BEAM_W, "target_precision": TARGET_PRECISION},
               "made_at": now()}
    atomic_json(out / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rules"}, indent=1, default=str), flush=True)


# ------------------------------------------------------------------ audit (CPU job)


def cmd_audit(args) -> None:
    """Confident learning (Northcutt et al. 2021, cleanlab's find_label_issues, binary case) on
    silver out-of-fold 10 ms frames: per-class thresholds = mean self-confidence of the frames
    given that label; a frame is an issue when the other class reaches its threshold. Shots are
    ranked by issue frames; each issue run is a span to look at in the review page."""
    import pandas as pd

    labels = intervals("silver")
    recs, spans = [], []
    data = []
    for d in sorted(p for g in args.oof for p in (ROOT / "oof").glob(g)):
        for f in sorted(d.glob("*.npz")):
            s = int(f.stem)
            if s not in labels:
                continue
            t = spec_times(s)
            with np.load(f) as z:
                p = z["prob"].astype(np.float64)
            n = int(math.ceil(t[-1] / FRAME_MS)) + 1
            _, m, cnt = frames(t, p, n)
            data.append((s, m, frame_labels(labels[s], n), cnt))
    allp = np.concatenate([m[(l >= 0) & (c > 0)] for _, m, l, c in data])
    ally = np.concatenate([l[(l >= 0) & (c > 0)] for _, m, l, c in data])
    t1, t0 = float(allp[ally == 1].mean()), float((1 - allp[ally == 0]).mean())
    source = set(pd.read_csv(REVIEW, usecols=["shot"]).shot)
    for s, m, lab, cnt in data:
        ok = (lab >= 0) & (cnt > 0)
        miss = ok & (lab == 1) & ((1 - m) >= t0)   # labeled present, model confidently absent
        extra = ok & (lab == 0) & (m >= t1)        # labeled absent, model confidently present
        issue = miss | extra
        recs.append({"shot": s, "label_source": "review" if s in source else "extension", "frames": int(ok.sum()),
                     "issue_frames": int(issue.sum()), "present_labeled_but_absent": int(miss.sum()),
                     "absent_labeled_but_present": int(extra.sum()),
                     "issue_frac": float(issue.sum() / max(ok.sum(), 1))})
        d = np.diff(np.concatenate([[0], issue.astype(np.int8), [0]]))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if b - a >= args.min_frames:
                spans.append({"shot": s, "t_start_ms": int(a * FRAME_MS), "t_end_ms": int(b * FRAME_MS),
                              "label": int(lab[a]), "mean_prob": float(m[a:b].mean())})
    out = ROOT / "audit"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(recs).sort_values(["issue_frames", "issue_frac"], ascending=False)
    df.to_csv(out / "silver_suspects.csv", index=False)
    pd.DataFrame(spans).to_csv(out / "silver_suspect_spans.csv", index=False)
    atomic_json(out / "audit.json", {"oof": args.oof, "thresholds": {"present": t1, "absent": t0},
                                     "shots": len(df), "issue_frames": int(df.issue_frames.sum()),
                                     "frames": int(df.frames.sum()), "spans": len(spans), "made_at": now()})
    print(f"thresholds present {t1:.3f} absent {t0:.3f}; {int(df.issue_frames.sum())} issue frames of "
          f"{int(df.frames.sum())} in {int((df.issue_frames > 0).sum())} of {len(df)} shots; {len(spans)} spans")
    print(df.head(15).to_string(index=False))


# ------------------------------------------------------------------ evaluate (CPU job)


def baseline_probs(shot: int):
    f = ROOT / "work" / f"{shot}.npz"
    if not f.exists():
        return None
    with np.load(f) as z:
        t = float(z["t_first_ms"]) + np.arange(int(z["n_columns"])) * float(z["hop_ms"])
        return t, z["prob"].astype(np.float64)


GATE_PR = 0.75  # the paper's release test on gold (catalog.tex, Label sets)


def release_gate(calls, labs, n_boot: int = 2000, seed: int = 0) -> dict:
    """The paper's release test on gold: precision and recall >= GATE_PR, and a paired 95 %
    interval (shots resampled) of F1 minus the best trivial baseline's F1 above 0. The best
    trivial baseline on a positive class is 'always present' (F1 = 2 pi / (1 + pi))."""
    rng = np.random.default_rng(seed)
    per = []
    for c, y in zip(calls, labs):
        m = y >= 0
        c, y = c[m].astype(bool), y[m] == 1
        per.append((int((c & y).sum()), int((c & ~y).sum()), int((~c & y).sum()), int(y.sum()), int(m.sum())))
    per = np.array(per, dtype=np.float64)

    def f1s(rows):
        tp, fp, fn, pos, n = rows.sum(0)
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        pi = pos / n if n else 0.0
        return f1, 2 * pi / (1 + pi) if pi else 0.0

    tp, fp, fn = per[:, 0].sum(), per[:, 1].sum(), per[:, 2].sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1, triv = f1s(per)
    diffs = np.array([np.subtract(*f1s(per[rng.integers(0, len(per), len(per))])) for _ in range(n_boot)])
    lo, hi = (float(v) for v in np.percentile(diffs, [2.5, 97.5]))
    ok = bool(p >= GATE_PR and r >= GATE_PR and lo > 0)
    return {"pass": ok, "precision": p, "recall": r, "f1": f1, "f1_always_present": triv,
            "f1_minus_trivial_95ci": [lo, hi], "shots": len(per)}


def student_holdout(d: Path):
    """(fold, folds) when every model behind preds/<name> is a student held out on one silver
    fold, else None."""
    try:
        paths = json.loads((d / "meta.json").read_text())["models"]
        info = [json.loads((Path(p).parent / "training.json").read_text()) for p in paths]
    except (OSError, KeyError, ValueError):
        return None
    if info and all(i.get("role") == "student" for i in info) and len({(i["fold"], i["folds"]) for i in info}) == 1:
        return info[0]["fold"], info[0]["folds"]
    return None


def cmd_evaluate(args) -> None:
    sets = {"stable (d3d_ae_activity_seldnet)": None}
    for name in args.preds:
        sets[name] = ROOT / "preds" / name
    report = {}
    have = spec_shots()
    for tier in ("gold", "silver_heldout"):
        labels = intervals("gold" if tier == "gold" else "silver")
        report[tier] = {}
        for name, d in sets.items():
            calls, means, labs, shots = [], [], [], []
            hold = student_holdout(d) if tier == "silver_heldout" and d is not None else None
            for s in sorted(labels):
                if tier == "silver_heldout" and d is not None and (hold is None or fold_of(s, hold[1]) != hold[0]):
                    continue  # silver: a student on its held-out fold; teachers through their oof (below)
                if d is None:
                    got = baseline_probs(s)
                    if got is None:
                        continue
                    t, p = got
                else:
                    f = d / f"{s}.npz"
                    if not f.exists() or s not in have:
                        continue
                    t = spec_times(s)
                    with np.load(f) as z:
                        p = z["mean"].astype(np.float64)
                n = int(math.ceil(t[-1] / FRAME_MS)) + 1
                c, m, _ = frames(t, p, n)
                calls.append(c), means.append(m), labs.append(frame_labels(labels[s], n)), shots.append(s)
            if shots:
                report[tier][name] = {"shots": len(shots), **scores(np.concatenate(calls), np.concatenate(means),
                                                                    np.concatenate(labs))}
                if tier == "gold":
                    report[tier][name]["gate"] = release_gate(calls, labs)
        holds = {student_holdout(d) for d in sets.values() if d is not None} - {None}
        if tier == "silver_heldout" and args.oof:
            for g in args.oof:
                try:
                    p, y, sh = oof_frames([g])
                except SystemExit:
                    continue
                report[tier][f"oof:{g}"] = {"shots": int(len(set(sh.tolist()))),
                                            **scores(p >= THRESHOLD, p, y.astype(np.int8))}
                for h, k in holds:  # the same shots a student held out, for a like-for-like row
                    m = np.isin(sh, [u for u in np.unique(sh) if fold_of(int(u), k) == h])
                    if m.any():
                        report[tier][f"oof:{g} fold {h}"] = {"shots": int(len(np.unique(sh[m]))),
                                                             **scores(p[m] >= THRESHOLD, p[m], y[m].astype(np.int8))}
    out = ROOT / "eval" / f"{args.tag}.json"
    atomic_json(out, {"report": report, "made_at": now(), "note": (
        "10 ms frames; call = at least half the frame's columns >= 0.5 (stable chain smoothing); "
        "auroc/auprc on the frame mean probability; gold = owner-reviewed, states 2-3 excluded; "
        "silver held-out = teachers' out-of-fold predictions against silver labels, which are mostly "
        "the stable model's own extension (not ground truth)")})
    for tier, rows in report.items():
        print(f"== {tier}")
        for name, r in rows.items():
            print(f"  {name:45s} shots {r['shots']:4d} frames {r['frames']:6d} pos {r['positive_frac']:.3f} "
                  f"auroc {r['auroc']:.3f} auprc {r['auprc']:.3f} f1 {r['f1']:.3f} "
                  f"(p {r['precision']:.3f} r {r['recall']:.3f})" + (
                      f" gate {'PASS' if r['gate']['pass'] else 'fail'} (F1 - always-present 95% CI "
                      f"{r['gate']['f1_minus_trivial_95ci'][0]:+.3f}..{r['gate']['f1_minus_trivial_95ci'][1]:+.3f})"
                      if "gate" in r else ""))
    print(f"wrote {out}")


# ------------------------------------------------------------------ main


def main(argv=None) -> None:
    global ROOT
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=ROOT)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("lists")
    t = sub.add_parser("train")
    t.add_argument("--name", default="teacher_f{fold}")
    t.add_argument("--fold", type=int, default=0, help="teacher: the held-out silver fold (SLURM_ARRAY_TASK_ID wins)")
    t.add_argument("--holdout", type=int, default=0, help="student: the silver fold held out for selection")
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--pseudo", default=None, help="student: a refined bronze dir (pseudo/<name>)")
    t.add_argument("--silver-frac", type=float, default=0.5)
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--steps", type=int, default=100)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--window", type=int, default=WINDOW)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--dropout", type=float, default=0.0)
    t.add_argument("--mixup", action="store_true", help="student mixup with a probabilistic-OR target (off: "
                   "the 2026-11-01 pilot student over-called with it, silver F1 .77 vs .87 without)")
    t.add_argument("--shard-gb", type=float, default=30.0,
                   help="a pool larger than this keeps a rotating shard of this size on the GPU")
    t.add_argument("--swap-frac", type=float, default=0.5, help="share of a rotating shard replaced each epoch")
    t.add_argument("--patience", type=int, default=10)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda")
    r = sub.add_parser("predict")
    r.add_argument("--models", nargs="+", required=True, help="checkpoint globs under <root>, e.g. models/teacher_f*/model.pt")
    r.add_argument("--name", required=True)
    r.add_argument("--list", default=None)
    r.add_argument("--views", default="id,f-3,f3,c0,c1,c2,c3")
    r.add_argument("--device", default="cuda")
    r.add_argument("--notch", action="store_true", help="notch fixed-frequency lines first (labeler.ae.lines)")
    f = sub.add_parser("refine")
    f.add_argument("--pred", required=True, help="preds/<name> of the teacher ensemble")
    f.add_argument("--oof", nargs="+", required=True, help="oof dir globs, e.g. teacher_f*")
    f.add_argument("--name", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--oof", nargs="+", required=True)
    a.add_argument("--min-frames", type=int, default=3)
    e = sub.add_parser("evaluate")
    e.add_argument("--preds", nargs="*", default=[])
    e.add_argument("--oof", nargs="*", default=[])
    e.add_argument("--tag", default="eval")
    args = p.parse_args(argv)
    ROOT = Path(args.root)
    {"lists": cmd_lists, "train": cmd_train, "predict": cmd_predict, "refine": cmd_refine,
     "audit": cmd_audit, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
