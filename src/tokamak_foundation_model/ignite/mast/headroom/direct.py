"""A plain transformer trained per task to regress the targets from the raw windows (headroom column C).

This is not the simulator and never becomes a headline number. It shows what a specialized model
reaches when nothing is quantized or truncated: no codec, no TokaMind coefficient truncation, no
frame means. Its tokens are deliberately the simulator's tokens (one channel over one 5 ms frame), so
a gap between this model and the later simulator points at the token path and not at how the data are
cut. A weak model is not a headroom finding: if its score is worse than our TokaMind replication on the
same task, the table marks the cell "not competitive".

Tokens. Each signal is cut along time into patches of `p = patch_len(dt, patch_s)` samples (the rule of
`cheap.frame_len` for the 5 ms default; a last partial patch is padded with NaN). One token is one
channel over one patch: a signal of `values_shape = (C, *rest)` gives `C` channels, and the `rest` dims
fold into the token's values (a spectrogram signal, `(3, 256)`, gives one token per probe and patch with
all 256 bins in it). The embedding of a token is the sum of: a per-signal `Linear` over the concatenation
of the values (NaN and inf replaced by 0) and their 0/1 validity; a learned vector per (signal name,
channel), shared by a signal's input, actuator and query tokens so that "summary-ip, channel 0" is one
identity in and out; a learned vector per role (input, actuator, query); and a `Linear` of a sinusoidal
encoding of the patch's centre time in seconds. Time enters in seconds, never as an index.

Time base. Time zero is the cut, the end of the input window. Input samples end at 0; the actuator window
ends at `delta + output_length` (its history part ends at 0, its future part is the part the task is
conditioned on); output samples start at `delta`. A token's time is the centre of the samples it covers.
These alignments are as the task spec states them and are not checked against the data's time stamps.

Queries. One query token per output channel and output patch, with the same identity and time embeddings
and no values. All tokens go through one stack of pre-norm blocks (`torch.nn.TransformerEncoderLayer`,
`norm_first=True`, GELU, feed-forward width `4 * d_model`, no dropout) and a final LayerNorm; a per-signal
`Linear` maps each query to its patch of output samples, patches are concatenated and cut to `T_out`.

Loss and selection. `masked_mse` in standardized units (NaN targets ignored, signals weighted equally),
AdamW (betas 0.9 and 0.95, decay on the weight matrices only) with linear warm-up then cosine to zero,
bfloat16 autocast on the GPU for the forward pass (the loss is computed in float32), gradient clipping at
1.0, an exponential moving average (EMA) of the weights whose decay is `min(ema, (1 + n) / (10 + n))` at
update n so that it does not lag behind during the first thousand steps. Every `val_every` steps (and at
the last step) the EMA weights are scored through `harness.score` on `val_shots` shots of `val_select`
spread evenly over its sorted list, first and last included (`validation_shots`; not the first ones: the
list is sorted by shot number, so those are the oldest campaign, which is not representative; measured
2026-11-22 on task_4-5, only 65 of the first 200 had windows and the training-mean baseline scored 0.30
on them against 0.685 on all shots); the lowest becomes `best.pt`, `last.pt` is the last. The trainer reports and logs a number
of distinct shots per training batch because a loader batch holds the windows of one worker's shots.

Data. `harness.train_windows(task, stride_s=0.005, shuffle=True, history_s=cfg.history_s)` restarted
when exhausted, behind a shuffle pool (`_Batches`): the loader delivers small batches, each from one
worker, which fill a pool of `POOL_FACTOR * batch_size` windows; each training batch is `batch_size`
windows drawn from the pool without replacement and replaced by new loader windows. Batches are
therefore built from the shots of many workers. A thread does this, so the loader and the GPU overlap.

Files of a run (`out_dir`): `config.json` (task, full `DirectConfig`, git revision), `train_log.csv`,
`best.pt`, `last.pt` (each holds the EMA weights and the configuration they were trained with), `val/`
(the scorer's files of each validation). Scoring a checkpoint rebuilds the model from the `config.json`
beside it, never from the command line, and refuses a checkpoint whose own configuration differs.

Reproducibility. `cfg.seed` seeds torch and the pool's sampling. Two runs are not bit-identical: the
loader's worker order is not deterministic.

`PREDICTOR_SHA256` is the sha256 of this file as loaded, read once at import: `run_headroom.py` stores it
in `score.json` so that a score from other code is not kept. It does not import `tokamark`.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tokamak_foundation_model.ignite.mast import baselines, harness, paths, tasks
from tokamak_foundation_model.ignite.mast.harness import PredictFn, WindowBatch
from tokamak_foundation_model.ignite.mast.headroom.cheap import FRAME_S, frame_len
from tokamak_foundation_model.ignite.mast.tasks import SignalSpec, TaskSpec

PREDICTOR_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

TRAIN_STRIDE_S = 0.005   # training windows start every 5 ms (the evaluation windows are TokaMark's)
POOL_FACTOR = 4          # the shuffle pool holds this many training batches of windows
LOADER_BATCH = 16        # windows per loader batch (each from one worker); small so that many workers mix
LOG_EVERY = 100          # steps between progress lines
MAX_SKIPPED_STEPS = 20   # non-finite gradients tolerated (the step is skipped) before the run aborts
TIME_PERIODS_S = (0.01, 2.0)   # shortest and longest period of the sinusoidal time encoding
TIME_FREQS = 24


@dataclass
class DirectConfig:
    d_model: int = 384
    depth: int = 8
    heads: int = 8
    patch_s: float = 0.005
    history_s: float | None = None        # None -> task.input_length
    batch_size: int = 256
    max_steps: int = 30_000
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup: int = 1_000
    ema: float = 0.999
    val_every: int = 2_500
    val_shots: int = 200
    seed: int = 0


def default_config(task: TaskSpec) -> DirectConfig:
    """The task's default: batches of 64 windows for the non-markovian (group 4) tasks, which have about
    6,000 tokens per window, otherwise 256."""
    return DirectConfig(batch_size=256 if task.markovian else 64)


# ---------------------------------------------------------------------------------------- tokens

def patch_len(dt: float, patch_s: float) -> int:
    """Samples per patch. For the 5 ms default this is `cheap.frame_len`, the rule of the cheap predictors."""
    return frame_len(dt) if patch_s == FRAME_S else max(1, round(patch_s / dt))


@dataclass(frozen=True)
class Layout:
    """How one signal is cut into tokens."""
    name: str
    values_shape: tuple[int, ...]
    channels: int                  # values_shape[0]
    inner: int                     # product of the other values dims (256 for a spectrogram's bins)
    n: int                         # samples in the window
    p: int                         # samples per patch
    centres: tuple[float, ...]     # seconds relative to the cut, per patch

    @property
    def n_patches(self) -> int:
        return len(self.centres)


def _layout(s: SignalSpec, n: int, p: int, t_start: float) -> Layout:
    patches = -(-n // p)
    centres = tuple(t_start + (k * p + min(n, (k + 1) * p)) / 2 * s.dt for k in range(patches))
    return Layout(s.name, tuple(s.values_shape), s.values_shape[0], int(np.prod(s.values_shape[1:], dtype=np.int64)), n, p, centres)


def layouts(task: TaskSpec, cfg: DirectConfig) -> tuple[list[Layout], list[Layout], list[Layout]]:
    """(input, actuator, output) layouts of the task at `cfg.history_s`."""
    inputs, actuators, outputs = [], [], []
    act_end = task.delta + task.output_length
    for s in task.inputs:
        n = task.n_in(s, cfg.history_s)
        inputs.append(_layout(s, n, patch_len(s.dt, cfg.patch_s), -n * s.dt))
    for s in task.actuators:
        n = task.n_act(s, cfg.history_s)
        actuators.append(_layout(s, n, patch_len(s.dt, cfg.patch_s), act_end - n * s.dt))
    for s in task.outputs:
        outputs.append(_layout(s, task.n_out(s), patch_len(s.dt, cfg.patch_s), task.delta))
    return inputs, actuators, outputs


def token_counts(task: TaskSpec, cfg: DirectConfig) -> dict[str, int]:
    """Tokens per window by role, from the task spec alone."""
    inputs, actuators, outputs = layouts(task, cfg)
    count = lambda ls: sum(l.channels * l.n_patches for l in ls)  # noqa: E731
    return {"input": count(inputs), "actuator": count(actuators), "query": count(outputs)}


def _time_features(centres: tuple[float, ...]) -> Tensor:
    """(patches, 2 * TIME_FREQS): sin and cos of the centre time in seconds at log-spaced periods."""
    periods = torch.logspace(math.log10(TIME_PERIODS_S[0]), math.log10(TIME_PERIODS_S[1]), TIME_FREQS,
                             dtype=torch.float64)
    angle = torch.tensor(centres, dtype=torch.float64)[:, None] * (2 * math.pi / periods)[None]
    return torch.cat([angle.sin(), angle.cos()], dim=-1).float()


_ROLES = {"input": 0, "actuator": 1, "query": 2}


class DirectModel(nn.Module):
    """Tokens of every input and actuator patch plus one query per output patch, one transformer."""

    def __init__(self, task: TaskSpec, cfg: DirectConfig):
        super().__init__()
        if cfg.d_model % cfg.heads:
            raise ValueError(f"d_model {cfg.d_model} is not a multiple of heads {cfg.heads}")
        self.task, self.cfg = task, cfg
        d = cfg.d_model
        self.in_layout, self.act_layout, self.out_layout = layouts(task, cfg)

        self.encode = nn.ModuleDict()
        self.head = nn.ModuleDict()
        self.identity = nn.ParameterDict()
        for prefix, group in (("i", self.in_layout), ("a", self.act_layout)):
            for l in group:
                self.encode[f"{prefix}_{l.name}"] = nn.Linear(2 * l.inner * l.p, d)
        for l in self.out_layout:
            self.head[l.name] = nn.Linear(d, l.inner * l.p)
        for l in (*self.in_layout, *self.act_layout, *self.out_layout):
            if l.name in self.identity:
                if self.identity[l.name].shape[0] != l.channels:
                    raise ValueError(f"signal {l.name} has different channel counts in different roles")
                continue
            self.identity[l.name] = nn.Parameter(0.02 * torch.randn(l.channels, d))
        self.role = nn.Parameter(0.02 * torch.randn(len(_ROLES), d))
        self.time_proj = nn.Linear(2 * TIME_FREQS, d)
        for prefix, group in (("i", self.in_layout), ("a", self.act_layout), ("o", self.out_layout)):
            for l in group:
                self.register_buffer(f"time_{prefix}_{l.name}", _time_features(l.centres), persistent=False)

        layer = nn.TransformerEncoderLayer(d, cfg.heads, 4 * d, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, cfg.depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def _tag(self, h: Tensor, l: Layout, prefix: str, role: str) -> Tensor:
        """Add identity, role and time to token features (B, C, P, d); returns (B, C * P, d)."""
        time = self.time_proj(getattr(self, f"time_{prefix}_{l.name}"))                    # (P, d)
        h = h + self.identity[l.name][None, :, None, :] + self.role[_ROLES[role]] + time[None, None]
        return h.reshape(h.shape[0], l.channels * l.n_patches, -1)

    def _values(self, l: Layout, prefix: str, role: str, x: Tensor) -> Tensor:
        if tuple(x.shape[1:]) != (*l.values_shape, l.n):
            raise ValueError(f"{l.name}: expected (B, {l.values_shape}, {l.n}), got {tuple(x.shape)}")
        B = x.shape[0]
        x = x.reshape(B, l.channels, l.inner, l.n)
        valid = torch.isfinite(x)
        v = torch.where(valid, x, torch.zeros((), dtype=x.dtype, device=x.device))
        pad = l.n_patches * l.p - l.n
        if pad:
            v, valid = F.pad(v, (0, pad)), F.pad(valid, (0, pad))
        shape = (B, l.channels, l.inner, l.n_patches, l.p)
        v = v.reshape(shape).permute(0, 1, 3, 2, 4).reshape(B, l.channels, l.n_patches, l.inner * l.p)
        valid = valid.reshape(shape).permute(0, 1, 3, 2, 4).reshape(B, l.channels, l.n_patches, l.inner * l.p)
        h = self.encode[f"{prefix}_{l.name}"](torch.cat([v, valid.to(v.dtype)], dim=-1))
        return self._tag(h, l, prefix, role)

    def forward(self, inputs: dict[str, Tensor], actuators: dict[str, Tensor]) -> dict[str, Tensor]:
        """Standardized tensors in (NaN allowed), standardized `(B, *values_shape, T_out)` out, always finite."""
        B = next(iter(inputs.values())).shape[0] if inputs else next(iter(actuators.values())).shape[0]
        tokens = [self._values(l, "i", "input", inputs[l.name]) for l in self.in_layout]
        tokens += [self._values(l, "a", "actuator", actuators[l.name]) for l in self.act_layout]
        queries = []
        for l in self.out_layout:
            zero = self.identity[l.name].new_zeros(1, l.channels, l.n_patches, self.cfg.d_model)
            queries.append(self._tag(zero, l, "o", "query").expand(B, -1, -1))
        n_context = sum(t.shape[1] for t in tokens)
        h = torch.cat([*tokens, *queries], dim=1)
        h = self.norm(self.blocks(h))[:, n_context:]
        out, start = {}, 0
        for l in self.out_layout:
            q = h[:, start:start + l.channels * l.n_patches]
            start += l.channels * l.n_patches
            y = self.head[l.name](q).float().reshape(B, l.channels, l.n_patches, l.inner, l.p)
            y = y.permute(0, 1, 3, 2, 4).reshape(B, l.channels, l.inner, l.n_patches * l.p)[..., :l.n]
            out[l.name] = y.reshape(B, *l.values_shape, l.n)
        return out


# ---------------------------------------------------------------------------------------- data to tensors

def _inputs_to_tensors(batch: WindowBatch, task: TaskSpec, device) -> tuple[dict, dict]:
    def convert(arrays, specs):
        return {s.name: s.standardize(torch.from_numpy(np.ascontiguousarray(arrays[s.name])).to(device))
                for s in specs}

    return convert(batch.inputs, task.inputs), convert(batch.actuators, task.actuators)


def to_tensors(batch: WindowBatch, task: TaskSpec, device) -> tuple[dict, dict, dict]:
    """(inputs, actuators, targets) as TokaMark-standardized float32 tensors on `device` (NaN stays NaN)."""
    inputs, actuators = _inputs_to_tensors(batch, task, device)
    targets = {s.name: s.standardize(torch.from_numpy(np.ascontiguousarray(batch.targets[s.name])).to(device))
               for s in task.outputs}
    return inputs, actuators, targets


def masked_mse(pred: dict[str, Tensor], target: dict[str, Tensor]) -> Tensor:
    """Mean over signals of each signal's mean squared error over its finite targets (float32).

    NaN (and inf) targets are ignored; a signal with no finite target in the batch does not count.
    With nothing finite at all the loss is a zero that still has a gradient path.
    """
    total, count = None, 0
    for name, p in pred.items():
        p = p.float()
        t = target[name]
        finite = torch.isfinite(t)
        n = finite.sum()
        if n == 0:
            term = (p * 0.0).sum()
        else:
            diff = torch.where(finite, p - torch.where(finite, t, torch.zeros_like(t)), torch.zeros_like(p))
            term = (diff ** 2).sum() / n
            count += 1
        total = term if total is None else total + term
    return total / max(count, 1)


# ---------------------------------------------------------------------------------------- checkpoints

def write_config(out_dir: Path, task: TaskSpec, cfg: DirectConfig) -> Path:
    path = Path(out_dir) / "config.json"
    record = {"task": task.name, "cfg": asdict(cfg), "git_revision": baselines._GIT_REVISION}
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def save_checkpoint(path: Path, state_source: nn.Module, cfg: DirectConfig, task: TaskSpec, *,
                    step: int, val_nrmse: float | None) -> None:
    """Atomically write the weights of `state_source` with the configuration they belong to."""
    state = {k: v.detach().cpu() for k, v in state_source.state_dict().items()}
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"state": state, "cfg": asdict(cfg), "task": task.name, "step": int(step),
                "val_nrmse": val_nrmse}, tmp)
    os.replace(tmp, path)


def _load_model(task: TaskSpec, checkpoint: Path, device) -> tuple[DirectModel, DirectConfig]:
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint.parent / "config.json").read_text())
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if config.get("task") != task.name or saved.get("task") != task.name:
        raise ValueError(f"{checkpoint} is not a checkpoint of {task.name}")
    if config["cfg"] != saved["cfg"]:
        raise ValueError(f"the config.json beside {checkpoint} differs from the configuration the "
                         "checkpoint was trained with")
    cfg = DirectConfig(**config["cfg"])
    model = DirectModel(task, cfg)
    model.load_state_dict(saved["state"])
    return model.to(device).eval(), cfg


def _autocast(device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda"))


def _predict_fn(model: DirectModel, task: TaskSpec, cfg: DirectConfig, device) -> PredictFn:
    def predict(batch: WindowBatch) -> dict[str, np.ndarray]:
        x, a = _inputs_to_tensors(batch, task, device)
        model.eval()
        with torch.no_grad(), _autocast(device):
            out = model(x, a)
        return {s.name: s.unstandardize(out[s.name].float()).cpu().numpy().astype(np.float32)
                for s in task.outputs}

    predict.history_s = cfg.history_s   # the history the scorer must cut the windows with
    return predict


def direct_predictor(task: TaskSpec, checkpoint: Path, device) -> PredictFn:
    """Physical units out. The model is rebuilt from the `config.json` beside `checkpoint`."""
    model, cfg = _load_model(task, checkpoint, device)
    return _predict_fn(model, task, cfg, device)


# ---------------------------------------------------------------------------------------- training data

def _rows(b: WindowBatch, sl: slice) -> WindowBatch:
    return WindowBatch(b.shot_ids[sl], b.window_indices[sl],
                       {k: v[sl] for k, v in b.inputs.items()}, {k: v[sl] for k, v in b.actuators.items()},
                       {k: v[sl] for k, v in b.targets.items()}, {})


class _Pool:
    """Fixed-size store of windows (arrays allocated at the first write)."""

    def __init__(self, size: int):
        self.size, self.arrays = size, None

    def write(self, slots: np.ndarray, b: WindowBatch) -> None:
        if self.arrays is None:
            def alloc(a):
                return np.full((self.size, *a.shape[1:]), np.nan, dtype=np.float32)
            self.arrays = {
                "inputs": {k: alloc(v) for k, v in b.inputs.items()},
                "actuators": {k: alloc(v) for k, v in b.actuators.items()},
                "targets": {k: alloc(v) for k, v in b.targets.items()},
                "shot_ids": np.zeros(self.size, dtype=np.int64),
                "window_indices": np.zeros(self.size, dtype=np.int64),
            }
        for role in ("inputs", "actuators", "targets"):
            for k, v in getattr(b, role).items():
                self.arrays[role][k][slots] = v
        self.arrays["shot_ids"][slots] = b.shot_ids
        self.arrays["window_indices"][slots] = b.window_indices

    def take(self, idx: np.ndarray) -> WindowBatch:
        a = self.arrays
        return WindowBatch(a["shot_ids"][idx], a["window_indices"][idx],
                           {k: v[idx] for k, v in a["inputs"].items()},
                           {k: v[idx] for k, v in a["actuators"].items()},
                           {k: v[idx] for k, v in a["targets"].items()}, {})


def _endless(task: TaskSpec, cfg: DirectConfig, num_workers: int, loader_batch: int):
    """Loader batches of the training split, restarted whenever the shots are exhausted."""
    while True:
        n = 0
        for b in harness.train_windows(task, stride_s=TRAIN_STRIDE_S, shuffle=True, history_s=cfg.history_s,
                                       batch_size=loader_batch, num_workers=num_workers):
            n += 1
            yield b
        if n == 0:
            raise RuntimeError("the training loader delivered no windows")


class _Batches:
    """Training batches drawn from a shuffle pool of loader windows, built on a thread.

    `get()` returns a `WindowBatch` (inputs, actuators, targets and shot ids only) or raises what the
    thread raised.
    """

    def __init__(self, task: TaskSpec, cfg: DirectConfig, num_workers: int, loader_batch: int, depth: int = 3):
        self._cfg, self._stop = cfg, threading.Event()
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._stream = _endless(task, cfg, num_workers, loader_batch)
        self._thread = threading.Thread(target=self._run, name="direct-batches", daemon=True)
        self._thread.start()

    def _put(self, item) -> None:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.5)
                return
            except queue.Full:
                pass

    def _run(self) -> None:
        cfg = self._cfg
        try:
            size = POOL_FACTOR * cfg.batch_size
            pool, rng = _Pool(size), np.random.default_rng(cfg.seed)
            free, spare = np.arange(size), None
            while not self._stop.is_set():
                while len(free) and not self._stop.is_set():
                    b = spare if spare is not None else next(self._stream)
                    k = min(len(b.shot_ids), len(free))
                    pool.write(free[:k], _rows(b, slice(0, k)))
                    spare = _rows(b, slice(k, None)) if k < len(b.shot_ids) else None
                    free = free[k:]
                if self._stop.is_set():
                    break
                idx = rng.choice(size, cfg.batch_size, replace=False)
                self._put(pool.take(idx))
                free = idx
        except BaseException as e:  # reported to the consumer, which is blocked on the queue
            self._put(e)
        finally:
            self._stream.close()

    def get(self) -> WindowBatch:
        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                if not self._thread.is_alive():
                    raise RuntimeError("the training batch thread ended without delivering a batch")
                continue
            if isinstance(item, BaseException):
                raise item
            return item

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=120)


# ---------------------------------------------------------------------------------------- training

LOG_COLUMNS = ["step", "train_loss", "val_nrmse", "seconds", "windows_per_s", "shots_per_batch"]


def _lr_factor(step: int, cfg: DirectConfig) -> float:
    """Linear warm-up over `warmup` steps, then cosine to zero at `max_steps`; `step` counts from 0."""
    if step < cfg.warmup:
        return (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.max_steps - cfg.warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay = [p for n, p in model.named_parameters() if p.ndim == 2 and n.endswith("weight")]
    ids = {id(p) for p in decay}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": rest, "weight_decay": 0.0}]


@torch.no_grad()
def _update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    e = list(ema.parameters())
    torch._foreach_lerp_(e, [p.detach() for p in model.parameters()], 1.0 - decay)


def validation_shots(val_shots: int) -> list[int]:
    """`val_shots` shots evenly spaced over the sorted `val_select` list, first and last included.

    Positions `linspace(0, n - 1, min(val_shots, n)).round()`: ascending, unique, deterministic; with
    `val_shots >= n` every shot once.
    """
    selection = tasks.split_shots("val_select")
    positions = np.linspace(0, len(selection) - 1, num=min(val_shots, len(selection))).round().astype(int)
    return [selection[i] for i in positions]


def train_direct(task: TaskSpec, cfg: DirectConfig, out_dir: Path, device: str, *, num_workers: int = 32,
                 loader_batch: int | None = None) -> Path:
    """Train on the training split and return the path of `best.pt` (the EMA of the best validation).

    `num_workers` is the loader's worker count for training and for each validation (the training
    loader's workers sit idle while a validation runs, because its queues are full).
    """
    out_dir = Path(out_dir)
    if (out_dir / "best.pt").exists():
        raise FileExistsError(f"{out_dir / 'best.pt'} exists; remove it (the command's --force) to train again")
    out_dir = paths.prepare_write_root(out_dir)
    write_config(out_dir, task, cfg)
    torch.manual_seed(cfg.seed)

    model = DirectModel(task, cfg).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    params = list(model.parameters())
    opt = torch.optim.AdamW(_param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=(0.9, 0.95))
    counts = token_counts(task, cfg)
    print(f"[direct] {task.name}: {sum(p.numel() for p in params) / 1e6:.1f} M parameters, "
          f"{sum(counts.values())} tokens per window {counts}, batch {cfg.batch_size}, "
          f"{cfg.max_steps} steps, device {device}", flush=True)

    best_path, last_path = out_dir / "best.pt", out_dir / "last.pt"
    log_path = out_dir / "train_log.csv"
    with log_path.open("w", newline="") as f:
        csv.writer(f).writerow(LOG_COLUMNS)

    def validate(step: int) -> float:
        shots = validation_shots(cfg.val_shots)
        result = harness.score(task, _predict_fn(ema, task, cfg, device), out_dir / "val" / f"step_{step:06d}",
                               split="val_select", shots=shots, history_s=cfg.history_s,
                               num_workers=num_workers)
        return float(result.task_nrmse)

    batches = _Batches(task, cfg, num_workers, loader_batch or min(LOADER_BATCH, cfg.batch_size))
    started = time.monotonic()
    best = math.inf
    step = skipped = 0
    window_loss, window_n = 0.0, 0         # since the last log row
    seg_seconds, seg_windows, seg_shots, seg_batches = 0.0, 0, 0, 0
    line_started, line_steps = time.monotonic(), 0
    try:
        while step < cfg.max_steps:
            t0 = time.monotonic()
            batch = batches.get()
            x, a, y = to_tensors(batch, task, device)
            for g in opt.param_groups:
                g["lr"] = cfg.lr * _lr_factor(step, cfg)
            model.train()
            with _autocast(device):
                out = model(x, a)
            loss = masked_mse(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            loss_value = float(loss.detach())
            if math.isfinite(loss_value) and math.isfinite(float(grad_norm)):
                opt.step()
                window_loss += loss_value
                window_n += 1
            else:
                skipped += 1
                if skipped > MAX_SKIPPED_STEPS:
                    raise RuntimeError(f"step {step}: loss or gradient is not finite ({skipped} steps skipped); "
                                       "the run is aborted")
            step += 1
            _update_ema(ema, model, min(cfg.ema, (1 + step) / (10 + step)))
            seg_seconds += time.monotonic() - t0
            seg_windows += len(batch.shot_ids)
            seg_shots += len(np.unique(batch.shot_ids))
            seg_batches += 1
            line_steps += 1

            if step % LOG_EVERY == 0:
                now = time.monotonic()
                peak = torch.cuda.max_memory_allocated(device) / 2**30 if str(device).startswith("cuda") else 0.0
                print(f"[direct] step {step} loss {window_loss / max(window_n, 1):.4f} "
                      f"{line_steps / (now - line_started):.2f} steps/s, "
                      f"{seg_windows / max(seg_seconds, 1e-9):.0f} windows/s, "
                      f"{seg_shots / seg_batches:.1f} shots per batch, peak {peak:.1f} GiB, skipped {skipped}",
                      flush=True)
                line_started, line_steps = now, 0
            if step % cfg.val_every == 0 or step == cfg.max_steps:
                nrmse = validate(step)
                row = [step, window_loss / max(window_n, 1), nrmse, f"{time.monotonic() - started:.1f}",
                       f"{seg_windows / max(seg_seconds, 1e-9):.1f}", f"{seg_shots / seg_batches:.2f}"]
                with log_path.open("a", newline="") as f:
                    csv.writer(f).writerow(row)
                save_checkpoint(last_path, ema, cfg, task, step=step, val_nrmse=nrmse)
                improved = math.isfinite(nrmse) and nrmse < best
                if improved:
                    best = nrmse
                    save_checkpoint(best_path, ema, cfg, task, step=step, val_nrmse=nrmse)
                print(f"[direct] validation at step {step}: nrmse {nrmse:.4f}"
                      f"{' (best)' if improved else ''}", flush=True)
                window_loss, window_n = 0.0, 0
                seg_seconds, seg_windows, seg_shots, seg_batches = 0.0, 0, 0, 0
                line_started, line_steps = time.monotonic(), 0
    finally:
        batches.close()
    if not best_path.exists():
        raise RuntimeError("no validation had a finite NRMSE; there is no best.pt")
    return best_path
