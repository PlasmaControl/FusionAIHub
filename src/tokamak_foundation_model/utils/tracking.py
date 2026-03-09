"""
Tracking utilities for training models.
This is based on https://github.com/descriptinc/audiotools/blob/master/audiotools/ml/decorators.py
"""

import math
import os
import time
from collections import defaultdict
from functools import wraps

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

import wandb


def default_list():
    return []


class Mean:
    """Keeps track of the running mean, along with the latest value."""

    def __init__(self) -> None:
        self.count: int = 0
        self.total: float = 0.

    def __call__(self):
        return self.total / max(self.count, 1)

    def reset(self) -> None:
        self.count = 0
        self.total = 0.

    def update(self, val: float) -> None:
        if math.isfinite(val):
            self.count += 1
            self.total += val


def when(condition):
    """Runs a function only when the condition is met."""
    def decorator(fn):
        @wraps(fn)
        def decorated(*args, **kwargs):
            if condition():
                return fn(*args, **kwargs)
        return decorated
    return decorator


def timer(prefix: str = "time"):
    """Adds execution time to the output dictionary of the decorated function."""
    def decorator(fn):
        @wraps(fn)
        def decorated(*args, **kwargs):
            s = time.perf_counter()
            output = fn(*args, **kwargs)
            assert isinstance(output, dict)
            e = time.perf_counter()
            output[f"{prefix}/{fn.__name__}"] = e - s
            return output
        return decorated
    return decorator


class Tracker:
    """Tracks training progress and logs metrics to console and TensorBoard."""

    def __init__(
        self,
        writer: SummaryWriter | None = None,
        rank: int = 0,
        step: int = 0,
    ):
        self.metrics: dict = {}
        self.history: dict = {}
        self.writer = writer
        self.rank = rank
        self.step = step
        self._progress: dict = {}  # label -> {completed, total}

    def _write(self, msg: str):
        print(msg)

    def print(self, msg: str):
        if self.rank == 0:
            self._write(str(msg))

    def update(self, label: str, fn_name: str):
        if self.rank != 0:
            return

        prog = self._progress.get(label, {})
        completed = prog.get("completed", 0)
        total = prog.get("total", "?")

        parts = [f"[{label}] {fn_name} | step {completed}/{total}"]
        for k, v in self.metrics[label]["value"].items():
            mean = self.metrics[label]["mean"][k]()
            parts.append(f"  {k}: {v:.6f} (mean: {mean:.6f})")

        self._write("\n".join(parts))

    def done(self, label: str, title: str):
        for lbl in self.metrics:
            for v in self.metrics[lbl]["mean"].values():
                v.reset()

        if self.rank == 0:
            self._write(f"\n{'='*40}\n{title}\n{'='*40}")
            for lbl in self.metrics:
                self._write(f"[{lbl}] Final means:")
                for k, m in self.metrics[lbl]["mean"].items():
                    self._write(f"  {k}: {m():.6f}")

    def track(
            self,
            label: str,
            length: int,
            completed: int = 0,
            log_every: int = 0,
            op: dist.ReduceOp.RedOpType = dist.ReduceOp.AVG,
            ddp_active: bool = "LOCAL_RANK" in os.environ,
    ):
        self._progress[label] = {"completed": completed, "total": length}
        self.metrics[label] = {
            "value": defaultdict(),
            "mean": defaultdict(lambda: Mean()),
        }

        def decorator(fn):
            @wraps(fn)
            def decorated(*args, **kwargs):
                output = fn(*args, **kwargs)
                self._progress[label]["completed"] += 1

                if not isinstance(output, dict):
                    if log_every > 0 and self._progress[label]["completed"] % log_every == 0:
                        self.update(label, fn.__name__)
                    return output

                scalar_keys = []
                for k, v in output.items():
                    if isinstance(v, (int, float)):
                        v = torch.tensor([v])
                    if not torch.is_tensor(v):
                        continue
                    if ddp_active and v.is_cuda:
                        dist.all_reduce(v, op=op)
                    output[k] = v.detach()
                    if torch.numel(v) == 1:
                        scalar_keys.append(k)
                        output[k] = v.item()

                for k, v in output.items():
                    if k not in scalar_keys:
                        continue
                    self.metrics[label]["value"][k] = v
                    self.metrics[label]["mean"][k].update(v)

                if log_every > 0 and self._progress[label]["completed"] % log_every == 0:
                    self.update(label, fn.__name__)
                return output

            return decorated

        return decorator

    def log(self, label: str, value_type: str = "value", history: bool = True):
        assert value_type in ["mean", "value"]
        if history and label not in self.history:
            self.history[label] = defaultdict(default_list)

        def decorator(fn):
            @wraps(fn)
            def decorated(*args, **kwargs):
                output = fn(*args, **kwargs)
                if self.rank == 0:
                    metrics = self.metrics[label][value_type]
                    for k, v in metrics.items():
                        v = v() if isinstance(v, Mean) else v
                        if self.writer is not None:
                            self.writer.add_scalar(f"{k}/{label}", v, self.step)
                        if wandb.run is not None:
                            wandb.log({f"{k}/{label}": v}, step=self.step)
                        if label in self.history:
                            self.history[label][k].append(v)
                    if label in self.history:
                        self.history[label]["step"].append(self.step)
                return output
            return decorated

        return decorator

    def is_best(self, label: str, key: str) -> bool:
        values = self.history[label][key]
        if not values:
            return False
        return values[-1] == min(values)

    def state_dict(self) -> dict:
        return {"history": self.history, "step": self.step}

    def load_state_dict(self, state_dict: dict) -> "Tracker":
        self.history = state_dict["history"]
        self.step = state_dict["step"]
        return self