"""The two parameter-free baselines of the TokaMark paper, and the file that records a score.

Mean: every output sample is the signal's mean (TokaMark's statistics file, `SignalSpec.mean`).
Persistence: the last finite input sample of each channel, repeated over the output window;
where a channel has no finite input sample, the signal's mean. That is TokaMark's rule
(`scripts/persistence.py` of the benchmark): forward fill over the whole history. The harness
computes it as `WindowBatch.last_valid`. It is defined only for tasks whose every output
signal is also an input.

Nothing here imports `tokamark`; the predictors work on a `WindowBatch` alone.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .harness import PredictFn, ScoreResult, WindowBatch
from .tasks import TaskSpec

_REPO = Path(__file__).resolve().parents[4]


def mean_predictor(task: TaskSpec) -> PredictFn:
    """Predict each output signal's mean everywhere (float32, shaped like the targets)."""
    means = {s.name: np.float32(s.mean) for s in task.outputs}

    def predict(batch: WindowBatch) -> dict[str, np.ndarray]:
        return {name: np.full(batch.targets[name].shape, mean, dtype=np.float32)
                for name, mean in means.items()}

    return predict


def persistence_predictor(task: TaskSpec) -> PredictFn:
    """Predict the last finite input sample of each channel; the signal's mean where there is none.

    Raises ``ValueError`` if an output signal of the task is not also an input.
    """
    input_names = {s.name for s in task.inputs}
    missing = [s.name for s in task.outputs if s.name not in input_names]
    if missing:
        raise ValueError(
            f"persistence is defined only when every output is also an input; task {task.name} "
            f"has outputs that are not inputs: {missing}"
        )
    means = {s.name: np.float32(s.mean) for s in task.outputs}

    def predict(batch: WindowBatch) -> dict[str, np.ndarray]:
        out = {}
        for name, mean in means.items():
            last = np.asarray(batch.last_valid[name], dtype=np.float32)
            filled = np.where(np.isfinite(last), last, mean).astype(np.float32)
            n_t = batch.targets[name].shape[-1]
            out[name] = np.ascontiguousarray(np.broadcast_to(filled[..., None], (*filled.shape, n_t)))
        return out

    return predict


def outside_tolerance(ours: float, published: float | None, tolerance: float) -> bool:
    """The gate's verdict on one score.

    True when `ours` is not finite (a NaN must never read as agreement: `abs(nan - x) > tol` is
    False). False when `published` is None (no difference is judged) and `ours` is finite.
    Otherwise True when `|ours - published| > tolerance`.
    """
    if not math.isfinite(ours):
        return True
    if published is None:
        return False
    return abs(ours - published) > tolerance


# The code that determines a baseline score: the harness (windows, scoring), the task specs
# (statistics, step counts, splits) and this module (the predictors).
_SCORING_FILES = ("harness.py", "tasks.py", "baselines.py")


def _hash_scoring_files() -> str:
    """sha256 over the bytes of the scoring files on disk now, each preceded by its name."""
    h = hashlib.sha256()
    for name in _SCORING_FILES:
        h.update(name.encode() + b"\n")
        h.update((Path(__file__).parent / name).read_bytes())
    return h.hexdigest()


def _git_revision() -> str:
    try:
        done = subprocess.run(["git", "-C", str(_REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True, timeout=30)
        return done.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# Taken once, here, within milliseconds of the scoring modules being loaded (`.harness` and
# `.tasks` are imported above) and never read again: a process that runs for hours must stamp its
# scores with the code it is running, not with whatever the files hold when a score is written.
# Observed 2026-11-22: scores computed by code loaded at 7eb42e9f carried git_revision e86aed5b
# because the revision and the hash were read at write time.
_CODE_SHA256 = _hash_scoring_files()
_GIT_REVISION = _git_revision()


def scoring_code_sha256() -> str:
    """The sha256 of the scoring code as loaded by this process (taken at import, not re-read).

    A stored score is the same score only if this value is unchanged.
    """
    return _CODE_SHA256


_SCORE_KEYS = ("task", "predictor", "split", "nrmse", "n_shots", "n_windows", "git_revision", "date",
               "code_sha256")


def read_kept_score(out_dir: Path, task: str, predictor: str, split: str) -> tuple[dict | None, str]:
    """The stored ``score.json`` of `out_dir` if it is the score asked for, else ``(None, reason)``.

    Kept only if the file exists, parses, has every key, names this task, predictor and split,
    and was written by the scoring code this process is running (`scoring_code_sha256`, as loaded).
    """
    path = Path(out_dir) / "score.json"
    if not path.exists():
        return None, "no score.json"
    try:
        stored = json.loads(path.read_text())
        if not isinstance(stored, dict) or any(k not in stored for k in ("task", "predictor", "split")):
            raise ValueError("not a score record")
    except (OSError, ValueError):
        return None, "unreadable score.json"
    if (stored["task"], stored["predictor"], stored["split"]) != (task, predictor, split):
        return None, f"score.json is for {stored['task']}/{stored['predictor']}/{stored['split']}"
    if stored.get("code_sha256") != scoring_code_sha256() or any(k not in stored for k in _SCORE_KEYS):
        return None, f"scoring code changed since {stored.get('git_revision', 'unknown')}"
    return stored, ""


def write_score_json(result: ScoreResult, predictor: str, split: str) -> Path:
    """Write ``score.json`` into ``result.out_dir`` beside TokaMark's evaluator files; return its path.

    The git revision and `code_sha256` describe the code as loaded at import, not the files at write time.
    """
    path = Path(result.out_dir) / "score.json"
    record = {
        "task": result.task,
        "predictor": predictor,
        "split": split,
        "nrmse": result.task_nrmse,
        "n_shots": int(len(result.per_shot)),
        "n_windows": int(result.n_windows),
        "git_revision": _GIT_REVISION,
        "date": datetime.now(timezone.utc).isoformat(),
        "code_sha256": _CODE_SHA256,
    }
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path
