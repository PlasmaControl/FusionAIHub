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

import json
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


def _git_revision() -> str:
    try:
        done = subprocess.run(["git", "-C", str(_REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True, timeout=30)
        return done.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_score_json(result: ScoreResult, predictor: str, split: str) -> Path:
    """Write ``score.json`` into ``result.out_dir`` beside TokaMark's evaluator files; return its path."""
    path = Path(result.out_dir) / "score.json"
    record = {
        "task": result.task,
        "predictor": predictor,
        "split": split,
        "nrmse": result.task_nrmse,
        "n_shots": int(len(result.per_shot)),
        "n_windows": int(result.n_windows),
        "git_revision": _git_revision(),
        "date": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path
