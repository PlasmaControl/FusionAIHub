"""Re-choose a frame model's threshold on its val shots, once (round three, T1;
the owner, 2026-09-30 01:28: "yes rechoose ntm and h mode threshold").

    python -m labeler.frames.rethreshold --method {hmode_frames,ntm_frames}

**Bins.** The model (`frames.VERSION`'s) is run over its split's val shots as
`train.fit` scored them at training: each val shot's prepared features
(`train.read_shot`), its P per bin pooled as the model was trained
(`train.bin_probs`), and the bins `train.bin_cells` counts, ABSENT or PRESENT_T
with every frame observed. The split must be the one the model was trained on
(its sha256, `evaluate._check`), and the val shots with features the ones its
checkpoint names. The test shots are never read.

**The check.** On training's grid (`train.THRESHOLDS`) the search must give the
model's own threshold, which shows the val bins are the training's; otherwise
nothing is written (`NotReproduced`, with both numbers).

**The search.** Then on `FINE_THRESHOLDS`, 0.01-0.99 in steps of 0.01, by the
spec's `threshold_rule` and the same tie rule (`train.pick_threshold`): the
present class's F1 for NTM, the mean of F1(H) and F1(L) for H-mode.

**Output.** `threshold.json` (`train.THRESHOLD_FILE`) beside `model.pt`, once:
the threshold, its rule and grid (`grid_of`); the trained threshold, its grid
and what that grid gave here (`reproduced`); the val score at each threshold
(`val_score`); the val shots and their scored bins by class
(`train.class_bins`); the model's path and sha256 and the split's sha256;
`WHY`, which says the re-choice was made after the v2 test scores were seen;
and `git_sha`. It has no clock. `train.load` then gives this threshold in
place of training's to every reader. As `labeler.paper.roc.write` does, it is
refused while tracked files differ from the commit (or git cannot say).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json

import numpy as np

from ..config import Paths, atomic_path, git_dirty, git_sha
from . import SPECS, VERSION, model_dir, prepare, shots_file
from . import evaluate as frames_evaluate
from . import train as frames_train

#: The methods the owner asked to re-choose the threshold of.
METHODS = ("hmode_frames", "ntm_frames")
FINE_THRESHOLDS = np.round(np.arange(0.01, 0.991, 0.01), 2)
WHY = (
    "re-chosen on the val shots at the owner's word (2026-09-30 01:28), after "
    "the v2 test scores were seen; the test shots did not choose it"
)


class NotReproduced(ValueError):
    """Training's grid does not give the model's own threshold on its val shots."""


def grid_of(grid) -> dict:
    """A grid's first and last threshold, its step and its number of points."""
    grid = np.asarray(grid, dtype=np.float64)
    return {
        "from": float(grid[0]),
        "to": float(grid[-1]),
        "step": float(np.round(np.diff(grid).mean(), 6)),
        "points": len(grid),
    }


def val_shots(paths: Paths, method: str, blob: dict) -> list[frames_train.Shot]:
    """The split's val shots with features, read as `train.fit` read them;
    refused unless the split is the model's and those shots are the ones its
    checkpoint names."""
    split_bytes = shots_file(paths, method, VERSION).read_bytes()
    split = prepare.split_shots(paths, method)
    frames_evaluate._check(blob, SPECS[method], split, split_bytes)
    shots = frames_train._load(
        paths,
        method,
        sorted(s for s, v in split.items() if v == "val"),
        0,
        prepare.dropped(paths, method),
        {},
    )
    named = [int(s) for s in blob.get("shots", {}).get("val", [])]
    if [s.shot for s in shots] != named:
        raise ValueError(
            f"{method}: the val shots with features are not the {len(named)} the "
            "model was stopped on"
        )
    return shots


def val_score(method: str, rule: str, probs, shots, threshold: float) -> dict:
    """The val bins at `threshold`: the pooled `[tp, fp, fn, tn]`, the rule's
    score (`train.rule_score`), and precision, recall and F1, H's and L's for
    H-mode (`evaluate.views`); None where undefined."""
    cells = np.zeros(4)
    for prob, shot in zip(probs, shots, strict=True):
        cells += frames_train.bin_cells(prob, shot.states, threshold, shot.observed)
    out = {
        "threshold": float(threshold),
        "cells": [int(c) for c in cells],
        "score": frames_evaluate._plain(frames_train.rule_score(cells, rule)),
    }
    for suffix, swap in frames_evaluate.views(method).items():
        view = frames_evaluate.swapped(cells) if swap else cells
        for name, metric in frames_evaluate.METRICS.items():
            out[name + suffix] = frames_evaluate._plain(metric(view))
    return out


def rethreshold(paths: Paths, method: str) -> dict:
    """Re-choose the method's threshold on its val shots (module docstring); the
    record written to `threshold.json`."""
    if method not in METHODS:
        raise ValueError(f"{method}: the owner asked for hmode_frames and ntm_frames")
    folder = model_dir(paths, method, VERSION)
    target = folder / frames_train.THRESHOLD_FILE
    if target.exists():
        raise FileExistsError(f"{target}: a threshold is re-chosen once")
    if git_dirty() is not False:
        raise RuntimeError(
            "tracked files differ from the commit (or git cannot say), so no "
            "record could name the code that computed it: commit first"
        )
    spec = SPECS[method]
    model_path = folder / "model.pt"
    data = model_path.read_bytes()
    model, blob = frames_train.load(io.BytesIO(data), folder=folder)
    trained = blob["trained_threshold"]
    shots = val_shots(paths, method, blob)
    per = frames_train.frames_per_bin(spec)
    probs = [frames_train.bin_probs(model, s.x, per, spec.pool) for s in shots]
    rule = spec.threshold_rule
    args = (probs, [s.states for s in shots], [s.observed for s in shots], rule)
    reproduced = frames_train.pick_threshold(*args, frames_train.THRESHOLDS)
    if reproduced != trained:
        raise NotReproduced(
            f"{method}: on its {len(shots)} val shots the trained grid gives "
            f"{reproduced}, not the model's threshold {trained}, so the val bins "
            "are not the training's; nothing is written"
        )
    threshold = frames_train.pick_threshold(*args, FINE_THRESHOLDS)
    record = {
        "method": method,
        "version": VERSION,
        "threshold": threshold,
        "threshold_rule": rule,
        "tie": "the one nearest 0.5",
        "grid": grid_of(FINE_THRESHOLDS),
        "trained_threshold": trained,
        "trained_grid": grid_of(frames_train.THRESHOLDS),
        "reproduced": reproduced,
        "val": {
            "at_threshold": val_score(method, rule, probs, shots, threshold),
            "at_trained_threshold": val_score(method, rule, probs, shots, trained),
            "shots": [s.shot for s in shots],
            "bins": frames_train.class_bins([s.states[s.observed] for s in shots]),
        },
        "model": {
            "path": str(model_path),
            "sha256": hashlib.sha256(data).hexdigest(),
        },
        "split_sha256": blob["split_sha256"],
        "why": WHY,
        "git_sha": git_sha(full=True),
    }
    with atomic_path(target) as tmp:
        tmp.write_text(json.dumps(record, indent=1) + "\n")
    return record


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--method", required=True, choices=list(METHODS))
    args = p.parse_args(argv)
    paths = Paths.from_env()
    try:
        record = rethreshold(paths, args.method)
    except (OSError, ValueError, RuntimeError) as error:
        p.error(str(error))
    line = {
        "method": args.method,
        "threshold": record["threshold"],
        "trained_threshold": record["trained_threshold"],
        "rule": record["threshold_rule"],
        "val": {k: record["val"][k] for k in ("at_threshold", "at_trained_threshold")},
        "bins": record["val"]["bins"],
    }
    print(json.dumps(line), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
