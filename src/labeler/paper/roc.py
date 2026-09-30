"""The ROC of each paper phenomenon's selected model on its test shots, once.

    python -m labeler.paper.roc [--method M ...] [--ae-version V]

For each phenomenon (`ORDER`) the model selected for main inference
(`SELECTED`): AE's `ae_xpower` at `--ae-version` (default `AE_VERSION`, v2),
and the other four's frame models (`roster.TABLES`) at `frames.VERSION`. Each
ROC is over exactly the frames or bins its evaluation scored its F1 on:

- **AE:** the chosen model (`evaluate.chosen_model`), read and checked as the
  test reads it (`evaluate.load_test`), over the split's reviewed test shots;
  each shot's 0-2 s frames from `evaluate.shot_frames`, with the shot's
  source-table label and no SELDnet. The score is P(AE) (`prob`), a frame is
  positive where the owner says present, and the frames are `scored`'s. A
  whole-window version (`WHOLE_WINDOW_VERSIONS`, v3) scores other frames, so
  it is refused.
- **NTM, H-mode, ELMing, sawteeth:** the test shots of `prepare.split_shots`,
  read with `evaluate.read_shots`; the score is `evaluate.model_probs`; the
  bins are the ones `evaluate.score` counts (`scored_bins`, the rule of
  `frames_train.bin_cells`): ABSENT or PRESENT_T, observed, with a finite P. A
  bin is positive where it is PRESENT_T. H-mode's ROC is `hmode_frames`' own,
  for H; its L-mode companion (`evaluate.LMODE`) is not drawn.

**The check.** The F1 at the model's own threshold (a unit said present where
its score reaches it) is recomputed from the same scores, `stats.f1` of the
pooled cells. It must equal the evaluation's to `F1_TOLERANCE`, or nothing is
written: AE's `methods.ae_xpower.f1` in `evaluation.json`, a frame model's own
F1 in `scores` (`f1_key`: `f1(H)` for H-mode). The model's sha256 must also be
the one the evaluation names. The check shows the ROC is over the scored set.

**The record** is `roc.json` beside the model's `evaluation.json`
(`roc_file`). It holds `auroc`, exact over every scored unit (the trapezoid
over the full curve, a point per distinct score: the rank statistic with ties
counted half); the curve thinned to at most `ROC_POINTS` points evenly spaced
along it (`thin`), keeping (0, 0) and (1, 1); `n_pos`, `n_neg` and `shots`;
the threshold's own point; the check; the model's and the evaluation's path
and sha256; `git_sha`; and `NOTE`. It has no clock: the same inputs give the
same bytes.

**Written once.** An existing `roc.json` is refused, as `evaluate` refuses a
second test. Every target is checked before any is computed. `--method`
(repeatable, or `all`, the default) picks which to write.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from .. import frames
from ..ae import xpower
from ..ae.xpower import WHOLE_WINDOW_VERSIONS
from ..ae.xpower import evaluate as ae_evaluate
from ..config import Paths, atomic_path, git_sha
from ..events.catalog.states import PRESENT
from ..frames import evaluate as frames_evaluate
from ..frames import prepare
from ..frames import train as frames_train
from ..frames.targets import ABSENT, PRESENT_T
from ..scoring import stats
from . import AE, LOGIN_THREADS, ORDER, roster

AE_METHOD = xpower.METHOD
#: The AE model used for main inference: ae_xpower v2 (the build's `--version v2`).
AE_VERSION = "v2"
#: Each paper phenomenon's model selected for main inference, in `ORDER`.
SELECTED = {c: AE_METHOD if c == AE else roster.TABLES[c] for c in ORDER}
ALL = "all"
ROC_FILE = "roc.json"
ROC_POINTS = 201
F1_TOLERANCE = 1e-9
NOTE = (
    "The ROC is a threshold-free look at the frozen model on the same test shots "
    "its evaluation scored once. Nothing is chosen from it: no threshold, model "
    "or version changes."
)


class F1Mismatch(ValueError):
    """The F1 recomputed from the ROC's scores is not the evaluation's."""


class Scored(NamedTuple):
    """A model's scores on its evaluation's scored units: the score and whether
    each is positive, the test shots, the model's threshold, its file and the
    sha256 of the bytes loaded, and what a unit is."""

    score: np.ndarray
    truth: np.ndarray
    shots: int
    threshold: float
    model: Path
    model_sha256: str
    unit: str


def category_of(method: str) -> str:
    [category] = [c for c, m in SELECTED.items() if m == method]
    return category


def evaluation_file(paths: Paths, method: str, ae_version: str = AE_VERSION) -> Path:
    """The selected model's `evaluation.json`: AE's at `ae_version`, a frame
    model's at `frames.VERSION`."""
    if method == AE_METHOD:
        return xpower.model_dir(paths, ae_version) / "evaluation.json"
    return frames.model_dir(paths, method, frames.VERSION) / "evaluation.json"


def roc_file(paths: Paths, method: str, ae_version: str = AE_VERSION) -> Path:
    """The selected model's `roc.json`, beside its `evaluation.json`."""
    return evaluation_file(paths, method, ae_version).with_name(ROC_FILE)


def f1_key(method: str) -> str:
    """The name of the method's own F1 in a frame evaluation: `f1`, or `f1(H)`
    for H-mode, whose scores are on H and on L (`evaluate.views`)."""
    return "f1" + next(iter(frames_evaluate.views(method)))


def f1_estimate(method: str, evaluation: dict) -> dict | None:
    """The evaluation's F1 of the method with its 95 % shot-bootstrap interval
    (a `stats.Estimate.as_json`); None where the record has none."""
    if method == AE_METHOD:
        return evaluation.get("methods", {}).get(method, {}).get("f1")
    return evaluation.get("intervals", {}).get(method, {}).get(f1_key(method))


def recorded_f1(method: str, evaluation: dict) -> tuple[str, float | None]:
    """Where the evaluation keeps the method's F1, and its value: AE's
    `methods.ae_xpower.f1`, a frame model's own F1 in `scores`."""
    if method == AE_METHOD:
        estimate = f1_estimate(method, evaluation) or {}
        return f"methods.{method}.f1.value", estimate.get("value")
    key = f1_key(method)
    return f"scores.{method}.{key}", evaluation["scores"][method].get(key)


def recorded_model(method: str, evaluation: dict) -> str | None:
    """The model sha256 the evaluation names."""
    if method == AE_METHOD:
        return evaluation.get("meta", {}).get("model_sha256")
    return evaluation.get("model", {}).get("sha256")


def curve(score, truth) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The full ROC, a point per distinct score from the highest down, from
    (0, 0) to (1, 1): `(fp, tp, fpr, tpr)`, the counts and their rates; a unit
    is said present where its score reaches the point's."""
    score = np.asarray(score, dtype=np.float64)
    truth = np.asarray(truth, dtype=bool)
    if score.shape != truth.shape or score.ndim != 1:
        raise ValueError("score and truth must be one-dimensional and alike")
    if not np.isfinite(score).all():
        raise ValueError("every scored unit must have a finite score")
    n_pos = int(truth.sum())
    n_neg = len(truth) - n_pos
    if not n_pos or not n_neg:
        raise ValueError(f"no ROC: {n_pos} positive and {n_neg} negative units")
    order = np.argsort(-score, kind="stable")
    s, t = score[order], truth[order]
    last = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    tp = np.r_[0, np.cumsum(t)[last]].astype(np.float64)
    fp = np.r_[0, last + 1 - tp[1:]].astype(np.float64)
    return fp, tp, fp / n_neg, tp / n_pos


def auroc(score, truth) -> float:
    """The area under the full ROC by the trapezoid, exact: the rank statistic
    with ties counted half. Integer sums, divided once."""
    fp, tp, _, _ = curve(score, truth)
    area2 = float(np.sum(np.diff(fp) * (tp[1:] + tp[:-1])))  # twice the area
    return area2 / (2.0 * tp[-1] * fp[-1])


def thin(fpr, tpr, n: int = ROC_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """At most `n` of the curve's points, evenly spaced along it (by FPR + TPR,
    which rises from 0 to 2 at every point), keeping (0, 0) and (1, 1)."""
    fpr, tpr = np.asarray(fpr, dtype=np.float64), np.asarray(tpr, dtype=np.float64)
    if len(fpr) <= n:
        return fpr, tpr
    along = fpr + tpr
    keep = np.unique(np.searchsorted(along, np.linspace(0, along[-1], n), "left"))
    return fpr[keep], tpr[keep]


def cells(score, truth, threshold: float) -> np.ndarray:
    """`[tp, fp, fn, tn]`, a unit said present where its score reaches
    `threshold`."""
    said = np.asarray(score, dtype=np.float64) >= threshold
    truth = np.asarray(truth, dtype=bool)
    return np.array(
        [
            np.sum(truth & said),
            np.sum(~truth & said),
            np.sum(truth & ~said),
            np.sum(~truth & ~said),
        ],
        dtype=np.float64,
    )


def check_f1(computed: float, recorded: float | None, where: str) -> None:
    """Refuse (`F1Mismatch`) an F1 that is not the evaluation's to `F1_TOLERANCE`."""
    if recorded is None or not abs(computed - float(recorded)) <= F1_TOLERANCE:
        raise F1Mismatch(
            f"{where}: the F1 recomputed from the ROC's scores, {computed!r}, is not "
            f"the evaluation's, {recorded!r}: the ROC would not be over the scored "
            "set"
        )


def scored_bins(prob, states, observed=None) -> tuple[np.ndarray, np.ndarray]:
    """The bins `evaluate.score` counts, by `frames_train.bin_cells`' rule:
    ABSENT or PRESENT_T, observed, with a finite P. Their P, and whether each
    is PRESENT_T."""
    prob, states = np.asarray(prob, dtype=np.float64), np.asarray(states)
    scored = np.isin(states, (ABSENT, PRESENT_T)) & np.isfinite(prob)
    if observed is not None:
        scored &= np.asarray(observed, bool)
    return prob[scored], states[scored] == PRESENT_T


def frame_scored(paths: Paths, method: str) -> Scored:
    """A frame model's P on the bins its evaluation scored (module docstring)."""
    spec = frames.SPECS[method]
    file = frames.model_dir(paths, method, frames.VERSION) / "model.pt"
    data = file.read_bytes()
    model, blob = frames_train.load(io.BytesIO(data))
    split = prepare.split_shots(paths, method)
    test = [s for s, v in split.items() if v == "test"]
    shots, _ = frames_evaluate.read_shots(paths, method, test)
    parts = [
        scored_bins(frames_evaluate.model_probs(model, spec, s), s.states, s.observed)
        for s in shots
    ]
    score = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0)
    truth = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0, bool)
    return Scored(
        score,
        truth,
        len(shots),
        float(blob["threshold"]),
        file,
        hashlib.sha256(data).hexdigest(),
        f"{spec.bin_ms:g} ms bins",
    )


def ae_scored(paths: Paths, version: str = AE_VERSION) -> Scored:
    """ae_xpower's P(AE) on the frames its evaluation scored (module
    docstring); a whole-window version is refused."""
    if version in WHOLE_WINDOW_VERSIONS:
        raise ValueError(
            f"ae_xpower {version} is scored over the owner's whole windows, not "
            "the 0-2 s frames this ROC reads"
        )
    models = xpower.model_dir(paths, version)
    inputs = ae_evaluate.load_test(paths, models, version)
    if inputs.file != ae_evaluate.chosen_model(models):
        raise ValueError(f"{inputs.file}: not chosen.json's model")
    scores, truths = [], []
    test = ae_evaluate._reviewed(inputs.split, "test", 0)
    for shot in test:
        source = inputs.source.labels.get(shot)
        if source is None:
            raise ValueError(f"test shot {shot} has no source-table label")
        f = ae_evaluate.shot_frames(
            shot,
            paths=paths,
            label=inputs.saved[shot],
            model=inputs.model,
            blob=inputs.blob,
            source=source,
            seldnet=None,
        )
        scores.append(np.asarray(f.prob, dtype=np.float64)[f.scored])
        truths.append((f.owner == PRESENT)[f.scored])
    return Scored(
        np.concatenate(scores) if scores else np.zeros(0),
        np.concatenate(truths) if truths else np.zeros(0, bool),
        len(test),
        float(inputs.blob["threshold"]),
        inputs.file,
        hashlib.sha256(inputs.snapshots["model.pt"]).hexdigest(),
        "10 ms frames of 0-2 s",
    )


def record(
    scored: Scored,
    evaluation: dict,
    *,
    method: str,
    version: str,
    evaluation_path: Path,
    evaluation_sha256: str,
) -> dict:
    """`roc.json`'s record, after the checks: the model's sha256 is the one
    the evaluation names, and the F1 at its threshold is the evaluation's
    (`check_f1`)."""
    named = recorded_model(method, evaluation)
    if named != scored.model_sha256:
        raise ValueError(
            f"{scored.model}: its sha256 is not the one {evaluation_path} names"
        )
    c = cells(scored.score, scored.truth, scored.threshold)
    computed = float(stats.f1(c))
    where, value = recorded_f1(method, evaluation)
    check_f1(computed, value, f"{evaluation_path} ({where})")
    _, _, fpr, tpr = curve(scored.score, scored.truth)
    thin_fpr, thin_tpr = thin(fpr, tpr)
    tp, fp, fn, tn = c
    return {
        "phenomenon": category_of(method),
        "method": method,
        "version": version,
        "unit": scored.unit,
        "shots": scored.shots,
        "n_pos": int(scored.truth.sum()),
        "n_neg": int((~scored.truth).sum()),
        "auroc": auroc(scored.score, scored.truth),
        "curve": {
            "fpr": thin_fpr.tolist(),
            "tpr": thin_tpr.tolist(),
            "points": len(thin_fpr),
            "full_points": len(fpr),
        },
        "threshold": {
            "value": scored.threshold,
            "fpr": fp / (fp + tn),
            "tpr": tp / (tp + fn),
        },
        "f1_check": {
            "computed": computed,
            "evaluation": value,
            "evaluation_key": where,
            "tolerance": F1_TOLERANCE,
            "cells": [int(x) for x in c],
        },
        "model": {"path": str(scored.model), "sha256": scored.model_sha256},
        "evaluation": {"path": str(evaluation_path), "sha256": evaluation_sha256},
        "git_sha": git_sha(full=True),
        "note": NOTE,
    }


def roc(paths: Paths, method: str, ae_version: str = AE_VERSION) -> dict:
    """The method's `roc.json` record, its evaluation read first."""
    path = evaluation_file(paths, method, ae_version)
    data = path.read_bytes()
    evaluation = json.loads(data)
    if method == AE_METHOD:
        scored, version = ae_scored(paths, ae_version), ae_version
    else:
        scored, version = frame_scored(paths, method), frames.VERSION
    return record(
        scored,
        evaluation,
        method=method,
        version=version,
        evaluation_path=path,
        evaluation_sha256=hashlib.sha256(data).hexdigest(),
    )


def write(
    paths: Paths, methods, *, ae_version: str = AE_VERSION, done=None
) -> dict[str, dict]:
    """Write each method's `roc.json`, once: refused, before any is computed,
    if one exists. `done(method, record, path, seconds)` is called after each."""
    targets = {m: roc_file(paths, m, ae_version) for m in methods}
    there = [str(t) for t in targets.values() if t.exists()]
    if there:
        raise FileExistsError(f"{', '.join(there)}: the ROC is recorded once")
    out = {}
    for method, target in targets.items():
        start = time.monotonic()
        found = roc(paths, method, ae_version)
        with atomic_path(target) as tmp:
            tmp.write_text(json.dumps(found, indent=1) + "\n")
        out[method] = found
        if done is not None:
            done(method, found, target, time.monotonic() - start)
    return out


def methods_of(named: list[str] | None) -> list[str]:
    """The methods `--method` names, in `ORDER`; all without one or with `all`."""
    if not named or ALL in named:
        return list(SELECTED.values())
    return [m for m in SELECTED.values() if m in named]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--method",
        action="append",
        choices=[*SELECTED.values(), ALL],
        help=f"repeatable; default {ALL}",
    )
    p.add_argument(
        "--ae-version",
        default=AE_VERSION,
        help=f"ae_xpower's version (default {AE_VERSION})",
    )
    args = p.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", LOGIN_THREADS)))
    paths = Paths.from_env()

    def done(method: str, found: dict, target: Path, seconds: float) -> None:
        line = {
            "method": method,
            "roc": str(target),
            "auroc": found["auroc"],
            "f1": found["f1_check"]["computed"],
            "n_pos": found["n_pos"],
            "n_neg": found["n_neg"],
            "seconds": round(seconds, 1),
        }
        print(json.dumps(line), flush=True)

    try:
        write(paths, methods_of(args.method), ae_version=args.ae_version, done=done)
    except (OSError, ValueError) as error:
        p.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
