"""Score a method's model on its test shots, once, against its baselines and its
bar (round three, Part B; spec §3.4).

    python -m labeler.frames.evaluate --method M [--pilot]

**Bins.** A shot's bins are its features' (`prepare`, of `frames.VERSION`, F1):
the whole `bin_ms` bins inside its window, and their states the original's
target with the owner's label over it (`targets.merged`, F2). A bin is scored
when its target is ABSENT or PRESENT_T and every one of its 10 ms frames is
observed (`score`); UNKNOWN and UNCERTAIN_T bins are not. The model says
present where its P, its frames' logits pooled as it was trained
(`model.bin_logits`), reaches the threshold chosen on the val shots
(`train.fit`).

**Baselines** (`baselines`, the spec's `baselines`), each a decision per bin:
- `always`: present everywhere;
- `elm_onsets`: `spans.elm_onsets`, the ELM clock's onsets inside its present
  spans, a bin present when it holds one: computed here, not read from a table
  (the legacy shots have no v2 row);
- `elm_clock`: `detect_elm`'s present spans (the same call's), a bin present
  when a span touches it, as the model's maximum pools onsets;
- `dalpha_lh`: `detect_hmode`'s present spans where its inputs are on disk
  (few legacy shots have CO2), a bin present when they cover half of it; it is
  reported and never gated.
A span baseline cannot see a bin outside what it measured; a shot whose inputs
are missing (`spans.INPUT_MISSING`) is left out of that baseline, with its
reason. Each baseline is scored on the bins it and the model both see, so its
paired difference with the model is over the same bins.

**Scores.** Precision, recall and F1 of the `[tp, fp, fn, tn]` pooled over the
test shots, with 95 % shot-bootstrap intervals (`scoring.stats`, 2000
replicates, seed 20260923), and each paired F1 difference, the model's less a
baseline's, on the same resampled shots. `hmode_frames` is scored on H and on L:
L is "labelled and not H" (D38), so its cells are H's with present and absent
swapped, and `lmode_frames` gets its own scores from them.

**The bar** (`verdict`): the spec's criteria, each a list of conditions on the
model's score or on the lower end of a paired difference's interval
(`lo(...)`); an undefined quantity fails. Whatever the verdict, the tier is
`suggestions`.

**Effectively always** (`always_like`, F6). The model is effectively the
`always` baseline when its calls equal "always present" on at least
`ALWAYS_AGREEMENT` (99 %) of the scored test bins, or when the upper end of the
paired F1 - always interval (F1(H) - always for H-mode) is below
`ALWAYS_MARGIN` (0.01). `evaluation.json`'s `effectively_always` says whether,
`always_like` the numbers and which of the two held, and the md says it in one
line; `apply` copies the flag into each suggestion table's meta, beside `bar`,
and the coverage marks such a model's suggestions.

**The owner's saves.** There is no owner split (F2 supersedes D40's): the
owner's saves are laid over the original's target, and the owner's shots in
test are scored with the rest, by the merged target. `owner` counts them (the
shots file's `owner` column) and those whose target is the owner's label alone
(`added_test_shots`: no original, added, not overriding), and `owner.flips`
what the owner's saves changed (`owner_flips`, F14), read here from the
frozen owner file and the original target (after `prepare.check_frozen`), not
from the features: over the scored test shots' observed bins (`test`), and
over every bin in the window of each of the owner's shots in the split
(`all`). For ELMs those saves are spans of ELMy time, not onsets, and the md
says how that changes E1 and E2 on the owner's shots.

**Output.** `evaluation.json` (the same bytes each time for the same inputs: no
clock in it) and `evaluation.md`, beside `model.pt` in `frames.model_dir`. The
test shots are scored once: an existing `evaluation.json` is refused, except
for a pilot's (`--pilot`, `train.pilot_dir`, under `runs/`). The md says what
the test cannot show: D44's limits for H/L, and the campaign years (`shot_years`)
of the split's shots (train, val, test; the legacy ones, outside the roster,
and the roster's) beside the population's, with the population's years
that no test shot is from. The md also counts the test shots left out for want
of features (`prepare`'s drops), with their reasons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import operator
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..ae.xpower import pilot_area
from ..config import Paths, atomic_path, git_sha
from ..events import spans
from ..events.catalog.population import read_pool
from ..events.catalog.states import PRESENT
from ..events.review import labels
from ..literature.osti import shot_year, year_starts
from ..scoring import stats
from . import (
    SEED,
    SPECS,
    VERSION,
    EventSpec,
    features_dir,
    model_dir,
    owner_file,
    prepare,
    shots_file,
    shots_meta_file,
    targets,
)
from . import train as frames_train
from .targets import ABSENT, PRESENT_T

METRICS = {"precision": stats.precision, "recall": stats.recall, "f1": stats.f1}
OPS = {">=": operator.ge, ">": operator.gt}
#: The method whose model also scores L, and the name L's scores go under.
LMODE = {"hmode_frames": "lmode_frames"}
ELM_OWNER_NOTE = (
    "On the owner's shots the target is ELMy time, not onsets: the owner's ELM "
    "saves are spans of ELMy time, so a bin inside one is present though it "
    "holds no onset, and an onset bin outside one is absent. There E1 reads a "
    "model that calls onsets alone as missing the spans' onset-free bins (its "
    "recall and F1 fall) and each onset it calls outside a span as false (its "
    "precision falls), and E2, the model less elm_onsets, which calls a bin only "
    "when it holds an onset, favours a model that calls ELMy time. Elsewhere the "
    "target is Hiro's onsets."
)
OWNER_NOTE = (
    "The owner's saves, frozen with the split, are laid over the original's "
    "target (F2) and scored with the rest."
)
#: The share of scored test bins on which a model calling "always present"
#: makes it effectively the `always` baseline, and the paired F1 - always
#: interval's upper end below which it is too (F6).
ALWAYS_AGREEMENT = 0.99
ALWAYS_MARGIN = 0.01
SAWTOOTH_NOTE = (
    "The target is the table of the ece_sawtooth v3 detector (ECE and SXR "
    "crashes; D56 as amended): the model is a distillation of that detector, and "
    "it is scored only against it."
)


def score(prob, states, threshold: float, observed=None) -> dict:
    """`cells` `[tp, fp, fn, tn]` over the bins ABSENT or PRESENT_T and observed
    (`observed`, and a finite `prob`), a bin said present when `prob` reaches
    `threshold`; and their precision, recall and F1 (None when undefined)."""
    cells = frames_train.bin_cells(prob, states, threshold, observed)
    return {"cells": [int(c) for c in cells]} | {
        name: _plain(metric(cells)) for name, metric in METRICS.items()
    }


def _plain(x) -> float | None:
    x = float(x)
    return x if np.isfinite(x) else None


def swapped(cells) -> np.ndarray:
    """L's cells from H's (D38): `[tn, fn, fp, tp]`."""
    cells = np.asarray(cells)
    return cells[..., [3, 2, 1, 0]]


def interval(cells, metric, *, seed: int = SEED) -> dict | None:
    """The metric of the pooled `cells` `(n_shots, 4)` with its 95 % shot-bootstrap
    interval (2000 replicates); None with no shot."""
    cells = np.asarray(cells, dtype=np.float64).reshape(-1, 4)
    if not len(cells):
        return None
    strata, weights = ["all"] * len(cells), np.ones(len(cells))
    return stats.estimate(
        cells, strata, weights, metric, n=stats.N_REPLICATES, seed=seed
    ).as_json()


def paired(a, b, metric, *, seed: int = SEED) -> dict | None:
    """`metric(a) - metric(b)` on the same resampled shots; None with no shot."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if not len(a):
        return None
    strata, weights = ["all"] * len(a), np.ones(len(a))
    return stats.difference(
        a, b, strata, weights, metric, n=stats.N_REPLICATES, seed=seed
    ).as_json()


class Shot:
    """One shot's features as scoring reads them: its bins, their states, and
    whether each bin's frames are all observed."""

    def __init__(self, shot: int, z) -> None:
        per = round(len(z["observed"]) / len(z["bins"]))
        self.shot = int(shot)
        self.x = z["x"]
        self.bins = z["bins"].astype(np.float64)
        self.states = z["states"]
        self.observed = z["observed"].reshape(-1, per).all(axis=1)
        self.window = tuple(int(v) for v in z["window"])


def read_shots(paths: Paths, method: str, shots) -> tuple[list[Shot], dict]:
    """The shots with features, and the others with their reasons."""
    gone, found, left = prepare.dropped(paths, method), [], {}
    for shot in sorted(int(s) for s in shots):
        path = features_dir(paths, method, VERSION) / f"{shot}.npz"
        if not path.is_file():
            left[str(shot)] = gone.get(shot, "not prepared")
            continue
        with np.load(path) as z:
            found.append(Shot(shot, z))
    return found, left


def _features(paths: Paths, spec: EventSpec, shots) -> list[Shot]:
    if shots and isinstance(next(iter(shots)), Shot):
        return list(shots)
    return read_shots(paths, spec.method, shots)[0]


def _seen(bins, bin_ms: float, measured) -> np.ndarray:
    """Whether each bin lies wholly inside one measured stretch."""
    ends = bins + bin_ms
    seen = np.zeros(len(bins), bool)
    for lo, hi in measured:
        seen |= (bins >= lo) & (ends <= hi)
    return seen


def span_bins(bins, bin_ms: float, found, pool: str) -> np.ndarray:
    """A span baseline's decision per bin: 1 where its present spans touch the
    bin ("max") or cover half of it ("mean"), 0 elsewhere, NaN where it did not
    measure the whole bin."""
    bins = np.asarray(bins, dtype=np.float64)
    cover = np.zeros(len(bins))
    for a, b, state in found.spans:
        if state == PRESENT:
            cover += np.clip(
                np.minimum(bins + bin_ms, b) - np.maximum(bins, a), 0, None
            )
    said = cover > 0 if pool == "max" else cover >= bin_ms / 2
    return np.where(_seen(bins, bin_ms, found.measured), said.astype(float), np.nan)


def onset_bins(bins, bin_ms: float, onsets) -> np.ndarray:
    """`elm_onsets` per bin: 1 where the bin holds an onset, 0 elsewhere, NaN where
    the clock's channel did not measure the whole bin."""
    bins = np.asarray(bins, dtype=np.float64)
    t = np.sort(np.asarray(onsets.times_ms, dtype=np.float64))
    held = np.searchsorted(t, bins + bin_ms, "left") > np.searchsorted(t, bins, "left")
    seen = _seen(bins, bin_ms, onsets.found.measured)
    return np.where(seen, held.astype(float), np.nan)


def _elm_onsets():
    found = getattr(spans, "elm_onsets", None)
    if found is None:
        raise RuntimeError(
            "labeler.events.spans has no elm_onsets: the ELM baselines need "
            "round-three-b's (dbdfb8f) merged first"
        )
    return found


def baselines(
    paths: Paths, spec: EventSpec, shots
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, dict[str, str]]]:
    """`(decisions, left_out)`: each of the spec's baselines' decisions per bin, by
    shot, over each shot's features' bins (module docstring), and by baseline the
    shots it could not run on, with their reasons. `shots` are shot numbers
    (their features are read) or `Shot`s."""
    shots = _features(paths, spec, shots)
    out: dict = {name: {} for name in spec.baselines}
    left: dict = {name: {} for name in spec.baselines}
    elm = {"elm_onsets", "elm_clock"} & set(spec.baselines)
    elm_onsets = _elm_onsets() if elm else None
    for shot in shots:
        if "always" in out:
            out["always"][shot.shot] = np.ones(len(shot.bins))
        if elm:
            try:
                onsets = elm_onsets(shot.shot, paths, shot.window)
            except spans.INPUT_MISSING as error:
                for name in elm:
                    left[name][str(shot.shot)] = f"{type(error).__name__}: {error}"
            else:
                if "elm_onsets" in out:
                    out["elm_onsets"][shot.shot] = onset_bins(
                        shot.bins, spec.bin_ms, onsets
                    )
                if "elm_clock" in out:
                    out["elm_clock"][shot.shot] = span_bins(
                        shot.bins, spec.bin_ms, onsets.found, "max"
                    )
        if "dalpha_lh" in out:
            try:
                found = spans.detect_hmode(shot.shot, paths, shot.window)
            except spans.INPUT_MISSING as error:
                left["dalpha_lh"][str(shot.shot)] = f"{type(error).__name__}: {error}"
            else:
                out["dalpha_lh"][shot.shot] = span_bins(
                    shot.bins, spec.bin_ms, found, "mean"
                )
    return out, left


def views(method: str) -> dict[str, bool]:
    """Each metric suffix and whether its cells are swapped: H and L for P(H)."""
    return {"(H)": False, "(L)": True} if method in LMODE else {"": False}


def _cells(said: dict, shots: list[Shot], threshold: float, mask=None):
    """Per-shot cells of the shots `said` has, over the bins the model sees and,
    with `mask` (another method's decisions), that one sees too."""
    rows, kept = [], []
    for shot in shots:
        if shot.shot not in said or (mask is not None and shot.shot not in mask):
            continue
        prob = np.array(said[shot.shot], dtype=np.float64)
        if mask is not None:
            prob[~np.isfinite(mask[shot.shot])] = np.nan
        rows.append(score(prob, shot.states, threshold, shot.observed)["cells"])
        kept.append(shot.shot)
    return np.asarray(rows, dtype=np.float64).reshape(-1, 4), kept


def table(method: str, said: dict, thresholds: dict, shots: list[Shot]) -> dict:
    """Every method's scores and intervals over `shots`, and the model's paired F1
    differences with each other method (`said`: decisions by method, then shot;
    the model's first)."""
    model = next(iter(said))
    out = {"scores": {}, "intervals": {}, "paired": {}, "shots": {}}
    for name, by_shot in said.items():
        cells, kept = _cells(by_shot, shots, thresholds[name])
        out["shots"][name] = kept
        total = cells.sum(axis=0)
        out["scores"][name] = {"shots": len(kept), "cells": [int(c) for c in total]}
        out["intervals"][name] = {}
        for suffix, swap in views(method).items():
            c = swapped(cells) if swap else cells
            for metric, fn in METRICS.items():
                value = _plain(fn(c.sum(axis=0))) if len(c) else None
                out["scores"][name][metric + suffix] = value
                out["intervals"][name][metric + suffix] = interval(c, fn)
        if name == model:
            continue
        ours, _ = _cells(said[model], shots, thresholds[model], mask=by_shot)
        theirs, _ = _cells(by_shot, shots, thresholds[name], mask=by_shot)
        for suffix, swap in views(method).items():
            a, b = (swapped(ours), swapped(theirs)) if swap else (ours, theirs)
            out["paired"][f"f1{suffix} - {name}"] = paired(a, b, stats.f1)
    if method in LMODE:
        cells, kept = _cells(said[model], shots, thresholds[model])
        cells = swapped(cells)
        name = LMODE[method]
        out["scores"][name] = {
            "shots": len(kept),
            "cells": [int(c) for c in cells.sum(0)],
        }
        out["scores"][name] |= {
            m: _plain(fn(cells.sum(axis=0))) if len(cells) else None
            for m, fn in METRICS.items()
        }
        out["intervals"][name] = {m: interval(cells, fn) for m, fn in METRICS.items()}
    return out


def quantity(record: dict, spec: EventSpec, name: str):
    """A bar quantity's value: the model's score, or `lo(...)` of a difference."""
    if name.startswith("lo(") and name.endswith(")"):
        found = record["paired"].get(name[3:-1])
        return None if found is None else found["low"]
    return record["scores"][spec.method].get(name)


def verdict(spec: EventSpec, record: dict) -> dict:
    """Each criterion of `spec.bar`, and "all"."""
    out = {}
    for criterion, conditions in spec.bar.items():
        out[criterion] = all(
            (value := quantity(record, spec, name)) is not None
            and OPS[op](value, bound)
            for name, op, bound in conditions
        )
    out["all"] = all(out.values())
    return out


def always_like(method: str, record: dict) -> dict:
    """Whether the model is effectively the `always` baseline (module docstring,
    F6): `value`, and `agreement`, the share of the scored test bins it calls
    present, and `high`, the upper end of the paired F1 - always interval
    (`key`), each with whether it `held`. An undefined number holds nothing."""
    tp, fp, fn, tn = record["scores"][method]["cells"]
    scored = tp + fp + fn + tn
    agreement = (tp + fp) / scored if scored else None
    key = f"f1{next(iter(views(method)))} - always"
    found = record["paired"].get(key)
    high = None if found is None else found.get("high")
    held = {
        "agreement": agreement is not None and agreement >= ALWAYS_AGREEMENT,
        "interval": high is not None and high < ALWAYS_MARGIN,
    }
    return {
        "value": any(held.values()),
        "agreement": agreement,
        "agreement_at_least": ALWAYS_AGREEMENT,
        "key": key,
        "high": high,
        "high_below": ALWAYS_MARGIN,
        "held": held,
    }


def model_probs(model, spec: EventSpec, shot: Shot) -> np.ndarray:
    """The model's P per bin of one shot, NaN where a frame is not observed."""
    per = frames_train.frames_per_bin(spec)
    prob = frames_train.bin_probs(model, shot.x, per, spec.pool).astype(np.float64)
    return np.where(shot.observed, prob, np.nan)


@dataclass(frozen=True)
class Years:
    """Campaign years by shot (`shot_years`): `known`, each shot's own, and
    `starts`, each year's first shot, the catalog's calendar."""

    known: dict[int, int]
    starts: dict[int, int]

    def of(self, shot) -> int | None:
        """The shot's own year, else the year the calendar puts it in (the last
        to start at or before it, `osti.shot_year`), else None."""
        shot = int(shot)
        if shot in self.known:
            return self.known[shot]
        return shot_year(shot, self.starts)


def shot_years(paths: Paths) -> Years:
    """The catalog's campaign years: a population shot's `year`
    (`population.csv`, which the coverage figure counts by), else a pool shot's
    own (`pool.csv`, from its run id, where the population's come from), and the
    calendar the literature links date shots by (`osti.year_starts` over the
    pool; the population's years' first shots without a pool). The corpus starts
    at 185601 (2021), so a shot before the calendar has no year here."""
    known: dict[int, int] = {}
    pool_path = paths.catalog / "pool.csv"
    pool = read_pool(pool_path) if pool_path.is_file() else None
    if pool is not None:
        fallback = pool.reasons.str.contains("session_fallback", regex=False)
        own = pool[pool.year.notna() & ~fallback]
        known |= dict(zip(own.shot.astype(int), own.year.astype(int), strict=True))
    path = paths.catalog / "population.csv"
    if path.is_file():
        frame = pd.read_csv(path, usecols=["shot", "year"])
        years = pd.to_numeric(frame.year, errors="coerce")
        keep = years.notna()
        known |= dict(
            zip(frame.shot[keep].astype(int), years[keep].astype(int), strict=True)
        )
    known = {int(k): int(v) for k, v in known.items()}
    if pool is not None:
        starts = year_starts(pool)
    else:
        starts = {}
        for shot, year in sorted(known.items()):
            starts.setdefault(year, shot)
    return Years(known, starts)


def year_counts(shots, years: Years) -> dict[str, int]:
    """The shots by campaign year (`Years.of`), in year order, "unknown" last."""
    found = Counter(years.of(s) for s in shots)
    out = {str(y): found[y] for y in sorted(y for y in found if y is not None)}
    if found[None]:
        out["unknown"] = found[None]
    return out


def _population_shots(paths: Paths) -> list[int]:
    """The population's non-blind shots (`spans.population`), which `apply` labels."""
    if not (paths.catalog / "population.csv").is_file():
        return []
    return [int(s) for s in spans.population(paths).shot]


def split_years(paths: Paths, method: str, years: Years) -> dict[str, dict]:
    """The split's shots by campaign year: each split's, the legacy shots' (outside
    the roster; their targets are the legacy tables') and the roster's."""
    frame = pd.read_csv(shots_file(paths, method, VERSION))
    out = {
        name: year_counts(frame.shot[frame.split == name], years)
        for name in ("train", "val", "test")
    }
    out["legacy"] = year_counts(frame.shot[frame.roster == 0], years)
    out["roster"] = year_counts(frame.shot[frame.roster == 1], years)
    return out


def labels_window(meta: dict) -> dict:
    """The "labels"-window shots' labelled bins outside the catalog's Ip window
    (Task 2.7's `labels_window`); not available when none has an Ip window."""
    found = dict(meta.get("labels_window") or {})
    if not found.get("with_ip_window"):
        return {
            "shots": int(found.get("shots", 0)),
            "outside_ip_window": "not available",
            "reason": "no labels-window shot has a window in catalog/ip.jsonl",
        }
    return found


def _original(paths: Paths, spec: EventSpec, shot: int):
    """The shot's original target, or None where there is none to read."""
    try:
        return targets.target_bins(paths, spec, shot)
    except (OSError, ValueError, KeyError):
        return None


def owner_flips(
    paths: Paths, spec: EventSpec, shots: list[Shot], owned, windows, saved
) -> tuple[dict, list[int]]:
    """`({"test": ..., "all": ...}, added)` (F14): what the owner's frozen saves
    `saved` changed in the targets (`targets.flips`, summed by
    `targets.flip_total`), each shot's original read again. `test` is over the
    observed bins of the scored test `shots` (a flip to 0 or 1 there is a
    scored bin, and one to uncertain left the score), `all` over every bin in
    the window (`windows`, the shots meta's) of each of the owner's shots in the
    split (`owned`); `added`, the scored test shots whose target is the owner's
    label alone."""
    bin_ms = spec.bin_ms

    def one(shot: int, k0: int, k1: int, mask=None):
        original = _original(paths, spec, shot)
        label = targets.label_bins(saved[shot], bin_ms)
        found = targets.flips(
            targets.on_bins(original, k0, k1, bin_ms),
            targets.on_bins(label, k0, k1, bin_ms),
            mask,
        )
        return found, original is None

    test, added = [], []
    for s in shots:
        if s.shot not in saved:
            continue
        k0 = round(float(s.bins[0]) / bin_ms) if len(s.bins) else 0
        found, alone = one(s.shot, k0, k0 + len(s.bins), s.observed)
        test.append(found)
        if alone:
            added.append(s.shot)
    every = []
    for shot in sorted(int(s) for s in owned):
        if shot in saved and str(shot) in windows:
            k0, k1 = targets.bin_range(windows[str(shot)][:2], bin_ms)
            every.append(one(shot, k0, k1)[0])
    found = {"test": targets.flip_total(test), "all": targets.flip_total(every)}
    return found, sorted(added)


def _check(blob: dict, spec: EventSpec, split: dict, split_bytes: bytes) -> None:
    sha = hashlib.sha256(split_bytes).hexdigest()
    if blob.get("split_sha256") != sha:
        raise ValueError(
            f"{spec.method}: the split changed since the model was trained"
        )
    test = {s for s, v in split.items() if v == "test"}
    trained = set(blob.get("shots", {}).get("train", [])) | set(
        blob.get("shots", {}).get("val", [])
    )
    if test & trained:
        raise ValueError(
            f"{spec.method}: test shots were trained on: {sorted(test & trained)}"
        )


def evaluate(paths: Paths, method: str, *, out: Path | None = None) -> dict:
    """Score the method's model once on its test shots (module docstring); the
    record written to `evaluation.json`."""
    spec = SPECS[method]
    out = model_dir(paths, method, VERSION) if out is None else Path(out)
    target = out / "evaluation.json"
    if target.exists() and not pilot_area(out, paths.runs):
        raise FileExistsError(f"{target}: the test shots are scored once")
    model_path = out / "model.pt"
    model, blob = frames_train.load(model_path)
    split_bytes = shots_file(paths, method, VERSION).read_bytes()
    split = prepare.split_shots(paths, method)
    _check(blob, spec, split, split_bytes)
    meta = json.loads(shots_meta_file(paths, method, VERSION).read_text())
    prepare.check_frozen(paths, method, meta)
    frame = pd.read_csv(shots_file(paths, method, VERSION))
    saved = {int(s) for s in frame.shot[frame.owner == 1]}
    frozen = labels.read_labels(owner_file(paths, method, VERSION))
    threshold = float(blob["threshold"])
    shots, left = read_shots(
        paths, method, [s for s, v in split.items() if v == "test"]
    )
    record = {
        "method": method,
        "version": VERSION,
        "tier": "suggestions",
        "model": {
            "path": str(model_path),
            "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "threshold": threshold,
            "threshold_rule": blob.get("threshold_rule", "f1"),
            "split_sha256": blob["split_sha256"],
        },
        "bin_ms": spec.bin_ms,
        "pool": spec.pool,
        "bar_criteria": {k: [list(c) for c in v] for k, v in spec.bar.items()},
        "replicates": stats.N_REPLICATES,
        "seed": SEED,
    }
    years = shot_years(paths)
    said = {method: {s.shot: model_probs(model, spec, s) for s in shots}}
    found, left_out = baselines(paths, spec, shots) if shots else ({}, {})
    for base in spec.baselines:
        said[base] = found.get(base, {})
    thresholds = {method: threshold} | dict.fromkeys(spec.baselines, 0.5)
    test = table(method, said, thresholds, shots) | {
        "left_out": left,
        "baseline_left_out": left_out,
        "bins": _bin_counts(shots),
        "years": year_counts([s.shot for s in shots], years),
    }
    record |= {k: test[k] for k in ("scores", "intervals", "paired")}
    record["bar"] = verdict(spec, record)
    record["always_like"] = always_like(method, record)
    record["effectively_always"] = record["always_like"]["value"]
    record["test"] = {
        k: test[k] for k in ("shots", "left_out", "baseline_left_out", "bins", "years")
    }
    flips, added = owner_flips(
        paths, spec, shots, saved, meta.get("windows", {}), frozen
    )
    record["owner"] = {
        "note": ELM_OWNER_NOTE if spec.method == "elm_frames" else OWNER_NOTE,
        "snapshot": meta.get("owner_snapshot"),
        "saved": meta.get("owner"),
        "test_shots": sorted(s for s in test["shots"][method] if s in saved),
        "added_test_shots": added,
        "flips": flips,
    }
    record["split_years"] = split_years(paths, method, years)
    record["population_years"] = year_counts(_population_shots(paths), years)
    record["labels_window"] = labels_window(meta)
    record["git_sha"] = git_sha()
    with atomic_path(target) as tmp:
        tmp.write_text(json.dumps(record, indent=1) + "\n")
    with atomic_path(out / "evaluation.md") as tmp:
        tmp.write_text(report_md(spec, record))
    return record


def _bin_counts(shots: list[Shot]) -> dict:
    scored = [s.observed & np.isin(s.states, (ABSENT, PRESENT_T)) for s in shots]
    present = [m & (s.states == PRESENT_T) for s, m in zip(shots, scored, strict=True)]
    absent = [m & (s.states == ABSENT) for s, m in zip(shots, scored, strict=True)]
    return {
        "scored": int(sum(m.sum() for m in scored)),
        "present": int(sum(m.sum() for m in present)),
        "absent": int(sum(m.sum() for m in absent)),
        "shots_with_present": int(sum(bool(m.any()) for m in present)),
        "shots_with_absent": int(sum(bool(m.any()) for m in absent)),
    }


def _fmt(e: dict | None) -> str:
    if e is None or e["value"] is None:
        return "n/a"
    lo, hi = e["low"], e["high"]
    band = "" if lo is None else f" [{lo:.3f}, {hi:.3f}]"
    return f"{e['value']:.3f}{band}"


def _score_table(method: str, part: dict) -> list[str]:
    names = [m + s for s in views(method) for m in METRICS]
    lines = [
        "| method | shots | " + " | ".join(names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for name, intervals in part["intervals"].items():
        if name in LMODE.values():
            continue
        shots = part["scores"][name]["shots"]
        cells = " | ".join(_fmt(intervals.get(n)) for n in names)
        lines.append(f"| {name} | {shots} | {cells} |")
    return lines


def _flips_said(c: dict) -> str:
    to_uncertain = c["absent_to_uncertain"] + c["present_to_uncertain"]
    unknown = c["unknown_to_absent"] + c["unknown_to_present"]
    return (
        f"0→1 {c['absent_to_present']}, 1→0 {c['present_to_absent']}, 0 or 1 to "
        f"uncertain {to_uncertain}, and unknown to 0 or 1 {unknown}, of the "
        f"{c['owned']} bins its label decides"
    )


def flips_line(flips: dict) -> str:
    """The md's line on what the owner's saves changed (F14)."""
    test, every = flips["test"], flips["all"]
    return (
        "The owner's flips, the original's state to the merged target's: on the "
        f"{test['bins']} observed bins of the {test['shots']} owner shots scored in "
        f"test, {_flips_said(test)}; over every bin in the windows of the owner's "
        f"{every['shots']} shots in the split, {_flips_said(every)}. A flip to 0 "
        "or 1 in test is a scored bin; one to uncertain left the score."
    )


def always_line(found: dict) -> str:
    """The md's one line on whether the model is effectively always (F6)."""
    agreement, high = found["agreement"], found["high"]
    parts = {
        "agreement": (
            "its calls equal always-present on "
            + ("no scored bin" if agreement is None else f"{agreement:.1%}")
            + f" of the scored test bins (effectively always from "
            f"{found['agreement_at_least']:.0%})"
        ),
        "interval": (
            f"the paired {found['key']} interval's upper end is "
            + ("undefined" if high is None else f"{high:.3f}")
            + f" (effectively always below {found['high_below']:g})"
        ),
    }
    held = [parts[k] for k, v in found["held"].items() if v]
    if held:
        return "Effectively always: yes, " + " and ".join(held) + "."
    return "Effectively always: no; " + "; ".join(parts.values()) + "."


def _years_line(years: dict) -> str:
    said = ", ".join(f"{k}: {v}" for k, v in years.items() if v)
    return said or "none"


def _span(years: dict) -> str:
    """The years' span, `2021-2023`, from counts by year."""
    known = sorted(int(k) for k, v in years.items() if k != "unknown" and v)
    if not any(years.values()):
        return "no shots"
    if not known:
        return "no known year"
    return str(known[0]) if known[0] == known[-1] else f"{known[0]}-{known[-1]}"


def years_md(record: dict) -> list[str]:
    """The md's year lines, from the counts alone (`split_years`)."""
    test, split = record["test"]["years"], record["split_years"]
    population = record["population_years"]
    lines = [
        (
            "Campaign years (the population's own, else the catalog's calendar, "
            "`shot_years`). The scored test shots: "
            f"{_span(test)} ({_years_line(test)})."
        ),
        "",
        "| shots | years | by year |",
        "|---|---|---|",
    ]
    names = {
        "train": "train",
        "val": "val",
        "test": "test (the split)",
        "legacy": "legacy (outside the roster)",
        "roster": "roster",
    }
    for key, name in names.items():
        counts = split[key]
        lines.append(f"| {name} | {_span(counts)} | {_years_line(counts)} |")
    lines.append(
        f"| the non-blind population it is applied to | {_span(population)} | "
        f"{_years_line(population)} |"
    )
    covered = {k for k, v in test.items() if v and k != "unknown"}
    missing = [
        k for k, v in population.items() if v and k != "unknown" and k not in covered
    ]
    if not covered:
        said = "No scored test shot has a known year: the test says nothing by year."
    elif missing:
        said = (
            f"No scored test shot is from {', '.join(missing)}, which hold "
            f"{sum(population[k] for k in missing)} of the population's shots: the "
            "test does not show how the model does on them."
        )
    else:
        said = "Every population year has a scored test shot."
    if test.get("unknown"):
        said += f" {test['unknown']} scored test shots have no known year."
    return [*lines, "", said]


def report_md(spec: EventSpec, record: dict) -> str:
    """`evaluation.md`: the test's tables, the bar, whether the model is
    effectively always, the owner's shots in test and what the test cannot
    show."""
    method, test = spec.method, record["test"]
    n = test["bins"]
    lines = [
        f"# {method} on the test shots",
        "",
        (
            f"{len(test['shots'][method])} test shots, {n['scored']} scored "
            f"{spec.bin_ms:g} ms bins ({n['present']} present, {n['absent']} "
            f"absent). Threshold {record['model']['threshold']} (chosen on the val "
            f"shots); bins pooled by their frames' {spec.pool}. 95 % shot-bootstrap "
            f"intervals, {record['replicates']} replicates, seed {record['seed']}."
        ),
        "",
        *_score_table(method, record),
        "",
        "| paired difference | value [95 %] |",
        "|---|---|",
        *(f"| {k} | {_fmt(v)} |" for k, v in record["paired"].items()),
        "",
    ]
    if method in LMODE:
        name = LMODE[method]
        lines += [
            f"{name} (L = labelled and not H, D38): "
            + ", ".join(f"{m} {_fmt(record['intervals'][name][m])}" for m in METRICS)
            + ".",
            "",
        ]
    said = ", ".join(
        f"{k} {'pass' if v else 'FAIL'}" for k, v in record["bar"].items() if k != "all"
    )
    lines += [
        (
            f"The bar: {said}; all {'pass' if record['bar']['all'] else 'FAIL'}. "
            "Tier: suggestions, whatever the bar."
        ),
        "",
        always_line(record["always_like"]),
        "",
    ]
    gone = Counter(test["left_out"].values())
    if gone:
        why = ", ".join(f"{k} ({v})" for k, v in sorted(gone.items()))
        lines.append(
            f"- {len(test['left_out'])} test shots left out for missing features: "
            f"{why}; the shots are in evaluation.json's test.left_out."
        )
    else:
        lines.append("- No test shot was left out for missing features.")
    for base, left in test["baseline_left_out"].items():
        ran = len(test["shots"].get(base, []))
        gated = "reported, never gated" if base == "dalpha_lh" else "compared"
        lines.append(
            f"- {base} ({gated}): ran on {ran} test shots, left out of {len(left)}."
        )
    owner = record["owner"]
    lines += [
        "",
        "## The owner's saves",
        "",
        (
            f"{len(owner['test_shots'])} of the {len(test['shots'][method])} scored "
            "test shots carry an owner label, "
            f"{len(owner['added_test_shots'])} of them the owner's alone (added, "
            "not overriding). " + owner["note"]
        ),
        "",
        flips_line(owner["flips"]),
        "",
        "## What this test can and cannot show",
        "",
    ]
    if method in LMODE:
        low_l = record["intervals"][method]["f1(L)"]
        width = (
            "undefined"
            if low_l is None or low_l["low"] is None
            else f"{low_l['high'] - low_l['low']:.2f} wide"
        )
        lines += [
            (
                f"D44: the {len(test['shots'][method])} test shots hold "
                f"{n['present']} H bins and {n['absent']} L bins, the L bins on "
                f"{n['shots_with_absent']} shots. The test can show whether the "
                "model finds H where Jalal Butt's table has it. It cannot rank "
                f"methods on L-mode: the F1(L) interval is {width}, and a shot "
                "bootstrap over so few L shots is coarse."
            ),
            "",
        ]
    if method == "sawtooth_frames":
        lines += [SAWTOOTH_NOTE, ""]
    lines += [*years_md(record), ""]
    window = record["labels_window"]
    outside = window["outside_ip_window"]
    lines += ["## Windows", ""]
    if outside == "not available":
        lines.append(
            f"{window['shots']} shots' windows are their target's hull; how many of "
            f"their labelled bins lie outside the plasma is not available: "
            f"{window['reason']}."
        )
    else:
        lines.append(
            f"{window['shots']} shots' windows are their target's hull; on the "
            f"{window['with_ip_window']} with a catalog Ip window, {outside} of "
            f"{window['labelled_bins']} labelled bins lie outside it."
        )
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    lines += ["", f"Written {stamp} at {record['git_sha']}.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--method", required=True, choices=list(SPECS))
    p.add_argument(
        "--pilot",
        action="store_true",
        help="the pilot's model, runs/frames/pilot/<method>; may be rescored",
    )
    args = p.parse_args(argv)
    paths = Paths.from_env()
    out = frames_train.pilot_dir(paths, args.method) if args.pilot else None
    try:
        record = evaluate(paths, args.method, out=out)
    except (OSError, ValueError, RuntimeError) as error:
        p.error(str(error))
    line = {"method": args.method, "bar": record["bar"], "tier": record["tier"]}
    print(json.dumps(line), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
