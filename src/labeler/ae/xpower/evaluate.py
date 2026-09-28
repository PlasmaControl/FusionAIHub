"""Choose the AE model on the validation shots, then score it on the test shots.

    python -m labeler.ae.xpower.evaluate --choose [--models DIR] [--limit N]
    python -m labeler.ae.xpower.evaluate [--test] [--models DIR] [--limit N]

The first scores every trained candidate of the version (`train.candidates`)
on the validation shots and writes `chosen.json`; the second scores the chosen
one, once, on the test shots and writes `evaluation.json` and `evaluation.md`,
all in `--models` (default `$LABELER_ROOT/models/ae_xpower/<version>`). A
version chosen by cross-validation (v2, `labeler.ae.xpower.cv`) is not chosen
here: `--choose` only checks that `chosen.json` is the one `train --from-cv`
wrote from `cv/choice.json`, and the test checks the model against both.
v2's test also reports v1's test shots (`SUBSET_OF`, the 58 of v1's chosen
model's `split.csv`, checked before scoring to be v2 test shots) as a second
table and a `v1_subset` block; the bar is judged on the whole split only.
The test is never scored in full (`--limit 0`) under `runs/`, where it could be
repeated, and a pilot there scores 20 test shots at most; `--version` must be
the models directory's name and the version its checkpoints record.

**Frames.** The 10 ms frames of 0-2 s that the owner called present or absent,
that TokEye's record covers, that the model's rows cover and that lie inside
the source table's window. 0-2 s is all TokEye and the earlier detector ever
saw, so every method is scored on the same frames.

**Methods.** `ae_xpower`, the chosen candidate at its validation threshold;
`seldnet`, the earlier detector (`ae_seldnet_threeway_sce.pt`), a frame present
when half its columns are over 0.5; `tokeye`, TokEye's coherent mask in
80-250 kHz on two chords for half the frame; `source`, the table the owner
started from (`format/`); `uci`, the UCI annotation as TokEye stores it,
unshifted; `always`, every frame present.

**Scores.** Frame precision, recall and F1 pooled over shots, with 95 %
shot-bootstrap intervals (`labeler.scoring.stats`, 2000 replicates, seed
20260923); the false-positive rate on MHD frames (the owner says absent and
TokEye sees a 0-60 kHz line, `data.mhd_frames`) and on the other absent frames;
paired differences against `seldnet` and `always`.

**The bar** (`verdict`): A1 F1 >= 0.90, precision and recall >= 0.75, and the
F1 difference from `seldnet` has a lower bound >= -0.03; A2 the MHD
false-positive rate is <= 0.05 and its difference from `seldnet`'s has an upper
bound < 0; A3 the F1 difference from `always` has a lower bound > 0. Passing or
not, what the model writes is a suggestion (v1 spec §3): the test shots were
reviewed from a starting table, not blind, and are 2017-2019 shots where the
extension runs on 2024-2025 ones (1,103 from 2024, 809 from 2025, and 2 from
2022 of the 1,914 CO2-eligible shots).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from ...config import Paths, atomic_path, git_sha, sha256_of
from ...events.catalog.states import ABSENT, PRESENT
from ...events.review import labels
from ...scoring import stats
from .. import model as seldnet_model
from . import (
    CV_VERSIONS,
    EVENT,
    LABEL_SNAPSHOTS,
    VERSION,
    check_bound,
    check_full,
    check_limit,
    check_snapshot,
    event_dir,
    model_dir,
    pilot_area,
    seldnet_dir,
    tokeye_masks,
)
from .data import (
    MIN_FRACTION,
    SEED,
    clean_path,
    frame_share,
    seldnet_split,
    store_rows,
    targets,
    tokeye_frames,
    window_frames,
)
from .train import candidates, load, probabilities, read_split

EVAL_FRAMES = (0, 200)  # 0-2 s
F1_MARGIN = 0.02
SELDNET_FILE = "ae_seldnet_threeway_sce.pt"
BAR = {
    "f1": 0.90,
    "precision": 0.75,
    "recall": 0.75,
    "f1_vs_seldnet_low": -0.03,
    "mhd_fp_rate": 0.05,
}
SUBSET_OF = {"v2": "v1"}  # the earlier version whose test shots are reported
METHODS = ("ae_xpower", "seldnet", "tokeye", "source", "uci", "always")


@dataclass
class ShotFrames:
    """One shot's scored frames and what each method said of them."""

    shot: int
    owner: np.ndarray  # (n,) frame states
    mhd: np.ndarray  # (n,) bool
    scored: np.ndarray  # (n,) bool
    said: dict[str, np.ndarray] = field(default_factory=dict)
    prob: np.ndarray | None = None  # model P(AE), retained for validation sweeps


def fp_rate(totals):
    """False-positive rate, `fp / (fp + tn)`, of `[tp, fp, fn, tn]` totals."""
    t = np.asarray(totals, dtype=float)
    den = t[..., 1] + t[..., 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, t[..., 1] / np.where(den > 0, den, 1.0), np.nan)


def mhd_absent(frames: ShotFrames) -> np.ndarray:
    return frames.mhd & (frames.owner == ABSENT)


def other_absent(frames: ShotFrames) -> np.ndarray:
    return ~frames.mhd & (frames.owner == ABSENT)


def cells(
    shots: Sequence[ShotFrames],
    method: str,
    where: Callable[[ShotFrames], np.ndarray] | None = None,
) -> np.ndarray:
    """`(n_shots, 4)` `[tp, fp, fn, tn]` of `method` over each shot's scored frames."""
    out = []
    for f in shots:
        mask = f.scored & (where(f) if where else True)
        truth, said = f.owner == PRESENT, f.said[method]
        out.append(
            [
                np.sum(mask & truth & said),
                np.sum(mask & ~truth & said),
                np.sum(mask & truth & ~said),
                np.sum(mask & ~truth & ~said),
            ]
        )
    return np.asarray(out, dtype=float).reshape(-1, 4)


def load_seldnet(paths: Paths):
    blob = torch.load(
        seldnet_dir(paths) / SELDNET_FILE, map_location="cpu", weights_only=False
    )
    net = seldnet_model.AeSeldNet(
        seldnet_model.AeSeldNetConfig.from_dict(blob["config"])
    )
    net.load_state_dict(blob["state_dict"])
    return net.eval()


def seldnet_said(net, spec_path, t_ms, first: int, n: int) -> np.ndarray:
    """The earlier detector's frames: half the frame's columns over 0.5."""
    with np.load(spec_path) as z:
        spec = np.ascontiguousarray(z["spec"].transpose(0, 2, 1))
    if spec.shape[1] != len(t_ms):
        raise ValueError(
            f"{spec_path}: {spec.shape[1]} columns, TokEye has {len(t_ms)}"
        )
    with torch.no_grad():
        prob = torch.sigmoid(net(torch.from_numpy(spec).float()[None])[0, :, 0]).numpy()
    return frame_share(t_ms, prob >= 0.5, first, n) >= 0.5


def shot_frames(
    shot: int,
    *,
    paths: Paths,
    label,
    model,
    blob: dict,
    source=None,
    seldnet=None,
    spec_path=None,
) -> ShotFrames:
    """Frames 0-2 s of one shot, and every method the arguments allow."""
    first, n = EVAL_FRAMES
    rows = store_rows(paths.spectrogram_file(EVENT, shot))
    prob, observed = probabilities(model, rows, first, n, band=blob["band_khz"])
    owner = targets(label, first, n)
    path = clean_path(tokeye_masks(paths), shot)
    if path is None:
        raise FileNotFoundError(f"no TokEye mask for {shot}")
    tk = tokeye_frames(path, first, n)
    scored = np.isin(owner, (ABSENT, PRESENT)) & tk["covered"] & observed
    said = {
        "ae_xpower": prob >= blob["threshold"],
        "tokeye": tk["ae"] >= MIN_FRACTION,
        "uci": tk["ann"] >= MIN_FRACTION,
        "always": np.ones(n, dtype=bool),
    }
    if source is not None:
        src = targets(source, first, n)
        said["source"] = src == PRESENT
        scored &= src >= 0
    if seldnet is not None:
        with np.load(path) as z:
            t_ms = np.asarray(z["t_ms"], dtype=np.float64)
        said["seldnet"] = seldnet_said(seldnet, spec_path, t_ms, first, n)
    mhd = tk["covered"] & (tk["low"] >= MIN_FRACTION)
    return ShotFrames(int(shot), owner, mhd, scored, said, prob)


def window_cells(shot: int, *, paths: Paths, label, model, blob: dict) -> np.ndarray:
    """`[tp, fp, fn, tn]` of the model over the owner's whole window, not only 0-2 s."""
    first, n = window_frames(label.window)
    rows = store_rows(paths.spectrogram_file(EVENT, shot))
    prob, observed = probabilities(model, rows, first, n, band=blob["band_khz"])
    owner = targets(label, first, n)
    scored = observed & np.isin(owner, (ABSENT, PRESENT))
    frames = ShotFrames(
        shot, owner, np.zeros(n, bool), scored, {"m": prob >= blob["threshold"]}
    )
    return cells([frames], "m")[0]


def _estimate(c, metric) -> dict:
    strata, weights = ["all"] * len(c), np.ones(len(c))
    return stats.estimate(
        c, strata, weights, metric, n=stats.N_REPLICATES, seed=SEED
    ).as_json()


def _difference(a, b, metric) -> dict:
    strata, weights = ["all"] * len(a), np.ones(len(a))
    return stats.difference(
        a, b, strata, weights, metric, n=stats.N_REPLICATES, seed=SEED
    ).as_json()


def score(shots: Sequence[ShotFrames], methods: Sequence[str]) -> dict:
    """Every method's frame scores and the model's paired differences."""
    out = {"methods": {}, "differences": {}}
    for m in methods:
        c = cells(shots, m)
        out["methods"][m] = {
            "precision": _estimate(c, stats.precision),
            "recall": _estimate(c, stats.recall),
            "f1": _estimate(c, stats.f1),
            "fp_rate_mhd": _estimate(cells(shots, m, mhd_absent), fp_rate),
            "fp_rate_other": _estimate(cells(shots, m, other_absent), fp_rate),
        }
    ours = cells(shots, "ae_xpower")
    for other in ("seldnet", "always"):
        if other in methods:
            out["differences"][f"f1_minus_{other}"] = _difference(
                ours, cells(shots, other), stats.f1
            )
    if "seldnet" in methods:
        out["differences"]["mhd_fp_minus_seldnet"] = _difference(
            cells(shots, "ae_xpower", mhd_absent),
            cells(shots, "seldnet", mhd_absent),
            fp_rate,
        )
    out["frames"] = {
        "shots": len(shots),
        "scored": int(sum(f.scored.sum() for f in shots)),
        "present": int(sum((f.scored & (f.owner == PRESENT)).sum() for f in shots)),
        "mhd_absent": int(sum((f.scored & mhd_absent(f)).sum() for f in shots)),
        "shots_with_mhd_absent": int(
            sum(bool((f.scored & mhd_absent(f)).any()) for f in shots)
        ),
    }
    return out


def _at_least(x, bound) -> bool:
    return x is not None and x >= bound


def verdict(scores: dict) -> dict:
    """The bar, A1-A3, from `score` output that includes `seldnet` and `always`."""
    ours, diff = scores["methods"]["ae_xpower"], scores["differences"]
    a1 = (
        _at_least(ours["f1"]["value"], BAR["f1"])
        and _at_least(ours["precision"]["value"], BAR["precision"])
        and _at_least(ours["recall"]["value"], BAR["recall"])
        and _at_least(diff["f1_minus_seldnet"]["low"], BAR["f1_vs_seldnet_low"])
    )
    mhd, gap = ours["fp_rate_mhd"]["value"], diff["mhd_fp_minus_seldnet"]["high"]
    a2 = mhd is not None and mhd <= BAR["mhd_fp_rate"] and gap is not None and gap < 0
    low = diff["f1_minus_always"]["low"]
    a3 = low is not None and low > 0
    return {
        "A1": bool(a1),
        "A2": bool(a2),
        "A3": bool(a3),
        "all": bool(a1 and a2 and a3),
    }


def choose(results: dict[str, dict], version: str = VERSION) -> tuple[str, str]:
    """The candidate with the lowest MHD false-positive rate among those within
    `F1_MARGIN` of the best validation F1; ties go to the earlier-listed one."""
    order = [c for c in candidates(version) if c in results]
    if not order:
        raise ValueError("no trained candidate to choose from")

    def f1(c):
        v = results[c]["f1"]["value"]
        return -1.0 if v is None else v

    best = max(f1(c) for c in order)
    near = [c for c in order if f1(c) >= best - F1_MARGIN]

    def mhd(c):
        v = results[c]["fp_rate_mhd"]["value"]
        return 1.0 if v is None else v

    picked = min(near, key=lambda c: (mhd(c), order.index(c)))
    why = (
        f"validation F1 {f1(picked):.3f} (best {best:.3f}, margin {F1_MARGIN}); "
        f"lowest MHD false-positive rate {mhd(picked):.3f} of {', '.join(near)}"
    )
    return picked, why


def _fmt(e: dict) -> str:
    if e["value"] is None:
        return "n/a"
    lo, hi = e["low"], e["high"]
    band = "" if lo is None else f" [{lo:.3f}, {hi:.3f}]"
    return f"{e['value']:.3f}{band}"


def report_md(scores: dict, bar: dict, meta: dict, subset: dict | None = None) -> str:
    """`evaluation.md`: the table the paper's AE score figure and table read."""
    n = scores["frames"]
    summary = (
        f"{n['shots']} shots, {n['scored']} frames of 0-2 s ({n['present']} present), "
        f"{n['mhd_absent']} MHD frames the owner called absent on "
        f"{n['shots_with_mhd_absent']} shots. Threshold {meta['threshold']}, band "
        f"{meta['band_khz']} kHz. 95 % shot-bootstrap intervals."
    )
    lines = [
        f"# AE model {meta['candidate']} on the test shots",
        "",
        summary,
        "",
        *_tables(scores),
    ]
    said = {k: "pass" if bar[k] else "FAIL" for k in ("A1", "A2", "A3")}
    verdict_line = (
        f"The bar: A1 {said['A1']}, A2 {said['A2']}, A3 {said['A3']}. "
        "Tier: suggestions."
    )
    lines += ["", verdict_line, ""]
    if subset is not None:
        lines += _subset_md(subset)
    return "\n".join(lines)


def _tables(scores: dict) -> list[str]:
    lines = [
        (
            "| method | precision | recall | F1 | FP rate, MHD frames | "
            "FP rate, other absent |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for m, s in scores["methods"].items():
        lines.append(
            f"| {m} | {_fmt(s['precision'])} | {_fmt(s['recall'])} | {_fmt(s['f1'])} | "
            f"{_fmt(s['fp_rate_mhd'])} | {_fmt(s['fp_rate_other'])} |"
        )
    lines += ["", "| paired difference | value [95 %] |", "|---|---|"]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in scores["differences"].items()]
    if "window" in scores:
        lines += [
            "",
            f"Over the owner's whole windows: F1 {_fmt(scores['window']['f1'])}.",
        ]
    return lines


def _subset_md(subset: dict) -> list[str]:
    """The second table: the same model on the earlier version's test shots."""
    name = subset["version"]
    lines = [f"## {name}'s test shots", ""]
    if "frames" not in subset:
        return [*lines, f"None of {name}'s test shots were scored.", ""]
    n = subset["frames"]
    lines += [
        (
            f"The same model and threshold on the {n['shots']} of {name}'s test "
            f"shots scored here ({len(subset['shots'])} in {name}'s split, its "
            f"model {subset['candidate']}): {n['scored']} frames "
            f"({n['present']} present), {n['mhd_absent']} MHD frames absent. "
            "Reported beside the whole split; not judged against the bar."
        ),
        "",
        *_tables(subset),
        "",
    ]
    return lines


def _reviewed(split: dict[int, str], which: str, limit: int) -> list[int]:
    shots = sorted(s for s, v in split.items() if v == which)
    return shots[:limit] if limit else shots


def run_choose(
    paths: Paths, models: Path, limit: int = 0, *, version: str = VERSION
) -> dict:
    check_limit(paths, models, limit)
    if version in CV_VERSIONS:
        # Chosen by `cv`; here the choice is only checked, and nothing written.
        check_bound(version, models)
        return cv_chosen(models, version)
    evaluation = models / "evaluation.json"
    if evaluation.exists() and not pilot_area(models, paths.runs):
        raise FileExistsError(
            f"{evaluation}: the version is evaluated; a new choice is a new version"
        )
    check_bound(version, models)
    results = {}
    for name in candidates(version):
        file = models / name / "model.pt"
        if not file.exists():
            continue
        model, blob = load(file)
        check_bound(version, models, blob, file)
        saved = labels.read_saved(file.parent)  # the labels it was trained on
        split = read_split(models / name / "split.csv")
        shots = [
            shot_frames(s, paths=paths, label=saved[s], model=model, blob=blob)
            for s in _reviewed(split, "val", limit)
        ]
        c = cells(shots, "ae_xpower")
        results[name] = {
            "f1": _estimate(c, stats.f1),
            "fp_rate_mhd": _estimate(cells(shots, "ae_xpower", mhd_absent), fp_rate),
            "shots": len(shots),
            "threshold": blob["threshold"],
        }
    picked, why = choose(results, version)
    record = {
        "candidate": picked,
        "version": version,
        "why": why,
        "validation": results,
        "git_sha": git_sha(),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with atomic_path(models / "chosen.json") as tmp:
        tmp.write_text(json.dumps(record, indent=1) + "\n")
    return record


def cv_chosen(models: Path, version: str) -> dict:
    """A cross-validated version's `chosen.json`, checked against `cv/choice.json`."""
    chosen_file, choice_file = models / "chosen.json", models / "cv" / "choice.json"
    chosen = json.loads(chosen_file.read_text())
    choice = json.loads(choice_file.read_bytes())
    wanted = {
        "version": version,
        "choice_sha256": sha256_of(choice_file),
        "candidate": choice["candidate"],
        "threshold": choice["threshold"],
        "labels_sha256": LABEL_SNAPSHOTS[version],
    }
    for key, value in wanted.items():
        if chosen.get(key) != value:
            raise ValueError(f"{chosen_file}: {key} differs from {choice_file}")
    return chosen


def check_cv_model(blob: dict, chosen: dict, labels_copy: bytes, file: Path) -> dict:
    """The final model of a cross-validated version is the one `chosen.json`
    names: its choice, threshold and label snapshot; the meta it adds."""
    version = chosen["version"]
    expected = LABEL_SNAPSHOTS[version]
    check_snapshot(hashlib.sha256(labels_copy).hexdigest(), version, file.parent)
    wanted = {
        "candidate": chosen["candidate"],
        "threshold": chosen["threshold"],
        "choice_sha256": chosen["choice_sha256"],
        "labels_sha256": expected,
        "snapshot_sha256": expected,
        "from_cv": True,
    }
    for key, value in wanted.items():
        if blob.get(key) != value:
            raise ValueError(f"{file}: {key} differs from chosen.json or the snapshot")
    return {
        "choice_sha256": chosen["choice_sha256"],
        "snapshot_sha256": expected,
        "fixed_epochs": blob.get("fixed_epochs"),
        "cv_branch": blob.get("cv_branch"),
    }


def earlier_test(paths: Paths, version: str, test: set[int]) -> dict | None:
    """The earlier version's test shots (`SUBSET_OF`), from its chosen model's
    split, refused unless every one is in this version's test split `test`."""
    earlier = SUBSET_OF.get(version)
    if earlier is None:
        return None
    models = model_dir(paths, earlier)
    chosen_bytes = (models / "chosen.json").read_bytes()
    name = json.loads(chosen_bytes)["candidate"]
    split_file = models / name / "split.csv"
    data = split_file.read_bytes()
    shots = sorted(
        s for s, v in read_split(split_file, data=data).items() if v == "test"
    )
    outside = sorted(set(shots) - test)
    if not shots or outside:
        raise ValueError(
            f"{split_file}: {earlier}'s test shots must be {version} test shots; "
            f"not: {', '.join(map(str, outside)) or 'none listed'}"
        )
    return {
        "version": earlier,
        "candidate": name,
        "shots": shots,
        "chosen_sha256": hashlib.sha256(chosen_bytes).hexdigest(),
        "split_sha256": hashlib.sha256(data).hexdigest(),
    }


def chosen_model(models: Path) -> Path:
    """The chosen candidate's `model.pt`, from `chosen.json`."""
    name = json.loads((models / "chosen.json").read_text())["candidate"]
    return models / name / "model.pt"


def run_test(
    paths: Paths, models: Path, limit: int = 0, *, version: str = VERSION
) -> dict:
    check_limit(paths, models, limit)
    check_full(paths, models, limit)
    evaluation = models / "evaluation.json"
    if evaluation.exists() and not pilot_area(models, paths.runs):
        raise FileExistsError(
            f"{evaluation}: the test shots are scored once; a retry is a new version"
        )
    check_bound(version, models)
    chosen_bytes = (models / "chosen.json").read_bytes()
    candidate = json.loads(chosen_bytes)["candidate"]
    file = models / candidate / "model.pt"
    # Load only these snapshots, so the hashes describe the bytes actually scored.
    snapshots = {
        "model.pt": file.read_bytes(),
        "split.csv": (file.parent / "split.csv").read_bytes(),
        "review/labels.csv": labels.labels_path(file.parent).read_bytes(),
    }
    with TemporaryDirectory(prefix="ae-evaluation-") as directory:
        frozen = Path(directory)
        for name, data in snapshots.items():
            target = frozen / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        saved = labels.read_saved(frozen)
        model, blob = load(frozen / "model.pt")
        split = read_split(frozen / "split.csv")
    check_bound(version, models, blob, file)
    cv_meta = {}
    if version in CV_VERSIONS:
        cv_meta = check_cv_model(
            blob, cv_chosen(models, version), snapshots["review/labels.csv"], file
        )
    subset = earlier_test(paths, version, {s for s, v in split.items() if v == "test"})
    source = labels.read_source(event_dir(paths))
    seldnet = load_seldnet(paths)
    splits = seldnet_split(tokeye_masks(paths))
    shots, windows = [], []
    for s in _reviewed(split, "test", limit):
        spec = paths.root / "ae" / "dataset" / f"{s}_{splits[s]}.npz"
        shots.append(
            shot_frames(
                s,
                paths=paths,
                label=saved[s],
                model=model,
                blob=blob,
                source=source.get(s),
                seldnet=seldnet,
                spec_path=spec,
            )
        )
        windows.append(
            window_cells(s, paths=paths, label=saved[s], model=model, blob=blob)
        )
    if any("source" not in f.said for f in shots):
        raise ValueError("a test shot has no source-table label")
    scores = score(shots, METHODS)
    scores["window"] = {"f1": _estimate(np.asarray(windows), stats.f1)}
    bar = verdict(scores)  # on the whole split only
    if subset is not None:
        within = [f for f in shots if f.shot in set(subset["shots"])]
        if within:
            wanted = {f.shot for f in within}
            subset |= score(within, METHODS)
            subset["window"] = {
                "f1": _estimate(
                    np.asarray([c for f, c in zip(shots, windows) if f.shot in wanted]),
                    stats.f1,
                )
            }
    meta = {
        "candidate": blob["candidate"],
        "version": version,
        "model_sha256": hashlib.sha256(snapshots["model.pt"]).hexdigest(),
        "chosen_sha256": hashlib.sha256(chosen_bytes).hexdigest(),
        "split_sha256": hashlib.sha256(snapshots["split.csv"]).hexdigest(),
        "labels_copy_sha256": hashlib.sha256(
            snapshots["review/labels.csv"]
        ).hexdigest(),
        "threshold": blob["threshold"],
        "band_khz": blob["band_khz"],
        "model_git_sha": blob["git_sha"],
        "labels_sha256": blob["labels_sha256"],
        "git_sha": git_sha(),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tier": "suggestions",
        "limit": limit,
        **cv_meta,
    }
    record = {"meta": meta, "bar": bar, "bar_thresholds": BAR, **scores}
    if subset is not None:
        record[f"{subset['version']}_subset"] = subset
    with atomic_path(models / "evaluation.json") as tmp:
        tmp.write_text(json.dumps(record, indent=1) + "\n")
    with atomic_path(models / "evaluation.md") as tmp:
        tmp.write_text(report_md(scores, bar, meta, subset))
    return record


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--choose", action="store_true", help="score candidates on validation"
    )
    mode.add_argument(
        "--test",
        action="store_true",
        help="score the chosen model once on the test shots (the default)",
    )
    p.add_argument(
        "--models", type=Path, help="default $LABELER_ROOT/models/ae_xpower/v1"
    )
    p.add_argument("--version", default=VERSION)
    p.add_argument("--limit", type=int, default=0, help="the first N shots (pilots)")
    args = p.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    paths = Paths.from_env()
    models = args.models or model_dir(paths, args.version)
    try:
        record = (
            run_choose(paths, models, args.limit, version=args.version)
            if args.choose
            else run_test(paths, models, args.limit, version=args.version)
        )
    except (OSError, ValueError) as error:
        p.error(str(error))
    if args.choose:
        print(f"chose {record['candidate']}: {record['why']}")
    else:
        said = {"bar": record["bar"], "frames": record["frames"]}
        for key in (f"{v}_subset" for v in SUBSET_OF.values()):
            if key in record:
                said[key] = record[key].get("frames")
        print(json.dumps(said))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
