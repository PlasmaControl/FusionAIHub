"""Score the AE segmentation once, on the test shots.

    python -m labeler.ae.seg.evaluate [--models DIR] [--limit N]

writes `evaluation.json` and `evaluation.md` beside the model (default
`$LABELER_ROOT/models/ae_seg/v1`).

**Pixels.** Every pixel a test shot's reviewed pseudo-mask scores (0 or 1:
80-250 kHz, inside the owner's present or absent frames, inside TokEye's
0-2 s). **Frames.** The owner's present and absent 10 ms frames of 0-2 s that
TokEye covers. A method calls a frame present when half its columns hold an
80-250 kHz pixel it calls AE (`MIN_FRACTION`, the rule `ae_xpower` scores TokEye
by).

**Methods.** `ae_seg`, the model at its validation threshold; `recipe`, the
pseudo-mask rule with the chosen `ae_xpower` model's frames in place of the
owner's (TokEye's lit pixels inside the frames that model calls present), which
is what could be drawn without this model; `tokeye`, TokEye's lit pixels in
80-250 kHz in every frame. `recipe` and `tokeye` score well on pixels by
construction, since the pseudo-masks are TokEye's pixels: they are the
baselines the model is read against, not rivals it must beat.

**Scores.** Pixel Dice (F1 over pixels) and frame precision, recall and F1,
pooled over shots with 95 % shot-bootstrap intervals (`labeler.scoring.stats`,
2000 replicates, seed 20260923); the false-positive rate on MHD frames (absent,
TokEye sees a 0-60 kHz line); paired differences `ae_seg - recipe`.

**The bar** (`verdict`): G1 pixel Dice >= 0.75 with a lower bound >= 0.65; G2
frame precision >= 0.90 (where the model draws a mode, the owner saw AE; its
frame recall is bounded by TokEye's, so it is reported, not gated); G3 MHD
false-positive rate <= 0.05. Passing or not, a mask is a suggestion, shown as a
point of interest, never a catalog label.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from ...config import Paths, atomic_path, git_sha
from ...events.catalog.states import ABSENT, PRESENT
from ...events.review import labels
from ...events.review.rows import Grid
from ...scoring import stats
from ...scoring.frames import FRAME_MS
from ..xpower import event_dir, tokeye_masks
from ..xpower import model_dir as ae_model_dir
from ..xpower.data import (
    BAND_KHZ,
    MIN_FRACTION,
    SEED,
    band_slice,
    clean_path,
    mhd_frames,
    store_rows,
    targets,
    tokeye_clean,
    window_frames,
)
from ..xpower.evaluate import (
    EVAL_FRAMES,
    ShotFrames,
    cells,
    chosen_model,
    fp_rate,
    mhd_absent,
)
from ..xpower.train import load as load_ae
from ..xpower.train import probabilities, read_split
from . import EVENT, model_dir, pseudo, regions, train

METHODS = ("ae_seg", "recipe", "tokeye")
BAR = {"dice": 0.75, "dice_low": 0.65, "frame_precision": 0.90, "mhd_fp_rate": 0.05}
HEADER = (
    "| method | pixel Dice | frame precision | frame recall | frame F1 | FP rate, MHD |"
)


def frame_calls(on: np.ndarray, grid: Grid, first: int, n: int) -> np.ndarray:
    """`(n,)` frames `first ..` present: at least `MIN_FRACTION` of the columns
    whose centre lies in the frame hold a pixel of `on` `(n_y, cols)`."""
    centres = grid.t0_ms + (np.arange(grid.n) + 0.5) * grid.dt_ms
    k = np.floor(centres / FRAME_MS).astype(np.int64) - first
    inside = (k >= 0) & (k < n)
    total = np.bincount(k[inside], minlength=n)
    lit = np.bincount(k[inside], weights=on.any(axis=0)[inside], minlength=n)
    return (total > 0) & (lit >= MIN_FRACTION * total)


def covered_frames(covered: np.ndarray, grid: Grid, first: int, n: int) -> np.ndarray:
    """`(n,)` frames `first ..` that hold at least one column, all of them `covered`."""
    centres = grid.t0_ms + (np.arange(grid.n) + 0.5) * grid.dt_ms
    k = np.floor(centres / FRAME_MS).astype(np.int64) - first
    inside = (k >= 0) & (k < n)
    total = np.bincount(k[inside], minlength=n)
    good = np.bincount(k[inside], weights=covered[inside], minlength=n)
    return (total > 0) & (good == total)


def column_frames(flags: np.ndarray, first: int, grid: Grid) -> np.ndarray:
    """`(cols,)`: each column takes the flag of the frame holding its centre."""
    centres = grid.t0_ms + (np.arange(grid.n) + 0.5) * grid.dt_ms
    k = np.floor(centres / FRAME_MS).astype(np.int64) - first
    out = np.zeros(grid.n, dtype=bool)
    inside = (k >= 0) & (k < len(flags))
    out[inside] = flags[k[inside]]
    return out


def shot_scores(paths: Paths, shot: int, *, label, decisions, seg, ae) -> dict:
    """One test shot: `{"pixels": {method: cells}, "frames": ShotFrames}`.
    `seg` and `ae` are `(model, blob)` pairs."""
    ex = train.load_example(paths, shot, decisions, margin=None)
    n_y = ex.y.shape[0]
    grid = Grid(ex.t0_ms, ex.dt_ms, ex.y.shape[1])
    in_band = np.zeros(n_y, dtype=bool)
    in_band[band_slice(ex.y0_khz, ex.dy_khz, n_y, BAND_KHZ)] = True
    t_ms, clean, _ = tokeye_clean(clean_path(tokeye_masks(paths), shot))
    lit = pseudo.pool_columns(pseudo.tokeye_rows(clean), t_ms, grid)[:n_y]
    covered = pseudo.covered_columns(t_ms, grid)
    tokeye_on = lit & in_band[:, None] & covered[None, :]
    first, n = window_frames(label.window)
    ae_model, ae_blob = ae
    prob, observed = probabilities(
        ae_model,
        store_rows(paths.spectrogram_file(EVENT, shot)),
        first,
        n,
        band=ae_blob["band_khz"],
    )
    said_ae = (prob >= ae_blob["threshold"]) & observed
    seg_model, seg_blob = seg
    on = {
        "ae_seg": (train.predict(seg_model, ex.x) >= seg_blob["threshold"])
        & in_band[:, None],
        "recipe": tokeye_on & column_frames(said_ae, first, grid)[None, :],
        "tokeye": tokeye_on,
    }
    pixels = {
        m: train.pixel_cells(v.astype(np.float32), ex.y, 0.5) for m, v in on.items()
    }
    lo = max(first, EVAL_FRAMES[0])
    hi = max(lo, min(first + n, EVAL_FRAMES[1]))
    owner = targets(label, lo, hi - lo)
    seen = covered_frames(covered, grid, lo, hi - lo)
    scored = np.isin(owner, (ABSENT, PRESENT)) & seen
    frames = ShotFrames(
        shot,
        owner,
        mhd_frames(clean_path(tokeye_masks(paths), shot), lo, hi - lo),
        scored,
        {m: frame_calls(v, grid, lo, hi - lo) for m, v in on.items()},
    )
    return {"pixels": pixels, "frames": frames}


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


def score(per_shot: list[dict]) -> dict:
    frames = [s["frames"] for s in per_shot]
    pixel = {m: np.array([s["pixels"][m] for s in per_shot]) for m in METHODS}
    out = {"methods": {}, "differences": {}}
    for m in METHODS:
        c = cells(frames, m)
        out["methods"][m] = {
            "dice": _estimate(pixel[m], stats.f1),
            "frame_precision": _estimate(c, stats.precision),
            "frame_recall": _estimate(c, stats.recall),
            "frame_f1": _estimate(c, stats.f1),
            "fp_rate_mhd": _estimate(cells(frames, m, mhd_absent), fp_rate),
        }
    out["differences"]["dice_minus_recipe"] = _difference(
        pixel["ae_seg"], pixel["recipe"], stats.f1
    )
    out["differences"]["frame_f1_minus_recipe"] = _difference(
        cells(frames, "ae_seg"), cells(frames, "recipe"), stats.f1
    )
    out["counts"] = {
        "shots": len(per_shot),
        "ae_pixels": int(sum(s["pixels"]["ae_seg"][[0, 2]].sum() for s in per_shot)),
        "scored_pixels": int(sum(s["pixels"]["ae_seg"].sum() for s in per_shot)),
        "frames": int(sum(f.scored.sum() for f in frames)),
        "present_frames": int(
            sum((f.scored & (f.owner == PRESENT)).sum() for f in frames)
        ),
        "mhd_absent_frames": int(sum((f.scored & mhd_absent(f)).sum() for f in frames)),
    }
    return out


def _at_least(x, bound) -> bool:
    return x is not None and x >= bound


def verdict(scores: dict) -> dict:
    ours = scores["methods"]["ae_seg"]
    g1 = _at_least(ours["dice"]["value"], BAR["dice"]) and _at_least(
        ours["dice"]["low"], BAR["dice_low"]
    )
    g2 = _at_least(ours["frame_precision"]["value"], BAR["frame_precision"])
    mhd = ours["fp_rate_mhd"]["value"]
    g3 = mhd is not None and mhd <= BAR["mhd_fp_rate"]
    return {
        "G1": bool(g1),
        "G2": bool(g2),
        "G3": bool(g3),
        "all": bool(g1 and g2 and g3),
    }


def _fmt(e: dict) -> str:
    if e["value"] is None:
        return "n/a"
    lo, hi = e["low"], e["high"]
    band = "" if lo is None else f" [{lo:.3f}, {hi:.3f}]"
    return f"{e['value']:.3f}{band}"


def report_md(scores: dict, bar: dict, meta: dict) -> str:
    c = scores["counts"]
    summary = (
        f"{c['shots']} shots; {c['scored_pixels']} scored pixels ({c['ae_pixels']} "
        f"AE); {c['frames']} frames of 0-2 s ({c['present_frames']} present, "
        f"{c['mhd_absent_frames']} MHD frames called absent). Threshold "
        f"{meta['threshold']}. 95 % shot-bootstrap intervals."
    )
    lines = [
        "# AE segmentation on the test shots",
        "",
        summary,
        "",
        HEADER,
        "|---|---|---|---|---|---|",
    ]
    for m, s in scores["methods"].items():
        cols = ("dice", "frame_precision", "frame_recall", "frame_f1", "fp_rate_mhd")
        lines.append(f"| {m} | " + " | ".join(_fmt(s[k]) for k in cols) + " |")
    lines += ["", "| paired difference | value [95 %] |", "|---|---|"]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in scores["differences"].items()]
    said = {k: "pass" if bar[k] else "FAIL" for k in ("G1", "G2", "G3")}
    gates = ", ".join(f"{k} {v}" for k, v in said.items())
    lines += ["", f"The bar: {gates}. Tier: suggestions.", ""]
    return "\n".join(lines)


def run_test(paths: Paths, models: Path, limit: int = 0) -> dict:
    seg = train.load(models / "model.pt")
    split = read_split(models / "split.csv")
    ae_file = chosen_model(ae_model_dir(paths))
    ae = load_ae(ae_file)
    saved = labels.read_saved(event_dir(paths))
    decisions = regions.read_decisions(event_dir(paths))
    shots = sorted(s for s, v in split.items() if v == "test")
    shots = shots[:limit] if limit else shots
    if not shots:
        raise ValueError(f"{models / 'split.csv'} has no test shot")
    per_shot = [
        shot_scores(paths, s, label=saved[s], decisions=decisions, seg=seg, ae=ae)
        for s in shots
    ]
    scores = score(per_shot)
    bar = verdict(scores)
    meta = {
        "threshold": seg[1]["threshold"],
        "model_git_sha": seg[1]["git_sha"],
        "inputs": seg[1]["inputs"],
        "ae_model": str(ae_file),
        "min_fraction": MIN_FRACTION,
        "git_sha": git_sha(),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tier": "suggestions",
        "limit": limit,
    }
    record = {"meta": meta, "bar": bar, "bar_thresholds": BAR, **scores}
    with atomic_path(models / "evaluation.json") as tmp:
        tmp.write_text(json.dumps(record, indent=1) + "\n")
    with atomic_path(models / "evaluation.md") as tmp:
        tmp.write_text(report_md(scores, bar, meta))
    return record


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--models", type=Path, help="default $LABELER_ROOT/models/ae_seg/v1")
    p.add_argument("--limit", type=int, default=0, help="the first N shots (pilots)")
    args = p.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    paths = Paths.from_env()
    record = run_test(paths, args.models or model_dir(paths), args.limit)
    print(json.dumps({"bar": record["bar"], "counts": record["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
