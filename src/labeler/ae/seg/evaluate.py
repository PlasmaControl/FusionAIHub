"""Score the AE segmentation once, on the test shots.

    python -m labeler.ae.seg.evaluate [--version V] [--models DIR] [--limit N]

writes `evaluation.json` and `evaluation.md` beside the model (default
`$LABELER_ROOT/models/ae_seg/<version>`, v1 the default version). A model is
scored only as the version its blob records (`train.blob_version`).

**Pixels.** Every pixel a test shot's reviewed pseudo-mask scores (0 or 1:
80-250 kHz, inside the owner's present or absent frames, inside TokEye's
0-2 s). **Frames.** The owner's present and absent 10 ms frames of 0-2 s that
TokEye covers. A method calls a frame present when half its columns hold an
80-250 kHz pixel it calls AE (`MIN_FRACTION`, the rule `ae_xpower` scores TokEye
by).

**Methods.** `ae_seg`, the model at its validation threshold; `recipe`,
pseudo-v1's rule with the chosen `ae_xpower` model's frames in place of the
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

**SegNet v2** (`--version v2`) is scored over the owner's whole windows: its
pixels are every pixel a test shot's pseudo-v2 mask scores (0-250 kHz, the band
its blob records, `train.blob_band`), and its frames the owner's present and
absent frames of the whole window that TokEye's whole-shot record
(`ae/masks-full`) covers. `ae_seg`, `recipe` and `tokeye` are all cut to the
blob's band, so they are scored on the same pixels. `recipe` is still pseudo-v1's
rule, with ae_xpower v3's chosen model's frames: TokEye's lit pixels inside
them, cut to the band. Neither it nor `tokeye` applies pseudo-v2's MHD-line or
bright rules, so below 80 kHz both keep every line TokEye lights, MHD lines
included. The bar is judged on that table. Beside it, not judged against it,
`window_0_2s` scores the same calls on the pixels and frames of 0-2 s, the
window SegNet v1 was scored on (the pixels stay pseudo-v2's, over the blob's
band).

**SegNet v3** (`--version v3`) is scored as v2, on pseudo-v3's pixels. Its test
shots are v2's, so evaluation.md states that its test is a second use of them,
made after v2's breakdown was seen (`reuse_note`). Dice and G3 cannot see an MHD
line drawn as AE in a present frame, since pseudo-v3 IGNORES it there, so
`mhd_lines` counts, per shot and pooled, SegNet's AE pixels on the pixels the
markers IGNORE in present frames (the mask's `mhd`), and its AE pixels below
80 kHz in the owner's window's columns inside a catalog NTM interval
(`markers.ntm_intervals`), beside TokEye's lit pixels there. Every report says
that `recipe`'s and `tokeye`'s pixel Dice are high by construction
(`BY_CONSTRUCTION`).

**SegNet v4** (`--version v4`) is scored as v1, over 0-2 s, on pseudo-v4's
pixels, over the band its blob records (60-250 kHz): every method and a frame's
present call are cut to that band, and evaluation.md names it. Its test shots
are v1's, so evaluation.md states that its test is a second use of them, made
after v1's test was scored (`reuse_note`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import torch

from ...config import Paths, atomic_path, git_sha, sha256_of
from ...events.catalog.states import ABSENT, PRESENT
from ...events.review import labels
from ...events.review.rows import Grid
from ...scoring import stats
from ...scoring.frames import FRAME_MS
from ..xpower import check_limit, pilot_area, tokeye_masks
from ..xpower import model_dir as ae_model_dir
from ..xpower.data import (
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
    fp_rate,
    mhd_absent,
)
from ..xpower.train import load as load_ae
from ..xpower.train import probabilities, read_split
from . import (
    EVENT,
    SEG_VERSIONS,
    VERSION,
    markers,
    model_dir,
    pseudo,
    regions,
    reuse_note,
    train,
)

METHODS = ("ae_seg", "recipe", "tokeye")
BAR = {"dice": 0.75, "dice_low": 0.65, "frame_precision": 0.90, "mhd_fp_rate": 0.05}
HEADER = (
    "| method | pixel Dice | frame precision | frame recall | frame F1 | FP rate, MHD |"
)
#: Said in every report beside the pixel Dice (module docstring, **Methods**).
BY_CONSTRUCTION = (
    "`recipe`'s and `tokeye`'s pixel Dice are high by construction: the "
    "pseudo-masks are TokEye's lit pixels, so these two are the baselines the "
    "model is read against, not rivals it must beat."
)
MHD_LINE_KEYS = ("mhd_px", "mhd_ae_px", "ntm_cols", "ntm_lit_px", "ntm_ae_px")


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


def mhd_line_counts(on, mhd, lit, low, ntm, window) -> dict:
    """What Dice and G3 cannot see, for one shot (`MHD_LINE_KEYS`): `mhd_px`,
    the IGNORED MHD-like pixels of present frames (`mhd`, `(n_y, n)`), and
    `mhd_ae_px`, those `on` calls AE; `ntm_cols`, the columns of the owner's
    window (`window`, `(n,)`) inside a catalog NTM interval (`ntm`, `(n,)`), and
    in them below 80 kHz (`low`, `(n_y,)`) TokEye's `lit` pixels (`ntm_lit_px`)
    and `on`'s (`ntm_ae_px`)."""
    cols = np.asarray(ntm, dtype=bool) & np.asarray(window, dtype=bool)
    box = np.asarray(low, dtype=bool)[:, None] & cols[None, :]
    mhd = np.asarray(mhd, dtype=bool)
    return {
        "mhd_px": int(mhd.sum()),
        "mhd_ae_px": int((on & mhd).sum()),
        "ntm_cols": int(cols.sum()),
        "ntm_lit_px": int((lit & box).sum()),
        "ntm_ae_px": int((on & box).sum()),
    }


def mhd_lines_summary(rows: dict) -> dict:
    """`rows` maps a shot to its `mhd_line_counts`: the pooled counts, the pooled
    shares (`mhd_ae_share` = mhd_ae_px / mhd_px, `ntm_ae_share` = ntm_ae_px /
    ntm_lit_px; None over nothing) and the rows by shot."""
    total = {k: int(sum(r[k] for r in rows.values())) for k in MHD_LINE_KEYS}

    def share(a: str, b: str) -> float | None:
        return total[a] / total[b] if total[b] else None

    return {
        **total,
        "mhd_ae_share": share("mhd_ae_px", "mhd_px"),
        "ntm_ae_share": share("ntm_ae_px", "ntm_lit_px"),
        "shots": {str(s): rows[s] for s in sorted(rows)},
    }


def mhd_lines_md(summary: dict, where: str) -> list[str]:
    """The MHD-lines section: the pooled counts and the shots that draw most."""

    def pct(x) -> str:
        return "n/a" if x is None else f"{x:.1%}"

    shots = summary["shots"]
    worst = sorted(shots, key=lambda s: (-shots[s]["mhd_ae_px"], int(s)))[:10]
    lines = [
        "## MHD lines in present frames (not scored by Dice or G3)",
        "",
        (
            f"SegNet's AE pixels on the {summary['mhd_px']} MHD-like pixels "
            f"{where} IGNORES in present frames: {summary['mhd_ae_px']} "
            f"({pct(summary['mhd_ae_share'])}). Below 80 kHz in the "
            f"{summary['ntm_cols']} columns of the owners' windows inside a "
            f"catalog NTM interval: {summary['ntm_ae_px']} AE pixels, of "
            f"{summary['ntm_lit_px']} TokEye lights there "
            f"({pct(summary['ntm_ae_share'])})."
        ),
        "",
        (
            "| shot | MHD-like px | AE on them | NTM columns | lit below 80 kHz "
            "there | AE there |"
        ),
        "|---|---|---|---|---|---|",
        *(
            "| {} | {mhd_px} | {mhd_ae_px} | {ntm_cols} | {ntm_lit_px} | "
            "{ntm_ae_px} |".format(s, **shots[s])
            for s in worst
        ),
        "",
        (
            "The ten shots with the most AE pixels on MHD-like pixels; every shot "
            "is in the JSON record."
        ),
    ]
    return lines


def shot_scores(
    paths: Paths,
    shot: int,
    *,
    label,
    decisions,
    seg,
    ae,
    pseudo_bytes=None,
    version: str = VERSION,
    ntm_spans=None,
    clicked: bytes | None = None,
) -> dict:
    """One test shot: `{"pixels": {method: cells}, "frames": ShotFrames}`.
    `seg` and `ae` are `(model, blob)` pairs. Every method is cut to the band the
    seg blob records. A whole-window version's frames are the owner's whole
    window, and it adds `"pixels_0_2s"` and `"frames_0_2s"`: the same calls on
    the pixels and frames of 0-2 s, as v1's are scored. `clicked` is the mask
    the shot's decision was made on, when training's manifest names it
    (`train.frozen_clicked`). When the pseudo-mask
    keeps its markers' pixels (`mhd`, pseudo-v3) it adds `"mhd_lines"`
    (`mhd_line_counts`), with the shot's NTM intervals `ntm_spans` (ms; none
    when None)."""
    spec = SEG_VERSIONS[version]
    masks = tokeye_masks(paths, spec.ae_version)
    ex = train.load_example(
        paths,
        shot,
        decisions,
        margin=None,
        pseudo_bytes=pseudo_bytes,
        version=version,
        clicked=clicked,
    )
    seg_model, seg_blob = seg
    n_y = ex.y.shape[0]
    grid = Grid(ex.t0_ms, ex.dt_ms, ex.y.shape[1])
    in_band = np.zeros(n_y, dtype=bool)
    in_band[band_slice(ex.y0_khz, ex.dy_khz, n_y, train.blob_band(seg_blob))] = True
    t_ms, clean, _ = tokeye_clean(clean_path(masks, shot))
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
    # recipe is pseudo-v1's rule on that model's frames, over the blob's band: on
    # a v2 blob it keeps the MHD lines pseudo-v2's rules ignore below 80 kHz.
    on = {
        "ae_seg": (train.predict(seg_model, ex.x) >= seg_blob["threshold"])
        & in_band[:, None],
        "recipe": tokeye_on & column_frames(said_ae, first, grid)[None, :],
        "tokeye": tokeye_on,
    }
    pixels = {
        m: train.pixel_cells(v.astype(np.float32), ex.y, 0.5) for m, v in on.items()
    }
    extra = {}
    if pseudo_bytes is None:
        pseudo_bytes = regions.pseudo_file(paths, shot, version).read_bytes()
    pm = pseudo.PseudoMask.load(BytesIO(pseudo_bytes))
    if pm.mhd is not None:
        centres = grid.t0_ms + (np.arange(grid.n) + 0.5) * grid.dt_ms
        window = (centres >= label.window[0]) & (centres < label.window[1])
        khz = ex.y0_khz + np.arange(n_y) * ex.dy_khz
        ntm = markers.ntm_columns(ntm_spans or (), grid.t0_ms, grid.dt_ms, grid.n)
        extra["mhd_lines"] = mhd_line_counts(
            on["ae_seg"], pm.mhd, lit, khz < markers.RULE_BELOW_KHZ, ntm, window
        )

    def frames(lo: int, hi: int) -> ShotFrames:
        """Frames `lo .. hi - 1`, scored where the owner says present or absent
        and TokEye's record covers them."""
        owner = targets(label, lo, hi - lo)
        seen = covered_frames(covered, grid, lo, hi - lo)
        scored = np.isin(owner, (ABSENT, PRESENT)) & seen
        return ShotFrames(
            shot,
            owner,
            mhd_frames(clean_path(masks, shot), lo, hi - lo),
            scored,
            {m: frame_calls(v, grid, lo, hi - lo) for m, v in on.items()},
        )

    lo = max(first, EVAL_FRAMES[0])
    hi = max(lo, min(first + n, EVAL_FRAMES[1]))
    if not spec.whole_window:
        return {"pixels": pixels, "frames": frames(lo, hi), **extra}
    # 0-2 s: the columns whose centre lies in EVAL_FRAMES' frames, as v1 scores.
    centres = grid.t0_ms + (np.arange(grid.n) + 0.5) * grid.dt_ms
    early = ex.y.copy()
    outside = (centres < EVAL_FRAMES[0] * FRAME_MS) | (
        centres >= EVAL_FRAMES[1] * FRAME_MS
    )
    early[:, outside] = pseudo.IGNORE
    return {
        "pixels": pixels,
        "frames": frames(first, first + n),
        "pixels_0_2s": {
            m: train.pixel_cells(v.astype(np.float32), early, 0.5)
            for m, v in on.items()
        },
        "frames_0_2s": frames(lo, hi),
        **extra,
    }


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


def _tables(scores: dict) -> list[str]:
    """The methods' table and the paired differences' table."""
    lines = [HEADER, "|---|---|---|---|---|---|"]
    for m, s in scores["methods"].items():
        cols = ("dice", "frame_precision", "frame_recall", "frame_f1", "fp_rate_mhd")
        lines.append(f"| {m} | " + " | ".join(_fmt(s[k]) for k in cols) + " |")
    lines += ["", "| paired difference | value [95 %] |", "|---|---|"]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in scores["differences"].items()]
    return lines


def report_md(scores: dict, bar: dict, meta: dict) -> str:
    """v1's report; a whole-window record's names its window and band and adds
    the 0-2 s tables after the bar."""
    c = scores["counts"]
    summary = (
        f"{c['shots']} shots; {c['scored_pixels']} scored pixels ({c['ae_pixels']} "
        f"AE); {c['frames']} frames of 0-2 s ({c['present_frames']} present, "
        f"{c['mhd_absent_frames']} MHD frames called absent). Threshold "
        f"{meta['threshold']}. 95 % shot-bootstrap intervals."
    )
    whole = meta.get("frames_window") == "whole"
    if not whole and meta.get("version", VERSION) != VERSION:
        # A later 0-2 s version (v4) names its band; v1's report is as it was.
        lo, hi = meta["band_khz"]
        summary = summary.replace(
            f"Threshold {meta['threshold']}.",
            f"Threshold {meta['threshold']}, band {lo:g}-{hi:g} kHz.",
        )
    if whole:
        lo, hi = meta["band_khz"]
        summary = (
            f"{c['shots']} shots; {c['scored_pixels']} scored pixels "
            f"({c['ae_pixels']} AE); {c['frames']} frames of the owner's whole "
            f"windows ({c['present_frames']} present, {c['mhd_absent_frames']} MHD "
            f"frames called absent). Threshold {meta['threshold']}, band "
            f"{lo:g}-{hi:g} kHz. 95 % shot-bootstrap intervals."
        )
    lines = ["# AE segmentation on the test shots", "", summary, "", *_tables(scores)]
    said = {k: "pass" if bar[k] else "FAIL" for k in ("G1", "G2", "G3")}
    gates = ", ".join(f"{k} {v}" for k, v in said.items())
    lines += ["", f"The bar: {gates}. Tier: suggestions.", "", BY_CONSTRUCTION, ""]
    note = reuse_note(meta["version"], c["shots"])
    if note is not None:
        lines += [note, ""]
    if whole:
        early = scores["window_0_2s"]
        e = early["counts"]
        masks = SEG_VERSIONS[meta["version"]].pseudo
        sentence = (
            f"The same model's pixels and frames of 0-2 s, the window SegNet v1 was "
            f"scored on; the pixels are still {masks}'s over {lo:g}-{hi:g} kHz "
            f"({e['scored_pixels']} scored pixels, {e['ae_pixels']} AE; "
            f"{e['frames']} frames, {e['present_frames']} present, "
            f"{e['mhd_absent_frames']} MHD frames called absent). For comparison, "
            "not judged against the bar."
        )
        lines += ["## 0-2 s", "", sentence, "", *_tables(early), ""]
    if "mhd_lines" in scores:
        where = SEG_VERSIONS[meta["version"]].pseudo
        lines += [*mhd_lines_md(scores["mhd_lines"], where), ""]
    return "\n".join(lines)


def run_test(
    paths: Paths, models: Path, limit: int = 0, version: str = VERSION
) -> dict:
    spec = SEG_VERSIONS[version]
    check_limit(paths, models, limit)
    evaluation = models / "evaluation.json"
    if evaluation.exists() and not pilot_area(models, paths.runs):
        raise FileExistsError(
            f"{evaluation}: the test shots are scored once; a retry is a new version"
        )
    model_file = models / "model.pt"
    model_bytes = model_file.read_bytes()
    seg = train.load(BytesIO(model_bytes))
    found = train.blob_version(seg[1])
    if found != version:
        raise ValueError(
            f"{model_file}: a SegNet {found} model; score it with --version {found}"
        )
    inputs = seg[1].get("inputs", {})
    archive = {
        "pseudo_masks_sha256": models / "pseudo_masks.json",
        "labels_sha256": labels.labels_path(models),
        "masks_sha256": regions.log_path(models),
    }
    frozen = {}
    evaluation_inputs = {}
    for key, path in archive.items():
        if not path.is_file() or not inputs.get(key):
            raise ValueError(
                f"{path}: trained before the frozen bundle; evaluate a new version"
            )
        frozen[key] = path.read_bytes()
        evaluation_inputs[key] = hashlib.sha256(frozen[key]).hexdigest()
        if evaluation_inputs[key] != inputs[key]:
            raise ValueError(f"{path}: archive hash differs from the training bundle")
    if not inputs.get("ae_model_sha256"):
        raise ValueError(
            f"{model_file}: trained before the frozen bundle; evaluate a new version"
        )
    split_bytes = (models / "split.csv").read_bytes()
    if hashlib.sha256(split_bytes).hexdigest() != inputs.get("split_sha256"):
        raise ValueError(
            f"{models / 'split.csv'}: split hash differs from the training bundle"
        )
    split = read_split(models / "split.csv", data=split_bytes)
    ae_models = ae_model_dir(paths, spec.ae_version)
    choice_bytes = (ae_models / "chosen.json").read_bytes()
    ae_file = ae_models / json.loads(choice_bytes)["candidate"] / "model.pt"
    ae_bytes = ae_file.read_bytes()
    evaluation_inputs["ae_model_sha256"] = hashlib.sha256(ae_bytes).hexdigest()
    evaluation_inputs["ae_chosen_sha256"] = hashlib.sha256(choice_bytes).hexdigest()
    if evaluation_inputs["ae_model_sha256"] != inputs["ae_model_sha256"]:
        raise ValueError(
            f"{ae_file}: chosen xpower model differs from the frozen bundle"
        )
    ae = load_ae(BytesIO(ae_bytes))
    saved, decisions = train.read_review_bytes(
        frozen["labels_sha256"], frozen["masks_sha256"]
    )
    manifest = json.loads(frozen["pseudo_masks_sha256"])
    shots = sorted(s for s, v in split.items() if v == "test")
    pseudo_bytes, clicked = {}, {}
    evaluation_inputs["pseudo_masks"] = {}
    for shot in shots:
        path = regions.pseudo_file(paths, shot, version)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if manifest.get(path.name) != digest:
            raise ValueError(f"{path}: pseudo-mask differs from the frozen bundle")
        pseudo_bytes[shot] = data
        evaluation_inputs["pseudo_masks"][path.name] = digest
        # The mask the reviewer clicked, when not this version's (`regions.transfer`).
        decision = decisions.get(shot)
        data = train.frozen_clicked(paths, shot, decision, manifest)
        if data is not None:
            clicked[shot] = data
            name = regions.clicked_name(shot, decision)
            evaluation_inputs["pseudo_masks"][name] = hashlib.sha256(data).hexdigest()
    shots = shots[:limit] if limit else shots
    if not shots:
        raise ValueError(f"{models / 'split.csv'} has no test shot")
    # A version made with pseudo-v3's markers is also read inside the catalog's
    # NTM intervals (`mhd_lines`).
    spans = {}
    if spec.gated:
        spans, evaluation_inputs["ntm_sha256"] = markers.ntm_intervals(paths)
    # Stores and TokEye remain external; detect drift across scoring. Targets
    # and models above are loaded from the exact bytes whose hashes we record.
    masks = tokeye_masks(paths, spec.ae_version)
    external = {}
    for s in shots:
        for path in (paths.spectrogram_file(EVENT, s), clean_path(masks, s)):
            if path is None:
                raise ValueError(f"{masks}: no TokEye mask for {s}")
            external[str(path)] = sha256_of(path)
    per_shot = [
        shot_scores(
            paths,
            s,
            label=saved[s],
            decisions=decisions,
            seg=seg,
            ae=ae,
            pseudo_bytes=pseudo_bytes[s],
            version=version,
            ntm_spans=spans.get(s, ()),
            clicked=clicked.get(s),
        )
        for s in shots
    ]
    for path, digest in external.items():
        if sha256_of(path) != digest:
            raise ValueError(f"{path}: evaluation input changed during scoring")
    evaluation_inputs["external_files"] = external
    scores = score(per_shot)
    if spec.whole_window:
        scores["window_0_2s"] = score(
            [{"pixels": s["pixels_0_2s"], "frames": s["frames_0_2s"]} for s in per_shot]
        )
    mhd_rows = {s: r["mhd_lines"] for s, r in zip(shots, per_shot) if "mhd_lines" in r}
    if mhd_rows:
        scores["mhd_lines"] = mhd_lines_summary(mhd_rows)
    bar = verdict(scores)
    meta = {
        "threshold": seg[1]["threshold"],
        "model_git_sha": seg[1]["git_sha"],
        "inputs": seg[1]["inputs"],
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "evaluation_inputs": evaluation_inputs,
        "ae_model": str(ae_file),
        "min_fraction": MIN_FRACTION,
        "git_sha": git_sha(),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tier": "suggestions",
        "limit": limit,
        "version": version,
        "band_khz": [float(b) for b in train.blob_band(seg[1])],
        "frames_window": "whole" if spec.whole_window else "0-2 s",
    }
    record = {"meta": meta, "bar": bar, "bar_thresholds": BAR, **scores}
    with atomic_path(models / "evaluation.json") as tmp:
        tmp.write_text(json.dumps(record, indent=1) + "\n")
    with atomic_path(models / "evaluation.md") as tmp:
        tmp.write_text(report_md(scores, bar, meta))
    return record


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--version",
        choices=sorted(SEG_VERSIONS),
        default=VERSION,
        help="the SegNet version the model is (default v1); v2 and v3 are scored "
        "over the owner's whole windows; v4 as v1, over 0-2 s at 60-250 kHz",
    )
    p.add_argument(
        "--models", type=Path, help="default $LABELER_ROOT/models/ae_seg/<version>"
    )
    p.add_argument("--limit", type=int, default=0, help="the first N shots (pilots)")
    args = p.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    paths = Paths.from_env()
    models = args.models or model_dir(paths, args.version)
    try:
        record = run_test(paths, models, args.limit, version=args.version)
    except (OSError, ValueError) as error:
        p.error(str(error))
    print(json.dumps({"bar": record["bar"], "counts": record["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
