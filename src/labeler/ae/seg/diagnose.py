"""Where SegNet fails on its test shots: a breakdown of its one test (side command).

    python -m labeler.ae.seg.diagnose [--version V] [--models DIR] [--limit N]

Not a new test and nothing re-selected: it reads the model, split, threshold and
frozen bundle that `evaluate` scored, refuses unless their sha256s are those
`evaluation.json` records (the model, the split, the labels, the mask decisions,
the pseudo-mask manifest and each test shot's pseudo-mask), makes the same calls
and breaks them down. It writes `diagnosis.json` and `diagnosis.md` beside the
model (default `$LABELER_ROOT/models/ae_seg/v2`):
- **pixels:** `ae_seg`'s pixel Dice, precision and recall per band
  (`DIAG_BANDS_KHZ`: 0-20, 20-40, 40-60, 60-80 and 80-250 kHz, and the blob's
  whole band) and per time (the owner's whole window, 0-2 s and after 2 s),
  pooled over the test shots with 95 % shot-bootstrap intervals, as `evaluate`
  pools them;
- **frames:** frame precision, recall and F1 and G3's MHD false-positive rate,
  with the calls over the blob's band (as `evaluate` makes them) and with the
  calls cut to 80-250 kHz (`CUT_KHZ`), over the same three times;
- **MHD lines in present frames** (`evaluate.mhd_line_counts`), which Dice and
  G3 cannot see: SegNet's AE pixels on the pixels its version IGNORES as MHD in
  present frames (v2: `mhdlines.mhd_like` under pseudo-v2's `rules.json`, & a
  present column & IGNORE; a pseudo-v3 mask: its `mhd`), and its AE pixels below
  80 kHz in the owner's window's columns inside a catalog NTM interval
  (`markers.ntm_intervals`).

The whole-band, whole-window Dice and MHD false-positive rate must equal
`evaluation.json`'s; `matches_evaluation` records whether they do, and the
report says so. The report carries `evaluate.BY_CONSTRUCTION`. Tier:
suggestions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import torch

from ...config import Paths, atomic_path, git_sha
from ...events.catalog.states import ABSENT, PRESENT
from ...events.review import labels
from ...events.review.rows import Grid
from ...scoring import stats
from ...scoring.frames import FRAME_MS
from ..xpower import tokeye_masks
from ..xpower.data import (
    BAND_KHZ,
    band_slice,
    clean_path,
    mhd_frames,
    targets,
    tokeye_clean,
    window_frames,
)
from ..xpower.evaluate import EVAL_FRAMES, ShotFrames, cells, fp_rate, mhd_absent
from ..xpower.train import read_split
from . import SEG_VERSIONS, markers, model_dir, pseudo, pseudo_dir, regions, train
from .evaluate import (
    BY_CONSTRUCTION,
    _estimate,
    _fmt,
    covered_frames,
    frame_calls,
    mhd_line_counts,
    mhd_lines_md,
    mhd_lines_summary,
)
from .mhdlines import RULE_BELOW_KHZ, Rules, mhd_like

VERSION = "v2"
DIAG_BANDS_KHZ = ((0.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, 80.0), (80.0, 250.0))
CUT_KHZ = BAND_KHZ  # G3 with the calls cut to the owner's view
TIMES = ("whole", "0-2 s", "after 2 s")
CALLS = ("ae_seg", "ae_seg_80_250")


def _band_rows(khz: np.ndarray, band) -> np.ndarray:
    """The bins whose centre lies in [lo, hi), and [lo, hi] for a band ending at
    the top of `DIAG_BANDS_KHZ`, so the bands partition 0-250 kHz."""
    lo, hi = band
    top = hi >= DIAG_BANDS_KHZ[-1][1]
    return (khz >= lo) & ((khz <= hi) if top else (khz < hi))


def _band_name(band) -> str:
    return "{:g}-{:g}".format(*band)


def _times(centres: np.ndarray) -> dict[str, np.ndarray]:
    """Each time's columns (or frames): by centre, 0-2 s as `evaluate` cuts it."""
    lo, hi = EVAL_FRAMES[0] * FRAME_MS, EVAL_FRAMES[1] * FRAME_MS
    return {
        "whole": np.ones(centres.shape, dtype=bool),
        "0-2 s": (centres >= lo) & (centres < hi),
        "after 2 s": centres >= hi,
    }


def shot_breakdown(
    paths: Paths,
    shot: int,
    *,
    label,
    decisions,
    seg,
    pseudo_bytes: bytes,
    version: str,
    rules: Rules | None,
    ntm_spans=(),
) -> dict:
    """One test shot: `pixels` {band: {time: [tp, fp, fn, tn]}}, `frames`
    {time: ShotFrames with CALLS}, and `mhd_lines` (`mhd_line_counts`). `rules`
    are pseudo-v2's (the MHD-like pixels of a mask without `mhd`)."""
    spec = SEG_VERSIONS[version]
    ex = train.load_example(
        paths, shot, decisions, margin=None, pseudo_bytes=pseudo_bytes, version=version
    )
    model, blob = seg
    n_y, n = ex.y.shape
    grid = Grid(ex.t0_ms, ex.dt_ms, n)
    khz = ex.y0_khz + np.arange(n_y) * ex.dy_khz
    in_band = np.zeros(n_y, dtype=bool)
    in_band[band_slice(ex.y0_khz, ex.dy_khz, n_y, train.blob_band(blob))] = True
    on = (train.predict(model, ex.x) >= blob["threshold"]) & in_band[:, None]
    cut = np.zeros(n_y, dtype=bool)
    cut[band_slice(ex.y0_khz, ex.dy_khz, n_y, CUT_KHZ)] = True
    centres = grid.t0_ms + (np.arange(n) + 0.5) * grid.dt_ms
    column_times = _times(centres)
    bands = {"all": in_band} | {
        _band_name(b): _band_rows(khz, b) & in_band for b in DIAG_BANDS_KHZ
    }
    pixels = {}
    for name, rows in bands.items():
        pixels[name] = {}
        for when, cols in column_times.items():
            y = np.full_like(ex.y, pseudo.IGNORE)
            box = rows[:, None] & cols[None, :]
            y[box] = ex.y[box]
            pixels[name][when] = train.pixel_cells(on.astype(np.float32), y, 0.5)
    masks = tokeye_masks(paths, spec.ae_version)
    t_ms, _, _ = tokeye_clean(clean_path(masks, shot))
    covered = pseudo.covered_columns(t_ms, grid)
    first, count = window_frames(label.window)
    owner = targets(label, first, count)
    seen = covered_frames(covered, grid, first, count)
    scored = np.isin(owner, (ABSENT, PRESENT)) & seen
    mhd = mhd_frames(clean_path(masks, shot), first, count)
    said = {
        "ae_seg": frame_calls(on, grid, first, count),
        "ae_seg_80_250": frame_calls(on & cut[:, None], grid, first, count),
    }
    frame_times = _times((first + np.arange(count) + 0.5) * FRAME_MS)
    frames = {
        when: ShotFrames(shot, owner, mhd, scored & inside, said)
        for when, inside in frame_times.items()
    }
    lines = pseudo.shot_lines(paths, shot, label)
    present = lines.state == PRESENT
    pm = pseudo.PseudoMask.load(BytesIO(pseudo_bytes))
    if pm.mhd is not None:
        ignored = np.asarray(pm.mhd, dtype=bool)
    else:
        ignored = mhd_like(lines, rules) & present[None, :]
    ignored &= ex.y == pseudo.IGNORE
    window = (centres >= label.window[0]) & (centres < label.window[1])
    ntm = markers.ntm_columns(ntm_spans, grid.t0_ms, grid.dt_ms, n)
    counts = mhd_line_counts(on, ignored, lines.lit, khz < RULE_BELOW_KHZ, ntm, window)
    return {"pixels": pixels, "frames": frames, "mhd_lines": counts}


def _pixel_scores(per_shot: list[dict]) -> dict:
    """{band: {time: {dice, precision, recall, ae_px, scored_px}}}."""
    out = {}
    for name in per_shot[0]["pixels"]:
        out[name] = {}
        for when in TIMES:
            c = np.array([s["pixels"][name][when] for s in per_shot])
            out[name][when] = {
                "dice": _estimate(c, stats.f1),
                "precision": _estimate(c, stats.precision),
                "recall": _estimate(c, stats.recall),
                "ae_px": int(c[:, [0, 2]].sum()),
                "scored_px": int(c.sum()),
            }
    return out


def _frame_scores(per_shot: list[dict]) -> dict:
    """{calls: {time: {precision, recall, f1, fp_rate_mhd, frames, mhd_absent}}}."""
    out = {}
    for m in CALLS:
        out[m] = {}
        for when in TIMES:
            frames = [s["frames"][when] for s in per_shot]
            c = cells(frames, m)
            out[m][when] = {
                "precision": _estimate(c, stats.precision),
                "recall": _estimate(c, stats.recall),
                "f1": _estimate(c, stats.f1),
                "fp_rate_mhd": _estimate(cells(frames, m, mhd_absent), fp_rate),
                "frames": int(sum(f.scored.sum() for f in frames)),
                "mhd_absent_frames": int(
                    sum((f.scored & mhd_absent(f)).sum() for f in frames)
                ),
            }
    return out


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def _check(meta: dict, models: Path, version: str) -> None:
    """Refuse a model or bundle whose sha256 is not the one `evaluation.json`
    records: nothing here is scored that its one test did not score."""
    if meta.get("version", "v1") != version:
        raise ValueError(
            f"{models / 'evaluation.json'}: a SegNet {meta.get('version')} test; "
            f"diagnose it with --version {meta.get('version')}"
        )
    for name, key in (("model.pt", "model_sha256"), ("split.csv", "split_sha256")):
        digest = hashlib.sha256((models / name).read_bytes()).hexdigest()
        if digest != meta.get(key):
            raise ValueError(f"{models / name}: not the file evaluation.json scored")


def run(paths: Paths, models: Path, version: str = VERSION, limit: int = 0) -> dict:
    spec = SEG_VERSIONS[version]
    evaluation = json.loads((models / "evaluation.json").read_text())
    meta = evaluation["meta"]
    _check(meta, models, version)
    model_bytes = (models / "model.pt").read_bytes()
    seg = train.load(BytesIO(model_bytes))
    if train.blob_version(seg[1]) != version:
        raise ValueError(f"{models / 'model.pt'}: not a SegNet {version} model")
    inputs = meta["evaluation_inputs"]
    archive = {
        "pseudo_masks_sha256": models / "pseudo_masks.json",
        "labels_sha256": labels.labels_path(models),
        "masks_sha256": regions.log_path(models),
    }
    frozen = {}
    for key, path in archive.items():
        frozen[key] = path.read_bytes()
        if hashlib.sha256(frozen[key]).hexdigest() != inputs.get(key):
            raise ValueError(f"{path}: not the bundle evaluation.json scored")
    saved, decisions = train.read_review_bytes(
        frozen["labels_sha256"], frozen["masks_sha256"]
    )
    split = read_split(models / "split.csv")
    shots = sorted(s for s, v in split.items() if v == "test")
    shots = shots[:limit] if limit else shots
    pseudo_bytes = {}
    for shot in shots:
        path = regions.pseudo_file(paths, shot, version)
        data = path.read_bytes()
        if inputs["pseudo_masks"].get(path.name) != hashlib.sha256(data).hexdigest():
            raise ValueError(f"{path}: not the pseudo-mask evaluation.json scored")
        pseudo_bytes[shot] = data
    rules, rules_sha256 = None, None
    rules_file = pseudo_dir(paths, version) / "rules.json"
    if not spec.gated:  # pseudo-v2's rules give the MHD-like pixels
        rules_bytes = rules_file.read_bytes()
        rules = Rules(**json.loads(rules_bytes)["rules"])
        rules_sha256 = hashlib.sha256(rules_bytes).hexdigest()
    spans, ntm_sha256 = markers.ntm_intervals(paths)
    per_shot = [
        shot_breakdown(
            paths,
            s,
            label=saved[s],
            decisions=decisions,
            seg=seg,
            pseudo_bytes=pseudo_bytes[s],
            version=version,
            rules=rules,
            ntm_spans=spans.get(s, ()),
        )
        for s in shots
    ]
    pixels = _pixel_scores(per_shot)
    frames = _frame_scores(per_shot)
    ours = evaluation["methods"]["ae_seg"]
    matches = not limit and (
        _same(pixels["all"]["whole"]["dice"]["value"], ours["dice"]["value"])
        and _same(
            frames["ae_seg"]["whole"]["fp_rate_mhd"]["value"],
            ours["fp_rate_mhd"]["value"],
        )
        and _same(
            frames["ae_seg"]["whole"]["precision"]["value"],
            ours["frame_precision"]["value"],
        )
    )
    return {
        "meta": {
            "version": version,
            "side_command": True,
            "tier": "suggestions",
            "shots": len(shots),
            "limit": limit,
            "threshold": seg[1]["threshold"],
            "band_khz": [float(b) for b in train.blob_band(seg[1])],
            "cut_khz": list(CUT_KHZ),
            "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
            "evaluation_sha256": hashlib.sha256(
                (models / "evaluation.json").read_bytes()
            ).hexdigest(),
            "rules": str(rules_file) if rules is not None else None,
            "rules_sha256": rules_sha256,
            "ntm_sha256": ntm_sha256,
            "git_sha": git_sha(),
            "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "matches_evaluation": bool(matches),
        "evaluation": {
            "dice": ours["dice"]["value"],
            "fp_rate_mhd": ours["fp_rate_mhd"]["value"],
            "frame_precision": ours["frame_precision"]["value"],
        },
        "pixels": pixels,
        "frames": frames,
        "mhd_lines": mhd_lines_summary(
            {s: r["mhd_lines"] for s, r in zip(shots, per_shot, strict=True)}
        ),
    }


def report_md(record: dict) -> str:
    """diagnosis.md: the pixel tables by time, the frame table, the MHD lines."""
    meta = record["meta"]
    version = meta["version"]
    where = SEG_VERSIONS[version].pseudo
    lo, hi = meta["cut_khz"]
    frames = record["frames"]
    full, cut = frames["ae_seg"]["whole"], frames["ae_seg_80_250"]["whole"]
    matches = (
        "They match evaluation.json's pooled Dice, frame precision and MHD "
        "false-positive rate."
        if record["matches_evaluation"]
        else "**They do not match evaluation.json's pooled numbers**"
        + (" (a --limit run)." if meta["limit"] else ".")
    )
    lines = [
        f"# SegNet {version}'s test, broken down (diagnosis)",
        "",
        (
            f"A side command, not a new test: SegNet {version}'s model, split, "
            f"threshold ({meta['threshold']}) and frozen bundle, checked by sha256 "
            f"against evaluation.json, on its {meta['shots']} test shots, with the "
            f"same calls broken down by band and time. Nothing is re-selected. "
            f"{matches} 95 % shot-bootstrap intervals. Tier: suggestions."
        ),
        "",
        BY_CONSTRUCTION,
        "",
        (
            f"With the calls cut to {lo:g}-{hi:g} kHz, G3's MHD false-positive rate "
            f"is {_fmt(cut['fp_rate_mhd'])} against {_fmt(full['fp_rate_mhd'])} "
            f"over the blob's band, and frame precision {_fmt(cut['precision'])} "
            f"against {_fmt(full['precision'])} (bar: FP rate <= 0.05)."
        ),
        "",
    ]
    for when in TIMES:
        lines += [
            f"## Pixels, {when}",
            "",
            "| band (kHz) | Dice | precision | recall | AE px | scored px |",
            "|---|---|---|---|---|---|",
        ]
        for name, by_time in record["pixels"].items():
            s = by_time[when]
            lines.append(
                f"| {name} | {_fmt(s['dice'])} | {_fmt(s['precision'])} | "
                f"{_fmt(s['recall'])} | {s['ae_px']} | {s['scored_px']} |"
            )
        lines.append("")
    lines += [
        "## Frames",
        "",
        (
            "| calls | time | precision | recall | F1 | FP rate, MHD | frames "
            "| MHD absent |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    names = {"ae_seg": "the blob's band", "ae_seg_80_250": f"cut to {lo:g}-{hi:g} kHz"}
    for m in CALLS:
        for when in TIMES:
            s = frames[m][when]
            lines.append(
                f"| {names[m]} | {when} | {_fmt(s['precision'])} | "
                f"{_fmt(s['recall'])} | {_fmt(s['f1'])} | {_fmt(s['fp_rate_mhd'])} "
                f"| {s['frames']} | {s['mhd_absent_frames']} |"
            )
    lines += ["", *mhd_lines_md(record["mhd_lines"], where), ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--version",
        choices=sorted(v for v in SEG_VERSIONS if SEG_VERSIONS[v].whole_window),
        default=VERSION,
        help="the SegNet version whose one test is broken down (default v2)",
    )
    p.add_argument(
        "--models", type=Path, help="default $LABELER_ROOT/models/ae_seg/<version>"
    )
    p.add_argument("--limit", type=int, default=0, help="the first N test shots")
    args = p.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    paths = Paths.from_env()
    models = args.models or model_dir(paths, args.version)
    try:
        record = run(paths, models, args.version, args.limit)
    except (OSError, ValueError, KeyError) as error:
        p.error(f"{type(error).__name__}: {error}")
    with atomic_path(models / "diagnosis.json") as tmp:
        tmp.write_text(json.dumps(record, indent=1) + "\n")
    with atomic_path(models / "diagnosis.md") as tmp:
        tmp.write_text(report_md(record))
    said = {
        "out": str(models / "diagnosis.md"),
        "matches_evaluation": record["matches_evaluation"],
        "dice_by_band": {
            k: v["whole"]["dice"]["value"] for k, v in record["pixels"].items()
        },
        "fp_rate_mhd_cut": record["frames"]["ae_seg_80_250"]["whole"]["fp_rate_mhd"][
            "value"
        ],
    }
    print(json.dumps(said))
    return 0 if record["matches_evaluation"] or args.limit else 1


if __name__ == "__main__":
    raise SystemExit(main())
