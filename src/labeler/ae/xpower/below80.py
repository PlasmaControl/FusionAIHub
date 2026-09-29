"""below80: the frames the owner called absent where AE may lie below 80 kHz.

    python -m labeler.ae.xpower.below80 [--shots 170720,176053] [--workers N]
        [--out FILE]

The owner labelled AE on the review page, which is cropped to 80-250 kHz
(`data.BAND_KHZ`), so a frame whose AE lies only below 80 kHz may be labelled
absent: the page never showed it there. This command lists such frames for the
owner to look at. A frame is listed when the owner calls it absent in ae_xpower
v3's label snapshot, v3 scores it (`evaluate.whole_frames`: TokEye's whole-shot
record and the model's rows cover it), and at least one reason holds:
- **model:** v3 says present, off an MHD frame (one where TokEye lights a
  0-60 kHz line for half the frame, as `data.mhd_frames` has it: there v3's
  call may be an MHD mode's harmonics);
- **tokeye_below80:** in at least half its TokEye columns, TokEye's whole-shot
  mask (`ae/masks-full`) lights two-chord pixels below 80 kHz (`FLOOR_KHZ`) and
  none at or above it, once pseudo-v2's steady lines are removed (`low_frames`:
  the steady rule of `labeler.ae.seg.mhdlines` on the pixels below 80 kHz, as
  pseudo-v2 applies it, with pseudo-v2's `rules.json`);
- **tokeye_lines:** in at least half its TokEye columns, TokEye lights
  two-chord pixels below 80 kHz that pseudo-v3's per-line MHD markers do not
  take (`line_frames`: `labeler.ae.seg.markers`, its fixed values, with the
  catalog's NTM intervals), whatever it lights at or above 80 kHz. The owner
  saw only 80-250 kHz, so an AE line running on above 80 kHz in an absent
  frame is no reason to drop the frame.
A run's `why` names every reason that holds in it, joined by "+", in
`REASONS` order (e.g. "model+tokeye_below80", once "both").

Every shot of v3's snapshot by default, or those `--shots` names, over
`--workers` processes (default `$SLURM_CPUS_PER_TASK`, else 1). Before any shot
the command refuses, naming the file:
- a chosen model (`models/ae_xpower/v3`) that cannot be read or whose checkpoint
  is not v3's, or an unreadable `split.csv` beside it;
- no `evaluation.json` in `models/ae_xpower/v3` naming the chosen model's
  sha256 (`gallery.check_tested`): below80.json gives v3's F1 on its test
  shots, so the command comes after v3's one test (`evaluate --test --version
  v3`), as the gallery's pictures do;
- pseudo-v2's `rules.json` missing (run `python -m labeler.ae.seg.pseudo
  --version v2` first) or unreadable;
- a label snapshot whose sha256 is not v3's, or a `--shots` shot not in it.

**Output.** `$LABELER_ROOT/suggestions/ae_xpower/v3/below80.csv` (or `--out`):
one row per run of consecutive listed frames of a shot with the same reason
(`COLUMNS`, by shot and `t_start`): the run's edges in ms and its frames, v3's
highest P(AE) in it, the lowest and highest frequency TokEye lights below
80 kHz in it (blank for a `model` run) and the shot's split in v3's model.
Beside it, `below80.md` and then `below80.json`: the frames by reason, the
failed shots (left out of the table and of the F1s), what was read by sha256
(the model, the rules, the NTM table, the snapshot), v3's F1 on its test shots
with and without the listed frames, and each reason's recall on the stretch the
owner cited (`OWNER_CITED`: 176041, absent at 424-1504 ms, AE at 40-80 kHz the
page did not show). The exit status is 1 if any shot failed.

**Tier:** suggestions, never labels. Nothing here changes the owner's labels or
the page's band: a listed frame is one for the owner to look at again. Test-shot
frames are listed too: relabelling them from this list would bias a later
version's test on the same shots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from dataclasses import asdict, replace
from datetime import UTC, datetime
from io import BytesIO
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import Paths, atomic_path, git_sha, sha256_of
from ...events.catalog.states import ABSENT
from ...events.review import labels
from ...scoring import stats
from ...scoring.frames import FRAME_MS
from ..seg import markers, mhdlines, pseudo, pseudo_dir
from ..seg.mhdlines import Rules
from . import (
    blob_version,
    evaluate,
    model_dir,
    read_snapshot,
    snapshot_file,
    suggestions_dir,
    tokeye_masks,
)
from .data import (
    BAND_KHZ,
    MIN_FRACTION,
    clean_path,
    frame_covered,
    frame_share,
    tokeye_clean,
    window_frames,
)
from .evaluate import ShotFrames, cells, chosen_model
from .gallery import check_tested, run_all
from .train import load, read_split

VERSION = "v3"
#: The page's crop. Steady lines stop at mhdlines.RULE_BELOW_KHZ, so the two must agree.
FLOOR_KHZ = 80.0  # BAND_KHZ[0]
#: The reasons, in the order a run's `why` joins them ("+").
REASONS = ("model", "tokeye_below80", "tokeye_lines")
#: The stretch the owner cited: (shot, start ms, end ms), absent at 80-250 kHz
#: with AE at 40-80 kHz. Each reason's recall there is in below80.json and .md.
OWNER_CITED = ((176041, 424.0, 1504.0),)
COLUMNS = (
    "shot",
    "split",
    "t_start",
    "t_end",
    "why",
    "f_lo_khz",
    "f_hi_khz",
    "p_max",
    "frames",
)
RULES_VERSION = "v2"  # the segmentation version whose rules.json is read
#: What reading the chosen model, v3's test record, pseudo-v2's rules or the
#: label snapshot can raise: a missing or damaged file (torch.load's errors
#: among them), or a record of another shape.
UNREADABLE = (
    OSError,
    ValueError,
    KeyError,
    TypeError,
    AttributeError,
    RuntimeError,
    EOFError,
    pickle.UnpicklingError,
)


def below80_file(paths: Paths) -> Path:
    """The table, `suggestions/ae_xpower/v3/below80.csv`; its record and report
    are beside it."""
    return suggestions_dir(paths, VERSION) / "below80.csv"


def _tokeye_file(paths: Paths, shot: int) -> Path:
    """The shot's clean file in v3's TokEye masks (masks-full), the record
    `evaluate.whole_frames` reads; FileNotFoundError without one."""
    masks = tokeye_masks(paths, VERSION)
    file = clean_path(masks, shot)
    if file is None:
        raise FileNotFoundError(f"{shot} has no TokEye mask in {masks}")
    return file


def why_name(code: int) -> str:
    """The reasons of a code (bit i: REASONS[i]) joined by "+"."""
    return "+".join(r for i, r in enumerate(REASONS) if code >> i & 1)


def _frame_flags(
    lines, gone, tokeye_bytes: bytes, first: int, n: int, *, clear_above: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(flag, f_lo, f_hi)` of `low_frames`, with the pixels of `gone` (the
    level-8 grid) removed from TokEye's own columns; a column flags when a pixel
    is left below FLOOR_KHZ and, with `clear_above`, none at or above it. f_lo
    and f_hi are over the pixels left below FLOOR_KHZ."""
    t_ms, clean, _ = tokeye_clean(BytesIO(tokeye_bytes))
    n_y = lines.lit.shape[0]
    native = pseudo.tokeye_rows(clean)[:n_y]
    cols = np.floor((t_ms - lines.t0_ms) / lines.dt_ms).astype(np.int64)
    on_grid = np.flatnonzero((cols >= 0) & (cols < gone.shape[1]))
    native[:, on_grid] &= ~gone[:, cols[on_grid]]
    khz = lines.y0_khz + np.arange(n_y) * lines.dy_khz
    below = khz < FLOOR_KHZ
    column = native[below].any(axis=0)
    if clear_above:
        column &= ~native[~below].any(axis=0)
    flag = frame_share(t_ms, column, first, n) >= MIN_FRACTION
    flag &= frame_covered(t_ms, first, n)
    f_lo, f_hi = np.full(n, np.inf), np.full(n, -np.inf)
    k = np.floor(t_ms / FRAME_MS).astype(np.int64) - first  # as frame_share
    use = np.flatnonzero(column & (k >= 0) & (k < n))
    if use.size:  # each flagging column's lowest and highest lit bin below
        lit = native[:, use] & below[:, None]
        top = int(np.flatnonzero(below).max())
        np.minimum.at(f_lo, k[use], khz[lit.argmax(axis=0)])
        np.maximum.at(f_hi, k[use], khz[top - lit[top::-1].argmax(axis=0)])
    return flag, np.where(flag, f_lo, np.nan), np.where(flag, f_hi, np.nan)


def low_frames(
    paths: Paths,
    shot: int,
    label: labels.Label,
    rules: Rules,
    first: int,
    n: int,
    *,
    tokeye_bytes: bytes | None = None,
    lines=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(flag, f_lo, f_hi)`, each `(n,)`, for frames `first .. first + n - 1`:
    whether TokEye lights only below FLOOR_KHZ there, and the lowest and highest
    frequency (kHz) it lights there, NaN where it does not flag.

    The shot's masks-full record (`tokeye_bytes`, its bytes, when given) is read
    as pseudo-v2 reads it (`pseudo.shot_lines`), and its steady lines are found
    on the store's level-8 grid as pseudo-v2 finds them (`mhdlines.mhd_like`):
    `mhdlines.steady` of `mhdlines.below(lit, ...)`, the pixels below 80 kHz, so
    a region crossing it is cut there, with the rules' `drift_khz`, `steady_ms`
    and `guard_khz`. In TokEye's own columns
    (`pseudo.tokeye_rows`: the page's bins at each TokEye time), a pixel whose
    grid column (`floor((t - t0) / dt)`, as `pseudo.pool_columns` maps it) holds
    a steady pixel in its bin is removed. A column flags when a pixel is left in
    a bin below FLOOR_KHZ (`y0 + k * dy < FLOOR_KHZ`) and none at or above it;
    a frame flags when at least MIN_FRACTION of its columns flag (`frame_share`)
    and TokEye's record covers it (`frame_covered`). f_lo and f_hi are the
    lowest and highest bin frequency of the flagging columns' pixels in each
    flagged frame (a flagging column has none at or above FLOOR_KHZ). `lines`,
    the shot's `pseudo.shot_lines` of the same bytes, when already read."""
    if tokeye_bytes is None:
        tokeye_bytes = _tokeye_file(paths, shot).read_bytes()
    if lines is None:
        lines = pseudo.shot_lines(paths, shot, label, tokeye_bytes=tokeye_bytes)
    gone = mhdlines.steady(
        mhdlines.below(lines.lit, lines.y0_khz, lines.dy_khz),
        lines.y0_khz,
        lines.dy_khz,
        lines.dt_ms,
        rules.drift_khz,
        rules.steady_ms,
        rules.guard_khz,
    )
    return _frame_flags(lines, gone, tokeye_bytes, first, n, clear_above=True)


def line_frames(
    paths: Paths,
    shot: int,
    label: labels.Label,
    first: int,
    n: int,
    *,
    spans=(),
    tokeye_bytes: bytes | None = None,
    lines=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The **tokeye_lines** reason, `(flag, f_lo, f_hi)` as `low_frames` gives
    them: the pixels pseudo-v3's markers take (`markers.mhd_lines` with the
    fixed `markers.Markers()`, below FLOOR_KHZ; `spans`, the shot's catalog NTM
    intervals in ms) are removed in place of the steady lines, and a column
    flags when a pixel is left below FLOOR_KHZ, whatever lies at or above it.
    f_lo and f_hi are the flagging columns' lowest and highest pixel below
    FLOOR_KHZ."""
    if tokeye_bytes is None:
        tokeye_bytes = _tokeye_file(paths, shot).read_bytes()
    if lines is None:
        lines = pseudo.shot_lines(paths, shot, label, tokeye_bytes=tokeye_bytes)
    ntm = markers.ntm_columns(spans, lines.t0_ms, lines.dt_ms, lines.lit.shape[1])
    gone = markers.mhd_lines(lines, markers.Markers(), ntm)["mhd"]
    return _frame_flags(lines, gone, tokeye_bytes, first, n, clear_above=False)


def _runs(codes: np.ndarray) -> list[tuple[int, int, int]]:
    """`(start, stop, code)` of each run of one nonzero code."""
    edges = np.flatnonzero(np.diff(np.r_[0, codes, 0]))
    return [(int(a), int(b), int(codes[a])) for a, b in pairwise(edges) if codes[a]]


def _khz(values: np.ndarray, pick) -> float | str:
    """`pick` (np.nanmin or np.nanmax) of `values` to 0.01 kHz; blank when all
    are NaN."""
    if np.isnan(values).all():
        return ""
    return round(float(pick(values)), 2)


_WORKER: dict = {}


def _init(
    model_file: str, root: str, label_tables: str, corpus: str, rules: dict
) -> None:
    """A worker's state: the chosen model and its split, v3's snapshot's labels,
    pseudo-v2's rules (`asdict` of them) and the catalog's NTM intervals."""
    import torch

    torch.set_num_threads(1)
    model, blob = load(model_file)
    paths = Paths(root=Path(root), label_tables=Path(label_tables), corpus=Path(corpus))
    _WORKER.update(
        paths=paths,
        model=model,
        blob=blob,
        split=read_split(Path(model_file).parent / "split.csv"),
        saved=read_snapshot(paths, VERSION)[1],
        rules=Rules(**rules),
        spans=markers.ntm_intervals(paths)[0],
    )


def shot_rows(shot: int) -> tuple[list[dict], ShotFrames, np.ndarray]:
    """A worker's shot, from `_init`'s state: its rows (`COLUMNS`), its frames
    (`evaluate.whole_frames`) and each frame's reasons code (0: not listed; bit
    i: REASONS[i] holds).

    A frame is listed when it is scored and the owner calls it absent, and v3
    says present off an MHD frame (`model`), or `low_frames` flags it
    (`tokeye_below80`), or `line_frames` does (`tokeye_lines`). Each run of
    consecutive frames with the same reasons is a row, `why` naming them
    (`why_name`): its frames' edges (ms) and count, round(max P(AE), 4), the
    nanmin of f_lo and the nanmax of f_hi of the two TokEye reasons to 0.01 kHz
    (blank when all NaN), and the shot's split in the model's split.csv."""
    w = _WORKER
    paths, label = w["paths"], w["saved"][shot]
    frames = evaluate.whole_frames(
        shot,
        paths=paths,
        label=label,
        model=w["model"],
        blob=w["blob"],
        version=VERSION,
    )
    first, n = window_frames(label.window)
    absent = frames.scored & (frames.owner == ABSENT)
    model = absent & frames.said["ae_xpower"] & ~frames.mhd
    tokeye = _tokeye_file(paths, shot).read_bytes()
    lines = pseudo.shot_lines(paths, shot, label, tokeye_bytes=tokeye)
    flag, lo2, hi2 = low_frames(
        paths, shot, label, w["rules"], first, n, tokeye_bytes=tokeye, lines=lines
    )
    per_line, lo3, hi3 = line_frames(
        paths,
        shot,
        label,
        first,
        n,
        spans=w["spans"].get(shot, ()),
        tokeye_bytes=tokeye,
        lines=lines,
    )
    low, held = absent & flag, absent & per_line
    # np.fmin and np.fmax skip a NaN: a frame keeps the reason that flags it.
    f_lo = np.where(low | held, np.fmin(lo2, lo3), np.nan)
    f_hi = np.where(low | held, np.fmax(hi2, hi3), np.nan)
    why = model.astype(np.int64) + 2 * low + 4 * held  # 0 is not listed
    rows = [
        {
            "shot": int(shot),
            "split": w["split"].get(shot, ""),
            "t_start": (first + a) * FRAME_MS,
            "t_end": (first + b) * FRAME_MS,
            "why": why_name(code),
            "f_lo_khz": _khz(f_lo[a:b], np.nanmin),
            "f_hi_khz": _khz(f_hi[a:b], np.nanmax),
            "p_max": round(float(frames.prob[a:b].max()), 4),
            "frames": b - a,
        }
        for a, b, code in _runs(why)
    ]
    return rows, frames, why


def _f1(frames: list[ShotFrames]) -> dict:
    """v3's frame F1 over these shots' scored frames and its 95 % shot-bootstrap
    interval, as its test estimates it (strata "all", weights 1)."""
    return evaluate._estimate(cells(frames, "ae_xpower"), stats.f1)


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def report_md(record: dict) -> str:
    """below80.md: the frames by reason and by `why`, the failed shots, v3's
    test F1 with and without the listed frames, and the recall on the stretch
    the owner cited."""
    frames, test, failed = record["frames"], record["test"], record["failed"]
    floor = f"{record['floor_khz']:g} kHz"
    page = "{:g}-{:g} kHz".format(*BAND_KHZ)
    holds = {
        "model": "v3 says AE, off an MHD frame",
        "tokeye_below80": f"TokEye lights only below {floor}, steady lines removed",
        "tokeye_lines": (
            f"TokEye lights below {floor} past pseudo-v3's per-line MHD markers, "
            f"whatever it lights at or above {floor}"
        ),
    }
    intro = (
        f"The owner labelled AE on the review page cropped to {page}, so a frame "
        f"whose AE lies only below {floor} may be labelled absent. Of the frames "
        "the owner called absent in v3's label snapshot that v3 scores on the "
        f"{record['shots']} shots read, {sum(frames.values())} are listed, in "
        f"{record['rows']} runs, for the owner to look at."
    )
    f1 = (
        f"v3's F1 on its {len(test['shots'])} test shots: "
        f"{evaluate._fmt(test['f1_with'])} with the listed frames, "
        f"{evaluate._fmt(test['f1_without'])} without the "
        f"{test['removed_frames']} listed there. 95 % shot-bootstrap intervals."
    )
    lines = [
        f"# AE frames the owner called absent: v3's AE, or TokEye's below {floor}",
        "",
        intro,
        "",
    ]
    if failed:
        shots = ", ".join(map(str, failed))
        lines += [f"Failed shots, left out of the table and of both F1s: {shots}.", ""]
    reasons = record["reasons"]
    cited = []
    for c in record["owner_cited"]:
        where = f"{c['shot']}'s owner-absent stretch at {c['t_ms'][0]:g}-" + (
            f"{c['t_ms'][1]:g} ms (the owner's example: AE at 40-80 kHz the page "
            "did not show)"
        )
        if c["frames"] is None:
            cited.append(f"{where}: the shot was not read.")
            continue
        each = ", ".join(
            f"{r} {c['listed'][r]} ({_pct(c['recall'][r])})" for r in REASONS
        )
        cited.append(
            f"Recall on {where}: {c['listed']['any']} of its {c['frames']} scored "
            f"absent frames are listed ({_pct(c['recall']['any'])}); by reason, "
            f"{each}."
        )
    lines += [
        "| reason | frames where it holds | what holds |",
        "|---|---|---|",
        *(f"| {r} | {reasons[r]} | {holds[r]} |" for r in REASONS),
        "",
        "| why (the reasons of a run) | frames |",
        "|---|---|",
        *(f"| {why} | {count} |" for why, count in frames.items()),
        "",
        *cited,
        "",
        f1,
        "",
        (
            "Test-shot frames are listed too: relabelling them from this list "
            "would bias a later version's test on the same shots."
        ),
        "",
        "Tier: suggestions. The owner's labels and the page's band are unchanged.",
    ]
    return "\n".join(lines) + "\n"


def owner_cited(shot: int, a: float, b: float, done, label) -> dict:
    """The recall on an owner-cited stretch: of the shot's scored frames the
    owner calls absent whose centre lies in [a, b) ms, how many are listed, by
    reason and at all ("any"). `done` is the shot's (frames, reasons codes), or
    None when it was not read; then `frames`, `listed` and `recall` are None."""
    entry = {"shot": shot, "t_ms": [a, b], "frames": None, "listed": None}
    entry["recall"] = None
    if done is None or label is None:
        return entry
    frames, why = done
    first, _ = window_frames(label.window)
    centre = (first + np.arange(len(why)) + 0.5) * FRAME_MS
    inside = frames.scored & (frames.owner == ABSENT) & (centre >= a) & (centre < b)
    listed = {
        r: int((inside & ((why >> i) & 1).astype(bool)).sum())
        for i, r in enumerate(REASONS)
    }
    listed["any"] = int((inside & (why > 0)).sum())
    k = int(inside.sum())
    entry.update(
        frames=k,
        listed=listed,
        recall={r: v / k if k else None for r, v in listed.items()},
    )
    return entry


def _read(p: argparse.ArgumentParser, file, read):
    """read(), or p.error naming `file` and what went wrong."""
    try:
        return read()
    except UNREADABLE as error:
        p.error(f"{file}: {type(error).__name__}: {error}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--shots",
        help="comma-separated, e.g. 170720,176053; default: every shot of v3's "
        "label snapshot",
    )
    p.add_argument(
        "--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    )
    p.add_argument(
        "--out",
        type=Path,
        help="default $LABELER_ROOT/suggestions/ae_xpower/v3/below80.csv",
    )
    args = p.parse_args(argv)
    paths = Paths.from_env()
    # Before any shot, each input is read and checked, and a failure names it.
    models = model_dir(paths, VERSION)
    model_file = _read(p, models / "chosen.json", lambda: chosen_model(models))
    blob = _read(p, model_file, lambda: load(model_file)[1])
    if blob_version(blob) != VERSION:
        p.error(
            f"{model_file}: the checkpoint's version is {blob_version(blob)}, "
            f"not {VERSION}"
        )
    # below80.json gives v3's F1 on its test shots, so the command comes after
    # the one test of this model, as the gallery's pictures do.
    evaluation = models / "evaluation.json"
    if not evaluation.is_file():
        p.error(
            f"{evaluation}: no test yet; below80 gives v3's test F1, so it runs "
            f"after evaluate --test --version {VERSION}"
        )
    _read(p, evaluation, lambda: check_tested(models, VERSION, model_file))
    model_sha256 = _read(p, model_file, lambda: sha256_of(model_file))
    split_file = model_file.parent / "split.csv"
    split = _read(p, split_file, lambda: read_split(split_file))
    rules_file = pseudo_dir(paths, RULES_VERSION) / "rules.json"
    if not rules_file.is_file():
        p.error(
            f"{rules_file}: no rules; run python -m labeler.ae.seg.pseudo "
            f"--version {RULES_VERSION} first"
        )
    rules_bytes = _read(p, rules_file, rules_file.read_bytes)
    rules = _read(p, rules_file, lambda: Rules(**json.loads(rules_bytes)["rules"]))
    ntm_file = paths.label_tables / markers.NTM_TABLE
    _, ntm_sha256 = _read(p, ntm_file, lambda: markers.ntm_intervals(paths))
    snapshot = snapshot_file(paths, VERSION)
    data, saved = _read(p, snapshot, lambda: read_snapshot(paths, VERSION))
    shots = sorted(saved)
    if args.shots:
        try:
            shots = sorted({int(s) for s in args.shots.split(",") if s.strip()})
        except ValueError:
            shots = []
        if not shots:
            p.error(f"--shots {args.shots!r}: comma-separated shot numbers")
        unknown = [s for s in shots if s not in saved]
        if unknown:
            p.error(f"{snapshot}: not shots of v3's label snapshot: {unknown}")
    init = (
        str(model_file),
        str(paths.root),
        str(paths.label_tables),
        str(paths.corpus),
        asdict(rules),
    )
    rows, done, failed = [], {}, []
    for shot, outcome in run_all(shot_rows, shots, args.workers, _init, init):
        if isinstance(outcome, Exception):
            failed.append(shot)
            print(f"{shot}: {type(outcome).__name__}: {outcome}", flush=True)
            continue
        found, frames, listed = outcome
        rows += found
        done[shot] = (frames, listed)
    table = pd.DataFrame(rows, columns=list(COLUMNS))
    table = table.sort_values(["shot", "t_start"], kind="stable", ignore_index=True)
    # The frames of each `why`, in code order, and of each reason.
    counts = {}
    for code in range(1, 2 ** len(REASONS)):
        n = sum(row["frames"] for row in rows if row["why"] == why_name(code))
        if n:
            counts[why_name(code)] = n
    reasons = {
        r: int(sum(((why >> i) & 1).sum() for _, why in done.values()))
        for i, r in enumerate(REASONS)
    }
    test = sorted(s for s in done if split.get(s) == "test")
    tested = [(f, why > 0) for f, why in (done[s] for s in test)]
    record = {
        "version": VERSION,
        "tier": "suggestions",
        "floor_khz": FLOOR_KHZ,
        "shots": len(done),
        "failed": failed,
        "rows": len(table),
        "frames": counts,
        "reasons": reasons,
        "owner_cited": [
            owner_cited(shot, a, b, done.get(shot), saved.get(shot))
            for shot, a, b in OWNER_CITED
        ],
        "model": str(model_file),
        "model_sha256": model_sha256,
        "rules": asdict(rules),
        "rules_sha256": hashlib.sha256(rules_bytes).hexdigest(),
        "markers": asdict(markers.Markers()),
        "ntm_table": str(ntm_file),
        "ntm_sha256": ntm_sha256,
        "labels_sha256": hashlib.sha256(data).hexdigest(),
        "git_sha": git_sha(),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "test": {
            "shots": test,
            "removed_frames": int(sum((f.scored & on).sum() for f, on in tested)),
            "f1_with": _f1([f for f, _ in tested]),
            "f1_without": _f1([replace(f, scored=f.scored & ~on) for f, on in tested]),
        },
    }
    out = args.out or below80_file(paths)
    with atomic_path(out) as tmp:
        table.to_csv(tmp, index=False)
    with atomic_path(out.with_suffix(".md")) as tmp:
        tmp.write_text(report_md(record))
    with atomic_path(out.with_suffix(".json")) as tmp:  # last: the run is whole
        tmp.write_text(json.dumps(record, indent=1) + "\n")
    said = {
        "out": str(out),
        "shots": len(done),
        "failed": failed,
        "rows": len(table),
        "frames": counts,
        "f1_with": record["test"]["f1_with"]["value"],
        "f1_without": record["test"]["f1_without"]["value"],
    }
    print(json.dumps(said))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
