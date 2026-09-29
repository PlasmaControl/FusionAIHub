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
- **both:** both hold.

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
(the model, the rules, the snapshot) and v3's F1 on its test shots with and
without the listed frames. The exit status is 1 if any shot failed.

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
from ..seg import mhdlines, pseudo, pseudo_dir
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
WHY = ("model", "tokeye_below80", "both")
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


def low_frames(
    paths: Paths,
    shot: int,
    label: labels.Label,
    rules: Rules,
    first: int,
    n: int,
    *,
    tokeye_bytes: bytes | None = None,
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
    flagged frame."""
    if tokeye_bytes is None:
        tokeye_bytes = _tokeye_file(paths, shot).read_bytes()
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
    t_ms, clean, _ = tokeye_clean(BytesIO(tokeye_bytes))
    n_y = lines.lit.shape[0]
    native = pseudo.tokeye_rows(clean)[:n_y]
    cols = np.floor((t_ms - lines.t0_ms) / lines.dt_ms).astype(np.int64)
    on_grid = np.flatnonzero((cols >= 0) & (cols < gone.shape[1]))
    native[:, on_grid] &= ~gone[:, cols[on_grid]]
    khz = lines.y0_khz + np.arange(n_y) * lines.dy_khz
    below = khz < FLOOR_KHZ
    column = native[below].any(axis=0) & ~native[~below].any(axis=0)
    flag = frame_share(t_ms, column, first, n) >= MIN_FRACTION
    flag &= frame_covered(t_ms, first, n)
    f_lo, f_hi = np.full(n, np.inf), np.full(n, -np.inf)
    k = np.floor(t_ms / FRAME_MS).astype(np.int64) - first  # as frame_share
    use = np.flatnonzero(column & (k >= 0) & (k < n))
    if use.size:  # each flagging column's lowest and highest lit bin
        lit = native[:, use]
        np.minimum.at(f_lo, k[use], khz[lit.argmax(axis=0)])
        np.maximum.at(f_hi, k[use], khz[n_y - 1 - lit[::-1].argmax(axis=0)])
    return flag, np.where(flag, f_lo, np.nan), np.where(flag, f_hi, np.nan)


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
    """A worker's state: the chosen model and its split, v3's snapshot's labels
    and pseudo-v2's rules (`asdict` of them)."""
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
    )


def shot_rows(shot: int) -> tuple[list[dict], ShotFrames, np.ndarray]:
    """A worker's shot, from `_init`'s state: its rows (`COLUMNS`), its frames
    (`evaluate.whole_frames`) and which of them are listed.

    A frame is listed when it is scored and the owner calls it absent, and v3
    says present off an MHD frame (`model`), or `low_frames` flags it
    (`tokeye_below80`), or both (`both`). Each run of consecutive frames with
    the same reason is a row: its frames' edges (ms) and count, round(max P(AE),
    4), the nanmin of f_lo and the nanmax of f_hi to 0.01 kHz (blank when all
    NaN), and the shot's split in the model's split.csv."""
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
    flag, f_lo, f_hi = low_frames(paths, shot, label, w["rules"], first, n)
    low = absent & flag
    why = model.astype(np.int64) + 2 * low  # WHY[why - 1]; 0 is not listed
    rows = [
        {
            "shot": int(shot),
            "split": w["split"].get(shot, ""),
            "t_start": (first + a) * FRAME_MS,
            "t_end": (first + b) * FRAME_MS,
            "why": WHY[code - 1],
            "f_lo_khz": _khz(f_lo[a:b], np.nanmin),
            "f_hi_khz": _khz(f_hi[a:b], np.nanmax),
            "p_max": round(float(frames.prob[a:b].max()), 4),
            "frames": b - a,
        }
        for a, b, code in _runs(why)
    ]
    return rows, frames, why > 0


def _f1(frames: list[ShotFrames]) -> dict:
    """v3's frame F1 over these shots' scored frames and its 95 % shot-bootstrap
    interval, as its test estimates it (strata "all", weights 1)."""
    return evaluate._estimate(cells(frames, "ae_xpower"), stats.f1)


def report_md(record: dict) -> str:
    """below80.md: the frames by reason, the failed shots, and v3's test F1 with
    and without the listed frames."""
    frames, test, failed = record["frames"], record["test"], record["failed"]
    floor = f"{record['floor_khz']:g} kHz"
    page = "{:g}-{:g} kHz".format(*BAND_KHZ)
    holds = {
        "model": "v3 says AE, off an MHD frame",
        "tokeye_below80": f"TokEye lights only below {floor}, steady lines removed",
        "both": "both",
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
    lines += [
        "| reason | frames | what holds |",
        "|---|---|---|",
        *(f"| {why} | {frames[why]} | {holds[why]} |" for why in WHY),
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
    counts = dict.fromkeys(WHY, 0)
    for row in rows:
        counts[row["why"]] += row["frames"]
    test = sorted(s for s in done if split.get(s) == "test")
    tested = [done[s] for s in test]
    record = {
        "version": VERSION,
        "tier": "suggestions",
        "floor_khz": FLOOR_KHZ,
        "shots": len(done),
        "failed": failed,
        "rows": len(table),
        "frames": counts,
        "model": str(model_file),
        "model_sha256": model_sha256,
        "rules": asdict(rules),
        "rules_sha256": hashlib.sha256(rules_bytes).hexdigest(),
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
