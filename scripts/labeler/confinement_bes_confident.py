#!/usr/bin/env python
"""Confident learning over the BES classifier's held-out predictions: which labelled
confinement intervals does the signal disagree with?

    PYTHONPATH=src pixi run --frozen --no-install -e labelmaker \
        python scripts/labeler/confinement_bes_confident.py [--row full_cum_abcdrge]

Reads the shot-grouped 5-fold predictions of one ablation row (every shot is predicted
by a model that never saw it), runs confident learning (Northcutt et al. 2021, written
out in ``labeler.confinement.confident`` since cleanlab is not installed) on the windows
the row scores (``score_ok``) that are also at least 20 ms inside their interval, and
flags an interval when at least half of its windows are confidently of one other class.
The row must be a gated one (the default, ``full_cum_abcdrge``, applies the paper's
beam gate and margins): an ungated row such as ``only_ge`` scores passive-BES windows,
where the 150L beam is off, so a "beam-valid" claim holds only for gated rows
(``confident_only_ge.json`` is that earlier, ungated run, kept as such). The analysis
is split by label status and by label source, and looks at QH against WPQH in
particular, since that boundary is the one the experts mark least sharply.

A flagged interval is a candidate for review, not a correction: the classifier sees BES
alone, and a window it calls L inside a labelled H interval may be a real dither. As a
second opinion that shares no input with it, the held-out 0D segmenter
(``confine-ours``, which never sees BES) is read over the same interval: an interval
that both put mostly outside the labelled class is the strongest candidate.

Writes ``confident_<row>.json`` (summary and the flagged list) next to the ablation
record, and the verdict on every interval to ``$LABELER_ROOT/round4/conf/confident/``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.confinement import bes_windows as bw
from labeler.confinement import confident

LABELER = Path(
    os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
)
WORK = LABELER / "round4/conf"
DEFAULT_OUT = REPO / "outputs/labeler/confinement/bes"
MARGIN_MS = 20.0
MIN_WINDOWS = 5
MIN_SHARE = 0.5
PAIRS = (("QH", "WP"), ("WP", "QH"), ("L", "H"), ("H", "L"), ("H", "QH"), ("QH", "H"))


def load_predictions(row_dir: Path, allow_partial: bool) -> pd.DataFrame:
    files = sorted(row_dir.glob("fold*_predictions.csv"))
    if not files or (len(files) < 5 and not allow_partial):
        raise SystemExit(f"{row_dir}: {len(files)} of 5 folds finished")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def windows_table(pred: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """Prediction windows joined to their interval, with ``keep`` set for the windows
    that count (scored by the row, outside the interval-end margins)."""
    iv = intervals[["shot", "interval", "t_start", "t_end"]]
    merged = pred.merge(iv, on=["shot", "interval"], how="left", validate="m:1")
    inside = (merged.center_ms >= merged.t_start + MARGIN_MS) & (
        merged.center_ms <= merged.t_end - MARGIN_MS
    )
    return merged.assign(keep=merged.score_ok.astype(bool) & inside)


def summarise_group(flags: pd.DataFrame, by: str) -> list[dict]:
    rows = []
    for name, g in flags.groupby(by):
        rows.append(
            {
                by: str(name),
                "intervals": len(g),
                "flagged": int(g.flagged.sum()),
                "flagged_share": float(g.flagged.mean()),
                "seconds": float(g.seconds.sum()),
                "flagged_seconds": float(g.seconds[g.flagged].sum()),
            }
        )
    return rows


def analyse(
    windows: pd.DataFrame, probs: np.ndarray, intervals: pd.DataFrame
) -> tuple[dict, pd.DataFrame]:
    labels = windows.label.to_numpy().astype(int)
    keep = windows.keep.to_numpy()
    joint, _ = confident.confident_joint(probs[keep], labels[keep])
    counts = np.bincount(labels[keep], minlength=4)
    calibrated = confident.calibrate_joint(joint, counts)
    off = calibrated.sum() - np.trace(calibrated)
    flags = confident.interval_issues(
        windows[["shot", "interval", "label", "keep"]],
        probs,
        margin_ms=MARGIN_MS,
        min_windows=MIN_WINDOWS,
        min_share=MIN_SHARE,
    )
    meta = intervals[
        ["shot", "interval", "t_start", "t_end", "sources", "status"]
    ].assign(seconds=lambda d: (d.t_end - d.t_start) / 1000.0)
    flags = flags.merge(meta, on=["shot", "interval"], how="left", validate="1:1")
    record = {
        "classes": list(confident.CLASSES),
        "kept_windows": int(keep.sum()),
        "kept_shots": int(windows.shot[keep].nunique()),
        "thresholds": {
            c: float(t)
            for c, t in zip(
                confident.CLASSES,
                confident.class_thresholds(probs[keep], labels[keep]),
                strict=True,
            )
        },
        "confident_joint": joint.tolist(),
        "calibrated_joint": np.round(calibrated, 1).tolist(),
        "estimated_label_noise": float(off / calibrated.sum()),
        "intervals_scored": len(flags),
        "intervals_flagged": int(flags.flagged.sum()),
        "by_status": summarise_group(flags, "status"),
        "by_source": summarise_group(flags, "sources"),
        "pairs": {
            f"{a}->{b}": {
                "flagged": int(
                    ((flags.given == a) & (flags.other == b))[flags.flagged].sum()
                ),
                "of_given": int((flags.given == a).sum()),
            }
            for a, b in PAIRS
        },
    }
    return record, flags


def add_ours(flags: pd.DataFrame, ours_dir: Path) -> pd.DataFrame:
    """Share of each interval's interior bins that confine-ours (held out by shot) puts
    in the given class and in the class the BES network prefers; NaN without a
    prediction for the shot."""
    probs: dict[int, np.ndarray] = {}
    for path in sorted(ours_dir.glob("fold*_pred.npz")):
        with np.load(path) as d:
            probs.update({int(k[1:]): d[k] for k in d.files})
    index = {c: i for i, c in enumerate(confident.CLASSES)}
    given, other = [], []
    for r in flags.itertuples():
        p = probs.get(int(r.shot))
        lo, hi = int(np.ceil(r.t_start + MARGIN_MS)), int(np.floor(r.t_end - MARGIN_MS))
        if p is None or hi <= lo:
            given.append(np.nan)
            other.append(np.nan)
            continue
        guess = p[:, lo : min(hi, p.shape[1])].argmax(0)
        given.append(float((guess == index[r.given]).mean()))
        other.append(
            float((guess == index[r.other]).mean()) if r.other in index else np.nan
        )
    return flags.assign(ours_share_given=given, ours_share_other=other)


def flagged_list(flags: pd.DataFrame, limit: int) -> dict[str, list[dict]]:
    cols = [
        "shot",
        "interval",
        "t_start",
        "t_end",
        "given",
        "other",
        "windows",
        "share_given",
        "share_other",
        "ours_share_given",
        "ours_share_other",
        "sources",
        "status",
    ]
    top = flags[flags.flagged].sort_values("share_other", ascending=False)
    qh_wp = top[top.given.isin(["QH", "WP"]) & top.other.isin(["QH", "WP"])]
    out = {
        "all_flagged": top.head(limit),
        "qh_wp": qh_wp.head(limit),
    }
    return {
        k: json.loads(v[cols].round(3).to_json(orient="records"))
        for k, v in out.items()
    }


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--row", default="full_cum_abcdrge")
    ap.add_argument("--runs-dir", type=Path, default=WORK / "ablation")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument(
        "--ours-dir",
        type=Path,
        default=WORK / "ours_runday",
        help="fold predictions of confine-ours (run-day folds), the second opinion",
    )
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args(argv)
    pred = load_predictions(args.runs_dir / args.row, args.allow_partial)
    intervals = bw.curated_intervals()
    windows = windows_table(pred, intervals)
    probs = windows[["p_L", "p_H", "p_QH", "p_WP"]].to_numpy()
    record, flags = analyse(windows, probs, intervals)
    flags = add_ours(flags, args.ours_dir)
    # ours also puts less than half of the interval in the labelled class
    both = flags[flags.flagged & (flags.ours_share_given < 0.5)]
    record["flagged_and_confine_ours_disagrees_too"] = {
        "intervals": len(both),
        "of_flagged": int(flags.flagged.sum()),
        "qh_wp": int(
            (both.given.isin(["QH", "WP"]) & both.other.isin(["QH", "WP"])).sum()
        ),
        "same_other_class_as_bes": int((both.ours_share_other > 0.5).sum()),
        "shots": sorted({int(s) for s in both.shot}),
    }
    record.update(
        {
            "row": args.row,
            "folds": int(pred.split.nunique()),
            "margin_ms": MARGIN_MS,
            "min_windows": MIN_WINDOWS,
            "min_share": MIN_SHARE,
            "git": git_sha(),
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "flagged": flagged_list(flags, args.limit),
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"confident_{args.row}.json").write_text(
        json.dumps(record, indent=1)
    )
    full = WORK / "confident"
    full.mkdir(parents=True, exist_ok=True)
    flags.to_csv(full / f"intervals_{args.row}.csv", index=False)
    print(
        f"{args.row}: {record['intervals_flagged']} of {record['intervals_scored']} "
        f"intervals flagged; estimated label noise "
        f"{100 * record['estimated_label_noise']:.1f} %"
    )
    for pair, v in record["pairs"].items():
        print(f"  {pair}: {v['flagged']} of {v['of_given']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
