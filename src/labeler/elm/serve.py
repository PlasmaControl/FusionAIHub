"""Serve `elm-ours` on shots the reviewed folds never labelled: a 10 ms-grid table.

    python -m labeler.elm.serve --name NAME --shots S ... [--also S ...] [--out-dir DIR]

A shot runs whole through the five fold models of the frozen run (`train.predict`
zero-pads it to the network's length multiple); its score is the mean of their event
probabilities, and it is present where the 50 ms moving mean of that score reaches the
mean of the five fold thresholds (`labels.runs_of`, the span rule of the evaluation).
The ensemble and that threshold have no out-of-fold score on the reviewed shots, since
every one is in some fold's training set (`docs/labeler/elm_ours.md`, serving
protocol); `docs/labeler/elm_heldout.md` scores them on 81 shots a later review
labelled, none in any fold.

The inputs are the fetched records `$LABELER_ROOT/benchmarks/elm/elmo/signals/<shot>.npz`
(filterscopes FS02-04 and the fast chords DENV2F and DENV3F, native rate), read by
`inputs.read_channels`; nothing is fetched here. A shot is published only when it has the
record, both density chords survive the failed-digitiser screen, and the median level of
each filterscope chord lies between the 5th and 95th percentile of the reviewed shots
the network trained on, widened by one sd of the training's level shift
(`input_problems`); otherwise it is skipped, with its reason in the sidecar.

A reviewed shot of the run's folds (`fold_shots`) is never published: four of the five
fold models trained on its label. With `--min-present F`, a shot under that fraction
present is scored but not published (`scored_not_published` in the sidecar; the owner's
rule of 2026-10-11 is 0.05).

`--shots` are published to `<out-dir>/<name>.csv` (the interval columns of
`docs/labeler/events/README.md`: present spans, and absent between them inside the window of
valid input, on the 10 ms grid) with `<name>.meta.json`; `--also` shots are scored and
their traces saved under `$LABELER_ROOT/round4/elm/serve/<name>/`, but not published.
A table (or meta) already at that path is not overwritten unless `--replace`: the
run stops before scoring anything.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import Paths, git_sha, sha256_of
from ..events import grid
from ..events.interval_tables import (
    INTERVAL_COLUMNS,
    validate_intervals,
    write_interval_table,
)
from . import inputs, labels, net, prepare, train

RUN = "cv2"
FOLDS = 5
CATEGORY = "edge_localized_mode"
ABSENT, PRESENT = 0, 1
#: Percentiles of the reviewed shots' per-chord median filterscope level that bound a
#: servable shot, widened by the run's level augmentation (the training shifts every
#: shot's D-alpha level by a normal offset, so levels that far beyond the reviewed
#: shots' own 5th-95th percentile were in the training inputs). The ends of the
#: reviewed range (a few shots each) are not enough support to bound it; the sidecar
#: records them beside the percentiles.
LEVEL_RANGE_PERCENTILES = (5.0, 95.0)
FULL_RANGE_PERCENTILES = (0.0, 100.0)


def load_models(run_dir: Path, device="cpu") -> tuple[list[net.ElmUNet], list[float]]:
    """The run's five fold models (eval mode) and their thresholds."""
    run_dir = Path(run_dir)
    dropout = json.loads((run_dir / "run.json").read_text())["config"]["dropout"]
    models, thresholds = [], []
    for k in range(FOLDS):
        model = net.ElmUNet(dropout=dropout)
        state = torch.load(
            run_dir / f"fold{k}/model.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(state)
        models.append(model.to(device).eval())
        info = json.loads((run_dir / f"fold{k}/fold.json").read_text())
        thresholds.append(float(info["threshold"]))
    return models, thresholds


def ensemble(models, x: np.ndarray, device="cpu") -> np.ndarray:
    """The mean event probability per ms of one shot's `(C, cells)` input."""
    return np.mean([train.predict(m, x, device)[0] for m in models], axis=0)


def level_medians(x: np.ndarray) -> np.ndarray:
    """Median log10 filterscope level of FS02-04 over the shot's valid cells."""
    valid = x[inputs.VALID] > 0
    level = x[list(inputs.FS_LEVEL)][:, valid] * inputs.FS_SCALE + inputs.FS_CENTRE
    return np.median(level, axis=1) if level.shape[1] else np.full(3, np.nan)


def fold_shots(run_dir: Path) -> set[int]:
    """The reviewed shots of the run's folds: each is in four folds' training."""
    folds = json.loads((Path(run_dir) / "run.json").read_text())["folds"]
    return {int(s) for fold in folds for s in fold}


def reference_levels(paths: Paths, run_dir: Path) -> np.ndarray:
    """`(shots, 3)` median levels of the reviewed shots the run's folds trained on."""
    shots = sorted(fold_shots(run_dir))
    return np.array(
        [level_medians(np.load(prepare.inputs_dir(paths) / f"{s}.npy")) for s in shots]
    )


def level_margin(run_dir: Path) -> float:
    """One sd of the run's D-alpha level augmentation, in decades."""
    config = json.loads((Path(run_dir) / "run.json").read_text())["config"]
    return float(config.get("level_shift", train.Config.level_shift)) * inputs.FS_SCALE


def input_problems(
    x: np.ndarray, reference: np.ndarray, margin: float = 0.0
) -> list[str]:
    """Why a shot's input is not servable; empty when it is.

    A filterscope chord's median level may lie `margin` decades beyond the
    `LEVEL_RANGE_PERCENTILES` of the reference shots' medians.
    """
    problems = []
    for name, row in zip(("DENV2F", "DENV3F"), inputs.DENSITY, strict=True):
        if not x[row].any():
            problems.append(f"{name} is empty or screened as a failed digitiser")
    lo, hi = np.percentile(reference, LEVEL_RANGE_PERCENTILES, axis=0)
    lo, hi = lo - margin, hi + margin
    level = level_medians(x)
    for j, name in enumerate(("FS02", "FS03", "FS04")):
        if not np.isfinite(level[j]):
            problems.append(f"{name} has no valid cell")
        elif not lo[j] <= level[j] <= hi[j]:
            problems.append(
                f"{name} median level 10^{level[j]:.2f} is outside the training range "
                f"10^{lo[j]:.2f}-10^{hi[j]:.2f}"
            )
    return problems


def sample_rates(record: Path) -> dict[str, float]:
    """The record's native sampling rates in kHz: filterscopes, then the fast chords."""
    with np.load(record) as z:
        return {
            "filterscope_khz": round(1 / float(np.median(np.diff(z["t_fs_ms"]))), 1),
            "interferometer_khz": round(
                1 / float(np.median(np.diff(z["t_int_ms"]))), 1
            ),
        }


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def shot_rows(
    shot: int, trace: np.ndarray, x: np.ndarray, threshold: float
) -> pd.DataFrame:
    """One shot's interval rows: present where the score reaches `threshold`, absent
    elsewhere inside each stretch of valid input, all on the 10 ms grid from t = 0.

    `confidence` is the mean ensemble probability of the row's own class over its
    milliseconds (the mean of one minus the probability for an absent row).
    """
    c0, c1 = labels.merge_intervals(*inputs.valid_intervals(x[inputs.VALID]))
    s0, s1 = labels.runs_of(trace, threshold)
    rows = []

    def row(category: int, a: int, b: int) -> None:
        i = int(np.clip(a - inputs.GRID0_MS, 0, trace.size))
        j = int(np.clip(b - inputs.GRID0_MS, i + 1, max(trace.size, i + 1)))
        p = float(np.mean(trace[i:j])) if trace.size else 0.0
        rows.append(
            (
                int(shot),
                category,
                float(a),
                float(b),
                round(p if category else 1.0 - p, 4),
            )
        )

    for a, b in zip(c0, c1, strict=True):
        lo_t, hi_t = max(float(a), 0.0), float(b)
        if hi_t <= lo_t:
            continue
        lo, hi = grid.snap_window(lo_t, hi_t)
        on = []
        for s, e in zip(s0, s1, strict=True):
            s, e = max(s, lo), min(e, hi)
            if e > s:
                u, v = grid.snap_span(s, e)
                u, v = max(u, lo), min(v, hi)
                if v > u:
                    on.append((u, v))
        cur = lo
        for u, v in _merge(on):
            if u > cur:
                row(ABSENT, cur, u)
            row(PRESENT, u, v)
            cur = v
        if cur < hi:
            row(ABSENT, cur, hi)
    return validate_intervals(pd.DataFrame(rows, columns=list(INTERVAL_COLUMNS)))


def present_fraction(rows: pd.DataFrame) -> float:
    length = rows.t_end - rows.t_start
    return (
        float(length[rows.category == PRESENT].sum() / length.sum())
        if len(rows)
        else 0.0
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--name", required=True, help="table name (<name>.csv, <name>.meta.json)"
    )
    ap.add_argument(
        "--shots", type=int, nargs="+", required=True, help="shots to publish"
    )
    ap.add_argument(
        "--also", type=int, nargs="*", default=[], help="score, do not publish"
    )
    ap.add_argument("--run", default=RUN)
    ap.add_argument(
        "--shot-list", help="name of the shot list, recorded in the sidecar"
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Paths.from_env().label_tables / CATEGORY / "extend_elm_ours",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--min-present",
        type=float,
        default=0.0,
        help="publish only shots at least this fraction present (scored otherwise)",
    )
    ap.add_argument(
        "--replace",
        action="store_true",
        help="overwrite an existing <name>.csv and its meta (refused otherwise)",
    )
    args = ap.parse_args(argv)
    out = args.out_dir / f"{args.name}.csv"
    existing = [p for p in (out, out.with_suffix(".meta.json")) if p.exists()]
    if existing and not args.replace:
        ap.error(
            f"{existing[0]} exists: a published table is not overwritten; "
            "name a new table, or pass --replace"
        )
    paths = Paths.from_env()
    run_dir = paths.root / "round4/elm/cv" / args.run
    models, thresholds = load_models(run_dir, args.device)
    threshold = float(np.mean(thresholds))
    reference = reference_levels(paths, run_dir)
    margin = level_margin(run_dir)
    trained = fold_shots(run_dir)
    traces = paths.root / "round4/elm/serve" / args.name
    traces.mkdir(parents=True, exist_ok=True)
    signals = prepare.signals_dir(paths)
    frames, skipped, records, near_empty = [], {}, {}, []
    for shot in [*args.shots, *args.also]:
        record = signals / f"{shot}.npz"
        if not record.is_file():
            gone = signals / f"{shot}.missing.json"
            skipped[shot] = (
                "fdp has no filterscope or interferometer record"
                if gone.is_file()
                else "no fetched record"
            )
            print(json.dumps({"shot": shot, "skipped": skipped[shot]}), flush=True)
            continue
        x = inputs.read_channels(record)
        trace = ensemble(models, x, args.device)
        np.savez_compressed(traces / f"{shot}.npz", p=trace.astype(np.float32))
        problems = input_problems(x, reference, margin)
        rows = shot_rows(shot, trace, x, threshold)
        fraction = present_fraction(rows)
        held = shot in trained or fraction < args.min_present
        print(
            json.dumps(
                {
                    "shot": shot,
                    "levels": level_medians(x).round(2).tolist(),
                    **sample_rates(record),
                    "present_fraction": round(fraction, 3),
                    "problems": problems,
                    "published": shot in args.shots and not problems and not held,
                }
            ),
            flush=True,
        )
        if shot not in args.shots:
            continue
        if problems:
            skipped[shot] = "; ".join(problems)
            continue
        if shot in trained:
            skipped[shot] = "a reviewed shot the run's folds trained on"
            continue
        if fraction < args.min_present:
            near_empty.append(shot)
            continue
        frames.append(rows)
        records[shot] = {"sha256": sha256_of(record), **sample_rates(record)}
    if not frames:
        print("nothing to publish", flush=True)
        return 1
    table = pd.concat(frames, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoints = [
        {
            "fold": k,
            "path": str(run_dir / f"fold{k}/model.pt"),
            "sha256": sha256_of(run_dir / f"fold{k}/model.pt"),
            "threshold": thresholds[k],
        }
        for k in range(FOLDS)
    ]
    meta = {
        "category": CATEGORY,
        "producer": "elm-ours",
        "status": "experimental",
        "model": "1D U-Net (401,714 parameters) over filterscopes FS02-04 and fast chords "
        "DENV2F/DENV3F; 5 shot-grouped folds of the 119 reviewed shots; "
        "docs/labeler/elm_ours.md",
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "made_by": "python -m labeler.elm.serve",
        "git_sha": git_sha(full=True),
        "code_sha256": sha256_of(Path(__file__)),
        "run": str(run_dir),
        "checkpoints": checkpoints,
        "threshold": threshold,
        "threshold_rule": "mean of the five fold thresholds; unevaluated",
        "score": "mean of the five fold models' event probability per ms, each shot run "
        "whole (zero-padded to the length multiple)",
        "spans": "present where the 50 ms centered moving mean of the score reaches the "
        "threshold (labels.runs_of), snapped to the 10 ms grid (events.grid.snap_span)",
        "window": "each stretch of valid input (filterscopes and both chords sampled) from "
        "t = 0 ms; absent rows fill it between the present spans",
        "confidence": "mean ensemble probability of the row's class over its milliseconds",
        "shot_list": args.shot_list or args.name,
        "shots_requested": sorted(args.shots),
        "shots_labelled": sorted(records),
        "shots_skipped": {
            str(s): why for s, why in sorted(skipped.items()) if s in args.shots
        },
        "input_records": {str(s): rec for s, rec in sorted(records.items())},
        "min_present": args.min_present,
        "input_check": {
            "reference": "median log10 filterscope level (FS02, FS03, FS04, over valid "
            f"cells) of the {len(reference)} reviewed shots the folds trained on",
            "range_percentiles": list(LEVEL_RANGE_PERCENTILES),
            "range": np.percentile(reference, LEVEL_RANGE_PERCENTILES, axis=0)
            .round(2)
            .tolist(),
            "margin_decades": round(margin, 3),
            "margin_rule": "one sd of the run's D-alpha level augmentation "
            "(level_shift x FS_SCALE), added beyond the range on both sides",
            "min_max": np.percentile(reference, FULL_RANGE_PERCENTILES, axis=0)
            .round(2)
            .tolist(),
        },
        "caveats": [
            (
                "the five-model ensemble and the mean threshold are unevaluated: every "
                "reviewed shot is in some fold's training set, and the headline numbers "
                "use the one fold model that never saw each shot"
            ),
            (
                "the label is occupancy of ELMy periods (97% of the training positives "
                "are crowd spans), not ELM onsets"
            ),
            "unreviewed model output; the owner's saved label wins over it on the review page",
            (
                "inputs are the native-rate fetched records, not the corpus 10 kHz "
                "resample; the reviewed shots' filterscopes are mostly 50 kS/s, and a shot "
                "recorded at another rate (input_records) has a different number of "
                "samples under each 0.1 ms maximum"
            ),
            (
                "shots come from run days the folds saw in training or from eras they did "
                "not (the reviewed shots span 185857-204937)"
            ),
        ],
    }
    if near_empty:
        meta["scored_not_published"] = {
            "shots": sorted(near_empty),
            "reason": f"under {args.min_present:.0%} present over their valid input; "
            "near-empty rows are not published (the owner's rule of 2026-10-11); the "
            f"five-fold mean traces are in round4/elm/serve/{args.name}/<shot>.npz",
        }
    write_interval_table(table, out, meta)
    print(
        f"wrote {out}: {len(table)} rows, {len(records)} shots, {len(skipped)} skipped"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
