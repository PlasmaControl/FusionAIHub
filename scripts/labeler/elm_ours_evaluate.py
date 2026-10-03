#!/usr/bin/env python
"""Score `elm-ours` against the reviewed ELM spans, beside ELM-O and the ELM clock.

    python scripts/labeler/elm_ours_evaluate.py --run cv2 [--out-dir DIR]

`elm-ours` (`labeler.elm`) is a 1D U-Net over filterscope D-alpha and line density.
Its predictions here are the out-of-fold ones of `python -m labeler.elm.train`: every
reviewed shot is predicted by the model of the fold it sat in, trained on other
shots, with the decision threshold (and the onset threshold) chosen on that fold's
inner-validation shots. No cohort test shot is read.

**Bins.** The benchmark's own: 50 ms bins wholly inside one absent, individual or
crowd span and inside analysed time (`labeler.elm.labels.scored_bins`, the rule of
`scripts/labeler/elmo_benchmark.bin_table`). Two sets:

* `bes73`: the 73 reviewed shots ELM-O runs on (it needs BES), analysed time = ELM-O's
  chunks (`review_coverage.csv`); `elm-ours`, ELM-O (the published setting) and the
  ELM clock are scored on identical bins. ELM-O's counts are checked against the
  benchmark's published ones.
* `all119`: every reviewed shot, analysed time = where the fetched D-alpha and
  density records have samples; `elm-ours` and the ELM clock (ELM-O cannot run).

**Scores.** Per bin: precision, recall, F1, false-alarm bin rate (hard calls: an
`elm-ours` bin is called present when its mean event probability reaches the fold's
threshold; ELM-O and the clock call a bin when a detected span touches it), AUROC and
AUPRC of the continuous score (ELM-O's from its threshold sweep). Per kind: the
share of crowd bins called, and the share of individual (single-ELM) spans a method
touches (for `elm-ours` the stretches where the same 50 ms mean reaches the same
threshold are its detected spans), plus the share of absent spans it touches. The
onset trace is scored separately: a detected onset (a peak of the onset head, the
start of a detected span for ELM-O and the clock) matches a reviewed individual-ELM
start within a tolerance. 95 % intervals are percentile intervals over 1000 shot
resamples, the same resamples for every method on a set, so differences are paired.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import elmo_benchmark as elmo

from labeler.config import Paths, git_sha
from labeler.elm import inputs, labels, methods, onset, prepare, score, train

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "labeler" / "elm" / "ours"
PUBLISHED_ELMO = {"tp": 2322, "fp": 443, "fn": 428, "tn": 3650}
NAMES = {"ours": "elm-ours", "elmo": "elm-elmo", "clock": "elm-clock"}


def areas(parts, draw) -> tuple[float, float]:
    truth, sc = score._pool(parts, draw)
    return score.roc_auc(truth, sc), score.average_precision(truth, sc)


def sweep_areas_boot(counts: np.ndarray, boot: np.ndarray):
    """ELM-O's AUROC/AUPRC (and the same per resample) from its threshold sweep."""
    strict = counts[:, ::-1]
    point = elmo.sweep_areas(strict.sum(axis=0))
    reps = np.array([elmo.sweep_areas(strict[d].sum(axis=0)) for d in boot])
    return point, reps


def evaluate_set(
    name, shots, data, bins_of, cover_of, oof, elmo_spans, clock_spans, boot
) -> dict:
    parts: dict[str, list[score.ShotScore]] = {"ours": [], "clock": []}
    if elmo_spans is not None:
        parts["elmo"] = []
    onsets: dict[str, dict[float, list]] = {m: {5.0: [], 10.0: []} for m in parts}
    for shot in shots:
        d, bins, cover = data[shot], bins_of[shot], cover_of[shot]
        cov0, cov1 = cover.t_start_ms.to_numpy(float), cover.t_end_ms.to_numpy(float)
        parts["ours"].append(
            methods.trace_part(
                d.spans, shot, bins, cover, oof.trace(shot)[0], oof.threshold[shot]
            )
        )
        c = clock_spans.get(shot, methods.span_frame([], []))
        parts["clock"].append(methods.span_part(d.spans, shot, bins, cover, c))
        found = {
            "ours": onset.peaks(oof.trace(shot)[1], oof.onset_threshold[shot])
            + inputs.GRID0_MS
            + 0.5,
            "clock": c.t_start_ms.to_numpy(float),
        }
        if elmo_spans is not None:
            e = elmo_spans.get(shot, methods.span_frame([], []))
            parts["elmo"].append(methods.span_part(d.spans, shot, bins, cover, e))
            found["elmo"] = e.t_start_ms.to_numpy(float)
        for m, f in found.items():
            for tol in onsets[m]:
                onsets[m][tol].append(
                    methods.onset_counts(
                        f, d.spans, d.dense.onset_mask, d.n_ms, cov0, cov1, tol
                    )
                )
    out: dict = {
        "shots": [int(s) for s in shots],
        "n_shots": len(shots),
        "bins": int(sum(len(parts["ours"][i].truth) for i in range(len(shots)))),
        "methods": {},
    }
    for m, plist in parts.items():
        res = score.summarise(plist, boot)
        out["methods"][NAMES[m]] = res
    if elmo_spans is not None:
        counts = out["methods"][NAMES["elmo"]]["counts"]
        out["elmo_counts_match_published"] = all(
            counts[k] == v for k, v in PUBLISHED_ELMO.items()
        )
    # the curve of ELM-O's threshold sweep supplies its AUROC and AUPRC
    paired: dict = {}
    if "elmo" in parts:
        sweep = pd.read_csv(Path(elmo.DEFAULT_WORK) / "review_sweep.csv.gz")
        cover_all = pd.concat(
            [cover_of[s].assign(shot=s) for s in shots], ignore_index=True
        )
        table = pd.concat([data[s].spans for s in shots], ignore_index=True)
        sweep_counts = elmo.sweep_counts(sweep, table, cover_all, list(shots))
        (a, p), reps = sweep_areas_boot(sweep_counts, boot)
        elmo_res = out["methods"][NAMES["elmo"]]
        elmo_res["point"]["auroc"], elmo_res["point"]["auprc"] = a, p
        elmo_res["ci95"]["auroc"] = score._ci(reps[:, 0])
        elmo_res["ci95"]["auprc"] = score._ci(reps[:, 1])
        elmo_res["score_source"] = "eta threshold sweep (elmo_benchmark.sweep_counts)"
        ours_reps = np.array([areas(parts["ours"], d) for d in boot])
        for i, metric in enumerate(("auroc", "auprc")):
            ours_point = out["methods"][NAMES["ours"]]["point"][metric]
            diff = ours_reps[:, i] - reps[:, i]
            paired[f"{NAMES['ours']} - {NAMES['elmo']}: {metric}"] = {
                "value": float(ours_point - elmo_res["point"][metric]),
                "ci95": score._ci(diff),
            }
    for other in ("elmo", "clock"):
        if other not in parts:
            continue
        for metric in (
            "f1",
            "precision",
            "recall",
            "false_alarm_bin_rate",
            "crowd_bin_recall",
            "individual_span_recall",
            "absent_span_alarm_rate",
        ):
            paired[f"{NAMES['ours']} - {NAMES[other]}: {metric}"] = (
                score.paired_difference(parts["ours"], parts[other], boot, metric)
            )
    out["paired"] = paired
    out["onset"] = {
        NAMES[m]: {
            f"tol_{int(tol)}ms": methods.onset_summary(np.array(rows), boot)
            for tol, rows in by_tol.items()
        }
        for m, by_tol in onsets.items()
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="a `labeler.elm.train` run name")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args(argv)
    paths = Paths.from_env()
    data = train.load(paths)
    oof = methods.Oof(paths.root / "round4" / "elm" / "cv" / args.run)
    shots_all = sorted(data)
    missing = [s for s in shots_all if s not in oof.fold_of]
    if missing:
        raise SystemExit(f"shots without out-of-fold predictions: {missing}")

    own_cover = {}
    for s in shots_all:
        c0, c1 = inputs.valid_intervals(
            np.load(prepare.inputs_dir(paths) / f"{s}.npy")[inputs.VALID]
        )
        own_cover[s] = methods.cover_frame(*labels.merge_intervals(c0, c1))
    own_bins = {s: data[s].bins for s in shots_all}

    cover = pd.read_csv(elmo.DEFAULT_WORK / "review_coverage.csv")
    shots_bes = sorted(int(s) for s in cover.shot.unique())
    elmo_cover = {
        s: methods.cover_frame(
            *labels.merge_intervals(
                *(g.sort_values("t_start_ms")[c] for c in ("t_start_ms", "t_end_ms"))
            )
        )
        for s, g in cover.groupby("shot")
    }
    elmo_bins = {
        s: labels.scored_bins(
            data[s].spans,
            elmo_cover[s].t_start_ms.to_numpy(float),
            elmo_cover[s].t_end_ms.to_numpy(float),
        )
        for s in shots_bes
    }
    found = pd.read_csv(elmo.DEFAULT_WORK / "review_elms.csv")
    found = found[found.variant == "paper"]
    elmo_spans = {
        int(s): g[["t_start_ms", "t_end_ms"]].reset_index(drop=True)
        for s, g in found.groupby("shot")
    }
    clock = pd.read_csv(elmo.ELM_CLOCK)
    clock = clock[clock.category == 1].rename(
        columns={"t_start": "t_start_ms", "t_end": "t_end_ms"}
    )
    clock_spans = {
        int(s): g[["t_start_ms", "t_end_ms"]].reset_index(drop=True)
        for s, g in clock.groupby("shot")
    }

    record = {
        "git": git_sha(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": args.run,
        "train_record": str(oof.dir / "run.json"),
        "reference": "expert-reviewed spans (review/labels.csv)",
        "cohort_test_shots_used": 0,
        "folds": oof.record["folds"],
        "event_thresholds": {
            r["fold"]: r["threshold"] for r in oof.record["fold_records"]
        },
        "onset_thresholds": {
            r["fold"]: r["onset_threshold"] for r in oof.record["fold_records"]
        },
        "config": oof.record["config"],
        "sets": {},
    }
    boot_bes = score.draws(len(shots_bes))
    record["sets"]["bes73"] = evaluate_set(
        "bes73",
        shots_bes,
        data,
        elmo_bins,
        elmo_cover,
        oof,
        elmo_spans,
        clock_spans,
        boot_bes,
    )
    boot_all = score.draws(len(shots_all))
    record["sets"]["all119"] = evaluate_set(
        "all119", shots_all, data, own_bins, own_cover, oof, None, clock_spans, boot_all
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "evaluation.json").write_text(json.dumps(record, indent=1))
    for name, s in record["sets"].items():
        print(name, s["n_shots"], "shots", s["bins"], "bins")
        for m, res in s["methods"].items():
            pt = res["point"]
            print(
                f"  {m:9s}",
                {k: round(v, 3) for k, v in pt.items() if k != "prevalence"},
            )
        if "elmo_counts_match_published" in s:
            print(
                "  elm-elmo counts match the published ones:",
                s["elmo_counts_match_published"],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
