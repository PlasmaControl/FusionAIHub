#!/usr/bin/env python
"""Score the lab's ELM survival model (`elm-dsm`) on the reviewed ELM spans.

    python scripts/labeler/elm_dsm_evaluate.py --run cv2 [--out-dir DIR] [--rescore]

The model is the labeler's refit of the lab's ELM time-to-event Deep Survival Machine
(`labeler.models.d3d_elm_time_to_event_dsm`, columns without BES). Three readings,
described in `labeler.elm.dsm`:

1. **Own target** (`own_target`): the published model on the test rows of its own
   split, AUROC at 5, 10, 20 and 50 ms of "an ELM within `h` ms" with 95 % shot
   intervals; the point values are checked against the training record's.
2. **elm-dsm, as published**: its 50 ms risk at the row before a bin, read as the bin's
   score ("an ELM starts in the next 50 ms"). The model is not retrained; the hard call
   uses the threshold that maximises F1 on the fold's inner-validation shots (the same
   shots `elm-ours` used), applied to the fold's held-out shots.
3. **elm-dsm-detect, objective changed to detection**: the same inputs and embedding
   with one logit head, trained on the reviewed spans to say whether the 50 ms ending
   at a row is present, on the same shot-grouped folds and inner-validation shots as
   `elm-ours`, a threshold per fold from its inner-validation shots. A variant starts
   from the published embedding (`elm-dsm-detect-init`).

All are compared with `elm-ours`, ELM-O (where it runs) and the ELM clock on common
bins (`labeler.elm.compare`); no cohort test shot is read. The fitted scores are saved
under `$LABELER_ROOT/round4/elm/dsm/` so the reference-swap script reuses them.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from labeler.config import Paths, git_sha
from labeler.elm import compare, dsm, methods, score, train
from labeler.models.d3d_elm_time_to_event_dsm import spec
from labeler.models.runners import dsm_pickle

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "labeler" / "elm" / "dsm"
NAME = compare.NAME
RECORDED_TOL = 5e-4


def own_target(paths: Paths) -> dict:
    model_dir = paths.models / dsm.SLUG
    res = dsm.legacy_own_target(model_dir, None)
    recorded = json.loads((model_dir / "training_no_bes.json").read_text())
    res["recorded_in_training_json"] = {}
    for key, row in res["horizons"].items():
        rec = recorded["metrics_test"][key]
        res["recorded_in_training_json"][key] = {
            "auroc": rec["auroc"],
            "agrees": abs(rec["auroc"] - row["auroc"]) < RECORDED_TOL
            and rec["n_cases"] == row["cases"],
        }
    return res


def load_rows(paths: Paths, shots, cache: Path, norm: dict):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(4) as pool:
        got = list(pool.map(lambda s: dsm.cached_rows(paths, s, norm, cache), shots))
    return dict(zip(shots, got, strict=True))


def fit_detectors(rows, data, oof, graph, cfg, log):
    """`{variant: ({shot: row scores}, {shot: threshold}, fold records)}`."""
    spans = {s: d.spans for s, d in data.items()}
    bins = {s: d.bins for s, d in data.items()}
    out = {}
    for key, init in (("detect", None), ("init", graph)):
        scores, thr, records = {}, {}, []
        for k in range(len(oof.record["folds"])):
            info = json.loads((oof.dir / f"fold{k}" / "fold.json").read_text())
            fold_cfg = replace(cfg, seed=cfg.seed + 100 * k)
            state, t, history = dsm.fit_fold(
                rows, spans, bins, info["train"], info["inner_val"], fold_cfg, init
            )
            model = dsm.Detector(dropout=cfg.dropout)
            model.load_state_dict(state)
            for s in info["test"]:
                scores[s] = dsm.predict(model, rows[s].x)
                thr[s] = float(t)
            best = max(history, key=lambda h: h["val_auprc"])
            records.append(
                {
                    "fold": k,
                    "threshold": float(t),
                    "best_epoch": best["epoch"],
                    "val_auprc": best["val_auprc"],
                    "val_f1": best["val_f1"],
                    "train_shots": len(info["train"]),
                    "inner_val_shots": len(info["inner_val"]),
                }
            )
            log(f"{NAME[key]} fold {k}: {records[-1]}")
        out[NAME[key]] = (scores, thr, records)
    return out


def published_thresholds(rows, risk, data, oof):
    """Per shot: the fold's F1-maximising threshold of the published 50 ms risk on the
    fold's inner-validation bins (the forecast row of each bin), and the fold records."""
    thr, records = {}, []
    for k in range(len(oof.record["folds"])):
        info = json.loads((oof.dir / f"fold{k}" / "fold.json").read_text())
        truth, sc = [], []
        for s in info["inner_val"]:
            b = dsm.bins_with_rows(
                data[s].bins, rows[s].usable, (compare.FORECAST_LAG_ROWS, 0)
            )
            kk = dsm.row_index(b, compare.FORECAST_LAG_ROWS)
            truth.append(b.truth)
            sc.append(risk[s][kk, -1])
        t, f1 = score.best_threshold(np.concatenate(truth), np.concatenate(sc))
        for s in info["test"]:
            thr[s] = float(t)
        records.append({"fold": k, "threshold": float(t), "val_f1": float(f1)})
    return thr, records


def evaluate_set(sdef, data, oof, dscores, elmo_spans, clock_spans, boot) -> dict:
    parts, bins_of = compare.common_parts(
        sdef, data, oof, dscores, elmo_spans, clock_spans
    )
    n_before = sum(len(sdef.bins[s].t0) for s in sdef.shots)
    n_after = sum(len(b.t0) for b in bins_of.values())
    out = {
        "shots": [int(s) for s in sdef.shots],
        "n_shots": len(sdef.shots),
        "bins": n_after,
        "bins_without_rows": n_before - n_after,
        "bins_before_restriction": n_before,
    }
    out.update(methods.summarise_methods(parts, boot, NAME["ours"]))
    # the published model's other horizons, read the same way (continuous score only)
    horizons = {}
    for j, h in enumerate(dsm.HORIZONS_MS):
        alt = []
        for s in sdef.shots:
            alt.append(
                methods.row_part(
                    data[s].spans,
                    s,
                    bins_of[s],
                    dsm.usable_cover(dscores.rows[s].usable, sdef.cover[s]),
                    dsm.ROW_T_MS,
                    dscores.risk[s][:, j],
                    0.5,
                    lag_rows=compare.FORECAST_LAG_ROWS,
                    ahead=True,
                )
            )
        horizons[f"risk_{int(h)}ms"] = methods.areas_summary(alt, boot)
    out["published_horizons"] = horizons
    out["by_co2_served"] = by_co2(sdef, parts, dscores)
    if sdef.has_elmo:
        out["note_elmo"] = (
            "elm-elmo's AUROC/AUPRC (threshold sweep) are not restricted to these "
            "bins; its hard-call metrics are"
        )
    return out


def by_co2(sdef, parts, dscores) -> dict:
    """The DSM methods and `elm-ours` on the shots whose CO2 density was served and on
    those where it was not (mean-filled), the model's most informative missing input."""
    served = [
        i
        for i, s in enumerate(sdef.shots)
        if not any(m.startswith("co2") for m in dscores.rows[s].missing)
    ]
    out = {}
    for tag, idx in (
        ("co2_served", served),
        ("co2_missing", [i for i in range(len(sdef.shots)) if i not in set(served)]),
    ):
        if len(idx) < 2:
            continue
        boot = score.draws(len(idx))
        out[tag] = {"shots": len(idx)}
        for name in (NAME["ours"], NAME["dsm"], NAME["detect"]):
            sub = [parts[name][i] for i in idx]
            res = score.summarise(sub, boot)
            out[tag][name] = {"point": res["point"], "ci95": res["ci95"]}
    return out


def row_table(rows) -> dict:
    out = {}
    for s, r in rows.items():
        out[str(s)] = {
            "usable_rows": int(r.usable.sum()),
            "in_filter_rows": int((r.usable & r.in_filter).sum()),
            "missing_features": list(r.missing),
            "archive_features": "archive" in set(r.resolvers.values()),
            "mean_filled_columns": len(r.filled),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="the `labeler.elm.train` run (folds)")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--epochs", type=int, default=dsm.FitConfig.epochs)
    ap.add_argument(
        "--rescore",
        action="store_true",
        help="score the saved fits (`round4/elm/dsm/`) again without training",
    )
    args = ap.parse_args(argv)
    paths = Paths.from_env()
    work = paths.root / "round4" / "elm" / "dsm"
    torch.set_num_threads(4)
    data = train.load(paths)
    oof = methods.Oof(paths.root / "round4" / "elm" / "cv" / args.run)
    norm = dsm.load_norm(paths)
    shots = sorted(data)
    rows = load_rows(paths, shots, work, norm)
    graph = dsm_pickle.load_dsm(paths.models / dsm.SLUG / spec.ARTIFACTS[0])
    risk = {s: dsm.published_risk(graph, rows[s].x) for s in shots}

    fits_json = work / "fits.json"
    if args.rescore:
        # the scores, thresholds and training records of an earlier run, rescored
        fits = json.loads(fits_json.read_text())
        dscores = compare.DsmScores.load(work, rows, variants=compare.VARIANTS)
    else:
        print("own-target scoring of the published model", flush=True)
        own = own_target(paths)
        overlap = {
            k: sorted(set(shots) & set(v)) for k, v in own["split_shots"].items()
        }
        own["reviewed_shots_in_published_split"] = {
            k: len(v) for k, v in overlap.items()
        }
        cfg = dsm.FitConfig(epochs=args.epochs)
        pub_thr, pub_records = published_thresholds(rows, risk, data, oof)
        trained = fit_detectors(
            rows, data, oof, graph, cfg, lambda m: print(m, flush=True)
        )
        dscores = compare.DsmScores(rows, risk)
        dscores.threshold[NAME["dsm"]] = pub_thr
        for name, (scores, thr, _) in trained.items():
            dscores.scores[name] = scores
            dscores.threshold[name] = thr
        dscores.save(work)
        fits = {
            "own_target": own,
            "published_thresholds": pub_records,
            "detector_config": cfg.__dict__,
            "detectors": {
                name: {"folds": records, "scores": str(work / f"scores_{name}.npz")}
                for name, (_, _, records) in trained.items()
            },
        }
        fits_json.write_text(json.dumps(fits, indent=1))
    own = fits["own_target"]

    sets = compare.load_sets(paths, data)
    elmo_spans, clock_spans = compare.load_detected(paths)
    record = {
        "git": git_sha(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": args.run,
        "reference": "expert-reviewed spans (review/labels.csv)",
        "cohort_test_shots_used": 0,
        "rescored_from_saved_scores": bool(args.rescore),
        "own_target": own,
        "published_thresholds": fits["published_thresholds"],
        "detector_config": fits["detector_config"],
        "detectors": fits["detectors"],
        "rows": row_table(rows),
        "sets": {},
    }
    for name, sdef in sets.items():
        boot = score.draws(len(sdef.shots))
        record["sets"][name] = evaluate_set(
            sdef, data, oof, dscores, elmo_spans, clock_spans, boot
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "evaluation.json").write_text(json.dumps(record, indent=1))
    for k, v in own["horizons"].items():
        print("own target", k, round(v["auroc"], 4), v["auroc_ci95"])
    for name, s in record["sets"].items():
        print(
            name,
            s["n_shots"],
            "shots",
            s["bins"],
            "bins",
            s["bins_without_rows"],
            "dropped",
        )
        for m, res in s["methods"].items():
            pt = res["point"]
            print(
                f"  {m:20s}",
                {k: round(v, 3) for k, v in pt.items() if k != "prevalence"},
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
