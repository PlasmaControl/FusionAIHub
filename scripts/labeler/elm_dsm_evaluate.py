#!/usr/bin/env python
"""Score the lab's ELM survival model (`elm-dsm`) on the reviewed ELM spans.

    python scripts/labeler/elm_dsm_evaluate.py --run cv2 [--out-dir DIR] [--rescore]

The model is the labeler's refit of the lab's ELM time-to-event Deep Survival Machine
(`labeler.models.d3d_elm_time_to_event_dsm`, columns without BES). Three readings,
described in `labeler.elm.dsm`:

1. **Own target** (`own_target`): the limited-input refit on its original early-stopping
   validation rows (upstream calls them test),
   split, AUROC at 5, 10, 20 and 50 ms of "an ELM within `h` ms" with 95 % shot
   intervals; the point values are checked against the training record's.
2. **elm-dsm refit**: an offline 50 ms forward-risk score with 25 ms centered-NBI
   lookahead, not a causal forecast. The model is not retrained; the hard call
   uses the threshold that maximises F1 on the fold's inner-validation shots (the same
   shots `elm-ours` used), applied to the fold's held-out shots.
3. **elm-dsm-detect, objective changed to detection**: the same inputs and embedding
   with one logit head, trained on the reviewed spans to say whether the 50 ms ending
   at a row is present, on the same shot-grouped folds and inner-validation shots as
   `elm-ours`, a threshold per fold from its inner-validation shots. A variant starts
   from the limited-input refit's embedding (`elm-dsm-detect-init`).

All are compared with `elm-ours`, ELM-O (where it runs) and the ELM clock on common
bins (`labeler.elm.compare`); reviewed-label detector CV excludes cohort test shots.
Original-refit phase IDs are decoded separately for its pretraining overlap audit.
The fitted scores are saved
under `$LABELER_ROOT/round4/elm/dsm/` so the reference-swap script reuses them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from labeler.config import Paths, git_sha, sha256_of
from labeler.elm import compare, dsm, methods, score, train
from labeler.models.d3d_elm_time_to_event_dsm import spec
from labeler.models.runners import dsm_pickle

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "labeler" / "elm" / "dsm"
NAME = compare.NAME
RECORDED_TOL = 5e-4
MODEL_CONTEXT_SCOPE = (
    "This record scores the limited-input adaptation; the original 124-input "
    "native 1 ms checkpoint is evaluated separately in native_evaluation.json "
    "(four exact-export scored shots and a 33-shot reconstructed sensitivity panel)."
)


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


def fit_detectors(rows, data, oof, cfg, work, log):
    """Confirmatory detector, with training-only normalization and random weights."""
    spans = {s: d.spans for s, d in data.items()}
    bins = {s: d.bins for s, d in data.items()}
    out = {}
    for key in ("detect",):
        scores, thr, records = {}, {}, []
        for k in range(len(oof.record["folds"])):
            info = json.loads((oof.dir / f"fold{k}" / "fold.json").read_text())
            fold_cfg = replace(cfg, seed=cfg.seed + 100 * k)
            norm = dsm.fit_detection_normalization(rows, spans, info["train"])
            fold_rows = {
                s: dsm.normalize_detection_rows(r, norm) for s, r in rows.items()
            }
            state, t, history = dsm.fit_fold(
                rows, spans, bins, info["train"], info["inner_val"], fold_cfg
            )
            model = dsm.Detector(dropout=cfg.dropout)
            model.load_state_dict(state)
            for s in info["test"]:
                scores[s] = dsm.predict(model, fold_rows[s].x)
                thr[s] = float(t)
            best = max(history, key=lambda h: h["val_auprc"])
            target = work / "isolated" / f"fold{k}"
            target.mkdir(parents=True, exist_ok=True)
            torch.save(state, target / "detector.pt")
            (target / "normalization.json").write_text(json.dumps(norm, indent=1))
            records.append(
                {
                    "fold": k,
                    "threshold": float(t),
                    "best_epoch": best["epoch"],
                    "val_auprc": best["val_auprc"],
                    "val_f1": best["val_f1"],
                    "train_shots": len(info["train"]),
                    "inner_val_shots": len(info["inner_val"]),
                    "train_shot_ids": info["train"],
                    "inner_val_shot_ids": info["inner_val"],
                    "test_shot_ids": info["test"],
                    "seed": fold_cfg.seed,
                    "normalization": norm,
                    "checkpoint": str(target / "detector.pt"),
                    "checkpoint_sha256": sha256_of(target / "detector.pt"),
                    "normalization_sha256": sha256_of(target / "normalization.json"),
                    "raw_rows_sha256": {
                        str(s): dsm.rows_digest(rows[s])
                        for s in info["train"] + info["inner_val"] + info["test"]
                    },
                    "initialization": "independent seeded random weights; no source parameters",
                    "source_parameters_reused": False,
                    "source_normalization_reused": False,
                    "history": history,
                }
            )
            log(f"{NAME[key]} fold {k}: epoch {best['epoch']}, threshold {t:.4f}")
        out[NAME[key]] = (scores, thr, records)
    return out


def published_thresholds(rows, risk, data, oof):
    """Per shot: the fold's F1-maximising threshold of the refit's 50 ms risk on the
    fold's inner-validation bins (the forecast row of each bin), and fold records."""
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
        records.append(
            {
                "fold": k,
                "display_name": dsm.DISPLAY_NAME,
                "threshold": float(t),
                "val_f1": float(f1),
                "inner_val_shots": info["inner_val"],
                "test_shots": info["test"],
                "val_risk_quantiles": dsm.risk_quantiles(np.concatenate(sc)),
            }
        )
    return thr, records


def evaluate_set(
    sdef, data, oof, dscores, elmo_spans, clock_spans, boot, elmo_sweep=None
) -> dict:
    parts, bins_of = compare.common_parts(
        sdef, data, oof, dscores, elmo_spans, clock_spans, elmo_sweep
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
    for name, row in out["methods"].items():
        row["display_name"] = compare.DISPLAY_NAME.get(name, name)
        if row["point"].get("recall", 0) >= 0.99:
            row["high_recall"] = True
    out["risk_quantiles_scored_forecast_bins"] = dsm.risk_quantiles(
        np.concatenate(
            [
                dscores.risk[s][dsm.row_index(bins_of[s], compare.FORECAST_LAG_ROWS), -1]
                for s in sdef.shots
            ]
        )
    )
    # The refit's other survival horizons (continuous score only).
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
    out["annotation_modes"] = methods.annotation_summary(
        parts, {s: data[s].spans for s in sdef.shots}
    )
    out["kind_metrics"] = {
        name: methods.kind_summary(part, boot) for name, part in parts.items()
    }
    if sdef.has_elmo:
        out["elmo_score_source"] = (
            "maximum hit-eta threshold of an overlapping swept detection on each "
            "common bin; hard calls retain the paper setting"
        )
    return out


def by_co2(sdef, parts, dscores) -> dict:
    """The DSM methods and `elm-ours` on the shots whose CO2 density was served and on
    those where it was not (mean-filled)."""
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
            out[tag][name] = {
                "display_name": compare.DISPLAY_NAME[name],
                "point": res["point"],
                "ci95": res["ci95"],
            }
    return out


def row_table(rows) -> dict:
    out = {}
    for s, r in rows.items():
        out[str(s)] = {
            "usable_rows": int(r.usable.sum()),
            "in_filter_rows": int((r.usable & r.in_filter).sum()),
            "outside_training_filter_usable_rows": int((r.usable & ~r.in_filter).sum()),
            "outside_training_filter_usable_row_share": float(
                (r.usable & ~r.in_filter).sum() / r.usable.sum()
            ) if r.usable.any() else None,
            "missing_features": list(r.missing),
            "archive_features": "archive" in set(r.resolvers.values()),
            "resolvers": r.resolvers,
            "mean_filled_columns": len(r.filled),
            "mean_filled_column_names": list(r.filled),
            "rows_sha256": dsm.rows_digest(r),
            "source_signature": r.source_signature,
        }
    return out


def provenance(paths, work, oof, rows):
    files = {
        "script": Path(__file__),
        "dsm_library": REPO / "src/labeler/elm/dsm.py",
        "compare_library": REPO / "src/labeler/elm/compare.py",
        "methods_library": REPO / "src/labeler/elm/methods.py",
        "score_library": REPO / "src/labeler/elm/score.py",
        "review_labels": paths.label_tables / "edge_localized_mode/review/labels.csv",
        "cohort": paths.catalog / "cohort.csv",
        "refit_checkpoint": paths.models / dsm.SLUG / spec.ARTIFACTS[0],
        "normalization": paths.models / dsm.SLUG / spec.ARTIFACTS[1],
        "refit_training_record": paths.models / dsm.SLUG / "training_no_bes.json",
        "fold_run": oof.dir / "run.json",
    }
    for k in range(len(oof.record["folds"])):
        files[f"fold{k}"] = oof.dir / f"fold{k}/fold.json"
    return {
        "producing_git": git_sha(full=True),
        "files": {
            name: {"path": str(path), "sha256": sha256_of(path)}
            for name, path in files.items()
        },
        "rows_sha256": {str(s): dsm.rows_digest(r) for s, r in rows.items()},
        "score_files": {
            name: {"path": str(work / name), "sha256": sha256_of(work / name)}
            for name in (
                "published_risk.npz",
                "scores_elm-dsm-detect.npz",
                "scores_elm-dsm-detect-init.npz",
                "scores_elm-dsm-detect-exposed.npz",
                "thresholds.json",
            )
            if (work / name).exists()
        },
    }


def annotate_prefetch_record():
    """Accurately label the retained pre-fetch result; leave all metrics unchanged."""
    path = OUT / "prefetch_evaluation.json"
    if not path.exists():
        return
    old = json.loads(path.read_text())
    if "row_diagnostics" in old:
        return
    old["display_name"] = dsm.DISPLAY_NAME
    old["method_display_names"] = compare.DISPLAY_NAME
    old["record_note"] = (
        "Archived pre-Ip/Bt-fetch limited-input refit evaluation; original metric "
        "values retained. Historical individual-kind metrics mean non-crowd spans."
    )
    old["own_target"]["display_name"] = dsm.DISPLAY_NAME
    for subset in old["sets"].values():
        for name, res in subset["methods"].items():
            res["display_name"] = compare.DISPLAY_NAME.get(name, name)
    baseline = Paths.from_env().root / "round4/elm/dsm/prefetch_baseline"
    old_rows = {int(s): dsm.load_rows(int(s), baseline / f"{s}.npz") for s in old["rows"]}
    if all(r is not None for r in old_rows.values()):
        with np.load(baseline / "published_risk.npz") as z:
            old_risk = {int(k[1:]): z[k] for k in z.files}
        old["row_diagnostics"] = dsm.row_diagnostics(old_rows, old_risk)
    old["archived_score_store"] = str(baseline)
    path.write_text(json.dumps(old, indent=1))


def verify_rescore(previous: dict, current: dict) -> dict:
    """Confirm all scientific results survive reload of the fitted score artifacts."""
    keys = (
        "own_target", "published_thresholds", "detector_config", "detectors",
        "rows", "row_diagnostics", "sets",
    )
    checks = {}
    for key in keys:
        old = json.dumps(previous[key], sort_keys=True)
        new = json.dumps(current[key], sort_keys=True)
        checks[key] = {
            "identical": old == new,
            "fit_result_sha256": hashlib.sha256(old.encode()).hexdigest(),
            "rescore_result_sha256": hashlib.sha256(new.encode()).hexdigest(),
        }
    return {
        "display_name": dsm.DISPLAY_NAME,
        "fit_producing_git": previous["git"],
        "rescore_git": current["git"],
        "fit_used_rescore_flag": previous["rescored_from_saved_scores"],
        "exact_results_reproduced": all(c["identical"] for c in checks.values()),
        "checks": checks,
    }


def refresh_own_target(paths: Paths, out_dir: Path) -> int:
    """Correct phase IDs/own-target intervals without fitting reviewed-label models.

    Preserve the original no-rescore and exact-rescore proof as separate snapshots.
    Correct both result records identically, retain their original producing git, and
    record the later correction revision. All reviewed-bin metrics remain unchanged.
    """
    work = paths.root / "round4/elm/dsm"
    result_path = out_dir / "evaluation.json"
    trained_path = out_dir / "evaluation_trained.json"
    result = json.loads(result_path.read_text())
    trained = json.loads(trained_path.read_text())
    cohort = pd.read_csv(paths.catalog / "cohort.csv")
    own = own_target(paths)
    reviewed = sorted(int(s) for s in trained["rows"])
    own.update(dsm.split_overlap(own, reviewed, cohort))
    correction = {
        "git": git_sha(full=True),
        "script_sha256": sha256_of(__file__),
        "dsm_sha256": sha256_of(REPO / "src/labeler/elm/dsm.py"),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "scope": (
            "phase-ID decoding, original-refit overlap, physical-shot own-target "
            "bootstrap and early-stopping selection interpretation; no new model fit"
        ),
        "original_no_rescore_producing_git": trained["git"],
        "reviewed_bin_results_unchanged": result["sets"] == trained["sets"],
        "upstream_identifier_source": (
            "/projects/EKOLEMEN/wpqh_elm_hiro/hiro_scripts/data_processing.ipynb:"
            "4321-4326 (<shot>_<phase> strings), and the original split pickle"
        ),
        "early_stopping_source": (
            str(paths.models / dsm.SLUG / "training_no_bes.json") + ":split_note"
        ),
    }
    files = [
        result_path, trained_path, out_dir / "prefetch_evaluation.json",
        out_dir / "reproducibility.json", work / "fits.json",
    ]
    for path in files:
        if path.exists():
            snapshot = path.with_name(path.stem + "_before_phase_fix.json")
            if not snapshot.exists():
                snapshot.write_bytes(path.read_bytes())
    correction["original_snapshots"] = {
        str(path): {
            "path": str(path.with_name(path.stem + "_before_phase_fix.json")),
            "sha256": sha256_of(path.with_name(path.stem + "_before_phase_fix.json")),
        }
        for path in files if path.exists()
    }
    for path in (result_path, trained_path, out_dir / "prefetch_evaluation.json", work / "fits.json"):
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        record["own_target"] = own
        record["own_target_correction"] = correction
        if "cohort_test_shots_used" in record:
            record["cohort_test_shots_used_scope"] = (
                "reviewed-label detector CV only; prior-refit original split overlap "
                "is reported in own_target.cohort_physical_shot_overlap"
            )
        path.write_text(json.dumps(record, indent=1))
    proof = verify_rescore(
        json.loads(trained_path.read_text()), json.loads(result_path.read_text())
    )
    proof.update(
        {
            "fit_record": str(trained_path),
            "rescore_record": str(result_path),
            "original_proof": str(out_dir / "reproducibility_before_phase_fix.json"),
            "correction": correction,
            "verification_scope": (
                "original fit/rescore reviewed-bin results retained; identical own-target "
                "correction applied to both records; not a new fit or full rescore"
            ),
        }
    )
    (out_dir / "reproducibility.json").write_text(json.dumps(proof, indent=1))
    (out_dir / "own_target_correction.json").write_text(
        json.dumps({"correction": correction, "own_target": own}, indent=1)
    )
    print(json.dumps({"split_counts": own["split_physical_shot_counts"], "reviewed_overlap": own["reviewed_shot_ids_in_published_split"], "cohort_overlap": own["cohort_physical_shot_overlap"]}))
    return 0


def refresh_context(out_dir: Path) -> int:
    """Correct companion-record scope without recomputing any scientific result."""
    correction = {
        "scope": "metadata only: link the separate native checkpoint comparison",
        "git": git_sha(full=True),
        "script_sha256": sha256_of(__file__),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "scientific_results_unchanged": True,
    }
    for name in ("evaluation.json", "evaluation_trained.json"):
        path = out_dir / name
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        record["model_context"]["scope"] = MODEL_CONTEXT_SCOPE
        record["model_context"]["native_comparison_record"] = str(
            out_dir / "native_evaluation.json"
        )
        record["context_correction"] = correction
        path.write_text(json.dumps(record, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="the `labeler.elm.train` run (folds)")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--epochs", type=int, default=dsm.FitConfig.epochs)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--refresh-context", action="store_true")
    ap.add_argument(
        "--rescore",
        action="store_true",
        help="score the saved fits (`round4/elm/dsm/`) again without training",
    )
    ap.add_argument(
        "--refresh-own-target",
        action="store_true",
        help="correct original-refit phase IDs and own-target intervals without fitting",
    )
    ap.add_argument(
        "--refresh-report",
        action="store_true",
        help="rescore saved fits with updated disclosures and span metrics",
    )
    args = ap.parse_args(argv)
    paths = Paths.from_env()
    if args.refresh_context:
        return refresh_context(args.out_dir)
    if args.refresh_own_target:
        return refresh_own_target(paths, args.out_dir)
    if args.refresh_report and not args.rescore:
        ap.error("--refresh-report requires --rescore")
    annotate_prefetch_record()
    work = paths.root / "round4" / "elm" / "dsm"
    torch.set_num_threads(4)
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            12 * 1024**3 / torch.cuda.get_device_properties(0).total_memory, 0
        )
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
        dscores = compare.DsmScores.load(work, rows, variants=tuple(fits["detectors"]))
    else:
        print("own-target scoring of " + dsm.DISPLAY_NAME, flush=True)
        own = own_target(paths)
        own.update(
            dsm.split_overlap(own, shots, pd.read_csv(paths.catalog / "cohort.csv"))
        )
        cfg = dsm.FitConfig(epochs=args.epochs, device=args.device)
        pub_thr, pub_records = published_thresholds(rows, risk, data, oof)
        previous = json.loads(fits_json.read_text()) if fits_json.exists() else {}
        historical_sources = dsm.historical_detector_sources(previous.get("detectors", {}))
        historical = (
            compare.DsmScores.load(work, rows, variants=tuple(historical_sources.values()))
            if historical_sources else None
        )
        for path in (fits_json, args.out_dir / "evaluation.json"):
            archive = path.with_name(path.stem + "_exposed.json")
            if path.exists() and not archive.exists() and historical_sources:
                archive.write_bytes(path.read_bytes())
        raw_rows = load_rows(paths, shots, work / "raw_rows", None)
        trained = fit_detectors(
            raw_rows, data, oof, cfg, work, lambda m: print(m, flush=True)
        )
        dscores = compare.DsmScores(rows, risk)
        dscores.threshold[NAME["dsm"]] = pub_thr
        for name, old_name in historical_sources.items():
            dscores.scores[name] = historical.scores[old_name]
            dscores.threshold[name] = historical.threshold[old_name]
        for name, (scores, thr, _) in trained.items():
            dscores.scores[name] = scores
            dscores.threshold[name] = thr
        dscores.save(work)
        # Evaluate the same serialized arrays as --rescore, including risk precision.
        dscores = compare.DsmScores.load(work, rows, variants=tuple(dscores.scores))
        fits = {
            "display_name": dsm.DISPLAY_NAME,
            "method_display_names": compare.DISPLAY_NAME,
            "provenance": provenance(paths, work, oof, rows),
            "own_target": own,
            "published_thresholds": pub_records,
            "detector_config": cfg.__dict__,
            "detectors": {
                name: {
                    "folds": records,
                    "scores": str(work / f"scores_{name}.npz"),
                    "role": "confirmatory source-isolated detection baseline",
                    "source_parameters_reused": False,
                    "source_normalization_reused": False,
                }
                for name, (_, _, records) in trained.items()
            },
        }
        for name, old_name in historical_sources.items():
            fits["detectors"][name] = {
                **previous["detectors"][old_name],
                "scores": str(work / f"scores_{name}.npz"),
                "role": "historical exposed supplemental comparison",
                "source_normalization_reused": True,
                "source_parameters_reused": name == NAME["init"],
                "original_fit_provenance": previous["detectors"][old_name].get(
                    "original_fit_provenance", previous.get("provenance")
                ),
            }
        fits_json.write_text(json.dumps(fits, indent=1))
    own = fits["own_target"]
    refit_training = json.loads(
        (paths.models / dsm.SLUG / "training_no_bes.json").read_text()
    )

    sets = compare.load_sets(paths, data)
    elmo_spans, clock_spans = compare.load_detected(paths)
    elmo_sweep = compare.load_elmo_sweep(paths)
    record = {
        "git": git_sha(full=True),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": args.run,
        "reference": "expert-reviewed spans (review/labels.csv)",
        "cohort_test_shots_used": 0,
        "cohort_test_shots_used_scope": (
            "reviewed-label detector CV only; original-refit overlap is reported separately"
        ),
        "rescored_from_saved_scores": bool(args.rescore),
        "display_name": dsm.DISPLAY_NAME,
        "method_display_names": compare.DISPLAY_NAME,
        "provenance": fits.get("provenance"),
        "evaluation_provenance": provenance(paths, work, oof, rows),
        "evaluation_role": "saved-score rescore" if args.rescore else "fresh fit",
        "own_target_correction": fits.get("own_target_correction"),
        "prefetch_evaluation": str(OUT / "prefetch_evaluation.json"),
        "model_context": {
            "original_input_columns": 124,
            "refit_input_columns": dsm.N_COLUMNS,
            "refit_checkpoint_epochs": refit_training["best_epoch"] + 1,
            "refit_run_epochs": refit_training["epochs_run"],
            "refit_best_epoch": refit_training["best_epoch"],
            "refit_checkpoint_note": "one-epoch checkpoint selected from a seven-epoch run",
            "photodiode_columns": "pcphd02 and pcphd03 mean-filled on every shot",
            "legacy_training_source": "wpqh_elm_hiro legacy onset/survival labels",
            "serving_changes": "50 ms-mean serving on a 25 ms grid of a 1 ms-trained "
            "model; no D-alpha input (pcphd02/03 mean-filled); CO2 missing on 75/119; "
            "|z| clipped at 10",
            "training_resolution_note": "1 ms refers to the source survival fit; "
            "detection heads are refitted on reviewed 50 ms-mean rows.",
            "serving": dsm.SERVING,
            "preprocessing_exposure": dsm.PREPROCESSING_EXPOSURE,
            "temporal_interpretation": "Offline risk score with 25 ms centered-NBI "
            "lookahead (not a causal forecast)",
            "normalization_exposure": "The confirmatory elm-dsm detection fits "
            "normalization only on optimizer-training shots in each fold and starts "
            "from independent random weights. Refit and historical detection exposed/"
            "init reuse upstream pre-split statistics including blind-cohort shots "
            "190646 and 190532; historical init additionally reuses refit weights.",
            "normalization_source": "/projects/EKOLEMEN/wpqh_elm_hiro/hiro_scripts/"
            "data_processing.ipynb:4406 (normalization before upstream split)",
            "scope": MODEL_CONTEXT_SCOPE,
            "native_comparison_record": str(args.out_dir / "native_evaluation.json"),
        },
        "own_target": own,
        "published_thresholds": fits["published_thresholds"],
        "detector_config": fits["detector_config"],
        "detectors": fits["detectors"],
        "rows": row_table(rows),
        "row_diagnostics": dsm.row_diagnostics(rows, dscores.risk),
        "sets": {},
    }
    for name, sdef in sets.items():
        boot = score.draws(len(sdef.shots))
        record["sets"][name] = evaluate_set(
            sdef, data, oof, dscores, elmo_spans, clock_spans, boot, elmo_sweep
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "evaluation.json").write_text(json.dumps(record, indent=1))
    trained_record = args.out_dir / "evaluation_trained.json"
    if not args.rescore:
        trained_record.write_text(json.dumps(record, indent=1))
    elif trained_record.exists() and not args.refresh_report:
        repeated = verify_rescore(json.loads(trained_record.read_text()), record)
        repeated["fit_record"] = str(trained_record)
        repeated["rescore_record"] = str(args.out_dir / "evaluation.json")
        (args.out_dir / "reproducibility.json").write_text(json.dumps(repeated, indent=1))
        if not repeated["exact_results_reproduced"]:
            raise RuntimeError("DSM rescore differs from the no-rescore evaluation")
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
