#!/usr/bin/env python
"""Prepare, train and score the AE supervision swap without evaluation selection.

Use CUDA phase-3 Python for training, Pixi for CPU work, and LABELER_NO_FETCH=1.
``prepare`` freezes current inputs and
the 100/20/60 split. ``train`` reuses ae_train's architecture and full recipe.
``evaluate`` reports completed arms and saved Garcia predictions; unavailable
arms remain explicitly missing. No threshold is selected on evaluation shots.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ae_train
from ae_baselines_evaluate import AE_CLASSES, check_order, load_older, nearest_bin

from labeler.ae.supervision import (
    BOOTSTRAP_SEED,
    CONVERGENCE_RULE,
    N_FRAMES,
    RECORD_MS,
    SPLIT_SEED,
    ShotMetric,
    clean_split,
    clock_prior,
    convergence_screen,
    dense_states,
    frame_mean,
    paired_scores,
    selection_threshold,
    sha256,
    training_conformance,
    validate_split,
    within_shot_scores,
)
from labeler.events.catalog.states import ABSENT, PRESENT
from labeler.events.review.labels import read_labels

ROOT = Path(os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker"))
TABLES = Path(
    os.environ.get(
        "LABELER_LABEL_TABLES", "/scratch/gpfs/nc1514/FusionAIHub/data/events"
    )
)
SUPERVISIONS = ("legacy", "dense", "threeway")


def write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temp.replace(path)


def provenance() -> dict:
    git = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "generated": datetime.now(UTC).isoformat(),
        "git": git,
        "code_sha256": {
            str(path.relative_to(REPO)): sha256(path)
            for path in (
                Path(__file__),
                REPO / "scripts/labeler/ae_train.py",
                REPO / "src/labeler/ae/supervision.py",
                REPO / "src/labeler/ae/model.py",
                REPO / "scripts/labeler/ae_supervision_swap_report.py",
            )
        },
    }


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    validate_split(manifest["split"], set(manifest["blind_gold_shots"]))
    for key in ("dense", "cohort"):
        source = manifest["inputs"][key]
        if sha256(Path(source["snapshot"])) != source["sha256"]:
            raise ValueError(f"{key} snapshot changed")
    return manifest


def references(manifest: dict) -> dict:
    owner = read_labels(Path(manifest["inputs"]["dense"]["snapshot"]))
    out = {}
    for shot, item in manifest["dataset"].items():
        path = Path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"dataset changed for shot {shot}")
        with np.load(path) as z:
            ann = frame_mean(z["annotated"]) >= 0.5
        out[int(shot)] = {"dense": dense_states(owner[int(shot)]), "legacy": ann}
    return out


def prepare(args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "manifest.json").exists():
        raise ValueError("manifest exists; use a new output directory to reprepare")
    inputs = {}
    for key, path in (("dense", args.dense), ("cohort", args.cohort)):
        snapshot = args.out_dir / "inputs" / f"{key}.csv"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        inputs[key] = {
            "path": str(path),
            "snapshot": str(snapshot),
            "sha256": sha256(snapshot),
        }
    dense_table = pd.read_csv(inputs["dense"]["snapshot"])
    cohort = pd.read_csv(inputs["cohort"]["snapshot"])
    gold = set(cohort.loc[cohort.split == "test", "shot"].astype(int))
    groups: dict[str, list[int]] = {"train": [], "valid": []}
    dataset = {}
    for path in sorted(args.dataset_dir.glob("*.npz")):
        shot = int(path.stem.split("_")[0])
        if str(shot) in dataset:
            raise ValueError(f"duplicate dataset shot {shot}")
        with np.load(path) as z:
            split = str(z["split"])
            if z["annotated"].shape != (7820,):
                raise ValueError(f"unexpected native grid for shot {shot}")
        groups[split].append(shot)
        dataset[str(shot)] = {"path": str(path), "sha256": sha256(path), "split": split}
    split = clean_split(groups["train"], groups["valid"], gold, args.split_seed)
    if set(map(int, dataset)) - set(dense_table.shot):
        raise ValueError("dense labels must cover every original AE shot")
    manifest = {
        **provenance(),
        "status": "prepared",
        "inputs": inputs,
        "dataset": dataset,
        "dataset_dir": str(args.dataset_dir),
        "split": split,
        "split_seed": args.split_seed,
        "blind_gold_shots": sorted(gold),
        "blind_gold_overlap": [],
        "seeds": args.seeds,
        "dense_counts": {
            "rows": len(dense_table),
            "shots": int(dense_table.shot.nunique()),
            "present_intervals": int((dense_table.category == PRESENT).sum()),
            "category_counts": {
                str(k): int(v) for k, v in dense_table.category.value_counts().items()
            },
        },
        "protocol": {
            "record_ms": [0, 2000],
            "frame_ms": 10,
            "native_columns": 7820,
            "legacy": "audit: >= half annotated columns; classes 1-4, LFM excluded",
            "dense": "catalog any-touch; only present/absent frames scored",
            "threeway": "original native annotation AND TokEye; disagreement ignored",
            "frequency": "identical native annotation&active finite-frequency weights",
            "selection": "minimum own-target loss on 20 selection shots, patience 5",
            "threshold": "maximum 10 ms selection F1 for own activity target",
            "evaluation_hard_call": "mean native probability >= frozen threshold",
            "recipe": "ae_train defaults: SCE, 710 frames, 8 windows/shot, batch 16, "
            "30 epochs, AdamW 1e-4, weight decay 1e-4, cosine min lr 1e-6",
        },
        "environment": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "python": platform.python_version(),
        },
    }
    refs = references(manifest)
    manifest["frame_counts"] = {
        group: {
            reference: {
                "frames": int(
                    sum(
                        np.isin(refs[s][reference], (ABSENT, PRESENT)).sum()
                        for s in shots
                    )
                ),
                "positive": int(
                    sum((refs[s][reference] == PRESENT).sum() for s in shots)
                ),
            }
            for reference in ("dense", "legacy")
        }
        for group, shots in split.items()
    }
    older = load_older(args.df2)
    if set(older["spec"]["rows"]) != set(older["xpow"]["rows"]):
        raise ValueError("older spectrogram and cross-power saved shot sets differ")
    manifest["older"] = {
        "root": str(args.df2),
        "aggregation": "max first four AE classes, then mean chords",
        "input": "spec",
        "trained_on": sorted(older["trained_on"]),
        "fair_evaluation": sorted(
            set(split["evaluation"]) & set(older["spec"]["rows"])
        ),
        "selection_available": sorted(
            set(split["selection"]) & set(older["spec"]["rows"])
        ),
        "evaluation_without_saved_predictions": sorted(
            set(split["evaluation"]) - set(older["spec"]["rows"])
        ),
        "order_check": check_order(older["spec"]),
    }
    relevant = set(split["selection"]) | set(split["evaluation"])
    if relevant & set(older["spec"]["rows"]) & older["trained_on"]:
        raise ValueError(
            "saved older validation predictions include older training shots"
        )
    arrays, thresholds = {}, {}
    for model in ("rcn", "lstm"):
        v = older["spec"]
        bins = nearest_bin(v[f"{model}_t"])
        for shot in sorted(relevant & set(v["rows"])):
            score = v[model][v["rows"][shot]][:, :, AE_CLASSES].max(axis=2).mean(axis=0)
            arrays[f"ae-{model}_{shot}"] = score[bins].astype(float)
        selected = manifest["older"]["selection_available"]
        if selected:
            threshold = selection_threshold(
                np.concatenate([arrays[f"ae-{model}_{s}"] for s in selected]),
                np.concatenate([refs[s]["legacy"] for s in selected]),
            )
            thresholds[f"ae-{model}"] = {
                **threshold,
                "shots": selected,
                "reference": "legacy",
            }
        else:
            thresholds[f"ae-{model}"] = {
                "threshold": {"rcn": 0.10, "lstm": 0.15}[model],
                "shots": [],
                "reference": "published operating point; selection unavailable",
            }
    saved = args.out_dir / "older_probabilities.npz"
    np.savez(saved, **arrays)
    older_files = [
        args.df2 / name
        for name in (
            "shots_val_7525.npy",
            "shots_train_7525.npy",
            "spec_dp_8_4/spec_time.npy",
            "results_dp/spec/lstm/time_rebin.npy",
            "results_dp/spec/lstm/y_val_rebin.npy",
            "results_dp/spec/lstm/y_pred_rebin.npy",
            "results_dp/spec/rcn/y_pred.npy",
        )
    ]
    manifest["older"]["source_sha256"] = {str(p): sha256(p) for p in older_files}
    manifest["older"]["probabilities"] = {"path": str(saved), "sha256": sha256(saved)}
    manifest["older"]["thresholds"] = thresholds
    write_json(args.out_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "split": split,
                "dense_counts": manifest["dense_counts"],
                "older_selection": manifest["older"]["selection_available"],
            },
            indent=2,
        )
    )


def run_dir(out: Path, supervision: str, seed: int) -> Path:
    return out / "models" / supervision / f"seed-{seed}"


def accepted_seeds(out: Path, manifest: dict) -> dict[str, list[int]]:
    """Seeds of each arm that count: the frozen list until a convergence plan exists."""
    path = out / "convergence.json"
    if path.exists():
        return json.loads(path.read_text())["accepted_seeds"]
    return {arm: list(manifest["seeds"]) for arm in SUPERVISIONS}


def extend_seeds(args) -> None:
    """Expand an untrained manifest without changing frozen inputs or shots."""
    path = args.out_dir / "manifest.json"
    manifest = load_manifest(path)
    if list((args.out_dir / "models").glob("**/run.json")):
        raise ValueError("cannot change the manifest after a completed run")
    if not set(manifest["seeds"]).issubset(args.seeds):
        raise ValueError("existing seeds must be retained")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    archive = args.out_dir / "run1_manifest.json"
    if not archive.exists():
        archive.write_bytes(path.read_bytes())
    manifest["seed_extension"] = {
        **provenance(),
        "previous_manifest_sha256": sha256(path),
        "previous_seeds": manifest["seeds"],
        "reason": "authorized GPU rerun with three seeds; inputs and split unchanged",
    }
    manifest["seeds"] = args.seeds
    write_json(path, manifest)
    print(json.dumps(manifest["seed_extension"], indent=2))


def verify(args) -> None:
    """Exercise all real-data label loaders without needing CUDA or training."""
    path = args.out_dir / "manifest.json"
    manifest = load_manifest(path)
    results, reference_frequency = {}, {}
    for supervision in SUPERVISIONS:
        records, _ = ae_train.load_swap_labels(path, supervision)
        results[supervision] = {}
        for group, split in (("train", "train"), ("selection", "valid")):
            selected = [rec for rec in records if rec.split == split]
            results[supervision][group] = {
                "shots": [int(rec.shot) for rec in selected],
                "native_columns": sum(rec.y.size for rec in selected),
                "weighted_columns": int(sum(rec.w.sum() for rec in selected)),
                "positive_columns": int(sum((rec.y * rec.w).sum() for rec in selected)),
                "frequency_columns": int(sum(rec.fw.sum() for rec in selected)),
            }
        for rec in records:
            if supervision == "legacy":
                reference_frequency[rec.shot] = (rec.ft.copy(), rec.fw.copy())
            else:
                for actual, frozen in zip(
                    (rec.ft, rec.fw), reference_frequency[rec.shot], strict=True
                ):
                    if not np.array_equal(actual, frozen):
                        raise ValueError("frequency supervision differs across arms")
            if supervision == "threeway":
                original = ae_train.build_targets(
                    rec.active, rec.annotated, rec.freq_khz, "threeway"
                )
                for actual, expected in zip(
                    (rec.y, rec.w, rec.ft, rec.fw), original, strict=True
                ):
                    if not np.array_equal(actual, expected):
                        raise ValueError("threeway differs from the original recipe")
    record = {
        **provenance(),
        "manifest_sha256": sha256(path),
        "split": manifest["split"],
        "status": "passed",
        "frequency_supervision_identical": True,
        "threeway_matches_original_recipe": True,
        "results": results,
    }
    write_json(args.out_dir / "verification.json", record)
    print(json.dumps(record, indent=2))


def probe(args) -> None:
    """Record the actual CUDA precision gate, including emulated support."""
    if not torch.cuda.is_available():
        raise SystemExit("probe requires the CUDA interpreter")
    supported = torch.cuda.is_bf16_supported()
    record = {
        **provenance(),
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "bf16_default": supported,
        "bf16_native": torch.cuda.is_bf16_supported(including_emulation=False),
        "selected_autocast_dtype": "bfloat16" if supported else "float32",
    }
    write_json(args.out_dir / "gpu_probe.json", record)
    print(json.dumps(record, indent=2))


def train(args) -> None:
    manifest_path = args.out_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    if args.seed not in accepted_seeds(args.out_dir, manifest)[args.supervision]:
        raise ValueError("seed is not in the frozen manifest or convergence plan")
    out = run_dir(args.out_dir, args.supervision, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        raise ValueError("run exists; do not overwrite selected checkpoints")
    record = {
        **provenance(),
        "model": "ae-ours",
        "supervision": args.supervision,
        "seed": args.seed,
        "manifest_sha256": sha256(manifest_path),
        "training_rule": CONVERGENCE_RULE["training"],
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "autocast_dtype": (
                "bfloat16" if torch.cuda.is_bf16_supported() else "float32"
            )
            if torch.cuda.is_available()
            else None,
        },
        "execution": {
            "python": sys.executable,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    if not torch.cuda.is_available():
        record.update(status="blocked", reason="training interpreter has no CUDA torch")
        write_json(out / "attempt.json", record)
        raise SystemExit(record["reason"])
    write_json(out / "attempt.json", {**record, "status": "running"})
    tag = f"seed{args.seed}"
    ae_train.main(
        [
            "--swap-manifest",
            str(manifest_path),
            "--target",
            args.supervision,
            "--loss",
            "sce",
            "--out-dir",
            str(out),
            "--seed",
            str(args.seed),
            "--tag",
            tag,
            "--patience",
            str(CONVERGENCE_RULE["training"]["patience"]),
            "--patience-start",
            str(CONVERGENCE_RULE["training"]["patience_start_epoch"]),
            "--num-workers",
            "8",
            "--device",
            "cuda",
        ]
    )
    ckpt = out / f"ae_seldnet_{args.supervision}_sce_{tag}.pt"
    metadata = out / f"training_{args.supervision}_sce_{tag}.json"
    training = json.loads(metadata.read_text())
    if training["status"] != "finished":
        raise ValueError("training did not finish")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model_data = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = ae_train.AeSeldNet(ae_train.AeSeldNetConfig.from_dict(model_data["config"]))
    model.load_state_dict(model_data["state_dict"])
    model.to(device)
    selected, _ = ae_train.load_swap_labels(manifest_path, args.supervision)
    selection = [s for s in selected if s.split == "valid"]
    all_labels = ae_train.load_labels(Path(manifest["dataset_dir"]), "threeway")
    evaluation = [
        s for s in all_labels if int(s.shot) in manifest["split"]["evaluation"]
    ]
    # Verify evaluation inputs before using them; their labels never selected the model.
    references(manifest)
    preds = ae_train.predict_records(model, selection + evaluation, device, dtype)
    arrays = {
        s.shot: frame_mean(preds[s.path.stem]["prob"]) for s in selection + evaluation
    }
    scores, truth = [], []
    for rec in selection:
        share = frame_mean(rec.w)
        keep = share >= 0.5
        target = frame_mean(rec.y) / np.maximum(share, 1e-12)
        scores.append(arrays[rec.shot][keep])
        truth.append(target[keep] >= 0.5)
    threshold = selection_threshold(np.concatenate(scores), np.concatenate(truth))
    threshold.update(shots=manifest["split"]["selection"], reference=args.supervision)
    probabilities = out / "probabilities.npz"
    np.savez(probabilities, **arrays)
    record.update(
        status="finished",
        checkpoint={"path": str(ckpt), "sha256": sha256(ckpt)},
        training={"path": str(metadata), "sha256": sha256(metadata)},
        probabilities={"path": str(probabilities), "sha256": sha256(probabilities)},
        threshold=threshold,
        selected_epoch=training["best_epoch"],
    )
    write_json(out / "run.json", record)
    print(json.dumps(record, indent=2))


def selection_parts(manifest_path: Path, arm: str, arrays: dict) -> list[tuple]:
    records, _ = ae_train.load_swap_labels(manifest_path, arm)
    parts = []
    for rec in records:
        if rec.split != "valid":
            continue
        share = frame_mean(rec.w)
        keep = share >= 0.5
        target = frame_mean(rec.y) / np.maximum(share, 1e-12)
        parts.append((arrays[int(rec.shot)][keep], target[keep] >= 0.5))
    return parts


def archive_run(out: Path, arm: str, seed: int) -> tuple[Path, int]:
    """Move a superseded run aside; its files are kept and never reused."""
    source = run_dir(out, arm, seed)
    n = 1
    while (target := source.with_name(f"seed-{seed}-superseded{n}")).exists():
        n += 1
    source.rename(target)
    return target, n


def audit_convergence(args) -> None:
    """Apply the declared rule to every record; freeze the rule before reading any."""
    path = args.out_dir / "convergence.json"
    manifest_path = args.out_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    if path.exists():
        plan = json.loads(path.read_text())
        if plan["rule"] != CONVERGENCE_RULE:
            raise ValueError("declared convergence rule cannot change")
    else:
        plan = {
            "declared": provenance(),
            "rule": CONVERGENCE_RULE,
            "manifest_sha256": sha256(manifest_path),
            "accepted_seeds": {arm: list(manifest["seeds"]) for arm in SUPERVISIONS},
            "audit": {},
            "superseded": {},
            "replacements": {},
        }
        write_json(path, plan)  # the rule is on disk before any checkpoint is read
    training_rule = CONVERGENCE_RULE["training"]
    for arm in SUPERVISIONS:
        seeds = plan["accepted_seeds"][arm]
        for seed in seeds[:]:
            name = f"ae-ours-{arm}-seed{seed}"
            run_path = run_dir(args.out_dir, arm, seed) / "run.json"
            if not run_path.exists():
                continue
            run = json.loads(run_path.read_text())
            training = json.loads(Path(run["training"]["path"]).read_text())
            losses = [row["valid_val_loss"] for row in training["history"]]
            conformance = training_conformance(
                losses,
                patience=training_rule["patience"],
                start=training_rule["patience_start_epoch"],
                min_delta=training_rule["min_delta"],
            )
            entry = {
                "training_rule": conformance,
                "run_sha256": sha256(run_path),
                "selected_epoch": run["selected_epoch"],
            }
            if not conformance["conforms"]:
                target, n = archive_run(args.out_dir, arm, seed)
                entry["status"] = "superseded: rerun with the same seed"
                entry["archived_to"] = str(target)
                plan["superseded"][f"{name}-superseded{n}"] = entry
                plan["audit"].pop(name, None)
                continue
            source = Path(run["probabilities"]["path"])
            if sha256(source) != run["probabilities"]["sha256"]:
                raise ValueError(f"probabilities changed for {name}")
            with np.load(source) as z:
                arrays = {s: z[str(s)].copy() for s in manifest["split"]["selection"]}
            entry["screen"] = convergence_screen(
                selection_parts(manifest_path, arm, arrays)
            )
            entry["status"] = "accepted" if entry["screen"]["passed"] else "excluded"
            plan["audit"][name] = entry
            if not entry["screen"]["passed"]:
                used = {
                    int(k.rsplit("seed", 1)[1])
                    for k in plan["audit"]
                    if k.startswith(f"ae-ours-{arm}-")
                }
                replacement = max(seeds + manifest["seeds"] + [2] + list(used)) + 1
                seeds[seeds.index(seed)] = replacement
                plan["replacements"][name] = f"ae-ours-{arm}-seed{replacement}"
    pending = [
        f"ae-ours-{arm}-seed{seed}"
        for arm in SUPERVISIONS
        for seed in plan["accepted_seeds"][arm]
        if not (run_dir(args.out_dir, arm, seed) / "run.json").exists()
    ]
    plan["pending_runs"] = pending
    write_json(path, plan)
    print(json.dumps(plan, indent=2))


def load_run(directory: Path, manifest_path: Path) -> tuple[dict, dict, dict]:
    """A finished run read from its own directory and verified by recorded hashes."""
    run = json.loads((directory / "run.json").read_text())
    if run["status"] != "finished" or run["manifest_sha256"] != sha256(manifest_path):
        raise ValueError(f"run in {directory} is unfinished or uses another manifest")
    training_path = directory / Path(run["training"]["path"]).name
    if sha256(training_path) != run["training"]["sha256"]:
        raise ValueError(f"training metadata changed in {directory}")
    probabilities = directory / "probabilities.npz"
    if sha256(probabilities) != run["probabilities"]["sha256"]:
        raise ValueError(f"probabilities changed in {directory}")
    with np.load(probabilities) as z:
        scores = {int(k): z[k].copy() for k in z.files}
    return run, json.loads(training_path.read_text()), scores


def reference_parts(refs: dict, shots: list[int], reference: str, prediction: dict):
    """(score, truth) per shot on the frames the reference can score."""
    parts = []
    for shot in shots:
        states = refs[shot][reference]
        keep = np.isin(states, (ABSENT, PRESENT))
        if prediction[shot].shape != (N_FRAMES,):
            raise ValueError(f"wrong frame shape for shot {shot}")
        parts.append((prediction[shot][keep], states[keep] == PRESENT))
    return parts


def matched_threshold(refs, shots, reference, prediction) -> dict:
    """Max-F1 threshold on selection shots against the reference being scored."""
    parts = reference_parts(refs, shots, reference, prediction)
    result = selection_threshold(
        np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    )
    return {**result, "shots": list(shots), "reference": reference}


def clock_models(refs: dict, manifest: dict) -> tuple[dict, dict]:
    """Input-free baselines: per-10 ms positive rate over train + selection shots."""
    split = manifest["split"]
    shots = split["train"] + split["selection"]
    priors, scores = {}, {}
    for reference, name in (("legacy", "clock-annotation"), ("dense", "clock-dense")):
        states = np.stack(
            [
                np.where(refs[s][reference] == PRESENT, PRESENT, ABSENT)
                if reference == "legacy"
                else refs[s][reference]
                for s in shots
            ]
        )
        priors[name] = {
            "reference": reference,
            "shots": len(shots),
            "prior": clock_prior(states).tolist(),
        }
        prior = np.asarray(priors[name]["prior"])
        scores[name] = {
            int(s): prior.copy() for s in split["selection"] + split["evaluation"]
        }
    return priors, scores


def dense_prevalence(manifest: dict, refs: dict) -> dict:
    """Reconcile the dense table's counts and prevalence with the paper's wording."""
    owner = read_labels(Path(manifest["inputs"]["dense"]["snapshot"]))
    split = manifest["split"]
    groups = {
        "train_selection_120": split["train"] + split["selection"],
        "train_100": split["train"],
        "evaluation_60": split["evaluation"],
    }
    out = {"crowd_present_rows": manifest["dense_counts"]["present_intervals"]}
    for name, shots in groups.items():
        seconds, frames, scorable = 0.0, 0, 0
        for shot in shots:
            label = owner[shot]
            union = np.zeros(RECORD_MS, bool)
            for start, stop, category in label.intervals:
                if category == PRESENT:
                    union[int(max(start, 0)) : int(min(stop, RECORD_MS))] = True
            seconds += union.sum() / 1000
            states = refs[shot]["dense"]
            frames += int((states == PRESENT).sum())
            scorable += int(np.isin(states, (ABSENT, PRESENT)).sum())
        out[name] = {
            "shots": len(shots),
            "present_seconds_union": seconds,
            "prevalence_by_duration": seconds / (RECORD_MS / 1000 * len(shots)),
            "prevalence_any_touch_frames": frames / scorable,
        }
    return out


def dense_history(manifest: dict) -> dict:
    """Who wrote the dense table and how many intervals its history holds."""
    path = Path(manifest["inputs"]["dense"]["path"]).parent / "history.jsonl"
    entries = [json.loads(line) for line in path.read_text().splitlines() if line]
    latest = {entry["shot"]: entry for entry in entries}
    return {
        "path": str(path),
        "sha256": sha256(path),
        "entries": len(entries),
        "shots": len(latest),
        "reviewers": sorted({str(entry["reviewer"]) for entry in entries}),
        "sources": sorted({str(entry["source"]) for entry in entries}),
        "latest_intervals_per_shot_total": sum(
            len(entry["intervals"]) for entry in latest.values()
        ),
        "all_entry_intervals_total": sum(len(entry["intervals"]) for entry in entries),
    }


def band_confound(manifest: dict) -> dict:
    """How much annotated time is BAE-only, and is it inside the model's band?"""
    masks = Path(manifest["dataset_dir"]).parent / "masks"
    split = manifest["split"]
    groups = {
        "train_selection_120": split["train"] + split["selection"],
        "train_100": split["train"],
        "evaluation_60": split["evaluation"],
        "fair_19": manifest["older"]["fair_evaluation"],
    }
    out = {
        "model_band_khz": [80.56649, 250.00024],
        "columns": "native annotation columns; classes 1-4 are BAE, EAE, RSAE, TAE",
        "source": f"{masks}/<shot>_<split>_clean.npz frame_labels, dataset active",
    }
    for name, shots in groups.items():
        annotated = bae = bae_active = other = other_active = 0
        for shot in shots:
            item = manifest["dataset"][str(shot)]
            with np.load(masks / f"{shot}_{item['split']}_clean.npz") as z:
                labels = z["frame_labels"].astype(bool)
            with np.load(item["path"]) as z:
                active = z["active"].astype(bool)
            ann = labels[1:5].any(axis=0)
            only = labels[1] & ~labels[2:5].any(axis=0)
            annotated += int(ann.sum())
            bae += int(only.sum())
            bae_active += int((only & active).sum())
            other += int((ann & ~only).sum())
            other_active += int((ann & ~only & active).sum())
        out[name] = {
            "annotated_columns": annotated,
            "bae_only_share": bae / annotated,
            "in_band_active_share_bae_only": bae_active / bae if bae else None,
            "in_band_active_share_other": other_active / other,
        }
    return out


def campaign_overlap(manifest: dict) -> dict:
    """How close the scored shots sit to each model's training shots (run blocks)."""
    split = manifest["split"]
    garcia = np.array(sorted(manifest["older"]["trained_on"]))
    fair = manifest["older"]["fair_evaluation"]

    def blocks(shots):
        counts: dict[str, int] = {}
        for shot in shots:
            counts[str(shot // 100)] = counts.get(str(shot // 100), 0) + 1
        return dict(sorted(counts.items()))

    ours = split["train"] + split["selection"]
    return {
        "block": "shot // 100 (shots of one run day sit in a block or two)",
        "garcia_training_shots": len(garcia),
        "evaluation_in_garcia_training": len(set(split["evaluation"]) & set(garcia)),
        "fair_nearest_garcia_training_shot_distance": sorted(
            int(np.abs(garcia - s).min()) for s in fair
        ),
        "ae_ours_training_shots": len(split["train"]),
        "ae_ours_training_by_block": blocks(split["train"]),
        "ae_ours_train_selection_in_garcia_training": len(set(ours) & set(garcia)),
        "evaluation_by_block": blocks(split["evaluation"]),
        "fair_by_block": blocks(fair),
    }


def pooled_block(refs, shots, reference, methods, thresholds, groups, replicates):
    """Paired pooled scores plus within-shot medians for one cohort and reference."""
    packed = {
        name: (
            reference_parts(refs, shots, reference, pred),
            thresholds[name]["threshold"],
        )
        for name, pred in methods.items()
    }
    result = paired_scores(packed, replicates, BOOTSTRAP_SEED, groups=groups)
    result["within_shot"] = {
        name: within_shot_scores(parts, replicates)
        for name, (parts, _) in packed.items()
    }
    return result, packed


def evaluate(args) -> None:
    manifest_path = args.out_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    refs = references(manifest)
    plan = json.loads((args.out_dir / "convergence.json").read_text())
    if plan["rule"] != CONVERGENCE_RULE:
        raise ValueError("convergence plan declares another rule")
    split = manifest["split"]
    scores, own, matched, missing, runs, v100 = {}, {}, {}, [], {}, {}
    for arm in SUPERVISIONS:
        for seed in plan["accepted_seeds"][arm]:
            name = f"ae-ours-{arm}-seed{seed}"
            directory = run_dir(args.out_dir, arm, seed)
            if not (directory / "run.json").exists():
                missing.append(name)
                continue
            run, training, scores[name] = load_run(directory, manifest_path)
            audit = plan["audit"].get(name)
            if audit is None or audit["status"] != "accepted":
                raise ValueError(f"{name} is not an accepted record of the plan")
            runs[name] = {
                **run,
                "training_environment": training["environment"],
                "epochs_completed": len(training["history"]),
                "audit": audit,
            }
            own[name] = run["threshold"]["threshold"]
            if "V100" in training["environment"]["gpu"] and seed != 0:
                v100.setdefault(arm, []).append(name)
    source = Path(manifest["older"]["probabilities"]["path"])
    if sha256(source) != manifest["older"]["probabilities"]["sha256"]:
        raise ValueError("saved older probabilities changed")
    with np.load(source) as z:
        for model in ("ae-rcn", "ae-lstm"):
            scores[model] = {
                int(k.removeprefix(model + "_")): z[k].copy()
                for k in z.files
                if k.startswith(model + "_")
            }
    for name, item in manifest["older"]["thresholds"].items():
        own[name] = item["threshold"]
    priors, clock_scores = clock_models(refs, manifest)
    scores.update(clock_scores)
    selection_shots = {
        name: (
            manifest["older"]["selection_available"]
            if name in ("ae-rcn", "ae-lstm")
            else split["selection"]
        )
        for name in scores
    }
    for name, prediction in scores.items():
        matched[name] = {
            reference: matched_threshold(
                refs, selection_shots[name], reference, prediction
            )
            for reference in ("dense", "legacy")
        }
    arms = {
        f"ae-ours-{arm}": [
            f"ae-ours-{arm}-seed{seed}" for seed in plan["accepted_seeds"][arm]
        ]
        for arm in SUPERVISIONS
    }
    complete = not missing
    blocks, sensitivity = {}, {}
    for group, shots in (
        ("all_60", split["evaluation"]),
        ("fair_19", manifest["older"]["fair_evaluation"]),
    ):
        eligible = {n: p for n, p in scores.items() if set(shots) <= set(p)}
        blocks[group] = {
            "shots": shots,
            "unavailable": sorted(set(scores) - set(eligible)) + missing,
            "references": {},
        }
        seed_groups = {
            **(arms if complete else {}),
            **{n: [n] for n in eligible if not n.startswith("ae-ours-")},
        }
        for reference in ("dense", "legacy"):
            result, packed = pooled_block(
                refs,
                shots,
                reference,
                eligible,
                {n: matched[n][reference] for n in eligible},
                seed_groups if complete else None,
                args.replicates,
            )
            result["calibration"] = {
                n: {
                    k: matched[n][reference][k]
                    for k in ("threshold", "f1", "n_frames", "n_positive", "shots")
                }
                for n in eligible
            }
            result["f1_own_target_threshold"] = {
                n: ShotMetric(packed[n][0], own[n]).values(np.ones(len(shots)))[2]
                for n in eligible
                if n in own
            }
            blocks[group]["references"][reference] = result
        if complete and all(len(v100.get(arm, [])) == 2 for arm in SUPERVISIONS):
            keep = [n for arm in SUPERVISIONS for n in v100[arm]]
            sensitivity[group] = {"runs": v100, "references": {}}
            for reference in ("dense", "legacy"):
                subset = {n: eligible[n] for n in keep}
                subset.update(
                    {n: p for n, p in eligible.items() if n in ("ae-rcn", "ae-lstm")}
                )
                packed = {
                    n: (
                        reference_parts(refs, shots, reference, p),
                        matched[n][reference]["threshold"],
                    )
                    for n, p in subset.items()
                }
                sensitivity[group]["references"][reference] = paired_scores(
                    packed,
                    args.replicates,
                    BOOTSTRAP_SEED,
                    groups={
                        **{f"ae-ours-{arm}": v100[arm] for arm in SUPERVISIONS},
                        **{n: [n] for n in subset if not n.startswith("ae-ours-")},
                    },
                )["seed_summary"]
    excluded = {}
    candidates = {
        name: Path(entry["archived_to"]) for name, entry in plan["superseded"].items()
    }
    for name, entry in plan["audit"].items():
        if entry["status"] == "excluded":
            arm, seed = name.split("-")[2], int(name.rsplit("seed", 1)[1])
            candidates[name] = run_dir(args.out_dir, arm, seed)
    for name, directory in candidates.items():
        run, training, prediction = load_run(directory, manifest_path)
        row = {
            "directory": str(directory),
            "selected_epoch": run["selected_epoch"],
            "epochs_completed": len(training["history"]),
            "gpu": training["environment"]["gpu"],
            "plan": plan["superseded"].get(name) or plan["audit"][name],
            "screen": convergence_screen(
                selection_parts(
                    manifest_path,
                    name.split("-")[2],
                    {s: prediction[s] for s in split["selection"]},
                )
            ),
            "results": {},
        }
        thr = {
            ref: matched_threshold(refs, split["selection"], ref, prediction)
            for ref in ("dense", "legacy")
        }
        for group, shots in (
            ("all_60", split["evaluation"]),
            ("fair_19", manifest["older"]["fair_evaluation"]),
        ):
            row["results"][group] = {}
            for reference in ("dense", "legacy"):
                parts = reference_parts(refs, shots, reference, prediction)
                scored = paired_scores(
                    {name: (parts, thr[reference]["threshold"])},
                    args.replicates,
                    BOOTSTRAP_SEED,
                )
                row["results"][group][reference] = {
                    **scored["methods"][name],
                    "within_shot": within_shot_scores(parts, args.replicates),
                }
        excluded[name] = row
    record = {
        **provenance(),
        "status": "finished" if complete else "incomplete",
        "missing_runs": missing,
        "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "convergence": plan,
        "runs": runs,
        "excluded_runs": excluded,
        "thresholds": {
            "calibration": (
                "every method: maximum 10 ms F1 on its selection shots (20; six for "
                "ae-rcn and ae-lstm) against the reference being scored"
            ),
            "own_target": own,
        },
        "clock": priors,
        "results": blocks,
        "sensitivity_v100_seeds_1_2": sensitivity,
        "band_confound": band_confound(manifest),
        "campaign_overlap": campaign_overlap(manifest),
        "dense_prevalence": dense_prevalence(manifest, refs),
        "dense_history": dense_history(manifest),
        "dense_counts": manifest["dense_counts"],
        "inputs": manifest["inputs"],
        "protocol": manifest["protocol"],
        "split": split,
        "limitations": [
            "Older detectors have no saved predictions for 41/60 evaluation shots.",
            "Older calibration uses only the six selection shots they did not train on.",
            "Three seeds per arm give limited precision for training variability.",
            (
                "Runs used A100 (legacy and dense seed 0) and V100S GPUs; a V100-only "
                "sensitivity analysis is recorded."
            ),
        ],
    }
    probe_path = args.out_dir / "gpu_probe.json"
    if probe_path.exists():
        record["gpu_probe"] = {
            "path": str(probe_path),
            "sha256": sha256(probe_path),
            "record": json.loads(probe_path.read_text()),
        }
    write_json(args.out_dir / "evaluation.json", record)
    write_json(args.record, record)
    if complete:
        from ae_supervision_swap_report import render_report

        render_report(record, manifest, args.out_dir, REPO)
    print(json.dumps({"status": record["status"], "missing": missing}, indent=2))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "round4/aeswap")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--dataset-dir", type=Path, default=ROOT / "ae/dataset")
    prep.add_argument(
        "--dense", type=Path, default=TABLES / "alfven_eigenmode/review/labels.csv"
    )
    prep.add_argument("--cohort", type=Path, default=TABLES / "catalog/cohort.csv")
    prep.add_argument(
        "--df2", type=Path, default=Path("/projects/EKOLEMEN/agarcia/df2")
    )
    prep.add_argument("--seeds", nargs="+", type=int, default=[0])
    prep.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    extend = commands.add_parser("extend-seeds")
    extend.add_argument("--seeds", nargs="+", type=int, required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--supervision", choices=SUPERVISIONS, required=True)
    fit.add_argument("--seed", type=int, default=0)
    commands.add_parser("verify")
    commands.add_parser("probe")
    commands.add_parser("audit-convergence")
    score = commands.add_parser("evaluate")
    score.add_argument("--replicates", type=int, default=1000)
    score.add_argument(
        "--record",
        type=Path,
        default=REPO / "outputs/labeler/ae/supervision_swap/evaluation.json",
    )
    args = parser.parse_args(argv)
    {
        "prepare": prepare,
        "extend-seeds": extend_seeds,
        "train": train,
        "verify": verify,
        "probe": probe,
        "audit-convergence": audit_convergence,
        "evaluate": evaluate,
    }[args.command](args)


if __name__ == "__main__":
    main()
