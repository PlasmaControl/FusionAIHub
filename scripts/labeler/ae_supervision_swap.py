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
import tempfile
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
    N_FRAMES,
    SPLIT_SEED,
    clean_split,
    dense_states,
    frame_mean,
    paired_scores,
    selection_threshold,
    sha256,
    validate_split,
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
    # multiprocessing adds /pymp-*/listener-* to TMPDIR; the prescribed scratch
    # path exceeds AF_UNIX's 108-byte limit. Actual files still live in TMPDIR.
    scratch = Path(os.environ["TMPDIR"]).resolve()
    alias = REPO / ".aeswap-tmp"
    if alias.is_symlink():
        if alias.resolve() != scratch:
            raise ValueError("temporary-directory alias points outside TMPDIR")
    elif not alias.exists():
        try:
            alias.symlink_to(scratch, target_is_directory=True)
        except FileExistsError:
            if not alias.is_symlink() or alias.resolve() != scratch:
                raise
    else:
        raise ValueError("temporary-directory alias is not a symlink")
    tempfile.tempdir = str(alias)
    manifest_path = args.out_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    if args.seed not in manifest["seeds"]:
        raise ValueError("seed not specified in the frozen manifest")
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


def evaluate(args) -> None:
    manifest_path = args.out_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    refs = references(manifest)
    sources, scores, thresholds, missing, attempts, runs = {}, {}, {}, [], {}, {}
    for supervision in SUPERVISIONS:
        for seed in manifest["seeds"]:
            name = f"ae-ours-{supervision}-seed{seed}"
            path = run_dir(args.out_dir, supervision, seed) / "run.json"
            if not path.exists():
                missing.append(name)
                attempt = path.with_name("attempt.json")
                if attempt.exists():
                    attempts[name] = {
                        "path": str(attempt),
                        "sha256": sha256(attempt),
                        "record": json.loads(attempt.read_text()),
                    }
                continue
            run = json.loads(path.read_text())
            if run["status"] != "finished" or run["manifest_sha256"] != sha256(
                manifest_path
            ):
                raise ValueError(f"run {name} is unfinished or uses another manifest")
            training_path = Path(run["training"]["path"])
            if sha256(training_path) != run["training"]["sha256"]:
                raise ValueError(f"training metadata changed for {name}")
            training = json.loads(training_path.read_text())
            runs[name] = {
                **run,
                "training_environment": training["environment"],
                "epochs_completed": len(training["history"]),
            }
            source = Path(run["probabilities"]["path"])
            if sha256(source) != run["probabilities"]["sha256"]:
                raise ValueError(f"probabilities changed for {name}")
            with np.load(source) as z:
                scores[name] = {int(k): z[k].copy() for k in z.files}
            thresholds[name] = run["threshold"]
            sources[name] = {"path": str(path), "sha256": sha256(path)}
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
    thresholds.update(manifest["older"]["thresholds"])
    blocks = {}
    for group, shots in (
        ("all_60", manifest["split"]["evaluation"]),
        ("fair_19", manifest["older"]["fair_evaluation"]),
    ):
        blocks[group] = {"shots": shots, "references": {}}
        eligible = {name: p for name, p in scores.items() if set(shots) <= set(p)}
        blocks[group]["unavailable"] = sorted(set(scores) - set(eligible)) + missing
        for reference in ("dense", "legacy"):
            methods = {}
            for name, prediction in eligible.items():
                parts = []
                for shot in shots:
                    states = refs[shot][reference]
                    keep = np.isin(states, (ABSENT, PRESENT))
                    if prediction[shot].shape != (N_FRAMES,):
                        raise ValueError(f"wrong frame shape for {name}, {shot}")
                    parts.append((prediction[shot][keep], states[keep] == PRESENT))
                methods[name] = (parts, thresholds[name]["threshold"])
            blocks[group]["references"][reference] = (
                paired_scores(
                    methods,
                    args.replicates,
                    BOOTSTRAP_SEED,
                    groups={
                        **{
                            f"ae-ours-{arm}": [
                                f"ae-ours-{arm}-seed{seed}"
                                for seed in manifest["seeds"]
                            ]
                            for arm in SUPERVISIONS
                            if all(
                                f"ae-ours-{arm}-seed{s}" in methods
                                for s in manifest["seeds"]
                            )
                        },
                        **{
                            name: [name]
                            for name in ("ae-rcn", "ae-lstm")
                            if name in methods
                        },
                    },
                )
                if methods
                else None
            )
    record = {
        **provenance(),
        "status": "finished" if not missing else "incomplete",
        "missing_runs": missing,
        "attempts": attempts,
        "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "sources": sources,
        "runs": runs,
        "thresholds": thresholds,
        "results": blocks,
        "dense_counts": manifest["dense_counts"],
        "inputs": manifest["inputs"],
        "protocol": manifest["protocol"],
        "split": manifest["split"],
        "limitations": [
            "Older detectors have no saved predictions for 41/60 evaluation shots.",
            "Older thresholds use only available, older-held-out selection shots.",
            (
                "Pooled intervals resample shots and the three observed seed IDs; "
                "three seeds give limited precision for training variability."
            ),
            (
                "GPU types differ across runs; actual capability and precision "
                "policy are recorded in gpu_probe.json and runtime metadata."
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
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Supervision & Dense AUROC & Dense F1 & Legacy AUROC & Legacy F1 \\",
        r"\midrule",
    ]
    block = blocks["fair_19"]["references"]

    def cell(name, reference, metric):
        result = block[reference]
        if result is None or name not in result["methods"]:
            return r"\textemdash"
        m = result["methods"][name][metric]
        if m["value"] is None:
            return r"\textemdash"
        lo, hi = m["ci95"]
        return f"{m['value']:.3f} [{lo:.3f}, {hi:.3f}]"

    for supervision in SUPERVISIONS:
        for seed in manifest["seeds"]:
            name = f"ae-ours-{supervision}-seed{seed}"
            cells = [
                cell(name, r, m) for r in ("dense", "legacy") for m in ("auroc", "f1")
            ]
            lines.append(
                f"ae-ours & {supervision} (seed {seed}) & " + " & ".join(cells) + r" \\"
            )
    for name in ("ae-rcn", "ae-lstm"):
        cells = [cell(name, r, m) for r in ("dense", "legacy") for m in ("auroc", "f1")]
        lines.append(f"{name} & legacy (Garcia saved) & " + " & ".join(cells) + r" \\")
    for supervision in SUPERVISIONS:
        name = f"ae-ours-{supervision}"
        cells = []
        for reference in ("dense", "legacy"):
            summary = (
                (block[reference] or {}).get("seed_summary", {}).get("methods", {})
            )
            for metric in ("auroc", "f1"):
                if name not in summary:
                    cells.append(r"\textemdash")
                    continue
                m = summary[name][metric]
                lo, hi = m["ci95"]
                cells.append(
                    f"{m['mean']:.3f} $\\pm$ {m['sd']:.3f} [{lo:.3f}, {hi:.3f}]"
                )
        lines.append(
            f"ae-ours & {supervision} (seed mean) & " + " & ".join(cells) + r" \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "% Fair held-out shots, 10 ms frames, 95% shot-bootstrap intervals.",
        "% Seed mean: mean +/- sample SD; CI resamples paired shots and seeds.",
        "% Individual-seed CIs resample shots only. Booktabs required.",
    ]
    (args.out_dir / "table_supervision_swap.tex").write_text("\n".join(lines) + "\n")
    if not missing:
        from ae_supervision_swap_report import render_report

        render_report(record, manifest, args.out_dir, REPO)
    print(
        json.dumps(
            {
                "status": record["status"],
                "missing": missing,
                "thresholds": thresholds,
                "results": blocks,
            },
            indent=2,
        )
    )


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
        "evaluate": evaluate,
    }[args.command](args)


if __name__ == "__main__":
    main()
