#!/usr/bin/env python
"""Task 7b: evaluate the trained AE SELDNets on the 60 validation shots.

For every checkpoint under ``--models-dir`` the whole 7820-frame record of each
validation shot goes through the network in one forward pass, and the frame
predictions are scored twice:

* **against `annotated`** - the human annotation (LFM excluded) resampled to
  the frame grid. This is the number that matters. Precision is *expected to
  be low*: the annotation under-counts (task 6/7a measured the cleaned mask at
  recall 0.867 for a 0.540 positive fraction against a 0.210 annotated
  fraction), so a frame the model calls active outside an annotated window is
  as likely to be an unlabelled mode as a false alarm. Recall and AUROC are
  the honest summaries; precision is reported for completeness and must not be
  read as an error rate.
* **against `active`** - task 7a's cleaned, notched mask label, i.e. how well
  the network reproduces its own teacher.

Frequency MAE is measured in kHz on ``annotated & active`` frames - the frames
where a human says there is a mode *and* the mask localised it, which is the
only population where the centroid target is trustworthy.

Writes ``evaluation.json`` next to the checkpoints and a six-shot figure set
under ``--fig-dir``.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for extra in (str(REPO / "src"), str(Path(__file__).resolve().parent)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from ae_train import (
    DEFAULT_DATASET,
    DEFAULT_OUT,
    auroc,
    load_labels,
    load_record,
)

from labelmaker.ae.model import (
    AeSeldNet,
    AeSeldNetConfig,
    denormalise_freq,
)

FIG_SHOTS = 6


def score(prob: np.ndarray, truth: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = prob >= threshold
    truth = truth.astype(bool)
    tp = int((pred & truth).sum())
    recall = float(tp / truth.sum()) if truth.sum() else float("nan")
    precision = float(tp / pred.sum()) if pred.sum() else float("nan")
    f1 = (
        float(2 * recall * precision / (recall + precision))
        if recall + precision > 0
        else 0.0
    )
    return {
        "auroc": auroc(prob, truth.astype(np.float64)),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "predicted_positive_frac": float(pred.mean()),
        "truth_frac": float(truth.mean()),
        "n_frames": int(prob.size),
    }


@torch.no_grad()
def run_checkpoint(
    ckpt_path: Path, shots: list, device: torch.device
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = AeSeldNet(AeSeldNetConfig.from_dict(blob["config"]))
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    preds: dict[str, dict[str, np.ndarray]] = {}
    for rec in shots:
        x = load_record(rec).to(device).float()
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            out = model(x)
        out = out.float()[0]
        preds[rec.path.stem] = {
            "prob": torch.sigmoid(out[:, 0]).cpu().numpy().astype(np.float64),
            "freq_khz": denormalise_freq(out[:, 1]).cpu().numpy().astype(np.float64),
        }
        del x, out
    prob = np.concatenate([preds[s.path.stem]["prob"] for s in shots])
    freq = np.concatenate([preds[s.path.stem]["freq_khz"] for s in shots])
    ann = np.concatenate([s.annotated for s in shots])
    act = np.concatenate([s.active for s in shots])
    ftrue = np.concatenate([s.freq_khz for s in shots])
    labelled = ann & act
    both_lab = (ann & act) | (~ann & ~act)
    conf = labelled & np.isfinite(ftrue)
    summary = {
        "checkpoint": str(ckpt_path),
        "target": blob.get("target"),
        "loss": blob.get("loss_config", {}).get("loss"),
        "best_epoch": blob.get("epoch"),
        "val_loss": blob.get("val_loss"),
        "n_shots": len(shots),
        "vs_annotated": score(prob, ann),
        "vs_active": score(prob, act),
        "vs_threeway_labelled": {
            **score(prob[both_lab], labelled[both_lab]),
            "labelled_frac": float(both_lab.mean()),
            "note": "only the frames the three-way target weights (annotated&active vs neither)",
        },
        "freq_mae_khz": float(np.abs(freq[conf] - ftrue[conf]).mean()) if conf.any() else float("nan"),
        "freq_median_ae_khz": float(np.median(np.abs(freq[conf] - ftrue[conf]))) if conf.any() else float("nan"),
        "freq_bias_khz": float((freq[conf] - ftrue[conf]).mean()) if conf.any() else float("nan"),
        "n_freq_frames": int(conf.sum()),
        "per_shot_recall_annotated_median": float(
            np.median(
                [
                    score(preds[s.path.stem]["prob"], s.annotated)["recall"]
                    for s in shots
                    if s.annotated.sum()
                ]
            )
        ),
    }
    return summary, preds


def make_figures(
    shots: list,
    all_preds: dict[str, dict[str, dict[str, np.ndarray]]],
    fig_dir: Path,
    frame_ms: float = 0.2557544757033248,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    # the six most-annotated validation shots, so the panels show something
    chosen = sorted(shots, key=lambda s: -int(s.annotated.sum()))[:FIG_SHOTS]
    written: list[str] = []
    for rec in chosen:
        with np.load(rec.path) as z:
            spec = z["spec"][:, :, ::10].astype(np.float32).mean(axis=0)
            bins = z["freq_khz_bins"]
        t = np.arange(rec.y.size) * frame_ms / 1000.0
        fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, height_ratios=[2, 1])
        ax = axes[0]
        ax.imshow(
            spec,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=(t[0], t[-1], float(bins[0]), float(bins[-1])),
            vmin=float(np.percentile(spec, 5)),
            vmax=float(np.percentile(spec, 99.5)),
        )
        ax.plot(t, np.where(rec.active, rec.freq_khz, np.nan), ".", ms=1.0, color="#4fd1ff",
                label="mask centroid (target)")
        for name, preds in all_preds.items():
            pr = preds[rec.path.stem]
            ax.plot(
                t,
                np.where(pr["prob"] >= 0.5, pr["freq_khz"], np.nan),
                ".",
                ms=0.9,
                alpha=0.7,
                label=f"{name} predicted f",
            )
        ax.set_ylabel("kHz")
        ax.set_ylim(float(bins[0]), float(bins[-1]))
        ax.legend(loc="upper right", fontsize=6, markerscale=6, framealpha=0.6)
        ax.set_title(f"{rec.path.stem}: AE activity and frequency, validation shot", fontsize=9)

        ax = axes[1]
        ax.fill_between(t, 0, rec.annotated.astype(float), color="#2ca02c", alpha=0.28,
                        step="mid", label="annotated")
        ax.fill_between(t, 0, rec.active.astype(float) * 0.5, color="#888888", alpha=0.35,
                        step="mid", label="mask active (x0.5)")
        for name, preds in all_preds.items():
            ax.plot(t, preds[rec.path.stem]["prob"], lw=0.8, label=f"{name} p(active)")
        ax.axhline(0.5, color="k", lw=0.5, ls=":")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("p(active)")
        ax.legend(loc="upper right", fontsize=6, ncol=2, framealpha=0.6)
        fig.tight_layout()
        out = fig_dir / f"valid_{rec.path.stem}.png"
        fig.savefig(out, dpi=100)
        plt.close(fig)
        written.append(out.name)
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--models-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--fig-dir", type=Path, default=REPO / "outputs" / "labelmaker" / "ae" / "training"
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    ckpts = sorted(args.models_dir.glob("ae_seldnet_*.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {args.models_dir}")

    results: dict[str, dict] = {}
    all_preds: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    valid_by_target: dict[str, list] = {}
    for ckpt in ckpts:
        blob_target = ckpt.stem.replace("ae_seldnet_", "").split("_")[0]
        if blob_target not in valid_by_target:
            shots = load_labels(args.dataset_dir, blob_target)
            valid_by_target[blob_target] = [s for s in shots if s.split == "valid"]
        shots = valid_by_target[blob_target]
        name = ckpt.stem.replace("ae_seldnet_", "")
        print(f"evaluating {name} on {len(shots)} validation shots on {device}", flush=True)
        summary, preds = run_checkpoint(ckpt, shots, device)
        results[name] = summary
        all_preds[name] = preds
        print(json.dumps(summary["vs_annotated"], indent=2), flush=True)

    figures: list[str] = []
    if not args.no_figures:
        any_target = next(iter(valid_by_target))
        figures = make_figures(valid_by_target[any_target], all_preds, args.fig_dir)

    payload = {
        "generated": datetime.now(UTC).astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset_dir": str(args.dataset_dir),
        "device": str(device),
        "threshold": 0.5,
        "n_valid_shots": len(next(iter(valid_by_target.values()))),
        "note": (
            "Precision against `annotated` is not an error rate: the annotation "
            "under-counts, so predicted-positive frames outside an annotated window "
            "may be unlabelled modes. Recall and AUROC are the honest summaries."
        ),
        "runs": results,
        "figures": figures,
    }
    out_path = args.models_dir / "evaluation.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(out_path)
    print(f"wrote {out_path} and {len(figures)} figures to {args.fig_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
