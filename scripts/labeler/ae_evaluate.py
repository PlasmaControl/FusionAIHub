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

**Named shot lists** (``--shotlist NAME[,NAME]``; docs/labeler/shotlists.md)
score each list the way the headline AE score is made
(``ae_baselines_evaluate.py``): every checkpoint's 0.256 ms probabilities go
onto the owner's 10 ms frames of 0-2 s (the mean in each frame; a hard call
where half the frame's columns are >= 0.5) and are scored against the
owner-reviewed AE table (``--reference``, default the live
``alfven_eigenmode/review/labels.csv``) on the frames it calls present or
absent, with 95 % shot-bootstrap intervals, and against the Heidbrink
annotation beside it. ``shotlists.plan`` picks the shots: a listed shot with
no npz is ``missing.no_input``; one of the train-split shots every checkpoint
was fitted on is reported under ``trained_on`` and is not scored unless
``--include-trained``, which marks the record leaky; one the reference does
not cover is ``missing.no_reference``, and so is one whose reviewed truth was
prefilled from this model's own output (``model_prefills``: the review history
beside the reference says so; ``unscorable`` names each such shot). DataLoader
worker processes read the
npz while the GPU runs every checkpoint on a batch of shots and takes the frame
means, in float32 as the corpus inference does (``--precision bf16`` is task
7b's autocast: its GRU falls off cuDNN and runs about 20 times slower; the
shipped checkpoint's frame means move by 2e-4 on average between the two).
Writes ``--out`` (each list's envelope under ``sets.<list>``, its scores
under ``results.<list>.<checkpoint>``) and the frame scores beside it
(``<out stem>.frames.npz``), so a list can be rescored without the GPU.

**The corpus chain** (``--shotlist ... --chain corpus``) scores the deployed
chain instead: the shipped checkpoint run by ``ae_gold_extend.py`` over each
shot's whole CO2 record (the model card's resample, spectrogram statistics over
the shot's window, 1,024-column windows, the 5-column moving average), read from
the per-shot caches under ``--corpus-work`` (default ``CORPUS_WORK``: the gold
table's, the silver table's and this evaluation's own run; a dir whose caches are
not shown to come from the card's checkpoint is refused). No network runs here.
A frame of 0-2 s is assessed where that table assesses it: a whole frame inside
the shot's window with STFT columns. The same reviewed truth and protocol score
it; the train-split shots and the shots whose truth this model prefilled stay
out. It writes the ``corpus`` block of ``--out``
(the dataset chain's numbers beside it are kept) and ``<out stem>.corpus_frames.npz``,
and compares the two chains on the shots both have (``corpus.agreement``).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
for extra in (str(REPO / "src"), str(Path(__file__).resolve().parent)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from ae_baselines_evaluate import (
    N_FRAMES,
    REPLICATES,
    SEED,
    SELD_FRAMES,
    bootstrap,
    frame_index,
    frame_mean,
    metrics,
)
from ae_train import (
    DEFAULT_DATASET,
    DEFAULT_OUT,
    auroc,
    load_labels,
    load_record,
)

from labeler.ae.model import (
    AeSeldNet,
    AeSeldNetConfig,
    denormalise_freq,
)
from labeler.ae.xpower.data import targets
from labeler.config import Paths, git_dirty, git_sha, sha256_of
from labeler.events import shotlists
from labeler.events.catalog.states import ABSENT, PRESENT
from labeler.events.review import labels as review_labels
from labeler.models.d3d_ae_activity_seldnet import spec as card

FIG_SHOTS = 6
EVENT = "alfven_eigenmode"
#: The checkpoint the model card and best_models.yaml name.
SHIPPED = "threeway_sce"
LIST_OUT = REPO / "outputs" / "labeler" / "ae" / "shotlists" / "evaluation.json"
#: The mark lines' clock, nvidia-smi's: the sbatch averages its sampler between them.
STAMP = "%Y/%m/%d %H:%M:%S.%f"
SMALL = 10
NOTES = {
    "reviewed": (
        "The truth: the owner-reviewed AE table (--reference), frames it calls present "
        "or absent; the others (uncertain, not observable, outside its window) are not "
        "counted. The reviewed labels were made with the detector's output on screen; a "
        "shot whose saved truth started as this model's own calls is not scored (prefilled)."
    ),
    "annotated": (
        "The Heidbrink hand annotation in the npz on the same frames (a frame is "
        "annotated where half its columns are). It under-counts, so an unannotated "
        "frame is not an absent one: precision against it is not an error rate."
    ),
    "no_input": (
        "The AE dataset holds npz only for the 180 shots of the CO2 hand annotation "
        "(events.yaml co2_detector_2021): ae_dataset.py builds them from the aemodes "
        "arrow cache, annotation and all. Any other shot needs its CO2 record fetched "
        "and put through the inference transform first; nothing here does that."
    ),
    "trained_on": (
        "Train-split npz: every checkpoint was fitted on them (the valid split chose "
        "its epoch: early stopping on val_loss)."
    ),
    "prefilled": (
        "A shot is not scored when a save in the review history beside the reference "
        "(review/history.jsonl) was prefilled from a table this model made: prefilled true, "
        "or not recorded (an old client's unknown provenance is never a blind review), "
        "from a table whose sidecar names a scored checkpoint's sha256, or that is or "
        "derives from such a layer (named by its path, its extend_* dir or its CSV's sha256; "
        "prefilled.model_layers, prefilled.layer_sha256). Once anchored, the shot stays out: "
        "the page opens a saved shot on its saved label, so every later save starts from the "
        "model's calls, and for this event a later save's prefilled flag only says whether "
        "the shot was opened on its source table in that sitting (app.js: a reopened saved "
        "shot records prefilled false), not that the label was drawn anew. Only an event "
        "whose page carries exposure across saves (EXPOSURE_TRACKED: detachment) clears the "
        "anchor on a save recorded prefilled false. If the owner re-labels such a shot from "
        "scratch, that is the owner's call to clear by hand: name it with --independent-truth "
        "SHOT (ae_evaluate.sbatch: INDEPENDENT_TRUTH=SHOT,...), and the record lists it under "
        "prefilled.cleared_by_hand. A prefilled save whose table or sidecar cannot be found "
        "is treated the same way (it may be this model's). Such a shot is under "
        "missing.no_reference when the plan reaches it (no_input and trained_on come first)."
    ),
}
#: The events whose review page carries a shot's exposure flags across saves
#: (app.js `independent()`), so a save recorded prefilled false there means the label
#: was not prefilled; on any other event's page it only means the shot was reopened.
EXPOSURE_TRACKED = frozenset({"detachment"})
#: --chain corpus: the deployed chain's per-shot caches (ae_gold_extend.py --work-dir),
#: under $LABELER_ROOT, each with the extend table built from it (its sidecar records the
#: checkpoint's sha256; None: no table); where two hold a shot, the first wins.
CORPUS_WORK = (
    # the gold table's (cpu chain, network on a CPU, 2026-10-12)
    ("runs/ae_gold_extend", f"{EVENT}/extend_ae_seldnet/gold_shots"),
    # the silver table's
    ("runs/silver_infer/ae_seldnet/work", f"{EVENT}/extend_ae_seldnet/silver_shots"),
    # the listed held-out shots neither holds (job 2965792, ae_gold_extend.py --chain gpu)
    ("runs/ae_eval_lists/work", None),
)
#: The checkpoint ae_gold_extend.py runs (its CKPT): the model card's.
CORPUS_CHECKPOINT = card.UPSTREAM / card.ARTIFACTS[0]
#: The 180 shots the card's checkpoint was fitted (120) and validated (60) on.
TRAINING_SHOTS = card.TRAINING_SHOTS
CORPUS_NOTES = {
    "chain": (
        "The deployed chain: the shipped checkpoint over the shot's whole CO2 record as "
        "ae_gold_extend.py runs it (the model card's 500 kHz resample and 80-250 kHz "
        "spectrogram with its statistics over the shot's window, 1,024-column windows "
        "with 128 of context, sigmoid, the card's 5-column moving average). Its input is "
        "not the dataset npz: the dataset chain standardises a 2 s record and runs it "
        "whole, unsmoothed."
    ),
    "frames": (
        "A 10 ms frame of 0-2 s is assessed where the extend table assesses it: a whole "
        "frame inside [0, window end] (the Ip >= 50 kA window, else the last EFIT02 time) "
        "with STFT columns. Reviewed frames after the window end are not scored, so a "
        "list's frames can be fewer than the dataset chain's."
    ),
    "no_input": "No corpus-chain npz for the shot in the work dirs (corpus.work_dirs).",
    "no_reference": (
        "The review table has no present or absent frame for the shot among the "
        "frames this chain assesses."
    ),
    "trained_on": (
        "The train split of the AE dataset: the shipped checkpoint was fitted on it (the "
        "valid split chose its epoch: early stopping on val_loss, so the valid shots are "
        "held out from the fit but not from the model selection). A card training shot "
        "with no npz under --dataset-dir, whose split is unknown, counts as trained on. A "
        "train-split shot without a corpus npz is reported as no_input (unscorable names "
        "both)."
    ),
    "prefilled": NOTES["prefilled"],
    "checkpoint": (
        "Each work dir's caches are taken as this checkpoint's when the checkpoint file "
        "was last modified before the dir's first npz (ae_gold_extend.py always runs that "
        "file) and, for a dir with an extend table, the table's sidecar names the same "
        "sha256 (work_dirs[].checkpoint). A dir that fails either is refused."
    ),
}


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


# --------------------------------------------------------------------------
# named shot lists


def list_names(values: list[str]) -> list[str]:
    """`--shotlist` values (repeated, each comma separated) as names, in order, once."""
    names: list[str] = []
    for value in values:
        for name in value.split(","):
            if name.strip() and name.strip() not in names:
                names.append(name.strip())
    return names


def npz_population(dataset_dir: Path) -> dict[int, tuple[Path, str]]:
    """Each shot with an npz under `dataset_dir` -> (path, split), from `<shot>_<split>.npz`."""
    found: dict[int, tuple[Path, str]] = {}
    for path in sorted(Path(dataset_dir).glob("*.npz")):
        shot, _, split = path.stem.partition("_")
        if not shot.isdigit() or split not in ("train", "valid"):
            raise SystemExit(f"{path}: not <shot>_<train|valid>.npz")
        if int(shot) in found:
            raise SystemExit(f"shot {shot} has two npz under {dataset_dir}")
        found[int(shot)] = (path, split)
    return found


def counted(states: np.ndarray) -> np.ndarray:
    """The frames a reference called present or absent."""
    return np.isin(states, (ABSENT, PRESENT))


def reference_states(path: Path) -> dict[int, np.ndarray]:
    """Each shot of a review table -> its state on the 10 ms frames of 0-2 s."""
    return {
        shot: targets(label, 0, N_FRAMES)
        for shot, label in review_labels.read_labels(path).items()
    }


def sidecar_of(table: Path) -> Path:
    return table.with_name(table.stem + ".meta.json")


def source_table(paths: Paths, event_dir: Path, source: str) -> Path | None:
    """The table a save's `source` names, as `silver.layer_file` maps an origin
    (`kind:name`); a bare file name, as old clients saved it, is that file under the
    event's format/. None when no such file exists."""
    kind, sep, name = source.partition(":")
    if not sep:
        path = Path(event_dir) / "format" / source
    elif kind == "suggest":
        path = Path(paths.root) / "suggestions" / f"{name}.csv"
    elif kind == "format":
        rel = Path(f"{name}.csv")
        path = Path(paths.label_tables) / rel.parent / "format" / rel.name
    else:
        path = Path(paths.label_tables) / f"{name}.csv"
    return path if path.is_file() else None


def model_layers(event_dir: Path, shas: set[str]) -> dict[str, str | None]:
    """The tables under `event_dir` this model made or that derive from one -> their
    CSV's sha256 (None without the CSV), each named by its path under `event_dir`
    without the suffix: a sidecar that names a sha256 of `shas`, then any sidecar naming
    one of those by that name, its `extend_*` dir or its CSV's sha256, to a fixpoint."""
    texts = {
        path.relative_to(event_dir).as_posix().removesuffix(".meta.json"): path.read_text(
            errors="replace"
        )
        for path in sorted(Path(event_dir).rglob("*.meta.json"))
    }
    found: dict[str, str | None] = {}
    markers = set(shas)
    while True:
        more = {n for n, text in texts.items() if n not in found and any(m in text for m in markers)}
        if not more:
            break
        for name in more:
            csv = Path(event_dir) / f"{name}.csv"
            found[name] = sha256_of(csv) if csv.is_file() else None
        markers |= more | {n.split("/")[0] for n in more if n.startswith("extend_")}
        markers |= {found[n] for n in more if found[n]}
    return dict(sorted(found.items()))


def model_prefills(
    paths: Paths, reference: Path, shas: set[str], *, event: str = EVENT,
    cleared: set[int] = frozenset(),
) -> tuple[dict[int, dict], dict]:
    """The shots whose saved truth in `reference` started as this model's own output
    (NOTES.prefilled) -> how, and the check's summary.

    `shas` are the scored checkpoints' sha256. The history is the `history.jsonl`
    beside the reference; with none, nothing can be checked (the summary says so). A
    save recorded prefilled false clears a shot's anchor only for an event in
    `EXPOSURE_TRACKED`; `cleared` are the shots the owner says were re-labelled from
    scratch (--independent-truth), kept under the summary's `cleared_by_hand`."""
    history = Path(reference).with_name("history.jsonl")
    event_dir = Path(reference).parent.parent
    found = model_layers(event_dir, shas)
    layers = sorted({*found, *(n.split("/")[0] for n in found if n.startswith("extend_"))})
    derived = {*layers, *(sha for sha in found.values() if sha)}
    tracked = event in EXPOSURE_TRACKED
    summary = {
        "history": str(history),
        "sha256": sha256_of(history) if history.is_file() else None,
        "checkpoint_sha256": sorted(shas),
        "model_layers": layers,
        "layer_sha256": found,
        "event": event,
        "prefilled_false_clears": tracked,
        "rule": NOTES["prefilled"],
    }
    if not history.is_file():
        summary["checked"] = False
        return {}, summary
    verdicts: dict[str, tuple[str, str | None]] = {}

    def verdict(source: str) -> tuple[str, str | None]:
        """-> ("model" | "unknown" | "independent", how)."""
        if source not in verdicts:
            table = source_table(paths, event_dir, source)
            named = any(layer in source for layer in layers)
            if table is not None and sidecar_of(table).is_file():
                text = sidecar_of(table).read_text(errors="replace")
                if any(sha in text for sha in shas):
                    verdicts[source] = ("model", "the table's sidecar names the checkpoint")
                elif named or any(marker in text for marker in derived):
                    verdicts[source] = ("model", "a layer of this model or derived from one")
                else:
                    verdicts[source] = ("independent", None)
            elif named:
                verdicts[source] = ("model", "a layer of this model by name (no sidecar)")
            else:
                verdicts[source] = ("unknown", "no table or sidecar for the source")
        return verdicts[source]

    anchors: dict[int, dict] = {}
    saves: dict[int, int] = {}
    lines = history.read_text().splitlines()
    for entry in (json.loads(line) for line in lines if line.strip()):
        shot = int(entry["shot"])
        saves[shot] = saves.get(shot, 0) + 1
        prefilled, source = entry.get("prefilled"), entry.get("source")
        if shot in anchors:
            if prefilled is False and tracked:
                anchors.pop(shot)  # the page carried exposure across saves: none here
            else:
                # opened on the anchored label: for an untracked event a reopened saved
                # shot records prefilled false all the same
                anchors[shot]["later_saves"] += 1
            continue
        if prefilled is False or source is None:
            continue  # nothing was prefilled
        kind, how = verdict(source)
        if kind == "independent":
            continue
        table = source_table(paths, event_dir, source)
        now = (None if table is None or entry.get("source_sha256") is None
               else sha256_of(table) == entry["source_sha256"])
        if now is False:
            how += "; the save was made against an earlier version of it (matched by its name)"
        anchors[shot] = {
            "reason": "truth prefilled from this model" if kind == "model"
            else "truth prefilled from a table of unknown producer (it may be this model)",
            "how": how,
            "source": source,
            "source_sha256": entry.get("source_sha256"),
            "table_now": now,
            "prefilled": prefilled,
            "saved_at": entry.get("saved_at"),
            "later_saves": 0,
        }
    for shot, anchor in anchors.items():
        anchor["saves"] = saves[shot]
    by_hand = {shot: anchors.pop(shot) for shot in sorted(set(cleared) & set(anchors))}
    summary.update(
        checked=True, n_shots=len(anchors),
        cleared_by_hand={str(shot): anchor for shot, anchor in by_hand.items()},
        cleared_not_anchored=sorted(set(cleared) - set(by_hand)),
    )
    return dict(sorted(anchors.items())), summary


class ShotSpecs(Dataset):
    """One item is one shot's whole record, read in a DataLoader worker while the GPU
    runs the batch before: its spectrogram ``(C, T, F)`` float16 and the Heidbrink
    annotation on the 10 ms frames."""

    def __init__(self, items: list[tuple[int, Path, str]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        shot, path, split = self.items[index]
        start = time.perf_counter()
        with np.load(path) as z:
            spec = np.ascontiguousarray(z["spec"].transpose(0, 2, 1))
            annotated = z["annotated"].astype(np.float64)
            stored = str(z["split"])
        if stored != split:
            raise ValueError(f"{path.name}: its split member says {stored!r}")
        if spec.shape[1] != SELD_FRAMES or annotated.size != SELD_FRAMES:
            # the frame scores place column k at (k + 0.5) x 2000 / SELD_FRAMES ms
            raise ValueError(
                f"{path.name}: {spec.shape[1]} frames and {annotated.size} annotation "
                f"columns, not the {SELD_FRAMES} of a 2 s record"
            )
        return {
            "shot": shot,
            "spec": torch.from_numpy(spec),
            "annotated": frame_mean(annotated) >= 0.5,
            "read_s": time.perf_counter() - start,
        }


def as_is(item: dict) -> dict:
    """The DataLoader's collate: one shot's item untouched (the main process batches)."""
    return item


def frame_scores(prob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(B, T)`` column probabilities -> ``(B, N_FRAMES)`` per 10 ms frame: the mean,
    and the share of columns >= 0.5 (`frame_mean` of both, on prob's device)."""
    index = torch.as_tensor(frame_index(prob.shape[1]), device=prob.device)
    count = torch.bincount(index, minlength=N_FRAMES).clamp(min=1).to(prob.dtype)
    zeros = torch.zeros(prob.shape[0], N_FRAMES, dtype=prob.dtype, device=prob.device)
    mean = zeros.index_add(1, index, prob) / count
    share = zeros.index_add(1, index, (prob >= 0.5).to(prob.dtype)) / count
    return mean, share


def mark(name: str) -> None:
    print(f"{name} {datetime.now(UTC).astimezone().strftime(STAMP)}", flush=True)


@torch.no_grad()
def predict_frames(
    models: dict[str, AeSeldNet], loader, device: torch.device, batch_shots: int,
    precision: str = "fp32",
) -> tuple[dict[str, dict[int, tuple[np.ndarray, np.ndarray]]], dict[int, np.ndarray], dict]:
    """Every checkpoint on every batch of `batch_shots` shots the loader yields, in
    float32 or (`precision` bf16, as task 7b scored) under bfloat16 autocast on CUDA.

    Returns each checkpoint's frame scores per shot (`frame_scores`, as numpy), each
    shot's annotation frames, and the loop's per-stage seconds. Between the
    LOOP_START and LOOP_END marks the GPU is meant to be busy; LOOP_WARM follows the
    first batch (CUDA start-up and cuDNN's first kernels are before it).
    """
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    frames: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {name: {} for name in models}
    annotated: dict[int, np.ndarray] = {}
    stages = dict.fromkeys(("wait", "read", "h2d", "gpu", "post"), 0.0)
    gpu_by_model = dict.fromkeys(models, 0.0)
    items = iter(loader)
    done = False
    batches = 0
    mark("LOOP_START")
    loop_start = time.perf_counter()
    while not done:
        pending, wait = [], 0.0
        while len(pending) < batch_shots:
            ask = time.perf_counter()
            try:
                pending.append(next(items))
            except StopIteration:
                done = True
                break
            finally:
                wait += time.perf_counter() - ask
        if not pending:
            break
        t0 = time.perf_counter()
        x = torch.stack([it["spec"].to(device, non_blocking=True) for it in pending]).float()
        sync()
        t1 = time.perf_counter()
        on_device = {}
        took = {}
        for name, model in models.items():
            start = time.perf_counter()
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda" and precision == "bf16",
            ):
                out = model(x)
            on_device[name] = frame_scores(torch.sigmoid(out.float()[..., 0]).double())
            sync()
            took[name] = time.perf_counter() - start
        t2 = time.perf_counter()
        shots = [it["shot"] for it in pending]
        for name, (mean, share) in on_device.items():
            mean, share = mean.cpu().numpy(), share.cpu().numpy()
            for i, shot in enumerate(shots):
                frames[name][shot] = (mean[i], share[i])
        for it in pending:
            annotated[it["shot"]] = it["annotated"]
        del x, out, on_device
        t3 = time.perf_counter()
        read = max(it["read_s"] for it in pending)
        for key, value in (("wait", wait), ("read", read), ("h2d", t1 - t0), ("gpu", t2 - t1), ("post", t3 - t2)):
            stages[key] += value
        for name, value in took.items():
            gpu_by_model[name] += value
        print(
            f"batch {batches}: {len(shots)} shots {shots[0]}..{shots[-1]} wait {wait:.3f} "
            f"read(max, in a worker) {read:.3f} h2d {t1 - t0:.3f} gpu {t2 - t1:.3f} "
            f"({' '.join(f'{n} {s:.3f}' for n, s in took.items())}) post {t3 - t2:.3f}",
            flush=True,
        )
        if batches == 0:
            mark("LOOP_WARM")
        batches += 1
    loop_s = time.perf_counter() - loop_start
    mark("LOOP_END")
    timing = {
        "loop_s": loop_s,
        "shots": len(annotated),
        "batches": batches,
        "stage_s": stages,
        "gpu_s_by_checkpoint": gpu_by_model,
    }
    return frames, annotated, timing


def score_shots(
    shots: list[int],
    scores: dict[int, tuple[np.ndarray, np.ndarray]],
    truths: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    interval: bool,
) -> dict:
    """`metrics` of the frame scores of `shots` against one truth (each shot's
    (counted frames, present frames)), with the shot-bootstrap interval if asked."""
    parts = []
    for shot in shots:
        keep, truth = truths[shot]
        mean, share = scores[shot]
        parts.append((mean[keep], share[keep] >= 0.5, truth[keep]))
    score, hard, truth = (np.concatenate([p[j] for p in parts]) for j in range(3))
    out = {
        **metrics(score, hard, truth),
        "n_shots": len(shots),
        "n_frames": int(truth.size),
        "truth_fraction": float(truth.mean()) if truth.size else float("nan"),
    }
    if interval:
        with warnings.catch_warnings():
            # a replicate with one class has no AUROC; a set of them, no interval
            warnings.simplefilter("ignore", RuntimeWarning)
            out["ci95"] = bootstrap(parts)
    return out


def score_lists(
    scored: dict[str, list[int]],
    frames: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
    states: dict[int, np.ndarray],
    annotated: dict[int, np.ndarray],
    workers: int = 0,
) -> dict[str, dict[str, dict]]:
    """`results.<list>.<checkpoint>.<truth>` for every list with a shot to score.

    Intervals are on the reviewed truth. Shots are taken in shot order, so lists
    holding the same shots in any order get the same numbers, computed once. With
    `workers` > 1 the scores are computed in that many processes, largest first
    (the bootstrap is most of the time; the GPU is done by then)."""
    truths = {
        "reviewed": {s: (counted(states[s]), states[s] == PRESENT) for s in annotated},
        "annotated": {s: (np.ones(N_FRAMES, bool), annotated[s]) for s in annotated},
    }
    keys = sorted(
        {
            (ckpt, truth_name, tuple(sorted(shots)))
            for shots in scored.values() if shots
            for ckpt in frames for truth_name in truths
        },
        key=lambda key: (-len(key[2]) * (key[1] == "reviewed"), key),
    )

    def task(key):
        ckpt, truth_name, shots = key
        return (list(shots), frames[ckpt], truths[truth_name])

    if workers > 1 and len(keys) > 1:
        fork = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(min(workers, len(keys)), mp_context=fork) as pool:
            futures = {
                key: pool.submit(score_shots, *task(key), interval=key[1] == "reviewed")
                for key in keys
            }
            done = {key: future.result() for key, future in futures.items()}
    else:
        done = {key: score_shots(*task(key), interval=key[1] == "reviewed") for key in keys}
    return {
        name: {
            ckpt: {t: done[(ckpt, t, tuple(sorted(shots)))] for t in truths}
            for ckpt in frames
        }
        for name, shots in scored.items() if shots
    }


def cautions(
    sets: dict[str, dict], results: dict[str, dict], *,
    trained_on: set[int] = frozenset(), prefilled: set[int] = frozenset(),
    checked: bool = True,
) -> dict[str, list[str]]:
    """What each list's numbers do not mean, from its envelope and scores
    (`trained_on`: the train-split shots; `prefilled`: the shots whose truth this
    model prefilled; `checked`: whether there was a review history to check that)."""
    out: dict[str, list[str]] = {}
    for name, env in sets.items():
        said = []
        if env["n_scored"] == 0:
            said.append("no shot to score: nothing below is a score")
        elif env["n_scored"] < SMALL:
            said.append(f"{env['n_scored']} shots: the shot-bootstrap interval is wide")
        if env["missing"]["no_input"]:
            train = len(set(env["missing"]["no_input"]) & set(trained_on))
            also = f", {train} of them train split (unscorable names them)" if train else ""
            said.append(
                f"{len(env['missing']['no_input'])} listed shots have no npz (notes.no_input){also}"
            )
        unscored = {*env["missing"]["no_input"], *env["missing"]["no_reference"], *env["trained_on"]}
        left = sorted(set(prefilled) & unscored)
        if left:
            alone = len(set(left) & set(env["missing"]["no_reference"]))
            said.append(
                f"{len(left)} listed shots have truth prefilled from this model, so they are not "
                f"scored (notes.prefilled; {alone} for that alone, unscorable gives the rest's "
                f"other reason): {', '.join(map(str, left))}"
            )
        if not checked:
            said.append("no review history beside the reference: prefill from this model is unchecked")
        if env["trained_on"]:
            how = "scored anyway: leaky" if env["leaky"] else "not scored"
            said.append(f"{len(env['trained_on'])} listed shots are trained-on, {how} (notes.trained_on)")
        shipped = (results.get(name) or {}).get(SHIPPED, {}).get("reviewed")
        if shipped and not 0.0 < shipped["truth_fraction"] < 1.0:
            said.append("the reviewed frames are of one class: AUROC and AUPRC are undefined")
        if said:
            out[name] = said
    return out


def clean(value):
    """The record without NaN or numpy scalars (NaN -> null), so it is plain JSON."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        write(fh)
    tmp.replace(path)


def plain_args(args) -> dict:
    """The command line as the record keeps it (paths as text)."""
    def plain(value):
        if isinstance(value, list):
            return [plain(v) for v in value]
        return str(value) if isinstance(value, Path) else value

    return {key: plain(value) for key, value in vars(args).items()}


def check_lists(paths: Paths, names: list[str]) -> None:
    """Stop with the known lists' names if one of `names` is not a shot list."""
    for name in names:
        try:
            shotlists.entry(paths, name)
        except (KeyError, shotlists.ShotListError) as err:
            known = ", ".join(shotlists.names(paths))
            raise SystemExit(f"shot list {name!r}: {err}; known lists: {known}") from None


def evaluate_lists(args) -> int:
    paths = Paths.from_env()
    names = list_names(args.shotlist)
    check_lists(paths, names)
    population = npz_population(args.dataset_dir)
    trained_on = {shot for shot, (_, split) in population.items() if split == "train"}
    reference = Path(args.reference or review_labels.labels_path(Path(paths.label_tables) / EVENT))
    states = reference_states(reference)
    covered = {shot for shot, st in states.items() if counted(st).any()}
    ckpts = sorted(args.models_dir.glob("ae_seldnet_*.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {args.models_dir}")
    shas = {path: sha256_of(path) for path in ckpts}
    prefilled, prefill_check = model_prefills(
        paths, reference, set(shas.values()), cleared=set(args.independent_truth)
    )
    try:
        scored, sets = shotlists.plan(
            paths, names, population, trained_on, covered - set(prefilled),
            include_trained=args.include_trained, scoring="heldout",
        )
    except shotlists.ShotListError as err:
        raise SystemExit(str(err)) from None
    for name, env in sets.items():
        print(
            f"{name}: {env['n_listed']} listed, {env['n_scored']} scored, "
            f"{len(env['trained_on'])} trained-on, "
            + ", ".join(f"{len(v)} {k}" for k, v in env["missing"].items()),
            flush=True,
        )
    print(f"{len(prefilled)} reviewed shots have truth prefilled from this model: "
          f"{sorted(prefilled)}", flush=True)

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    models: dict[str, AeSeldNet] = {}
    checkpoints: dict[str, dict] = {}
    for path in ckpts:
        name = path.stem.removeprefix("ae_seldnet_")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model = AeSeldNet(AeSeldNetConfig.from_dict(blob["config"]))
        model.load_state_dict(blob["state_dict"])
        models[name] = model.to(device).eval()
        checkpoints[name] = {
            "path": str(path),
            "sha256": shas[path],
            "target": blob.get("target"),
            "loss": blob.get("loss_config", {}).get("loss"),
            "best_epoch": blob.get("epoch"),
            "val_loss": blob.get("val_loss"),
        }

    todo = sorted({shot for shots in scored.values() for shot in shots})
    items = [(shot, *population[shot]) for shot in todo]
    print(f"{len(todo)} shots to run through {len(models)} checkpoints on {device}, "
          f"{args.workers} loader workers, {args.batch_shots} shots a batch", flush=True)
    frames: dict = {name: {} for name in models}
    annotated: dict = {}
    timing: dict = {"loop_s": 0.0, "shots": 0, "batches": 0}
    if items:
        loader = DataLoader(
            ShotSpecs(items), batch_size=None, collate_fn=as_is, num_workers=args.workers,
            prefetch_factor=2 if args.workers else None, pin_memory=device.type == "cuda",
        )
        frames, annotated, timing = predict_frames(
            models, loader, device, args.batch_shots, args.precision
        )
    timing.update(workers=args.workers, batch_shots=args.batch_shots)
    start = time.perf_counter()
    results = score_lists(scored, frames, states, annotated, args.workers)
    timing["scoring_s"] = time.perf_counter() - start

    out = Path(args.out)
    record = {
        "generated": datetime.now(UTC).astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": git_sha(),
        "git_dirty": git_dirty(),
        "args": plain_args(args),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "event": EVENT,
        "dataset_dir": str(args.dataset_dir),
        "population": {
            "npz": len(population),
            "train": len(trained_on),
            "valid": len(population) - len(trained_on),
        },
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": args.precision if device.type == "cuda" else "fp32",
        "reference": {
            "path": str(reference),
            "sha256": sha256_of(reference),
            "n_shots": len(states),
            "n_covered": len(covered),
            "covered": "a present or absent frame in 0-2 s",
        },
        "frames": {
            "frame_ms": 10,
            "span_ms": [0, 2000],
            "score": "mean of the record's 0.256 ms column probabilities in each frame",
            "hard": "half the frame's columns >= 0.5",
            "threshold": 0.5,
        },
        "bootstrap": {"replicates": REPLICATES, "seed": SEED, "over": "shots", "truth": "reviewed"},
        "scoring": "heldout",
        "include_trained": bool(args.include_trained),
        "leaky": any(env["leaky"] for env in sets.values()),
        "shipped": SHIPPED,
        "checkpoints": checkpoints,
        "prefilled": {**prefill_check, "shots": {str(k): v for k, v in prefilled.items()}},
        "sets": sets,
        "results": results,
        "unscorable": {
            name: unscorable(env, trained_on, covered, set(states), prefilled, no_input="no dataset npz")
            for name, env in sets.items()
        },
        "cautions": cautions(
            sets, results, prefilled=set(prefilled), checked=prefill_check.get("checked", False)
        ),
        "notes": NOTES,
        "frame_scores": out.with_name(out.stem + ".frames.npz").name,
        "timing": timing,
    }
    kept = json.loads(out.read_text()).get("corpus") if out.is_file() else None
    kept_ref = ((kept or {}).get("reference") or {}).get("sha256")
    if kept and kept_ref == record["reference"]["sha256"]:
        # the corpus chain's block is its own run's (--chain corpus): keep it, and say
        # what moved under it
        kept["kept_by_dataset_run"] = {
            "generated": record["generated"],
            "slurm_job": record["slurm_job"],
            "same_reference": True,
            "note": (
                "The dataset chain was rerun after this block was written: the block's "
                "agreement used the dataset frame scores of that time. Rerun --chain corpus "
                "to refresh it."
            ),
        }
        record["corpus"] = kept
    elif kept:
        # one record, one truth: a block scored against another reference goes
        record["corpus_dropped"] = {
            "generated": kept.get("generated"),
            "slurm_job": kept.get("slurm_job"),
            "reference_sha256": kept_ref,
            "note": (
                "The corpus block scored another reference than this dataset run, so it was "
                "dropped. Rerun --chain corpus on this reference to restore it."
            ),
        }
        print(f"dropped the corpus block of {out}: it scored reference {kept_ref}; rerun "
              "--chain corpus", flush=True)
    arrays = {"shots": np.asarray(todo, dtype=np.int64)}
    if todo:
        arrays["annotated"] = np.stack([annotated[s] for s in todo])
        for name, scores in frames.items():
            arrays[f"{name}_mean"] = np.stack([scores[s][0] for s in todo])
            arrays[f"{name}_share"] = np.stack([scores[s][1] for s in todo])
    write_atomic(out.with_name(out.stem + ".frames.npz"), lambda fh: np.savez(fh, **arrays))
    text = json.dumps(clean(record), indent=1, allow_nan=False) + "\n"
    write_atomic(out, lambda fh: fh.write(text.encode()))
    for name in results:
        got = results[name].get(SHIPPED, {}).get("reviewed")
        if got:
            print(f"{name} ({SHIPPED}, reviewed): AUROC {got['auroc']:.3f} AUPRC {got['auprc']:.3f} "
                  f"F1 {got['f1']:.3f} on {got['n_shots']} shots, {got['n_frames']} frames", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


# --------------------------------------------------------------------------
# the corpus chain (--chain corpus)


def corpus_population(dirs: list[Path]) -> tuple[dict[int, Path], dict[int, list[str]]]:
    """Each shot with a corpus-chain cache `<shot>.npz` in `dirs` -> its path (the
    first dir that holds it wins), and the shots more than one dir holds -> those dirs."""
    found: dict[int, Path] = {}
    held: dict[int, list[str]] = {}
    for folder in dirs:
        for path in sorted(Path(folder).glob("*.npz")):
            if not path.stem.isdigit():
                continue
            shot = int(path.stem)
            held.setdefault(shot, []).append(str(folder))
            found.setdefault(shot, path)
    return found, {shot: where for shot, where in held.items() if len(where) > 1}


def corpus_frames(
    t_ms: np.ndarray, prob: np.ndarray, end_ms: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One shot's deployed-chain column probabilities on the 10 ms frames of 0-2 s, as
    ae_gold_extend.py frames them: the mean probability and the share of columns >= 0.5
    in each frame, and whether the frame is assessed (a whole frame inside [0, end_ms]
    with at least one column)."""
    k = np.floor(np.asarray(t_ms, np.float64) / 10.0).astype(np.int64)
    ok = (k >= 0) & (k < N_FRAMES)
    prob = np.asarray(prob, np.float64)[ok]
    count = np.bincount(k[ok], minlength=N_FRAMES)
    mean = np.bincount(k[ok], weights=prob, minlength=N_FRAMES) / np.maximum(count, 1)
    share = np.bincount(k[ok], weights=(prob >= 0.5).astype(float), minlength=N_FRAMES)
    assessed = (np.arange(N_FRAMES) < int(end_ms // 10)) & (count > 0)
    return mean, share / np.maximum(count, 1), assessed


def read_corpus(path: Path) -> dict:
    """A corpus-chain cache on the 10 ms frames (`corpus_frames`) and how it was made."""
    with np.load(path) as z:
        mean, share, assessed = corpus_frames(z["t_ms"], z["prob"], float(z["plasma_end_ms"]))
        run_on = str(z["run_on"]) if "run_on" in z.files else "not recorded"
        end = float(z["plasma_end_ms"])
    return {"mean": mean, "share": share, "assessed": assessed, "run_on": run_on, "end_ms": end}


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa of two boolean calls (NaN when both say one class throughout)."""
    seen = float((a == b).mean())
    chance = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (seen - chance) / (1 - chance) if chance < 1 else float("nan")


def agreement(
    shots: list[int], dataset: dict[int, tuple[np.ndarray, np.ndarray]],
    corpus: dict[int, dict], states: dict[int, np.ndarray],
) -> dict:
    """The two chains on the frames of `shots` the corpus chain assesses: how close the
    frame means and hard calls are, and both chains scored against the reviewed truth
    on the same frames (shot-bootstrap intervals)."""
    if not shots:
        return {"n_shots": 0}
    per_shot, parts = {}, []
    for shot in shots:
        keep = corpus[shot]["assessed"]
        d_mean, d_share = (x[keep] for x in dataset[shot])
        c_mean, c_share = corpus[shot]["mean"][keep], corpus[shot]["share"][keep]
        parts.append((d_mean, c_mean, d_share >= 0.5, c_share >= 0.5))
        per_shot[str(shot)] = {
            "n_frames": int(keep.sum()),
            "mean_abs_diff": float(np.abs(d_mean - c_mean).mean()) if keep.any() else None,
            "call_agreement": float(((d_share >= 0.5) == (c_share >= 0.5)).mean()) if keep.any() else None,
        }
    d_mean, c_mean, d_call, c_call = (np.concatenate([p[j] for p in parts]) for j in range(4))
    keep_rev = {s: counted(states[s]) & corpus[s]["assessed"] for s in shots}
    truths = {s: (keep_rev[s], states[s] == PRESENT) for s in shots}
    corpus_scores = {s: (corpus[s]["mean"], corpus[s]["share"]) for s in shots}
    return {
        "n_shots": len(shots),
        "n_frames": int(d_mean.size),
        "pearson_r": float(np.corrcoef(d_mean, c_mean)[0, 1]) if d_mean.size > 1 else None,
        "mean_abs_diff": float(np.abs(d_mean - c_mean).mean()),
        "call_agreement": float((d_call == c_call).mean()),
        "kappa": kappa(d_call, c_call),
        "present_fraction": {"dataset": float(d_call.mean()), "corpus": float(c_call.mean())},
        "reviewed": {
            "dataset": score_shots(shots, dataset, truths, interval=True),
            "corpus": score_shots(shots, corpus_scores, truths, interval=True),
        },
        "per_shot": per_shot,
    }


def unscorable(
    env: dict, trained_on: set[int], covered: set[int], reviewed: set[int],
    prefilled: dict[int, dict] | None = None, *, no_input: str = "no corpus-chain npz",
) -> dict[str, str]:
    """Each listed shot a chain does not score -> why (`covered`: the shots with a
    present or absent reviewed frame in 0-2 s; `reviewed`: those in the table;
    `prefilled`: the shots whose truth this model prefilled -> `model_prefills`' entry)."""
    prefilled = prefilled or {}
    why: dict[str, str] = {}
    for shot in env["missing"]["no_input"]:
        also = " (and a train-split shot)" if shot in trained_on else ""
        why[str(shot)] = f"{no_input}{also}"
    if not env["leaky"]:
        for shot in env["trained_on"]:
            why[str(shot)] = "train split: the checkpoint was fitted on it"
    for shot in env["missing"]["no_reference"]:
        if shot in prefilled:
            why[str(shot)] = prefilled[shot]["reason"]
        elif shot in covered:
            why[str(shot)] = "its reviewed frames all lie outside the chain's window"
        elif shot in reviewed:
            why[str(shot)] = "no present or absent reviewed frame in 0-2 s"
        else:
            why[str(shot)] = "not in the review table"
    for shot, reason in why.items():
        if int(shot) in prefilled and reason != prefilled[int(shot)]["reason"]:
            why[shot] = f"{reason}; and {prefilled[int(shot)]['reason']}"
    return why


def corpus_trained_on(dataset_dir: Path) -> tuple[set[int], dict]:
    """The shots the corpus chain may not score as held out: the AE dataset's train
    split, and any card training shot (`TRAINING_SHOTS`) with no npz there, whose split
    is unknown. Refuses a dataset dir with no train split: a wrong --dataset-dir would
    otherwise score the fitted shots as held out."""
    population = npz_population(dataset_dir)
    train = {shot for shot, (_, split) in population.items() if split == "train"}
    if not train:
        raise SystemExit(
            f"{dataset_dir}: no <shot>_train.npz, so the trained-on guard would be empty; "
            "--dataset-dir must be the AE dataset the checkpoint was fitted on"
        )
    unknown = set(TRAINING_SHOTS) - set(population)
    return train | unknown, {
        "dataset_dir": str(dataset_dir),
        "npz": len(population),
        "train": len(train),
        "valid": len(population) - len(train),
        "card_training_shots_without_npz": sorted(unknown),
        "rule": "the train split, and any card training shot without an npz (split unknown)",
    }


def cache_checkpoint(
    folder: Path, table: str | None, ckpt: Path, ckpt_sha: str, label_tables: Path
) -> dict:
    """Whether a work dir's caches came from `ckpt` (CORPUS_NOTES.checkpoint): the file
    unchanged since the dir's first npz, and the sidecar of the table built from the dir
    (`table`, under the label tables) naming the same sha256."""
    stamps = [path.stat().st_mtime for path in Path(folder).glob("*.npz")]
    unchanged = bool(stamps) and ckpt.stat().st_mtime < min(stamps)
    out: dict = {
        "checkpoint_mtime": datetime.fromtimestamp(ckpt.stat().st_mtime, UTC).isoformat(timespec="seconds"),
        "first_npz": datetime.fromtimestamp(min(stamps), UTC).isoformat(timespec="seconds") if stamps else None,
        "unchanged_since_first_npz": unchanged,
        "sidecar": None,
    }
    ok = unchanged
    if table is not None:
        side = sidecar_of(Path(label_tables) / f"{table}.csv")
        recorded = (
            (json.loads(side.read_text()).get("model") or {}).get("checkpoint_sha256")
            if side.is_file() else None
        )
        out["sidecar"] = {"path": str(side), "checkpoint_sha256": recorded, "same": recorded == ckpt_sha}
        ok = ok and recorded == ckpt_sha
    out["same_checkpoint"] = ok
    return out


def evaluate_corpus(args) -> int:
    paths = Paths.from_env()
    names = list_names(args.shotlist)
    check_lists(paths, names)
    trained_on, trained_from = corpus_trained_on(args.dataset_dir)
    reference = Path(args.reference or review_labels.labels_path(Path(paths.label_tables) / EVENT))
    states = reference_states(reference)
    covered = {shot for shot, st in states.items() if counted(st).any()}
    ckpt = Path(CORPUS_CHECKPOINT)
    if not ckpt.is_file():
        raise SystemExit(f"{ckpt}: the chain's checkpoint is not there; its caches cannot be checked")
    ckpt_sha = sha256_of(ckpt)
    if args.corpus_work:
        work = [(Path(d), None) for d in args.corpus_work]
    else:
        work = [(Path(paths.root) / d, table) for d, table in CORPUS_WORK]
    dirs = [d for d, _ in work]
    population, duplicates = corpus_population(dirs)
    prefilled, prefill_check = model_prefills(
        paths, reference, {ckpt_sha}, cleared=set(args.independent_truth)
    )

    out = Path(args.out)
    record = json.loads(out.read_text()) if out.is_file() else {"event": EVENT}
    dataset_ref = (record.get("reference") or {}).get("sha256")
    reference_sha = sha256_of(reference)
    if dataset_ref is not None and dataset_ref != reference_sha:
        raise SystemExit(
            f"{out}'s dataset chain scored a reference with sha256 {dataset_ref}, this run's "
            f"is {reference_sha}: rerun the dataset chain on it first, or write another --out"
        )

    start = time.perf_counter()
    listed = sorted({s for name in names for s in shotlists.shots(paths, name)} & set(population))
    checks = {
        str(d): cache_checkpoint(d, table, ckpt, ckpt_sha, Path(paths.label_tables))
        for d, table in work
    }
    wrong = [d for d in dirs if any(population[s].parent == d for s in listed)
             and not checks[str(d)]["same_checkpoint"]]
    if wrong:
        raise SystemExit(
            f"caches not shown to come from {ckpt} (sha256 {ckpt_sha[:12]}): "
            + "; ".join(f"{d}: {checks[str(d)]}" for d in wrong)
        )
    corpus = {shot: read_corpus(population[shot]) for shot in listed}
    read_s = time.perf_counter() - start
    # covered here: a present or absent reviewed frame among the frames the chain
    # assesses, on a shot whose truth this model did not prefill
    in_window = {
        s for s in listed
        if s in states and s not in prefilled and (counted(states[s]) & corpus[s]["assessed"]).any()
    }
    try:
        scored, sets = shotlists.plan(
            paths, names, population, trained_on,
            in_window | (covered - set(population) - set(prefilled)),
            include_trained=args.include_trained, scoring="heldout",
        )
    except shotlists.ShotListError as err:
        raise SystemExit(str(err)) from None
    for name, env in sets.items():
        print(
            f"corpus {name}: {env['n_listed']} listed, {env['n_scored']} scored, "
            f"{len(env['trained_on'])} trained-on, "
            + ", ".join(f"{len(v)} {k}" for k, v in env["missing"].items()),
            flush=True,
        )
    print(f"{len(prefilled)} reviewed shots have truth prefilled from this model: "
          f"{sorted(prefilled)}", flush=True)

    start = time.perf_counter()
    truths = {s: (counted(states[s]) & corpus[s]["assessed"], states[s] == PRESENT) for s in in_window}
    scores = {s: (corpus[s]["mean"], corpus[s]["share"]) for s in in_window}
    keys = sorted({tuple(sorted(shots)) for shots in scored.values() if shots}, key=lambda k: (-len(k), k))
    if args.workers > 1 and len(keys) > 1:
        fork = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(min(args.workers, len(keys)), mp_context=fork) as pool:
            futures = {k: pool.submit(score_shots, list(k), scores, truths, interval=True) for k in keys}
            done = {k: future.result() for k, future in futures.items()}
    else:
        done = {k: score_shots(list(k), scores, truths, interval=True) for k in keys}
    results = {
        name: {"reviewed": done[tuple(sorted(shots))]} for name, shots in scored.items() if shots
    }

    dataset_frames = Path(args.dataset_frames or out.with_name(out.stem + ".frames.npz"))
    shared: list[int] = []
    dataset: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if dataset_frames.is_file():
        with np.load(dataset_frames) as z:
            if f"{SHIPPED}_mean" in z.files:
                for i, shot in enumerate(z["shots"].tolist()):
                    dataset[shot] = (z[f"{SHIPPED}_mean"][i], z[f"{SHIPPED}_share"][i])
        shared = sorted(
            s for s in dataset if s in in_window and (args.include_trained or s not in trained_on)
        )
    compared = agreement(shared, dataset, corpus, states)
    timing = {"read_s": read_s, "scoring_s": time.perf_counter() - start, "workers": args.workers}

    shipped_sha = (record.get("checkpoints") or {}).get(SHIPPED, {}).get("sha256")
    dataset_prefill = record.get("prefilled") or {}
    record.pop("corpus_dropped", None)  # this run restores the block
    run_on = sorted({corpus[s]["run_on"] for s in listed})
    record["corpus"] = {
        "generated": datetime.now(UTC).astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": git_sha(),
        "git_dirty": git_dirty(),
        "args": plain_args(args),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "chain": "ae_gold_extend.py (the model card's chain over the whole CO2 record)",
        "checkpoint": {
            "path": str(ckpt),
            "sha256": ckpt_sha,
            "same_as_dataset_chain_shipped": None if shipped_sha is None else ckpt_sha == shipped_sha,
        },
        "work_dirs": [
            {"path": str(d), "npz": len(list(d.glob("*.npz"))),
             "listed_used": sum(1 for s in listed if population[s].parent == d),
             "checkpoint": checks[str(d)]}
            for d in dirs
        ],
        "duplicates": {str(s): where for s, where in sorted(duplicates.items()) if s in set(listed)},
        "run_on": {how: sum(1 for s in listed if corpus[s]["run_on"] == how) for how in run_on},
        "population": {"npz": len(population), "listed": len(listed)},
        "trained_on_from": trained_from,
        "reference": {
            "path": str(reference),
            "sha256": reference_sha,
            "n_shots": len(states),
            "covered": "a present or absent frame among the frames the chain assesses",
            "same_as_dataset_chain": None if dataset_ref is None else dataset_ref == reference_sha,
        },
        "prefilled": {
            **prefill_check,
            "shots": {str(k): v for k, v in prefilled.items()},
            "same_as_dataset_chain": None if "shots" not in dataset_prefill
            else sorted(dataset_prefill["shots"]) == sorted(map(str, prefilled)),
        },
        "frames": {
            "frame_ms": 10,
            "span_ms": [0, 2000],
            "score": "mean of the smoothed column probabilities in each frame",
            "hard": "half the frame's columns >= 0.5",
            "assessed": "a whole frame inside [0, window end] with STFT columns",
            "threshold": 0.5,
        },
        "bootstrap": {"replicates": REPLICATES, "seed": SEED, "over": "shots", "truth": "reviewed"},
        "scoring": "heldout",
        "include_trained": bool(args.include_trained),
        "leaky": any(env["leaky"] for env in sets.values()),
        "sets": sets,
        "results": results,
        "unscorable": {
            name: unscorable(env, trained_on, covered, set(states), prefilled)
            for name, env in sets.items()
        },
        "cautions": cautions(
            sets, {name: {SHIPPED: got} for name, got in results.items()},
            trained_on=trained_on, prefilled=set(prefilled),
            checked=prefill_check.get("checked", False),
        ),
        "agreement": {"dataset_frames": str(dataset_frames), "checkpoint": SHIPPED, **compared},
        "notes": CORPUS_NOTES,
        "frame_scores": out.with_name(out.stem + ".corpus_frames.npz").name,
        "timing": timing,
    }
    arrays = {"shots": np.asarray(listed, dtype=np.int64)}
    if listed:
        for key in ("mean", "share", "assessed"):
            arrays[key] = np.stack([corpus[s][key] for s in listed])
    write_atomic(out.with_name(out.stem + ".corpus_frames.npz"), lambda fh: np.savez(fh, **arrays))
    text = json.dumps(clean(record), indent=1, allow_nan=False) + "\n"
    write_atomic(out, lambda fh: fh.write(text.encode()))
    for name, got in results.items():
        got = got["reviewed"]
        print(f"corpus {name} (reviewed): AUROC {got['auroc']:.3f} AUPRC {got['auprc']:.3f} "
              f"F1 {got['f1']:.3f} on {got['n_shots']} shots, {got['n_frames']} frames", flush=True)
    if shared:
        print(f"agreement on {compared['n_shots']} shots, {compared['n_frames']} frames: "
              f"r {compared['pearson_r']:.3f}, calls agree {compared['call_agreement']:.3f}, "
              f"kappa {compared['kappa']:.3f}", flush=True)
    print(f"wrote the corpus block of {out}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--models-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--fig-dir", type=Path, default=REPO / "outputs" / "labeler" / "ae" / "training"
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument(
        "--shotlist", action="append", default=[], metavar="NAME[,NAME]",
        help="score these named shot lists instead of the validation split (repeatable)",
    )
    p.add_argument(
        "--include-trained", action="store_true",
        help="--shotlist: score the trained-on shots too and mark the record leaky",
    )
    p.add_argument(
        "--reference", type=Path, default=None,
        help="--shotlist: the reviewed AE table (default the event's review/labels.csv)",
    )
    p.add_argument("--out", type=Path, default=LIST_OUT, help="--shotlist: the record")
    p.add_argument(
        "--workers", type=int, default=max(len(os.sched_getaffinity(0)) - 1, 0),
        help="--shotlist: DataLoader processes reading the npz, then scoring processes "
        "(default: CPUs less one)",
    )
    p.add_argument("--batch-shots", type=int, default=8, help="--shotlist: shots a forward pass")
    p.add_argument(
        "--precision", choices=("fp32", "bf16"), default="fp32",
        help="--shotlist on CUDA: float32 (as the corpus inference runs), or bfloat16 "
        "autocast (task 7b's scoring; its GRU misses cuDNN and runs ~20x slower)",
    )
    p.add_argument(
        "--chain", choices=("dataset", "corpus"), default="dataset",
        help="--shotlist: dataset (default) runs every checkpoint on the dataset npz; corpus "
        "scores the deployed chain's per-shot caches (--corpus-work) and writes --out's corpus block",
    )
    p.add_argument(
        "--corpus-work", type=Path, action="append", default=[], metavar="DIR",
        help="--chain corpus: an ae_gold_extend.py work dir (repeatable; the first that holds a "
        "shot wins; default CORPUS_WORK under $LABELER_ROOT)",
    )
    p.add_argument(
        "--independent-truth", action="append", default=[], metavar="SHOT[,SHOT]",
        help="--shotlist: shots the owner re-labelled from scratch after a prefill from this "
        "model; they count again (the owner's call; the record lists them under "
        "prefilled.cleared_by_hand)",
    )
    p.add_argument(
        "--dataset-frames", type=Path, default=None,
        help="--chain corpus: the dataset chain's frame scores to compare with (default "
        "<out stem>.frames.npz)",
    )
    args = p.parse_args(argv)
    if args.chain == "dataset" and (args.corpus_work or args.dataset_frames):
        p.error("--corpus-work and --dataset-frames go with --chain corpus")
    if args.shotlist:
        if not list_names(args.shotlist):
            p.error("--shotlist names no list")
        try:
            args.independent_truth = sorted({int(s) for s in list_names(args.independent_truth)})
        except ValueError:
            p.error("--independent-truth takes shot numbers")
        return evaluate_corpus(args) if args.chain == "corpus" else evaluate_lists(args)
    if args.include_trained or args.reference or args.chain == "corpus" or args.independent_truth:
        p.error("--include-trained, --reference, --independent-truth and --chain corpus go "
                "with --shotlist")

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
