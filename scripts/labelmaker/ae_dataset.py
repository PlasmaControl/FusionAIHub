"""Build the frame-level AE activity labels and training dataset (task 7a).

Task 6 measured tokeye's ``big_tf_unet`` coherent mask against the 180
hand-annotated shots and said GO, but it also showed why the raw mask cannot be
the label as it stands: 61.3 % of all frames were called active at occupancy
0.01 (the annotation says 21.0 %), because broadband ELM-like streaks leak into
the coherent channel. This script cleans the mask first, with rules fixed by the
task 7a brief and implemented in ``labelmaker.ae.labels``:

    coherent = sigmoid(ch0) >= 0.2
    transient = sigmoid(ch1) >= 0.2
    mode mask = coherent & ~transient
    binary opening along TIME only, PERSIST_FRAMES = 20 (5.1 ms), per bin
    notch (per channel and bin) any bin lit in > NOTCH threshold of the record
    active[f] = band occupancy over bins 164-511 >= the chosen occupancy threshold

and then fixes the two thresholds by stated rules rather than by eye:

notch
    sweep {0.5, 0.6, 0.7, 0.8, 0.9}; take the SMALLEST threshold at which no bin
    inside an annotated AE window's centroid +-3 bins is removed on more than 2
    shots. Smaller means more aggressive, so the rule buys as much instrument
    line rejection as the protected bins allow.

occupancy
    sweep {0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1}; take the LOWEST threshold
    whose predicted-positive frame fraction is <= 0.40 while pooled recall of
    annotated frames stays >= 0.80. If none satisfies both, take the one with
    recall >= 0.80 and the smallest positive fraction and say so.

Human review (brief item 5): every shot where the notch removes a bin below
190 kHz - `176526_train`'s 124.0-127.9 kHz in `r0` is the case task 6 found -
is written to ``notch_review.json`` with ``needs_review: true``, and that
channel's notch is NOT applied to that shot's label; the other channels' notches
still are.

Stages
------
``probs`` (GPU)
    per shot: read the cached arrow time series, run ``tokeye``'s transform per
    channel, standardise per array as ``tokeye.inference.model_infer`` does, run
    ``big_tf_unet`` once per batch of channels, and write BOTH sigmoid channels
    as uint8 (floor(p * 255), so ``>= 51`` is exactly ``p >= 0.2``) under
    ``<root>/masks/<shot>_probs.npz``, the cleaned and raw packed masks and the
    small per-shot series under ``<root>/masks/<shot>_clean.npz``, and the
    band-restricted standardised spectrogram into ``<root>/dataset/<shot>.npz``.
``dataset`` (CPU, no GPU needed - this is why the probabilities are saved)
    the two threshold sweeps, ``notch_review.json`` / ``notch_review.png``, the
    per-shot labels written back into ``<root>/dataset/<shot>.npz``, and
    ``dataset.json``.
``jobstats``
    merge a captured ``jobstats`` report into ``dataset.json`` after the job.

Nothing is written into ``/scratch/gpfs/nc1514/aemodes`` or
``/scratch/gpfs/nc1514/tokeye``; both are read-only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labelmaker.ae.labels import (  # the sys.path insert above makes this work
    BAND_HI_BIN,
    BAND_LO_BIN,
    HOP,
    N_BINS,
    N_FFT,
    PERSIST_FRAMES,
    PROB_THRESHOLD,
    apply_notch,
    band_occupancy,
    bin_active_fraction,
    bin_freqs_khz,
    centroid_khz,
    clean_mask,
    contiguous_runs,
    notch_bins,
    persist_open,
    power_weights,
)

# --- constants fixed by the spec and the task 7a brief -----------------------

TS_CACHE = Path("/scratch/gpfs/nc1514/aemodes/data/.cache/ae_timeseries")
CHECKPOINT = Path("/scratch/gpfs/nc1514/tokeye/model/big_tf_unet_251210.pt")

CHANNELS = ("r0", "v1", "v2", "v3")
N_LABELS = 5
LABEL_NAMES = ("lfm", "bae", "eae", "rsae", "tae")
AE_LABELS = (1, 2, 3, 4)  # label_0 (LFM) is excluded from the AE annotation

TIME_RANGE_MS = 2000.0
N_FRAMES = 7820
N_SAMPLES = 1000001
TRANSFORM = "tokeye"

#: uint8 code of the 0.2 probability threshold; floor(p*255) >= 51 <=> p >= 0.2.
PROB_CODE = 51

NOTCH_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
PROTECT_HALF_WIDTH = 3  # bins either side of an annotated window's centroid
PROTECT_MAX_SHOTS = 2  # "removed on more than 2 shots" fails the threshold

OCC_THRESHOLDS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1)
MAX_POSITIVE_FRAC = 0.40
MIN_POOLED_RECALL = 0.80

REVIEW_MAX_KHZ = 190.0  # a notched bin below this needs a human look

# --- shot list and grids -----------------------------------------------------


def shot_stems() -> list[str]:
    """`<shot>_<split>` stems of every cached shot, in shot order."""
    stems = [p.stem for p in TS_CACHE.glob("*.arrow")]
    stems.sort(key=lambda s: int(s.rsplit("_", 1)[0]))
    return stems


def split_of(stem: str) -> str:
    return stem.rsplit("_", 1)[1]


def frame_grid(n_samples: int = N_SAMPLES):
    """Frame times (ms), the nearest sample per frame, and fs in kHz."""
    from scipy import signal

    fs_khz = n_samples / TIME_RANGE_MS
    win = signal.get_window("hann", N_FFT)
    sft = signal.ShortTimeFFT(win, hop=HOP, fs=fs_khz)
    t_ms = sft.t(n_samples)
    idx = np.clip(np.rint(t_ms * fs_khz).astype(np.int64), 0, n_samples - 1)
    return t_ms, idx, fs_khz


def read_labels(arrow: Path) -> np.ndarray:
    from pyarrow import feather

    table = feather.read_table(
        arrow, columns=[f"label_{i}" for i in range(N_LABELS)], memory_map=True
    )
    return np.stack(
        [table.column(f"label_{i}").to_numpy() > 0 for i in range(N_LABELS)], axis=0
    )


def read_signals(arrow: Path) -> np.ndarray:
    from pyarrow import feather

    table = feather.read_table(arrow, columns=list(CHANNELS), memory_map=True)
    return np.stack(
        [table.column(c).to_numpy().astype(np.float64) for c in CHANNELS], axis=0
    )


def spectrogram_tokeye(signals: np.ndarray):
    """`tokeye.transforms.compute_stft` per channel -> raw, standardised, mean, std.

    One channel at a time, so no cross-spectrum is ever formed; standardised
    exactly the way `tokeye.inference.model_infer` does it, per array.
    """
    from tokeye.transforms import compute_stft

    raw = np.stack(
        [
            compute_stft(signals[ch : ch + 1], n_fft=N_FFT, hop=HOP).astype(np.float32)
            for ch in range(signals.shape[0])
        ],
        axis=0,
    )
    mean = raw.mean(axis=(1, 2))
    std = raw.std(axis=(1, 2))
    norm = (raw - mean[:, None, None]) / (std[:, None, None] + 1e-6)
    return raw, norm.astype(np.float32), mean.astype(np.float64), std.astype(np.float64)


# --- inference ---------------------------------------------------------------


def load_model(device: str):
    from tokeye import hub

    return hub.load_model(CHECKPOINT, device)


def mask_probs(model, norm: np.ndarray, device: str, batch: int) -> np.ndarray:
    """sigmoid of both output channels -> (n_ch, 2, bins, frames) uint8.

    The channels are batched into one forward pass when they fit; task 6's run
    was host-I/O bound with the GPU at 20 %, so this is the cheap part of the
    fix. On OOM the batch is halved, permanently, and the fallback is printed.
    """
    import torch

    n_ch = norm.shape[0]
    out = np.empty((n_ch, 2, norm.shape[1], norm.shape[2]), dtype=np.uint8)
    start = 0
    while start < n_ch:
        stop = min(start + batch, n_ch)
        chunk = torch.from_numpy(np.ascontiguousarray(norm[start:stop]))
        chunk = chunk.unsqueeze(1).float().to(device)
        try:
            with torch.no_grad():
                prob = torch.sigmoid(model(chunk)[0][:, :2])
        except torch.OutOfMemoryError:
            del chunk
            torch.cuda.empty_cache()
            if batch == 1:
                raise
            batch = max(batch // 2, 1)
            print(f"OOM: falling back to a channel batch of {batch}", flush=True)
            continue
        prob = (prob.clamp(0.0, 1.0) * 255.0).floor().to(torch.uint8).cpu().numpy()
        out[start:stop] = prob
        start = stop
    return out


# --- stage 1: probabilities and cleaned masks --------------------------------


def window_centroids(
    raw: np.ndarray, mask: np.ndarray, freqs: np.ndarray, ann: np.ndarray
) -> list[float]:
    """Centroid kHz of each annotated AE window, from the un-notched clean mask.

    Weighted by LINEAR power, `power_weights(raw) = expm1(raw)**2`, the same
    weighting the published `freq_khz` uses.
    """
    band = slice(BAND_LO_BIN, BAND_HI_BIN)
    weight = power_weights(raw[:, band, :]) * mask[:, band, :]
    num = (weight * freqs[band][None, :, None]).sum(axis=(0, 1))
    den = weight.sum(axis=(0, 1))
    out = []
    for lo, hi in contiguous_runs(ann):
        total = float(den[lo:hi].sum())
        if total > 0:
            out.append(float(num[lo:hi].sum() / total))
    return out


def protected_bins(centroids: list[float], freqs: np.ndarray) -> np.ndarray:
    """Bins within +-3 of any annotated window's centroid: never notch these."""
    keep = np.zeros(N_BINS, dtype=bool)
    for khz in centroids:
        centre = int(np.argmin(np.abs(freqs - khz)))
        lo = max(centre - PROTECT_HALF_WIDTH, 0)
        keep[lo : centre + PROTECT_HALF_WIDTH + 1] = True
    return keep


def run_probs(args) -> None:
    import torch

    mask_dir = Path(args.root) / "masks"
    data_dir = Path(args.root) / "dataset"
    mask_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    stems = select_stems(args)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} shots={len(stems)} batch={args.batch}", flush=True)
    model = load_model(device)
    t_ms, sample_idx, fs_khz = frame_grid()
    freqs = bin_freqs_khz(fs_khz)

    started = time.time()
    for n, stem in enumerate(stems, start=1):
        clean_path = mask_dir / f"{stem}_clean.npz"
        prob_path = mask_dir / f"{stem}_probs.npz"
        spec_path = data_dir / f"{stem}.npz"
        if (
            clean_path.exists()
            and prob_path.exists()
            and spec_path.exists()
            and not args.overwrite
        ):
            continue
        arrow = TS_CACHE / f"{stem}.arrow"
        frame_labels = read_labels(arrow)[:, sample_idx]
        ann = frame_labels[list(AE_LABELS)].any(axis=0)
        lfm = frame_labels[0]

        raw, norm, mean, std = spectrogram_tokeye(read_signals(arrow))
        if raw.shape != (len(CHANNELS), N_BINS, N_FRAMES):
            raise ValueError(f"{stem}: unexpected spectrogram shape {raw.shape}")
        prob = mask_probs(model, norm, device, args.batch)

        coherent = prob[:, 0] >= PROB_CODE
        mode = clean_mask(prob[:, 0], prob[:, 1], threshold=PROB_CODE)
        clean = persist_open(mode, PERSIST_FRAMES)
        centroids = window_centroids(raw, clean, freqs, ann)

        np.savez_compressed(prob_path, prob=prob, ann=ann)
        np.savez_compressed(
            clean_path,
            mask_clean=np.packbits(clean, axis=-1),
            mask_raw=np.packbits(coherent, axis=-1),
            active_frac_clean=bin_active_fraction(clean).astype(np.float32),
            active_frac_raw=bin_active_fraction(coherent).astype(np.float32),
            n_pixels_raw=coherent.sum(axis=(1, 2)).astype(np.int64),
            n_pixels_no_transient=mode.sum(axis=(1, 2)).astype(np.int64),
            n_pixels_clean=clean.sum(axis=(1, 2)).astype(np.int64),
            protected=protected_bins(centroids, freqs),
            window_centroids_khz=np.asarray(centroids, dtype=np.float32),
            ann=ann,
            lfm=lfm,
            frame_labels=frame_labels,
            t_ms=t_ms.astype(np.float32),
            freqs=freqs.astype(np.float32),
            spec_mean=mean,
            spec_std=std,
        )
        # The spectrogram half of the per-shot dataset file; the label half is
        # written by the dataset stage, once the two thresholds are fixed.
        save_npz(
            spec_path,
            spec=norm[:, BAND_LO_BIN:BAND_HI_BIN, :].astype(np.float16),
            annotated=ann.astype(np.uint8),
            split=np.asarray(split_of(stem)),
            spec_mean=mean,
            spec_std=std,
            freq_khz_bins=freqs[BAND_LO_BIN:BAND_HI_BIN].astype(np.float32),
        )
        rate = (time.time() - started) / n
        print(
            f"[{n}/{len(stems)}] {stem} ann={ann.mean():.3f} "
            f"clean/raw pixels={clean.sum() / max(coherent.sum(), 1):.3f} "
            f"{rate:.1f}s/shot",
            flush=True,
        )
    if device == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


def save_npz(path: Path, **arrays) -> None:
    """Uncompressed npz written atomically (float16 spectrograms barely compress)."""
    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp, path)


def select_stems(args) -> list[str]:
    stems = shot_stems()
    if args.shots:
        wanted = [s.strip() for s in args.shots.split(",") if s.strip()]
        missing = [s for s in wanted if s not in stems]
        if missing:
            raise SystemExit(f"unknown shots: {missing}")
        return wanted
    if args.limit:
        return stems[: args.limit]
    return stems


# --- stage 2: thresholds, review and labels ----------------------------------


def load_clean(mask_dir: Path, stem: str) -> dict:
    with np.load(mask_dir / f"{stem}_clean.npz") as z:
        return {k: z[k] for k in z.files}


def notch_sweep(shots: dict) -> dict:
    """Per-threshold notch counts and the protected-bin violations."""
    rows = []
    for thr in NOTCH_THRESHOLDS:
        per_channel = []
        violations = []
        for stem, d in shots.items():
            notched = notch_bins(d["active_frac_clean"], thr)
            per_channel.append(notched.sum(axis=1))
            hit = notched & d["protected"][None, :]
            if hit.any():
                violations.append(
                    {
                        "shot": stem,
                        "channels": [
                            CHANNELS[c] for c in np.flatnonzero(hit.any(axis=1))
                        ],
                        "bins": [int(b) for b in np.flatnonzero(hit.any(axis=0))],
                        "khz": [
                            round(float(d["freqs"][b]), 3)
                            for b in np.flatnonzero(hit.any(axis=0))
                        ],
                    }
                )
        counts = np.asarray(per_channel, dtype=float)  # (n_shots, 4)
        union = np.asarray(
            [
                int(notch_bins(d["active_frac_clean"], thr).any(axis=0).sum())
                for d in shots.values()
            ],
            dtype=float,
        )
        removed = {}
        for stem, d in shots.items():
            notched = notch_bins(d["active_frac_clean"], thr)
            entry = {
                CHANNELS[ci]: {
                    "bins": [int(b) for b in np.flatnonzero(notched[ci])],
                    "khz": [
                        round(float(d["freqs"][b]), 3)
                        for b in np.flatnonzero(notched[ci])
                    ],
                }
                for ci in range(len(CHANNELS))
                if notched[ci].any()
            }
            if entry:
                removed[stem] = entry
        rows.append(
            {
                "threshold": thr,
                "removed_bins_per_shot": removed,
                "notched_bins_per_channel_mean": float(counts.mean()),
                "notched_bins_per_channel_max": float(counts.max()),
                "notched_bins_union_mean": float(union.mean()),
                "notched_bins_union_max": float(union.max()),
                "shots_with_any_notch": int((union > 0).sum()),
                "shots_with_protected_bin_removed": len(violations),
                "protected_violations": violations,
                "passes": len(violations) <= PROTECT_MAX_SHOTS,
            }
        )
    passing = [r["threshold"] for r in rows if r["passes"]]
    return {
        "rule": (
            "sweep {0.5, 0.6, 0.7, 0.8, 0.9} and take the SMALLEST threshold at "
            "which no bin inside an annotated AE window's centroid +-"
            f"{PROTECT_HALF_WIDTH} bins is removed on more than {PROTECT_MAX_SHOTS} "
            "shots (a smaller threshold notches more aggressively)"
        ),
        "protect_half_width_bins": PROTECT_HALF_WIDTH,
        "protect_max_shots": PROTECT_MAX_SHOTS,
        "table": rows,
        "passing": passing,
        "chosen": min(passing) if passing else max(NOTCH_THRESHOLDS),
        "chosen_by": (
            "smallest passing threshold"
            if passing
            else "no threshold passed; fell back to the least aggressive"
        ),
    }


def build_review(shots: dict, threshold: float) -> dict:
    """Per-shot notch record; a bin below 190 kHz makes that channel reviewable."""
    per_shot = []
    for stem, d in shots.items():
        notched = notch_bins(d["active_frac_clean"], threshold)
        entry = {"shot": stem, "split": split_of(stem), "channels": {}}
        review = False
        for ci, name in enumerate(CHANNELS):
            bins = np.flatnonzero(notched[ci])
            if not bins.size:
                continue
            khz = [round(float(d["freqs"][b]), 3) for b in bins]
            low = [f for f in khz if f < REVIEW_MAX_KHZ]
            entry["channels"][name] = {
                "bins": [int(b) for b in bins],
                "khz": khz,
                "occupancy": [
                    round(float(d["active_frac_clean"][ci, b]), 4) for b in bins
                ],
                "in_band": [bool(BAND_LO_BIN <= b < BAND_HI_BIN) for b in bins],
                "needs_review": bool(low),
                "notch_applied": not low,
                "reason": (
                    f"removes {len(low)} bin(s) below {REVIEW_MAX_KHZ:g} kHz "
                    f"({min(low):.1f}-{max(low):.1f} kHz), inside the AE centroid "
                    "range; the notch is not applied to this channel until a human "
                    "looks"
                )
                if low
                else "",
            }
            review = review or bool(low)
        entry["needs_review"] = review
        if entry["channels"]:
            per_shot.append(entry)
    flagged = [e["shot"] for e in per_shot if e["needs_review"]]
    return {
        "notch_threshold": threshold,
        "rule": (
            "a (channel, bin) lit in more than the threshold fraction of the "
            "7820 frames of the cleaned mask is receiver pickup and its row is "
            "zeroed before the activity decision and before the centroid"
        ),
        "review_rule": (
            f"any shot whose notch removes a bin below {REVIEW_MAX_KHZ:g} kHz is "
            "flagged needs_review and that channel's notch is NOT applied to the "
            "label; the other channels' notches still are"
        ),
        "shots_needing_review": flagged,
        "n_shots_with_any_notch": len(per_shot),
        "shots": per_shot,
    }


def effective_notch(d: dict, review: dict, stem: str, threshold: float) -> np.ndarray:
    """Notched bins actually applied to the label, review exclusions removed."""
    notched = notch_bins(d["active_frac_clean"], threshold)
    entry = next((e for e in review["shots"] if e["shot"] == stem), None)
    if entry is not None:
        for ci, name in enumerate(CHANNELS):
            record = entry["channels"].get(name)
            if record is not None and not record["notch_applied"]:
                notched[ci] = False
    return notched


def occupancy_of(mask_dir: Path, stem: str, d: dict, notched: np.ndarray, key: str):
    """Band occupancy per frame, averaged over the 4 channels."""
    with np.load(mask_dir / f"{stem}_clean.npz") as z:
        packed = z[key]
    mask = np.unpackbits(packed, axis=-1, count=N_FRAMES).astype(bool)
    return band_occupancy(apply_notch(mask, notched)).mean(axis=0), mask


def threshold_table(occ: np.ndarray, ann: np.ndarray) -> list[dict]:
    rows = []
    for thr in OCC_THRESHOLDS:
        pred = occ >= thr
        rows.append(
            {
                "threshold": thr,
                "pooled_recall": float(pred[ann].mean()),
                "precision": float(ann[pred].mean()) if pred.any() else float("nan"),
                "predicted_positive_frac": float(pred.mean()),
                "recall_ok": bool(pred[ann].mean() >= MIN_POOLED_RECALL),
                "positive_frac_ok": bool(pred.mean() <= MAX_POSITIVE_FRAC),
            }
        )
    return rows


def choose_occupancy(rows: list[dict]) -> dict:
    both = [r for r in rows if r["recall_ok"] and r["positive_frac_ok"]]
    if both:
        chosen = min(both, key=lambda r: r["threshold"])
        how = (
            "lowest threshold with predicted-positive fraction <= "
            f"{MAX_POSITIVE_FRAC} and pooled recall >= {MIN_POOLED_RECALL}"
        )
        satisfied = True
    else:
        recall_ok = [r for r in rows if r["recall_ok"]]
        satisfied = False
        if recall_ok:
            chosen = min(recall_ok, key=lambda r: r["predicted_positive_frac"])
            how = (
                "NO threshold satisfied both halves of the rule; fell back to the "
                f"one with recall >= {MIN_POOLED_RECALL} and the smallest positive "
                "fraction, as the rule's own escape clause says"
            )
        else:
            # Outside the rule's escape clause: nothing to fall back on but the
            # highest-recall threshold, and the caller has to say so out loud.
            chosen = min(rows, key=lambda r: r["threshold"])
            how = (
                f"NO threshold even reaches pooled recall {MIN_POOLED_RECALL}; fell "
                "back to the lowest threshold (highest recall). The cleaning rule "
                "needs revisiting before this label is trusted"
            )
    return {
        "rule": (
            f"choose the LOWEST threshold at which the predicted-positive frame "
            f"fraction is <= {MAX_POSITIVE_FRAC} while pooled recall stays >= "
            f"{MIN_POOLED_RECALL}; if none satisfies both, the one with recall >= "
            f"{MIN_POOLED_RECALL} and the smallest positive fraction"
        ),
        "table": rows,
        "chosen": chosen["threshold"],
        "chosen_by": how,
        "rule_satisfied": satisfied,
        "at_chosen": chosen,
    }


def agreement(active: np.ndarray, annotated: np.ndarray) -> dict:
    both = int((active & annotated).sum())
    return {
        "n_frames": int(active.size),
        "active_frac": float(active.mean()),
        "annotated_frac": float(annotated.mean()),
        "recall_of_annotated": float(active[annotated].mean()),
        "precision_against_annotated": float(annotated[active].mean())
        if active.any()
        else float("nan"),
        "jaccard": float(both / max(int((active | annotated).sum()), 1)),
        "frame_agreement": float((active == annotated).mean()),
        "active_and_annotated_frac": float(both / active.size),
        "active_not_annotated_frac": float((active & ~annotated).mean()),
        "annotated_not_active_frac": float((~active & annotated).mean()),
    }


def split_balance(members, stems, active_all, ann_all):
    """`agreement` over one split, or None when the split is empty (smoke runs)."""
    if not members:
        return None
    return agreement(
        np.concatenate([active_all[stems.index(s)] for s in members]),
        np.concatenate([ann_all[s] for s in members]),
    )


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001 - provenance must never fail a run
        return f"unavailable: {exc}"


def notch_figure(shots: dict, review: dict, threshold: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stems = list(shots)
    freqs = next(iter(shots.values()))["freqs"]
    grid = np.zeros((len(stems), N_BINS), dtype=np.float32)
    for row, (stem, d) in enumerate(shots.items()):
        grid[row] = notch_bins(d["active_frac_clean"], threshold).sum(axis=0)
    # only the shots that notch anything get a row: the rest would be 126 blank
    # lines with unreadable tick labels.
    rows = [i for i in range(len(stems)) if grid[i].any()]
    grid = grid[rows] if rows else grid[:1]
    names = [stems[i] for i in rows] if rows else ["(no shot notches anything)"]
    used = np.flatnonzero(grid.any(axis=0))
    lo = max(int(used.min()) - 8, 0) if used.size else BAND_LO_BIN
    hi = min(int(used.max()) + 9, N_BINS) if used.size else BAND_HI_BIN

    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    img = ax.imshow(
        grid[:, lo:hi],
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0,
        vmax=4,
        extent=(float(freqs[lo]), float(freqs[hi - 1]), -0.5, len(names) - 0.5),
    )
    fig.colorbar(img, ax=ax, label="CO2 channels notching this bin", shrink=0.6)
    ax.axvline(REVIEW_MAX_KHZ, color="#00d0ff", lw=1.2, ls="--")
    ax.text(
        REVIEW_MAX_KHZ,
        len(names) - 0.5,
        f"{REVIEW_MAX_KHZ:g} kHz ",
        color="#00d0ff",
        fontsize=8,
        va="top",
        ha="right",
    )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6)
    flagged = set(review["shots_needing_review"])
    for label in ax.get_yticklabels():
        if label.get_text() in flagged:
            label.set_color("#0a8f00")
            label.set_fontweight("bold")
    ax.set_xlabel("frequency [kHz]")
    ax.set_ylabel("shot")
    ax.set_title(
        f"notched bins at threshold {threshold:g} (cleaned mask, per CO2 channel)\n"
        f"{review['n_shots_with_any_notch']} of {len(stems)} shots notch anything; "
        f"{len(flagged)} in bold green remove a bin below {REVIEW_MAX_KHZ:g} kHz and "
        "have that channel's notch withheld",
        fontsize=9,
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_dataset(args) -> None:
    root = Path(args.root)
    mask_dir = root / "masks"
    data_dir = root / "dataset"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = [s for s in select_stems(args) if (mask_dir / f"{s}_clean.npz").exists()]
    print(f"dataset over {len(stems)} shots", flush=True)
    shots = {s: load_clean(mask_dir, s) for s in stems}

    sweep = notch_sweep(shots)
    notch_thr = sweep["chosen"]
    review = build_review(shots, notch_thr)
    print(f"notch threshold {notch_thr:g}; review shots {review['shots_needing_review']}")

    notch_figure(shots, review, notch_thr, out_dir / "notch_review.png")
    (out_dir / "notch_review.json").write_text(json.dumps(review, indent=2))

    occ_clean, occ_raw, ann_all = {}, {}, {}
    for stem, d in shots.items():
        notched = effective_notch(d, review, stem, notch_thr)
        occ_clean[stem], _ = occupancy_of(mask_dir, stem, d, notched, "mask_clean")
        occ_raw[stem], _ = occupancy_of(mask_dir, stem, d, notched, "mask_raw")
        ann_all[stem] = d["ann"].astype(bool)

    occ = np.concatenate([occ_clean[s] for s in stems])
    ann = np.concatenate([ann_all[s] for s in stems])
    study = choose_occupancy(threshold_table(occ, ann))
    occ_thr = study["chosen"]
    study["raw_mask_table_for_comparison"] = threshold_table(
        np.concatenate([occ_raw[s] for s in stems]), ann
    )
    print(f"occupancy threshold {occ_thr:g} ({study['chosen_by']})", flush=True)
    if not study["rule_satisfied"]:
        print("=" * 72)
        print("WARNING: the occupancy rule was NOT satisfied - " + study["chosen_by"])
        print("=" * 72, flush=True)

    # --- per-shot labels -----------------------------------------------------
    band = slice(BAND_LO_BIN, BAND_HI_BIN)
    per_shot, active_all = [], []
    for stem, d in shots.items():
        notched = effective_notch(d, review, stem, notch_thr)
        with np.load(mask_dir / f"{stem}_clean.npz") as z:
            mask = np.unpackbits(z["mask_clean"], axis=-1, count=N_FRAMES).astype(bool)
        mask = apply_notch(mask, notched)
        active = (occ_clean[stem] >= occ_thr).astype(np.uint8)
        with np.load(data_dir / f"{stem}.npz") as z:
            spec = z["spec"]
            annotated = z["annotated"]
            mean, std = z["spec_mean"], z["spec_std"]
            bins_khz = z["freq_khz_bins"]
        # raw = spec * std + mean recovers the un-standardised transform values
        # (standardisation is per channel and affine); power_weights then
        # inverts the log1p and squares, so the centroid weights by linear
        # power, not by the log values.
        raw = spec.astype(np.float64) * std[:, None, None] + mean[:, None, None]
        freq = centroid_khz(
            power_weights(raw), mask[:, band, :], bins_khz
        ).astype(np.float32)
        freq[active == 0] = np.nan
        save_npz(
            data_dir / f"{stem}.npz",
            spec=spec,
            active=active,
            annotated=annotated,
            freq_khz=freq,
            notched_bins=notched,
            notch_excluded=np.asarray(
                [
                    bool(
                        notch_bins(d["active_frac_clean"], notch_thr)[ci].any()
                        and not notched[ci].any()
                    )
                    for ci in range(len(CHANNELS))
                ]
            ),
            split=np.asarray(split_of(stem)),
            spec_mean=mean,
            spec_std=std,
            freq_khz_bins=bins_khz,
            occupancy=occ_clean[stem].astype(np.float32),
        )
        active_all.append(active.astype(bool))
        good = np.isfinite(freq)
        per_shot.append(
            {
                "shot": stem,
                "split": split_of(stem),
                "active_frac": float(active.mean()),
                "annotated_frac": float(annotated.mean()),
                "recall_of_annotated": float(
                    active.astype(bool)[annotated.astype(bool)].mean()
                )
                if annotated.any()
                else float("nan"),
                "median_freq_khz": float(np.median(freq[good])) if good.any() else None,
                "notched_bins_union": int(notched.any(axis=0).sum()),
                "needs_review": stem in review["shots_needing_review"],
                "pixels_raw": [int(v) for v in d["n_pixels_raw"]],
                "pixels_no_transient": [int(v) for v in d["n_pixels_no_transient"]],
                "pixels_clean": [int(v) for v in d["n_pixels_clean"]],
            }
        )

    active = np.concatenate(active_all)
    splits = {sp: [s for s in stems if split_of(s) == sp] for sp in ("train", "valid")}
    cleaning = {
        stem: {
            "pixels_coherent": int(np.sum(d["n_pixels_raw"])),
            "pixels_after_transient_veto": int(np.sum(d["n_pixels_no_transient"])),
            "pixels_after_persistence": int(np.sum(d["n_pixels_clean"])),
            "kept_frac_after_transient_veto": float(
                np.sum(d["n_pixels_no_transient"]) / max(np.sum(d["n_pixels_raw"]), 1)
            ),
            "kept_frac_after_cleaning": float(
                np.sum(d["n_pixels_clean"]) / max(np.sum(d["n_pixels_raw"]), 1)
            ),
        }
        for stem, d in shots.items()
    }
    total_raw = sum(v["pixels_coherent"] for v in cleaning.values())
    total_clean = sum(v["pixels_after_persistence"] for v in cleaning.values())

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "git_sha": git_sha(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "environment": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "pythonpath": os.environ.get("PYTHONPATH", ""),
        },
        "inputs": {
            "checkpoint": str(CHECKPOINT),
            "timeseries": str(TS_CACHE),
            "dataset_dir": str(data_dir),
            "masks_dir": str(mask_dir),
        },
        "construction": {
            "transform": TRANSFORM,
            "transform_detail": (
                "tokeye.transforms.compute_stft per channel (log1p(|STFT|) then a "
                "1/99 percentile clip), Hann 1024 / hop 128, DC bin dropped, "
                "standardised per array as tokeye.inference.model_infer does"
            ),
            "model": "big_tf_unet (tokeye), channel 0 coherent, channel 1 transient",
            "prob_threshold": PROB_THRESHOLD,
            "prob_uint8_code": PROB_CODE,
            "persist_frames": PERSIST_FRAMES,
            "persist_ms": PERSIST_FRAMES * TIME_RANGE_MS / N_FRAMES,
            "band_bins": [BAND_LO_BIN, BAND_HI_BIN - 1],
            "band_khz": [80.56648681640625, 250.00025],
            "n_frames": N_FRAMES,
            "frame_ms": TIME_RANGE_MS / N_FRAMES,
            "freq_khz_weighting": (
                "intensity-weighted centroid over masked band pixels, weighted by "
                "LINEAR power: power_weights(x) = expm1(x)**2 applied to the "
                "tokeye transform's clip(log1p(|STFT|), p1, p99) values, so the "
                "weight is |STFT|**2 after the percentile clip. Weighting by the "
                "stored log values is effectively unweighted (they span 46-63, "
                "CV 0.026) and is what task 6 did"
            ),
            "label": (
                "active[f] = band occupancy of the cleaned, notched coherent mask "
                ">= the chosen occupancy threshold; annotated[f] = label_1|2|3|4 "
                "(LFM excluded) resampled to the frame grid"
            ),
        },
        "notch": sweep,
        "notch_review": {
            "threshold": notch_thr,
            "shots_needing_review": review["shots_needing_review"],
            "n_shots_with_any_notch": review["n_shots_with_any_notch"],
            "file": "notch_review.json",
        },
        "occupancy_study": study,
        "counts": {
            "n_shots": len(stems),
            "n_train": len(splits["train"]),
            "n_valid": len(splits["valid"]),
            "n_frames": int(active.size),
            "spec_shape": [len(CHANNELS), BAND_HI_BIN - BAND_LO_BIN, N_FRAMES],
        },
        "class_balance": {
            "pooled": agreement(active, ann),
            "train": split_balance(splits["train"], stems, active_all, ann_all),
            "valid": split_balance(splits["valid"], stems, active_all, ann_all),
        },
        "cleaning_effect": {
            "note": (
                "active mask pixels over all 4 channels x 512 bins x 7820 frames, "
                "before the notch"
            ),
            "pooled_kept_frac_after_cleaning": float(
                total_clean / max(total_raw, 1)
            ),
            "reported_shots": {
                stem: cleaning[stem]
                for stem in ("176042_train", "175978_train")
                if stem in cleaning
            },
            "per_shot": cleaning,
        },
        "per_shot": per_shot,
        "jobstats": None,
    }
    out_json = out_dir / "dataset.json"
    out_json.write_text(json.dumps(payload, indent=2, default=float))
    print(json.dumps(payload["class_balance"]["pooled"], indent=2))
    print(f"wrote {out_json}")


def run_jobstats(args) -> None:
    path = Path(args.out_dir) / "dataset.json"
    payload = json.loads(path.read_text())
    payload["jobstats"] = Path(args.jobstats_file).read_text()
    if args.job_id:
        payload["slurm_job_id"] = args.job_id
    path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"merged jobstats into {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage", choices=("probs", "dataset", "all", "jobstats"), default="all"
    )
    p.add_argument("--root", default="/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/ae")
    p.add_argument(
        "--out-dir",
        default="/scratch/gpfs/nc1514/FusionAIHub/outputs/labelmaker/ae/dataset",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--batch", type=int, default=4, help="CO2 channels per forward pass")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shots", default="", help="comma-separated stems, e.g. 176042_train")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--jobstats-file", default="")
    p.add_argument("--job-id", default="")
    args = p.parse_args()
    if args.stage in ("probs", "all"):
        run_probs(args)
    if args.stage in ("dataset", "all"):
        run_dataset(args)
    if args.stage == "jobstats":
        run_jobstats(args)


if __name__ == "__main__":
    main()
