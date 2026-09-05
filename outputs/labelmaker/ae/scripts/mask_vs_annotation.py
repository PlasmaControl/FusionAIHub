"""Measure tokeye's ``big_tf_unet`` coherent mask against the 180 hand-annotated AE shots.

Phase 3 task 6 / spec section 5.8 item 1: decide, with numbers, whether the
coherent-mode mask can carry the AE label, and which spectrogram transform to use
(spec section 7 question 3).

Two transforms are compared, both fed to the same released checkpoint
(``tokeye/model/big_tf_unet_251210.pt``, registry name ``big_tf_unet``):

``aemodes``
    the regenerated spectrogram tif on disk
    (``aemodes/data/.cache/step_0a/spectrograms/<shot>_<mode>.tif``,
    ``log1p(|STFT|**2)``, Hann 1024 / hop 128, DC bin dropped), standardised with
    the global ``step_0a/stats.json`` mean 54.637 / std 2.732.

``tokeye``
    ``tokeye.transforms.compute_stft`` run on the cached arrow time series, one
    channel at a time (``log1p(|STFT|)`` then a 1/99 percentile clip), standardised
    the way ``tokeye.inference.model_infer`` does it: per array,
    ``(x - x.mean()) / (x.std() + 1e-6)``.

``tokeye.inference.model_infer`` does not window the column axis - it feeds the
whole spectrogram in one forward pass - so the default here is the full 7820-column
width. ``--width`` chunks the column axis (with overlap, centre-cropped on
reassembly) if a device cannot hold the full width.

Stage ``masks`` writes one small npz of per-frame series plus one npz of packed
mask bits per (shot, transform) under ``<out-root>/masks``. Stage ``metrics``
reads those back, writes ``mask_vs_annotation.json`` and the figures. Neither
stage writes anything into the aemodes or tokeye trees.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- constants fixed by the spec / brief -------------------------------------

TS_CACHE = Path("/scratch/gpfs/nc1514/aemodes/data/.cache/ae_timeseries")
SPEC_CACHE = Path("/scratch/gpfs/nc1514/aemodes/data/.cache/step_0a/spectrograms")
STATS_JSON = Path("/scratch/gpfs/nc1514/aemodes/data/.cache/step_0a/stats.json")
CHECKPOINT = Path("/scratch/gpfs/nc1514/tokeye/model/big_tf_unet_251210.pt")

CHANNELS = ("r0", "v1", "v2", "v3")
N_LABELS = 5
LABEL_NAMES = ("lfm", "bae", "eae", "rsae", "tae")
AE_LABELS = (1, 2, 3, 4)  # label_0 (LFM) is excluded from the AE annotation

N_FFT = 1024
HOP = 128
TIME_RANGE_MS = 2000.0
N_BINS = 512  # after the DC bin is dropped
N_FRAMES = 7820

BAND_LO_BIN = 164  # 80 kHz (exactly 80.566 kHz on this grid)
BAND_HI_BIN = 512  # exclusive; bin 511 = 250.000 kHz
PROB_THRESHOLD = 0.2  # step_1_make_semantic.py's coherent-mask threshold
NOTCH_ACTIVE_FRAC = 0.8  # a bin lit in > 80 % of frames is an instrument line
OCC_THRESHOLDS = (0.005, 0.01, 0.02, 0.05)
REPORT_OCC = 0.01  # the threshold the unannotated-fraction split is reported at
TRANSFORMS = ("aemodes", "tokeye")

# --- small helpers -----------------------------------------------------------


def shot_files() -> list[tuple[str, Path, Path]]:
    """(stem, arrow path, tif path) for every cached shot, sorted by shot number."""
    out = []
    for arrow in sorted(TS_CACHE.glob("*.arrow")):
        stem = arrow.stem
        tif = SPEC_CACHE / f"{stem}.tif"
        out.append((stem, arrow, tif))
    out.sort(key=lambda row: int(row[0].rsplit("_", 1)[0]))
    return out


def frame_grid(n_samples: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Frame times (ms), nearest sample index per frame, and fs in kHz."""
    from scipy import signal

    fs_khz = n_samples / TIME_RANGE_MS
    win = signal.get_window("hann", N_FFT)
    sft = signal.ShortTimeFFT(win, hop=HOP, fs=fs_khz)
    t_ms = sft.t(n_samples)
    idx = np.clip(np.rint(t_ms * fs_khz).astype(np.int64), 0, n_samples - 1)
    return t_ms, idx, fs_khz


def bin_freqs_khz(fs_khz: float) -> np.ndarray:
    """Centre frequency of each of the 512 bins after the DC bin is dropped."""
    return (np.arange(N_BINS, dtype=np.float64) + 1.0) * fs_khz / N_FFT


def read_labels(arrow: Path) -> np.ndarray:
    """(5, n_samples) bool label array from the arrow cache."""
    import pyarrow.feather as feather

    table = feather.read_table(
        arrow, columns=[f"label_{i}" for i in range(N_LABELS)], memory_map=True
    )
    return np.stack(
        [table.column(f"label_{i}").to_numpy() > 0 for i in range(N_LABELS)], axis=0
    )


def read_signals(arrow: Path) -> np.ndarray:
    """(4, n_samples) float64 CO2 channels from the arrow cache."""
    import pyarrow.feather as feather

    table = feather.read_table(arrow, columns=list(CHANNELS), memory_map=True)
    return np.stack(
        [table.column(c).to_numpy().astype(np.float64) for c in CHANNELS], axis=0
    )


# --- spectrograms ------------------------------------------------------------


def spectrogram_aemodes(tif: Path) -> tuple[np.ndarray, np.ndarray]:
    """Stored regenerated tif -> (raw (4, 512, T), standardised (4, 512, T))."""
    import tifffile

    stats = json.loads(STATS_JSON.read_text())
    raw = tifffile.imread(tif).astype(np.float32)
    std = ((raw - np.float32(stats["mean"])) / np.float32(stats["std"])).astype(np.float32)
    return raw, std


def spectrogram_tokeye(signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """tokeye ``compute_stft`` per channel -> (raw, standardised) (4, 512, T)."""
    from tokeye.transforms import compute_stft

    raw_channels = []
    for ch in range(signals.shape[0]):
        sxx = compute_stft(signals[ch : ch + 1], n_fft=N_FFT, hop=HOP)
        raw_channels.append(sxx.astype(np.float32))
    raw = np.stack(raw_channels, axis=0)
    std = np.empty_like(raw)
    for ch in range(raw.shape[0]):
        # exactly tokeye.inference.model_infer's standardisation, per channel
        std[ch] = (raw[ch] - raw[ch].mean()) / (raw[ch].std() + 1e-6)
    return raw, std


# --- inference ---------------------------------------------------------------


def load_model(device: str):
    from tokeye import hub

    return hub.load_model(CHECKPOINT, device)


_OOM_FALLBACK_WIDTH = 2048
_forced_width: list[int] = []


def coherent_prob(model, std_spec: np.ndarray, device: str, width: int) -> np.ndarray:
    """sigmoid(model(x))[coherent channel] for one (512, T) standardised spectrogram.

    Falls back to chunked columns once, permanently, if the device cannot hold the
    full width; the fallback is reported in the run log so it is never silent.
    """
    import torch

    if _forced_width:
        width = _forced_width[0]
    try:
        return _coherent_prob(model, std_spec, device, width)
    except torch.OutOfMemoryError:
        if width > 0:
            raise
        torch.cuda.empty_cache()
        _forced_width.append(_OOM_FALLBACK_WIDTH)
        print(
            f"OOM at full width; falling back to column windows of {_OOM_FALLBACK_WIDTH}",
            flush=True,
        )
        return _coherent_prob(model, std_spec, device, _OOM_FALLBACK_WIDTH)


def _coherent_prob(model, std_spec: np.ndarray, device: str, width: int) -> np.ndarray:
    import torch

    height, total = std_spec.shape
    out = np.empty((height, total), dtype=np.float32)
    if width <= 0 or width >= total:
        spans = [(0, total, 0, total)]
    else:
        # overlap by width // 4 on each side and keep only the centre of each chunk
        pad = max(width // 4, 32)
        spans = []
        start = 0
        while start < total:
            stop = min(start + width, total)
            lo = max(start - pad, 0)
            hi = min(stop + pad, total)
            spans.append((lo, hi, start, stop))
            start = stop
    with torch.no_grad():
        for lo, hi, keep_lo, keep_hi in spans:
            chunk = torch.from_numpy(np.ascontiguousarray(std_spec[:, lo:hi]))
            chunk = chunk.unsqueeze(0).unsqueeze(0).float().to(device)
            logits = model(chunk)[0]  # (1, 2, H, W); channel 0 = coherent
            prob = torch.sigmoid(logits[0, 0]).cpu().numpy()
            out[:, keep_lo:keep_hi] = prob[:, keep_lo - lo : keep_hi - lo]
    return out


# --- per-shot mask products --------------------------------------------------


@dataclass(frozen=True)
class ShotResult:
    occ: np.ndarray  # (T,) band occupancy averaged over channels
    occ_ch: np.ndarray  # (4, T)
    centroid: np.ndarray  # (T,) intensity-weighted centroid frequency, kHz, nan if empty
    notched: np.ndarray  # (4, 512) bool
    active_frac: np.ndarray  # (4, 512) fraction of frames each bin is lit, pre-notch
    packed: np.ndarray  # packbits of the notched mask, (4, 512, T)


def shot_masks(
    model, raw: np.ndarray, std: np.ndarray, freqs: np.ndarray, device: str, width: int
) -> ShotResult:
    n_ch, _, n_frames = std.shape
    occ_ch = np.zeros((n_ch, n_frames), dtype=np.float32)
    notched = np.zeros((n_ch, N_BINS), dtype=bool)
    active_frac = np.zeros((n_ch, N_BINS), dtype=np.float32)
    num = np.zeros(n_frames, dtype=np.float64)
    den = np.zeros(n_frames, dtype=np.float64)
    masks = np.zeros((n_ch, N_BINS, n_frames), dtype=bool)
    band_freqs = freqs[BAND_LO_BIN:BAND_HI_BIN][:, None]

    for ch in range(n_ch):
        prob = coherent_prob(model, std[ch], device, width)
        mask = prob >= PROB_THRESHOLD
        # first-pass notch: a bin lit across more than 80 % of frames is an
        # instrument line, not a mode. Zero the row before anything else uses it.
        frac = mask.mean(axis=1)
        active_frac[ch] = frac
        line = frac > NOTCH_ACTIVE_FRAC
        notched[ch] = line
        mask[line, :] = False
        masks[ch] = mask
        band = mask[BAND_LO_BIN:BAND_HI_BIN]
        occ_ch[ch] = band.mean(axis=0)
        weight = raw[ch, BAND_LO_BIN:BAND_HI_BIN] * band
        num += (weight * band_freqs).sum(axis=0)
        den += weight.sum(axis=0)

    occ = occ_ch.mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        centroid = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return ShotResult(
        occ=occ.astype(np.float32),
        occ_ch=occ_ch,
        centroid=centroid.astype(np.float32),
        notched=notched,
        active_frac=active_frac,
        packed=np.packbits(masks, axis=-1),
    )


def run_masks(args) -> None:
    import torch

    out_root = Path(args.out_root) / "masks"
    out_root.mkdir(parents=True, exist_ok=True)
    shots = shot_files()
    if args.limit:
        shots = shots[: args.limit]
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} shots={len(shots)} transforms={args.transforms} width={args.width}")
    model = load_model(device)
    t_ms, sample_idx, fs_khz = frame_grid(1000001)
    freqs = bin_freqs_khz(fs_khz)

    started = time.time()
    for n, (stem, arrow, tif) in enumerate(shots, start=1):
        labels = read_labels(arrow)
        frame_labels = labels[:, sample_idx]  # (5, T) nearest-sample resample
        ann = frame_labels[list(AE_LABELS)].any(axis=0)
        lfm = frame_labels[0]
        signals = None
        for transform in args.transforms:
            small = out_root / f"{stem}_{transform}.npz"
            packed_path = out_root / f"{stem}_{transform}_mask.npz"
            if small.exists() and packed_path.exists() and not args.overwrite:
                continue
            if transform == "aemodes":
                raw, std = spectrogram_aemodes(tif)
            else:
                if signals is None:
                    signals = read_signals(arrow)
                raw, std = spectrogram_tokeye(signals)
            if raw.shape != (len(CHANNELS), N_BINS, N_FRAMES):
                raise ValueError(f"{stem}/{transform}: unexpected shape {raw.shape}")
            res = shot_masks(model, raw, std, freqs, device, args.width)
            np.savez_compressed(
                small,
                occ=res.occ,
                occ_ch=res.occ_ch,
                centroid=res.centroid,
                notched=res.notched,
                active_frac=res.active_frac,
                ann=ann,
                lfm=lfm,
                frame_labels=frame_labels,
                t_ms=t_ms.astype(np.float32),
                freqs=freqs.astype(np.float32),
            )
            np.savez_compressed(packed_path, packed=res.packed)
        rate = (time.time() - started) / n
        print(f"[{n}/{len(shots)}] {stem} ann={ann.mean():.3f} {rate:.1f}s/shot", flush=True)
    if device == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


# --- metrics -----------------------------------------------------------------


def auroc(scores: np.ndarray, truth: np.ndarray) -> float:
    """Rank-based AUROC with tie handling; nan when a class is absent."""
    pos = int(truth.sum())
    neg = int(truth.size - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(s.size, dtype=np.float64)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    rank_sum = ranks[truth[order]].sum()
    return float((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def roc_curve(scores: np.ndarray, truth: np.ndarray, n_points: int = 512):
    order = np.argsort(-scores, kind="mergesort")
    t = truth[order]
    tps = np.cumsum(t)
    fps = np.cumsum(~t)
    pos = max(int(truth.sum()), 1)
    neg = max(int(truth.size - truth.sum()), 1)
    tpr = np.concatenate([[0.0], tps / pos])
    fpr = np.concatenate([[0.0], fps / neg])
    keep = np.unique(np.linspace(0, tpr.size - 1, n_points).astype(int))
    return fpr[keep], tpr[keep]


def _pct(values: np.ndarray, q: float) -> float:
    v = values[np.isfinite(values)]
    return float(np.percentile(v, q)) if v.size else float("nan")


def load_series(out_root: Path, stem: str, transform: str):
    with np.load(out_root / "masks" / f"{stem}_{transform}.npz") as z:
        return {k: z[k] for k in z.files}


def transform_metrics(out_root: Path, stems: list[str], transform: str) -> dict:
    occ_all, ann_all, lfm_all = [], [], []
    per_shot = []
    for stem in stems:
        d = load_series(out_root, stem, transform)
        occ, ann, lfm = d["occ"], d["ann"], d["lfm"]
        occ_all.append(occ)
        ann_all.append(ann)
        lfm_all.append(lfm)
        centroid = d["centroid"]
        cen_active = centroid[ann & np.isfinite(centroid)]
        row = {
            "shot": stem,
            "n_frames": int(occ.size),
            "ann_frac": float(ann.mean()),
            "lfm_frac": float(lfm.mean()),
            "auroc": auroc(occ.astype(np.float64), ann),
            "median_centroid_khz": float(np.median(cen_active)) if cen_active.size else float("nan"),
            "notched_bins_per_channel": [int(v) for v in d["notched"].sum(axis=1)],
            "notched_bins_union": int(d["notched"].any(axis=0).sum()),
            "notched_bins_union_in_band": int(d["notched"][:, BAND_LO_BIN:BAND_HI_BIN].any(axis=0).sum()),
            "max_bin_active_frac": float(d["active_frac"].max()),
            "max_bin_active_frac_in_band": float(d["active_frac"][:, BAND_LO_BIN:BAND_HI_BIN].max()),
            "mean_occ": float(occ.mean()),
        }
        for thr in OCC_THRESHOLDS:
            pred = occ >= thr
            key = f"{thr:g}"
            row[f"recall@{key}"] = float(pred[ann].mean()) if ann.any() else float("nan")
            row[f"precision@{key}"] = float(ann[pred].mean()) if pred.any() else float("nan")
            row[f"predpos@{key}"] = float(pred.mean())
        per_shot.append(row)

    occ = np.concatenate(occ_all).astype(np.float64)
    ann = np.concatenate(ann_all)
    lfm = np.concatenate(lfm_all)

    pooled = {
        "n_frames": int(occ.size),
        "n_shots": len(stems),
        "ann_positive_frac": float(ann.mean()),
        "lfm_positive_frac": float(lfm.mean()),
        "auroc": auroc(occ, ann),
        "thresholds": {},
    }
    for thr in OCC_THRESHOLDS:
        pred = occ >= thr
        key = f"{thr:g}"
        unann = pred & ~ann
        pooled["thresholds"][key] = {
            "recall": float(pred[ann].mean()),
            "precision": float(ann[pred].mean()) if pred.any() else float("nan"),
            "predicted_positive_frac": float(pred.mean()),
            "unannotated_frac_of_predicted": float(unann.mean() / pred.mean()) if pred.any() else float("nan"),
            "unannotated_inside_lfm_frac": float((unann & lfm).sum() / pred.sum()) if pred.any() else float("nan"),
            "unannotated_outside_any_frac": float((unann & ~lfm).sum() / pred.sum()) if pred.any() else float("nan"),
        }

    auc_vals = np.array([r["auroc"] for r in per_shot], dtype=float)
    finite_auc = auc_vals[np.isfinite(auc_vals)]
    per_shot_summary = {
        "n_shots_with_annotation": int(finite_auc.size),
        "auroc_median": float(np.median(finite_auc)) if finite_auc.size else float("nan"),
        "auroc_p10": _pct(auc_vals, 10),
        "auroc_below_0.6": int((finite_auc < 0.6).sum()),
        "auroc_min": float(finite_auc.min()) if finite_auc.size else float("nan"),
        "auroc_max": float(finite_auc.max()) if finite_auc.size else float("nan"),
        "recall": {},
    }
    for thr in OCC_THRESHOLDS:
        key = f"{thr:g}"
        rec = np.array([r[f"recall@{key}"] for r in per_shot], dtype=float)
        finite = rec[np.isfinite(rec)]
        per_shot_summary["recall"][key] = {
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "p10": _pct(rec, 10),
            "frac_ge_0.5_of_annotated_shots": float((finite >= 0.5).mean()) if finite.size else float("nan"),
            "frac_ge_0.5_of_all_180": float((np.nan_to_num(rec, nan=0.0) >= 0.5).mean()),
            "n_ge_0.5": int((finite >= 0.5).sum()),
        }

    cen = np.array([r["median_centroid_khz"] for r in per_shot], dtype=float)
    finite_cen = cen[np.isfinite(cen)]
    notch = np.array([r["notched_bins_union"] for r in per_shot], dtype=float)
    peak_frac = np.array([r["max_bin_active_frac"] for r in per_shot], dtype=float)
    peak_frac_band = np.array([r["max_bin_active_frac_in_band"] for r in per_shot], dtype=float)
    notch_ch = np.array(
        [r["notched_bins_per_channel"] for r in per_shot], dtype=float
    )
    extras = {
        "centroid_khz": {
            "median_of_per_shot_medians": float(np.median(finite_cen)) if finite_cen.size else float("nan"),
            "p10": _pct(cen, 10),
            "p90": _pct(cen, 90),
            "min": float(finite_cen.min()) if finite_cen.size else float("nan"),
            "max": float(finite_cen.max()) if finite_cen.size else float("nan"),
            "n_shots_in_80_250": int(((finite_cen >= 80.0) & (finite_cen <= 250.0)).sum()),
            "n_shots_with_centroid": int(finite_cen.size),
        },
        "notched_bins": {
            "union_median": float(np.median(notch)),
            "union_mean": float(notch.mean()),
            "union_max": float(notch.max()),
            "union_min": float(notch.min()),
            "per_channel_mean": float(notch_ch.mean()),
            "per_channel_max": float(notch_ch.max()),
            "n_shots_with_zero_notched": int((notch == 0).sum()),
            "in_band_union_mean": float(np.mean([r["notched_bins_union_in_band"] for r in per_shot])),
            "in_band_union_max": float(np.max([r["notched_bins_union_in_band"] for r in per_shot])),
            "threshold": NOTCH_ACTIVE_FRAC,
            "max_bin_active_frac_median": float(np.median(peak_frac)),
            "max_bin_active_frac_max": float(peak_frac.max()),
            "max_bin_active_frac_in_band_median": float(np.median(peak_frac_band)),
            "max_bin_active_frac_in_band_max": float(peak_frac_band.max()),
        },
    }
    return {"pooled": pooled, "per_shot_summary": per_shot_summary, **extras, "per_shot": per_shot}


def verdict_for(metrics: dict) -> dict:
    """GO if pooled recall >= 0.70 at some threshold AND >= 80 % of the 180 shots
    have per-shot recall >= 0.5 at that same threshold. Precision, AUROC and the
    unannotated fraction are reported but never fail the verdict."""
    passing = []
    rows = []
    for thr in OCC_THRESHOLDS:
        key = f"{thr:g}"
        pooled_recall = metrics["pooled"]["thresholds"][key]["recall"]
        shot_frac_all = metrics["per_shot_summary"]["recall"][key]["frac_ge_0.5_of_all_180"]
        shot_frac_ann = metrics["per_shot_summary"]["recall"][key]["frac_ge_0.5_of_annotated_shots"]
        ok = pooled_recall >= 0.70 and shot_frac_all >= 0.80
        rows.append(
            {
                "threshold": thr,
                "pooled_recall": pooled_recall,
                "frac_shots_recall_ge_0.5_of_180": shot_frac_all,
                "frac_shots_recall_ge_0.5_of_annotated": shot_frac_ann,
                "passes": bool(ok),
            }
        )
        if ok:
            passing.append(thr)
    return {
        "rule": (
            "GO if pooled recall of annotated-active frames >= 0.70 at some occupancy "
            "threshold AND per-shot recall >= 0.5 on at least 80 % of the 180 shots at "
            "that threshold. Precision, AUROC and the unannotated fraction are reported "
            "but never fail the verdict."
        ),
        "per_threshold": rows,
        "passing_thresholds": passing,
        "verdict": "GO" if passing else "NO-GO",
    }


# --- figures -----------------------------------------------------------------


def _shrink_png(path: Path, colors: int = 128) -> None:
    """Palette-quantise a saved PNG so the committed figures stay modest (~0.5 MB)."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    img.quantize(colors=colors, method=Image.Quantize.FASTOCTREE).save(path, optimize=True)


def _decimate(arr: np.ndarray, factor: int, how: str) -> np.ndarray:
    if factor <= 1:
        return arr
    n = (arr.shape[-1] // factor) * factor
    blocks = arr[..., :n].reshape(*arr.shape[:-1], n // factor, factor)
    return blocks.max(axis=-1) if how == "max" else blocks.mean(axis=-1)


def figure_shot(
    out_root: Path, fig_dir: Path, stem: str, transform: str, dpi: int, decim: int = 4
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = load_series(out_root, stem, transform)
    with np.load(out_root / "masks" / f"{stem}_{transform}_mask.npz") as z:
        packed = z["packed"]
    mask = np.unpackbits(packed, axis=-1, count=N_FRAMES).astype(bool)  # (4, 512, T)
    t_ms, freqs = d["t_ms"], d["freqs"]
    if transform == "aemodes":
        raw, _ = spectrogram_aemodes(SPEC_CACHE / f"{stem}.tif")
    else:
        raw, _ = spectrogram_tokeye(read_signals(TS_CACHE / f"{stem}.arrow"))
    band = slice(BAND_LO_BIN, BAND_HI_BIN)
    spec = _decimate(raw[0][band], decim, "mean")
    m = _decimate(mask[0][band].astype(np.float32), decim, "max")
    extent = (float(t_ms[0]), float(t_ms[-1]), float(freqs[BAND_LO_BIN]), float(freqs[BAND_HI_BIN - 1]))

    fig, axes = plt.subplots(
        3, 1, figsize=(11, 7), sharex=True, height_ratios=[3.0, 1.0, 1.0], constrained_layout=True
    )
    ax = axes[0]
    # display only: subtract each bin's median over time so modes stand out of the
    # broadband background. The model saw the un-subtracted spectrogram.
    disp = spec - np.median(spec, axis=1, keepdims=True)
    lo, hi = np.percentile(disp, [10, 99.8])
    ax.imshow(disp, origin="lower", aspect="auto", extent=extent, cmap="gray", vmin=lo, vmax=hi)
    overlay = np.zeros((*m.shape, 4), dtype=np.float32)
    overlay[m >= 0.5] = (0.0, 1.0, 0.4, 0.55)
    ax.imshow(overlay, origin="lower", aspect="auto", extent=extent)
    for b in np.flatnonzero(d["notched"][0]):
        if BAND_LO_BIN <= b < BAND_HI_BIN:
            ax.axhline(float(freqs[b]), color="red", lw=0.9, ls="--", alpha=0.9)
    ax.set_ylabel("frequency [kHz]")
    n_notch = int(d["notched"][0].sum())
    auc = auroc(d["occ"].astype(np.float64), d["ann"]) if d["ann"].any() else float("nan")
    ax.set_title(
        f"{stem} | {transform} transform | channel r0 | per-shot AUROC {auc:.3f} | "
        f"green = coherent mask (prob>={PROB_THRESHOLD}) | notched bins in r0: {n_notch} "
        f"(red dashed if in band)\nper-bin median removed and time decimated x{decim} "
        f"for display only; the model saw the raw spectrogram",
        fontsize=8,
    )

    ax = axes[1]
    t_dec = _decimate(t_ms, decim, "mean")
    ax.plot(t_dec, _decimate(d["occ"], decim, "max"), lw=0.6, color="k", label="band occupancy")
    for thr, colour in zip(OCC_THRESHOLDS, ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]):
        ax.axhline(thr, color=colour, lw=0.7, ls=":", label=f"occ={thr:g}")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.0)
    ax.set_ylabel("occupancy")
    ax.legend(fontsize=6, ncol=5, loc="upper right")

    ax = axes[2]
    fl = _decimate(d["frame_labels"].astype(np.float32), decim, "max") > 0.5
    for i, name in enumerate(LABEL_NAMES):
        colour = "0.6" if i == 0 else plt.cm.tab10(i)
        ax.fill_between(t_dec, i, i + 0.8, where=fl[i], color=colour, step="mid", lw=0)
    ax.fill_between(
        t_dec, 5.1, 5.9, where=_decimate(d["ann"].astype(np.float32), decim, "max") > 0.5,
        color="k", step="mid", lw=0,
    )
    ax.set_yticks([i + 0.4 for i in range(5)] + [5.5])
    ax.set_yticklabels([f"label_{i} {n}" for i, n in enumerate(LABEL_NAMES)] + ["AE (1|2|3|4)"], fontsize=7)
    ax.set_ylim(0, 6)
    ax.set_xlabel("time [ms]")

    path = fig_dir / f"{stem}_mask.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    _shrink_png(path)
    return path


def figure_pooled(out_root: Path, fig_dir: Path, stems: list[str], metrics: dict, dpi: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    ax = axes[0]
    for transform, colour in zip(TRANSFORMS, ["#1f77b4", "#d62728"]):
        occ = np.concatenate([load_series(out_root, s, transform)["occ"] for s in stems]).astype(np.float64)
        ann = np.concatenate([load_series(out_root, s, transform)["ann"] for s in stems])
        fpr, tpr = roc_curve(occ, ann)
        ax.plot(fpr, tpr, color=colour, lw=1.4, label=f"{transform} (AUROC {metrics[transform]['pooled']['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], color="0.7", lw=0.8, ls="--")
    ax.set_xlabel("false positive rate (unannotated frames)")
    ax.set_ylabel("true positive rate (annotated AE frames)")
    ax.set_title(f"pooled ROC, occupancy vs annotation ({len(stems)} shots)")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    bins = np.linspace(0.0, 1.0, 41)
    for transform, colour in zip(TRANSFORMS, ["#1f77b4", "#d62728"]):
        vals = np.array([r["auroc"] for r in metrics[transform]["per_shot"]], dtype=float)
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=bins, histtype="step", lw=1.4, color=colour, label=f"{transform} (median {np.median(vals):.3f})")
    ax.axvline(0.5, color="0.7", lw=0.8, ls="--")
    ax.axvline(0.6, color="k", lw=0.8, ls=":")
    ax.set_xlabel("per-shot AUROC")
    ax.set_ylabel("shots")
    ax.set_title("per-shot AUROC distribution")
    ax.legend(fontsize=8, loc="upper left")

    path = fig_dir / "occupancy_vs_annotation.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def run_metrics(args) -> None:
    out_root = Path(args.out_root)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    shots = shot_files()
    if args.limit:
        shots = shots[: args.limit]
    stems = [s for s, _, _ in shots]
    stems = [
        s
        for s in stems
        if all((out_root / "masks" / f"{s}_{t}.npz").exists() for t in TRANSFORMS)
    ]
    print(f"metrics over {len(stems)} shots")

    metrics = {t: transform_metrics(out_root, stems, t) for t in TRANSFORMS}
    better = max(TRANSFORMS, key=lambda t: metrics[t]["pooled"]["auroc"])
    verdicts = {t: verdict_for(metrics[t]) for t in TRANSFORMS}

    order = sorted(
        [r for r in metrics[better]["per_shot"] if np.isfinite(r["auroc"])],
        key=lambda r: r["auroc"],
    )
    worst = [r["shot"] for r in order[:3]]
    best = [r["shot"] for r in order[-3:]][::-1]
    best = [s for s in best if s not in worst]

    figures = []
    if not args.no_figures:
        for stem in best + worst:
            figures.append(str(figure_shot(out_root, fig_dir, stem, better, args.dpi, args.decim)))
        figures.append(str(figure_pooled(out_root, fig_dir, stems, metrics, args.dpi)))

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "environment": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "pythonpath": os.environ.get("PYTHONPATH", ""),
        },
        "inputs": {
            "checkpoint": str(CHECKPOINT),
            "spectrograms": str(SPEC_CACHE),
            "timeseries": str(TS_CACHE),
            "stats_json": json.loads(STATS_JSON.read_text()),
        },
        "settings": {
            "prob_threshold": PROB_THRESHOLD,
            "notch_active_frac": NOTCH_ACTIVE_FRAC,
            "band_bins": [BAND_LO_BIN, BAND_HI_BIN - 1],
            "band_khz": [80.56648681640625, 250.00025],
            "occ_thresholds": list(OCC_THRESHOLDS),
            "n_frames": N_FRAMES,
            "coherent_channel": 0,
            "column_window": (
                f"OOM fallback: column windows of {_forced_width[0]}"
                if _forced_width
                else ("full width (7820) - tokeye.inference does not window" if args.width <= 0 else args.width)
            ),
            "standardisation": {
                "aemodes": "global step_0a/stats.json mean 54.637 / std 2.732",
                "tokeye": "per-channel (x - mean) / (std + 1e-6), as tokeye.inference.model_infer",
            },
        },
        "shots": stems,
        "better_transform": better,
        "better_transform_chosen_by": "higher pooled AUROC",
        "verdict": verdicts[better]["verdict"],
        "verdicts": verdicts,
        "metrics": metrics,
        "figures": figures,
        "best_shots": best,
        "worst_shots": worst,
    }
    out_json = Path(args.json_out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=float))
    print(json.dumps({t: {"auroc": metrics[t]["pooled"]["auroc"], "verdict": verdicts[t]["verdict"]} for t in TRANSFORMS}, indent=2))
    print(f"better transform: {better}; verdict: {verdicts[better]['verdict']}")
    print(f"wrote {out_json}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("masks", "metrics", "all"), default="all")
    p.add_argument("--out-root", default="/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/ae")
    p.add_argument(
        "--fig-dir", default="/scratch/gpfs/nc1514/FusionAIHub/outputs/labelmaker/ae/figures"
    )
    p.add_argument(
        "--json-out",
        default="/scratch/gpfs/nc1514/FusionAIHub/outputs/labelmaker/ae/mask_vs_annotation.json",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--width", type=int, default=0, help="column window; 0 = full width")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--transforms", default=",".join(TRANSFORMS))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--dpi", type=int, default=130)
    p.add_argument("--decim", type=int, default=4, help="time decimation for shot figures")
    args = p.parse_args()
    args.transforms = tuple(t for t in args.transforms.split(",") if t)
    if args.stage in ("masks", "all"):
        run_masks(args)
    if args.stage in ("metrics", "all"):
        run_metrics(args)


if __name__ == "__main__":
    main()
