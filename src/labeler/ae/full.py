"""TokEye over whole shots: its input, frames, annotation and tiles, and the check
against v1.

    python -m labeler.ae.full --check [--shots 170720,176053] [--no-seldnet]

v1's TokEye masks (`ae/masks`, made by `scripts/labeler/ae_dataset.py`) cover
0-2 s only, because the aemodes arrow cache they were made from holds 0-2 s
only. The owner labelled AE to the end of the shot, so round three runs TokEye
over the whole record into `ae/masks-full` and `ae/dataset-full`, under v1's
names and keys plus `EXTRA_KEYS`. The GPU job, `scripts/labeler/ae_masks_full.py`,
runs in the tokeye venv and imports this module for everything that does not
need that venv:

- **the input** (`input_signals`): the raw cache's CO2 (`raw.raw_signal`, about
  -1.5 to 7.5 s at 1.667 MHz), cut to [0, min(end, `CROP_END_MS`)] with a
  `MARGIN_MS` margin either side, resampled to 500 kHz by the review rows' own
  lines (`alfven.resample`), and trimmed back to [0, min(end, `CROP_END_MS`)];
- **the frames** (`frame_times`): v1's `frame_grid` rule on any record;
- **the annotation** (`annotation`): UCI's `ann`, `lfm` and `frame_labels`,
  which exist over 0-2 s only, taken from v1's clean file at the nearest v1
  frame and False after v1's last frame;
- **the tiles** (`tiles`, `tiled_probs`) of the fallback, used only if a whole
  shot does not fit on the GPU or the check fails: frames cut into runs, each
  run given a margin either side and only its middle kept;
- **the check** against v1 over 0-2 s (`check_shot`, `check_bar`, `main`).

**The check.** The frames compared are the whole-shot record's inside v1's
record and before `CHECK_UNTIL_MS`, each against its nearest v1 frame. Per
shot: the IoU of the pixels lit on at least `MIN_CHORDS` of the four chords in
the un-notched cleaned mask, in 0-250 kHz and in 80-250 kHz (`BANDS`); each
record's notched bins (per record: a bin lit in over `NOTCH` of its frames);
and, unless skipped, how often SELDNet, run on each record's spectrogram, says
the same of the 0-2 s frames both records cover. **The bar** (`check_bar`), in
both bands: a median IoU of at least `IOU_MEDIAN`, and at most `IOU_LOW_SHARE` of
the shots below `IOU_LOW`. A shot with nothing lit in a band on either record is
left out of that band. The bar is reported in `masks-full/check.{json,md}`, not
raised: the command succeeds either way.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ..config import Paths, atomic_path
from ..events import raw
from ..events.review import alfven
from .labels import (
    BAND_HI_BIN,
    BAND_LO_BIN,
    N_BINS,
    N_FFT,
    bin_active_fraction,
    bin_freqs_khz,
    notch_bins,
)
from .transform import short_time_fft
from .xpower import tokeye_masks
from .xpower.data import MIN_CHORDS, NOTCH, frame_covered, seldnet_split

#: The input's end: every owner window ends by 5448 ms, and 6000 ms is where
#: the review rows' quiet span ends.
CROP_END_MS = 6000.0
#: Record kept either side of [0, end] through the resampler, then cut off, so
#: its filter's edges fall outside the input.
MARGIN_MS = 10.0
#: The check compares frames before this, where v1's record ends.
CHECK_UNTIL_MS = 2000.0
#: The check's two bands, as TokEye bins [lo, hi): 0-250 and 80-250 kHz.
BANDS = {"0-250": (0, N_BINS), "80-250": (BAND_LO_BIN, BAND_HI_BIN)}
#: The bar, in both bands: a median IoU of at least IOU_MEDIAN, and at most
#: IOU_LOW_SHARE of the shots below IOU_LOW.
IOU_MEDIAN = 0.75
IOU_LOW = 0.50
IOU_LOW_SHARE = 0.10
#: The four CO2 chords, in the masks' channel order (`ae_dataset.CHANNELS`).
CHANNEL_NAMES = ("r0", "v1", "v2", "v3")
#: What a masks-full clean file holds beyond v1's keys.
EXTRA_KEYS = ("ann_until_ms", "t0_input_ms", "fs_hz", "tile_ms")
#: The rate the input is resampled to, in kHz: the check's frequency axis.
FS_KHZ = alfven.RATE_HZ / 1000


def masks_full_dir(paths: Paths) -> Path:
    """TokEye's whole-shot masks, `<shot>_<split>_{clean,probs}.npz`."""
    return paths.root / "ae" / "masks-full"


def dataset_full_dir(paths: Paths) -> Path:
    """SELDNet's whole-shot inputs, `<shot>_<split>.npz`."""
    return paths.root / "ae" / "dataset-full"


def input_signals(shot: int, *, paths: Paths) -> tuple[np.ndarray, float, float]:
    """TokEye's input: the four CO2 chords `(4, n)` float64 at ~500 kHz over
    [0, min(record end, CROP_END_MS)] ms, the time of the first sample (ms), and
    the exact rate (Hz)."""
    a = raw.raw_signal(shot, "co2", paths=paths)
    if len(a.y) != len(CHANNEL_NAMES):
        raise ValueError(f"shot {int(shot)}: {len(a.y)} CO2 chords, not 4")
    end = min(float(a.x[-1]), CROP_END_MS)
    keep = (a.x >= -MARGIN_MS) & (a.x <= end + MARGIN_MS)
    time_ms = a.x[keep]
    x, fs = alfven.resample(time_ms, a.y[:, keep])
    t = time_ms[0] + np.arange(x.shape[1]) * 1000 / fs
    inside = np.flatnonzero((t >= 0) & (t <= end))
    if inside.size == 0:
        raise ValueError(f"shot {int(shot)}: no CO2 sample in [0, {end}] ms")
    lo, hi = inside[0], inside[-1] + 1
    return x[:, lo:hi].astype(np.float64), float(t[lo]), float(fs)


def frame_times(n_samples: int, t0_ms: float, fs_hz: float) -> np.ndarray:
    """The frame times (ms) of TokEye's STFT of `n_samples` samples from `t0_ms`
    at `fs_hz`: v1's `frame_grid` rule, `ShortTimeFFT(hann(N_FFT), HOP).t`, on
    any record."""
    return t0_ms + short_time_fft(fs_hz / 1000).t(n_samples)


def nearest(t_from: np.ndarray, t_to: np.ndarray) -> np.ndarray:
    """For each time in `t_to`, the index of the nearest time in sorted `t_from`,
    the earlier on a tie."""
    t_from = np.asarray(t_from, dtype=np.float64)
    t_to = np.asarray(t_to, dtype=np.float64)
    if t_from.size == 0:
        raise ValueError("no times to take the nearest of")
    if t_from.size == 1:
        return np.zeros(t_to.shape, dtype=np.intp)
    right = np.clip(np.searchsorted(t_from, t_to), 1, t_from.size - 1)
    left = right - 1
    return np.where(t_to - t_from[left] <= t_from[right] - t_to, left, right)


def annotation(v1_clean: Path, t_ms: np.ndarray) -> dict:
    """UCI's `ann`, `lfm` and `frame_labels` `(5, T)` at the frames `t_ms`, from
    v1's clean file: each frame takes the nearest v1 frame's, and is False after
    v1's last frame. Also "ann_until_ms", the time of that last frame."""
    t_ms = np.asarray(t_ms, dtype=np.float64)
    with np.load(v1_clean) as z:
        t_v1 = np.asarray(z["t_ms"], dtype=np.float64)
        ann = np.asarray(z["ann"], dtype=bool)
        lfm = np.asarray(z["lfm"], dtype=bool)
        frame_labels = np.asarray(z["frame_labels"], dtype=bool)
    near = nearest(t_v1, t_ms)
    inside = t_ms <= t_v1[-1]
    return {
        "ann": ann[near] & inside,
        "lfm": lfm[near] & inside,
        "frame_labels": frame_labels[:, near] & inside,
        "ann_until_ms": float(t_v1[-1]),
    }


def tiles(
    n_frames: int, tile_frames: int, margin: int = 128
) -> list[tuple[int, int, int, int]]:
    """`(lo, hi, keep_lo, keep_hi)` per tile: the keep ranges cut [0, n_frames)
    into runs of `tile_frames` (the last shorter), and each tile runs `margin`
    frames beyond its keep range either side, inside the record. `tile_frames`
    <= 0 is one tile, the whole record."""
    if tile_frames <= 0:
        return [(0, n_frames, 0, n_frames)]
    cuts = []
    for keep_lo in range(0, n_frames, tile_frames):
        keep_hi = min(n_frames, keep_lo + tile_frames)
        lo, hi = max(0, keep_lo - margin), min(n_frames, keep_hi + margin)
        cuts.append((lo, hi, keep_lo, keep_hi))
    return cuts


def tiled_probs(
    mask_probs,
    model,
    norm: np.ndarray,
    device: str,
    batch: int,
    cuts: list[tuple[int, int, int, int]],
) -> np.ndarray:
    """TokEye's probabilities `(C, 2, bins, T)` uint8, one `mask_probs` call per
    cut on its frames `lo:hi`, of which the frames `keep_lo:keep_hi` are kept."""
    n_ch, n_bins, n_frames = norm.shape
    out = np.zeros((n_ch, 2, n_bins, n_frames), dtype=np.uint8)
    for lo, hi, keep_lo, keep_hi in cuts:
        prob = mask_probs(model, norm[..., lo:hi], device, batch)
        if prob.shape != (n_ch, 2, n_bins, hi - lo):
            raise ValueError(
                f"frames {lo}-{hi}: mask_probs gave {prob.shape}, not "
                f"{(n_ch, 2, n_bins, hi - lo)}"
            )
        out[..., keep_lo:keep_hi] = prob[..., keep_lo - lo : keep_hi - lo]
    return out


def lit(clean: np.ndarray) -> np.ndarray:
    """`(bins, T)`: the pixels lit on at least MIN_CHORDS of the `(C, bins, T)`
    chords."""
    return np.asarray(clean, dtype=bool).sum(axis=0, dtype=np.uint8) >= MIN_CHORDS


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """`|a & b| / |a | b|`; nan when both are empty."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    union = np.count_nonzero(a | b)
    if union == 0:
        return math.nan
    return np.count_nonzero(a & b) / union


def _read_clean(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """A clean file's times (ms) and un-notched cleaned mask `(C, bins, T)`."""
    with np.load(path) as z:
        t = np.asarray(z["t_ms"], dtype=np.float64)
        clean = np.unpackbits(z["mask_clean"], axis=-1, count=len(t)).astype(bool)
    return t, clean


def _read_times(path: Path) -> np.ndarray:
    with np.load(path) as z:
        return np.asarray(z["t_ms"], dtype=np.float64)


def _notched_khz(clean: np.ndarray) -> dict[str, list[float]]:
    """Per chord of CHANNEL_NAMES, the bins the notch takes from this record,
    in kHz to 0.01."""
    notched = notch_bins(bin_active_fraction(clean), NOTCH)
    freqs = bin_freqs_khz(FS_KHZ)
    return {
        name: [round(float(f), 2) for f in freqs[row]]
        for name, row in zip(CHANNEL_NAMES, notched, strict=True)
    }


def check_shot(v1_clean: Path, full_clean: Path) -> dict:
    """One shot's whole-shot record against v1's over 0-2 s: the frames
    compared, the IoU of the lit pixels per band of BANDS, and each record's
    notched bins."""
    t_v1, clean_v1 = _read_clean(v1_clean)
    t_full, clean_full = _read_clean(full_clean)
    compared = (t_full >= t_v1[0]) & (t_full <= t_v1[-1]) & (t_full < CHECK_UNTIL_MS)
    lit_v1 = lit(clean_v1)[:, nearest(t_v1, t_full[compared])]
    lit_full = lit(clean_full[..., compared])
    return {
        "frames": int(compared.sum()),
        "iou": {
            band: iou(lit_v1[lo:hi], lit_full[lo:hi])
            for band, (lo, hi) in BANDS.items()
        },
        "notch_v1": _notched_khz(clean_v1),
        "notch_full": _notched_khz(clean_full),
    }


def check_bar(results: list[dict]) -> dict:
    """The bar over the shots' IoUs, per band and in both."""
    bands = {}
    for band in BANDS:
        values = np.array([float(r["iou"][band]) for r in results], dtype=np.float64)
        empty = int(np.isnan(values).sum())
        values = values[~np.isnan(values)]
        if values.size:
            median = float(np.median(values))
            below = int((values < IOU_LOW).sum())
            share = below / values.size
            passed = median >= IOU_MEDIAN and share <= IOU_LOW_SHARE
        else:
            median, below, share, passed = math.nan, 0, math.nan, False
        bands[band] = {
            "median": median,
            "below_half": below,
            "share_below_half": share,
            "empty": empty,
            "passed": bool(passed),
        }
    return {"bands": bands, "passed": all(b["passed"] for b in bands.values())}


def _number(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.3f}"


def _notch_cell(notch: dict[str, list[float]]) -> str:
    """Each chord's notched kHz, a run of adjacent bins as its first and last,
    as in `r0 124.02-127.93; v2 60.06`; "-" when nothing is notched."""
    step = FS_KHZ / N_FFT
    parts = []
    for name, khz in notch.items():
        if not khz:
            continue
        runs = [[khz[0], khz[0]]]
        for f in khz[1:]:
            if f - runs[-1][1] < 1.5 * step:
                runs[-1][1] = f
            else:
                runs.append([f, f])
        text = ", ".join(f"{a:.2f}" if a == b else f"{a:.2f}-{b:.2f}" for a, b in runs)
        parts.append(f"{name} {text}")
    return "; ".join(parts) or "-"


def check_md(results: list[dict], bar: dict) -> str:
    """The check as Markdown: the bar per band, then one row per shot."""
    changed = sum(r["notch_v1"] != r["notch_full"] for r in results)
    lines = [
        "# Whole-shot TokEye against v1, 0-2 s",
        "",
        (
            f"The IoU of the pixels lit on at least {MIN_CHORDS} of the four chords "
            "(the un-notched cleaned mask), over the whole-shot frames inside v1's "
            f"record and before {CHECK_UNTIL_MS:.0f} ms, each against its nearest "
            f"v1 frame. The bar, in each band: a median IoU of at least "
            f"{IOU_MEDIAN:.2f} and at most {IOU_LOW_SHARE:.0%} of the shots below "
            f"{IOU_LOW:.2f}. A shot with nothing lit in a band on either record is "
            "left out of that band (empty)."
        ),
        "",
        (
            f"**The bar {'passed' if bar['passed'] else 'failed'}.** The notched "
            f"bins differ between the records on {changed} of {len(results)} shots."
        ),
        "",
        (
            f"| band (kHz) | shots | median IoU | below {IOU_LOW:.2f} "
            f"| share below {IOU_LOW:.2f} | empty | passed |"
        ),
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for band, b in bar["bands"].items():
        lines.append(
            f"| {band} | {len(results) - b['empty']} | {_number(b['median'])} "
            f"| {b['below_half']} | {_number(b['share_below_half'])} "
            f"| {b['empty']} | {'yes' if b['passed'] else 'no'} |"
        )
    ious = " | ".join(f"IoU {band}" for band in BANDS)
    lines += [
        "",
        (
            f"| shot | split | frames | {ious} | notched, v1 (kHz) "
            "| notched, whole shot (kHz) | SELDNet agreement |"
        ),
        "|---:|---|---:|" + "---:|" * len(BANDS) + "---|---|---:|",
    ]
    for r in results:
        values = " | ".join(_number(r["iou"][band]) for band in BANDS)
        agreement = _number(r["seldnet_agreement"]) if "seldnet_agreement" in r else "-"
        lines.append(
            f"| {r['shot']} | {r['split']} | {r['frames']} | {values} "
            f"| {_notch_cell(r['notch_v1'])} | {_notch_cell(r['notch_full'])} "
            f"| {agreement} |"
        )
    return "\n".join(lines) + "\n"


def seldnet_agreement(net, paths: Paths, stem: str) -> float:
    """The share of the 0-2 s frames both records cover where SELDNet says the
    same on v1's `ae/dataset/<stem>.npz` (v1's frames) and on
    `dataset-full/<stem>.npz` (masks-full's frames); nan when no frame is
    covered by both."""
    from .xpower import evaluate

    first, n = evaluate.EVAL_FRAMES
    covered, said = [], []
    for masks, dataset in (
        (tokeye_masks(paths), paths.root / "ae" / "dataset"),
        (masks_full_dir(paths), dataset_full_dir(paths)),
    ):
        t = _read_times(masks / f"{stem}_clean.npz")
        covered.append(frame_covered(t, first, n))
        said.append(evaluate.seldnet_said(net, dataset / f"{stem}.npz", t, first, n))
    both = covered[0] & covered[1]
    if not both.any():
        return math.nan
    return float(np.mean(said[0][both] == said[1][both]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--check", action="store_true", help="check masks-full against v1, 0-2 s"
    )
    p.add_argument("--shots", help="comma-separated; default every AE180 shot")
    p.add_argument(
        "--no-seldnet", action="store_true", help="skip SELDNet's frame agreement"
    )
    args = p.parse_args(argv)
    if not args.check:
        p.error("nothing to do: pass --check")
    paths = Paths.from_env()
    v1, full = tokeye_masks(paths), masks_full_dir(paths)
    split = seldnet_split(v1)
    if not split:
        p.error(f"{v1}: no v1 TokEye clean files")
    if not full.is_dir():
        p.error(f"{full}: no whole-shot masks (scripts/labeler/ae_masks_full.py)")
    shots = sorted(split)
    if args.shots:
        try:
            shots = sorted({int(s) for s in args.shots.split(",") if s.strip()})
        except ValueError:
            p.error(f"--shots {args.shots!r}: comma-separated shot numbers")
        unknown = [s for s in shots if s not in split]
        if unknown:
            p.error(f"not among v1's {len(split)} AE shots: {unknown}")
    stems = [f"{s}_{split[s]}" for s in shots]
    # With SELDNet, a record also needs its dataset file: a job cut between
    # the two writes leaves a clean file alone, which counts as missing.
    present = [
        s
        for s in stems
        if (full / f"{s}_clean.npz").is_file()
        and (args.no_seldnet or (dataset_full_dir(paths) / f"{s}.npz").is_file())
    ]
    missing = [s for s in stems if s not in present]
    net = None
    if present and not args.no_seldnet:
        from .xpower import evaluate  # torch and SELDNet, only when asked for

        net = evaluate.load_seldnet(paths)
    results = []
    for stem in present:
        shot = int(stem.split("_")[0])
        result = {
            "shot": shot,
            "split": split[shot],
            **check_shot(v1 / f"{stem}_clean.npz", full / f"{stem}_clean.npz"),
        }
        if net is not None:
            result["seldnet_agreement"] = seldnet_agreement(net, paths, stem)
        results.append(result)
    bar = check_bar(results)
    # A record that shares no 0-2 s frame with v1's has nothing to compare, so
    # its NaN IoUs must not pass for "nothing lit": it fails the bar.
    no_frames = [f"{r['shot']}_{r['split']}" for r in results if r["frames"] == 0]
    bar["passed"] = bool(bar["passed"] and not no_frames)
    record = {"shots": results, "bar": bar, "missing": missing}
    record["no_frames"] = no_frames
    with atomic_path(full / "check.json") as tmp:
        tmp.write_text(json.dumps(record, indent=1) + "\n")
    with atomic_path(full / "check.md") as tmp:
        tmp.write_text(check_md(results, bar))
    summary = {"bar": bar, "checked": len(results), "missing": missing}
    print(json.dumps({**summary, "no_frames": no_frames}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
