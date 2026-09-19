#!/usr/bin/env python3
"""Write the adjusted AE labels; run with the labelmaker environment.

Two changes to the original annotations, and nothing else:

1. COLLAPSE. The five original classes become one binary presence label, the
   union of the four AE classes. LFM is excluded - it is not an Alfven
   eigenmode - which is the same rule `format_ae_records` applies.

2. SHIFT. The annotations run late against the spectrograms they were drawn
   on, so the label array is ROLLED earlier by SHIFT_MS and `time_ms` stays
   the plain 0-2000 ms grid: the correction is built into the labels, and a
   reader needs to know nothing about it. Rolling vacates the tail - the last
   `|shift_frames|` bins have no annotation behind them - so `valid` is False
   there and `label` is 0. Absent and unannotated are NOT the same claim.
   Cross-correlating a narrowband ridge trace (80-250 kHz, detrended per
   bin to drop receiver lines and per frame to drop broadband bursts) against
   the labels peaks at -278 ms aggregated over 179 shots, with a per-shot
   median of -245 ms. The objective is flat from about -200 to -300 ms
   (mean r 0.334, 0.343, 0.347, 0.343 at -200, -245, -278, -300 against 0.169
   unshifted), so SHIFT_MS is a round number inside that flat top rather than
   a value the data singles out. A single shift improves 74% of shots.

   It is NOT a recovered clock offset: a real time-base error would be the
   same on every shot, and the per-shot IQR is -339..-74 ms. The shift also
   absorbs the difference between what a ridge detector marks and what an
   annotator marked. Good enough to line plots up; not a per-shot correction.

The spectrograms in the original pickle are dropped. This file carries labels
and time only, so it is under a megabyte instead of 2.9 GB.
"""

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from labeler.config import Paths  # noqa: E402

#: Original class order in the pickle's `(3905, 5)` label arrays.
AE_CLASSES = ("lfm", "bae", "eae", "rsae", "tae")
#: Rows of the original label grid, and the window they are taken to span.
N_FRAMES, WINDOW_MS = 3905, (0.0, 2000.0)
#: See the module docstring. Set to 0.0 for the pickle's own pairing.
SHIFT_MS = -250.0
SOURCE_STEM = "co2_detector_2021"
OUTPUT_NAME = "co2_250_detector_adjust.pkl"


def collapse(labels):
    """Five classes -> one binary presence label, LFM excluded."""
    labels = np.asarray(labels)
    if labels.shape != (N_FRAMES, len(AE_CLASSES)):
        raise ValueError(f"expected {(N_FRAMES, len(AE_CLASSES))}, got {labels.shape}")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("expected the original binary labels")
    return (labels[:, 1:].max(axis=1) > 0).astype(np.uint8)


def frame_times_ms():
    """Centre of each of the 3905 bins, unshifted - the shift is in the labels."""
    lo, hi = WINDOW_MS
    edges = np.linspace(lo, hi, N_FRAMES + 1)
    return edges[:-1] + np.diff(edges) / 2.0


def shift_frames(shift_ms=SHIFT_MS):
    lo, hi = WINDOW_MS
    return int(round(shift_ms / ((hi - lo) / N_FRAMES)))


def roll(labels, frames):
    """Move labels earlier by `frames`, leaving the vacated tail unannotated."""
    rolled = np.roll(np.asarray(labels), frames, axis=-1)
    valid = np.ones(rolled.shape[-1], bool)
    if frames < 0:
        rolled[..., frames:] = 0
        valid[frames:] = False
    elif frames > 0:
        rolled[..., :frames] = 0
        valid[:frames] = False
    return rolled, valid


def main():
    root = Paths.from_env().label_tables
    manifest = yaml.safe_load((root / "events.yaml").read_text())
    source = next(r for r in manifest["raw_datasets"] if r["stem"] == SOURCE_STEM)
    source_path = root / source["path"]
    output_path = source_path.parent / OUTPUT_NAME

    print(f"reading {source_path} (2.9 GB; a minute or two)")
    with source_path.open("rb") as stream:
        dataset = pickle.load(stream)

    shots, splits, labels = [], [], []
    for split, offset in (("train", 0), ("valid", 3)):
        part_shots, _, part_labels = dataset[offset : offset + 3]
        for shot, label in zip(part_shots, part_labels):
            shots.append(int(shot))
            splits.append(split)
            labels.append(collapse(label))

    order = np.argsort(shots, kind="stable")
    frames = shift_frames()
    rolled, valid = roll(np.stack(labels)[order], frames)
    adjusted = {
        "shots": np.asarray(shots, int)[order],
        "split": np.asarray(splits)[order],
        "label": rolled,
        "valid": valid,
        "time_ms": frame_times_ms(),
        "shift_frames": frames,
        "categories": {0: "absent", 1: "present"},
        "shift_ms": SHIFT_MS,
        "source": source["path"],
        "collapsed_classes": AE_CLASSES[1:],
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with output_path.open("wb") as stream:
        pickle.dump(adjusted, stream, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "name": OUTPUT_NAME,
        "derived_from": source["path"],
        "written_by": "alfven_eigenmode/adjust_labels.py",
        "written": adjusted["written"],
        "shots": len(adjusted["shots"]),
        "frames_per_shot": N_FRAMES,
        "window_ms": list(WINDOW_MS),
        "shift_ms": SHIFT_MS,
        "shift_frames": int(shift_frames()),
        "shift_applied": "label array rolled; time_ms is the unshifted grid",
        "unannotated_tail_bins": int(abs(shift_frames())),
        "shift_basis": (
            "ridge/label cross-correlation, peak -278 ms over 179 shots, "
            "per-shot median -245 ms, objective flat from -200 to -300 ms"
        ),
        "collapsed_classes": list(AE_CLASSES[1:]),
        "excluded_classes": ["lfm"],
        "spectrograms": "dropped; labels and time only",
        "categories": {"0": "absent", "1": "present"},
    }
    output_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    frac = adjusted["label"][:, adjusted["valid"]].mean()
    print(f"wrote {output_path} ({output_path.stat().st_size / 1e6:.2f} MB)")
    print(f"  {len(adjusted['shots'])} shots, {frac * 100:.1f}% of valid bins present, "
          f"time {adjusted['time_ms'][0]:.1f} to {adjusted['time_ms'][-1]:.1f} ms, "
          f"rolled {frames} frames, {int(~adjusted['valid'].sum() * 0 + (~adjusted['valid']).sum())} "
          f"tail bins unannotated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
