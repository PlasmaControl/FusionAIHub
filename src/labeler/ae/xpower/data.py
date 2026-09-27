"""The owner's AE review as 10 ms frames: inputs, targets, MHD flags and the split.

**Inputs** are the review page's own AE rows (`review/alfven.py`): CO2 R0 x V1,
V2 and V3 cross-power, 0.256 ms columns, 257 bins to 250 kHz, each 0-255 over
-3..27 dB above its bin's quiet-time median, cut to a band: 80-250 kHz (the AE
band, and the earlier detector's input) or 0-250 kHz (which lets a model see an
MHD mode's 0-60 kHz fundamental beside its harmonics, and learn that the pair is
not AE). `train.CANDIDATES` holds both; the validation shots choose. Columns are
averaged onto 2 ms sub-frames, five to each frame of `labeler.scoring.frames`.

**Targets** are the owner's saved labels, `review/labels.csv`, as frame states
(`frame_states`): present, absent, uncertain, not observable, or outside.

**MHD frames** are where TokEye's cleaned coherent mask (`$LABELER_ROOT/ae/masks`)
lights a line at or under ~60 kHz on at least two of the four chords for at least
half of a frame's columns. TokEye covers 0-2 s. They are the hard negatives
training up-weights and the named check `evaluate` runs.

**The split.** The test shots are the reviewed shots of the earlier detector's
own validation block (SELDNet's `valid`, a block of sessions), so every method
is compared on shots none of them trained on. The rest are drawn into train and
validation by seed 20260923.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from ...events.review import alfven
from ...events.review.labels import Label
from ...events.review.rows import Grid
from ...scoring.frames import FRAME_MS, Assessment, frame_states
from ..labels import apply_notch, bin_active_fraction, notch_bins

BAND_KHZ = (80.0, 250.0)
FULL_BAND_KHZ = (0.0, 250.0)
SUB_MS = 2.0
SUBS = int(FRAME_MS / SUB_MS)
#: Input frames either side of a labelled window, so its edges see context.
CONTEXT_FRAMES = 20
SEED = 20260923
N_VAL = 16
CROSS_ROWS = tuple(f"R0x{chord}" for chord in alfven.CHORDS[1:])
#: TokEye bins (0.488 kHz) at or under ~60 kHz, and its AE band, 80-250 kHz.
LOW_BINS = (0, 122)
AE_BINS = (164, 512)
MIN_CHORDS = 2
MIN_FRACTION = 0.5
NOTCH = 0.8


@dataclass(frozen=True)
class Shot:
    """One shot's frames `first .. first + n - 1`, context included."""

    shot: int
    first: int
    x: np.ndarray  # (3, n_bins, SUBS * n) float16 in [0, 1]
    states: np.ndarray  # (n,) int8, `frame_states`
    mhd: np.ndarray  # (n,) bool
    observed: np.ndarray  # (n,) bool: the rows cover the frame

    @property
    def n(self) -> int:
        return len(self.states)


def window_frames(window) -> tuple[int, int]:
    """The first frame and the number of whole frames inside `window` (ms)."""
    lo, hi = int(window[0]), int(window[1])
    first = -(-lo // FRAME_MS)
    return first, max(0, hi // FRAME_MS - first)


def band_slice(y0: float, dy: float, n_y: int, band=BAND_KHZ) -> slice:
    """The bins whose centre lies inside `band`, kHz."""
    lo = max(0, math.ceil((band[0] - y0) / dy - 1e-9))
    hi = min(n_y, math.floor((band[1] - y0) / dy + 1e-9) + 1)
    if hi <= lo:
        raise ValueError(f"no bin of {n_y} from {y0} kHz every {dy} lies in {band}")
    return slice(lo, hi)


def store_rows(path, level: int = 1) -> tuple[Grid, np.ndarray, float, float]:
    """The three cross-power rows of a review store: grid, `(3, n_y, n)`, y0, dy."""
    with h5py.File(path, "r") as f:
        t0, dt = float(f.attrs["t0_ms"]), float(f.attrs["dt_ms"])
        values = np.stack([f["rows"][name][str(level)][...] for name in CROSS_ROWS])
        meta = json.loads(f["rows"][CROSS_ROWS[0]].attrs["meta"])
    grid = Grid(t0, dt * level, values.shape[-1])
    return grid, values, float(meta["y0"]), float(meta["dy"])


def raw_rows(time_ms, chords) -> tuple[Grid, np.ndarray, float, float]:
    """The same rows computed from the four CO2 chords (`alfven.spectrogram_rows`)."""
    grid, rows = alfven.spectrogram_rows(time_ms, chords)
    by_name = {row.name: row for row in rows}
    values = np.stack([by_name[name].values for name in CROSS_ROWS])
    return grid, values, float(rows[0].y0), float(rows[0].dy)


def frame_inputs(
    values, grid: Grid, first: int, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Inputs: `(C, n_y, SUBS * n)` float32 in [0, 1] for frames `first ..`.

    Also returns `(n,)` observed.

    Each sub-frame is the mean of the columns whose centre falls in it; one no
    column falls in is 0. A frame is observed when half its sub-frames are.
    """
    values = np.asarray(values)
    subs = SUBS * n
    out = np.zeros((*values.shape[:-1], subs), dtype=np.float32)
    centres = grid.t0_ms + (np.arange(grid.n) + 0.5) * grid.dt_ms
    index = np.floor((centres - first * FRAME_MS) / SUB_MS).astype(np.int64)
    cols = np.flatnonzero((index >= 0) & (index < subs))
    counts = np.bincount(index[cols], minlength=subs)
    if cols.size:
        which = index[cols]
        starts = np.flatnonzero(np.r_[True, np.diff(which) > 0])
        sums = np.add.reduceat(values[..., cols].astype(np.float32), starts, axis=-1)
        out[..., which[starts]] = sums / counts[which[starts]]
    observed = (counts > 0).reshape(n, SUBS).mean(axis=1) >= 0.5
    return out / 255.0, observed


def targets(label: Label, first: int, n: int) -> np.ndarray:
    """The owner's state of frames `first .. first + n - 1`; -1 outside the window."""
    return frame_states(Assessment.from_label(label), first, n)


def clean_path(masks_dir, shot: int) -> Path | None:
    """TokEye's `<shot>_<split>_clean.npz`, or None."""
    found = sorted(Path(masks_dir).glob(f"{int(shot)}_*_clean.npz"))
    return found[0] if found else None


def frame_share(t_ms, flags, first: int, n: int) -> np.ndarray:
    """Per frame `first ..`: the share of the columns at `t_ms` where `flags` holds."""
    k = np.floor(np.asarray(t_ms, dtype=np.float64) / FRAME_MS).astype(np.int64) - first
    ok = (k >= 0) & (k < n)
    count = np.bincount(k[ok], minlength=n)
    hits = np.bincount(k[ok], weights=np.asarray(flags)[ok].astype(float), minlength=n)
    return hits / np.maximum(count, 1)


def frame_covered(t_ms, first: int, n: int) -> np.ndarray:
    """Whether the columns at `t_ms` (one step apart) span each whole frame."""
    t = np.asarray(t_ms, dtype=np.float64)
    half = (t[1] - t[0]) / 2 if len(t) > 1 else 0.0
    starts = (first + np.arange(n)) * FRAME_MS
    return (starts >= t[0] - half) & (starts + FRAME_MS <= t[-1] + half)


def tokeye_clean(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TokEye's times (ms), notched coherent mask `(4, 512, T)` and UCI annotation."""
    with np.load(path) as z:
        t = np.asarray(z["t_ms"], dtype=np.float64)
        clean = np.unpackbits(z["mask_clean"], axis=-1, count=len(t)).astype(bool)
        ann = np.asarray(z["ann"]).astype(bool)
    return t, apply_notch(clean, notch_bins(bin_active_fraction(clean), NOTCH)), ann


def tokeye_frames(path, first: int, n: int) -> dict[str, np.ndarray]:
    """Per frame: the share of TokEye columns with a low line, an AE-band line and
    the UCI annotation, and whether TokEye's record covers the whole frame."""
    t, clean, ann = tokeye_clean(path)
    low = clean[:, LOW_BINS[0] : LOW_BINS[1]].any(axis=1).sum(axis=0) >= MIN_CHORDS
    ae = clean[:, AE_BINS[0] : AE_BINS[1]].any(axis=1).sum(axis=0) >= MIN_CHORDS
    return {
        "low": frame_share(t, low, first, n),
        "ae": frame_share(t, ae, first, n),
        "ann": frame_share(t, ann, first, n),
        "covered": frame_covered(t, first, n),
    }


def mhd_frames(path, first: int, n: int) -> np.ndarray:
    """Frames TokEye covers where a 0-60 kHz line holds for half the frame."""
    if path is None:
        return np.zeros(n, dtype=bool)
    frames = tokeye_frames(path, first, n)
    return frames["covered"] & (frames["low"] >= MIN_FRACTION)


def seldnet_split(masks_dir) -> dict[int, str]:
    """`train` or `valid` per AE180 shot, from TokEye's file names."""
    found = {}
    for path in Path(masks_dir).glob("*_clean.npz"):
        shot, split = path.name.split("_")[:2]
        found[int(shot)] = split
    return found


def make_split(
    reviewed: Iterable[int],
    seldnet: Mapping[int, str],
    *,
    seed: int = SEED,
    n_val: int = N_VAL,
) -> dict[int, str]:
    """`train`, `val` or `test` per reviewed shot; `test` is SELDNet's `valid`."""
    reviewed = sorted({int(s) for s in reviewed})
    missing = [s for s in reviewed if s not in seldnet]
    if missing:
        raise ValueError(f"no SELDNet split for {len(missing)} shots: {missing[:5]}")
    pool = [s for s in reviewed if seldnet[s] == "train"]
    rng = np.random.default_rng(seed)
    val = {int(s) for s in rng.permutation(pool)[:n_val]}
    return {
        s: "test" if seldnet[s] == "valid" else "val" if s in val else "train"
        for s in reviewed
    }


def load_shot(
    shot: int,
    label: Label,
    rows,
    masks_dir=None,
    *,
    context: int = CONTEXT_FRAMES,
    band=BAND_KHZ,
) -> Shot:
    """One reviewed shot as frames; `rows` is `store_rows` or `raw_rows` output."""
    grid, values, y0, dy = rows
    first, n = window_frames(label.window)
    lo, count = first - context, n + 2 * context
    band = band_slice(y0, dy, values.shape[1], band)
    x, observed = frame_inputs(values[:, band], grid, lo, count)
    path = None if masks_dir is None else clean_path(masks_dir, shot)
    return Shot(
        shot=int(shot),
        first=lo,
        x=x.astype(np.float16),
        states=targets(label, lo, count),
        mhd=mhd_frames(path, lo, count),
        observed=observed,
    )
