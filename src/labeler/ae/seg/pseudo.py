"""Pseudo-masks: TokEye's coherent mask inside the owner's AE frames, 80-250 kHz.

    python -m labeler.ae.seg.pseudo [--shots S ...]

Drawing an AE mode pixel by pixel is slow, and a mode can cover a large part of
the spectrogram. Two things already say where it is. TokEye's cleaned coherent
mask (`$LABELER_ROOT/ae/masks/<shot>_<split>_clean.npz`, 0-2 s, four chords)
lights every coherent line, AE or not. The owner's saved frames say when AE is
present. Where both agree, inside 80-250 kHz, the pixel is very likely the mode.

**Grid.** The review store's level 8: 2.048 ms columns, the page's 257 bins of
0.977 kHz to 250 kHz. A TokEye column falls in the store column that holds its
time. A page bin takes TokEye's two 0.488 kHz bins whose centres lie in it. A
pixel is lit where TokEye lights at least two of the four chords (the rule
`labeler.ae.xpower.data` uses for MHD frames), in any of its TokEye columns.

**Values.** 1 is AE, 0 is background, 255 is ignored by training and scoring:
- in a frame the owner calls absent, every 80-250 kHz pixel is 0, TokEye's
  lines included: those are the coherent modes that are not AE (the hard
  negatives, an MHD mode's harmonics among them);
- in a frame the owner calls present, a lit 80-250 kHz pixel is 1 and an unlit
  one 0, except a ring `RING_BINS` bins and `RING_COLS` columns wide around the
  lit pixels, which is ignored (a mode's faint edge);
- a present column with no lit pixel is ignored: TokEye missed the mode there,
  and 0 would teach the network that there is none;
- a lit region of fewer than `MIN_AREA` pixels (8-connected) is ignored;
- everything else is ignored: uncertain, not observable, outside the window,
  outside TokEye's 0-2 s, below 80 kHz.

The reviewer can reject a region on the review page (`regions`); training then
takes that region as background.

**Output.** `$LABELER_ROOT/segmentation/alfven_eigenmode/pseudo-v1/<shot>.npz`
(`mask`, `t0_ms`, `dt_ms`, `y0_khz`, `dy_khz`) and `index.csv` beside them.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy import ndimage

from ...config import Paths, atomic_path, git_sha, sha256_of
from ...events.catalog.states import ABSENT, PRESENT
from ...events.review import labels
from ...scoring.frames import FRAME_MS, OUTSIDE
from ..xpower import event_dir, tokeye_masks
from ..xpower.data import (
    BAND_KHZ,
    MIN_CHORDS,
    band_slice,
    clean_path,
    store_rows,
    targets,
    tokeye_clean,
    window_frames,
)
from . import EVENT, PSEUDO, pseudo_dir

LEVEL = 8
IGNORE = 255
RING_BINS = 2
RING_COLS = 1
MIN_AREA = 6
EIGHT = np.ones((3, 3), dtype=bool)
INDEX_COLUMNS = (
    "shot",
    "file",
    "ae_px",
    "background_px",
    "ignored_px",
    "regions",
    "ae_cols",
    "present_cols_unlit",
)


@dataclass(frozen=True)
class PseudoMask:
    """One shot's mask: columns `t0_ms + k * dt_ms`, bins `y0_khz + j * dy_khz`."""

    shot: int
    t0_ms: float
    dt_ms: float
    y0_khz: float
    dy_khz: float
    mask: np.ndarray  # (n_y, n) uint8: 0, 1 or IGNORE
    #: Columns the owner calls present where TokEye lit nothing in the band.
    present_unlit: int = 0

    def save(self, path) -> None:
        with atomic_path(Path(path)) as tmp, open(tmp, "wb") as f:
            np.savez_compressed(
                f,
                shot=np.int64(self.shot),
                t0_ms=self.t0_ms,
                dt_ms=self.dt_ms,
                y0_khz=self.y0_khz,
                dy_khz=self.dy_khz,
                mask=self.mask,
                present_unlit=np.int64(self.present_unlit),
            )

    @classmethod
    def load(cls, path) -> PseudoMask:
        with np.load(path) as z:
            return cls(
                int(z["shot"]),
                float(z["t0_ms"]),
                float(z["dt_ms"]),
                float(z["y0_khz"]),
                float(z["dy_khz"]),
                z["mask"],
                int(z["present_unlit"]),
            )


def tokeye_rows(clean: np.ndarray, min_chords: int = MIN_CHORDS) -> np.ndarray:
    """TokEye's `(chords, 512, T)` mask on the page's 257 bins, `(257, T)`.

    TokEye bin b is centred on (b + 1) x 0.488 kHz (its DC bin is dropped) and
    page bin k on k x 0.977 kHz, so page bin k takes TokEye bins 2k - 2 and 2k - 1.
    """
    lit = np.asarray(clean).sum(axis=0) >= min_chords
    padded = np.zeros((514, lit.shape[1]), dtype=bool)
    padded[2:] = lit
    return padded.reshape(257, 2, -1).any(axis=1)


def pool_columns(lit: np.ndarray, t_ms, grid) -> np.ndarray:
    """`(n_y, T)` flags at times `t_ms` onto the grid's columns, OR within each."""
    t = np.asarray(t_ms, dtype=np.float64)
    cols = np.floor((t - grid.t0_ms) / grid.dt_ms).astype(np.int64)
    keep = np.flatnonzero((cols >= 0) & (cols < grid.n))
    out = np.zeros((lit.shape[0], grid.n), dtype=bool)
    if keep.size:
        which = cols[keep]
        starts = np.flatnonzero(np.r_[True, np.diff(which) > 0])
        out[:, which[starts]] = np.logical_or.reduceat(lit[:, keep], starts, axis=1)
    return out


def covered_columns(t_ms, grid) -> np.ndarray:
    """The grid columns whose whole span lies inside TokEye's record."""
    t = np.asarray(t_ms, dtype=np.float64)
    half = (t[1] - t[0]) / 2 if len(t) > 1 else 0.0
    starts = grid.t0_ms + np.arange(grid.n) * grid.dt_ms
    return (starts >= t[0] - half) & (starts + grid.dt_ms <= t[-1] + half)


def column_states(label: labels.Label, grid) -> np.ndarray:
    """The owner's state of the frame holding each column's centre; OUTSIDE if none."""
    first, n = window_frames(label.window)
    states = targets(label, first, n)
    centres = grid.t0_ms + (np.arange(grid.n) + 0.5) * grid.dt_ms
    k = np.floor(centres / FRAME_MS).astype(np.int64) - first
    out = np.full(grid.n, OUTSIDE, dtype=np.int64)
    inside = (k >= 0) & (k < n)
    out[inside] = states[k[inside]]
    return out


def build(
    shot: int, label: labels.Label, grid, n_y: int, y0: float, dy: float, tokeye
) -> PseudoMask:
    """The pseudo-mask on `grid`; `tokeye` is `tokeye_clean`'s `(t_ms, clean, ann)`."""
    t_ms, clean, _ = tokeye
    band = band_slice(y0, dy, n_y, BAND_KHZ)
    in_band = np.zeros(n_y, dtype=bool)
    in_band[band] = True
    lit = pool_columns(tokeye_rows(clean), t_ms, grid)[:n_y]
    state = column_states(label, grid)
    state[~covered_columns(t_ms, grid)] = OUTSIDE
    present, absent = state == PRESENT, state == ABSENT
    positive = lit & in_band[:, None] & present[None, :]
    regions, count = ndimage.label(positive, structure=EIGHT)
    sizes = np.bincount(regions.ravel(), minlength=count + 1)
    small = (sizes < MIN_AREA)[regions] & positive
    positive &= ~small
    ring = ndimage.binary_dilation(
        positive, structure=np.ones((2 * RING_BINS + 1, 2 * RING_COLS + 1), bool)
    )
    mask = np.full((n_y, grid.n), IGNORE, dtype=np.uint8)
    scored = in_band[:, None] & (absent | present)[None, :]
    mask[scored] = 0
    mask[ring & ~positive & present[None, :] & in_band[:, None]] = IGNORE
    mask[small] = IGNORE
    unlit = present & ~positive.any(axis=0)
    mask[:, unlit] = IGNORE
    mask[positive] = 1
    return PseudoMask(
        int(shot), grid.t0_ms, grid.dt_ms, float(y0), float(dy), mask, int(unlit.sum())
    )


def summary(pm: PseudoMask, file: str) -> dict:
    mask = pm.mask
    _, count = ndimage.label(mask == 1, structure=EIGHT)
    positive_cols = (mask == 1).any(axis=0)
    return {
        "shot": pm.shot,
        "file": file,
        "ae_px": int((mask == 1).sum()),
        "background_px": int((mask == 0).sum()),
        "ignored_px": int((mask == IGNORE).sum()),
        "regions": int(count),
        "ae_cols": int(positive_cols.sum()),
        "present_cols_unlit": pm.present_unlit,
    }


def make(paths: Paths, shot: int, label: labels.Label) -> PseudoMask:
    grid, values, y0, dy = store_rows(paths.spectrogram_file(EVENT, shot), LEVEL)
    tokeye = clean_path(tokeye_masks(paths), shot)
    if tokeye is None:
        raise FileNotFoundError(f"{shot} has no TokEye mask")
    return build(shot, label, grid, values.shape[1], y0, dy, tokeye_clean(tokeye))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--shots", type=int, nargs="*", help="default: every saved shot")
    args = p.parse_args(argv)
    paths = Paths.from_env()
    directory = event_dir(paths)
    saved = labels.read_saved(directory)
    shots = args.shots or sorted(saved)
    out = pseudo_dir(paths)
    out.mkdir(parents=True, exist_ok=True)
    rows, failed = [], []
    for shot in shots:
        try:
            pm = make(paths, shot, saved[shot])
        except (KeyError, OSError, ValueError) as error:
            failed.append(shot)
            print(f"{shot}: {type(error).__name__}: {error}", flush=True)
            continue
        pm.save(out / f"{shot}.npz")
        rows.append(summary(pm, f"{shot}.npz"))
    with atomic_path(out / "index.csv") as tmp, open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    labels_file = labels.labels_path(directory)
    meta = {
        "pseudo": PSEUDO,
        "shots": len(rows),
        "failed": failed,
        "labels": str(labels_file),
        "labels_sha256": sha256_of(labels_file) if labels_file.is_file() else None,
        "git_sha": git_sha(),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with atomic_path(out / "meta.json") as tmp:
        tmp.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {len(rows)} pseudo-masks to {out}; {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
