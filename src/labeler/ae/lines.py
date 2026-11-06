"""Fixed-frequency lines in the AE input, and a notch over them.

Pickup lines (a carrier, a power supply, a digitizer clock) sit at one frequency and draw horizontal
lines across the CO2 spectrogram; after the per-shot z-score the network can read them as modes. An
Alfven eigenmode moves with the density and the q profile and is several bins wide, so it rarely holds
one or two bins for long. A line is therefore a run of at most ``MAX_WIDTH`` bins whose level (its
median over frames, ``LEVEL_Q``) stands above the running median of its neighbours (``BASE_BINS``
bins) by more than ``K_MAD`` robust deviations and at least ``MIN_EXCESS``, over the whole shot, in
some chord. A line found that way is a known frequency for every chord of the shot: there it is
tested again at the weaker ``K_CROSS``, over the whole shot and over each block of about 500 ms (a
line that starts or stops mid-shot, or is drowned in plasma noise for a while). A narrow run seen in
one block only, at a frequency no chord holds over the shot, is left alone: that is how a slowly
drifting mode looks. The notch replaces the flagged bins, frame by frame, by the straight line
between the nearest kept bins below and above.

Works on the model input after ``gpu.model_input`` / ``transform.model_input`` (z-scored, band bins
164:512, frames 0.256 ms apart): one chord ``(frames, bins)`` (:func:`line_bins`,
:func:`notch_chord`), a shot ``(frames, chords, bins)`` as ``scripts/labeler/ae_tiers.py`` stores it
(:func:`notch_shot`), or the same shot as a torch tensor on the GPU (:func:`notch_shot_torch`, used by
``scripts/labeler/ae_fdp_label.py infer --notch``: the medians and the notch on the device, the
line tests on the small level vectors on the host; the same masks as :func:`notch_shot`).

Checked 2026-11-06 with the ae_tiers student (dev/ae_notch): gold F1 .857 either way, silver held-out
F1 .913 -> .912, and AE called while the beams had been off 50 ms or more (no fast ions) on 15 shots
an era: 120000s .61 -> .11, 125000s .68 -> .14, 135000s .18 -> .01, 140000s .29 -> .01, 145000 on
unchanged.
"""
from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise

import numpy as np

LEVEL_Q = 0.5  # a line holds its level over at least half of the frames looked at
BASE_BINS = 15  # running median over bins: the floor a line stands on (~7 kHz of 0.49 kHz bins)
K_MAD = 3.5  # robust deviations above the floor, over the whole shot
K_CROSS = 2.0  # the same at a frequency already found as a line in some chord (whole shot or block)
MIN_EXCESS = 0.4  # and at least this, in z-score units
MAX_WIDTH = 4  # bins: a wider run is not a line (a mode, or the floor bending)
DILATE = 1  # bins either side (the Hann window's main lobe)
MAX_SHARE = 0.25  # more than this share of the band flagged is not a set of lines: notch nothing
BLOCK_FRAMES = 1953  # ~500 ms at 0.256 ms a frame


def _flag(level: np.ndarray, k: float) -> np.ndarray:
    from scipy.ndimage import median_filter

    level = np.asarray(level, dtype=np.float32)
    floor = median_filter(level, size=BASE_BINS, mode="nearest")
    excess = level - floor
    mad = 1.4826 * float(np.median(np.abs(excess - np.median(excess))))
    hit = excess > max(k * mad, MIN_EXCESS)
    if hit.any():  # drop runs wider than MAX_WIDTH
        edges = np.flatnonzero(np.diff(np.r_[0, hit.astype(np.int8), 0]))
        for a, b in zip(edges[::2], edges[1::2], strict=True):
            if b - a > MAX_WIDTH:
                hit[a:b] = False
    return hit


def block_edges(n_frames: int, block: int = BLOCK_FRAMES) -> np.ndarray:
    """Frame edges of the blocks: about ``block`` frames each, the last one not short."""
    n = max(1, round(n_frames / block))
    return np.linspace(0, n_frames, n + 1).astype(int)


def _masks(level: np.ndarray, block_level: Callable[[int], np.ndarray], n_blocks: int,
           known: np.ndarray | None) -> np.ndarray:
    """One chord's ``(blocks, bins)`` line bins from its whole-shot level ``(bins,)`` and
    ``block_level(i)``, block i's level (asked for only when the shot has known lines)."""
    from scipy.ndimage import binary_dilation

    whole = _flag(level, K_MAD)
    known = whole if known is None else np.asarray(known, dtype=bool)
    if known.any():  # the neighbour bins too: a line can sit between two bins
        known = binary_dilation(known, iterations=1)
    whole = whole | (_flag(level, K_CROSS) & known)
    out = np.zeros((n_blocks, level.size), dtype=bool)
    for i in range(n_blocks):
        m = whole | (_flag(block_level(i), K_CROSS) & known) if known.any() else whole
        if m.any() and DILATE:
            m = binary_dilation(m, iterations=DILATE)
        out[i] = m if m.mean() <= MAX_SHARE else False
    return out


def _weights(mask: np.ndarray):
    """(bad, left, right, w): each masked bin, its nearest kept bins below and above (the nearest
    one at the band's edges) and its weight on the upper one."""
    good, bad = np.flatnonzero(~mask), np.flatnonzero(mask)
    right_at = np.searchsorted(good, bad)
    left = good[np.clip(right_at - 1, 0, good.size - 1)]
    right = good[np.clip(right_at, 0, good.size - 1)]
    span = right - left
    w = np.where(span > 0, (bad - left) / np.maximum(span, 1), 0.0).astype(np.float32)
    return bad, left, right, w


def whole_lines(chord: np.ndarray) -> np.ndarray:
    """``(frames, bins)`` -> boolean ``(bins,)``: the lines that hold over the whole shot."""
    return _flag(np.quantile(np.asarray(chord, dtype=np.float32), LEVEL_Q, axis=0), K_MAD)


def line_bins(chord: np.ndarray, block: int = BLOCK_FRAMES, known: np.ndarray | None = None) -> np.ndarray:
    """``(frames, bins)`` -> boolean ``(blocks, bins)``: each block's line bins. ``known``: the
    shot's line frequencies (:func:`whole_lines` of every chord, or'ed; this chord's own if None),
    tested here at ``K_CROSS`` over the whole shot and each block."""
    x = np.asarray(chord, dtype=np.float32)
    edges = block_edges(x.shape[0], block)
    return _masks(np.quantile(x, LEVEL_Q, axis=0),
                  lambda i: np.quantile(x[edges[i]:edges[i + 1]], LEVEL_Q, axis=0), edges.size - 1, known)


def notch(chord: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``chord`` ``(frames, bins)`` with the ``mask`` bins replaced, in every frame, by linear
    interpolation between the nearest unmasked bins (the nearest one at the band's edges)."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any() or mask.all():
        return chord
    bad, left, right, w = _weights(mask)
    out = np.array(chord, copy=True)
    src = np.asarray(chord, dtype=np.float32)
    out[:, bad] = ((1 - w) * src[:, left] + w * src[:, right]).astype(out.dtype)
    return out


def notch_chord(chord: np.ndarray, block: int = BLOCK_FRAMES,
                known: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One chord ``(frames, bins)`` notched block by block, and its ``(blocks, bins)`` masks."""
    masks = line_bins(chord, block, known)
    edges = block_edges(chord.shape[0], block)
    out = np.array(chord, copy=True)
    for m, a, b in zip(masks, edges[:-1], edges[1:], strict=True):
        if m.any():
            out[a:b] = notch(chord[a:b], m)
    return out, masks


def notch_shot(spec: np.ndarray, block: int = BLOCK_FRAMES) -> tuple[np.ndarray, list[np.ndarray]]:
    """A shot's input ``(frames, chords, bins)`` with each chord's lines notched (the lines of every
    chord known to all), and each chord's ``(blocks, bins)`` masks."""
    known = np.zeros(spec.shape[2], dtype=bool)
    for c in range(spec.shape[1]):
        known |= whole_lines(spec[:, c, :])
    out = np.array(spec, copy=True)
    masks = []
    for c in range(spec.shape[1]):
        out[:, c, :], m = notch_chord(spec[:, c, :], block, known)
        masks.append(m)
    return out, masks


def _median0(x):
    """torch: the median over dim 0 as ``np.quantile(x, 0.5, axis=0)`` takes it (the mean of the two
    middle values when the count is even)."""
    s = x.sort(dim=0).values
    n = s.shape[0]
    return (s[(n - 1) // 2] + s[n // 2]) / 2


def notch_shot_torch(spec, block: int = BLOCK_FRAMES):
    """:func:`notch_shot` for a torch ``(frames, chords, bins)`` tensor on any device: a notched
    float32 copy on the same device, and each chord's ``(blocks, bins)`` masks."""
    import torch

    x = spec.float()
    n_frames, n_chords, n_bins = x.shape
    level = _median0(x).cpu().numpy()  # (chords, bins)
    known = np.zeros(n_bins, dtype=bool)
    for c in range(n_chords):
        known |= _flag(level[c], K_MAD)
    edges = block_edges(n_frames, block)
    blocks = (np.stack([_median0(x[a:b]).cpu().numpy() for a, b in pairwise(edges)])
              if known.any() else None)  # (blocks, chords, bins)
    out = x.clone()
    masks = []
    for c in range(n_chords):
        m = _masks(level[c], lambda i, c=c: blocks[i, c], edges.size - 1, known)
        for mi, a, b in zip(m, edges[:-1], edges[1:], strict=True):
            if mi.any() and not mi.all():
                bad, left, right, w = (torch.as_tensor(v, device=x.device) for v in _weights(mi))
                src = x[a:b, c]
                out[a:b, c, bad] = (1 - w) * src[:, left] + w * src[:, right]
        masks.append(m)
    return out, masks
