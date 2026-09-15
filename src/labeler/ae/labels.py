"""Turn tokeye's coherent/transient mask into an AE activity label.

The rules are fixed by the task 7a brief and by spec section 5.4; none of them
is fitted:

1. **mode mask** - ``coherent >= 0.2 and not transient >= 0.2``. Channel 0 of
   ``big_tf_unet`` is coherent activity and channel 1 is transient activity
   (``tokeye/README.md:205``); a pixel the model also calls transient is an
   ELM-like streak, not a coherent mode.
2. **persistence** - a coherent mode lasts, a streak does not, so the mask is
   opened along TIME only with a run length of ``PERSIST_FRAMES = 20`` frames
   (5.1 ms at 0.256 ms/frame). Task 6 measured why this is needed: broadband
   vertical streaks lifted the band occupancy to 0.05-0.3 for hundreds of ms.
3. **notch** - a frequency bin lit in more than a threshold fraction of the
   record is receiver pickup, not a mode; the whole row is zeroed before the
   activity decision and before the centroid, per (channel, bin).
4. **band** - only bins 164-511 (80.57-250.00 kHz on this grid) count towards
   the activity decision and the centroid.

Pure numpy: no scipy, no torch, no I/O. ``persist_open`` is a run-length
rewrite of ``scipy.ndimage.binary_opening(mask, structure=np.ones((1, frames)))``
and ``tests/labeler/test_ae_labels.py`` pins the two together on random
input; the rewrite is used because the dataset script opens 180 x 4 arrays of
(512, 7820).
"""
from __future__ import annotations

import numpy as np

#: sigmoid threshold on both mask channels (``step_1_make_semantic.py``).
PROB_THRESHOLD = 0.2
#: minimum run of consecutive frames a coherent mode must occupy in one bin.
PERSIST_FRAMES = 20
#: STFT size and hop of the shared 7820-frame grid (spec section 5.3).
N_FFT = 1024
HOP = 128
#: number of frequency bins once the DC bin is dropped.
N_BINS = 512
#: AE band, in bins: 164 is 80.57 kHz, 511 is 250.00 kHz (exclusive upper edge).
BAND_LO_BIN = 164
BAND_HI_BIN = 512
#: bins either side of an annotated window's centroid that the notch must not take.
NOTCH_PROTECT_HALF_WIDTH = 3
#: a protected BIN removed on more than this many shots fails a notch threshold.
NOTCH_PROTECT_MAX_SHOTS = 2


def clean_mask(coherent, transient, *, threshold: float = PROB_THRESHOLD):
    """Coherent-mode mask: lit where coherent fires and transient does not.

    `coherent` and `transient` are sigmoid probabilities of the same shape.
    Both comparisons are inclusive (``>= threshold``), as in task 6.
    """
    coherent = np.asarray(coherent)
    transient = np.asarray(transient)
    if coherent.shape != transient.shape:
        raise ValueError(
            f'channel shapes differ: {coherent.shape} vs {transient.shape}'
        )
    return (coherent >= threshold) & ~(transient >= threshold)


def persist_open(mask, frames: int = PERSIST_FRAMES):
    """Binary opening along the last (time) axis with a run of `frames`.

    Equivalent to ``scipy.ndimage.binary_opening`` with a ``(1, frames)``
    structure and its default zero border: a pixel survives only if it sits in
    a run of at least `frames` consecutive lit frames within its own bin. The
    time axis is last; every leading axis (bins, and optionally channels) is
    independent.
    """
    mask = np.asarray(mask, dtype=bool)
    if frames <= 1:
        return mask.copy()
    if mask.ndim < 1:
        raise ValueError('mask needs at least a time axis')
    n_time = mask.shape[-1]
    flat = mask.reshape(-1, n_time)
    # Pad each row with a dark frame so runs never join across rows once
    # flattened; then mark the runs that are long enough with a +1/-1 cumsum.
    padded = np.zeros((flat.shape[0], n_time + 2), dtype=np.int8)
    padded[:, 1:-1] = flat
    edges = np.diff(padded.ravel())
    starts = np.flatnonzero(edges == 1) + 1
    stops = np.flatnonzero(edges == -1) + 1
    keep = (stops - starts) >= frames
    marks = np.zeros(padded.size + 1, dtype=np.int32)
    np.add.at(marks, starts[keep], 1)
    np.add.at(marks, stops[keep], -1)
    opened = np.cumsum(marks[:-1]) > 0
    return opened.reshape(padded.shape)[:, 1:-1].reshape(mask.shape)


def band_occupancy(mask, lo_bin: int = BAND_LO_BIN, hi_bin: int = BAND_HI_BIN):
    """Fraction of the band's bins lit in each frame.

    `mask` is ``(..., bins, frames)``; the returned array is ``(..., frames)``
    float64. `hi_bin` is exclusive.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim < 2:
        raise ValueError('mask needs a bin axis and a time axis')
    return mask[..., lo_bin:hi_bin, :].mean(axis=-2, dtype=np.float64)


def bin_active_fraction(mask):
    """Fraction of frames each bin is lit in, ``(..., bins)`` float64."""
    return np.asarray(mask, dtype=bool).mean(axis=-1, dtype=np.float64)


def notch_bins(occupancy, threshold: float):
    """Bins to notch: lit in strictly more than `threshold` of the frames.

    `occupancy` is the per-bin lit fraction from `bin_active_fraction`, so
    ``(..., bins)``; the result has the same shape. Strict ``>`` matches task 6.
    """
    return np.asarray(occupancy, dtype=np.float64) > threshold


def apply_notch(mask, notched):
    """Copy of `mask` with every notched ``(..., bin)`` row zeroed."""
    mask = np.asarray(mask, dtype=bool)
    notched = np.asarray(notched, dtype=bool)
    return mask & ~notched[..., None]


def protected_removal_counts(notched_per_shot, protected_per_shot, n_bins=N_BINS):
    """How many SHOTS each protected bin is notched away on, per bin.

    The notch rule is stated per BIN, not per shot: "no bin inside an annotated
    window's centroid +-3 bins is removed on more than 2 shots". Counting shots
    that lose any protected bin is a different (and much stricter) quantity, so
    the count is accumulated bin by bin here and read by
    `notch_threshold_passes`.

    `notched_per_shot` is one boolean array per shot, either ``(bins,)`` or
    ``(..., bins)`` (channels are OR-ed: a bin removed in any channel is
    removed on that shot). `protected_per_shot` is the matching ``(bins,)``
    protected mask for the same shot. The result is ``(bins,)`` int64.
    """
    counts = np.zeros(int(n_bins), dtype=np.int64)
    for notched, protected in zip(notched_per_shot, protected_per_shot, strict=True):
        notched = np.asarray(notched, dtype=bool)
        if notched.ndim > 1:
            notched = notched.any(axis=tuple(range(notched.ndim - 1)))
        protected = np.asarray(protected, dtype=bool)
        if notched.shape != counts.shape or protected.shape != counts.shape:
            raise ValueError(
                f'expected ({n_bins},) per shot, got notched {notched.shape} '
                f'and protected {protected.shape}'
            )
        counts += notched & protected
    return counts


def notch_threshold_passes(counts, max_shots: int = NOTCH_PROTECT_MAX_SHOTS) -> bool:
    """True when NO protected bin is removed on more than `max_shots` shots.

    `counts` is `protected_removal_counts`' per-bin shot count. One bin over
    the limit fails the threshold, however many other bins are clean; a bin
    removed on exactly `max_shots` shots is still allowed ("more than").
    """
    counts = np.asarray(counts)
    return not bool(np.any(counts > max_shots))


def power_weights(log_spec):
    """Linear power ``|STFT|**2`` from the transform's ``log1p(|STFT|)`` values.

    ``tokeye.transforms.compute_stft`` returns ``clip(log1p(|STFT|), p1, p99)``,
    so ``expm1`` inverts the log and recovers the (percentile-clipped)
    amplitude, and squaring it gives power. This matters: the log values are
    nearly constant across a shot - measured 24.2-31.9 with a coefficient of
    variation of 0.034-0.039 for the tokeye transform on the 180 AE shots, and
    46-63 / CV 0.026 for the aemodes log-power tif - so weighting a centroid by
    them is very nearly weighting every masked pixel equally. Computed in
    float64: ``expm1(63)**2`` is 1e54, far outside float32. The percentile
    clip means the top 1 % of pixels all carry the same weight; that ceiling is
    part of the transform, not of this function.
    """
    amplitude = np.expm1(np.asarray(log_spec, dtype=np.float64))
    return amplitude * amplitude


def centroid_khz(spec, mask, bin_khz):
    """Weighted centroid frequency of the masked pixels, per frame.

    `spec` and `mask` are ``(..., bins, frames)`` over the SAME bins as
    `bin_khz` (the caller restricts to the band first); every leading axis is
    pooled, so the four CO2 channels contribute to one centroid. `spec` carries
    the weights: pass `power_weights(raw_transform)` for the linear-power
    centroid the AE label uses, never the raw log values. Frames with no masked
    pixel - or with zero total weight - are NaN, because the centroid of an
    empty mask is not a number.
    """
    spec = np.asarray(spec, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    bin_khz = np.asarray(bin_khz, dtype=np.float64)
    if spec.shape != mask.shape:
        raise ValueError(f'spec {spec.shape} and mask {mask.shape} differ')
    if bin_khz.shape != (spec.shape[-2],):
        raise ValueError(f'bin_khz {bin_khz.shape} does not match {spec.shape}')
    weight = spec * mask
    axes = tuple(range(spec.ndim - 1))
    num = (weight * bin_khz[:, None]).sum(axis=axes)
    den = weight.sum(axis=axes)
    good = den > 0
    out = np.full(num.shape, np.nan, dtype=np.float64)
    np.divide(num, den, out=out, where=good)
    return out


def bin_freqs_khz(fs_khz: float, n_bins: int = N_BINS):
    """Centre frequency of each bin, in kHz, once the DC bin has been dropped."""
    return (np.arange(n_bins, dtype=np.float64) + 1.0) * fs_khz / N_FFT


def contiguous_runs(flags) -> list[tuple[int, int]]:
    """Half-open ``(start, stop)`` spans of the True runs in a 1-D flag array."""
    flags = np.asarray(flags, dtype=bool)
    if flags.ndim != 1:
        raise ValueError('contiguous_runs takes a 1-D array')
    padded = np.concatenate(([False], flags, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [(int(a), int(b)) for a, b in zip(starts, stops)]
