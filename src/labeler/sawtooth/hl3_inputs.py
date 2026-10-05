"""The HL-3 network's offline input set on DIII-D, assembled without labels.

OuYang et al. (PPCF 67, 105004) classify 20 ms windows of ten 10 kHz channels:
plasma current, line-integrated density, a Mirnov pair, soft X-ray (SXR) core and
edge, stored energy, the electron-temperature perturbation, beam power and ECH
power. This module holds the pure parts of building that set for one shot: the
label-free choice of the two SXR chords, the beam and ECH sums, the clock check
that lets a float32 time axis through, and the stacking onto one native grid. It
reads no files and no labels; the script that calls it does the reading.

SXR chord choice. DIII-D's SXR fans carry no impact parameters in MDSplus or in
imas_composer (probed 2026-10-05), so the core and edge chords come from the data:
the chord with the highest 100 Hz to 2 kHz variance is the core, and among the lit
chords further from the fan centre than it, the chord most anti-correlated with it
in that band is the edge, because a sawtooth inverts across the inversion radius.
Only chords whose band variance is at least twice the white-noise share expected
from the same chord's 3 to 4.5 kHz band (``STRUCTURE_RATIO``) compete: on a noisy
chord the plain variance is the noise, and it picked a structureless chord (chord
15, a broadband-noise chord on many shots) on the first shots tried. Both steps read the whole analysis window of the shot and nothing
else, so no label, no crash time and no q profile enters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

FS = 10000.0
#: Row order of the stacked input. The paper's table lists ten rows for nine
#: channels; here the two beam powers are one total, which makes nine.
CHANNELS = (
    "ece_core_te",
    "mirnov_pair_mean",
    "ip",
    "line_density",
    "sxr_core",
    "sxr_edge",
    "stored_energy",
    "nbi_power",
    "ech_power",
)
#: Divisor applied to each channel when stacking, so every row is of order one: the
#: benchmark stores its training windows as float16 (largest finite value 65504),
#: and a stored energy in joules or a power in watts overflows it to infinity, which
#: the normaliser then reads as a missing channel. Stored energy is in MJ and the
#: two powers in MW; the CO2 chord V2 is divided by 1e14 (its corpus unit gives
#: about 1e14 to 4e14); the SXR chords are already in units of their own noise.
UNIT_SCALE = {
    "ece_core_te": 1.0,
    "mirnov_pair_mean": 1.0,
    "ip": 1.0,
    "line_density": 1e14,
    "sxr_core": 1.0,
    "sxr_edge": 1.0,
    "stored_energy": 1e6,
    "nbi_power": 1e6,
    "ech_power": 1e6,
}
BAND_HZ = (100.0, 2000.0)
#: Above every sawtooth harmonic that matters here; its variance, scaled by the
#: bandwidth ratio, is the white-noise share of the 100 Hz to 2 kHz band.
NOISE_BAND_HZ = (3000.0, 4500.0)
#: A chord carries structure when its 100 Hz to 2 kHz variance is at least this
#: multiple of the white-noise share that same chord's noise band predicts.
STRUCTURE_RATIO = 2.0
#: The 32 chords of a fan are numbered 0..31; the middle of that numbering.
FAN_CENTRE = 15.5
MIN_LIT_CHORDS = 8
LIT_FRACTION = 0.5
MIN_JOINT_SAMPLES = 200
METHOD = (
    "label_free_noise_corrected_highpass_variance_core_and_anticorrelated_outer_edge"
)


def uniform_clock(x):
    """The ideal uniform clock of a native time axis, or ValueError.

    A float32 axis at a few seconds has a resolution near or above the step of a
    several-hundred-kHz digitiser: neighbouring stored times repeat, and a chunk-wise
    check against the first sample or a strict-increase test rejects good records.
    The check here is global: the axis must be finite, never decrease, and stay
    within four float eps of the line through its end points.
    """
    clock = np.asarray(x, dtype=np.float64)
    if len(clock) < 32 or not np.isfinite(clock).all() or (np.diff(clock) < 0).any():
        raise ValueError("native clock must be finite and non-decreasing")
    dt = (clock[-1] - clock[0]) / (len(clock) - 1)
    if dt <= 0:
        raise ValueError("native clock must advance")
    ideal = clock[0] + np.arange(len(clock)) * dt
    eps = np.finfo(np.asarray(x).dtype).eps
    tolerance = max(1e-7, 4 * eps * float(np.max(np.abs(clock))))
    if np.max(np.abs(clock - ideal)) > tolerance:
        raise ValueError("nonuniform waveform")
    return ideal


def interpolate_rows(t, tx, y):
    """Rows of ``y`` on the grid ``t``; NaN outside the source's own extent."""
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    return np.stack([np.interp(t, tx, row, left=np.nan, right=np.nan) for row in y])


def interpolate_gapped(t, tx, y, *, max_gap_s):
    """``y`` on ``t`` from its finite samples; NaN outside them and across gaps.

    Slow signals (EFIT's stored energy, 20 ms) are bridged between neighbouring
    samples, but a hole longer than ``max_gap_s`` stays unknown rather than being
    drawn as a straight line.
    """
    tx, y = np.asarray(tx, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(tx) & np.isfinite(y)
    tx, y = tx[ok], y[ok]
    if len(tx) < 2:
        return np.full(len(t), np.nan)
    out = np.interp(t, tx, y, left=np.nan, right=np.nan)
    right = np.clip(np.searchsorted(tx, t), 1, len(tx) - 1)
    out[(tx[right] - tx[right - 1]) > max_gap_s] = np.nan
    return out


def sum_sources(y):
    """Total power of several sources: positive finite samples summed.

    A NaN or negative sample counts as no power (the repo's ECH rule); a source
    with no finite sample at all is ignored; with no finite source the total is
    unknown (None), so the input is masked rather than zero.
    """
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    live = np.isfinite(y).any(axis=1)
    if not live.any():
        return None
    y = y[live]
    return np.where(np.isfinite(y) & (y > 0), y, 0.0).sum(axis=0)


def _fill(values):
    """Linear fill of NaN gaps (edges held) for filtering, plus the finite mask."""
    finite = np.isfinite(values)
    if finite.all():
        return values, finite
    out = values.copy()
    index = np.arange(len(values))
    out[~finite] = np.interp(index[~finite], index[finite], values[finite])
    return out, finite


def bandpass(y, *, fs=FS, band=BAND_HZ):
    """Zero-phase 100 Hz to 2 kHz band of each row, with its finite mask.

    Gaps are filled by interpolation before filtering and reported through the
    mask, so a variance or correlation never reads a filled sample.
    """
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    sos = butter(2, band, btype="bandpass", fs=fs, output="sos")
    filtered = np.full(y.shape, np.nan)
    masks = np.zeros(y.shape, dtype=bool)
    for i, row in enumerate(y):
        filled, finite = _fill(row)
        if finite.sum() < 3 * int(fs / band[0]):
            continue
        filtered[i] = sosfiltfilt(sos, filled - np.mean(filled[finite]))
        masks[i] = finite
    return filtered, masks


@dataclass(frozen=True)
class SxrPair:
    """The chosen core and edge chord indices (within the fan), or None."""

    core: int | None
    edge: int | None
    lit: tuple[int, ...]
    variance: float | None
    correlation: float | None
    noise_floor: float | None = None
    structured_chords: int | None = None
    structured_core: bool | None = None
    structured_edge: bool | None = None
    method: str = METHOD

    def to_json(self):
        return {
            "core_chord": self.core,
            "edge_chord": self.edge,
            "lit_chords": len(self.lit),
            "chords_with_structure": self.structured_chords,
            "core_has_structure": self.structured_core,
            "edge_has_structure": self.structured_edge,
            "core_highpass_variance": self.variance,
            "core_noise_floor_variance": self.noise_floor,
            "edge_core_correlation": self.correlation,
            "method": self.method,
        }


def select_sxr_chords(y, *, chords=None, fs=FS, centre=FAN_CENTRE):
    """Core and edge chord of one fan from the shot's own signals, no labels.

    ``y`` is ``(rows, samples)`` with NaN for a missing sample; ``chords`` gives
    each row's chord number in the fan (default: the row index) and the answer is
    in those numbers. A chord is lit when at least half its samples are finite;
    fewer than eight lit chords give no pair. A lit chord has structure when its
    100 Hz to 2 kHz variance is at least ``STRUCTURE_RATIO`` times the white-noise
    share its own 3 to 4.5 kHz band predicts. Core: the chord with structure and
    the largest band variance (with none, the chord of the highest ratio, flagged).
    Edge: among chords with structure strictly further from ``centre`` than the
    core (with none, among all such lit chords, flagged), the one with the lowest
    Pearson correlation with the core in the band.
    """
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    numbers = np.arange(len(y)) if chords is None else np.asarray(chords, dtype=int)
    if numbers.shape != (len(y),):
        raise ValueError("one chord number per row")
    rows = np.flatnonzero(np.isfinite(y).mean(axis=1) >= LIT_FRACTION)
    lit = tuple(int(numbers[i]) for i in rows)
    if len(rows) < MIN_LIT_CHORDS:
        return SxrPair(None, None, lit, None, None)
    band, finite = bandpass(y[rows], fs=fs)
    noise, noise_finite = bandpass(y[rows], fs=fs, band=NOISE_BAND_HZ)
    usable = finite.any(axis=1) & noise_finite.any(axis=1)
    if not usable.any():
        return SxrPair(None, None, lit, None, None)
    width = (BAND_HZ[1] - BAND_HZ[0]) / (NOISE_BAND_HZ[1] - NOISE_BAND_HZ[0])
    variance = np.full(len(rows), np.nan)
    floor = np.full(len(rows), np.nan)
    for i in np.flatnonzero(usable):
        variance[i] = np.var(band[i][finite[i]])
        floor[i] = width * np.var(noise[i][noise_finite[i]])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(usable, variance / np.where(floor > 0, floor, np.nan), np.nan)
    ratio = np.where(usable & ~np.isfinite(ratio), np.inf, ratio)
    structured = usable & (ratio >= STRUCTURE_RATIO)
    if structured.any():
        core = int(np.argmax(np.where(structured, variance, -np.inf)))
    else:
        core = int(np.argmax(np.where(usable, ratio, -np.inf)))
    reach = abs(numbers[rows[core]] - centre)
    outer = usable & (np.abs(numbers[rows] - centre) > reach)
    outer[core] = False
    pool = outer & structured if (outer & structured).any() else outer
    best, best_correlation = None, None
    for i in np.flatnonzero(pool):
        joint = finite[core] & finite[i]
        if joint.sum() < MIN_JOINT_SAMPLES:
            continue
        a, b = band[core][joint], band[i][joint]
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        correlation = float(np.corrcoef(a, b)[0, 1])
        if best_correlation is None or correlation < best_correlation:
            best, best_correlation = i, correlation
    return SxrPair(
        int(numbers[rows[core]]),
        None if best is None else int(numbers[rows[best]]),
        lit,
        float(variance[core]),
        best_correlation,
        float(floor[core]),
        int(structured.sum()),
        bool(structured[core]),
        None if best is None else bool(structured[best]),
    )


def robust_standardise(values):
    """Centre one chord on its median and divide by its robust spread (label-free).

    SXR amplifier gain, filtering and brightness differ between shots and chords by
    more than a factor of ten, so the raw level is not comparable across shots; the
    waveform in units of its own noise is. Spread is 1.4826 times the median
    absolute deviation; a flat chord is centred and left unscaled.
    """
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return values
    centre = float(np.median(values[finite]))
    spread = 1.4826 * float(np.median(np.abs(values[finite] - centre)))
    return (values - centre) / spread if spread > 0 else values - centre


def stack_inputs(t, parts):
    """Stack channels onto the native grid ``t`` in `CHANNELS` order.

    ``parts`` maps a channel name to ``None`` (missing on the shot), an array
    already on ``t``, or a ``(tx, values)`` pair on its own clock, in the units the
    reader produced (J, W, the CO2 corpus unit); each row is divided by its
    `UNIT_SCALE`. Returns the ``(9, len(t))`` float32 matrix (NaN where unknown) and
    each channel's finite fraction over the analysis window.
    """
    unknown = set(parts) - set(CHANNELS)
    if unknown:
        raise ValueError(f"unknown input channels: {sorted(unknown)}")
    matrix = np.full((len(CHANNELS), len(t)), np.nan, dtype=np.float32)
    for row, name in enumerate(CHANNELS):
        part = parts.get(name)
        if part is None:
            continue
        if isinstance(part, tuple):
            values = interpolate_rows(
                t, np.asarray(part[0], dtype=np.float64), part[1]
            )[0]
        else:
            values = np.asarray(part, dtype=np.float64)
            if values.shape != (len(t),):
                raise ValueError(f"{name} must be on the native grid")
        matrix[row] = values / UNIT_SCALE[name]
    return matrix, {
        name: float(np.isfinite(matrix[row]).mean())
        for row, name in enumerate(CHANNELS)
    }
