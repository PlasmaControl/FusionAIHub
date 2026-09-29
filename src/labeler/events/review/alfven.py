"""The Heidbrink AE rows: CO2 cross-power of R0 against each other chord.

The CO2 interferometer's four chords (R0, V1, V2, V3) are resampled to
500 kHz and given a Hann STFT of 512 samples every 128: 0.256 ms columns,
257 bins 0.977 kHz apart up to 250 kHz. Each frequency bin is flattened by
its median over 0-6 s, so the colour is dB above that bin's own background
and one scale (-3 to 27 dB) serves every shot. Each row averages
R0 x conj(chord) over 8 columns before the magnitude: a coherent mode adds
up, incoherent noise cancels. The chords' own power rows are not kept; noise
on one chord is what the cross-power removes.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

from ...config import Paths
from .. import raw
from .rows import Grid, ImageRow

RATE_HZ = 500_000
NPERSEG = 512
HOP = 128
CROSS_COLUMNS = 8
QUIET_MS = (0.0, 6000.0)
Z_DB = (-3.0, 27.0)
BAND_KHZ = (80.0, 250.0)
CHORDS = ("R0", "V1", "V2", "V3")
# Stores built before 2026-09-23 also hold each chord's own power, named by chord.
DROPPED = frozenset(CHORDS)
PARAMS = {
    "rate_hz": RATE_HZ,
    "nperseg": NPERSEG,
    "hop": HOP,
    "cross_columns": CROSS_COLUMNS,
    "quiet_ms": list(QUIET_MS),
    "z_db": list(Z_DB),
    "band_khz": list(BAND_KHZ),
    "chords": list(CHORDS),
}


def resample(time_ms, chords) -> tuple[np.ndarray, float]:
    """`chords` (C, n) at the rate their `time_ms` span implies, resampled to
    RATE_HZ by `signal.resample_poly` with the ratio limited to denominator 100;
    the float32 samples (C, m) and their exact rate in Hz. Exactly the lines
    `spectrogram_rows` held, which now calls this.

    Its own function because `labeler.ae.full` resamples TokEye's whole-shot
    input with it: the model then sees CO2 at the rate the review rows drew.
    """
    time_ms = np.asarray(time_ms, dtype=np.float64)
    # The rate from the span, not a median step: float32 time vectors quantise.
    rate = (len(time_ms) - 1) / ((time_ms[-1] - time_ms[0]) / 1000)
    ratio = Fraction(RATE_HZ / rate).limit_denominator(100)
    x = signal.resample_poly(
        np.asarray(chords, dtype=np.float32),
        ratio.numerator,
        ratio.denominator,
        axis=-1,
    )
    return x, rate * ratio.numerator / ratio.denominator


def spectrogram_rows(time_ms, chords) -> tuple[Grid, list[ImageRow]]:
    """Three cross-power image rows, R0 against V1, V2 and V3, on one grid."""
    time_ms = np.asarray(time_ms, dtype=np.float64)
    x, fs = resample(time_ms, chords)
    freq, t_s, spec = signal.stft(
        x, fs=fs, window="hann", nperseg=NPERSEG, noverlap=NPERSEG - HOP,
        boundary=None, padded=False, axis=-1,
    )
    dt_ms = HOP / fs * 1000
    centres = time_ms[0] + t_s * 1000
    quiet = (centres >= QUIET_MS[0]) & (centres <= QUIET_MS[1])
    if not quiet.any():
        quiet[:] = True
    cross = []
    for k in range(1, len(CHORDS)):
        product = spec[0] * np.conj(spec[k])
        mean = uniform_filter1d(product.real, CROSS_COLUMNS, axis=-1) + 1j * (
            uniform_filter1d(product.imag, CROSS_COLUMNS, axis=-1)
        )
        cross.append(10 * np.log10(np.abs(mean) + 1e-30))
    names = [f"R0x{c}" for c in CHORDS[1:]]
    titles = [f"R0 × {c}" for c in CHORDS[1:]]
    grid = Grid(float(centres[0] - dt_ms / 2), float(dt_ms), len(centres))
    rows = [
        ImageRow(
            name, title, _quantise(db, quiet),
            y0=float(freq[0] / 1000), dy=float((freq[1] - freq[0]) / 1000),
            y_units="kHz", z_lo=Z_DB[0], z_hi=Z_DB[1], z_units="dB", band=BAND_KHZ,
        )
        for name, title, db in zip(names, titles, cross)
    ]
    return grid, rows


def _quantise(db: np.ndarray, quiet: np.ndarray) -> np.ndarray:
    """dB above each bin's quiet-time median, onto 0-255 over `Z_DB`."""
    flat = db - np.median(db[:, quiet], axis=1, keepdims=True)
    scaled = np.rint((flat - Z_DB[0]) * 255 / (Z_DB[1] - Z_DB[0]))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def build(event: str, shot: int, paths: Paths) -> tuple[Grid, list[ImageRow], dict]:
    co2 = raw.raw_signal(int(shot), "co2", paths=paths)
    tier = co2.attrs["tier"]
    if tier == "corpus":
        path = paths.corpus_file(shot)
    else:
        path = raw.cache_path(shot, paths=paths)
    stat = path.stat()
    grid, rows = spectrogram_rows(co2.x, co2.y)
    source = {"tier": tier, "path": str(path), "size": stat.st_size,
              "mtime_ns": stat.st_mtime_ns}
    return grid, rows, {"params": PARAMS, "source": source}
