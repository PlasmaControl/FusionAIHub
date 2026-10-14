"""Toroidal-n MHD: the magnetics, the n=1..3 amplitude, q_min and q95, beta_N.

One page serves the taxonomy's `mhd_n1`, `mhd_n2` and `mhd_n3`: the Mirnov
spectrogram and toroidal-n map of the tearing editor
(`neoclassical_tearing_mode`), a row that says how loud each n is, where the
rational surfaces q = m/n sit against the q profile, and beta_N.

The n=1..3 amplitude row is, per time column and for each k, the loudest cell
between 1 and 30 kHz whose best-fitting n is k, whose fit (the probes'
phase-coherence score, `mode_numbers(..., with_score=True)`) clears the shot's
gate, and that sits on a coherent line, in dB above the same floor the map uses.
A column with no such cell is drawn at 0 dB. `amplitude_by_n` keeps it NaN,
which the candidate rule (`labeler.events.candidates`, `n_amplitude`) reads.

Six probes cannot tell noise from a mode on the fit alone (the median
n = 1 cell of a phase-scrambled shot sits as high as a real one), so each shot
calibrates itself against a null: the same map with the probes' angles randomly
permuted (`NULL_DRAWS` draws, fixed seed), over the plasma columns. A cell needs a fit
of at least `MIN_FIT` and above the null's `NULL_PERCENTILE`th percentile fit
(the gate); it counts only inside a line that holds for `COHERENT_COLUMNS` columns
within `COHERENT_BINS` bin of a frequency; and the amplitude a mode must reach is
`THRESHOLD_DB` or the null's own percentile amplitude, whichever is larger,
per n. The row draws those thresholds as its lines.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d

from ...config import Paths
from .. import equilibrium
from ..verify import NoDataError, Panel
from . import neoclassical_tearing_mode as ntm
from ._shared import MISSING, above_floor_db, betan_panel, optional, with_caveat

#: The n the amplitude row follows, and what a cell must fit to count for one.
AMPLITUDE_NS = (1, 2, 3)
MIN_FIT = 0.95
#: The band the amplitude is the maximum over, kHz.
BAND_KHZ = (1.0, 30.0)
#: The amplitude that makes a mode present, dB above the floor. The candidate
#: tables apply it for at least `candidates.MIN_MS`.
THRESHOLD_DB = 12.0
#: The null: draws of the probes' angles permuted (each moves them this far on
#: average, degrees, so it is not a near-identity), the seed, and the percentile of
#: its fit and its amplitude a cell must exceed.
NULL_DRAWS = 3
NULL_SEED = 0
NULL_MIN_SHIFT_DEG = 45.0
NULL_PERCENTILE = 99.0
#: A mode is a line: its cells hold on for this many columns, each with a
#: qualifying cell within this many frequency bins of the one before.
COHERENT_COLUMNS = 3
COHERENT_BINS = 1
#: The rational surfaces q = m/n the taxonomy names (1/1, 4/3, 3/2, 2/1, 3/1):
#: `configs/labeler/events.yaml`'s mhd_n1/n2/n3 children. Hard-coded, not read from
#: that file, so this module does not depend on the taxonomy's loader.
Q_LINES = (1.0, 4 / 3, 1.5, 2.0, 3.0)

AMPLITUDE_TITLE = "n=1..3 amplitude"
Q_TITLE = "q_min, q95 (EFIT01)"
TITLES = (*ntm.TITLES[:2], AMPLITUDE_TITLE, Q_TITLE, "beta_N")


def coherent(mask, *, columns=COHERENT_COLUMNS, bins=COHERENT_BINS) -> np.ndarray:
    """The cells of `mask` `(F, T)` that lie on a line: in a run of at least
    `columns` (odd) columns in which every column holds a cell of `mask` within
    `bins` frequency bins. A cheap run-length check, no tracking of the line."""
    mask = np.asarray(mask, dtype=np.uint8)
    near = maximum_filter1d(mask, 2 * bins + 1, axis=0, mode="constant")
    runs = minimum_filter1d(near, columns, axis=1, mode="constant")
    return (maximum_filter1d(runs, columns, axis=1, mode="constant") > 0) & (mask > 0)


def amplitude_by_n(
    db,
    modes,
    score,
    f_khz,
    *,
    ns=AMPLITUDE_NS,
    min_fit=MIN_FIT,
    band=BAND_KHZ,
    line=False,
) -> np.ndarray:
    """`(len(ns), T)` float32: each column's loudest qualifying cell per n, in dB.

    `db`, `modes` and `score` are `(F, T)` (the map's dB above its floor,
    `mode_numbers`' best n and its fit), `f_khz` the F frequencies. A cell
    qualifies for k when its best n is k and its fit is at least `min_fit` and,
    with `line`, it lies on a coherent line (`coherent`); a column with none
    for k is NaN.
    """
    db, modes, score = (np.asarray(a) for a in (db, modes, score))
    inband = (np.asarray(f_khz) >= band[0]) & (np.asarray(f_khz) <= band[1])
    out = np.full((len(ns), db.shape[1]), np.nan, dtype=np.float32)
    if not inband.any():
        return out
    db, modes, score = db[inband], modes[inband], score[inband]
    for i, k in enumerate(ns):
        cells = (modes == k) & (score >= min_fit)
        if line:
            cells = coherent(cells)
        loudest = np.where(cells, db, -np.inf).max(axis=0)
        out[i] = np.where(np.isfinite(loudest), loudest, np.nan)
    return out


def null_angles(phi_deg, *, draws=NULL_DRAWS, seed=NULL_SEED) -> list[np.ndarray]:
    """`draws` random permutations of the probes' angles (fixed `seed`), each
    one moving them `NULL_MIN_SHIFT_DEG` on average, the identity never."""
    phi = np.asarray(phi_deg, dtype=float)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        for _ in range(50):
            moved = phi[rng.permutation(len(phi))]
            shift = np.abs((moved - phi + 180.0) % 360.0 - 180.0).mean()
            if shift >= NULL_MIN_SHIFT_DEG:
                break
        out.append(moved)
    return out


class Amplitude(NamedTuple):
    """The n=1..3 amplitude of one shot, and the null it was calibrated against."""

    db: np.ndarray  # (len(AMPLITUDE_NS), T) float32, NaN where nothing qualifies
    null_db: np.ndarray  # (len(AMPLITUDE_NS),) the null's amplitude percentile, nan if none
    gate: float  # the fit a cell needed: max(MIN_FIT, the null's fit percentile)

    @property
    def threshold_db(self) -> np.ndarray:
        """Per n, the amplitude a mode must reach: `THRESHOLD_DB` or the null's."""
        return np.fmax(THRESHOLD_DB, self.null_db)


def amplitude(s: ntm.Spectra, modes, score) -> Amplitude:
    """The n=1..3 amplitude of the spectra `s` over the n map's own floor (the
    0.2 quantile of its plasma columns), calibrated against a null of the same
    map with the probes' angles permuted (the module docstring)."""
    db = above_floor_db(s.power.mean(axis=0), ntm.FLOOR_QUANTILE, columns=s.plasma)
    inside = (s.f_khz >= BAND_KHZ[0]) & (s.f_khz <= BAND_KHZ[1])
    nulls = [
        ntm.mode_numbers(s.spec, angles, with_score=True)
        for angles in null_angles(list(ntm.PROBES.values()))
    ]
    fits = np.concatenate([sc[inside][:, s.plasma].ravel() for _, sc in nulls])
    gate = MIN_FIT
    if fits.size:
        gate = max(MIN_FIT, float(np.percentile(fits, NULL_PERCENTILE)))
    loud = np.concatenate(
        [
            amplitude_by_n(db, nm, sc, s.f_khz, min_fit=gate, line=True)[:, s.plasma]
            for nm, sc in nulls
        ],
        axis=1,
    )
    null_db = np.full(len(AMPLITUDE_NS), np.nan)
    for i, row in enumerate(loud):
        row = row[np.isfinite(row)]
        if row.size:
            null_db[i] = np.percentile(row, NULL_PERCENTILE)
    return Amplitude(
        amplitude_by_n(db, modes, score, s.f_khz, min_fit=gate, line=True), null_db, gate
    )


def amplitude_panel(s: ntm.Spectra, modes, score) -> Panel:
    """The n=1..3 amplitude row of the spectra `s`, with each n's threshold."""
    got = amplitude(s, modes, score)
    lines = sorted({round(float(v), 1) for v in got.threshold_db})
    return with_caveat(
        Panel(
            title=AMPLITUDE_TITLE,
            x=s.t_ms,
            y=np.nan_to_num(got.db, nan=0.0),
            ylabel="dB",
            legend=[f"n={k}" for k in AMPLITUDE_NS],
            hlines=lines,
        ),
        f"Loudest {BAND_KHZ[0]:g}-{BAND_KHZ[1]:g} kHz cell whose best n is k, whose "
        f"fit is at least {got.gate:.2f} and above a phase-scrambled null's "
        f"{NULL_PERCENTILE:g}th percentile, on a line of {COHERENT_COLUMNS}+ columns, "
        "dB above the n map's floor; 0 where no cell qualifies. The lines are the "
        f"{THRESHOLD_DB:g} dB candidate threshold or the null's amplitude, per n: "
        "null-calibrated per shot",
    )


def magnetics_panels(shot, *, t_range=None, paths=None) -> list[Panel]:
    """MPI66M322D's power, the toroidal-n map and the n=1..3 amplitude."""
    s = ntm.spectra(shot, t_range=t_range, paths=paths)
    modes, score = ntm.mode_numbers(
        s.spec, list(ntm.PROBES.values()), with_score=True
    )
    return [*ntm.heatmaps(s, modes), amplitude_panel(s, modes, score)]


def q_panel(shot, *, t_range=None, paths=None) -> list[Panel]:
    """q_min and q95 on one row, with a line at each rational surface."""
    paths = Paths.from_env() if paths is None else paths
    arrays, missing = {}, []
    for name in ("qmin", "q95"):
        try:
            arrays[name] = equilibrium.plot_arrays(
                equilibrium.signal(int(shot), name, paths, fetch=True)
            )
        except MISSING:
            missing.append(name)
    if not arrays:
        raise NoDataError(f"shot {int(shot)}: no q_min or q95")
    x = next(iter(arrays.values()))[0]
    rows, legend = [], []
    for name, label in (("qmin", "q_min"), ("q95", "q95")):
        if name not in arrays:
            continue
        xs, ys = arrays[name]
        same = np.array_equal(xs, x)
        rows.append(ys[0] if same else np.interp(x, xs, ys[0], np.nan, np.nan))
        legend.append(label)
    y = np.stack(rows).astype(np.float32)
    if t_range is not None:
        keep = (x >= t_range[0]) & (x <= t_range[1])
        x, y = x[keep], y[:, keep]
    panel = Panel(
        title=Q_TITLE,
        x=x,
        y=y,
        ylabel="q",
        legend=legend,
        hlines=list(Q_LINES),
    )
    return [
        with_caveat(
            panel,
            "Lines at q = 1, 4/3, 3/2, 2 and 3, the rational surfaces m/n of the "
            "taxonomy",
            f"{', '.join(missing)} not available" if missing else "",
        )
    ]


def panels(shot, *, t_range=None, paths=None):
    kwargs = {"t_range": t_range, "paths": paths}
    return (
        optional("magnetics", shot, lambda: magnetics_panels(shot, **kwargs))
        + optional("q", shot, lambda: q_panel(shot, **kwargs))
        + optional("beta_N", shot, lambda: betan_panel(shot, **kwargs))
    )
