"""What the mask never sees: sawteeth, L->H, actuator state, the QH proxy.

`tracks.py` and `transients.py` read a U-Net's opinion of a spectrogram.
Four phenomena this pipeline needs are not in that picture at all, and each
is here for its own reason.

**A sawtooth crash is a RADIAL fact, not a spectral one.** It is a
synchronised fast step across the ECE array whose sign INVERTS along the
channel index - the core loses Te, the plasma outside the inversion radius
gains it - and no single channel's spectrogram can tell it from an ELM,
because on one channel the two look the same. `sawtooth_events` is
`omnimode.mrms.ece` ported: `inversion_block` and `has_inversion` verbatim
(their thresholds were calibrated on shot 193273 and this module has no
better ones), and the crash search RESTRUCTURED. The reference computes the
1 ms envelope inside `crash_times` and then again inside `crash_steps` for
every candidate, which on shot 198658's 429 candidates is 430 passes over a
594 MB record - about three and a half minutes. Here the envelope is
computed ONCE and every candidate reads it: 0.4 s for the same answer. The
mapping-free design is the reference's too, and is the point - the corpus
carries no ECE frequency-to-radius conversion, and the inversion test needs
none.

The acceptance for that port is **47 +/- 3 crashes with a 69 +/- 5 ms
median period on shot 198658, in under a second**. Those are the
REFERENCE's own numbers, checked crash-for-crash by
`scripts/labeler/sawtooth_reference_check.py` (its 198658 output is
committed at `tests/labeler/data/sawtooth_198658_reference.json`); the
plan's "45 sawteeth, 76 ms" was a different measurement of the same shot
and is not what this detector - or the reference it is a port of -
produces. The median is not a stable statistic here in any case: the
inter-crash intervals on 198658 run 11 ms to 897 ms.

**An L->H transition is a coincidence of three signals.** The D-alpha drop
alone is a gas event, the density rise alone is a fuelling change, and NBI
alone is a beam. `lh_transitions` requires all three within their own
windows AND requires the drop to hold - a type-I ELM's D-alpha burst is
milliseconds wide, so the level really does step down after every ELM, and
what an ELM does not do is still be down 20 to 50 ms later. It reports the
drop it measured and the fraction the level held at, so a consumer can weigh
a marginal one. It reads filterscope channels 0-7, which are the ones that carry real
D-alpha at 10 kHz; channels 8 and up are NaN on every shot (plan V4) and are
the caller's to drop - passing them raises rather than quietly taking a
median of NaN.

**Actuator state is a hysteresis question.** "The beams are on" is not
`pinj > threshold` sample by sample: a beam notching down for 5 ms has not
turned off, and a threshold crossed by noise turns on and off ten times a
second. `actuator_intervals` is a 2:1 Schmitt trigger - on at the threshold,
off at half of it - with short gaps bridged and short intervals dropped, and
it takes CANONICAL features (`features/namespace.py` names and units) that
the caller has already resolved. Nothing here opens a file.

**The QH proxy is definitional, and says so.** QH-mode is, in this
pipeline's terms, an EHO in an ELM-free NBI-heated flat-top - so a "QH
detector" built from those four things detects nothing that was not already
assumed. `qh_candidates` emits it anyway, because the association is useful
to rank on, with `attrs["note"]` saying what it is: a consumer must not read
these rows as independent evidence for the EHO/QH association (Appendix C
item 8), and no classifier may be trained to predict one from the others.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import medfilt

from .coverage import (
    UNKNOWN,
    clip_point_to_coverage,
    clip_to_coverage,
    clipped_attrs,
    finite_span,
    intersect,
)
from .schema import Event
from .tracks import Track

# ---------------------------------------------------------------- constants

#: Envelope bin, in ms. The reference's: a crash anchor needs times to the
#: millisecond, not the ~100 us the crash physics happens on.
ENV_MS = 1.0
#: Fractional envelope drop, in one bin, that counts as a step.
DROP_FRAC = 0.02
#: Channels that have to step together before it is a crash candidate.
MIN_CHANNELS = 2
#: How close two crashes may be. Candidates are thinned to this, worst first.
MIN_INTERVAL_MS = 10.0
#: Exclusion zone either side of the crash when the step is measured, and
#: how far out the before/after means reach.
STEP_GAP_MS = 2.0
STEP_SPAN_MS = 8.0
#: Classification thresholds, relative to the largest |step| in the profile.
#: `REL_POS` is the heat-pulse test: measured on 193273 the true sawtooth's
#: rise is 126% of its drop, the ELM impostor's mid-array wiggle 21%.
REL_NEG = 0.25
REL_POS = 0.3
#: Channels a block needs; single-channel spikes are dead-channel artefacts.
MIN_BLOCK = 3
#: Neutral channels allowed between the dropping block and the rising one.
MAX_GAP = 2
#: The DIII-D ECE array. Fixed, because the inversion test is a statement
#: about a radial ordering and a different array is a different statement.
N_ECE_CHANNELS = 48

#: Filterscope channels that carry real D-alpha (plan V4); 8 and up are NaN
#: on every shot in the corpus.
N_DALPHA_CHANNELS = 8
#: Fractional D-alpha drop, within `LH_DROP_WINDOW_MS`, that is a transition.
LH_DROP_FRAC = 0.30
LH_DROP_WINDOW_MS = 5.0
#: Fractional density rise required over the hold window below - `[t + 20
#: ms, t + 50 ms]` - rather than at the moment of the drop. The window is
#: longer than the drop's, and later, because the pedestal builds over tens
#: of milliseconds while the recycling light falls in one.
LH_NE_RISE_FRAC = 0.05
#: NBI power the transition needs, in kW - the canonical `pinj_total` unit.
LH_MIN_PINJ_KW = 500.0
#: The drop a full-confidence transition makes. 60% is what a clean L->H
#: does on a filterscope; a 30% one is half as believable and says so.
LH_FULL_DROP = 0.6
#: The window, in ms either side of the step, over which the transition has
#: to HOLD - and the fraction of the pre-transition level the post window's
#: median has to stay under. This is what separates a transition from an
#: ELM: a type-I ELM's D-alpha burst is 2-5 ms wide, most of the 5 ms drop
#: window, so the LEVEL genuinely steps up and back down at every ELM and
#: the drop gate alone fires once per ELM (measured on the synthetic shot:
#: 4 ms bursts every 15 ms give a candidate per ELM, eleven of which pass
#: the density gate - four `lh_transition` and seven `hl_transition` - in
#: 0.8 s). An ELM's level is back where it was 20 ms later; a transition's
#: is not.
LH_HOLD_LO_MS = 20.0
LH_HOLD_HI_MS = 50.0
#: 0.70 (and the H->L's `1 / 0.70`) is a CALIBRATION, not a fit: it is what
#: separated the transitions from the ELMs on a ten-shot look, and nobody
#: has swept it against labelled transitions. Treat it as a knob with one
#: measurement behind it.
#:
#: What the gate ASSUMES is that the inter-ELM baseline is stationary over
#: the +/-50 ms it compares, and a shot with a gas puff or a density ramp
#: under it is not: a baseline falling by ~30% per 70 ms is lower after
#: every ELM than it was before, so every ELM passes the hold and the gate
#: becomes an ELM counter again. Measured on the synthetic shot: a half-
#: life of 100 ms or less under 4 ms bursts every 15 ms gives four false
#: `lh_transition`s in 0.8 s, one per ELM that also clears the density
#: gate; at 150 ms and 300 ms it gives none. `test_matrix_a_*` pins that.
#:
#: Comparing the post window with the pre window's own TREND instead of its
#: median cancels the drift out of the ratio, and was tried twice - a
#: linear extrapolation of two half-window medians, then a geometric one,
#: both bounded by a drift-like test and a floor. Both were WITHDRAWN. The
#: pre window of a transition whose fall begins before the detected edge is
#: not a baseline at all: extrapolating it predicts the rest of the fall,
#: so the transition explains itself away and the detector reports nothing.
#: Measured, on a 45% step a third of which is taken in a linear lead-in
#: before the edge: the linear trend loses it at lead-ins of 40, 50 and 70
#: ms and the geometric one at 40 and 50 ms, and both lose the H->L mirror
#: at 40 ms. A dithering or slow transition is not exotic, and a silent
#: miss is worse than a train of false positives a consumer can see. The
#: whole matrix is `test_matrix_a_*` through `test_matrix_e_*`.
LH_HOLD_FRAC = 0.70

#: Actuator thresholds, in the canonical units of `features/namespace.py`.
NBI_ON_KW = 500.0
ECH_ON_W = 1.0e5
RMP_ON_KA = 0.5
GAS_ON_V = 0.5
#: On at the threshold, off at half of it.
HYSTERESIS_RATIO = 2.0
#: An interval shorter than this is a glitch; a gap shorter than this is a
#: notch in one interval rather than the end of it.
ACTUATOR_MIN_MS = 20.0
ACTUATOR_GAP_MS = 20.0

#: What an EHO looks like (spec A4): 2-20 kHz, at least 100 ms, flat.
QH_F_LO_KHZ = 2.0
QH_F_HI_KHZ = 20.0
QH_MIN_MS = 100.0
QH_MAX_CHIRP_KHZ_PER_MS = 0.02
#: The one string that keeps a definitional association honest.
QH_NOTE = (
    "proxy: eho and elm_free and nbi and flattop "
    "(definitional, see Appendix C item 8)"
)

#: The fraction of its own peak the current has to hold for the record to
#: be in flat-top. 0.9 is the convention the q-min regimes were measured
#: under, and it is loose enough to ride over the sawtooth-scale ripple of
#: a real Ip trace while still excluding both ramps.
FLATTOP_FRAC = 0.9

#: The q-min regime boundaries, from the label sheet's own note - "qmin >
#: 0.95 = hybrid, > 1.5 = elevated, > 2 = high ... 500 ms". One line each:
#:
#: 0.95 is where the q = 1 surface leaves the plasma, so the sawtooth goes
#: with it and the discharge is a hybrid rather than a standard scenario.
QMIN_HYBRID = 0.95
#: 1.5 clears the 3/2 surface, the first tearing-prone rational surface
#: inside a conventional profile; above it the scenario is "elevated".
QMIN_ELEVATED = 1.5
#: 2.0 clears the 2/1 as well, the last rational surface worth naming:
#: above it the profile is advanced-tokamak territory.
QMIN_HIGH = 2.0
#: A band shorter than this is a stretch of a transition and not a regime
#: the shot was IN. 500 ms is the sheet's own figure, and it is exactly 25
#: sampling intervals of EFIT01's 20 ms q-min, so a real band lands on the
#: boundary often - see `qmin_regimes` for which side it falls on.
QMIN_MIN_MS = 500.0

#: `(phenomenon, lower bound EXCLUSIVE, upper bound INCLUSIVE or None)`.
#: Exclusive and exhaustive above `QMIN_HYBRID`: a sample is in at most one
#: band, and a q-min sitting exactly on a boundary belongs to the band
#: BELOW it, so the regimes partition rather than overlap.
QMIN_BANDS: tuple[tuple[str, float, float | None], ...] = (
    ("qmin_hybrid", QMIN_HYBRID, QMIN_ELEVATED),
    ("qmin_elevated", QMIN_ELEVATED, QMIN_HIGH),
    ("qmin_high", QMIN_HIGH, None),
)

#: Which equilibrium the q-min was reconstructed from, on every row. The
#: canonical `qmin` feature is `\efit01::top.results.aeqdsk:qmin` today and
#: the sheet asks for EFIT02 or CAKE; when a higher-fidelity source joins
#: `features/namespace.py` this attribute is what tells a stored row which
#: one it was computed from.
QMIN_EFIT = "efit01"

#: Who claims what. All four are `KNOWN_SOURCES` of the events table.
SAWTOOTH_SOURCE = "ece_sawtooth"
SAWTOOTH_PHENOMENON = "sawtooth"
LH_SOURCE = "dalpha_lh"
LH_PHENOMENON = "lh_transition"
HL_PHENOMENON = "hl_transition"
ACTUATOR_SOURCE = "actuator"
QH_SOURCE = "qh_proxy"
QH_PHENOMENON = "qh"
QMIN_SOURCE = "qmin_rule"
#: The canonical feature every `qmin_rule` row was measured on, written
#: into `diag` so a row says which axis its coverage is the coverage of.
QMIN_DIAG = "qmin"


# ----------------------------------------------------------------- envelope

def envelope(y, t_s, env_ms: float = ENV_MS):
    """`(C, n)` traces -> `((C, m) bin means, (m,) bin centres in seconds)`.

    `omnimode.mrms.ece.envelope`'s answer, to the bit, by a different route.
    The reference assigns every SAMPLE a bin with `searchsorted` and then
    runs one `np.bincount` per channel; this searches the m bin EDGES in the
    (sorted) time axis instead - m is a few thousand where n is a few
    million - and sums each bin's contiguous run with `np.add.reduceat`,
    accumulating in float64 whatever the record's dtype is. That is the same
    numbers added in the same order, and it reads the record once rather
    than once per channel and without materialising the float64 copy
    `np.asarray(X, float)` makes of a float32 record.

    A bin no sample falls in - a gap in the digitiser record - is zero, as
    it is in the reference; `reduceat` would otherwise return the sample at
    the empty run's start and make a gap look like one sample repeated.

    `env_ms` is milliseconds because that is the unit the reference's
    thresholds were set in; everything crossing this module's boundary is
    seconds.
    """
    y = np.atleast_2d(y)
    t_s = np.asarray(t_s, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError(f"expected (channels, samples), got {y.shape}")
    if t_s.ndim != 1 or t_s.size < 2:
        raise ValueError(f"a time axis needs at least two samples, got {t_s.shape}")
    if y.shape[1] != t_s.size:
        raise ValueError(
            f"{y.shape[1]} samples per channel against {t_s.size} times"
        )
    if not float(env_ms) > 0.0:
        raise ValueError(f"env_ms must be positive; got {env_ms}")
    # In MILLISECONDS, on the reference's own arithmetic. Binning the same
    # record on edges built in seconds is the same answer to a part in 1e16
    # and a different answer for any sample that lands within that of an
    # edge, of which a 3-million-sample record has a few - and on a 2%
    # threshold a single sample changing bins can add or drop a crash.
    t_ms = t_s * 1e3
    nbin = max(1, round((float(t_ms[-1]) - float(t_ms[0])) / float(env_ms)))
    edges = float(t_ms[0]) + np.arange(nbin + 1, dtype=np.float64) * float(env_ms)
    # `side="left"`: a sample sitting exactly on an edge opens the bin, which
    # is where the reference's `searchsorted(edges, t, "right") - 1` puts it.
    starts = np.searchsorted(t_ms, edges[:-1], side="left")
    counts = np.diff(np.append(starts, t_ms.size))
    sums = np.empty((y.shape[0], nbin), dtype=np.float64)
    safe = np.minimum(starts, t_ms.size - 1)
    for c in range(y.shape[0]):
        np.add.reduceat(y[c], safe, dtype=np.float64, out=sums[c])
    sums[:, counts == 0] = 0.0
    env = sums / np.maximum(counts, 1)
    return env, (edges[:-1] + float(env_ms) / 2.0) * 1e-3


# ------------------------------------------------------- the inversion test

def _runs(mask) -> list[tuple[int, int]]:
    """`[(start, end_exclusive)]` of contiguous True runs. The reference's."""
    idx = np.flatnonzero(
        np.diff(np.concatenate([[0], np.asarray(mask).view(np.int8), [0]]))
    )
    return list(zip(idx[::2].tolist(), idx[1::2].tolist(), strict=True))


def inversion_block(steps, rel_neg: float = REL_NEG, rel_pos: float = REL_POS,
                    min_block: int = MIN_BLOCK, max_gap: int = MAX_GAP):
    """`(a, b)` channel bounds (end-exclusive) of the core block of a
    STRUCTURED sawtooth inversion, or None when the step profile is not one.

    `omnimode.mrms.ece.inversion_block`, ported verbatim - its thresholds
    were calibrated on dev shot 193273 and nothing here has better ones. Its
    own note: any-both-signs was measured too loose (it accepted ELM edge
    crashes and a startup transient with one broken channel). The structured
    rule is that after 3-channel median smoothing the run containing the
    deepest drop must span at least `min_block` channels, sit INTERIOR to
    the array (an ELM's edge block terminates at the end of the channel
    range), and have an adjacent - within `max_gap` channels - positive
    block of at least `min_block` channels at `rel_pos` of the profile
    maximum: the displaced heat has to actually arrive somewhere next door.

    The returned interval is the DROPPING block. On the axis-crossing DIII-D
    array it spans both q=1 crossings, which is the region a 1/1 kink's
    phase flip must sit in; a per-boundary point reference was measured
    unusable, landing on whichever side's crossing matched first.

    Non-finite steps - a dead or NaN channel - are treated as zero, so one
    dead channel cannot make the array's profile unreadable.
    """
    s = np.asarray(steps, dtype=np.float64)
    if np.isfinite(s).sum() < 2 * min_block:
        return None
    sm = medfilt(np.where(np.isfinite(s), s, 0.0), 3)
    peak = float(np.max(np.abs(sm)))
    if peak <= 0:
        return None
    neg = sm < -rel_neg * peak
    if not neg.any():
        return None
    imin = int(np.argmin(sm))
    dom = next(((a, b) for a, b in _runs(neg) if a <= imin < b), None)
    if dom is None or dom[1] - dom[0] < min_block:
        return None
    if dom[0] == 0 or dom[1] == len(sm):        # touches an array end
        return None
    for a, b in _runs(sm > rel_pos * peak):
        if b - a < min_block:
            continue
        if (a >= dom[1] and a - dom[1] <= max_gap) or (
            b <= dom[0] and dom[0] - b <= max_gap
        ):
            return dom
    return None


def has_inversion(steps, **kw) -> bool:
    """True when the step profile is a structured sawtooth inversion.

    Delegates to `inversion_block` so the two can never disagree. The
    reference's, verbatim.
    """
    return inversion_block(steps, **kw) is not None


# ----------------------------------------------------------------- sawteeth

def _bin_drops(env: np.ndarray) -> np.ndarray:
    """`(C, m - 1)` fractional DROP from each envelope bin to the next.

    Positive is a fall. A non-finite ratio - a NaN channel, an envelope bin
    that sits at zero - is no change at all, so a dead channel neither votes
    for a crash nor blocks one.

    **This is where the port deviates from the reference, deliberately.**
    The reference finishes with `np.nan_to_num(d)`, which maps NaN to 0 but
    -inf to -1.8e308 - so a bin that is EMPTY (a gap in the digitiser
    record: the envelope of no samples is 0) followed by a bin below zero
    divides by zero, comes out -inf, and is counted as a channel dropping
    by 1.8e308. Here `np.where(np.isfinite(d), d, 0.0)` calls the same bin
    no change. A gap in the record is not a crash, and a detector that
    fires on the resumption of the digitiser is finding the digitiser. The
    divergence is reachable only through an empty envelope bin, which no
    contiguous corpus record has.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        drop = -np.diff(env, axis=1) / np.abs(env[:, :-1])
    return np.where(np.isfinite(drop), drop, 0.0)


def _crash_bins(drop: np.ndarray, *, env_ms: float, drop_frac: float,
                min_channels: int, min_interval_ms: float) -> list[int]:
    """Bin drops -> the bin indices a candidate sits after, in time order.

    The reference's `crash_times` with the envelope handed in: the count of
    channels falling by more than `drop_frac` between one bin and the next,
    and a greedy thin to `min_interval_ms` that keeps the most synchronised
    candidate first and drops whatever falls inside its shadow.

    The separation is measured in BINS rather than between two bin times.
    The bins are `env_ms` apart by construction, so `|i - k| * env_ms` is
    the same quantity exactly, where the difference of two times a few
    seconds into the shot carries a rounding of ~1e-13 ms - and with a 1 ms
    grid and a 10 ms threshold, candidates land EXACTLY on that boundary
    all day, so which side of it a rounding falls is which crashes a shot
    has.
    """
    score = (drop > float(drop_frac)).sum(axis=0)
    cand = np.flatnonzero(score >= int(min_channels))
    order = cand[np.argsort(score[cand])[::-1]]
    kept: list[int] = []
    for i in order.tolist():
        if all(
            abs(i - k) * float(env_ms) >= float(min_interval_ms) for k in kept
        ):
            kept.append(i)
    return sorted(kept)


def _step_offsets(env_ms: float, gap_ms: float,
                  span_ms: float) -> np.ndarray:
    """Bin offsets `d` with `gap_ms <= d * env_ms <= span_ms`, ascending.

    The step windows as INTEGER counts of bins, which is what the reference
    is selecting when it writes `te >= tc - span_ms`: the envelope grid is
    `env_ms` apart by construction, so the only thing that comparison can
    say is how many bins away a bin is. Saying it in bins says it exactly.

    "Exactly" is why the bounds are converted to bins ONCE and compared as
    integers, rather than each candidate `d * env_ms` being compared with
    `gap_ms` and `span_ms` in floating point. On the pipeline's 1 ms grid
    the two agree; on any grid whose spacing does not divide the bounds
    they need not, because `d * env_ms` lands a bit either side of a bound
    that is `d` bins away exactly - `3 * 0.7 == 2.0999999999999996` is not
    `>= 2.1`, and `8 * 0.7 == 5.6000000000000005` is not `<= 5.6`, so both
    end bins of a seven-bin window disappear.

    The bounds are converted to the NEAREST bin (`round`), not to the
    nearest bin strictly INSIDE the window (`ceil`/`floor`), which is what
    the float comparison amounted to. On any grid whose spacing divides
    both bounds - the pipeline's 1 ms one, and every grid this was written
    against - the two are the same integers, so nothing about the reference
    comparison changes. Off such a grid they differ by a bin at each end,
    and nearest is the one the comparison MEANT. `round` is banker's, so a
    bound landing exactly on a half-bin resolves to the even bin; that is
    arbitrary, and it is a half-bin either way.
    """
    env_ms, gap_ms, span_ms = float(env_ms), float(gap_ms), float(span_ms)
    lo = max(1, round(gap_ms / env_ms))
    hi = round(span_ms / env_ms)
    return np.arange(lo, hi + 1, dtype=np.intp)


def _crash_step(env: np.ndarray, k: int, *, env_ms: float = ENV_MS,
                gap_ms: float = STEP_GAP_MS,
                span_ms: float = STEP_SPAN_MS) -> np.ndarray:
    """Per-channel envelope step across the crash: mean(after) - mean(before).

    The reference's `crash_steps`, reading the envelope it is given rather
    than recomputing it, and selecting its windows in BIN SPACE - the bins
    `gap_ms` to `span_ms` either side of bin `k` - rather than by comparing
    bin centres in seconds against `tc - span`. The two are the same seven
    bins in exact arithmetic; in floating point the seconds version rounds
    the centre and the boundary independently, and on a 7000-bin record that
    dropped one of the seven bins from 86 before-windows and 78 after-
    windows. A step measured over six bins instead of seven is a different
    number, and `inversion_block`'s thresholds are relative to it.

    `nan` for every channel when either window falls off the end of the
    record entirely, which is the reference's answer too; a window that is
    merely SHORT there keeps the bins it has, as the reference's does.
    """
    d = _step_offsets(env_ms, gap_ms, span_ms)
    m = env.shape[1]
    before = int(k) - d[::-1]
    after = int(k) + d
    before = before[(before >= 0) & (before < m)]
    after = after[(after >= 0) & (after < m)]
    if before.size == 0 or after.size == 0:
        return np.full(env.shape[0], np.nan)
    return env[:, after].mean(axis=1) - env[:, before].mean(axis=1)


def sawtooth_events(
    ece_y,
    ece_t_s,
    *,
    shot: int,
    drop_frac: float = DROP_FRAC,
    min_channels: int = MIN_CHANNELS,
    min_interval_ms: float = MIN_INTERVAL_MS,
    gap_ms: float = STEP_GAP_MS,
    span_ms: float = STEP_SPAN_MS,
    t_cov: tuple[float, float],
) -> list[Event]:
    """The 48-channel ECE array -> one point event per sawtooth crash.

    Candidates are bins where at least `min_channels` channels lose more
    than `drop_frac` of their level in one 1 ms bin; a candidate is a
    sawtooth when the step profile across it is a structured inversion
    (`inversion_block`). The envelope is computed ONCE here and read by
    every candidate - the reference recomputes it per candidate, which is
    the whole cost of the detector.

    The crash time is the reference's: the centre of the first bin that is
    fully after the step, so it lags the crash by up to one envelope bin.

    `confidence` is the fraction of the channels that were LOOKED AT - the
    finite ones - that took part in the inversion, dropping or rising. A
    crash the whole array sees is worth more than one three channels see,
    and a NaN channel is neither evidence for nor against it.

    `attrs["inversion_channel_lo"]` and `attrs["inversion_channel_stop"]`
    are the dropping block's bounds and are END-EXCLUSIVE, like every other
    channel range in this package and like `inversion_block`'s return:
    channels `lo <= c < stop` dropped. The name says `stop` rather than
    `hi` because a reader who takes `hi` for the last dropping channel is
    off by one, and the inversion RADIUS - the thing anyone reads this for -
    sits between `stop - 1` and `stop`.
    """
    ece_y = np.atleast_2d(ece_y)
    if ece_y.shape[0] != N_ECE_CHANNELS:
        raise ValueError(
            f"the inversion test is a claim about the {N_ECE_CHANNELS}-channel "
            f"ECE array; got {ece_y.shape[0]} channels"
        )
    env, t_env_s = envelope(ece_y, ece_t_s)
    drop = _bin_drops(env)
    n_finite = int(np.isfinite(env).any(axis=1).sum())
    cov = (float(t_cov[0]), float(t_cov[1]))

    out: list[Event] = []
    for i in _crash_bins(
        drop, env_ms=ENV_MS, drop_frac=drop_frac, min_channels=min_channels,
        min_interval_ms=min_interval_ms,
    ):
        tc_s = float(t_env_s[i + 1])
        steps = _crash_step(
            env, i + 1, env_ms=ENV_MS, gap_ms=gap_ms, span_ms=span_ms
        )
        block = inversion_block(steps)
        if block is None:
            continue
        smoothed = medfilt(np.where(np.isfinite(steps), steps, 0.0), 3)
        peak = float(np.max(np.abs(smoothed)))
        n_dropping = int((smoothed < -REL_NEG * peak).sum())
        n_rising = int((smoothed > REL_POS * peak).sum())
        confidence = min(1.0, (n_dropping + n_rising) / max(n_finite, 1))
        out.append(
            Event(
                shot=int(shot),
                source=SAWTOOTH_SOURCE,
                evidence_kind="heuristic",
                phenomenon=SAWTOOTH_PHENOMENON,
                t0_s=tc_s,
                t1_s=tc_s,
                confidence=float(confidence),
                diag="ece",
                channel=-1,
                attrs={
                    "inversion_channel_lo": int(block[0]),
                    "inversion_channel_stop": int(block[1]),
                    "n_channels_dropping": n_dropping,
                    "n_channels_rising": n_rising,
                    "drop_frac_max": float(drop[:, i].max()),
                },
                t_cov0_s=cov[0],
                t_cov1_s=cov[1],
            )
        )
    return out


def sawtooth_summary(events: Sequence[Event]) -> dict[str, float]:
    """`n` crashes and the median and mean interval between them, in ms.

    The period is what a consumer actually asks of a sawtooth train, and it
    is a property of the LIST rather than of any one crash, so it is not an
    event attribute. NaN for both periods under two crashes: one crash has
    no period, and saying 0 would be saying something false.
    """
    times = np.sort(
        np.array(
            [e.t0_s for e in events if e.phenomenon == SAWTOOTH_PHENOMENON],
            dtype=np.float64,
        )
    )
    if times.size < 2:
        return {
            "n": int(times.size),
            "median_period_ms": math.nan,
            "mean_period_ms": math.nan,
        }
    periods_ms = np.diff(times) * 1e3
    return {
        "n": int(times.size),
        "median_period_ms": float(np.median(periods_ms)),
        "mean_period_ms": float(periods_ms.mean()),
    }


# ------------------------------------------------------------ L->H and H->L

def _at(t_s, y, when: float) -> float:
    """`y` at `when`, linearly interpolated; NaN outside the trace."""
    if t_s is None or y is None:
        return math.nan
    t = np.asarray(t_s, dtype=np.float64)
    v = np.asarray(y, dtype=np.float64)
    if t.size == 0 or when < t[0] or when > t[-1]:
        return math.nan
    return float(np.interp(when, t, v))


def _finite_in(t_s, y, lo: float, hi: float) -> np.ndarray:
    """The finite samples of `y` whose time is in `[lo, hi]`.

    Selecting first and reducing after, rather than reaching for a `nan*`
    reduction: `np.nanmean` of a window that is entirely NaN - the head of
    a filterscope record, a dead coil - warns rather than returning, and a
    warning is not an answer this module can act on.
    """
    if t_s is None or y is None:
        return np.zeros(0, dtype=np.float64)
    t = np.asarray(t_s, dtype=np.float64)
    v = np.asarray(y, dtype=np.float64)
    # `searchsorted` rather than a boolean mask: this is asked once per
    # candidate transition, and a mask over a 4.5-million-sample CO2 record
    # would make the detector quadratic in the number of candidates.
    a = int(np.searchsorted(t, lo, side="left"))
    b = int(np.searchsorted(t, hi, side="right"))
    seg = v[a:b]
    return seg[np.isfinite(seg)]


def _window_mean(t_s, y, lo: float, hi: float) -> float:
    """Mean of `y` over `[lo, hi]`, or NaN when nothing finite is in it."""
    got = _finite_in(t_s, y, lo, hi)
    return float(got.mean()) if got.size else math.nan


def _window_max(t_s, y, lo: float, hi: float) -> float:
    """Largest `y` over `[lo, hi]`, or NaN when nothing finite is in it."""
    got = _finite_in(t_s, y, lo, hi)
    return float(got.max()) if got.size else math.nan


def _window_median(t_s, y, lo: float, hi: float) -> float:
    """Median of `y` over `[lo, hi]`, or NaN when nothing finite is in it.

    The median rather than the mean because the windows it is asked for
    straddle ELMs: the mean of an ELMy phase is its duty cycle times its
    spikes, which moves when the ELM frequency does, while the median is
    the inter-ELM level whatever the ELMs are doing.
    """
    got = _finite_in(t_s, y, lo, hi)
    return float(np.median(got)) if got.size else math.nan


def _sample_s(t: np.ndarray) -> float:
    """Seconds between samples, from the axis itself."""
    step = float(t[-1] - t[0]) / max(t.size - 1, 1)
    if not step > 0.0:
        raise ValueError(f"times must increase; got {t[0]} to {t[-1]}")
    return step


def _usable_span(t: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(t, d)` trimmed to the finite span, with interior dropouts filled.

    Every filterscope record begins and ends in NaN, and the ends are simply
    not record - `masks.read_waveform` strips them the same way. An INTERIOR
    dropout is a different thing, and it is interpolated across rather than
    dropped: the rolling median below is not defined over a NaN, and a gap
    of a few samples cannot make a step in the level by itself while
    dropping it would move every later sample's window.
    """
    finite = np.flatnonzero(np.isfinite(d))
    if finite.size == 0:
        return t[:0], d[:0]
    lo, hi = int(finite[0]), int(finite[-1]) + 1
    t, d = t[lo:hi], d[lo:hi].copy()
    bad = ~np.isfinite(d)
    if bad.any():
        d[bad] = np.interp(t[bad], t[~bad], d[~bad])
    return t, d


def _shift(a: np.ndarray, by: int) -> np.ndarray:
    """`a` moved `by` samples later (positive) or earlier, ends held."""
    if by == 0 or a.size == 0:
        return a
    out = np.empty_like(a)
    if by > 0:
        out[:by] = a[0]
        out[by:] = a[:a.size - by]
    else:
        out[by:] = a[-1]
        out[:by] = a[-by:]
    return out


def _levels(d: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """The median level of the `width` samples before and after each sample.

    A ROLLING MEDIAN, not the samples themselves, and this is the whole
    difference between an L->H detector and an ELM detector. A D-alpha
    record in an ELMy phase steps down by far more than 30% within 5 ms
    after every single ELM - the spike falls back to the inter-ELM baseline
    - so comparing the sample at `t` with the one 5 ms later fires on every
    ELM in the shot (measured: 17,989 qualifying samples of 80,001 on shot
    185946). A median is unmoved by a spike a small fraction of the window
    wide, so what is compared is the LEVEL before against the LEVEL after,
    and only a step in the level survives.
    """
    m = median_filter(d, size=int(width), mode="nearest")
    half = int(width) // 2
    return _shift(m, half), _shift(m, -half)


def _step_runs(t: np.ndarray, before: np.ndarray, after: np.ndarray, *,
               frac_min: float, rising: bool,
               width: int) -> list[tuple[float, float]]:
    """`(time, fraction)` of each step of at least `frac_min` in the level.

    Every sample whose two windows straddle the step qualifies, so a step
    yields a RUN of them - symmetric about the step for an ideal one, since
    the leading window's median flips at the same distance the trailing
    one's does - and the run's midpoint is the step. That is as precise as
    the window, which is what a level comparison can be; a sub-window time
    would be a precision the median has already averaged away.

    Runs less than `width` apart are ONE step: a transition the plasma
    takes in two stages, or a run broken by a sample where the fraction
    dips under the threshold, is one transition, and reporting it three
    times two milliseconds apart says something false about how many there
    were.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (after - before) / before
    if not rising:
        frac = -frac
    ok = np.isfinite(frac) & (before > 0.0) & (frac >= float(frac_min))
    return [
        (0.5 * (float(t[a]) + float(t[b - 1])), float(frac[a:b].max()))
        for a, b in _runs(_bridge_mask(ok, max(int(width), 1)))
    ]


def _bridge_mask(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Close False gaps of at most `max_gap` samples between True runs."""
    filled = np.asarray(mask, dtype=bool).copy()
    for (_, prev_stop), (next_start, _) in itertools.pairwise(_runs(filled)):
        if next_start - prev_stop <= max_gap:
            filled[prev_stop:next_start] = True
    return filled


def lh_transitions(
    dalpha_t_s,
    dalpha_y,
    *,
    ne_t_s,
    ne_y,
    betan_t_s,
    betan_y,
    pinj_t_s,
    pinj_y,
    shot: int,
    drop_frac: float = LH_DROP_FRAC,
    drop_window_ms: float = LH_DROP_WINDOW_MS,
    ne_rise_frac: float = LH_NE_RISE_FRAC,
    min_pinj_kw: float = LH_MIN_PINJ_KW,
    t_cov: tuple[float, float],
) -> list[Event]:
    """D-alpha, density and NBI -> the L->H transitions, and the way back.

    An L->H is claimed where four things happen at once: the channel-median
    D-alpha falls by `drop_frac` of itself within `drop_window_ms`, it is
    STILL down `LH_HOLD_LO_MS` to `LH_HOLD_HI_MS` later, the line-averaged
    density is higher over that same later window, and the NBI is above
    `min_pinj_kw` in the window before it. No beams, no claim - and `[]`
    rather than an exception, because a shot with no NBI trace is a shot
    this heuristic cannot speak about.

    **The hold is the gate that makes this an L->H detector rather than an
    ELM detector.** A rolling median rejects a SPIKE, but a type-I ELM's
    D-alpha burst is 2-5 ms wide - most of the 5 ms drop window - so the
    level itself steps up at the burst and back down after it, and the drop
    gate fires on the way down from every single ELM. What an ELM does not
    do is stay: 20 to 50 ms later the level is exactly where it was before,
    while an L->H's is still down. So the median over `[t + 20 ms, t + 50
    ms]` must be at most `LH_HOLD_FRAC` of the median over the mirror
    window `[t - 50 ms, t - 20 ms]`, and the density test is read over that
    same post window rather than at one point 50 ms out. A transition
    within 50 ms of either end of the D-alpha record therefore cannot be
    claimed - the window it would have to hold over is not in the record.

    That comparison assumes the inter-ELM baseline is STATIONARY over the
    +/-50 ms, which a shot with a gas puff or a density ramp under it is
    not: a baseline falling by ~30% per 70 ms is lower after every ELM than
    it was before, so every ELM passes the hold and the gate becomes an ELM
    counter again (measured: a 100 ms half-life under 4 ms bursts every 15
    ms, four false `lh_transition`s in 0.8 s). That is a KNOWN LIMITATION
    and not an oversight: comparing against the pre window's own trend
    instead of its median removes it, and was tried twice and withdrawn
    both times, because the pre window of a transition whose fall starts
    before the edge is not a baseline and extrapolating it makes the
    transition explain itself away. See `LH_HOLD_FRAC` for the matrix.

    The H->L back transition is the same step upward, held the same way
    (the level `LH_HOLD_HI_MS` later at least `1 / LH_HOLD_FRAC` of the
    level before), with the density falling. Its density test is only a
    DIRECTION, not a fraction: the pedestal collapses in a millisecond but
    the density it built decays over tens, so a symmetric threshold would
    find no back transitions at all. NBI is required for it too, and
    measured in the window BEFORE the step, because the commonest back
    transition in the corpus is the beams turning off.

    `betan` decides nothing. It goes into `attrs` because a transition with
    betan already at 2 is a different event from one at 0.5, and that is a
    consumer's judgement to make and not this function's.

    Every measured quantity is in the `attrs`, so a marginal transition can
    be re-weighed without re-running anything.

    `t_cov` is the coverage the row declares - the pipeline passes the
    INTERSECTION of the D-alpha, the density and the beams, since the
    claim needs all three. A `when` past it by no more than the gate
    window (the beam record ended within `drop_window_ms` of the step) is
    written AT the bound with `attrs["clipped"] = true` and the measured
    instant in `attrs["t_measured_s"]`; see `coverage.clip_point_to_coverage`.
    """
    y = np.atleast_2d(np.asarray(dalpha_y, dtype=np.float64))
    if y.shape[0] != N_DALPHA_CHANNELS:
        raise ValueError(
            f"D-alpha is filterscope channels 0-{N_DALPHA_CHANNELS - 1}; got "
            f"{y.shape[0]} channels. Channels {N_DALPHA_CHANNELS} and up are "
            "NaN on every shot and are the caller's to drop"
        )
    t = np.asarray(dalpha_t_s, dtype=np.float64)
    if t.ndim != 1 or t.size < 2 or y.shape[1] != t.size:
        raise ValueError(
            f"{y.shape[1]} D-alpha samples against {t.size} times"
        )
    if pinj_t_s is None or pinj_y is None:
        return []
    pinj = np.asarray(pinj_y, dtype=np.float64)
    if not np.any(pinj >= float(min_pinj_kw)):
        return []

    # The channel median, column by column, without `np.nanmedian`'s warning
    # on the all-NaN columns every filterscope record starts and ends with.
    d = np.full(y.shape[1], np.nan)
    lit = np.isfinite(y).any(axis=0)
    if lit.any():
        d[lit] = np.nanmedian(y[:, lit], axis=0)
    t, d = _usable_span(t, d)
    if t.size < 2:
        return []
    window_s = float(drop_window_ms) * 1e-3
    width = max(1, round(window_s / _sample_s(t)))
    before_level, after_level = _levels(d, width)
    cov = (float(t_cov[0]), float(t_cov[1]))
    hold_lo_s = LH_HOLD_LO_MS * 1e-3
    hold_hi_s = LH_HOLD_HI_MS * 1e-3

    out: list[Event] = []
    for rising, phenomenon in ((False, LH_PHENOMENON), (True, HL_PHENOMENON)):
        for when, measured in _step_runs(
            t, before_level, after_level, frac_min=float(drop_frac),
            rising=rising, width=width,
        ):
            before = _window_mean(t, d, when - window_s, when)
            after = _window_mean(t, d, when, when + window_s)
            # The hold, over the two mirror windows: an ELM's level comes
            # back and a transition's does not.
            held0 = _window_median(t, d, when - hold_hi_s, when - hold_lo_s)
            held1 = _window_median(t, d, when + hold_lo_s, when + hold_hi_s)
            if not (math.isfinite(held0) and math.isfinite(held1)) or (
                held0 <= 0.0
            ):
                continue
            hold_frac = held1 / held0
            if rising:
                if not hold_frac >= 1.0 / LH_HOLD_FRAC:
                    continue
            elif not hold_frac <= LH_HOLD_FRAC:
                continue
            # The density over the SAME post window, so the two gates are
            # asking about one interval rather than two.
            ne0 = _window_mean(ne_t_s, ne_y, when - window_s, when)
            ne1 = _window_mean(
                ne_t_s, ne_y, when + hold_lo_s, when + hold_hi_s
            )
            if not (math.isfinite(ne0) and math.isfinite(ne1)) or ne0 == 0.0:
                continue
            ne_change = (ne1 - ne0) / abs(ne0)
            if rising:
                if not ne_change < 0.0:
                    continue
            elif not ne_change >= float(ne_rise_frac):
                continue
            pinj_kw = _window_max(pinj_t_s, pinj, when - window_s, when)
            if not (math.isfinite(pinj_kw) and pinj_kw >= float(min_pinj_kw)):
                continue
            betan = _at(betan_t_s, betan_y, when)
            # The gates above read windows AROUND `when`, so a transition up
            # to `drop_window_ms` after the beam record's last sample clears
            # them with every input measured - and lies past the
            # intersection coverage the pipeline hands in. Moved onto the
            # bound, said so, the measured instant kept; see
            # `coverage.clip_point_to_coverage`.
            at, clipped = clip_point_to_coverage(when, cov)
            out.append(
                Event(
                    shot=int(shot),
                    source=LH_SOURCE,
                    evidence_kind="heuristic",
                    phenomenon=phenomenon,
                    t0_s=at,
                    t1_s=at,
                    confidence=float(min(1.0, measured / LH_FULL_DROP)),
                    diag="filterscopes",
                    channel=-1,
                    attrs=clipped_attrs({
                        **({"t_measured_s": float(when)} if clipped else {}),
                        "drop_frac": float(measured),
                        "dalpha_before": float(before),
                        "dalpha_after": float(after),
                        "ne_change_frac": float(ne_change),
                        "hold_frac": float(hold_frac),
                        "dalpha_held_before": float(held0),
                        "dalpha_held_after": float(held1),
                        "pinj_kw": float(pinj_kw),
                        "betan": None if not math.isfinite(betan) else float(betan),
                        "window_ms": float(drop_window_ms),
                        "hold_lo_ms": float(LH_HOLD_LO_MS),
                        "hold_hi_ms": float(LH_HOLD_HI_MS),
                    }, clipped),
                    t_cov0_s=cov[0],
                    t_cov1_s=cov[1],
                )
            )
    return sorted(out, key=lambda e: e.t0_s)


# ---------------------------------------------------------------- actuators

#: `(phenomenon, canonical feature, threshold, units, take |.|, many
#: channels)`. The thresholds and units are `features/namespace.py`'s:
#: `pinj_total` is kW, `ech_power_total` W, `gas` V - the first two are
#: TOTALS and a caller handing over the eight beams instead has made a unit
#: error this refuses rather than silently reads as one beam. `rmp` has no
#: namespace entry yet and is expected in kA, which is how the coil currents
#: are quoted; it and `gas` are per-coil and per-valve, and one energised
#: coil is the whole set being on.
_ACTUATORS = (
    ("nbi_on", "pinj_total", NBI_ON_KW, "kW", False, False),
    ("ech_on", "ech_power_total", ECH_ON_W, "W", False, False),
    ("rmp_on", "rmp", RMP_ON_KA, "kA", True, True),
    ("gas_on", "gas", GAS_ON_V, "V", False, True),
)


def _level(name: str, y, *, absolute: bool = False,
           multi: bool = False) -> np.ndarray:
    """A feature's trace -> the one level per sample its threshold applies to.

    The loudest channel for the per-coil and per-valve features, because
    "the RMP coils are on" is a claim about the set of them. Written as a
    max over a `-inf` fill rather than as `np.nanmax`, which warns on an
    all-NaN column - a dead coil - instead of answering; here that column
    comes back NaN, and `_schmitt` holds the state across it.
    """
    y = np.asarray(y, dtype=np.float64)
    if absolute:
        y = np.abs(y)
    if y.ndim == 1:
        return y
    if y.ndim != 2 or not multi:
        raise ValueError(
            f"{name} must be (channels, samples); got {y.shape}" if multi
            else f"{name} is a total, not one trace per channel; "
                 f"got {y.shape}"
        )
    lit = np.isfinite(y)
    level = np.where(lit, y, -np.inf).max(axis=0)
    return np.where(lit.any(axis=0), level, np.nan)


def _schmitt(level: np.ndarray, thr: float,
             ratio: float = HYSTERESIS_RATIO) -> np.ndarray:
    """On at `thr`, off below `thr / ratio`, and holding in between.

    A beam notching to 400 kW has not turned off and a threshold crossed by
    noise has not turned on twice. A NaN sample is neither, so it holds
    whatever the actuator was doing; the state starts off.
    """
    high = level >= float(thr)
    low = level < float(thr) / float(ratio)
    decisive = np.where(high, 1, np.where(low, 0, -1))
    last = np.maximum.accumulate(
        np.where(decisive >= 0, np.arange(level.size), 0)
    )
    return decisive[last] == 1


def _samples_under(span_ms: float, step_s: float) -> int:
    """How many samples fit STRICTLY inside `span_ms`.

    `n * step < span`, so an exact ratio - which a 1 kHz clock and a 20 ms
    threshold give constantly - answers one less than the ratio rather than
    the ratio. The `1 - 1e-12` is what makes the exact case exact: the
    ratio's own float error is a part in 1e16, and no digitiser rate is
    within a part in 1e12 of putting a sample on the boundary by accident.
    """
    return max(0, math.floor(float(span_ms) * 1e-3 / step_s * (1.0 - 1e-12)))


def _mask_intervals(t: np.ndarray, mask: np.ndarray, *, min_ms: float,
                    gap_ms: float) -> list[tuple[int, int]]:
    """True runs of `mask` -> half-open sample bounds, bridged and filtered.

    Gaps first, then durations: a pulse notched in the middle is one pulse
    of its full length, and asking the question the other way round would
    throw away both halves before they could be joined.

    **Both are measured EDGE TO EDGE**, on the convention that a sample
    owns half a sampling step either side of itself: a run of `n` samples
    is `n * step` long and a gap of `n` samples is `n * step` wide, which
    partitions the record with nothing left over. So a gap is bridged when
    it is STRICTLY under `gap_ms` and an interval is kept when it is at
    least `min_ms`, and at 1 kHz that is 19 off-samples bridged and 20 not,
    20 on-samples kept and 19 not. The event's `t0_s`/`t1_s` are still the
    first and last SAMPLE - half a step inside the measured span at each
    end - because they are the times something was measured at, and a
    consumer intersecting two actuator intervals wants sample times.
    """
    if t.size == 0:
        return []
    step_s = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    if not step_s > 0.0:
        return []
    gap = _samples_under(gap_ms, step_s)
    keep = _samples_under(min_ms, step_s) + 1
    return [
        (a, b) for a, b in _runs(_bridge_mask(mask, gap))
        if b - a >= keep
    ]


def _actuator_event(shot: int, phenomenon: str, t: np.ndarray,
                    level: np.ndarray, bounds: tuple[int, int], units: str,
                    cov: tuple[float, float], *, name: str = "") -> Event:
    """One interval of one actuator, with the level it held inside it.

    `name` is the canonical feature the interval was measured on and goes
    into `diag`, so the row says WHICH axis its `t_cov` is the coverage of.
    An `nbi_on` row and a `gas_on` row are read off different digitisers
    with different spans - 0 to 13.10 s and -10 to 94.86 s on shot 198658 -
    and a table in which both said only "actuator" could not be asked
    whether the beams were off after 13 s or simply not measured.
    """
    a, b = bounds
    inside = level[a:b]
    inside = inside[np.isfinite(inside)]
    mean = float(inside.mean()) if inside.size else math.nan
    top = float(inside.max()) if inside.size else math.nan
    # The trigger holds its state across a NaN, which is right for a
    # dropout inside a record and turns into an overrun at the end of one;
    # the coverage is the finite span, so the extent is trimmed into it.
    t0_s, t1_s, clipped = clip_to_coverage(float(t[a]), float(t[b - 1]), cov)
    return Event(
        shot=int(shot),
        source=ACTUATOR_SOURCE,
        evidence_kind="heuristic",
        phenomenon=phenomenon,
        t0_s=t0_s,
        t1_s=t1_s,
        diag=str(name),
        channel=-1,
        attrs=clipped_attrs({
            "mean_level": 0.0 if not math.isfinite(mean) else mean,
            "max_level": 0.0 if not math.isfinite(top) else top,
            "units": units,
        }, clipped),
        t_cov0_s=cov[0],
        t_cov1_s=cov[1],
    )


def actuator_intervals(
    features: Mapping[str, tuple[Any, Any]],
    *,
    shot: int,
    t_cov: tuple[float, float] | Mapping[str, tuple[float, float]],
) -> list[Event]:
    """Canonical actuator features -> the intervals each one was on for.

    `features` maps a `features/namespace.py` name to `(t_s, y)`, already
    resolved by the caller: this module opens nothing and knows no locators.
    A feature that is absent produces no events, because a shot without RMP
    coils is the common case and is not an error.

    `t_cov` is PER FEATURE - `{name: (t0, t1)}`, as
    `coverage.feature_spans` builds it - and each row gets the coverage of
    the axis it was measured on. A single `(t0, t1)` is still accepted and
    is given to every row, which is what the old callers did and what the
    iteration-0 critic found wrong on shot 198658: the gas recorder's
    -10 to 94.86 s axis became the declared coverage of an `nbi_on` row
    whose digitiser stopped at 13.10 s. `nbi_counter` needs the torque,
    the injected power and the current at once, so it takes the
    INTERSECTION of those three rather than any one of them.

    Every interval is a 2:1 Schmitt trigger on the feature's own threshold,
    with gaps under `ACTUATOR_GAP_MS` bridged and intervals under
    `ACTUATOR_MIN_MS` dropped, and carries no confidence: it is a threshold
    on a measured actuator, and its uncertainty is the actuator's.

    `nbi_counter` is the one that is not a threshold. It marks where the
    injected torque opposes the plasma current while the beams are on, which
    is the circumstance a QH-mode is usually found in - and it needs `ip`,
    so without `ip` it is simply not claimed rather than guessed.
    """
    def cov_of(*names: str) -> tuple[float, float]:
        """The coverage of one input, or the intersection of several."""
        if not isinstance(t_cov, Mapping):
            return (float(t_cov[0]), float(t_cov[1]))
        spans = [t_cov.get(name, UNKNOWN) for name in names]
        return spans[0] if len(spans) == 1 else intersect(spans)

    out: list[Event] = []
    for phenomenon, name, thr, units, absolute, multi in _ACTUATORS:
        if name not in features:
            continue
        t_raw, y_raw = features[name]
        t = np.asarray(t_raw, dtype=np.float64)
        level = _level(name, y_raw, absolute=absolute, multi=multi)
        if t.size != level.size:
            raise ValueError(
                f"{name}: {level.size} samples against {t.size} times"
            )
        on = _schmitt(level, thr)
        cov = cov_of(name)
        out.extend(
            _actuator_event(shot, phenomenon, t, level, bounds, units, cov,
                            name=name)
            for bounds in _mask_intervals(
                t, on, min_ms=ACTUATOR_MIN_MS, gap_ms=ACTUATOR_GAP_MS
            )
        )

    if not {"tinj_total", "ip", "pinj_total"} <= set(features):
        return sorted(out, key=lambda e: (e.t0_s, e.phenomenon))
    t_tinj = np.asarray(features["tinj_total"][0], dtype=np.float64)
    tinj = _level("tinj_total", features["tinj_total"][1])
    pinj_t, pinj_y = features["pinj_total"]
    ip_t, ip_y = features["ip"]
    # Both onto the torque's clock: the three come off different digitisers
    # and a sign comparison between two grids is a sign comparison between
    # two different moments.
    pinj_on_tinj = np.interp(
        t_tinj, np.asarray(pinj_t, dtype=np.float64),
        _level("pinj_total", pinj_y), left=np.nan, right=np.nan,
    )
    beams = _schmitt(pinj_on_tinj, NBI_ON_KW)
    ip = np.interp(
        t_tinj, np.asarray(ip_t, dtype=np.float64),
        _level("ip", ip_y), left=np.nan, right=np.nan,
    )
    # Only where all three were actually MEASURED. `_schmitt` holds its
    # state across a NaN, which is right for a dropout inside a record and
    # wrong at the end of one: `ip` and `pinj_total` come off other
    # digitisers than the torque and routinely stop earlier, and holding
    # the state there is how a beam that stopped being measured goes on
    # injecting counter-current torque to the end of the shot.
    covered = np.isfinite(pinj_on_tinj) & np.isfinite(ip) & np.isfinite(tinj)
    with np.errstate(invalid="ignore"):
        counter = (
            covered & beams & (np.sign(tinj) != 0) & (np.sign(ip) != 0)
            & (np.sign(tinj) != np.sign(ip))
        )
    counter_cov = cov_of("tinj_total", "pinj_total", "ip")
    out.extend(
        _actuator_event(
            shot, "nbi_counter", t_tinj, np.abs(tinj), bounds, "N m",
            counter_cov, name="tinj_total",
        )
        for bounds in _mask_intervals(
            t_tinj, counter, min_ms=ACTUATOR_MIN_MS, gap_ms=ACTUATOR_GAP_MS
        )
    )
    return sorted(out, key=lambda e: (e.t0_s, e.phenomenon))


# --------------------------------------------------------------- the QH proxy

def _as_intervals(pairs) -> np.ndarray:
    """`(n, 2)` float64, sorted by start. `(0, 2)` for nothing."""
    arr = np.asarray(pairs, dtype=np.float64).reshape(-1, 2)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return arr[np.argsort(arr[:, 0], kind="stable")]


def _intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The intersection of two interval sets, as `(m, 2)`.

    Both are assumed sorted by start and non-overlapping within themselves,
    which every producer here guarantees. Zero-length overlaps are dropped:
    two intervals that merely touch share no time.
    """
    out: list[tuple[float, float]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i, 0], b[j, 0])
        hi = min(a[i, 1], b[j, 1])
        if hi > lo:
            out.append((float(lo), float(hi)))
        if a[i, 1] < b[j, 1]:
            i += 1
        else:
            j += 1
    return np.array(out, dtype=np.float64).reshape(-1, 2)


def _is_eho(track: Track) -> bool:
    """Whether a track looks like an EHO: in band, long enough, and flat."""
    return (
        QH_F_LO_KHZ <= track.f_centroid_khz <= QH_F_HI_KHZ
        and track.duration_ms >= QH_MIN_MS
        and abs(track.chirp_khz_per_ms) < QH_MAX_CHIRP_KHZ_PER_MS
    )


def qh_candidates(
    eho_tracks: Sequence[Track],
    elm_free,
    nbi,
    ip_flattop,
    *,
    shot: int,
    t_cov: tuple[float, float],
) -> list[Event]:
    """EHO tracks and three interval sets -> QH-mode candidate intervals.

    The intersection, piece by piece: where an EHO-like track runs inside an
    ELM-free interval, inside an NBI-on interval, inside the Ip flat-top.
    `confidence` is the weaker of the track's own and the fraction of the
    TRACK that is ELM-free - a mode that is quiet for a tenth of its life is
    not a QH-mode however clean the tenth is.

    This is a proxy and `attrs["note"]` says so. QH-mode is defined in this
    pipeline as an EHO in an ELM-free NBI-heated flat-top, so these rows
    restate their inputs and are evidence for ranking candidates to LOOK at,
    never evidence that EHOs and QH-modes go together (Appendix C item 8).
    """
    free = _as_intervals(elm_free)
    beams = _as_intervals(nbi)
    flattop = _as_intervals(ip_flattop)
    cov = (float(t_cov[0]), float(t_cov[1]))
    out: list[Event] = []
    for track in eho_tracks:
        if not _is_eho(track):
            continue
        span = np.array([[track.t0_s, track.t1_s]], dtype=np.float64)
        duration = float(track.t1_s - track.t0_s)
        quiet = _intersect(span, free)
        free_frac = (
            float(np.sum(quiet[:, 1] - quiet[:, 0]) / duration)
            if duration > 0.0 else 0.0
        )
        for lo, hi in _intersect(_intersect(quiet, beams), flattop).tolist():
            out.append(
                Event(
                    shot=int(shot),
                    source=QH_SOURCE,
                    evidence_kind="heuristic",
                    phenomenon=QH_PHENOMENON,
                    t0_s=float(lo),
                    t1_s=float(hi),
                    f0_khz=float(track.f0_khz),
                    f1_khz=float(track.f1_khz),
                    confidence=float(
                        min(max(track.conf, 0.0), 1.0, max(free_frac, 0.0))
                    ),
                    diag="",
                    channel=-1,
                    pass_name="zoom",
                    attrs={
                        "note": QH_NOTE,
                        "eho_t0_s": float(track.t0_s),
                        "eho_t1_s": float(track.t1_s),
                        "f_centroid_khz": float(track.f_centroid_khz),
                        "chirp_khz_per_ms": float(track.chirp_khz_per_ms),
                        "track_conf": float(track.conf),
                        "elm_free_fraction": free_frac,
                    },
                    t_cov0_s=cov[0],
                    t_cov1_s=cov[1],
                )
            )
    return sorted(out, key=lambda e: e.t0_s)


# ------------------------------------------- the Ip flat-top and the q-min rule

def _one_trace(name: str, y) -> np.ndarray:
    """A scalar canonical feature's `(1, T)` or `(T,)` record as `(T,)`.

    The features store keeps every quantity as `(channels, samples)`, so a
    scalar arrives one row deep. Anything wider is refused rather than
    reduced: `ip` and `qmin` are single-valued, and a caller handing over
    several rows has resolved something else.
    """
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0]
    raise ValueError(f"{name} is one trace, not {np.shape(y)}")


def ip_flattop(t_s, ip_y) -> tuple[float, float]:
    """`(t0, t1)` of the plasma-current flat-top, or `UNKNOWN`.

    The longest CONTIGUOUS stretch of samples whose |Ip| is over
    `FLATTOP_FRAC` of the record's own peak, as `(first sample, last
    sample)`. Everything about it is deliberately crude, and that is the
    point: the flat-top here is a GATE, not a measurement. It exists so
    that a regime rule is asked about the part of the record where there
    was a steady plasma, and the cost of the gate being a hundred
    milliseconds out at either end is one sample of a 20 ms q-min, while
    the cost of not having it at all is the whole rule - `qmin > 0.95`
    fires on 99.4% of shots ungated, because every ramp passes through it.
    
    Magnitude, not sign: DIII-D runs both current directions and a
    reversed-Ip shot's flat-top is the same stretch of record. Ties go to
    the earlier stretch. `UNKNOWN` when nothing in the record is finite,
    because a flat-top computed from no samples is not one.

    This is the one thing `ip` was missing for: `ip` is not a corpus group,
    so before the features store reached the events stage there was no
    flat-top, `qh_candidates` intersected with the empty set and claimed
    nothing on every shot, and `actuator_intervals` could not compare the
    torque's sign with the current's.
    """
    t = np.asarray(t_s, dtype=np.float64).ravel()
    y = _one_trace("ip", ip_y)
    if t.size != y.size:
        raise ValueError(f"ip: {y.size} samples against {t.size} times")
    level = np.abs(y)
    finite = np.isfinite(level) & np.isfinite(t)
    if not finite.any():
        return UNKNOWN
    peak = float(level[finite].max())
    with np.errstate(invalid="ignore"):
        on = finite & (level > FLATTOP_FRAC * peak)
    best: tuple[float, float] | None = None
    longest = -1.0
    for a, b in _runs(on):
        span = float(t[b - 1] - t[a])
        # Strictly greater, so the FIRST of two equally long stretches
        # wins and a record with two identical halves is not decided by
        # iteration order.
        if span > longest:
            longest, best = span, (float(t[a]), float(t[b - 1]))
    return UNKNOWN if best is None else best


def qmin_rule_coverage(qmin_t, qmin_y, flattop) -> tuple[float, float]:
    """What a `qmin_rule` row's coverage is: the flat-top, where q-min was.

    The intersection of the Ip flat-top with the finite span of the q-min
    record, and not either one alone. Outside the flat-top the rule DOES
    NOT LOOK - a ramp's q-min is not a regime - so declaring the whole
    q-min record as coverage would turn a deliberate abstention into an
    observed absence, which is the one thing the coverage columns exist to
    prevent.
    """
    if flattop is None:
        return UNKNOWN
    return intersect([
        finite_span(qmin_t, _one_trace("qmin", qmin_y)),
        (float(flattop[0]), float(flattop[1])),
    ])


def _qmin_event(shot: int, phenomenon: str, lo: float, hi: float | None,
                t: np.ndarray, q: np.ndarray, bounds: tuple[int, int],
                cov: tuple[float, float]) -> Event:
    """One q-min band as a row, with the extremes it held inside it."""
    a, b = bounds
    inside = q[a:b]
    t0_s, t1_s, clipped = clip_to_coverage(float(t[a]), float(t[b - 1]), cov)
    return Event(
        shot=int(shot),
        source=QMIN_SOURCE,
        evidence_kind="heuristic",
        phenomenon=phenomenon,
        t0_s=t0_s,
        t1_s=t1_s,
        # No confidence, deliberately. A threshold on a reconstructed
        # scalar has no calibrated probability behind it, and writing 1.0
        # would let a ranker read a rule as a perfectly-confident detector.
        diag=QMIN_DIAG,
        channel=-1,
        attrs=clipped_attrs({
            "qmin_min": float(inside.min()),
            "qmin_max": float(inside.max()),
            "efit": QMIN_EFIT,
            # `None` and not `inf` for the open-topped band: `attrs` is
            # stored as strict JSON, which has no word for an infinity.
            "thresholds": {"lo": float(lo),
                           "hi": None if hi is None else float(hi),
                           "min_ms": float(QMIN_MIN_MS)},
        }, clipped),
        t_cov0_s=cov[0],
        t_cov1_s=cov[1],
    )


def qmin_regimes(qmin_t, qmin_y, flattop, *, shot: int) -> list[Event]:
    """The q-min record and the Ip flat-top -> exclusive regime intervals.

    `qmin_hybrid` (0.95 < q <= 1.5), `qmin_elevated` (1.5 < q <= 2) and
    `qmin_high` (q > 2), each claimed over a contiguous run of q-min
    samples that is INSIDE the flat-top and lasts at least `QMIN_MIN_MS`.
    A sample belongs to at most one band, so the rows of one shot never
    overlap; a non-finite sample belongs to none, which splits a band
    around a dropout rather than interpolating over it.

    Three things this rule is not. It has **no confidence**: it is a
    threshold on a reconstructed scalar, and the honest value is NaN.
    It is **gated**, and the gate is most of the rule - ungated,
    `q > 0.95` fires on 497 of the 500 `recommender_v1` shots (99.4%),
    because the current ramp takes every shot through every band on its
    way up. And it is **EFIT01's** q-min today (`attrs["efit"]`), which
    the label sheet notes is not the equilibrium it would have asked for.

    MEASURED, on those 500 shots through the features store: hybrid 271,
    elevated 60, high 50 shots. The duration test is `>=`, edge to edge on
    the SAMPLE times - a band of exactly 25 of EFIT01's 20 ms slices
    counts - and that is not a free choice: `>` instead gives 268 / 60 /
    48. The same knife-edge cuts the other way once in the 500 (shot
    190509's elevated band is 25 slices long but subtracts to
    0.4999999999999999 s in float64 and is dropped), which is the price of
    deciding a physical claim with a bare float comparison and is recorded
    here rather than discovered later.
    """
    t = np.asarray(qmin_t, dtype=np.float64).ravel()
    q = _one_trace("qmin", qmin_y)
    if t.size != q.size:
        raise ValueError(f"qmin: {q.size} samples against {t.size} times")
    cov = qmin_rule_coverage(t, q, flattop)
    if not (math.isfinite(cov[0]) and math.isfinite(cov[1])):
        return []
    inside = np.isfinite(q) & (t >= float(flattop[0])) & (t <= float(flattop[1]))
    least_s = QMIN_MIN_MS * 1e-3
    out: list[Event] = []
    for phenomenon, lo, hi in QMIN_BANDS:
        with np.errstate(invalid="ignore"):
            band = inside & (q > lo)
            if hi is not None:
                band &= q <= hi
        out.extend(
            _qmin_event(shot, phenomenon, lo, hi, t, q, (a, b), cov)
            for a, b in _runs(band)
            if float(t[b - 1] - t[a]) >= least_s
        )
    return sorted(out, key=lambda e: (e.t0_s, e.phenomenon))
