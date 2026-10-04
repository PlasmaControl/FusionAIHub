"""The Prad,div indicator: radiated power below the X-point over the heating power.

`f_div = Prad,div,L / P_in` with the lower-divertor radiated power from the
calibrated bolometer (`\\BOLOM::PRAD_DIVL`, Eldon 2019's Prad,div,L) and P_in the
beam, ohmic and ECH power. Thresholds and their sources are in `thresholds.py`.
Prad,div is a radiation measure, not a detachment measure, and its level differs
between shots. It is not part of the exported label: it is a within-shot
corroborator reported per shot, and a sensitivity variant adds its vote to the
rule (`label_model.SECOND_VOTERS`). It never votes MARFE and never corroborates
one: a MARFE moves the radiation out of the Prad,div,L region (above the
X-point), so a MARFE bin can read the same f_div as a detached one or lower.

Two votes are computed from the same value. The relative one (`relative_fdiv`:
f_div over the shot's own baseline, so the shot's input power and seeding cancel;
`with_relative_vote`) is the indicator's vote (`prad_vote`). The absolute vote
(`fdiv_vote` with the shot-201081-anchored global cutoffs) is recorded beside it
(`prad_abs_vote`): those cutoffs do not carry over from the anchor shot (Opus
review 5, I1).
"""

from __future__ import annotations

import numpy as np

from . import thresholds as th
from .core import (
    ABSTAIN,
    ATTACHED,
    BIN_MS,
    DETACHED,
    Indicator,
    assemble,
    bin_centres,
    bin_fraction,
    bin_mean,
    elm_at,
    elm_bin_known,
    sample_windows_known,
)


def fdiv_vote(
    f: np.ndarray,
    attached_max: float | None = None,
    detached_min: float | None = None,
) -> np.ndarray:
    """Attached at or below `attached_max` (default PRAD_ATTACHED_MAX), detached at
    or above `detached_min` (default PRAD_DETACHED_MIN), abstain between (and where
    `f` is not finite). The cutoffs are arguments so the sweeps can vary them."""
    attached_max = th.PRAD_ATTACHED_MAX if attached_max is None else attached_max
    detached_min = th.PRAD_DETACHED_MIN if detached_min is None else detached_min
    f = np.asarray(f, dtype=float)
    vote = np.full(f.shape, ABSTAIN, dtype=np.int8)
    with np.errstate(invalid="ignore"):
        vote[f <= attached_max] = ATTACHED
        vote[f >= detached_min] = DETACHED
    vote[~np.isfinite(f)] = ABSTAIN
    return vote


def relative_fdiv(
    f: np.ndarray,
    valid: np.ndarray,
    p_in_w: np.ndarray | None = None,
    width_ms: float = BIN_MS,
) -> np.ndarray:
    """f_div over the shot's own baseline, NaN where invalid or without a baseline.

    The baseline is the `PRAD_BASELINE_QUANTILE` of the shot's valid f_div (its
    unseeded level: seeding and heating only raise the ratio), and needs at least
    `PRAD_BASELINE_MIN_MS` of valid bins at `width_ms`. When `p_in_w` is given, the baseline uses
    only the bins at the shot's flat-top input power (at least
    `PRAD_BASELINE_POWER_FRACTION` of its 90th percentile), so the beam ramp-up,
    where the ratio is low for want of power, is not read as the unseeded level.
    A shot detached throughout has no attached baseline and reads as attached: a
    stated limitation of this vote.
    """
    f = np.asarray(f, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(f)
    out = np.full(f.shape, np.nan)
    basis = valid
    if p_in_w is not None:
        p_in_w = np.asarray(p_in_w, dtype=float)
        if valid.any() and np.isfinite(p_in_w[valid]).any():
            top = np.nanquantile(p_in_w[valid], 0.9)
            basis = valid & (p_in_w >= th.PRAD_BASELINE_POWER_FRACTION * top)
    if basis.sum() < th.min_bins(th.PRAD_BASELINE_MIN_MS, width_ms):
        return out
    base = float(np.quantile(f[basis], th.PRAD_BASELINE_QUANTILE))
    if base <= 0:
        return out
    out[valid] = f[valid] / base
    return out


def relative_vote(ratio: np.ndarray) -> np.ndarray:
    """Vote on `relative_fdiv` with the anchor-derived relative cutoffs."""
    return fdiv_vote(ratio, th.PRAD_REL_ATTACHED_MAX, th.PRAD_REL_DETACHED_MIN)


def with_relative_vote(indicator: Indicator, ratio: np.ndarray) -> Indicator:
    """The exported f_div indicator: the absolute f_div value, voting on `ratio`.

    A bin that is valid but has no baseline (`ratio` not finite: the shot has too
    few flat-top bins) cannot vote: it becomes invalid, reason `no_baseline`.
    """
    ratio = np.asarray(ratio, dtype=float)
    valid = indicator.valid & np.isfinite(ratio)
    reason = np.where(indicator.valid & ~valid, "no_baseline", indicator.reason)
    vote = np.where(valid, relative_vote(ratio), ABSTAIN)
    return assemble(indicator.name, indicator.value, valid, reason, vote)


def elm_window_known(edges, elm_t_ms, elm_flag) -> np.ndarray:
    """D-alpha availability over the centered radiation averaging windows."""
    if elm_t_ms is None or elm_flag is None:
        return np.zeros(len(edges) - 1, bool)
    centres = bin_centres(edges)
    half = th.PRAD_AVERAGING_MS / 2
    return sample_windows_known(
        elm_t_ms,
        np.isfinite(elm_flag),
        centres - half,
        centres + half,
        closed_right=True,
    )


def prad_indicator(
    edges: np.ndarray,
    prad_t_ms: np.ndarray | None,
    prad_w: np.ndarray | None,
    power_t_ms: np.ndarray | None,
    p_in_w: np.ndarray | None,
    elm_t_ms: np.ndarray | None = None,
    elm_flag: np.ndarray | None = None,
) -> Indicator:
    """Prad indicator on a bin grid.

    Radiation and heating use centered 250 ms means (an acausal local tau_E-scale
    choice). Radiation drops measured ELM samples; uncovered heating/radiation
    windows stay invalid. D-alpha availability must cover the entire radiation
    averaging window as well as the native bin. A bin mostly in ELMs is invalid.
    Reasons include
    `no_bolometer`, `no_input_power`, `low_power`, `elm`, `elm_unknown`,
    `no_samples`, `negative_radiation`. Negative radiation below the documented
    0.05 MW offset tolerance in either the native label bin or the smoothed
    window is rejected; smaller negative offsets become zero.
    """
    n = len(edges) - 1
    value = np.full(n, np.nan)
    reason = np.full(n, "no_bolometer", dtype=object)
    if prad_t_ms is None or prad_w is None or not np.isfinite(prad_w).any():
        return assemble("prad", value, np.zeros(n, bool), reason, np.zeros(n))
    reason[:] = "no_input_power"
    if power_t_ms is None or p_in_w is None:
        return assemble("prad", value, np.zeros(n, bool), reason, np.zeros(n))
    prad_t_ms = np.asarray(prad_t_ms, dtype=float)
    in_elm = elm_at(prad_t_ms, elm_t_ms, elm_flag)
    from .signals import window_mean

    centres = bin_centres(edges)
    prad = window_mean(prad_t_ms, prad_w, centres, th.PRAD_AVERAGING_MS, keep=~in_elm)
    native_prad, count = bin_mean(prad_t_ms, prad_w, edges, keep=~in_elm)
    p_in = window_mean(power_t_ms, p_in_w, centres, th.PRAD_AVERAGING_MS)
    elm_share = (
        bin_fraction(prad_t_ms, in_elm, edges) if elm_t_ms is not None else np.zeros(n)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        corrected = np.where(
            prad >= -th.RADIATION_NEGATIVE_TOL_W, np.maximum(prad, 0.0), prad
        )
        value = corrected / p_in
    reason[:] = ""
    reason[~np.isfinite(p_in)] = "no_input_power"
    reason[np.isfinite(p_in) & (p_in < th.MIN_INPUT_POWER_W)] = "low_power"
    reason[~np.isfinite(prad) & (reason == "")] = "no_samples"
    reason[
        (prad < -th.RADIATION_NEGATIVE_TOL_W)
        | (native_prad < -th.RADIATION_NEGATIVE_TOL_W)
    ] = "negative_radiation"
    reason[np.nan_to_num(elm_share) > th.MAX_ELM_FRACTION] = "elm"
    reason[(count == 0) & (reason == "")] = "no_samples"
    elm_known = elm_bin_known(edges, elm_t_ms, elm_flag) & elm_window_known(
        edges, elm_t_ms, elm_flag
    )
    reason[~elm_known] = "elm_unknown"
    valid = (reason == "") & np.isfinite(value)
    return assemble("prad", value, valid, reason, fdiv_vote(value))
