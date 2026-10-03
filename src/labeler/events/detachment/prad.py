"""The Prad,div indicator: radiated power below the X-point over the heating power.

`f_div = Prad,div,L / P_in` with the lower-divertor radiated power from the
calibrated bolometer (`\\BOLOM::PRAD_DIVL`, Eldon 2019's Prad,div,L) and P_in the
beam, ohmic and ECH power. Thresholds and their sources are in `thresholds.py`.
Prad,div is a radiation measure, not a detachment measure; the indicator is a
weak voter and never votes MARFE (the integrals carry no position).
"""

from __future__ import annotations

import numpy as np

from . import thresholds as th
from .core import (
    ABSTAIN,
    ATTACHED,
    DETACHED,
    Indicator,
    assemble,
    bin_centres,
    bin_fraction,
    bin_mean,
    elm_at,
    elm_bin_known,
)


def fdiv_vote(f: np.ndarray) -> np.ndarray:
    """Attached at or below PRAD_ATTACHED_MAX, detached at or above
    PRAD_DETACHED_MIN, abstain between (and where `f` is not finite)."""
    f = np.asarray(f, dtype=float)
    vote = np.full(f.shape, ABSTAIN, dtype=np.int8)
    with np.errstate(invalid="ignore"):
        vote[f <= th.PRAD_ATTACHED_MAX] = ATTACHED
        vote[f >= th.PRAD_DETACHED_MIN] = DETACHED
    vote[~np.isfinite(f)] = ABSTAIN
    return vote


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
    windows stay invalid. A bin mostly in ELMs is invalid. Reasons include
    `no_bolometer`, `no_input_power`, `low_power`, `elm`, `elm_unknown`,
    `no_samples`, `negative_radiation`. Negative radiation below the documented
    0.05 MW offset tolerance is rejected; smaller negative offsets become zero.
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
    prad = window_mean(
        prad_t_ms, prad_w, centres, th.PRAD_AVERAGING_MS, keep=~in_elm
    )
    _, count = bin_mean(prad_t_ms, prad_w, edges, keep=~in_elm)
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
    reason[prad < -th.RADIATION_NEGATIVE_TOL_W] = "negative_radiation"
    reason[np.nan_to_num(elm_share) > th.MAX_ELM_FRACTION] = "elm"
    reason[(count == 0) & (reason == "")] = "no_samples"
    reason[~elm_bin_known(edges, elm_t_ms, elm_flag)] = "elm_unknown"
    valid = (reason == "") & np.isfinite(value)
    return assemble("prad", value, valid, reason, fdiv_vote(value))
