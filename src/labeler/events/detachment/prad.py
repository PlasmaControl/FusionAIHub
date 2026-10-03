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
    bin_fraction,
    bin_mean,
    elm_at,
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

    Bolometer samples inside an ELM are dropped before the bin mean (a mean, not a
    median: it is a power); a bin whose samples are mostly in ELMs is invalid. The
    reason on an invalid bin is one of `no_bolometer`, `no_input_power`,
    `low_power`, `elm`, `no_samples`.
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
    prad, count = bin_mean(prad_t_ms, prad_w, edges, keep=~in_elm)
    p_in, _ = bin_mean(power_t_ms, p_in_w, edges)
    elm_share = (
        bin_fraction(prad_t_ms, in_elm, edges) if elm_t_ms is not None else np.zeros(n)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        value = prad / p_in
    reason[:] = ""
    reason[~np.isfinite(p_in)] = "no_input_power"
    reason[np.isfinite(p_in) & (p_in < th.MIN_INPUT_POWER_W)] = "low_power"
    reason[np.nan_to_num(elm_share) > th.MAX_ELM_FRACTION] = "elm"
    reason[(count == 0) & (reason == "")] = "no_samples"
    valid = (reason == "") & np.isfinite(value)
    return assemble("prad", value, valid, reason, fdiv_vote(value))
