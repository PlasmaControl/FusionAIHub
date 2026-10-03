"""The Afrac indicator: divertor ion saturation current against its attached value.

Eldon 2022 defines `Afrac = Jsat / (C <ne>^2 q_par^(-3/7))`, the measured ion
saturation current at the outer strike point over the value the two-point model
predicts for an attached divertor with the same upstream density and power; it is
1 / DOD of Eldon 2021 (attached 1, detached below 0.5).

What is different here, and why. The corpus holds raw swept-probe records with no
calibration and no probe positions (`langmuir.py`), so:

* Jsat is the PEAK over the live probes of each bin's median inter-ELM ion current (the
  strike point moves along the array; the probe that sees it carries the maximum).
  Between two probes the peak under-reads: a source of false "detached" votes.
* `C` is not Eldon's absolute constant but the shot's own attached level, the
  `AFRAC_REFERENCE_QUANTILE` of the model-normalised Jsat over its valid bins.
  A shot detached throughout is mis-called attached in its top tail.
* `<ne>` is the line-integrated CO2 density (V2 chord) and `P_SOL` is the heating
  power minus dW/dt, core radiation not subtracted; both enter only as ratios.

Votes: attached at or above `AFRAC_ATTACHED_MIN`, detached at or below
`AFRAC_DETACHED_MAX`, abstain between. It never votes MARFE.
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
    bin_median,
    elm_at,
)


def afrac_vote(a: np.ndarray) -> np.ndarray:
    """Attached at or above AFRAC_ATTACHED_MIN, detached at or below
    AFRAC_DETACHED_MAX, abstain between (and where `a` is not finite)."""
    a = np.asarray(a, dtype=float)
    vote = np.full(a.shape, ABSTAIN, dtype=np.int8)
    with np.errstate(invalid="ignore"):
        vote[a >= th.AFRAC_ATTACHED_MIN] = ATTACHED
        vote[a <= th.AFRAC_DETACHED_MAX] = DETACHED
    vote[~np.isfinite(a)] = ABSTAIN
    return vote


def peak_jsat(edges, t_ms, jsat) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin peak over probes of the per-bin median Jsat, and the probe index."""
    per_probe = np.vstack([bin_median(t_ms, row, edges)[0] for row in jsat])
    filled = np.where(np.isfinite(per_probe), per_probe, -np.inf)
    best = filled.argmax(axis=0)
    peak = filled.max(axis=0)
    peak[~np.isfinite(peak)] = np.nan
    return peak, best


def ramp_rate(t_ms, ip_a, edges) -> np.ndarray:
    """|dIp/dt| in MA/s per bin from a (smoothed) plasma-current record."""
    t = np.asarray(t_ms, dtype=float)
    ip = np.convolve(
        np.nan_to_num(np.asarray(ip_a, dtype=float)), np.ones(5) / 5, "same"
    )
    rate = np.abs(np.gradient(ip, t / 1000.0)) / 1e6
    rate[:3] = rate[-3:] = 0.0
    return bin_median(t, rate, edges)[0]


def afrac_indicator(
    edges: np.ndarray,
    probe_t_ms: np.ndarray | None,
    jsat: np.ndarray | None,
    ne_t_ms: np.ndarray | None,
    ne: np.ndarray | None,
    power_t_ms: np.ndarray | None,
    p_sol_w: np.ndarray | None,
    ip_t_ms: np.ndarray | None = None,
    ip_a: np.ndarray | None = None,
    elm_t_ms: np.ndarray | None = None,
    elm_flag: np.ndarray | None = None,
) -> Indicator:
    """Afrac indicator on a bin grid.

    Reasons on an invalid bin: `no_probes`, `no_density`, `no_power`, `low_power`,
    `ramp`, `elm`, `no_samples`, `short_reference` (under AFRAC_MIN_MS of valid bins
    to set the attached level on).
    """
    n = len(edges) - 1
    value = np.full(n, np.nan)
    nothing = np.zeros(n, dtype=bool)
    reason = np.full(n, "no_probes", dtype=object)
    if probe_t_ms is None or jsat is None or len(jsat) == 0:
        return assemble("afrac", value, nothing, reason, np.zeros(n))
    reason[:] = "no_density"
    if ne_t_ms is None or ne is None:
        return assemble("afrac", value, nothing, reason, np.zeros(n))
    reason[:] = "no_power"
    if power_t_ms is None or p_sol_w is None:
        return assemble("afrac", value, nothing, reason, np.zeros(n))

    # sweeps inside an ELM are dropped: the bin's median is the inter-ELM level
    in_elm = elm_at(probe_t_ms, elm_t_ms, elm_flag)
    jsat = np.where(in_elm[None, :], np.nan, jsat)
    peak, _ = peak_jsat(edges, probe_t_ms, jsat)
    density = bin_median(ne_t_ms, ne, edges)[0]
    p_sol = bin_median(power_t_ms, p_sol_w, edges)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = peak / (density**2 * np.power(p_sol, -3.0 / 7.0))
    raw[~np.isfinite(raw)] = np.nan
    ramp = (
        ramp_rate(ip_t_ms, ip_a, edges)
        if ip_t_ms is not None and ip_a is not None
        else np.zeros(n)
    )
    elm_share = (
        bin_fraction(elm_t_ms, elm_flag, edges) if elm_t_ms is not None else np.zeros(n)
    )
    reason[:] = ""
    reason[~(np.nan_to_num(density) > 0)] = "no_density"
    reason[~np.isfinite(p_sol)] = "no_power"
    reason[np.isfinite(p_sol) & (p_sol < th.MIN_INPUT_POWER_W)] = "low_power"
    reason[np.nan_to_num(ramp) > th.RAMP_DIP_MAX_MA_PER_S] = "ramp"
    reason[np.nan_to_num(elm_share) > th.MAX_ELM_FRACTION] = "elm"
    reason[(reason == "") & ~(np.nan_to_num(peak) > 0)] = "no_samples"
    ok = (reason == "") & np.isfinite(raw)
    if ok.sum() * float(edges[1] - edges[0]) < th.AFRAC_MIN_MS:
        reason[ok] = "short_reference"
        return assemble("afrac", value, nothing, reason, np.zeros(n))
    reference = np.quantile(raw[ok], th.AFRAC_REFERENCE_QUANTILE)
    value = raw / reference
    return assemble("afrac", value, ok, reason, afrac_vote(value))
