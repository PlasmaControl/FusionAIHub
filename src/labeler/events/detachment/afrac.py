"""Uncalibrated Jsat ratio (local proxy), not published Eldon Afrac.

The bin exporter requires a positioned SOL-side processed probe. Its attached
reference remains a whole-shot quantile, independent of the camera votes.

The Afrac indicator: divertor ion saturation current against its attached value.

Eldon 2022 defines `Afrac = Jsat / (C <ne>^2 q_par^(-3/7))`, the measured ion
saturation current at the outer strike point over the value the two-point model
predicts for an attached divertor with the same upstream density and power; it is
1 / DOD of Eldon 2021 (attached 1, detached below 0.5).

The bin exporter selects a positioned processed SOL-side probe before calling
this numerical helper. Unpositioned corpus sweeps cannot establish that provenance
and do not vote. Differences from published Afrac remain:

* The exporter supplies the nearest qualified SOL-side probe's median inter-ELM
  current. This helper can also accept a probe array, using its per-bin peak;
  such an array must have passed independent position/flux selection upstream.
  A finite probe spacing can under-read the current and bias detached votes.
* `C` is not Eldon's fitted attached-current constant but a local proxy level, the
  `AFRAC_REFERENCE_QUANTILE` of the model-normalised Jsat over its valid bins.
  At least 3 s of eligible samples are needed. A shot detached throughout can
  be mis-called attached in its top tail.
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
    elm_bin_known,
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
    *,
    pre_masked: bool = False,
) -> Indicator:
    """Afrac indicator on a bin grid.

    Reasons on an invalid bin: `no_probes`, `no_density`, `no_power`, `low_power`,
    `ramp`, `elm`, `elm_unknown`, `no_samples`, `short_reference` (under
    AFRAC_MIN_MS of valid bins
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
    if not pre_masked:
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
    reason[~elm_bin_known(edges, elm_t_ms, elm_flag)] = "elm_unknown"
    ok = (reason == "") & np.isfinite(raw)
    if ok.sum() * float(edges[1] - edges[0]) < th.AFRAC_MIN_MS:
        reason[ok] = "short_reference"
        return assemble("afrac", value, nothing, reason, np.zeros(n))
    reference = np.quantile(raw[ok], th.AFRAC_REFERENCE_QUANTILE)
    value = raw / reference
    return assemble("afrac", value, ok, reason, afrac_vote(value))


def select_sol_probe(jsat, positions, strike, psi_n):
    """Nearest finite positive outer-target current safely on the SOL side.

    Require R_probe - R_strike >= 5 mm, distance <= 2 cm and psiN >= 1.01.
    The local position/flux guard bands are explicit uncertainty margins, not
    a claim that the EFIT/probe positions were independently calibrated.
    Returns selected index (-1 if none), validity and the abstention reason.
    """
    positions, strike = np.asarray(positions), np.asarray(strike)
    psi_n, jsat = np.asarray(psi_n), np.asarray(jsat)
    distance = np.linalg.norm(positions[:, None, :] - strike[None, :, :], axis=2)
    close = np.isfinite(distance) & (distance <= th.PROBE_MAX_DISTANCE_M)
    sol = positions[:, None, 0] - strike[None, :, 0] >= th.PROBE_STRIKE_MARGIN_M
    sol &= np.isfinite(psi_n) & (psi_n >= th.PROBE_SOL_PSI_N_MIN)
    usable = close & sol & np.isfinite(jsat) & (jsat > 0)
    ranked = np.where(usable, distance, np.inf)
    valid = usable.any(axis=0)
    which = np.where(valid, ranked.argmin(axis=0), -1)
    reason = np.full(len(which), "", object)
    reason[~valid] = "no_probe_samples"
    reason[~(close & sol).any(axis=0)] = "probe_not_sol"
    reason[~(close & np.isfinite(psi_n)).any(axis=0)] = "probe_flux_unknown"
    reason[~close.any(axis=0)] = "probe_far_from_strike"
    return which, valid, reason
