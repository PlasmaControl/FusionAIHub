"""Uncalibrated Jsat ratio (local proxy), not published Eldon Afrac.

The Afrac indicator: divertor ion saturation current against its attached value.

Eldon 2022 defines `Afrac = Jsat / (C <ne>^2 q_par^(-3/7))`, the measured ion
saturation current at the outer strike point over the value the two-point model
predicts for an attached divertor with the same upstream density and power; it is
1 / DOD of Eldon 2021 (attached 1, detached below 0.5).

What this module measures, and why it is built this way. The target current falls
steeply with distance from the separatrix and the probes are 1-4 cm apart, so a
current read from "the probe the selection rule happened to pick" measures that
probe's flux position as much as the plasma (Opus review 5, C1: the earlier peak
probe inside 1.000 < psiN <= 1.05 with one whole-shot reference read 0.79 near
the separatrix and 0.10 further out for the same TangTV state). Two changes
remove that dependence:

* each probe has its OWN attached reference: the `AFRAC_REFERENCE_QUANTILE` of
  its own model-normalised current over the bins where it is within
  `AFRAC_PSI_WINDOW` of the separatrix (and ELM-, ramp- and L-mode-free), so a
  probe's position and calibration cancel; a probe with fewer than
  `AFRAC_REFERENCE_MIN_MS` of such bins has no reference and does not vote;
* the probe read on a bin is the one nearest the separatrix in flux
  (`|psiN - 1|` smallest) among those with a reference, inside the same window,
  rather than the largest current. The strike-point probe is the one that shows
  the rollover, and the old outboard margin excluded it.

Differences from published Afrac remain:

* `C` is not Eldon's fitted attached-current constant but a per-probe upper-tail
  level of the probe's own near-separatrix bins. There is no literature-backed
  minimum-duration gate: a probe that stays detached through every bin it is
  within the window is mis-called attached in its top tail.
* The window is one decision (`AFRAC_PSI_WINDOW`), chosen from the shelf's flux
  expansion (about 0.5 psiN per metre, so 0.01 is 2 cm) and swept in the records,
  not tuned to a score; wider windows admit probes whose psiN is mis-mapped by
  the EFIT strike-point error.
* `<ne>` is the line-integrated CO2 density (V2 chord) and `P_SOL` is the heating
  power minus dW/dt, core radiation not subtracted; both enter only as ratios. The
  density is the bin median; `P_SOL`, which sits on the 20-25 ms EFIT time base, is
  interpolated to the bin centre, so a 20 ms bin is not left without a sample.
* Eldon 2022's model needs L-mode excluded: bins in a known L-mode stretch
  abstain (reason `l_mode`), and do not enter any reference. Bins where the regime
  is not known are not gated (the caller records which).

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
    elm_bin_known,
    interp_at_centres,
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
    jsat: np.ndarray | None,
    psi_n: np.ndarray | None,
    ne_t_ms: np.ndarray | None,
    ne: np.ndarray | None,
    power_t_ms: np.ndarray | None,
    p_sol_w: np.ndarray | None,
    ip_t_ms: np.ndarray | None = None,
    ip_a: np.ndarray | None = None,
    elm_t_ms: np.ndarray | None = None,
    elm_flag: np.ndarray | None = None,
    lmode: np.ndarray | None = None,
    *,
    window: float = th.AFRAC_PSI_WINDOW,
    min_bins: int | None = None,
) -> tuple[Indicator, np.ndarray, np.ndarray]:
    """Afrac indicator on a bin grid: `(indicator, probe index per bin, references)`.

    `jsat` is (probe, bin), the per-bin median inter-ELM current of each positioned
    probe (the caller drops ELM samples before binning); `psi_n` is (probe, bin),
    each probe's normalised flux on the bin. `lmode` is a boolean per bin, True
    where the shot is known to be in L-mode (those bins abstain and are left out
    of every reference); None gates nothing. The index is the probe read on the
    bin (-1 where none), the references are each probe's attached level in the
    model-normalised current (NaN for a probe with too few near-separatrix bins).

    Reasons on an invalid bin: `no_probes`, `no_density`, `no_power`, `low_power`,
    `ramp`, `elm`, `l_mode`, `elm_unknown`, `probe_flux_unknown` (no probe has a
    flux value), `probe_off_separatrix` (none is within `window`),
    `no_probe_samples` (some are, none has a current in the bin) and
    `short_reference` (a probe is in the window with a current, but none has
    `min_bins` bins to define its reference).
    `min_bins` defaults to `AFRAC_REFERENCE_MIN_MS` of bins at the grid's own width.
    """
    n = len(edges) - 1
    if min_bins is None:
        width_ms = float(np.median(np.diff(edges)))
        min_bins = th.min_bins(th.AFRAC_REFERENCE_MIN_MS, width_ms)
    value = np.full(n, np.nan)
    nothing = np.zeros(n, dtype=bool)
    none_read = np.full(n, -1)
    reason = np.full(n, "no_probes", dtype=object)
    if jsat is None or psi_n is None or len(jsat) == 0:
        return (
            assemble("afrac", value, nothing, reason, np.zeros(n)),
            none_read,
            np.array([]),
        )
    jsat, psi_n = np.asarray(jsat, dtype=float), np.asarray(psi_n, dtype=float)
    references = np.full(len(jsat), np.nan)
    reason[:] = "no_density"
    if ne_t_ms is None or ne is None:
        return (
            assemble("afrac", value, nothing, reason, np.zeros(n)),
            none_read,
            references,
        )
    reason[:] = "no_power"
    if power_t_ms is None or p_sol_w is None:
        return (
            assemble("afrac", value, nothing, reason, np.zeros(n)),
            none_read,
            references,
        )

    density = bin_median(ne_t_ms, ne, edges)[0]
    p_sol = interp_at_centres(power_t_ms, p_sol_w, edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = jsat / (density**2 * np.power(p_sol, -3.0 / 7.0))[None, :]
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
    if lmode is not None:
        reason[np.asarray(lmode, dtype=bool) & (reason == "")] = "l_mode"
    reason[~elm_bin_known(edges, elm_t_ms, elm_flag)] = "elm_unknown"
    gated = reason == ""

    known = np.isfinite(psi_n)
    near = known & (np.abs(psi_n - 1.0) <= window)
    sampled = near & np.isfinite(raw) & (jsat > 0)
    candidate = sampled & gated[None, :]
    level = np.full(raw.shape, np.nan)
    for i in range(len(jsat)):
        if candidate[i].sum() >= min_bins:
            references[i] = np.quantile(
                raw[i, candidate[i]], th.AFRAC_REFERENCE_QUANTILE
            )
            if references[i] > 0:
                level[i, candidate[i]] = raw[i, candidate[i]] / references[i]
    distance = np.where(np.isfinite(level), np.abs(psi_n - 1.0), np.inf)
    which = distance.argmin(axis=0)
    ok = np.isfinite(distance.min(axis=0))
    column = np.arange(n)
    value = np.where(ok, level[which, column], np.nan)
    why = np.full(n, "short_reference", dtype=object)
    why[~sampled.any(axis=0)] = "no_probe_samples"
    why[~near.any(axis=0)] = "probe_off_separatrix"
    why[~known.any(axis=0)] = "probe_flux_unknown"
    reason = np.where(gated & ~ok, why, reason)
    return (
        assemble("afrac", value, ok & gated, reason, afrac_vote(value)),
        np.where(ok & gated, which, -1),
        references,
    )


def reported_probe(psi_n, which, valid):
    """The probe a bin's provenance describes, valid or not.

    The probe read where the bin is valid; otherwise the probe nearest the
    separatrix in flux with a known value, so an invalid bin still says how far
    from the separatrix the closest probe was. -1 where there is none.
    """
    psi_n = np.asarray(psi_n, dtype=float)
    distance = np.where(np.isfinite(psi_n), np.abs(psi_n - 1.0), np.inf)
    nearest = np.where(np.isfinite(distance).any(axis=0), distance.argmin(axis=0), -1)
    nearest = np.where(np.isfinite(distance.min(axis=0)), nearest, -1)
    return np.where(np.asarray(valid), which, nearest)
