"""Every threshold of the detachment label, with where it comes from.

Nothing here was fitted to a score. Each number is read off a published worked
example or a physical definition; the digests cited are in
`.tmp/label_papers/` and the DOIs are in `docs/labeler/detachment.md`. A number
that is a judgement call says so.

Durations that are counted in bins (the length of a reference, of a baseline, of a
persistence) are stated in ms here and converted with `min_bins`, so the bin width
changes the resolution and not how much data a statistic is asked to rest on.
"""

import math


def min_bins(duration_ms: float, width_ms: float) -> int:
    """Bins needed to cover `duration_ms` at `width_ms` (at least one)."""
    return max(1, math.ceil(duration_ms / width_ms - 1e-9))


# --- Afrac (Eldon 2022, PPCF 64 075002; Eldon 2021, NME 27 100963) -----------------
#: Afrac = Jsat / (C <ne>^2 q_par^(-3/7)) = 1 / DOD.  Eldon 2021 reference shot
#: 180257: DOD 1 (attached, 2.4 s), 2 (detaching, 3.6 s), 4 (detached, 4.8 s), i.e.
#: Afrac 1, 0.5, 0.25. The offline Jroll tracker over-estimates the attached level
#: by ~10 %, so "attached" reads 0.8-0.9 there. The margin below is an operational
#: choice, not proof of calibration for the local whole-shot proxy reference.
AFRAC_ATTACHED_MIN = 0.75
#: Detached below Afrac 0.5 (DOD >= 2): the first reference-shot window the paper
#: calls detaching, and the operating point Eldon 2022 controls to. Between 0.5 and
#: 0.75 the strike point is partially detached: the indicator abstains.
AFRAC_DETACHED_MAX = 0.5
#: Each probe's own attached reference for Eldon's C: the shot's own quantile of the
#: probe's model-normalised Jsat over the bins where that probe is near the
#: separatrix (the detachment only ever lowers the ratio, so the upper tail is the
#: attached level). A probe detached through every one of its near-separatrix bins
#: is therefore mis-called attached in its top tail: a stated limitation of this
#: indicator, which is why it never decides alone.
AFRAC_REFERENCE_QUANTILE = 0.90
#: A probe is read, and its reference defined, only where it is within this psiN of
#: the separatrix (|psiN - 1| <= window). The shelf's flux expansion is about 0.5
#: psiN per metre, so 0.01 is about 2 cm of target. Chosen from the probe spacing
#: (1-4 cm) and the width of the target profile, not from a score; 0.015 and 0.02
#: are swept in `docs/labeler/results/detachment_afrac_check.json`. Wider windows
#: admit probes whose psiN is mis-mapped by the EFIT strike-point error.
AFRAC_PSI_WINDOW = 0.01
#: A probe needs at least this much near-separatrix time to define its reference,
#: in ms; the bin count is `min_bins(AFRAC_REFERENCE_MIN_MS, width_ms)`, so a finer
#: or coarser grid asks for the same duration (the bin width is a resolution, not
#: a change in how much data a reference needs).
AFRAC_REFERENCE_MIN_MS = 1000.0


# --- Prad,div (Eldon 2019, NME 18 285; Chen 2026 NF 66 036014) ---------------------
#: Prad,div,L / P_in, with P_in = neutral beams + EFIT ohmic + ECH (never a typed-in
#: power). Eldon 2019 defines the sensor, not classification thresholds, and its
#: digest recommends normalising before thresholding: by the input power, or as a
#: ratio to the shot's own unseeded baseline. Both are used, as follows.
#:
#: SENSITIVITY (absolute) cutoffs are anchored on the one shot whose attached and
#: detached Prad,div,L are published together with its divertor Thomson Te cliffs:
#: Chen 2026 shot 201081 (published Prad,div,L 1.6 MW attached, 2.2 MW detached;
#: Te cliffs at ~2650 and ~4450 ms). The anchor values below are MEASURED on that
#: shot with the labels' own 250 ms ELM-masked averaging, in windows set by the
#: published Te-cliff times and not by TangTV (`detach_prad_anchor.py`, record
#: `docs/labeler/results/detachment_prad_anchor.json`; a test checks agreement).
#: The cutoffs sit BETWEEN the measured attached and detached values: the
#: midpoint plus or minus `PRAD_BAND_MW`, divided by the measured P_in (beams from
#: PTDATA BMSPINJ + EFIT POH + ECH). Nothing is fitted to TangTV, Afrac or any
#: other indicator, so the vote is not circular. They are GLOBAL cutoffs on the
#: ratio (not per shot) and do not carry over from the anchor shot: median f_div
#: in TangTV-attached bins is 0.41-0.58 on most shots (Opus review 5, I1). Neither
#: this vote nor the relative one below is part of the exported label (f_div is a
#: within-shot corroborator, `label_model.SECOND_VOTERS`); both are sensitivities.
PRAD_ANCHOR_SHOT = 201081
PRAD_ANCHOR_ATTACHED_MW = 1.621
PRAD_ANCHOR_DETACHED_MW = 1.990
#: The anchor shot's input power, one number used everywhere: the median of the
#: 250 ms-averaged P_in over the 32 50 ms bins of its attached window (MW).
PRAD_ANCHOR_P_IN_MW = 4.274
#: Half-width of the abstention band around the midpoint, in MW of Prad,div,L.
#: The published operating values are quoted to 0.1 MW, so a band of 0.1 MW keeps
#: every cutoff clear of that rounding. The band is swept in the records
#: (`detach_prad_sensitivity.py`), not tuned.
PRAD_BAND_MW = 0.1


def prad_cutoffs(
    p_in_mw: float = PRAD_ANCHOR_P_IN_MW, band_mw: float = PRAD_BAND_MW
) -> tuple[float, float]:
    """`(attached_max, detached_min)` of f_div = Prad,div,L / P_in.

    The midpoint of the anchor shot's attached and detached Prad,div,L divided by
    its measured input power, minus and plus `band_mw` of radiated power.
    """
    mid = 0.5 * (PRAD_ANCHOR_ATTACHED_MW + PRAD_ANCHOR_DETACHED_MW)
    return (mid - band_mw) / p_in_mw, (mid + band_mw) / p_in_mw


PRAD_ATTACHED_MAX, PRAD_DETACHED_MIN = prad_cutoffs()

#: Relative cutoffs: f_div over the shot's own baseline (the
#: `PRAD_BASELINE_QUANTILE` of its valid f_div, the unseeded level), so a shot's
#: input power and seeding cancel. Same anchor, power-free: the anchor shot's
#: detached/attached ratio is 2.2/1.6, the midpoint 1.9/1.6 and the same band
#: (`PRAD_BAND_MW` / attached MW) either side. This is the f_div vote the data
#: carry (`prad_vote`); the exported label does not use it (it ranks the bins of one
#: shot but is at or below chance pooled over shots). A shot without a baseline (fewer than
#: `PRAD_BASELINE_MIN_MS` of flat-top bins) has no f_div vote: the bin is invalid,
#: reason `no_baseline`.
PRAD_BASELINE_QUANTILE = 0.10
#: A shot needs this much valid time (ms) for a baseline to mean anything; the
#: bin count is `min_bins(PRAD_BASELINE_MIN_MS, width_ms)`.
PRAD_BASELINE_MIN_MS = 2000.0
#: The baseline bins are those at the flat-top input power: at least this fraction
#: of the shot's 90th-percentile P_in (the beam ramps are excluded).
PRAD_BASELINE_POWER_FRACTION = 0.9


def prad_relative_cutoffs(band_mw: float = PRAD_BAND_MW) -> tuple[float, float]:
    """`(attached_max, detached_min)` of f_div / baseline, from the same anchor."""
    mid = 0.5 * (PRAD_ANCHOR_ATTACHED_MW + PRAD_ANCHOR_DETACHED_MW)
    return (
        (mid - band_mw) / PRAD_ANCHOR_ATTACHED_MW,
        (mid + band_mw) / PRAD_ANCHOR_ATTACHED_MW,
    )


PRAD_REL_ATTACHED_MAX, PRAD_REL_DETACHED_MIN = prad_relative_cutoffs()

#: Below this input power the ratio is noise (the bolometer offset, ~0.05 MW, is a
#: tenth of it), and Prad,div cannot be normalised.
MIN_INPUT_POWER_W = 0.5e6
#: Local acausal tau_E-scale averaging choice, not a fitted time constant. Both
#: radiation and heating use the same centered 250 ms window to avoid beam-blip
#: labels. Chen 2026 reports Prad leading DZ by ~50 ms; smoothing is not a lag fix.
PRAD_AVERAGING_MS = 250.0
#: Operational offset tolerance, not a measured calibration uncertainty: 0.05 MW,
#: consistent with the offset scale used above for the input-power floor. More
#: negative native-bin OR 250 ms averaged radiation is invalid, so smoothing
#: cannot hide a bad bin. Small negative offsets are clipped to 0 in the vote.
RADIATION_NEGATIVE_TOL_W = 0.05e6

# --- TangTV DZ (Chen 2026) ---------------------------------------------------------
#: DZ = 1 - (ZX - ZE)/(ZX - ZS): 0 at the strike point (attached), 1 at the X-point.
#: The Te cliff sits at DZ ~ 0.5 (shot 201081). Attached below 0.35; detached from
#: 0.5 to 1.2 under the owner's candidate-MARFE margin; the 0.35-0.5 band is the cliff,
#: where the divertor dithers (Eldon
#: 2017: 2.5 ms jumps), so the indicator abstains rather than call it.
DZ_ATTACHED_MAX = 0.35
DZ_DETACHED_MIN = 0.5
#: Candidate MARFE margin above the X-point. A MARFE vote additionally requires
#: persistence over at least `MARFE_MIN_MS` of adjacent valid bins, psiN<1 at the
#: inversion peak near/above the X-point, and the density cue
#: (`GREENWALD_CUE_MIN`). Prad,div and Afrac do not corroborate a MARFE.
DZ_MARFE_MIN = 1.2
MIN_LEG_M = 0.10
#: DZ below this is unphysical (Chen 2026: "DZ < 0 is unphysical"); half a leg of
#: slack for the 5 mm grid and the EFIT strike-point error.
DZ_UNPHYSICAL_MIN = -0.25
#: Emissivity above which a pixel counts as bright; plasma_tv's EMISSION_THRESHOLD.
EMISSION_THRESHOLD = 0.1

# --- Geometry gate for TangTV (Chen 2026; the owner's rule) ------------------------
#: plasma_tv's regression and its Redge = 1.35 m correction were built on shots whose
#: outer strike point is on the upper shelf (Z = -1.25 m, R > 1.37 m). The
#: lower shelf (Z = -1.363 m, R < 1.37 m) is another geometry; Victor & Scotti 2024 needed a
#: separate model for it. SHELF_WALL_R is plasma_tv's own `SHELF_WALL_R`.
SHELF_WALL_R = 1.37
SHELF_Z = -1.25
SHELF_Z_TOL = 0.05
#: Lower single null: the primary X-point is below this height (m).
LSN_ZX_MAX = -0.5
#: EFIT carries -0.89 (and 0, -9.99) where a point does not exist.
EFIT_SENTINELS = (-0.89, -9.99, 0.0)

# --- Probable L-mode (regime proxy for the TangTV gate) -----------------------------
#: A shot or phase whose regime the confinement table and the D-alpha detector do not
#: give is `probable_L` when its ELM coverage is known, the D-alpha detector flags no
#: ELM anywhere in the window (the bins where the ELM coverage and the input power
#: are both known), and its median input power is below this. The cut sits
#: in the L-H threshold range of the multi-machine scaling (Martin 2008, P_LH =
#: 0.0488 n_e,20^0.717 B_T^0.803 S^0.941 MW), about 1.7-2.4 MW for typical DIII-D
#: parameters (B_T about 2 T, n_e 3-5 x 10^19 m^-3): 2 MW is the middle of that
#: range. P_in (beams + ohmic + ECH) is an upper bound of the power across the
#: separatrix, so below the cut the plasma is under the threshold for most of the
#: range. It is a proxy, not a measurement of the regime. A priori: fixed from the
#: scaling before any Te was looked at.
PROBABLE_L_MAX_P_IN_W = 2.0e6
#: The window needs this much time (ms) of bins with known ELM coverage and a
#: measured input power; the bin count is `min_bins(PROBABLE_REGIME_MIN_MS, width_ms)`.
PROBABLE_REGIME_MIN_MS = 1000.0

# --- Common ------------------------------------------------------------------------
#: A fine-grid sample belongs to an ELM when the divertor D-alpha rises this many
#: robust sigmas over its low-passed baseline; the samples inside ELMs are dropped
#: from Afrac and Prad (Eldon 2017's detector; Leonard 2018: analyse between ELMs).
ELM_SIGMA = 4.0
#: ... and by at least this fraction of the baseline itself: a quiet, noisy D-alpha
#: (a detached or MARFE plasma has none of the ELMs' spikes) has a tiny robust sigma,
#: and noise alone would otherwise be flagged as ELMs.
ELM_MIN_REL_RISE = 0.5
#: A bin with more than this fraction of its samples inside an ELM is invalid; the
#: samples inside ELMs are dropped from the rest (so a bin keeps its inter-ELM time).
MAX_ELM_FRACTION = 0.8
#: Local inter-ELM mask: +/-2 ms around the D-alpha excursion. Camera exposures
#: integrate ELMs as in Chen 2026 and do not reuse this mask as an overlap veto.
ELM_MASK_HALF_WIDTH_MS = 2.0
#: Distance of a probe from the outer strike point, kept only as provenance. The
#: probe is read by its flux position (`AFRAC_PSI_WINDOW`), not by this distance.
#: Every processed LANGMUIR probe is on the outer target (R >= 1.22 m).
#: Density-limit cue of a MARFE; local conservative cue, not a universal MARFE
#: boundary. Spatial evidence and sustained height remain mandatory.
GREENWALD_CUE_MIN = 0.8
#: Time (ms) over which the MARFE evidence must persist, as adjacent valid bins:
#: `min_bins(MARFE_MIN_MS, width_ms)`, two bins at the exported 50 ms width.
MARFE_MIN_MS = 100.0
#: |dIp/dt| above this (MA/s) is a ramp: Afrac's model is not valid (Eldon 2022).
RAMP_DIP_MAX_MA_PER_S = 1.0
