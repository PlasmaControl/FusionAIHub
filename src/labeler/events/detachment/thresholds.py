"""Every threshold of the detachment label, with where it comes from.

Nothing here was fitted to a score. Each number is read off a published worked
example or a physical definition; the digests cited are in
`.tmp/label_papers/` and the DOIs are in `docs/labeler/detachment.md`. A number
that is a judgement call says so.
"""

# --- Afrac (Eldon 2022, PPCF 64 075002; Eldon 2021, NME 27 100963) -----------------
#: Afrac = Jsat / (C <ne>^2 q_par^(-3/7)) = 1 / DOD.  Eldon 2021 reference shot
#: 180257: DOD 1 (attached, 2.4 s), 2 (detaching, 3.6 s), 4 (detached, 4.8 s), i.e.
#: Afrac 1, 0.5, 0.25. The offline Jroll tracker over-estimates the attached level
#: by ~10 %, so "attached" reads 0.8-0.9 there; the reference level here is set the
#: same way (on the shot's own attached window), so the same margin applies.
AFRAC_ATTACHED_MIN = 0.75
#: Detached below Afrac 0.5 (DOD >= 2): the first reference-shot window the paper
#: calls detaching, and the operating point Eldon 2022 controls to. Between 0.5 and
#: 0.75 the strike point is partially detached: the indicator abstains.
AFRAC_DETACHED_MAX = 0.5
#: The uncalibrated probes (no gain, no position on disk) cannot give Eldon's
#: absolute C. The attached reference is the shot's own: this quantile of the
#: model-normalised Jsat over its valid bins (the detachment only ever lowers the
#: ratio, so the upper tail is the attached level). A shot detached throughout is
#: therefore mis-called attached in its top tail: a stated limitation of this
#: indicator, which is why it never decides alone.
AFRAC_REFERENCE_QUANTILE = 0.90
#: Least valid time (ms, any bin width) the reference quantile may rest on.
AFRAC_MIN_MS = 3000.0

# --- Prad,div (Eldon 2019, NME 18 285; Chen 2026 NF 66 036014) ---------------------
#: The indicator is f_div = Prad,div,L / P_in, the radiated power below the X-point
#: (BOLOM PRAD_DIVL, calibrated post-shot) over the heating power (beams + ohmic +
#: ECH). Normalised by input power, not by a baseline, because a baseline fails for
#: a shot that starts detached, and Prad,div scales with the power fed to the
#: boundary (Eldon 2019, 2.1). Chen 2026 quotes Prad,div,L 1.6 MW attached -> 2.2 MW
#: detached across 201081 (P_in 4.4 MW incl. ohmic: f_div 0.36 -> 0.50) and 1.8 ->
#: 2.4 MW for 201085. The thresholds are those two end states rounded outward, so a
#: bin has to look at least as attached / detached as the published worked example:
#: between them the indicator abstains. Prad is a radiation measure, not a detachment
#: measure (it rises before the Te cliff and stays up after reattachment; the
#: post-shot estimate is good to ~20 %), so it is a weak voter and abstains in a wide
#: band. It never votes MARFE: the bolometer integrals carry no position.
PRAD_ATTACHED_MAX = 0.35
PRAD_DETACHED_MIN = 0.50
#: Below this input power the ratio is noise (the bolometer offset, ~0.05 MW, is a
#: tenth of it), and Prad,div cannot be normalised.
MIN_INPUT_POWER_W = 0.5e6

# --- TangTV DZ (Chen 2026) ---------------------------------------------------------
#: DZ = 1 - (ZX - ZE)/(ZX - ZS): 0 at the strike point (attached), 1 at the X-point.
#: The Te cliff sits at DZ ~ 0.5 (shot 201081). Attached below 0.35; detached from
#: 0.5 to 1.0; the 0.35-0.5 band is the cliff, where the divertor dithers (Eldon
#: 2017: 2.5 ms jumps), so the indicator abstains rather than call it.
DZ_ATTACHED_MAX = 0.35
DZ_DETACHED_MIN = 0.5
#: DZ > 1: the C-III emission centroid is above the X-point: MARFE territory (Chen
#: 2026; Leonard 2018 stage 5).
DZ_MARFE_MIN = 1.0
#: DZ below this is unphysical (Chen 2026: "DZ < 0 is unphysical"); half a leg of
#: slack for the 5 mm grid and the EFIT strike-point error.
DZ_UNPHYSICAL_MIN = -0.25
#: Emissivity above which a pixel counts as bright; plasma_tv's EMISSION_THRESHOLD.
EMISSION_THRESHOLD = 0.1

# --- Geometry gate for TangTV (Chen 2026; the owner's rule) ------------------------
#: plasma_tv's regression and its Redge = 1.35 m correction were built on shots whose
#: outer strike point is on the lower divertor SHELF (Z = -1.25 m, R > 1.37 m). The
#: floor (Z = -1.363 m, R < 1.37 m) is another geometry; Victor & Scotti 2024 needed a
#: separate model for it. SHELF_WALL_R is plasma_tv's own `SHELF_WALL_R`.
SHELF_WALL_R = 1.37
SHELF_Z = -1.25
SHELF_Z_TOL = 0.05
#: Lower single null: the primary X-point is below this height (m).
LSN_ZX_MAX = -0.5
#: EFIT carries -0.89 (and 0, -9.99) where a point does not exist.
EFIT_SENTINELS = (-0.89, -9.99, 0.0)

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
#: |dIp/dt| above this (MA/s) is a ramp: Afrac's model is not valid (Eldon 2022).
RAMP_DIP_MAX_MA_PER_S = 1.0
