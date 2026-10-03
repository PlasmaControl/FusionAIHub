# Detachment

A redundant lower-divertor diagnostic state: attached, detached, MARFE, or
uncertain. This is an **unverified weak label**, awaiting expert review.
Missing intervals are unassessed. Posteriors are uncalibrated; TangTV's
model-implied accuracy, including any bound solution, is not physical accuracy.

## Data Provenance

### Dataset 1

**Dataset File(s)**: `raw/emission_structure_*.sav` and matching `_raw.sav`
inputs; packed inversions are outside git under
`$LABELER_ROOT/round4/detach/inversions/`.

**Author**: Nathaniel Chen (plasma_tv).

**Description**: C-III TangTV inversions on the upper and lower shelves.
The upper shelf is Z≈−1.25 m, R≥1.37 m; the lower shelf is
Z≈−1.363 m, R<1.37 m. Lower-shelf inversion heights use plasma_tv's 2026
`r_max=1.37 m` window and the lower-shelf strike point. The raw SAV video
provides camera examples. The surrogate is evaluated with nested shot holdouts
and intensity normalization; deployment abstains when camera/filter/exposure
provenance is unverified.

**Publications**: Chen et al., Nuclear Fusion 66, 036014 (2026).

### Dataset 2

**Dataset File(s)**: read-only corpus diagnostics and fetched records under
`$LABELER_ROOT/round4/detach/{cache,processed_probes,geometry02,efit}/`.

**Author**: DIII-D diagnostic teams.

**Description**: Processed LANGMUIR Jsat, probe R/Z and millisecond time;
bolometric lower-divertor radiation; beams, ECH, ohmic power, line density,
D-alpha and EFIT geometry/flux/LIM. EFIT02 camera geometry matches plasma_tv;
missing or sparse geometry has a named EFIT01 fallback for inversions only.
Confinement cues use Jalal Butt's tracked source labels. Real-time Thomson
points are not geometrically localized and all old temperature-validation
claims are withdrawn. IRTV heat flux was explicitly probed and returned NODATA;
usable bolometer chord geometry was unavailable.

**Publications**: Eldon et al., NME 18, 285 (2019); NME 27, 100963 (2021);
PPCF 64, 075002 (2022).

### Dataset 3

**Dataset File(s)**: none received.

**Author**: Cheolsik Byun (detachment reference contact).

**Description**: No independent expert table is available. `shots.csv` is an
unverified review roster, not a gold reference.

## Models

**stable**: none.

**latest**: detach_vote | 2026_10_03 (unverified).

**all**:

<!-- MODELS -->

- detach_vote | 2026_10_03: 3862 assessed bins/53 shots; 344 certain/23 shots; 0 confirmed MARFE.
- detach_rule | 2026_10_03: the same observed states with transparent compatible-vote support.
- detach-ours | 2026_10_03: CV 344 bins/23 shots; agreement accuracy 0.762 [0.658, 0.857]; kappa 0.522 [0.279, 0.709]. Reference: unverified combined label, all inversion-sourced.
- detach-victor | 2026_10_03: CV 213 bins/14 shots; agreement accuracy 0.756 [0.484, 0.978]; kappa 0.527 [0.116, 0.950]. Reference: unverified combined label, all inversion-sourced.

These are weak-reference agreement scores. LOO and source-specific results, bootstrap intervals and exact denominators are in the protocol. No primary blind-test or MARFE accuracy is available.

<!-- /MODELS -->

## Inputs

`detach_vote` and `detach_rule` use processed or raw target-current traces,
Prad,div/heating power, true TangTV inversions (or strictly gated surrogate
predictions), EFIT strike/X geometry, filterscope ELM masks, gas-flow timing,
line density, Ip and explicit L/H labels. A self-referenced target-current
trace is named **uncalibrated Jsat ratio (local proxy)**. Separate pre-puff
L/H attached references, when available, are flagged `eldon_pre_puff_LH`.

`detach-ours` reads current, heating power, relative line density and D-alpha,
ELM share, and five global EFIT scalars. It excludes strike/X geometry,
Langmuir currents, bolometry and camera pixels. `detach-victor` reads a raw
TangTV frame; its intensity normalization is fitted within each training fold.

## Method

Each 50 ms bin carries indicator values, validity/reasons and compatible votes.
A certain state needs valid in-domain TangTV agreeing with another cast vote,
no compatible-state conflict, and the chosen state's posterior ≥0.7. The
Afrac+Prad-only agreement is exported as uncertain with
`tier=low_confidence_pair`. Other uncertain tiers distinguish conflict,
low posterior, insufficient votes and candidate MARFE. Invalid geometry never
supplies a cast vote.

Target-current thresholds are ratio ≥0.75 attached / ≤0.5 detached. The
radiation thresholds ≤0.36 attached / ≥0.50 detached are local settings,
validated on cohort-training and external-development inversion bins; Eldon publishes the sensor,
not these universal thresholds. TangTV requires a ≥10 cm leg and valid shelf
geometry; DZ<0.35 votes attached, 0.5≤DZ<1.2 detached. A high front alone is
candidate MARFE. MARFE requires DZ≥1.2 for two adjacent bins, accepted-frame
inversion-peak psiN<1 near/above X, and an independent density-limit or H–L cue.
Unknown density units do not create a Greenwald cue.

The camera ELM mask includes integration and recovery; ELM share >50% invalidates
TangTV. Primary states are unsmoothed. Temporal neighbour-fill suggestions are
separate exports and may not be promoted to observed certainty. Interval
`attrs` includes `tier`; the full bin table has a `tier` column. Confidence is
not calibrated. The rule uses the same compatibility and support conditions.

See [the current protocol](../../../docs/labeler/detachment.md) for complete
gates, fitting, exact contributing populations, source sensitivity, published
settings, results and limitations. Figure 2 coverage input is
`docs/labeler/results/detachment_figure2.json`; figures and `HANDOFF.md` stay
under `$LABELER_ROOT/round4/detach/`.
