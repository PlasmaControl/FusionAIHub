# Detachment protocol

This is an **unverified diagnostic consensus**, awaiting the detachment review
stream. It is suitable for studying agreement and coverage; its posterior and
model weights are not calibrated physical probabilities or diagnostic accuracy.
The fixed cohort's test shots are excluded from fitting, threshold validation,
surrogate selection and bin-policy analysis.

## Definitions and export

The scalar lower-divertor state is attached (1), detached (2), MARFE (3), or
uncertain (4). Absence (0 internally) means not assessed; missing time is never
interpreted as attached. Attached means the accepted diagnostics support a
low radiation front and an attached target. Detached means the front has lifted
and another compatible indicator supports reduced target current or enhanced
divertor radiation. MARFE means a persistent high front with localized emission
inside the separatrix near/above the X-point and a separate operational cue.
These definitions express the protocol, not independently established truth.

Bins are 50 ms, following the catalog grid and published diagnostic response
scales. A full TangTV camera frame is approximately 30 Hz; the corpus 50 fps
resampling does not create independent frames. Thus a bin holds roughly 1.5
independent camera frames, and a median alone cannot reject ELM contamination.

`data/events/detachment/extend_detach_vote/detach_shots.csv` contains assessed
intervals with the standard shot/category/start/end/confidence fields. Its
`attrs` JSON includes `tier`. The large `labels_bins.csv.gz` has an explicit
`tier` column, indicator value/valid/reason/vote columns, source and geometry
provenance, posterior columns, and a separate `state_temporal_imputation`.
The owned `shots.csv` is an unverified camera-review roster, not a gold table.
The interval table, sparse grids and per-shot indicator CSVs derive from the
same observed bin labels. Time bases are milliseconds in these outputs; corpus
HDF5 time bases are converted from seconds.

## Indicators and validity

### Target current

The processed path reads `LANGMUIR::TOP.PROBE_*:{R,Z,JSAT,TIME}`. Invalid/zero
positions are excluded. It uses the probe nearest the EFIT outer strike point,
with a maximum distance of 2 cm, and inter-ELM bin medians of processed Jsat
(amps/cm²). Following the Eldon 2021 density-only DOD construction, the attached
reference is `C ne²`, fitted separately for explicit L and H confinement labels
in an attached pre-puff window. The window ends at the first corpus total-gas
rise exceeding 10% of its shot excursion above the tenth-percentile floor.
It needs six quality-valid bins with an attached, valid TangTV vote per regime.
Unknown confinement, missing gas timing, bad geometry, low power, ramping or
ELM contamination cannot fit C. This operational pre-puff recipe is explicit
and should be checked by a human before interpreting C as a physical reference.

If that reference is unavailable, the output is **uncalibrated Jsat ratio
(local proxy)**. Positioned processed traces use a nearest-probe density-scaled
whole-shot 90th-percentile reference; missing processed traces use the original
raw swept-probe maximum with density and power scaling. Neither self-reference
establishes a wholly detached shot's attached current. The `afrac_method` field
separates `eldon_pre_puff_LH` from `local_proxy`; benchmark terminology defaults
to the local proxy. It votes attached at ratio ≥0.75, detached at ≤0.5, and
abstains between. It cannot vote MARFE. The nearest-probe quality and geometry
gates still apply to the processed fallback.

### Divertor radiation

`f_div = PRAD_DIVL / P_in` uses calibrated lower-divertor bolometry and heating
power (beams, ECH and ohmic power). It votes attached at ≤0.36 and detached at
≥0.50, abstaining between, and never votes MARFE. Eldon 2019 supplies the
sensor definition, **not universal state thresholds**. These fixed thresholds
come from the local Chen 201081 worked example and are checked on cohort training and external development
shots with true inversions; they are not optimized on that check.
The validation counts and binary kappa are reported below. Validity requires
input power ≥0.5 MW and acceptable samples/ELM share. The processed-target path
uses density-only scaling; the legacy raw proxy uses P_SOL = P_in − dW/dt,
without subtracting core radiation, another limitation of that fallback.

### TangTV front

C-III emissivity inversions give the SSA height ZE and
`DZ = 1 − (ZX − ZE)/(ZX − ZS)`. The **upper shelf** is near Z=−1.25 m with
outer strike R≥1.37 m. The **lower shelf** is near Z=−1.363 m with R<1.37 m.
Lower-shelf inversions use plasma_tv's 2026 window `RX ≤ R < SHELF_WALL_R`
(`r_max=1.37 m`) and the actual lower-shelf strike height, excluding upper-shelf
emission. All local lower-shelf inversion shots are attempted and checked
against the other indicators below; unlocalized Thomson temperatures are not
used as confirmation.

Both sources require lower-single-null geometry, a positive outer leg of at
least 10 cm, a valid shelf strike, finite emission, DZ≥−0.25, and a nearest
EFIT slice within 40 ms. Camera geometry preferentially uses EFIT02, matching
plasma_tv. Missing or single-slice EFIT02 records have an explicit EFIT01
fallback for inversions, with source exported; the surrogate rejects this
fallback. Heating/current calculations retain their original EFIT01 source.
A bin requires at least half its camera frames to pass its frame gates.

The surrogate is a ridge regression of block-averaged raw frames to inversion
ZE. Features subtract the black level, take square-root block means and apply
per-frame mean-zero/unit-standard-deviation normalization (ZE_Norm), plus RX.
Deployment requires leg length, outer strike R and X-point R inside the
accepted inversion training envelope, training-range brightness, less than 1%
saturation, and verified camera/filter provenance. Camera provenance is verified
only on matching SAV/corpus training shots. Predictions on other shots abstain:
there is no warranted exposure/camera transfer claim. Alpha selection is nested
by shot: each outer held-out shot is excluded from three grouped inner folds,
then evaluated after a fit on the outer training shots. Final deployment alpha
is selected on development data only. The reported error is a surrogate-to-
inversion error, not detachment or MARFE accuracy.

TangTV votes attached for DZ<0.35 and detached for 0.5≤DZ<1.2. The transition
band abstains. DZ≥1.2 alone is a **candidate MARFE**. A MARFE vote needs at least
two consecutive bins satisfying high DZ, accepted-frame spatial evidence and
a second cue. Spatial evidence requires the inversion's global emission peak
to have `0<psiN<1`, within 30 cm radially of X and between ZX−2 cm and ZX+30 cm,
using an EFIT02 flux map within 40 ms. Only frames accepted for DZ contribute;
an excluded ELM frame cannot supply spatial evidence. The second cue is a
recorded H–L back-transition within 200 ms, or fG≥0.8 from unit-confirmed line
density and Ip. Cached UF density lacks confirmed units, so it supplies no
Greenwald cue. The implemented future SI-density chord approximation is not
used to invent present fG values. Surrogate fronts lack spatial inversions and
cannot confirm MARFE. The spatial window is an operational candidate test,
not proof of a localized instability or an expert MARFE label.

### ELM mask

Robust divertor D-alpha spikes are widened by 17 ms before and 50 ms after,
covering half a 30 Hz integration plus a conservative post-ELM recovery window.
Accepted TangTV frames exclude this mask; bins with ELM share >50% are invalid
and cannot become certain through TangTV. Target-current and radiation bins
retain their explicit quality masks. The fixed recovery window is a conservative
protocol assumption and merits shot-specific development validation.

## Redundancy, model and tiers

The label model exactly enumerates three states and indicator propensities,
weights and optional pair correlations. Correlation structure uses shot-grouped
development folds and the simplest structure within one standard error of the
best held-out marginal likelihood. The primary weight fit requests true
inversion anchors with all three indicators valid and a fixed 1/2, 1/4, 1/4
class prior. With fewer than 300 anchors it uses an all-bin fallback, explicitly
unidentified. Weight saturation is recorded. In either case TangTV's physical
accuracy is **not identified**, and all posteriors are **uncalibrated**.

After the posterior, a certain state requires at least two valid cast votes,
a valid in-domain TangTV vote agreeing with another indicator, no incompatible
vote, a posterior-selected state compatible with those votes, and that state's
posterior ≥0.7. `COMPATIBLE` treats a detached target-current/radiation vote as
compatible with a TangTV MARFE vote; an attached vote conflicts with either
not-attached state. Invalid votes do not participate. Thus all available cast
votes must have a common compatible state. Afrac plus Prad alone never produce
a certain state, even when both agree or the posterior is high.

| Tier | Exported state | Meaning |
|---|---|---|
| `certain` | 1/2/3 | All certainty and physical gates pass |
| `low_confidence_pair` | 4 | Target-current/radiation agreement without TangTV support |
| `candidate_marfe` | 4 | High front without the full MARFE evidence/certainty chain |
| `conflict` | 4 | Cast votes have no common compatible state |
| `low_posterior` | 4 | Insufficient compatible vote support or posterior below threshold |
| `no_vote` | 4 | No sufficient supported state |
| `not_assessed` | absent | Fewer than two valid indicators |

Candidate MARFE takes precedence in the tier field; indicator votes still reveal
any conflict. The transparent `detach_rule` uses the same compatibility and
physical support conditions without fitted confidence. Primary states are
**not smoothed**. A one-bin neighbour-fill suggestion is exported separately;
it may cross a validity/confidence boundary and must never be promoted to an
observed certain label. Its count is audited separately.

## Evaluation and current results

<!-- RESULTS -->

The survey selects 474 candidates: 433 cohort shots with real bolo and real TangTV or Langmuir records, union all local inversions and four explicit Eldon/Victor examples. Stubs do not count. The exact union and shot IDs are in [the fix audit](results/detachment_fix.json).

There are 50,032 discharge bins on 470 cached live shots. Eligibility requires at least 20 valid bins in each of two indicators and 20 assessed bins. The export assesses 3,862 bins on 53 eligible shots: 165 attached, 179 detached, 0 MARFE and 3,518 uncertain. The 344 certain bins involve 23 shots. The rule produces the same state counts. [Coverage](../../data/events/detachment/extend_detach_vote/records/coverage.json), [model](../../data/events/detachment/extend_detach_vote/records/label_model.json).

| Indicator | Valid bins / contributing shots |
|---|---:|
| Uncalibrated Jsat ratio (local proxy) | 3,590 / 45 |
| Prad,div local thresholds | 21,408 / 456 |
| TangTV inversion | 1,096 / 28 |

These indicator populations include all cached live shots, before export eligibility. No bin obtained a fitted pre-puff L/H reference; all target-current ratios therefore retain the local-proxy name. The model uses 730 inversion-only anchor bins on 15 development shots. TangTV still has a bound weight solution; it is not evidence of 96% physical accuracy. Mixed versus inversion-only anchors change 0 current states because no surrogate bin supplies an anchor. The old sensitivity is retained in the appendix and audit.

| Assessed tier | Bins |
|---|---:|
| `low_posterior` | 1,380 |
| `conflict` | 950 |
| `low_confidence_pair` | 924 |
| `certain` | 344 |
| `no_vote` | 154 |
| `candidate_marfe` | 110 |

All certainty/conflict/MARFE/ELM invariant violations are zero. There are 60 separate temporal suggestions; they do not alter observed labels.

| TangTV source / inversion envelope | All bins | Valid | Candidate MARFE | MARFE votes / labels |
|---|---:|---:|---:|---:|
| inversion / inside | 1,704 | 598 | 88 | 0 / 0 |
| inversion / outside | 2,813 | 498 | 22 | 0 / 0 |
| surrogate / inside | 1,837 | 0 | 0 | 0 / 0 |
| surrogate / outside | 23,148 | 0 | 0 | 0 / 0 |
| none / inside | 846 | 0 | 0 | 0 / 0 |
| none / outside | 19,684 | 0 | 0 | 0 / 0 |

The upper-shelf surrogate training envelope is a deployment restriction for the surrogate only; true lower-shelf inversions can be outside that envelope while satisfying their own valid geometry. Domain rows use all discharge bins, not just eligible export bins.

All 14 lower-shelf inversion shots were attempted. They contribute 803 lower-geometry bins; 402 bins on 9 shots pass TangTV validity. Their indicator checks are against the other votes with the scored vote withheld:

| Lower-shelf indicator | Cast comparison bins / shots | Binary kappa [95% shot CI] |
|---|---:|---|
| afrac | 48 / 4 | 0.862 [0.155, 1.000] |
| prad | 50 / 5 | 0.783 [0.182, 1.000] |
| tangtv | 52 / 5 | 0.712 [0.059, 1.000] |

The fixed radiation thresholds have 840 valid development inversion bins/27 shots; 202 bins/25 shots cast both binary votes, kappa 0.531 [0.224, 0.764]. The cohort-train subset contributes zero usable pairs after the gates; the check uses the external inversion campaigns already treated as development training inputs. Val/test are excluded. Thresholds were fixed before the check, not selected from it. This is diagnostic agreement, not independently validated classification accuracy.

Nested surrogate LOSO uses 1720 frames on 11 development shots. SAV ZE MAE is 1.234 cm [0.823, 1.828]. Only 1 held-out shot has matched corpus inputs (210 frames), with ZE MAE 3.397 cm and vote kappa 0.000. A single shot cannot establish transfer accuracy; no new surrogate shot passes the camera provenance gate. [Nested fit record](results/detachment_tangtv_surrogate.json).

Single-indicator leave-one-out comparisons use two compatible, valid other votes, with posterior threshold 0.7 and a conflict veto. The TangTV-withheld reference is the low-confidence Jsat/Prad pair, not a primary certain state. Failure analysis uses these references. All intervals below are 1000-replicate shot bootstraps.

| Indicator | Reference bins | Valid reference bins | Cast comparison bins / shots | Binary kappa [95% CI] |
|---|---:|---:|---:|---|
| afrac | 143 | 71 | 62 / 9 | 0.765 [0.161, 1.000] |
| prad | 273 | 273 | 66 / 12 | 0.641 [0.000, 0.881] |
| tangtv | 1013 | 96 | 63 / 10 | 0.736 [0.231, 1.000] |

[Full benchmark](results/detachment_benchmark.json) includes split and source strata, all metric denominators and source-specific failure analysis. Combined-label agreement includes the scored vote and is circular.

| Baseline / CV reference | Bins / shots | Accuracy [95% CI] | Kappa [95% CI] |
|---|---:|---|---|
| detach-ours: combined; all inversion | 344 / 23 | 0.762 [0.658, 0.857] | 0.522 [0.279, 0.709] |
| detach-ours: LOO; inversion only | 223 / 14 | 0.404 [0.221, 0.588] | -0.194 [-0.486, 0.214] |
| detach-ours: LOO; all sources | 974 / 42 | 0.366 [0.210, 0.523] | -0.059 [-0.248, 0.105] |
| detach-victor: combined; all inversion | 213 / 14 | 0.756 [0.484, 0.978] | 0.527 [0.116, 0.950] |
| detach-victor: LOO; inversion only | 78 / 7 | 0.923 [0.721, 1.000] | 0.589 [0.274, 1.000] |
| detach-victor: LOO; all sources | 465 / 21 | 0.716 [0.473, 0.944] | 0.425 [0.113, 0.862] |

The combined-label CV populations are certain labels only. For LOO evaluation the full assessed datasets retain 3862 windows/53 shots and 1982 frames/29 shots. Surrogate-source combined-label and blind-test combined-label populations are both empty. There is no MARFE training/evaluation class; its score is undefined (JSON null), not zero. Fixed-epoch CNNs train only on certain labels. Normalization is learned within each fold, and majority-class predictions are selected within each training fold. [Ours](results/detachment_ours.json), [Victor-style adaptation](results/detachment_victor.json).

Blind-test LOO agreement is supported by only 39 bins/2 shots for ours and 28 bins/1 shot for Victor. These weak-reference populations do not support a gold-test accuracy claim.

Bin sensitivity uses 77 non-test shots, excluding 192154, 196153, 201107, 202188. It is descriptive, not a test-driven choice of bin width.

| Width (ms) | All bins | Assessed bins / shots | Certain bins / shots |
|---|---:|---:|---:|
| 20 | 20,726 | 2694 / 21 | 491 / 17 |
| 50 | 8,327 | 1283 / 23 | 222 / 16 |
| 100 | 4,180 | 635 / 23 | 105 / 13 |

[Width record](results/detachment_bin_sensitivity.json). Comparison agreement applies only to bins certain at both resolutions; it does not measure preservation of coverage or uncertain intervals.

<!-- /RESULTS -->

## Figures and reproduction

`scripts/labeler/detach_figure.py` writes two appendix figures at 6.75-inch
text width: side-by-side camera and inversion views, and indicator
traces. EFIT flux surfaces and the fetched g-file LIM vessel/divertor wall
appear on inversion panels. The Jsat ratio axis is logarithmic and does not clip
excursions. Minimum font size is 7 pt at that width; neither figure has a global
title. Candidate high-front examples are labelled uncertain when appropriate.

The bolometer row is omitted because usable chord geometry was not found in
the surveyed BOLOM tree or local resources; chord index is not a spatial ray.
The explicit `IRTV::HEATFLUX` read returned NODATA and is recorded in the fetch
audit, so no invented heat-flux panel is shown. `detach_paper_panel.py` supplies
a separate legacy-vs-Tokamak-SI coverage panel; its small input is
`results/detachment_figure2.json`. Coverage is not accuracy. The controller can
place it in paper Figure 2 without this stream editing the manuscript.

Regenerate with the committed `detach_tv_surrogate.py`, `detach_bins.py`,
`detach_label.py`, `detach_benchmark.py`, `detach_fix_records.py`,
`detach_sensitivity.py`, `detach_ours.py`, `detach_victor.py`, and figure scripts.
All large artifacts stay under `$LABELER_ROOT/round4/detach/`. The external
`HANDOFF.md` describes time units, columns and review paths. Only the allowed
login-node fdp fetches read remote diagnostics, with a shared authentication
stop flag, at most three fetch workers and one-second pacing.

## Limitations

There is no expert-reviewed reference, identified TangTV accuracy, posterior
calibration or validated campaign transfer. Proxy-pair agreement remains weak
evidence even in leave-one-out evaluation. The processed probe fetch does not
guarantee a usable attached pre-puff reference. Geometric fallbacks and sparse
camera sampling reduce coverage. MARFE absence in the revised export cannot be
read as proof that no MARFE occurred. Processed DTS with chord coordinates,
psiN>1, position below X and Te≤100 eV would support a new temperature check;
all former unlocalized Te medians/AUROCs and cold-target claims are withdrawn.
Priorities are expert review of inversion examples, diagnostic unit/camera
metadata verification, attached-reference validation and calibrated uncertainty.

## Appendix: fix history

The original primary export used surrogate anchors, allowed proxy-pair certainty,
followed TangTV across conflicts and short legs, and smoothed after confidence
thresholding. Its source provenance was 919 inversion anchor bins/19 shots and
1,638 surrogate bins/21 shots, rather than 40 inversion shots. Reproducing the
original rule and smoothing after an inversion-only refit changes
16,245/24,786 original assessed labels. This is a sensitivity diagnostic for the
superseded model, not a comparison to truth. Source-stratified counts and the
old/new weight fits are preserved in `results/detachment_fix.json`.

The old baseline datasets contained 10,289 windows/257 shots and 5,877
frames/107 shots; the actual non-test CV populations were 9,400/231 and
5,435/99. Old leave-one-out contributing shot counts were 41, 41 and 36,
with one test contributor each. These historical denominators are recorded in
that same audit; they must not be attached to the revised results.

## References

- Chen et al., *Nuclear Fusion* 66, 036014 (2026),
  [doi:10.1088/1741-4326/ae3972](https://doi.org/10.1088/1741-4326/ae3972).
- Eldon et al., *Nuclear Materials and Energy* 27, 100963 (2021),
  [doi:10.1016/j.nme.2021.100963](https://doi.org/10.1016/j.nme.2021.100963).
- Eldon et al., *Plasma Physics and Controlled Fusion* 64, 075002 (2022),
  [doi:10.1088/1361-6587/ac6ff9](https://doi.org/10.1088/1361-6587/ac6ff9).
- Victor and Scotti, *Review of Scientific Instruments* 95, 083503 (2024),
  [doi:10.1063/5.0218724](https://doi.org/10.1063/5.0218724).

These references motivate diagnostic constructions; our operational gates and
thresholds, restricted coverage and redundant weak label are stated separately.
