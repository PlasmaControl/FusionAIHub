# Detachment protocol

<!-- SUMMARY -->

**Exploratory coverage and indicator agreement; no independent benchmark.** Certain labels use two indicators in practice: upper-shelf C-III front height and local Prad,div/P_in corroboration. They do not establish physical state accuracy.

The export has 1,737 assessed bins on 36 shots, including 307 certain bins on 16 shots: 0 attached on 0 shots, 300 detached on 16 shots, and 7 MARFE on 1 shot(s). No shot has both certain attached and detached bins; the export cannot support a study of attached-to-detached transitions.

Afrac is not reproduced: EFIT flux maps cover 42 shots versus 404 positioned-probe files; 33,044 bins have unknown probe flux. No upper-shelf SOL probe passes the position/flux gate (0 bins). The local Jsat proxy is uncalibrated and casts votes on 235 bins only (99 lower-shelf, 136 without TangTV). Its former 3000 ms minimum-reference gate had no source and was removed. The Snorkel model is vestigial: no three-source upper-shelf anchors identify its accuracies, and it never decides certainty.

Upper-shelf Prad–TangTV agreement uses 356 cast pairs on 17 shots: three-state κ 0.000 [0.000, 0.000]; binary attached/not-attached κ 0.000 [0.000, 0.000]. The 95% intervals resample shots. Raw agreement is 0.843 [0.628, 0.973]. Prad casts only 'detached' on the upper shelf, so κ is zero by construction and agreement cannot establish discrimination between states; its 105 attached votes all fall on bins without TangTV, so none can make an attached label certain. In the 49 pairs where TangTV votes attached, Prad votes detached in 49 (100%). Lower-shelf measurements are reported separately as provisional.

Normalising Prad,div and P_in over a centered 250 ms window (instead of the 50 ms bin) and repairing the missing-data and negative-radiation gates moved the certain set from 321 bins (11 attached on 3 shots, 298 detached, 12 MARFE) to 307 (0 attached, 300 detached, 7 MARFE). None of the earlier attached bins survives. The change combines all repairs; it is not an isolated ablation.

The certain composition is set by the Prad cutoffs: attached/detached/MARFE bins are 0/499/23 at -0.10, 0/465/23 at -0.05, 0/300/7 as exported, 38/80/0 at +0.05 and 241/2/0 at +0.10. Certainty therefore means a TangTV vote plus f_div on the same side of one local, single-shot-derived cutoff; it is not a threshold-independent state.

MARFE bins are **single-shot MARFE candidates within threshold uncertainty** on 199172: fG=0.804–0.866 against a 0.8 cue and f_div=0.501–0.511 against 0.50. Chord-based fG has about 10–20% geometric uncertainty. The published MARFE on 199166 (3.705 s) is missed: its onset bin fails the spatial gate and fG=0.721 is below the 0.8 cue. Shifting the fG cue by ±0.1 changes the MARFE bins from 7 to 17 (0.70) or 0 (0.90); no threshold is selected from this.

The Prad cutoffs 0.36/0.50 come from one worked example in Chen 2026 (shot 201081; Chen reports 4 MW NBI, while 4.4 MW total heating was assumed here and is unsourced). The 201081 `pinj` fetch failure is a PTDATA client configuration error, not missing data.

<!-- /SUMMARY -->

The fixed 500-shot cohort's test split is excluded from fitting, threshold
selection, surrogate selection and bin-policy analysis.

## Definitions and export

The lower-divertor states are attached (1), detached (2), MARFE (3), and uncertain
(4). Missing intervals are unassessed, never attached. Attached requires a low
accepted C-III front and another compatible vote. Detached requires a lifted
front and support from target current or divertor radiation. MARFE additionally
requires persistent high DZ, spatially localized emission inside the separatrix
near/above X, and an independent operational cue. These are operational definitions.

Bins are 50 ms, following the catalog grid and diagnostic response scales.
The 30 Hz camera supplies about 1.5 independent frames per bin. Chen 2026 explicitly
integrates ELMs over these exposures; resampling does not supply extra independent
frames. This implementation accepts integrated frames and documents that choice.

`data/events/detachment/extend_detach_vote/detach_shots.csv` retains the standard
shot/category/start/end/confidence/attrs schema. `attrs` carries `tier`.
`$LABELER_ROOT/round4/detach/labels_bins.csv.gz` contains values, votes, validity
reasons, geometry and probe provenance. The legacy `state_lm` column now aliases
the primary `state_rule`; the fitted model lives in `state_model_diagnostic`.
`state_lower_shelf_window` is a separate provisional suggestion, never a certain
primary label. `state_temporal_imputation` also remains separate. Rule confidence
is null; diagnostic posteriors retain their named columns. Sparse grids and
`indicators/<shot>.csv` derive from the same primary rule. Outputs use milliseconds;
corpus HDF5 xdata uses seconds. `shots.csv` is an unverified review roster.

## Indicators and validity

### Target current

The processed path uses positioned `LANGMUIR::TOP.PROBE_*:{R,Z,JSAT,TIME}` data.
Only a SOL-side probe at least 5 mm outboard of the strike point, within 2 cm,
and with normalized flux ≥1.01 may vote. The 5 mm and 0.01 margins are explicit
operational allowances for EFIT position uncertainty, not fitted error bars.
The chosen probe,
coordinates, strike geometry, distance, normalized flux, flux-map source and
selection validity are exported per bin. Exact margins and sample counts are in
`thresholds.py` and the fix audit. A probe without usable position/flux evidence
abstains; private-flux current cannot corroborate detachment.

The retained indicator is an **uncalibrated Jsat ratio (local proxy)**: inter-ELM
current times P_SOL^(3/7), divided by relative density squared, then normalized
to its own shot's 90th-percentile reference. A reference requires eligible
SOL-side samples; no arbitrary minimum duration is imposed. This follows Eldon's
density/power scaling,
but it does not
reproduce a separately fitted attached pre-puff L/H reference or identify attached
current in an entirely detached shot. Current construction does not fit against
TangTV-attached bins, preserving independence of the current construction from TangTV.
Ratio ≥0.75 votes attached, ≤0.5 detached, and the transition band abstains.
It cannot distinguish detached from MARFE. `afrac_method=local_proxy` preserves the
historical field name without claiming the published Afrac setting was reproduced.

### Divertor radiation

`f_div = PRAD_DIVL / P_in` uses calibrated lower-divertor radiation and beam,
ECH and ohmic heating. The numerator and denominator are averaged over the same
centered **250 ms** window before taking their ratio, to avoid interpreting
beam blips as attached states. Heating coverage must be complete, and radiation
uses covered inter-ELM samples. D-alpha availability must cover the full 250 ms
radiation window, including samples outside the native 50 ms label bin;
`aux_prad_elm_window_known` records that gate. Radiation below −0.05 MW in either the native
50 ms label-bin mean or the 250 ms mean is invalid; a tolerated 250 ms mean between
−0.05 MW and zero is clipped to zero in the ratio. That tolerance is an
operational allowance,
not a calibrated error bar. Missing beam samples remain unknown, never zero power.
The ratio votes attached at ≤0.36, detached at ≥0.50, and abstains
between. Eldon 2019 supplies the diagnostic construction, not universal state
thresholds. These thresholds are fixed local settings motivated by Chen's 201081
worked example. Chen reports **4 MW NBI** and radiation of 1.6 → 2.2 MW. The
local thresholds instead assumed **4.4 MW total heating**: 1.6/4.4≈0.36 and
2.2/4.4=0.50. The extra 0.4 MW is not independently sourced, so these are
exploratory local settings, not published calibrated thresholds. The failed
201081 `pinj` fetch reports `PTSERVER/ptserver` absent from `/etc/services`, a
PTDATA client configuration error; it does not establish absent experimental
data. Their check uses external development inversion shots; the fixed
cohort-train subset has **zero usable upper-shelf reference bins**. These checks
measure agreement with the imaging indicator. Input power must be ≥0.5 MW with
valid samples
and acceptable inter-ELM coverage. No radiation vote resolves MARFE.

### TangTV front and geometry

The inversion supplies the SSA emission height ZE and
`DZ = 1 - (ZX - ZE)/(ZX - ZS)`. The upper shelf is near Z=−1.25 m, outer strike
R≥1.37 m. The lower shelf is near Z=−1.363 m, R<1.37 m. The owner's
`plasma_tv/python/make_labels_2026.make_labels_for_file(...,
r_max=SHELF_WALL_R=1.37)` explicitly supports the restricted lower-shelf window;
`compare_labels_2026.py` uses it as well. That extraction is retained, but every
lower-shelf primary bin remains uncertain in the **`lower_shelf_window` tier**, pending
owner sign-off. Its votes and provisional states are reported separately.

Inversion frames require lower-single-null shelf geometry, a positive outer leg
≥10 cm, finite emission, DZ≥−0.25, and a nearest EFIT slice within 40 ms. At least
half the frames in a bin must pass. Camera geometry prefers EFIT02; missing or
single-slice EFIT02 has an explicit EFIT01 fallback. The surrogate rejects that
fallback. Heating/current remain EFIT01. Accepted-frame geometry determines each
bin's shelf provenance.

TangTV votes attached for DZ<0.35 and detached for 0.5≤DZ<1.2. The intermediate
band abstains. DZ≥1.2 is candidate MARFE until all evidence gates pass for two
consecutive bins. Spatial evidence uses a close multi-slice EFIT02 map, or a
named EFIT01 fallback, and the global inversion peak with `0<psiN<1`, within
30 cm radially of X and ZX−2 cm to ZX+30 cm vertically. The second cue is a recorded
H–L back-transition within 200 ms or fG≥0.8. Density units must be confirmed from
source metadata: `fG = nbar_e / nG`, `nG = Ip(MA)/(pi*a(m)^2)` in 10²⁰ m⁻³.
The V2 line integral uses its confirmed `m/cm3` units, converted to m⁻²;
an elliptical chord length at R=1.94 m estimates the line average. This neglects
triangularity and has about 10–20% geometric uncertainty, so fG is a corroborating
cue, not MARFE truth.
A missing or unit-ambiguous density cannot produce that cue. The provenance and
explicit check of Chen's published MARFE on 199166 at 3.705 s appear below.
A surrogate without a spatial inversion cannot confirm MARFE.

The ridge surrogate has nested shot-held-out ZE checks against inversion heights.
Camera/filter/exposure provenance and training geometry/brightness gates prevent
unsupported deployment. It contributes **zero valid bins** in this export and
therefore extends neither certainty nor cohort coverage. Its training error is
an inversion-surrogate error, not detachment accuracy.

### ELM coverage

Prad and Jsat use their own **±2 ms** D-alpha ELM masks and inter-ELM samples.
TangTV accepts the 30 Hz ELM-integrated exposures, matching the measurement choice
in Chen 2026. Integrated camera emission can still differ from the
inter-ELM target state, so redundancy and conflict abstention remain necessary.
All three require known filterscope time coverage. FS01–FS04 are fetched for
shots lacking corpus filterscopes when possible. Missing ELM information is
`elm_unknown`, never clean. A covered group containing a complete missing
50 ms window also remains unknown. `aux_elm_share` remains NaN when unknown and is
counted explicitly in the audit. An unknown-coverage bin cannot be certain.

## Primary rule and diagnostic model

The label intersects the compatible sets of all valid cast votes. A certain state
requires TangTV, at least one other cast vote, a common compatible state, known ELM
coverage and upper-shelf geometry. An attached vote conflicts with detached/MARFE;
current/radiation detached votes are compatible with a TangTV MARFE. Afrac+Prad
alone are `low_confidence_pair`, exported uncertain. Abstentions do not supply
support. Lower-shelf results remain provisional regardless of agreement.

The Snorkel-style exact-enumeration model is **vestigial, only a diagnostic**. Its source
weights, posterior threshold sweep, shot-grouped structure comparison and agreement
with the rule are recorded; fitted posterior values never promote a primary bin.
Weights and posteriors are uncalibrated and do not identify physical accuracy.
Model/rule agreement supplies no extra physical validation because both use the
defining votes. Primary labels are unsmoothed. Neighbor-fill suggestions
may cross validity boundaries and cannot become observed certainty.

| Tier | Primary state | Meaning |
|---|---|---|
| `certain` | 1/2/3 | Upper-shelf, known ELM coverage, compatible redundant votes |
| `lower_shelf_window` | 4 | Owner's restricted extraction, pending sign-off |
| `elm_unknown` | 4 | ELM coverage cannot be established |
| `low_confidence_pair` | 4 | Current/radiation votes without TangTV support |
| `candidate_marfe` | 4 | High front without the complete evidence chain |
| `conflict` | 4 | Cast votes have no common compatible state |
| `insufficient_support` / `no_vote` | 4 | Too few usable cast votes |
| `not_assessed` | absent | Fewer than two valid indicators |

## Evaluation and current results

<!-- RESULTS -->

Full discharge extraction: 50,032 bins/470 shots. Eligibility requires **20 valid bins per indicator for at least two indicators and 20 jointly assessed bins per shot**. This one-second minimum narrows the requested every-shot coverage; shorter overlaps are not exported.

Certain coverage: 15.35 s/42 intervals; 11 are single 50 ms bins. Fixed-cohort certain bins: train 0, validation 50, test 0. True inversions exist on 41 shots; surrogate valid bins: 0.

| Certain state | Bins | Shots | Shot IDs | Single-bin intervals |
|---|---|---|---|---|
| attached | 0 | 0 | — | 0 |
| detached | 300 | 16 | 189057, 189061, 189062, 189081, 189088, 189090, 189093, 189094, 190109, 190110, 190113, 190115, 190116, 199166, 199172, 200977 | 11 |
| marfe | 7 | 1 | 199172 | 0 |

Certain composition by shot (zeros denote absent state support):

| Shot | Attached bins | Detached bins | MARFE bins |
|---|---|---|---|
| 189057 | 0 | 32 | 0 |
| 189061 | 0 | 50 | 0 |
| 189062 | 0 | 37 | 0 |
| 189081 | 0 | 2 | 0 |
| 189088 | 0 | 30 | 0 |
| 189090 | 0 | 51 | 0 |
| 189093 | 0 | 2 | 0 |
| 189094 | 0 | 34 | 0 |
| 190109 | 0 | 25 | 0 |
| 190110 | 0 | 9 | 0 |
| 190113 | 0 | 2 | 0 |
| 190115 | 0 | 14 | 0 |
| 190116 | 0 | 2 | 0 |
| 199166 | 0 | 6 | 0 |
| 199172 | 0 | 3 | 7 |
| 200977 | 0 | 1 | 0 |

Survey checks time length >1 on 16,909 shots: usable TangTV 7,233, bolometer 13,846, Langmuir 6,239, IRTV 250. Group presence alone is insufficient.

| Indicator, full discharge | Valid bins / shots | Cast vote bins |
|---|---|---|
| Local f_div | 38,631 / 449 | 31377 |
| Inversion DZ | 1,791 / 41 | 1520 |

| Primary export tier | Bins |
|---|---|
| insufficient_support | 708 |
| lower_shelf_window | 459 |
| certain | 307 |
| candidate_marfe | 84 |
| conflict | 69 |
| no_vote | 67 |
| low_confidence_pair | 43 |

Pairwise cast votes are compared before consensus selection and stratified by accepted `tangtv_tier`. The upper shelf is the paper population; lower-shelf primary bins remain uncertain.

| TangTV tier | Pair | Cast pairs / shots | Agreement [95% shot CI] | Three-state κ [95% shot CI] | Binary κ [95% shot CI] |
|---|---|---|---|---|---|
| lower_shelf_window | afrac / prad | 19 / 6 | 0.737 [0.250, 0.963] | 0.481 [-0.358, 0.917] | 0.481 [-0.358, 0.917] |
| lower_shelf_window | afrac / tangtv | 55 / 7 | 0.836 [0.480, 0.971] | 0.671 [0.000, 0.941] | 0.671 [0.000, 0.941] |
| lower_shelf_window | prad / tangtv | 114 / 9 | 0.825 [0.533, 0.957] | 0.642 [0.101, 0.879] | 0.707 [0.121, 0.956] |
| none | afrac / prad | 63 / 7 | 0.683 [0.434, 0.864] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| upper_shelf | prad / tangtv | 356 / 17 | 0.843 [0.628, 0.973] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

On the upper shelf, Prad votes detached in 49 of 49 cast pairs where TangTV votes attached. There are 416 such bins with valid Prad measurements; Prad abstains in 367. Every upper-shelf cast Prad vote is detached. Radiation corroboration cannot establish a physical classifier.

Across 1144 upper-shelf bins with both measurements valid, f_div has median 0.484 and 5th–95th percentiles 0.398–0.564. Local cutoff placement therefore affects which imaging states receive radiation corroboration.

Threshold margins in the certain set:

| State | Bins | f_div range | Margin Q25 / median | Within .05 | Within .1 |
|---|---|---|---|---|---|
| detached | 300 | 0.500–0.618 | 0.015 / 0.028 | 220 | 298 |
| marfe | 7 | 0.501–0.511 | 0.002 / 0.008 | 7 | 7 |

Prad margins are signed distances from 0.36 for attached and 0.50 for detached/MARFE. Sensitivity shifts both Prad cutoffs together. Greenwald changes recompute the complete adjacent-bin spatial/cue MARFE gate and retain the H–L cue. Measurement gates stay fixed. Only non-test eligible shots enter this descriptive analysis.

| Family | Shift | Cutoffs / cue | Attached | Detached | MARFE | Certain bins / shots |
|---|---|---|---|---|---|---|
| prad | -0.10 | 0.26 / 0.40 | 0 | 499 | 23 | 522 / 22 |
| prad | -0.05 | 0.31 / 0.45 | 0 | 465 | 23 | 488 / 22 |
| prad | +0.00 | 0.36 / 0.50 | 0 | 300 | 7 | 307 / 16 |
| prad | +0.05 | 0.41 / 0.55 | 38 | 80 | 0 | 118 / 10 |
| prad | +0.10 | 0.46 / 0.60 | 241 | 2 | 0 | 243 / 17 |
| greenwald | -0.10 | fG≥0.70 | 0 | 300 | 17 | 317 / 16 |
| greenwald | +0.00 | fG≥0.80 | 0 | 300 | 7 | 307 / 16 |
| greenwald | +0.10 | fG≥0.90 | 0 | 300 | 0 | 300 / 16 |

The certain composition is set by the Prad cutoffs: attached/detached/MARFE bins are 0/499/23 at -0.10, 0/465/23 at -0.05, 0/300/7 as exported, 38/80/0 at +0.05 and 241/2/0 at +0.10. Certainty therefore means a TangTV vote plus f_div on the same side of one local, single-shot-derived cutoff; it is not a threshold-independent state.

Bin-width sensitivity (descriptive; non-test shots with a TangTV-eligible grid):

| Bin width (ms) | Assessed bins / shots | Certain bins / shots |
|---|---|---|
| 20 | 2854 / 27 | 744 / 16 |
| 50 | 1144 / 27 | 307 / 16 |
| 100 | 586 / 27 | 156 / 14 |

The vestigial Snorkel diagnostic has 0 three-source anchors on 7 shots. Its uncalibrated model/rule agreement supplies no physical validation.

| Held-shot diagnostic / control | Bins / shots | Accuracy [95% shot CI] | κ [95% shot CI] | Fixed-class macro-F1 [95% shot CI] |
|---|---|---|---|---|
| detach-ours (ours population) | 231 / 12 | 0.835 [0.549, 1.000] | -0.052 [-0.170, 0.000] | 0.455 [0.335, 0.493] |
| Fold majority (ours population) | 231 / 12 | 0.970 [0.877, 1.000] | 0.000 [0.000, 0.000] | 0.492 [0.464, 0.494] |
| detach-victor (victor population) | 69 / 8 | 0.899 [0.631, 1.000] | 0.000 [0.000, 0.000] | 0.473 [0.377, 0.483] |
| Fold majority (victor population) | 69 / 8 | 0.899 [0.631, 1.000] | 0.000 [0.000, 0.000] | 0.473 [0.377, 0.483] |

Class support in each scored population:

| CNN | Reference class | Bins | Shots |
|---|---|---|---|
| detach-ours | attached | 0 | 0 |
| detach-ours | detached | 224 | 12 |
| detach-ours | marfe | 7 | 1 |
| detach-victor | attached | 0 | 0 |
| detach-victor | detached | 62 | 8 |
| detach-victor | marfe | 7 | 1 |

detach-ours, ours population: rows are reference, columns prediction (attached: no reference bins, never predicted).

| Reference | detached | marfe |
|---|---|---|
| detached | 193 | 31 |
| marfe | 7 | 0 |

Fold majority, ours population: rows are reference, columns prediction (attached: no reference bins, never predicted).

| Reference | detached | marfe |
|---|---|---|
| detached | 224 | 0 |
| marfe | 7 | 0 |

detach-victor, victor population: rows are reference, columns prediction (attached: no reference bins, never predicted).

| Reference | detached | marfe |
|---|---|---|
| detached | 62 | 0 |
| marfe | 7 | 0 |

Fold majority, victor population: rows are reference, columns prediction (attached: no reference bins, never predicted).

| Reference | detached | marfe |
|---|---|---|
| detached | 62 | 0 |
| marfe | 7 | 0 |

These CNNs predict weak rule labels, so their scores are exploratory agreement diagnostics. Neither establishes learning beyond the adjacent fold-majority control. The fold holding the sole MARFE shot has no MARFE training examples; **MARFE transfer is unsupported**. Epochs and architecture are fixed, folds group by shot, and normalization uses training-fold data. Macro-F1 keeps the population's class set fixed across draws; draws missing a required class are excluded and counted in the JSON. Ours uses complete finite windows without heating inputs, missingness channels or imputation. The scored populations differ and cannot rank model quality.

The independent published reference has 3 points on 2 shots. The primary rule abstains on all three points, including 199166 MARFE at 3.705 s; no positive independent state validation is available.

199166/3705 ms: DZ=1.238; fG=0.721 fails the fixed 0.8 cue, and no H–L back-transition occurs. Spatial evidence fails at onset; nearby bins pass spatial evidence but still fail fG. The primary state is uncertain/candidate_marfe despite usable EFIT01 maps and confirmed density units. The cue was not retuned.

Manual front check: 49 valid paired bins/1 shot(s), DZ MAE 0.017 on 58 annotated bins. These points reuse the imaging modality and do not independently validate states.

Sources: [current coverage/margins](results/detachment_round3.json), [tier-stratified agreement](results/detachment_benchmark.json), [published reference](results/detachment_reference.json), [ours CNN](results/detachment_ours.json) and [Victor CNN](results/detachment_victor.json). These are exploratory records; **no independent benchmark** is available.

<!-- /RESULTS -->

## Figures and reproduction

The two appendix figures, `fig_detachment_views` and `fig_detachment_timeline`,
are reserved for **full text width, 6.75 inches**. Their ≥7 pt text is specified
at that final placement size; they must not be scaled to a single column.
Camera/inversion examples use an upper-shelf shot. Inversion panels show EFIT
separatrix/flux contours, X/strike markers and the g-file wall. Raw pixels have
no invented camera-space magnetic calibration. A bolometer chord-index profile
is a diagnostic row, not a spatial inversion. The timeline plots the voted
`f_div` and DZ with their thresholds, and legends identify selected-time markers.
PDFs are vector and PNGs 150 dpi.

Figure 2 uses upper-shelf indicator coverage/agreement with an explicit **no
independent benchmark** annotation. `docs/labeler/figure2_detach.json` and
`results/detachment_figure2.json` record the selected population and agreement
source; the lower shelf remains a separate provisional stratum. No detector
ranking or corpus-wide physical classification accuracy is asserted.

Regenerate in order with `detach_bins.py` (full 50 ms extraction) and
`detach_regenerate_widths.py` (20/50/100 ms grids; its 50 ms arrays must equal the
primary ones), `detach_label.py`, then `detach_reference.py`,
`detach_round2_records.py`, `detach_physics_audit.py`, `detach_sensitivity.py`,
`detach_benchmark.py`, `detach_fix_records.py` and `detach_round3_records.py`;
`detach_ours.py prep`/`refresh` and `detach_victor.py prep`/`refresh` (retrain on a
GPU only if the prepared arrays change), `detach_figure.py`, `detach_paper_panel.py`,
`detach_handoff.py`, `detach_protocol.py` and `detach_validate_outputs.py`. Full
outputs remain under `$LABELER_ROOT/round4/detach/`; the external `HANDOFF.md`
documents additive fields and per-shot diagnostic availability.
Fetches use the allowed login-node fdp path, ≤3 workers, one-second pacing and a
shared authentication-stop flag. No remote fetch continues after an auth failure.

## Limitations

The published anchors are too small and concentrated to support population
accuracy, calibration or campaign transfer. Manual emission-front points test DZ
localization, not independently reviewed attached/detached states. Agreement between
indicators can share systematic biases. The local current reference does not identify
true attached current, lower-shelf geometry awaits sign-off, and the surrogate supplies
no usable deployment coverage. Flux-map/density availability can prevent MARFE even
where a paper observes one; abstention and the exact failed gates are disclosed.
No primary gold-test performance is available. Future work needs an independent
expert reference, attached-current calibration and more validated inversion coverage.

## References

- Chen et al., *Nuclear Fusion* 66, 036014 (2026),
  [doi:10.1088/1741-4326/ae3972](https://doi.org/10.1088/1741-4326/ae3972).
- Eldon et al., *Nuclear Materials and Energy* 27, 100963 (2021),
  [doi:10.1016/j.nme.2021.100963](https://doi.org/10.1016/j.nme.2021.100963).
- Eldon et al., *Plasma Physics and Controlled Fusion* 64, 075002 (2022),
  [doi:10.1088/1361-6587/ac6ff9](https://doi.org/10.1088/1361-6587/ac6ff9).
- Victor and Scotti, *Review of Scientific Instruments* 95, 083503 (2024),
  [doi:10.1063/5.0218724](https://doi.org/10.1063/5.0218724).

Published constructions motivate the indicators. Local thresholds, quality gates,
restricted coverage and agreement scores are stated separately.
