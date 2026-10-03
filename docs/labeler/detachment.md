# Detachment protocol

Paper-facing result: **exploratory agreement and coverage**. The primary label is
an unverified compatibility rule with TangTV required. A small reference assembled
from explicit published statements is reported separately; it does not establish
campaign-wide physical accuracy. Expert review and owner sign-off remain pending.
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
Only a SOL-side probe sufficiently outboard of the strike point and with normalized
flux above one plus the EFIT uncertainty margin may vote. The chosen probe,
coordinates, strike geometry, distance, normalized flux, flux-map source and
selection validity are exported per bin. Exact margins and sample counts are in
`thresholds.py` and the fix audit. A probe without usable position/flux evidence
abstains; private-flux current cannot corroborate detachment.

The retained indicator is an **uncalibrated Jsat ratio (local proxy)**: inter-ELM
current divided by relative density squared, then normalized to its own shot's
90th-percentile reference. This follows Eldon's density scaling, but it does not
reproduce a separately fitted attached pre-puff L/H reference or identify attached
current in an entirely detached shot. The inactive fit against TangTV-attached bins
has been removed, preserving independence of the current construction from TangTV.
Ratio ≥0.75 votes attached, ≤0.5 detached, and the transition band abstains.
It cannot distinguish detached from MARFE. `afrac_method=local_proxy` preserves the
historical field name without claiming the published Afrac setting was reproduced.

### Divertor radiation

`f_div = PRAD_DIVL / P_in` uses calibrated lower-divertor radiation and beam,
ECH and ohmic heating. It votes attached at ≤0.36, detached at ≥0.50, and abstains
between. Eldon 2019 supplies the diagnostic construction, not universal state
thresholds. These thresholds are fixed local settings motivated by Chen's 201081
worked example. Their check uses external development inversion shots; the fixed
cohort-train subset has **zero usable reference bins**. It is agreement, not
independent physical validation. Input power must be ≥0.5 MW with valid samples
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
in Chen 2026; it does not apply the former −17/+50 ms camera window to itself or
the faster diagnostics. Integrated camera emission can still differ from the
inter-ELM target state, so redundancy and conflict abstention remain necessary.
All three require known filterscope time coverage. FS01–FS04 are fetched for
shots lacking corpus filterscopes when possible. Missing ELM information is
`elm_unknown`, never clean. `aux_elm_share` remains NaN when unknown and is
counted explicitly in the audit. An unknown-coverage bin cannot be certain.

## Primary rule and diagnostic model

The label intersects the compatible sets of all valid cast votes. A certain state
requires TangTV, at least one other cast vote, a common compatible state, known ELM
coverage and upper-shelf geometry. An attached vote conflicts with detached/MARFE;
current/radiation detached votes are compatible with a TangTV MARFE. Afrac+Prad
alone are `low_confidence_pair`, exported uncertain. Abstentions do not supply
support. Lower-shelf results remain provisional regardless of agreement.

The Snorkel-style exact-enumeration model is **only a diagnostic**. Its source
weights, posterior threshold sweep, shot-grouped structure comparison and agreement
with the rule are recorded; fitted posterior values never promote a primary bin.
Weights and posteriors are uncalibrated and do not identify physical accuracy.
The current model/rule agreement is reported below, rather than treating identical
outputs as extra validation. Primary labels are unsmoothed. Neighbor-fill suggestions
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

Regeneration in progress.

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

Figure 2 uses `docs/labeler/figure2_detach.json`: local single-indicator and
held-shot `detach-victor` F1 against the primary label, with shot-bootstrap CIs.
Published settings that were not reproduced are unavailable. These F1 comparisons
include scored indicators in the reference and are circular agreement diagnostics.
`results/detachment_figure2.json` separately records measurement, vote and consensus
coverage on explicit populations. The F1 panel is 3.25 inches with ≥7 pt text.
No corpus-wide physical classification accuracy is asserted.

Regenerate with `detach_bins.py`, `detach_label.py`, `detach_benchmark.py`,
`detach_reference.py`, `detach_fix_records.py`, `detach_round2_records.py`,
`detach_sensitivity.py`, `detach_ours.py`, `detach_victor.py`, `detach_figure.py`,
`detach_paper_panel.py`, and `detach_protocol.py`. Full outputs remain under
`$LABELER_ROOT/round4/detach/`; the external `HANDOFF.md` documents additive fields.
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
