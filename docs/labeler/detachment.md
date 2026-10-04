# Detachment protocol

<!-- SUMMARY -->

**Exploratory label set with an independent temperature check; no gold-standard benchmark.** Each 50 ms bin on an eligible shot is attached, detached, MARFE or uncertain. Three indicators vote: the target-current ratio (Afrac proxy), the lower-divertor radiated fraction f_div = Prad,div,L / P_in, and the TangTV C-III front height DZ. A bin is certain attached or detached when TangTV votes and a second indicator agrees with it and none disagrees; it is certain MARFE on the TangTV MARFE vote (spatial cue, density cue, persistence). Certain labels measure indicator agreement, not physical accuracy.

**Coverage.** 2,873 assessed bins on 37 shots (144 s); 471 certain bins on 24 shots: attached 78 bins on 9 shots, detached 370 bins on 21 shots, MARFE 23 bins on 1 shot. The rest is uncertain, with the reason in the `tier` column.

**Indicator agreement (upper shelf).** f_div ranks TangTV-detached above TangTV-attached bins with AUROC 0.79 [0.65, 0.92] pooled (1,039 bins, 28 shots; shot-bootstrap 95% intervals) and 0.80 [0.58, 0.98] within shots. Cohen's kappa is 0.12 [-0.07, 0.37] (three states) and 0.14 [-0.06, 0.41] (attached against not attached). The Afrac proxy is near chance against TangTV: AUROC 0.55 [0.44, 0.68].

**Divertor Thomson Te check.** 1,498 of the 2,873 assessed bins have a processed divertor Thomson Te near the target. Median Te is 15.5 eV in certain attached bins and 2.1 eV in certain detached bins; -Te separates them with AUROC 0.95 [0.92, 0.99]. Against Te, the TangTV vote has AUROC 0.95 [0.91, 0.98], f_div 0.75 [0.68, 0.81] and Afrac 0.60 [0.52, 0.67].

**Reference shot 201081.** Measured P_in is 4.27 MW. The f_div cutoffs 0.399 / 0.446 are the midpoint of the attached (1.621 MW) and detached (1.990 MW) Prad,div,L levels over that P_in, with a stated band of +-0.1 MW. Of 7 published points on 201081, 3 are certain and correct, 4 are uncertain and 0 are wrong; the Afrac proxy votes detached on 4 of the 4 attached points. The Te cliffs fall at 2575 and 4425 ms against 2650 and 4450 ms published.

**Composition under the cutoffs.** Certain attached / detached / MARFE bins are 78 / 370 / 23 with the absolute cutoffs, 141 / 219 / 23 with the per-shot relative f_div (sensitivity), and 47 / 501 / 23 when the Afrac proxy does not vote. The band sweep is in the results table.

**MARFE.** Certain on one shot (199172). 84 further bins on 6 shots carry a TangTV MARFE signal without the full evidence (tier `candidate_marfe`). The published MARFE on 199166 (3.705 s) is not labelled: DZ exceeds 1.2 from 3.700 s, but the emission peak is outside the separatrix at the onset and fG = 0.72-0.75 is below the 0.8 cue.

**Afrac proxy.** It reads the peak-Jsat probe on the scrape-off side of the outer strike. Its votes disagree with TangTV often enough that conflict is the largest uncertain tier (1,056 bins); without its vote the certain composition is 47 / 501 / 23.

**Three-state label set.** Not offered: certain attached and certain detached bins coexist on 6 shots, 0 of them in the fixed cohort, against a requirement of at least three shots including one cohort shot. The result is an indicator-agreement appendix.

<!-- /SUMMARY -->

The fixed 500-shot cohort's test split is excluded from fitting, threshold
selection, surrogate selection and bin-policy analysis.

## Definitions and export

The lower-divertor states are attached (1), detached (2), MARFE (3) and uncertain
(4). Missing intervals are unassessed, never attached. These are operational
definitions: attached is a low accepted C-III front with a compatible second
indicator; detached is a lifted front with the same support; MARFE is a persistent
high front with an emission peak inside the separatrix near the X-point and a
density or H-L cue.

Bins are 50 ms, following the catalog grid. The 30 Hz camera gives about 1.5
frames per bin; Chen 2026 integrates ELMs over these exposures and so does this
export, without resampling extra frames.

Files:

- `data/events/detachment/extend_detach_vote/detach_shots.csv` holds the standard
  shot/category/start/end/confidence/attrs intervals; `attrs` carries `tier`.
- `$LABELER_ROOT/round4/detach/labels_bins.csv.gz` holds every bin with values,
  votes, validity reasons, geometry and Afrac probe provenance (exported for invalid
  bins too). `state_rule` is the primary label; `state_lm` is its legacy alias;
  `state_rule_relative_prad` is the sensitivity column with per-shot relative f_div
  cutoffs; `state_model_diagnostic` is the vestigial fitted model;
  `state_temporal_imputation` is a separate suggestion.
- The lower-shelf TangTV window is not a label column. It is absent from the
  interval table, the grids, `labels_bins.csv.gz`, `indicators/<shot>.csv` and the
  detach-ui handoff; its bins carry `state_rule = 4` and `tier =
  lower_shelf_window`. The per-indicator vote and value columns of those bins
  remain.
- Outputs use milliseconds; corpus HDF5 time axes use seconds. `shots.csv` is an
  unverified review roster.

## Indicators

**Afrac proxy.** Processed `LANGMUIR::TOP.PROBE_*:{R,Z,JSAT,TIME}` Jsat of one probe
per bin: among the probes on the scrape-off side (psiN > 1.000 by the nearest
multi-slice EFIT map, at least 5 mm outboard of the outer strike point, psiN <=
1.05) the one with the peak Jsat. Inter-ELM current times P_SOL^(3/7), divided by
relative density squared, normalised to the shot's own 90th-percentile reference.
It is an uncalibrated local proxy, not the published Afrac reference. Ratio >= 0.75
votes attached, <= 0.5 detached, the band between abstains. Probe, position, psiN,
distance to the strike and the number of eligible probes are exported per bin.

**f_div = Prad,div,L / P_in.** Calibrated lower-divertor radiation over input power,
both averaged over the same centered 250 ms window of inter-ELM samples; D-alpha
must cover the whole window. P_in is the neutral beams plus the EFIT ohmic power plus
ECH, at least 0.5 MW. Where the corpus beam group is empty (26 shots) the beam
total is PTDATA `BMSPINJ` (1 V = 1 MW); the corpus total over `BMSPINJ` is 0.97 to
1.01 where both exist (`detachment_nbi_calibration.json`). Radiation below -0.05 MW
is invalid. The cutoffs are absolute and come from shot 201081: the midpoint of
the attached and detached Prad,div,L levels (measured in 250 ms inter-ELM windows
placed from the published Te cliffs) over the measured P_in, with a stated band of
+-0.1 MW between them (`detachment_prad_anchor.json`). Attached below the lower
cutoff, detached above the upper one. The sensitivity column replaces them with
per-shot relative cutoffs on f_div over a baseline (the 10th percentile over
flat-top-power bins). f_div does not resolve MARFE and is not a MARFE corroborator.

**TangTV front height.** The inversion gives the emission height ZE and
`DZ = 1 - (ZX - ZE)/(ZX - ZS)`. Frames need lower-single-null shelf geometry, an
outer leg of at least 10 cm, finite emission, DZ >= -0.25 and an EFIT slice within
40 ms; half the frames of a bin must pass. Camera geometry prefers EFIT02, with a
named EFIT01 fallback. DZ < 0.35 votes attached and 0.5 <= DZ < 1.2 detached. DZ >=
1.2 for two adjacent bins, with the global emission peak inside the separatrix near
the X-point (within 30 cm radially, ZX-2 cm to ZX+30 cm vertically) and a density
cue fG >= 0.8 (or a recorded H-L back-transition within 200 ms), is the MARFE vote.
fG uses unit-confirmed line-averaged density over the Greenwald density with an
elliptical chord, with about 10-20% geometric uncertainty; unknown units give no
cue. The upper shelf is Z near -1.25 m, R >= 1.37 m. The lower shelf (R < 1.37 m) is
extracted by the owner's code and never yields a state. The ridge surrogate
contributes no valid bins.

**ELM coverage.** Afrac and f_div use inter-ELM samples with +-2 ms D-alpha ELM
masks; TangTV accepts the integrated frames. Missing filterscope coverage gives
`elm_unknown`, which can never be certain.

## Primary rule

A bin is certain attached or detached when TangTV votes, at least one other cast
vote agrees, no cast vote disagrees, the ELM coverage is known and the geometry is
the upper shelf. A conflicting Afrac or f_div vote therefore blocks certainty. A
bin is certain MARFE on the TangTV MARFE vote unless Afrac or f_div votes attached;
Afrac and f_div are not MARFE corroborators. Primary states are unsmoothed.

| Tier | State | Meaning |
|---|---|---|
| `certain` | 1/2/3 | compatible redundant votes on the upper shelf, known ELM coverage |
| `conflict` | 4 | cast votes with no common state |
| `insufficient_support` | 4 | TangTV votes but no second compatible vote |
| `low_confidence_pair` | 4 | two votes without TangTV |
| `no_vote` | 4 | no indicator casts a vote |
| `candidate_marfe` | 4 | TangTV MARFE signal without the full evidence |
| `lower_shelf_window` | 4 | restricted lower-shelf extraction, never a state |
| `geometry_unknown` / `elm_unknown` | 4 | shelf geometry or ELM coverage unknown |
| `not_assessed` | absent | fewer than two valid indicators |

The Snorkel-style label model is vestigial. Its weights are uncalibrated, they do not
identify physical accuracy, and it never decides a label.

## Results

<!-- RESULTS -->

### Coverage

| Group | Bins / shots | Seconds |
|---|---|---|
| Assessed | 2,873 / 37 | 143.7 |
| Certain | 471 / 24 | 23.6 |
| Certain attached | 78 / 9 | 3.90 |
| Certain detached | 370 / 21 | 18.50 |
| Certain MARFE | 23 / 1 | 1.15 |
| Tier `candidate_marfe` | 84 / 6 | 4.20 |
| Tier `conflict` | 1,056 / 36 | 52.80 |
| Tier `insufficient_support` | 401 / 34 | 20.05 |
| Tier `low_confidence_pair` | 364 / 36 | 18.20 |
| Tier `lower_shelf_window` | 459 / 9 | 22.95 |
| Tier `no_vote` | 38 / 17 | 1.90 |

Assessed means at least two valid indicators on an eligible shot (at least 20 assessed bins and 20 valid bins per contributing indicator). Tiers are disjoint and sum to the assessed bins; the `lower_shelf_window` tier holds the owner's restricted lower-shelf extraction, which never yields a state.

### Indicator coverage, all extracted shots

| Indicator | Measurement | Valid | Vote | In a certain label |
|---|---|---|---|---|
| afrac | 3,078 / 42 | 2,970 / 42 | 2,381 / 42 | 352 / 22 |
| prad | 43,081 / 455 | 39,061 / 454 | 36,604 / 454 | 382 / 23 |
| tangtv | 1,813 / 41 | 1,791 / 41 | 1,520 / 41 | 471 / 24 |

Cells are bins / shots over 50,032 extracted bins on 470 shots.

### Indicator agreement, upper shelf

| Statistic | Value [95% shot CI] | Chance | Bins | Shots |
|---|---|---|---|---|
| AUROC of f_div, pooled | 0.79 [0.65, 0.92] | 0.5 | 1039 | 28 |
| AUROC of f_div, mean over shots | 0.80 [0.58, 0.98] | 0.5 | - | 8 |
| Spearman rho of f_div with DZ, pooled | 0.50 [0.32, 0.63] | 0.0 | 1214 | 28 |
| Spearman rho, mean over shots | 0.34 [0.19, 0.50] | 0.0 | - | 28 |
| Cohen's kappa, attached against not attached | 0.14 [-0.06, 0.41] | 0.0 | 837 | 28 |
| Cohen's kappa, three states | 0.12 [-0.07, 0.37] | 0.0 | 837 | 28 |

AUROC is the probability that a TangTV-detached or MARFE bin has a larger f_div than a TangTV-attached bin. Intervals resample shots (1000 replicates, seed 0). Kappa is null, and not drawn, where either rater used one class. The same statistics for each pair of indicators:

| Shelf | Pair | AUROC [CI] | Spearman rho [CI] | Bins | Shots |
|---|---|---|---|---|---|
| upper shelf | f_div against TangTV | 0.79 [0.65, 0.92] | 0.50 [0.32, 0.63] | 1039 | 28 |
| upper shelf | Afrac against TangTV | 0.55 [0.44, 0.68] | 0.13 [-0.05, 0.30] | 1070 | 28 |
| upper shelf | f_div against Afrac | 0.47 [0.34, 0.59] | 0.01 [-0.21, 0.22] | 933 | 28 |
| lower shelf window | f_div against TangTV | 0.86 [0.69, 0.97] | 0.47 [0.07, 0.77] | 387 | 9 |
| lower shelf window | Afrac against TangTV | 0.49 [0.30, 0.77] | -0.06 [-0.35, 0.38] | 382 | 9 |
| lower shelf window | f_div against Afrac | 0.55 [0.27, 0.88] | 0.10 [-0.27, 0.53] | 341 | 9 |

### Divertor Thomson Te check

The processed divertor Thomson Te (`\ELECTRONS::TSTE_DIV`, 14 or 16 chords at R = 1.485 m, about 20 ms) is a temperature that none of the three indicators uses. A bin's Te is the median of the valid samples of the chords 1.5 to 5 cm above the outer shelf (Z = -1.25 m) that lie on the scrape-off side, 1.000 < psiN <= 1.05, by the multi-slice EFIT map; samples outside 0.1-50 eV are rejected. No threshold is fitted: the bands were fixed from the literature before scoring.

| Group | Bins / shots | Median Te, eV (10-90%) | Share in band |
|---|---|---|---|
| certain attached | 39 / 7 | 15.5 (7.6-23.3) | 0.77 |
| certain detached | 206 / 18 | 2.1 (1.4-8.2) | 0.77 |
| tangtv votes attached | 379 / 18 | 16.0 (6.6-22.1) | 0.83 |
| tangtv votes detached | 308 / 20 | 2.1 (1.4-9.2) | 0.77 |
| prad votes attached | 316 / 26 | 14.2 (2.6-27.3) | 0.68 |
| prad votes detached | 889 / 27 | 5.2 (1.4-17.0) | 0.49 |
| afrac votes attached | 394 / 26 | 12.1 (1.7-22.1) | 0.62 |
| afrac votes detached | 840 / 27 | 6.8 (1.4-20.9) | 0.44 |

Share in band is the fraction of bins with Te >= 10 eV for attached or <= 5 eV for detached. AUROC of -Te for detached against attached bins (pooled, shot-bootstrap interval):

| Reference | AUROC [CI] | Bins | Shots |
|---|---|---|---|
| certain labels | 0.95 [0.92, 0.99] | 245 | 21 |
| tangtv vote | 0.95 [0.91, 0.98] | 687 | 25 |
| prad vote | 0.75 [0.68, 0.81] | 1205 | 27 |
| afrac vote | 0.60 [0.52, 0.67] | 1234 | 27 |

27 of the 37 assessed shots have a Thomson Te in an assessed bin; the other shots have no chord near the target or no Thomson data.

On 201081 the Te first falls below the detached band at 2575 ms and last does at 4425 ms (published cliffs 2650 and 4450 ms). No certain MARFE bin has a Thomson Te: the check does not cover MARFE.

### Reference shot 201081 and published points

P_in = neutral beams + EFIT ohmic + ECH, median 4.28 MW over the flat top (beams 4.09 MW from PTDATA BMSPINJ, because the corpus `pinj` group is a stub on this shot). Prad,div,L is 1.621 MW attached and 1.990 MW detached in 250 ms inter-ELM windows placed from the published Te cliffs (32 attached and 28 detached bins).

| Shot | Time, ms | Published | Afrac | f_div | TangTV | Label |
|---|---|---|---|---|---|---|
| 180257 | 2400 | attached | abstain | abstain | abstain | unassessed |
| 180257 | 4800 | detached | abstain | attached | abstain | unassessed |
| 199166 | 3705 | marfe | detached | detached | abstain | uncertain |
| 201081 | 2300 | attached | detached | attached | attached | uncertain |
| 201081 | 2500 | attached | detached | abstain | abstain | uncertain |
| 201081 | 3000 | detached | detached | detached | detached | detached |
| 201081 | 3500 | detached | detached | detached | detached | detached |
| 201081 | 4000 | detached | detached | detached | detached | detached |
| 201081 | 4900 | attached | detached | attached | attached | uncertain |
| 201081 | 5100 | attached | detached | attached | attached | uncertain |

The ten points are published statements (Chen 2026; Eldon 2021). They overlap the sources that motivated the indicators, and 201081 also anchors the f_div cutoffs, so the table is a sanity check, not a benchmark. A vote of `abstain` means the indicator does not vote in that bin.

### Composition under the f_div cutoffs

| Variant | Cutoffs | Certain bins (att / det / MARFE) | Certain shots | Shots with both | Cohort shots with both |
|---|---|---|---|---|---|
| primary_absolute | 0.399 / 0.446 | 78 / 370 / 23 | 9 / 21 / 1 | 6 | 0 |
| relative | 1.052 / 1.176 | 141 / 219 / 23 | 15 / 18 / 1 | 9 | 0 |
| primary_absolute_afrac_abstains | 0.399 / 0.446 | 47 / 501 / 23 | 4 / 23 / 1 | 2 | 0 |
| relative_afrac_abstains | 1.052 / 1.176 | 92 / 123 / 23 | 14 / 12 / 1 | 7 | 0 |
| absolute_published_anchor_values | 0.421 / 0.468 | 116 / 360 / 23 | 15 / 19 / 1 | 8 | 0 |
| absolute_band_0.025_mw | 0.417 / 0.428 | 52 / 371 / 23 | 8 / 21 / 1 | 4 | 0 |
| relative_band_0.025_mw | 1.098 / 1.129 | 134 / 198 / 11 | 13 / 18 / 1 | 9 | 0 |
| absolute_band_0.05_mw | 0.411 / 0.434 | 60 / 370 / 23 | 8 / 21 / 1 | 5 | 0 |
| relative_band_0.05_mw | 1.083 / 1.145 | 141 / 209 / 22 | 13 / 18 / 1 | 9 | 0 |
| absolute_band_0.1_mw | 0.399 / 0.446 | 78 / 370 / 23 | 9 / 21 / 1 | 6 | 0 |
| relative_band_0.1_mw | 1.052 / 1.176 | 141 / 219 / 23 | 15 / 18 / 1 | 9 | 0 |
| absolute_band_0.15_mw | 0.387 / 0.458 | 91 / 372 / 23 | 10 / 21 / 1 | 7 | 0 |
| relative_band_0.15_mw | 1.021 / 1.206 | 161 / 239 / 23 | 16 / 19 / 1 | 9 | 0 |
| absolute_band_0.2_mw | 0.376 / 0.469 | 107 / 370 / 23 | 15 / 21 / 1 | 9 | 0 |
| relative_band_0.2_mw | 0.990 / 1.237 | 149 / 254 / 23 | 15 / 19 / 1 | 7 | 0 |
| absolute_band_0.3_mw | 0.352 / 0.493 | 150 / 355 / 23 | 15 / 20 / 1 | 8 | 0 |
| relative_band_0.3_mw | 0.929 / 1.299 | 162 / 262 / 23 | 15 / 19 / 1 | 7 | 0 |

`absolute` cutoffs (primary) are the 201081 midpoint over the measured P_in; `relative` cutoffs scale a per-shot baseline (the 10th percentile of f_div over flat-top-power bins); `band_X` sets the half-width of the uncertain band to X MW (the primary variant keeps 0.1 MW); `afrac_abstains` removes the Afrac vote; `published_anchor_values` uses the published 1.6 / 2.2 MW.

Decision criterion: certain attached AND certain detached bins on at least 3 of the same shots, at least one of them a cohort shot (split train, val or test). Certain attached and certain detached bins coexist on 6 shots (0 in the cohort), so the criterion is not met and the label set is presented as an indicator-agreement appendix, not as a three-state label set. No variant in the table meets it.

### Bin width

| Width | Assessed bins | Certain bins / shots | Uncertain share | Flicker per s | Agreement with 50 ms |
|---|---|---|---|---|---|
| 20ms | 5,099 | 1065 / 25 | 0.79 | 0.047 | 0.999 |
| 50ms | 2,086 | 428 / 23 | 0.79 | 0.047 | - |
| 100ms | 1,060 | 207 / 22 | 0.80 | 0.048 | 0.995 |

The same rule on the 77 roster shots at each width, without the export's eligibility filter, so counts differ from the coverage table. Where both a 50 ms and a finer or coarser bin are certain, the states agree: 0.999 at 20ms (905 bins), 0.995 at 100ms (190 bins).

### Afrac probe selection

Probes eligible for Afrac are on the scrape-off side (psiN > 1.000, at least 5 mm outboard of the outer strike point, psiN <= 1.05); the peak-Jsat eligible probe is read, and its position, flux, distance and eligible-probe count are exported for every bin, valid or not. Earlier exports reported no upper-shelf probe. The cause was a gate that required a probe within 2 cm of the outer strike and psiN >= 1.01 together: only 249 of 3,377 bins with a selected probe meet both. It was not sparse EFIT flux maps.

### MARFE witness

Published MARFE on 199166 at 3705 ms (Chen 2026): the TangTV front exceeds the 1.2 cutoff from the 3700 ms bin (DZ 1.24) but the emission peak lies outside the separatrix (psiN 1.48 at 3565 ms, EFIT01) and fG is 0.72 at 3700 ms and 0.75 at 3800 ms, below the 0.8 cue. The bin is `candidate_marfe` (0 MARFE votes). The thresholds were not tuned to this shot.

<!-- /RESULTS -->

## Figures and reproduction

Figures are vector PDF and 150 dpi PNG with text of at least 7 pt, in colour-blind
safe colours, drawn at the 6.75 inch text width unless noted:

- `fig_detachment_views`: camera inversion, EFIT contours, Prad,div,L and P_rad
  traces for shot 201081 at time-ordered columns from sustained attached, detached
  and attached intervals (at least five bins). No 2D bolometer emissivity exists
  (`detachment_bolometer_availability.json`), so the radiation row shows the
  PRAD_DIVL and PRAD_TOT traces.
- `fig_detachment_timeline`: label, Afrac/f_div/TangTV vote strips, f_div and DZ
  with their cutoffs, and the divertor Thomson Te.
- `fig_detachment_marfe_witness`: the 199172 MARFE evidence and the missed
  199166 onset.
- `fig_detachment_figure2` (3.25 inch panel): threshold-free agreement and per-state
  coverage; the source is `docs/labeler/figure2_detach.json`, the one canonical
  record.

Regenerate in order: `detach_fetch_round4.py` (login node, `fdp run`, one worker,
stops at the first authentication error), `detach_bins.py`, `detach_label.py`,
`detach_prad_anchor.py`, `detach_reference.py`, `detach_benchmark.py`,
`detach_te_check.py`, `detach_prad_sensitivity.py`, `detach_current_state.py`,
`detach_physics_audit.py`, `detach_probe_bolom.py`, `detach_bins.py --width-ms 20/100`
and `detach_sensitivity.py`, `detach_ours.py` and `detach_victor.py` (refresh),
`detach_figure.py`, `detach_paper_panel.py`, `detach_handoff.py`,
`detach_docs_tables.py` and `detach_validate_outputs.py`. Large outputs and
`HANDOFF.md` are under `$LABELER_ROOT/round4/detach/`.

## Limitations

The published points are few and concentrated on one shot, so they support neither
accuracy nor calibration. The Afrac proxy is near chance against TangTV and the
divertor Thomson Te, and its votes block many bins. Certain attached and detached bins
never coexist on a cohort shot, so the set cannot support the cohort benchmark or a
study of attached-to-detached transitions. MARFE is certain on one shot, and the
published 199166 MARFE is missed. The Thomson check covers attached and detached
only, on the shots that have a chord near the target. Indicator agreement can share
systematic biases. Future work needs an independent expert state table and more
inversion coverage.

## Appendix: learned baselines

<!-- BASELINES -->

Two learned baselines were trained to predict the labels with shot-grouped folds: `detach-ours` on windows of current, density, D-alpha, ELM share and EFIT scalars, and `detach-victor` on raw camera frames. Neither beats its fold-majority control by a margin that survives the shot bootstrap, and the fold holding the only MARFE shot has no MARFE training examples, so MARFE transfer is unsupported. They are not listed among the dataset's models and are not paper-facing results.

| Model | Bins / shots | Accuracy [95% shot CI] | Kappa [95% shot CI] | Macro-F1 [95% shot CI] |
|---|---|---|---|---|
| detach-ours | 373 / 22 | 0.638 [0.433, 0.827] | 0.182 [-0.137, 0.512] | 0.437 [0.262, 0.540] |
| detach-ours fold majority | 373 / 22 | 0.777 [0.600, 0.921] | 0.000 [0.000, 0.000] | 0.292 [0.249, 0.312] |
| detach-victor | 245 / 12 | 0.861 [0.671, 0.984] | 0.129 [-0.028, 0.740] | 0.426 [0.268, 0.588] |
| detach-victor fold majority | 245 / 12 | 0.857 [0.676, 0.972] | 0.000 [0.000, 0.000] | 0.308 [0.265, 0.313] |

Records: `docs/labeler/results/detachment_ours.json` and `detachment_victor.json`; scripts `detach_ours.py` and `detach_victor.py`.

<!-- /BASELINES -->

## References

- Chen et al., *Nuclear Fusion* 66, 036014 (2026),
  [doi:10.1088/1741-4326/ae3972](https://doi.org/10.1088/1741-4326/ae3972).
- Eldon et al., *Nuclear Materials and Energy* 27, 100963 (2021),
  [doi:10.1016/j.nme.2021.100963](https://doi.org/10.1016/j.nme.2021.100963).
- Eldon et al., *Plasma Physics and Controlled Fusion* 64, 075002 (2022),
  [doi:10.1088/1361-6587/ac6ff9](https://doi.org/10.1088/1361-6587/ac6ff9).
- Victor and Scotti, *Review of Scientific Instruments* 95, 083503 (2024),
  [doi:10.1063/5.0218724](https://doi.org/10.1063/5.0218724).
