# Detachment protocol

<!-- SUMMARY -->

**Label set: the geometry-gated TangTV state, validated by divertor Thomson Te. Exploratory; no gold-standard benchmark.** Each 50 ms bin on an eligible shot is attached, detached or uncertain; MARFE is never a state. Three indicators vote: the target-current ratio (Afrac proxy: a probe's Jsat over that probe's own attached level, the probe nearest the separatrix, known L-mode bins abstain), the lower-divertor radiated fraction f_div = Prad,div,L / P_in over the shot's own baseline, and the TangTV C-III front height DZ. A bin is `certain` when TangTV votes and a second indicator (relative f_div or Afrac) agrees with it and none disagrees; it is `tangtv_only` (silver) when TangTV votes and every other indicator abstains or is invalid; every other assessed bin is uncertain, with the reason in the `tier` column. Certain labels measure indicator agreement, not physical accuracy.

**Coverage.** 1,779 assessed bins on 32 shots (89 s). Certain: 275 bins on 19 shots (attached 132 on 15 shots, detached 143 on 13 shots). TangTV only: 398 bins on 22 shots (attached 219, detached 179). The other 1,106 bins are uncertain; 447 of them are `conflict`. Of the 32 assessed shots 1 is in the fixed cohort (val split, 57 assessed bins of which 0 certain and 13 TangTV only) and none is a test shot.

**Divertor Thomson Te check.** AUROC of -Te for a detached against an attached vote (pooled, with shot-bootstrap 95% intervals): TangTV alone 0.95 [0.90, 0.98] (668 bins, 23 shots), certain tier 0.91 [0.77, 0.96] (195 bins, 17 shots), TangTV-only tier 0.95 [0.90, 1.00] (248 bins, 19 shots). Within shots: TangTV alone 0.97 [0.94, 0.99] over 6 shot(s), certain 0.95 [0.93, 0.97] over 2 shot(s), TangTV only 0.80 [0.80, 0.80] over 1 shot(s). **The second vote does not improve the Te agreement of the TangTV state**: certain minus TangTV alone is -0.036 [-0.165, +0.011] over 23 shots (not distinguishable); the point estimate is lower for the certain tier. What the second vote buys is agreement between indicators, not a better match to Te.

**Afrac proxy rebuilt.** The earlier proxy read the peak-current probe against one whole-shot reference and followed the probe's flux position (median 0.79 within 0.005 of the separatrix and 0.17 at psiN 1.02-1.03, for the same plasma). Each probe now has its own attached reference and the probe nearest the separatrix inside |psiN - 1| <= 0.01 is read; the results show the median Afrac against the selected-probe psiN before and after. Against TangTV its AUROC is 0.86 [0.66, 0.95] (228 bins, 7 shots); against Te 0.75 by vote [0.45, 0.86] and 0.76 by value. It stays in the vote (rule: removed below 0.65), but only as a weak second indicator: it is valid on 909 of 50,032 extracted bins (14 of 470 shots), the Te interval includes 0.5, the regime is known on 10 of 37 shots (so the L-mode gate removes 780 bins and misses the rest) and a wider window (0.02) puts the value-based AUROC at 0.64, below the removal bound.

**f_div is per shot.** The exported f_div vote is the shot-relative one (f_div over the shot's own baseline, cutoffs 1.052 / 1.176). The 201081-anchored absolute cutoffs (0.399 / 0.446, one pair for every shot; P_in 4.274 MW) are a sensitivity row. Per-shot medians of the absolute f_div in TangTV-attached bins run from 0.39 to 0.58 (19 shots) and in TangTV-detached bins from 0.40 to 0.57 (21 shots), with P_in from 1.1 to 10.9 MW: the two classes overlap across shots, so no single cutoff pair fits. The relative vote does not rank TangTV states across shots (AUROC 0.41 [0.25, 0.57] pooled, 1,013 bins on 26 shots) but does within shots (0.80 [0.56, 0.98] over 8 shot(s)). The absolute f_div ranks better pooled (0.79 [0.64, 0.92]) but its cutoffs vote detached on most of the TangTV-attached bins where it votes. Against Te the relative f_div vote has AUROC 0.42 [0.23, 0.59], i.e. no Te support pooled. The published detached point 180257 at 4800 ms is not assessed (TangTV has no valid geometry, the relative f_div has no baseline, Afrac reads `probe_off_separatrix`); the absolute cutoffs would call it attached (f_div 0.17 at P_in 7.2 MW), a miss of that published point.

**Reference shot 201081.** Measured P_in is 4.274 MW (the median over the 32 attached-window bins of the 250 ms-averaged P_in); Prad,div,L is 1.621 MW attached and 1.990 MW detached. Of 7 published points on 201081, 6 are labelled in agreement with the publication, 1 uncertain and 0 contradicted.

**MARFE.** No bin is exported as MARFE: the density cue (fG >= 0.8) has no literature source (Dong 2025 gives fG of about 0.5 or more on HL-3 from a core-point density, with a core-Te condition and a gate of 0.40, and says thresholds are device-specific). A persistent high TangTV front is the uncertain tier `candidate_marfe`: 70 bins on 4 shots; the full TangTV MARFE vote (front, emission peak inside the separatrix and the density cue) fires on 23 bins of 1 shot only. **Recall of the published MARFE onset on 199166 (3705 ms): 0/1 as a label**; the bin holding it is `candidate_marfe` (1/1 as a candidate), with fG = 0.72 there against the 0.8 cue.

**Three-state label set.** Not offered: certain attached and certain detached bins coexist on 9 shots, 0 of them in the fixed cohort, against a requirement of at least three shots including one cohort shot. The result is an indicator-agreement appendix.

<!-- /SUMMARY -->

The fixed 500-shot cohort's test split is excluded from fitting, threshold
selection, surrogate selection and bin-policy analysis.

## Definitions and export

The label set is the **geometry-gated TangTV state, validated by divertor
Thomson Te**. The exported lower-divertor states are attached (1), detached (2) and
uncertain (4); the interval schema keeps code 3 (MARFE) but no bin carries it.
Missing intervals are unassessed, never attached. These are operational
definitions: attached is a low accepted C-III front (DZ < 0.35 on the upper shelf),
detached a lifted one (0.5 <= DZ < 1.2). The tier says what supports the state:
`certain` when a second indicator (the shot-relative f_div or Afrac) agrees and none
disagrees, `tangtv_only` (silver) when TangTV is the only vote. A persistent high
front, the MARFE candidate, is the uncertain tier `candidate_marfe`: the density
cue that would make it a MARFE has no literature source (see the MARFE results).

The original plan for this label set called for a benchmark of each single indicator
against the combined label, and optionally for learned detectors. That is replaced by the
independent divertor Thomson Te check: scoring an indicator against a label built
from it is circular, while Te is read by none of the three indicators. The
single-indicator agreement tables remain as agreement diagnostics, and the learned
baselines were not retrained on this label set (appendix).

Bins are 50 ms, following the catalog grid. The 30 Hz camera gives about 1.5
frames per bin; Chen 2026 integrates ELMs over these exposures and so does this
export, without resampling extra frames.

Files:

- `data/events/detachment/extend_detach_vote/detach_shots.csv` holds the standard
  shot/category/start/end/confidence/attrs intervals; `attrs` carries `tier`
  (`certain` or `tangtv_only` on attached and detached intervals, the reason on
  uncertain ones).
- `$LABELER_ROOT/round4/detach/labels_bins.csv.gz` holds every bin with values,
  votes, validity reasons, geometry and Afrac probe provenance (exported for invalid
  bins too). `state_rule` is the primary label (relative f_div); `state_lm` is its
  legacy alias; `state_rule_absolute_prad` and `tier_absolute_prad` are the
  sensitivity columns that vote f_div with the 201081-anchored absolute cutoffs
  (`prad_abs_valid`, `prad_abs_vote`); `state_model_diagnostic` is the vestigial
  fitted model; `state_temporal_imputation` is a separate suggestion.
- The lower-shelf TangTV window is not a label column. It is absent from the
  interval table, the grids, `labels_bins.csv.gz`, `indicators/<shot>.csv` and the
  detach-ui handoff; its bins carry `state_rule = 4` and `tier =
  lower_shelf_window`. The per-indicator vote and value columns of those bins
  remain.
- Outputs use milliseconds; corpus HDF5 time axes use seconds. `shots.csv` is an
  unverified review roster.

## Indicators

**Afrac proxy.** Processed `LANGMUIR::TOP.PROBE_*:{R,Z,JSAT,TIME}` Jsat of every
positioned probe, as the inter-ELM median per bin and probe. A probe's current times
P_SOL^(3/7), divided by relative density squared, is divided by that probe's OWN
attached level: the 0.9-quantile of the quantity over the bins where the probe lies
within |psiN - 1| <= 0.01 of the separatrix (nearest multi-slice EFIT map; at least
20 such bins, or the probe has no reference and does not vote; ELM, ramp,
low-power and known-L-mode bins are left out). A bin's Afrac is that ratio on the
probe nearest the separatrix in flux among those with a reference, inside the same
window. Known L-mode bins abstain (reason `l_mode`; the regime is the confinement
suggestion table, then the D-alpha H-mode detector, and a bin of unknown regime is
not gated). Strike-point positions that are sentinel values are ignored. The
ratio is an uncalibrated local proxy, not the published Afrac, which needs
Eldon's fitted attached-current model: >= 0.75 votes attached, <= 0.5 detached,
the band between abstains. The probe, its position, psiN and distance from the
separatrix are exported per bin. The Afrac check records the median Afrac against
the selected probe's psiN before and after this design, a window sweep, and its
agreement with TangTV and with Te; Afrac stays in the vote only while its AUROC
against Te is at least 0.65.

**f_div = Prad,div,L / P_in.** Calibrated lower-divertor radiation over input power,
both averaged over the same centered 250 ms window of inter-ELM samples; D-alpha
must cover the whole window. P_in is the neutral beams plus the EFIT ohmic power plus
ECH, at least 0.5 MW. Where the corpus beam group is empty (26 shots) the beam
total is PTDATA `BMSPINJ` (1 V = 1 MW); the corpus total over `BMSPINJ` is 0.97 to
1.01 where both exist (`detachment_nbi_calibration.json`). Radiation below -0.05 MW
is invalid. The exported vote is the shot-RELATIVE one: f_div over the shot's own
baseline (the 10th percentile of f_div over its flat-top-power bins; a shot with
fewer than 40 such bins has no baseline and no vote, reason `no_baseline`), attached
below 1.052 and detached above 1.176. Those two global numbers are the band edges
of shot 201081 (the midpoint of its measured attached and detached Prad,div,L, plus
or minus 0.1 MW, measured in 250 ms inter-ELM windows placed from the published Te
cliffs) over its attached level (`detachment_prad_anchor.json`). The same anchor with
ABSOLUTE cutoffs, 0.399 and 0.446 of P_in (the band edges over the anchor's single
measured P_in of 4.274 MW, the median over the 32 attached-window bins of its
250 ms-averaged P_in), is the sensitivity alternative, exported as
`prad_abs_valid`/`prad_abs_vote`: the per-shot f_div tables show it does not
carry over from the anchor shot. f_div does not resolve MARFE and is not a MARFE
corroborator.

**TangTV front height.** The inversion gives the emission height ZE and
`DZ = 1 - (ZX - ZE)/(ZX - ZS)`. Frames need lower-single-null shelf geometry, an
outer leg of at least 10 cm, finite emission, DZ >= -0.25 and an EFIT slice within
40 ms; half the frames of a bin must pass. Camera geometry prefers EFIT02, with a
named EFIT01 fallback. DZ < 0.35 votes attached and 0.5 <= DZ < 1.2 detached. DZ >=
1.2 for two adjacent bins, with the global emission peak inside the separatrix near
the X-point (within 30 cm radially, ZX-2 cm to ZX+30 cm vertically) and a density
cue fG >= 0.8 (or a recorded H-L back-transition within 200 ms), is the TangTV MARFE
vote; it is exported as the uncertain tier `candidate_marfe`, never as a state.
fG uses unit-confirmed line-averaged density over the Greenwald density with an
elliptical chord, with about 10-20% geometric uncertainty; unknown units give no
cue. The upper shelf is Z near -1.25 m, R >= 1.37 m. The lower shelf (R < 1.37 m) is
extracted by the owner's code; its TangTV vote is invalid (reason
`lower_shelf_window`) and it never yields a state. The ridge surrogate
contributes no valid bins.

**ELM coverage.** Afrac and f_div use inter-ELM samples with +-2 ms D-alpha ELM
masks; TangTV accepts the integrated frames. Missing filterscope coverage gives
`elm_unknown`, which can never be certain.

## Primary rule

A bin takes the TangTV state when TangTV votes attached or detached on the upper
shelf with known ELM coverage. It is `certain` when at least one other cast vote
(the shot-relative f_div or Afrac) agrees and none disagrees, and `tangtv_only`
(silver) when TangTV is the only cast vote. A conflicting Afrac or f_div vote
blocks both: the bin is uncertain, tier `conflict`. A persistent high TangTV
front is `candidate_marfe`, an uncertain tier. Primary states are unsmoothed.

| Tier | State | Meaning |
|---|---|---|
| `certain` | 1/2 | TangTV vote plus an agreeing second indicator (relative f_div or Afrac), no conflict, upper shelf, known ELM coverage |
| `tangtv_only` | 1/2 | TangTV vote alone (silver): every other indicator abstains or is invalid |
| `conflict` | 4 | a cast vote disagrees with TangTV's, or the two others disagree with each other |
| `insufficient_support` | 4 | TangTV casts no attached or detached vote and the other votes do not make a pair |
| `low_confidence_pair` | 4 | the other two indicators agree and TangTV casts no vote |
| `no_vote` | 4 | no indicator casts a vote |
| `candidate_marfe` | 4 | TangTV MARFE vote or persistent high front (MARFE candidate); never a state |
| `lower_shelf_window` | 4 | restricted lower-shelf extraction, TangTV invalid, never a state |
| `geometry_unknown` / `elm_unknown` | 4 | shelf geometry or ELM coverage unknown |
| `not_assessed` | absent | fewer than two valid indicators and no TangTV vote |

The Snorkel-style label model is vestigial. Its weights are uncalibrated, they do not
identify physical accuracy, and it never decides a label.

## Results

<!-- RESULTS -->

### Coverage

| Group | Bins / shots | Seconds |
|---|---|---|
| Assessed | 1,779 / 32 | 89.0 |
| Certain or TangTV only | 673 / 25 | 33.6 |
| Certain attached | 132 / 15 | 6.60 |
| Certain detached | 143 / 13 | 7.15 |
| TangTV only attached | 219 / 15 | 10.95 |
| TangTV only detached | 179 / 13 | 8.95 |
| Tier `candidate_marfe` | 70 / 4 | 3.50 |
| Tier `certain` | 275 / 19 | 13.75 |
| Tier `conflict` | 447 / 29 | 22.35 |
| Tier `insufficient_support` | 163 / 24 | 8.15 |
| Tier `low_confidence_pair` | 67 / 11 | 3.35 |
| Tier `lower_shelf_window` | 325 / 6 | 16.25 |
| Tier `no_vote` | 34 / 15 | 1.70 |
| Tier `tangtv_only` | 398 / 22 | 19.90 |

A bin is assessed when at least two indicators are valid on it or TangTV votes; a shot is eligible when at least two indicators are each valid on at least 20 bins and it has at least 20 assessed bins. Tiers are disjoint and sum to the assessed bins; `lower_shelf_window` holds the owner's restricted lower-shelf extraction, whose TangTV vote is invalid (the shelf geometry does not hold) so it never yields a state.

### Indicator coverage, all extracted shots

| Indicator | Measurement | Valid | Vote | In a certain label |
|---|---|---|---|---|
| afrac | 909 / 14 | 909 / 14 | 722 / 14 | 128 / 6 |
| prad | 43,081 / 455 | 23,421 / 241 | 17,955 / 241 | 212 / 19 |
| tangtv | 1,813 / 41 | 1,291 / 28 | 1,098 / 28 | 275 / 19 |

Cells are bins / shots over 50,032 extracted bins on 470 shots.

### What corroborates the certain bins

| Certain bins | Count |
|---|---|
| all | 275 |
| relative f_div casts a vote | 212 |
| Afrac casts a vote | 128 |
| corroborated by f_div only | 147 |
| corroborated by Afrac only | 63 |
| corroborated by both | 65 |

### Divertor Thomson Te check

The processed divertor Thomson Te (`\ELECTRONS::TSTE_DIV`, 14 or 16 chords at R = 1.485 m, about 20 ms) is a temperature that none of the three indicators uses. A bin's Te is the median of the valid samples of the chords 1.5 to 5 cm above the outer shelf (Z = -1.25 m) that lie on the scrape-off side, 1.000 < psiN <= 1.05, by the multi-slice EFIT map; samples outside 0.1-50 eV are rejected. No threshold is fitted: the bands were fixed from the literature before scoring.

| Group | Bins / shots | Median Te, eV (10-90%) | Share in band |
|---|---|---|---|
| certain attached | 105 / 14 | 16.0 (7.8-22.1) | 0.78 |
| certain detached | 90 / 9 | 3.0 (1.4-10.7) | 0.54 |
| TangTV-only attached | 197 / 15 | 16.3 (8.2-22.7) | 0.86 |
| TangTV-only detached | 51 / 9 | 2.3 (1.4-6.8) | 0.88 |
| TangTV vote alone, attached | 393 / 18 | 16.0 (6.8-22.1) | 0.83 |
| TangTV vote alone, detached | 275 / 18 | 2.3 (1.4-9.4) | 0.74 |
| Afrac vote attached | 89 / 7 | 12.5 (2.2-20.3) | 0.67 |
| Afrac vote detached | 99 / 7 | 2.7 (1.3-17.5) | 0.61 |
| relative f_div vote attached | 264 / 21 | 4.9 (1.5-19.0) | 0.31 |
| relative f_div vote detached | 241 / 16 | 10.1 (1.5-18.3) | 0.32 |

Share in band is the fraction of bins with Te >= 10 eV for attached or <= 5 eV for detached. AUROC of -Te for detached against attached votes (pooled with a shot-bootstrap interval; within shots is the mean over the shots that have at least 5 bins of each class, with the number of such shots):

| Reference | Pooled AUROC [CI] | Bins / shots | Within-shot mean [CI] | Shots (within) |
|---|---|---|---|---|
| TangTV vote alone | 0.95 [0.90, 0.98] | 668 / 23 | 0.97 [0.94, 0.99] | 6 |
| certain tier | 0.91 [0.77, 0.96] | 195 / 17 | 0.95 [0.93, 0.97] | 2 |
| TangTV-only tier | 0.95 [0.90, 1.00] | 248 / 19 | 0.80 [0.80, 0.80] | 1 |
| TangTV vote in `conflict` bins | 0.96 [0.91, 0.99] | 225 / 23 | undefined | 0 |
| Afrac vote | 0.76 [0.44, 0.86] | 188 / 8 | 0.65 [0.33, 0.89] | 3 |
| relative f_div vote | 0.42 [0.23, 0.59] | 505 / 25 | 0.81 [0.65, 0.92] | 6 |

**Does the second vote improve the Te agreement?** No. Certain minus TangTV alone is -0.036 [-0.165, +0.011] over 23 shots (not distinguishable); against the TangTV votes that are TangTV only or in conflict it is -0.051 [-0.175, +0.007]. The within-shot figure for the certain tier rests on 2 shot(s), too few for any statement.

25 of the 32 assessed shots have a Thomson Te in an assessed bin; for the others no selected chord had a valid sample, or no Thomson data exist.

On 201081 the Te first falls below the detached band at 2575 ms and last does at 4425 ms (published cliffs 2650 and 4450 ms). No MARFE bin is exported, so the check does not cover MARFE.

### Indicator agreement, upper shelf

| Statistic | Value [95% shot CI] | Chance | Bins | Shots |
|---|---|---|---|---|
| AUROC of relative f_div, pooled | 0.41 [0.25, 0.57] | 0.5 | 1013 | 26 |
| AUROC of relative f_div, mean over shots | 0.80 [0.56, 0.98] | 0.5 | - | 8 |
| Spearman rho of relative f_div with DZ, pooled | -0.09 [-0.30, 0.18] | 0.0 | 1167 | 26 |
| Spearman rho, mean over shots | 0.36 [0.19, 0.54] | 0.0 | - | 26 |
| Cohen's kappa, attached against not attached | -0.16 [-0.44, 0.14] | 0.0 | 554 | 25 |
| Cohen's kappa, three states | -0.16 [-0.44, 0.14] | 0.0 | 554 | 25 |

AUROC is the probability that a TangTV-detached bin has a larger relative f_div than a TangTV-attached bin. Intervals resample shots (1000 replicates, seed 0). Kappa is null, and not drawn, where either rater used one class. The within-shot rows average over the few shots that have enough bins of both classes (the shot count is in the last column of the table). The same statistics for each pair of indicators:

| Shelf | Pair | AUROC [CI] | Bins | Shots | Within-shot AUROC | Shots (within) |
|---|---|---|---|---|---|---|
| upper shelf | relative f_div against TangTV | 0.41 [0.25, 0.57] | 1013 | 26 | 0.80 | 8 |
| upper shelf | absolute f_div against TangTV (sensitivity) | 0.79 [0.64, 0.92] | 1013 | 26 | 0.80 | 8 |
| upper shelf | Afrac against TangTV | 0.86 [0.66, 0.95] | 228 | 7 | 0.98 | 2 |
| upper shelf | relative f_div against Afrac | 0.64 [0.30, 0.87] | 191 | 7 | 0.87 | 3 |
| lower shelf window | relative f_div against TangTV | undefined | 0 | 0 | undefined | 0 |
| lower shelf window | absolute f_div against TangTV (sensitivity) | undefined | 0 | 0 | undefined | 0 |
| lower shelf window | Afrac against TangTV | undefined | 0 | 0 | undefined | 0 |
| lower shelf window | relative f_div against Afrac | 0.96 [0.80, 1.00] | 253 | 6 | 0.99 | 3 |

### Afrac proxy: dependence on the probe read

Median Afrac against the flux position of the probe it was read from, all assessed bins, and split by the TangTV vote. Before: the round-4 definition (peak-current probe against one whole-shot reference). After: each probe against its own attached reference, the probe nearest the separatrix within |psiN - 1| <= 0.01.

Before:

| psiN of the selected probe | Bins / shots | Median Afrac | TangTV attached (bins) | TangTV detached (bins) |
|---|---|---|---|---|
| (1, 1.005] | 761 / 37 | 0.79 | 0.69 (209) | 0.72 (107) |
| (1.005, 1.01] | 561 / 37 | 0.82 | 0.83 (74) | 0.87 (118) |
| (1.01, 1.02] | 1019 / 37 | 0.45 | 0.50 (114) | 0.49 (310) |
| (1.02, 1.03] | 347 / 26 | 0.17 | 0.34 (37) | 0.15 (28) |
| (1.03, 1.05] | 171 / 19 | 0.18 | 0.25 (16) | 0.08 (34) |

After:

| psiN of the selected probe | Bins / shots | Median Afrac | TangTV attached (bins) | TangTV detached (bins) |
|---|---|---|---|---|
| (0.98, 0.995] | 46 / 10 | 0.28 | 0.04 (5) | 0.08 (9) |
| (0.995, 1] | 285 / 13 | 0.72 | 0.95 (30) | 0.23 (54) |
| (1, 1.005] | 439 / 14 | 0.78 | 0.82 (51) | 0.26 (30) |
| (1.005, 1.01] | 139 / 14 | 0.68 | 0.68 (37) | 0.82 (12) |

Spearman rank correlation of Afrac with the selected probe's psiN, TangTV-attached bins: before -0.49 (450 bins, 19 shots), after -0.14 (123 bins, 6 shots); TangTV-detached bins: before -0.61 (597 bins, 23 shots), after 0.50 (105 bins, 5 shots). The dependence in the attached class is much reduced; the detached class keeps a positive rank correlation on few bins and shots (its median is 0.22 at psiN <= 1.005 on 93 bins and the 12 detached bins at 1.005-1.01 read 0.82), so the proxy is not free of position. On 201081 the median Afrac is 1.05 before the first cliff, 0.21 between the cliffs and 0.89 after the second, all from one probe.

The cause of the earlier failure: The earlier proxy read the peak-current probe among those with psiN in (1.000, 1.05] and at least 5 mm outboard of the outer strike point, against one whole-shot reference. The strike-point probe, which shows the rollover, was excluded, and which probe was read followed millimetre-scale strike-point motion; probe spacing (1.1 to 4.4 cm) is wider than the current peak, so a fixed-psiN interpolation was not an alternative. The proxy then varied with the read probe's flux position, not with the plasma state.

Window sensitivity (the primary window is 0.01; wider windows admit probes the EFIT strike-point error can mis-map):

| Window | Valid bins / shots | AUROC(-Afrac) vs TangTV [CI] | Shots | AUROC(-Te), vote [CI] | AUROC(-Afrac) cold vs warm Te [CI] | Shots |
|---|---|---|---|---|---|---|
| 0.01 | 909 / 14 | 0.86 [0.69, 0.95] | 7 | 0.75 [0.45, 0.86] | 0.76 [0.43, 0.88] | 8 |
| 0.015 | 1,367 / 23 | 0.74 [0.58, 0.86] | 15 | 0.70 [0.50, 0.83] | 0.67 [0.45, 0.81] | 12 |
| 0.02 | 1,750 / 28 | 0.68 [0.56, 0.80] | 20 | 0.66 [0.52, 0.78] | 0.64 [0.50, 0.76] | 21 |

Decision rule: Afrac stays in the vote unless its AUROC against Te is below 0.65 (the lower of the vote-based and the value-based point estimates at the primary window). Statistic 0.746; Afrac stays in the vote. Known-L-mode bins (780 bins) abstain; the regime is known for 1,005 of 4,124 extracted bins on 10 of 37 shots (confinement suggestion table, then the D-alpha H-mode detector) and unknown regime is not gated.

### f_div per shot

Per-shot f_div in the TangTV-attached and TangTV-detached upper-shelf bins of the 14 shots that have both (table), with the bins counted in brackets. P_in is the shot's median; the baseline is the 10th percentile of f_div over the shot's flat-top-power bins (the denominator of the relative f_div). Over all shots, the median absolute f_div in TangTV-attached bins ranges from 0.39 to 0.58 (19 shots) and in TangTV-detached bins from 0.40 to 0.57 (21 shots), while P_in ranges from 1.1 to 10.9 MW; the anchor shot's attached level is 0.38. The classes overlap across shots, so one pair of absolute cutoffs does not carry over from the anchor shot.

| Shot | P_in, MW | Baseline f_div | Attached f_div (bins) | Detached f_div (bins) | Attached relative | Detached relative |
|---|---|---|---|---|---|---|
| 189057 | 6.1 | 0.43 | 0.44 (10) | 0.54 (39) | 1.02 | 1.27 |
| 189081 | 9.3 | 0.47 | 0.51 (48) | 0.52 (2) | 1.09 | 1.10 |
| 189088 | 9.5 | 0.50 | 0.47 (3) | 0.52 (34) | 0.94 | 1.05 |
| 189090 | 9.5 | 0.53 | 0.58 (3) | 0.57 (51) | 1.09 | 1.07 |
| 189093 | 9.1 | 0.42 | 0.46 (39) | 0.52 (2) | 1.10 | 1.23 |
| 189094 | 9.1 | 0.48 | 0.48 (4) | 0.54 (36) | 1.01 | 1.14 |
| 189100 | 9.1 | 0.38 | 0.42 (13) | 0.40 (29) | 1.11 | 1.06 |
| 189101 | 9.2 | 0.39 | 0.44 (6) | 0.40 (42) | 1.13 | 1.02 |
| 199166 | 1.6 | 0.40 | 0.45 (25) | 0.52 (6) | 1.14 | 1.31 |
| 199352 | 1.5 | 0.41 | 0.46 (32) | 0.48 (2) | 1.12 | 1.19 |
| 199353 | 1.5 | 0.40 | 0.45 (19) | 0.49 (10) | 1.13 | 1.23 |
| 199354 | 1.5 | 0.40 | 0.45 (18) | 0.49 (13) | 1.12 | 1.21 |
| 200977 | 1.7 | 0.42 | 0.42 (15) | 0.48 (53) | 1.00 | 1.13 |
| 201081 | 4.3 | 0.34 | 0.39 (26) | 0.46 (40) | 1.15 | 1.36 |

Votes of f_div in the TangTV-voted upper-shelf bins:

| TangTV | f_div cutoffs | Bins / shots | Voted attached | Voted detached | Abstained |
|---|---|---|---|---|---|
| TangTV attached | relative | 442 / 19 | 92 | 104 | 246 |
| TangTV attached | absolute | 442 / 19 | 47 | 237 | 158 |
| TangTV detached | relative | 548 / 21 | 235 | 123 | 190 |
| TangTV detached | absolute | 548 / 21 | 29 | 475 | 44 |

The relative f_div measures change within a shot, not the state: a shot detached throughout has a detached baseline and reads about 1, so across shots the relative vote misses it (pooled AUROC below chance), while the absolute value ranks the states across shots but with cutoffs that fit one shot.

Published detached point 180257 at 4800 ms: P_in 7.2 MW, absolute f_div 0.171, which the absolute cutoffs vote attached against the publication; the relative f_div has no baseline (`no_baseline`), Afrac reads `probe_off_separatrix`, TangTV has no valid geometry (tier `none`) and the bin is not assessed. The absolute cutoffs therefore miss this point rather than rescue it, and the exported label makes no statement about it.

### Reference shot 201081 and published points

P_in = neutral beams + EFIT ohmic + ECH, 4.274 MW: the median over the 32 attached-window bins of the 250 ms-averaged P_in on 201081 (beams from PTDATA BMSPINJ, because the corpus `pinj` group is a stub on this shot). Prad,div,L is 1.621 MW attached and 1.990 MW detached in 250 ms inter-ELM windows placed from the published Te cliffs (32 attached and 28 detached bins).

| Shot | Time, ms | Published | Afrac | f_div rel. | f_div abs. | TangTV | Label | Tier |
|---|---|---|---|---|---|---|---|---|
| 180257 | 2400 | attached | abstain | abstain | abstain | abstain | unassessed | bin_not_assessed |
| 180257 | 4800 | detached | abstain | abstain | attached | abstain | unassessed | bin_not_assessed |
| 199166 | 3705 | marfe | abstain | detached | detached | abstain | uncertain | candidate_marfe |
| 201081 | 2300 | attached | attached | abstain | attached | attached | attached | certain |
| 201081 | 2500 | attached | attached | detached | abstain | abstain | uncertain | conflict |
| 201081 | 3000 | detached | detached | detached | detached | detached | detached | certain |
| 201081 | 3500 | detached | detached | detached | detached | detached | detached | certain |
| 201081 | 4000 | detached | detached | detached | detached | detached | detached | certain |
| 201081 | 4900 | attached | attached | abstain | attached | attached | attached | certain |
| 201081 | 5100 | attached | attached | abstain | attached | attached | attached | certain |

The ten points are published statements (Chen 2026; Eldon 2021). They overlap the sources that motivated the indicators, and 201081 also anchors the absolute f_div cutoffs, so the table is a sanity check, not a benchmark: the label casts 6 votes of 10 points, and 100% of those agree with the publication. The uncast points are 180257 (not assessed) and the MARFE on 199166 (uncertain) and the conflict bin on 201081. A vote of `abstain` means the indicator does not vote in that bin.

### Composition under the f_div cutoffs

| Variant | Cutoffs | Certain bins (att / det) | Certain shots (att / det) | TangTV-only bins (att / det) | Uncertain bins |
|---|---|---|---|---|---|
| primary_relative | 1.052 / 1.176 | 132 / 143 | 15 / 13 | 219 / 179 | 1106 |
| absolute | 0.399 / 0.446 | 56 / 474 | 5 / 20 | 178 / 52 | 1019 |
| primary_relative_afrac_abstains | 1.052 / 1.176 | 92 / 123 | 14 / 12 | 277 / 206 | 1081 |
| absolute_afrac_abstains | 0.399 / 0.446 | 47 / 475 | 4 / 21 | 189 / 60 | 1008 |
| absolute_published_anchor_values | 0.421 / 0.468 | 113 / 463 | 14 / 20 | 217 / 39 | 947 |
| relative_band_0.025_mw | 1.098 / 1.129 | 178 / 178 | 18 / 13 | 76 / 60 | 1287 |
| absolute_band_0.025_mw | 0.417 / 0.428 | 82 / 485 | 10 / 20 | 85 / 24 | 1103 |
| relative_band_0.05_mw | 1.083 / 1.145 | 167 / 149 | 18 / 13 | 131 / 111 | 1221 |
| absolute_band_0.05_mw | 0.411 / 0.434 | 68 / 478 | 6 / 20 | 122 / 36 | 1075 |
| relative_band_0.1_mw | 1.052 / 1.176 | 132 / 143 | 15 / 13 | 219 / 179 | 1106 |
| absolute_band_0.1_mw | 0.399 / 0.446 | 56 / 474 | 5 / 20 | 178 / 52 | 1019 |
| relative_band_0.15_mw | 1.021 / 1.206 | 106 / 131 | 12 / 9 | 294 / 314 | 934 |
| absolute_band_0.15_mw | 0.387 / 0.458 | 43 / 472 | 6 / 20 | 237 / 73 | 954 |
| relative_band_0.2_mw | 0.990 / 1.237 | 64 / 112 | 6 / 7 | 355 / 428 | 820 |
| absolute_band_0.2_mw | 0.376 / 0.469 | 38 / 458 | 4 / 18 | 296 / 94 | 893 |
| relative_band_0.3_mw | 0.929 / 1.299 | 62 / 96 | 7 / 5 | 378 / 458 | 785 |
| absolute_band_0.3_mw | 0.352 / 0.493 | 46 / 349 | 4 / 17 | 363 / 206 | 815 |

`primary_relative` is the exported label: the per-shot relative f_div, cutoffs scaling the shot baseline. `absolute` uses the 201081-anchored global cutoffs (the midpoint of the measured attached and detached Prad,div,L over its P_in; one pair for every shot). `band_X` sets the half-width of the uncertain band to X MW (the primary keeps 0.1 MW). `afrac_abstains` removes the Afrac vote. `published_anchor_values` uses the published 1.6 / 2.2 MW.

Decision criterion: attached AND detached bins on at least 3 of the same shots, at least one of them a cohort shot (split train, val or test). Certain attached and certain detached bins coexist on 9 shots (0 in the cohort); counting the TangTV-only tier as well, 13 shots (0 in the cohort). The criterion is not met, so the label set is presented as an indicator-agreement appendix, not as a three-state label set.

### Bin width

| Width | Certain bins (att / det) | Certain shots (att / det) | TangTV-only bins (att / det) | Assessed bins | Uncertain share | Flicker per s | Agreement with 50 ms (bins) |
|---|---|---|---|---|---|---|---|
| 20ms | 310 / 361 | 15 / 18 | 481 / 318 | 3,428 | 0.57 | 0.476 | 0.993 (1359) |
| 50ms | 110 / 103 | 14 / 12 | 219 / 217 | 1,237 | 0.48 | 0.154 | - |
| 100ms | 25 / 9 | 4 / 1 | 205 / 279 | 538 | 0.04 | 0.212 | 0.991 (317) |

The same compatibility rule at each width on the 77 non-test shots of the width roster (27 of them assessed at 50 ms; the roster is not the exported shot list, so its 50 ms row differs from the coverage table above; the widths are comparable with each other). The certain tier depends on the width: 20 ms gives 310 attached / 361 detached certain bins (15 / 18 shots), 50 ms 110 attached / 103 detached certain bins (14 / 12 shots) and 100 ms 25 attached / 9 detached certain bins (4 / 1 shots). The second indicator votes on fewer of the wider bins, so the TangTV-only tier takes over. Where two widths both label a bin the states agree: 0.993 at 20ms (1359 bins), 0.991 at 100ms (317 bins).

### MARFE witness

**Recall of the published MARFE on 199166 at 3705 ms (Chen 2026): 0/1 as a state** (no bin is exported as MARFE). The bin holding that time (3700-3750 ms) is the uncertain tier `candidate_marfe` (1/1 as a candidate). In the bins from 3700 ms the TangTV front is at DZ 1.24, 1.30, 1.30 (cutoff 1.2), the emission peak is inside the separatrix from 3750 ms, and fG is 0.72, 0.74, 0.75, below the 0.8 cue, so the full MARFE vote does not fire on this shot. The cue's value has no literature source (Dong 2025: fG of about 0.5 or more on HL-3 with a core-point density and a core-Te condition), so even where it passes the MARFE stays a candidate in the exports and figures; the full vote fires on 23 bins of one other shot. The witness figure shows the frame at 3699 ms. The thresholds were not tuned to this shot.

<!-- /RESULTS -->

## Figures and reproduction

Figures are vector PDF and 150 dpi PNG with text of at least 7 pt, in colour-blind
safe colours, drawn at the 6.75 inch text width unless noted:

- `fig_detachment_views`: camera inversion, EFIT contours, Prad,div,L and P_rad
  traces for shot 201081 at time-ordered columns taken from sustained TangTV-attached,
  TangTV-detached and TangTV-attached intervals (at least five bins). Each column is
  titled by its TangTV vote, with the exported label of the bins in the interval in
  brackets; the legend and caption say so. The X-point and the outer strike point
  are marked. No 2D bolometer emissivity exists
  (`detachment_bolometer_availability.json`), so the radiation row shows the
  PRAD_DIVL and PRAD_TOT traces.
- `fig_detachment_timeline`: label, Afrac/relative f_div/TangTV vote strips, relative
  f_div and DZ with their cutoffs, and the divertor Thomson Te; below, per-shot bars
  of certain, TangTV-only and uncertain bins with the shot IDs on the axis.
- `fig_detachment_marfe_witness`: the 199172 MARFE candidate (no certain MARFE) and
  the published 199166 onset, with the X-point marked, the frame time, the label
  bin and its tier, and the recall (0 of 1).
- `fig_detachment_figure2` (3.25 inch panel): coverage by class and tier,
  the AUROC of -Te per vote source and the threshold-free agreement of the
  relative f_div with TangTV, with the number of shots beside every row; the
  source is `docs/labeler/figure2_detach.json`, the one canonical record.

Regenerate in order: `detach_fetch_round4.py` (login node, `fdp run`, one worker,
stops at the first authentication error), `detach_bins.py`, `detach_label.py`,
`detach_prad_anchor.py`, `detach_reference.py`, `detach_benchmark.py`,
`detach_te_check.py`, `detach_afrac_check.py`, `detach_prad_sensitivity.py`,
`detach_current_state.py`, `detach_physics_audit.py`, `detach_probe_bolom.py`,
`detach_bins.py --width-ms 20/50/100` and `detach_sensitivity.py`,
`detach_figure.py`, `detach_paper_panel.py`, `detach_handoff.py`,
`detach_docs_tables.py` and `detach_validate_outputs.py`. Large outputs and
`HANDOFF.md` are under `$LABELER_ROOT/round4/detach/`.

## Limitations

The published points are few (ten, on three shots) and share their sources with the
indicator definitions, so they support neither accuracy nor calibration. The
certain tier measures agreement between indicators, not physical accuracy, and the
second vote does not improve its agreement with divertor Thomson Te over the TangTV
state alone, and the silver tier is as Te-consistent as the certain tier. The
Afrac proxy is a weak second indicator: it is valid on few shots, its Te
interval includes chance, its result depends on the flux window, and the L-mode
gate knows the regime on a minority of shots. The relative f_div vote agrees with
TangTV within shots but not across shots (pooled AUROC below 0.5), and the
absolute cutoffs do not carry over from their anchor shot either; many bins are
therefore `conflict` bins. Only one assessed shot is in the fixed cohort and none
is a test shot; certain attached and detached bins never coexist on a cohort
shot, so the set cannot support the cohort benchmark or a study of
attached-to-detached transitions. No bin is exported as MARFE: the one published
MARFE (199166 at 3705 ms) is a `candidate_marfe` bin, recalled 0 of 1 as a
state. The Thomson check covers attached and detached only, on the shots that have
a chord near the target; the within-shot Te figures rest on a handful of shots. The
width of the bins changes the certain tier strongly. Indicator agreement can share
systematic biases. Future work needs an independent expert state table, more
inversion coverage and a literature-backed density cue for MARFE.

## Appendix: learned baselines

<!-- BASELINES -->

Two learned baselines, `detach-ours` (windows of current, density, D-alpha, ELM share and EFIT scalars) and `detach-victor` (raw camera frames), were trained on an earlier build of the labels (labels sha256 `df8747c48aa1`), before the Afrac proxy, the per-shot f_div vote and the tier names changed. They have not been retrained on the labels described here, so no score of either is quoted: a score against a different label set would not describe this one. They are not listed among the dataset's models and are not paper-facing results. Their records (`docs/labeler/results/detachment_ours.json` and `detachment_victor.json`, scripts `detach_ours.py` and `detach_victor.py`) stay in the repository for the earlier build; retraining them is open work.

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
