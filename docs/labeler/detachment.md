# Detachment protocol

<!-- SUMMARY -->

**Label set: the geometry-gated TangTV state, validated by divertor Thomson Te. Exploratory; no gold-standard benchmark.** Each 50 ms bin on an eligible shot is attached, detached or uncertain; MARFE is never a state. Two indicators vote: the TangTV C-III front height DZ and the target-current ratio (Afrac proxy: a probe's Jsat over that probe's own attached level, the probe nearest the separatrix, known L-mode bins abstain). The lower-divertor radiated fraction f_div = Prad,div,L / P_in is measured on every bin but is not a vote: it corroborates within a shot and creates no conflicts (below). A bin is `certain` when TangTV votes and a valid Afrac vote agrees with it; `tangtv_only` (silver) when TangTV votes and Afrac abstains or is invalid; `conflict` when TangTV and Afrac vote differently; every other assessed bin is uncertain, with the reason in the `tier` column. Certain labels measure indicator agreement, not physical accuracy.

**Coverage.** 1,779 assessed bins on 32 shots (89 s). Certain: 147 bins on 7 shots (attached 60 on 5 shots, detached 87 on 4 shots). TangTV only: 861 bins on 26 shots (attached 393, detached 468); in 809 of them Afrac is invalid (no reference yet, known L-mode, probe off the separatrix) and in 52 it sits between its cutoffs. The other 771 bins are uncertain; 29 of them are `conflict`. Of the 32 assessed shots 1 is in the fixed cohort (val split, 57 assessed bins of which 0 certain and 57 TangTV only) and none is a test shot.

**Divertor Thomson Te check.** AUROC of -Te for a detached against an attached vote (pooled, with shot-bootstrap 95% intervals): TangTV alone 0.95 [0.90, 0.98] (668 bins, 23 shots), certain tier 0.94 [0.04, 0.97] (104 bins, 6 shots; the interval is uninformative on so few shots), TangTV-only tier 0.95 [0.91, 0.98] (550 bins, 22 shots). Within shots: TangTV alone 0.97 [0.94, 0.99] over 6 shot(s), certain 0.95 [0.95, 0.95] over 2 shot(s), TangTV only 0.98 [0.94, 1.00] over 4 shot(s). **The second vote does not improve the Te agreement of the TangTV state**: certain minus TangTV alone is -0.008 [-0.923, +0.027] over 23 shots (not distinguishable). What the second vote buys is agreement between two indicators, not a better match to Te.

**Afrac proxy rebuilt.** The earlier proxy read the peak-current probe against one whole-shot reference and followed the probe's flux position (median 0.79 within 0.005 of the separatrix and 0.17 at psiN 1.02-1.03, for the same plasma). Each probe now has its own attached reference and the probe nearest the separatrix inside |psiN - 1| <= 0.01 is read; the results show the median Afrac against the selected-probe psiN before and after. Against TangTV its AUROC is 0.86 [0.66, 0.95] (228 bins, 7 shots); against Te 0.75 by vote [0.45, 0.86] and 0.76 by value. It stays the second vote (rule: removed below 0.65), but only as a weak one: it is valid on 909 of 50,032 extracted bins (14 of 470 shots), the Te interval includes 0.5, the regime is known on 10 of 37 shots (so the L-mode gate removes 780 bins and misses the rest) and a wider window (0.02) puts the value-based AUROC at 0.64, below the removal bound.

**f_div is a within-shot corroborator, not a vote.** Pooled over shots its level moves with the shot, not with the state: per-shot medians of the absolute f_div in TangTV-attached bins run from 0.39 to 0.58 (19 shots) and in TangTV-detached bins from 0.40 to 0.57 (21 shots), with P_in from 1.1 to 10.9 MW, so no cutoff pair carries over between shots. Counted as a second vote (shot-relative cutoffs 1.052 / 1.176) it would give 275 certain bins but 447 conflicts, against 147 and 29; it is a sensitivity row. As a corroborator, per shot with at least 5 bins of each class: the relative f_div ranks TangTV-detached above TangTV-attached bins with a within-shot AUROC of 0.80 [0.58, 0.98] over 8 shots (6 above 0.5), and cold above warm divertor Thomson Te with 0.69 [0.54, 0.86] over 7 shots (4 above 0.5); pooled over shots the same scores give 0.41 [0.25, 0.57] on 990 bins, 26 shots against TangTV and 0.40 [0.20, 0.58] on 690 bins, 25 shots against Te, i.e. no support once the shots are mixed. The absolute f_div (201081-anchored cutoffs 0.399 / 0.446, P_in 4.274 MW) ranks the same bins within a shot and better pooled (against TangTV 0.78 [0.63, 0.92] on 990 bins, 26 shots), but its cutoffs vote detached on most TangTV-attached bins where it votes. Counting only the relative votes it casts, their AUROC against Te is 0.42 [0.25, 0.59] pooled. The published detached point 180257 at 4800 ms is not assessed (TangTV has no valid geometry, the relative f_div has no baseline, Afrac reads `probe_off_separatrix`); the absolute cutoffs would call it attached (f_div 0.17 at P_in 7.2 MW), a miss of that published point.

**Reference shot 201081.** Measured P_in is 4.274 MW (the median over the 32 attached-window bins of the 250 ms-averaged P_in); Prad,div,L is 1.621 MW attached and 1.990 MW detached. Of 7 published points on 201081, 6 are labelled in agreement with the publication, 1 uncertain and 0 contradicted.

**MARFE.** No bin is exported as MARFE: the density cue (fG >= 0.8) has no literature source (Dong 2025 gives fG of about 0.5 or more on HL-3 from a core-point density, with a core-Te condition and a gate of 0.40, and says thresholds are device-specific). A persistent high TangTV front is the uncertain tier `candidate_marfe`: 70 bins on 4 shots; the full TangTV MARFE vote (front, emission peak inside the separatrix and the density cue) fires on 23 bins of 1 shot only. **Recall of the published MARFE onset on 199166 (3705 ms): 0/1 as a label**; the bin holding it is `candidate_marfe` (1/1 as a candidate), with fG = 0.72 there against the 0.8 cue.

**Three-state label set.** Not offered: certain attached and certain detached bins coexist on 2 shots, 0 of them in the fixed cohort, against a requirement of at least three shots including one cohort shot. The result is an indicator-agreement appendix.

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
`certain` when a valid Afrac vote agrees with TangTV, `tangtv_only` (silver) when
TangTV votes and Afrac abstains or is invalid, `conflict` (uncertain) when Afrac
votes against TangTV. f_div is not a vote: it is reported per shot as a
corroborator (see the results). A persistent high front, the MARFE candidate, is
the uncertain tier `candidate_marfe`: the density cue that would make it a MARFE
has no literature source (see the MARFE results). In the preceding build f_div was
a third voter; its tiers are the `with_relative_fdiv` row of the composition table.

The original plan for this label set called for a benchmark of each single indicator
against the combined label, and optionally for learned detectors. That is replaced by the
independent divertor Thomson Te check: scoring an indicator against a label built
from it is circular, while Te is read by none of the three indicators. The
single-indicator agreement tables remain as agreement diagnostics, and the learned
baselines are retrained on this label set (appendix).

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
  bins too). `state_rule` is the primary label (TangTV with Afrac as the second
  vote); `state_lm` is its legacy alias; `state_rule_relative_prad` /
  `tier_relative_prad` and `state_rule_absolute_prad` / `tier_absolute_prad` are
  the sensitivity columns that add f_div as a second voter next to Afrac, with the
  shot-relative or the 201081-anchored absolute cutoffs (`prad_abs_valid`,
  `prad_abs_vote`); `state_model_diagnostic` is the vestigial fitted model;
  `state_temporal_imputation` is a separate suggestion.
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
agreement with TangTV and with Te; Afrac stays the second vote only while its AUROC
against Te is at least 0.65.

**f_div = Prad,div,L / P_in.** Calibrated lower-divertor radiation over input power,
both averaged over the same centered 250 ms window of inter-ELM samples; D-alpha
must cover the whole window. P_in is the neutral beams plus the EFIT ohmic power plus
ECH, at least 0.5 MW. Where the corpus beam group is empty (26 shots) the beam
total is PTDATA `BMSPINJ` (1 V = 1 MW); the corpus total over `BMSPINJ` is 0.97 to
1.01 where both exist (`detachment_nbi_calibration.json`). Radiation below -0.05 MW
is invalid. f_div is a within-shot corroborator and is NOT a vote of the exported
label: pooled over shots its level moves with the shot (input power, seeding,
strike-point geometry), and counted as a vote it only created conflicts. Its
shot-RELATIVE value is f_div over the shot's own baseline (the 10th percentile of
f_div over its flat-top-power bins; a shot with fewer than 40 such bins has no
baseline and no value, reason `no_baseline`); the relative votes, attached below
1.052 and detached above 1.176, stay as columns and as a sensitivity. Those two global numbers are the band edges
of shot 201081 (the midpoint of its measured attached and detached Prad,div,L, plus
or minus 0.1 MW, measured in 250 ms inter-ELM windows placed from the published Te
cliffs) over its attached level (`detachment_prad_anchor.json`). The same anchor with
ABSOLUTE cutoffs, 0.399 and 0.446 of P_in (the band edges over the anchor's single
measured P_in of 4.274 MW, the median over the 32 attached-window bins of its
250 ms-averaged P_in), is the second sensitivity, exported as
`prad_abs_valid`/`prad_abs_vote`: the per-shot f_div tables show it does not
carry over from the anchor shot. `detach_fdiv_check.py` scores f_div within each
shot against TangTV and against Te (counts of shots beside every summary). f_div
does not resolve MARFE and is not a MARFE corroborator.

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
shelf with known ELM coverage. It is `certain` when a valid Afrac vote agrees with
it, and `tangtv_only` (silver) when Afrac casts no vote (it abstains between its
cutoffs or is invalid). An Afrac vote against TangTV's makes the bin uncertain,
tier `conflict`. Only TangTV and Afrac vote; f_div never supports or blocks a
state. A persistent high TangTV front is `candidate_marfe`, an uncertain tier.
Primary states are unsmoothed.

| Tier | State | Meaning |
|---|---|---|
| `certain` | 1/2 | TangTV vote plus an agreeing valid Afrac vote, no Afrac conflict, upper shelf, known ELM coverage |
| `tangtv_only` | 1/2 | TangTV vote alone (silver): Afrac abstains or is invalid |
| `conflict` | 4 | TangTV and Afrac cast different votes |
| `insufficient_support` | 4 | TangTV casts no attached or detached vote; Afrac votes alone |
| `no_vote` | 4 | neither TangTV nor Afrac casts a vote |
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
| Certain or TangTV only | 1,008 / 26 | 50.4 |
| Certain attached | 60 / 5 | 3.00 |
| Certain detached | 87 / 4 | 4.35 |
| TangTV only attached | 393 / 17 | 19.65 |
| TangTV only detached | 468 / 20 | 23.40 |
| Tier `candidate_marfe` | 70 / 4 | 3.50 |
| Tier `certain` | 147 / 7 | 7.35 |
| Tier `conflict` | 29 / 5 | 1.45 |
| Tier `insufficient_support` | 224 / 13 | 11.20 |
| Tier `lower_shelf_window` | 325 / 6 | 16.25 |
| Tier `no_vote` | 123 / 24 | 6.15 |
| Tier `tangtv_only` | 861 / 26 | 43.05 |

A bin is assessed when at least two indicators are valid on it or TangTV votes; a shot is eligible when at least two indicators are each valid on at least 20 bins and it has at least 20 assessed bins. Tiers are disjoint and sum to the assessed bins; `lower_shelf_window` holds the owner's restricted lower-shelf extraction, whose TangTV vote is invalid (the shelf geometry does not hold) so it never yields a state.

### Indicator coverage, all extracted shots

| Indicator | Measurement | Valid | Vote | In a certain label |
|---|---|---|---|---|
| afrac | 909 / 14 | 909 / 14 | 722 / 14 | 147 / 7 |
| prad (f_div, not a vote of the label) | 43,081 / 455 | 23,421 / 241 | 17,955 / 241 | 84 / 7 |
| tangtv | 1,813 / 41 | 1,291 / 28 | 1,098 / 28 | 147 / 7 |

Cells are bins / shots over 50,032 extracted bins on 470 shots. For f_div the Vote column counts its relative votes (a sensitivity: the label does not use them) and the last column the certain bins in which it casts one.

### What supports the labelled bins

| Labelled bins | Count |
|---|---|
| certain bins (TangTV vote and an agreeing Afrac vote) | 147 |
| TangTV-only bins | 861 |
|   Afrac valid, between its cutoffs (abstains) | 52 |
|   Afrac invalid | 809 |
|     reason `short_reference` | 311 |
|     reason `l_mode` | 277 |
|     reason `probe_off_separatrix` | 195 |
|     reason `elm` | 22 |
|     reason `no_power` | 3 |
|     reason `probe_flux_unknown` | 1 |

A TangTV-only bin keeps TangTV's state because Afrac does not vote there; it is not evidence against it. Most of these bins have no Afrac because the probe's reference is too short, the shot is in L-mode or no probe sits near the separatrix.

### Divertor Thomson Te check

The processed divertor Thomson Te (`\ELECTRONS::TSTE_DIV`, 14 or 16 chords at R = 1.485 m, about 20 ms) is a temperature that none of the three indicators (TangTV, Afrac, f_div) uses. A bin's Te is the median of the valid samples of the chords 1.5 to 5 cm above the outer shelf (Z = -1.25 m) that lie on the scrape-off side, 1.000 < psiN <= 1.05, by the multi-slice EFIT map; samples outside 0.1-50 eV are rejected. No threshold is fitted: the bands were fixed from the literature before scoring.

| Group | Bins / shots | Median Te, eV (10-90%) | Share in band |
|---|---|---|---|
| certain attached | 50 / 5 | 13.1 (7.8-21.6) | 0.74 |
| certain detached | 54 / 3 | 2.0 (1.2-9.0) | 0.78 |
| TangTV-only attached | 335 / 16 | 16.4 (7.8-22.4) | 0.86 |
| TangTV-only detached | 215 / 16 | 2.3 (1.5-9.6) | 0.73 |
| TangTV vote alone, attached | 393 / 18 | 16.0 (6.8-22.1) | 0.83 |
| TangTV vote alone, detached | 275 / 18 | 2.3 (1.4-9.4) | 0.74 |
| Afrac vote attached | 89 / 7 | 12.5 (2.2-20.3) | 0.67 |
| Afrac vote detached | 99 / 7 | 2.7 (1.3-17.5) | 0.61 |
| relative f_div vote (unused) attached | 264 / 21 | 4.9 (1.5-19.0) | 0.31 |
| relative f_div vote (unused) detached | 241 / 16 | 10.1 (1.5-18.3) | 0.32 |

Share in band is the fraction of bins with Te >= 10 eV for attached or <= 5 eV for detached. AUROC of -Te for detached against attached votes (pooled with a shot-bootstrap interval; within shots is the mean over the shots that have at least 5 bins of each class, with the number of such shots):

| Reference | Pooled AUROC [CI] | Bins / shots | Within-shot mean [CI] | Shots (within) |
|---|---|---|---|---|
| TangTV vote alone | 0.95 [0.90, 0.98] | 668 / 23 | 0.97 [0.94, 0.99] | 6 |
| certain tier | 0.94 [0.04, 0.97] | 104 / 6 | 0.95 [0.95, 0.95] | 2 |
| TangTV-only tier | 0.95 [0.91, 0.98] | 550 / 22 | 0.98 [0.94, 1.00] | 4 |
| TangTV vote in `conflict` bins | 0.98 [0.94, 1.00] | 14 / 5 | undefined | 0 |
| Afrac vote | 0.76 [0.47, 0.87] | 188 / 8 | 0.65 [0.33, 0.89] | 3 |
| relative f_div votes (unused) | 0.42 [0.25, 0.59] | 505 / 25 | 0.81 [0.67, 0.92] | 6 |

**Does the second vote improve the Te agreement?** No. Certain minus TangTV alone is -0.008 [-0.923, +0.027] over 23 shots (not distinguishable); against the TangTV votes that are TangTV only or in conflict it is -0.010 [-0.938, +0.036]. The within-shot figure for the certain tier rests on 2 shot(s), too few for any statement.

25 of the 32 assessed shots have a Thomson Te in an assessed bin; for the others no selected chord had a valid sample, or no Thomson data exist.

On 201081 the Te first falls below the detached band at 2575 ms and last does at 4425 ms (published cliffs 2650 and 4450 ms). No MARFE bin is exported, so the check does not cover MARFE.

### f_div and the indicator pairs, upper shelf

| Statistic | Value [95% shot CI] | Chance | Bins | Shots |
|---|---|---|---|---|
| AUROC of relative f_div, pooled | 0.41 [0.25, 0.57] | 0.5 | 1013 | 26 |
| AUROC of relative f_div, mean over shots | 0.80 [0.56, 0.98] | 0.5 | - | 8 |
| Spearman rho of relative f_div with DZ, pooled | -0.09 [-0.30, 0.18] | 0.0 | 1167 | 26 |
| Spearman rho, mean over shots | 0.36 [0.19, 0.54] | 0.0 | - | 26 |
| Cohen's kappa, attached against not attached | -0.16 [-0.44, 0.14] | 0.0 | 554 | 25 |
| Cohen's kappa, three states | -0.16 [-0.44, 0.14] | 0.0 | 554 | 25 |

Agreement diagnostics; none of them decides a label. AUROC is the probability that a TangTV-detached (or TangTV-MARFE-vote) bin has a larger relative f_div than a TangTV-attached bin; the per-shot f_div corroborator table below counts the detached votes alone. Intervals resample shots (1000 replicates, seed 0). Kappa is null, and not drawn, where either rater used one class. The within-shot rows average over the few shots that have enough bins of both classes (the shot count is in the last column of the table). The same statistics for each pair of indicators:

| Shelf | Pair | AUROC [CI] | Bins | Shots | Within-shot AUROC | Shots (within) |
|---|---|---|---|---|---|---|
| upper shelf | relative f_div against TangTV (corroborator) | 0.41 [0.25, 0.57] | 1013 | 26 | 0.80 | 8 |
| upper shelf | absolute f_div against TangTV (sensitivity) | 0.79 [0.64, 0.92] | 1013 | 26 | 0.80 | 8 |
| upper shelf | Afrac against TangTV | 0.86 [0.66, 0.95] | 228 | 7 | 0.98 | 2 |
| upper shelf | relative f_div against Afrac | 0.64 [0.30, 0.87] | 191 | 7 | 0.87 | 3 |
| lower shelf window | relative f_div against TangTV (corroborator) | undefined | 0 | 0 | undefined | 0 |
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

### f_div as a within-shot corroborator

f_div is not a vote, so it creates no conflicts and no certain bins. It is scored as a ranking: per shot, the AUROC of f_div for the TangTV-detached bins against the TangTV-attached ones (upper shelf), and for the divertor-Thomson-cold bins (Te <= 5 eV) against the warm ones (>= 10 eV); a shot counts when each class has at least 5 bins; the mean is over those shots, with a shot-bootstrap interval, and the pooled AUROC mixes the shots. The relative f_div is the ratio over the shot's own baseline, valid where the baseline exists; the absolute ranks the bins of one shot the same way (it differs by a per-shot constant) and is the 201081-anchored sensitivity. Record: `docs/labeler/results/detachment_fdiv_check.json`.

| Score against reference | Within-shot mean [CI] | Shots | Shots above 0.5 | Pooled AUROC [CI] | Pooled bins / shots |
|---|---|---|---|---|---|
| relative f_div against TangTV | 0.80 [0.58, 0.98] | 8 | 6 | 0.41 [0.25, 0.57] | 990 / 26 |
| relative f_div against Te | 0.69 [0.54, 0.86] | 7 | 4 | 0.40 [0.20, 0.58] | 690 / 25 |
| absolute f_div against TangTV (sensitivity) | 0.80 [0.56, 0.98] | 8 | 6 | 0.78 [0.63, 0.92] | 990 / 26 |
| absolute f_div against Te (sensitivity) | 0.69 [0.54, 0.86] | 7 | 4 | 0.78 [0.69, 0.86] | 690 / 25 |

Per shot (relative f_div; cells are AUROC with the detached or cold / attached or warm bin counts in brackets; shots with fewer than 5 bins in a class are left out of a column):

| Shot | Split | Against TangTV | Against Te |
|---|---|---|---|
| 189057 | outside | 0.93 (38 / 10) | 0.79 (25 / 11) |
| 189081 | outside | - | 0.47 (7 / 13) |
| 189090 | outside | - | 0.43 (23 / 6) |
| 189093 | outside | - | 0.69 (17 / 30) |
| 189097 | outside | - | 0.50 (5 / 42) |
| 189100 | outside | 0.39 (29 / 13) | - |
| 189101 | outside | 0.16 (42 / 6) | - |
| 199166 | outside | 1.00 (6 / 25) | 1.00 (8 / 29) |
| 199353 | outside | 0.95 (10 / 19) | - |
| 199354 | outside | 0.93 (13 / 18) | - |
| 200977 | outside | 1.00 (53 / 15) | - |
| 201081 | outside | 1.00 (40 / 26) | 0.96 (29 / 49) |

### f_div per shot, the absolute cutoffs

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

The relative f_div measures change within a shot, not the state: a shot detached throughout has a detached baseline and reads about 1, so across shots the relative scores miss it (pooled AUROC below chance), while the absolute value ranks the states across shots but with cutoffs that fit one shot. Neither is used by the label.

Published detached point 180257 at 4800 ms: P_in 7.2 MW, absolute f_div 0.171, which the absolute cutoffs vote attached against the publication; the relative f_div has no baseline (`no_baseline`), Afrac reads `probe_off_separatrix`, TangTV has no valid geometry (tier `none`) and the bin is not assessed. The absolute cutoffs therefore miss this point rather than rescue it, and the exported label makes no statement about it.

### Reference shot 201081 and published points

P_in = neutral beams + EFIT ohmic + ECH, 4.274 MW: the median over the 32 attached-window bins of the 250 ms-averaged P_in on 201081 (beams from PTDATA BMSPINJ, because the corpus `pinj` group is a stub on this shot). Prad,div,L is 1.621 MW attached and 1.990 MW detached in 250 ms inter-ELM windows placed from the published Te cliffs (32 attached and 28 detached bins).

| Shot | Time, ms | Published | Afrac | f_div rel. (unused) | f_div abs. (unused) | TangTV | Label | Tier |
|---|---|---|---|---|---|---|---|---|
| 180257 | 2400 | attached | abstain | abstain | abstain | abstain | unassessed | bin_not_assessed |
| 180257 | 4800 | detached | abstain | abstain | attached | abstain | unassessed | bin_not_assessed |
| 199166 | 3705 | marfe | abstain | detached | detached | abstain | uncertain | candidate_marfe |
| 201081 | 2300 | attached | attached | abstain | attached | attached | attached | certain |
| 201081 | 2500 | attached | attached | detached | abstain | abstain | uncertain | insufficient_support |
| 201081 | 3000 | detached | detached | detached | detached | detached | detached | certain |
| 201081 | 3500 | detached | detached | detached | detached | detached | detached | certain |
| 201081 | 4000 | detached | detached | detached | detached | detached | detached | certain |
| 201081 | 4900 | attached | attached | abstain | attached | attached | attached | certain |
| 201081 | 5100 | attached | attached | abstain | attached | attached | attached | certain |

The ten points are published statements (Chen 2026; Eldon 2021). They overlap the sources that motivated the indicators, and 201081 also anchors the absolute f_div cutoffs, so the table is a sanity check, not a benchmark: the label casts 6 votes of 10 points, and 100% of those agree with the publication. The uncast points are 180257 at 2400 ms (bin_not_assessed), 180257 at 4800 ms (bin_not_assessed), 199166 at 3705 ms (candidate_marfe), 201081 at 2500 ms (insufficient_support). A vote of `abstain` means the indicator does not vote in that bin; the f_div columns are shown for reference, the label does not use them.

### Composition under other second voters

| Variant | Second voters | f_div cutoffs | Certain bins (att / det) | Certain shots (att / det) | TangTV-only bins (att / det) | Conflict bins | Uncertain bins |
|---|---|---|---|---|---|---|---|
| primary | afrac | - | 60 / 87 | 5 / 4 | 393 / 468 | 29 | 771 |
| tangtv_alone | none | - | 0 / 0 | 0 / 0 | 473 / 564 | 0 | 742 |
| with_relative_fdiv | afrac, prad | 1.052 / 1.176 | 132 / 143 | 15 / 13 | 219 / 179 | 447 | 1106 |
| with_absolute_fdiv | afrac, prad | 0.399 / 0.446 | 56 / 474 | 5 / 20 | 178 / 52 | 355 | 1019 |
| with_relative_fdiv_afrac_abstains | prad | 1.052 / 1.176 | 92 / 123 | 14 / 12 | 277 / 206 | 339 | 1081 |
| with_absolute_fdiv_afrac_abstains | prad | 0.399 / 0.446 | 47 / 475 | 4 / 21 | 189 / 60 | 266 | 1008 |
| with_absolute_fdiv_published_anchor_values | afrac, prad | 0.421 / 0.468 | 113 / 463 | 14 / 20 | 217 / 39 | 278 | 947 |
| with_relative_fdiv_band_0.025_mw | afrac, prad | 1.098 / 1.129 | 178 / 178 | 18 / 13 | 76 / 60 | 654 | 1287 |
| with_absolute_fdiv_band_0.025_mw | afrac, prad | 0.417 / 0.428 | 82 / 485 | 10 / 20 | 85 / 24 | 451 | 1103 |
| with_relative_fdiv_band_0.05_mw | afrac, prad | 1.083 / 1.145 | 167 / 149 | 18 / 13 | 131 / 111 | 574 | 1221 |
| with_absolute_fdiv_band_0.05_mw | afrac, prad | 0.411 / 0.434 | 68 / 478 | 6 / 20 | 122 / 36 | 418 | 1075 |
| with_relative_fdiv_band_0.1_mw | afrac, prad | 1.052 / 1.176 | 132 / 143 | 15 / 13 | 219 / 179 | 447 | 1106 |
| with_absolute_fdiv_band_0.1_mw | afrac, prad | 0.399 / 0.446 | 56 / 474 | 5 / 20 | 178 / 52 | 355 | 1019 |
| with_relative_fdiv_band_0.15_mw | afrac, prad | 1.021 / 1.206 | 106 / 131 | 12 / 9 | 294 / 314 | 270 | 934 |
| with_absolute_fdiv_band_0.15_mw | afrac, prad | 0.387 / 0.458 | 43 / 472 | 6 / 20 | 237 / 73 | 288 | 954 |
| with_relative_fdiv_band_0.2_mw | afrac, prad | 0.990 / 1.237 | 64 / 112 | 6 / 7 | 355 / 428 | 149 | 820 |
| with_absolute_fdiv_band_0.2_mw | afrac, prad | 0.376 / 0.469 | 38 / 458 | 4 / 18 | 296 / 94 | 222 | 893 |
| with_relative_fdiv_band_0.3_mw | afrac, prad | 0.929 / 1.299 | 62 / 96 | 7 / 5 | 378 / 458 | 92 | 785 |
| with_absolute_fdiv_band_0.3_mw | afrac, prad | 0.352 / 0.493 | 46 / 349 | 4 / 17 | 363 / 206 | 139 | 815 |

`primary` is the exported label: Afrac is the only second voter. `tangtv_alone` has no second voter, so every TangTV vote is TangTV only. The `with_*` rows add f_div as a second voter next to Afrac (`with_relative_fdiv` reproduces the label of the earlier build, where f_div was a vote: it makes many more certain bins and many more conflicts); `relative` takes cutoffs that scale the shot baseline, `absolute` the 201081-anchored global cutoffs (the midpoint of the measured attached and detached Prad,div,L over its P_in; one pair for every shot). `band_X` sets the half-width of the uncertain band to X MW (the primary keeps 0.1 MW). `afrac_abstains` leaves f_div as the only second voter. `published_anchor_values` uses the published 1.6 / 2.2 MW. A two-voter variant also has the tier `low_confidence_pair` (the two second voters agree and TangTV casts none), folded here into the uncertain bins.

Decision criterion: attached AND detached bins on at least 3 of the same shots, at least one of them a cohort shot (split train, val or test). Certain attached and certain detached bins coexist on 2 shots (0 in the cohort); counting the TangTV-only tier as well, 13 shots (0 in the cohort). The criterion is not met, so the label set is presented as an indicator-agreement appendix, not as a three-state label set.

### Bin width

| Width | Certain bins (att / det) | Certain shots (att / det) | TangTV-only bins (att / det) | Assessed bins | Uncertain share | Flicker per s | Agreement with 50 ms (bins) |
|---|---|---|---|---|---|---|---|
| 20ms | 152 / 430 | 7 / 13 | 859 / 770 | 3,428 | 0.36 | 0.498 | 0.993 (2114) |
| 50ms | 34 / 51 | 4 / 3 | 393 / 502 | 1,237 | 0.21 | 0.163 | - |
| 100ms | 18 / 9 | 3 / 1 | 212 / 279 | 538 | 0.04 | 0.212 | 0.992 (481) |

The same compatibility rule at each width on the 77 non-test shots of the width roster (27 of them assessed at 50 ms; the roster is not the exported shot list, so its 50 ms row differs from the coverage table above; the widths are comparable with each other). The certain tier depends on the width: 20 ms gives 152 attached / 430 detached certain bins (7 / 13 shots), 50 ms 34 attached / 51 detached certain bins (4 / 3 shots) and 100 ms 18 attached / 9 detached certain bins (3 / 1 shots); the certain share of the labelled bins is 0.26 at 20ms, 0.09 at 50ms, 0.05 at 100ms and the TangTV-only tier takes over. Where two widths both label a bin the states agree: 0.993 at 20ms (2114 bins), 0.992 at 100ms (481 bins).

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
- `fig_detachment_timeline`: label, Afrac and TangTV vote strips, the relative f_div
  (a corroborator, marked "not a vote") and DZ with their cutoffs, and the divertor
  Thomson Te; below, per-shot bars of certain, TangTV-only and uncertain bins with
  the shot IDs on the axis.
- `fig_detachment_marfe_witness`: the 199172 MARFE candidate (no certain MARFE) and
  the published 199166 onset, with the X-point marked, the frame time, the label
  bin and its tier, and the recall (0 of 1).
- `fig_detachment_figure2` (3.25 inch panel): coverage by class and tier,
  the AUROC of -Te per vote source and, as a corroborator, the within-shot and
  pooled AUROC of the relative f_div against TangTV and against Te, with the number
  of shots beside every row; the source is `docs/labeler/figure2_detach.json`, the
  one canonical record.

Regenerate in order: `detach_fetch_round4.py` (login node, `fdp run`, one worker,
stops at the first authentication error), `detach_bins.py`, `detach_label.py`,
`detach_prad_anchor.py`, `detach_reference.py`, `detach_benchmark.py`,
`detach_te_check.py`, `detach_afrac_check.py`, `detach_prad_sensitivity.py`,
`detach_fdiv_check.py`, `detach_current_state.py`, `detach_physics_audit.py`,
`detach_probe_bolom.py`, `detach_bins.py --width-ms 20/50/100` and
`detach_sensitivity.py`, `detach_ours.py` and `detach_victor.py` (the learned
baselines, on the labels just written), `detach_figure.py`,
`detach_paper_panel.py`, `detach_handoff.py`, `detach_docs_tables.py` and
`detach_validate_outputs.py`. Large outputs and
`HANDOFF.md` are under `$LABELER_ROOT/round4/detach/`.

## Limitations

The published points are few (ten, on three shots) and share their sources with the
indicator definitions, so they support neither accuracy nor calibration. The
certain tier measures agreement between two indicators, not physical accuracy: the
second vote does not improve the label's agreement with divertor Thomson Te over the
TangTV state alone, and the silver tier is as Te-consistent as the certain tier. The
Afrac proxy is a weak second vote: it is valid on few shots, its Te interval
includes chance, its result depends on the flux window, and the L-mode gate knows the
regime on a minority of shots; most labelled bins are therefore `tangtv_only`, not
because Afrac disagrees but because it has no valid measurement there. f_div
corroborates TangTV and Te within the few shots that have both classes but not
pooled over shots (AUROC below 0.5), and the absolute cutoffs do not carry over from
their anchor shot either, so it is reported per shot and does not vote; the shot
counts beside every f_div figure are small. Only one assessed shot is in the fixed
cohort and none is a test shot; certain attached and detached bins never coexist on
a cohort shot, so the set cannot support the cohort benchmark or a study of
attached-to-detached transitions. No bin is exported as MARFE: the one published
MARFE (199166 at 3705 ms) is a `candidate_marfe` bin, recalled 0 of 1 as a
state. The Thomson check covers attached and detached only, on the shots that have
a chord near the target; the within-shot Te figures rest on a handful of shots. The
width of the bins changes the certain tier strongly. Indicator agreement can share
systematic biases. Future work needs an independent expert state table, more
inversion coverage and a literature-backed density cue for MARFE.

## Appendix: learned baselines

<!-- BASELINES -->

Two learned baselines, retrained on the labels described here (labels sha256 `b7daaa3cb943`; the architecture and epochs are unchanged), predict the labelled states, certain and TangTV only together (1,008 bins), with 5-fold shot-grouped cross-validation over the non-test shots; test shots are never fitted or scored here. `detach-ours` reads windows of 0-D signals (current, density, D-alpha, ELM share, EFIT scalars) and `detach-victor` raw TangTV frames. Scores are agreement with constructed labels, not physical accuracy: `detach-victor` reads the same camera that sets the TangTV state, so its agreement partly reproduces the label from its own input. The fold-majority control predicts each held-out fold's bins as the majority class of the training fold; with shot-grouped folds that class is often the minority of the held-out shots, so the control scores below 0.5 and kappa 0 (a constant class) is the plainer reference. Intervals are shot-bootstrap (1000 replicates); bins are those with complete inputs or a camera frame, and the interval of `detach-ours` has a lower kappa end near 0.

| Model | Bins / shots | Accuracy [95% shot CI] | Kappa [95% shot CI] | Macro-F1 [95% shot CI] |
|---|---|---|---|---|
| detach-ours | 857 / 26 | 0.67 [0.53, 0.80] | 0.33 [0.06, 0.59] | 0.67 [0.52, 0.79] |
| detach-ours fold-majority control | 857 / 26 | 0.28 [0.15, 0.43] | -0.43 [-0.67, -0.12] | 0.28 [0.14, 0.42] |
| detach-victor | 445 / 13 | 0.77 [0.58, 0.92] | 0.52 [0.18, 0.83] | 0.76 [0.54, 0.91] |
| detach-victor fold-majority control | 445 / 13 | 0.38 [0.18, 0.58] | -0.28 [-0.57, 0.04] | 0.34 [0.18, 0.49] |

MARFE transfer is unsupported (no MARFE label exists). Records: `docs/labeler/results/detachment_ours.json` and `detachment_victor.json`; scripts `detach_ours.py` and `detach_victor.py`. They are not listed among the dataset's models: they are a comparison of what 0-D signals and TangTV frames can reproduce of the constructed labels.

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
