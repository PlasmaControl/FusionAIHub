# Detachment protocol

<!-- SUMMARY -->

**Label set: the geometry-gated TangTV state, checked against divertor Thomson Te. Exploratory; no gold-standard benchmark.** Each 50 ms bin on an eligible shot is attached, detached or uncertain; MARFE is never a state. Two indicators vote: the TangTV C-III front height DZ and the target-current ratio (Afrac proxy: a probe's Jsat over that probe's own attached level, the probe nearest the separatrix, known L-mode bins abstain). The lower-divertor radiated fraction f_div = Prad,div,L / P_in is measured on every bin but is not a vote: it corroborates within a shot and creates no conflicts (below). A bin is `certain` when TangTV votes and a valid Afrac vote agrees with it (called **TangTV + Afrac agreement** in text: two indicators agree, this is not a confidence level, and its Te agreement is not higher than that of the TangTV-only tier, below); `tangtv_only` (silver) when TangTV votes and Afrac abstains or is invalid; `conflict` when TangTV and Afrac vote differently. A TangTV detached vote on a bin of a known L-mode phase or of a probable-L phase is `tangtv_only_lmode`, uncertain: the DZ cutoffs come from an H-mode shot, so the detached vote is not trusted as a state in L-mode. Only the detached vote is gated, and that choice is a priori: L-mode inner-SOL leakage biases DZ upward, so a low DZ stays trustworthy. A phase of unknown regime is probable-L when its ELM coverage is known, its ELM share is zero over the window and its median P_in is below 2 MW (the L-H threshold power of Martin 2008's multi-machine scaling is about 1.7 to 2.4 MW at typical DIII-D parameters); the window is the bins of unknown regime that have both an ELM flag and a measured P_in, at least 1 s of them. A phase of unknown regime with an ELM share is probable-H, which is reported and not gated. The gate is fixed from the regime alone and was not tuned on Te; a bin of any other regime keeps the TangTV vote. Every other assessed bin is uncertain, with the reason in the `tier` column. The agreement tier measures indicator agreement, not physical accuracy.

**Coverage.** Labelled (a state, attached or detached): 821 bins on 23 shots (41 s): TangTV + Afrac agreement 138 bins on 7 shots (attached 62 on 5 shots, detached 76 on 4 shots) and TangTV only 683 bins on 23 shots (attached 392, detached 291); in 631 of the TangTV-only bins Afrac is invalid (no reference yet, known L-mode, probe off the separatrix) and in 52 it sits between its cutoffs. By regime, the labelled detached bins are H-mode 56, L-mode 0, probable-L 0, probable-H 303 and bins of unknown regime 8; every agreement-tier bin is on a shot that has no known regime (probable-H here), so the regime gate of Afrac never acts where the agreement tier lives. Of 1,780 assessed bins on 32 shots (89 s), 566 bins on 13 shots can never carry a state (TangTV is invalid there), and 9 assessed shots have no labelled bin. The other 959 assessed bins are uncertain: 29 `conflict`, 187 `tangtv_only_lmode` on 10 shots (93 on 6 shots of a known L-mode phase and 94 on 4 shots of a probable-L phase), and the rest by the tier column. Of the 32 assessed shots 1 is in the fixed cohort (val split, 57 assessed bins of which 0 agreement tier and 57 TangTV only) and none is a test shot.

**Divertor Thomson Te check.** AUROC of -Te for a detached against an attached vote (pooled, with shot-bootstrap 95% intervals; one bootstrap per statistic, so a population has one interval wherever it appears; an interval over fewer than three shots is printed `n/a`). **The check is the TangTV vote before the L-mode gate** (every valid upper-shelf TangTV vote, called `tangtv_vote_before_gate` in the records): AUROC 0.95 [0.91, 0.98] (668 bins, 23 shots), and 0.94 [0.90, 0.98] on 22 shots without the anchor shot 201081 that sets the DZ cutoffs. The exported tiers follow the L-mode gate, which was introduced after the Te check flagged the L-mode bins, so their scores are post-selection and are not an independent test of the gate: the labelled bins 0.95 [0.91, 0.99] (553 bins, 21 shots), TangTV + Afrac agreement 0.94 [0.81, 0.97] (110 bins, 6 shots), TangTV only 0.96 [0.90, 0.99] (443 bins, 20 shots). Within shots: the vote before the gate 0.97 [0.94, 0.99] over 6 shots, agreement tier 0.95 [n/a (2 shots)], TangTV only 0.92 [n/a (1 shot)]. Share of bins in the expected Te band (attached >= 10 eV, detached <= 5 eV; in brackets bins / shots), before the gate: attached 0.83 (393 / 18) and detached 0.74 (275 / 18); after it (post-selection), in the agreement tier attached 0.73 (52 / 5) and detached 0.79 (58 / 3) against attached 0.86 (334 / 16) and detached 0.87 (109 / 10) in the TangTV-only tier: the agreement tier is not more Te-consistent. **The second vote does not improve the Te agreement of the TangTV state**: the paired difference (agreement tier minus the vote before the gate) is not estimable on 6 shots (the agreement tier rests on 6 shots; 39 of 1000 bootstrap replicates are dropped because their agreement-tier resample has one class, so the interval [-0.13, +0.03] is not a result). What the second vote buys is agreement between two indicators, not a better match to Te. **By regime**, before the gate (the regime source is the label model's; the probable-L and probable-H rows are the proxies above): on known L-mode phases the TangTV vote ranks Te almost perfectly, AUROC 1.00 [0.99, 1.00] (200 bins, 6 shots), but its detached bins sit at 8.6 eV (median; 3 of 31 bins at or below 5 eV), in L-mode the DZ cutoff of 0.5 marks detaching, not detached, so these bins are `tangtv_only_lmode`, uncertain. On probable-L phases (3 shots with a Te) 52 of 69 detached bins (0.75) are at or below 5 eV: the gate withholds mostly cold bins there, because it is a priori and does not look at Te. It withholds 94 detached votes on 4 shots, 66 of them with DZ >= 0.8 (32 on 190110, 32 on 190113, 2 on 190116), cold where a Te exists (47 of the 47 bins that have a Te are at or below 5 eV), and 28 warm ones with DZ 0.5 to 0.65 (28 on 190109; 5 of its 22 bins with a Te are at or below 5 eV, median 6.8 eV). The H-mode bins are one shot with 20 detached bins and no attached ones, so its AUROC is not estimable; the probable-H phases (report only) give 0.93 [0.86, 0.97] (373 bins, 13 shots), with 0.83 of 152 detached bins at or below 5 eV; 6 bins on shots with neither a regime nor a proxy are too few to score. The gate is a priori (the cutoffs come from an H-mode shot; Chen 2026 notes that the outboard-of-X-point window excludes the inner SOL only in H-mode); it was not tuned on Te.

**Afrac proxy rebuilt.** The earlier proxy read the peak-current probe against one whole-shot reference and followed the probe's flux position (median 0.79 within 0.005 of the separatrix and 0.17 at psiN 1.02-1.03, for the same plasma). Each probe now has its own attached reference and the probe nearest the separatrix inside |psiN - 1| <= 0.01 is read; the results show the median Afrac against the selected-probe psiN before and after. Against TangTV its AUROC is 0.84 [0.60, 0.95] (219 bins, 7 shots); against Te 0.76 by vote [0.54, 0.86] (201 extracted bins with a Te, 8 shots; the Te check below scores the assessed bins instead, a different population) and 0.76 by value. It stays the second vote (rule: removed below 0.65), but only as a weak one: it is valid on 898 of 50,032 extracted bins (14 of 470 shots), the Te interval includes 0.5, the regime is known on 10 of 37 shots (so the L-mode gate removes 774 bins and misses the rest) and a wider window (0.02) puts the value-based AUROC at 0.64, below the removal bound.

**f_div is a within-shot corroborator, not a vote.** Pooled over shots its level moves with the shot, not with the state: per-shot medians of the absolute f_div in TangTV-attached bins run from 0.39 to 0.58 (19 shots) and in TangTV-detached bins from 0.40 to 0.57 (21 shots), with P_in from 1.1 to 10.9 MW, so no cutoff pair carries over between shots. Counted as a second vote (shot-relative cutoffs 1.052 / 1.176) it would give 240 certain bins but 374 conflicts, against 138 and 29; it is a sensitivity row. As a corroborator, per shot with at least 5 bins of each class: the relative f_div ranks TangTV-detached above TangTV-attached bins with a within-shot AUROC of 0.80 [0.58, 0.98] over 8 shots (6 above 0.5), and cold above warm divertor Thomson Te with 0.69 [0.53, 0.84] over 7 shots (4 above 0.5); pooled over shots the same scores give 0.41 [0.26, 0.59] on 990 bins, 26 shots against TangTV (TangTV detached votes against attached votes) and 0.39 [0.22, 0.59] on 691 bins, 25 shots against Te, i.e. no support once the shots are mixed. The absolute f_div (201081-anchored cutoffs 0.399 / 0.446, P_in 4.274 MW) ranks the same bins within a shot and better pooled (against TangTV, same population: 0.78 [0.62, 0.92] on 990 bins, 26 shots), but its cutoffs vote detached on most TangTV-attached bins where it votes. Counting only the relative votes it casts, their AUROC against Te is 0.42 [0.23, 0.58] pooled. The published detached point 180257 at 4800 ms is not assessed (TangTV has no valid geometry, the relative f_div has no baseline, Afrac reads `probe_off_separatrix`); the absolute cutoffs would call it attached (f_div 0.17 at P_in 7.2 MW), a miss of that published point.

**Reference shot 201081.** Measured P_in is 4.274 MW (the median over the 32 attached-window bins of the 250 ms-averaged P_in); Prad,div,L is 1.621 MW attached and 1.990 MW detached. Of 7 published points on 201081, 6 are labelled in agreement with the publication, 1 uncertain and 0 contradicted.

**MARFE.** No bin is exported as MARFE: the density cue (fG >= 0.8) has no literature source (Dong 2025 gives fG of about 0.5 or more on HL-3 from a core-point density, with a core-Te condition and a gate of 0.40, and says thresholds are device-specific). A persistent high TangTV front is the uncertain tier `candidate_marfe`: 70 bins on 4 shots; the full TangTV MARFE vote (front, emission peak inside the separatrix and the density cue) fires on 23 bins of 1 shot only. **Recall of the published MARFE onset on 199166 (3705 ms): 0/1 as a label**; the bin holding it is `candidate_marfe` (1/1 as a candidate), with fG = 0.72 there against the 0.8 cue.

**Three-state label set.** Not offered: agreement-tier attached and detached bins coexist on 2 shots, 0 of them in the fixed cohort, against a requirement of at least three shots including one cohort shot. The result is an indicator-agreement appendix.

<!-- /SUMMARY -->

The fixed 500-shot cohort's test split is excluded from fitting, threshold
selection, surrogate selection and bin-policy analysis.

## Definitions and export

The label set is the **geometry-gated TangTV state, checked against divertor
Thomson Te**. The exported lower-divertor states are attached (1), detached (2) and
uncertain (4); the interval schema keeps code 3 (MARFE) but no bin carries it.
Missing intervals are unassessed, never attached. These are operational
definitions: attached is a low accepted C-III front (DZ < 0.35 on the upper shelf),
detached a lifted one (0.5 <= DZ < 1.2). The tier says what supports the state:
`certain` when a valid Afrac vote agrees with TangTV (the paper text calls this
tier **TangTV + Afrac agreement**: two indicators agree, it is not a confidence
level), `tangtv_only` (silver) when TangTV votes and Afrac abstains or is invalid,
`conflict` (uncertain) when Afrac votes against TangTV, and `tangtv_only_lmode`
(uncertain) when TangTV votes detached on a bin of a known L-mode phase or of a
probable-L phase: the DZ cutoffs come from one H-mode shot (201081), Chen 2026
notes that the outboard-of-X-point window excludes the inner SOL only in H-mode,
and the TangTV detached bins on known L-mode phases are not cold by the Te
criterion (see the L-mode section of the results). Only the detached vote is
gated, and that is a priori: in L-mode the inner SOL leaks into the window and
biases DZ upward, so a low DZ stays trustworthy. The gate uses the regime alone (a
priori; it was not tuned on Te, which would be circular) and a bin of any other
regime keeps the TangTV vote.

The regime of every bin is exported in the `regime` column and its source in
`regime_source`. `H` and `L` come from the confinement suggestion table (curated
Gill and Butt intervals where they exist, else the rows of the dalpha_lh
detector; source `regime_table`) or from the D-alpha H-mode detector (source
`dalpha_detector`); on the assessed shots all of this information comes from the
D-alpha detectors (see the coverage results). A bin of unknown regime gets a
proxy from its ELM flag and input power over a window of such bins that have
both an ELM flag and a measured P_in, at least 1 s of them: `probable_L` when the
ELM share is zero over the window and the median P_in is under 2 MW (the L-H
threshold power of Martin 2008's multi-machine scaling is about 1.7 to 2.4 MW at
typical DIII-D parameters), `probable_H` when any bin carries an ELM share
(report only, no gate acts on it). Both proxy labels are written to `regime` and
`regime_source`. f_div is not a vote: it is reported per shot as a
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

Bins are 50 ms, following the catalog grid. The camera is a 60 Hz interlaced
camera captured as 30 Hz full frames (Chen 2026), but the parked inversions are
spaced 16.7 ms apart (about 3 per bin, measured on all 32 assessed shots) and the
raw VID frames in the SAV files are spaced the same; the corpus's raw TangTV movie
is resampled to 20 ms (2.5 frames per bin). Chen 2026 integrates ELMs over these
exposures and so does this export, without resampling extra frames. The width
section of the results shows that the labelled states do not depend on the bin width.

Files:

- `data/events/detachment/extend_detach_vote/detach_shots.csv` holds the standard
  shot/category/start/end/confidence/attrs intervals; `attrs` carries `tier`
  (`certain` or `tangtv_only` on attached and detached intervals, the reason on
  uncertain ones).
- `$LABELER_ROOT/round4/detach/labels_bins.csv.gz` holds every bin with values,
  votes, validity reasons, geometry and Afrac probe provenance (exported for invalid
  bins too). `state_rule` is the primary label (TangTV with Afrac as the second
  vote); `regime` and `regime_source` are the regime of each bin and where it
  comes from; `state_lm` is its legacy alias; `state_rule_relative_prad` /
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
1 s of such bins, 20 at 50 ms, or the probe has no reference and does not vote; ELM, ramp,
low-power and known-L-mode bins are left out). A bin's Afrac is that ratio on the
probe nearest the separatrix in flux among those with a reference, inside the same
window. Known L-mode bins abstain (reason `l_mode`; the regime is the confinement
suggestion table (curated Gill and Butt intervals where they exist, else the
rows of the dalpha_lh detector), then the D-alpha H-mode detector; a bin of
unknown regime is not gated, and the probable-L proxy gates the TangTV vote, not
Afrac). Strike-point positions that are sentinel values are ignored. The
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
f_div over its flat-top-power bins; a shot with less than 2 s of such bins, 40 at
50 ms, has no baseline and no value, reason `no_baseline`); the relative votes, attached below
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
1.2 for at least 100 ms (two adjacent bins at 50 ms), with the global emission peak inside the separatrix near
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
shelf with known ELM coverage. A TangTV detached vote on a known L-mode phase or a
probable-L phase makes the bin uncertain, tier `tangtv_only_lmode`. Otherwise it is `certain` (TangTV +
Afrac agreement) when a valid Afrac vote agrees with it, and `tangtv_only` (silver)
when Afrac casts no vote (it abstains between its cutoffs or is invalid). An Afrac
vote against TangTV's makes the bin uncertain, tier `conflict`. Only TangTV and
Afrac vote; f_div never supports or blocks a
state. A persistent high TangTV front is `candidate_marfe`, an uncertain tier.
Primary states are unsmoothed.

| Tier | State | Meaning |
|---|---|---|
| `certain` | 1/2 | TangTV + Afrac agreement: TangTV vote plus an agreeing valid Afrac vote, no Afrac conflict, upper shelf, known ELM coverage (not a confidence level) |
| `tangtv_only` | 1/2 | TangTV vote without a second vote (silver): Afrac abstains or is invalid |
| `tangtv_only_lmode` | 4 | TangTV votes detached (0.5 <= DZ < 1.2) on a known L-mode or probable-L phase; the state is withheld by the a-priori regime gate (`regime` column: `L` or `probable_L`) |
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
| Labelled (TangTV + Afrac agreement or TangTV only) | 821 / 23 | 41.0 |
| TangTV + Afrac agreement, attached | 62 / 5 | 3.10 |
| TangTV + Afrac agreement, detached | 76 / 4 | 3.80 |
| TangTV only, attached | 392 / 17 | 19.60 |
| TangTV only, detached | 291 / 12 | 14.55 |
| Assessed | 1,780 / 32 | 89.0 |
| Assessed, can never carry a state (no valid TangTV vote) | 566 / 13 | 28.30 |
| Tier `candidate_marfe` | 70 / 4 | 3.50 |
| Tier `certain` | 138 / 7 | 6.90 |
| Tier `conflict` | 29 / 5 | 1.45 |
| Tier `insufficient_support` | 226 / 13 | 11.30 |
| Tier `lower_shelf_window` | 325 / 6 | 16.25 |
| Tier `no_vote` | 122 / 24 | 6.10 |
| Tier `tangtv_only` | 683 / 23 | 34.15 |
| Tier `tangtv_only_lmode` | 187 / 10 | 9.35 |

A bin is assessed when at least two indicators are valid on it or TangTV votes; a shot is eligible when at least two indicators are each valid for at least 1 s (20 bins at 50 ms) and it has at least 1 s of assessed bins. The assessed count includes bins that have no TangTV vote (the row `can never carry a state`), so the labelled count is the one to quote. Tiers are disjoint and sum to the assessed bins; `lower_shelf_window` holds the owner's restricted lower-shelf extraction, whose TangTV vote is invalid (the shelf geometry does not hold) so it never yields a state.

### Labelled bins by regime

| Regime | Labelled attached (bins / shots) | Labelled detached (bins / shots) | TangTV detached votes before the gate | Votes withheld by the gate | Agreement-tier bins |
|---|---|---|---|---|---|
| H-mode | 0 / 0 | 56 / 1 | 56 / 1 | - | 0 / 0 |
| L-mode | 184 / 7 | 0 / 0 | 93 / 6 | 93 / 6 | 0 / 0 |
| probable L-mode (gated) | 0 / 0 | 0 / 0 | 94 / 4 | 94 / 4 | 0 / 0 |
| probable H-mode (report only) | 264 / 12 | 303 / 10 | 313 / 11 | - | 138 / 7 |
| no regime, no proxy | 6 / 1 | 8 / 2 | 8 / 2 | - | 0 / 0 |

The regime comes from the confinement suggestion table where it covers the shot (curated Gill and Butt intervals where they exist, else the dalpha_lh D-alpha detector's rows), else from the D-alpha H-mode detector, else it is unknown. On the 32 assessed shots (9 with a known regime) all regime information comes from the D-alpha detectors: no assessed shot has a curated interval, the table's rows for 189061 are the dalpha_lh detector's own (56 bins) and the other known-L shots come from the H-mode detector (350 bins). A shot or phase of unknown regime then gets a proxy label from its ELM flag and input power (probable-L: no ELM share and median P_in under 2 MW, gated like known L-mode; probable-H: any ELM share, report only; both written to the `regime` and `regime_source` columns): 140 assessed bins are probable-L and 1,220 probable-H, 14 stay without a regime because their ELM flag or P_in is missing. The gate moves 187 TangTV detached votes to `tangtv_only_lmode` (93 on known L-mode phases, 94 on probable-L phases); no labelled detached bin is on a known or probable-L phase. The probable-L window is the unknown-regime bins that have both an ELM flag and a measured P_in, so a bin that lacks either keeps the plain rule: 8 labelled detached bins on 2 shots (190110 and 190113, which otherwise type as probable-L) lie outside the window.

### Indicator coverage, all extracted shots

| Indicator | Measurement | Valid | Vote | In an agreement-tier label |
|---|---|---|---|---|
| afrac | 898 / 14 | 898 / 14 | 716 / 14 | 138 / 7 |
| prad (f_div, not a vote of the label) | 43,081 / 455 | 23,421 / 241 | 17,955 / 241 | 77 / 6 |
| tangtv | 1,813 / 41 | 1,291 / 28 | 1,098 / 28 | 138 / 7 |

Cells are bins / shots over 50,032 extracted bins on 470 shots. For f_div the Vote column counts its relative votes (a sensitivity: the label does not use them) and the last column the agreement-tier bins in which it casts one.

### What supports the labelled bins

| Labelled bins | Count |
|---|---|
| TangTV + Afrac agreement bins (TangTV vote and an agreeing Afrac vote) | 138 |
| TangTV-only bins | 683 |
|   Afrac valid, between its cutoffs (abstains) | 52 |
|   Afrac invalid | 631 |
|     reason `short_reference` | 271 |
|     reason `l_mode` | 184 |
|     reason `probe_off_separatrix` | 149 |
|     reason `elm` | 22 |
|     reason `no_power` | 4 |
|     reason `probe_flux_unknown` | 1 |

A TangTV-only bin keeps TangTV's state because Afrac does not vote there; it is not evidence against it. Most of these bins have no Afrac because the probe's reference is too short, the shot is in L-mode or no probe sits near the separatrix.

### Divertor Thomson Te check

The processed divertor Thomson Te (`\ELECTRONS::TSTE_DIV`, 14 or 16 chords at R = 1.485 m, about 20 ms) is a temperature that none of the three indicators (TangTV, Afrac, f_div) uses; the labels are checked against it, not validated by it. A bin's Te is the median of the valid samples of the chords 1.5 to 5 cm above the outer shelf (Z = -1.25 m) that lie on the scrape-off side, 1.000 < psiN <= 1.05, by the multi-slice EFIT map; samples outside 0.1-50 eV are rejected. No threshold is fitted: the bands were fixed from the literature before scoring. **The check is the TangTV vote before the L-mode gate** (every valid upper-shelf TangTV vote). Te has informed rule decisions (the Afrac keep rule, the removal of f_div as a voter, the L-mode gate), so the rows after the gate are post-selection and the whole table is a consistency check, not an independent validation.

| Group | Bins / shots | Median Te, eV (10-90%) | Share in band |
|---|---|---|---|
| agreement tier attached | 52 / 5 | 12.9 (6.9-22.0) | 0.73 |
| agreement tier detached | 58 / 3 | 1.9 (1.2-8.5) | 0.79 |
| TangTV-only attached | 334 / 16 | 16.4 (7.9-22.3) | 0.86 |
| TangTV-only detached | 109 / 10 | 2.1 (1.4-5.8) | 0.87 |
| labelled attached | 386 / 18 | 16.0 (7.8-22.1) | 0.84 |
| labelled detached | 167 / 11 | 2.0 (1.3-6.6) | 0.84 |
| TangTV vote before the gate, attached | 393 / 18 | 16.0 (6.8-22.1) | 0.83 |
| TangTV vote before the gate, detached | 275 / 18 | 2.3 (1.4-9.4) | 0.74 |
| TangTV vote before the gate without 201081, attached | 367 / 17 | 16.0 (6.7-22.2) | 0.83 |
| TangTV vote before the gate without 201081, detached | 235 / 17 | 2.3 (1.4-9.5) | 0.74 |
| vote before the gate, known H-mode, attached | 0 / 0 | - | - |
| vote before the gate, known H-mode, detached | 20 / 1 | 1.8 (1.3-2.5) | 1.00 |
| vote before the gate, known L-mode, attached | 169 / 6 | 17.7 (13.6-22.5) | 1.00 |
| vote before the gate, known L-mode, detached | 31 / 4 | 8.6 (5.0-11.9) | 0.10 |
| vote before the gate, probable L-mode, attached | 0 / 0 | - | - |
| vote before the gate, probable L-mode, detached | 69 / 3 | 2.2 (1.5-7.9) | 0.75 |
| vote before the gate, probable H-mode (report only), attached | 221 / 12 | 13.6 (5.0-21.5) | 0.71 |
| vote before the gate, probable H-mode (report only), detached | 152 / 10 | 2.1 (1.4-7.5) | 0.83 |
| vote before the gate, no regime, no proxy, attached | 3 / 1 | 12.5 (10.5-23.0) | 0.67 |
| vote before the gate, no regime, no proxy, detached | 3 / 1 | 1.9 (1.6-1.9) | 1.00 |
| TangTV detached vote in the gated bins (known L and probable L), detached | 100 / 7 | 3.6 (1.6-10.2) | 0.55 |
| Afrac vote attached | 92 / 7 | 12.5 (2.2-21.3) | 0.67 |
| Afrac vote detached | 105 / 7 | 2.6 (1.4-16.8) | 0.63 |
| relative f_div vote (unused) attached | 265 / 21 | 4.9 (1.5-19.0) | 0.31 |
| relative f_div vote (unused) detached | 241 / 16 | 10.1 (1.5-18.3) | 0.32 |

Share in band is the fraction of bins with Te >= 10 eV for attached or <= 5 eV for detached; the rows of the tiers (agreement tier, TangTV only, labelled) are after the L-mode gate and so post-selection. AUROC of -Te for detached against attached votes (pooled with a shot-bootstrap interval; within shots is the mean over the shots that have at least 5 bins of each class, with the number of such shots; an interval over fewer than 3 shots is `n/a`, and a population with a single class is `not estimable`):

| Reference | Pooled AUROC [CI] | Bins / shots | Within-shot mean [CI] | Shots (within) |
|---|---|---|---|---|
| labelled bins (agreement tier and TangTV only) | 0.95 [0.91, 0.99] | 553 / 21 | 0.94 [0.92, 0.96] | 3 |
| TangTV + Afrac agreement tier | 0.94 [0.81, 0.97] | 110 / 6 | 0.95 [n/a (2 shots)] | 2 |
| TangTV-only tier | 0.96 [0.90, 0.99] | 443 / 20 | 0.92 [n/a (1 shot)] | 1 |
| TangTV vote before the gate (the check) | 0.95 [0.91, 0.98] | 668 / 23 | 0.97 [0.94, 0.99] | 6 |
| TangTV vote before the gate, 201081 left out | 0.94 [0.90, 0.98] | 602 / 22 | 0.97 [0.94, 1.00] | 5 |
| vote before the gate, known H-mode | not estimable | 20 / 1 | undefined | 0 |
| vote before the gate, known L-mode | 1.00 [0.99, 1.00] | 200 / 6 | 0.99 [0.99, 1.00] | 3 |
| vote before the gate, probable L-mode | not estimable | 69 / 3 | undefined | 0 |
| vote before the gate, probable H-mode (report only) | 0.93 [0.86, 0.97] | 373 / 13 | 0.94 [0.92, 0.96] | 3 |
| vote before the gate, no regime, no proxy | 1.00 [n/a (2 shots)] | 6 / 2 | undefined | 0 |
| TangTV vote in `conflict` bins | 0.96 [0.90, 1.00] | 15 / 5 | undefined | 0 |
| Afrac vote (assessed bins) | 0.77 [0.54, 0.86] | 197 / 8 | 0.66 [0.36, 0.89] | 3 |
| relative f_div votes (unused) | 0.42 [0.23, 0.58] | 506 / 25 | 0.81 [0.66, 0.92] | 6 |

**Does the second vote improve the Te agreement?** It does not, and the paired difference cannot be estimated: agreement tier minus the TangTV vote before the gate is not estimable on 6 shots (the agreement tier has 6 shots with a Te; 39 of 1000 bootstrap replicates are dropped because the resample of those shots has a single class, which leaves the interval [-0.13, +0.03] degenerate; point difference -0.005). Against the TangTV votes that are TangTV only or in conflict the same difference is -0.018 [-0.15, +0.02] (not estimable on 6 shots). The within-shot figure for the agreement tier rests on 2 shots, too few for any statement. In the share-in-band column the agreement tier is less Te-consistent than the TangTV-only tier (attached 0.73 (52 / 5) against 0.86 (334 / 16)), so the tier is agreement of two indicators and not a confidence level.

### L-mode and probable L-mode: where the TangTV detached vote is gated

The DZ cutoffs (0.35 and 0.5) come from one H-mode shot (201081). On known L-mode phases the TangTV detached vote is not cold by the Te criterion: 3 of its 31 bins with a Te (0.10) are at or below 5 eV (median 8.6 eV), against 0.83 on probable-H phases, although the ranking inside L-mode is near perfect (1.00 [0.99, 1.00] (200 bins, 6 shots)): the state boundary sits at about 10 eV there, not 5 eV. Chen 2026 notes that the outboard-of-X-point window excludes the inner SOL only in H-mode, so a cutoff set on an H-mode shot need not carry over to L-mode. Only the detached vote is gated, and that is a priori: in L-mode the inner SOL leaks into the window and biases DZ upward, so a low DZ stays trustworthy. The label withholds the state (tier `tangtv_only_lmode`, uncertain) for a TangTV detached vote on a known L-mode phase, from the regime alone; **the gate was not tuned on Te**, which would make this check circular. The regime is known on 8 assessed shots in L-mode and 1 in H-mode, so the gate also acts on a proxy: a phase of unknown regime with known ELM coverage, no ELM share and a median P_in below 2 MW (probable-L; the L-H threshold power of Martin 2008's scaling is about 1.7 to 2.4 MW at typical DIII-D parameters) is gated the same way, also from the regime alone. The proxy window is the bins of unknown regime that have both an ELM flag and a measured P_in. A phase with an ELM share is probable-H, reported here and not gated.

**What the probable-L gate withholds.** 94 TangTV detached votes on 4 shots: 66 with DZ >= 0.8 (32 on 190110, 32 on 190113, 2 on 190116), where 47 of the 47 bins that have a Te are at or below 5 eV, so the gate withholds cold bins too, because the rule does not look at Te (of all 94 gated votes 52 of the 69 that have a Te are at or below 5 eV), and 28 with DZ 0.5 to 0.65 (28 on 190109), the detaching-not-detached case the gate is for: 5 of its 22 bins with a Te are at or below 5 eV, median 6.8 eV. In the labels these bins are uncertain (`tangtv_only_lmode`). The gate by shot:

| Regime | Shot | Gated bins | DZ >= 0.8 | DZ < 0.8 | With a Te | Te <= 5 eV |
|---|---|---|---|---|---|---|
| known L-mode | 199166 | 6 | 2 | 4 | 6 | 2 |
| known L-mode | 199172 | 9 | 4 | 5 | 0 | 0 |
| known L-mode | 199352 | 2 | 0 | 2 | 2 | 0 |
| known L-mode | 199353 | 10 | 0 | 10 | 10 | 0 |
| known L-mode | 199354 | 13 | 0 | 13 | 13 | 1 |
| known L-mode | 200977 | 53 | 16 | 37 | 0 | 0 |
| probable L-mode | 190109 | 28 | 0 | 28 | 22 | 5 |
| probable L-mode | 190110 | 32 | 32 | 0 | 26 | 26 |
| probable L-mode | 190113 | 32 | 32 | 0 | 21 | 21 |
| probable L-mode | 190116 | 2 | 2 | 0 | 0 | 0 |

Te of the TangTV detached votes by DZ band and regime (before the gate, shot counts beside each cell):

| DZ band | known H-mode | known L-mode | probable L-mode | probable H-mode (report only) | no regime, no proxy |
|---|---|---|---|---|---|
| 0.5-0.65 | 3 bins / 1 shot, median 2.1 eV, in band 1.00 | 21 bins / 4 shots, median 9.3 eV, in band 0.05 | 22 bins / 1 shot, median 6.8 eV, in band 0.23 | 46 bins / 9 shots, median 3.2 eV, in band 0.72 | - |
| 0.65-0.8 | 10 bins / 1 shot, median 1.9 eV, in band 1.00 | 8 bins / 3 shots, median 7.4 eV, in band 0.00 | - | 70 bins / 7 shots, median 1.9 eV, in band 0.86 | - |
| 0.8-1.2 | 7 bins / 1 shot, median 1.4 eV, in band 1.00 | 2 bins / 1 shot, median 2.7 eV, in band 1.00 | 47 bins / 2 shots, median 1.9 eV, in band 1.00 | 36 bins / 5 shots, median 1.8 eV, in band 0.92 | 3 bins / 1 shot, median 1.9 eV, in band 1.00 |

Exported detached and attached bins by regime after the gate: known H-mode 0 attached / 20 detached; known L-mode 169 attached / 0 detached; probable L-mode 0 attached / 0 detached; probable H-mode (report only) 214 attached / 144 detached; no regime, no proxy 3 attached / 3 detached. The probable-H phases, where the gate does not act, give 0.93 [0.86, 0.97] (373 bins, 13 shots) before the gate (the ELMy shots of unknown regime are H-mode by the ELM signature; report only).

25 of the 32 assessed shots have a Thomson Te in an assessed bin; for the others no selected chord had a valid sample, or no Thomson data exist.

On 201081 the Te first falls below the detached band at 2575 ms and last does at 4425 ms (published cliffs 2650 and 4450 ms). No MARFE bin is exported, so the check does not cover MARFE.

### f_div and the indicator pairs, upper shelf

| Statistic | Value [95% shot CI] | Chance | Bins | Shots |
|---|---|---|---|---|
| AUROC of relative f_div, pooled (detached or MARFE vote) | 0.41 [0.26, 0.58] | 0.5 | 1013 | 26 |
| AUROC of relative f_div, mean over shots | 0.80 [0.58, 0.98] | 0.5 | - | 8 |
| Spearman rho of relative f_div with DZ, pooled | -0.09 [-0.31, 0.18] | 0.0 | 1167 | 26 |
| Spearman rho, mean over shots | 0.36 [0.20, 0.52] | 0.0 | - | 26 |
| Cohen's kappa, attached against not attached | -0.16 [-0.44, 0.14] | 0.0 | 554 | 25 |
| Cohen's kappa, three states | -0.16 [-0.44, 0.14] | 0.0 | 554 | 25 |

Agreement diagnostics; none of them decides a label. AUROC is the probability that a TangTV-detached (or TangTV-MARFE-vote) bin has a larger relative f_div than a TangTV-attached bin; the per-shot f_div corroborator table below counts the detached votes alone. Intervals resample shots (1000 replicates, seed 0). Kappa is null, and not drawn, where either rater used one class. The within-shot rows average over the few shots that have enough bins of both classes (the shot count is in the last column of the table). The pooled f_div figures against TangTV come in two populations: the corroborator population (TangTV detached votes against attached votes; 990 bins on 26 shots; relative 0.41 [0.26, 0.59], absolute 0.78 [0.62, 0.92]) and the agreement-diagnostics population, which also counts the TangTV MARFE-vote bins as detached (1,013 bins on 26 shots; relative 0.41 [0.26, 0.58], absolute 0.79 [0.63, 0.92]). The same statistics for each pair of indicators:

| Shelf | Pair | AUROC [CI] | Bins | Shots | Within-shot AUROC [CI] | Shots (within) |
|---|---|---|---|---|---|---|
| upper shelf | relative f_div against TangTV (detached or MARFE vote) | 0.41 [0.26, 0.58] | 1013 | 26 | 0.80 [0.58, 0.98] | 8 |
| upper shelf | absolute f_div against TangTV (sensitivity; detached or MARFE vote) | 0.79 [0.63, 0.92] | 1013 | 26 | 0.80 [0.58, 0.98] | 8 |
| upper shelf | Afrac against TangTV | 0.84 [0.60, 0.95] | 219 | 7 | 0.98 [n/a (2 shots)] | 2 |
| upper shelf | relative f_div against Afrac | 0.71 [0.37, 0.88] | 183 | 7 | 0.87 [0.68, 1.00] | 3 |
| lower shelf window | relative f_div against TangTV (detached or MARFE vote) | undefined | 0 | 0 | undefined | 0 |
| lower shelf window | absolute f_div against TangTV (sensitivity; detached or MARFE vote) | undefined | 0 | 0 | undefined | 0 |
| lower shelf window | Afrac against TangTV | undefined | 0 | 0 | undefined | 0 |
| lower shelf window | relative f_div against Afrac | 0.97 [0.83, 1.00] | 258 | 6 | 0.99 [0.96, 1.00] | 3 |

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
| (0.98, 0.995] | 45 / 10 | 0.30 | 0.04 (5) | 0.08 (8) |
| (0.995, 1] | 282 / 13 | 0.72 | 0.94 (30) | 0.24 (50) |
| (1, 1.005] | 434 / 14 | 0.79 | 0.85 (51) | 0.24 (27) |
| (1.005, 1.01] | 137 / 14 | 0.68 | 0.67 (37) | 0.78 (11) |

Spearman rank correlation of Afrac with the selected probe's psiN, TangTV-attached bins: before -0.49 (450 bins, 19 shots), after -0.13 (123 bins, 6 shots); TangTV-detached bins: before -0.61 (597 bins, 23 shots), after 0.50 (96 bins, 5 shots). The dependence in the attached class is much reduced; the detached class keeps a positive rank correlation on few bins and shots (its median is 0.22 at psiN <= 1.005 on 85 bins and the 11 detached bins at 1.005-1.01 read 0.78), so the proxy is not free of position. On 201081 the median Afrac is 1.05 before the first cliff, 0.21 between the cliffs and 0.89 after the second, all from one probe.

The cause of the earlier failure: The earlier proxy read the peak-current probe among those with psiN in (1.000, 1.05] and at least 5 mm outboard of the outer strike point, against one whole-shot reference. The strike-point probe, which shows the rollover, was excluded, and which probe was read followed millimetre-scale strike-point motion; probe spacing (1.1 to 4.4 cm) is wider than the current peak, so a fixed-psiN interpolation was not an alternative. The proxy then varied with the read probe's flux position, not with the plasma state.

Window sensitivity (the primary window is 0.01; wider windows admit probes the EFIT strike-point error can mis-map):

| Window | Valid bins / shots | AUROC(-Afrac) vs TangTV [CI] | Shots | AUROC(-Te), vote [CI] | AUROC(-Afrac) cold vs warm Te [CI] | Shots |
|---|---|---|---|---|---|---|
| 0.01 | 898 / 14 | 0.84 [0.60, 0.95] | 7 | 0.76 [0.54, 0.86] | 0.76 [0.51, 0.88] | 8 |
| 0.015 | 1,344 / 23 | 0.74 [0.59, 0.86] | 14 | 0.70 [0.48, 0.83] | 0.67 [0.43, 0.81] | 12 |
| 0.02 | 1,744 / 28 | 0.68 [0.57, 0.80] | 20 | 0.66 [0.52, 0.78] | 0.64 [0.48, 0.75] | 21 |

Decision rule: Afrac stays in the vote unless its AUROC against Te is below 0.65 (the lower of the vote-based and the value-based point estimates at the primary window). Statistic 0.758; Afrac stays in the vote. Known-L-mode bins (774 bins) abstain; the regime is known for 1,005 of 4,124 extracted bins on 10 of 37 shots (confinement suggestion table, then the D-alpha H-mode detector) and unknown regime is not gated. The window table scores the Afrac vote against Te on 201 extracted upper-shelf bins with a Te (8 shots, AUROC 0.758); the Te check above scores the assessed bins instead (197 bins, 8 shots, AUROC 0.771); these are two populations, not two estimates of one number, and both intervals include 0.5.

### f_div as a within-shot corroborator

f_div is not a vote, so it creates no conflicts and no agreement-tier bins. It is scored as a ranking: per shot, the AUROC of f_div for the TangTV-detached bins against the TangTV-attached ones (upper shelf), and for the divertor-Thomson-cold bins (Te <= 5 eV) against the warm ones (>= 10 eV); a shot counts when each class has at least 5 bins; the mean is over those shots, with a shot-bootstrap interval, and the pooled AUROC mixes the shots. The relative f_div is the ratio over the shot's own baseline, valid where the baseline exists; the absolute ranks the bins of one shot the same way (it differs by a per-shot constant) and is the 201081-anchored sensitivity. The pooled column here is the corroborator population (TangTV detached votes against attached votes), not the agreement-diagnostics population above, which also counts the MARFE-vote bins. Record: `docs/labeler/results/detachment_fdiv_check.json`.

| Score against reference | Within-shot mean [CI] | Shots | Shots above 0.5 | Pooled AUROC [CI] | Pooled bins / shots |
|---|---|---|---|---|---|
| relative f_div against TangTV | 0.80 [0.58, 0.98] | 8 | 6 | 0.41 [0.26, 0.59] | 990 / 26 |
| relative f_div against Te | 0.69 [0.53, 0.84] | 7 | 4 | 0.39 [0.22, 0.59] | 691 / 25 |
| absolute f_div against TangTV (sensitivity) | 0.80 [0.58, 0.98] | 8 | 6 | 0.78 [0.62, 0.92] | 990 / 26 |
| absolute f_div against Te (sensitivity) | 0.69 [0.53, 0.84] | 7 | 4 | 0.78 [0.70, 0.86] | 691 / 25 |

Per shot (relative f_div; cells are AUROC with the detached or cold / attached or warm bin counts in brackets; shots with fewer than 5 bins in a class are left out of a column):

| Shot | Split | Against TangTV | Against Te |
|---|---|---|---|
| 189057 | outside | 0.93 (38 / 10) | 0.79 (25 / 11) |
| 189081 | outside | - | 0.47 (7 / 13) |
| 189090 | outside | - | 0.41 (24 / 6) |
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

| Variant | Second voters | f_div cutoffs | Agreement-tier bins (att / det) | Agreement-tier shots (att / det) | TangTV-only bins (att / det) | Conflict bins | Uncertain bins |
|---|---|---|---|---|---|---|---|
| primary | afrac | - | 62 / 76 | 5 / 4 | 392 / 291 | 29 | 959 |
| no_second_voter | none | - | 0 / 0 | 0 / 0 | 473 / 377 | 0 | 930 |
| with_relative_fdiv | afrac, prad | 1.052 / 1.176 | 134 / 106 | 15 / 7 | 218 / 102 | 374 | 1220 |
| with_absolute_fdiv | afrac, prad | 0.399 / 0.446 | 56 / 287 | 5 / 10 | 178 / 51 | 356 | 1208 |
| with_relative_fdiv_afrac_abstains | prad | 1.052 / 1.176 | 92 / 90 | 14 / 6 | 277 / 126 | 265 | 1195 |
| with_absolute_fdiv_afrac_abstains | prad | 0.399 / 0.446 | 47 / 289 | 4 / 11 | 189 / 59 | 266 | 1196 |
| with_absolute_fdiv_published_anchor_values | afrac, prad | 0.421 / 0.468 | 113 / 279 | 14 / 10 | 218 / 35 | 278 | 1135 |
| with_relative_fdiv_band_0.025_mw | afrac, prad | 1.098 / 1.129 | 180 / 114 | 18 / 7 | 75 / 35 | 556 | 1376 |
| with_absolute_fdiv_band_0.025_mw | afrac, prad | 0.417 / 0.428 | 82 / 297 | 10 / 10 | 85 / 24 | 452 | 1292 |
| with_relative_fdiv_band_0.05_mw | afrac, prad | 1.083 / 1.145 | 168 / 113 | 18 / 7 | 131 / 52 | 482 | 1316 |
| with_absolute_fdiv_band_0.05_mw | afrac, prad | 0.411 / 0.434 | 68 / 291 | 6 / 10 | 122 / 35 | 419 | 1264 |
| with_relative_fdiv_band_0.1_mw | afrac, prad | 1.052 / 1.176 | 134 / 106 | 15 / 7 | 218 / 102 | 374 | 1220 |
| with_absolute_fdiv_band_0.1_mw | afrac, prad | 0.399 / 0.446 | 56 / 287 | 5 / 10 | 178 / 51 | 356 | 1208 |
| with_relative_fdiv_band_0.15_mw | afrac, prad | 1.021 / 1.206 | 108 / 100 | 12 / 6 | 293 / 194 | 234 | 1085 |
| with_absolute_fdiv_band_0.15_mw | afrac, prad | 0.387 / 0.458 | 43 / 286 | 6 / 10 | 237 / 71 | 289 | 1143 |
| with_relative_fdiv_band_0.2_mw | afrac, prad | 0.990 / 1.237 | 66 / 94 | 6 / 5 | 354 / 265 | 143 | 1001 |
| with_absolute_fdiv_band_0.2_mw | afrac, prad | 0.376 / 0.469 | 39 / 278 | 4 / 8 | 296 / 86 | 222 | 1081 |
| with_relative_fdiv_band_0.3_mw | afrac, prad | 0.929 / 1.299 | 64 / 81 | 7 / 4 | 377 / 286 | 91 | 972 |
| with_absolute_fdiv_band_0.3_mw | afrac, prad | 0.352 / 0.493 | 49 / 277 | 4 / 8 | 361 / 90 | 139 | 1003 |

`primary` is the exported label: Afrac is the only second voter. `no_second_voter` has none, so every TangTV vote that survives the L-mode gate is TangTV only (473 attached and 377 detached bins); the gate still applies, so it is not the TangTV vote before the gate of the Te rows (`tangtv_vote_before_gate`, 668 bins with a Te, with no gate), which includes the gated detached votes. The `with_*` rows add f_div as a second voter next to Afrac (`with_relative_fdiv` reproduces the label of the earlier build, where f_div was a vote: it makes many more agreement-tier bins and many more conflicts); `relative` takes cutoffs that scale the shot baseline, `absolute` the 201081-anchored global cutoffs (the midpoint of the measured attached and detached Prad,div,L over its P_in; one pair for every shot). `band_X` sets the half-width of the uncertain band to X MW (the primary keeps 0.1 MW). `afrac_abstains` leaves f_div as the only second voter. `published_anchor_values` uses the published 1.6 / 2.2 MW. A two-voter variant also has the tier `low_confidence_pair` (the two second voters agree and TangTV casts none), folded here into the uncertain bins.

Decision criterion: attached AND detached bins on at least 3 of the same shots, at least one of them a cohort shot (split train, val or test). Agreement-tier attached and detached bins coexist on 2 shots (0 in the cohort); counting the TangTV-only tier as well, 8 shots (0 in the cohort). The criterion is not met, so the label set is presented as an indicator-agreement appendix, not as a three-state label set.

### Bin width

The durations are fixed and converted to bins by the width: the Afrac reference needs 1 s (50 bins at 20ms, 20 bins at 50ms, 10 bins at 100ms), the f_div baseline 2 s (100, 40, 20), a persistent MARFE front 100 ms (5, 2, 1) and shot eligibility 1 s (50, 20, 10). Cells in brackets are shots.

| Width | Agreement tier bins (att / det) | TangTV-only bins (att / det) | Agreement tier, s (shots) | TangTV only, s (shots) | Agreement share of labelled seconds | Afrac valid, s (shots) | Afrac short reference, s | Afrac no power, s | Flicker per s | Agreement with 50 ms (bins) |
|---|---|---|---|---|---|---|---|---|---|---|
| 20ms | 66 / 106 | 997 / 687 | 3.44 (6) | 33.7 (22) | 0.093 | 9.4 (6) | 25.2 | 3.5 | 0.89 | 0.989 (1757) |
| 50ms | 36 / 40 | 392 / 287 | 3.80 (6) | 34.0 (22) | 0.101 | 11.8 (6) | 24.1 | 3.6 | 0.21 | - |
| 100ms | 24 / 27 | 199 / 140 | 5.10 (6) | 33.9 (22) | 0.131 | 13.0 (6) | 22.8 | 4.1 | 0.18 | 0.989 (362) |

The same compatibility rule at each width on the 77 non-test shots of the width roster (25 of them assessed at 50 ms; the roster is not the exported shot list, so its 50 ms row differs from the coverage table above; the widths are comparable with each other, and the 50 ms rows reproduce the exported bins on the shots they share). Shot eligibility is applied at each width (the same duration, counted in bins): 52 of the 77 loaded shots are ineligible at every width. With the durations held fixed the labelled time is about the same at every width (37.1 s at 20ms, 37.8 s at 50ms, 39.0 s at 100ms), the time in which Afrac has no long enough reference is stable (25.2 s, 24.1 s, 22.8 s), and where two widths both label a bin the states agree: 0.989 at 20ms (1757 bins), 0.989 at 100ms (362 bins). What moves is the agreement tier: 3.44 s on 6 shots at 20ms, 3.80 s on 6 shots at 50ms, 5.10 s on 6 shots at 100ms (a share of the labelled seconds of 0.093, 0.101, 0.131). Afrac's heating-power input is interpolated to the bin centres (as the f_div window means are), so a narrow bin is not lost for want of a sample: Afrac is invalid for 3.5 s at 20 ms, 3.6 s at 50 ms and 4.1 s at 100 ms for `no_power` (a bin centre outside the power record or in a gap of it). The agreement tier still grows with the width because Afrac is valid for longer (9.4 s, 11.8 s, 13.0 s at 20, 50 and 100 ms): the time it loses to the ELM exclusion falls from 3.8 s at 20 ms to 1.3 s at 100 ms and the time to a short reference from 25.2 s to 22.8 s (the power input is no longer the cause, and the rest is not attributed here). A reference length counted in bins instead of seconds (20 bins) would have been 0.4 s at 20 ms and 2 s at 100 ms and produced a strong width dependence that is not in the data. The 20 ms labels flicker 0.89 times per second against 0.21 at 50 ms and 0.18 at 100 ms (a flicker is a change of labelled state between consecutive bins).

**Why 50 ms.** The camera is a 60 Hz interlaced camera captured as 30 Hz full frames (Chen 2026), but the parked inversions are spaced 16.7 ms (median of 32 shots; every shot between 16.6 and 16.7 ms), about 3 inversions in a 50 ms bin, and the raw VID frames in the SAV files are 16.7 ms apart as well; the corpus's raw TangTV movie is resampled to 20 ms (2.5 frames per 50 ms bin; 13 shots checked). A 50 ms bin therefore holds at least two camera frames of either kind (a 20 ms bin holds about one inversion), and the 250 ms radiation averaging window spans five bins. It is the width at which the exports are built; the table above is the evidence that the states do not depend on it.

### MARFE witness

**Recall of the published MARFE on 199166 at 3705 ms (Chen 2026): 0/1 as a state** (no bin is exported as MARFE). The bin holding that time (3700-3750 ms) is the uncertain tier `candidate_marfe` (1/1 as a candidate). In the bins from 3700 ms the TangTV front is at DZ 1.24, 1.30, 1.30 (cutoff 1.2), the emission peak is inside the separatrix from 3750 ms, and fG is 0.72, 0.74, 0.75, below the 0.8 cue, so the full MARFE vote does not fire on this shot. The cue's value has no literature source (Dong 2025: fG of about 0.5 or more on HL-3 with a core-point density and a core-Te condition), so even where it passes the MARFE stays a candidate in the exports and figures; the full vote fires on 23 bins of one other shot. The witness figure shows the inversion at 3716 ms, inside the bin 3700-3750 ms whose DZ and fG it quotes (the nearest frame to 3705 ms, at 3699 ms, is in the neighbouring bin, which is TangTV only detached with DZ 0.91). The thresholds were not tuned to this shot.

<!-- /RESULTS -->

## Figures and reproduction

Figures are vector PDF and 150 dpi PNG with text of at least 7 pt, in colour-blind
safe colours, drawn at the 6.75 inch text width unless noted:

- `fig_detachment_views`: camera inversion, EFIT contours, Prad,div,L and P_rad
  traces for shot 201081 at time-ordered columns taken from sustained TangTV-attached,
  TangTV-detached and TangTV-attached intervals (at least five bins). Each column is
  titled by its TangTV vote, with the exported label of the bins in the interval in
  brackets (TangTV + Afrac agreement, TangTV only, uncertain); the legend and caption
  say so. The X-point and the outer strike point are marked. The raw TangTV frames
  carry no EFIT overlay because no camera projection calibration exists (the
  inversion panels carry the overlay), and no IRTV heat-flux row is drawn because
  the IRTV HEATFLUX fetch returned NODATA. No 2D bolometer emissivity exists
  (`detachment_bolometer_availability.json`), so the radiation row shows the
  PRAD_DIVL and PRAD_TOT traces.
- `fig_detachment_timeline`: label, Afrac and TangTV vote strips, the relative f_div
  (a corroborator, marked "not a vote") and DZ with their cutoffs, and the divertor
  Thomson Te; below, per-shot bars of the TangTV + Afrac agreement tier, the
  TangTV-only tier, the L-mode-gate uncertain bins (green; the TangTV detached
  votes the gate withholds on known and probable L-mode phases) and the other
  uncertain bins, with the shot IDs on the axis. The caption repeats the two notes of the
  views figure (no IRTV row, no overlay on the raw frames).
- `fig_detachment_marfe_witness`: the 199172 MARFE candidate (no certain MARFE) and
  the 199166 bin that holds the published onset (3700-3750 ms), each showing an
  inversion inside its named bin (the 199166 frame is at 3716 ms) with the X-point
  marked, the frame time, the label bin, its DZ and fG, its tier, and the recall
  (0 of 1).
- `fig_detachment_figure2` (3.25 inch panel): coverage by class and tier, labelled
  bins first and the number of assessed bins that can never carry a state; the AUROC
  of -Te per vote source, with the number of shots beside every row and a marker
  legend (open markers: the label tiers and the indicator votes; filled markers:
  the TangTV vote before the L-mode gate, which is the check). The rows are the
  labelled bins, the agreement tier, the TangTV-only tier, the TangTV vote before
  the gate, the same with 201081 left out, then the TangTV vote before the gate by
  regime (known H-mode, known L-mode, probable L-mode, probable H-mode; the
  unknown-regime remainder is too small to score), the Afrac vote and the f_div
  votes (unused). Under each regime row the share of the detached votes at or
  below 5 eV is written (known L-mode 3/31), and a row with no attached bins (known
  H-mode, probable L-mode) has no AUROC and says so ("n/a: no attached bins"). A
  row over fewer than three shots has a value and no interval ("n/a (N shots)" in
  the record) and is not drawn. Below it, as a corroborator, the within-shot and
  pooled AUROC of the relative and absolute f_div against TangTV and against Te
  (filled blue diamond: pooled; open blue triangle: mean over shots), with the shot
  counts. Each `te_tangtv_vote_before_gate_regime_*` row of the record carries
  `share_in_band` (attached and detached, with bin counts) and a note. The source is
  `docs/labeler/figure2_detach.json`, the one canonical record.

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

<!-- LIMITATIONS -->

The published points are few (ten, on three shots) and share their sources with the indicator definitions, so they support neither accuracy nor calibration. The TangTV + Afrac agreement tier (`certain`) measures agreement between two indicators, not physical accuracy and not confidence: the second vote does not improve the label's agreement with divertor Thomson Te over the TangTV vote before the gate (the paired difference is not estimable on its 6 shots), and the TangTV-only tier is more Te-consistent than the agreement tier. Te has informed rule decisions (the Afrac keep rule, the removal of f_div as a voter, the L-mode gate), so the comparison with Te is a consistency check, not a validation, and the shares in band after the gate are post-selection. All agreement-tier bins are on shots with no known regime.

The TangTV DZ cutoffs come from one H-mode shot (201081). The L-mode gate withholds 93 TangTV detached votes on 6 shots of a known L-mode phase (3 of the 31 that have a Te are at or below 5 eV) and 94 on 4 shots of a probable-L phase. The regime is known on 9 of the 32 assessed shots, so the probable-L proxy (unknown regime, ELM coverage known, no ELM share, median P_in under 2 MW) does much of the work; it is a rule on two signals, not a regime measurement, and it has a price: it withholds 66 detached votes with DZ >= 0.8 (32 on 190110, 32 on 190113, 2 on 190116), cold where a Te exists (47 of the 47 bins that have a Te are at or below 5 eV), together with 28 warm votes (28 on 190109; 5 of its 22 bins with a Te are at or below 5 eV, median 6.8 eV). Its zero-ELM-share condition is strict: 189448 (6 flagged bins of 90, at most 1.2 MW; median P_in 1.0 MW); 189451 (4 flagged bins of 96, at most 0.7 MW; median P_in 1.1 MW) type as probable-H on a few ELM flags in the current ramp although their median power is under the cut; they carry 0 detached votes, so no label depends on it. The window counts the bins that have both an ELM flag and a measured P_in; a first definition counted every bin with ELM coverage and typed 190113 probable-H through one flag on a bin without input power, so the window was narrowed. That is a definitional change (a bin without power cannot support a power criterion), but it was made after seeing that outcome. 8 labelled detached bins on 2 shots lie outside the window and keep the plain rule.

The Afrac proxy is a weak second vote: it is valid on few shots, its Te interval includes chance, its result depends on the flux window, and the L-mode gate of Afrac itself knows the regime on a minority of shots; most labelled bins are therefore `tangtv_only`, not because Afrac disagrees but because it has no valid measurement there. f_div corroborates TangTV and Te within the few shots that have both classes but not pooled over shots (AUROC below 0.5), and the absolute cutoffs do not carry over from their anchor shot either, so it is reported per shot and does not vote; the shot counts beside every f_div figure are small. Of the assessed bins, 566 on 13 shots can never carry a state (TangTV is invalid there), so the labelled set (821 bins on 23 shots) is the number to quote. Only one assessed shot is in the fixed cohort and none is a test shot; agreement-tier attached and detached bins never coexist on a cohort shot, so the set cannot support the cohort benchmark or a study of attached-to-detached transitions. No bin is exported as MARFE: the one published MARFE (199166 at 3705 ms) is a `candidate_marfe` bin, recalled 0 of 1 as a state. The Thomson check covers attached and detached only, on the shots that have a chord near the target; the within-shot Te figures rest on a handful of shots, and an interval over fewer than 3 shots is not given. The labelled states do not depend on the bin width once the minimum durations are fixed in seconds (labelled time and states agree across 20, 50 and 100 ms); the agreement tier does (3.4 s at 20ms, 3.8 s at 50ms, 5.1 s at 100ms), through the time Afrac is valid, not through its power input. Indicator agreement can share systematic biases. Future work needs an independent expert state table, more inversion coverage, an a-priori regime for every shot and a literature-backed density cue for MARFE.

<!-- /LIMITATIONS -->

## Appendix: learned baselines

<!-- BASELINES -->

Two learned baselines, retrained on the labels described here (labels sha256 `a683b477dca0`; the architecture and epochs are unchanged), predict the labelled states, agreement tier and TangTV only together (821 bins), with 5-fold shot-grouped cross-validation over the non-test shots; test shots are never fitted or scored here. `detach-ours` reads windows of 0-D signals (current, density, D-alpha, ELM share, EFIT scalars) and `detach-victor` raw TangTV frames. Scores are agreement with constructed labels, not physical accuracy: `detach-victor` reads the same camera that sets the TangTV state, so its agreement partly reproduces the label from its own input. The fold-majority control predicts each held-out fold's bins as the majority class of the training fold; with shot-grouped folds that class is often the minority of the held-out shots, so the control scores below 0.5 and kappa 0 (a constant class) is the plainer reference. Intervals are shot-bootstrap (1000 replicates); bins are those with complete inputs or a camera frame, and the lower ends of the kappa intervals are 0.12 (`detach-ours`) and 0.00 (`detach-victor`).

| Model | Bins / shots | Accuracy [95% shot CI] | Kappa [95% shot CI] | Macro-F1 [95% shot CI] |
|---|---|---|---|---|
| detach-ours | 706 / 22 | 0.73 [0.58, 0.87] | 0.45 [0.12, 0.73] | 0.72 [0.55, 0.87] |
| detach-ours fold-majority control | 706 / 22 | 0.40 [0.23, 0.57] | -0.30 [-0.58, 0.04] | 0.34 [0.19, 0.49] |
| detach-victor | 258 / 10 | 0.55 [0.28, 0.83] | 0.23 [0.00, 0.53] | 0.54 [0.25, 0.75] |
| detach-victor fold-majority control | 258 / 10 | 0.81 [0.59, 1.00] | 0.00 [0.00, 0.00] | 0.45 [0.37, 0.50] |

MARFE transfer is unsupported (no MARFE label exists). Records: `docs/labeler/results/detachment_ours.json` and `detachment_victor.json`; scripts `detach_ours.py` and `detach_victor.py`. They are not listed among the dataset's models: they are a comparison of what 0-D signals and TangTV frames can reproduce of the constructed labels.

<!-- /BASELINES -->

## References

- Chen et al., *Nuclear Fusion* 66, 036014 (2026),
  [doi:10.1088/1741-4326/ae3972](https://doi.org/10.1088/1741-4326/ae3972).
- Eldon et al., *Nuclear Materials and Energy* 27, 100963 (2021),
  [doi:10.1016/j.nme.2021.100963](https://doi.org/10.1016/j.nme.2021.100963).
- Eldon et al., *Plasma Physics and Controlled Fusion* 64, 075002 (2022),
  [doi:10.1088/1361-6587/ac6ff9](https://doi.org/10.1088/1361-6587/ac6ff9).
- Martin et al., *Journal of Physics: Conference Series* 123, 012033 (2008),
  [doi:10.1088/1742-6596/123/1/012033](https://doi.org/10.1088/1742-6596/123/1/012033).
- Victor and Scotti, *Review of Scientific Instruments* 95, 083503 (2024),
  [doi:10.1063/5.0218724](https://doi.org/10.1063/5.0218724).
