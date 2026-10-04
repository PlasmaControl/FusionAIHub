# Detachment

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

Missing intervals are unassessed. Expert review is pending.

## Data Provenance

### Dataset 1

**Dataset File(s)**: `raw/emission_structure_*.sav` and matching raw SAVs, packed
outside git in `$LABELER_ROOT/round4/detach/inversions/`.

**Author**: Nathaniel Chen (plasma_tv).

**Description**: C-III inversions on about 41 shots. The upper shelf is at
Z=-1.25 m, R>=1.37 m; the lower shelf at Z=-1.363 m, R<1.37 m. Only the upper
shelf yields a state. The owner's lower-shelf extraction
(`make_labels_2026.make_labels_for_file(..., r_max=1.37)`) is kept as the tier
`lower_shelf_window`: its bins are uncertain, and it is not a label column and not
in the detach-ui handoff. TangTV accepts the ELM-integrated 30 Hz frames as Chen
2026 does, and requires known ELM coverage. The camera surrogate contributes no
valid bins.

**Publications**: Chen et al., Nuclear Fusion 66, 036014 (2026).

### Dataset 2

**Dataset File(s)**: the read-only corpus and the fetched
`$LABELER_ROOT/round4/detach/{cache,processed_probes,geometry02,efit,dts}/`.

**Author**: DIII-D diagnostic teams.

**Description**: Positioned scrape-off-layer processed Jsat, lower-divertor
radiation, heating, density, filterscopes and EFIT geometry and flux (EFIT02 where
available, close multi-slice EFIT01 as a named fallback). Input power is the neutral
beams plus the EFIT ohmic power plus ECH; where the corpus beam group is empty, the
beam total is read from PTDATA `BMSPINJ`. The processed divertor Thomson
temperature (`\ELECTRONS::TSTE_DIV`) is used only as an independent check, never as
a vote.

**Publications**: Eldon et al., NME 18, 285 (2019); NME 27, 100963 (2021);
PPCF 64, 075002 (2022).

### Dataset 3

**Dataset File(s)**: `docs/labeler/results/detachment_reference.json`; the owner's
manual emission-front points are read-only in the adjacent plasma_tv repository.

**Description**: Ten published shot/time/state points (shots 180257, 199166, 201081)
and a one-shot manual DZ check. The set is small, shares its sources with the
indicator definitions, and cannot establish campaign-wide accuracy. No independent
expert state table has been received. `shots.csv` is an unverified review roster.

## Models

**stable**: none.

**latest**: detach_vote | 2026_10_03 (unverified compatibility rule).

**all**:

<!-- MODELS -->

- detach_vote | 2026_10_03: exploratory geometry-gated TangTV state validated by divertor Thomson Te; per-shot relative f_div and a per-probe Afrac proxy as second indicators; 1,779 assessed bins/32 shots; certain 275 bins/19 shots (attached 132, detached 143); TangTV only (silver) 398 bins/22 shots (attached 219, detached 179). AUROC of -Te against the vote: TangTV alone 0.95 [0.90, 0.98], certain tier 0.91 [0.77, 0.96]. No MARFE state is exported. No independent benchmark.
- detach_rule | 2026_10_03: alias of the primary rule.

Class support, agreement tables and intervals are in [the protocol](../../../docs/labeler/detachment.md) and its linked JSON records.

<!-- /MODELS -->

## Inputs

- **Afrac proxy**: inter-ELM Jsat of each positioned probe, scaled by P_SOL^(3/7) over
  relative density squared and divided by that probe's OWN attached level (its
  0.9-quantile over the bins within |psiN - 1| <= 0.01 of the separatrix); the probe
  nearest the separatrix is read. Known L-mode bins abstain. Uncalibrated. Votes
  attached at >= 0.75 and detached at <= 0.5. The probe, its position and flux are
  exported per bin.
- **f_div**: Prad,div,L over P_in, both as centered 250 ms inter-ELM means, taken
  over the shot's own baseline (the shot-relative f_div). Votes attached below
  1.052 and detached above 1.176, band edges of shot 201081 over its attached
  level (see the protocol). The 201081-anchored absolute cutoffs (0.399 and 0.446
  of P_in, global) are the sensitivity alternative, exported as `prad_abs_valid`,
  `prad_abs_vote`, `state_rule_absolute_prad` and `tier_absolute_prad`.
- **TangTV DZ**: the C-III front height from the inversion on the upper shelf.
  Attached below 0.35, detached from 0.5 to 1.2; DZ >= 1.2 for two adjacent bins
  with an emission peak inside the separatrix near the X-point and fG >= 0.8 (or a
  recorded H-L cue) is the TangTV MARFE vote, exported only as the uncertain tier
  `candidate_marfe`. The lower-shelf TangTV window is invalid.
- **Divertor Thomson Te** (`\ELECTRONS::TSTE_DIV`): not an indicator; the independent
  check of the states.

## Method

The label set is the geometry-gated TangTV state, validated by divertor Thomson Te.
Each 50 ms bin carries each indicator's value, validity and vote. A bin takes the
TangTV state (attached or detached) when TangTV votes on the upper shelf with known
ELM coverage. It is `certain` when at least one other indicator (the shot-relative
f_div or Afrac) casts the same vote and none disagrees, and `tangtv_only` (silver)
when TangTV is the only vote. No bin is a MARFE: a persistent high front is the
uncertain tier `candidate_marfe`. Every other bin is uncertain with its reason in
`tier`: `conflict`, `insufficient_support`, `low_confidence_pair`, `no_vote`,
`candidate_marfe`, `lower_shelf_window`, `geometry_unknown` or `elm_unknown`.
Primary states are unsmoothed; `confidence` is null.

A shot is exported when at least two indicators are each valid on at least 20 bins
and it has at least 20 assessed bins. See [the protocol](../../../docs/labeler/detachment.md) for
the definitions, the results tables and the reproduction order. Figure 2 reads
`docs/labeler/figure2_detach.json`; large outputs and `HANDOFF.md` are under
`$LABELER_ROOT/round4/detach/`.
