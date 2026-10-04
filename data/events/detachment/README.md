# Detachment

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

**latest**: detach_vote | 2026_10_04 (unverified compatibility rule).

**all**:

<!-- MODELS -->

- detach_vote | 2026_10_04: exploratory geometry-gated TangTV state validated by divertor Thomson Te; a per-probe Afrac proxy is the second vote and f_div a within-shot corroborator; 1,779 assessed bins/32 shots; certain 147 bins/7 shots (attached 60, detached 87); TangTV only (silver) 861 bins/26 shots (attached 393, detached 468). AUROC of -Te against the vote: TangTV alone 0.95 [0.90, 0.98] on 23 shots, certain tier 0.94 [0.04, 0.97] on 6 shots. No MARFE state is exported. No independent benchmark.
- detach_rule | 2026_10_04: alias of the primary rule.

Class support, agreement tables and intervals are in [the protocol](../../../docs/labeler/detachment.md) and its linked JSON records.

<!-- /MODELS -->

## Inputs

- **Afrac proxy**: inter-ELM Jsat of each positioned probe, scaled by P_SOL^(3/7) over
  relative density squared and divided by that probe's OWN attached level (its
  0.9-quantile over the bins within |psiN - 1| <= 0.01 of the separatrix); the probe
  nearest the separatrix is read. Known L-mode bins abstain. Uncalibrated. Votes
  attached at >= 0.75 and detached at <= 0.5. The probe, its position and flux are
  exported per bin.
- **f_div** (a within-shot corroborator, not a vote of the label): Prad,div,L over
  P_in, both as centered 250 ms inter-ELM means, also taken over the shot's own
  baseline (the shot-relative f_div). It would vote attached below 1.052 and
  detached above 1.176, band edges of shot 201081 over its attached level (see the
  protocol); those votes, and the 201081-anchored absolute cutoffs (0.399 and 0.446
  of P_in, global), are sensitivities, exported as `prad_vote`, `prad_abs_valid`,
  `prad_abs_vote`, `state_rule_relative_prad`, `tier_relative_prad`,
  `state_rule_absolute_prad` and `tier_absolute_prad` (f_div added as a second voter
  next to Afrac). Within each shot it ranks TangTV-detached above TangTV-attached
  bins and cold above warm divertor Thomson Te; the protocol reports it per shot.
- **TangTV DZ**: the C-III front height from the inversion on the upper shelf.
  Attached below 0.35, detached from 0.5 to 1.2; DZ >= 1.2 for two adjacent bins
  with an emission peak inside the separatrix near the X-point and fG >= 0.8 (or a
  recorded H-L cue) is the TangTV MARFE vote, exported only as the uncertain tier
  `candidate_marfe`. The lower-shelf TangTV window is invalid.
- **Divertor Thomson Te** (`\ELECTRONS::TSTE_DIV`): not an indicator; the independent
  check of the states.

## Method

The label set is the geometry-gated TangTV state, validated by divertor Thomson Te.
Each 50 ms bin carries each indicator's value, validity and vote. Two indicators vote,
TangTV and Afrac. A bin takes the TangTV state (attached or detached) when TangTV
votes on the upper shelf with known ELM coverage. It is `certain` when a valid Afrac
vote agrees with it, and `tangtv_only` (silver) when Afrac casts no vote (it
abstains between its cutoffs or is invalid); when Afrac votes against TangTV the bin
is uncertain, tier `conflict`. No bin is a MARFE: a persistent high front is the
uncertain tier `candidate_marfe`. Every other bin is uncertain with its reason in
`tier`: `conflict`, `insufficient_support`, `no_vote`, `candidate_marfe`,
`lower_shelf_window`, `geometry_unknown` or `elm_unknown`.
Primary states are unsmoothed; `confidence` is null.

A shot is exported when at least two indicators are each valid on at least 20 bins
and it has at least 20 assessed bins. See [the protocol](../../../docs/labeler/detachment.md) for
the definitions, the results tables and the reproduction order. Figure 2 reads
`docs/labeler/figure2_detach.json`; large outputs and `HANDOFF.md` are under
`$LABELER_ROOT/round4/detach/`.
