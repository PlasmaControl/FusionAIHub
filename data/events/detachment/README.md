# Detachment

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

- detach_vote | 2026_10_03: exploratory TangTV-led compatibility rule over three indicators; 2,873 assessed bins/37 shots; 471 certain bins/24 shots (attached 78, detached 370, MARFE 23). Agreement of the TangTV vote with divertor Thomson Te: AUROC 0.95 [0.91, 0.98]. No independent benchmark.
- detach_rule | 2026_10_03: alias of the primary rule.

Class support, agreement tables and intervals are in [the protocol](../../../docs/labeler/detachment.md) and its linked JSON records.

<!-- /MODELS -->

## Inputs

- **Afrac proxy**: inter-ELM Jsat of the peak-Jsat probe on the scrape-off side of
  the outer strike (psiN > 1.000, at least 5 mm outboard of the strike point,
  psiN <= 1.05), scaled by P_SOL^(3/7) over relative density squared, divided by
  the shot's own 90th-percentile reference. Uncalibrated. Votes attached at >=0.75
  and detached at <=0.5. The probe, its position and flux are exported per bin.
- **f_div**: Prad,div,L over P_in, both as centered 250 ms inter-ELM means. Votes
  attached below the lower cutoff and detached above the upper one; the cutoffs
  come from shot 201081 (see the protocol). `state_rule_relative_prad` repeats the
  rule with per-shot relative cutoffs.
- **TangTV DZ**: the C-III front height from the inversion on the upper shelf.
  Attached below 0.35, detached from 0.5 to 1.2; DZ >= 1.2 for two adjacent bins
  with an emission peak inside the separatrix near the X-point and fG >= 0.8 (or a
  recorded H-L cue) is the MARFE vote.

## Method

Each 50 ms bin carries each indicator's value, validity and vote. A bin is certain
attached or detached when TangTV votes, at least one other indicator casts the same
vote, none disagrees, the ELM coverage is known and the geometry is the upper
shelf. It is certain MARFE on the TangTV MARFE vote unless Afrac or f_div votes
attached. Afrac and f_div never corroborate a MARFE. Every other bin is uncertain
with its reason in `tier`: `conflict`, `insufficient_support`,
`low_confidence_pair`, `no_vote`, `candidate_marfe`, `lower_shelf_window`,
`geometry_unknown` or `elm_unknown`. Primary states are unsmoothed; `confidence`
is null.

A shot is exported when it has at least 20 assessed bins and 20 valid bins for each
contributing indicator. See [the protocol](../../../docs/labeler/detachment.md) for
the definitions, the results tables and the reproduction order. Figure 2 reads
`docs/labeler/figure2_detach.json`; large outputs and `HANDOFF.md` are under
`$LABELER_ROOT/round4/detach/`.
