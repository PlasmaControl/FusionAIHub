# Detachment

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

Missing intervals are unassessed. Expert review is pending; the fitted Snorkel
model is vestigial and its probabilities are uncalibrated.

## Data Provenance

### Dataset 1

**Dataset File(s)**: `raw/emission_structure_*.sav` and matching raw SAVs, packed
outside git in `$LABELER_ROOT/round4/detach/inversions/`.

**Author**: Nathaniel Chen (plasma_tv).

**Description**: C-III inversions on about 41 shots. Upper shelf Z≈−1.25 m,
R≥1.37 m; lower shelf Z≈−1.363 m, R<1.37 m. The owner's
`make_labels_2026.make_labels_for_file(..., r_max=SHELF_WALL_R=1.37)` extracts the
restricted lower-shelf window. Its votes remain in `lower_shelf_window`, pending
owner sign-off, and cannot enter primary certainty. TangTV accepts ELM-integrated
30 Hz frames as Chen 2026 does, while requiring known ELM coverage. The surrogate
contributes **zero valid bins** and does not extend coverage.

**Publications**: Chen et al., Nuclear Fusion 66, 036014 (2026).

### Dataset 2

**Dataset File(s)**: read-only corpus and fetched
`$LABELER_ROOT/round4/detach/{cache,processed_probes,geometry02,efit}/`.

**Author**: DIII-D diagnostic teams.

**Description**: Positioned SOL-side processed Jsat, lower-divertor radiation,
heating, density, filterscopes and EFIT geometry/flux. EFIT02 is preferred; close
multi-slice EFIT01 maps are named fallbacks where appropriate. Unknown ELM
coverage vetoes certainty. Unit-confirmed density is required for Greenwald
fraction. No spatially localized Thomson temperature reference is available.

**Publications**: Eldon et al., NME 18, 285 (2019); NME 27, 100963 (2021);
PPCF 64, 075002 (2022).

### Dataset 3

**Dataset File(s)**: `docs/labeler/results/detachment_reference.json`; owner's
manual emission-front points are read-only in the adjacent plasma_tv repository.

**Description**: Three explicit published shot/time/state anchors, including
199166 MARFE at 3.705 s, plus a separate one-shot manual DZ check. This small
reference is independent of the compatibility calculation but cannot establish
campaign-wide accuracy. No independent expert state table has been received.
`shots.csv` is an unverified review roster.

## Models

**stable**: none.

**latest**: detach_vote | 2026_10_03 (unverified compatibility rule).

**all**:

<!-- MODELS -->

- detach_vote | 2026_10_03: unverified two-indicator rule; 1737 assessed bins/36 shots; 307 certain bins/16 shots. Exploratory coverage/agreement; no independent benchmark.
- detach_rule | 2026_10_03: alias of the primary compatibility rule.
- detach-ours | 2026_10_03: held-shot weak-label accuracy 0.835 [0.549, 1.000]; adjacent fold-majority 0.970 [0.877, 1.000]; κ -0.052 [-0.170, 0.000]. No established learning beyond majority; MARFE transfer unsupported.
- detach-victor | 2026_10_03: held-shot weak-label accuracy 0.899 [0.631, 1.000]; adjacent fold-majority 0.899 [0.631, 1.000]; κ 0.000 [0.000, 0.000]. No established learning beyond majority; MARFE transfer unsupported.

Class support, model/control confusion matrices and intervals are in [the protocol](../../../docs/labeler/detachment.md) and linked JSON records. The primary rule abstains on all three published reference points; state validation remains unavailable.

<!-- /MODELS -->

## Inputs

The rule uses positioned SOL-side target current, Prad,div/heating, true TangTV
inversions, EFIT strike/X/flux, filterscope coverage, density and Ip. The current
indicator is an **uncalibrated Jsat ratio (local proxy)**; no fitted Eldon attached
L/H reference is available. Selected probe provenance is exported per bin.

`detach-ours` reads only complete finite windows of current, relative
density, D-alpha, ELM share and global EFIT scalars. No missingness channels or
imputed values are used. `detach-victor` is a raw-camera CNN adaptation
with shot-grouped evaluation and training-fold intensity normalization.

## Method

Each 50 ms bin carries values, validity/reasons and compatible votes. Certain
means upper-shelf TangTV plus another compatible cast vote, no conflict and known
ELM coverage. Afrac+Prad alone remain uncertain. Lower-shelf extraction remains
`lower_shelf_window`; its separate provisional state never becomes primary
certainty. The fitted model's posterior does not decide the primary label.

Current ratio ≥0.75 votes attached / ≤0.5 detached. Prad thresholds ≤0.36 / ≥0.50
are local settings, checked on external development inversions. Prad and input
power use matched centered **250 ms averages**, with D-alpha availability required
over the full radiation window (`aux_prad_elm_window_known`). Chen's 201081 example reports
4 MW NBI; local threshold derivation assumed an independently unsourced 4.4 MW
total heating (1.6/4.4≈0.36, 2.2/4.4=0.50). The 201081 `pinj` fetch failed because
`PTSERVER/ptserver` is absent from `/etc/services`, a PTDATA client configuration
error, not evidence that experimental power data are missing. **Cohort-train
has zero upper-shelf reference bins**. Eldon publishes the sensor construction;
classification thresholds are local. Prad/Jsat use their own ±2 ms ELM masks. Missing
filterscopes produce `elm_unknown`, not clean samples.

TangTV requires ≥10 cm leg, valid shelf geometry and a close EFIT slice.
DZ<0.35 votes attached; 0.5≤DZ<1.2 detached. High DZ is candidate MARFE;
confirmation needs two adjacent high-front bins, localized inversion emission
inside the separatrix near/above X and unit-confirmed fG≥0.8 or a recorded H–L
cue. Unknown density units do not create a cue. Primary states are unsmoothed;
temporal suggestions are separate and interval confidence is null.

Shot eligibility additionally requires at least **20 valid bins each for two
indicators and 20 jointly assessed bins**, a one-second minimum that narrows the
requested every-shot coverage.

See [the protocol](../../../docs/labeler/detachment.md) for coverage, threshold
margin/sensitivity tables and geometry-stratified indicator agreement. Figure 2
coverage/agreement input is `docs/labeler/figure2_detach.json`; its source record is
`docs/labeler/results/detachment_figure2.json`. Appendix figures require full
6.75-inch text width with ≥7 pt text. Large outputs and `HANDOFF.md` remain under
`$LABELER_ROOT/round4/detach/`.
