# Detachment

Attached, detached, MARFE and uncertain lower-divertor states from an unverified
compatibility rule with TangTV required. Paper use is **exploratory agreement and
coverage**. Missing intervals are unassessed. Expert review is pending; the fitted
Snorkel model is only a diagnostic and its probabilities are uncalibrated.

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
fraction. Unlocalized Thomson temperature claims remain withdrawn.

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

- detach_vote | 2026_10_03: compatibility rule; 1648 assessed bins/36 shots, 321 certain/23 shots, 12 MARFE; Snorkel is only a diagnostic.
- detach_rule | 2026_10_03: alias of the primary compatibility rule.
- detach-ours | 2026_10_03: combined-label CV κ 0.092 [-0.097, 0.586] (233 bins/21 shots); **LOO κ N/A** (0 bins/0 shots), inversion-only N/A; test-LOO accuracy N/A (0 bins/0 shots). Reference: bins where the other two indicators agree. Complete finite inputs only; no missingness channels.
- detach-victor | 2026_10_03: combined-label CV κ 0.000 [0.000, 0.000] (93 bins/11 shots); **LOO κ N/A** (0 bins/0 shots), inversion-only N/A; test-LOO accuracy N/A (0 bins/0 shots). Reference: bins where the other two indicators agree. Reviewed pre-fix Victor test-LOO was 0/28 on one shot.

The fixed 500-shot cohort contains **43 certain bins** on 1 validation shot(s); train has 0 and test has 0. Restored ELM coverage on 189061 supersedes the reviewed export's zero-cohort count. True TangTV inversions exist on only 41 shots; the surrogate supplies 0 valid bins. Current certainty totals 16.05 s; 46/98 certain intervals are single 50 ms bins. The reviewed pre-fix export had 56/113 single-bin intervals and 17.2 s total; those historical counts are superseded. [Before/after population record](../../../docs/labeler/results/detachment_round2.json).

Strict SOL/reference gates yield zero valid Jsat votes, so certainty currently uses Prad and TangTV. The primary consensus abstains on all 3 published reference points; independent state validation remains unavailable.

Corpus usable records out of 16,909: TangTV 7,233, bolo 13,846, Langmuir 6,239, IRTV 250. Source: detachment_round2.json. Scores are exploratory agreement and coverage; no primary gold-test accuracy is available.

<!-- /MODELS -->

## Inputs

The rule uses positioned SOL-side target current, Prad,div/heating, true TangTV
inversions, EFIT strike/X/flux, filterscope coverage, density and Ip. The current
indicator is an **uncalibrated Jsat ratio (local proxy)**; no fitted Eldon attached
L/H reference is available. Selected probe provenance is exported per bin.

`detach-ours` reads only complete finite windows of current, heating, relative
density, D-alpha, ELM share and global EFIT scalars. Missingness channels and
imputed values have been removed. `detach-victor` is a raw-camera CNN adaptation
with shot-grouped evaluation and training-fold intensity normalization.

## Method

Each 50 ms bin carries values, validity/reasons and compatible votes. Certain
means upper-shelf TangTV plus another compatible cast vote, no conflict and known
ELM coverage. Afrac+Prad alone remain uncertain. Lower-shelf extraction remains
`lower_shelf_window`; its separate provisional state never becomes primary
certainty. The fitted model's posterior does not decide the primary label.

Current ratio ≥0.75 votes attached / ≤0.5 detached. Prad thresholds ≤0.36 / ≥0.50
are local settings, checked on external development inversions; **cohort-train
has zero upper-shelf reference bins**. Eldon publishes the sensor construction;
classification thresholds are local. Prad/Jsat use their own ±2 ms ELM masks. Missing
filterscopes produce `elm_unknown`, not clean samples.

TangTV requires ≥10 cm leg, valid shelf geometry and a close EFIT slice.
DZ<0.35 votes attached; 0.5≤DZ<1.2 detached. High DZ is candidate MARFE;
confirmation needs two adjacent high-front bins, localized inversion emission
inside the separatrix near/above X and unit-confirmed fG≥0.8 or a recorded H–L
cue. Unknown density units do not create a cue. Primary states are unsmoothed;
temporal suggestions are separate and interval confidence is null.

See [the protocol](../../../docs/labeler/detachment.md) for coverage, paired
agreement and selected-reference benchmarks. Figure 2 F1 input is
`docs/labeler/figure2_detach.json`; the separate coverage table is
`docs/labeler/results/detachment_figure2.json`. Appendix figures require full
6.75-inch text width with ≥7 pt text. Large outputs and `HANDOFF.md` remain under
`$LABELER_ROOT/round4/detach/`.
