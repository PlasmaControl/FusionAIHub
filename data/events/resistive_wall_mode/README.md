# Resistive Wall Mode

## Description
The resistive wall mode (RWM) is the external kink (mostly n = 1) that a perfectly
conducting wall would stabilise but a real, resistive wall only slows: the mode
grows on the wall's flux-penetration time tau_w (milliseconds on DIII-D) instead of
the Alfvén time. It exists in the window between the no-wall and ideal-wall beta
limits,

    beta_N^no-wall  <  beta_N  <  beta_N^ideal-wall

(empirically beta_N^no-wall ~ 4 l_i on conventional DIII-D plasmas; this proxy is uncertain for the high-qmin, low-li roster here), and is stabilised by plasma rotation
and kinetic resonances or by active feedback with the C- and I-coils. Its onset
often ends a high-beta_N discharge in a disruption; error-field amplification
(resonant field amplification, RFA) is its marginally stable precursor.

First identified in the mid 1990s in DIII-D wall-stabilised high-beta experiments
and on the HBT-EP and PBX-M devices.

RWMs are locked or slowly rotating (≪ 100 Hz), detected with low-frequency n = 1
magnetics such as saddle loops and poloidal probe arrays near the no-wall limit.
The 0–10 kHz band describes tearing modes, rather than RWM rotation.

## Data Provenance
### Dataset 1

**Dataset File(s)**: `rwm_onsets_2017.csv`, `rwm_onsets_2024.csv`

**Author**: Jeremy Hanson

**Description**: Reference dataset from Jeremy Hanson: `rwm_onsets_2017.csv` (20 shots, 156785-158023) and `rwm_onsets_2024.csv` (13 shots, 176067-176092), 56 onsets over 33 distinct shots in `raw/`.

**Publications**:

### Dataset 2

**Dataset File(s)**: `extend_rwm_onset_window/rwm_windows.csv`, `rwm_windows.meta.json`, `rwm_windows.shots.csv`

**Author**: built here from Dataset 1 and DIII-D inputs fetched for 208 candidate shots (raw cache `$LABELER_ROOT/raw`).

**Description**: One minimal category-1 present slice [o, o+10 ms) per onset records presence without measuring duration; category-2 pre-onset windows remain uncertain (ONSET_TIME detection meaning unconfirmed). The export also contains assumed-absent spans before the first Hanson precursor, explicit category 4 unassessed precursor/post-onset time, and unlabelled screen spans on matched comparison shots. The review reader preserves the interval CSV's JSON `attrs.evidence_tier`; only Dataset 1's onset points are verified evidence. No post-onset physical absence is inferred without termination evidence. Counts, matching, frozen-cohort checks and input audit: `outputs/labeler/rwm/shots.json`.

**Publications**:

## Models
**stable**: none

**latest**: rwm-brf | 2026_10_03 (onset-derived forecasting baseline)

**all** (reference split (seed 0); Hanson primary 100 ms forecasts on 10 ms slices; assumed negatives):

- rwm-brf | 2026_10_03 | Primary AUROC: 0.760 [0.706, 0.809] (ties elapsed time) | Broad AUROC: 0.740 [0.668, 0.799] | AUPRC: 0.163 [0.123, 0.220] | F1: 0.275 [0.227, 0.337]
- rwm-rule-elapsed-time | 2026_10_03 | Primary AUROC: 0.759 [0.712, 0.815] | Broad AUROC: 0.390 [0.333, 0.440] | AUPRC: 0.228 [0.212, 0.307] | F1: 0.267 [0.217, 0.331]
- rwm-rule-betan | 2026_10_03 | Primary AUROC: 0.707 [0.639, 0.772] | Broad AUROC: 0.715 [0.641, 0.776] | AUPRC: 0.166 [0.128, 0.246] | F1: 0.246 [0.194, 0.312]
- rwm-rule-betan-over-li | 2026_10_03 | Primary AUROC: 0.723 [0.664, 0.784] | Broad AUROC: 0.752 [0.688, 0.815] | AUPRC: 0.159 [0.127, 0.232] | F1: 0.261 [0.212, 0.328]
- rwm-rule-rwm-candidates | 2026_10_03 | Primary AUROC: 0.500 [0.500, 0.500] | Broad AUROC: 0.498 [0.495, 0.500] | AUPRC: 0.084 [0.070, 0.102] | F1: 0.000 [0.000, 0.000]

No AUROC advantage over the strongest scalar, or onset-specific warning skill,
was established. Within shot, the forest is **0.02–0.07 below βN and βN/li on both
masks (one of four unadjusted intervals excludes zero)**. Intervals are exploratory,
unadjusted for multiple comparisons.
Primary/broad point means are forest **0.784/0.719**, beta_N **0.814/0.739**,
beta_N/l_i **0.855/0.786**, elapsed time **0.931/0.417**. Elapsed time's **0.931**
primary mean is an artefact of the mask's cutoff at the last onset (median **1.0**).
Sources: `evaluation.json/configs/<model>/within_shot_auroc` and
`evaluation.json/paired/rwm-brf - <scalar>/within_shot_auroc`.


Within campaign and 200 ms elapsed-time bins, primary AUROC is **0.544
[0.453, 0.629]** for the forest, versus **0.582** elapsed time, **0.541** βN and
**0.595** βN/li. All paired forest-minus-scalar intervals include zero. Source:
`evaluation.json/{configs/<model>,paired/rwm-brf - <scalar>}/phase_controlled_auroc`.

In 2014 high-beta, the forest is below chance on the reference split
(**0.311 [0.22, 0.41]**); the scalar rules are near chance. Forest detection ranges
**0.125–0.271** over five splits; no improvement over the approximate rate-matched
random reference was established. All four rules were compared on seed 0 only;
rule alarms were not replayed over seeds 1–4. Sources:
`evaluation.json/configs/<model>/by_campaign/2014/metrics/high_beta_auroc`,
`split_sensitivity/alarm_ranges`. Full results and limitations are in
[the current protocol](../../../docs/labeler/rwm_baseline.md) and
[evaluation.json](../../../outputs/labeler/rwm/evaluation.json).

`nnpu.py`, `evaluate.Nnpu`, and `Brf(use_comparison=True)` are excluded
development code and contribute to none of these results.

## Inputs
**rwm-brf** (stored scalars and profiles; trailing/held features on a 10 ms grid):

- `betan`, `li`, `q95`, `qmin`, `wmhd`, `ip`, with beta_N/l_i and beta_N-4l_i derived.
- `n1rms`, `n2rms`: postprocessed magnetic RMS, trailing mean, peak and log-slope calculations; upstream timing is uncertain. N1RMS is not a direct RWM sensor.
- ZIPFIT toroidal rotation at fixed rho=0.25 (core) and rho=0.625 (mid-radius, not an identified q=2 surface); its upstream time smoothing is acausal, with timing bias unbounded here. The inherited `rot_*_khz` columns retain recorded `units_from_source=kHz`; krad/s is an unresolved physical-unit hypothesis, and no conversion is applied. A reference-split CV without rotation yields primary AUROC 0.752, change −0.008 [−0.021, +0.005], and broad AUROC 0.754, change +0.014 [−0.001, +0.028] (no rotation minus original forest; `outputs/labeler/rwm/rotation_ablation.json/{metrics,paired_change}`). This sensitivity measures input dependence, not upstream timing bias.
- `dusbradial` is excluded: it is zero on most 2014 traces and flagged corrupted for 176030-176912, including all 2018 Hanson shots. The exact zero/nonzero audit is in `shots.json/input_audit`. The isolated OPERATIONS CN1BAMP/ILN1BAMP/IUN1BAMP probe succeeded on three Hanson shots, but corrected RWM-sensor semantics remain unverified; no candidate was added to model inputs (see `outputs/labeler/rwm/sensor_probe.json`).

**rules**: beta_N, beta_N/l_i, elapsed time since the first |Ip| ≥0.5 MA sample, and the existing `rwm_candidates` call. The analysis span and the candidate screen's whole-flat-top median/MAD threshold are retrospective; the screen is not a causal alarm comparator.

## Method
Hanson onset points support one minimal present slice **[o, o+10 ms)**
(category 1, `onset_point_minimal`); this does not measure mode duration.
**[o−20 ms, o)** remains uncertain (category 2). Time before the first
100 ms precursor is assumed absent, conditional on list completeness;
remaining time is unassessed. Comparisons are unlabelled, with uncertain screen
spans. Forecast labels remain separate: a 100 ms horizon, pre-last-onset
negatives, and aftermath/n=2 exclusions. Five outer and three inner shot-grouped
folds keep tuning inside training shots; five split seeds and four run-record
holdouts assess sensitivity. Offline magnetic timing and acausal ZIPFIT limit
online interpretation. Per-shot Detected/Early/Missed uses the first considered
alarm; any-alarm precedence is a labelled sensitivity.
See [protocol and reproduction](../../../docs/labeler/rwm_baseline.md) and
[evaluation.json](../../../outputs/labeler/rwm/evaluation.json) for scores and intervals.

## Alias
resistive wall mode, rwm, resonant field amplification (precursor), rfa

## Future Implementations

## Reference
- M. S. Chu and M. Okabayashi, "Stabilization of the external kink and the
  resistive wall mode", Plasma Phys. Control. Fusion 52, 123001 (2010).
- A. M. Garofalo et al., "Sustained stabilization of the resistive-wall mode by
  plasma rotation in the DIII-D tokamak", Phys. Rev. Lett. 89, 235001 (2002).

## Contact
- **Jeremy Hanson** (original dataset): hansonjm [at] fusion [dot] gat [dot] com
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu
