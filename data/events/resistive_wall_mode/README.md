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

RWMs are locked or slowly rotating, of order 1/tau_w (tens of Hz), detected
with low-frequency n = 1 magnetics such as saddle loops and poloidal probe arrays
near the no-wall limit. Rotating tearing modes on DIII-D typically sit at kHz to
tens of kHz.

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

**Category-2 reader warning:** category 2 covers both Hanson pre-onset windows
and comparison screen spans. Distinguish them only through `attrs.evidence_tier`:
`onset_window_uncertain` versus `unlabelled_screen`; category alone mixes the two.
Neither tier supplies verified negatives. Category codes are unchanged.

**Publications**:

## Models
**stable**: none

**latest**: rwm-brf

**all**:

- rwm-brf | 2026_10_03 | Phase-controlled AUROC: 0.534 [0.438, 0.624] | Primary AUROC: 0.760 [0.706, 0.809] (no established skill beyond phase)
- rwm-rule-elapsed-time | 2026_10_03 | Phase-controlled AUROC: 0.522 [0.520, 0.549] | Primary AUROC: 0.759 [0.712, 0.815]
- rwm-rule-betan | 2026_10_03 | Phase-controlled AUROC: 0.529 [0.417, 0.634] | Primary AUROC: 0.707 [0.639, 0.772]
- rwm-rule-betan-over-li | 2026_10_03 | Phase-controlled AUROC: 0.588 [0.476, 0.691] | Primary AUROC: 0.723 [0.664, 0.784]

Reference split (seed 0), Hanson primary forecasts, assumed negatives, 100 ms phase bins with at least 5 slices. Pooled primary AUROC is phase-confounded; elapsed time's residual AUROC is the phase floor. Intervals are exploratory shot-bootstrap intervals. Adding the unverified comparisons as label-noisy training negatives gives phase-controlled AUROC **0.587 [0.505, 0.660]**, a paired change of **+0.053 [-0.028, 0.127]**. It alarms on **3/132** comparison shots and warns **5/48** onsets. The paired interval includes zero; whether verified stable-shot negatives would help is untested. See [protocol and results](../../../docs/labeler/rwm_baseline.md) and [comparison sensitivity](../../../outputs/labeler/rwm/comparison_sensitivity.json).

## Inputs
**rwm-brf** (stored scalars and profiles; trailing/held features on a 10 ms grid):

- `betan`, `li`, `q95`, `qmin`, `wmhd`, `ip`, with beta_N/l_i and beta_N-4l_i derived.
- `n1rms`, `n2rms`: postprocessed magnetic RMS, trailing mean, peak and log-slope calculations; upstream timing is uncertain. N1RMS is not a direct RWM sensor.
- ZIPFIT toroidal rotation at fixed rho=0.25 (core) and rho=0.625 (mid-radius, not an identified q=2 surface); its upstream time smoothing is acausal, with timing bias unbounded here. The `TROTFIT` node reports kHz, as measured from its units field in `src/labeler/features/namespace.py:331-339`; `rot_*_khz` columns retain those units without conversion. A reference-split CV without rotation yields primary AUROC 0.752, change −0.008 [−0.021, +0.005], and broad AUROC 0.754, change +0.014 [−0.001, +0.028] (no rotation minus original forest; `outputs/labeler/rwm/rotation_ablation.json/{metrics,paired_change}`). This sensitivity measures input dependence, not upstream timing bias.
- `dusbradial` is excluded: it is zero on most 2014 traces and flagged corrupted for 176030-176912, including all 2018 Hanson shots. The exact zero/nonzero audit is in `shots.json/input_audit`. The isolated OPERATIONS CN1BAMP/ILN1BAMP/IUN1BAMP probe succeeded on three Hanson shots, but corrected RWM-sensor semantics remain unverified; no candidate was added to model inputs (see `outputs/labeler/rwm/sensor_probe.json`).

PTDATA `ONSBRADIAL`, disruption-py's reported fallback for unavailable `DUSBRADIAL`,
is the most promising next input; none is cached on the 79 zero-DUSBRADIAL 2014
shots checked, so no probe or fetch was made (`cached_sensor_audit.json`,
`LABELER_NO_FETCH=1`; fallback behavior is not independently verified here).

**rules**: beta_N, beta_N/l_i and elapsed time since the first |Ip| ≥0.5 MA sample. The `rwm_candidates` screen is a concurrent review aid; its whole-flat-top median/MAD threshold and the analysis span are retrospective.

## Method
Dataset 1 loads as point events, preserving `NTOR` and `MODE_TYPE` in `attrs`
with unknown confidence and coverage; an unlisted shot is not a negative.
The `extend_rwm/recommender_v1.csv` scan of the fixed 500 project shots is
header-only because none contains a listed Hanson onset, a completed zero rather
than an absence claim (see `src/labeler/events/rwm.py`).

The primary `rwm-brf` trains only on Hanson RWM shots: no stable discharge supplies
negatives, so it learns **when an RWM comes in a shot that has one**.
The measured comparison-negative sensitivity is reported under Models;
whether verified stable-shot negatives would help is untested.

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
The minimal **[o−20,o)/[o,o+10 ms)** physical windows are a deviation from the
brief's growth-window positives: growth duration and pre-onset presence are
unverified. Onset-derived interval `attrs.source_onsets` preserves contributing
`NTOR`, `MODE_TYPE` and original onset times, including merged duplicates.
The repository evaluation is a summary below 0.5 MB. Its `external_details`
pointer names the complete record under `$LABELER_ROOT/round4/rwm/`, including
per-shot outcomes and within-shot details; read it with
`labeler.rwm.records.load_evaluation(path, details=True)` to verify the SHA-256.
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
