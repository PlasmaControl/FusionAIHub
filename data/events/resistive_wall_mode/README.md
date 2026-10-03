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

Typically found via low-frequency (0-10 kHz) n = 1 magnetics - saddle loops and
poloidal probe arrays - as a slowly growing, slowly rotating or locked mode near the
no-wall limit.

## Data Provenance
### Dataset 1

**Dataset File(s)**: `rwm_onsets_2017.csv`, `rwm_onsets_2024.csv`

**Author**: Jeremy Hanson

**Description**: Reference dataset from Jeremy Hanson: `rwm_onsets_2017.csv` (20 shots, 156785-158023) and `rwm_onsets_2024.csv` (13 shots, 176067-176092), 56 onsets over 33 distinct shots in `raw/`.

**Publications**:

### Dataset 2

**Dataset File(s)**: `extend_rwm_onset_window/rwm_windows.csv`, `rwm_windows.meta.json`, `rwm_windows.shots.csv`

**Author**: built here from Dataset 1 and DIII-D inputs fetched for 208 candidate shots (raw cache `$LABELER_ROOT/raw`).

**Description**: Onset-derived uncertain windows (ONSET_TIME detection meaning unconfirmed), assumed-absent spans before the first Hanson precursor, explicit category 4 unassessed precursor/post-onset time, and unlabelled screen spans on matched comparison shots. The review reader preserves the interval CSV's JSON `attrs.evidence_tier`; only Dataset 1's onset points are verified evidence. No post-onset physical absence is inferred without termination evidence. Counts, matching, frozen-cohort checks and input audit: `outputs/labeler/rwm/shots.json`.

**Publications**:

## Models
**stable**: none

**latest**: rwm-brf | 2026_10_03 (onset-derived forecasting baseline)

**all** (reference split (seed 0); Hanson primary 100 ms forecasts on 10 ms slices; assumed negatives):

- rwm-brf | 2026_10_03 | AUROC: 0.760 [0.706, 0.809] | AUPRC: 0.163 [0.123, 0.220] | F1: 0.275 [0.227, 0.337]
- rule-elapsed-time | 2026_10_03 | AUROC: 0.759 [0.712, 0.815] | AUPRC: 0.228 [0.212, 0.307] | F1: 0.267 [0.217, 0.331]
- rule-betan | 2026_10_03 | AUROC: 0.707 [0.639, 0.772] | AUPRC: 0.166 [0.128, 0.246] | F1: 0.246 [0.194, 0.312]
- rule-betan-over-li | 2026_10_03 | AUROC: 0.723 [0.664, 0.784] | AUPRC: 0.159 [0.127, 0.232] | F1: 0.261 [0.212, 0.328]
- rule-rwm-candidates | 2026_10_03 | AUROC: 0.500 [0.500, 0.500] | AUPRC: 0.084 [0.070, 0.102] | F1: 0.000 [0.000, 0.000]

Primary negatives end at the last n=1 onset, so elapsed time ranks within-shot almost perfectly (median AUROC **1.0**, mean **0.93**, **30** two-class shots; forest median **0.83**). Broad-mask forest minus elapsed time is **+0.350 [0.287, 0.414]** (run-record holdout **+0.362 [0.304, 0.428]**); forest minus beta_N/l_i is **−0.013 [−0.092, 0.072]**. **The forest matches the best single scalar under either mask (elapsed time on primary, beta_N/l_i on broad); no onset-specific skill.** Sources: `evaluation.json/configs/<model>/within_shot_auroc`, `paired/rwm-brf - <rule>/broad_auroc`, `leave_one_run_record_out/paired_time/broad_auroc`.

The Piccione-style forest on generic 0D inputs has five-split high-beta AUROC 0.602–0.696 pooled, **about chance or below in 2014 (0.31–0.53; split-0 CI [0.22, 0.41]) and below elapsed time on every split (point estimates; CI excludes zero on 2 of 5 seeds)**, versus 0.698–0.747 in 2018; **7/26** 2014 targets versus **1/22** in 2018 have no high-beta slice in their 100 ms forecast window (`evaluation.json/onset_physics/by_campaign`). Forest detection ranges **0.125–0.271**, detection minus the rate-matched random reference **−0.002 to +0.057**, and median warning **135–356 ms** across five splits: **no improvement over the approximate random reference was established (all five CIs include 0)**.

Primary forest-minus-elapsed-time differences span **0.001–0.047**; seed 3 is borderline (**0.047 [−0.0001, 0.098]**). Run-record holdout detection is **0.167**, reference difference **0.014 [−0.025, 0.053]**, median warning **214 ms**; the reference-split beta_N rule exceeds its reference by **0.058 [0.014, 0.115]** on **4/48** warned onsets. All results and the noncomparable NSTX Legacy reference, exact source pointers, onset physics, offline input and coverage caveats are in [the current protocol](../../../docs/labeler/rwm_baseline.md) and [evaluation.json](../../../outputs/labeler/rwm/evaluation.json).

## Inputs
**rwm-brf** (stored scalars and profiles; trailing/held features on a 10 ms grid):

- `betan`, `li`, `q95`, `qmin`, `wmhd`, `ip`, with beta_N/l_i and beta_N-4l_i derived.
- `n1rms`, `n2rms`: postprocessed magnetic RMS, trailing mean, peak and log-slope calculations; upstream timing is uncertain. N1RMS is not a direct RWM sensor.
- ZIPFIT toroidal rotation at fixed rho=0.25 (core) and rho=0.625 (mid-radius, not an identified q=2 surface); its upstream time smoothing is acausal, with timing bias unbounded here. The inherited `rot_*_khz` columns have values matching krad/s rather than kHz; units remain unresolved. No rotation ablation is needed because no skill or rotation benefit is claimed.
- `dusbradial` is excluded: it is zero on most 2014 traces and flagged corrupted for 176030-176912, including all 2018 Hanson shots. The exact zero/nonzero audit is in `shots.json/input_audit`. The isolated OPERATIONS CN1BAMP/ILN1BAMP/IUN1BAMP probe succeeded on three Hanson shots, but corrected RWM-sensor semantics remain unverified; no candidate was added to model inputs (see `outputs/labeler/rwm/sensor_probe.json`).

**rules**: beta_N, beta_N/l_i, elapsed time since the first |Ip| ≥0.5 MA sample, and the existing `rwm_candidates` call. The analysis span and the candidate screen's whole-flat-top median/MAD threshold are retrospective; the screen is not a causal alarm comparator.

## Method
Only the curated onset points are confirmed evidence. Original database events retain NaN confidence and NaN coverage; onset listing does not establish an examined interval. The original 500-shot screen export is header-only, not an absence claim.

The review editor's `rwm_candidates` screen requires n=1 RMS above the whole-flat-top median plus **6 MAD**, **beta_N > 4 l_i**, and **at least 10 ms** duration. Screen candidates are uncertain, other time is unassessed, and missing inputs are not observable. N1RMS cannot distinguish RWM from tearing or applied-field response. See [the editor guide](../../../docs/labeler/equilibrium_review.md).

The supplied Hanson CSVs/README do not define `ONSET_TIME` as growth-start versus detection/threshold crossing; the local Piccione digests concern NSTX and cannot confirm the DIII-D convention (searched sources: `rwm_windows.meta.json/onset_time_provenance`, `sensor_probe.json/source_search`). The baseline interval table therefore writes **[o−20 ms, o) as UNCERTAIN (category 2)**, with zero present duration rows: neither direction nor extent is measured. Immediately post-onset time stays **category 4**, pending extent/termination evidence; the matched random-time N1RMS slope search cannot repair this gap. Time before the first onset's 100 ms precursor is **assumed absent**, conditional on list completeness. Precursors and all post-onset physical time are explicitly **unassessed (category 4)** apart from later uncertain windows; comparisons remain unlabelled, with uncertain screen spans and category 4 elsewhere. These physical categories are separate from forecast labels.

All features are trailing calculations on offline inputs: holding samples does not establish causal availability of postprocessed N1RMS/N2RMS or acausal ZIPFIT. [Onset-physics tables](../../../outputs/labeler/rwm/tables.md) give beta_N, l_i, beta_N/l_i and elapsed time at every merged n=1 onset, and separately at the first slice in its uncertain 20 ms window; six below-proxy and three missing-EFIT flags apply to the latter snapshot, not exact onset.

Forecasting uses the paper-inspired 100 ms horizon and only pre-last-n=1-onset assumed negatives; immediate aftermath and n=2 surroundings are excluded, and n=2-only shots have no primary n=1 negatives. This future-onset target is separate from physical state. Five outer and three inner shot-grouped folds keep thresholds and alarm choices inside training shots, repeated over five seeds; leave-one-run-record-out is a separate sensitivity. Primary Hanson alarm tuning/scoring ends at the last n=1/n=2 onset +100 ms, while comparison traces retain full span; the former full-trace objective is separately retuned. Alarm tuning targets n=1 separately from n=1/n=2 events used to explain alarms. Rebuild and score with `scripts/labeler/rwm_build.py`, `rwm_growth.py`, `rwm_evaluate.py`, `rwm_tables.py` and `rwm_figure.py`, following the stream environment rules. This round uses `rwm_evaluate.py --rescore-saved --workers 5 --replicates 1000` to replay existing predictions/rules without refitting or fetching. Protocol, every result and paired CIs are in `outputs/labeler/rwm/evaluation.json`.

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
