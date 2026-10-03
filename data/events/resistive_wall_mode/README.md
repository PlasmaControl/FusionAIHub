# Resistive Wall Mode

## Description
The resistive wall mode (RWM) is the external kink (mostly n = 1) that a perfectly
conducting wall would stabilise but a real, resistive wall only slows: the mode
grows on the wall's flux-penetration time tau_w (milliseconds on DIII-D) instead of
the Alfvén time. It exists in the window between the no-wall and ideal-wall beta
limits,

    beta_N^no-wall  <  beta_N  <  beta_N^ideal-wall

(empirically beta_N^no-wall ~ 4 l_i on DIII-D), and is stabilised by plasma rotation
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

**Dataset File(s)**: `extend_rwm_growth/rwm_windows.csv`, `rwm_windows.meta.json`, `rwm_windows.shots.csv`

**Author**: built here from Dataset 1 and DIII-D inputs fetched for 208 candidate shots (raw cache `$LABELER_ROOT/raw`).

**Description**: Onset-derived conventional weak windows, assumed-absent spans before the first Hanson precursor, explicit category 4 unassessed precursor/post-onset time, and unlabelled screen spans on matched comparison shots. The review reader preserves the interval CSV's JSON `attrs.evidence_tier`; only Dataset 1's onset points are verified evidence. No post-onset physical absence is inferred without termination evidence. Counts, matching, frozen-cohort checks and input audit: `outputs/labeler/rwm/shots.json`.

**Publications**:

## Models
**stable**: none

**latest**: rwm-brf | 2026_10_03 (onset-derived forecasting baseline)

**all** (reference split 0; five-split headline ranges below):

- rwm-brf | 2026_10_03 | AUROC: 0.760 [0.706, 0.809] | AUPRC: 0.163 [0.123, 0.220] | F1: 0.275 [0.227, 0.337] | bin set: Hanson primary, 100 ms forecast / 10 ms slices (pre-last-n=1-onset mask; assumed negatives)
- rule-time-since-flattop | 2026_10_03 | AUROC: 0.759 [0.712, 0.815] | AUPRC: 0.228 [0.212, 0.307] | F1: 0.267 [0.217, 0.331] | bin set: Hanson primary, 100 ms forecast / 10 ms slices (pre-last-n=1-onset mask; assumed negatives)
- rule-betan | 2026_10_03 | AUROC: 0.707 [0.639, 0.772] | AUPRC: 0.166 [0.128, 0.246] | F1: 0.246 [0.194, 0.312] | bin set: Hanson primary, 100 ms forecast / 10 ms slices (pre-last-n=1-onset mask; assumed negatives)
- rule-betan-over-li | 2026_10_03 | AUROC: 0.723 [0.664, 0.784] | AUPRC: 0.159 [0.127, 0.232] | F1: 0.261 [0.212, 0.328] | bin set: Hanson primary, 100 ms forecast / 10 ms slices (pre-last-n=1-onset mask; assumed negatives)
- rule-rwm-candidates | 2026_10_03 | AUROC: 0.500 [0.500, 0.500] | AUPRC: 0.084 [0.070, 0.102] | F1: 0.000 [0.000, 0.000] | bin set: Hanson primary, 100 ms forecast / 10 ms slices (pre-last-n=1-onset mask; assumed negatives)

The primary rwm-brf trains on Hanson positives and assumed negatives; comparison slices remain unlabelled. Its 100 ms future-onset target is separate from the physical catalog's unassessed post-onset state. Across **five shot-fold splits (seeds 0–4)**, pooled AUROC is **0.760–0.806** and **high-beta conditional AUROC is 0.602–0.696 (about 0.60–0.70)**. High-beta AUROC is about chance (**0.311–0.527**) in 2014 and **0.698–0.747 in 2018**; campaign primary AUROC ranges are 0.663–0.735 and 0.819–0.848 respectively. Above-proxy conditional AUROC ranges are 0.546–0.657 pooled, 0.277–0.508 in 2014 and 0.647–0.706 in 2018. Ranges are point-estimate variation across splits, not CIs.

The forest is **not distinguishable from the elapsed-time rule in AUROC in any of the three pooled strata**: paired forest-minus-time point differences across seeds range 0.001–0.047 (primary), −0.040 to +0.054 (high-beta) and −0.051 to +0.061 (above-proxy); every paired 95% interval includes zero. These conditional strata retain discharge-phase information. **Leave-one-run-record-out** has pooled AUROC 0.779 [0.731, 0.823], AUPRC 0.181 [0.149, 0.248], F1 0.285 [0.233, 0.344], high-beta AUROC 0.621 [0.536, 0.700] and above-proxy AUROC 0.579 [0.475, 0.677]. Its campaign high-beta scores are 0.441 [0.369, 0.582] in 2014 and 0.661 [0.576, 0.746] in 2018. The four records remain separate: 20180314 and 20180314A share a date, so this does not claim calendar-day isolation or new-run-population performance. The broader negative-mask sensitivity uses identical split-0 scores: AUROC 0.740 [0.668, 0.799], AUPRC 0.065 [0.043, 0.103].

Primary split-0 alarms are tuned within training folds: **9/48** onsets warned; per-shot **6 Detected / 22 Missed / 2 Early** among 30 n=1 target shots. Detection **0.188 [0.049, 0.333]** equals the rate-matched random-alarm reference **0.190** within uncertainty (difference **−0.002 [−0.058, 0.052]**): no better than randomly placed alarms at the same rate under this approximate reference. Median warning **356 ms [286, 389]** is near the 400 ms acceptance limit, consistent with phase tracking.

Comparison FP is alarm incidence **19/132**, 0.144 [0.083, 0.212], without verified stable-shot ground truth. The 2014 matched beta_N/l_i p95 mean is **4.43 versus 5.07 for Hanson** (unchosen pool 4.65), biasing this incidence low for a beta-tracking model. Comparison alarm traces retain their **full span**, while Hanson traces end at the last n=1/n=2 onset +100 ms; exposure is asymmetric. The separately retuned full-trace sensitivity retains 1/48 detections and 1/132 comparison incidence. The retrospective screen never fires before an onset on these Hanson traces, is zero on primary slices, and has no paired forest-minus-screen row. Elapsed time has higher point AUPRC, but the paired **basic shot-bootstrap** interval includes zero (forest minus time −0.065 [−0.100, 0.016]); no AUPRC advantage is established.

**Legacy — published NSTX RUS forest, Piccione et al. (2022):** AUROC **0.918**, TPR **92.4%**, FPR **21.4%**, **10/11 detected**, **2/17 FP**; **different machine, expert-reviewed stable shots, not comparable**. Different NSTX inputs and training/test validation; F1/intervals unavailable in source digest. Source: `outputs/labeler/rwm/evaluation.json/legacy`, transcribed by the evaluation script from the Piccione digest in the main checkout `.tmp/label_papers/` (doi:10.1088/1741-4326/ac44af).

Source for model, campaign and alarm cells: `outputs/labeler/rwm/evaluation.json/configs.<model>.{metrics,counts,by_campaign,split_seeds,full_trace_alarm_sensitivity}`, `split_sensitivity.{auroc_ranges,paired_time_ranges,paired_time_by_seed}`, `leave_one_run_record_out`, `paired`, and `screen_audit`. Comparison balance: `outputs/labeler/rwm/shots.json/comparison.balance.2014`. Individual/campaign CIs use 1,000 percentile shot resamples at fixed predictions; paired CIs use the basic method. They do not include refitting or fold-selection uncertainty. nnPU is excluded because earlier development used an outer fold and its U prior was unidentified. Full protocol, all score/alarm tables, and paper outputs: [rwm_baseline.md](../../../docs/labeler/rwm_baseline.md).

## Inputs
**rwm-brf** (stored scalars and profiles; trailing/held features on a 10 ms grid):

- `betan`, `li`, `q95`, `qmin`, `wmhd`, `ip`, with beta_N/l_i and beta_N-4l_i derived.
- `n1rms`, `n2rms`: magnetic RMS, trailing mean, peak and log-slope features. N1RMS is not a direct RWM sensor.
- ZIPFIT toroidal rotation at the configured radii; its upstream time smoothing is mildly acausal.
- `dusbradial` is excluded: it is zero on most 2014 traces and flagged corrupted for 176030-176912, including all 2018 Hanson shots. The exact zero/nonzero audit is in `shots.json/input_audit`. No validated low-frequency n=1 saddle-loop/Bp-amplitude locator was available through approved `FETCH_SPECS` or the feature namespace.

**rules**: beta_N, beta_N/l_i, causal elapsed time since a fixed 0.5 MA current crossing, and the existing `rwm_candidates` call. The analysis span and the candidate screen's whole-flat-top median/MAD threshold are retrospective; the screen is not a causal alarm comparator.

## Method
Only the curated onset points are confirmed evidence. Original database events retain NaN confidence and NaN coverage; onset listing does not establish an examined interval. The original 500-shot screen export is header-only, not an absence claim.

The review editor's `rwm_candidates` screen requires n=1 RMS above the whole-flat-top median plus **6 MAD**, **beta_N > 4 l_i**, and **at least 10 ms** duration. Screen candidates are uncertain, other time is unassessed, and missing inputs are not observable. N1RMS cannot distinguish RWM from tearing or applied-field response. See [the editor guide](../../../docs/labeler/equilibrium_review.md).

The baseline interval table expands each onset into a **conventional weak** category-1 window [o−20 ms, o] (half-open in CSV/readers), motivated by tau_w, not a measured growth time. This is a prescribed pre-onset convention: a point label supplies no post-onset duration, so the few milliseconds immediately after onset are **category 4** pending extent/termination evidence. The same maximum-slope search at random flat-top centres gives comparable N1RMS maxima (see `outputs/labeler/rwm/growth.json`), so it cannot validate physical extent. Only time before the first onset's 100 ms precursor is **assumed absent**, conditional on list completeness. Precursors and all post-onset physical time are explicitly **unassessed (category 4)**, apart from later weak windows; no recovery/termination evidence establishes physical absence. Comparison shots remain **unlabelled**, with uncertain screen spans and category 4 elsewhere, never primary training negatives.

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
