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

**Description**: Onset-derived conventional weak windows, assumed-absent spans on Hanson shots, and unlabelled screen spans on matched comparison shots. The interval CSV's JSON `attrs.evidence_tier` distinguishes these; only Dataset 1's onset points are verified evidence. The assumed-absent spans do not establish reviewed coverage. Counts, matching, frozen-cohort checks and input audit: `outputs/labeler/rwm/shots.json`.

**Publications**:

## Models
**stable**: none

**latest**: rwm-brf | 2026_10_03 (onset-derived forecasting baseline)

**all**:

- rwm-brf | AUROC: 0.760 [0.706, 0.809] | AUPRC: 0.163 [0.123, 0.220] (Piccione-style pre-onset scoring; Hanson assumed negatives)
- rule-time-since-flattop | AUROC: 0.759 [0.712, 0.815] | AUPRC: 0.228 [0.212, 0.307] (Piccione-style pre-onset scoring; Hanson assumed negatives)
- rule-betan | AUROC: 0.707 [0.639, 0.772] | AUPRC: 0.166 [0.128, 0.246] (Piccione-style pre-onset scoring; Hanson assumed negatives)
- rule-betan-over-li | AUROC: 0.723 [0.664, 0.784] | AUPRC: 0.159 [0.127, 0.232] (Piccione-style pre-onset scoring; Hanson assumed negatives)
- rule-rwm-candidates | AUROC: 0.500 [0.500, 0.500] | AUPRC: 0.084 [0.070, 0.102] (Piccione-style pre-onset scoring; Hanson assumed negatives)

The primary rwm-brf trains on Hanson-shot positives and Hanson-shot assumed negatives only. Comparison slices remain unlabelled. The forecast target is a listed n=1 onset within 100 ms; primary negative scoring stops at the last n=1 onset. The broader Hanson-negative sensitivity uses the same predictions: AUROC 0.740 [0.668, 0.799], AUPRC 0.065 [0.043, 0.103]. The forest mainly separates discharge phases: high-beta conditional AUROC 0.602 [0.491, 0.688], above-no-wall-proxy conditional AUROC 0.546 [0.416, 0.642]; within-phase discrimination is weak and no AUROC gain over elapsed time or the beta rules is established. Sources: `outputs/labeler/rwm/evaluation.json/configs.<model>.metrics` and `paired.rwm-brf - <rule>`. CIs resample shots; split sensitivity is separate. Alarm incidence on comparison shots is not a false positive rate. nnPU is excluded from formal results because its earlier development used an outer fold and its U prior was unidentified. Full scope, alarm results, conditional scores and paper outputs: [rwm_baseline.md](../../../docs/labeler/rwm_baseline.md).

## Inputs
**rwm-brf** (stored scalars and profiles; trailing/held features on a 10 ms grid):

- `betan`, `li`, `q95`, `qmin`, `wmhd`, `ip`, with beta_N/l_i and beta_N-4l_i derived.
- `n1rms`, `n2rms`: magnetic RMS, trailing mean, peak and log-slope features. N1RMS is not a direct RWM sensor.
- ZIPFIT toroidal rotation at the configured radii; its upstream time smoothing is mildly acausal.
- `dusbradial` is excluded: it is zero on most 2014 traces and flagged corrupted for 176030-176912, including all 2018 Hanson shots. The exact zero/nonzero audit is in `shots.json/input_audit`. No validated low-frequency n=1 saddle-loop/Bp-amplitude locator was available through approved `FETCH_SPECS` or the feature namespace.

**rules**: beta_N, beta_N/l_i, causal elapsed time since a fixed 0.5 MA current crossing, and the existing `rwm_candidates` call. The analysis span and the candidate screen's whole-flat-top median/MAD threshold are retrospective; the screen is not a causal alarm comparator.

## Method
Only the curated onset points are confirmed evidence. Original database events retain NaN confidence and NaN coverage; onset listing does not establish an examined interval. The original 500-shot screen export is header-only, not an absence claim.

The baseline interval table expands each onset into a **conventional weak** 20 ms pre-onset window motivated by tau_w, not a measured growth time. The same maximum-slope search at random flat-top centres gives comparable N1RMS maxima (see `outputs/labeler/rwm/growth.json`), so it cannot validate the interval's physical extent. Hanson time away from listed onsets is **assumed absent**, conditional on list completeness. Comparison shots remain **unlabelled**, with only uncertain screen spans, never primary training negatives.

Forecasting uses the paper-inspired 100 ms horizon and only pre-last-n=1-onset assumed negatives; immediate aftermath and n=2 surroundings are excluded, and n=2-only shots have no primary n=1 negatives. Five outer and three inner shot-grouped folds keep thresholds and alarm choices inside training shots. Alarm tuning targets n=1 separately from n=1/n=2 events used to explain alarms. Rebuild and score with `scripts/labeler/rwm_build.py`, `rwm_growth.py`, `rwm_evaluate.py`, `rwm_tables.py` and `rwm_figure.py`, following the stream environment rules. Protocol, every result and paired CIs are in `outputs/labeler/rwm/evaluation.json`.

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
