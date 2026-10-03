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

**Description**: Growth-window labels (54 windows of 20 ms before the 54 distinct listed onsets, 48 of mode number n = 1 and 6 of n = 2; 84 absent spans) on the 33 Hanson shots, the screen's uncertain spans (128) on 132 matched comparison shots, and the roster of the 208 candidate shots (`selected` marks the 165 used). The comparison shots are four per Hanson shot, from the same campaign, nearest in flat-top beta_N and beta_N / l_i; they are unlabelled, not negatives. None of the 165 shots is in the corpus or the frozen cohort.

**Publications**:

## Models
**stable**: none

**latest**: rwm-brf | 2026_10_03 (a baseline, not a detector)

**all**:
- rwm-brf | 2026_10_03 | AUROC: 0.798 | AUPRC: 0.084 | F1: 0.149 (balanced random forest after Piccione et al. 2022, adapted to DIII-D)
- rwm-nnpu | 2026_10_03 | AUROC: 0.794 | AUPRC: 0.083 | F1: 0.136 (same features, non-negative positive-unlabelled risk, Kiryo et al. 2017)
- rule-betan-over-li | 2026_10_03 | AUROC: 0.752 | AUPRC: 0.074 | F1: 0.141 (beta_N / l_i read off as a score, nothing fitted)
- rwm_candidates (uncertain review screen; uncalibrated) | AUROC: 0.498 | AUPRC: 0.034 | F1: 0.000 (flags none of the 480 positive slices)

Scores are for the labelled time slices (every 10 ms, flat-top) of the 33 Hanson shots against the target "an n = 1 onset within the next 100 ms" (480 positive and 13,848 negative slices from 48 onsets; 591 slices around onsets are excluded), by 5-fold shot-grouped cross-validation, with the cutoff for F1 chosen on the training shots. 95 % shot-bootstrap intervals (1000 replicates): rwm-brf AUROC 0.798 [0.740, 0.845], AUPRC 0.084 [0.061, 0.116]; the single feature beta_N / l_i is 0.752 [0.688, 0.815]. Legacy, the published NSTX forest on its 28 test shots: AUROC 0.918, TPR 92.4 %, FPR 21.4 %. As a warning system they are not shown to beat chance: rwm-brf warns of 17 of 48 onsets, where randomly placed alarms at its alarm rate would warn of 33 %, and it raises an unexplained alarm on 21 of 33 Hanson shots. The 48 onsets (33 shots, two campaigns) cannot separate the models from each other or from beta_N / l_i. Protocol, every number and its source JSON, and caveats: [rwm_baseline.md](../../../docs/labeler/rwm_baseline.md).

## Inputs
**rwm-brf**, **rwm-nnpu** (fetched for the 208 candidate shots into the raw cache, causal 10 ms slices, `scripts/labeler/rwm_fetch.py`):
- `betan`, `li`, `q95`, `qmin`, `wmhd`, `ip` (EFIT and magnetics scalars; beta_N / l_i and beta_N - 4 l_i derived)
- `n1rms`, `n2rms` (n = 1 and n = 2 magnetic RMS: 5 ms RMS, 20 ms peak and log growth rate)
- ZIPFIT toroidal rotation at rho = 0.25 and 0.625
- `dusbradial` (locked-mode detector) is fetched but kept out of the model: it is exactly zero on every 2014 shot and nonzero on 2018 ones, so it only tags the campaign

**rwm_candidates**:
- `n1rms`, `betan`, `li`, and `ip`.
- Review rows additionally show `n2rms` where available.

## Method
No calibrated detector writes `rwm`. The two curated onset tables are the only
confirmed evidence; each
row becomes a point event with `evidence_kind = database`,
`source = database:rwm_onsets_<year>`, NaN confidence and NaN coverage - a listing
says nothing about the interval anybody examined, so a shot absent from the tables is
never a negative. `NTOR` (toroidal mode number) and `MODE_TYPE` travel in `attrs`. The
`extend_rwm/recommender_v1.csv` scan over the 500 project shots is header-only:
none of the 33 listed shots is in the corpus, and that is a completed zero, not an
absence claim.

The review editor shows the original database onsets and a conservative
`rwm_candidates` screen: an n=1 magnetic RMS excursion above the flat-top median
plus six MAD, beta_N > 4 l_i, and at least 10 ms duration. A screen candidate is
uncertain; other measured time is unassessed, and missing inputs are not
observable. N1RMS does not distinguish RWM from tearing or applied-field response.
The roster includes the database's 33 reference shots alongside the cohort.
See [the editor guide](../../../docs/labeler/equilibrium_review.md).

`extend_rwm_growth/rwm_windows.csv` is the baseline's label table and expands the
onsets, which the review tables above do not. Each onset (mode number 1 or 2) becomes a
present window of 20 ms before it, about two e-foldings of the measured n = 1 growth
(median e-folding time 9 ms; the 1 kHz RMS cannot place the start more sharply, so the
length is a convention resting on the wall time). On the 33 Hanson shots the
high-current window (Ip at least half its peak) outside 100 ms either side of every onset
is absent; the shots are taken to have been examined, which is an assumption the tables
do not state. Other shots stay unlabelled: the 132 matched comparison shots carry only
the screen's uncertain spans. The forecast target of the baselines is Piccione et al.'s:
a slice is positive when an n = 1 onset follows within 100 ms. Rebuild with
`scripts/labeler/rwm_pool.py`, `rwm_fetch.py` and `rwm_build.py`; `rwm_evaluate.py` scores
the baselines.

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
