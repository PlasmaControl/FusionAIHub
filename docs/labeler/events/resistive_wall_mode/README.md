# Resistive Wall Mode

> The label data is `data/events/resistive_wall_mode/`: the `raw/`, `format/`, `review/` and `extend_*/` paths
> below are relative to it. It is not in git. `scripts/labeler/hf_publish.py pull`
> restores the label tables (`format/`, `review/`, `extend_*/`), but not `raw/`: the
> third-party originals are not in the Hugging Face release; they are on the Stellar
> checkout's disk and archived in `/scratch/gpfs/EKOLEMEN/nc1514/labeler/raw_archive/`.

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

**Description**: One minimal category-1 present slice [o, o+10 ms) per onset records presence without measuring duration; category-2 pre-onset windows remain uncertain (ONSET_TIME detection meaning unconfirmed). The export also contains assumed-absent spans before the first Hanson precursor, explicit category 4 unassessed precursor/post-onset time, and unlabelled screen spans on matched comparison shots. The review reader preserves the interval CSV's JSON `attrs.evidence_tier`; only Dataset 1's onset points are verified evidence. A βN collapse (at least 40% within 100 ms from βN of at least 2) inside an assumed-absent span is flagged in `attrs.unexplained_beta_collapse` (4 of 33 Hanson shots; beam power is not cached for them, so the span keeps category 0). No post-onset physical absence is inferred without termination evidence. Counts, matching, frozen-cohort checks and input audit: `outputs/labeler/rwm/shots.json`.

**Category-2 reader warning:** category 2 covers both Hanson pre-onset windows
and comparison screen spans. Distinguish them only through `attrs.evidence_tier`:
`onset_window_uncertain` versus `unlabelled_screen`; category alone mixes the two.
Neither tier supplies verified negatives. Category codes are unchanged.

**Review note:** the shared review reader's `save()` refuses sources whose shots
carry evidence attributes, so pointing `review/source.json` at `rwm_windows.csv`
loads the labels but cannot save edits to them; expert review of these labels
needs a tier-preserving save path first.

**Publications**:

## Models
**stable**: none

**latest**: rwm-ours (RWM presence per 10 ms bin, `docs/labeler/results/rwm_classifier.json`)

**all**:

- rwm-ours | 2026_10_14 | F1: 0.529 [0.432, 0.609] | AUROC: 0.967 [0.946, 0.986] | AUPRC: 0.523 [0.403, 0.644] | Limitation: a presence classifier at t (the balanced forest below on the v5 inputs at t, δBp included, no gap) against `rwm_onsets` v5 on Hanson's 33 shots: 93 present and 14,082 negative 10 ms bins (prevalence 0.0066: always-present F1 0.013, AUPRC 0.0066), uncertain time left out, one calendar run day held out at a time, the F1 cut chosen on the training days. It reads the labels' own sensor (the present spans end on δBp), so the score is partly agreement with the v5 rule, and the negatives are assumed (Hanson's lists taken as complete). With uncertain time scored negative: F1 0.260 [0.189, 0.339], AUROC 0.954 [0.930, 0.975], AUPRC 0.228 [0.136, 0.357]
- rwm-brf | 2026_10_14 | F1: 0.034 [0.014, 0.060] | AUROC: 0.711 [0.661, 0.760] | AUPRC: 0.014 [0.009, 0.021] | Limitation: the NSTX forest's recipe (Piccione 2022: balanced random forest, 300 trees, depth 8) on the round-four inputs (EFIT01 scalars, N1RMS and N2RMS, ZIPFIT rotation; no δBp), retrained as a presence classifier on the same bins and folds. rwm-ours minus rwm-brf (paired shot bootstrap): F1 +0.495 [+0.403, +0.577], AUROC +0.257 [+0.200, +0.315], AUPRC +0.509 [+0.392, +0.625], so δBp is what detects the mode. The NSTX forest itself reported slice AUROC 0.918 for a 100 ms forecast on NSTX, another machine, and no F1 or AUPRC

Both are scored on Hanson's shots only. The rerun on the table rebuilt once the roster shots from 2021 on were fetched (`docs/labeler/results/rwm_classifier_roster.json`) gives the same scores, because Hanson's rows did not change: the roster adds no present time and no tested absence. Since 2026-10-14 every RWM model is a presence classifier (owner: "everything is just classification"). The v5 forecaster run that morning was dropped, with its 20 ms onset gap, gap-aware alarms, 100 ms horizon and phase-controlled AUROC. Round four's forecasters follow as a record (a 100 ms horizon on the same Hanson shots, `outputs/labeler/rwm/evaluation.json`):

- rwm-brf (round four) | 2026_10_03 | Phase-controlled AUROC: 0.534 [0.436, 0.625] | AUROC: 0.760 [0.706, 0.809] (primary; phase-confounded) | AUPRC: 0.163 [0.123, 0.220] | F1: 0.275 [0.227, 0.337] | Limitation: no demonstrated skill beyond elapsed time or βN/li under phase control; phase-controlled margin over elapsed time +0.012 to +0.107 over 5 seeds (interval excluding zero on 1/5); within shot, forest minus elapsed time, βN and βN/li −0.147, −0.031 and −0.071, a ranking dominated by phase (elapsed time 0.931); run-record holdout 2018 −0.049, 2014 +0.019; equilibrium-scalar timing baseline, no input senses the RWM
- rwm-rule-elapsed-time | 2026_10_03 | Phase-controlled AUROC: 0.522 [0.511, 0.536] | AUROC: 0.759 [0.712, 0.815] (primary; phase-confounded) | AUPRC: 0.228 [0.212, 0.307] | F1: 0.267 [0.217, 0.331] | Limitation: its phase-controlled AUROC is the residual-phase floor
- rwm-rule-betan | 2026_10_03 | Phase-controlled AUROC: 0.529 [0.410, 0.638] | AUROC: 0.707 [0.639, 0.772] (primary; phase-confounded) | AUPRC: 0.166 [0.128, 0.246] | F1: 0.246 [0.194, 0.312] | Limitation: single-scalar rule, not an RWM sensor
- rwm-rule-betan-over-li | 2026_10_03 | Phase-controlled AUROC: 0.588 [0.471, 0.690] | AUROC: 0.723 [0.664, 0.784] (primary; phase-confounded) | AUPRC: 0.159 [0.127, 0.232] | F1: 0.261 [0.212, 0.328] | Limitation: single-scalar rule, not an RWM sensor
- rwm-rule-rwm-candidates | 2026_10_03 | Phase-controlled AUROC: 0.500 [0.500, 0.500] | AUROC: 0.500 [0.500, 0.500] (primary; phase-confounded) | AUPRC: 0.084 [0.070, 0.102] | F1: 0.000 [0.000, 0.000] | Limitation: retrospective review-editor screen (n=1 RMS above the flat-top median plus 6 MAD, βN > 4 li, at least 10 ms) with 0 calls on primary Hanson slices, so its AUROC is the no-information value; warns 0/48 onsets and alarms on 36/132 (27%) unlabelled comparison shots; a review aid, not a forecast comparator
- rwm-brf-onsb | 2026_10_14 | Phase-controlled AUROC: 0.529 [0.433, 0.626] | AUROC: 0.760 [0.703, 0.811] (primary; phase-confounded) | AUPRC: 0.158 [0.123, 0.208] | F1: 0.285 [0.230, 0.357] | Limitation: rwm-brf plus four ONSBRADIAL columns, which exist on the 2018 shots only (2014 imputed); paired change in phase-controlled AUROC -0.005 [-0.035, 0.020], slice AUROC -0.000 [-0.010, 0.010], F1 0.010 [-0.014, 0.027]: the sensor adds no measurable forecast skill
- rwm-det-brf | 2026_10_14 | Detection AUROC: 0.798 [0.740, 0.849] | AUPRC: 0.049 [0.034, 0.078] (positive base rate 0.015) | Limitation: detection benchmark, onset ±20 ms on held-out Hanson slices (the reference folds, no tuning); single columns: N1RMS 20 ms maximum 0.580, elapsed time 0.402; the ranking may draw on post-onset βN and N1RMS changes, none RWM-specific
- rwm-det-brf-onsb | 2026_10_14 | Detection AUROC: 0.801 [0.745, 0.851] | AUPRC: 0.047 [0.033, 0.073] (positive base rate 0.015) | Limitation: rwm-det-brf plus the ONSBRADIAL columns; difference 0.003 [-0.007, 0.013]; ONSBRADIAL's 20 ms maximum alone 0.575 on the 2018 shots

Round four's forecaster showed no skill beyond elapsed time or βN/li under phase control; it was an equilibrium-scalar timing baseline and no input sensed the RWM. Phase-controlled AUROC, seed 0 (the forest's weakest split): forest 0.534 [0.436, 0.625], elapsed time 0.522, βN/li 0.588. Over 5 seeds the forest's margin over elapsed time is +0.012 to +0.107, positive on 5/5 seeds and the holdout, with an interval excluding zero on 1/5. Within shot, forest minus elapsed time, βN and βN/li is −0.147 [−0.200, −0.084], −0.031 [−0.109, 0.045] and −0.071 [−0.133, −0.010]; primary-mask within-shot ranking is dominated by phase (elapsed time 0.931). Run-record holdout, forest minus elapsed time (primary AUROC): 2018 −0.049 [−0.073, −0.013], 2014 +0.019 [−0.052, 0.096], pooled +0.020 [−0.033, 0.080].

Round four: reference split (seed 0), Hanson primary forecasts, assumed negatives, 100 ms phase bins with at least 5 slices; intervals are exploratory 95% shot-bootstrap intervals. Definitions, the split sensitivities and the post-hoc bin choice are in [protocol and results](../../rwm_baseline.md).

## Inputs
**rwm-brf** (the round-four inputs, which round four's forecaster also read; stored scalars and profiles; trailing/held features on a 10 ms grid):

- `betan`, `li`, `q95`, `qmin`, `wmhd`, `ip`, with beta_N/l_i and beta_N-4l_i derived.
- `n1rms`: postprocessed magnetic RMS, with trailing mean, peak and log slope; `n2rms`: trailing mean only. Upstream timing is uncertain, and N1RMS is not a direct RWM sensor.
- ZIPFIT toroidal rotation at fixed rho=0.25 (core) and rho=0.625 (mid-radius, not an identified q=2 surface); its upstream time smoothing is acausal, with timing bias unbounded here. The `TROTFIT` units field says kHz, but the core magnitudes (median 75, maximum 174) match krad/s, the ZIPFIT convention; kHz would imply supersonic toroidal velocity. Legacy `rot_*_khz` names retain raw values without conversion and do not establish physical units; the forest is invariant to a positive constant unit conversion (`outputs/labeler/rwm/evaluation.json#/forecast_label_audit/rotation_raw`). A reference-split CV without rotation yields primary AUROC 0.752, change −0.008 [−0.021, +0.005], and broad AUROC 0.754, change +0.014 [−0.001, +0.028] (no rotation minus original forest; `outputs/labeler/rwm/rotation_ablation.json/{metrics,paired_change}`). This sensitivity measures input dependence, not upstream timing bias.
- `dusbradial` is excluded: it is zero on most 2014 traces and flagged corrupted for 176030-176912, including all 2018 Hanson shots. The exact zero/nonzero audit is in `shots.json/input_audit`. The isolated OPERATIONS CN1BAMP/ILN1BAMP/IUN1BAMP probe succeeded on three Hanson shots, but these are applied-field amplitudes (about 12-14 G I-coil flat-top medians on the 2014 shots, a roughly 19.6 G C-coil plateau on the 2018 shot; `outputs/labeler/rwm/data_audit.json`), not a plasma response, and corrected RWM-sensor semantics remain unverified; no candidate was added to model inputs (see `outputs/labeler/rwm/sensor_probe.json`).

**No round-four input senses the RWM** (rwm-ours adds the δBp sensor, below). N1RMS shows no detectably larger maximum slope at onsets (median maximum log slope 111.7/s at the 48 n=1 onsets versus 111.7/s at 2,675 flat-top controls before each shot's first onset; rank AUC 0.545, Mann-Whitney p 0.28; most of the 7,704 controls fall after the first onset, so only the pre-onset ones are compared, `outputs/labeler/rwm/growth.json#/slope_rank_test_n1`), and `dusbradial` is unusable, so `rwm-brf` can only learn the beta_N/l_i-and-time trajectory of a Hanson shot. The near-flat 2018 C-coil n=1 plateau points to a pre-programmed or static applied field (for example error-field correction) rather than active feedback; this is a physical reason to stratify every result by campaign. Round four found no cached PTDATA `ONSBRADIAL`, disruption-py's reported fallback for unavailable `DUSBRADIAL`, on the 79 zero-DUSBRADIAL 2014 shots (`cached_sensor_audit.json`).

**ONSBRADIAL (round five).** `scripts/labeler/rwm_sensor_fetch.py` fetched it for the 207 study shots: all 85 2018 shots (13 Hanson, 72 comparison) return a 0.25 ms record in native units, and none of the 122 2014 shots does. On every 2014 shot fdp fails with a PTSERVER lookup error. An invented point name on a 2018 shot fails the same way: the PTDATA client falls back to a live PTSERVER, which stellar cannot reach, whenever the archive's pointname index does not list a point. The index fdp reads (its newest version, `json_indexes_2026-06-23`) holds every 2014 shot, with 11,874 to 14,938 points each, and lists `DUSBRADIAL` but not `ONSBRADIAL` on all 122. On 156785 it lists no ONS* point at all. Each shot's `<shot>.missing.json` records the check, so ONSBRADIAL is absent from the archive fdp reads for 2014. Its semantics and calibration are not independently verified. The separation probe (`scripts/labeler/rwm_sensor_separation.py`, `outputs/labeler/rwm/sensor_separation.json`) defines an onset jump as ln of the largest 5 ms mean of |ONSBRADIAL| in [o, o+20 ms] over the median 5 ms mean in [o−120, o−20 ms]. Against 84 matched comparison times on 47 shots, the jump at 22 merged n=1 onsets on 12 shots has AUC 0.659 [0.545, 0.777], just above the 0.65 bar fixed before the run. On the same onsets and controls, N1RMS gives 0.755 [0.565, 0.918], and against flat-top times on the same Hanson shots ONSBRADIAL gives 0.619 [0.497, 0.753]. ONSBRADIAL therefore separates no better than N1RMS. Adding four ONSBRADIAL columns to the forest (`rwm-brf-onsb`, `scripts/labeler/rwm_sensor_model.py`, `outputs/labeler/rwm/sensor_model.json`) changes phase-controlled AUROC by −0.005 [−0.035, 0.020], and onset ±20 ms detection AUROC by 0.003 [−0.007, 0.013] (Models).

**rwm-ours** adds to those, on the same 10 ms grid (`labeler.rwm.features.BP_FEATURES`, built by `scripts/labeler/rwm_v5_build.py` into `$LABELER_ROOT/round5/rwm/slices_v5.parquet`):

- `bp_n1_g`, `bp_n1_max_g`, `bp_n1_growth_per_s`: the n=1 δBp amplitude of the midplane probe differences MPID067U/097U/127U/157U (`labeler.rwm.bp`, the sensor under Method) with the causal 100 ms trailing median and 2 ms trailing boxcar; 5 ms trailing mean, 20 ms trailing maximum and 20 ms log slope. The pickup coupling is fitted once per shot over the whole shot (a static calibration, trimmed of bursts) and applied causally.
- `icoil_dev_ka`, `icoil_level_ka`: I-coil activity, the largest deviation of the six I-coil currents from their causal 100 ms median (5 ms trailing maximum) and their root mean square (5 ms trailing mean), kA.
- `rot_q2_khz`: ZIPFIT rotation at the outermost q=2 surface of the EFIT01 q profile (`qpsi` on normalised psi, mapped to rho by `rhovn`), raw ZIPFIT units as `rot_*_khz`. It is NaN where q_min ≥ 2, which is 52% of the 2014 and 86% of the 2018 slices; finite on 23% of all slices.
- beta_N − 4 l_i was already an input (`betan_minus_4li`); N1RMS stays as context.

Each slice at t is the 10 ms bin [t − 10, t), whose trailing inputs it reads; δBp at t is an input and nothing is masked around an onset (`labeler.rwm.classify`, `scripts/labeler/rwm_classify.py`). Because the v5 present spans end where δBp falls back under the screen threshold and the uncertain stretches are δBp runs, rwm-ours reads the labels' own sensor.

**rules**: beta_N, beta_N/l_i and elapsed time since the first |Ip| ≥0.5 MA sample. The `rwm_candidates` screen is a concurrent review aid; its whole-flat-top median/MAD threshold and the analysis span are retrospective.

## Method
The review page's source is `rwm_onsets` v5 (round five; `review/source.json`, built by `scripts/labeler/rwm_onsets_source.py --version v5`, counts in `outputs/labeler/rwm/onsets_source_v5.json`). v5 measures the mode on the n=1 δBp of the midplane poloidal-field probes, the RWM sensor of Okabayashi et al. 2005, instead of N1RMS. N1RMS (`\MHD::N1RMS`) is the RMS of the n=1 part of the automated Mirnov spectrum from its 0.5 kHz bin up, so it sees rotating modes and not an RWM rotating below 30 Hz. On Hanson's onsets, rank AUC against flat-top windows is 0.95 (2014) and 0.99 (2018) for δBp, 0.66 and 0.83 for N1RMS.

**Sensor** (`labeler.rwm.bp`). It uses the differences B(φ) − B(φ+180°) of the midplane probes at φ = 67.5, 97.4, 127.0 and 157.6° (PTDATA MPID067U/097U/127U/157U). These are binned to 0.5 ms and converted to G, and each has its centred 100 ms running median subtracted, which removes the 5–20 G static offsets. Coil pickup is removed next: per shot and probe, a two-pass least-squares fit (residuals beyond 5 MAD dropped on the second pass) on two sets of regressors, each less its running median:
- the six I-coil currents IU/IL30, 90 and 150;
- the odd C-coil differences C79−C259, C139−C319 and C199−C19.

A regressor that varies by less than 5 A is skipped. Finally D(φ) = 2A cos(φ − φ0) is fitted, giving the amplitude A (G) and the phase φ0, and A is smoothed with a 2 ms boxcar. The differences cancel even n, so the sensor sees n=1 (and n=3) but never n=2.

**Single probes (2021 on).** PTDATA has no MPID difference points from 2021 on. There each difference is 0.5 × (MPI66M{φ}F − its partner 180° away): 247, 277 and 307 for 067, 097 and 127, and 340 for 157, since the 337° probe is gone from 2018 on (`bp.SINGLES`, `DIFFERENCE_GAIN`). On 176088 and 178872, which have both, the singles' difference matches MPID with gain 0.46–0.49 and correlation 0.98–0.99 after the running median (`$LABELER_ROOT/round5/rwm/checks/fb_compare.out`). The meta names the probes each shot used (`probes`).

**Screen.** The measured time is where δBp is finite and |Ip| ≥ 50 kA. The threshold is the larger of the measured δBp's median + **6 MAD** and **5 G**. A candidate is a run above the threshold of at least **3 ms**, where a dip shorter than 3 ms with every sample finite does not end the run. It must have βN > **2.4** li (nearest EFIT01 samples) somewhere in the 20 ms before it starts; where EFIT01 has no sample there (it often ends before a terminal mode), the condition is untested and the candidate kept. The fraction of the candidate's N1RMS samples above N1RMS's median + 6 MAD is recorded (`n1rms_cover` in the meta) but drops nothing. v4's Mirnov n=1 track gate (below) then applies. Candidates are uncertain, never present.

**The start condition.** Until the 2026-10-14 follow-up the screen asked βN > 4li and dropped a candidate as rotating when N1RMS covered half of it. Neither holds on the reference RWMs (Reference set, below): βN > 4li is false at all of them, and the N1RMS check dropped five of the seven, because the N1RMS crash of a terminal mode falls inside its run. 2.4 is the largest round margin that keeps all seven references (lowest βN/li 2.76, 114329) and all 44 of Hanson's onsets that a δBp run reaches within [o−20, o+25] (lowest 2.46, 156785; the runs at 156797's 1598 ms, 176069's 3235 ms and 176070's 3013.5 ms onsets have no EFIT01 sample before them and are kept untested); the survey is `$LABELER_ROOT/round5/rwm/checks/rule_survey.csv` (script beside it). No rule of this kind also rejects the two negatives: 119666's δBp runs have βN/li 3.7–5.9 and 120290's internal mode at 1680 ms 4.0, and their N1RMS before and during the run (5–58 G) overlaps the references' and Hanson's (up to 29 G). Okabayashi et al. told 120290's mode apart on the Mirnov dδBp/dt, which the screen does not read. So both stay uncertain candidates: the rule errs towards review rather than towards dropping an RWM.

**Onsets.** [o−20, o) stays uncertain. An n=1 onset is present from o to the first of:
- δBp back under the threshold: the first sample at or after the later of o and the rise that lies outside the runs above it, counted only when δBp was above the threshold somewhere in [o−20, o+**25** ms];
- the βN collapse;
- the end of Ip;
- o + 300 ms.

An end within 10 ms of o is moved to o + 10 ms. If δBp does not respond, o is present for 10 ms and the time to the other ends is uncertain. n=2 onsets keep v4's N2RMS end.

**Results.**
- 48 of the 50 n=1 onsets respond. Their rise comes −2.2, 1.1 and 2.1 ms after o (quartiles).
- Of the 50 n=1 onsets, 27 end at the 10 ms minimum, 13 on δBp, 8 on the βN collapse, and 2 show no response.
- Of the 6 n=2 onsets, 3 end on N2RMS, 2 show no response and 1 ends on the βN collapse.
- The median present duration is 10 ms (quartiles 10 and 21.4 ms, maximum 70.5 ms). Present time is 0.99 s, against 2.55 s in v4.
- The 25 ms rise window and the thresholds were chosen on the first v5 domain, which also required βN and li (the follow-up dropped that, which moves each shot's median and MAD a little and lets six more onsets respond). The 25 ms rise window was chosen from the measured lags. Of the 50 n=1 onsets, the first run starts within −21 to +8 ms of 39, at +10.6, +15.5 and +22 ms on three, and at +31 to +37 ms (or never) on eight.
- The thresholds come from a survey of the 208 study shots (`$LABELER_ROOT/round5/rwm/checks/screen_stats_all.json`, with `rise_lag.csv`; scripts beside them). With the 5 G floor, every K from 4 to 10 reaches the same 39 of 50 n=1 onsets within [o−20, o+25]. A 3 G floor reaches 43 but has 2.5 times the candidate time on the comparison shots, and an 8 G floor reaches 32.

**Screen results.**
- On Hanson's 33 shots the screen finds 358 candidates (5.9 s, on every shot; median 9.5 ms). 348 pass the start condition and 10 are untested; 134 δBp runs fail it. All 358 stay uncertain and unchecked, because no Hanson shot has a Mirnov record in the MHD cache.
- On the 534 roster shots from 2021 on that could be screened, it finds 1,945 candidates (23.0 s). The Mirnov gate drops 833 as co-rotating n=1 tracks and keeps 1,112 (13.1 s on 196 shots; median 9 ms; 16 of them have no Mirnov record). The 25 older roster shots give 5 candidates.
- On round four's comparison shots (not in the table; `outputs/labeler/rwm/onsets_source_v5.json#/comparison`), it keeps 247 candidates (3.3 s) on all 20 same-day shots and 558 (6.2 s) on 109 of 154 same-experiment shots, none Mirnov-checked. The comparison shots are therefore not checked RWM-free.

**Coverage.** The table covers the same 628 shots as v4. δBp is screened on 592 of them: Hanson's 33 (MPID), 25 roster shots up to 2018 (MPID) and 534 from 2021 on (single probes). v4 screened 248 on N1RMS. The other 36 are not observable outside the onset stretches:
- 32 have an Ip record on an ambiguous legacy clock (23 older, 9 from 2021 on).
- 4 older shots (132240, 132710, 138388, 142111) have probe records on clocks the resolver refuses.

The table holds 3,832 rows: 54 present, 1,500 uncertain, 2,048 unassessed and 230 not observable (`$LABELER_ROOT/round5/rwm/checks/coverage_v5c.json` by role).

**Reference set.** The 2003-04 shots of Okabayashi et al. 2005 (`$LABELER_ROOT/round5/rwm/checks/paper_sensor.json` and `paper_check.json`; their multi-rate PTDATA clocks were prepared into the raw cache by `scripts/paper_prep.py` there) are where the start margin was chosen (above); nothing else was tuned on them:
- δBp reaches every one of its six RWMs within [o−20, o+25] with a run of at least 3 ms (peaks 47–84 G), and 119631's terminal mode too.
- The screen flags all seven (`$LABELER_ROOT/round5/rwm/checks/paper_screen_v5b.json`). The first v5 screen flagged none: EFIT01 puts βN below 4li at every onset (βN 1.9–2.3 against 4li 2.6–2.9; on these shots the 4li proxy sits above the no-wall limit), and on the terminal modes the N1RMS crash at the disruption covers 0.55–1.0 of the run, which dropped five of them as rotating.
- 120290's bursts at ~1390 and ~1588 ms peak at 5.1 and 4.2 G, at the floor, and are missed.
- The two negatives are not rejected: 120290's 10 kHz internal mode at 1680 ms (βN/li 4.0) and three δBp runs on the feedback-on control 119666 (βN/li 3.7–5.9) are uncertain candidates. These shots have no Mirnov record in the MHD cache, so the track gate cannot reach them either.

**Limitations.**
- The 100 ms running median absorbs any perturbation longer than about 50 ms, so a slowly growing or long-lasting mode's tail is cut.
- A locked tearing mode also produces an n=1 δBp. Only the Mirnov gate sets it apart, while it still rotates; the N1RMS cover is recorded for the reviewer but drops nothing.
- n=3 passes the odd-n differences.

v4, the previous source (counts in `outputs/labeler/rwm/onsets_source_v4.json`), measured the mode as follows. At each Hanson onset o, **[o−20, o)** is uncertain and the mode is present from o to the first of: the 50 ms centred mean of the onset's RMS amplitude back at or below its pre-onset baseline (its mean over [o−70, o−20)) after rising above it within **25 ms** of o; the first βN sample at most 60% of the last one at or before o; the first |Ip| sample below 50 kA; o + **300 ms**. The amplitude is N1RMS for an n=1 onset. For an n=2 onset it is N2RMS when the shot has N2RMS cached with at least one sample in [o−70, o−20), where the baseline is taken; otherwise it is N1RMS, and the onset's record in the table's meta (`per_shot.<shot>.onsets`) carries `end_rms: n1rms` and `n2_end_on_n1rms: true`. All 6 n=2 onsets (158015, 158023, 176085, 176088, 176089, 176092) have N2RMS samples there, so none is flagged; v1 and v2 measured them on N1RMS. An end within 10 ms of o is moved to o + 10 ms. If the amplitude's mean does not rise above the baseline within 25 ms of o, nothing measured the mode after its onset: o is present for 10 ms, and the time from there to the first of the other ends is uncertain. All 56 onsets get a present span. 17 end on N1RMS, 3 on N2RMS, 4 on the βN collapse, 3 at the 300 ms cap and 2 at the 10 ms minimum. The other 27 show no response (25 on N1RMS, 2 on N2RMS); their uncertain stretches have a median length of 37.0 ms. The median present duration is 10.0 ms (quartiles 10.0 and 28.5 ms, maximum 300 ms), and 29 of the 56 present spans are 10 ms long. The median depends on the rise window (`rise_sensitivity`): 10.0 ms with a 10 ms or a 25 ms window, 17.8 ms with 50 ms and 38.5 ms with no bound. v3 moved only the n=2 onsets against v2: their median present duration fell from 68.5 ms on N1RMS to 31.8 ms on N2RMS, and present time in the table fell from 3.136 s to 2.548 s. For example, 158015 at 2613 ms (N1RMS never came back: 300 ms) is now present for 10 ms with no N2RMS response, then uncertain to the cap, and 176088 at 4133.5 ms went from 300 ms to 70.5 ms. No n=1 onset's span changed. v4 draws every onset's spans as v3 did; it changes only the screen (below). v2 measured every onset on N1RMS; its median was 12.1 ms (quartiles 10.0 and 40.4 ms, `onsets_source_v2.json`). v1 had no rise bound. It took the first rise however late, and drafted an onset whose N1RMS never rose as present up to its other ends. Its median was 38.5 ms (quartiles 23.8 and 167.4 ms). When N1RMS was below its baseline at the onset, v1 ended the span on a later excursion. For example, 176070 at 3013.5 ms (the gold RWM shot) was present to 3180 ms in v1 and is present for 10 ms since v2. Of the 26 onsets with no response in v2, v1 ended 17 on the βN collapse, 5 at the cap and 4 on N1RMS (`onsets_source_v1.json`). Onsets whose spans touch share one row, so the v4 table holds 54 present rows, beside 94 uncertain rows. The table covers 628 shots: the RWM review queue, Hanson's 33 shots, and the silver and gold lists, less the catalog cohort's blind shots (none is in the table). 381 of them are not observable throughout, 377 because they have no N1RMS cached, so on most silver and gold shots the source shows nothing. Hanson's shots, and silver and gold shots outside the population, are windowed by their own Ip (at least 50 kA), not by last onset + 500 ms. The page drafts each onset as one Individual annotation from o to the end of its present span, beside the uncertain Spans before it and, without a response, after it (`labels.with_onsets`). Only the onset time is curated evidence. The end is measured on N1RMS or N2RMS, βN and Ip, none of them RWM-specific, and N1RMS also rises with rotating n=1 modes, so a present span can include tearing or locked-mode time. The earlier `rwm_candidates` v1 table (the screen alone, no present rows) and the `rwm_onsets` v1 to v3 tables stay on disk.

v4's screen required n=1 RMS above the whole-flat-top median plus **6 MAD**, **beta_N > 4 l_i**, and **at least 10 ms** duration. Screen candidates are uncertain, never present; other time is unassessed, and missing inputs are not observable. N1RMS cannot distinguish RWM from tearing or applied-field response. Since `rwm_onsets` v3, a candidate is dropped when at least **half** of its time lies inside n=1 Mirnov tracks: a rotating mode is not an RWM. Its time becomes unassessed. In v4 the tracks are every n=1 track that is not a fixed-frequency line, the same tracks that make `mhd_n1` present, whatever their resonance status (m/n, not resonant, no rotation, q_min above, uncertain, or not modelled at all). v3 counted only the tracks whose co-rotation P(resonant) was at least 0.9. The tracks come from the mhd_modes v4 pipeline cache (`$LABELER_ROOT/mhd/labeler/v4/`, read, not rerun), and their frames are spanned as the MHD tables span them (2 ms pad, 20 ms merge, 20 ms minimum). Only track time inside the shot's Mirnov record counts (`mirnov_ms` less `mirnov_gaps` in the cache). A candidate that is not dropped counts as checked only if it lies wholly inside that record; it is kept unchecked when the cache has no record of the shot, did not read its Mirnov, or when part of the candidate lies outside the record. The table's meta (`screen_gate`) names every cache file read with its sha256, gives one sha256 over them, and counts the candidates; `spans.run` writes it, so rerunning the table reproduces it. Of the screen's 47 candidates (2.0 s on 16 shots), v4 drops 27 (1.285 s on 7 shots: 185945, 185961, 186636, 189272, 195016, 195024 and 203959), keeps 4 after the check (0.080 s on 4 shots) and keeps 16 unchecked (0.632 s on 7 shots). v3 dropped 13 (0.389 s on 5 shots). The 14 further drops turn 0.907 s of uncertain time into unassessed time on 185945, 185961, 195016, 195024 and 203959; no present span changes. The unchecked candidates are all on Hanson shots: none of the 33 Hanson shots has a Mirnov record in the cache (11 are in it as "no corpus record", 22 are not in it), so the check cannot reach any candidate near a Hanson onset. The `rwm_candidates` v1 table keeps the ungated screen. See [the editor guide](../../equilibrium_review.md).

Dataset 1 loads as point events, preserving `NTOR` and `MODE_TYPE` in `attrs`
with unknown confidence and coverage; an unlisted shot is not a negative.
The `extend_rwm/recommender_v1.csv` scan of the fixed 500 project shots is
header-only because none contains a listed Hanson onset, a completed zero rather
than an absence claim (see `src/labeler/events/rwm.py`).

Round four's forecaster (`rwm-brf` until 2026-10-14) is an equilibrium-scalar timing baseline on Hanson's onset list, not an
RWM predictor. It trains only on Hanson RWM shots: no stable discharge supplies
negatives, so it learns **when an RWM comes in a shot that has one**.
The measured comparison-negative sensitivity is reported under Models;
whether verified stable-shot negatives would help is untested.

In the Dataset 2 export that round four's forecaster reads (not the review source above),
Hanson onset points support one minimal present slice **[o, o+10 ms)**
(category 1, `onset_point_minimal`); this does not measure mode duration.
**[o−20 ms, o)** remains uncertain (category 2); the 20 ms length assumes a wall
time of about 5 ms, an uncited assumption (Piccione 2022 gives only "milliseconds"). Time before the first
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
The repository evaluation is a summary below 0.6 MB. Its `external_details`
pointer names the complete record under `$LABELER_ROOT/round4/rwm/`, including
per-shot outcomes and within-shot details; read it with
`labeler.rwm.records.load_evaluation(path, details=True)` to verify the SHA-256.
See [protocol and reproduction](../../rwm_baseline.md) and
[evaluation.json](../../../../outputs/labeler/rwm/evaluation.json) for scores and intervals.

## Alias
resistive wall mode, rwm, resonant field amplification (precursor), rfa

## Future Implementations
Phase-matched negative sampling, which draws training negatives at the same elapsed time and campaign as each positive instead of ending every labelled span in a positive, would test skill beyond phase more directly than the comparison-negative sensitivity, and is left as future work.

## Reference
- M. S. Chu and M. Okabayashi, "Stabilization of the external kink and the
  resistive wall mode", Plasma Phys. Control. Fusion 52, 123001 (2010).
- A. M. Garofalo et al., "Sustained stabilization of the resistive-wall mode by
  plasma rotation in the DIII-D tokamak", Phys. Rev. Lett. 89, 235001 (2002).

## Contact
- **Jeremy Hanson** (original dataset): hansonjm [at] fusion [dot] gat [dot] com
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu
