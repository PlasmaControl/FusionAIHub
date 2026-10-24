# Tearing Mode (TM)

> The label data is `data/events/neoclassical_tearing_mode/`: the `raw/`, `format/`, `review/` and `extend_*/` paths
> below are relative to it. It is not in git. `scripts/labeler/hf_publish.py pull`
> restores the label tables (`format/`, `review/`, `extend_*/`), but not `raw/`: the
> third-party originals are not in the Hugging Face release; they are on the Stellar
> checkout's disk and archived in `/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/raw_archive/`.

## Description
These labels include classical and neoclassical tearing modes. The abbreviation
is `tm`; the historical directory name and `ntm` alias remain readable by
existing annotations and models.

Tearing modes are resistive MHD instabilities that connect the field at the rational surface q = m/n. This creates a magnetic island, flattening the pressure across it and degrading confinement. The neoclassical tearing mode (NTM), governed by the
modified Rutherford equation

$$(\tau_R / r_s) \frac{dw}{dt} \approx r_s \Delta'(w) + a_bs * \beta_p * (w / (w^2 + w_d^2)) - ...$$

which is metastable: it needs a seed (sawtooth crash, ELM, fishbone) above a
threshold width and then saturates. The 3/2 degrades confinement by 10-30%; the
2/1 rotates at a few kHz, slows by wall drag, LOCKS (frequency -> 0) and usually
disrupts.

First described by Furth, Killeen and Rosenbluth (1963); NTMs identified on TFTR in
1995 and set the beta limit on DIII-D and JET.

Typically found via Mirnov / MHR magnetics as a coherent line below ~30 kHz with
its toroidal mode number from the probe array, on ECE as a flattened island
signature, and on the radial saddle loops once locked.

## Data Provenance
### Dataset 1

**Dataset File(s)**:

**Author**: Jaemin Seo

**Description**: Jaemin Seo tearing archive (`tm_label`, 8,505 shots 147000-190997 for the CNN's training; 1,503 overlap shots with the FAITH corpus for evaluation). Models are upstream PlasmaControl artefacts, re-served by labeler with pinned preprocessing. The review page's shot list `n1_seo` holds the archive's shots with a positive sample (1,801 less 3 blind test shots: 1,798; 161 are in the event catalog).

**Publications**:

### Dataset 2

**Dataset File(s)**: `tm_labels.tar`, `tm_labels.h5`

**Author**:

**Description**: `raw/tm_labels.tar` and `raw/tm_labels.h5` now contain original sampled labels from `/projects/EKOLEMEN/survival_tm_2/` (Hiro Farre's directory): one 1 ms `<shot>_ntm.npy` in the tar for each of 28,521 shots (140000-195501), 4,820 with a positive label, the shots of the format table; the h5 holds 18,064 of them at 20 ms. The review page's shot list `n1_farre` holds the tar's shots with a positive label (4,814 after 6 blind test shots; 429 are in the event catalog) and `n1_farre_seo` the two lists together (4,943; 1,669 in both). The two label sets are not the same: 8,487 shots are in both, and 1,671 of Seo's 1,801 positive shots are also positive in the tar (`docs/labeler/shotlists.md`).

**Publications**:

## Models
**stable**: d3d_tearing_onset_cnn1d | 2022_12_01

**latest**: none (the detectors below are saved development cross-validation ensembles, not registry models; nothing here has a deployed adapter)

**review source**: tm_mhd | v2 (`review/source.json` since 2026-10-14, after v1 earlier that day and `window` v1 before it; the rule labels joined with the `mhd_modes` v5 mode tracks; unreviewed; see "MHD-resolved source")

**all**:
- d3d_tearing_onset_cnn1d | 2022_12_01 (upstream training date; presence at t+25 ms)
- d3d_tearing_time_to_event_dsm | 2026_09_05 (survival forecast; 250 ms / 500 ms / 1 s)
- d3d_tearing_time_to_event_dsm_continued | 2026_09_05 (continued-training variant)
- tm-onsetcnn-retrained | 2026_10_04 (prior CNN architecture retrained for detection at t)
- tm-dsm-retrained | 2026_10_04 (prior survival embedding with a detection head at t)
- tm-ours | 2026_10_04 (Mirnov spectrogram detector, 10 ms bins; saved CV ensembles)
- tm-ours-rms | 2026_10_04 (the same detector with the RMS as an input; circular ablation)

The short names identify benchmark training scripts and saved cross-validation
predictions, not registry models. Saved weights: only the two Mirnov detectors keep
their fold ensembles, normalization and thresholds, under
`$LABELER_ROOT/round4/tm/checkpoints/tm_ours_magnetics/` and
`tm_ours_magnetics_rms/`. The retrained CNN and survival-embedding detectors keep
only their out-of-fold predictions and thresholds (`results/oof_tm_prior_retrained_*`),
not weights. All are experimental CV artifacts without a deployed registry adapter. Published
weights remain available through the three original model IDs above. A two-line RMS
baseline, `max(n1 RMS / 12 G, n2 RMS / 6 G)`, needs no training and is the reference
the detectors are measured against.

## Inputs
**d3d_tearing_onset_cnn1d**:
- at t+25 ms: `bt`, `ip`, `pinj_total`, `tinj_total`, `R0_EFITRT1`, `kappa_EFITRT1`, `tritop_EFIT01`, `tribot_EFIT01`, `gapin_EFIT01`, `ech_pwr_total`, `EC.RHO_ECH`
- profiles at t: `thomson_density_mtanh_1d`, `thomson_temp_mtanh_1d`, `1/qpsi_EFITRT1`, `pres_EFIT01`, `cer_rot_csaps_1d`

**d3d_tearing_time_to_event_dsm**:
- `bmspinj`, `bmstinj`, `betan_EFITRT2`, `qmin_EFITRT2`, `ech_pwr_total`, `ip`, `PCBCOIL`, `li_EFITRT2`, `aminor_EFITRT2`, `rmaxis_EFITRT2`, `tribot_EFITRT2`, `tritop_EFITRT2`, `kappa_EFITRT2`, `volume_EFITRT2`
- profiles: `thomson_temp_mtanh_1d`, `cer_temp_csaps_1d`, `thomson_density_mtanh_1d`, `cer_rot_csaps_1d`, `qpsi_EFITRT2`, `pres_EFITRT2`

**d3d_tearing_time_to_event_dsm_continued**:
- `bmspinj`, `bmstinj`, `betan_EFITRT2`, `qmin_EFITRT2`, `ech_pwr_total`, `ip`, `PCBCOIL`, `li_EFITRT2`, `aminor_EFITRT2`, `rmaxis_EFITRT2`, `tribot_EFITRT2`, `tritop_EFITRT2`, `kappa_EFITRT2`, `volume_EFITRT2`
- profiles: `thomson_temp_mtanh_1d`, `cer_temp_csaps_1d`, `thomson_density_mtanh_1d`, `cer_rot_csaps_1d`, `qpsi_EFITRT2`, `pres_EFITRT2`

**ntm_frames (historical model ID, round three)**:
- `MPI66M322D power` (image, 32 frequency groups)
- `toroidal n, MPI66M probes` (modes row, 10 n)

## Method
Three claims:

1. **TokEye track event** `tokeye_track` / `coherent_mode` with band <= 30 kHz and
   bandwidth <= 30 kHz on mhr, ece or co2 (a locked mode sits at 0 kHz and is
   invisible to this route).
2. **Model label** `d3d_tearing_onset_cnn1d/tm_prob`: probability a tearing mode is
   present 25 ms from now, from 0-D scalars and profiles (PlasmaControl ensemble
   of ten 12,086-parameter networks).
3. **Forecasts** `d3d_tearing_time_to_event_dsm/tm_risk_{250ms,500ms,1s}`
   (+ isotonic variants): the Deep-Survival-Machines risk of an onset within the
   horizon; a forecast is never an observed event.

`validate.alarm_quality` scores the published labels against archived onsets
(`tm_label` on the 1,503 shots shared with the tearing archive).

**Whole-interval labels** (`extend_tm_interval/tm_interval.csv`, 450 development shots;
the lab's earlier labels are onsets or forecasts, these say when a mode is present):
a rule on the n = 1 and n = 2 magnetic RMS (`\MHD::N1RMS`, `N2RMS`, gauss, 1 kHz; the
processed magnetic traces), `labeler.tearing.rule`. The label is **a strong rotating
n=1/n=2 mode (tearing-mode proxy)**, without independent island identification.
A seed must exceed 12 G (n1, Farre-Kaga et al. 2025) or 6 G (n2, local extension)
continuously for at least 50 ms in both raw and 5 ms median RMS, before any runs
are joined. A stable rotating n-resolved line from 1.5 kHz up to the cap (30 kHz for
n = 1, 60 kHz for n = 2) must support that seed (`N1FREQ`/`N2FREQ`, or Mirnov
coherence when frequency is unavailable; the Mirnov line search covers 1 to 30 kHz).
Hysteresis extends each qualified seed to max(1 G, 10% of its peak); release dips
up to 50 ms merge only across available data. Merged components retain their own
release levels; the stored summary release is the minimum, the peak the largest.
The n = 2 frequency cap scales with n (the Mirnov features stop at 30 kHz, so the
n = 2 cap acts through `N2FREQ` only). The development-only harmonic veto drops an
n = 2 seed unless n2/n1 > 0.57, the 99th percentile (0.561, rounded up) of that ratio
over bins whose n = 2 line sits at twice the n = 1 frequency and whose Mirnov best-fit
toroidal number there is 2: the harmonics of a rotating, non-sinusoidal n = 1 mode
carry n = 2. Toroidal phase cannot separate such a harmonic from a co-rotating,
frequency-coupled n = 2 mode (a 3/2 mode locked to the 2/1), so **the veto is a
heuristic and may also remove real 3/2 modes**; <!-- gen:veto -->it removes 9 of 86 n = 2 seeds on the development shots; the finished rule has 20 n = 2
intervals, 19 without the veto, because a removed seed can change how its neighbours
merge (net effect on intervals +1)<!-- /gen:veto -->.
Unsupported seeds, high-frequency or chirping bursts, and sustained coherent sub-seed lines are category 2 (uncertain).
Weak uncertainty tracks require a continuous 100 ms coherent core above the frozen
development quiet-amplitude p95, then follow that line down to this amplitude
floor itself; the same weak screen runs over ramp-up and flat-top. Brief evidence interruptions up to 50 ms can join; acquisition gaps cannot.
Frequency drops alone are `locked_candidate`, with all candidate times retained;
only independent locked-mode confirmation sets `locked=true` and truncates the
rotating span at the confirmed time. An abrupt fall from above the seed to below
release within 5 ms is never
`decay`: it ends `locked` when radial-field evidence confirms, otherwise `unknown`.
The subsequent phase is uncertain until the lock signal falls or the discharge ends;
without that signal it stays uncertain to the discharge end. Lock confirmation uses
the independently fetched n=1 `DUSBRADIAL` radial-field amplitude (native ptdata
units, treated as gauss by disruption-py; the unit is not verified here): a lock is
a **step**. At each candidate time (a frequency drop at least 50 ms after the seed
starts, an abrupt collapse, an interval's end) the median |DUSBRADIAL| over 20 to
120 ms after must exceed its median over 200 to 20 ms before by at least 5, so a
field that is already high or ramps slowly confirms nothing. Candidates are checked at every interval end
and for rejected ones. A sustained rise (100 ms above the quiet median plus 5, with a
step at its start) in otherwise-absent flat-top time is uncertain with reason
`locked_unseeded`: a field event with no mode seen. Coverage is recorded in the label metadata; shots with no
`DUSBRADIAL` record keep an unknown lock status. `locked_known` is true only for a
confirmed lock (the interval is then also `locked`); false says that none was
confirmed (no record, no step, or n = 2), not that the lock was checked and found
absent. The onset is a point event
(`iscrowd` 0, at the interval's start) with an `onset_window_ms` attribute (the start
of the preceding same-n weak track), the interval a span (`iscrowd` 1); both carry
`n`. `m` requires EFIT q at an independently observed island radius, such as an ECE
flattening location. No island radius is resolved here, so `m` is empty; a unique
candidate rational surface alone does not identify it. The rest of each shot's
window is absent, the ramp-up uncertain where the rule fires in it, and every RMS
acquisition gap unobservable (preserved as NaN). tm-ours reads the same Mirnov array
as the label's N1RMS; this benchmark measures recovery of an RMS-based rule.
Counts, thresholds, the agreement with Seo's and the survival onsets and
the detector benchmark are in
[tearing_detection.md](../../tearing_detection.md). The detectors
trained on these labels are `tm-ours` (Mirnov-array spectrogram features, per 10 ms
bin) and the two prior architectures retrained for detection, `tm-onsetcnn-retrained` and
`tm-dsm-retrained`. Their targets are mode presence at t, with horizon zero.

These labels omit fast-locking and brief modes, and weak modes. Recall of the lab's
archived onsets, counted when an onset lies in one of our intervals or within 100 ms
of its edge, is Seo **12/26** and survival **16/67** on the development shots; that
tests whether the label holds the historical onset, not the accuracy of its timing
(the onset error against Seo has a median of 130 ms). Cohort "absent" can still
contain weak modes that fail the weak-line screen. The uncertainty mask is partly
Mirnov-derived and shares `tm-ours` inputs; it covers 34.0% of the observable
catalog-window time of the development shots, and a co-primary benchmark group scores
it as negative. A lock is confirmed from `DUSBRADIAL`, which is on file for 327 of the
450 development shots (12 confirmed locks); the rest keep an unknown lock status. The blind split carries no tearing-mode labels: its 50 shots are never opened
for labels, features or scores. The benchmark tests recovery of a magnetic rule, not
superiority as a TM detector, and the untrained two-line RMS baseline is the reference
it is paired with: <!-- gen:paired -->On all 450 development shots, which both models score on identical bins, the paired
difference `tm-ours` minus the two-line baseline is, with uncertain time excluded
(162840 bins), AUROC 0.000 [-0.001, +0.001] (not resolved); AUPRC +0.009 [+0.003,
+0.015] (`tm-ours` ahead); F1 +0.020 [-0.011, +0.046] (not resolved); segmental F1
+0.215 [+0.111, +0.309] (`tm-ours` ahead); with uncertain time scored as negative
(246801 bins, 450 shots), AUROC -0.015 [-0.029, -0.003] (the two-line baseline ahead);
AUPRC -0.120 [-0.222, -0.019] (the two-line baseline ahead); F1 -0.084 [-0.104, -0.065]
(the two-line baseline ahead); segmental F1 +0.038 [+0.010, +0.068] (`tm-ours` ahead).
The order reverses between the groups on AUPRC (`tm-ours` ahead with uncertain time
excluded, the two-line baseline ahead with it scored as negative); `tm-ours` is ahead in
both groups on segmental F1; the order is not resolved in at least one group on AUROC
and F1. The untuned seed-level rule (the same score at its seed levels, no training, no
tuning) has, with uncertain time scored as negative, F1 0.578 [0.507,0.643] and
segmental F1 0.269 [0.204,0.341] against 0.363 [0.290,0.431] and 0.229 [0.182,0.281] for
`tm-ours` (paired difference `tm-ours` minus the seed-level rule: F1 -0.215 [-0.305,
-0.124] (the seed-level rule ahead); segmental F1 -0.040 [-0.107, +0.029] (not
resolved); with uncertain time excluded, F1 +0.315 [+0.238, +0.393] (`tm-ours` ahead);
segmental F1 +0.368 [+0.231, +0.493] (`tm-ours` ahead)). With uncertain time scored as
negative the seed-level rule has the higher F1 and segmental F1 (F1 resolved; segmental
F1 not resolved), so the segmental-F1 lead of `tm-ours` holds only against the tuned
baseline.<!-- /gen:paired --> No TM coverage gain is
claimed: on the survival-matched shots the interval labels cover 464.4 s against
799.3 s for the legacy labels.
Population weak screening uses the same criteria where inputs exist; unscreened time
above the weak RMS thresholds is uncertain rather than absent.

The `extend_` table must be converted before promotion to `review/`: state 3 is
outside the TM catalog schema, n-specific rows overlap, onset points have zero
length, and boundaries need whole-millisecond conversion. The focused table test
checks interval geometry only, not full catalog validity.
Historical `results/*_test.json` files were moved without inspection to
`$LABELER_ROOT/round4/tm/results/quarantine_blind_test/`; they are excluded from all
current processing. Superseded all-cohort audit/agreement source snapshots are
quarantined there as well, without inspection, and are removed from the active
benchmark sources. These historical records must not be used for model selection
or evaluation. Blind shots are never opened for labeling, galleries or scores.

### MHD-resolved source
The review page opens an unsaved shot on `tm_mhd` v2 (round 5, decision S5 as amended
on 2026-10-14; `scripts/labeler/tm_source.py`, rule in `labeler.tearing.source`, table
`$LABELER_ROOT/suggestions/tm_mhd/v2/`, whose meta records the rule tables, the sha256
of every input, the rule text and each shot's equilibrium tree; v1 stays on disk). It
joins the rule labels above with the mode tracks of the MHD analysis (`mhd_modes` v5,
read from its cache, not rerun); `tm_interval.csv` is unchanged. Present is the rule's
present time plus every co-rotating resonant track labelled 2/1, 3/1, 3/2, 4/3 or 4/2
(status m/n; 4/2 marks an island at q = 2), its frames joined across gaps of up to
50 ms and kept when the span is at least 50 ms long. A resonant track with another
label (4/1, 5/2, 5/3, 7/3: outside the named classes, where the resonance law was not
fit) is unsure, and so is an n >= 2 track with at least half its frames at n times the
frequency of a simultaneous 1/1 track (within the harmonic lock's tolerance, the larger
of 0.3 kHz and 2%): an unlocked harmonic of the 1/1, or a mode that cannot be told from
one. Unsure time is uncertain, never absent. Rule-uncertain time becomes absent where
the MHD analysis observed it and saw only 1/1 or non-resonant tracks, except in the
rule's locked phases (`confirmed_locked_phase`, `post_collapse_lock_unknown`): a locked
island does not rotate, so the rotating-mode tracks cannot rule it out. Observed means
that the Mirnov record covers the time, that the shot has an equilibrium tree (EFIT02,
EFIT02ER or EFIT01) with a slice within 30 ms, and that a rotation profile lies within
60 ms. Rule-uncertain time stays uncertain where an unsure track or a resonant track too
short for a span covers it, where there is no track, where the MHD did not observe, and
in a locked phase. Rule absent and not observable time stay as they are unless a
resonant span covers them. A shot without rule labels is present on the resonant spans,
not observable where the Mirnov was never read (as for the MHD events), absent where the
MHD observed and found no other track that counts, and uncertain elsewhere.

The table covers 606 shots (the roster, silver and gold; no blind shot) in 4,943 rows:
1,728.8 s absent, 720.3 s present, 541.9 s uncertain and 368.3 s not observable; 346
shots have present time. The tree is EFIT02 on 493 shots, EFIT01 on 95, EFIT02ER on 15
and none on 3. On the 487 shots with rule labels, present time grows from 158.7 s (the
rule alone) to 679.9 s and uncertain time falls from 915.1 s to 377.8 s: 459.9 s of
rule-uncertain time became present, 77.4 s became absent, and 61.3 s of rule-absent
time became present. The uncertain time left has an unsure MHD track (240.5 s, of which
31.8 s lies under an out-of-set label and 22.6 s under a 1/1 harmonic; they overlap),
no MHD observation (68.3 s), no track (68.9 s) or a locked phase (0.1 s). The 119 shots
without rule labels (silver and gold) have 40.4 s present, 105.2 s absent, 164.1 s
uncertain and 333.8 s not observable. The resonant spans are mostly 3/2 tracks
(407.9 s), then 2/1 (154.7 s), 4/3 (153.4 s), 3/1 (82.8 s) and 4/2 (9.7 s; labels
overlap). The RMS rule does see n = 2, but leaves its sub-seed and vetoed lines
uncertain: of the 459.9 s of rule-uncertain time that became present, 381.6 s lies in
sub-seed rows (n = 2 262.9 s, n = 1 118.7 s), 70.2 s in short seeds and 7.8 s in vetoed
lines. By track, 307.0 s of it lies under 3/2 tracks (250.9 s in n = 2 rows), 92.4 s
under n = 3 tracks (4/3), which the rule does not look at at all, 84.9 s under 2/1 and
43.5 s under 3/1 (labels overlap). So the added present time is mostly sub-seed or
vetoed 3/2, plus n = 3 time, which the rule never examines, a wider definition of the event than the rule's. Against
v1, which also counted the out-of-set labels and the 1/1 harmonics as present and left
the time without a Mirnov reading uncertain, present time falls from 780.5 s to
720.3 s and uncertain time from 828.6 s to 541.9 s, and not observable time rises from
34.5 s to 368.3 s. Gold shot 199597 is present at 2790-6038 ms (2/1), 199603 at
3574-6094 ms and 199607 at 3678-4034, 4306-4786 and 5058-5606 ms. No one has reviewed
this table.

The benchmark adds a sensitivity view, `mhd_resolved` (key `mhd_resolved` in
`benchmark/tm_benchmark.json`, from `tm_mhd` v2: records `*_mhd_resolved_v2_dev.json`;
the v1 records `*_mhd_resolved_dev.json` stay beside them). It uses the same models,
scores and thresholds as the primary rows and the rule's targets, except that a
rule-uncertain bin is positive where this table is present and negative where it is
absent. The primary rows and every other key are unchanged. On the 450 development
shots it scores 49,068 of the 83,961 rule-uncertain bins (41,918 positive, 7,150
negative), so positive bins grow from 14,403 to 56,321. `tm-ours` has AUROC 0.977
[0.970, 0.983], AUPRC 0.941 [0.923, 0.957], F1 0.843 [0.814, 0.870] and segmental F1
0.603 [0.556, 0.649] (primary 0.999, 0.994, 0.960, 0.843). The two-line baseline has
0.968, 0.923, 0.739 and 0.514 (primary 0.998, 0.986, 0.940, 0.629), and the retrained
onset CNN 0.900, 0.891, 0.694 and 0.612 (primary 0.982, 0.953, 0.844, 0.683). The
thresholds were tuned on the rule's targets, so the F1 values here are not tuned for
this view.

## Alias
tearing mode, tearing, tm, ntm, neoclassical tearing mode, 2/1, 3/2, locked mode, magnetic island

## Future Implementations

## Reference
- R. J. La Haye, "Neoclassical tearing modes and their control", Phys. Plasmas 13,
  055501 (2006).
- H. P. Furth, J. Killeen and M. N. Rosenbluth, "Finite-resistivity instabilities
  of a sheet pinch", Phys. Fluids 6, 459 (1963).

## Contact
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu
