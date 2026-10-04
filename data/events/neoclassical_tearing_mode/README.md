# Tearing Mode (TM)

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

**Description**: Jaemin Seo tearing archive (`tm_label`, 8,505 shots 147000-190997 for the CNN's training; 1,503 overlap shots with the FAITH corpus for evaluation). Models are upstream PlasmaControl artefacts, re-served by labeler with pinned preprocessing.

**Publications**:

### Dataset 2

**Dataset File(s)**: `tm_labels.tar`, `tm_labels.h5`

**Author**:

**Description**: `raw/tm_labels.tar` and `raw/tm_labels.h5` now contain original sampled labels from `/projects/EKOLEMEN/survival_tm_2/`.

**Publications**:

## Models
**stable**: d3d_tearing_onset_cnn1d | 2022_12_01

**latest**: none (the detectors below are saved development cross-validation ensembles, not registry models; nothing here has a deployed adapter)

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
heuristic and may also remove real 3/2 modes**; <!-- gen:veto -->x<!-- /gen:veto -->.
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
`DUSBRADIAL` record keep an unknown lock status. The onset is a point event
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
[tearing_detection.md](../../../docs/labeler/tearing_detection.md). The detectors
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
it is paired with: <!-- gen:paired -->x<!-- /gen:paired --> No TM coverage gain is
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
