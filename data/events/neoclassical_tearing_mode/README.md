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

**latest**: d3d_tearing_time_to_event_dsm | 2026_09_05

**all**:
- d3d_tearing_onset_cnn1d | 2022_12_01 (upstream training date; presence at t+25 ms)
- d3d_tearing_time_to_event_dsm | 2026_09_05 (survival forecast; 250 ms / 500 ms / 1 s)
- d3d_tearing_time_to_event_dsm_continued | 2026_09_05 (continued-training variant)

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
