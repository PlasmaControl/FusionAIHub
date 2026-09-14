# Tearing Mode

## Description
Tearing modes are resistive MHD instabilities that reconnect the field at a rational
surface q = m/n and open a magnetic island there, flattening the pressure across it
and degrading confinement. In high-beta plasmas the loss of bootstrap current inside
the island drives it further - the neoclassical tearing mode (NTM), governed by the
modified Rutherford equation

    (tau_R / r_s) dw/dt ~= r_s Delta'(w) + a_bs * beta_p * (w / (w^2 + w_d^2)) - ...

which is metastable: it needs a seed (sawtooth crash, ELM, fishbone) above a
threshold width and then saturates. The 3/2 degrades confinement by 10-30%; the
2/1 rotates at a few kHz, slows by wall drag, LOCKS (frequency -> 0) and usually
disrupts.

First described by Furth, Killeen and Rosenbluth (1963); NTMs identified on TFTR in
1995 and set the beta limit on DIII-D and JET.

Typically found via Mirnov / MHR magnetics as a coherent line below ~30 kHz with
its toroidal mode number from the probe array, on ECE as a flattened island
signature, and on the radial saddle loops once locked.

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

## Provenance
Detection archive: the PlasmaControl tearing archive (`tm_label`, 8,505 shots
147000-190997 for the CNN's training; 1,503 overlap shots with the FAITH corpus for
evaluation). Models are upstream PlasmaControl artefacts, re-served by labelmaker
with pinned preprocessing. No curated onset table has been placed in `raw/` yet.

## Models
**stable**: d3d_tearing_onset_cnn1d | 2022_12_01

**latest**: d3d_tearing_time_to_event_dsm | 2026_09_05

**all**:
- d3d_tearing_onset_cnn1d | 2022_12_01 (upstream training date; presence at t+25 ms)
- d3d_tearing_time_to_event_dsm | 2026_09_05 (survival forecast; 250 ms / 500 ms / 1 s)
- d3d_tearing_time_to_event_dsm_continued | 2026_09_05 (continued-training variant)

## Alias
- tearing mode
- tearing
- ntm
- neoclassical tearing mode
- 2/1
- 3/2
- locked mode
- magnetic island

## Reference
- R. J. La Haye, "Neoclassical tearing modes and their control", Phys. Plasmas 13,
  055501 (2006).
- H. P. Furth, J. Killeen and M. N. Rosenbluth, "Finite-resistivity instabilities
  of a sheet pinch", Phys. Fluids 6, 459 (1963).

## Contact
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu

## Tables

Inventory row: Tearing Mode; lexicon id: `tearing`.

The scope inventory is [`discrete_labels.csv`](../discrete_labels.csv).
`raw/` holds the untouched provided lists; `format/` holds their
common-schema CSVs and metadata. Each `extend_<model>/` holds one
producing source's output on the project shot list. Categories without
a producer have no `extend_*` directory. See the [table guide](../README.md).

Regenerate registered raw tables from the repository root:

```bash
PYTHONPATH=src python scripts/labelmaker/labels_format.py
```

No raw table is registered for this category yet.
