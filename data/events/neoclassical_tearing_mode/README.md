# Neoclassical Tearing Mode

## Description
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
Jaemin Seo tearing archive (`tm_label`, 8,505 shots 147000-190997 for the CNN's training; 1,503 overlap shots with the FAITH corpus for
evaluation). Models are upstream PlasmaControl artefacts, re-served by labeler with pinned preprocessing. `raw/tm_labels.tar` and `raw/tm_labels.h5` now contain original sampled labels from `/projects/EKOLEMEN/survival_tm_2/`.

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
pixi run -e labelmaker python data/events/neoclassical_tearing_mode/formatter.py
```

Raw sources and the combined output are registered in `../events.yaml`.

## Local formatter and example

[`formatter.py`](formatter.py) reads `../events.yaml` and writes a CSV containing
only `shot,category,t_start,t_end,confidence` under `format/`. Source provenance and
conversion assumptions are stored in its `.meta.json` sidecar. Raw files stay
unchanged. The shared conversion implementation is in
`src/labeler/events/source_formatters.py`.

[`example.ipynb`](example.ipynb) opens the saved per-shot sparse labels and plots the rho–time grid.
It also plots the original annotations through a shared helper and supports
available `extend_*` datasets. For these sources
without radial localization, each time label is broadcast across 20 rho bins.
Use the labelmaker Python environment to rerun it. See the [storage guide](../README.md)
for the sparse per-shot grid format used by extensions.


## Category

The CSV `category` column and grid values use integer IDs. The same mapping
is recorded in each JSON sidecar under `categories`.

| ID | Label |
| --- | --- |
| 0 | Absent |
| 1 | Present |

Unknown or unclassified grid cells are stored separately from 0. A dataset
containing only positive annotations does not establish absence elsewhere.
Sampled grids use 50 ms bins and 20 rho bins.

Per-shot labels are saved in `format/shots/<shot>.npz`. Each file includes
time and rho coordinates, sparse integer values, unknown-cell coordinates,
and the category ID-to-name mapping. The formatted plot reads these saved files. A separate original-label plot
reads the source annotations; neither plot reruns the formatter.

The per-shot grids include all shots with sampled source labels, including
negative-only shots absent from the positive-interval CSV. Samples from HDF5
and the archive are aggregated into 50 ms bins; any positive sample wins.
Bins with supplied zeros remain 0, and unsampled bins remain unknown.

The notebook's last cell plots the category's original annotations alongside
the saved 50 ms grid. Original-label plots require the source files; the
formatted and extended plots continue to read only their selected NPZ files.

## Verification

[`verification.ipynb`](verification.ipynb) plots one shot's signals against its
saved labels and takes back corrections. The review roster is
[`shots.csv`](shots.csv). See the [table guide](../README.md) for the roster
schema.
