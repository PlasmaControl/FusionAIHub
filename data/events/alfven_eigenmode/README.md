# Alfven Eigenmode

## Description
Alfvén eigenmodes (AEs) are weakly damped shear-Alfvén waves and are driven unstable by fast ions (beam ions, fusion alphas, ICRF tails) whose velocity resonates with the wave. The Alfvén speed is

$$v_A = \frac{B}{\sqrt{\mu_0 \rho}}$$

and the toroidicity-induced gap mode (TAE) sits near

$$f_{TAE} = \frac{v_A}{4\pi qR}$$

which on DIII-D is roughly 80-250 kHz. Sub-families are named by the gap or the profile feature that hosts them: TAE (toroidicity), EAE (ellipticity), BAE (beta-induced, tens of kHz), RSAE (reversed-shear, chirps up as q_min falls). AEs redistribute and eject fast ions, so they show up as neutron-rate deficits and beam-ion losses.

First predicted from ideal MHD gap theory (Cheng, Chen and Chance 1985) and observed on TFTR and DIII-D in 1991.

Typically found via magnetics (Mirnov / MHR probes), CO2 interferometer chords, and ECE as coherent clusters around the 80-250 kHz band of a spectrogram.

## Method
1. Raw (`raw/co2_250_detector.pkl`) are collpased (`rase`,`tae`,`bae`,`eae`) into (`raw/co2_250_detector_combine.pkl`).
2. Adjusted labels for proper coverage (`raw/co2_250_detector_adjust.pkl`).
3. CO2 spectrograms for `r0`,`v0`,`v1`,`v2` made with decimation to 500kHz, stft `window=512,hop=128,taper=hann`.
4. Train classifier and create labels
6. If label appears on any channel, mark positive for AE.

## Provenance
Obtained using reference dataset from William W Heidbrink: 180 hand-annotated DIII-D shots (170659-178879) with five classes (`lfm`, `bae`, `eae`, `rsae`, `tae`) in `raw/co2_250_detector.pkl`.

## Models
**stable**: d3d_ae_activity_seldnet | 2026_09_06

**latest**: d3d_ae_activity_seldnet | 2026_09_06

**all**:
- d3d_ae_activity_seldnet | 2026_09_06 (ours; `threeway_sce` run, SLURM 2924037)

## Alias
- alfven eigenmode
- alfvén eigenmode
- ae
- ae mode
- tae
- rsae
- bae
- eae

## Future Implementations
- Seperate RSAE, TAE, BAE, EAE, LFM
- Include rho (location) using ECE

## Reference
- W. W. Heidbrink, "Basic physics of Alfvén instabilities driven by energetic
  particles", Phys. Plasmas 15, 055501 (2008).
- C. Z. Cheng, L. Chen and M. S. Chance, "High-n ideal and resistive shear Alfvén
  waves in tokamaks", Ann. Phys. 161, 21 (1985).

## Contact
- **Alvin Garcia**: alvin [dot] garcia [at] uci [dot] edu
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu

## Tables

Inventory row: AE Mode; lexicon id: `ae`.

The scope inventory is [`discrete_labels.csv`](../discrete_labels.csv).
`raw/` holds the untouched provided lists; `format/` holds their
common-schema CSVs and metadata. Each `extend_<model>/` holds one
producing source's output on the project shot list. Categories without
a producer have no `extend_*` directory. See the [table guide](../README.md).

Regenerate registered raw tables from the repository root:

```bash
pixi run -e labelmaker python data/events/alfven_eigenmode/formatter.py
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

The original pickle has no timestamps. Its time mapping is explicitly reconstructed
as uniform bins over the upstream reader's 0–2 s window, configurable with
`--start-s` and `--stop-s`. The four AE classes are combined into binary presence/absence, excluding LFM.
Any positive sample makes its 50 ms bin category 1; other annotated bins are 0. This is not a recovered STFT time axis.


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

The notebook's last cell plots the category's original annotations alongside
the saved 50 ms grid. Original-label plots require the source files; the
formatted and extended plots continue to read only their selected NPZ files.

## Verification

[`verification.ipynb`](verification.ipynb) plots one shot's signals against its
saved labels and takes back corrections. The review roster is
[`shots.csv`](shots.csv). See the [table guide](../README.md) for the roster
schema.
