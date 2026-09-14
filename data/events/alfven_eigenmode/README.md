# Alfven Eigenmode

## Description
Alfvén eigenmodes (AEs) are weakly damped shear-Alfvén waves that live in gaps of
the continuous Alfvén spectrum and are driven unstable by fast ions (beam ions,
fusion alphas, ICRF tails) whose velocity resonates with the wave. The Alfvén speed
is

    v_A = B / sqrt(mu_0 * rho)

and the toroidicity-induced gap mode (TAE) sits near

    f_TAE = v_A / (4 * pi * q * R)

which on DIII-D is roughly 80-250 kHz. Sub-families are named by the gap or the
profile feature that hosts them: TAE (toroidicity), EAE (ellipticity), BAE
(beta-induced, tens of kHz), RSAE (reversed-shear, chirps up as q_min falls).
AEs redistribute and eject fast ions, so they show up as neutron-rate deficits and
beam-ion losses.

First predicted from ideal MHD gap theory (Cheng, Chen and Chance 1985) and observed
on TFTR and DIII-D in 1991.

Typically found via magnetics (Mirnov / MHR probes), CO2 interferometer chords,
ECE and BES as coherent lines in the 80-250 kHz band of a spectrogram; RSAEs are
recognised by their upward chirp, BAEs by their lower frequency.

## Method
Two independent claims, both written by `labelmaker`:

1. **Model label** `d3d_ae_activity_seldnet/ae_active`: per-25 ms probability that
   an AE is present in 80.6-250 kHz of the four CO2 chords (STFT -> log-power ->
   SELDnet, 440,514 parameters). `ae_frequency` is a probability-weighted mean
   frequency and is unvalidated. Against the human annotation alone the model is
   AUROC 0.62 (recall 0.88, precision 0.26), so it is evidence, not truth.
2. **TokEye track event** `tokeye_track` / `coherent_mode` with a band at or above
   40 kHz and bandwidth <= 100 kHz on mhr, ece or co2 (see
   `configs/ideate/phenomena.yaml`, id `ae`).

Nothing above the 250 kHz Nyquist of the 500 kHz groups is observable; a mode there
is reported as quiet, not unknown.

## Provenance
Obtained using reference dataset from William W Heidbrink: 180 hand-annotated DIII-D
shots (170659-178879) with five classes (`lfm`, `bae`, `eae`, `rsae`, `tae`), the
`aemodes` project's annotation cache. The SELDnet was trained on 120 of them and
validated on 60. None of the 180 has a FAITH corpus file, so every corpus label is
on a held-out shot. `raw/` is still empty: the annotation table has not yet been
placed here in its original form.

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
- ae-free

## Reference
- W. W. Heidbrink, "Basic physics of Alfvén instabilities driven by energetic
  particles", Phys. Plasmas 15, 055501 (2008).
- C. Z. Cheng, L. Chen and M. S. Chance, "High-n ideal and resistive shear Alfvén
  waves in tokamaks", Ann. Phys. 161, 21 (1985).

## Contact
- **Alvin Garcia**: alvin [dot] garcia [at] uci [dot] edu
- **Azarakhsh Jalalvand**: aj17 [at] princeton [dot] edu
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
PYTHONPATH=src python scripts/labelmaker/labels_format.py
```

No raw table is registered for this category yet.
