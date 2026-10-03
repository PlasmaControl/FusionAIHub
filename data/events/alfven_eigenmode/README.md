# Alfven Eigenmode

## Description
Alfvén eigenmodes (AEs) are driven unstable by fast ions (beam ions, fusion alphas, ICRF tails) whose velocity resonates with the wave. The Alfvén speed is

$$v_A = \frac{B}{\sqrt{\mu_0 \rho}}$$

Sub-families are named by the gap or the profile feature that hosts them: TAE (toroidicity), EAE (ellipticity), BAE (beta-induced, tens of kHz), RSAE (reversed-shear, chirps up as q_min falls). AEs redistribute and eject fast ions, so they show up as neutron-rate deficits and beam-ion losses.

First predicted from ideal MHD gap theory (Cheng, Chen and Chance 1985) and observed on TFTR and DIII-D in 1991.

Typically found via magnetics (Mirnov / MHR probes), CO2 interferometer chords, and ECE as coherent clusters around the 80-250 kHz band of a spectrogram.

## Data Provenance
### Dataset 1

**Dataset File(s)**: `co2_250_detector.pkl`

**Author**: William W Heidbrink

**Description**: 180 hand-annotated event detection DIII-D shots (170659-178879) with five classes (`lfm`, `bae`, `eae`, `rsae`, `tae`)

**Publications**:

### Dataset 2

**Dataset File(s)**: `ece_all_labels/labels_<shot>.txt` (26 files)

**Author**:

**Description**: 26 hand-annotated DIII-D shots (132240-178872) on ECE spectrograms, not CO2: 1,330 time-frequency boxes (`tae` 441, `rsae` 415, `bae` 329, `lfm` 143, `eae` 2) on ECE channels 1-40, 0-250 kHz, 0.21-1.9 s. Copied unchanged from `/projects/EKOLEMEN/ece_cnn/all_labels/` (files dated 2022-01-24; the annotator is not recorded in them). 13 of the 26 shots are also in Dataset 1. Boxes only: a time with no box is unknown, not absent.

**Publications**:

## Models
**stable**: d3d_ae_activity_seldnet

**latest**: d3d_ae_activity_seldnet

**all**:
- d3d_ae_activity_seldnet | 2026_09_06 | AUROC: 0.982 | AUPRC: 0.992 | F1: 0.944
- d3d_ae_co2_rcn | 2022_10_05 | AUROC: 0.836 | AUPRC: 0.918 | F1: 0.774
- d3d_ae_co2_lstm | 2022_10_04 | AUROC: 0.714 | AUPRC: 0.842 | F1: 0.715
- d3d_ae_co2_rcn_xpow | 2023_03_03 | AUROC: 0.767 | AUPRC: 0.886 | F1: 0.725
- d3d_ae_co2_lstm_xpow | 2022_11_21 | AUROC: 0.699 | AUPRC: 0.825 | F1: 0.700
- ae-ours | legacy supervision | clean 100/20 selection retrain pending
- ae-ours | dense supervision | clean 100/20 selection retrain pending
- ae-ours | threeway supervision | clean 100/20 selection retrain pending

The supervision-swap entries are planned experiments; their SLURM attempts
failed because the required Pixi environment has CPU-only PyTorch. Current
saved-baseline scores, input hashes, and continuation commands are in
[ae_supervision_swap.md](../../../docs/labeler/ae_supervision_swap.md).

Scores are against the owner-reviewed labels, on 10 ms frames of the 19 validation shots the older detectors did not train on (SELDnet at 0.5, RCN at 0.10, LSTM at 0.15; each chord or chord pair a sample). On all 60 validation shots the SELDnet scores AUROC 0.980, AUPRC 0.992, F1 0.951. Against the raw Heidbrink annotation, which under-counts, the order reverses: AUROC 0.68 for the SELDnet and 0.86 to 0.90 for the older detectors. Protocol, caveats and the older detectors' files: [ae_baselines_benchmark.md](../../../docs/labeler/ae_baselines_benchmark.md).

## Inputs
**d3d_ae_activity_seldnet**:
- `CO2` interferometer chords `R0`, `V1`, `V2`, `V3` (4 ch) 

**d3d_ae_co2_lstm**, **d3d_ae_co2_rcn** (Alvin Garcia, UCI):
- `CO2` interferometer chord `R0`, `V1`, `V2` or `V3`, one at a time, 20-250 kHz spectrogram

**d3d_ae_co2_lstm_xpow**, **d3d_ae_co2_rcn_xpow**:
- `CO2` cross-power of a chord pair (10 pairs of `R0`, `V1`, `V2`, `V3`), 20-250 kHz

**ae_xpower (round three)**:
- `CO2 R0×V1`, `CO2 R0×V2`, `CO2 R0×V3` cross-power (from CO2 chords `R0`, `V1`, `V2`, `V3`)

**ae_seg (round three)**:
- `CO2 R0×V1`, `CO2 R0×V2`, `CO2 R0×V3` cross-power (from CO2 chords `R0`, `V1`, `V2`, `V3`)

## Method
1. Raw labels are obtained from `raw/co2_250_detector.pkl`
1. The labels `rase`,`tae`,`bae`,`eae` are collapsed into a single `ae` label and saved to `raw/co2_250_detector_combine.pkl`
1. Labels are adjusted for proper coverage and saved to `raw/co2_250_detector_adjust.pkl`
1. CO2 spectrograms for `r0`,`v0`,`v1`,`v2` made with decimation to 500kHz, stft `window=512,hop=128,taper=hann`.
1. Train classifier and create downstream labels
1. Human supervision re-adjusts labels

Dataset 2 (ECE boxes): the `tae`, `rsae`, `bae` and `eae` boxes of every ECE channel are collapsed into one `ae` span per shot, the union of their time spans (`lfm` ignored), written to `format/alfven_eigenmode_ece_format_2026_v1.csv` with 50 ms grids in `format/ece_shots/`. Nothing is marked absent, and ECE channel and frequency are not kept. Most boxes start at exactly 0.30 s, so a mode already present earlier is cut there. It is a second table beside Dataset 1's; the pinned CO2 table `format/alfven_eigenmode_format_2026_v1.csv` is unchanged.

Benchmark: the older CO2 detectors, Alvin Garcia's LSTM and echo-state network (RCN), are scored beside the SELDnet from their saved predictions, on the 19 validation shots none of them trained on, against both the reviewed labels and the raw annotation (`scripts/labeler/ae_baselines_evaluate.py`; protocol and caveats in [ae_baselines_benchmark.md](../../../docs/labeler/ae_baselines_benchmark.md)).

## Alias
alfven eigenmode, alfvén eigenmode, ae, ae mode, tae, rsae, bae, eae

## Future Implementations
- Seperate RSAE, TAE, BAE, EAE, LFM
- Include radial location

## Reference
- W. W. Heidbrink, "Basic physics of Alfvén instabilities driven by energetic
  particles", Phys. Plasmas 15, 055501 (2008).
- C. Z. Cheng, L. Chen and M. S. Chance, "High-n ideal and resistive shear Alfvén
  waves in tokamaks", Ann. Phys. 161, 21 (1985).

## Contact
- **Alvin Garcia**: alvin [dot] garcia [at] uci [dot] edu
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu
