# AE benchmark: the older CO2 detectors

Status: **run 2026-10-01.** The owner asked for the CO2 detectors that came before the
SELDnet (an LSTM and an "RCN", the echo state network) to be benchmarked against it, and
the numbers added to the AE README. Script: `scripts/labeler/ae_baselines_evaluate.py`
(from the repository root, `pixi run --frozen -e labelmaker python
scripts/labeler/ae_baselines_evaluate.py`); record:
`outputs/labeler/ae/baselines/evaluation.json` (git `2b4da1d`).

## The older detectors

They are **Alvin Garcia's** (UCI), in `/projects/EKOLEMEN/agarcia/df2/`, not the Aza models
the owner first suspected: Aza's CO2 weights (`rcn_co2_spectra_aza.pkl`,
`weights_nathan.pkl`, named in the notebook `aemodes/.archive/rcn/rcn_co2.ipynb`) are not on
disk, and the Aza-made RCNs that are (`/projects/EKOLEMEN/pcs-diiid-ae-rcn-aza/`) read ECE,
not CO2. No CO2 CNN exists, only a crashed prototype.

| | LSTM | RCN |
|---|---|---|
| Net | Keras SavedModel (TensorFlow 2.1, Oct 2022): 20 spectrogram bins in, 3 x LSTM(64), dropout 0.5, 3 x Dense(128), 5 sigmoid outputs (eae, tae, rsae, bae, lfm); 254,213 parameters; BCE, Adam 1e-4, 200 epochs | PyRCN `ESNClassifier` pickles (scikit-learn 0.23): two layers, 4,000 to 16,000 reservoir neurons in the first and 500 in the second, hyperparameters those of the paper's excerpt |
| Versions here | spectrogram 2022_10_04, cross-power 2022_11_21 | spectrogram 2022_10_05, cross-power 2023_03_03 |
| Input | one CO2 chord (R0, V1, V2 or V3), 20-250 kHz, 142 bins of 14.1 ms over 0-2 s; or the cross-power of a chord pair (10 pairs) | the same |
| Output grid | 28 bins of about 70 ms | 142 (spectrogram) or 141 (cross-power) bins of 14.1 ms |
| Operating point | 0.15 (Alvin's notebook; 0.10 and 0.16 elsewhere) | 0.10 |
| Training | random 75/25 split over shots: 801 shots train, 268 validation; labels are the UCI time-point flags widened to +-125 ms, the same on every chord, five classes | same |

Neither can be run here (no TensorFlow, no scikit-learn in any environment), so what is scored
is what they saved: their predictions on their own 268 validation shots
(`results_dp/spec/` and `results_dp/xpow/`, in the order of `shots_val_7525.npy`, and, for the
cross-power rows, of the last 2,680 entries of `xpow_shots_chords.npy[irand_7525_xpow.npy]`).
The order was checked: the LSTM's saved truth is identical across a shot's rows (spread 0), and
the RCN output agrees with the LSTM's more on the same row than on a shuffled one (r
0.76 against
0.65 for the spectrogram,
0.73 against
0.60 for cross-power), a weak check
because every shot's AE output has much the same shape. Their own score,
`performance.csv`, uses a tolerant match (about +-280 ms for the LSTM) over five classes
(LSTM TPR 0.94 / FPR 0.11, RCN 0.90 / 0.10) and is not comparable with the frame scores below.

## Protocol

- **Shots.** The SELDnet's 60 validation shots (170659-178879) that are in the older detectors'
  validation set: **19**, the only ones neither trained on. The other 41 were in their
  training set and have no saved predictions; the SELDnet scored on all 60 is shown for reference.
  The 19: 170660, 170661, 170663, 170666, 170669, 170677, 170678, 170718, 170725, 170729, 170730, 170792, 170793, 170798, 170801, 170803, 175241, 175245, 178879.
- **Frames.** The owner's 10 ms frames of 0-2 s (`labeler.scoring.frames`), as in the ae_xpower
  test. SELDnet probabilities (0.256 ms) are averaged in a frame, and its hard call is half the
  columns >= 0.5 (reproducing the ae_xpower v2 test on all 60 shots: P 0.974, R 0.928, F1 0.951).
  An older detector's value is that of the bin nearest the frame's centre. Its AE activity is the
  largest output over eae, tae, rsae and bae; LFM is left out, as in ours.
- **Truth 1, reviewed.** The owner-reviewed AE labels (`review/labels.csv`, the snapshot the
  ae_xpower v2 test used), frames called present or absent: 68% present on these shots.
- **Truth 2, annotated.** The Heidbrink hand annotation (`annotated`, labels 1-4) over all frames:
  21% present. It under-counts, so an unannotated frame is not an absent one: precision is
  not an error rate, and a detector that finds every real mode scores worse than one that
  learned the annotator's selectivity.
- **Scores.** AUROC and AUPRC (average precision) of the continuous output; precision, recall and
  F1 at the detector's own operating point (SELDnet 0.5); the best F1 over every threshold on
  these frames (tuned on the test frames, so optimistic). 95 % intervals bootstrap the shots
  (2000 replicates, seed 20260923). An older detector's "each" rows score every
  chord (or pair) of a shot as a sample, as their own evaluation does; "averaged" rows average a
  shot's inputs first.

## Results, against the reviewed labels

| Method | Frames | AUROC | AUPRC | Precision | Recall | F1 | Best F1 |
|---|---|---|---|---|---|---|---|
| SELDnet (`d3d_ae_activity_seldnet`) | 3,800 | 0.982 [0.96, 0.99] | 0.992 [0.98, 1.00] | 0.982 | 0.908 | 0.944 [0.90, 0.98] | 0.948 |
| LSTM, chord spectrogram (each chord) | 15,200 | 0.714 [0.65, 0.77] | 0.842 [0.82, 0.87] | 0.785 | 0.656 | 0.715 [0.68, 0.75] | 0.838 |
| LSTM, chord spectrogram (4 chords averaged) | 3,800 | 0.736 [0.66, 0.80] | 0.855 [0.84, 0.89] | 0.798 | 0.686 | 0.738 [0.69, 0.79] | 0.851 |
| RCN, chord spectrogram (each chord) | 15,200 | 0.836 [0.79, 0.88] | 0.918 [0.89, 0.95] | 0.893 | 0.682 | 0.774 [0.73, 0.81] | 0.849 |
| RCN, chord spectrogram (4 chords averaged) | 3,800 | 0.862 [0.81, 0.91] | 0.930 [0.91, 0.96] | 0.908 | 0.699 | 0.790 [0.74, 0.84] | 0.871 |
| LSTM, chord-pair cross-power (each pair) | 38,000 | 0.699 [0.61, 0.77] | 0.825 [0.80, 0.86] | 0.793 | 0.626 | 0.700 [0.65, 0.74] | 0.829 |
| LSTM, chord-pair cross-power (10 pairs averaged) | 3,800 | 0.714 [0.61, 0.80] | 0.841 [0.81, 0.88] | 0.772 | 0.629 | 0.693 [0.65, 0.74] | 0.841 |
| RCN, chord-pair cross-power (each pair) | 38,000 | 0.767 [0.72, 0.82] | 0.886 [0.86, 0.92] | 0.859 | 0.628 | 0.725 [0.68, 0.77] | 0.816 |
| RCN, chord-pair cross-power (10 pairs averaged) | 3,800 | 0.778 [0.73, 0.84] | 0.896 [0.87, 0.93] | 0.857 | 0.660 | 0.746 [0.69, 0.80] | 0.827 |
| every frame present | 3,800 | 0.500 | 0.683 | 0.683 | 1.000 | 0.812 | 0.812 |
| SELDnet on all 60 validation shots (reference) | 12,000 | 0.980 | 0.992 | 0.974 | 0.928 | 0.951 | 0.954 |

## Results, against the Heidbrink annotation

| Method | Frames | AUROC | AUPRC | Precision | Recall | F1 | Best F1 |
|---|---|---|---|---|---|---|---|
| SELDnet (`d3d_ae_activity_seldnet`) | 3,800 | 0.680 [0.54, 0.80] | 0.351 [0.19, 0.55] | 0.297 | 0.894 | 0.445 [0.34, 0.57] | 0.448 |
| LSTM, chord spectrogram (each chord) | 15,200 | 0.857 [0.81, 0.89] | 0.598 [0.46, 0.71] | 0.347 | 0.946 | 0.508 [0.43, 0.58] | 0.599 |
| LSTM, chord spectrogram (4 chords averaged) | 3,800 | 0.898 [0.85, 0.93] | 0.668 [0.52, 0.78] | 0.354 | 0.992 | 0.522 [0.44, 0.59] | 0.647 |
| RCN, chord spectrogram (each chord) | 15,200 | 0.896 [0.86, 0.93] | 0.667 [0.53, 0.77] | 0.391 | 0.975 | 0.559 [0.46, 0.65] | 0.651 |
| RCN, chord spectrogram (4 chords averaged) | 3,800 | 0.911 [0.87, 0.94] | 0.691 [0.55, 0.80] | 0.399 | 1.000 | 0.570 [0.47, 0.66] | 0.676 |
| LSTM, chord-pair cross-power (each pair) | 38,000 | 0.867 [0.83, 0.89] | 0.558 [0.43, 0.66] | 0.376 | 0.968 | 0.541 [0.46, 0.62] | 0.608 |
| LSTM, chord-pair cross-power (10 pairs averaged) | 3,800 | 0.892 [0.86, 0.92] | 0.607 [0.47, 0.72] | 0.374 | 0.995 | 0.544 [0.45, 0.62] | 0.654 |
| RCN, chord-pair cross-power (each pair) | 38,000 | 0.867 [0.82, 0.91] | 0.585 [0.43, 0.73] | 0.393 | 0.935 | 0.553 [0.45, 0.64] | 0.614 |
| RCN, chord-pair cross-power (10 pairs averaged) | 3,800 | 0.886 [0.83, 0.93] | 0.616 [0.46, 0.77] | 0.388 | 0.974 | 0.554 [0.45, 0.65] | 0.645 |
| every frame present | 3,800 | 0.500 | 0.209 | 0.209 | 1.000 | 0.346 | 0.346 |
| SELDnet on all 60 validation shots (reference) | 12,000 | 0.625 | 0.275 | 0.260 | 0.882 | 0.402 | 0.407 |

## Reading them

- On the owner's reviewed labels the SELDnet is far ahead of every older detector (AUROC about 0.98
  against 0.70 to 0.86), and no older detector's F1 at its own threshold reaches the 0.81 of
  calling every frame present, because 68 % of these frames are present. Their best-threshold F1
  (0.82 to 0.87) is only a little above it. The best older one is the RCN on the chord spectrogram.
- Against the raw annotation the order reverses (AUROC 0.68 for the SELDnet, 0.86 to 0.91 for the
  older ones). The annotation under-counts (the SELDnet's model card says so of its own low
  precision there), and the older detectors were trained on labels of this kind (the UCI time
  points, widened to +-125 ms), so they probably follow where annotators marked a mode more closely
  than the SELDnet, which responds to every active mode. The reviewed labels exist because of that
  under-count; both truths are reported.
- Averaging an older detector's chords or pairs helps a little; cross-power does not beat the
  chord spectrogram here.
- Nineteen shots is small; intervals are wide (shots are the unit). The older detectors predict on
  14 to 70 ms bins from a +-125 ms label convention, so a 10 ms frame score is harder for them than
  the task they were built for.

## Not done

- The `pl` signal variants (`results_pl/`), the 2024 `results_pl_new/`, and the small-window LSTM:
  same recipe, other CO2 signal variants.
- Re-running or re-training the older architectures on the SELDnet's 120 training shots (the fair
  same-data comparison); it would need TensorFlow and PyRCN, or a rewrite in PyTorch and numpy.
- The ECE-based RCNs (`pcs-diiid-ae-rcn-aza/`, `ece-reservoircomputing-nf2021`): different input.
