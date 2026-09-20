---
title: The DIII-D corpus
sidebar_position: 1
---

The FAITH corpus is a set of per-shot HDF5 files from the DIII-D tokamak.
Each file holds one shot's diagnostics, actuators and video, at their native
sampling rates, plus whatever text (operator logbook entries, run
summaries) is associated with that shot. `shot_design` and the labeler
detectors read from it directly; the foundation-model training pipeline
(`src/tokamak_foundation_model/data/data_loader.py`) turns it into fixed-
duration training chunks.

## File layout

Every signal group in a shot file carries `xdata` (timestamps, in
milliseconds) and `ydata` (the values). `TokamakH5Dataset`
(`data_loader.py`) reads a shot file and, for each requested modality, runs
the same pipeline: load raw data at native rate, optionally compute an STFT
magnitude spectrogram, resample to a common `target_time_frames` grid
(linear interpolation for time series, trilinear for video), then apply a
configured preprocessing transform (`standardize`, `normalize`,
`log_standardize`, `log`, or `none`).

Two modes read this data:

- **Standard mode** returns a flat `{modality_name: tensor}` dict over one
  `[t_start, t_start + chunk_duration_s)` window.
- **Prediction mode** loads `chunk_duration_s + prediction_horizon_s`
  seconds, processes it as one window, then splits the result into
  `{"inputs": {...}, "targets": {...}}` at the prediction horizon.

All signal tensors follow `(batch, channels, freq_bins_or_1, time_frames)`:
spectrograms after STFT are `(B, C, n_fft//2+1, 100)`, time series are
`(B, C, 1, 100)` (a singleton frequency dimension), and video is
`(B, T, H, W)` grayscale.

## Modalities

Ten signal groups, three video diagnostics, and text:

| Group | Signals | Channels | Processing |
|-------|---------|----------|------------|
| Spectrograms | mhr(8), ece(48), co2(4) | STFT → 2D | `log_standardize` or `standardize` |
| Actuators | gas(5), ech(11), pin(8), tin(8) | 1D timeseries | `standardize` |
| Diagnostics | d_alpha(6), mse(69), ts_core_density(44) | 1D timeseries | `standardize` or `none` |
| Video | bolo(80×120), irtv(513×640), tangtv(240×720) | grayscale @ 50fps | trilinear resampling |
| Text | log entries | string | DistilBERT tokenization |

This is the layout `dummy_model.py`'s `MultiModalTokamakModel` and
`MultiModalPredictionModel` encode against, and the layout
`compute_preprocessing_stats()` computes normalization statistics for
(saved to `data/preprocessing_stats.pt`). The IGNITE dynamics model (see
[IGNITE](../models/ignite.md)) reads a different, tokenized view of a
mostly-overlapping signal set — its modality table (14 modalities in the v2
generation, 15 with `mirnov` in v4) is defined independently in
`configs/shot_design/ignite_modalities.yaml`, not derived from this table.

## Where it lives

| Location | What |
|---|---|
| `/scratch/gpfs/EKOLEMEN/foundation_model` (Stellar) | per-shot HDF5, `<shot>_processed.h5`, 16,911 shots, 53 TB |
| `/lustre/orion/fus187/proj-shared/foundation_model` (Frontier) | the same corpus, already present |
| `/scratch/gpfs/EKOLEMEN/big_d3d_data/foundation_model_text` (Stellar) | logbook / text corpus, 27 GB |
| `/lustre/orion/fus187/proj-shared/foundation_model_text` (Frontier) | the Frontier `shotsummary` text bundles `shot_design` reads directly (no `sql/` import there) |
| `configs/shot_list/` | named shot lists consumed by the data-generation Hydra configs (`train_debug`, `train_small`, `train_medium`, `train_full`, `validation`) |
| `configs/shot_design/shot_lists/` | shot lists consumed by `shot_design corpus select` (`recommender_v1` on Stellar, `recommender_frontier_v1` on Frontier — see [Database build](../shot-design/database-build.md)) |
| `data/preprocessing_stats.pt` | normalization statistics computed by `compute_preprocessing_stats()` |

`shot_design`'s own corpus-facing commands (`corpus scan`, `corpus summary`,
`corpus select`) census this same HDF5 store and its text bundles into
`db/corpus_coverage.parquet`; see
[Database build](../shot-design/database-build.md) for that pipeline and
[Adding a cluster](../clusters/adding-a-cluster.md) for how
`SHOT_DESIGN_CORPUS` points at the right root per cluster.
