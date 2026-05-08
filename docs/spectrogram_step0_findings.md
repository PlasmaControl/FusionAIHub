# Phase B Step 0 — Data Verification Findings

**Date:** 2026-05-06
**Shots inspected (5):** 200003, 200004, 200005, 200006, 200007
**Generator:** `inspect_spectrograms/step0_inspect.py`
**Figures:** `../inspect_spectrograms/figures/` (relative to this doc)

This is the documentation artefact for Phase B Step 0 of
`docs/spectrogram_tokenizer_plan.md`. Re-running `step0_inspect.py`
overwrites this file in place (and refreshes the figures).

---

## Confirmed shapes

| modality | C (sliced) | observed shape (C, F, T) | matches plan `[C, 512, 98]`? |
|---|---:|---|:---:|
| ece | 40 | (40, 512, 98) | ✓ |
| co2 | 4 | (4, 512, 98) | ✓ |
| bes | 16 | (16, 512, 98) | ✓ |

All five shots produced identical shapes per modality. Axis order is
`(channels, frequency, time)` — DC bin removed by the data loader,
512 freq bins, 98 STFT time frames at `n_fft=1024, hop=256` on a
50 ms × 500 kHz window. The plan's earlier `[94, 513]` /
`(time, freq)` claim was wrong on all three counts and was
corrected.

## Per-channel preprocessing-stats sanity

| modality | C in stats | NaN(mean) | NaN(std) | std min | std max |
|---|---:|---:|---:|---:|---:|
| ece | 40 | 0 | 0 | 0.1245 | 0.1954 |
| co2 | 4  | 0 | 0 | 0.6263 | 0.7038 |
| bes | 16 | 0 | 0 | 0.1355 | 0.2423 |

Sanity-checked against
`/projects/EKOLEMEN/foundation_model/preprocessing_stats.pt` for the
16 selected BES channels (`slice(48, 64)`). No NaN, no zero-std.
ECE and BES log-stats sit in nearly identical ranges; CO2 sits on a
different log scale (mean ≈ 12 vs ≈ 0.2 for ECE/BES) — fine for
training because `log_standardize` flattens the per-channel
distribution to ~unit variance globally, and per-batch
standardisation in the trainer flattens per-window distributions
on top of that.

## Modality presence at the shot level

Across 50 random shots:
- ECE present: **94 %**
- CO2 present: **44 %**
- BES present: **36 %**

Only ~36 % of shots have all three. The plan's earlier "no missing
data" assumption was wrong at the shot level. Per-modality
`<name>_valid > 0` indicators are emitted by the data loader and
routed through to the model's missing-modality token (Phase C
tangtv pattern). Spectrogram loss is excluded for absent modalities.

## Figures

- Per-shot panels (1 s window, all channels stacked, log-magnitude):
  - `200003_ece.png`, `200003_co2.png`, `200003_bes.png`
  - `200004_…`, `200005_…`, `200006_…`, `200007_…` (15 files total)
- `freq_energy.png` — per-frequency mean log-magnitude averaged over
  channels, time, and shots.
- `bes_correlation.png` — pairwise correlation between BES 16
  channels' time-averaged log-magnitude spectra; black lines split
  the proposed 49–56 vs 57–64 spatial rows.

All paths relative to `../inspect_spectrograms/figures/`.

## Resolved status — open questions (closed 2026-05-06)

1. **Frequency cutoff: keep full 0–250 kHz range.**
   `freq_energy.png` does show ECE/BES energy concentrated below
   ~50 kHz with a flat-ish noise floor above and faint features at
   ~130 kHz and ~210 kHz. Cropping the freq axis was considered as
   an optimisation (could reduce tokens by ~80 %) but rejected — the
   high-frequency features may be physics, and the model can learn
   to suppress noise channels through standardisation. Token budget
   stays at 96 / 192 / 192 (CO2 / ECE / BES).

2. **BES grid orientation: not applicable.**
   The BES array is moved radially per session and channel
   configurations vary by session-leader request, so
   channel-to-(R, Z) mapping is non-stationary across shots. There
   is no fixed (R, Z) orientation to align to; (R, Z) is not in the
   dataset. The plan's Step 0 row-major / column-major checkbox is
   marked **n/a** — the Conv3d fallback (Risk #4) would have to use
   logical adjacency only, not physical layout.

3. **Physics features visible: yes.**
   Per-shot panels show coherent low-frequency content for ECE and
   BES; CO2 shows persistent horizontal banding across all 4
   channels (visible only at the 1 s window — the 50 ms training
   window is too narrow). Confirms the spectrograms carry
   real plasma signal, not just noise.

## BES anomaly note (informational, not actionable)

The 5 inspected shots all show channels 50 (1-indexed 51, 3rd row in
the panel) and 57 (1-indexed 58, 10th row) with distinctly lower
correlation to their neighbours in `bes_correlation.png`. Per
discussion with domain expertise: BES has campaign-dependent dead
channels even within the historically-safe 49–64 selection.
`log_standardize` flattens the amplitude difference, so these
channels train through without runtime detection. No code-level
mitigation needed.
