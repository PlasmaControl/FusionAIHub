---
title: "IGNITE on MAST: TokEye judges the Mirnov spectrograms"
---

# IGNITE on MAST: TokEye judges the Mirnov spectrograms

**Status: measured (2026-11-24) on all 11,177 cached MAST shots. OMV keeps 9,502 of 10,498 (shot, signal) pairs and CC 5,433 of 7,240; the dropped pairs leave the dynamics loss through `presence_tokeye.json`. TokEye also measures that the Mirnov codec keeps the modes (mask F1 0.93) and that IGNITE's rollouts lose them (F1 0.14, no better than holding the last seed frame), for run 5 and run 6 alike.****

## What this is

dev-peter's advice for the spectrogram tokenizer that works on DIII-D was to find
shots with enough mode content to train it on, to inspect the training data
carefully, and to watch the compression rate. MAST's two Mirnov spectrograms
(TokaMark's `magnetics-b_field_pol_probe_omv_voltage`, three poloidal probes, and
`magnetics-b_field_tor_probe_cc_field`, three toroidal probes) were used whole so
far: every probe of every shot went into the codec fit and into the dynamics loss.
This page adds a judge. TokEye's segmenter (`big_tf_unet`, the weights
`labeler.events.unet` pins by sha256 `4afc3948...`) marks coherent modes and
transients in each probe's spectrogram, and three statistics in time say whether
a probe carries plasma signal at all. The judge's outputs are used three ways:

1. **Keep or drop a (shot, signal) pair in the dynamics loss.** A probe that is
   white noise in time is a target no model can predict, so a signal whose noise
   probes outnumber its good ones leaves the loss (`presence_tokeye.json`, passed
   as `--presence_path` with `--mask_absent`).
2. **Repeat the training windows where modes are active** (`frames.parquet`,
   `port_train.py --tokeye_frames ... --tokeye_factor K`).
3. **Measure whether modes survive** the tokenizer and the rollout: TokEye's mask
   on the truth against its mask on the codec ceiling, on an IGNITE rollout and on
   a persistence baseline (`tokeye_mode_fidelity.py`).

The code is `src/tokamak_foundation_model/ignite/mast/port/specquality.py`
(scores, verdicts, mask agreement), `port/curate.py` (window oversampling) and
the scripts `scripts/mast/tokeye_spec_quality.py` (sweep),
`scripts/mast/tokeye_keep.py` (verdicts) and
`scripts/mast/tokeye_mode_fidelity.py` (agreement). Tests:
`tests/ignite/mast/port/test_specquality.py` and `test_curate.py`, with a
stand-in segmenter, so they pin the bookkeeping, not TokEye's weights.

## The input TokEye sees

Cache v1 stores TokaMark's spectrogram: non-overlapping 512-sample Hann windows of
the 500 kHz probe signal, `|FFT|^2`, 256 bins of 0.977 kHz, one column every
1.024 ms. TokEye's own transform is `log1p|STFT|` (Hann 1024, hop 128), DC
dropped, clipped to the 1st and 99th percentiles and z-scored. The judge applies
the same steps to what the cache holds: `0.5 log P`, DC dropped, the frequency
axis interpolated twice so that a row is 0.488 kHz (TokEye's wide pass), then
the clip and z-score over the valid columns only. Invalid columns (fewer than 90%
finite positive bins) are set to 0 after the z-score, so a gap is flat rather
than a cliff the segmenter could call a transient. The input is reflect-padded
to a multiple of 16 for the U-Net's four pools.

Checked by eye on shots 24934, 23403 and 20384 (2026-11-24): the coherent mask
follows the chirping modes on OMV, and the transient mask lights the ELM bursts.
Upsampling time four times as well gave sharper but more fragmented tracks at
four times the cost, so time stays at 1.024 ms.

**Pickup lines are not modes.** TokEye lights any line, including the stationary
instrument pickup that MAST's Mirnov spectrograms carry before, during and after
the plasma (shot 23403's CC probes: lines at 20, 40 and 80 kHz from 0 to 0.47 s,
while OMV shows chirping modes). A row is pickup if it is lit for at least 30%
of the valid columns outside the plasma window (when there are at least 20 of
them), or for at least 60% of the plasma window's valid columns. Pickup rows are
dilated by two rows and left out of the coherent activity.

## Per-probe statistics and the verdict

For each probe, over the plasma window of the frame cache (the frames where
|Ip| is above the cache's threshold):

| Statistic | Definition | Source |
|---|---|---|
| `valid` | share of plasma frames with a valid column | |
| `ac1` | lag-1 autocorrelation in time of log10 power, mean-removed per bin, longest valid run | dev-peter's `spectro_final_fig.band_stats` |
| `coh5` | share of the variance kept by a 5-column boxcar in time | dev-peter's `coh_frac` |
| `coh` | share of non-pickup rows TokEye lights coherent, per frame | TokEye |
| `contrast_db` | median plasma power over median off-plasma power | |

`ac1` and `coh5` are computed in 8 frequency bands of about 31 kHz, and the
verdict uses the largest band value. White noise gives `ac1 = 0` and
`coh5 = 1/5`. Whole-band numbers were tried first and were wrong for weak but
real probes: on shots 13780 to 16737 an OMV probe whose plasma signal sits below
about 30 kHz reads `ac1` about 0.1 over all 255 bins, because the many bins at
the noise floor dominate, while its lowest band is clearly coherent (contact
sheet, 2026-11-24).

| Class | Rule |
|---|---|
| absent | `valid < 0.2` (no data in the plasma window, or stored as zeros) |
| noise | largest band `ac1 < 0.15` and largest band `coh5 < 0.3` |
| ok | otherwise |

A (shot, signal) pair is **kept** when at least one probe is ok and no more
probes are noise than ok. An absent probe codes a constant, which costs nothing
to predict; a noise probe is a target nothing can predict. A frame is
**mode-active** when the mean TokEye coherent activity of the signal's ok probes
is at least 0.02.

**TokEye rescue (OMV only).** Late MAST shots (about 28,000 on) carry weak OMV
probes: a white background with faint falling chirps for part of the shot. Their
band statistics read noise (`ac1` about 0.1 in every band), but TokEye lights the
chirps. A noise probe is rescued to ok when the 90th percentile of its per-frame
coherent activity is at least 0.02 and the median is at most 0.3 of it, that is,
activity that comes and goes as a mode does. On a contact sheet of 12 random
OMV rescues all 12 show real chirps. The same rule on CC would rescue 290
probes, and its contact sheet shows mostly broadband interference (vertical
bars, bright horizontal bands) and intermittent lines that the pickup rule
misses, with an occasional real mode. On CC, TokEye activity is therefore not
taken as evidence of plasma signal, and CC has no rescue. TokEye's transient
channel does not separate the two (median transient activity 0.001 for the
OMV rescues, 0.004 for the CC candidates).

## Results (measured 2026-11-24, all 11,177 shots)

The sweep (`scores_v2`, 48 CPU workers) took 56 minutes with no errors; 10,867
shots hold at least one Mirnov spectrogram.

| | OMV | CC |
|---|---|---|
| probes ok / noise / absent | 29,058 / 2,328 / 108 | 14,048 / 4,234 / 3,438 |
| noise probes rescued by TokEye | 310 | 0 (no rescue on CC) |
| (shot, signal) kept | 9,502 of 10,498 (90.5%) | 5,433 of 7,240 (75.0%) |
| kept train / val / test | 7,583 / 965 / 954 of 8,386 / 1,064 / 1,048 | 4,339 / 558 / 536 of 5,803 / 732 / 705 |
| train frames kept | 589,433 of 673,820 | 346,618 of 471,512 |
| pairs set absent in `presence_tokeye.json` | 994 | 1,806 |
| median share of mode-active frames, kept signals | 0.32 | 0.18 |

The drops are not spread evenly. OMV keeps 96 to 100% of its pairs below shot
28,000, then 44% for shots 28,000 to 29,999 and 22% from 30,000, where the
probes that are not rescued are white noise. CC keeps nothing below shot 12,000
(two probes stored as zeros, the third white noise), 45% and 25% for shots
20,000 to 23,999 (interference), and 70 to 100% elsewhere.

The frame cache marks a CC signal stored as zeros as present, so until this
page its codes were in the dynamics loss.

**Whole-band statistics were wrong first.** The first sweep (`scores_v1`, the
statistics over all 255 bins) classed about 17% of OMV probes noise, and the
contact sheet showed many of them with signal below 30 kHz. Per-band
statistics bring that to 8.4% before the rescue and 7.4% after it.

### Do the modes survive the tokenizer and the rollout?

`tokeye_mode_fidelity.py` on the 12 val shots with the most OMV mode activity,
3 windows per shot where the predicted frames are mode-active, 32-frame windows
(12 true seed frames, 20 predicted), real actuators, the probes classed ok
(selection from the first, partial keep map, 2026-11-24). `F1` is TokEye's
coherent mask on the truth against its mask on the other spectrogram, one
pixel of slack (0.49 kHz, 1.02 ms). Peak contrast is the median over columns and
32-bin bands of the 95th percentile minus the median of `10 log10 P`.

| OMV, 108 probe-windows | mask F1 | recall | peak contrast |
|---|---|---|---|
| truth | | | 9.1 dB |
| codec ceiling (the true codes decoded) | 0.93 | 0.99 | 6.1 dB |
| last seed frame held (persistence) | 0.15 | 0.26 | 4.6 dB |
| run 6, step 3000, temperature 1.0 | 0.14 | 0.15 | 2.7 dB |
| run 6, step 3000, temperature 0.5 | 0.14 | 0.17 | 3.5 dB |
| run 5, step 30,000, temperature 1.0 | 0.14 | 0.13 | 3.0 dB |

CC (27 probe-windows) reads the same way: ceiling 0.93, persistence 0.12,
rollouts 0.09 to 0.14.

The codec keeps the modes TokEye sees. It is constant within each 5 ms frame,
so a chirp becomes a staircase of 5 ms steps and a third of the peak contrast
goes (9.1 to 6.1 dB). The rollouts lose the modes: within a frame or two of the
seed the predicted spectrum is dim and flat, with half the ceiling's peak
contrast, also when the seed frames carry strong chirps. A model four times
larger (run 6, d1024 x 16, batch 256) and a colder temperature do not change
that, so it is not a question of model size or sampling. Each Mirnov array is
432 independent cosine coefficients per frame; a spectral line is spread over
many of them, and a model that predicts each coefficient's distribution washes
lines out. This is where dev-peter's peak-and-baseline advice points.

## Use in training

Not applied to a run yet (2026-11-24). The presence file and the window
oversampling are wired (`port_train.py`, off unless given) and tested on the
synthetic three-shot cache. They change what the loss sees, not how the
spectrogram is coded, so on their own they cannot fix rollouts that wash the
modes out; they go into the next run together with a new Mirnov coding.

## How to run

```bash
S=/models/intermediate/mast/stage2r
# 1. scores (CPU, 48 workers x 4 threads, ~45 min for 11,177 shots)
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= scripts/mast/py.sh scripts/mast/tokeye_spec_quality.py \
    --out $S/tokeye/scores_v2 --workers 48 --threads 4
# 2. verdicts, keep map, presence file
scripts/mast/py.sh scripts/mast/tokeye_keep.py --scores $S/tokeye/scores_v2
# 3. mode fidelity of the codec and of a checkpoint's rollouts
IGNITE_HISTORY_CACHE=1 scripts/mast/py.sh scripts/mast/tokeye_mode_fidelity.py \
    --keep $S/tokeye/scores_v2 --out $S/tokeye/fidelity_<run>_<step> --ckpt <run>/dynamics_step<N>.pt
# 4. training with the judge (port_train.py options, off unless given)
... port_train.py ... --presence_path $S/tokeye/scores_v2/presence_tokeye.json --mask_absent \
    --tokeye_frames $S/tokeye/scores_v2/frames.parquet --tokeye_factor 2
```
