---
title: "IGNITE on MAST: TokEye judges the Mirnov spectrograms"
---

# IGNITE on MAST: TokEye judges the Mirnov spectrograms

**Status: measured (2026-11-24) on all 11,177 cached MAST shots. OMV keeps 9,502 of 10,498 (shot, signal) pairs and CC 5,433 of 7,240; the dropped pairs leave the dynamics loss through `presence_tokeye.json`. TokEye also measures that the Mirnov codec keeps the modes (mask F1 0.93) and that IGNITE's rollouts lose them (F1 0.14, no better than holding the last seed frame), for run 5 and run 6 alike. A peak + baseline Mirnov codec (`port.peakspec`, dev-peter's advice) keeps the modes nearly as well (F1 0.89) and is in training as run 7.**

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

## A peak + baseline Mirnov codec

`port/peakspec.py` (`PeakSpec`) codes each probe's frame spectrum the way
dev-peter advised for the DIII-D spectrogram tokenizer: a baseline plus peaks.
It works in the same centred log10 power as the band-DCT codec (the `logspec`
fit's floor and mean per bin are reused) and per probe and 5 ms frame keeps:

- **baseline**: the first 24 cosine coefficients of the 256-bin spectrum,
  fitted robustly (a second pass replaces the bins where the first residual tops
  1 dB by the first baseline, so a strong line does not lift its surroundings);
- **peaks**: the bins fall into 32 bands of 8 bins (7.8 kHz). Each band holds at
  most one peak, the highest local maximum of the residual above 1 dB: its
  frequency (sub-bin, a parabola through three bins), its height and its
  Gaussian width (from the width at half height). An empty band codes height 0.

A band slot always means the same band, so a mode drifting in frequency stays in
its slot while it stays in its band. That is the bet against washed-out
rollouts: a sampled peak token always draws a sharp line, where hundreds of
independently sampled cosine coefficients do not. Each array is 3 probes x 120
values = 360 tokens a frame (the band-DCT codec has 432 live ones).

The grids (64 levels, `peak_grids`) are fitted by `fit_peakspec` on 300 training
(shot, signal) pairs per array that the judge keeps, half of them the pairs with
the most mode activity, using only probes classed ok (dev-peter: fit the
spectrogram tokenizer on shots with enough mode content). The fit
(`peakspec_fit_v1`, 2026-11-24, 6 s) found a peak above 1 dB in 95% (OMV) and
97% (CC) of band slots, median height 4.4 and 4.3 dB. It is stored as
`logspec_fit.{npz,json}` with `"codec": "peakspec"`, so a frame-code manifest
records it in `logspec_fit_path` and `logspec.build_logspec_codecs` builds it.

How much of the modes the codec keeps, measured end to end (encode, the
frame-cache codes, decode) on 36 val probes classed ok, 24 OMV and 12 CC, two
thirds with mode activity (2026-11-24):

| mode-rich probes | OMV mask F1 | OMV peak contrast | CC mask F1 | CC peak contrast | values a probe |
|---|---|---|---|---|---|
| truth | | 7.8 dB | | 7.2 dB | |
| frame mean (no coding) | 0.92 | 5.5 dB | 0.88 | 4.9 dB | 256 |
| band-DCT codec (`LogBandDCT`) | 0.93 | 5.3 dB | 0.88 | 4.7 dB | 144 |
| `PeakSpec` | 0.89 | 5.3 dB | 0.86 | 4.7 dB | 120 |

The peak codec loses a few points of mask F1 at the same peak contrast and the
same median |dlog10| (0.44 against 0.43 on OMV). Fewer bands cost more
(prototype, 16 bands: F1 0.82). Run 7 tests whether rollouts on it keep the
modes.

## Use in training

**Run 7** (`runs/v1b_peakspec_cam_live_da_f32_d512L12`, started 2026-11-24
20:42 UTC on GPUs 2-3) is run 5's recipe (d512 x 12, 32-frame windows, 30k
steps) on the frame cache `frame_codes_v1b_peakspec_cam_live_da` (`PeakSpec`
for both Mirnov arrays, the D-alpha level + spikes codec, the camera codes;
1,396 tokens a frame), with the judge: `--presence_path presence_tokeye.json
--mask_absent` and `--tokeye_frames frames.parquet --tokeye_factor 2`. The
oversampling adds two copies of each of the 249,348 mode-active training
windows (at least 4 of 32 frames mode-active), so 78% of the 961,093 training
windows are mode-active (54% without it). Run 7 therefore changes the coding and
what the loss sees at once; it is compared with run 5 on the mode fidelity of
its rollouts, not on loss.

### Run 7 at step 5,000, and why it was stopped

Mode fidelity with the final keep map (12 val shots, 3 windows each, the probes
classed ok; 102 OMV probe-windows where the rollout decodes), measured
2026-11-24. "Mode share" is the fraction of pixels TokEye marks coherent in the
spectrogram compared (truth 0.085), a realism check that does not ask where the
modes are.

| OMV | mask F1 | rollout beats persistence | mode share | peak contrast |
|---|---|---|---|---|
| truth | | | 0.085 | 12.6 dB |
| codec ceiling (band-DCT) | 0.957 | | 0.100 | 9.7 dB |
| last seed frame held (persistence) | 0.189 | | 0.159 | 6.0 dB |
| run 5, step 30,000, temperature 1.0 | 0.163 | 44 of 102 | 0.116 | 5.8 dB |
| run 7, step 5,000, temperature 1.0 | 0.095 | 16 of 102 | 0.036 | 3.8 dB |

Paired on the same windows, run 7 is 0.049 below run 5 in mask F1 (95% shot
bootstrap −0.095 to −0.010) and better on 28 of 102. Its rollouts are dim: a
third of run 5's mode share. CC reads the same way (0.055 against 0.144). Over
all 40 signals its median rollout NRMSE at step 5,000 is 0.180, against run 5's
0.142 at the same step (better on 15 of 40).

Why the peak codec does not help: its peak slots are filled whether or not a
mode is there. On 60 val shots with OMV (4,619 complete frames of the
`frame_codes_v1b_peakspec_cam_live_da` cache), a band holds a peak in 97.9% of
quiet frames (TokEye mode share under 0.02) and 95.8% of mode-active ones, with
median heights of 0.40 and 0.45 decade; 27% and 39% of bands hold a peak of 0.5
decade or more. The 1 dB threshold sits far below the bumps of a 5 ms frame mean,
so most of the 96 peak slots a frame carry noise, which no model can predict,
and a higher threshold alone would still keep a quarter of the bands on quiet
frames. Telling modes from noise would need persistence across frames or a
TokEye-informed gate in the codec. Run 7 was stopped at step about 5,800
(ruling R25; checkpoints kept).

Run 5's rollouts are not empty of modes either: TokEye lights more of them than
of the truth (0.116 against 0.085), but in other places and with less contrast
than the codec allows (5.8 against 9.7 dB).

**Run 8** (`runs/v1b_logspec_cam_live_da_tokeye_f32_d512L12`, started 21:52
UTC) is run 7's launcher on `frame_codes_v1b_logspec_cam_live_da`: the band-DCT
Mirnov codec of runs 5 and 6, the D-alpha level + spikes codec, and the same
TokEye selection (the presence file was built on this cache; same 961,093
windows). Run 7 against run 8 at equal steps isolates the Mirnov codec; run 8
against run 5 is D-alpha v2 plus the selection.

### Decode options (no retraining)

dev-peter's sampler has options that act only at rollout time.
`tokeye_mode_fidelity.py` passes them through, all off by default: `--gate`,
`--consensus-n/-q/-mode` and `--decode-steps` (MaskGIT unmasking steps per frame;
0 keeps the checkpoint's, 10). On run 5's final checkpoint, same windows as
above:

| OMV, run 5 step 30,000 | mask F1 | rollout beats persistence | identical to persistence | mode share | peak contrast |
|---|---|---|---|---|---|
| temperature 1.0 (as trained and plotted) | 0.163 | 44 of 102 | 0 | 0.116 | 5.8 dB |
| temperature 0.7 | 0.177 | 44 of 99 | 0 | 0.117 | 6.1 dB |
| temperature 0.5 | 0.176 | 45 of 102 | 0 | 0.118 | 6.1 dB |
| gate 0.2 | 0.189 | 27 of 101 | 57 | 0.159 | 6.0 dB |
| gate 0.5 | 0.189 | 15 of 102 | 83 | 0.155 | 6.0 dB |
| 30 unmasking steps a frame (10 trained) | 0.181 | 45 of 102 | 0 | 0.116 | 6.0 dB |

The gate keeps the previous frame's code wherever the sampled code had a
probability under the gate. At Mirnov positions almost every code is that
uncertain, so the gated rollout is the last seed frame held, in 57 and 83 of 102
windows exactly. Once it samples the missing code it holds that too (shot 20738,
the spectrum blanks from 212 ms on). A colder temperature moves mask F1 by about
0.01 and peak contrast by 0.3 dB, still under persistence. Three times as many
unmasking steps (each code then conditioned on more already-drawn codes of its
frame) gives the same: paired +0.007 against 10 steps (95% shot bootstrap −0.002
to +0.017). Sampling is not what keeps the rollouts off the measured modes.

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
#    decode options, off by default: --temperature 0.7 --gate 0.2 --consensus-n 4 --decode-steps 30
# 4. the peak + baseline Mirnov codec and a frame cache with it (CPU, about 3 minutes)
scripts/mast/py.sh scripts/mast/port_fit_peakspec.py --frames-dir $S/frame_codes_v1b_logspec_cam_live_da \
    --keep-dir $S/tokeye/scores_v2 --out $S/peakspec_fit_v1 --workers 32
C=/models/intermediate/mast/camera
CUDA_VISIBLE_DEVICES= scripts/mast/py.sh scripts/mast/port_build_cache.py --logspec-fit $S/peakspec_fit_v1 \
    --dalpha-fit $S/dalpha_fit_v1 --camera-codec $C/codec_rgb_v2/codec_final.pt --camera-fit $C/camera_fit_v1 \
    --camera-frames $C/frames_rgb_v1 --out $S/frame_codes_v1b_peakspec_cam_live_da --workers 48 --live-only
# 5. training with the judge (port_train.py options, off unless given); run 8's launcher:
#    $S/runs/v1b_logspec_cam_live_da_tokeye_f32_d512L12/launch.sh (run 7's: .../v1b_peakspec_cam_live_da_f32_d512L12/)
... port_train.py ... --presence_path $S/tokeye/scores_v2/presence_tokeye.json --mask_absent \
    --tokeye_frames $S/tokeye/scores_v2/frames.parquet --tokeye_factor 2
```
