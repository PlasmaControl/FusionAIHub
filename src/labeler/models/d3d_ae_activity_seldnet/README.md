---
language: en
license: other
library_name: pytorch
pipeline_tag: audio-classification
tags:
  - diii-d
  - tokamak
  - alfven-eigenmode
  - interferometry
  - seldnet
  - spectrogram
datasets:
  - plasmacontrol/d3d-faith-corpus
metrics:
  - roc_auc
model-index:
  - name: d3d-ae-activity-seldnet
    results: []
labelmaker:
  status: implemented
  slug: d3d_ae_activity_seldnet
  card_id: plasmacontrol/d3d-ae-activity-seldnet
  framework: torch_pt
  time_step_ms: 25.0
  ensemble_n: 1
  upstream:
    path: /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/d3d_ae_activity_seldnet
    trained: 2026-09-06
    training_code: scripts/labelmaker/ae_train.py
    reference_harness: scripts/labelmaker/ae_evaluate.py
    artifacts:
    - ae_seldnet_threeway_sce.pt
    sha256:
      ae_seldnet_threeway_sce.pt: f2d5314a7673a53c49e287cdbb2d5e7232bbf192b8ba830d3215c8c9124079ab
    notes: >-
      Ours, not a third party's: the network is src/labelmaker/ae/model.py (AeSeldNet,
      440,514 parameters, pool_sizes (6, 2, 29)) and the checkpoint is task 7b's
      `threeway_sce` run (SLURM job 2924037, best epoch 6 of 12 run, early stopped,
      val_loss 0.4065, git_sha 4779ee6). Trained on task 7a's dataset: the 180
      hand-annotated DIII-D shots 170659-178879, 120 train / 60 validation, split as the
      upstream aemodes cache splits them. Those 180 shots are committed beside this card
      as training_shots.txt; NONE of them has a FAITH corpus file (checked shot by shot
      against /scratch/gpfs/EKOLEMEN/foundation_model/<shot>_processed.h5 - 0 of 180
      hits), so every corpus label this model writes is on a held-out shot.
  inputs:
  - co2_density <- co2
  outputs:
  - name: ae_active
    task: binary
    activation: none
    units: ''
  - name: ae_frequency
    task: regression
    activation: none
    units: 'kHz'
  approximations:
  - >-
    250 kHz is the instrument's ceiling, not a modelling choice. The CO2 record is
    sampled at 500 kHz (measured 499,999.98 Hz on shot 198279 from the record's span),
    so its Nyquist is 250 kHz and nothing above that is observable at all. The band this
    model reads is bins 164:512, 80.57-250.00 kHz. An Alfven eigenmode above 250 kHz -
    and DIII-D runs them - is invisible here and will be reported as quiet, not as
    unknown.
  - >-
    LFM is excluded from the label. The annotation carries five classes in the order
    ['lfm', 'bae', 'eae', 'rsae', 'tae'] (aemodes/.archive/early_fusion.py:50) and the
    training target is `label_1 | label_2 | label_3 | label_4` only. `ae_active` therefore
    does NOT fire on a low-frequency mode, by construction; upstream's own
    step_1_make_semantic.py collapses all five and is a different label.
  - >-
    The training label is a product of two noisy sources, not a hand annotation.
    Frame-level truth was built in task 7a as tokeye big_tf_unet's coherent mask
    (sigmoid(ch0) >= 0.2 AND NOT sigmoid(ch1) >= 0.2), opened along time with a 20-frame
    (5.1 ms) run, notched per (channel, bin) at 0.8, restricted to bins 164:511, and
    called active where band occupancy >= 0.01. The three-way target then trains only on
    frames where that mask and the human annotation AGREE: 1 on annotated-and-active,
    0 on not-annotated-and-not-active, weight 0 on the 52.4% of validation frames where
    they disagree. Every number below that says "labelled" is on the agreeing frames.
  - >-
    The spectrogram is NOT notched at inference. The 0.8 notch above is a rule over the
    mask, and there is no mask at inference time, so there is nothing to threshold. The
    network sees the un-notched band. Each label group carries `notch_rule` saying what
    the training label had removed and `notched_bins_khz: none` saying what inference
    removed.
  - >-
    The transform's statistics are taken over a longer record than training used. The
    1st/99th percentile clip and the per-channel standardisation are both per-array, and
    the trained records were 2 s of plasma while a corpus record is 8-11 s. Inference
    clips the record to the output grid's span ([-25 ms, 6.0 s]) before transforming,
    which removes the pre-shot baseline and the long post-shot tail, but 6 s is still
    three times the trained window. This is an un-priced approximation: no experiment has
    measured how much it moves a probability.
  - >-
    A whole record is evaluated in 1,024-frame windows with 128 frames of recurrent
    context on each side, not in one pass. The network pools nothing along time, so one
    pass is legal, but a 23,000-frame record's first convolution would need tens of GB.
    Everywhere except within 128 frames of a window seam the recurrent layers see the
    same neighbourhood a single pass would give them.
  - >-
    The 25 ms row summarises the causal window (t - 25 ms, t], which holds 97.6 frames at
    0.256 ms/frame. `ae_active` is the MEAN of those frames' probabilities - so a row is
    a duty cycle, not a maximum, and a 5 ms burst inside a 25 ms window reads about 0.2
    even if the model was certain during it. A row whose window holds fewer than 50
    frames is marked invalid; row 0's window ends at t = 0 and normally has none.
  - >-
    `ae_frequency` is UNVALIDATED. On the 81,859 annotated-and-active validation frames
    the frequency head's MAE is 17.63 kHz, against 16.79 kHz for predicting the global
    median - it does not beat a constant. It is published as a diagnostic because the
    activity head is what was validated and the two share a forward pass; do not read a
    number off it as a measurement.
---

# d3d_ae_activity_seldnet

Frame-level Alfven-eigenmode activity from the four CO2 interferometer chords,
aggregated onto labelmaker's 25 ms grid. Two labels per shot:

| label | task | units | meaning |
|---|---|---|---|
| `ae_active` | binary | - | mean over the window's frames of P(an AE is present in 80-250 kHz) |
| `ae_frequency` | regression | kHz | probability-weighted mean of the frequency head over the window's frames; NaN where `ae_active < 0.5` |

Both are probabilities/quantities, never thresholded, and both are `(1, 240)`
float32 beside a `_valid` mask.

## What it is

`labelmaker.ae.model.AeSeldNet`: three 2-D convolution blocks that pool along
**frequency only** (6, then 2, then 29 - 348 bins to 1), two bidirectional
GRUs whose directions are multiplied, and a small feed-forward head giving two
outputs per frame. 440,514 parameters. Because nothing pools time, the output
has one row per input frame and a record of any length is a legal input.

## Input chain

    co2 (4, ~4.5e6) float32 at 500 kHz, corpus group `co2`
      -> clip to [-25 ms, 6.0 s]                          (clip_to_grid)
      -> per chord: scipy ShortTimeFFT(hann 1024, hop 128),
         |.|, log1p, drop DC, clip to the array's 1/99 percentiles
      -> per-channel standardisation over all 512 bins
      -> bins 164:512                                  -> (4, 348, ~23,400)
      -> AeSeldNet in 1,024-frame windows              -> (frames, 2)
      -> sigmoid(col 0), 5-frame centred moving average
      -> col 1 * 170 + 80                              -> kHz
      -> mean / probability-weighted mean over (t-25 ms, t]

The transform is `src/labelmaker/ae/transform.py`, a port of
`tokeye.transforms.compute_stft` plus `tokeye.inference.model_infer`'s
normalisation. labelmaker must not import tokeye (it lives in a read-only venv
with its own torch), so the port is **pinned against tokeye's own output on a
real corpus record**: shot 198279, 0-6 s, `(4, 512, 23445)`, values 23.60 to
29.40 - **max absolute difference 0.0** at every stage (raw STFT,
standardised, band-restricted), across two different Python and scipy versions
(3.13/scipy 1.18.0 in the tokeye venv against 3.12/scipy 1.17.1 in the pixi
env). Evidence: `outputs/labelmaker/ae/scripts/pin_transform.py` and
`outputs/labelmaker/ae/transform_pin.json`.

## How it scored

Validation is the 60 held-out shots of task 7a's dataset, whole 7,820-frame
records, 469,200 frames. Three different frame populations appear below and
they are not interchangeable:

| population | frames | share |
|---|---|---|
| **labelled** - mask and annotation agree; the frames the loss was computed on | 223,443 | 47.6% |
| **annotated** - the human annotation calls an AE present | 91,306 | 19.5% |
| **active** - the cleaned mask calls the band occupied | 318,164 | 67.8% |

| metric | population | SCE (shipped) | BCE (not shipped) |
|---|---|---|---|
| AUROC | labelled | **0.9908** | 0.9897 |
| AUROC | active | 0.9857 | 0.9836 |
| AUROC | annotated | 0.6209 | 0.6147 |
| recall | annotated | 0.8804 | 0.8734 |
| precision | annotated | 0.2584 | 0.2595 |
| F1 | annotated | 0.3995 | 0.4001 |
| predicted positive fraction | all | 0.663 | 0.6551 |
| frequency MAE (kHz) | annotated & active | 17.63 | 16.44 |

Read the three AUROCs together, because they are the honest picture. Against
the target it was trained on the model is excellent (0.991). Against the mask
alone it is nearly as good (0.986) - unsurprising, since the mask is half of
that target. Against the **human annotation alone** it is 0.62: it recalls
88% of annotated frames but only a quarter of what it calls active is
annotated, because the mask over-calls (67.8% of frames active against the
annotation's 19.5%) and the model learned the mask's generosity along with the
annotation's discipline. `ae_active` is therefore a good detector of *coherent
band activity that a human would often call an AE*, and a poor estimator of
*the fraction of a shot a human would annotate*.

Symmetric cross entropy was the experiment task 7b ran to see whether label
noise was worth defending against. It wins on every AUROC and loses on
frequency MAE, all by margins in the third decimal; the SCE checkpoint is
shipped on the AUROCs. The two `val_loss` values (0.4065 SCE, 0.1429 BCE) are
on different loss scales and must not be compared.

## The notch rule, stated once

Task 7a swept notch thresholds {0.5, 0.6, 0.7, 0.8, 0.9} under the rule "take
the smallest threshold at which no bin within +-3 bins of an annotated
window's centroid is removed on more than 2 shots". The rule never bound - no
protected bin was removed on more than 2 shots at ANY swept threshold - so it
degenerated to "take the smallest", 0.5. A bin lit in half of a 2 s record can
still be a real long-lived mode, so the applied notch was fixed by the
controller at **0.8**, task 6's first-pass value. That is a property of the
**training label**; inference notches nothing.

## Coverage

The corpus fills `co2` on 12 of 24 shots sampled with
`sample_shots(corpus_shots, 24, seed=0)`; the other 12 carry the `(4, 1)`
absent-signal sentinel, and their rows are written invalid rather than
skipped. Every filled shot in that sample is above shot 198279. There is no
fdp fallback: a shot without a corpus `co2` group gets no AE label.

## Known-bad reading

Do not read `ae_active` as "an AE was present for this fraction of the
window": it is a mean of probabilities, and a probability of 0.5 held for the
whole window and a probability of 1.0 held for half of it give the same 0.5.
Do not read a high `ae_active` on a shot with no ECH or NBI as evidence of a
mode - nothing in this model sees the actuators, and the 250 kHz ceiling and
the LFM exclusion above both bias what it can see.
