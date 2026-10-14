# elm-inst: ELM onsets at 0.1 ms beside ELMy occupancy

Status: **run 2026-10-08 (run `round4/elm/inst/inst1`; scripts not yet committed).** One
network gives two outputs: ELMy occupancy per 1 ms, as `elm-ours` does, and a 0.1 ms
heatmap whose peaks are individual ELM onsets. It is the first step towards the
instance-plus-crowd model (instance segmentation's `is_crowd`): occupancy covers the group,
the heatmap the individual ELMs.

## Why

The two earlier onset heads fail on the DIII-D challenge key (`elm_challenge.md`) for
reasons that can be read off their design:

- they work at 1 ms with peaks at least 10 ms apart, and 63 % of the key's ELMs have
  another within 10 ms;
- the Smith head saw only Smith's windows around isolated ELMs and no quiet time, so it
  fires in quiet windows (266 false detections in the key's 12 windows with few events);
- the review head learned the reviewed non-crowd span starts and fires early (median
  -4.8 ms) and rarely (35 detections in 24 windows).

## Design (`src/labeler/elm/inst.py`, `scripts/labeler/elm_inst.py`)

- **Network.** The `cv2` U-Net trunk (`net.ElmUNet`, same 11 inputs) gives its 1 ms decoder
  features to an occupancy head (1x1 conv) and, upsampled to 10 kHz and joined with a
  two-layer 10 kHz convolution branch over the raw inputs, to the instance head.
- **Instance target** (CenterNet style): a Gaussian of 0.3 ms sigma at each onset, peak
  cell 1; penalty-reduced focal loss normalised by the batch's peak count. Decoding: local
  maxima at least 2 ms apart above a threshold. Each detection becomes an event of the
  median clock span in the fold's training present spans (1.05 to 1.17 ms).
- **Training data**, both sources in every batch (8 + 8 crops of 2,048 ms):
  - the 119 reviewed shots: occupancy from the review labels; absent spans, 5 ms in from
    their edges, are instance negatives; inside crowd and non-crowd spans the ELM clock's
    onsets are pseudo-targets at half weight and the rest is ignored;
  - Smith's 2,316 windows on 211 shots: his onsets inside his windows, the rest ignored.
- **Folds.** Fold k uses `cv2`'s split of the reviewed shots and the Smith run's split of
  his shots, so no fold model sees its test shots of either source. 40 epochs of 100
  steps; the kept epoch maximises the trailing three-epoch mean of (occupancy AUPRC +
  onset F1) / 2 on inner validation, where onset F1 counts every detection in guarded
  absent time as false. Thresholds are inner-validation F1 maxima (onset 0.35 to 0.65).
- **Served on the key**: the mean of the five fold models' traces, at the mean onset
  threshold (0.51). Whether to gate onsets by occupancy was decided on development data
  before the key was scored: ungated (development F1 with absent alarms 0.748 against
  0.732 gated).

## Development results (out of fold; `outputs/labeler/elm/inst/evaluation.json`)

| view | elm-inst | comparison |
|---|---|---|
| occupancy, reviewed 50 ms bins (12,409) | AUROC 0.946, AUPRC 0.911, F1 0.831 | `cv2`: 0.941, 0.877, 0.830 |
| Smith's windows, onsets within 2 ms | P 0.994, R 0.807, F1 0.891 [0.855, 0.924], median offset 0.05 ms | Smith head: 0.998, 0.933, 0.964, 0.25 ms |
| detections in guarded reviewed absent time (403 s, 341 spans) | 802, in 37 spans (11 %) | Smith head 3,682 (63 %); clock 15,782 (37 %); `cv2` onset 212 (25 %) |

On the reviewed crowd spans, where only the clock's onsets exist as a reference, elm-inst
finds 40 % of those with another clock onset within 10 ms and 60 % of the rest (at the fold
thresholds; 74 % and 92 % at 0.2, with more absent alarms).

## Challenge key (held out; scored with the challenge merit function at commit b71f991)

Full tables in `elm_challenge.md`. In short:

- **All 24 windows: merit 2,460**, the first of our methods below no detection (2,863;
  clock 26,659, Smith head 5,315). 96 detections, 94 of them a key ELM within 2 ms
  (precision 0.979, recall 0.303, median offset 0.06 ms).
- **12 windows with few events: 115** against 116 for no detection; one detection, a real
  ELM. The clock and the Smith head fire 2,324 and 275 times there.
- **12 busy windows: 2,346**, better than the Smith head (2,449) and no detection (2,747),
  worse than the clock (2,120): recall 0.304 against the clock's 0.634.
- **ELM trains are what it misses:** of the 194 key ELMs with another within 10 ms it finds
  22 (clock 111, Smith head 53); of the 28 alone by 50 ms or more, 19 (clock 24, Smith 26).
- **Occupancy bins:** F1 0.817 [0.642, 0.925] on 24 shots, beside `elm-ours`' 0.827.

## What was tried after the key result, on development data only

A two-level decoding (a lower threshold where occupancy is on, the served threshold
elsewhere) was searched over the threshold grid on the out-of-fold development traces,
scoring the mean of the Smith-window F1 with absent alarms and the F1 against the clock's
onsets in crowd spans. The best pair is a single threshold of 0.45 (0.675 against 0.668 at
0.5); the gate adds nothing. That is no material change, so the key was not scored again.
Recall in trains needs a training change, not a decoding change: the only instance targets
inside crowds are the clock's onsets at half weight, and the clock itself finds 63 % of the
key's ELMs in busy windows.

## Reproduce

Training needs a GPU; the labeler pixi env's torch is CPU-only, so `train` and `predict`
ran with `$LABELER_ROOT/envs/phase3/bin/python` and `PYTHONPATH=src`:

    python scripts/labeler/elm_inst.py prepare          # clock pseudo-onsets, pixi env
    python scripts/labeler/elm_inst.py train --fold K --device cuda:0   # K = 0..4
    python scripts/labeler/elm_inst.py predict --device cuda:0
    python scripts/labeler/elm_inst.py evaluate         # development record
    python scripts/labeler/elm_challenge_evaluate.py    # key scores, as elm_challenge.md

Inputs: the frozen `round4/elm/cv/cv2` and `round4/elm/smith/cv` fold files and inputs
under `$LABELER_ROOT`. Run outputs go to `$LABELER_ROOT/round4/elm/inst/inst1/`.
