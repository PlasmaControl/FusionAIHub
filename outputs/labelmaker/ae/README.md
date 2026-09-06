# AE go/no-go: `big_tf_unet`'s coherent mask against the 180 hand-annotated shots

Phase 3 task 6. Answers spec section 5.8 item 1 and spec section 7 question 3.
Measured 2026-09-05, SLURM job **2923882** on stellar.

> ## Verdict: **GO**, using the **`tokeye`** transform.
>
> Under the `tokeye` transform the coherent mask recovers **91.2 %** of all
> annotated-active frames at an occupancy threshold of 0.01, and **172 of the 180
> shots (95.6 %)** have a per-shot recall of at least 0.5 at that same threshold.
> The rule is recall >= 0.70 pooled and >= 0.5 per shot on >= 80 % of shots; both
> clear it with margin, and they clear it at three of the four thresholds tested.
> The `aemodes` transform also passes (0.902 / 95.0 % at the same threshold); the
> two are within 0.004 AUROC of each other, and `tokeye` is ahead on every recall
> number, so `tokeye` is the choice. The tie-breaker beyond the numbers is that
> `compute_stft` is tokeye's own inference contract for these weights
> (`transforms.py` records that `hop = 128` "matches the released model's training
> recipe"), so training and corpus inference can inherit one recipe instead of two.
>
> Precision is **0.31** at that threshold and pooled AUROC is **0.721**: the mask
> marks 61 % of all frames active where the annotation marks 21 %. Per Nathan's
> 2026-09-05 amendment those numbers are reported, not disqualifying - the hand
> annotations under-count real activity. But see "What the numbers do not say"
> before treating the mask as a finished label.

## What was measured

`tokeye`'s `big_tf_unet` is a 1 -> 2 channel U-Net whose **channel 0 is coherent
activity** (`tokeye/README.md:205`). The AE label in spec section 5.5 rests on that
channel agreeing with the hand annotations where the two overlap. This run produced
the coherent mask for all **180 annotated shots (170659-178879, 120 train / 60
valid)** under **two spectrogram transforms**, reduced it to a per-frame band
occupancy, and scored that against the annotation. Nothing about the label is
fitted here; this is only the go/no-go.

| item | value |
|---|---|
| checkpoint | `/scratch/gpfs/nc1514/tokeye/model/big_tf_unet_251210.pt` (registry `big_tf_unet`), loaded from the local path with `tokeye.hub.load_model` - no HF download |
| coherent channel | output channel 0, `sigmoid(logits)` |
| mask | `prob >= 0.2`, the threshold `aemodes/pipeline/step_1_make_semantic.py` uses |
| notch | first pass, per (CO2 channel, frequency bin): zero the bin if it is lit in more than **80 %** of the 7820 frames |
| band | bins **164-511** = **80.57-250.00 kHz** on this grid (`(bin+1) x 500.0005 kHz / 1024`) |
| occupancy | `occ[f] = mean over bins 164-511 of (prob >= 0.2)`, per CO2 channel after notching, then averaged over the 4 channels |
| annotation | `label_1 \| label_2 \| label_3 \| label_4` (BAE, EAE, RSAE, TAE). **`label_0` (LFM) excluded**, reported separately |
| annotation resampling | nearest sample: frame time from `ShortTimeFFT.t(1000001)` in ms, sample index `round(t_ms x 500.0005)` |
| frames | 7820 per shot at 0.256 ms, **1,407,600** frames over the 180 shots; 21.0 % annotated active, 1.40 % inside an LFM window |

## The two transforms

**`aemodes`** - the regenerated spectrogram already on disk,
`aemodes/data/.cache/step_0a/spectrograms/<shot>_<mode>.tif`, `(4, 512, 7820)`
float32. That is `log1p(|STFT|^2)` (`ShortTimeFFT.spectrogram`, Hann `nperseg=1024`,
`hop=128`, DC bin dropped). Standardised with the **global** `step_0a/stats.json`
statistics, mean 54.637 / std 2.732.

**`tokeye`** - `tokeye.transforms.compute_stft` run on the cached arrow time series
one channel at a time (`n_fft=1024`, `hop=128`, Hann, DC clipped), which is
`log1p(|STFT|)` - amplitude, not power - followed by a **1st/99th percentile clip**.
Standardised the way `tokeye.inference.model_infer` does it, **per array**:
`(x - x.mean()) / (x.std() + 1e-6)`.

`compute_stft` forms a cross-spectrum only when handed two rows; each CO2 channel is
fed on its own, so **no cross-spectrum is formed anywhere in this run**.

**Windowing: none.** `tokeye/src/tokeye/inference.py::model_infer` feeds the whole
spectrogram in one forward pass and does not window the column axis, so the full
7820-column width was used, matching tokeye's own contract. The script keeps a
`--width` chunking path with an automatic fallback on CUDA OOM; it never fired -
the full width peaked at 5.5 GiB of torch allocation (10.5 GiB of the A100's 40 GiB
including the CUDA context and cache).

**The two transforms differ in two ways at once**: magnitude + percentile clip vs
power, and per-array vs global standardisation. That is what the brief specifies
(each recipe with its own standardisation, which is what a training run would
actually do), but it means the 0.004 AUROC gap is attributed to the transform as a
whole and not to either half of it. Given how small the gap is, that ambiguity does
not change any conclusion here.

## Results

Everything below is over all 180 shots and all 1,407,600 frames. The full per-shot
table is in `mask_vs_annotation.json`.

### Pooled

| metric | aemodes | tokeye |
|---|---|---|
| frames | 1,407,600 | 1,407,600 |
| annotated-active frame fraction | 0.210 | 0.210 |
| **pooled AUROC** | 0.7166 | **0.7207** |
| recall @ occ>=0.005 | 0.930 | **0.938** |
| recall @ occ>=0.01 | 0.902 | **0.912** |
| recall @ occ>=0.02 | 0.857 | **0.869** |
| recall @ occ>=0.05 | 0.698 | **0.720** |
| precision @ occ>=0.005 | 0.282 | 0.268 |
| precision @ occ>=0.01 | 0.325 | 0.313 |
| precision @ occ>=0.02 | 0.353 | 0.350 |
| precision @ occ>=0.05 | 0.358 | 0.361 |
| predicted-positive frame fraction @ occ>=0.005 | 0.694 | 0.736 |
| predicted-positive frame fraction @ occ>=0.01 | 0.584 | 0.613 |
| predicted-positive frame fraction @ occ>=0.02 | 0.510 | 0.522 |
| predicted-positive frame fraction @ occ>=0.05 | 0.409 | 0.419 |

The predicted-positive fraction is the honest denominator for the recall: a coin
that flipped positive 61.3 % of the time would recover 61.3 % of annotated frames.
The mask's 91.2 % is a **lift of 1.49** over that, rising to 1.72 at occ>=0.05
(recall 0.720 against a 0.419 positive rate).

### Unannotated predicted-positive frames, split

Of the frames the mask calls active but the annotation does not, essentially none
fall inside an LFM window - LFM covers only 1.40 % of frames, and 94 % of LFM frames
are already inside an AE window (18,563 of 19,656).

| threshold | transform | unannotated share of predicted positives | inside an LFM window | outside any annotation |
|---|---|---|---|---|
| occ>=0.005 | aemodes | 0.718 | 0.0003 | 0.718 |
| occ>=0.005 | tokeye | 0.732 | 0.0003 | 0.732 |
| occ>=0.01 | aemodes | 0.675 | 0.0003 | 0.675 |
| occ>=0.01 | tokeye | 0.687 | 0.0003 | 0.687 |
| occ>=0.02 | aemodes | 0.647 | 0.0003 | 0.647 |
| occ>=0.02 | tokeye | 0.650 | 0.0003 | 0.650 |
| occ>=0.05 | aemodes | 0.642 | 0.0001 | 0.641 |
| occ>=0.05 | tokeye | 0.639 | 0.0001 | 0.639 |

At occ>=0.01 under `tokeye` that is 592,706 unannotated frames, of which 236 are in
an LFM window and 592,470 are outside any annotation at all.

### Per shot

179 of the 180 shots have at least one annotated-active frame; **172025_train** has
none and drops out of every per-shot recall and AUROC statistic (it is still counted
in the 180 denominator of the verdict's 80 % rule, i.e. counted as failing).

| metric | aemodes | tokeye |
|---|---|---|
| AUROC median | 0.7947 | 0.7978 |
| AUROC 10th percentile | 0.4167 | 0.3914 |
| AUROC min / max | 0.2232 / 0.9901 | 0.2104 / 0.9910 |
| shots with AUROC < 0.6 | 44 | 47 |
| shots with AUROC < 0.5 | 36 | 36 |
| recall median @ occ>=0.005 | 0.991 | 0.997 |
| recall median @ occ>=0.01 | 0.972 | 0.982 |
| recall median @ occ>=0.02 | 0.914 | 0.923 |
| recall median @ occ>=0.05 | 0.722 | 0.760 |
| recall 10th pct @ occ>=0.005 | 0.775 | 0.809 |
| recall 10th pct @ occ>=0.01 | 0.701 | 0.727 |
| recall 10th pct @ occ>=0.02 | 0.578 | 0.620 |
| recall 10th pct @ occ>=0.05 | 0.326 | 0.356 |
| **shots with recall >= 0.5 @ occ>=0.005** | 176/180 (0.978) | **177/180 (0.983)** |
| **shots with recall >= 0.5 @ occ>=0.01** | 171/180 (0.950) | **172/180 (0.956)** |
| **shots with recall >= 0.5 @ occ>=0.02** | 168/180 (0.933) | **168/180 (0.933)** |
| **shots with recall >= 0.5 @ occ>=0.05** | 136/180 (0.756) | 142/180 (0.789) |

Under `tokeye`, per-shot recall at occ>=0.01 has a minimum of 0.375 and only 7 shots
fall below 0.5.

Both AUROC < 0.5 counts were recomputed from the per-shot AUROCs in
`mask_vs_annotation.json` (task 7a review, 2026-09-05): **36 for each transform**, not
the 33 first written for `aemodes`. The two transforms disagree about *which* shots
they are - the union is larger than 36 - but the count is the same. `auroc_below_0.5`
now sits beside `auroc_below_0.6` in `metrics.<transform>.per_shot_summary`, so the
number no longer has to be recounted by hand.

### Centroid frequency check

The intensity-weighted centroid frequency of the masked pixels, taken over
annotated-active frames only and then median-reduced per shot:

| metric | aemodes | tokeye |
|---|---|---|
| median of the per-shot medians | 123.4 kHz | 124.2 kHz |
| 10th / 90th percentile | 107.9 / 148.3 kHz | 110.4 / 149.7 kHz |
| min / max over shots | 100.3 / 170.4 kHz | 100.9 / 169.2 kHz |
| shots whose median centroid is inside 80-250 kHz | 179/179 | 179/179 |

Every shot's median centroid lands inside the band, clustered at 100-170 kHz - where
TAE/EAE activity on DIII-D belongs.

**How this centroid is weighted (corrected 2026-09-05).** It is weighted by the
transform's own un-standardised values, which are *logarithmic*: `tokeye`'s
`clip(log1p(|STFT|), p1, p99)` and `aemodes`' log-power tif. Those values barely vary -
measured over task 7a's dataset the `tokeye` ones span 24.2-31.9 with a coefficient of
variation of 0.034-0.039 per (shot, channel), and the `aemodes` tif spans 46-63 at
CV 0.026 - so a log-weighted centroid is **effectively an unweighted mean over the
masked pixels**, not an intensity-weighted one. That, and not a real agreement about
where the power is, is why the two transforms land within ~1 kHz of each other here.
Task 7a therefore weights `freq_khz` by **linear power**, `power_weights(x) =
expm1(x)**2`, which spreads the weights over a ~5e6 dynamic range; it moves the pooled
median centroid down about 11 kHz (124.2 kHz here to 113.0 kHz in
`dataset/dataset.json`). Read the numbers in this section as unweighted centroids.

### Notch

| metric | aemodes | tokeye |
|---|---|---|
| notched bins per shot, union over the 4 CO2 channels: mean / median / max | 0.31 / 0 / 9 | 0.41 / 0 / 12 |
| notched bins per single channel, mean | 0.08 | 0.10 |
| all notched bins were inside 80-250 kHz | yes | yes |
| shots with zero notched bins | 167/180 | 164/180 |
| peak per-bin active fraction, median over shots | 0.450 | 0.466 |
| peak per-bin active fraction, max over shots | 0.954 | 0.966 |

**What it actually removed.** Under `tokeye` the notch fires on 16 of 180 shots, and
what it removes is mostly one line: a contiguous **201.2-203.1 kHz** block in CO2
channel `v3`, present on 14 of those 16 shots (4-5 bins each). That is the receiver
pickup section 5.4 describes, found without being told where to look. The one
exception is **176526_train**, where it also removes bins **124.0-127.9 kHz** in `r0`
(9 bins) - and that is squarely inside where the AE centroids live, so it is exactly
the case section 5.4 warns about: a bin lit for 80 % of a 2-second record could be a
line, or it could be a mode that simply never went away. That shot needs a human look
before the rule is fixed, and the per-shot notch record in `mask_vs_annotation.json`
(`notched_bins_per_channel`) plus the packed masks are what makes that possible.

**The 0.8 rule is quiet**: it fires on 16 of 180 shots under `tokeye` and removes a
median of zero bins. The typical shot's most-lit bin is active in 47 % of frames -
nowhere near the threshold - and only the genuinely line-like bins approach it. That
is evidence for spec section 5.8 item 2 that 0.8 is a safe, conservative first-pass
threshold: it will not silently delete real modes, but it also cannot be the whole
instrument-line story, and a per-shot review is still needed on the 16 shots where
it does fire.

## Figures

Six shots, three best and three worst by per-shot AUROC under `tokeye`. Each shows
CO2 channel `r0` over the 80-250 kHz band with the coherent mask in green, notched
bins as red dashed lines, the band occupancy with the four thresholds, and the
per-class annotation bars. The spectrogram has its per-bin median removed and its
time axis decimated x4 **for display only** - the model saw the raw spectrogram.

| figure | per-shot AUROC | what it shows |
|---|---|---|
| `figures/175978_train_mask.png` | 0.991 | best. Chirping 100-170 kHz structures sit exactly inside the annotated 300-850 ms TAE/EAE window and the occupancy steps up and down with it. |
| `figures/171998_train_mask.png` | 0.988 | best; recall 1.000 at occ>=0.01 |
| `figures/175964_train_mask.png` | 0.984 | best |
| `figures/176042_train_mask.png` | 0.210 | worst, and the clearest picture of why. The annotation covers 100-350 ms only; the mask finds a strong rising 80-120 kHz branch from 1200 ms to the end of the shot that no one annotated. This is Nathan's under-counting, visible. |
| `figures/170722_valid_mask.png` | 0.238 | worst - yet its per-shot recall at occ>=0.01 is 1.000. A low AUROC here means the mask is *more* active outside the annotated window than inside it, not that it missed the mode. |
| `figures/170803_valid_mask.png` | 0.248 | worst; per-shot recall 1.000 at occ>=0.01 |
| `figures/occupancy_vs_annotation.png` | - | pooled ROC for both transforms (they overlie each other) and the per-shot AUROC histogram, which is strongly bimodal: a mode near 0.85-0.95 and a tail below 0.5. |

None of the six figure shots had a notched bin in `r0`, so no red dashed line appears
on any of them; the shots where the notch does fire are listed under "Notch" below.

## What the numbers do not say

Three things a reader should not take from a GO:

1. **Pooled AUROC 0.72 with a bimodal per-shot distribution.** 47 shots score below
   0.6 and 36 below 0.5 - on those the occupancy series is uninformative about, or
   anti-correlated with, the annotation. `176042_train` shows the benign reason
   (real unannotated activity later in the shot inverts the ranking), and two of the
   three worst shots still have per-shot recall 1.000, but it has not
   been checked shot by shot that every low-AUROC shot is benign, and a shot where
   the mask fires on something that is not an AE would look identical in these
   aggregates.
2. **The recall is bought with a wide mask.** 61 % of all frames are called active.
   The recall clears the bar comfortably, but the lift over the predicted-positive
   base rate is 1.49, not 4. Where the occupancy threshold ends up (spec 5.8 item 3)
   matters a great deal, and 0.05 - the only threshold that fails the per-shot half
   of the rule at 78.9 % - is where the mask starts becoming selective.
3. **Broadband transients leak into the coherent channel.** In `176042_train` the
   500-1150 ms stretch is full of narrow vertical stripes that the coherent channel
   marks. Whatever those are (ELMs, sawteeth, pellets), they are not eigenmodes, and
   channel 1 (transient) is not being used here to suppress them. Using
   `channel 0 & ~channel 1`, or a minimum time-extent on mask components, is the
   obvious next experiment and is not part of this task.

None of these fail the brief's verdict rule, which is deliberately recall-only, and
none are hidden here.

## Environment, run and cost

- **Python**: `/scratch/gpfs/nc1514/tokeye/.venv/bin/python` (3.13.1, torch
  2.9.1+cu128, `tokeye` importable), used **read-only**. `pyarrow` is not installed
  there and the tokeye tree must not be written to, so `pyarrow==25.0.1` was
  installed with `uv pip install --target
  /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/ae/pylibs` and put on `PYTHONPATH`.
  Nothing was written into `/scratch/gpfs/nc1514/tokeye` or
  `/scratch/gpfs/nc1514/aemodes`. The phase3 venv did not exist yet when this ran.
- **Smoke test**: 2 shots on the login node, CPU, both transforms, 68 s/shot, before
  submitting.
- **SLURM job 2923882**, partition `gpu`, 1x A100 40 GB, 4 CPUs, 32 GB, on
  `stellar-m01g2`. Wall clock **10:21** of a 3 h limit; 3.2 s per shot for both
  transforms and all 4 channels.
- **Utilisation, sampled twice while running and once at the end**: GPU 9.2 % at
  1:31, 2.1 % at 5:45, 5.6 % at 6:03, **20.5 % averaged over the completed job**;
  GPU memory high-water **10.5 GB of 40 GB**; CPU efficiency **23.0 %** (00:09:31 of
  00:41:24 core-walltime); CPU memory high-water **4.40 GB of 32 GB**.
- **The GPU was mostly idle, and that is expected here, not a bug.** 1440 forward
  passes of a 7.9 M-parameter U-Net take a few milliseconds each on an A100; the job
  is dominated by single-threaded host work - reading 180 x 64 MB tifs and 180 x
  16 MB arrow tables, and computing 720 STFTs plus their percentile clips in scipy.
  Fixing it would mean prefetching shots on the 4 allocated cores and batching the 4
  CO2 channels into one forward pass. At a 10-minute total runtime that optimisation
  is not worth writing; a job that ran this over thousands of corpus shots would
  need it.

## Files

| path | what |
|---|---|
| `scripts/mask_vs_annotation.py` | the whole measurement: stage `masks` (GPU) and stage `metrics` (CPU, re-runnable without the GPU) |
| `scripts/mask_vs_annotation.sbatch` | the launcher used for job 2923882 |
| `mask_vs_annotation.json` | every number above plus the full 180-row per-shot table, both transforms |
| `figures/*.png` | the six shot figures and the pooled ROC / AUROC histogram |
| `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/ae/masks/<shot>_<transform>{,_mask}.npz` | per-shot occupancy, centroid, notch, annotation series and the packed mask bits (73 MB total). Not in the repo. |
| `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/2923882.out` | the run log |

Re-running the metrics and figures without a GPU:

```bash
PYTHONPATH=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/ae/pylibs \
/scratch/gpfs/nc1514/tokeye/.venv/bin/python \
  outputs/labelmaker/ae/scripts/mask_vs_annotation.py --stage metrics
```
