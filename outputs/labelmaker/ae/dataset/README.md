# AE frame labels and dataset (task 7a)

Frame-level Alfven-eigenmode activity labels for the 180 annotated shots, built from
`big_tf_unet`'s coherent mask **after cleaning**, with the notch and the activity
threshold fixed by stated rules. Spec section 5; the mask itself was validated in
task 6 (`../README.md`, `../mask_vs_annotation.json`).

Produced by `scripts/labelmaker/ae_dataset.py` / `.sbatch`, SLURM job **2923913**.
Every number below is in `dataset.json`; its `git_sha` is HEAD at generation time, which
is the parent of the commit that adds these files - a file cannot carry its own hash.

## Construction

    coherent  = sigmoid(big_tf_unet(x)[0]) >= 0.2        # channel 0 = coherent
    transient = sigmoid(big_tf_unet(x)[1]) >= 0.2        # channel 1 = transient
    mode      = coherent & ~transient
    clean     = binary opening of `mode` along TIME only, PERSIST_FRAMES = 20
                (5.1 ms at 0.256 ms/frame), per (channel, frequency bin)
    notch     = zero every (channel, bin) lit in > 0.5 of the 7820 frames of `clean`
    active[f] = mean over the 4 CO2 channels of the fraction of band bins 164-511
                lit in frame f, thresholded at >= 0.01

Transform: `tokeye.transforms.compute_stft` per channel on the cached arrow time
series - `clip(log1p(|STFT|), p1, p99)`, Hann 1024 / hop 128, DC bin dropped - then
standardised per array as `tokeye.inference.model_infer` does, `(x - mean) / (std + 1e-6)`.
One channel at a time, so no cross-spectrum is ever formed. Band bins 164-511 are
**80.566-250.000 kHz**; 348 bins, 7820 frames, 0.256 ms per frame.

`annotated[f] = label_1 | label_2 | label_3 | label_4` (BAE, EAE, RSAE, TAE) resampled
to the frame grid by nearest sample; **`label_0` (LFM) is excluded** per spec 5.1.

`freq_khz[f]` is the **linear-power**-weighted centroid over the masked band pixels:
weights are `power_weights(x) = expm1(x)**2` applied to the transform's own
un-standardised values, i.e. `|STFT|**2` after the percentile clip, pooled over the
four channels. NaN wherever `active` is 0. Weighting by the stored `log1p` values
instead - what task 6 did - is very nearly unweighted: measured over this dataset those
values span 24.2-31.9 with a coefficient of variation of 0.034-0.039 (the aemodes
log-power tif task 6 also used spans 46-63, CV 0.026). Power weighting spreads that into
a 5e6 dynamic range, so the centroid follows the brightest coherent pixels. The p99
clip means the top 1 % of pixels all carry the same weight; that ceiling belongs to the
transform.

The rules live in `src/labelmaker/ae/labels.py` (numpy only, no torch, no I/O) and are
unit-tested in `tests/labelmaker/test_ae_labels.py`, including that `persist_open`
agrees exactly with `scipy.ndimage.binary_opening(mask, structure=np.ones((1, 20)))`.

## Notch: swept, then fixed at 0.5

Rule: sweep {0.5, 0.6, 0.7, 0.8, 0.9} and take the **smallest** threshold at which no
bin inside an annotated AE window's centroid +-3 bins is removed on more than 2 shots.
Smaller notches more aggressively, so the rule buys as much line rejection as the
protected bins allow.

| threshold | notched bins/channel, mean (max) | union/shot, mean (max) | shots notching anything | shots with a protected bin removed | passes |
|---|---|---|---|---|---|
| **0.5** | **0.379 (11)** | **1.456 (19)** | **54** | **0** | **yes** |
| 0.6 | 0.196 (11) | 0.783 (16) | 29 | 0 | yes |
| 0.7 | 0.126 (9) | 0.506 (12) | 19 | 0 | yes |
| 0.8 | 0.082 (7) | 0.328 (7) | 13 | 0 | yes |
| 0.9 | 0.051 (5) | 0.206 (5) | 10 | 0 | yes |

**Zero violations at every threshold**, so the rule takes 0.5. The centroids it
protects are power-weighted and therefore narrow, which is part of why nothing is hit.
`dataset.json:notch.table[*].removed_bins_per_shot` lists, for each swept threshold,
every bin each shot and channel loses: 273 bins over 63 channel-notches at 0.5, down to
37 over 10 at 0.9. Note that `176526_train`'s `r0` block keeps 5 bins at
125.0-127.0 kHz even at 0.9, so task 6's concern about that shot is not an artefact of
the threshold - the line is there at every setting swept.

## Notch review: 34 shots, 39 channel-notches withheld

`notch_review.json` records, per shot and channel, which bins the notch removes, their
kHz and their occupancy. Per the brief, **any shot whose notch removes a bin below
190 kHz is flagged `needs_review: true` and that channel's notch is not applied** to
the label; the other channels' notches still are. 34 of the 54 notching shots are
flagged and 39 of the 63 channel-notches are withheld; 24 are applied.

- `176526_train` `r0` **123.5-128.4 kHz, 11 bins**, occupancy up to 0.95 - the case
  task 6 raised (it saw 124.0-127.9 kHz at threshold 0.8). Withheld. Its `v1`
  166.5-167.5 kHz is withheld too; its `v3` 201-203 kHz line is applied.
- The dominant line the notch does remove is **`v3` 200.7-203.6 kHz on 24 shots** -
  above 190 kHz, so it is applied everywhere it fires. This is task 6's 201 kHz line.
- The dominant **withheld** line is **`v1` 166.0-167.5 kHz on ~15 shots**, a 3-bin line
  at 0.52-0.57 occupancy. It looks like instrument pickup, but it is inside the AE
  band and below 190 kHz, so the rule leaves it in the label until Nathan says.
- 11 of the 34 flagged shots are flagged **only** by bins below the 80.57 kHz band
  edge (`170671_valid`, `170790/170792/170793/170799/170801/170803/170808_valid`,
  `176043/176045/176060_train`). Those bins never enter the occupancy or the centroid,
  so withholding those notches changes no label; the flag is the literal rule firing.
  15 of the 39 withheld channel-notches are entirely below the band.

`notch_review.png` is the shot x bin heat map (only the 54 shots that notch anything;
colour is how many CO2 channels notch that bin; bold green shot names need review).

**Consequence worth stating:** a single un-notched line bin adds `1/348/4 = 0.0007` to
the occupancy per channel it lights, and a 5-bin line in one channel adds 0.0036 -
a third of the 0.01 activity threshold. The withheld notches therefore do bias the
label towards active on those 23 in-band shots.

## Activity threshold: swept, rule not satisfied, 0.01 by the escape clause

Rule: the **lowest** threshold whose predicted-positive frame fraction is <= 0.40 while
pooled recall of annotated frames stays >= 0.80; if none satisfies both, the one with
recall >= 0.80 and the smallest positive fraction, and say so.

Cleaned + notched mask, 180 shots, 1,407,600 frames, 21.03 % annotated active:

| occupancy threshold | pooled recall | precision | predicted-positive frac | recall >= 0.80 | pp <= 0.40 |
|---|---|---|---|---|---|
| 0.005 | 0.903 | 0.304 | 0.625 | yes | no |
| **0.01** | **0.864** | **0.340** | **0.535** | **yes** | **no** |
| 0.02 | 0.793 | 0.363 | 0.460 | no | no |
| 0.03 | 0.724 | 0.368 | 0.414 | no | no |
| 0.05 | 0.564 | 0.369 | 0.321 | no | yes |
| 0.08 | 0.275 | 0.353 | 0.164 | no | yes |
| 0.1 | 0.136 | 0.352 | 0.081 | no | yes |

**No threshold satisfies both halves**: recall falls below 0.80 between 0.01 and 0.02,
while the positive fraction only falls below 0.40 at 0.05. The escape clause applies -
of the thresholds with recall >= 0.80, 0.01 has the smaller positive fraction - so
**the label is `occupancy >= 0.01`, at recall 0.864 and a 53.5 % positive fraction**.
That is 2.5x the annotation's 21.0 %. Per Nathan's under-counting amendment a label
with more positives than the annotation is acceptable; per this rule it is still
outside the 0.40 target and is recorded as such (`occupancy_study.rule_satisfied`
is `false` in `dataset.json`).

### What the cleaning bought

Pixel-wise it removes a third of the mask: pooled, **65.7 %** of coherent pixels
survive. Per the two shots the brief names (all 4 channels x 512 bins x 7820 frames,
before the notch):

| shot | coherent pixels | after transient veto | after persistence | kept |
|---|---|---|---|---|
| `176042_train` (task 6's streakiest, AUROC 0.210) | 1,765,346 | 1,589,183 (90.0 %) | 1,099,560 | **62.3 %** |
| `175978_train` (task 6's cleanest, AUROC 0.991) | 454,628 | 443,274 (97.5 %) | 342,646 | **75.4 %** |

The cleaning bites harder on the streaky shot than on the clean one, which is the
intended direction. **At the frame level it buys almost nothing**, though: the same
sweep run on the raw coherent mask (same notch) gives recall 0.868 at a 0.522 positive
fraction (threshold 0.02) against the cleaned mask's 0.864 at 0.535 - the same
trade-off, shifted along the threshold axis. The table is in
`dataset.json:occupancy_study.raw_mask_table_for_comparison`. The 20-frame opening
removes streak pixels but not the frames they sit in, because those frames usually
have other lit bins.

## Dataset

`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/ae/dataset/<shot>_<split>.npz`, 180 files,
3.7 GB (not committed - group storage):

| array | shape | dtype | meaning |
|---|---|---|---|
| `spec` | (4, 348, 7820) | float16 | band-restricted standardised tokeye-transform spectrogram |
| `active` | (7820,) | uint8 | the label: band occupancy of the cleaned, notched mask >= 0.01 |
| `annotated` | (7820,) | uint8 | `label_1\|2\|3\|4` on the frame grid, LFM excluded |
| `freq_khz` | (7820,) | float32 | linear-power centroid of the masked band pixels; NaN where inactive |
| `notched_bins` | (4, 512) | bool | bins actually zeroed, per channel (withheld notches are all-False) |
| `notch_excluded` | (4,) | bool | channels whose notch was withheld pending review |
| `occupancy` | (7820,) | float32 | the occupancy trace `active` was thresholded from |
| `spec_mean`, `spec_std` | (4,) | float64 | per-channel standardisation, so `spec * std + mean` is the raw transform |
| `freq_khz_bins` | (348,) | float32 | centre frequency of each `spec` row |
| `split` | () | str | `train` / `valid`, from the pickle's own lists via the file suffix |

Both sigmoid channels are kept as uint8 probabilities under
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/ae/masks/<shot>_probs.npz` (1.6 GB with the
packed masks), quantised with `floor(p * 255)` so that `>= 51` reproduces `p >= 0.2`
exactly - the threshold study can be re-run without a GPU.

### Class balance

| set | shots | frames | active | annotated | recall of annotated | precision vs annotated | frame agreement |
|---|---|---|---|---|---|---|---|
| pooled | 180 | 1,407,600 | 0.535 | 0.210 | 0.864 | 0.340 | 0.618 |
| train | 120 | 938,400 | 0.463 | 0.218 | 0.850 | 0.400 | 0.689 |
| valid | 60 | 469,200 | 0.678 | 0.195 | 0.897 | 0.257 | 0.476 |

Jaccard(active, annotated) pooled 0.323; 18.2 % of frames are both, 35.3 % active only,
2.9 % annotated only. **The valid split is the more active half** (0.678 vs 0.463) -
worth knowing before reading task 7b's validation numbers.

Per shot: recall of annotated frames median 0.918, p10 0.604, >= 0.5 on 169 of 180;
active fraction median 0.466 (min 0.036, max 0.931). `freq_khz` per-shot median 112.6 kHz
(p10 92.6, p90 132.5, min 87.5, max 161.3) and **all 180 shots sit inside 80-250 kHz**.

## Run

`gpu` partition, 1 A100 40 GB on `stellar-m01g4`, 4 CPUs, 32 GB, 2 h limit; wall clock
**00:27:10**, 8.4 s/shot. The 4 CO2 channels go through `big_tf_unet` in **one batched
forward pass** (task 6 ran them one at a time): peak torch allocation 21.93 GiB, 38.6 GB
of the card including cache, so batch 4 at full 7820-column width is the largest that
fits and the script falls back to 2 then 1 on OOM. CPU efficiency 23.9 % (00:25:58 of
01:48:40 core-time), memory 4.95 GB of 32 GB, **GPU utilisation 0.6 %**. The job is
still host-bound - one core doing scipy STFTs and zlib-compressing 32 MB of
probabilities per shot - so batching the forward pass did not move the needle; only
prefetching shots across the 4 cores would. At 27 minutes it is not worth writing.

## Open items for Nathan

1. The 23 shots whose in-band sub-190 kHz notch is withheld, above all the `v1`
   166-167 kHz line - is it pickup (notch it) or a real mode (leave it)?
2. The activity rule's 0.40 positive-fraction target is not reachable at recall 0.80
   with this construction. Either the target moves, or the label needs something the
   occupancy statistic does not have.
