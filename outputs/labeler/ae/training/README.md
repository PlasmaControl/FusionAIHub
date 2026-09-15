# AE activity + frequency SELDNet (task 7b)

A frame-level Alfven-eigenmode detector: 4 CO2 interferometer spectrogram channels in,
one activity probability and one frequency in kHz out, per 0.256 ms frame. Trained on
the 120 training shots of task 7a's dataset and scored on its 60 validation shots.

Checkpoints, `training_<run>.json` and `evaluation.json` live on group storage at
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/d3d_ae_activity_seldnet/` (not in git).
The code is `src/labelmaker/ae/model.py`, `scripts/labelmaker/ae_train.{py,sbatch}` and
`scripts/labelmaker/ae_evaluate.{py,sbatch}`.

## Results

Four runs on the same split (120 training / 60 validation shots; validation is a single
forward pass over each whole 7820-frame record) and the same recipe: AdamW lr 1e-4, weight
decay 1e-4, cosine to 1e-6, bf16 autocast, 16 windows of 710 frames per step, early stop on
validation loss with patience 5. All four stopped early. Every number below is read from
`evaluation.json` and the four `training_<run>.json` files; the activity threshold is 0.5.

| run | target | loss | pool sizes | epochs run | best epoch | val loss | AUROC, labelled frames | freq MAE (kHz) | params |
|---|---|---|---|---|---|---|---|---|---|
| `threeway_sce` (shipped) | three-way | SCE (α 1.0, β 0.5) | (6, 2, 29) | 12 | 6 | 0.4065 | **0.9908** | 17.63 | 440,514 |
| `threeway_bce` | three-way | BCE | (6, 2, 29) | 13 | 7 | 0.1429 | 0.9897 | 16.44 | 440,514 |
| `mask_sce` | mask only | SCE (α 1.0, β 0.5) | (6, 2, 29) | 17 | 11 | 0.4717 | 0.9885 | 17.39 | 440,514 |
| `threeway_sce_f29` | three-way | SCE (α 1.0, β 0.5) | (6, 2) | 20 | 14 | 0.5232 | 0.9803 | **12.90** | 1,779,714 |

"AUROC, labelled frames" is against the three-way target on the frames that target weights
(`annotated & active` against `~annotated & ~active`): 223,443 of the 469,200 validation
frames, 47.6 %. The `val_loss` column is **not** comparable across rows - SCE and BCE are
different objectives, and the mask run averages over every frame while the three-way runs
average only over the weighted ones.

Frequency MAE is measured on the 81,859 `annotated & active` frames that have a finite
centroid. The constant baseline on exactly those frames is **16.79 kHz**: the MAE of the
best possible single number, the median target 114.09 kHz (the mean, 118.33 kHz, gives
17.11 kHz; the band centre 165 kHz gives 47.99 kHz). Three of the four runs sit at or above
that baseline. With pool sizes (6, 2, 29) the conv stack collapses the frequency axis to a
single bin before the GRUs, so the frequency head has essentially nothing left to regress
from and settles near the marginal median.

Against the raw human annotation all four runs behave alike, and as the brief predicted:
recall 0.873-0.892 at precision 0.258-0.273, predicted-positive fraction 0.62-0.67 against
an annotated fraction of 0.195, median per-shot recall 1.000. That precision is not an
error rate - the annotation under-counts, so a frame called active outside an annotated
window is as likely to be an unlabelled mode as a false alarm. Recall and the
labelled-frame AUROC are the honest summaries.

### What is shipped, and why

`ae_seldnet_threeway_sce.pt` - the three-way target with the symmetric cross-entropy loss.
It is the checkpoint named in `src/labelmaker/models/d3d_ae_activity_seldnet/README.md`
(SLURM job 2924037, best epoch 6 of 12 run, `git_sha` 4779ee6).

* **Three-way over mask-only.** The activity gap is small (labelled-frame AUROC 0.9908 vs
  0.9885, AUROC against the annotation 0.621 vs 0.602), but the mask-only run is taught
  that every frame the mask misses is a negative, including the under-count region the
  three-way target deliberately leaves unweighted. The three-way run reaches its best epoch
  in half the epochs (6 vs 11) and matches or beats the mask run on every activity metric,
  so the label design costs nothing and removes supervision known to be wrong.
* **SCE over BCE.** SCE leads on labelled-frame AUROC (0.9908 vs 0.9897) and on AUROC
  against the annotation (0.621 vs 0.615), and its reverse-cross-entropy term is the point
  of the comparison: the label is noisy in both directions. BCE's much smaller `val_loss`
  is a property of the objective, not evidence about the model.

The margins between the three (6, 2, 29) runs are third-decimal. Read the SCE-BCE gap as
evidence that the label noise does not dominate at this scale, not as a decisive win.

### The `_f29` variant, and why it is not shipped

`threeway_sce_f29` is the same three-way SCE recipe with pool sizes (6, 2) instead of
(6, 2, 29), so 29 frequency bins - not 1 - reach the GRUs, at 4x the parameters (1,779,714
vs 440,514). It is the only run whose frequency head beats the constant baseline:
**12.90 kHz MAE against 16.79 kHz** (median absolute error 9.34 kHz, against 15.29 kHz for
the shipped run), and it has by far the best AUROC against the raw annotation (0.723 vs
0.621). It pays for that in agreement with its own teacher: labelled-frame AUROC 0.9803 vs
0.9908, recall against `active` 0.885 vs 0.953.

It is **not shipped**. Per Nathan's decision (2026-09-06) the AE model is closed out with
the checkpoint the card already names; no checkpoint is swapped. The variant is recorded
here because it identifies the fix should the frequency output ever be needed: keep the
frequency axis alive through the conv stack. Until then the shipped model's frequency
output should be treated as unusable - it is no better than quoting 114 kHz on every frame.

### Figures

`valid_<shot>_valid.png` - the six most-annotated validation shots (170663, 170671, 170672,
170678, 178872, 178879). Top panel: the channel-mean CO2 spectrogram with the mask centroid
target and each run's predicted frequency on the frames it calls active (p >= 0.5). Bottom
panel: the `annotated` and `active` label bands with each run's p(active) over the record.
`evaluation.json` here is a copy of the artifact written next to the checkpoints.

