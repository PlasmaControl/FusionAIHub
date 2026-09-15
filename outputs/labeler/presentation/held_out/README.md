# The same figures over the shots the survival model was never trained on

Rendered 2026-09-05 with
`scripts/make_presentation.py --subset held_out --out-dir held_out`, from the
same `pool_rows.npz` as the figures one level up.

`d3d_tearing_time_to_event_dsm` was fitted on 8,923 DIII-D shots
(140444-193373; the list is committed as
`src/labelmaker/models/d3d_tearing_time_to_event_dsm/training_shots.txt`).
**214 of the 500 pool shots are in it**, so every figure in the parent
directory mixes shots the model has seen with shots it has not. These are the
272 aligned pool shots it has NOT seen.

Two things this does **not** fix:

- **The CNN panels are in-sample either way.** The archive scored against IS
  `d3d_tearing_onset_cnn1d`'s training store, so blue and orange here are the
  same in-sample comparison as above, on a smaller shot set. Only the violet
  survival series changes meaning.
- **`04_reconstruction_fidelity.png` is not redrawn.** It is a pooled
  per-feature measurement from the CNN's `reconstruction.json`, which has no
  held-out/in-training split of its own; publishing it under a population it
  does not have would be worse than leaving it out. Use the one above.

The example shots differ from the parent directory's: they are picked from the
drawn subset, so `05a`/`05b` here are held-out shots (186523 and 186554)
rather than 187199 (a training shot) and 186545.

## all / held-out / in-training, side by side

Shot counts differ by row: `pool shots scored` and everything below it down to
the alarm rates come from `alarm_quality.json`, whose denominator is the 463
shots with usable pre-onset rows; the CNN and coverage rows come from
`summary.json`, whose denominator is the 486 aligned shots (486 / 272 / 214).
ECE rows come from `calibration_study.json`, measured on the shots its
held-out fit half did not use.

| quantity | all | held_out | in_training |
|---|---:|---:|---:|
| pool shots scored | 463 | 255 | 208 |
| of them tearing | 80 | 39 | 41 |
| pre-onset rows | 28,290 | 16,750 | 11,540 |
| AUROC `tm_risk_250ms` | 0.8100 | 0.8387 | 0.7608 |
| AUROC `tm_risk_500ms` | 0.7869 | 0.8168 | 0.7380 |
| AUROC `tm_risk_1s` | 0.7584 | 0.8006 | 0.6964 |
| IPCW AUC `tm_risk_250ms` | 0.8082 | 0.8379 | 0.7569 |
| IPCW AUC `tm_risk_500ms` | 0.7845 | 0.8153 | 0.7339 |
| IPCW AUC `tm_risk_1s` | 0.7582 | 0.8020 | 0.6912 |
| raw ECE all_pre_onset, 250 ms | 0.0017 | 0.0029 | 0.0007 |
| raw ECE all_pre_onset, 500 ms | 0.0050 | 0.0055 | 0.0057 |
| raw ECE all_pre_onset, 1 s | 0.0197 | 0.0145 | 0.0237 |
| raw ECE onset_shots_only, 250 ms | 0.1011 | 0.0931 | 0.1049 |
| raw ECE onset_shots_only, 500 ms | 0.2204 | 0.2086 | 0.2259 |
| raw ECE onset_shots_only, 1 s | 0.4358 | 0.4205 | 0.4429 |
| median warning at 0.2, `tm_risk_1s` (s) | 0.35 | 0.2875 | 0.375 |
| final-label FPR at 0.2, `tm_risk_1s` | 0.0653 | 0.0694 | 0.0599 |
| final-label FNR at 0.2, `tm_risk_1s` | 0.6625 | 0.5897 | 0.7317 |
| any-row FPR at 0.2, `tm_risk_1s` | 0.1462 | 0.1019 | 0.2036 |
| any-row FNR at 0.2, `tm_risk_1s` | 0.6000 | 0.5128 | 0.6829 |
| CNN `tm_prob` AUROC, archived inputs | 0.9315 | 0.9613 | 0.8834 |
| CNN `tm_prob` AUROC, reconstruction | 0.8972 | 0.9388 | 0.8307 |
| median lead time, CNN / survival (s) | +0.000 / +0.375 | -0.037 / +0.275 | +0.000 / +0.387 |

## What it says

**The in-sample half of the pool is the harder half, not the flattered one.**
Held-out AUROC is 0.03 to 0.10 *higher* than in-training AUROC at every
horizon, for the survival model and for the CNN alike, and IPCW AUC agrees.
The overlap was therefore not carrying the pool numbers; it was dragging them
down. The reason is population, not memorisation: 41 of the 208 in-training
scored shots tear (19.7%, and 6.8% of their pre-onset rows are positive at
1 s) against 39 of 255 held-out shots (15.3%, 4.7%), and coverage is lower on
them too (45% against 55% of timesteps published).

What the split does change is the alarm trade-off. On held-out shots the
survival model's final-label FNR at 0.2 falls to 0.59 from 0.73 in-sample and
its any-row FPR halves (0.10 against 0.20), while its median lead time is
0.10 s shorter. Any single number quoted for "the pool" sits between the two
columns and belongs to neither population.

`summary.json` in this directory holds the machine-readable version for the
held-out subset; the `in_training` rendering is not published (it is one
command away: `--subset in_training`).
