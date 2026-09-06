# The retrained model over the shots it was never trained on

Rendered 2026-09-05 with
`scripts/make_presentation.py --dsm-slug d3d_tearing_time_to_event_dsm_continued --subset held_out --out-dir held_out`,
from the same `pool_rows.npz` as the figures one level up.

`d3d_tearing_time_to_event_dsm_continued` continued the shipped checkpoint's
fit on the shipped checkpoint's own rows, so **its training set is the base
model's**: the 8,923 shots of
`src/labelmaker/models/d3d_tearing_time_to_event_dsm/training_shots.txt`,
214 of which are in the 500-shot pool. These are the 272 aligned pool shots
neither checkpoint has seen. `../held_out/` is the same rendering for the
shipped model, on exactly the same shots, so the two are comparable row for
row.

Two things this does **not** fix:

- **The CNN panels are in-sample either way.** The archive scored against IS
  `d3d_tearing_onset_cnn1d`'s training store, so blue and orange are the same
  in-sample comparison as above, on a smaller shot set. They are also
  identical to `../../presentation/held_out/`'s: only the violet survival
  series differs between the two directories.
- **`04_reconstruction_fidelity.png` is not redrawn.** It is a pooled
  per-feature measurement with no held-out/in-training split of its own.

The example shots are picked from the drawn subset, so `05a`/`05b` here are
held-out shots (186523 and 186554) rather than 187199 (a training shot) and
186545.

## all / held-out / in-training, side by side

Shot counts differ by row: `pool shots scored` down to the alarm rates come
from `alarm_quality.json`, whose denominator is the 463 shots with usable
pre-onset rows; the CNN and coverage rows come from `summary.json`, whose
denominator is the 486 aligned shots (486 / 272 / 214). ECE rows come from
`calibration_study.json`, measured on the shots its held-out fit half did not
use.

| quantity | all | held_out | in_training |
|---|---:|---:|---:|
| pool shots scored | 463 | 255 | 208 |
| of them tearing | 80 | 39 | 41 |
| pre-onset rows | 28,290 | 16,750 | 11,540 |
| AUROC `tm_risk_250ms` | 0.6813 | 0.6918 | 0.6671 |
| AUROC `tm_risk_500ms` | 0.6695 | 0.6719 | 0.6553 |
| AUROC `tm_risk_1s` | 0.6802 | 0.6880 | 0.6499 |
| IPCW AUC `tm_risk_250ms` | 0.6796 | 0.6910 | 0.6640 |
| IPCW AUC `tm_risk_500ms` | 0.6644 | 0.6667 | 0.6494 |
| IPCW AUC `tm_risk_1s` | 0.6733 | 0.6817 | 0.6404 |
| raw ECE all_pre_onset, 250 ms | 0.0096 | 0.0081 | 0.0108 |
| raw ECE all_pre_onset, 500 ms | 0.0273 | 0.0217 | 0.0316 |
| raw ECE all_pre_onset, 1 s | 0.0614 | 0.0439 | 0.0748 |
| raw ECE onset_shots_only, 250 ms | 0.1008 | 0.1085 | 0.0983 |
| raw ECE onset_shots_only, 500 ms | 0.2187 | 0.2302 | 0.2134 |
| raw ECE onset_shots_only, 1 s | 0.4290 | 0.4380 | 0.4248 |
| median warning at 0.2, `tm_risk_1s` (s) | 0.5 | 0.325 | 0.6 |
| final-label FPR at 0.2, `tm_risk_1s` | 0.1175 | 0.0833 | 0.1617 |
| final-label FNR at 0.2, `tm_risk_1s` | 0.6250 | 0.7179 | 0.5366 |
| any-row FPR at 0.2, `tm_risk_1s` | 0.3420 | 0.2454 | 0.4671 |
| any-row FNR at 0.2, `tm_risk_1s` | 0.3750 | 0.4103 | 0.3415 |
| CNN `tm_prob` AUROC, archived inputs | 0.9315 | 0.9613 | 0.8834 |
| CNN `tm_prob` AUROC, reconstruction | 0.8972 | 0.9388 | 0.8307 |
| median lead time, CNN / survival (s) | +0.000 / +0.950 | -0.037 / +1.413 | +0.000 / +0.600 |

## What it says

**Splitting reverses the one number that flattered the retrained model.**
Pooled over every scored shot it looked like the continuation traded false
positives for false negatives - final-label FNR 0.625 at threshold 0.2 against
the shipped model's 0.6625. On held-out shots alone its FNR is **0.7179**
against the shipped model's **0.5897**, so it is worse on both error rates on
the population that answers the question. Its any-row FPR is nearly twice as
high on shots it was trained on (0.4671) as on shots it was not (0.2454),
which is what a model whose risks have moved up on remembered rows looks like.

The AUROC verdict does not change and gets slightly sharper: held-out, the
retrained model ranks at 0.6918 / 0.6719 / 0.6880 against the shipped model's
0.8387 / 0.8168 / 0.8006 at 250 ms / 500 ms / 1 s - a gap of 0.147, 0.145 and
0.113, wider at every horizon than the pooled 0.129 / 0.117 / 0.078.
Calibration on all pre-onset rows is worse than the shipped model's on both
halves (raw ECE at 1 s, held-out 0.0439 against 0.0145).

Nothing here changes the default: `analyze_default.yaml` still points at
`d3d_tearing_time_to_event_dsm`.

`summary.json` in this directory holds the machine-readable version for the
held-out subset; the `in_training` rendering is not published
(`--subset in_training` produces it).
