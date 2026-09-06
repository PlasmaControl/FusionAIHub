# labelmaker, both tearing models against the archived truth

Rendered 2026-09-05 over the 500-shot pool (`$LABELMAKER_ROOT/shots_500.txt`): 486
shots align to their archived training rows, 31,257 of those rows are
published as valid, 8.5% of them tearing-positive. Blue is the CNN on its own
archived training inputs (its ceiling), orange is labelmaker's reconstruction (what gets
published), violet is the survival model.

Rebuild: `scripts/pool_rows.py` writes the per-row arrays, `scripts/make_presentation.py`
draws these seven; `scripts/pick50.py` chooses the shot-level set.

**214 of the 500 pool shots are in the survival model's own training set**, so
every figure here mixes in-sample and held-out shots. The same figures over the
held-out shots alone, with an all / held-out / in-training table, are in
`held_out/`; `make_presentation.py --subset {all,held_out,in_training}` renders
any of the three.

| file | what it shows |
|---|---|
| `01_roc_pr.png` | ROC and precision-recall for both models. CNN `tm_prob` AUROC 0.932 archived, 0.897 reconstructed; survival risk 0.81 / 0.79 / 0.76 at 250 ms / 500 ms / 1 s. |
| `02_f1_vs_threshold.png` | F1 against the decision threshold. Neither model's reported threshold is its best one. |
| `03_calibration.png` | Reliability and probability histograms. The CNN is over-confident; the survival risk is calibrated over every aligned shot and badly under-confident once restricted to shots that do get an onset. |
| `04_reconstruction_fidelity.png` | Per-feature median relative difference to the training inputs, grouped by provenance, with the survival model's six unpriceable features marked. |
| `05a_example_shot_187199.png` | A shot where the CNN calls the mode before the archived label does. |
| `05b_example_shot_186545.png` | A shot where it calls the mode only after the label already says so. |
| `06_prediction_shift.png` | Row by row: what the reconstruction does to the CNN, and whether either prediction rises as the onset approaches. |
| `07_coverage_and_lead_time.png` | Published coverage per shot (median 40% CNN, 50% survival) and lead time on the 86 shots with an onset. |

## What the numbers say

- The reconstruction costs the CNN 0.034 of AUROC and 0.086 of best F1; `betan` is barely
  affected.
- The survival risk ranks pre-onset rows better than the CNN's present-mode probability does
  (0.81 against 0.79 at 250 ms), and it fires early far more often:
  its median lead time is +0.38 s and 82% of its crossings come before
  the archived onset, against +0.00 s and 40% for the CNN.
- Coverage is the ECH story: the survival model needs no deposition location, so it publishes
  a median 50% of timesteps against the CNN's 40%.
- Two row sets are in play for the survival model and they answer different questions. Every
  aligned shot's pre-onset rows (28,290 rows, 5.5%
  positive at 1 s) asks "which shots and times are heading for a mode". Restricting to the shots
  that do get one asks "when", and AUROC falls to 0.68 / 0.66 / 0.64. Figures 01, 02 and 06-left
  use the first; figure 06-right and 07-right use the second; figure 03 shows both.

## Per shot, on the 50 with an onset

`scripts/agg50.py` rolls up the `truth` block of each per-shot analysis
(`../analysis/per_shot_scores.json`). Medians over the shots where the number is
defined:

| | CNN `tm_prob` at 0.5 | survival `tm_risk_1s` at 0.2 |
|---|---|---|
| AUROC within the shot | 0.970 (43 shots) | 0.958 (39 shots) |
| precision | 1.00 | 1.00 |
| recall | 0.75 | 0.00 |
| F1 | 0.52 | 0.00 |
| lead time | -0.05 s | +0.34 s |
| crosses its threshold at all | 39 of 50 | 22 of 50 |

`betan` median RMSE is 0.111 per shot.

The survival model ranks within a shot as well as the CNN does (0.958 against
0.970) but almost never reaches 0.2, so at that threshold its recall is zero on
half the shots. Its scale, not its ranking, is the problem - the same
under-confidence figure 03 shows. Lowering the threshold trades coverage for
lead time cleanly, measured on the 86 pool shots with an onset:

| threshold | shots that cross | median lead time | fires early |
|---|---|---|---|
| 0.05 | 77 of 86 | 1.35 s | 94% |
| 0.10 | 64 of 86 | 0.82 s | 81% |
| 0.15 | 50 of 86 | 0.54 s | 80% |
| 0.20 | 39 of 86 | 0.38 s | 82% |

0.10 looks like the better default for this label; the config in
`src/labelmaker/analyze_default.yaml` currently ships 0.2.

Per-shot views (`pixi run -e labelmaker label <shot>`) go to `../analysis/<shot>/`; the
committed example is 199597. `summary.json` here holds the numbers quoted above.
