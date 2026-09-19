# The retrained survival model on the same pool, beside the shipped one

The seven Phase 2 figures re-rendered with
`d3d_tearing_time_to_event_dsm_continued` in place of
`d3d_tearing_time_to_event_dsm`. Everything else is identical: the same
500-shot pool (`$LABELMAKER_ROOT/shots_500.txt`), the same 486 aligned shots,
the same 31,257 published CNN rows, the same CNN panels. Only the violet
survival series changes.

Rebuild: `scripts/pool_rows.py pool_rows.npz --dsm-slug <slug>` then
`scripts/make_presentation.py --dsm-slug <slug> --out-dir <dir>`. Both are
copies of `../presentation/scripts/` with a `--dsm-slug` argument added; the
originals were not edited.

The retrained checkpoint is `rt_fixed_rot_continued.pkl`: the shipped fit
continued from its own weights for 300 more epochs at lr 1e-4 (SLURM job
2923879, 29 min 38 s), validation NLL 0.471321 -> 0.287652 on the split the
shipped model was fitted on. Details in
`src/labelmaker/models/d3d_tearing_time_to_event_dsm_continued/README.md`.

## Shipped against continued

Row sets, so the columns are not mixed up: **AUROC / IPCW AUC / lead time /
FPR / FNR** come from `alarm_quality.json` over **all 28,290 pre-onset valid
rows of the 463 scored shots** (383 quiet, 80 tearing); **ECE** comes from
`calibration_study.json` over the **held-out report half** of that pool (128
shots; 8,809 all_pre_onset rows, of which 685 belong to onset shots), raw
risks, no post-hoc map. Both models were scored with the same seed-0 split,
which since 2026-09-05 is drawn from the 255 pool shots neither checkpoint was
trained on.

| quantity | shipped | continued | difference |
|---|---:|---:|---:|
| AUROC `tm_risk_250ms` (all pre-onset rows) | 0.810043 | 0.681327 | **-0.128716** |
| AUROC `tm_risk_500ms` | 0.786852 | 0.669523 | **-0.117329** |
| AUROC `tm_risk_1s` | 0.758441 | 0.680208 | **-0.078233** |
| IPCW AUC `tm_risk_250ms` | 0.808217 | 0.679597 | -0.128620 |
| IPCW AUC `tm_risk_500ms` | 0.784507 | 0.664393 | -0.120115 |
| IPCW AUC `tm_risk_1s` | 0.758210 | 0.673280 | -0.084930 |
| ECE `tm_risk_250ms`, all_pre_onset | 0.002928 | 0.008132 | +0.005204 |
| ECE `tm_risk_500ms`, all_pre_onset | 0.005542 | 0.021728 | +0.016186 |
| ECE `tm_risk_1s`, all_pre_onset | 0.014508 | 0.043870 | +0.029362 |
| ECE `tm_risk_250ms`, onset_shots_only | 0.093080 | 0.108482 | +0.015402 |
| ECE `tm_risk_500ms`, onset_shots_only | 0.208588 | 0.230165 | +0.021577 |
| ECE `tm_risk_1s`, onset_shots_only | 0.420512 | 0.438004 | +0.017492 |
| median lead time, `tm_risk_1s` at 0.2 (warning time over final-label TPs) | 0.350 s | 0.500 s | +0.150 s |
| median lead time, first crossing of 0.2 (figure 07) | 0.375 s (39 of 86 shots cross) | 0.950 s (61 of 86 cross) | +0.575 s |
| final-label FPR at 0.2, `tm_risk_1s` | 0.065274 | 0.117493 | +0.052219 |
| final-label FNR at 0.2, `tm_risk_1s` | 0.662500 | 0.625000 | -0.037500 |

**These pool numbers are roughly half in-sample for both models.** The
survival training set is 8,923 unique DIII-D shots spanning 140444-193373;
214 of the 500 pool shots, 208 of the 463 aligned scored shots and 41 of the
80 shots with an archived onset are training shots. Every figure here draws
all of them. The same figures over the held-out shots alone, and the
side-by-side all / held-out / in-training table, are in `held_out/`.

## What the comparison says

The continuation is a large likelihood gain on its own validation split and a
**loss of ranking power on the pool**: AUROC falls at all three horizons, by
0.078 to 0.129, and IPCW AUC falls with it. Calibration on all pre-onset rows
gets worse at all three horizons (ECE 0.0029 -> 0.0081, 0.0055 -> 0.0217,
0.0145 -> 0.0439), and so does calibration restricted to onset shots
(0.4205 -> 0.4380 at 1 s). Pooled over every scored shot rather than the
held-out report half, the onset-only figure moves the other way
(0.4358 -> 0.4290): which sign you get depends on the population, and the
held-out one is the one that answers the question.

The same picture shows in the alarm rates: the retrained model calls more
shots at threshold 0.2, so its FNR falls (0.6625 -> 0.6250 at 1 s) while its
FPR nearly doubles (0.0653 -> 0.1175), and it crosses 0.2 on 61 of the 86
onset shots against 39, with a longer median lead time. Firing earlier and
more often is not the same as ranking better, and the AUROC column is the one
that separates the two.

Why the gain did not transfer is recorded in both cards: the split the shipped
checkpoint was fitted on is `SurvivalModel.fit`'s default 15% **row** sample,
not a by-shot holdout, so 8,685 of the 8,690 shots in it also have training
rows. Training to convergence against that objective improves the leaky
likelihood and does not improve out-of-sample discrimination.

**Nothing here recommends switching the default.** `analyze_default.yaml`
still points at `d3d_tearing_time_to_event_dsm`, and the open question in
`.claude/superpowers/specs/2026-09-05-labelmaker-phase3-design.md` section 7
(question 5) stays open with the shipped checkpoint as the default.

| file | what it shows |
|---|---|
| `01_roc_pr.png` | ROC and precision-recall. CNN panels unchanged; survival AUROC 0.68 / 0.67 / 0.68 at 250 ms / 500 ms / 1 s, against 0.81 / 0.79 / 0.76 shipped. |
| `02_f1_vs_threshold.png` | F1 against threshold for both models. |
| `03_calibration.png` | Reliability and probability histograms on both survival row sets. |
| `04_reconstruction_fidelity.png` | Per-feature reconstruction difference. Identical to the Phase 2 figure: it does not involve the survival model. |
| `05a_example_shot_187199.png` | The retrained risk trace on the early-calling example shot. |
| `05b_example_shot_186545.png` | The retrained risk trace on the late-calling example shot. |
| `06_prediction_shift.png` | Row by row, and whether the risk rises as the onset approaches. |
| `07_coverage_and_lead_time.png` | Coverage per shot (unchanged, median 50%) and lead time at 0.2 (median +0.95 s, 82% early). |

`summary.json` holds the machine-readable version of the numbers quoted here;
the full threshold sweeps are in
`$LABELMAKER_ROOT/validation/d3d_tearing_time_to_event_dsm_continued/{alarm_quality,calibration_study}.json`.
