# labelmaker outputs

Three sets, all rendered 2026-09-05:

- `presentation/`: the pool-level figures 01-07 over the 500-shot pool, both tearing models,
  with their own README and the scripts that build them;
- `analysis/<shot>/`: `python -m labelmaker.run analyze` on 61 shots - the 50 with archived
  ground truth and a real onset, plus the 2024 tearing-mode shots 199597-199607 - each a JSON
  summary and one figure with a panel per label, both models, the archived truth overlaid and
  scored;
- `dsm_onset_quality.json` (+ the script that made it): the survival model's onset risk scored
  against the tearing archive's onset times on the 486 aligned shots (86 have an onset), pre-onset
  rows only - AUROC 0.81 / 0.79 / 0.76 at 250 ms / 500 ms / 1 s, against 0.79 / 0.75 / 0.67 for
  the CNN's `tm_prob` read as a predictor of the same truth;
- `tabpfn/`: the TabPFN-against-the-CNN study (Study A on the CNN's archived inputs, Study B on
  labelmaker's reconstructed inputs), with its own README.

Throughout, blue is a model on its own archived training inputs (its ceiling), orange is
labelmaker's reconstruction (what gets published), and violet is either the survival model or
the archived truth, as each figure's legend says. Every number comes from
`validation/<slug>/*.json` or from the per-row predictions rebuilt by the scripts in
`presentation/scripts/`. The per-figure table is in `presentation/README.md`.
