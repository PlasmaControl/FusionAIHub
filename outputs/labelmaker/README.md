# labelmaker outputs

- `analysis/<shot>/`: what `pixi run -e labelmaker label <shot>` leaves behind - the labels
  file (`<shot>_labels.h5`, `<model>/<label>/{xdata,ydata}` plus `_spread`/`_valid`), the same
  as numpy (`<shot>_labels.npz`), a JSON summary per label, and one figure with a panel per
  prediction from every servable model. The committed example is 199597, a 2024 tearing-mode
  shot that has every input the five models need. `per_shot_scores.json` is `presentation/scripts/agg50.py`'s
  roll-up of the `truth` blocks over the 50 shots with archived truth and an onset; the
  per-shot analyses it was rolled up from were two-model renders and were removed once the
  `label` command superseded them (rerun `label` on a shot to regenerate its view);
- `presentation/`, `presentation_continued/`: the pool-level figures 01-07 over the 500-shot
  pool for the two tearing survival models (`held_out/` restricts to the shots outside the
  survival model's training set), with their own READMEs and the scripts that build them;
- `ae/`: the AE dataset build, mask-vs-annotation check, transform pin, corpus coverage and
  the SELDNet training comparison (`ae/README.md`, `ae/training/README.md`);
- `tabpfn/`: the TabPFN-against-the-CNN study (Study A on the CNN's archived inputs, Study B on
  labelmaker's reconstructed inputs), with its own README.

In the presentation figures, blue is a model on its own archived training inputs (its
ceiling), orange is labelmaker's reconstruction (what gets published), and violet is either
the survival model or the archived truth, as each figure's legend says. Every number comes
from `$LABELMAKER_ROOT/validation/<slug>/*.json` or from the per-row predictions rebuilt by
the scripts in `presentation/scripts/`.
