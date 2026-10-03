# Sawtooth fix-round implementation plan

Goal: regenerate physics labels and converged GPU benchmarks after both reviews.
The user brief, binding implementer rules, and both reviews define acceptance.

- [ ] Detector: add failing regression cases for soft q conflicts, inversion
  stability, core relative drop, low Te, missing support, edge-only ELMs,
  calibrated direction and isolated candidates; implement and run covering tests.
- [ ] Reader: test native-rate filtering against aliasing and chunk boundaries;
  implement bounded-memory FIR decimation, NaN auxiliaries, neutron/Mirnov
  evidence, density cutoff proxy and four-state records.
- [ ] Freeze all guards using only an explicit fixed train-shot development set;
  record every guard's provenance and correct the mixed-split pilot claim.
- [ ] Generate cohort and population with the identical rule; remove tracked CSV
  shards and export untracked tables to the worktree and stream output root.
- [ ] Run the original sawtooth_events rule on all cohort/expert shots read-only;
  compare per expert shot on identical observable support, list all-shot agreement,
  and record reference-shot availability and train-shot antialiasing impact.
- [ ] Train both models on the assigned CUDA device until inner-fold early
  stopping; select thresholds on inner shots, score outer shots with bootstrap
  intervals and expert shots individually, including abstention coverage.
- [ ] Produce whole-shot and crash-window galleries and inspect every PNG;
  render source-backed docs and append review-to-change mapping to the report.
- [ ] Run covering tests and Ruff only, sweep temporary storage after long jobs,
  audit small-summary size, and commit with the requested prefix and trailer.

File owners: detector/its regression tests; benchmark/metrics/its tests;
gallery/results/docs; root owns preprocessing, pipeline, freeze and validation.
Shared record schema: full four-state spans; separate present and uncertain
intervals; signal arrays carry observable and assessed masks. Missing and
uncertain samples never enter target losses or scoring denominators.
