# Sawtooth second-fix implementation plan

Goal: regenerate explicitly unvalidated research labels and matched model
benchmarks after addressing the two re-reviews. The user brief and
`saw-opus2.md` / `saw-sol2.md` are the specification.

Constraints: fixed shot splits; no test tuning; read-only corpus; no fetching;
GPU 1; all large artifacts under `round4/saw/fix2`; CSV labels untracked;
covering tests only; `labeler:` commits with the user-specified Codex trailer.

- [x] Detector (`physics.py`, `preprocessing.py`, covering tests): reproduce
  ambiguous gaps, missing/noisy auxiliary and distant gain-block failures; make
  absence require candidate exclusion and a recorded core relaxation test;
  use positive per-crash auxiliary evidence and adjacent profile blocks.
- [x] Geometry/export (`geometry.py`, driver, covering tests): search actual
  frequency metadata, map only where supported, otherwise mask implausible
  channels and derive a per-shot hottest-channel core proxy. Preserve train
  IDs, periods and nullable inversion coordinates in canonical present rows.
- [x] Models (benchmark and covering tests): use identical observability inputs
  at fit/inference and loss-only assessment masks; match sampling, widen z,
  train with inner-selection patience, report confusion/recalls/F1/baseline.
- [x] Regeneration: freeze explicit prior provenance on the original train
  development shots, run cohort then population, compare state seconds and
  absent support in short holes with the prior run, rerun validation/models.
- [x] Owner review: queue 15 nonexpert validation shots stratified by explicit
  heating/period proxies because cached H/L is unavailable, write blind protocol,
  render and inspect one
  300 ms train/val example at 3.25 inches with fonts at least 7 pt.
- [x] Evidence/docs: render current JSON-backed results, state expert anchoring
  and possible edge-originated relaxations on 190637, add brief history appendix,
  append Fix round 2 report, run covering tests/ruff and commit on r4-saw.

The final absence audit also protects significant negative-core phases without
period bounds. Saved edge records were refined with inputs retained, and both
models were retrained after cohort assessment targets changed. All final figures
were viewed and the covering tests, Ruff and required formatting checks passed.

Shared interfaces: `Detection.absent_mask/absence_diagnostics` feed the driver;
`state_spans(..., absent=...)` defaults unresolved bins to uncertainty. Driver
signal caches feed both model training and figures. Model JSON feeds results.

Review focus: candidate holes; incomplete/noisy auxiliary support; nonthermal
channels near the apparent core; uncertain bins retained in observable model
inputs; all scientific metrics remain agreement with unvalidated labels.

Execution: three disjoint implementation workers, coordinated integration and
reproduction by the primary agent. Owner-away authorization in the brief covers
routine implementation decisions and reruns; no approval pause is needed.
