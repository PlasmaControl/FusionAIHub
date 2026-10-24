# ELM models on Ricardo Shousha's review of 2026-10-23

Status: **run 2026-10-24 (SLURM CPU job 2977016, 2 min 44 s; jobstats gate passed at CPU
79.3 %, memory 98.1 %; git dbabdb4c).** The frozen ELM models, scored without refitting
anything on the 81 shots Ricardo Shousha reviewed on 2026-10-23. Record:
`outputs/labeler/elm/heldout_2026_10_23/evaluation.json` (copy at
`$LABELER_ROOT/round4/elm/heldout/2026-10-23/evaluation.json`). Code:
`src/labeler/elm/heldout.py`, `scripts/labeler/elm_heldout_evaluate.py`,
`scripts/labeler/elm_heldout.sbatch`.

## What this shows, in four lines

- **No model trained on any of his shots, and none of them is in a cross-validation fold,
  Smith's 211 shots or the cohort's blind test split.** This is the first score of the
  deployed `elm-ours` ensemble (five `cv2` fold models at the mean threshold 0.417) on
  shots outside its training. It is not the blind split's score.
- **His labels are not independent of `elm_jump`.** His review opened on the `elm_jump` v2
  onsets as a prefill and he kept 17,109 of his 17,128 individual rows as they were. The
  occupancy scores below are near their ceiling for every method that behaves like
  `elm_jump` (B1 F1 0.955 to 0.972). The independent information is his edits to the spans,
  his absent calls and the 23 challenge-key shots (B2), where the models do worse and
  differ.
- **Onset agreement is agreement with the accepted `elm_jump` prefill**, not onset accuracy.
  `elm_jump` scores 1.000 on it by construction.
- The old reviewed set (119 shots) reproduces the published records exactly, so the new
  numbers sit on the same code as the old ones; they are not comparable with them (other
  shots, other labelling procedure).

## Inputs

| input | file | sha256 |
|---|---|---|
| old review labels (pin 2026-10-17) | `$LABELER_ROOT/benchmarks/elm/review_pins/2026-10-17/labels.csv` (592 rows) | `d483fed8...6276` |
| his review labels (pin 2026-10-23) | `.../review_pins/2026-10-23/labels.csv` (19,079 rows, 201 shots) | `b54c1a27...8e02` |
| his save history | `.../review_pins/2026-10-23/history.jsonl` | `e95ceb0c...b5fd` (the evaluation record holds the full hashes) |
| `elm_jump` v2 table (his prefill) | `suggestions/elm_jump/v2/edge_localized_mode_suggest_elm_jump_v2.csv` | `aa936d3c...45c4` (the `source_sha256` on all 81 of his last saves) |

Both pins are read-only copies with a `SHA256SUMS`; the script re-hashes them
(`heldout.verify_pin`) before anything is scored. The 2026-10-23 table holds the same rows
for the old pin's shots as the 2026-10-17 pin does (`pins_agree_on_the_old_shots`; the old
pin has 120 shots; the published panel scores 119 of them).

**Pinned labels.** Every ELM benchmark and training script now reads
`prepare.review_csv(paths)`, which returns the 2026-10-17 pin, so a later review does not
change an old number. `prepare.review_csv(paths, None)` is the live table and
`prepare.review_csv(paths, "2026-10-23")` his. A pin that does not exist is an error, not a
fallback. `train.check_no_holdout` refuses any `protect: holdout` shot (the challenge key's
24) in `train.load` and in `elm_inst.py run_prepare`, and fails closed when the shot-list
registry cannot be read.

## His shots

| set | shots | what | scored |
|---|---|---|---|
| B1 | 58 | his shots that are not on a holdout list | 57 (203401 has no `\BCI::DENV2F`, fdp `%TREE-E-NODATA`, so no model input) |
| B2 | 23 | 23 of the challenge key's 24 shots; evaluation only | 23 |

B1 and B2 are never pooled. Inputs for his shots that the corpus lacked were fetched on the
login node with `elmo_fetch.py` (11 shots; fdp exited 245 after fetching all 11, which was
checked by counting the files). Nothing is fetched inside the evaluation.

## What his labels are

| | B1 (58 shots) | B2 (23 shots) |
|---|---|---|
| individual 10 ms rows | 14,104 | 3,024 |
| of them the accepted prefill | 14,104 | 3,005 |
| of them edited (a cell the prefill did not have) | 0 | 19 (141474: 1, 161558: 1, 192855: 17) |
| prefilled onsets he removed | 0 | 19 (all in 163518) |
| crowd rows | 84 | 43 |
| uncertain time (ms; in no score) | 6,140 (10 shots) | 5,750 (141474, 163518) |
| his present time against the `elm_jump` draft's (shots scored) | 164,070 ms against 141,914 | 44,020 ms against 37,141 |
| draft time he kept (precision) | 0.976 | 0.969 |
| his time the draft had (recall) | 0.844 | 0.818 |

A row is `prefill_accepted` when its 10 ms cell contains an onset of the prefill table inside
its save's window, `edited` otherwise (`heldout.individual_provenance`; every row with its
provenance is in `individual_provenance.csv` next to the traces). The page flags the save
that wrote the prefill, but a later save of the same shot is not flagged, so the per-row
provenance and not the flag is the evidence. His independent work is the added present
time (about 16 to 19 % more than the draft) and the absent calls; his onset rows add almost
nothing the prefill did not hold.

## Models (frozen; no refit, no threshold change)

| model | what is run |
|---|---|
| elm-ours | mean event probability of the five `cv2` fold models on the whole shot, threshold 0.4173 (mean of the fold thresholds); checked against the traces `labeler.elm.serve` saved for 46 shots (largest difference 2.4e-7) |
| elm-ours onset | peaks of the mean second head at the mean fold onset threshold (0.35) |
| elm-inst | mean occupancy of the five `inst1` fold models at 0.3483; its onsets at 0.51, ungated, the decoding its development record chose |
| ELM-O | `elmo_benchmark.review_shot` at the published setting (eta 0.997), only where BES exists (30 of B1, 3 of B2) |
| elm-clock, elm-clock-v2, elm-jump | the suggestion tables (`elm_clock` v1 and v2, `elm_jump` v2); the tables do not cover all key shots (elm-jump covers 20 of B2's 23) |
| elm-clock-peaks | the ELM clock's peaks on the first finite filterscope, each the span of its half-prominence width (`challenge.clock_spans`), run on all 80 scored shots; added because the clock tables lack the key shots |
| always present | every bin present (a floor) |

## Scoring

- **Occupancy bins**, as for the old review: 50 ms bins, present where a crowd row touches
  the bin, absent where only absent rows do. His 10 ms individual rows give no bin, and
  uncertain rows are in no score. Intervals are 95 % shot bootstraps (1,000 resamples, seed
  20261003), the same code as `score.summarise`. The table's false alarm is the fraction of
  absent bins a method calls present.
- **Onset agreement**: detections within 5, 10 or 20 ms of one of his rows, matched one to
  one, nearest first. A detection counts as a false alarm in judged-present time or in
  absent time; none counts inside uncertain time or outside his labelled window. The
  detections are collapsed to the 10 ms cells his rows sit on before matching (a method
  firing three times in one cell is one detection), and the raw count is in the record.
- The `bes` panel keeps only the bins inside the chunks ELM-O analysed.
- `without a shared run day` drops the shots that share a run day with the reviewed 119
  (the fold models' training shots; none shares one with Smith's 211) and, in B2, the three
  shots with no run day on record (180445, 192855, 207788).

## Results: occupancy

Reference, the old reviewed set (pin 2026-10-17; out of fold, each shot's own fold model and
threshold; 119 shots, 12,481 bins). The `elm-ours`, `elm-clock`, `elm-clock-v2` and
`elm-jump` F1 and both bin counts equal the published record
(`outputs/labeler/elm/sources/evaluation.json`, sha256 `ed88aa55...2b63c`) to the last digit.
The `elm-inst` row is new on these bins (its development table scored the 12,409-bin
version, F1 0.831).

| method | shots | F1 | crowd-bin recall | absent-bin false alarm | AUROC |
|---|---|---|---|---|---|
| elm-ours | 119 | 0.828 [0.776, 0.869] | 0.851 [0.789, 0.902] | 0.115 [0.077, 0.155] | 0.940 [0.905, 0.966] |
| elm-inst | 119 | 0.829 [0.775, 0.874] | 0.873 [0.804, 0.933] | 0.130 [0.086, 0.178] | 0.946 [0.911, 0.972] |
| elm-jump | 119 | 0.823 [0.768, 0.869] | 0.889 [0.838, 0.937] | 0.161 [0.115, 0.210] | -- |
| elm-clock | 119 | 0.755 [0.679, 0.824] | 0.680 [0.580, 0.789] | 0.064 [0.034, 0.102] | -- |
| always present | 119 | 0.546 [0.487, 0.600] | 1.000 | 1.000 | 0.500 |

**B1: his 57 scored shots outside the key** (5,297 bins: 2,744 crowd, 2,553 absent; serving
ensemble).

| method | shots | F1 | crowd-bin recall | absent-bin false alarm | AUROC |
|---|---|---|---|---|---|
| elm-ours | 57 | 0.966 [0.948, 0.980] | 0.942 [0.908, 0.967] | 0.008 [0.002, 0.017] | 0.996 [0.990, 0.999] |
| elm-inst | 57 | 0.955 [0.920, 0.979] | 0.923 [0.862, 0.968] | 0.010 [0.004, 0.019] | 0.994 [0.988, 0.998] |
| elm-jump | 57 | 0.972 [0.953, 0.986] | 0.950 [0.916, 0.978] | 0.006 [0.003, 0.009] | -- |
| elm-clock (v1 and v2 alike) | 57 | 0.736 [0.623, 0.831] | 0.640 [0.499, 0.761] | 0.106 [0.024, 0.210] | -- |
| elm-clock-peaks | 57 | 0.800 [0.711, 0.866] | 0.868 [0.783, 0.929] | 0.324 [0.177, 0.454] | -- |
| always present | 57 | 0.683 [0.607, 0.742] | 1.000 | 1.000 | 0.500 |

B1 with BES (30 shots, 2,263 bins) and without a shared run day (36 shots, 3,123 bins):

| method | BES: F1 | BES: crowd recall | BES: false alarm | no shared day: F1 | no shared day: false alarm |
|---|---|---|---|---|---|
| elm-ours | 0.963 [0.933, 0.982] | 0.933 | 0.008 | 0.965 [0.938, 0.982] | 0.009 |
| elm-inst | 0.976 [0.952, 0.989] | 0.959 | 0.011 | 0.968 [0.943, 0.984] | 0.010 |
| elm-jump | 0.966 [0.944, 0.982] | 0.941 | 0.012 | 0.975 [0.955, 0.986] | 0.006 |
| elm-clock | 0.618 [0.442, 0.754] | 0.477 | 0.105 | 0.652 [0.465, 0.788] | 0.140 |
| elm-clock-peaks | 0.822 [0.733, 0.891] | 0.818 | 0.267 | 0.744 [0.601, 0.850] | 0.369 |
| ELM-O | 0.862 [0.756, 0.928] | 0.774 | 0.034 | -- | -- |
| always present | 0.756 | 1.000 | 1.000 | 0.600 | 1.000 |

**B2: the 23 challenge-key shots** (2,203 bins: 798 crowd, 1,405 absent; evaluation only).
The shot count differs by method (`elm-jump` has a table for 20 of the 23), so compare F1
across rows with care.

| method | shots | F1 | crowd-bin recall | absent-bin false alarm | AUROC |
|---|---|---|---|---|---|
| elm-ours | 23 | 0.856 [0.673, 0.969] | 0.883 [0.761, 0.964] | 0.103 [0.001, 0.286] | 0.976 [0.929, 0.998] |
| elm-inst | 23 | 0.916 [0.817, 0.975] | 0.926 [0.852, 0.980] | 0.055 [0.005, 0.131] | 0.991 [0.972, 0.999] |
| elm-jump | 20 | 0.923 [0.848, 0.981] | 0.860 [0.738, 0.967] | 0.003 [0.000, 0.006] | -- |
| elm-clock-peaks | 23 | 0.709 [0.497, 0.892] | 0.922 [0.881, 0.958] | 0.385 [0.123, 0.598] | -- |
| always present | 23 | 0.532 [0.359, 0.671] | 1.000 | 1.000 | 0.500 |

B2 with BES is 3 shots (264 bins: 33 crowd, 231 absent), so point estimates only: elm-ours
F1 0.596, elm-inst 0.738, elm-jump 0.985 (2 shots), ELM-O 0.881, clock-peaks 0.259, always
present 0.222. B2 without a shared run day (20 shots, 1,778 bins): elm-ours 0.935 [0.858,
0.978], elm-inst 0.951 [0.901, 0.982], elm-jump 0.923 [0.848, 0.981], clock-peaks 0.892
[0.770, 0.953].

What the numbers say, and do not:

- **B1 is near ceiling because the labels are largely the `elm_jump` draft** (which the
  models and the draft agree with: ours 0.966, jump 0.972). A high B1 score is agreement
  with a prefill that was accepted, not evidence the models are right where the prefill was
  wrong. No run-day effect shows on B1: the 36 shots with no shared day score the same as
  all 57.
- **B1's 0.966 is not a gain over the old set's 0.828.** The labels, the shots and the model
  (serving ensemble, not out of fold) all differ.
- **B2 is the harder, less prefill-bound test.** `elm-inst` (0.916) beats `elm-ours`
  (0.856) on all 23, and on the 20 shots with a known day (0.951 against 0.935). The
  intervals overlap widely; with 23 shots this is a tendency, not a ranking.
- **`elm-ours` loses most of its B2 score on one shot.** 207788 has a single absent row in
  his review (-1,510 to 8,500 ms; no ELMs). `elm-ours` calls 140 of its 150 bins present
  (92 % of the shot's time above threshold) and `elm-inst` 60 of 150: 140 of `elm-ours`'s 145
  and 60 of `elm-inst`'s 78 false-alarm bins on B2. This was not diagnosed. The `without a
  shared run day` B2 set differs from the all-23 set only by 207788, 180445 and 192855
  (no run day on record), so the jump from 0.856 to 0.935 is mostly this shot, not a
  run-day effect: no B2 shot shares a run day with a trained set.
- **Where B2's crowd bins are missed** (per-shot counts, `elm-ours` / `elm-inst` called of
  the shot's crowd bins): 141474 2/31 and 14/31; 204116 14/33 and 24/33; 206709 71/96 and
  77/96; 132651 18/23 and 22/23. The other shots with crowd bins are at or above 89 % for both models.
- **`elm_jump` on B2 has lower crowd-bin recall (0.860) than on B1 (0.950)**, which fits
  his adding about 19 % more present time than the draft held there (16 % on B1); the
  added time is where his edits show up.
- **ELM-O is lower than `elm-ours` and `elm-inst` on B1's 30 BES shots** (0.862 against
  0.963 and 0.976) and higher on B2's three (0.881 against 0.596 and 0.738; with three
  shots that is an anecdote). On the old set the three are within noise of each other
  (bes73: 0.841, 0.843, 0.847).

## Results: onsets (agreement with the accepted `elm_jump` prefill)

Tolerance 10 ms, detections collapsed to his 10 ms cells. The 5 and 20 ms scores are in the
record; they differ from these by at most 0.02 except `elm-ours`'s precision (0.486 at
5 ms, 0.770 at 20 ms on B1). Recall is over his rows inside the scored window.

| set | method | shots | precision | recall | F1 | detections in absent time |
|---|---|---|---|---|---|---|
| B1 | elm-ours (onset head) | 57 | 0.750 | 0.008 | 0.016 [0.008, 0.026] | 33 |
| B1 | elm-inst | 57 | 0.966 | 0.344 | 0.507 [0.361, 0.621] | 4 |
| B1 | elm-jump | 57 | 1.000 | 1.000 | 1.000 | 0 |
| B1 | elm-clock-peaks | 57 | 0.642 | 0.658 | 0.650 [0.553, 0.727] | 4,154 |
| B1 BES | ELM-O | 30 | 0.930 | 0.351 | 0.510 [0.406, 0.586] | 60 |
| B2 | elm-ours (onset head) | 23 | 0.750 | 0.019 | 0.037 [0.015, 0.068] | 18 |
| B2 | elm-inst | 23 | 0.995 | 0.357 | 0.526 [0.271, 0.725] | 0 |
| B2 | elm-jump | 20 | 1.000 | 0.999 | 1.000 [0.999, 1.000] | 0 |
| B2 | elm-clock-peaks | 23 | 0.443 | 0.752 | 0.557 [0.356, 0.764] | 2,827 |

- **`elm_jump` is circular** here (the rows are its own onsets, accepted). Unreduced to his
  cells, it fires about 2.3 times per cell, so counted raw it scores precision 0.434 (B1).
- **`elm-ours`'s onset head scores near 0 recall** because it was trained on the starts of
  isolated (non-crowd) reviewed spans, and his rows sit mostly in dense trains. That is
  expected and says nothing against its occupancy output, which is what the catalog uses.
- **`elm-inst`'s precision is high (0.97 to 0.99) and its recall is 0.34 to 0.36**: it finds
  about a third of the prefilled cells, with almost no detections in absent time.
- **The recall of his 19 edited B2 rows** is 0 of 19 for `elm-ours` and `elm-inst`, and 18
  of 19 for `elm-clock-peaks` at 10 ms (it fires at every filterscope peak, absent time
  included: 2,827 detections there). Too few rows (19, in 3 shots) to rank methods on.

## Caveats

- **Circularity** (above). Read his rows as a reviewed `elm_jump` output with edits, in
  which spans carry his judgement and the onsets carry the draft's.
- **Run-day overlap.** 21 of his 58 B1 shots share a run day with the reviewed 119 (the
  fold models' training set), none with Smith's 211; no B2 shot shares one, and three B2
  shots have no run day on record. The run date is the operator log's, else the cohort's
  `run_id`. No effect shows in B1's scores, and the 36-shot set without a shared day is
  reported beside every B1 table.
- **The serving ensemble is not out of fold on the old set.** Panel A therefore uses each
  shot's own fold model and threshold; B1 and B2 use the ensemble, since none of his shots
  is in any fold.
- **Small sets.** B2 has 23 shots and its BES panel 3; its intervals are wide, and one shot
  (207788) moves `elm-ours` by 0.08 in F1.
- **`elm-clock-peaks` is a method defined here** (see Models), not a table in the catalog.
  `elm-clock` v1 and v2 call the same bins on every set, so they score alike.
- **Two cv2 hashes.** The `cv2` run's `run.json` records review labels with sha256
  `6ae3497e...` (13,783 bytes), not the 2026-10-17 pin's. The pin reproduces the published
  numbers exactly (above), so the scored bins are the same; the difference pre-dates this
  work (the 2026-10-11 regrid).
- **The models score 0.83 on the old set and 0.96 on B1**; the difference is as much the
  labels and shots as the models. Nothing here is retrained or tuned (standing decision:
  report, do not chase).

## Reproduce

```bash
export LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labeler
export SHOT_DESIGN_DATA_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/shot_design
export LABELER_LABEL_TABLES=/scratch/gpfs/EKOLEMEN/nc1514/events
# login node, logged in to fdp: his shots' missing inputs (skips shots already fetched)
pixi run --frozen -e labeler fdp run python scripts/labeler/elmo_fetch.py \
    --from review --review-pin 2026-10-23
# one SLURM CPU job (1 CPU, 900M; the sizing comment in the script records the gate runs)
sbatch --output=$LABELER_ROOT/runs/slurm/%j.out --error=$LABELER_ROOT/runs/slurm/%j.out \
    scripts/labeler/elm_heldout.sbatch
python -m labeler.jobstats --job-id <id> --cpu-only
```

The job holds the `elm_jump` table, both pins and the old panel's published record
(`--published`, default the main checkout's `outputs/labeler/elm`) as read-only inputs and
changes nothing else. The `--skip-old` and `--shots` flags make a pilot. The memory use
(98 % of 900M) is tight; give a rerun `--mem=1000M`. Two runs gave identical panels, per-shot
records, models and run-day sets.

## Suggested README lines (not applied)

`docs/labeler/events/edge_localized_mode/README.md`'s Models block is generated by
`scripts/labeler/elm_protocol.py`, which does not read this evaluation, and was not edited.
For the owner to place:

- elm-ours | 2026_10_24 | AUROC: 0.996 [0.990, 0.999] | AUPRC: -- | F1: 0.966 [0.948, 0.980]
  (frozen serving ensemble at 0.417 on 57 of Ricardo Shousha's 2026-10-23 shots, none in
  any training set, 5,297 bins; his rows are largely the accepted elm_jump prefill; on the
  23 challenge-key shots, evaluation only: AUROC 0.976 [0.929, 0.998], F1 0.856 [0.673,
  0.969]; elm_heldout.md)
- elm-inst | 2026_10_24 | AUROC: 0.994 [0.988, 0.998] | AUPRC: -- | F1: 0.955 [0.920, 0.979]
  (the same 57 shots; on the 23 key shots AUROC 0.991 [0.972, 0.999], F1 0.916 [0.817,
  0.975])
- the README's sentence "A frozen-model score on the blind split awaits review of those
  shots" would read: "A frozen-model score on 57 shots outside every training set (not the
  blind split) is in elm_heldout.md; his labels are largely the accepted elm_jump prefill."
