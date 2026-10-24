# ELM models on Ricardo Shousha's review of 2026-10-23

Status: **run 2026-10-24 as one SLURM CPU job (job 2977033, 900M, code at git 8d9339c4;
jobstats gate passed, CPU 78.6 %, memory 72.8 %; the sizing runs are in the comment of
`scripts/labeler/elm_heldout.sbatch`).** The frozen ELM models, scored
without refitting anything on the 81 shots Ricardo Shousha reviewed on 2026-10-23.
Record: `outputs/labeler/elm/heldout_2026_10_23/evaluation.json` (copy at
`$LABELER_ROOT/round4/elm/heldout/2026-10-23/evaluation.json`). Code:
`src/labeler/elm/heldout.py`, `scripts/labeler/elm_heldout_evaluate.py`,
`scripts/labeler/elm_heldout.sbatch`.

## What this shows

- **No model trained on any of his shots**: none is in a cross-validation fold, in Smith's
  211, or in the cohort's blind test split. This is the first score of the `elm-ours`
  serving ensemble (five `cv2` fold models at the mean threshold 0.417) outside its
  training. It is not the blind split's score.
- **The deployed score is the score on the servable shots.** `labeler.elm.serve` publishes
  a shot only when `serve.input_problems` finds nothing wrong with its input. 15 of his 80
  scored shots fail that screen (8 of the 23 challenge-key shots, 7 of 57 others), mostly a
  filterscope with no signal. On the 15 servable key shots (B2) `elm-ours` has F1 0.922
  [0.826, 0.982] and calls 0.3 % of the absent bins present; on all 23, unscreened, F1
  0.856 and 10.3 %. Screened-out shots are not what the catalog would hold.
- **His labels are not independent of `elm_jump`.** His review opened on the `elm_jump` v2
  onsets as a prefill and he kept 17,109 of his 17,128 individual rows as they were. The
  other-than-key shots (B1) score near ceiling for every method that behaves like
  `elm_jump` (F1 0.955 to 0.972); his independent information is his added present time,
  his absent calls and the key shots.
- **Onset agreement is agreement with the accepted `elm_jump` prefill**, not onset
  accuracy; `elm_jump` scores 1.000 on it by construction.
- The old reviewed set reproduces the published `sources` record (below); the new numbers
  are not comparable with it (other shots, another labelling procedure).

## Inputs and pinned labels

| input | file | sha256 |
|---|---|---|
| old review labels, pin 2026-10-10 (default of the ELM scripts) | `$LABELER_ROOT/benchmarks/elm/review_pins/2026-10-10/labels.csv` (119 shots, no history kept) | `6ae3497e...e380` |
| old review labels, pin 2026-10-17 (panel A, `labeler.elm.sources`) | `.../2026-10-17/labels.csv` (120 shots, regridded) | `d483fed8...6276` |
| his review labels, pin 2026-10-23 | `.../2026-10-23/labels.csv` (19,079 rows, 201 shots) and `history.jsonl` | `b54c1a27...8e02`, `e95ceb0c...b5fd` |
| `elm_jump` v2 table (his prefill) | `suggestions/elm_jump/v2/edge_localized_mode_suggest_elm_jump_v2.csv` | `aa936d3c...45c4`, the `source_sha256` of all 81 of his last saves |
| `elm_jump` v2 onsets | the table's `_onsets.csv` | `d831967a...07ac`; his saves do not hash it, so it is assumed to be the prefill's source (the rows it explains support that: 17,109 of 17,128) |

Every pin is a read-only copy with a `SHA256SUMS`; the script re-hashes them
(`heldout.verify_pin`) before scoring. The 2026-10-23 table holds the same rows for the
2026-10-17 pin's shots (`pins_agree_on_the_old_shots`).

**Which pin is which.** `prepare.review_csv(paths)` returns the **2026-10-10 pin**, the
119 shots as they stood before the 10 ms regrid of 2026-10-10. Those are the bytes the
`cv2` run trained on (its `run.json` records the same sha256 and 13,783 bytes) and the
ones the README's Models lines of 2026_10_03, 2026_10_01 and 2026_09_13 were scored on
(12,409 bins; bes73 6,843), and every shot has a prepared input, so an unfiltered
`train.load` runs. The **2026-10-17 pin** is the regridded table (12,481 bins) plus 189093,
which has no input; `labeler.elm.sources` and panel A read it. `prepare.review_csv(paths,
None)` is the live table. A pin that does not exist is an error, never a fallback.
`train.check_no_holdout` refuses any `protect: holdout` shot (the challenge key's 24) in
`train.load` and in `elm_inst.py prepare`, failing closed when the registry cannot be read;
`elm_dsm_train.py` reads neither (see `elm_challenge.md`).

## His shots and labels

| | B1 (58 shots) | B2 (23 shots) |
|---|---|---|
| what | not on a holdout list | 23 of the challenge key's 24 shots, evaluation only |
| scored | 57 (203401 has no `\BCI::DENV2F`, `%TREE-E-NODATA`) | 23 |
| servable (`serve.input_problems` empty) | 50 | 15 |
| individual 10 ms rows | 14,104 | 3,024 |
| of them the accepted prefill / edited | 14,104 / 0 | 3,005 / 19 (141474: 1, 161558: 1, 192855: 17) |
| prefilled onsets he removed | 0 | 19 (all in 163518) |
| crowd rows | 84 | 43 |
| uncertain time (ms; in no score) | 6,140 (10 shots) | 5,750 (141474, 163518) |
| his present time against the draft's | 164,070 ms against 141,914 | 44,020 ms against 37,141 |
| draft time he kept / his time the draft had | 0.976 / 0.844 | 0.969 / 0.818 |

B1 and B2 are never pooled. The 11 shots whose inputs the corpus lacked were fetched on the
login node with `elmo_fetch.py` (fdp exited 245 after fetching all 11, checked by counting
the files); nothing is fetched inside the evaluation. A row is `prefill_accepted` when its
10 ms cell holds an onset of the prefill table inside its save's window and `edited`
otherwise (`heldout.individual_provenance`; every row with its provenance is in
`individual_provenance.csv` next to the traces). The page flags only the save that wrote the
prefill, so the per-row provenance, not the flag, is the evidence.

## Models (frozen; no refit, no threshold change)

| model | what is run |
|---|---|
| elm-ours | mean event probability of the five `cv2` fold models on the whole shot at 0.4173 (mean of the fold thresholds); equals the traces `serve` saved for 46 shots (largest difference 2.4e-7) |
| elm-ours onset | peaks of the mean second head at the mean fold onset threshold (0.35) |
| elm-inst | mean occupancy of the five `inst1` fold models at 0.3483; onsets at 0.51, ungated, the decoding its development record chose |
| ELM-O | `elmo_benchmark.review_shot` at the published setting (eta 0.997), where BES exists (30 of B1, 3 of B2) |
| elm-clock, elm-clock-v2, elm-jump | the suggestion tables; they do not cover all key shots (`elm-jump` holds 20 of B2's 23) |
| elm-clock-peaks | the clock's peaks on the first finite filterscope, each the span of its half-prominence width (`challenge.clock_spans`), run on every scored shot; defined here because the clock tables lack the key shots |
| always present | every bin present (a floor) |

**The input screen.** Each shot's input goes through `serve.input_problems` as `serve`
applies it: both density chords present and not screened as a failed digitiser, and each
filterscope's median level inside the 5th to 95th percentile of the 119 reviewed shots'
levels, widened by 0.6 decade. The `servable` tables below are on the shots that pass; the
`all` tables are the models' unscreened output. Every shot's problems are in the record
(`per_shot`, `input_screen`). A median of exactly 10^12.00 is the input's clip floor, a
channel with no signal above it.

| set | screened-out shots (problem) |
|---|---|
| B2 (8) | 163518 (FS02 10^12.30, below the range), 165703 (FS02 and FS04 near the floor), 166828 (FS02 floor), 171322 (FS03 and FS04 floor), 180257 and 180264 (FS02 10^16.20 and 10^16.21, 0.01 decade above the limit), 194479 (both density chords empty), 207788 (FS02 floor) |
| B1 (7) | 185953 (FS02 floor), 193802, 194025 and 194401 (both density chords empty), 196632, 199907 and 202205 (FS03 floor) |

## Scoring

- **Occupancy bins**, as for the old review: the 50 ms bins of `labels.scored_bins`. A bin is
  scored only when it lies wholly inside one crowd or one absent row and inside analysed
  time: crowd bins are positives, absent bins negatives. His 10 ms individual rows give no
  bin and uncertain rows are in no score. Intervals are 95 % shot bootstraps (1,000
  resamples, seed 20261003, `score.summarise`). The false alarm is the share of absent bins
  a method calls present.
- **Onset agreement**: detections matched one to one, nearest first, to his rows. A
  detection is a false alarm in judged-present or absent time; none counts in uncertain
  time or outside his labelled window. Detections are collapsed to the 10 ms cells his rows
  sit on (a method firing three times in one cell is one detection; the raw count is in the
  record), and cells are compared by their centres: a **10 ms tolerance is the same or an
  adjacent cell** (a raw time difference of up to nearly 20 ms), **5 ms the same cell**, 20 ms
  within two cells.
- The `bes` panel keeps the bins inside the chunks ELM-O analysed. `without a shared run
  day` drops the shots that share a run day with the 119 reviewed shots. The `bes` and run-day
  tables are on all scored shots, not only the servable ones.

## Results: occupancy

**Reference, the old reviewed set** (pin 2026-10-17, out of fold: each shot's own fold model
and threshold; 119 shots, 12,481 bins). The `elm-ours`, `elm-clock`, `elm-clock-v2` and
`elm-jump` F1 and the bin counts (all119 12,481, bes73 6,881) equal
`outputs/labeler/elm/sources/evaluation.json` (2026-10-14) to the last digit. They are not
the README Models lines of 2026_10_03 and earlier (elm-ours 0.830, elm-clock 0.757, elm-elmo
0.842), which were scored on the 2026-10-10 pin (12,409 and 6,843 bins).

| method | shots | F1 | crowd-bin recall | absent-bin false alarm | AUROC |
|---|---|---|---|---|---|
| elm-ours | 119 | 0.828 [0.776, 0.869] | 0.851 [0.789, 0.902] | 0.115 [0.077, 0.155] | 0.940 [0.905, 0.966] |
| elm-inst | 119 | 0.829 [0.775, 0.874] | 0.873 [0.804, 0.933] | 0.130 [0.086, 0.178] | 0.946 [0.911, 0.972] |
| elm-jump | 119 | 0.823 [0.768, 0.869] | 0.889 [0.838, 0.937] | 0.161 [0.115, 0.210] | -- |
| elm-clock | 119 | 0.755 [0.679, 0.824] | 0.680 [0.580, 0.789] | 0.064 [0.034, 0.102] | -- |
| always present | 119 | 0.546 [0.487, 0.600] | 1.000 | 1.000 | 0.500 |

**B2, the challenge-key shots, servable (15 shots, 1,532 bins: 601 crowd, 931 absent).**
`elm-jump` has a table for 13 of the 15.

| method | shots | F1 | crowd-bin recall | absent-bin false alarm | AUROC |
|---|---|---|---|---|---|
| elm-ours (deployed) | 15 | 0.922 [0.826, 0.982] | 0.860 [0.705, 0.968] | 0.003 [0.000, 0.009] | 0.993 [0.982, 0.999] |
| elm-inst | 15 | 0.946 [0.889, 0.986] | 0.920 [0.827, 0.989] | 0.016 [0.003, 0.036] | 0.996 [0.989, 0.999] |
| elm-jump | 13 | 0.910 [0.819, 0.989] | 0.837 [0.698, 0.984] | 0.003 [0.000, 0.007] | -- |
| elm-clock-peaks | 15 | 0.730 [0.477, 0.926] | 0.922 [0.877, 0.962] | 0.389 [0.070, 0.642] | -- |
| always present | 15 | 0.564 [0.374, 0.694] | 1.000 | 1.000 | 0.500 |

**B2, all 23 scored shots, unscreened model output** (2,203 bins: 798 crowd, 1,405 absent;
`elm-jump` 20 shots).

| method | shots | F1 | crowd-bin recall | absent-bin false alarm | AUROC |
|---|---|---|---|---|---|
| elm-ours | 23 | 0.856 [0.673, 0.969] | 0.883 [0.761, 0.964] | 0.103 [0.001, 0.286] | 0.976 [0.929, 0.998] |
| elm-inst | 23 | 0.916 [0.817, 0.975] | 0.926 [0.852, 0.980] | 0.055 [0.005, 0.131] | 0.991 [0.972, 0.999] |
| elm-jump | 20 | 0.923 [0.848, 0.981] | 0.860 [0.738, 0.967] | 0.003 [0.000, 0.006] | -- |
| elm-clock-peaks | 23 | 0.709 [0.497, 0.892] | 0.922 [0.881, 0.958] | 0.385 [0.123, 0.598] | -- |
| always present | 23 | 0.532 [0.359, 0.671] | 1.000 | 1.000 | 0.500 |

**B1, servable (50 shots, 4,942 bins: 2,646 crowd, 2,296 absent)** and all 57 scored
(5,297 bins), F1 with the false alarm:

| method | servable F1 | servable crowd recall | servable false alarm | all 57: F1 | all 57: false alarm |
|---|---|---|---|---|---|
| elm-ours | 0.970 [0.953, 0.982] | 0.949 | 0.009 | 0.966 [0.948, 0.980] | 0.008 |
| elm-inst | 0.957 [0.922, 0.981] | 0.926 | 0.010 | 0.955 [0.920, 0.979] | 0.010 |
| elm-jump | 0.971 [0.953, 0.986] | 0.949 | 0.006 | 0.972 [0.953, 0.986] | 0.006 |
| elm-clock (v1 and v2 alike) | 0.751 [0.643, 0.841] | 0.663 | 0.118 | 0.736 [0.623, 0.831] | 0.106 |
| elm-clock-peaks | 0.810 [0.734, 0.878] | 0.892 | 0.358 | 0.800 [0.711, 0.866] | 0.324 |
| always present | 0.697 | 1.000 | 1.000 | 0.683 | 1.000 |

On B1's 30 BES shots (2,263 bins) F1 is 0.963 for `elm-ours`, 0.976 for `elm-inst`, 0.966 for
`elm-jump`, 0.862 [0.756, 0.928] for ELM-O (crowd recall 0.774, false alarm 0.034) and 0.618
for the clock. On B1's 36 shots without a shared run day (3,123 bins) F1 is 0.965 for
`elm-ours`, 0.968 for `elm-inst`, 0.975 for `elm-jump` and 0.652 for the clock. Both subsets
are on all scored shots, not only the servable ones.

What the numbers say, and do not:

- **B1 is near ceiling because the labels are largely the `elm_jump` draft**; a high B1
  score is agreement with an accepted prefill, not evidence the models are right where the
  prefill was wrong. Screening changes B1 little (0.966 to 0.970). It is also not a gain over
  the old set's 0.828: other labels, other shots, the ensemble rather than out-of-fold models.
- **B2 is the informative test, and on the servable shots `elm-ours` is a good detector.**
  `elm-inst` is ahead by 0.02 and the intervals overlap fully; with 15 shots that is a
  tendency, not a ranking.
- **The unscreened B2 loss is the input screen at work.** `elm-ours` on 207788 (a single
  absent row in his review, no ELMs) calls 140 of 150 bins present; its FS02 median is
  10^12.00, the clip floor, a dead channel, and `serve` would not publish the shot. That one
  shot is 140 of `elm-ours`'s 145 false-alarm bins on all 23 shots (the 15 servable shots
  have 3), so the unscreened false alarm of 0.103 is that shot. 163518, 165703, 166828 and
  171322 also have a channel at or near the floor; 180257 and 180264 fail by 0.01 decade, so
  the screen's edge, not dead data, decides them.
- **The screen is the deployed pipeline's, not a model result**: a shot refused here has no
  `elm-ours` row in the catalog, so the all-shots rows are for diagnosis only. No B2 shot
  shares a run day with a trained set; 180445, 192855 and 207788 have no run day on record.
- **`elm_jump` has lower crowd-bin recall on B2 (0.860 on 20 shots) than on B1 (0.950)**,
  consistent with his adding about 19 % more present time than the draft held there (16 %
  on B1): the added time is where his edits show.
- **ELM-O is lower than `elm-ours` and `elm-inst` on B1's 30 BES shots** (0.862 against
  0.963 and 0.976) and higher on B2's three (0.881 against 0.596 and 0.738, point estimates
  only; elm-jump 0.985 on two). On the old set the three are within noise (bes73: 0.841,
  0.843, 0.847).

## Results: onsets (agreement with the accepted `elm_jump` prefill)

Tolerance 10 ms, detections collapsed to his 10 ms cells (all scored shots; the servable
subsets differ by at most 0.07 in precision or recall, in the record). The 5 and 20 ms
scores differ from these by at most 0.02 except `elm-ours`'s precision (0.486 at 5 ms,
0.770 at 20 ms on B1). Recall is over his rows inside the scored window.

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

- **`elm_jump` is circular** here: the rows are its own onsets, accepted. Unreduced to his
  cells it fires about 2.3 times per cell and scores precision 0.434 (B1).
- **`elm-ours`'s onset head scores near 0 recall**: it was trained on the starts of isolated
  (non-crowd) reviewed spans and his rows sit mostly in dense trains. That says nothing
  against its occupancy output, which is what the catalog uses.
- **`elm-inst` finds about a third of the prefilled cells** (precision 0.97 to 0.99) with
  almost no detections in absent time.
- **His 19 edited B2 rows** are found 0 of 19 by `elm-ours` and `elm-inst` and 18 of 19 by
  `elm-clock-peaks` (it fires at every filterscope peak, absent time included): too few rows,
  in 3 shots, to rank methods on.

## Caveats

- **Circularity** (above): read his rows as a reviewed `elm_jump` output with edits, whose
  spans carry his judgement and whose onsets carry the draft's.
- **Run-day overlap.** 21 of his 58 B1 shots share a run day with the 119 reviewed shots (the
  fold models' training shots), none with Smith's 211; no B2 shot shares one. The run date is
  the operator log's, else the cohort's `run_id`. No effect shows in B1.
- Panel A uses each old shot's own fold model and threshold (the ensemble would be in
  sample); B1 and B2 use the ensemble. B2 has 23 shots (15 servable) and a 3-shot BES panel,
  so its intervals are wide. `elm-clock-peaks` is defined here, not a catalog table; `elm-clock`
  v1 and v2 call the same bins on every set. Nothing is retrained or tuned.

## Reproduce

```bash
export LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labeler
export SHOT_DESIGN_DATA_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/shot_design
export LABELER_LABEL_TABLES=/scratch/gpfs/EKOLEMEN/nc1514/events
# login node, logged in to fdp: his shots' missing inputs (skips shots already fetched)
pixi run --frozen -e labeler fdp run python scripts/labeler/elmo_fetch.py \
    --from review --review-pin 2026-10-23
# one SLURM CPU job; the sbatch header records the sizing and the gate runs
sbatch --output=$LABELER_ROOT/runs/slurm/%j.out --error=$LABELER_ROOT/runs/slurm/%j.out \
    scripts/labeler/elm_heldout.sbatch
python -m labeler.jobstats --job-id <id> --cpu-only
```

`--published` (default `outputs/labeler/elm` of the checkout) names the records the job reads:
`inst/evaluation.json` for the onset decoding and `sources/evaluation.json` for panel A's
check; a checkout without them passes the main checkout's. The job reads the pins and the
tables and changes nothing else. `--skip-old` and `--shots` make a pilot. Repeated runs gave
identical panels and per-shot records.

## Suggested README lines (not applied)

`docs/labeler/events/edge_localized_mode/README.md`'s Models block is generated by
`scripts/labeler/elm_protocol.py`, which does not read this evaluation, and was not edited.
For the owner to place:

- elm-ours | 2026_10_24 | AUROC: 0.993 [0.982, 0.999] | AUPRC: -- | F1: 0.922 [0.826, 0.982]
  (frozen serving ensemble at 0.417 on the 15 servable of the 23 ELM challenge-key shots,
  which no model trained on, reviewed by Ricardo Shousha 2026-10-23, 1,532 bins; false alarm
  0.003; on his 50 servable other shots, 4,942 bins, F1 0.970, near ceiling because his rows
  are largely the accepted elm_jump prefill; elm_heldout.md)
- elm-inst | 2026_10_24 | AUROC: 0.996 [0.989, 0.999] | AUPRC: -- | F1: 0.946 [0.889, 0.986]
  (the same 15 key shots; his 50 other servable shots F1 0.957)
- the README's sentence "A frozen-model score on the blind split awaits review of those
  shots" would read: "A frozen-model score on 65 servable shots outside every training set
  (not the blind split) is in elm_heldout.md; his labels are largely the accepted elm_jump
  prefill."
