# ELM models on Ricardo Shousha's review of 2026-10-23

Status: **run 2026-10-24 as one SLURM CPU job (job 2977052, 900M, code at git 85f85adf;
jobstats gate passed, CPU 73.7 %, memory 74.1 %; the sizing runs are in the comment of
`scripts/labeler/elm_heldout.sbatch`).** The frozen ELM models, scored without refitting
anything on the 81 shots he reviewed. Record:
`outputs/labeler/elm/heldout_2026_10_23/evaluation.json` (copy at
`$LABELER_ROOT/round4/elm/heldout/2026-10-23/evaluation.json`). Code:
`src/labeler/elm/heldout.py`, `scripts/labeler/elm_heldout_evaluate.py`,
`scripts/labeler/elm_heldout.sbatch`.

## What this shows

- **None of the scored models trained on any of his shots**: none is in a cross-validation
  fold, in Smith's 211, or in the cohort's blind test split. (`elm-dsm`, not scored here,
  did train on two: 163518 in B2 and 190627 in B1, `training_membership.json`.) This is the
  first score of the `elm-ours` serving ensemble (five `cv2` fold models at the mean
  threshold 0.417) on review labels of shots outside its training; it is not the blind
  split's score. `elm_challenge.md` had scored the same ensemble at the same threshold on
  the 24 key windows against Eldon's key: F1 0.827 [0.660, 0.926] on 934 touch bins
  (`outputs/labeler/elm/challenge/evaluation.json`).
- **Servable means input-screened, not published.** `labeler.elm.serve` publishes a shot
  only if its input passes `serve.input_problems` (15 of his 80 scored shots fail it,
  mostly a filterscope with no signal) and, the owner's rule of 2026-10-11, it is at least
  5 % present. **Servable** applies the first rule only and is the basis of the scores
  below; **published** (9 of the 15 key shots, 38 of the 50 others) is given beside it. The
  near-empty shots stay in the servable scores: a shot where the model finds nothing adds
  almost only true negatives, and filtering on the model's own output would hide those
  misses.
- **The occupancy score does not see his isolated ELMs, and both models miss them.** A 50 ms
  bin must lie wholly inside one crowd or absent row, so his individual rows outside his
  crowd spans give no bin. On the servable shots `elm-ours`'s present runs touch 0.949 of
  the 11,015 individual rows inside his crowd spans in B1 and 0.028 of the 1,826 outside
  them (`elm-inst` 0.920 and 0.033); in B2 0.815 and 0.107 of 168. On 11 servable shots
  whose only present labels are individual rows it touches none of 1,191. `elm-ours` is a
  detector of ELMing periods, not of single ELMs. Whether it misses these ELMs or the rows
  are prefill nobody checked is not known.
- **His labels are not independent of `elm_jump`.** His review opened on the `elm_jump` v2
  onsets as a prefill and he kept 17,109 of his 17,128 individual rows as they were. B1
  scores near ceiling for every method that behaves like `elm_jump` (F1 0.955 to 0.972); his
  independent information is his added present time, his absent calls and the key shots.
  Onset agreement is agreement with the accepted prefill, not onset accuracy: `elm_jump`
  scores 1.000 on it by construction.
- On the 15 servable key shots (B2) `elm-ours` has F1 0.922 [0.826, 0.982] and calls 0.3 %
  of the absent bins present (all 23, unscreened: 0.856 and 10.3 %, the shots the screen
  refuses), and F1 0.908 [0.782, 0.969] on the key's own bins (below). The old reviewed set
  reproduces the published `sources` record (below); the new numbers are not comparable
  with it.

## Inputs and pinned labels

| input | file under `$LABELER_ROOT/benchmarks/elm/review_pins/` | sha256 |
|---|---|---|
| old labels, pin 2026-10-10 (default of the ELM scripts) | `2026-10-10/labels.csv`: 119 shots, no history; file mtime 2026-10-01, before the regrid the pin is named for | `6ae3497e...e380` |
| old labels, pin 2026-10-17 (panel A, `labeler.elm.sources`) | `2026-10-17/labels.csv`: 120 shots, regridded | `d483fed8...6276` |
| his labels, pin 2026-10-23 | `2026-10-23/labels.csv` (19,079 rows, 201 shots) and `history.jsonl` | `b54c1a27...8e02`, `e95ceb0c...b5fd` |
| `elm_jump` v2 table (his prefill) | `suggestions/elm_jump/v2/edge_localized_mode_suggest_elm_jump_v2.csv` | `aa936d3c...45c4`, the `source_sha256` of all 81 of his last saves |
| `elm_jump` v2 onsets | the table's `_onsets.csv` | `d831967a...07ac`; not hashed by his saves, assumed to be the prefill's source (it explains 17,109 of 17,128 rows) |

Every pin is a read-only copy (files and folder) with a `SHA256SUMS`, re-hashed by
`heldout.verify_pin` before scoring; the 2026-10-23 table holds the 2026-10-17 pin's rows
unchanged. `prepare.review_csv(paths)` returns the 2026-10-10 pin: the bytes the `cv2` run
trained on (same sha256 and 13,783 bytes in its `run.json`) and the README Models lines of
2026_10_03 and earlier were scored on (12,409 bins, bes73 6,843); every shot has a prepared
input. The 2026-10-17 pin is the regridded table (12,481 bins) plus 189093, which has no
input; `review_csv(paths, None)` is the live table, a missing pin is an error.
`train.check_no_holdout` refuses any `protect: holdout` shot (the challenge key's 24) in
`train.load` and `elm_inst.py prepare`, failing closed (`elm_dsm_train.py` reads neither; see
`elm_challenge.md`).

## His shots and labels

| | B1 (58 shots) | B2 (23 shots) |
|---|---|---|
| what | not on a holdout list | 23 of the challenge key's 24 shots, evaluation only |
| scored / servable / published | 57 (203401 has no `\BCI::DENV2F`) / 50 / 38 | 23 / 15 / 9 |
| individual 10 ms rows: accepted prefill / edited | 14,104: 14,104 / 0 | 3,024: 3,005 / 19 (141474: 1, 161558: 1, 192855: 17) |
| prefilled onsets he removed; crowd rows | 0; 84 | 19 (all in 163518); 43 |
| servable shots with no crowd row (rows) | 8 (1,114) | 3 (77): 192048, 192855, 203788 |
| uncertain time (ms; in no score) | 6,140 (10 shots) | 5,750 (141474, 163518) |
| his present time against the draft's (B1: the 57 scored) | 164,070 ms against 141,914 | 44,020 ms against 37,141 |

B1 and B2 are never pooled: B2 is the key's shots by the vendored key, and the holdout
registry must name the same shots or the run stops (`heldout.split_panels`). The 11 shots
whose inputs the corpus lacked were fetched on the login node with `elmo_fetch.py` (fdp
exited 245 after fetching all 11, checked by counting the files); nothing is fetched inside
the evaluation. A row is `prefill_accepted` when its 10 ms cell holds an onset of the prefill
table inside its save's window and `edited` otherwise (`heldout.individual_provenance`;
every row with its provenance is in `individual_provenance.csv` beside the traces); the page
flags only the save that wrote the prefill, so the per-row provenance is the evidence.

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

**The input screen** is `serve.input_problems` as `serve` applies it: both density chords
present and not a failed digitiser, and each filterscope's median level inside the 5th to
95th percentile of the 119 reviewed shots' levels, widened by 0.6 decade; `published` also
needs `serve.present_fraction` of the shot at least 0.05. A median of exactly 10^12.00 is the
input's clip floor, a channel with no signal. Screened out, B2 (8): 163518 (FS02 below the
range), 165703 (FS02 and FS04 near the floor), 166828 (FS02 floor), 171322 (FS03 and FS04
floor), 180257 and 180264 (FS02 0.01 decade above the limit), 194479 (both density chords
empty), 207788 (FS02 floor); B1 (7): 185953 (FS02 floor), 193802, 194025 and 194401 (both
density chords empty), 196632, 199907 and 202205 (FS03 floor). The `all` tables are the
models' unscreened output, for diagnosis: a refused shot has no `elm-ours` row in the catalog.

## Scoring

- **Occupancy bins**, as for the old review: the 50 ms bins of `labels.scored_bins`. A bin is
  scored only when it lies wholly inside one crowd or one absent row and inside analysed
  time: crowd bins are positives, absent bins negatives. His 10 ms individual rows give no
  bin and uncertain rows are in no score. Intervals are 95 % shot bootstraps (1,000
  resamples, seed 20261003, `score.summarise`). The false alarm is the share of absent bins
  a method calls present.
- **Individual-row presence** (`heldout.row_presence_set`): of his individual 10 ms rows
  inside analysed time, the share a method's present runs overlap at all, split by whether
  the row's centre lies inside one of his crowd rows. Present runs are the evaluation's
  detection rule (the 50 ms moving mean of a trace at its threshold, `labels.runs_of`, or a
  table's spans); `raw 1 ms` reads the trace itself, a second reading that moves each
  figure by at most 0.02, up or down (the largest move in the record is 0.017). Intervals
  are 95 % shot bootstraps. It is the only score of an isolated ELM; the rows outside his
  crowd spans give no occupancy bin.
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
`elm-jump` F1 and the bin counts equal `outputs/labeler/elm/sources/evaluation.json`
(2026-10-14) to the last digit. They are not the README Models lines of 2026_10_03 and
earlier (elm-ours 0.830), which were scored on the 2026-10-10 pin (12,409 bins).

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
| elm-ours | 15 | 0.922 [0.826, 0.982] | 0.860 [0.705, 0.968] | 0.003 [0.000, 0.009] | 0.993 [0.982, 0.999] |
| elm-inst | 15 | 0.946 [0.889, 0.986] | 0.920 [0.827, 0.989] | 0.016 [0.003, 0.036] | 0.996 [0.989, 0.999] |
| elm-jump | 13 | 0.910 [0.819, 0.989] | 0.837 [0.698, 0.984] | 0.003 [0.000, 0.007] | -- |
| elm-clock-peaks | 15 | 0.730 [0.477, 0.926] | 0.922 [0.877, 0.962] | 0.389 [0.070, 0.642] | -- |
| always present | 15 | 0.564 [0.374, 0.694] | 1.000 | 1.000 | 0.500 |

**B1, servable (50 shots, 4,942 bins: 2,646 crowd, 2,296 absent)** and all 57 scored
(5,297 bins):

| method | servable F1 | servable crowd recall | servable false alarm | all 57: F1 | all 57: false alarm |
|---|---|---|---|---|---|
| elm-ours | 0.970 [0.953, 0.982] | 0.949 | 0.009 | 0.966 [0.948, 0.980] | 0.008 |
| elm-inst | 0.957 [0.922, 0.981] | 0.926 | 0.010 | 0.955 [0.920, 0.979] | 0.010 |
| elm-jump | 0.971 [0.953, 0.986] | 0.949 | 0.006 | 0.972 [0.953, 0.986] | 0.006 |
| elm-clock (v1 and v2 alike) | 0.751 [0.643, 0.841] | 0.663 | 0.118 | 0.736 [0.623, 0.831] | 0.106 |
| elm-clock-peaks | 0.810 [0.734, 0.878] | 0.892 | 0.358 | 0.800 [0.711, 0.866] | 0.324 |
| always present | 0.697 | 1.000 | 1.000 | 0.683 | 1.000 |

**Published** (B2 9 shots, 982 bins; B1 38 shots, 4,228 bins), F1: B2 `elm-ours` 0.947
[0.878, 0.989], `elm-inst` 0.960 [0.910, 0.993], `elm-jump` 0.904, **`elm-clock-peaks`
0.949 [0.924, 0.972]** (false alarm 0.022); B1 `elm-ours` 0.971 [0.956, 0.983], `elm-inst`
0.958 [0.923, 0.982], `elm-jump` 0.972, `elm-clock-peaks` 0.851 [0.787, 0.899]. On B2's 9
the clock ties `elm-ours`: there is no margin over it there. The view is chosen by
`elm-ours`'s own present fraction, and the 6 B2 shots it drops hold 353 of the clock's 362
false-alarm bins on the servable set (F1 0.730, false alarm 0.389). The 18 shots it drops
(6 of B2's servable, 12 of B1's; fractions in the record) hold 1,264 scored bins, 36 of
them crowd. **Unscreened B2** (23 shots, 2,203 bins): `elm-ours` F1 0.856 [0.673, 0.969],
false alarm 0.103; `elm-inst` 0.916 [0.817, 0.975], 0.055; `elm-jump` (20 shots) 0.923.

On B1's 30 BES shots (2,263 bins) F1 is 0.963 for `elm-ours`, 0.976 for `elm-inst`, 0.966 for
`elm-jump`, 0.862 [0.756, 0.928] for ELM-O (crowd recall 0.774, false alarm 0.034) and 0.618
for the clock. On B1's 36 shots without a shared run day (3,123 bins) F1 is 0.965 for
`elm-ours`, 0.968 for `elm-inst`, 0.975 for `elm-jump` and 0.652 for the clock. Both subsets
are on all scored shots, not only the servable ones.

- **B1 is near ceiling because the labels are largely the `elm_jump` draft**: a high B1
  score is agreement with an accepted prefill, not evidence the models are right where the
  prefill was wrong, and no gain over the old set's 0.828 (other labels, other shots, the
  ensemble rather than out-of-fold models).
- **B2 is the better test, not an independent one.** B1 saturates, but his B2 spans keep
  0.969 of the draft's present time (35,995 of 37,141 ms) and 0.818 of his present time is
  the draft's (35,995 of 44,020 ms; `span_agreement_with_elm_jump_draft` in `per_shot`).
  The independent check is the challenge key, the one B2 annotation that is not the draft.
  On the key's own bins `elm-ours` has F1 0.908 [0.782, 0.969] on the 15 servable shots
  (433 bins, 72 present: tp 64, fp 5, fn 8), 0.928 [0.838, 0.977] on the 9 published (145
  bins) and 0.701 on the 8 screened out (499 bins; tp 27, fp 11, fn 12), which agrees with
  0.922 on his labels within the intervals. The bin rule differs: a key bin is present when
  an ELM touches it, and the key holds short hand-timed windows, not whole shots. The 23
  shots give tp 91, fp 16, fn 20 against that record's 91, 16, 22 on 24 (161202 holds the
  other 2 fn bins). Recomputed in memory from the saved traces and the key grids with
  `elm_challenge_evaluate.ours_bins` (95 % shot bootstrap as in `pooled_with_ci`); it is
  not in `evaluation.json`.
- On the servable shots `elm-ours` detects ELMing periods well (F1 0.922, false alarm
  0.003) but not isolated ELMs (next section). `elm-inst` is ahead by 0.02 in F1 and the
  intervals overlap fully; with 15 shots that is a tendency, not a ranking.
- **The unscreened B2 loss is the input screen at work**: `elm-ours` on 207788 (a single
  absent row, no ELMs) calls 140 of 150 bins present, its FS02 median is the 10^12.00 clip
  floor (a dead channel), and `serve` would not publish it. That shot is 140 of the 145
  false-alarm bins on all 23 shots (the 15 servable shots have 3). 180257 and 180264 fail the
  screen by 0.01 decade, so its edge, not dead data, decides them.
- **`elm_jump` has lower crowd-bin recall on B2 (0.860, 20 shots) than on B1 (0.950)**,
  consistent with his adding about 19 % more present time than the draft held there (16 % in
  B1): the added time is where his edits show.
- **ELM-O is lower than `elm-ours` and `elm-inst` on B1's 30 BES shots** (0.862 against 0.963
  and 0.976). B2's three BES shots (192855, 194479, 204116) are not a comparison: all 33 of
  their crowd bins are in 204116, the other two hold absent bins only, and 194479 fails the
  input screen. On the old set the three are within noise (bes73: 0.841, 0.843, 0.847).

## Results: his individual rows, inside and outside his crowd spans

The share of his individual rows a method's present runs touch (`heldout.row_presence_set`;
rows inside analysed time, split by the row's centre in or out of one of his crowd rows).

| set | method | shots | inside crowd spans (rows) | outside crowd spans [95 % CI] (rows hit of rows) |
|---|---|---|---|---|
| B1 servable | elm-ours | 50 | 0.949 (11,015) | 0.028 [0.001, 0.088] (52 of 1,826) |
| B1 servable | elm-inst | 50 | 0.920 (11,015) | 0.033 [0.004, 0.096] (61 of 1,826) |
| B1 servable | elm-jump | 50 | 0.984 (11,015) | 0.157 [0.049, 0.318] (287 of 1,826) |
| B1 servable | elm-clock-peaks | 50 | 0.769 (11,015) | 0.359 [0.199, 0.509] (656 of 1,826) |
| B1 published | elm-ours | 38 | 0.950 (10,998) | 0.080 [0.002, 0.254] (45 of 560) |
| B1 published | elm-inst | 38 | 0.921 (10,998) | 0.102 [0.010, 0.300] (57 of 560) |
| B2 servable | elm-ours | 15 | 0.815 (2,060) | 0.107 [0.000, 0.291] (18 of 168) |
| B2 servable | elm-inst | 15 | 0.891 (2,060) | 0.179 [0.020, 0.444] (30 of 168) |
| B2 servable | elm-jump | 13 | 0.950 (2,060) | 0.662 [0.349, 0.877] (100 of 151) |
| B2 published | elm-ours | 9 | 0.873 (1,903) | 0.103 [0.000, 0.450] (6 of 58) |
| B2 published | elm-inst | 9 | 0.928 (1,903) | 0.224 [0.026, 0.724] (13 of 58) |

- **Both models touch almost none of the rows outside his crowd spans**, on the servable
  shots and on those `serve` would publish, in B1 and B2. The occupancy tables do not show
  it: a bin must lie wholly inside one crowd or absent row, so these rows give no bin.
- **Eleven servable shots have no present label but individual rows** (B1: 192277, 196101,
  200012, 202098, 203174, 203349, 203563, 204687; B2: 192048, 192855, 203788). `elm-ours`
  touches 0 of their 1,191 rows (1,114 in B1, 77 in B2), `elm-inst` 1. He drew no crowd row
  on any of them, so they add absent bins only to the occupancy score. On 192855 his 17
  rows are his own additions (`edited`) and neither model touches one.
- **Not known**: whether the models miss these ELMs or the rows are prefill nobody checked.
  All 1,826 of B1's outside rows are the accepted `elm_jump` prefill, and the draft's own
  table has present time at only 0.157 of them. Read isolated ELMs as not detected, not
  as absent.
- `elm-clock-peaks` touches more of them because it fires at every filterscope peak, absent
  time included (false alarm 0.36 in B1).

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
| B1 BES | ELM-O | 30 | 0.930 | 0.351 | 0.510 [0.406, 0.586] | 60 |
| B2 | elm-ours (onset head) | 23 | 0.750 | 0.019 | 0.037 [0.015, 0.068] | 18 |
| B2 | elm-inst | 23 | 0.995 | 0.357 | 0.526 [0.271, 0.725] | 0 |
| B2 | elm-jump | 20 | 1.000 | 0.999 | 1.000 [0.999, 1.000] | 0 |

- **`elm_jump` is circular** here: the rows are its own onsets, accepted. Unreduced to his
  cells it fires about 2.3 times per cell and scores precision 0.434 (B1).
- **`elm-ours`'s onset head scores near 0 recall**: it was trained on the starts of isolated
  (non-crowd) reviewed spans and his rows sit mostly in dense trains.
- **`elm-inst` finds about a third of the prefilled cells** (precision 0.97 to 0.99) with
  almost no detections in absent time. His 19 edited B2 rows are found 0 of 19 by `elm-ours`
  and `elm-inst` and 18 of 19 by `elm-clock-peaks` (it fires at every filterscope peak: precision 0.64 in B1,
  0.44 in B2, thousands of detections in absent time): too few rows, in 3 shots, to rank.

## Caveats

- **Run-day overlap.** 21 of his 58 B1 shots share a run day with the 119 reviewed shots (the
  fold models' training shots), none with Smith's 211; no B2 shot shares one (180445, 192855
  and 207788 have no run day on record). The run date is the operator log's, else the
  cohort's `run_id`. No effect shows in B1.
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

`--published` (default `outputs/labeler/elm` of the checkout) names the records the job reads
(`inst/evaluation.json`, `sources/evaluation.json`); a checkout without them passes the main
checkout's. `--skip-old` and `--shots` make a pilot. Repeated runs gave identical panels
and per-shot records.

## Suggested README lines (not applied)

`docs/labeler/events/edge_localized_mode/README.md`'s Models block is generated by
`scripts/labeler/elm_protocol.py`, which does not read this evaluation, and was not edited.
The F1 on the 9 published shots (0.947) is left out of the lines: `elm-clock-peaks` scores
0.949 on them. For the owner to place:

- elm-ours | 2026_10_24 | AUROC: 0.993 [0.982, 0.999] | AUPRC: 0.991 [0.969, 0.998] | F1: 0.922 [0.826, 0.982]
  (frozen serving ensemble at 0.417 on the 15 input-screened (servable) of the 23 ELM
  challenge-key shots, which none of the scored models trained on, reviewed by Ricardo
  Shousha 2026-10-23, 1,532 bins; false alarm 0.003; his B2 spans are mostly the elm_jump
  draft, and on the challenge key's own bins the 15 score F1 0.908 [0.782, 0.969]; on his
  50 servable other shots F1 0.970, near ceiling because his rows are largely the accepted
  elm_jump prefill. An ELMing-period detector: its present runs touch 0.107 of his
  individual rows outside his crowd spans, 0.815 inside; elm_heldout.md)
- elm-inst | 2026_10_24 | AUROC: 0.996 [0.989, 0.999] | AUPRC: 0.994 [0.979, 0.999] | F1: 0.946 [0.889, 0.986]
  (the same 15 key shots; 0.179 of the individual rows outside his crowd spans; his 50 other
  servable shots F1 0.957)
- the README's sentence "A frozen-model score on the blind split awaits review of those
  shots" would read: "A frozen-model score on 65 input-screened shots that none of the
  scored models trained on (not the blind split) is in elm_heldout.md; his labels are
  largely the accepted elm_jump prefill, and both models miss the isolated ELMs outside his
  crowd spans."
