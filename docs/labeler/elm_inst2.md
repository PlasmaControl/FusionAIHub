# elm-inst2: ELM trains as separate spikes, from reviewed onsets

Status: **trained and scored on the challenge key 2026-10-24 (run `round4/elm/inst/inst2`;
scripts not yet committed). Round 1 (type 5): neither decoding beats `elm-inst`. Round 2
(type-2 threshold and size typing, below): `elm-inst2-sized` scores 1,635.7, the best so
far.**
`elm-inst` (`elm_inst.md`) with two changes the owner asked for on 2026-10-24: train it on
Ricardo Shousha's ELM review of 2026-10-23, and give it a loss that rewards spikes (the
old one was "too smooth": a train came out as a low plateau).

## Why

On the DIII-D challenge key `elm-inst` is the only method below no detection (merit 2,460
against 2,863 on all 24 windows; `elm_challenge.md`), but it is a precision result: P 0.979,
R 0.303, and 22 of the 194 key ELMs that have another within 10 ms. Inside crowds its only
instance targets were the clock's onsets at half weight, and its focal loss rewards a high
heatmap at an onset cell without asking it to fall before the next ELM.

## Design (`src/labeler/elm/spikes.py`, `scripts/labeler/elm_inst2.py`)

- **Network and the two old sources unchanged**: `inst.InstNet`; the 119 reviewed shots and
  Smith's 2,316 windows, split and targeted as for `elm-inst`.
- **His shots** (pin `2026-10-23`): the 57 of his 81 that are on no hold-out list and have
  an input (the 23 challenge-key shots and 203401, which has no `DENV2F`, are left out).
  - Onsets: the `elm_jump` v2 onsets inside his individual 10 ms rows (all 13,776 rows on
    these shots are the accepted prefill), merged so no kept onset follows another by
    less than 3 ms: 33,664 onsets in his rows, 22,607 kept. Half the gaps between the
    table's onsets are under 3 ms (a second jump on one ELM's rise; the histogram peaks
    at 1.0-1.5 ms), against 2.8 % of the clock's crowd intervals. Peaks are decoded 3 ms
    apart to match.
  - Targets: each onset a full-weight Gaussian peak known within 1.5 ms, as Smith's; his
    absent time known empty 5 ms in from its edges, as the review's; the rest of his
    crowd and row time ignored by the focal loss. Occupancy from his spans and rows, his
    absent rows under 50 ms (the gaps between rows) ignored.
  - Folds: dealt by run day into five groups (11, 12, 11, 12, 11 shots); fold k tests on
    group k and validates on group k + 1.
- **Spike loss** (`spikes.spike_loss`, weight 1, beside the focal loss): at each of
  Smith's onsets and his, the heatmap's maximum within 0.5 ms should be 1 (a peak a few
  cells off still counts), and its minimum on each side, 0.8 to 2.5 ms away, should be 0.
  A train must come out as a row of peaks, not a plateau.
- **Focal loss balance**: his crops hold hundreds of onsets each and the focal loss is
  normalised by the onsets in the batch, so it is taken over his crops (a third of each
  batch) and over the rest apart; the reviewed absent time keeps the weight it had.
- **Fold thresholds**: each fold records its inner-validation maximum of what the
  detections would save on the challenge merit function's published costs
  (`spikes.merit_gain`). A large ELM found with the detection-only type saves
  10 x (1.20 - 0.70) = 5.0 plus the event's length. A false detection costs
  10 x 0.95 = 9.5 plus its length. Break-even precision is about 0.63. The F1 maximum
  (what `elm-inst` served) is recorded beside it. On his shots a detection is judged
  false only in guarded absent time or within 3 ms of a kept onset it does not match (a
  second detection on one ELM); the others are unjudged (`spikes.judge`).
- **Served threshold** (fixed by `evaluate` before the key is read): the grid threshold
  where the out-of-fold counts of all three sources, pooled, save the most by a
  conservative rule (`rule_gains`). That rule changes the fold rule in two places. An
  ELM of his found inside a crowd is priced as small (type 2: it saves only
  10 x (0.40 - 0.35) = 0.5 plus its length), because crowd ELMs are small: they rise
  0.11 to 0.34 decades, against 0.94 for Smith's isolated ones. And his detections in a
  present 10 ms cell where he kept no onset (`empty`) count as false. `elm-inst2-alt`
  is the fold rule's mean threshold. Both are declared in the development record, and
  both are scored once (as `elm-inst` and `elm-inst-alt` were); `elm-inst2` is the
  primary.
- **Served on the key**: the five fold models' mean heatmap at the served threshold,
  ungated, peaks 3 ms apart, each event the training shots' median clock event length,
  as `elm-inst`. `key` first rebuilds `elm-inst`'s graded files from its key traces
  through the same decoding and checks that they match the graded files and their
  merit (2,460.3, no detection 2,862.7). Only then does it take a lock under the data
  root (`round4/elm/inst/inst2/key.lock`), which stops any checkout from scoring a
  second time. The key traces and checkpoints are hashed in the development record and
  checked before decoding.

## Development results (out of fold; `outputs/labeler/elm/inst2/evaluation.json`)

Training was job 2977040: an A100, five folds at once, 10 min 37 s, GPU 65.4 % over the
training loop. Its jobstats gate **failed**: CPU 57.7 % and GPU 62.3 %, against jobstats'
70 % bars. It passes the project's GPU rule (at least 50 %); a rerun should ask for 6 CPUs,
not 8. The development record is job 2977055.

| Decoding | His onsets: recall (95 % CI) | His judged precision | Smith windows: F1 (recall, precision) | Alarms per 10 s of reviewed absent time |
|---|---|---|---|---|
| elm-inst2, fold thresholds (0.40-0.75) | 0.575 (0.47-0.66) | 0.907 | 0.966 (0.991, 0.942) | 75.9 |
| elm-inst2-alt, 0.53 | 0.556 (0.45-0.66) | 0.906 | 0.971 (0.990, 0.953) | 72.0 |
| elm-inst2 served, 0.90 | 0.131 (0.07-0.21) | 0.939 | 0.740 (0.588, 0.997) | 4.4 |
| elm-inst, fold thresholds (his shots: its five-model mean at 0.51) | 0.261 (0.16-0.37) | 0.932 | 0.891 (0.807, 0.994) | 19.9 |

Occupancy on the reviewed bins is unchanged: elm-inst2 has AUROC 0.938, AUPRC 0.909 and
F1 0.838; elm-inst has 0.946, 0.911 and 0.831.

The pooled trade-off (`tradeoff_pooled_out_of_fold`) favours elm-inst2 under one rule
and elm-inst under the other:

- **Fold rule:** elm-inst2's best is at 0.60 and saves 46,198. elm-inst's best is at 0.35
  and saves 41,931.
- **Conservative rule:** elm-inst2's best is at 0.90 and saves 7,982. elm-inst's best is
  at 0.65 and saves 10,201. Priced this way, the crowd ELMs elm-inst2 adds save less than
  its extra alarms cost.

At about the same number of alarms, elm-inst2 finds somewhat more of his onsets. At
0.75 it gives 985 alarms in the reviewed absent time and his recall 0.347. elm-inst at
0.50 gives 974 alarms and his recall 0.274.

## Challenge key (held out; scored with the challenge merit function at commit b71f991, once)

Job 2977056 (`outputs/labeler/elm/inst2/key.json`; a copy is at
`round4/elm/inst/inst2/key_record.json`). This is not an official leaderboard result.
Before taking the lock, the job rebuilt elm-inst's 24 graded files from its key traces.
They were identical to the graded files, and the merit function gave 2,460.328 again
(no detection: 2,862.745).

| Method | Merit, all 24 windows (lower is better) | Windows with 10+ key events | Other windows | Onsets within 2 ms: P, R, F1 | Crowded key ELMs found (of 194) |
|---|---|---|---|---|---|
| elm-inst2 (primary, 0.90) | 2,652.4 | 2,537.6 | 114.8 | 1.000, 0.148, 0.258 | 4 |
| elm-inst2-alt (0.53) | 2,515.2 | 2,312.3 | 202.9 | 0.885, 0.523, 0.657 | 78 |
| elm-inst | **2,460.3** | 2,345.5 | 114.8 | 0.979, 0.303, 0.463 | 22 |
| no detection | 2,862.7 | 2,746.9 | 115.9 | - | - |

A crowded key ELM has another within 10 ms. It counts as found when a detection starts
within 2 ms of it; this matching is not one to one.

**Neither elm-inst2 decoding beats elm-inst.**

- **The alternative (0.53)** finds ELMs inside trains, as intended: 78 of the 194 crowded
  key ELMs against 22, and onset F1 0.66 against 0.46. On the windows with 10 or more key
  events it scores below elm-inst (2,312 against 2,346), although 178555 and 169908 cost
  39 and 21 more there. It loses more than that on the windows with fewer than 10 key
  events, 88 in all, nearly all in 163518 (106 more: 11 detections where the key has 5
  events). On the merit function's costs, a found small ELM saves 0.5 and a false
  detection costs 9.5, so 21 false detections outweigh 68 more ELMs found.
- **The primary (0.90)** was too conservative for the key: 46 detections, all correct,
  against elm-inst's 96.

Another decoding scored on this key would be tuning on it. A new choice needs data of
our own that no model has seen.

Run days: key shot 192048 shares 2022-09-15 with his training shot 192045. The key has
one event in that window, and no method detects anything there. Three key shots
(180445, 192855, 207788) have no run day on record.

## Caveats

- His onsets are `elm_jump` v2 onsets a reviewer kept, not hand timing: agreement with them
  on his shots is agreement with a reviewed `elm_jump`.
- The 3 ms merge is a choice: some real ELMs follow another by less than 3 ms.
- The fold rule (`elm-inst2-alt`) prices every ELM found as large (type 1). Where ELMs
  are small it fires too often. The served rule prices his crowd ELMs as small and his
  empty-cell detections as false, but it still takes Smith's ELMs and his ELMs outside
  crowds as large. It also counts every alarm in the reviewed absent time as false,
  even where the review may have missed ELMs: 192229's 6 s absent span holds 927
  clock peaks.
- The development views score single out-of-fold models; the served five-model mean is
  never scored on development data. `elm-inst` on his shots is its five-model mean,
  which never saw them. The bootstrap intervals of two methods are not paired, so
  overlapping intervals say nothing about their difference.
- The spike loss is dominated by his anchors: his crops hold hundreds of onsets each
  against a few in Smith's.
- One key shot shares a run day with a training shot (`key.json`, `run_days`):
  192048 and his 192045, both on 2022-09-15.
- The key is scored: `round4/elm/inst/inst2/key.lock` refuses `evaluate` and `key`
  from any checkout.

## Reproduce

    export LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labeler
    export SHOT_DESIGN_DATA_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/shot_design
    export LABELER_LABEL_TABLES=/scratch/gpfs/EKOLEMEN/nc1514/events
    export REPO=$PWD    # the checkout to run; the sbatch default is the main one
    S="sbatch --output=$LABELER_ROOT/runs/slurm/%j.out --error=$LABELER_ROOT/runs/slurm/%j.out"
    STAGE=prepare $S scripts/labeler/elm_inst2_cpu.sbatch     # CPU, seconds
    $S scripts/labeler/elm_inst2.sbatch                       # GPU: five folds at once, predict
    STAGE=evaluate $S scripts/labeler/elm_inst2_cpu.sbatch    # development record
    STAGE=key $S scripts/labeler/elm_inst2_cpu.sbatch         # the key, once

## Round 2: type-2 prices and size typing (`scripts/labeler/elm_inst2_typed.py`)

Status: **scored once, job 2977241 (`outputs/labeler/elm/inst2/key_typed.json`; a copy
is at `round4/elm/inst/inst2/key2_record.json`, locked by `key2.lock`).** The owner asked
for this on 2026-10-24, after the survey in `elm_score_methods.md`. The development
record is job 2977240, which reran 2977224 after an Opus review and is identical but for
its hashes.

### Why

Round 1 reported every detection as the detection-only type 5 and chose its thresholds
at type-5 prices. At those prices the merit function says "do not report small ELMs":

| Type reported | Large ELM found | Small ELM found | False detection |
|---|---|---|---|
| 5 (round 1) | +5.0 | +0.5 | -9.5 |
| 2, small ELM | +7.0 | +4.0 | -3.0 |
| 1, large ELM | +12.0 | +1.0 | -10.0 |

These are the rows of `miscounting_penalties` at commit b71f991, times 10
(`spikes.PENALTY`, `spikes.prices`). Each detection also gains or costs about its own
length (1.1) through the timing term. `elm-inst` retyped from 5 to 2 scored 2,206.0
against 2,460.3 (graded by another session on the owner's word, 2026-10-24).

### Design

- **Thresholds:**
  - `evaluate` recomputes round 1's out-of-fold counts and stops unless they are
    identical.
  - It prices them with type-2 costs (`elm_inst2.rule_gains(..., reported=2)`), using
    the same two rules.
  - `elm-inst2-t2` is at the conservative rule's maximum, `elm-inst2-t2-alt` at the
    as-trained rule's. Both report every detection as type 2.
- **Size typing** (`elm-inst2-sized`, the primary): `elm-inst2-t2`'s detections, each
  reported as type 1 or 2, whichever costs less in expectation (`spikes.report_type`).
  - Inputs: three size measures at the onset (`sizes.features`):
    - the filterscope rise in decades over 1 ms;
    - the peak contrast against the 0.5 s running median;
    - the line-density drop over 5 ms.

    None of them asks how close the next ELM is.
  - P(large | ELM): a logistic regression of Smith's onsets found (large) against his
    onsets found inside a crowd (small). The two classes are weighted equally, so the
    key's mix of large and small is not used.
  - P(no event): a logistic regression of detections judged false against detections
    that found an onset, at their development mix. False means unmatched in Smith's
    windows, in guarded reviewed absent time, or his `fp` and `empty`.
  - Report type 1 when 7 P(none) < (1 - P(none)) (8 P(large | ELM) - 3). For a sure
    ELM that is P(large | ELM) > 3/8.
- **Key:**
  - `key` checks the record against round 1's and rebuilds `elm-inst`'s graded files as
    round 1 did.
  - It decodes and types every key detection.
  - It checks the code, the key shots' input records and the `elm-inst-as2` files
    against the hashes in the development record (`frozen`).
  - It builds every submission and checks that the merit function would read each
    event's intended type (`read_types`).
  - Only then does it take `round4/elm/inst/inst2/key2.lock` (round 1's `key.lock`
    stays), write the three folders and grade them beside `elm-inst`, `elm-inst-as2`
    and no detection.
  - The reproduction check grades `elm-inst` and no detection before the lock, as round
    1 did; nothing of round 2 is graded before it.

### Development results (out of fold; `outputs/labeler/elm/inst2/evaluation_typed.json`)

| Rule at type-2 prices | Best threshold | Saves |
|---|---|---|
| Conservative (`elm-inst2-t2`) | 0.50 | 63,241 |
| As trained (`elm-inst2-t2-alt`) | 0.40 | 108,841 |

Priced the same way, elm-inst's best is 57,422 at 0.30 (conservative) and 102,541 at
0.20 (as trained). At type-5 prices, round 1's served elm-inst2 (0.90) saved 7,982.

**Size models at 0.50** (single out-of-fold models; 2,298 large, 12,974 small, 90 of
his ELMs outside crowds, 5,436 false, 62,774 unjudged):

- **P(large | ELM):** AUROC 0.941.

  | Predicted P(large) | 0.0-0.2 | 0.2-0.4 | 0.4-0.6 | 0.6-0.8 | 0.8-1.0 |
  |---|---|---|---|---|---|
  | Mean predicted | 0.06 | 0.29 | 0.49 | 0.70 | 0.93 |
  | Large share (balanced) | 0.03 | 0.26 | 0.53 | 0.78 | 0.93 |

- **P(no event):** AUROC 0.58, nearly a base rate. It lowers P(no event) only for large
  rises, to about 0.1 at a 0.95-decade rise.
- **Typing out of fold:**
  - typed large: 2,017 of the 2,298 large ELMs, 1,600 of the 12,974 small ones, and 477
    of the 5,436 false detections;
  - against every detection typed 2, that is +1,946 at the development mix
    (5 x 2,017 - 3 x 1,600 - 7 x 477);
  - of the unjudged detections, 32 % are typed large.
- **Median measures:**

  | Detections | Rise (decades) | Contrast (decades) | Density drop |
  |---|---|---|---|
  | Large | 0.95 | 0.90 | 0.050 |
  | Small | 0.38 | 0.34 | 0.024 |
  | His, outside crowds | 0.38 | 0.33 | 0.013 |
  | False | 0.23 | 0.20 | 0.023 |

### Challenge key, round 2 (scored with the challenge merit function at commit b71f991, once)

This is not an official leaderboard result, and not a held-out one (see the caveats).
Lower is better.

| Method | Merit, all 24 | Timing | Classification | Missed and false | Detections | Onsets within 2 ms: P, R, F1 | Crowded key ELMs found (of 194) |
|---|---|---|---|---|---|---|---|
| **elm-inst2-sized** (primary) | **1,635.7** | 588.1 | 103.2 | 944.5 | 206 | 0.850, 0.565, 0.678 | 95 |
| elm-inst2-t2 (0.50, all type 2) | 1,907.7 | 588.1 | 410.2 | 909.5 | 206 | same | 95 |
| elm-inst2-t2-alt (0.40, all type 2) | 2,094.9 | 666.3 | 439.6 | 989.0 | 325 | 0.643, 0.661, 0.652 | 125 |
| elm-inst-as2 (elm-inst typed 2) | 2,206.0 | 619.8 | 371.2 | 1,215.0 | 96 | 0.979, 0.303, 0.463 | 22 |
| elm-inst (type 5) | 2,460.3 | 619.8 | 593.0 | 1,247.5 | 96 | same | 22 |
| no detection | 2,862.7 | 698.5 | 0 | 2,164.2 | 0 | - | - |

- **Step 1, the type-2 threshold (2,206.0 to 1,907.7):**
  - At 0.50, elm-inst2 makes 206 detections against elm-inst's 96.
  - Missed events fall from 257 to 170, at the cost of 30 isolated false detections and
    1 double count.
  - It finds 95 of the 194 crowded key ELMs, against 22.
- **Step 2, size typing (1,907.7 to 1,635.7):**
  - 106 detections were typed large and 100 small.
  - The classification cost fell from 410.2 to 103.2, with 34 events still typed wrong
    against 84.
  - Typing helped most in 178555 (-83), 169908 (-75), 154749 (-70) and 171322 (-65).
  - It cost 51 in 180257 and 12 in 132651, where detections typed large are small or
    false by the key.
- **The alternative (0.40)** finds more (recall 0.66), but its 113 false detections
  cost more than its 39 extra ELMs save.
- **Reproduction check:** before the lock, elm-inst's 24 graded files were rebuilt
  identically and scored 2,460.328 again (no detection 2,862.745).

### Caveats (round 2)

- **Proxy labels:** large means "like Smith's isolated ELMs" and small means "like his
  crowd ELMs". The key's own criterion is not known, and typing pays only if more than
  3/8 of the ELMs typed large are large by it.
- **Single models:** the size models are fitted on single out-of-fold models'
  detections, but the key's detections come from the five-model mean.
- **Not held out:** the key has been graded before, in round 1, as `elm-inst` retyped,
  and as the key itself fed back as a submission. Round 2's numbers are therefore not a
  held-out result.
- **The no-event model sets most of the typing.** It barely separates its classes
  (AUROC 0.58), so it acts as a fixed prior at the development mix. At P(no event) 0.2
  to 0.3, a detection needs P(large | ELM) of about 0.59 to 0.75 to be typed large. The
  five-model mean on the key has another precision.
- **Isolation may leak into the rise.** In a dense train, the rise's baseline sits on
  the last ELM's decay. Since the proxy classes are isolated (Smith) against crowded
  (his), the 0.94 AUROC partly measures isolation. Against that, his ELMs found
  outside crowds have a median P(large) of 0.08.
- **Other key types are not priced.** `report_type` prices no event, large and small
  only. The key's other, pellet and unspecified events cost type 1 and type 2
  differently (a hidden pellet: 9.5 against 2.5).
