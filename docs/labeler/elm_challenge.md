# ELM benchmark: the DIII-D ELM detection challenge key

Status: **run 2026-10-08 (key commit b71f991; elm-inst added the same day; the review page's
jump-and-hold onsets added 2026-10-13, every earlier number reproduced exactly; scripts not
yet committed).** David Eldon's
[`elm_detection_challenge`](https://github.com/DIII-D/elm_detection_challenge) holds a
hand-timed, hand-classified ELM answer key and a merit function. The maintainers told the
owner the repository may be used freely. We use it twice, without copying their code:

- **as a label source**: the key is vendored unchanged in
  `data/events/edge_localized_mode/raw/elm_challenge_key_b71f991.json` and formatted as
  Dataset 4 of the ELM README (`edge_localized_mode_challenge_format_2026_v1`);
- **as a held-out benchmark**: its 24 DIII-D shots are `holdout` rows of the ELM roster
  (`shots.csv`, tier `unverified`). None is one of the 119 reviewed shots, no method is
  trained or tuned on them (the key is public), and only 163518 is also in Dataset 1.
  The challenge's merit function runs unchanged from a pinned clone outside the repository.

The numbers below are **scored with the challenge's merit function at commit b71f991**.
They are not official leaderboard results.

## The key and the scoring

- 24 DIII-D windows (46.97 s in all, 354 events): ELMs (types 1, 2, 6), "other" transients
  (3, 4), pellets (5, 7). Six windows hold no event. Each event edge has an uncertainty.
  The key's 4 NSTX cases are skipped. Shot 169908 lists one event twice (keys 5 and 11).
- **Merit function** (lower is better, not normalised; 0 is a perfect score): milliseconds
  of mismatched event flag (weighted down within an edge's uncertainty), isolated false
  positives, over-counting, false negatives and event-type confusion. A case with no
  submission costs 20,000, so sums here are over graded cases only. A detection that
  gives no type pays the penalty of an unclassified event; none of our methods classifies,
  so each submits the challenge's detection-only type (5), as its samples do.
- **Our boundary matching** (`scoring/events.py`, one to one, nearest first) of detected
  onsets to the key's ELM onsets (types 1, 2, 6) inside each window. A detection that
  matches no ELM but starts within the tolerance of an uncertain key event (types 3, 4, 8)
  is left out, as `scoring.events` leaves out detections beside the other reader's
  uncertain spans.
- **50 ms occupancy bins**, as for the review labels: 1 where an ELM touches the bin, 0
  where only absent segments do; bins touched only by uncertain events, and bins outside a
  window, are not scored. This is the one view `elm-ours` (a 1 ms occupancy model, not an
  event detector) can be scored on.

## Methods

| method | what it submits | inputs on the challenge's `signal_list.yml` |
|---|---|---|
| no-detection | no event on any shot | none |
| always-present (bins only) | every bin present | none |
| elm-clock | each D-alpha peak of the clock as the span of its half-prominence width | yes: FS01 on all 24 shots |
| jump-and-hold | the review page's "ELM onset" markers (`panels.edge_localized_mode.markers`: rise over the next 0.25 ms over its p99.5 in the Ip flat-top, threshold 0.12, on D-alpha summed over the strike-point chords; D. Eldon's OMFIT method, reimplemented), each onset a fixed 1 ms event; constants as on the page, none changed for the key | partly: the chords are on the list on 20 shots (FS07 or FS08 on 154749, 161558, 194479, 207173); the chord choice reads RVSIN, which is not; 4 shots with no cached strike points read FS01 |
| Eldon's samples | his seven sample submissions, read in place from the clone | his own; `filterscopes_used` per file |
| elm-ours (bins only) | mean of the five frozen `cv2` fold models, at the mean of the fold thresholds (0.417) | partly: FS02-FS04 yes; the fast chords DENV2F/DENV3F no (the list has PCDENV2/PCDENV3) |
| elm-ours-onset-review | peaks of the mean onset trace of the same five `cv2` models (trained on the reviewed non-crowd span starts), at the mean of their onset thresholds (0.35); each peak a fixed 1 ms event | as elm-ours |
| elm-ours-onset-smith | peaks of the mean onset trace of the five Smith-only fold models (`round4/elm/smith/cv`), at the mean of their thresholds (0.81); each peak a fixed 1 ms event | as elm-ours |
| elm-inst | peaks (2 ms apart) of the mean 0.1 ms instance trace of the five `inst1` fold models (`elm_inst.md`), at the mean of their thresholds (0.51), each an event of the mean fold event duration (1.10 ms); not gated by occupancy, as chosen on development data | as elm-ours |
| elm-inst-alt | the same, kept only where the mean occupancy's 50 ms centred mean reaches the mean occupancy threshold (0.35) | as elm-ours |
| elm-elmo | not run: needs BES, which is not on the list | no |

The two onset heads are the baseline for the instance side of a later instance-plus-crowd
model (as in instance segmentation's `is_crowd`); none of the key's shots is in any fold's
test list or among the reviewed ones. Peaks are local maxima of the averaged trace at least
10 ms apart (`onset.peaks`), kept inside the shot's valid input and the key's window. The
heads have no end output, so every peak is submitted as a 1 ms event, a choice made before
scoring: their timing-mismatch terms are therefore not a measure of event width. Neither
head treats crowds specially yet.

The clock reads FS01 on every shot (`spans.dalpha_channel`). Nineteen shots' filterscopes
come from the raw cache fetched on 2026-10-08 at the diagnostic's native 50 kHz and are
resampled to the corpus's 10 kHz grid by linear interpolation before the clock reads them
(`challenge.resample`); the five corpus shots (192048, 192855, 194479, 203788, 204116) are
read as stored. The clock's rules were not changed for the key. Its point peaks become
spans of their half-prominence width because the clock has no other extent.

## Result

Windows are split by what the key holds (12 with at least 10 events, 12 with fewer), not by
any method's result.

**Merit function** (sum over the graded cases; the cases' counts of ELMs and detections):

| cases | no-detection | elm-clock | jump-and-hold | onset-review | onset-smith | elm-inst | Eldon's samples |
|---|---|---|---|---|---|---|---|
| all 24 (354 events) | 2,863 | 26,659 (2,531 detections) | 5,391 (584) | 3,220 (35) | 5,315 (441) | **2,460** (96) | |
| 12 windows with 10 or more events (342 events) | 2,747 | **2,120** (207 detections) | 3,088 (365) | 2,989 (24) | 2,449 (166) | 2,346 (95) | |
| 12 windows with fewer (12 events) | 116 | 24,540 (2,324 detections) | 2,303 (219) | 231 (11) | 2,866 (275) | **115** (1) | |
| his 7 sample shots (149 events) | 1,355 | 2,013 | 1,630 | 1,554 | 1,856 | **1,172** | 17,911 |
| the same, without 166828 | 1,352 | 1,039 | 1,628 | 1,520 | 1,782 | 1,170 | **905** |

elm-inst-alt (gated) scores within one point of elm-inst in every row (2,461 on all 24).

- **On ELMy windows the clock beats no detection** (2,120 against 2,747): 139 events
  missed against 342, timing mismatch 462 ms against 662. 203 of its detections count as
  unclassified.
- **On windows with few events the clock fires on every D-alpha bump:** 2,324 detections
  against 12 events and 2,307 isolated false positives, which is 92 % of its total score. The
  cause (noise above 3 % of the trace's range, a quiet baseline, or real small transients
  the key does not mark) was not examined. This is the clock's known limit, not new: its
  peak rule has no amplitude floor beyond the trace's own range.
- **Eldon's sample for 166828 reports 1,071 events in a window with one** (17,007 of his
  17,911); the challenge's README says some samples are poor. His other six, 905 against
  1,039 for the clock, are the fairer comparison, and the clock and his detector are close.
  This says nothing about his best detector.
- **The Smith onset head beats no detection on the ELMy windows** (2,449 against 2,747;
  191 events missed against 342) but not the clock (2,120), and it loses on the windows with
  few events (2,866 against 116): 266 of its 275 detections there are isolated false
  positives, and 187 detections sit in two windows (180445, 123; 163518, 64). 180445 is also where the clock fires 1,824
  times and the key marks nothing; why was not examined.
- **Jump-and-hold, the review page's own onset detector, finds the most ELMs and still
  scores worse than no detection** (5,391 against 2,863 on all 24 windows; 3,088 against
  2,747 on the busy ones). On the busy windows the merit function counts 66 of 342 events
  missed (the clock 139), but each of its 365 detections pays the unclassified-type cost
  (272 of its busy-window total) and, being 1 ms long, a timing mismatch (548 ms against
  the clock's 462). In the quiet windows it makes 219 detections against 12 events (204
  isolated false positives); four shots make 193 of them (192048, 194479, 204116 and
  203788, 44 to 55 detections each against 0 to 3 key events).
  It was scored once, as the page runs it; no constant was changed.
- **The review-trained `cv2` onset head is close to no detection** (3,220 against 2,863): it
  fires 35 times in 24 windows and misses 353 of the 354 events by the merit function. It
  scored F1 0.09 to 0.22 on its own folds' reviewed shots at 5 ms (`onset_val_f1` in the fold
  files), so this is not a surprise. Where it does fire next to a key ELM it fires early:
  for the 26 key ELMs with one of its detections within 20 ms the median signed offset is
  -4.8 ms. Why it is early (a different onset convention in the reviewed labels, or the
  model) was not examined.
- **elm-inst is the only method below no detection on all 24 windows** (2,460 against
  2,863). It makes 96 detections in 24 windows and almost none is false (2 against the key's
  ELMs; one detection, a real ELM, in the 12 windows with few events, 115 against 116). It
  misses most ELMs (246 of 342 by the merit function in the busy windows), so on those the
  clock stays ahead (2,120 against 2,346). Its false-positive term there is 5, as the
  clock's (Smith head 23).

**Our boundary matching** (onset tolerance 2 ms; 1 and 5 ms are in `evaluation.json`: on
the 12 busy windows precision 0.935 and 0.970, recall 0.614 and 0.641):

| cases | method | precision | recall | matched ELMs / false / missed | median offset |
|---|---|---|---|---|---|
| 12 busy windows | elm-clock | 0.965 | 0.634 | 194 / 7 / 112 | 0.15 ms |
| 12 other windows | elm-clock | 0.002 | 1.0 | 4 / 2,308 / 0 | 0.12 ms |
| 7 sample shots | elm-clock | 0.479 | 0.710 | 93 / 101 / 38 | 0.36 ms |
| 12 busy windows | jump-and-hold | 0.725 | 0.817 | 250 / 95 / 56 | 0.05 ms |
| 12 other windows | jump-and-hold | 0.019 | 1.0 | 4 / 209 / 0 | 0.21 ms |
| 7 sample shots | jump-and-hold | 0.644 | 0.786 | 103 / 57 / 28 | 0.06 ms |
| 7 sample shots | Eldon's samples | 0.058 | 0.504 | 66 / 1,075 / 65 | 0.03 ms |
| 12 busy windows | onset-smith | 0.876 | 0.461 | 141 / 20 / 165 | 0.26 ms |
| 12 other windows | onset-smith | 0.015 | 1.0 | 4 / 266 / 0 | 0.44 ms |
| 7 sample shots | onset-smith | 0.429 | 0.412 | 54 / 72 / 77 | 0.25 ms |
| 12 busy windows | onset-review (5 ms) | 0.571 | 0.039 | 12 / 9 / 294 | 4.7 ms |
| 12 busy windows | elm-inst | 0.979 | 0.304 | 93 / 2 / 213 | 0.06 ms |
| 12 other windows | elm-inst | 1.0 | 0.25 | 1 / 0 / 3 | 0.01 ms |
| 7 sample shots | elm-inst | 1.0 | 0.244 | 32 / 0 / 99 | 0.07 ms |

elm-inst's matching is the same at 1, 2 and 5 ms: every detection that matches is within
1 ms.

The `cv2` onset head matches nothing within 1 or 2 ms (0 of 310 key ELMs, 35 detections);
in the table, 5 ms is the first tolerance where it matches anything (13 of 310 over all 24
windows).

**Individual against crowded ELMs.** 194 of the key's 310 ELM onsets (63 %) have another key
ELM onset within 10 ms, the onset heads' minimum peak separation. Of the key ELMs each
method finds (nearest detection within 2 ms, not one to one), by the distance to the nearest
other key ELM (`evaluation.json`, `onset_by_crowding`):

| nearest other ELM | key ELMs | onset-smith | elm-clock | elm-inst | jump-and-hold |
|---|---|---|---|---|---|
| under 10 ms | 194 | 53 | 111 | 22 | 160 |
| 10 to 20 ms | 51 | 45 | 49 | 32 | 50 |
| 20 to 50 ms | 37 | 26 | 33 | 23 | 35 |
| 50 ms or more, or alone | 28 | 26 | 24 | 19 | 26 |

The onset-smith head finds most ELMs that stand apart and little of the trains. Two peaks
cannot be closer than 10 ms in its output, which alone limits what it can find there;
whether a smaller separation would recover them was not tried. This is the group-detection
gap behind the baseline: the merit function scores each ELM of a train, which a head built
for separated onsets does not give. How many of the 194 our reviewers would call a crowd was
not checked. elm-inst allows peaks 2 ms apart and still finds the fewest ELMs in trains (22
of 194): separation was not the only limit. Its only instance targets inside crowds are the
clock's onsets at half weight (`elm_inst.md`). Jump-and-hold, which merges onsets only
within 1 ms, finds 160 of the 194: most of the train ELMs the other methods miss, at the
cost of the quiet-window false positives above. Its matching moves by at most a few
events between 1, 2 and 5 ms (busy windows: recall 0.814, 0.817, 0.820).

**Occupancy bins.** elm-inst's occupancy output, at the mean of its fold thresholds, scores
F1 0.817 [0.642, 0.925] on the 24 shots against `elm-ours`' 0.827 [0.660, 0.926] (busy
windows 0.889 and 0.881; sample shots 0.920 and 0.939): the same within the intervals.

## Caveats

- 24 shots and 12 busy windows: intervals are wide and the split is post hoc (by key
  content, not tuned).
- The key marks only short hand-inspected windows, so absent time is what the annotator
  inspected, not whole discharges, and "no detection" scores well there (2,863 for 354
  events, about 8 per event).
- The merit function is not normalised and a single shot can dominate a sum (166828 for
  Eldon, 180445 and 203788 for the clock: 9,493 and 11,207 of its 26,659). Read per-case
  numbers (`evaluation.json`, `merit.<method>.<scope>.per_case`).
- On these windows an isolated false positive costs about as much as a missed ELM (the
  clock's extra 2,307 in the quiet windows cost 10.6 each; no detection pays 8.0 per busy
  window event). elm-inst's lead on all 24 windows comes from the quiet windows: it is a
  precision result, not a recall one.
- The onset heads' events are 1 ms by construction, so their timing-mismatch terms count
  the missing width as well as the missing events; read their false-negative and
  false-positive terms and the onset matching first.
- No method classifies event type; type confusion costs are paid by all. A classifier would
  be graded on types 1-8 and the pellet types, which the key holds in small numbers.
- The key is still growing upstream. A newer commit is a new dated dataset beside this one;
  re-run against it, never in place.
- The merit function and the clone are David Eldon's and are used as published; if the
  maintainers change the scoring, scores at another commit are not comparable.

## Reproduce

Fetches need fdp (login node, logged in): `scripts/labeler/elmo_fetch.py --shots ...` for
the fast signals `elm-ours` reads, and
`python -m labeler.events.raw --event edge_localized_mode --groups filterscopes ...` for the
clock's filterscopes (logs in `$LABELER_ROOT/runs/elm_challenge_fetch_2026-10-08/`).
Scoring needs a clone of `elm_detection_challenge` pinned at b71f991 and a venv with numpy,
scipy, pandas and `uncertainties` (no labeler), both outside the repo:

    pixi run --frozen --no-install -e labeler python scripts/labeler/elm_challenge_evaluate.py

The onset heads need the frozen `round4/elm/cv/cv2` and `round4/elm/smith/cv` folds under
`$LABELER_ROOT`; elm-inst needs `round4/elm/inst/inst1/key/` (from `elm_inst.py predict`)
and its development record `outputs/labeler/elm/inst/evaluation.json`. It writes the submissions to `$LABELER_ROOT/benchmarks/elm/challenge/submissions/` (never
the clone's `submission/`), grades them with `scripts/labeler/elm_challenge_merit.py`
(refuses a key that differs from the clone's, a dirty clone, or an empty folder, which the
challenge would replace with its samples) and records everything in
`outputs/labeler/elm/challenge/evaluation.json`, with the key's and the scripts' sha256.
