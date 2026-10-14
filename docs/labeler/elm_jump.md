# ELM review source: `elm_jump` v2

Status: **v2 run 2026-10-14; the ELM review source since 2026-10-14 (v1 earlier that day;
`elm_clock` v1 from 2026-09-27); code not yet committed.** Code:
`src/labeler/events/elm_jump.py` (`python -m labeler.events.elm_jump`); scores:
`python -m labeler.elm.sources` (`outputs/labeler/elm/sources/evaluation.json`).

## The rule

`elm_jump` turns the review page's own "ELM onset" markers into ELMy spans. The run rule
and the gate were chosen among five jump-and-hold variants scored on these 119 reviewed
shots before v1 was built (10 ms frames, F1 .733 to .805: a 200 ms gap ungated, the
D-alpha L-H gate, 50 ms / 5 onsets with the confine-ours gate, 100 ms / 5 with it, 50 ms /
5 ungated; investigation map of 2026-10-13), so its scores below are in-sample for that
choice.

- **Onsets**: jump-and-hold on D-alpha summed over the filterscope chords that see the
  outer strike point, upper or lower divertor, chosen from the EFIT01 strike points
  (`panels.edge_localized_mode`: rise over the next 0.25 ms over the shot's flat-top
  p99.5 rise, threshold 0.12, onsets 1 ms apart merged, inside the Ip flat-top; after
  D. Eldon's OMFIT ELM module). Where the strike points cannot be read, the one
  filterscope the ELM spans read (FS01 first) is used (side "fallback").
- **Present**: each run of at least 5 onsets no more than 50 ms apart, padded by 5 ms on
  both sides.
- **Gate**: less the time the shot's confinement layers call L-mode (2) or QH-mode (3).
  The layers are painted on the 10 ms grid as the silver export paints them (the first
  layer that assessed a frame wins: the owner's saves, folded saves, format tables,
  extend tables, the review source, other suggestions). confine-ours'
  `roster_review.csv` and `gold_shots.csv` are extend tables there; its `roster.csv`,
  which the silver page cannot read, comes last. Nothing the D-alpha L-H detector
  (`dalpha_lh`) drew is read: no layer named for it, and not the rows the confinement
  review source (`regimes` v1) took from it, which are its rows on every shot outside
  the curated Gill and Butt table (`dalpha_lh_shots`; 438 of its 839 shots). The
  clock's gate is that detector, and it reads the same lower-divertor filterscope.
- **Not observable**: where the D-alpha trace has no finite sample or one value repeats
  for 200 ms or more, and the whole window of a shot whose D-alpha cannot be read.
- **Absent**: the rest of the window (the catalog window, else the shot's Ip window).

Each onset's size, the largest normalised jump in the 1 ms after it, is in the onsets
table beside the suggestion table (`edge_localized_mode_suggest_elm_jump_v2_onsets.csv`:
shot, t_ms, jump, in_span), written after the table has been validated and written. It is on the shot's own scale and is not a class. The meta
records per shot the side, the chords, the onset count, the gate layers that decided a
frame and the present time the gate removed.

## The table

`$LABELER_ROOT/suggestions/elm_jump/v2/edge_localized_mode_suggest_elm_jump_v2.csv`
covers the ELM roster, the silver list and the gold list less the blind cohort: 630
shots, 627 in the table (3,821 rows).

- 626 shots drafted: 385 read the lower divertor, 219 the upper, 22 the fallback
  filterscope (21 because no strike point was found while Ip was high, 165703 because its
  strike points are not cached). Of the 500 silver and gold shots, 499 are drafted
  (303 lower, 179 upper, 17 fallback); 178872 has no filterscope record (fdp has no
  FS02 for it) and its window is not observable. 3 roster shots (180445, 192855,
  207788, ELM challenge shots) have no window and are not in the table.
- 316,640 onsets, 272,470 of them inside present spans; 538 shots have a present span.
  Present 1,313.5 s, absent 2,138.5 s, not observable 19.2 s.
- Gate: silver confinement layers decided frames on 509 shots, confine-ours' tables
  alone on 23; 94 shots have no confinement layer (69, and the 25 shots new in v2).
  The gate removed 184.5 s of present time on 235 shots.

**v1 and v2.** Same run rule and onsets; nothing was tuned. v1
(`suggestions/elm_jump/v1`, kept) marked 26 silver shots not observable because their
filterscopes were not yet in the raw cache; 25 of them were fetched and their review
stores built on 2026-10-14, and v2 drafts them (11 upper, 8 lower, 6 fallback).
v1's gate also read the regimes rows the D-alpha L-H detector supplied; v2 leaves them
out. That changes the rows of 10 shots, which lose 1.8 s of present time in all, and
moves the in_span flag of 214 of their onsets; the onsets themselves are identical on
every shot v1 drafted. Of the 119 scored reviewed shots, 116 have identical rows; on
186427, 195016 and 198806 the present time drops by 10, 49 and 69 ms. On these shots
every F1 and interval below moves by at most 0.001 from v1's, and the absent-span touch
rate by 0.006 (v1: 51.3 %, 31.7 % trimmed); the pointer rule's outcome is the same.

## Scores on the reviewed shots

The benchmark's 50 ms bins and sets (`labeler.elm.compare`); a span method calls a bin
that a present span touches; elm-ours is its `cv2` out-of-fold trace at each fold's
threshold. 95 % percentile intervals over 1000 shot resamples, shared by every method
on a scope. Reference: the live review labels (10 ms grid). The table's 120th reviewed
shot, 189093, has no fetched inputs and is left out, as the benchmark leaves it.

| scope | elm_jump v2 F1 | elm-clock v1 F1 | elm-ours F1 | elm_jump − clock |
|---|---|---|---|---|
| all119 (12,481 bins) | 0.823 [0.768, 0.869] | 0.755 [0.679, 0.824] | 0.828 [0.776, 0.869] | +0.068 [−0.016, +0.151] |
| bes73 (6,881 bins) | 0.807 [0.732, 0.862] | 0.705 [0.600, 0.798] | 0.843 [0.784, 0.894] | +0.102 [−0.000, +0.218] |
| lower divertor, 75 shots | 0.831 [0.761, 0.889] | 0.696 [0.580, 0.796] | 0.793 [0.714, 0.851] | +0.135 [+0.013, +0.250] |
| upper divertor, 44 shots | 0.811 [0.718, 0.887] | 0.825 [0.723, 0.897] | 0.877 [0.807, 0.931] | −0.014 [−0.099, +0.083] |

- On all119 elm_jump has precision 0.768 [0.687, 0.840] and recall 0.886 [0.836, 0.934];
  the clock 0.862 [0.785, 0.926] and 0.671 [0.573, 0.776]. elm_jump's false-alarm bin
  rate is 0.161 against the clock's 0.064, and it touches 50.7 % of absent spans (31.4 %
  with 25 ms trimmed from each edge) against the clock's 29.0 % (9.7 %).
- The gain is on lower-divertor shots, where the clock calls 55.9 % of positive bins
  (precision 0.920): its L-mode gate, the D-alpha L-H detector on the same lower
  filterscope, removes ELMy time there. On upper-divertor shots the two agree within
  their intervals.
- elm_jump and elm-ours differ by −0.005 [−0.050, +0.043] in F1 on all119.
- `elm_clock` v2 calls the same bins as v1 on these shots (F1 0.755; only one absent-span
  touch differs).
- Against the committed review labels (before the 10 ms regrid) the clock reproduces its
  published F1 0.757 [0.680, 0.827] and elm-ours 0.830 [0.778, 0.871]; elm_jump scores
  0.824 [0.768, 0.870] (v1 and v2 alike), and elm_jump minus the clock is +0.067
  [−0.017, +0.149].
- The review was seeded by the clock's spans, so the reviewed edges favour the clock.
- The run rule and the gate were picked from variants scored on these same shots (see
  "The rule"), so elm_jump's scores here are in-sample for that choice and favour
  elm_jump; elm-ours' are out-of-fold.

**Decision.** The rule for this round was to move the ELM review source to elm_jump v1 if
its all119 F1 beats the clock v1's. It does (0.823 against 0.755), and
`data/events/edge_localized_mode/review/source.json` pointed to v1 from 2026-10-14 04:17
UTC and points to v2 since 08:11 UTC (`labels.write_pointer`; v2 scores 0.823 too). The paired interval of the difference includes zero, so the
F1 gain is not significant on 119 shots; the recall gain (+0.215 [+0.095, +0.331]) is.
Both are in-sample for the choice of the run rule and the gate.

## On the ELM challenge key

The same onsets, each submitted as a 1 ms event, were scored once on the DIII-D ELM
detection challenge key (held out, no tuning): merit 5,391 on all 24 shots against 2,863
for no detection and 26,659 for the clock, with the highest recall of our methods
(0.817 on the 12 busy windows, precision 0.725). See `elm_challenge.md`.
