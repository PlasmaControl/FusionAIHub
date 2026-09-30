---
title: "AE from cross-power"
---

# AE from cross-power (`ae_xpower`)

A small frame model that says, every 10 ms of a shot, whether an Alfvén
eigenmode (AE) is present. It learns from the owner's reviewed AE labels, is
chosen by a rule written down before any test number exists, is scored once on
held-out shots against a fixed bar, and then (only if it passes) labels the
population's shots as suggestions. This page follows it from its inputs to its
last record.

Code: `src/labeler/ae/xpower/` (`data`, `model`, `train`, `cv`, `evaluate`,
`gallery`, `extend`, `posthoc`). Tests: `tests/labeler/test_ae_xpower_*.py`.
Jobs: `scripts/labeler/ae_xpower_*.sbatch`. `$LABELER_ROOT` is
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker`.

There are four versions. **v1** trained three candidates, chose one on its
validation shots and was tested once (A2 failed: it mistook MHD for AE). **v2**
is the one this page describes in full (v3 is v2's recipe over the owner's
whole window at 0-250 kHz, v4 v2's at 60-250 kHz, below): the same model,
chosen by five-fold cross-validation with a rule that weighs the MHD mistake (the ledger's
Deviation 11). Every command takes `--version`; it must be the models
directory's name and the version the checkpoint records, or the command
refuses.

**v4** (2026-09-30, the owner's ask after the review page's AE band was lowered
to 60 kHz) is v2 with the band at 60-250 kHz: candidates `band60-mhd3`,
`band60-mhd10`, `band60-mhd30` (`data.BAND60_KHZ`), and everything else v2's:
v2's label snapshot (the same sha256, `LABEL_SNAPSHOTS["v4"]`), v2's split,
folds and 60 test shots, 0-2 s scoring, the choice rule and the final-training
rule. Before its folds, copy the snapshot in (there is no command for it; v2's
was copied the same way) and check its sha256:

```bash
mkdir -p $LABELER_ROOT/models/ae_xpower/v4/review
cp $LABELER_ROOT/models/ae_xpower/v2/review/labels.csv $LABELER_ROOT/models/ae_xpower/v4/review/labels.csv
sha256sum $LABELER_ROOT/models/ae_xpower/v4/review/labels.csv   # 5f52a268...
```

`cv --version v4 --folds` refuses unless its folds are v2's `cv/folds.csv`,
byte for byte, and `evaluate --test --version v4` unless its test shots are
v2's. Its test is a second use of v2's test shots, after v2's was scored, so
`training.json`, `chosen.json`, the model and `evaluation.{json,md}` carry
`test_reuse` (`xpower.reuse_note`), and the second look (`evaluate.SUBSET_OF`)
names v2's scoring; its `v2_subset` table is its whole test. v2's own records
are untouched: `evaluate --test --version v2` still refuses a second scoring.
The jobs are v2's with `VERSION=v4` (each script's header has the v4 lines).
v4's baselines stay at v2's bands: TokEye's AE call and the MHD-frame rule
(`data`'s fixed TokEye bins) are not moved to 60 kHz, so only the model's band
differs from v2. SegNet v4 (`labeler.ae.seg`: SegNet v1 at 60-250 kHz, on the
live labels, which the owner has edited on 6 of its 162 split shots since
SegNet v1) is separate.

## Inputs

- **The owner's labels, frozen.** v2 reads only its label snapshot,
  `$LABELER_ROOT/models/ae_xpower/v2/review/labels.csv`: the owner's review
  file as of 2026-09-27 22:05 EDT, 180 shots. Its sha256 (`5f52a268…`) is
  written in the code (`LABEL_SNAPSHOTS`), and every step refuses a file with
  another. The live review file (`data/events/alfven_eigenmode/review/labels.csv`)
  is never read by v2. (v1 read the live file and kept a copy beside each model.)
- **The source table**, `$LABELER_LABEL_TABLES/alfven_eigenmode/format/`, the
  table the owner started reviewing from. Its windows limit which frames are
  scored, and it is one of the methods compared. Every record names it by sha256.
- **TokEye's masks**, `$LABELER_ROOT/ae/masks/<shot>_<split>_clean.npz`: TokEye's
  cleaned coherent mask for the 180 "AE180" shots over 0-2 s. They give the MHD
  frames (below), TokEye's own AE call, the UCI annotation, and, from the file
  names, SELDNet's `train`/`valid` split.
- **The model's input**, the review page's AE rows in
  `$LABELER_ROOT/spectrograms/alfven_eigenmode/<shot>.h5`: CO2 R0 × V1, V2 and V3
  cross-power, 0-250 kHz, on the page's colour scale. v2's candidates all read
  the 80-250 kHz band.
- **The corpus**, `/scratch/gpfs/EKOLEMEN/foundation_model/<shot>_processed.h5`,
  read only by the extension, for its CO2 channels. Nothing is fetched.
- **The earlier detector**, SELDNet (`$LABELER_ROOT/models/d3d_ae_activity_seldnet/`,
  inputs in `$LABELER_ROOT/ae/dataset/`), scored beside the model.

## Frames and the MHD rule

Everything is scored on 10 ms frames. A frame is **scored** when the owner
called it present or absent, TokEye's record covers it, the model's input rows
cover it, and it lies inside the source table's window for that shot. Only
frames in 0-2 s are scored, since that is all TokEye and SELDNet ever saw, so
every method is compared on the same frames.

A frame is an **MHD frame** when TokEye's mask lights a line at or under about
60 kHz on at least two of the four chords for at least half of the frame. An
MHD frame the owner called absent is the mistake the earlier models made: MHD
seen as AE. Training up-weights these frames by the candidate's MHD weight, and
the MHD false-positive rate (the share of such frames the model calls present)
is half of the bar.

## The split and the folds

The **test shots** are the reviewed shots of SELDNet's own validation block
(its `valid` sessions), so no method is scored on a shot it trained on. On v2's
180 shots that gives 60 test shots (v1's 58 and 2 of the 18 new ones) and 120
**pool** shots. (The owner had approved 58 and 122; the rule gave 60 and 120,
and the ledger records this as Deviation 14.)

The pool is cut into **five folds** before any training: the sorted pool,
permuted with seed 20260923, dealt in turn, 24 shots each.
`python -m labeler.ae.xpower.cv --version v2 --folds` writes them once, to
`cv/folds.csv` (shot, split, fold) and `cv/folds.json`. Every later step
re-derives the folds from the snapshot and TokEye's masks and refuses unless
the file matches byte for byte; the choice, the final model and `chosen.json`
name its sha256.

## The candidates and the choice

Three candidates, the same FrameCNN on 80-250 kHz with v1's training settings,
differing only in how much an MHD frame the owner called absent weighs in the
loss: `band80-mhd3`, `band80-mhd10` and `band80-mhd30` (weights 3, 10 and 30).

**A fold task** (one per candidate and fold, 15 in all) predicts fold K with a
model that trained on three other folds and stopped early on fold (K + 1) mod 5
(validation F1 at a 0.5 threshold, patience 10, at most 60 epochs). Fold K's
shots never train or stop the model that predicts them. Each task saves, per
fold-K shot, P(AE) on the frames the test would score, to
`cv/<candidate>/fold<K>.npz` and the record `fold<K>.json`.

**The choice** pools the five folds' out-of-fold frames per candidate, so all
120 pool shots count once, and scores every (candidate, threshold) pair, with
thresholds 0.10 to 0.90 in steps of 0.05, for pooled F1 and MHD false-positive
rate. Then, in order:

1. If some pair has an MHD false-positive rate of 0.05 or less, take the one
   with the highest F1 among those.
2. Otherwise, if some pair has an F1 of 0.90 or more, take the one with the
   lowest MHD false-positive rate among those.
3. Otherwise, take the highest F1.

Ties go to the lower MHD weight, then to the threshold nearest 0.5, then to the
lower threshold. This rule was fixed in the ledger (Deviation 11) before any v2
number existed. `python -m labeler.ae.xpower.cv --version v2 --choose` applies
it: it refuses while any of the 15 records is missing or differs from what its
task should hold, and writes `cv/choice.json` and the full table in
`cv/frontier.md`. Run again, it checks the saved choice and changes nothing.

v2 chose **band80-mhd10 at 0.70 by branch 2**: no pair reached MHD FP 0.05, and
among those with F1 ≥ 0.90 this one had the lowest MHD FP (out-of-fold F1 0.902,
MHD FP 0.075).

## The final model

`python -m labeler.ae.xpower.train --version v2 --from-cv` trains the chosen
candidate on all 120 pool shots, with no early stopping, for a fixed number of
epochs: the median of its five folds' best epochs (v2: 10, 10, 15, 23, 28, so
15). It is saved at the chosen threshold, with the choice's, the folds' and the
snapshot's sha256, into `models/ae_xpower/v2/band80-mhd10/`, and `chosen.json`
is written beside it.

## The test, once

`python -m labeler.ae.xpower.evaluate --test --version v2` scores the final
model on the 60 test shots, with SELDNet, TokEye, the source table, the UCI
annotation and "always present" on the same frames: precision, recall and F1
pooled over shots, the MHD and other-absent false-positive rates, 95 %
shot-bootstrap intervals (2000 replicates, seed 20260923), and paired
differences against SELDNet and "always present". It writes `evaluation.json`
and `evaluation.md`, and reports v1's 58 test shots as a second table.
(`--choose` for v2 only checks `chosen.json` against `cv/choice.json` and
writes nothing.)

It refuses, and scores nothing, when:

- `evaluation.json` already exists outside `runs/`: the test shots are scored
  once, and a retry is a new version;
- it would be a full scoring (`--limit 0`) under `runs/`, where a record can be
  replaced, or a pilot there asks for more than 20 test shots; or `--limit` is
  negative, or positive outside `runs/`;
- the models directory is not v2's own (a copy elsewhere, by any path), or under
  `runs/` it holds a full choice's model rather than a pilot's;
- `--version` is not the directory's name or the checkpoint's version;
- `chosen.json` differs from `cv/choice.json` (candidate, threshold, the
  choice's sha256, the snapshot, the folds), or the choice was made from other
  folds, or `cv/folds.csv` is not what the snapshot and TokEye's masks give now;
- `model.pt`'s sha256 is not `chosen.json`'s, its split does not put the folds'
  test shots in test and every pool shot in train, or the checkpoint was not
  trained as the choice says (candidate, threshold, epochs, band, training
  settings, snapshot);
- the labels beside the model are not the snapshot;
- the source table is not the one the fold records scored;
- one of v1's test shots is not a v2 test shot, or a test shot has no source
  table label.

## The bar and the tier

The bar, judged on the 60 shots only:

- **A1:** F1 ≥ 0.90, precision and recall ≥ 0.75, and the lower bound of F1
  minus SELDNet's F1 ≥ −0.03.
- **A2:** MHD false-positive rate ≤ 0.05, and the upper bound of that rate
  minus SELDNet's below 0.
- **A3:** the lower bound of F1 minus "always present"'s F1 above 0.

v2 recorded F1 0.933 [0.917, 0.948] and MHD FP 0.029 [0.011, 0.049]: **A1
fails** (F1 minus SELDNet's lower bound is −0.032, 0.002 short of −0.03),
**A2 and A3 pass**. Whatever the scores, what the model writes is a
**suggestion**, never a label: the test shots were reviewed from a starting
table, not blind, and they are 2017-2019 shots, while the extension runs on
2024-2025 ones.

## The gallery

`python -m labeler.ae.xpower.gallery --version v2 --workers 1` draws one JPEG
per AE180 shot into
`$LABELER_ROOT/gallery/alfven_eigenmode/ae_xpower-v2/reviewed/<shot>.jpg`
(`unreviewed/` for a shot the owner has not saved), and `index.csv` beside them.
A picture shows the three cross-power rows, the owner's frames, and the model's
P(AE) with its threshold and the MHD frames ticked. A reviewed picture's title
gives the model's F1 against the owner over the frames the test scores,
"F1 vs owner, 0-2 s"; `index.csv` has it as `f1_0_2s` and, as before, the F1
over the owner's whole window as `f1_vs_owner`. v3 is scored over the owner's
whole windows, so its title says "F1 vs owner, whole window"; `index.csv` names
the window each version is scored on (`scored_window`, "0-2 s" or "whole") and
its F1 there (`f1_scored`), and keeps `f1_0_2s` for every version. For v3,
`f1_scored` counts only the frames its test scores (those the whole-shot TokEye
record covers and the store observes), where `f1_vs_owner` counts every frame
of the window. Only a version scored on 0-2 s draws the "scored: 0-2 s" line,
and no gallery or POI picture draws an 80 kHz line (the owner, 2026-09-28: that
floor was only the labelling view). Because a picture shows F1 on test shots,
a cross-validated version's gallery (v2, v3) refuses until `evaluation.json`
has scored the model it draws, and it draws only from that version's own
models directory (a pilot's under
`runs/` draws into its own `gallery/`).

## The extension and its gate (D47)

`python -m labeler.ae.xpower.extend --shard K --of 4 --version v2` runs the
model over the population's shots with at least 2 s of corpus CO2 (1,901 after
the cohort's blind test shots are removed), as suggestion rows per frame, and
draws each shot into the gallery's `extension/`. `--merge --of 4` then checks
every shard and writes one table. **The gate (D47):** the extension runs, and
merges, only when the version's full test evaluation passes A1 and A2 and names
the model and the choice by sha256; an evaluation under `runs/` never counts.
The merge also refuses a shard whose shots do not match the eligible population,
failures above 2 % of it, and a shard that was asked for pictures but lacks a
listed, on-disk picture for any shot it labelled.

**v2's extension did not run**, since A1 failed. A2 is not relaxed without the
owner's word.

## The post-hoc record and the seed study

Both come after the test and change nothing that was decided: the choice, the
verdict and the tier stand.

**The post-hoc record,** `python -m labeler.ae.xpower.posthoc --version v2`,
recomputes P(AE) from `model.pt` for the test shots through the test's own
loading and checks, and refuses unless `evaluation.json` names the same model,
`chosen.json`, split, labels and source table by sha256 (and, in a full run,
the same frame counts). It writes only `models/ae_xpower/v2/posthoc.json` and
`posthoc.md`, both headed as post hoc, and never touches `evaluation.*`:

- **(a)** the test's MHD false-positive rate by where the MHD frame lies: before
  the owner's first present frame (lead-in), between the first and last present
  frames (inside AE), after the last (after AE), and in shots with no present
  frame. Given for every method, with intervals and each method's difference
  from SELDNet;
- **(b)** the share of scored frames that are present, and that are MHD frames
  the owner called absent, in the pooled out-of-fold frames (40.9 % and 13.3 %)
  and in the test frames (69.6 % and 8.75 %);
- **(c)** the seed study, below, or "not run";
- **(d)** the second look: 58 of the 60 test shots were v1's test shots, scored
  with v1's model before v2 was chosen.

A pilot (`--limit 3`) writes only to `$LABELER_ROOT/runs/ae_xpower/posthoc/v2/`.

**The seed study** asks how much the chosen configuration's out-of-fold scores
move with the training seed alone, beside A1's 0.03 margin that the test missed
by 0.002. `python -m labeler.ae.xpower.cv --version v2 --candidate band80-mhd10
--fold K --seed S` retrains the chosen candidate's fold K with `TrainConfig.seed`
S (20260924, 20260925 and 20260926; the default is 20260923), with the same
folds, stop rule, epoch cap and data. It reads v2's `cv/folds.csv`, checked
against `chosen.json`'s `folds_sha256`; it trains only the chosen candidate;
it writes only to `$LABELER_ROOT/runs/ae_xpower/seeds/v2/seed<S>/cv/` (a pilot
to `seed<S>/pilot/`), refusing any other directory and any record already
there; and each record carries its seed. The post-hoc record's (c) pools each
seed's five folds at 0.70: F1, precision, recall, MHD FP, whether F1 ≥ 0.90,
and the range and standard deviation across seeds.

## Caveats for the paper

- **The choice sits on seed noise.** At 0.70, band80-mhd10's pooled out-of-fold
  F1 is 0.902 at the default seed and 0.889, 0.888 and 0.886 at the three study
  seeds, so only the default seed clears branch 2's F1 ≥ 0.90. The study
  retrains only the chosen candidate, so it does not measure whether the choice
  rule would still pick band80-mhd10 at 0.70 under another seed.
- **Best epochs move with the seed.** The five folds' best epochs have medians of
  15 at the default seed and 14, 20 and 15 at the study seeds. Fold 0's best
  epoch is 23 at the default seed and 7 at 20260924.
- **The final model's epochs are longer than a fold's.** A fold's model trains on
  three folds (72 shots; the next fold early-stops it), so at 8 crops a shot an
  epoch is 576 crops. The final model trains on all 120 shots, 960 crops an
  epoch, so its 15 epochs are about 1.67 times the training steps of a fold at
  the median, and it is saved at the threshold chosen on the folds.
- **The test shots were looked at before.** 58 of the 60 are v1's test shots.
  v1's A2 failure on them (MHD FP 0.241) is why v2 weighs MHD frames. v2's
  choice read only out-of-fold frames, but its design answered a result on the
  test shots, so the test is not fully fresh. A clean confirmation needs shots
  that no version has seen.

## The jobs, in run order

Each script's header gives its sizing and the measured runs behind it. Prefix
variables (`VERSION`, `PILOT`, `SEED`, `MODELS`, `LIMIT`, `SHOTS`, `OF`) go on
the sbatch line only. Pilots come first; each job is gated with
`python -m labeler.jobstats --job-id <id> --cpu-only`. The `sbatch` lines run
from `scripts/labeler/`; the `python -m` lines run on the login node, from the
repository root, in the `labelmaker` environment (`pixi run -e labelmaker`).

```bash
# 1. The folds, once, on the login node (seconds).
python -m labeler.ae.xpower.cv --version v2 --folds
# 2. The 15 fold tasks (3 candidates x 5 folds).
python -m labeler.ae.xpower.cv --version v2 --folds --pilot 10   # the pilot's folds
PILOT=10 sbatch ae_xpower_cv.sbatch
sbatch ae_xpower_cv.sbatch
# 3. The choice, then the final model.
PILOT=10 sbatch ae_xpower_final.sbatch
sbatch --dependency=afterok:<cv array id> ae_xpower_final.sbatch
# 4. The test, once.
VERSION=v2 MODELS=$LABELER_ROOT/runs/ae_xpower/pilot/v2 LIMIT=3 sbatch ae_xpower_evaluate.sbatch
VERSION=v2 sbatch ae_xpower_evaluate.sbatch
# 5. The gallery.
VERSION=v2 SHOTS="170659 170813" sbatch ae_xpower_gallery.sbatch
VERSION=v2 sbatch ae_xpower_gallery.sbatch
# 6. The extension, only if A1 and A2 passed (v2: not run).
VERSION=v2 LIMIT=20 sbatch --array=0 --cpus-per-task=4 --mem=7G ae_xpower_extend.sbatch
VERSION=v2 sbatch ae_xpower_extend.sbatch
python -m labeler.ae.xpower.extend --merge --of 4 --version v2   # login node
# 7. The seed study: band80-mhd10's folds are tasks 5-9.
SEED=20260924 PILOT=10 sbatch --array=5-9 ae_xpower_cv.sbatch
SEED=20260924 sbatch --array=5-9 ae_xpower_cv.sbatch
SEED=20260925 sbatch --array=5-9 ae_xpower_cv.sbatch
SEED=20260926 sbatch --array=5-9 ae_xpower_cv.sbatch
# 8. The post-hoc record (again after the seed study, for its (c)).
LIMIT=3 sbatch ae_xpower_posthoc.sbatch
sbatch ae_xpower_posthoc.sbatch
```

v1 ran `ae_xpower_train.sbatch` (its three candidates, each with its own
validation shots), then `ae_xpower_evaluate.sbatch` (`--choose` on validation:
the lowest MHD false-positive rate within 0.02 of the best F1; then `--test`),
then the gallery.

## The production files

**`$LABELER_ROOT/models/ae_xpower/<version>/`**

| file | what it is |
|---|---|
| `review/labels.csv` | v2's frozen label snapshot, the only labels v2 reads |
| `cv/folds.csv`, `cv/folds.json` | the five folds (shot, split, fold) and their record: seed, counts, sha256 |
| `cv/<candidate>/fold<K>.npz` | fold K's out-of-fold frames per shot: P(AE), the owner's states, MHD, scored |
| `cv/<candidate>/fold<K>.json` | the fold task's record: shots, stop fold, training history and settings, shas |
| `cv/choice.json`, `cv/frontier.md` | the choice: the table of every (candidate, threshold), the branch, the final epochs |
| `<candidate>/model.pt` | the final model: weights, threshold, band, and the shas of its choice, folds and labels |
| `<candidate>/split.csv` | its shots: the pool as `train`, the test shots as `test` (v1: `train`, `val`, `test`) |
| `<candidate>/training.json` | how it trained: shot counts, history, best or fixed epochs, threshold, shas, peak memory |
| `<candidate>/review/labels.csv` | the labels it trained on (for v2, the snapshot's bytes) |
| `chosen.json` | the chosen candidate and threshold, why, and the shas of the choice, folds, labels and model |
| `evaluation.json`, `evaluation.md` | the one test: every method's scores, the differences, the bar, and what was scored by sha256 |
| `posthoc.json`, `posthoc.md` | the post-hoc record, after its job has run |

v1's directory has one `<candidate>/` per candidate and `validation_frontier.{csv,md}`
(a validation-only threshold sweep), and no `cv/` or `review/` of its own.

**`$LABELER_ROOT/gallery/alfven_eigenmode/ae_xpower-<version>/`**

| file | what it is |
|---|---|
| `reviewed/<shot>.jpg` | a shot the owner has saved, with F1 vs owner over the scored window (0-2 s; v3's whole window) in its title |
| `unreviewed/<shot>.jpg` | a shot not yet saved; the strip is the source table's |
| `extension/<shot>.jpg` | an extension shot (none for v2) |
| `index.csv` | one row per picture: shot, group, split, file, window, present frames, `f1_vs_owner`, threshold, candidate, version, `snapshot_sha256`, `f1_0_2s`, `scored_window`, `f1_scored` |

**`$LABELER_ROOT/runs/ae_xpower/`** holds pilots and post-hoc runs, where a
record may be replaced (except the seed study's) and nothing counts as the test:

| directory | what it is |
|---|---|
| `pilot/` | v1's pilot: a candidate, its `chosen.json` and a 3-shot `evaluation.*` |
| `pilot/v2/` | v2's pilot: its folds, fold records and choice in `cv/`, its choice's model and `chosen.json`, a 3-shot `evaluation.*` |
| `seeds/v2/seed<S>/cv/band80-mhd10/` | the seed study's fold records for seed S, never replaced; a pilot's under `seed<S>/pilot/` |
| `posthoc/v2/` | the post-hoc pilot's `posthoc.{json,md}` |

The extension writes to `$LABELER_ROOT/suggestions/ae_xpower/<version>/`
(`shards/`, then the merged table, `summary.csv` and `failed.jsonl`); v2 has
none. Job logs are in `$LABELER_ROOT/runs/slurm/`.
