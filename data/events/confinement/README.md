# Shared confinement labels

Untouched labels attributed to **Kevin Gill** (the XLSX workbook and BES-time
HDF5 files) and **Jalal Butt** (the April 2024 CSV). This directory is a shared
source collection, not an additional catalog category (`events.yaml` lists it
under `non_category_dirs`). L/H/QH/WP remain distinct in the reconciled interval
table; the binary detector treats H/QH/WP as high confinement.

## Review

The review page offers confinement as one event, `confinement`, in place of
`high_confinement_mode`, `low_confinement_mode`,
`quiescent_high_confinement_mode` and
`wide_pedestal_quiescent_high_confinement_mode` (it neither lists nor opens
those four; their tables and rosters are untouched, and the frame models and the
paper still read them). A span is 1 high, 2 low, 3 qh, 4 wpqh or 5 uncertain;
time left unmarked is category 0 (absent), and nothing in it is observable.

- `shots.csv` is the roster: the H-mode roster's 450 shots, in its order, then
  389 curated shots outside it (unverified, not held out). They are the shots of
  the merged interval table below that have a D-alpha record, in the corpus or in
  the raw cache (the cache holds the rest, fetched while logged into fdp). Not on
  it: 2 blind shots, and 45 shots without a D-alpha record (6 have none to fetch,
  174601, 174602, 175658, 179212, 179216 and 179314; the other 39 were not reached
  before the fetch hit its time limit).
- `review/source.json` points at the suggestion an unsaved shot opens on,
  `suggestions/regimes/v1/confinement_suggest_regimes_v1.csv` under
  `$LABELER_ROOT`, built once by hand from two sources:
  - the curated intervals, `runs/labeler/confinement/v1/merged_intervals.csv`:
    H is 1 high, L is 2 low, QH is 3 qh, WP is 4 wpqh, and the one H|L conflict is
    5 uncertain. This is what the 401 shots that have any (the 12 on the 450 and
    the 389 added) open on, tiled over the shot's window with the rest unmarked.
    A shot outside the cohort has the population's window when it has one, else 0
    to 500 ms past its last interval, as the resistive-wall-mode reference shots
    do;
  - for every other shot, the H-mode D-alpha table renumbered
    (`suggestions/dalpha_lh/v1/confinement_suggest_dalpha_lh_v1.csv`): high stays
    1, uncertain becomes 5, and the table's no-H-mode and not-observable stretches
    are left unmarked, because the detector never established L-mode. Only
    curated shots show low, qh or wpqh.

  Its `.meta.json` records both sources' sha256 and the mapping.
- `review/labels.csv` and `review/history.jsonl` are written by the page when a
  shot is saved.
- The panels are the H-mode event's (D-alpha, density, NBI, beta_N). They are
  built once under `spectrograms/high_confinement_mode/` and shared
  (`labeler.config.STORE_EVENTS`).

From the repository root:

```bash
pixi run --frozen -e labelmaker python -m labeler.confinement labels
```

Writes the exact source union, conflicts, conservative 50 ms targets, shot splits,
class/shot-balanced training weights and source hashes under
`runs/labeler/confinement/v1`. Unknown samples and incomplete bins are excluded;
held-out class prevalence is preserved. Raw `Test only` assignments apply across
all sources for a shot. Raw files and corpus stores are read-only.

The workbook's labelled intervals exactly match entries in Jalal's CSV;
matching files must not be treated as independent expert annotations. See the
[analysis](../../../docs/labeler/confinement_analysis.md) for measured overlap,
the within-source conflict, source completeness and scientific review records.

The [executed notebook](example.ipynb) includes source statistics, ELM consistency,
diagnostic eligibility and frozen detector results. The
[model card](../../../docs/labeler/confinement_model_card.md) documents training,
no-BES inference, the measured limits and commands. Merged complementary H/L
exports are under `runs/labeler/confinement/v1/format/`; trained models and
their evaluation are in that same run directory.

## Models

**stable**: none

**latest**: confine-ours (experimental, see below)

**all**:
- confine-ours | 2026_10_03 | AUROC: 0.964 | AUPRC: 0.913 | F1: 0.885
- confine-cnn | 2026_10_03 | AUROC: 0.889 | AUPRC: 0.713 | F1: 0.703
- confine-cnn | 2026_10_01 | AUROC: 0.926 | AUPRC: 0.701 | F1: 0.678 (read 2 blind test shots; the same recipe without them, 117 shots: 0.717)

Scores are macro means over the four classes (L, H, QH, WPQH), per window or bin, out of
sample, with 95 % shot-bootstrap intervals in the records. **confine-ours** (a 1D U-Net over 0D
signals, no BES; labels the whole roster, see
[confinement_ours.md](../../../docs/labeler/confinement_ours.md)) is **experimental**: its score is
per 1 ms bin on 401 curated shots in 5-fold cross-validation whose folds hold out whole run days:
F1 0.885 [0.855, 0.914]. (Folds that split shots at random scored 0.934; that was not out of
sample in the sense a new campaign is, because 169 of 218 pairs of neighbouring shots straddled
folds.) On the same BES windows its margin over the BES network, whose folds hold out shots but not
run days, is small and depends on the windows: paired
macro-F1 difference ours minus `confine-cnn` -0.029 [-0.128, +0.071] on the 117 shots the corpus
holds BES for, +0.046 [+0.006, +0.087] on 400 shots without a beam gate, +0.092 [-0.044, +0.243]
on the 124 test shots and gated windows of the protocol-matched BES row. Its roster labels,
`extend_confine_ours/roster.csv`, have only 42 % of their label time on the curated shots inside a
curated interval (the rest the experts left unlabelled); they
put every segment under a confidence of 0.7 in category 5 (uncertain), mark QH and WPQH on shots
without curated labels as an unreviewed tier, and flag shots after 196493, the last of the training
range, as extrapolated; they are not yet reviewed.
**confine-cnn** is the BES network of Gill et al. (2024), rebuilt from the paper's description (its
code is not on disk) and retrained on our labels. The 2026_10_03 line is the protocol-matched
score, **our reimplementation under the paper's selection and split**: the paper's data selection
(beam gate, margins, optimiser, a 6 x 8 block chosen from the channel positions, its architecture
and 60,000 steps with no early stopping), native 1 MHz BES, 401 shots, split by discharge; macro F1
0.703 [0.587, 0.794] on 142 distinct test shots (five random splits whose test sets overlap: 200
shot-tests, 152,863 windows pooled of which 106,232 are distinct; one split scores 0.684 +/- 0.074,
each shot counted once 0.736, probabilities averaged over the repeats 0.747; corpus shots 0.463 on
34, other shots 0.742 on 108), against the paper's 0.94 on its own 44-shot test set. The remainder
is unexplained. The 2026_10_01 line is the first retrain, with none of that protocol, on the 119
shots the corpus holds BES for at 500 kHz, and it read two blind test shots: reading them out, the
same recipe scores 0.717 [0.63, 0.78] on 117 shots, and the first retrain's 0.678 is not a
benchmark number. Protocols, the
ablation of the gap to 0.94 and the confident-learning list of doubtful intervals:
[confinement_bes_benchmark.md](../../../docs/labeler/confinement_bes_benchmark.md). The
curated intervals (the merged tables of Gill and Butt) are the reference for every score. A model
that reads the traces the experts used (`confine-ours`) is not independent of them.
