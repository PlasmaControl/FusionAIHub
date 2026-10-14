# Gold set

> The label data is `data/events/gold/`: the `raw/`, `format/`, `review/` and `extend_*/` paths
> below are relative to it. It is not in git. `scripts/labeler/hf_publish.py pull`
> restores the label tables (`format/`, `review/`, `extend_*/`), but not `raw/`: the
> third-party originals are not in the Hugging Face release; they are on the Stellar
> checkout's disk and archived in `/scratch/gpfs/EKOLEMEN/nc1514/labeler/raw_archive/`.

## Description
50 DIII-D shots meant for evaluation. Each is 170000 or later and is not one of the
catalog cohort's blind shots. Together they cover every observed class, about
evenly, and are split about evenly over the shot-number decades the corpus has:
13 shots from 17xxxx, 12 from 18xxxx, 12 from 19xxxx and 13 from 20xxxx (the
owner asked for the same number from 17xxxx to 21xxxx; no 21xxxx shot exists, as
the operator log ends at 208495 and the corpus at 204999). They come from 13 run
days, a few shots from each, so a model trained elsewhere gives up few run days to
be evaluated here.

Evidence behind them: 42 carry a hand label or a curated source, meaning a saved
review label, a format table from another source, the I-mode logbook, or a row of
the confinement source table (`regimes`: the curated Gill and Butt intervals, else
the D-alpha detector's rows). Eight do not. Three are the owner's own picks, 199597,
199603 and 199607, from the 2024-07-11 tearing-mode stabilization experiment (shots
199597-199608, run `20240711A`, the subject of the 2026 Physics of Plasmas paper
"Interpreting AI for fusion"). Five are the 2022-04-29 detachment shots 189081,
189088, 189090, 189094 and 189097, kept from the first pick for the detachment classes: only a
detachment voting table (a model's output) labels detachment, and the eight shots that carry it and
are not in the silver list are all from 2022-04-29 (gold keeps seven). The rest were picked by the script.

The owner's review page offers them with the `Gold set` box next to the event
list, on every event, whether or not the event's own roster has the shot. The 450-shot list reviewed for every event is the silver set
(`../silver/`), which shares none of these shots.

`gold_shots.csv` predates the Alfven eigenmode table `../alfven_eigenmode/extend_ae_seldnet/gold_shots.csv`, which gives `d3d_ae_activity_seldnet` frames (unreviewed model output) for the 37 gold shots without an AE label (the 13 from 2017-2019 carry one). The `sources` and `classes` columns here do not list it.

`../confinement/extend_confine_ours/gold_shots.csv` gives `confine-ours` regimes (unreviewed model output) for 27 gold shots, from fetched density, beta_N and W_MHD traces (`gold_shots.meta.json` lists the inputs): of the 28 with no confinement label, all but 204497, which had no neutral beams and so has no beam-on bins to label. 199597 has its labels on `../confinement/extend_confine_ours/roster_review.csv` (the roster in the layout the page reads, made 2026-10-11) and was dropped from `gold_shots.csv`. The other 21 carry a regimes row.

The edge localized mode table `../edge_localized_mode/extend_elm_ours/gold_shots.csv` gives `elm-ours` occupancy frames (unreviewed model output) for 7 gold shots that have no ELM label or suggestion (176070, the owner's 199597, 199603, 199607, and the I-mode shots 204501, 204503, 204505). Shots the model read as under 5 % present are not published: 13 from 2017-2019 were dropped at the owner's word on 2026-10-11, and of the 20xxxx shots added on 2026-10-12 that no ELM table covers (the five I-mode shots), 204497 and 204499 came out 0.0 present and were scored but not published; shots that already carry one are not in it, and the `sources` and `classes` columns here do not list it either.

**ELM training overlap (2026-10-14).** Seven gold shots, 192751, 195016, 195024, 195102, 195103, 195111 and 200954, are among the 119 reviewed ELM shots whose saved labels trained `elm-ours`: they sit in folds 1 to 4 of its `cv2` run, so four of its five fold models trained on each. A gold-set score of `elm-ours`, or of any model trained on the ELM review labels, is therefore contaminated on these seven: report it on the other 43 shots, or give the seven apart. `elm_jump`'s run rule and gate were also chosen on those 119 shots. The seven stay in the gold set (the owner's call). `elm-ours` is not served on them: `../edge_localized_mode/extend_elm_ours/silver_gold.csv` (2026-10-14) gives its rows for 11 other gold shots (189081, 189088, 189090, 189094, 189097, 189100, 189580, 189602, 189603, 195034, 195035), and `labeler.elm.serve` refuses any fold shot.

## Data Provenance
### Dataset 1

**Dataset File(s)**: `gold_shots.csv`, picked 2026-10-11 (`pick_gold.py`) and rebalanced by shot-number decade on
2026-10-12 (`pick_gold_v2.py`; the first list is kept there as `gold_shots_v1.csv`). The picking scripts
and their inputs are kept in
`/scratch/gpfs/EKOLEMEN/nc1514/labeler/gold/v1/selection/`; the sha256 of each is
in `gold_shots.meta.json`.

The columns are `shot, run_day, year, classes, sources, notes`:
- `run_day` is the DIII-D run day (`YYYYMMDD`, from the operator log);
- `classes` lists the `event:category` labels some table already gives the shot;
- `sources` lists the kinds of table behind them (saved, format, source,
  logbook, extend, suggest). "source" is the first pick's loose reading: any
  confinement table, a frames suggestion included, counted as a curated source.
  The Description above gives the strict count (42 shots);
- `notes` says "not in the corpus (fetched)" for the 13 shots outside
  `/scratch/gpfs/EKOLEMEN/foundation_model`; their data was fetched with the
  silver set's old shots.

**Selection, first pick** (`pick_gold.py`): a greedy pick over run days. The pool is every
corpus shot, plus the fetched old shots, from 170000 up, not blind, with a run day.
A class adds w · r · (log(1 + n + 1) − log(1 + n)) for each shot, where n is how many
picked shots carry it, w is the evidence weight (saved 3, format, logbook and the
confinement source 2, extend 1.5, suggest 1) and r = 1/√(shots in the pool that carry
the class), so rare classes come first. A run day is scored by the gain of its best
7 shots; days are taken until 50 shots, and a class no chosen day carries (RWM
present, three shots in the pool) is swapped in for the most redundant shot.
Last, the owner's three go in, each in place of the most redundant pick that
holds no saved label and is not the last shot of a class (195113, 195032, 192766).
That list held 14 shots from 17xxxx, 16 from 18xxxx, 20 from 19xxxx and none from 20xxxx.

**Selection, rebalanced** (`pick_gold_v2.py`, 2026-10-12): the quota is 13, 12, 12 and 13 shots from
17xxxx, 18xxxx, 19xxxx and 20xxxx. The silver list's 450 shots leave the pool, so the two sets stay
disjoint. Where a decade is over its quota the most redundant shots go, by the same class
loss (never an owner pick or a saved-label shot, and the last holder of a class only when nothing else
is left): 170670; 189093, 189598, 189599, 189607; 192755, 192760, 192763, 192768; 195037, 195040,
195101, 195107. The 20xxxx quota is filled the way the first pick was, a greedy over run days under the
same class gain, but only from shots with a hand label or a curated source in the strict sense above
(saved, format, the I-mode logbook, or a `regimes` row) and with no two added shots on consecutive shot
numbers, since a run's neighbours repeat each other: 2025-07-14 (the I-mode logbook run), 2024-08-12, 2024-12-02 and
2024-09-23. Silver already holds 31 of the 32 shots from 200000 up that carry a saved label (ELM), so
the 20xxxx shots here lean on the confinement source and the I-mode logbook; the other classes on them
come from the owner's pointed suggestion tables. 16 shots carry a saved label. Every class the pool carries
still has a gold shot.

| run day | shots | shot numbers |
|---|---|---|
| 2017-05-04 | 6 | 170659, 170660, 170672, 170677, 170678, 170679 |
| 2018-03-14 | 1 | 176070 |
| 2019-06-20 | 6 | 178631, 178633, 178636, 178637, 178640, 178641 |
| 2022-04-29 | 8 | 189081, 189085, 189088, 189090, 189094, 189097, 189100, 189101 |
| 2022-05-19 | 4 | 189580, 189597, 189602, 189603 |
| 2022-11-09 | 2 | 192751, 192761 |
| 2023-04-05 | 4 | 195016, 195024, 195034, 195035 |
| 2023-04-07 | 3 | 195102, 195103, 195111 |
| 2024-07-11 | 3 | 199597, 199603, 199607 |
| 2024-08-12 | 3 | 200007, 200009, 200013 |
| 2024-09-23 | 2 | 200954, 200958 |
| 2024-12-02 | 3 | 202098, 202100, 202102 |
| 2025-07-14 | 5 | 204497, 204499, 204501, 204503, 204505 |

| class | gold shots | in the pool |
|---|---|---|
| alfven_eigenmode.rsae:present | 10 | 11 |
| alfven_eigenmode.tae:present | 10 | 11 |
| alfven_eigenmode:present | 12 | 23 |
| confinement:high | 29 | 4498 |
| confinement:imode | 8 | 32 |
| confinement:low | 26 | 4039 |
| confinement:qh | 4 | 18 |
| confinement:wpqh | 4 | 13 |
| detachment:attached | 7 | 8 |
| detachment:detached | 6 | 6 |
| edge_localized_mode:present | 20 | 2888 |
| mhd_n1.m1:present | 35 | 6934 |
| mhd_n1.m2:present | 2 | 2 |
| mhd_n1.m3:present | 5 | 5 |
| mhd_n1:present | 36 | 7472 |
| mhd_n2.m3:present | 2 | 2 |
| mhd_n2:present | 13 | 66 |
| mhd_n3:present | 11 | 41 |
| minimum_safety_factor:elevated | 5 | 47 |
| minimum_safety_factor:high | 7 | 45 |
| minimum_safety_factor:hybrid | 4 | 286 |
| minimum_safety_factor:low | 8 | 45 |
| poloidal_beta:present | 7 | 11 |
| resistive_wall_mode:present | 1 | 1 |

| shot numbers | 17xxxx | 18xxxx | 19xxxx | 20xxxx |
|---|---|---|---|---|
| gold shots | 13 | 12 | 12 | 13 |

Classes with few gold shots are the rare ones: RWM present is on one shot, the 2/1 child of
n=1 and the 3/2 child of n=2 are on two, and the 3/1 child of n=1, QH, WPQH and the hybrid q_min
regime are on four or five. "In the pool" counts the shots outside the silver list, so
the pool holds no more RWM, 2/1, 3/1 or 3/2 shot than gold has.
