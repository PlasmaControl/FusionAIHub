# `data/labels/` — curated label tables

Label **data** lives here and never under `src/`. Code that reads it is
`src/labelmaker/events/databases.py`, and the only thing that stands between a
CSV somebody emailed us and an event row is `tables.yaml`.

```
data/labels/
  README.md
  tables.yaml                      # the manifest: one entry per CSV
  Recommender System - Discrete Labels.csv     # the label roadmap (a sheet, not a table)
  resistive_wall_mode/
    rwm_onsets_2017.csv
    rwm_onsets_2024.csv
  <phenomenon>/<table>.csv         # every future table
```

## What a curated table is, and what it is not

A table is somebody's **list**: a shot, a time, and whatever they recorded
about it. It is knowledge we did not compute and cannot re-derive, so:

* **A CSV stays exactly as its author sent it.** No renamed columns, no unit
  conversion on disk, no de-duplication, no sorting. Everything that
  reconciles a table with labelmaker's schema is in the manifest and the
  loader. A duplicate `(shot, time)` row is kept — two onsets at the same time
  is a claim the table makes, not a mistake to tidy.
* **A listing is not a coverage claim.** Every event a table produces carries
  `t_cov0_s = t_cov1_s = NaN`, because a database that names a shot says
  nothing about which interval of that shot anybody examined. A shot **absent**
  from a table is therefore **not** a negative, and the NaN coverage is what
  stops every downstream consumer from reading it as one. A shot no table names
  gets no event row *and no source row*: nobody looked.
* **A curated list has no calibrated probability.** `confidence` is NaN.
  Writing 1.0 would let a ranker treat a human list as a perfectly confident
  detector.

## Adding a table

1. Put the file at `data/labels/<phenomenon_dir>/<stem>.csv`, untouched. One
   directory per phenomenon; the directory name is prose (`resistive_wall_mode`),
   the manifest's `phenomenon:` is the join key (`rwm`).
2. Add one entry to `tables.yaml`:

   ```yaml
     - stem: wpqh_2023           # the file is <dir>/<stem>.csv; also the source suffix
       dir: quiescent_h_mode
       phenomenon: qh            # an events/lexicons.yaml id, validated at load
       kind: point               # point (t_col) | interval (t0_col, t1_col)
       shot_col: SHOT
       t_col: START_TIME
       t_units: ms               # ms | s
       attr_cols: [NTOR]         # carried verbatim into attrs
       attr_types: {NTOR: int}   # int | float | str
       provenance: "Who made it, from what, in which campaign"
   ```

   A stem that is not unique across the manifest is an error, and so is a
   `phenomenon` that is not a `lexicons.yaml` id: the lexicon is the single
   vocabulary, and a typo that loaded silently would be a phenomenon with no
   evidence.
3. Ingest it, over any shot list, without the corpus or the GPU path:

   ```bash
   python -m labelmaker.run events --databases-only \
       --shot-file $LABELMAKER_ROOT/recommender_v1.txt
   ```

   The run prints `N of M shots are named by any table`. Zero is a normal
   answer — the RWM tables span 156785–176092 and the corpus starts at 185601,
   so on `recommender_v1` the honest answer is 0 of 500.
4. No code changes. If a table needs one, the manifest is missing a field and
   that is the thing to add.

## Committing

`.gitignore` excludes `data/*` and re-includes `data/labels/` — the glob and
not `data/`, because git never descends into an excluded *directory* and a
`!data/labels/` after `data/` would un-ignore nothing. So a new table is a
plain `git add` and needs no `-f`:

```bash
git add data/labels/tables.yaml data/labels/<dir>/<stem>.csv
```

(A checkout whose `.git/info/exclude` also carries `data/` will still hide
them — that file is per-clone and not tracked; drop the line there once.)

Only commit a table that is small and that we are allowed to redistribute. One
that is neither lives on `/scratch` and the root moves with it:

```bash
export LABELMAKER_LABEL_TABLES=/scratch/gpfs/EKOLEMEN/nc1514/label_tables
```

`config.Paths.label_tables` is the only place that names this directory.

## What is here now

| table | rows | shots | span | provenance |
|---|---|---|---|---|
| `rwm_onsets_2017` | 30 | 20 | 156785–158023, 0.856–3.061 s | Jeremy Hansen, 2017 campaign |
| `rwm_onsets_2024` | 26 | 13 | 176067–176092, 1.654–4.400 s | Jeremy Hansen, 2024 campaign |

56 onsets over 33 distinct shots. **None of the 33 is in the corpus**
(185601–204999), in `recommender_v1`, or in the features store. The loader
lands on them anyway: they are the first database-backed label and they define
the contract every later curated list arrives through.

`Recommender System - Discrete Labels.csv` is the label **roadmap** — the
user's editable sheet of what to build — not a curated table, and it is
deliberately absent from `tables.yaml`.
