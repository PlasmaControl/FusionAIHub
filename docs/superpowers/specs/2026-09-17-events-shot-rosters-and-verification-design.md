# Event shot rosters and human verification

Date: 2026-09-17
Status: design, approved for planning

Every phenomenon under `data/events/` gets a shot-level review roster
(`shots.csv`) and a notebook (`verification.ipynb`) that shows a reviewer the
signals that decide whether the event is real, lets them correct the label
intervals by dragging on the plot, and records who reviewed what.

## Problem

`data/events/<category>/` holds interval CSVs and per-shot label grids, and
`example.ipynb` plots them. Nothing records whether a human has ever *looked*
at a shot's labels, and nothing lets a human change them. Label quality is
therefore unmeasured: the ELM table has 576 shots and the NTM table 4,820, all
of them at the same unknown standard as each other.

## Scope

Fifteen category directories under `data/events/`, each gaining:

- `shots.csv` — the review roster, three placeholder rows.
- `verification.ipynb` — the review surface.

Three categories get real, hand-written panel definitions:
`minimum_safety_factor`, `sawtooth_oscillation`, `alfven_eigenmode`. The other
twelve get a generic fallback panel set until somebody writes theirs.

Out of scope: choosing the actual gold shots (none are chosen yet), rerunning
any formatter, and the repository-wide `labelmaker`/`ideate` rename sweep.

## Roster schema: `data/events/<category>/shots.csv`

```csv
shot,tier,reviewers,verified_on,notes
000001,gold,alice;bob,2026-01-01,EXAMPLE - replace
000002,silver,alice,2026-01-01,EXAMPLE - replace
000003,unverified,,,EXAMPLE - replace
```

| Column | Meaning |
| --- | --- |
| `shot` | DIII-D shot number, integer, unique within the file |
| `tier` | `gold`, `silver`, or `unverified` |
| `reviewers` | `;`-separated reviewer ids, in the order they reviewed |
| `verified_on` | ISO date of the most recent review; blank when unverified |
| `notes` | free text, one line |

Tier is a count of *independent* reviews, not a judgement of the labels:

- `gold` — two or more reviewers, each having pressed Verify on this shot.
- `silver` — exactly one reviewer.
- `unverified` — nobody has reviewed it.

`tier` is written explicitly because people scan the file by eye, and it is
redundant with `reviewers` by construction. `validate_shots` enforces the
agreement — `len(reviewers) >= 2` iff `gold`, `== 1` iff `silver`, `== 0` iff
`unverified` — the same way `validate_intervals` guards the interval schema.
A reviewer id is `$USER`, and a reviewer appears at most once per shot;
pressing Verify twice updates `verified_on` and does not promote the tier.

Ten gold shots per category is the target, not a rule the file enforces. The
roster is not the category's shot list: a shot enters it when someone puts it
up for review, and the interval tables under `format/` and `extend_*/` remain
the record of what is labelled.

The three placeholder rows ship in every category and are meant to be deleted.

## Corrections: `data/events/<category>/review/<shot>.csv`

A reviewer's edits are written per shot, in the public interval schema already
documented in `data/events/README.md`:

```csv
shot,category,t_start,t_end,confidence
170815,1,450.0,700.0,
```

The review directory is a separate claim from `format/` and `extend_*/`: it is
what a human asserts after looking, and no formatter reads or overwrites it.
Merging reviews back into a formatted table is a later decision, deliberately
not made here.

## Verification notebook

One `verification.ipynb` per category. Shared machinery lives in
`src/labeler/events/verify.py`; the panel definitions live **inline in each
notebook**, because deciding which traces settle a phenomenon is case-by-case
work that does not generalise.

### Shared: `src/labeler/events/verify.py`

```python
corpus_signal(shot, group, channels=None, t_range=None) -> FeatureArray
```

Reads one group from `<corpus>/<shot>_processed.h5` with h5py slicing. ECE is
`(48, 3.1e6)` float32 and CO2 `(4, 4.5e6)`; neither is ever read whole. Raises
`NoDataError` naming the shot and the corpus span when the file is absent or
the group carries the `(C, 1)` absent-signal sentinel.

```python
review(event, shot, panels, *, source="format/shots") -> ReviewSession
```

Builds a stacked `plotly.graph_objects.FigureWidget`: one row per panel, a
shared time axis in milliseconds, and a final row showing the current labels
read from the category's `.npz` grid. Box-select on any row captures a time
range; `Mark present` / `Mark absent` turn that range into a correction row;
`Verify` appends `$USER` to the roster and promotes the tier; `Save` writes
both files. Nothing is written until a button is pressed.

`FigureWidget` requires `anywidget`, which is not currently installed — one
entry in `[tool.pixi.feature.labelmaker.pypi-dependencies]` and a lock update.

### Panels: `minimum_safety_factor`

`qmin` from the feature store (all 337 rostered shots have it), `qpsi` as a
profile heatmap, and `ip` for context. The existing `qmin_rule` classes
(0 absent, 1 low, 2 hybrid, 3 elevated, 4 high) are the label row.

### Panels: `sawtooth_oscillation`

Raw ECE from the **corpus** `ece` group — `(48, ~3.1e6)` at roughly 500 kHz —
in four rows of four channels each, covering channels 20–36. The feature
store's `ece` is decimated to 1 ms and is useless here: a sawtooth crash is
100 µs to 1 ms, so the decimated record cannot show one. Channels are
overplotted within a row so the inversion radius shows as the phase flip
between rows.

### Panels: `alfven_eigenmode`

CO2 crosspower spectrograms for the chord pairs R0×V1, R0×V2, R0×V3, computed
with `scipy.signal.csd` over the 500 kHz record, plotted as log magnitude
against time and frequency with the 80–250 kHz AE band marked. The corpus
`co2` group carries the four chords in order `r0, v1, v2, v3`.

Two data facts constrain this panel, and the notebook states both rather than
failing silently:

1. **The 180 annotated AE shots are outside the corpus.** They span
   170659–178879; the corpus covers 185601–204999. None of the 180 has a
   corpus file or a feature file.
2. **Corpus `co2` is sparse.** It is filled on roughly half of corpus shots,
   and the filled ones measured so far are all above shot 198279.

So `alfven_eigenmode` gets an fdp fallback: when the corpus has no CO2 for a
shot, fetch `DENR0UF`, `DENV1UF`, `DENV2UF`, `DENV3UF` through
`toksearch_d3d.PtDataSignal` and cache the result under the review directory.
This is new code — the `co2` feature declares `sources=("corpus",)` and has no
fdp resolver today. The fetch requires the kernel to have been launched under
the `fdp run` wrapper and a valid SciToken; without them PTDATA fails with
`getservbyname failed for task 'PTSERVER'`, so the fallback checks first and
raises a message naming the wrapper rather than surfacing that error.

### Panels: the other twelve categories

`ip`, `betan`, and `ne_line` from the feature store, plus the label row. Enough
to confirm a shot exists and its labels sit inside the discharge; not enough to
verify a phenomenon. Each notebook says so in its first markdown cell.

## Moving `plot_original` out of `src/`

`src/labeler/events/notebooks.py` is 278 lines, of which `plot_original` is
189 across six `if event ==` branches. Each branch reads one category's
original annotation format and exists only to serve that category's
`example.ipynb`. The branches move into their own notebooks as plain cells.

`notebooks.py` keeps `load_shot` and `plot_shot`, which are generic over the
saved grid format and are what the new `verify.py` builds on.

## Dead package directories

`src/labelmaker/` and `src/ideate/` are the pre-R1 names. Both now contain
`__pycache__` and nothing else — zero `.py` files — and neither is tracked by
git. Both are deleted.

Stale references inside `data/events/**` are corrected: the module path
`labelmaker.events.*` becomes `labeler.events.*`, and `src/labelmaker/`
becomes `src/labeler/`.

Three things keep their old names on purpose, matching the R1 decision:

- the pixi environments `labelmaker` and `ideate` (pixi rejects underscores,
  so `shot_design` is not available as an environment name);
- the data roots `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker` and
  `.../nc1514/ideate`;
- `.meta.json` sidecars, which are provenance records of what produced a file.

The wider sweep — about 190 files, mostly historical run records under
`outputs/` — is a separate job and is not part of this work.

## Documentation

`data/events/README.md` gains a section covering the roster schema, the tier
rules, the review directory, and how to run a verification notebook. Each
category README links its `verification.ipynb` beside the existing
`example.ipynb` link.

## Testing

`tests/labeler/test_events_verify.py` covers, without a display or a kernel:

- `validate_shots` accepts the placeholder file and rejects each way tier and
  reviewer count can disagree, a duplicate shot, and a duplicate reviewer;
- the roster writer promotes `unverified` to `silver` to `gold` across two
  reviewers, and a repeated Verify by the same reviewer updates `verified_on`
  without promoting;
- corrections round-trip through `validate_intervals`;
- `corpus_signal` slices rather than loading, and raises `NoDataError` for an
  absent file and for the `(C, 1)` sentinel;
- the AE fallback raises its wrapper message when toksearch is unavailable.

Every `verification.ipynb` is checked to parse as JSON and to import cleanly;
none is executed in the suite, because execution needs the corpus.

## What this does not decide

Which shots become gold. How a review merges back into `format/`. Whether a
third reviewer's disagreement demotes a shot. Panel definitions for the
twelve categories on the generic fallback.
