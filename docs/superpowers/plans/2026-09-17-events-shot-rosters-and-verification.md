# Event shot rosters and verification notebooks — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every phenomenon under `data/events/` a shot-level review roster and a notebook that shows a reviewer the signals that decide whether the event is real, lets them correct the label intervals by dragging on the plot, and records who reviewed what.

**Architecture:** Two new modules under `src/labeler/events/`. `rosters.py` owns the `shots.csv` schema — pure pandas, no plotting, no HDF5. `verify.py` owns the review surface — it reads corpus signals, assembles a stacked `plotly.FigureWidget`, and writes corrections plus roster updates. Neither knows any plasma physics: the notebook computes the arrays and hands them over as `Panel` objects, so each category's panel definitions stay inline in its own `verification.ipynb` where they can be edited case by case.

**Tech Stack:** Python 3.11, pandas, numpy, h5py, scipy.signal, plotly 6.9 `FigureWidget` (needs `anywidget`), ipywidgets 8, nbformat 5.11, pytest, pixi.

## Global Constraints

- Run everything through pixi: `pixi run -e labelmaker <cmd>`. Never invoke `.pixi/envs/*/bin/python` directly.
- The installed package is `labeler` (renamed from `labelmaker`). The pixi *environment* is still named `labelmaker`, and the data roots are still `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker` and `.../nc1514/ideate`. Do not rename either.
- Data roots are read-only. `/scratch/gpfs/EKOLEMEN/foundation_model` (corpus) and `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/features` are never written to.
- Ruff, 88-character lines, four-space indent, `snake_case` functions, `PascalCase` classes.
- Times are milliseconds in every CSV and every plot axis. The corpus stores seconds on `xdata`; convert at the boundary.
- Commit after every task. Prefix subjects `labeler:`. End each commit message with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- Never execute a `verification.ipynb` in the test suite; execution needs the corpus.

## File Structure

| File | Responsibility |
| --- | --- |
| `src/labeler/events/rosters.py` | **Create.** The `shots.csv` schema: columns, tier rules, read/validate/write, recording one review. |
| `src/labeler/events/verify.py` | **Create.** Corpus signal reads, `Panel`, `ReviewSession`, `review()`, corrections I/O. |
| `src/labeler/events/notebooks.py` | **Modify.** Delete `plot_original` (lines 90–278). Keep `load_shot` and `plot_shot`. |
| `scripts/labeler/make_verification_notebook.py` | **Create.** Scaffolds a generic `verification.ipynb` for a category that has none. |
| `data/events/*/shots.csv` | **Create** (16). Placeholder roster, three example rows. |
| `data/events/*/verification.ipynb` | **Create** (16). Four hand-written, twelve scaffolded. |
| `data/events/fishbone/README.md` | **Create.** New category documentation. |
| `data/events/*/example.ipynb` | **Modify** (7). Inline the `plot_original` branch that category used. |
| `data/events/README.md` | **Modify.** Roster schema, tier rules, review workflow. |
| `pyproject.toml` | **Modify.** Declare `plotly` and `anywidget` for the labelmaker feature. |
| `tests/labeler/test_events_rosters.py` | **Create.** Roster schema and tier promotion. |
| `tests/labeler/test_events_verify.py` | **Create.** Corpus reads, corrections, notebook well-formedness. |

---

### Task 1: Declare the notebook plotting dependencies

`plotly` is present in the `labelmaker` environment only transitively — nothing in `pyproject.toml` asks for it — and `anywidget`, which `plotly.graph_objects.FigureWidget` imports at construction time, is absent entirely. Both get declared.

**Files:**
- Modify: `pyproject.toml:164-172` (the `[tool.pixi.feature.labelmaker.pypi-dependencies]` block)

**Interfaces:**
- Consumes: nothing.
- Produces: a `labelmaker` environment in which `plotly.graph_objects.FigureWidget()` constructs.

- [ ] **Step 1: Confirm the failure first**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -c "import plotly.graph_objects as go; go.FigureWidget()"
```

Expected: `ImportError: Please install anywidget to use the FigureWidget class`

- [ ] **Step 2: Add the two dependencies**

Append to the `[tool.pixi.feature.labelmaker.pypi-dependencies]` block in `pyproject.toml`, after the `torchvision` line:

```toml
# The verification notebooks are built on plotly's FigureWidget: a stacked
# figure whose box-select callback turns a dragged time range into a label
# correction. `plotly` currently reaches this environment only transitively,
# and `FigureWidget` imports `anywidget` at construction time, so a bare
# `import plotly` succeeding says nothing about whether the widget works.
# Declare both rather than depend on another package's requirements.
plotly = ">=6.9,<7"
anywidget = ">=0.9,<1"
```

- [ ] **Step 3: Install and verify**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi install -e labelmaker
pixi run -e labelmaker python -c "
import plotly.graph_objects as go
w = go.FigureWidget()
print('FigureWidget OK', type(w).__name__)
"
```

Expected: `FigureWidget OK FigureWidget`

If the solve fails on `protobuf`, read the comment block at `pyproject.toml:138-163` — that feature already pins `protobuf = "<7"` for exactly this reason. Do not remove the pin; report the error instead.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml pixi.lock
git commit -m "$(cat <<'EOF'
labeler: declare plotly and anywidget for the verification notebooks

FigureWidget imports anywidget at construction, and plotly reached the
labelmaker environment only transitively.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: The roster schema

**Files:**
- Create: `src/labeler/events/rosters.py`
- Test: `tests/labeler/test_events_rosters.py`

**Interfaces:**
- Consumes: `labeler.config.Paths`, `labeler.events.databases.DatabaseError`.
- Produces:
  - `ROSTER_COLUMNS: tuple[str, ...] = ("shot", "tier", "reviewers", "verified_on", "notes")`
  - `TIERS: tuple[str, ...] = ("unverified", "silver", "gold")`
  - `REVIEWER_SEPARATOR: str = ";"`
  - `tier_for(reviewers: Sequence[str]) -> str`
  - `split_reviewers(value) -> list[str]`
  - `roster_path(event: str, *, root: Path | None = None) -> Path`
  - `read_roster(path) -> pd.DataFrame`
  - `validate_roster(frame: pd.DataFrame) -> pd.DataFrame`
  - `write_roster(frame: pd.DataFrame, path) -> None`
  - `record_review(path, shot: int, reviewer: str, *, on: date | None = None, notes: str | None = None) -> pd.DataFrame`

- [ ] **Step 1: Write the failing tests**

Create `tests/labeler/test_events_rosters.py`:

```python
"""The shots.csv review roster: its schema, and what a review does to it."""

from datetime import date

import pandas as pd
import pytest

from labeler.events.databases import DatabaseError
from labeler.events.rosters import (
    ROSTER_COLUMNS,
    read_roster,
    record_review,
    tier_for,
    validate_roster,
    write_roster,
)


def _frame(rows):
    return pd.DataFrame(rows, columns=list(ROSTER_COLUMNS))


def test_tier_counts_reviewers_not_quality():
    assert tier_for([]) == "unverified"
    assert tier_for(["alice"]) == "silver"
    assert tier_for(["alice", "bob"]) == "gold"
    assert tier_for(["alice", "bob", "carol"]) == "gold"


def test_a_valid_roster_round_trips(tmp_path):
    path = tmp_path / "shots.csv"
    frame = _frame([
        [170815, "gold", "alice;bob", "2026-01-01", "retimed onset"],
        [178631, "silver", "alice", "2026-01-02", ""],
        [185945, "unverified", "", "", ""],
    ])
    write_roster(frame, path)
    got = read_roster(path)
    assert list(got.columns) == list(ROSTER_COLUMNS)
    assert got.shot.tolist() == [170815, 178631, 185945]
    assert got.tier.tolist() == ["gold", "silver", "unverified"]


def test_tier_must_agree_with_the_reviewer_count():
    with pytest.raises(DatabaseError, match="tier"):
        validate_roster(_frame([[1, "gold", "alice", "2026-01-01", ""]]))
    with pytest.raises(DatabaseError, match="tier"):
        validate_roster(_frame([[1, "silver", "alice;bob", "2026-01-01", ""]]))
    with pytest.raises(DatabaseError, match="tier"):
        validate_roster(_frame([[1, "unverified", "alice", "2026-01-01", ""]]))
    with pytest.raises(DatabaseError, match="tier"):
        validate_roster(_frame([[1, "silver", "", "", ""]]))


def test_an_unknown_tier_is_rejected():
    with pytest.raises(DatabaseError, match="tier"):
        validate_roster(_frame([[1, "bronze", "alice", "2026-01-01", ""]]))


def test_a_shot_appears_once():
    with pytest.raises(DatabaseError, match="duplicate shot"):
        validate_roster(_frame([
            [1, "silver", "alice", "2026-01-01", ""],
            [1, "silver", "bob", "2026-01-01", ""],
        ]))


def test_a_reviewer_appears_once_per_shot():
    with pytest.raises(DatabaseError, match="duplicate reviewer"):
        validate_roster(_frame([[1, "gold", "alice;alice", "2026-01-01", ""]]))


def test_a_verified_row_carries_a_date():
    with pytest.raises(DatabaseError, match="verified_on"):
        validate_roster(_frame([[1, "silver", "alice", "", ""]]))
    with pytest.raises(DatabaseError, match="verified_on"):
        validate_roster(_frame([[1, "unverified", "", "2026-01-01", ""]]))


def test_two_reviewers_promote_a_shot_to_gold(tmp_path):
    path = tmp_path / "shots.csv"
    write_roster(_frame([[170815, "unverified", "", "", ""]]), path)

    record_review(path, 170815, "alice", on=date(2026, 1, 1))
    row = read_roster(path).iloc[0]
    assert row.tier == "silver"
    assert row.reviewers == "alice"
    assert row.verified_on == "2026-01-01"

    record_review(path, 170815, "bob", on=date(2026, 1, 2))
    row = read_roster(path).iloc[0]
    assert row.tier == "gold"
    assert row.reviewers == "alice;bob"
    assert row.verified_on == "2026-01-02"


def test_the_same_reviewer_twice_redates_without_promoting(tmp_path):
    path = tmp_path / "shots.csv"
    write_roster(_frame([[170815, "silver", "alice", "2026-01-01", ""]]), path)
    record_review(path, 170815, "alice", on=date(2026, 3, 9))
    row = read_roster(path).iloc[0]
    assert row.tier == "silver"
    assert row.reviewers == "alice"
    assert row.verified_on == "2026-03-09"


def test_reviewing_an_unrostered_shot_adds_it(tmp_path):
    path = tmp_path / "shots.csv"
    write_roster(_frame([]), path)
    record_review(path, 199999, "alice", on=date(2026, 1, 1), notes="new")
    got = read_roster(path)
    assert got.shot.tolist() == [199999]
    assert got.tier.tolist() == ["silver"]
    assert got.notes.tolist() == ["new"]


def test_notes_stay_on_one_line(tmp_path):
    path = tmp_path / "shots.csv"
    write_roster(_frame([[1, "unverified", "", "", ""]]), path)
    with pytest.raises(DatabaseError, match="one line"):
        record_review(path, 1, "alice", on=date(2026, 1, 1), notes="two\nlines")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker pytest tests/labeler/test_events_rosters.py -q
```

Expected: collection error, `ModuleNotFoundError: No module named 'labeler.events.rosters'`

- [ ] **Step 3: Write the module**

Create `src/labeler/events/rosters.py`:

```python
"""The per-category review roster: who has looked at which shot.

`data/events/<category>/shots.csv` is a shot-level roster, not an interval
table. A shot enters it when somebody puts it up for review; the interval
tables under `format/` and `extend_*/` stay the record of what is labelled.

`tier` counts independent reviews and says nothing about label quality:
two or more reviewers is `gold`, one is `silver`, none is `unverified`. It is
redundant with `reviewers` by construction, and is written out anyway because
people read this file by eye. `validate_roster` is what keeps the two honest.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from ..config import Paths
from .databases import DatabaseError

ROSTER_COLUMNS = ("shot", "tier", "reviewers", "verified_on", "notes")
TIERS = ("unverified", "silver", "gold")
REVIEWER_SEPARATOR = ";"
ROSTER_NAME = "shots.csv"


def tier_for(reviewers: Sequence[str]) -> str:
    """The tier a shot has after this many independent reviews."""
    return TIERS[min(len(reviewers), 2)]


def split_reviewers(value) -> list[str]:
    """Reviewer ids from one cell, in the order they reviewed."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [part for part in str(value).split(REVIEWER_SEPARATOR) if part]


def roster_path(event: str, *, root: Path | None = None) -> Path:
    """The roster for one category, independently of the working directory."""
    root = Paths.from_env().label_tables if root is None else Path(root)
    return root / event / ROSTER_NAME


def validate_roster(frame: pd.DataFrame) -> pd.DataFrame:
    """Check the public roster schema; return it with `shot` as int64."""
    if tuple(frame.columns) != ROSTER_COLUMNS:
        raise DatabaseError(f"Expected columns {ROSTER_COLUMNS}")
    result = frame.copy()
    for column in ("tier", "reviewers", "verified_on", "notes"):
        result[column] = result[column].fillna("").astype(str)
    shots = pd.to_numeric(result["shot"], errors="coerce")
    if not (shots.notna() & (shots % 1 == 0) & (shots >= 0)).all():
        raise DatabaseError("shot must be a nonnegative integer")
    result["shot"] = shots.astype("int64")
    duplicated = result["shot"][result["shot"].duplicated()]
    if len(duplicated):
        raise DatabaseError(f"duplicate shot {int(duplicated.iloc[0])}")
    for row in result.itertuples():
        reviewers = split_reviewers(row.reviewers)
        if len(set(reviewers)) != len(reviewers):
            raise DatabaseError(f"duplicate reviewer on shot {row.shot}")
        if row.tier not in TIERS:
            raise DatabaseError(f"tier {row.tier!r} is not one of {TIERS}")
        if row.tier != tier_for(reviewers):
            raise DatabaseError(
                f"shot {row.shot}: tier {row.tier!r} disagrees with "
                f"{len(reviewers)} reviewer(s), which is {tier_for(reviewers)!r}"
            )
        if bool(reviewers) != bool(row.verified_on):
            raise DatabaseError(
                f"shot {row.shot}: verified_on and reviewers must agree"
            )
        if "\n" in row.notes:
            raise DatabaseError(f"shot {row.shot}: notes must be one line")
    return result


def read_roster(path) -> pd.DataFrame:
    """Read and validate one roster."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not len(frame.columns):
        frame = pd.DataFrame(columns=list(ROSTER_COLUMNS))
    return validate_roster(frame)


def write_roster(frame: pd.DataFrame, path) -> None:
    """Validate, sort by shot, and write one roster."""
    validated = validate_roster(frame).sort_values("shot", ignore_index=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(path, index=False)


def record_review(
    path,
    shot: int,
    reviewer: str,
    *,
    on: date | None = None,
    notes: str | None = None,
) -> pd.DataFrame:
    """Append one reviewer to a shot's row, promoting its tier.

    A reviewer who has already reviewed this shot re-dates the row and does
    not promote it: `gold` means two *independent* reviews. A shot not yet in
    the roster is added. Returns the roster as written.
    """
    frame = read_roster(path)
    on = date.today() if on is None else on
    rows = frame.index[frame["shot"] == int(shot)]
    if len(rows):
        index = rows[0]
    else:
        index = len(frame)
        frame.loc[index] = [int(shot), "unverified", "", "", ""]
    reviewers = split_reviewers(frame.at[index, "reviewers"])
    if reviewer not in reviewers:
        reviewers.append(reviewer)
    frame.at[index, "reviewers"] = REVIEWER_SEPARATOR.join(reviewers)
    frame.at[index, "tier"] = tier_for(reviewers)
    frame.at[index, "verified_on"] = on.isoformat()
    if notes is not None:
        frame.at[index, "notes"] = notes
    write_roster(frame, path)
    return read_roster(path)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker pytest tests/labeler/test_events_rosters.py -q
```

Expected: `11 passed`

- [ ] **Step 5: Lint**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker ruff check src/labeler/events/rosters.py tests/labeler/test_events_rosters.py
```

Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/labeler/events/rosters.py tests/labeler/test_events_rosters.py
git commit -m "$(cat <<'EOF'
labeler: the shots.csv review roster schema

gold/silver/unverified counts independent reviews. validate_roster keeps the
written tier honest against the reviewer list; record_review promotes.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Placeholder rosters in all sixteen categories

**Files:**
- Create: `data/events/<category>/shots.csv` × 16
- Modify: `tests/labeler/test_events_rosters.py` (append one test)

**Interfaces:**
- Consumes: `write_roster`, `read_roster`, `roster_path` from Task 2.
- Produces: a `shots.csv` in every category directory, each validating.

The sixteen categories are: `alfven_eigenmode`, `detachment`, `edge_localized_mode`, `fishbone`, `high_confinement_mode`, `improved_energy_confinement_mode`, `locked_mode`, `low_confinement_mode`, `minimum_safety_factor`, `neoclassical_tearing_mode`, `poloidal_beta`, `quiescent_high_confinement_mode`, `resistive_wall_mode`, `sawtooth_oscillation`, `vertical_displacement_event`, `wide_pedestal_quiescent_high_confinement_mode`.

Five of them already hold an empty (0-byte) `shots.csv`; those get overwritten.

- [ ] **Step 1: Write the failing test**

Append to `tests/labeler/test_events_rosters.py`:

```python
def test_every_category_has_a_valid_roster():
    from labeler.config import Paths
    from labeler.events.rosters import ROSTER_NAME, read_roster

    root = Paths.from_env().label_tables
    categories = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert len(categories) == 16
    for category in categories:
        path = root / category / ROSTER_NAME
        assert path.is_file(), f"{category} has no {ROSTER_NAME}"
        frame = read_roster(path)
        assert len(frame) == 3, f"{category} should ship three example rows"
        assert frame.tier.tolist() == ["gold", "silver", "unverified"]
```

- [ ] **Step 2: Run it to verify it fails**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker pytest tests/labeler/test_events_rosters.py::test_every_category_has_a_valid_roster -q
```

Expected: FAIL — `alfven_eigenmode has no shots.csv` is not the message; the empty files fail earlier in `read_roster` with a pandas `EmptyDataError`, or the count assertion fails. Either failure is the expected red.

- [ ] **Step 3: Write the sixteen rosters**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python - <<'PY'
from pathlib import Path

import pandas as pd

from labeler.config import Paths
from labeler.events.rosters import ROSTER_COLUMNS, ROSTER_NAME, write_roster

PLACEHOLDER = [
    [1, "gold", "alice;bob", "2026-01-01", "EXAMPLE - replace with a real shot"],
    [2, "silver", "alice", "2026-01-01", "EXAMPLE - replace with a real shot"],
    [3, "unverified", "", "", "EXAMPLE - replace with a real shot"],
]

root = Paths.from_env().label_tables
frame = pd.DataFrame(PLACEHOLDER, columns=list(ROSTER_COLUMNS))
for directory in sorted(p for p in root.iterdir() if p.is_dir()):
    write_roster(frame, directory / ROSTER_NAME)
    print("wrote", directory.name)
PY
```

Expected: sixteen `wrote <category>` lines.

Then check one by eye:

```bash
cat data/events/fishbone/shots.csv
```

Expected:

```csv
shot,tier,reviewers,verified_on,notes
1,gold,alice;bob,2026-01-01,EXAMPLE - replace with a real shot
2,silver,alice,2026-01-01,EXAMPLE - replace with a real shot
3,unverified,,,EXAMPLE - replace with a real shot
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker pytest tests/labeler/test_events_rosters.py -q
```

Expected: `12 passed`

- [ ] **Step 5: Commit**

```bash
git add data/events/*/shots.csv tests/labeler/test_events_rosters.py
git commit -m "$(cat <<'EOF'
labeler: placeholder shots.csv in all sixteen event categories

Three example rows each, one per tier, meant to be deleted once real shots
are picked.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Reading corpus signals for review

**Files:**
- Create: `src/labeler/events/verify.py`
- Test: `tests/labeler/test_events_verify.py`

**Interfaces:**
- Consumes: `labeler.config.Paths`, `labeler.features.store.FeatureArray`.
- Produces:
  - `class NoDataError(RuntimeError)`
  - `corpus_path(shot: int, *, corpus: Path | None = None) -> Path`
  - `corpus_signal(shot: int, group: str, *, channels: Sequence[int] | None = None, t_range: tuple[float, float] | None = None, corpus: Path | None = None) -> FeatureArray` — `x` is **milliseconds**, `y` is `(C, T)` float32.

`FeatureArray` is the existing `(x, y, attrs)` container from `labeler.features.store`; reusing it means anything that already reads a feature reads a review panel's input too.

- [ ] **Step 1: Write the failing tests**

Create `tests/labeler/test_events_verify.py`:

```python
"""Reading corpus signals for a review, and what a review writes."""

import h5py
import numpy as np
import pytest

from labeler.events.verify import NoDataError, corpus_signal


def _corpus(tmp_path, shot, groups):
    """One corpus-shaped file: `<shot>_processed.h5`, xdata seconds, ydata (C, T)."""
    path = tmp_path / f"{shot}_processed.h5"
    with h5py.File(path, "w") as f:
        for name, (x, y) in groups.items():
            group = f.create_group(name)
            group.create_dataset("xdata", data=np.asarray(x, dtype="float32"))
            group.create_dataset("ydata", data=np.asarray(y, dtype="float32"))
    return path


def test_a_signal_comes_back_in_milliseconds(tmp_path):
    seconds = np.linspace(0.0, 2.0, 2001)
    _corpus(tmp_path, 185601, {"ece": (seconds, np.zeros((48, 2001)))})
    got = corpus_signal(185601, "ece", corpus=tmp_path)
    assert got.y.shape == (48, 2001)
    assert got.x[0] == pytest.approx(0.0)
    assert got.x[-1] == pytest.approx(2000.0)


def test_channels_select_rows(tmp_path):
    seconds = np.linspace(0.0, 1.0, 101)
    values = np.arange(48 * 101, dtype="float32").reshape(48, 101)
    _corpus(tmp_path, 185601, {"ece": (seconds, values)})
    got = corpus_signal(185601, "ece", channels=[20, 24, 28], corpus=tmp_path)
    assert got.y.shape == (3, 101)
    assert got.y[0] == pytest.approx(values[20])
    assert got.attrs["channels"] == "20,24,28"


def test_a_time_range_slices_rather_than_loading(tmp_path):
    seconds = np.linspace(0.0, 6.0, 6001)
    _corpus(tmp_path, 185601, {"ece": (seconds, np.zeros((48, 6001)))})
    got = corpus_signal(185601, "ece", t_range=(2000.0, 2100.0), corpus=tmp_path)
    assert got.x[0] >= 2000.0
    assert got.x[-1] <= 2100.0
    assert got.y.shape[1] == got.x.shape[0] < 200


def test_a_missing_file_names_the_corpus_span(tmp_path):
    with pytest.raises(NoDataError, match="170815"):
        corpus_signal(170815, "co2", corpus=tmp_path)


def test_the_absent_signal_sentinel_is_not_data(tmp_path):
    _corpus(tmp_path, 185601, {"co2": (np.zeros(1), np.zeros((4, 1)))})
    with pytest.raises(NoDataError, match="co2"):
        corpus_signal(185601, "co2", corpus=tmp_path)


def test_a_missing_group_is_named(tmp_path):
    _corpus(tmp_path, 185601, {"ece": (np.zeros(10), np.zeros((48, 10)))})
    with pytest.raises(NoDataError, match="mhr"):
        corpus_signal(185601, "mhr", corpus=tmp_path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker pytest tests/labeler/test_events_verify.py -q
```

Expected: collection error, `ModuleNotFoundError: No module named 'labeler.events.verify'`

- [ ] **Step 3: Write the module**

Create `src/labeler/events/verify.py`:

```python
"""The human review surface: show a shot's signals, take back corrections.

This module knows nothing about plasma physics. A notebook computes whatever
arrays settle its own phenomenon - a spectrogram, a set of raw channels, a
scalar trace - and hands them over as `Panel`s. Deciding which traces settle
which phenomenon is case-by-case work that does not generalise, so it lives
in each category's `verification.ipynb` rather than here.

What is shared is everything else: reading the corpus without loading it,
stacking the panels on one time axis, turning a dragged range into an
interval, and writing the two files a review produces - the corrections under
`review/` and the roster row in `shots.csv`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..config import Paths
from ..features.store import FeatureArray

#: A corpus group whose `ydata` is this narrow carries the absent-signal
#: sentinel - `(C, 1)` - rather than a record. `resolve_corpus` writes it for
#: a diagnostic that did not run, and it is not data.
SENTINEL_WIDTH = 1


class NoDataError(RuntimeError):
    """A signal a panel asked for is not on disk for this shot."""


def corpus_path(shot: int, *, corpus: Path | None = None) -> Path:
    """The corpus file for one shot."""
    corpus = Paths.from_env().corpus if corpus is None else Path(corpus)
    return corpus / f"{int(shot)}_processed.h5"


def corpus_signal(
    shot: int,
    group: str,
    *,
    channels: Sequence[int] | None = None,
    t_range: tuple[float, float] | None = None,
    corpus: Path | None = None,
) -> FeatureArray:
    """One corpus group, sliced: `x` in milliseconds, `y` as `(C, T)` float32.

    ECE is `(48, 3.1e6)` and CO2 `(4, 4.5e6)`; neither is ever read whole.
    `t_range` is milliseconds and is applied with an h5py slice, so a 100 ms
    window costs a 100 ms read. `channels` selects rows by index.
    """
    import h5py

    path = corpus_path(shot, corpus=corpus)
    if not path.is_file():
        raise NoDataError(
            f"shot {int(shot)} has no corpus file at {path}. The corpus covers "
            f"185601-204999; a shot outside it has to be fetched."
        )
    with h5py.File(path, "r") as f:
        if group not in f:
            raise NoDataError(f"shot {int(shot)} has no {group!r} group in {path}")
        x = f[group]["xdata"]
        y = f[group]["ydata"]
        if y.shape[-1] <= SENTINEL_WIDTH:
            raise NoDataError(
                f"shot {int(shot)} carries the absent-signal sentinel for "
                f"{group!r}: ydata is {y.shape}, so this diagnostic did not run"
            )
        start, stop = 0, x.shape[0]
        if t_range is not None:
            times = np.asarray(x, dtype="float64") * 1000.0
            start = int(np.searchsorted(times, t_range[0], side="left"))
            stop = int(np.searchsorted(times, t_range[1], side="right"))
            if stop <= start:
                raise NoDataError(
                    f"shot {int(shot)} {group!r} has no samples in "
                    f"{t_range[0]}-{t_range[1]} ms"
                )
        rows = list(range(y.shape[0])) if channels is None else list(channels)
        values = np.stack([y[row, start:stop] for row in rows]).astype("float32")
        times_ms = np.asarray(x[start:stop], dtype="float64") * 1000.0
    return FeatureArray(
        x=times_ms,
        y=values,
        attrs={
            "group": group,
            "shot": str(int(shot)),
            "channels": ",".join(str(row) for row in rows),
            "units": "ms",
        },
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker pytest tests/labeler/test_events_verify.py -q
```

Expected: `6 passed`

- [ ] **Step 5: Lint and commit**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker ruff check src/labeler/events/verify.py tests/labeler/test_events_verify.py
git add src/labeler/events/verify.py tests/labeler/test_events_verify.py
git commit -m "$(cat <<'EOF'
labeler: corpus_signal, the read behind every verification panel

Slices rather than loads - ECE is (48, 3.1e6) - and refuses the (C, 1)
absent-signal sentinel instead of plotting it as a record.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Corrections and the review session

**Files:**
- Modify: `src/labeler/events/verify.py`
- Modify: `tests/labeler/test_events_verify.py`

**Interfaces:**
- Consumes: `corpus_signal`, `NoDataError` (Task 4); `record_review`, `roster_path` (Task 2); `labeler.events.interval_tables.{validate_intervals, INTERVAL_COLUMNS, read_label_grid}`; `labeler.events.notebooks.load_shot`.
- Produces:
  - `@dataclass Panel(title: str, x: np.ndarray, y: np.ndarray, kind: str = "line", z: np.ndarray | None = None, ylabel: str = "", legend: Sequence[str] | None = None, bands: Sequence[tuple[float, float]] = ())` — `kind` is `"line"` or `"heatmap"`; for a heatmap `y` is the vertical axis and `z` is `(len(y), len(x))`.
  - `review_path(event: str, shot: int, *, root: Path | None = None) -> Path` — `data/events/<event>/review/<shot>.csv`
  - `read_corrections(path) -> pd.DataFrame`
  - `write_corrections(frame: pd.DataFrame, path) -> None`
  - `class ReviewSession` with `.event`, `.shot`, `.corrections -> pd.DataFrame`, `.mark(t_start, t_end, category)`, `.verify(*, notes=None)` (the reviewer is fixed at construction; `ReviewSession(...)` is keyword-only), `.save()`, `.figure`, `.controls`, `._ipython_display_()`
  - `review(event: str, shot: int, panels: Sequence[Panel], *, source: str = "format/shots", root: Path | None = None, reviewer: str | None = None) -> ReviewSession`

- [ ] **Step 1: Write the failing tests**

Append the test functions below to `tests/labeler/test_events_verify.py`. Their
imports go into the **existing import block at the top of the file**, not above
the new functions — ruff's E402 rejects a module-level import that follows a
definition.

Add to the top of the file:

```python
import pandas as pd

from labeler.events.interval_tables import INTERVAL_COLUMNS
from labeler.events.rosters import ROSTER_COLUMNS, read_roster, write_roster
from labeler.events.verify import (
    Panel,
    ReviewSession,
    read_corrections,
    review_path,
    write_corrections,
)
```

Then append the tests:

```python
def _roster(root, event):
    path = root / event / "shots.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_roster(
        pd.DataFrame(
            [[185601, "unverified", "", "", ""]], columns=list(ROSTER_COLUMNS)
        ),
        path,
    )
    return path


def test_corrections_round_trip_through_the_interval_schema(tmp_path):
    path = tmp_path / "review" / "185601.csv"
    frame = pd.DataFrame(
        [[185601, 1, 1200.0, 1450.0, ""], [185601, 0, 1450.0, 1600.0, ""]],
        columns=list(INTERVAL_COLUMNS),
    )
    write_corrections(frame, path)
    got = read_corrections(path)
    assert list(got.columns) == list(INTERVAL_COLUMNS)
    assert got.t_start.tolist() == [1200.0, 1450.0]
    assert got.category.tolist() == [1, 0]


def test_review_path_is_under_the_category(tmp_path):
    got = review_path("fishbone", 185601, root=tmp_path)
    assert got == tmp_path / "fishbone" / "review" / "185601.csv"


def test_marking_a_range_then_saving_writes_both_files(tmp_path):
    _roster(tmp_path, "fishbone")
    session = ReviewSession(
        event="fishbone", shot=185601, panels=[], root=tmp_path, reviewer="alice"
    )
    session.mark(1200.0, 1450.0, category=1)
    session.mark(1450.0, 1600.0, category=0)
    session.verify()
    session.save()

    corrections = read_corrections(review_path("fishbone", 185601, root=tmp_path))
    assert len(corrections) == 2
    assert corrections.shot.tolist() == [185601, 185601]

    roster = read_roster(tmp_path / "fishbone" / "shots.csv")
    assert roster.iloc[0].tier == "silver"
    assert roster.iloc[0].reviewers == "alice"


def test_nothing_is_written_before_save(tmp_path):
    _roster(tmp_path, "fishbone")
    session = ReviewSession(
        event="fishbone", shot=185601, panels=[], root=tmp_path, reviewer="alice"
    )
    session.mark(1200.0, 1450.0, category=1)
    assert not review_path("fishbone", 185601, root=tmp_path).exists()
    assert read_roster(tmp_path / "fishbone" / "shots.csv").iloc[0].tier == "unverified"


def test_a_backwards_range_is_refused(tmp_path):
    _roster(tmp_path, "fishbone")
    session = ReviewSession(
        event="fishbone", shot=185601, panels=[], root=tmp_path, reviewer="alice"
    )
    with pytest.raises(ValueError, match="t_end"):
        session.mark(1450.0, 1200.0, category=1)


def test_a_figure_stacks_one_row_per_panel(tmp_path):
    _roster(tmp_path, "fishbone")
    panels = [
        Panel(title="mhr B1", x=np.arange(10.0), y=np.zeros((1, 10)), ylabel="T/s"),
        Panel(
            title="spectrogram",
            kind="heatmap",
            x=np.arange(10.0),
            y=np.arange(5.0),
            z=np.zeros((5, 10)),
            ylabel="kHz",
        ),
    ]
    session = ReviewSession(
        event="fishbone", shot=185601, panels=panels, root=tmp_path, reviewer="alice"
    )
    assert len(session.figure.data) >= 2
    assert "185601" in session.figure.layout.title.text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker pytest tests/labeler/test_events_verify.py -q
```

Expected: FAIL with `ImportError: cannot import name 'Panel' from 'labeler.events.verify'`

- [ ] **Step 3: Extend the module**

Append to `src/labeler/events/verify.py`:

```python
REVIEW_DIRECTORY = "review"


@dataclass
class Panel:
    """One row of a review figure, already computed by the notebook.

    `kind="line"` draws each row of `y` against `x`. `kind="heatmap"` draws
    `z`, shaped `(len(y), len(x))`, with `y` as the vertical axis - which is
    how a spectrogram arrives. `bands` shades horizontal regions of interest,
    such as the 80-250 kHz AE band.
    """

    title: str
    x: np.ndarray
    y: np.ndarray
    kind: str = "line"
    z: np.ndarray | None = None
    ylabel: str = ""
    legend: Sequence[str] | None = None
    bands: Sequence[tuple[float, float]] = ()


def review_path(event: str, shot: int, *, root: Path | None = None) -> Path:
    """Where one shot's corrections live."""
    root = Paths.from_env().label_tables if root is None else Path(root)
    return root / event / REVIEW_DIRECTORY / f"{int(shot)}.csv"


def read_corrections(path):
    """Read one shot's corrections through the public interval schema."""
    import pandas as pd

    from .interval_tables import validate_intervals

    return validate_intervals(pd.read_csv(path, keep_default_na=False))


def write_corrections(frame, path) -> None:
    """Validate and write one shot's corrections."""
    from .interval_tables import validate_intervals

    validated = validate_intervals(frame)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(path, index=False)


class ReviewSession:
    """One reviewer, one shot: the figure, the marks, and what gets written.

    Marks accumulate in memory. `save()` is the only thing that touches disk,
    and it writes two files: the corrections under `review/` and the roster
    row in `shots.csv`. `verify()` records the intent to promote; without it
    `save()` writes corrections alone, which is what a half-finished review
    should leave behind.
    """

    def __init__(
        self,
        *,
        event: str,
        shot: int,
        panels: Sequence[Panel],
        root: Path | None = None,
        reviewer: str | None = None,
        source: str = "format/shots",
    ) -> None:
        import os

        self.event = event
        self.shot = int(shot)
        self.panels = list(panels)
        self.source = source
        self.root = Paths.from_env().label_tables if root is None else Path(root)
        self.reviewer = reviewer or os.environ.get("USER", "unknown")
        self._marks: list[tuple[float, float, int]] = []
        self._verify = False
        self._notes: str | None = None
        self.figure = self._build_figure()
        self.controls = self._build_controls()

    @property
    def corrections(self):
        import pandas as pd

        from .interval_tables import INTERVAL_COLUMNS

        return pd.DataFrame(
            [
                [self.shot, category, t_start, t_end, ""]
                for t_start, t_end, category in self._marks
            ],
            columns=list(INTERVAL_COLUMNS),
        )

    def mark(self, t_start: float, t_end: float, category: int = 1) -> None:
        """Record one corrected interval, in milliseconds."""
        if t_end < t_start:
            raise ValueError(f"t_end {t_end} precedes t_start {t_start}")
        self._marks.append((float(t_start), float(t_end), int(category)))

    def verify(self, *, notes: str | None = None) -> None:
        """Say this reviewer has looked; `save()` then promotes the tier."""
        self._verify = True
        self._notes = notes

    def save(self) -> None:
        """Write the corrections, and the roster row when verified."""
        from .rosters import record_review, roster_path

        if self._marks:
            write_corrections(
                self.corrections, review_path(self.event, self.shot, root=self.root)
            )
        if self._verify:
            record_review(
                roster_path(self.event, root=self.root),
                self.shot,
                self.reviewer,
                notes=self._notes,
            )

    def _build_figure(self):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        rows = max(len(self.panels), 1)
        figure = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=[panel.title for panel in self.panels] or [""],
        )
        for index, panel in enumerate(self.panels, start=1):
            if panel.kind == "heatmap":
                figure.add_trace(
                    go.Heatmap(x=panel.x, y=panel.y, z=panel.z, showscale=False),
                    row=index,
                    col=1,
                )
            else:
                names = panel.legend or [f"ch {i}" for i in range(len(panel.y))]
                for values, name in zip(panel.y, names, strict=False):
                    figure.add_trace(
                        go.Scattergl(x=panel.x, y=values, name=name, mode="lines"),
                        row=index,
                        col=1,
                    )
            for low, high in panel.bands:
                figure.add_hrect(
                    y0=low, y1=high, line_width=0, fillcolor="black", opacity=0.08,
                    row=index, col=1,
                )
            figure.update_yaxes(title_text=panel.ylabel, row=index, col=1)
        figure.update_xaxes(title_text="Time (ms)", row=rows, col=1)
        figure.update_layout(
            title=f"{self.event} - shot {self.shot} ({self.source})",
            height=200 * rows + 120,
            dragmode="select",
            selectdirection="h",
            showlegend=False,
            margin=dict(l=60, r=20, t=60, b=40),
        )
        return go.FigureWidget(figure)

    def _build_controls(self):
        import ipywidgets as widgets

        present = widgets.Button(description="Mark present", button_style="primary")
        absent = widgets.Button(description="Mark absent")
        verify = widgets.Button(description="Verify", button_style="success")
        save = widgets.Button(description="Save", button_style="warning")
        status = widgets.HTML(value=self._status())

        def selected() -> tuple[float, float]:
            selection = getattr(self.figure.layout, "selections", None)
            if not selection:
                raise ValueError("drag a time range on the figure first")
            box = selection[0]
            return float(min(box.x0, box.x1)), float(max(box.x0, box.x1))

        def on_mark(category):
            def handler(_):
                try:
                    t_start, t_end = selected()
                    self.mark(t_start, t_end, category)
                except ValueError as error:
                    status.value = f"<b style='color:#b2182b'>{error}</b>"
                    return
                status.value = self._status()

            return handler

        def on_verify(_):
            self.verify()
            status.value = self._status()

        def on_save(_):
            self.save()
            status.value = self._status() + " <b>saved</b>"

        present.on_click(on_mark(1))
        absent.on_click(on_mark(0))
        verify.on_click(on_verify)
        save.on_click(on_save)
        return widgets.VBox(
            [widgets.HBox([present, absent, verify, save]), status]
        )

    def _status(self) -> str:
        verified = "verified" if self._verify else "not verified"
        return (
            f"{len(self._marks)} correction(s), {verified}, "
            f"reviewer <code>{self.reviewer}</code>"
        )

    def _ipython_display_(self):
        from IPython.display import display

        display(self.figure, self.controls)


def review(
    event: str,
    shot: int,
    panels: Sequence[Panel],
    *,
    source: str = "format/shots",
    root: Path | None = None,
    reviewer: str | None = None,
) -> ReviewSession:
    """Open a review of one shot, with the label row appended to the panels."""
    panels = list(panels)
    try:
        panels.append(label_panel(event, shot, source=source, root=root))
    except (FileNotFoundError, OSError):
        pass
    return ReviewSession(
        event=event,
        shot=shot,
        panels=panels,
        root=root,
        reviewer=reviewer,
        source=source,
    )


def label_panel(
    event: str, shot: int, *, source: str = "format/shots", root: Path | None = None
) -> Panel:
    """The saved label grid as a heatmap row, unknown cells left as NaN."""
    from .notebooks import load_shot

    grid = load_shot(event, shot, source=source, root=root)
    return Panel(
        title=f"labels ({source})",
        kind="heatmap",
        x=grid["time_ms"],
        y=grid["rho_edges"][:-1],
        z=grid["label"].T,
        ylabel="rho",
    )
```

Add to the module's imports at the top:

```python
from dataclasses import dataclass
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker pytest tests/labeler/test_events_verify.py -q
```

Expected: `12 passed`

- [ ] **Step 5: Lint and commit**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker ruff check src/labeler/events/verify.py tests/labeler/test_events_verify.py
git add src/labeler/events/verify.py tests/labeler/test_events_verify.py
git commit -m "$(cat <<'EOF'
labeler: ReviewSession - stacked panels, dragged ranges, two files on save

Marks accumulate in memory; save() is the only thing that touches disk. A
review without verify() leaves corrections alone, which is what a
half-finished one should leave behind.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: The generic notebook scaffold

**Files:**
- Create: `scripts/labeler/make_verification_notebook.py`
- Create: `data/events/<category>/verification.ipynb` × 12 (the categories with no hand-written panels)
- Modify: `tests/labeler/test_events_verify.py`

**Interfaces:**
- Consumes: `nbformat`, `labeler.config.Paths`.
- Produces: `build(event: str) -> nbformat.NotebookNode` and a CLI writing `data/events/<event>/verification.ipynb`; the script refuses to overwrite an existing notebook unless `--force` is given.

The twelve scaffolded categories: `detachment`, `edge_localized_mode`, `high_confinement_mode`, `improved_energy_confinement_mode`, `locked_mode`, `low_confinement_mode`, `neoclassical_tearing_mode`, `poloidal_beta`, `quiescent_high_confinement_mode`, `resistive_wall_mode`, `vertical_displacement_event`, `wide_pedestal_quiescent_high_confinement_mode`.

- [ ] **Step 1: Write the failing test**

Append to `tests/labeler/test_events_verify.py`:

```python
def test_every_category_has_a_verification_notebook():
    import json

    from labeler.config import Paths

    root = Paths.from_env().label_tables
    categories = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert len(categories) == 16
    for category in categories:
        path = root / category / "verification.ipynb"
        assert path.is_file(), f"{category} has no verification.ipynb"
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4
        sources = "".join(
            "".join(cell["source"]) for cell in notebook["cells"]
        )
        assert "from labeler.events.verify import" in sources, category
        # The kernel is still called "Python (FAITH labelmaker)" on purpose;
        # what must not survive the rename is the MODULE path.
        assert "labelmaker.events" not in sources, category
        assert "from labelmaker" not in sources, category
```

- [ ] **Step 2: Run it to verify it fails**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker pytest tests/labeler/test_events_verify.py::test_every_category_has_a_verification_notebook -q
```

Expected: FAIL — `alfven_eigenmode has no verification.ipynb`

- [ ] **Step 3: Write the scaffold script**

Create `scripts/labeler/make_verification_notebook.py`:

```python
"""Scaffold a category's verification.ipynb.

The generic notebook plots `ip`, `betan` and `pinj_total` from the feature store
against the saved labels. That is enough to confirm a shot exists and that its
labels sit inside the discharge, and nowhere near enough to verify a
phenomenon - which is the point. Whoever takes a category on replaces the
panel cell with the traces that actually settle it, the way
`minimum_safety_factor`, `sawtooth_oscillation`, `alfven_eigenmode` and
`fishbone` already have.

    pixi run -e labelmaker python scripts/labeler/make_verification_notebook.py \
        --event detachment
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat

from labeler.config import Paths

HEADER = """# {title} - verification

Use the **Python (FAITH labelmaker)** kernel. Set `shot` below, drag a time
range on any panel, then press *Mark present* / *Mark absent*, *Verify* and
*Save*.

**These are generic panels.** `ip`, `betan` and `pinj_total` show that the shot
exists and that its labels sit inside the discharge. They do not show whether
{title} actually happened. Replace the panel cell below with the traces that
settle this phenomenon, then delete this paragraph.

*Save* writes two files: the corrected intervals to
`review/<shot>.csv`, and your review to `shots.csv`, which promotes the shot
to `silver` (one reviewer) or `gold` (two).
"""

SETUP = """%load_ext autoreload
%autoreload 2"""

PANELS = '''import numpy as np

from labeler.config import Paths
from labeler.features.store import read_feature
from labeler.events.verify import Panel, review

event = "{event}"
shot = 1  # replace with a shot from shots.csv
source = "format/shots"

features = Paths.from_env().features / f"{{shot}}_features.h5"

def trace(name, ylabel):
    """One scalar feature as a panel; seconds on disk, milliseconds on the plot."""
    array = read_feature(features, name)
    return Panel(title=name, x=array.x * 1000.0, y=array.y, ylabel=ylabel)

panels = [
    trace("ip", "A"),
    trace("betan", ""),
    trace("pinj_total", "kW"),
]'''

REVIEW = """session = review(event, shot, panels, source=source)
session"""

FOOTER = """After pressing *Save*, check what was written:

```python
from labeler.events.verify import read_corrections, review_path
read_corrections(review_path(event, shot))
```
"""


def build(event: str) -> nbformat.NotebookNode:
    """The generic verification notebook for one category."""
    title = event.replace("_", " ")
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell(HEADER.format(title=title)),
        nbformat.v4.new_code_cell(SETUP),
        nbformat.v4.new_code_cell(PANELS.format(event=event)),
        nbformat.v4.new_code_cell(REVIEW),
        nbformat.v4.new_markdown_cell(FOOTER),
    ]
    notebook.metadata["kernelspec"] = {
        "display_name": "Python (FAITH labelmaker)",
        "language": "python",
        "name": "faith-labelmaker",
    }
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()

    root = Paths.from_env().label_tables if args.root is None else args.root
    path = root / args.event / "verification.ipynb"
    if path.exists() and not args.force:
        raise SystemExit(f"{path} exists; pass --force to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build(args.event), path)
    print("wrote", path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Generate the twelve generic notebooks**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
for event in detachment edge_localized_mode high_confinement_mode \
             improved_energy_confinement_mode locked_mode low_confinement_mode \
             neoclassical_tearing_mode poloidal_beta \
             quiescent_high_confinement_mode resistive_wall_mode \
             vertical_displacement_event \
             wide_pedestal_quiescent_high_confinement_mode; do
  pixi run -e labelmaker python scripts/labeler/make_verification_notebook.py --event "$event"
done
```

Expected: twelve `wrote data/events/<event>/verification.ipynb` lines.

- [ ] **Step 5: Confirm the test still fails, for the right reason**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker pytest tests/labeler/test_events_verify.py::test_every_category_has_a_verification_notebook -q
```

Expected: FAIL — `alfven_eigenmode has no verification.ipynb`. The four hand-written notebooks come in Tasks 7–10; this test goes green at the end of Task 10.

- [ ] **Step 6: Commit**

```bash
git add scripts/labeler/make_verification_notebook.py data/events/*/verification.ipynb tests/labeler/test_events_verify.py
git commit -m "$(cat <<'EOF'
labeler: generic verification notebook scaffold, and twelve of them

The generic panels show a shot exists, not that a phenomenon happened. Each
notebook says so and asks to be replaced.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: `minimum_safety_factor` verification notebook

**Files:**
- Create: `data/events/minimum_safety_factor/verification.ipynb`

**Interfaces:**
- Consumes: `Panel`, `review` (Task 5); `labeler.features.store.read_feature`.
- Produces: nothing other tasks read.

All 337 rostered q-min shots have `qmin` and `qpsi` in the feature store, so this notebook reads the feature store and never the corpus.

- [ ] **Step 1: Write the notebook**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python - <<'PY'
import nbformat

from labeler.config import Paths

HEADER = """# Minimum safety factor - verification

Use the **Python (FAITH labelmaker)** kernel. Set `shot`, drag a time range on
any panel, then press *Mark present* / *Mark absent*, *Verify* and *Save*.

`qmin` is the offline EFIT01 `aeqdsk:qmin` standing in for `qmin_EFITRT2`, and
every shot in this category's roster has it. The classes are the `qmin_rule`
ones: 0 absent, 1 low, 2 hybrid, 3 elevated, 4 high, with thresholds at
qmin > 0.95 (hybrid), > 1.5 (elevated) and > 2 (high) - the three dashed lines
on the first panel.

What to check: that the flat-top is where the rule says it is, and that a
class change sits on a real qmin crossing rather than on an EFIT excursion of
one or two frames.
"""

PANELS = '''import numpy as np

from labeler.config import Paths
from labeler.features.store import read_feature
from labeler.events.verify import Panel, review

event = "minimum_safety_factor"
shot = 1  # replace with a shot from shots.csv
source = "extend_qmin_rule/recommender_v1"

features = Paths.from_env().features / f"{shot}_features.h5"

qmin = read_feature(features, "qmin")
qpsi = read_feature(features, "qpsi")
ip = read_feature(features, "ip")

panels = [
    Panel(
        title="qmin (EFIT01 aeqdsk)",
        x=qmin.x * 1000.0,
        y=qmin.y,
        ylabel="q",
        # the rule's class thresholds, as three thin shaded lines
        bands=[(0.95, 0.96), (1.5, 1.51), (2.0, 2.01)],
    ),
    Panel(
        title="q profile",
        kind="heatmap",
        x=qpsi.x * 1000.0,
        y=np.linspace(0.0, 1.0, qpsi.y.shape[0]),
        z=qpsi.y,
        ylabel="rho",
    ),
    Panel(title="ip", x=ip.x * 1000.0, y=ip.y, ylabel="A"),
]'''

REVIEW = """session = review(event, shot, panels, source=source)
session"""

FOOTER = """`category` on a correction is the q-min class, not a binary flag:
0 absent, 1 low, 2 hybrid, 3 elevated, 4 high. *Mark present* writes 1, so for
any other class call `session.mark(t_start, t_end, category=3)` directly.

```python
from labeler.events.verify import read_corrections, review_path
read_corrections(review_path(event, shot))
```
"""

notebook = nbformat.v4.new_notebook()
notebook.cells = [
    nbformat.v4.new_markdown_cell(HEADER),
    nbformat.v4.new_code_cell("%load_ext autoreload\n%autoreload 2"),
    nbformat.v4.new_code_cell(PANELS),
    nbformat.v4.new_code_cell(REVIEW),
    nbformat.v4.new_markdown_cell(FOOTER),
]
notebook.metadata["kernelspec"] = {
    "display_name": "Python (FAITH labelmaker)",
    "language": "python",
    "name": "faith-labelmaker",
}
path = Paths.from_env().label_tables / "minimum_safety_factor" / "verification.ipynb"
nbformat.write(notebook, path)
print("wrote", path)
PY
```

Expected: `wrote .../minimum_safety_factor/verification.ipynb`

- [ ] **Step 2: Check it parses and its panel code compiles**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python - <<'PY'
import json

path = "data/events/minimum_safety_factor/verification.ipynb"
notebook = json.loads(open(path).read())
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if source.startswith("%"):
            continue
        compile(source, path, "exec")
print("ok:", len(notebook["cells"]), "cells")
PY
```

Expected: `ok: 5 cells`

- [ ] **Step 3: Commit**

```bash
git add data/events/minimum_safety_factor/verification.ipynb
git commit -m "$(cat <<'EOF'
labeler: minimum_safety_factor verification notebook

qmin against the rule's three class thresholds, the q profile, and ip. Reads
the feature store, which covers every rostered shot.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: `sawtooth_oscillation` verification notebook

**Files:**
- Create: `data/events/sawtooth_oscillation/verification.ipynb`

**Interfaces:**
- Consumes: `Panel`, `review`, `corpus_signal` (Tasks 4–5).
- Produces: nothing other tasks read.

This notebook must read the **corpus** `ece` group, not the feature store's `ece`. The feature is decimated to 1 ms; a sawtooth crash is 100 µs to 1 ms, so the decimated record cannot show one. The corpus group is `(48, ~3.1e6)` at roughly 500 kHz, which is why every read is windowed.

- [ ] **Step 1: Write the notebook**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python - <<'PY'
import nbformat

from labeler.config import Paths

HEADER = """# Sawtooth oscillation - verification

Use the **Python (FAITH labelmaker)** kernel. Set `shot` and `window`, drag a
time range on any panel, then press *Mark present* / *Mark absent*, *Verify*
and *Save*.

Raw ECE, four channels to a row, channels 20-36. A sawtooth crash is a
simultaneous **drop** in the core channels and **rise** in the outer ones; the
row where the sign flips brackets the inversion radius, which is the q = 1
surface, typically rho ~ 0.3-0.6 on DIII-D.

This reads the corpus `ece` group at its native rate, **not** the feature
store's `ece`, which is decimated to 1 ms and cannot resolve a crash. The
group is `(48, ~3.1e6)`, so always keep `window` to a few hundred
milliseconds - `corpus_signal` slices, and a whole-shot read is 600 MB.
"""

PANELS = '''from labeler.events.verify import Panel, corpus_signal, review

event = "sawtooth_oscillation"
shot = 1        # replace with a shot from shots.csv
window = (2000.0, 2300.0)   # milliseconds; keep it short, this is 500 kHz data
source = "format/shots"

# Channels 20-36 in four groups of four. Overplotted within a row so the
# inversion shows up as the phase flip between rows.
groups = [range(20, 24), range(24, 28), range(28, 32), range(32, 36)]

panels = []
for channels in groups:
    signal = corpus_signal(shot, "ece", channels=list(channels), t_range=window)
    panels.append(
        Panel(
            title=f"ece ch {channels.start}-{channels.stop - 1}",
            x=signal.x,
            y=signal.y,
            ylabel="V",
            legend=[f"ch {c}" for c in channels],
        )
    )'''

REVIEW = """session = review(event, shot, panels, source=source)
session"""

FOOTER = """A crash is a point event: mark it with an equal start and end, which
the interval schema stores as a point.

```python
session.mark(2143.0, 2143.0, category=1)
```

The rule detector this category ships, `ece_sawtooth`, also records which
channels took part, in `attrs["inversion_channel_lo"]` and
`attrs["inversion_channel_stop"]`. If your eye disagrees with the rule about
the inversion channels, put that in the roster `notes`.
"""

notebook = nbformat.v4.new_notebook()
notebook.cells = [
    nbformat.v4.new_markdown_cell(HEADER),
    nbformat.v4.new_code_cell("%load_ext autoreload\n%autoreload 2"),
    nbformat.v4.new_code_cell(PANELS),
    nbformat.v4.new_code_cell(REVIEW),
    nbformat.v4.new_markdown_cell(FOOTER),
]
notebook.metadata["kernelspec"] = {
    "display_name": "Python (FAITH labelmaker)",
    "language": "python",
    "name": "faith-labelmaker",
}
path = Paths.from_env().label_tables / "sawtooth_oscillation" / "verification.ipynb"
nbformat.write(notebook, path)
print("wrote", path)
PY
```

Expected: `wrote .../sawtooth_oscillation/verification.ipynb`

- [ ] **Step 2: Check it parses and compiles**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python - <<'PY'
import json

path = "data/events/sawtooth_oscillation/verification.ipynb"
notebook = json.loads(open(path).read())
for cell in notebook["cells"]:
    if cell["cell_type"] == "code" and not "".join(cell["source"]).startswith("%"):
        compile("".join(cell["source"]), path, "exec")
print("ok:", len(notebook["cells"]), "cells")
PY
```

Expected: `ok: 5 cells`

- [ ] **Step 3: Commit**

```bash
git add data/events/sawtooth_oscillation/verification.ipynb
git commit -m "$(cat <<'EOF'
labeler: sawtooth_oscillation verification notebook

Raw corpus ECE, channels 20-36 in four rows, windowed. The feature store's
ece is decimated to 1 ms and cannot resolve a crash.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: `alfven_eigenmode` verification notebook

**Files:**
- Create: `data/events/alfven_eigenmode/verification.ipynb`

**Interfaces:**
- Consumes: `Panel`, `review`, `corpus_signal`, `NoDataError` (Tasks 4–5); `toksearch_d3d.PtDataSignal` at runtime.
- Produces: nothing other tasks read.

The CO2 fetch lives in this notebook, not in the feature namespace: the `co2` feature declares `sources=("corpus",)` and has no fdp resolver. The 180 annotated AE shots span 170659–178879 and the corpus covers 185601–204999, so for this category the fetch is the normal path, not the fallback. The kernel must have been started under the `fdp run` wrapper.

- [ ] **Step 1: Write the notebook**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python - <<'PY'
import nbformat

from labeler.config import Paths

HEADER = """# Alfven eigenmode - verification

Use the **Python (FAITH labelmaker)** kernel, started under the `fdp run`
wrapper:

```bash
pixi run -e labelmaker fdp run python -m ipykernel_launcher -f <connection-file>
```

or, more simply, launch Jupyter itself under it. Without the wrapper PTDATA
fails with `getservbyname failed for task 'PTSERVER'`.

CO2 crosspower for the chord pairs R0xV1, R0xV2, R0xV3, as log magnitude
against time and frequency, with the 80-250 kHz AE band shaded. TAEs sit in
that band as coherent clusters; an RSAE chirps upward as q_min falls; a BAE is
lower, in the tens of kHz.

**This category fetches.** Its 180 annotated shots are 170659-178879 and the
corpus covers 185601-204999, so none of them is on disk. Where the corpus does
carry `co2` it is filled on about half its shots, all above 198279. The fetch
below pulls the four unfiltered chords from PTDATA and caches them next to the
corrections.
"""

FETCH = '''from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal

from labeler.events.verify import NoDataError, corpus_signal, review_path

CHORDS = ["DENR0UF", "DENV1UF", "DENV2UF", "DENV3UF"]


def co2_chords(shot):
    """The four CO2 chords for one shot: (4, T) float32 and time in ms.

    Corpus first. When the corpus has nothing - which is every shot in this
    category's original annotation set - fetch from PTDATA and cache the
    arrays beside the corrections, keyed by shot.
    """
    try:
        array = corpus_signal(shot, "co2")
        return array.x, array.y
    except NoDataError:
        pass

    cache = review_path("alfven_eigenmode", shot).with_name(f"{shot}_co2.npz")
    if cache.is_file():
        with np.load(cache) as stored:
            return stored["time_ms"], stored["chords"]

    try:
        from toksearch_d3d import PtDataSignal
    except ImportError as error:
        raise NoDataError(
            "shot is outside the corpus and toksearch_d3d is unavailable. "
            "Start the kernel under `pixi run -e labelmaker fdp run ...`"
        ) from error

    fetched = [PtDataSignal(name).fetch(int(shot)) for name in CHORDS]
    time_ms = np.asarray(fetched[0]["times"], dtype="float64")
    chords = np.stack(
        [np.asarray(one["data"], dtype="float32") for one in fetched]
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, time_ms=time_ms, chords=chords)
    return time_ms, chords


def crosspower(time_ms, a, b, *, nperseg=2048):
    """Log |cross-spectral density| of two chords, as (freq_khz, t_ms, z)."""
    # Take the rate from the SPAN, never from a median diff: the corpus time
    # vector is float32, whose spacing at t ~ 3 s quantises.
    rate = (len(time_ms) - 1) / ((time_ms[-1] - time_ms[0]) / 1000.0)
    freq, times, spectrum = scipy_signal.spectrogram(
        a * b, fs=rate, nperseg=nperseg, noverlap=nperseg // 2, mode="complex"
    )
    return freq / 1000.0, times * 1000.0 + time_ms[0], np.log10(np.abs(spectrum) + 1e-30)'''

PANELS = '''from labeler.events.verify import Panel, review

event = "alfven_eigenmode"
shot = 1  # replace with a shot from shots.csv
source = "format/shots"

time_ms, chords = co2_chords(shot)

panels = []
for index, label in [(1, "R0xV1"), (2, "R0xV2"), (3, "R0xV3")]:
    freq_khz, t_ms, power = crosspower(time_ms, chords[0], chords[index])
    keep = freq_khz <= 300.0
    panels.append(
        Panel(
            title=f"co2 crosspower {label}",
            kind="heatmap",
            x=t_ms,
            y=freq_khz[keep],
            z=power[keep],
            ylabel="kHz",
            bands=[(80.0, 250.0)],
        )
    )'''

REVIEW = """session = review(event, shot, panels, source=source)
session"""

FOOTER = """The original annotation has five classes - `lfm`, `bae`, `eae`,
`rsae`, `tae` - and the formatted labels collapse them to binary presence,
excluding LFM. Corrections here are binary too. If you can name the
sub-family, put it in the roster `notes`.

Nothing above 250 kHz is observable: that is this record's Nyquist. A mode
there is reported as quiet, not unknown.

```python
from labeler.events.verify import read_corrections, review_path
read_corrections(review_path(event, shot))
```
"""

notebook = nbformat.v4.new_notebook()
notebook.cells = [
    nbformat.v4.new_markdown_cell(HEADER),
    nbformat.v4.new_code_cell("%load_ext autoreload\n%autoreload 2"),
    nbformat.v4.new_code_cell(FETCH),
    nbformat.v4.new_code_cell(PANELS),
    nbformat.v4.new_code_cell(REVIEW),
    nbformat.v4.new_markdown_cell(FOOTER),
]
notebook.metadata["kernelspec"] = {
    "display_name": "Python (FAITH labelmaker)",
    "language": "python",
    "name": "faith-labelmaker",
}
path = Paths.from_env().label_tables / "alfven_eigenmode" / "verification.ipynb"
nbformat.write(notebook, path)
print("wrote", path)
PY
```

Expected: `wrote .../alfven_eigenmode/verification.ipynb`

- [ ] **Step 2: Check it parses and compiles**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python - <<'PY'
import json

path = "data/events/alfven_eigenmode/verification.ipynb"
notebook = json.loads(open(path).read())
for cell in notebook["cells"]:
    if cell["cell_type"] == "code" and not "".join(cell["source"]).startswith("%"):
        compile("".join(cell["source"]), path, "exec")
print("ok:", len(notebook["cells"]), "cells")
PY
```

Expected: `ok: 6 cells`

- [ ] **Step 3: Commit**

```bash
git add data/events/alfven_eigenmode/verification.ipynb
git commit -m "$(cat <<'EOF'
labeler: alfven_eigenmode verification notebook

CO2 crosspower R0xV1/V2/V3 with the 80-250 kHz band shaded. The 180 annotated
shots are outside the corpus, so the notebook fetches the four unfiltered
chords from PTDATA and caches them beside the corrections.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: The `fishbone` category

**Files:**
- Create: `data/events/fishbone/README.md`
- Create: `data/events/fishbone/raw/.gitkeep`
- Create: `data/events/fishbone/format/.gitkeep`
- Create: `data/events/fishbone/verification.ipynb`

**Interfaces:**
- Consumes: `Panel`, `review`, `corpus_signal` (Tasks 4–5).
- Produces: a complete category directory; `test_every_category_has_a_verification_notebook` (Task 6) goes green.

`data/events/fishbone/` exists and is empty. Its roster arrived in Task 3. The lexicon entry `fishbone` is already in `src/labeler/events/lexicons.yaml:126-131` and the inventory row `Fishbone` is already in `data/events/discrete_labels.csv` — neither changes. Nothing is registered in `events.yaml`: that file lists datasets a formatter produces, and fishbone has neither a raw table nor a formatter.

- [ ] **Step 1: Write the README**

Create `data/events/fishbone/README.md`:

```markdown
# Fishbone

## Description
Fishbones are bursting m/n = 1/1 internal kink modes driven by fast ions. The
resonance is with the trapped fast ions' toroidal precession, so the mode sits
near the precession frequency rather than near an Alfven gap, and each burst
chirps **downward** as the resonant fast-ion energy falls. On DIII-D a burst is
roughly 2-30 kHz and lasts a few ms, repeating through the beam-heated part of
the discharge. The name is the shape: the burst envelope on a Mirnov trace
looks like a fish skeleton.

Two kinds are distinguished by what sets the frequency: precession-frequency
fishbones under near-perpendicular neutral beams, and diamagnetic-frequency
fishbones (f ~ omega\*i) under tangential beams.

They expel fast ions, which shows up as neutron-rate drops and beam-ion losses,
and they can seed sawteeth and NTMs. A fishbone is close kin to a sawtooth
precursor - both are the 1/1 kink - and the two are easy to confuse on
magnetics alone.

First observed on PDX during near-perpendicular NBI (McGuire et al. 1983).

Typically found via the magnetic spectrogram (Mirnov / MHR probes) as a
repeated downward chirp in the 2-30 kHz band, confirmed as n = 1 from a
toroidal probe array, with supporting drops in the neutron rate.

## Method
Use the magnetic spectrogram and look for the n = 1 chirp: bursts in the
2-30 kHz band that sweep downward in frequency, on the beam-heated part of the
discharge, with toroidal mode number n = 1.

No detector exists yet. This is the stated method, not a description of one
that runs.

## Provenance
None. No curated table has been obtained, so `raw/` is empty and `format/`
holds nothing.

## Models
**stable**: none

**latest**: none

**all**: none

## Alias
- fishbone
- fishbones

## Reference
- K. McGuire et al., "Study of high-beta magnetohydrodynamic modes and
  fast-ion losses in PDX", Phys. Rev. Lett. 50, 891 (1983).
- L. Chen, R. B. White and M. N. Rosenbluth, "Excitation of internal kink modes
  by trapped energetic beam ions", Phys. Rev. Lett. 52, 1122 (1984).

## Contact
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu

## Tables

Inventory row: Fishbone; lexicon id: `fishbone`.

The scope inventory is [`discrete_labels.csv`](../discrete_labels.csv).
`raw/` holds the untouched provided lists; `format/` holds their
common-schema CSVs and metadata. Each `extend_<model>/` holds one
producing source's output on the project shot list. Categories without
a producer have no `extend_*` directory. See the [table guide](../README.md).

No raw table is registered for this category yet, so nothing appears in
`../events.yaml` and there is no `formatter.py`.

## Category

The CSV `category` column and grid values use integer IDs.

| ID | Label |
| --- | --- |
| 0 | Absent |
| 1 | Present |

Unknown or unclassified grid cells are stored separately from 0.

## Verification

[`verification.ipynb`](verification.ipynb) plots the magnetic spectrogram for
one shot and takes back corrections. The review roster is
[`shots.csv`](shots.csv). See the [table guide](../README.md) for the tier
rules.
```

- [ ] **Step 2: Create the empty directories**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
mkdir -p data/events/fishbone/raw data/events/fishbone/format
touch data/events/fishbone/raw/.gitkeep data/events/fishbone/format/.gitkeep
ls data/events/fishbone
```

Expected: `README.md  format  raw  shots.csv`

- [ ] **Step 3: Write the notebook**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python - <<'PY'
import nbformat

from labeler.config import Paths

HEADER = """# Fishbone - verification

Use the **Python (FAITH labelmaker)** kernel. Set `shot` and `window`, drag a
time range on any panel, then press *Mark present* / *Mark absent*, *Verify*
and *Save*.

The magnetic spectrogram. What you are looking for is a burst that chirps
**downward** through the 2-30 kHz band - the shaded region - repeating on the
beam-heated part of the discharge. A synthetic fishbone in this repository's
test fixtures sweeps 20 -> 12 kHz at -0.8 kHz/ms, which is the shape.

**The toroidal mode number is judged, not measured.** The corpus does not
record the MHR probes' toroidal angles, so the second panel shows the
cross-phase between a probe pair and leaves n = 1 to your eye. Nothing here
computes n.

Fishbones and sawtooth precursors are both the 1/1 kink and look alike on
magnetics. If a burst sits immediately before a crash in
`sawtooth_oscillation`, say so in the roster `notes`.
"""

PANELS = '''import numpy as np
from scipy import signal as scipy_signal

from labeler.events.verify import Panel, corpus_signal, review

event = "fishbone"
shot = 1        # replace with a shot from shots.csv
window = (1500.0, 3500.0)   # milliseconds
source = "format/shots"
probes = [0, 4]             # B1 and B5, the pair used for the cross-phase

mhr = corpus_signal(shot, "mhr", channels=probes, t_range=window)

# Take the rate from the SPAN, never from a median diff: xdata is float32 and
# its spacing quantises at t ~ 3 s.
rate = (mhr.x.shape[0] - 1) / ((mhr.x[-1] - mhr.x[0]) / 1000.0)
nperseg = 4096

freq, times, power = scipy_signal.spectrogram(
    mhr.y[0], fs=rate, nperseg=nperseg, noverlap=nperseg // 2
)
_, _, cross = scipy_signal.spectrogram(
    mhr.y[0] + 1j * mhr.y[1], fs=rate, nperseg=nperseg,
    noverlap=nperseg // 2, mode="complex", return_onesided=False,
)
keep = freq <= 40000.0

panels = [
    Panel(
        title=f"mhr B{probes[0] + 1} spectrogram",
        kind="heatmap",
        x=times * 1000.0 + mhr.x[0],
        y=freq[keep] / 1000.0,
        z=np.log10(power[keep] + 1e-30),
        ylabel="kHz",
        bands=[(2.0, 30.0)],
    ),
    Panel(
        title=f"cross-phase B{probes[0] + 1} x B{probes[1] + 1}",
        kind="heatmap",
        x=times * 1000.0 + mhr.x[0],
        y=freq[keep] / 1000.0,
        z=np.angle(cross[: keep.sum()]),
        ylabel="kHz",
        bands=[(2.0, 30.0)],
    ),
]'''

REVIEW = """session = review(event, shot, panels, source=source)
session"""

FOOTER = """This category has no formatted labels yet, so the label row will be
missing and `review()` simply leaves it out. Your corrections are the first
annotation it has.

```python
from labeler.events.verify import read_corrections, review_path
read_corrections(review_path(event, shot))
```
"""

notebook = nbformat.v4.new_notebook()
notebook.cells = [
    nbformat.v4.new_markdown_cell(HEADER),
    nbformat.v4.new_code_cell("%load_ext autoreload\n%autoreload 2"),
    nbformat.v4.new_code_cell(PANELS),
    nbformat.v4.new_code_cell(REVIEW),
    nbformat.v4.new_markdown_cell(FOOTER),
]
notebook.metadata["kernelspec"] = {
    "display_name": "Python (FAITH labelmaker)",
    "language": "python",
    "name": "faith-labelmaker",
}
path = Paths.from_env().label_tables / "fishbone" / "verification.ipynb"
nbformat.write(notebook, path)
print("wrote", path)
PY
```

Expected: `wrote .../fishbone/verification.ipynb`

- [ ] **Step 4: Run the whole verification test file**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker pytest tests/labeler/test_events_verify.py -q
```

Expected: `13 passed` — all sixteen categories now have both files.

- [ ] **Step 5: Commit**

```bash
git add data/events/fishbone tests/labeler/test_events_verify.py
git commit -m "$(cat <<'EOF'
labeler: the fishbone category

README, empty raw/ and format/, and a verification notebook on the magnetic
spectrogram. n = 1 is judged by eye: the corpus does not record the MHR
probes' toroidal angles, so nothing here claims to measure it.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Move `plot_original` into the notebooks that use it

**Files:**
- Modify: `src/labeler/events/notebooks.py` (delete lines 90–278)
- Modify: `data/events/minimum_safety_factor/example.ipynb`
- Modify: `data/events/alfven_eigenmode/example.ipynb`
- Modify: `data/events/edge_localized_mode/example.ipynb`
- Modify: `data/events/neoclassical_tearing_mode/example.ipynb`
- Modify: `data/events/resistive_wall_mode/example.ipynb`
- Modify: `data/events/high_confinement_mode/example.ipynb`
- Modify: `data/events/low_confinement_mode/example.ipynb`
- Modify: `tests/labeler/test_events_verify.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `labeler.events.notebooks` exporting only `load_shot` and `plot_shot`.

Reading which traces settle a category is case-by-case work, and so is reading that category's original annotation format. `plot_original` is 189 of the 278 lines in `notebooks.py` and six `if event ==` branches, each serving exactly one notebook. Each branch moves into the notebook it serves.

`data/events/detachment/example.ipynb` does not call `plot_original` and is not touched.

- [ ] **Step 1: Write the failing test**

Append to `tests/labeler/test_events_verify.py`:

```python
def test_notebooks_module_is_only_the_generic_helpers():
    from labeler.events import notebooks

    assert hasattr(notebooks, "load_shot")
    assert hasattr(notebooks, "plot_shot")
    assert not hasattr(notebooks, "plot_original"), (
        "per-category original-label plotting belongs in each example.ipynb"
    )


def test_no_example_notebook_imports_plot_original():
    import json

    from labeler.config import Paths

    root = Paths.from_env().label_tables
    for path in sorted(root.glob("*/example.ipynb")):
        sources = "".join(
            "".join(cell["source"]) for cell in json.loads(path.read_text())["cells"]
        )
        assert "plot_original" not in sources, path
        assert "labelmaker.events" not in sources, path
        assert "from labelmaker" not in sources, path
```

- [ ] **Step 2: Run it to verify it fails**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker pytest tests/labeler/test_events_verify.py -q -k "plot_original or generic_helpers"
```

Expected: 2 failed — `plot_original` is still exported, and the example notebooks still import it.

- [ ] **Step 3: Move each branch into its notebook**

For each of the seven categories, open its `example.ipynb` and:

1. Change the import cell from

   ```python
   from labelmaker.events.notebooks import load_shot, plot_shot, plot_original
   ```

   to

   ```python
   from labeler.events.notebooks import load_shot, plot_shot
   ```

2. Replace the `plot_original(grid);` cell with a cell holding that category's branch, lifted verbatim from `src/labeler/events/notebooks.py` and de-indented, with `grid`, `event` and `shot` taken from the notebook's own variables.

The branches, by category and by their line range in the current `notebooks.py`:

| Category | Lines | What it reads |
| --- | --- | --- |
| `minimum_safety_factor` | 117–152 | `read_events` on the rule's `<shot>_events.parquet`, broken_barh per class |
| `alfven_eigenmode` | 158–182 | `read_ae` on the pickle, `imshow` of the five classes against sample index |
| `edge_localized_mode` | 183–196 | `read_elm`, a 1 ms binary trace |
| `neoclassical_tearing_mode` | 197–228 | `read_tm_h5` and `read_tm_tar`, one subplot per source |
| `resistive_wall_mode` | 229–246 | `pd.read_csv` on the onset CSVs, `scatter` of onset time against `NTOR` |
| `high_confinement_mode`, `low_confinement_mode` | 247–270 | `pd.read_csv` on the regime table, broken_barh for L/H/QH/WP |

Each of the six non-qmin branches needs this preamble, which the shared
function used to supply:

```python
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from labeler.config import Paths

root = Paths.from_env().label_tables
manifest = yaml.safe_load((root / "events.yaml").read_text())
spec = next(row for row in manifest["format_datasets"] if row["name"] == event)
raw = {row["stem"]: root / row["path"] for row in manifest["raw_datasets"]}
paths = {stem: raw[stem] for stem in spec["sources"]}
fig, ax = plt.subplots(figsize=(10, 3.5), layout="constrained")
```

and the `minimum_safety_factor` branch needs:

```python
import json

import matplotlib.pyplot as plt

from labeler.config import Paths
from labeler.events.schema import read_events

root = Paths.from_env().label_tables
folder = root / event / source
meta = json.loads(folder.with_suffix(".meta.json").read_text())
events_root = Path(meta["full_events_root"])
if not events_root.is_absolute():
    events_root = Paths.from_env().root / events_root
```

Replace every `raise ValueError(...)` / `plt.close(fig)` guard with a plain
`print(...)`: in a notebook an absent original annotation is a fact to read,
not an exception to handle.

- [ ] **Step 4: Delete `plot_original` from the module**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python - <<'PY'
from pathlib import Path

path = Path("src/labeler/events/notebooks.py")
lines = path.read_text().splitlines(keepends=True)
start = next(i for i, line in enumerate(lines) if line.startswith("def plot_original"))
path.write_text("".join(lines[:start]).rstrip() + "\n")
print("notebooks.py is now", len(path.read_text().splitlines()), "lines")
PY
```

Expected: `notebooks.py is now 88 lines` (or 87–89; the exact count depends on trailing blanks).

Then update the module docstring, first line of the file, to:

```python
"""Loading and plotting saved event-label grids.

Reading a category's ORIGINAL annotation format is case-by-case work - six
formats, six readers, one consumer each - so it lives in each category's
`example.ipynb` rather than here. What stays is what is generic over the saved
grid: opening one, and drawing it.
"""
```

- [ ] **Step 5: Run the tests**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker pytest tests/labeler/test_events_verify.py tests/labeler/test_events_rosters.py -q
pixi run -e labelmaker ruff check src/labeler/events/notebooks.py
```

Expected: `27 passed` (15 in `test_events_verify.py`, 12 in
`test_events_rosters.py`) and `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/labeler/events/notebooks.py data/events/*/example.ipynb tests/labeler/test_events_verify.py
git commit -m "$(cat <<'EOF'
labeler: move plot_original out of src and into the seven notebooks that use it

Six original annotation formats, six readers, one consumer each. notebooks.py
keeps load_shot and plot_shot, which are generic over the saved grid.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Delete the pre-rename package directories

**Files:**
- Delete: `src/labelmaker/` (recursively)
- Delete: `src/ideate/` (recursively)
- Modify: `data/events/*/README.md`, `data/events/*/formatter.py` — stale module paths

**Interfaces:**
- Consumes: nothing.
- Produces: an `src/` holding only `faith`, `labeler`, `shot_design` and `tokamak_foundation_model`.

Both directories hold `__pycache__` and nothing else — zero `.py` files — and neither is tracked by git. Verify that before deleting.

Three things keep their old names on purpose, matching the R1 rename decision: the pixi environments `labelmaker` and `ideate`, the data roots `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker` and `.../nc1514/ideate`, and the `.meta.json` sidecars, which are provenance records of what produced a file. The wider sweep — about 190 files, mostly historical run records under `outputs/` — is a separate job.

- [ ] **Step 1: Confirm both directories are dead**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
find src/labelmaker src/ideate -name '*.py' | wc -l
git ls-files src/labelmaker src/ideate | wc -l
```

Expected: `0` and `0`. **If either is not zero, stop** and report — the directories are not dead and this task's premise is wrong.

- [ ] **Step 2: Delete them**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
rm -rf src/labelmaker src/ideate
ls src/
```

Expected: `faith  labeler  shot_design  tokamak_foundation_model`

- [ ] **Step 3: Verify the package still imports**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -c "
import labeler, shot_design
from labeler.events import notebooks, rosters, verify
print('ok')
"
```

Expected: `ok`

- [ ] **Step 4: Fix the stale module paths under `data/events/`**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
grep -rln 'labelmaker\.\|src/labelmaker' data/events --include='*.md' --include='*.py' --include='*.ipynb'
```

For each file, replace `labelmaker.` with `labeler.` and `src/labelmaker/` with `src/labeler/`:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
grep -rl 'labelmaker\.\|src/labelmaker' data/events --include='*.md' --include='*.py' --include='*.ipynb' \
  | xargs sed -i 's#src/labelmaker/#src/labeler/#g; s#labelmaker\.events#labeler.events#g; s#labelmaker\.config#labeler.config#g; s#scripts/labelmaker/#scripts/labeler/#g'
```

Then check what is left, which should be only the pixi environment name:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
grep -rn 'labelmaker' data/events --include='*.md' --include='*.py' --include='*.ipynb' | grep -v -- '-e labelmaker' | grep -v 'FAITH labelmaker' | grep -v 'faith-labelmaker'
```

Expected: no output. Every surviving `labelmaker` is either `pixi run -e labelmaker`, the kernel display name `Python (FAITH labelmaker)`, or the kernel id `faith-labelmaker` — all of which are correct.

- [ ] **Step 5: Run the labeler suite**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker pytest tests/labeler -q -m "not real_data"
```

Expected: no new failures against the pre-task baseline. Record the counts; five known-spurious failures are expected in this suite and are not caused by this task.

- [ ] **Step 6: Commit**

```bash
git add -A src data/events
git commit -m "$(cat <<'EOF'
labeler: delete the pre-rename src/labelmaker and src/ideate

Both held __pycache__ and nothing else, and neither was tracked. Module paths
under data/events follow. The pixi environment names, the data roots and the
.meta.json provenance records keep their old names on purpose.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: Document the roster and the review workflow

**Files:**
- Modify: `data/events/README.md`
- Modify: `data/events/<category>/README.md` × 15 (fishbone's was written in Task 10)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing other tasks read.

- [ ] **Step 1: Add the roster section to the table guide**

Insert into `data/events/README.md`, immediately after the `## Interval CSVs` section and before `## Per-shot sampled grids`:

```markdown
## Review rosters

`shots.csv` in each category is a shot-level roster of who has looked at what.
It is not the category's shot list: a shot enters it when somebody puts it up
for review, and the interval tables stay the record of what is labelled.

```csv
shot,tier,holdout,reviewers,verified_on,notes
170815,gold,false,nc1514;aj17,2026-09-17,retimed first onset -30 ms
178631,silver,false,nc1514,2026-09-17,
185945,unverified,false,,,
```

`tier` is a curation judgement, set by hand, and says nothing about how many
people have reviewed a shot: `gold`/`silver`/`unverified` are legal values,
not a count. `holdout` is a required `true`/`false` reserving a shot from
training and tuning for final evaluation only; blank is invalid. Reviewer
ids are `$USER`, separated by `;`, in the order they reviewed. A reviewer
appears at most once per shot, so pressing Verify twice re-dates the row
without touching `tier`. `verified_on` is the most recent review, blank when
unverified. `notes` is one line.

Ten gold shots per category is the target, not something the file enforces.
The three placeholder rows every category ships are meant to be deleted.

`labeler.events.rosters.validate_roster` validates this schema: legal `tier`
and `holdout` values, no duplicate shots or reviewers, and `verified_on`
agreeing with `reviewers`. It does not check `tier` against the reviewer
list.

## Verification notebooks

`verification.ipynb` in each category shows one shot's signals against its
saved labels and takes back corrections:

```bash
pixi run -e labelmaker jupyter lab data/events/<category>/verification.ipynb
```

Set `shot`, drag a time range on any panel, then press *Mark present* /
*Mark absent*, *Verify* and *Save*. Nothing touches disk until *Save*, which
writes two files: the corrected intervals to `review/<shot>.csv`, in the same
five-column schema as `format/`, and the reviewer's name into `shots.csv`.

Corrections under `review/` are a separate claim from `format/` and
`extend_*/`: they are what a human asserts after looking. No formatter reads
or overwrites them, and merging them back into a formatted table is not yet
decided.

Four categories have panels chosen for the phenomenon -
`minimum_safety_factor` (qmin against the rule's class thresholds),
`sawtooth_oscillation` (raw ECE channels 20-36, four to a row),
`alfven_eigenmode` (CO2 crosspower R0xV1/V2/V3) and `fishbone` (the magnetic
spectrogram). The rest carry generic `ip`/`betan`/`pinj_total` panels, which show
that a shot exists and not that a phenomenon happened; each says so and asks
to be replaced. Scaffold a new one with:

```bash
pixi run -e labelmaker python scripts/labeler/make_verification_notebook.py --event <category>
```

`alfven_eigenmode` fetches: its 180 annotated shots are 170659-178879 and the
corpus covers 185601-204999, so its notebook pulls the CO2 chords from PTDATA
and needs its kernel started under the `fdp run` wrapper.
```

- [ ] **Step 2: Add a Verification section to each category README**

For each of the fifteen categories other than `fishbone`, append to
`data/events/<category>/README.md`:

```markdown
## Verification

[`verification.ipynb`](verification.ipynb) plots one shot's signals against its
saved labels and takes back corrections. The review roster is
[`shots.csv`](shots.csv). See the [table guide](../README.md) for the tier
rules.
```

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
for readme in data/events/*/README.md; do
  grep -q '^## Verification' "$readme" && continue
  cat >> "$readme" <<'MD'

## Verification

[`verification.ipynb`](verification.ipynb) plots one shot's signals against its
saved labels and takes back corrections. The review roster is
[`shots.csv`](shots.csv). See the [table guide](../README.md) for the tier
rules.
MD
  echo "appended $readme"
done
```

Expected: fifteen `appended ...` lines (fishbone already has the section).

Note that `locked_mode` and `quiescent_high_confinement_mode` have no
`README.md` at all. Create one for each first, with just a title and the
Verification section:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
for event in locked_mode quiescent_high_confinement_mode; do
  [ -f "data/events/$event/README.md" ] && continue
  title=$(echo "$event" | tr '_' ' ')
  printf '# %s\n\nNo description, method, provenance or table has been written for this\ncategory yet. The scope inventory row is in\n[`discrete_labels.csv`](../discrete_labels.csv).\n' "$title" > "data/events/$event/README.md"
  echo "created data/events/$event/README.md"
done
```

Run this **before** the append loop above.

- [ ] **Step 3: Check every category README links both files**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python - <<'PY'
from labeler.config import Paths

root = Paths.from_env().label_tables
missing = []
for directory in sorted(p for p in root.iterdir() if p.is_dir()):
    readme = directory / "README.md"
    if not readme.is_file():
        missing.append(f"{directory.name}: no README.md")
        continue
    text = readme.read_text()
    for link in ("verification.ipynb", "shots.csv"):
        if link not in text:
            missing.append(f"{directory.name}: no link to {link}")
print("\n".join(missing) or "all sixteen READMEs link both files")
PY
```

Expected: `all sixteen READMEs link both files`

- [ ] **Step 4: Run the full labeler suite one more time**

Run:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker pytest tests/labeler -q -m "not real_data"
```

Expected: no new failures against the Task 12 baseline.

- [ ] **Step 5: Commit**

```bash
git add data/events/README.md data/events/*/README.md
git commit -m "$(cat <<'EOF'
labeler: document the review rosters and the verification notebooks

The tier rules, what Save writes, and why corrections under review/ are a
separate claim from format/.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review

**Spec coverage.** Roster schema → Task 2; placeholder rosters → Task 3; corrections directory → Task 5; shared `verify.py` → Tasks 4–5; qmin/sawtooth/AE/fishbone panels → Tasks 7–10; generic panels → Task 6; `plot_original` move → Task 11; dead directories and stale module paths → Task 12; documentation → Task 13; `anywidget` → Task 1; tests → distributed through Tasks 2–6 and 10–11.

**Deviation from the spec, deliberate.** The spec put `validate_shots` in `verify.py`. This plan splits it into `rosters.py`: the roster schema is pure pandas and is read by things that never draw a figure, and keeping it out of `verify.py` means importing the schema does not pull in plotly, h5py and ipywidgets. The function is named `validate_roster` rather than `validate_shots` to match its module.

**Known gap.** The spec's test list includes "the AE fallback raises its wrapper message when toksearch is unavailable". That check lives inside the AE notebook's `co2_chords`, which the suite does not import, so it is covered by reading rather than by a test. Pulling it into `verify.py` to test it would put a category-specific fetch in the shared module, which is the boundary this design exists to hold.
