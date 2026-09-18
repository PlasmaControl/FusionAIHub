# Browser Verification Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace sixteen per-event `verification.ipynb` notebooks with one browser application that renders a full shot at any zoom, sourcing raw signals through a corpus-first cache.

**Architecture:** Three new units. `labeler.events.raw` is the only way to get a raw signal: corpus, then the project's `.cache/raw/`, then a live fdp fetch written into that cache. `labeler.events.panels` is a registry of per-event functions turning a shot and a time window into `Panel`s. `labeler.events.ui` is a FastAPI app ported structure-for-structure from `src/shot_design/ui/`, serving those panels to a vanilla-JS plotly page behind a token gate on the loopback.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, h5py, numpy, scipy, plotly (browser CDN-free: the existing `plotly` conda package ships `plotly.min.js`), pytest, pixi.

**Spec:** `docs/superpowers/specs/2026-09-18-verification-ui-design.md`

## Global Constraints

- **`src/labeler/events/verify.py` is never edited.** `Panel`, `label_panel`, `write_corrections`, `correction_path`, `corrections_for`, `ReviewSession`, `record_review` and `NoDataError` are imported, not changed. `tests/labeler/test_events_verify.py` (and any sibling covering it) must pass unmodified at every commit.
- **Nothing in this plan writes `shots.csv`.** The app reads it. `ReviewSession.save()` is never called by the app, because it calls `record_review`, which writes that file.
- **Corrections are append-only.** `write_corrections` refuses an existing path; nothing added here may weaken that.
- **Corpus file layout** is flat groups, each holding `xdata` (1-D float32, **seconds**) and `ydata` `(C, T)` float32, no attributes anywhere in the file. Cache files written here use exactly that layout so promotion is a move.
- **`FeatureArray.x` returned by `corpus_signal` and `fdp_signal` is MILLISECONDS**, despite the corpus storing seconds and `FeatureArray`'s own docstring saying seconds. `raw_signal` keeps the milliseconds convention. Converting on write (`ms / 1000.0`) and on read (`s * 1000.0`) is the seam.
- **Sample rate is always derived from the span**, `(n - 1) / ((x[-1] - x[0]) / 1000.0)`, never from a median of successive differences: a float32 time vector quantises its spacing at t ~ 3 s.
- **Cache root** is `<repo>/.cache/raw/`, gitignored at `.gitignore:49`. Never the corpus root: training globs `*_processed.h5` there (`src/tokamak_foundation_model/ignite/spike.py:220`, `fastts_train.py:214`, `train_codec.py:633`) and a partial shot would be handed to training with groups missing.
- **Live fdp fetches only work under the wrapper** `pixi run -e labelmaker fdp run ...`. Outside it PTDATA fails with `getservbyname failed for task 'PTSERVER'` and MDSplus with `TREE-E-FOPENR`.
- **Tests run as** `pixi run -e labelmaker pytest tests/labeler/<file> -q`. Never the bare `.pixi` interpreter.
- Ruff, `line-length = 88`.

## File Structure

| Path | Responsibility |
|---|---|
| `src/labeler/config.py` | +`Paths.raw_cache`, the `.cache/raw/` root |
| `src/labeler/events/raw.py` | three-tier `raw_signal`, corpus-layout writer, `promote`, `clean`, CLI |
| `src/labeler/events/panels/__init__.py` | `build()`, `BUILDERS`, `PanelSet` |
| `src/labeler/events/panels/_generic.py` | `ip` / `betan` / `pinj_total`, the fallback |
| `src/labeler/events/panels/alfven_eigenmode.py` | CO2 crosspower, `crosspower()` |
| `src/labeler/events/panels/fishbone.py` | mhr spectrogram + cross-phase |
| `src/labeler/events/panels/sawtooth_oscillation.py` | ECE channel rows |
| `src/labeler/events/panels/minimum_safety_factor.py` | qmin / qpsi / ip |
| `src/labeler/events/ui/app.py` | FastAPI, token gate, JSON endpoints |
| `src/labeler/events/ui/serve.py` | launcher, loopback bind, token link |
| `src/labeler/events/ui/__main__.py` | `python -m labeler.events.ui` |
| `src/labeler/events/ui/static/{index.html,app.js,style.css}` | the page |
| `tests/labeler/test_events_raw.py` | tiers, writer, promote, clean |
| `tests/labeler/test_events_panels.py` | registry, fallback, crosspower rate |
| `tests/labeler/test_events_ui.py` | gate, endpoints, the shots.csv guarantee |

---

### Task 1: The cache root

**Files:**
- Modify: `src/labeler/config.py:18-68`
- Test: `tests/labeler/test_config.py` (create if absent)

**Interfaces:**
- Produces: `Paths.raw_cache: Path`, defaulting to `<repo>/.cache/raw`, overridden by `LABELER_RAW_CACHE`.

- [ ] **Step 1: Write the failing test**

Append to `tests/labeler/test_config.py` (create the file with `from labeler.config import DEFAULT_RAW_CACHE, Paths` at the top if it does not exist):

```python
def test_raw_cache_defaults_under_the_repo():
    # Label DATA lives under data/events for the same reason: a run from a
    # SLURM scratch directory must find the same place a run from the repo
    # does, so this resolves off this file, never off cwd.
    assert Paths().raw_cache == DEFAULT_RAW_CACHE
    assert DEFAULT_RAW_CACHE.name == "raw"
    assert DEFAULT_RAW_CACHE.parent.name == ".cache"


def test_raw_cache_honours_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LABELER_RAW_CACHE", str(tmp_path / "elsewhere"))
    assert Paths.from_env().raw_cache == tmp_path / "elsewhere"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pixi run -e labelmaker pytest tests/labeler/test_config.py -q`
Expected: FAIL, `ImportError: cannot import name 'DEFAULT_RAW_CACHE'`

- [ ] **Step 3: Implement**

In `src/labeler/config.py`, after `DEFAULT_LABEL_TABLES` (around line 44):

```python
#: Where a live fetch parks a shot's raw record. Deliberately the PROJECT
#: directory and not the corpus: EKOLEMEN is the long-term home for bulk raw
#: signal data and has the capacity for it, while this directory is meant for
#: temporary and smaller things. A fetch lands here as scratch and stays
#: scratch until `raw.promote` moves it. `.cache` is gitignored, and deleting
#: this directory at any time is safe - the next read refetches.
DEFAULT_RAW_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "raw"
```

Add the field to `Paths` after `label_tables`:

```python
    raw_cache: Path = DEFAULT_RAW_CACHE
```

And to `from_env`, after the `label_tables` entry:

```python
            raw_cache=Path(getenv("LABELER_RAW_CACHE", str(DEFAULT_RAW_CACHE))),
```

- [ ] **Step 4: Run it and watch it pass**

Run: `pixi run -e labelmaker pytest tests/labeler/test_config.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/labeler/config.py tests/labeler/test_config.py
git commit -m "labeler: give the raw fetch cache a root of its own"
```

---

### Task 2: Reading the two on-disk tiers

**Files:**
- Create: `src/labeler/events/raw.py`
- Test: `tests/labeler/test_events_raw.py`

**Interfaces:**
- Consumes: `Paths.raw_cache` (Task 1); `corpus_signal`, `NoDataError` from `labeler.events.verify`.
- Produces:
  - `raw_signal(shot: int, group: str, *, channels: Sequence[int] | None = None, t_range: tuple[float, float] | None = None, paths: Paths | None = None) -> FeatureArray` — `x` milliseconds, `y` `(C, T)` float32.
  - `cache_path(shot: int, *, paths: Paths | None = None) -> Path`
  - `write_group(path: Path, group: str, times_ms: np.ndarray, values: np.ndarray) -> None`
  - `groups_in(path: Path) -> set[str]`

- [ ] **Step 1: Write the failing test**

Create `tests/labeler/test_events_raw.py`:

```python
"""The three-tier raw read, and the cache file it writes."""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from labeler.config import Paths
from labeler.events import raw
from labeler.events.verify import NoDataError


def write_corpus_file(path, group, times_ms, values):
    """A corpus file by hand: seconds on xdata, no attributes anywhere."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group(group)
        g.create_dataset("xdata", data=np.asarray(times_ms, "float32") / 1000.0)
        g.create_dataset("ydata", data=np.asarray(values, "float32"))


@pytest.fixture
def roots(tmp_path):
    return Paths(corpus=tmp_path / "corpus", raw_cache=tmp_path / "cache")


def test_tier_one_is_the_corpus(roots):
    times, values = np.arange(10.0), np.arange(20.0).reshape(2, 10)
    write_corpus_file(roots.corpus / "1_processed.h5", "co2", times, values)
    got = raw.raw_signal(1, "co2", paths=roots)
    assert np.allclose(got.x, times)
    assert np.allclose(got.y, values)
    assert got.attrs["tier"] == "corpus"


def test_tier_two_is_the_cache(roots):
    times, values = np.arange(10.0), np.arange(20.0).reshape(2, 10)
    write_corpus_file(roots.raw_cache / "2_processed.h5", "co2", times, values)
    got = raw.raw_signal(2, "co2", paths=roots)
    assert np.allclose(got.y, values)
    assert got.attrs["tier"] == "cache"


def test_the_corpus_wins_over_the_cache(roots):
    """Both tiers hold shot 3; the corpus copy is the one served."""
    write_corpus_file(roots.corpus / "3_processed.h5", "co2",
                      np.arange(4.0), np.zeros((1, 4)))
    write_corpus_file(roots.raw_cache / "3_processed.h5", "co2",
                      np.arange(4.0), np.ones((1, 4)))
    assert np.allclose(raw.raw_signal(3, "co2", paths=roots).y, 0.0)


def test_channels_and_t_range_slice_the_way_the_corpus_does(roots):
    times = np.arange(100.0)
    values = np.arange(400.0).reshape(4, 100)
    write_corpus_file(roots.corpus / "4_processed.h5", "co2", times, values)
    got = raw.raw_signal(4, "co2", channels=[1, 3], t_range=(10.0, 20.0),
                         paths=roots)
    assert got.y.shape == (2, 11)
    assert np.allclose(got.x, np.arange(10.0, 21.0))
    assert np.allclose(got.y[0], values[1, 10:21])


def test_a_group_the_file_lacks_and_cannot_be_fetched_raises(roots):
    write_corpus_file(roots.corpus / "5_processed.h5", "co2",
                      np.arange(4.0), np.zeros((1, 4)))
    with pytest.raises(NoDataError, match="no fetch route"):
        raw.raw_signal(5, "not_a_group", paths=roots)


def test_write_group_is_additive(roots, tmp_path):
    path = tmp_path / "6_processed.h5"
    raw.write_group(path, "co2", np.arange(4.0), np.zeros((1, 4)))
    raw.write_group(path, "ece", np.arange(4.0), np.ones((1, 4)))
    assert raw.groups_in(path) == {"co2", "ece"}


def test_write_group_leaves_no_partial_file_when_it_fails(roots, tmp_path):
    path = tmp_path / "7_processed.h5"
    with pytest.raises(ValueError):
        # y must be (C, T); a 1-D y is rejected BEFORE anything is created,
        # so a reader never meets a half-written shot.
        raw.write_group(path, "co2", np.arange(4.0), np.zeros(4))
    assert not path.exists()
    assert list(tmp_path.glob("*")) == []


def test_write_group_stores_seconds_so_the_corpus_can_read_it(tmp_path):
    path = tmp_path / "8_processed.h5"
    raw.write_group(path, "co2", np.array([0.0, 1000.0]), np.zeros((1, 2)))
    with h5py.File(path, "r") as f:
        assert np.allclose(f["co2"]["xdata"][:], [0.0, 1.0])
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pixi run -e labelmaker pytest tests/labeler/test_events_raw.py -q`
Expected: FAIL, `ImportError: cannot import name 'raw' from 'labeler.events'`

- [ ] **Step 3: Implement**

Create `src/labeler/events/raw.py`:

```python
"""One way to get a raw signal, whatever tier it happens to live on.

Three places are tried in order: the corpus, the project's fetch cache, and
a live fetch that writes the cache. A caller cannot tell which one answered
except by looking at `attrs["tier"]`, which exists for diagnostics and for
the promote command, not for branching.

The split between the two on-disk roots is about what the storage is FOR.
EKOLEMEN holds the long-term bulk raw record and has the capacity for it.
The project directory has room but is meant for temporary and smaller
things, so a fetch lands there as scratch. Nothing here ever writes the
corpus; `promote` does, deliberately and by hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..config import Paths
from ..features.store import FeatureArray
from .verify import NoDataError, corpus_signal

#: A group whose `ydata` is this narrow carries the corpus' absent-signal
#: sentinel rather than a record.
SENTINEL_WIDTH = 1


def cache_path(shot: int, *, paths: Paths | None = None) -> Path:
    """Where a fetched shot parks, in the corpus' own naming."""
    paths = Paths.from_env() if paths is None else paths
    return paths.raw_cache / f"{int(shot)}_processed.h5"


def groups_in(path) -> set[str]:
    """The group names one corpus-layout file holds; empty if it is absent."""
    import h5py

    path = Path(path)
    if not path.is_file():
        return set()
    with h5py.File(path, "r") as f:
        return set(f.keys())


def write_group(path, group: str, times_ms, values) -> None:
    """Add one group to a corpus-layout file, atomically and additively.

    Additive because the server fetches groups one panel at a time: asking
    for `ece` on a shot whose cache already holds `co2` must not throw the
    240 MB of CO2 away. Atomic because it is doing that concurrently with
    itself, and a reader must never meet a half-written group.

    `times_ms` arrives in milliseconds, the convention `corpus_signal` and
    `fdp_signal` both return, and is stored in SECONDS, the convention the
    corpus file itself uses. That conversion is the whole reason this
    function exists rather than a bare h5py call.
    """
    import h5py

    path = Path(path)
    times_ms = np.asarray(times_ms, dtype="float64")
    values = np.asarray(values, dtype="float32")
    # Validated before any file is touched, so a bad call leaves nothing on
    # disk at all - not an empty file, not a temp file.
    if values.ndim != 2:
        raise ValueError(f"{group!r} ydata must be (C, T), got {values.shape}")
    if values.shape[-1] != times_ms.shape[-1]:
        raise ValueError(
            f"{group!r} xdata {times_ms.shape} and ydata {values.shape} disagree"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(f".{path.name}.{id(values):x}.tmp")
    try:
        if path.is_file():
            # h5py cannot add to a file another process may be reading, and
            # cannot shrink one in place either. Copy, add, rename.
            import shutil

            shutil.copy2(path, scratch)
        with h5py.File(scratch, "a") as f:
            if group in f:
                del f[group]
            g = f.create_group(group)
            g.create_dataset("xdata", data=(times_ms / 1000.0).astype("float32"))
            # Chunked per channel: a whole-channel read touches contiguous
            # chunks and a time slice touches one chunk per channel. No
            # compression - the record is broadband noise that gzip barely
            # shrinks while costing minutes per shot.
            g.create_dataset(
                "ydata",
                data=values,
                chunks=(1, min(values.shape[-1], 1 << 20)),
            )
        scratch.replace(path)
    finally:
        scratch.unlink(missing_ok=True)


def raw_signal(
    shot: int,
    group: str,
    *,
    channels: Sequence[int] | None = None,
    t_range: tuple[float, float] | None = None,
    paths: Paths | None = None,
) -> FeatureArray:
    """One group of one shot: `x` milliseconds, `y` `(C, T)` float32.

    Corpus, then cache, then a live fetch that fills the cache.
    """
    paths = Paths.from_env() if paths is None else paths
    for tier, root in (("corpus", paths.corpus), ("cache", paths.raw_cache)):
        try:
            array = corpus_signal(
                shot, group, channels=channels, t_range=t_range, corpus=root
            )
        except NoDataError:
            continue
        return FeatureArray(
            x=array.x, y=array.y, attrs={**array.attrs, "tier": tier}
        )
    return _fetch(shot, group, channels=channels, t_range=t_range, paths=paths)


def _fetch(shot, group, *, channels, t_range, paths) -> FeatureArray:
    # Filled in by Task 3. Until then, say so rather than returning
    # something a panel would silently plot.
    raise NoDataError(
        f"shot {int(shot)} has no {group!r} in the corpus or the cache, and "
        f"there is no fetch route for {group!r}"
    )
```

- [ ] **Step 4: Run it and watch it pass**

Run: `pixi run -e labelmaker pytest tests/labeler/test_events_raw.py -q`
Expected: PASS, 8 passed

- [ ] **Step 5: Confirm nothing in `verify.py` moved**

Run: `pixi run -e labelmaker pytest tests/labeler -q -k verify`
Expected: PASS, no failures

- [ ] **Step 6: Commit**

```bash
git add src/labeler/events/raw.py tests/labeler/test_events_raw.py
git commit -m "labeler: read a raw signal from the corpus or the fetch cache"
```

---

### Task 3: Fetch on miss

**Files:**
- Modify: `src/labeler/events/raw.py` (replace `_fetch`)
- Test: `tests/labeler/test_events_raw.py` (append)

**Interfaces:**
- Consumes: `fdp_signal`, `CO2_CHORDS`, `ECE_POINT`, `ECE_TREE` from `labeler.events.verify`.
- Produces: `FETCH_SPECS: dict[str, FetchSpec]`; `FetchSpec(exprs, via, tree)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/labeler/test_events_raw.py`:

```python
def test_a_miss_fetches_and_fills_the_cache(roots, monkeypatch):
    """The fetch runs once; the second read is served off disk."""
    calls = []

    def fake_fdp_signal(shot, exprs, *, tree, via, t_range=None, **kwargs):
        calls.append((shot, tuple(exprs), via))
        times = np.arange(50.0)
        return FeatureArray(
            x=times,
            y=np.arange(4 * 50, dtype="float32").reshape(4, 50),
            attrs={"units": "ms"},
        )

    monkeypatch.setattr(raw, "fdp_signal", fake_fdp_signal)
    first = raw.raw_signal(9, "co2", paths=roots)
    assert first.attrs["tier"] == "fetch"
    assert len(calls) == 1
    assert calls[0][2] == "ptdata"

    assert raw.cache_path(9, paths=roots).is_file()
    second = raw.raw_signal(9, "co2", paths=roots)
    assert second.attrs["tier"] == "cache"
    assert len(calls) == 1, "a cached shot must not refetch"
    assert np.allclose(second.y, first.y)


def test_a_fetch_honours_channels_and_t_range_after_caching_everything(
    roots, monkeypatch
):
    """The cache holds the WHOLE record; the slice is applied to the return."""
    def fake_fdp_signal(shot, exprs, *, tree, via, t_range=None, **kwargs):
        return FeatureArray(
            x=np.arange(100.0),
            y=np.arange(400.0, dtype="float32").reshape(4, 100),
            attrs={"units": "ms"},
        )

    monkeypatch.setattr(raw, "fdp_signal", fake_fdp_signal)
    got = raw.raw_signal(10, "co2", channels=[0, 2], t_range=(5.0, 9.0),
                         paths=roots)
    assert got.y.shape == (2, 5)
    with h5py.File(raw.cache_path(10, paths=roots), "r") as f:
        assert f["co2"]["ydata"].shape == (4, 100), "the cache is not sliced"


def test_ece_fetches_over_mds_not_ptdata(roots, monkeypatch):
    seen = {}

    def fake_fdp_signal(shot, exprs, *, tree, via, t_range=None, **kwargs):
        seen.update(via=via, tree=tree, n=len(exprs))
        return FeatureArray(
            x=np.arange(10.0),
            y=np.zeros((len(exprs), 10), dtype="float32"),
            attrs={"units": "ms"},
        )

    monkeypatch.setattr(raw, "fdp_signal", fake_fdp_signal)
    raw.raw_signal(11, "ece", paths=roots)
    assert seen["via"] == "mds"
    assert seen["tree"] == "D3D"
    assert seen["n"] == 48
```

Add to that file's imports: `from labeler.features.store import FeatureArray`.

- [ ] **Step 2: Run it and watch it fail**

Run: `pixi run -e labelmaker pytest tests/labeler/test_events_raw.py -q -k fetch`
Expected: FAIL, `NoDataError: ... there is no fetch route for 'co2'`

- [ ] **Step 3: Implement**

In `src/labeler/events/raw.py`, extend the import from `.verify`:

```python
from .verify import (
    CO2_CHORDS,
    ECE_POINT,
    ECE_TREE,
    NoDataError,
    corpus_signal,
    fdp_signal,
)
```

Add the dataclass import (`from dataclasses import dataclass`) and, after `SENTINEL_WIDTH`:

```python
#: ECE is 48 radiometer channels, matching the corpus group's channel order
#: so a fetched shot and a corpus shot index the same.
ECE_CHANNELS = tuple(range(1, 49))


@dataclass(frozen=True)
class FetchSpec:
    """How to fetch one corpus group live, when neither tier has it."""

    exprs: tuple[str, ...]
    via: str
    tree: str = ECE_TREE


#: Only groups listed here can be fetched. An unlisted group missing from
#: both tiers raises rather than guessing at a point name: a wrong guess
#: puts the wrong diagnostic in front of a reviewer, which is worse than a
#: refusal.
FETCH_SPECS: dict[str, FetchSpec] = {
    "co2": FetchSpec(exprs=CO2_CHORDS, via="ptdata"),
    "ece": FetchSpec(
        exprs=tuple(ECE_POINT.format(channel=c) for c in ECE_CHANNELS),
        via="mds",
    ),
}
```

Replace `_fetch` entirely:

```python
def _fetch(shot, group, *, channels, t_range, paths) -> FeatureArray:
    """Fetch one group live, cache the WHOLE record, return the slice.

    The cache is never the `t_range` window. Widening a window later would
    otherwise refetch a shot that is already on disk - minutes, and hundreds
    of megabytes over the wire, for a drag of the mouse.
    """
    spec = FETCH_SPECS.get(group)
    if spec is None:
        raise NoDataError(
            f"shot {int(shot)} has no {group!r} in the corpus or the cache, "
            f"and there is no fetch route for {group!r}; known routes are "
            f"{sorted(FETCH_SPECS)}"
        )
    fetched = fdp_signal(
        int(shot), list(spec.exprs), tree=spec.tree, via=spec.via
    )
    write_group(cache_path(shot, paths=paths), group, fetched.x, fetched.y)
    array = corpus_signal(
        shot, group, channels=channels, t_range=t_range, corpus=paths.raw_cache
    )
    return FeatureArray(x=array.x, y=array.y, attrs={**array.attrs, "tier": "fetch"})
```

- [ ] **Step 4: Run it and watch it pass**

Run: `pixi run -e labelmaker pytest tests/labeler/test_events_raw.py -q`
Expected: PASS, 11 passed

- [ ] **Step 5: Commit**

```bash
git add src/labeler/events/raw.py tests/labeler/test_events_raw.py
git commit -m "labeler: fetch a missing shot into the cache, whole record at a time"
```

---

### Task 4: `promote` and `clean`

**Files:**
- Modify: `src/labeler/events/raw.py`
- Test: `tests/labeler/test_events_raw.py` (append)

**Interfaces:**
- Produces:
  - `CORPUS_GROUPS: tuple[str, ...]` — the 32 group names a complete shot holds.
  - `promote(shot: int, *, partial: bool = False, paths: Paths | None = None) -> Path`
  - `clean(*, paths: Paths | None = None) -> int` — bytes removed.
  - `main(argv: Sequence[str] | None = None) -> int`

- [ ] **Step 1: Write the failing test**

Append to `tests/labeler/test_events_raw.py`:

```python
def test_promote_refuses_a_partial_shot_and_names_what_is_missing(roots):
    raw.write_group(raw.cache_path(12, paths=roots), "co2",
                    np.arange(4.0), np.zeros((1, 4)))
    with pytest.raises(ValueError) as caught:
        raw.promote(12, paths=roots)
    message = str(caught.value)
    assert "1 of 32" in message
    assert "ece" in message, "the refusal must name groups that are missing"
    assert "--partial" in message
    assert raw.cache_path(12, paths=roots).is_file(), "nothing moved"


def test_promote_moves_a_partial_shot_when_told_to(roots):
    raw.write_group(raw.cache_path(13, paths=roots), "co2",
                    np.arange(4.0), np.ones((1, 4)))
    landed = raw.promote(13, partial=True, paths=roots)
    assert landed == roots.corpus / "13_processed.h5"
    assert landed.is_file()
    assert not raw.cache_path(13, paths=roots).exists(), "a move, not a copy"


def test_a_promoted_shot_reads_identically_from_tier_one(roots):
    times, values = np.arange(10.0), np.arange(20.0).reshape(2, 10)
    raw.write_group(raw.cache_path(14, paths=roots), "co2", times, values)
    before = raw.raw_signal(14, "co2", paths=roots)
    assert before.attrs["tier"] == "cache"
    raw.promote(14, partial=True, paths=roots)
    after = raw.raw_signal(14, "co2", paths=roots)
    assert after.attrs["tier"] == "corpus"
    assert np.allclose(before.x, after.x)
    assert np.allclose(before.y, after.y)


def test_promote_refuses_to_clobber_a_corpus_shot(roots):
    write_corpus_file(roots.corpus / "15_processed.h5", "co2",
                      np.arange(4.0), np.zeros((1, 4)))
    raw.write_group(raw.cache_path(15, paths=roots), "co2",
                    np.arange(4.0), np.ones((1, 4)))
    with pytest.raises(FileExistsError):
        raw.promote(15, partial=True, paths=roots)


def test_clean_empties_the_cache_and_a_read_refetches(roots, monkeypatch):
    calls = []

    def fake_fdp_signal(shot, exprs, *, tree, via, t_range=None, **kwargs):
        calls.append(shot)
        return FeatureArray(x=np.arange(10.0),
                            y=np.zeros((4, 10), dtype="float32"),
                            attrs={"units": "ms"})

    monkeypatch.setattr(raw, "fdp_signal", fake_fdp_signal)
    raw.raw_signal(16, "co2", paths=roots)
    assert raw.clean(paths=roots) > 0
    assert not raw.cache_path(16, paths=roots).exists()
    raw.raw_signal(16, "co2", paths=roots)
    assert len(calls) == 2
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pixi run -e labelmaker pytest tests/labeler/test_events_raw.py -q -k "promote or clean"`
Expected: FAIL, `AttributeError: module 'labeler.events.raw' has no attribute 'promote'`

- [ ] **Step 3: Implement**

Append to `src/labeler/events/raw.py`:

```python
#: The 32 groups a complete corpus shot holds, from
#: `src/tokamak_foundation_model/data/config/modalities/modalities.yaml`.
#: `promote` compares against this to decide whether a cache entry is a
#: whole shot or a verification fetch of one diagnostic.
CORPUS_GROUPS: tuple[str, ...] = (
    "mhr", "ece", "co2",
    "gas", "gas_raw", "ech", "ech_raw", "pin", "tin",
    "d_alpha", "mse", "ts_core_density", "ts_core_temp",
    "ts_tan_density", "ts_tan_temp", "cer_rot", "filterscopes",
    "ip", "betan", "pinj", "tinj", "li", "q95", "qmin", "qpsi",
    "kappa", "tritop", "tribot", "aminor", "rmaxis", "zmaxis", "wmhd",
)


def promote(shot: int, *, partial: bool = False, paths: Paths | None = None) -> Path:
    """Move a cached shot into the corpus, where it lives long-term.

    Manual and separate from anything the reviewer clicks: this moves
    hundreds of megabytes, and a Save that did it as a side effect would be
    a Save that can half-fail.

    The default refuses an incomplete shot. Training loaders glob
    `*_processed.h5` in the corpus root, and a verification fetch
    materialises the one group a panel asked for - so a partial file there
    is one those globs hand to training with the rest of the groups
    missing.
    """
    paths = Paths.from_env() if paths is None else paths
    source = cache_path(shot, paths=paths)
    if not source.is_file():
        raise FileNotFoundError(f"shot {int(shot)} is not in {paths.raw_cache}")
    target = paths.corpus / source.name
    if target.exists():
        raise FileExistsError(
            f"{target} already exists; promote never overwrites a corpus "
            f"shot. Inspect both and remove one by hand."
        )
    present = groups_in(source)
    if not partial and len(present) < len(CORPUS_GROUPS):
        missing = sorted(set(CORPUS_GROUPS) - present)
        raise ValueError(
            f"shot {int(shot)} holds {len(present)} of {len(CORPUS_GROUPS)} "
            f"groups; missing {', '.join(missing)}. Training globs "
            f"*_processed.h5 in the corpus root and would read this as a "
            f"whole shot. Pass --partial if that is what you want."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    # `replace` is atomic within a filesystem and falls back to copy+unlink
    # across one, which the cache and the corpus may well be.
    try:
        source.replace(target)
    except OSError:
        import shutil

        shutil.copy2(source, target)
        source.unlink()
    return target


def clean(*, paths: Paths | None = None) -> int:
    """Delete the whole fetch cache; return the bytes recovered."""
    import shutil

    paths = Paths.from_env() if paths is None else paths
    root = paths.raw_cache
    if not root.is_dir():
        return 0
    freed = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    shutil.rmtree(root)
    return freed


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m labeler.events.raw promote 178642 [--partial]` / `clean`."""
    import argparse

    parser = argparse.ArgumentParser(prog="labeler.events.raw")
    sub = parser.add_subparsers(dest="command", required=True)
    move = sub.add_parser("promote", help="move a cached shot into the corpus")
    move.add_argument("shot", type=int)
    move.add_argument(
        "--partial", action="store_true",
        help="promote a shot that does not hold all 32 groups",
    )
    sub.add_parser("clean", help="delete the whole fetch cache")
    args = parser.parse_args(argv)

    paths = Paths.from_env()
    if args.command == "clean":
        freed = clean(paths=paths)
        print(f"removed {freed / 1e9:.2f} GB from {paths.raw_cache}")
        return 0
    try:
        landed = promote(args.shot, partial=args.partial, paths=paths)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        print(f"error: {error}")
        return 1
    print(f"promoted {args.shot} -> {landed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run it and watch it pass**

Run: `pixi run -e labelmaker pytest tests/labeler/test_events_raw.py -q`
Expected: PASS, 16 passed

- [ ] **Step 5: Add the pixi task**

In `pyproject.toml`, under `[tool.pixi.feature.labelmaker.tasks]`, after the `label` entry:

```toml
# `pixi run -e labelmaker labeler-raw promote 178642 --partial`: move a
# fetched shot out of the project's .cache/raw and into EKOLEMEN, which is
# where bulk raw signal data lives long-term.
labeler-raw = "python -m labeler.events.raw"
```

- [ ] **Step 6: Commit**

```bash
git add src/labeler/events/raw.py tests/labeler/test_events_raw.py pyproject.toml
git commit -m "labeler: promote a cached shot to EKOLEMEN by hand, never by accident"
```

---

### Task 5: The panel registry and its generic fallback

**Files:**
- Create: `src/labeler/events/panels/__init__.py`, `src/labeler/events/panels/_generic.py`
- Test: `tests/labeler/test_events_panels.py`

**Interfaces:**
- Consumes: `Panel` from `labeler.events.verify`; `read_feature` from `labeler.features.store`; `Paths.features_file`.
- Produces:
  - `build(event: str, shot: int, *, t_range: tuple[float, float] | None = None, paths: Paths | None = None) -> list[Panel]`
  - `guidance(event: str) -> str`
  - `BUILDERS: dict[str, Builder]`, where a builder is a module exposing `panels(shot, *, t_range, paths)` and `GUIDANCE: str`.

- [ ] **Step 1: Write the failing test**

Create `tests/labeler/test_events_panels.py`:

```python
"""The per-event panel registry."""
from __future__ import annotations

import numpy as np
import pytest

from labeler.events import panels as registry


def test_an_unregistered_event_falls_back_to_the_generic_builder(monkeypatch):
    calls = []
    monkeypatch.setattr(
        registry._generic, "panels",
        lambda shot, *, t_range=None, paths=None: calls.append(shot) or [],
    )
    assert registry.build("no_such_event", 42) == []
    assert calls == [42]


def test_every_registered_event_has_guidance():
    for event in registry.BUILDERS:
        assert registry.guidance(event).strip(), event


def test_guidance_for_an_unregistered_event_says_it_is_generic():
    assert "generic" in registry.guidance("no_such_event").lower()


def test_the_registry_covers_the_events_with_bespoke_panels():
    assert set(registry.BUILDERS) == {
        "alfven_eigenmode",
        "fishbone",
        "minimum_safety_factor",
        "sawtooth_oscillation",
    }
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pixi run -e labelmaker pytest tests/labeler/test_events_panels.py -q`
Expected: FAIL, `ImportError: cannot import name 'panels' from 'labeler.events'`

- [ ] **Step 3: Implement the fallback**

Create `src/labeler/events/panels/_generic.py`:

```python
"""Panels for an event whose own traces have not been decided yet.

`ip`, `betan` and `pinj_total` show that the shot exists and that its labels
sit inside the discharge. They do NOT show whether the phenomenon happened.
An event still on these is an event waiting for someone who knows which
traces settle it.
"""

from __future__ import annotations

from ...config import Paths
from ...features.store import read_feature
from ..verify import Panel

GUIDANCE = (
    "<b>These are generic panels.</b> <code>ip</code>, <code>betan</code> and "
    "<code>pinj_total</code> show that the shot exists and that its labels sit "
    "inside the discharge. They do not show whether this phenomenon actually "
    "happened. Until someone adds a builder for this event under "
    "<code>labeler/events/panels/</code>, treat a review here as provisional."
)

TRACES = (("ip", "A"), ("betan", ""), ("pinj_total", "kW"))


def panels(shot, *, t_range=None, paths=None):
    """One panel per scalar feature; seconds on disk, milliseconds on screen."""
    paths = Paths.from_env() if paths is None else paths
    features = paths.features_file(int(shot))
    built = []
    for name, ylabel in TRACES:
        array = read_feature(features, name)
        x = array.x * 1000.0
        y = array.y
        if t_range is not None:
            keep = (x >= t_range[0]) & (x <= t_range[1])
            x, y = x[keep], y[:, keep]
        built.append(Panel(title=name, x=x, y=y, ylabel=ylabel))
    return built
```

Create `src/labeler/events/panels/__init__.py`:

```python
"""Which traces settle which phenomenon.

Deciding that is case-by-case work that does not generalise, which is why it
used to live in sixteen notebook cells. It lives here instead so it can be
imported, tested, and called by a server - and so there is one copy of it.

A builder module exposes `panels(shot, *, t_range, paths) -> list[Panel]` and
a `GUIDANCE` string, the prose a reviewer needs beside the figure. An event
with no module gets `_generic`, which is what twelve of those notebooks
already were.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...config import Paths
from ..verify import Panel
from . import (
    _generic,
    alfven_eigenmode,
    fishbone,
    minimum_safety_factor,
    sawtooth_oscillation,
)

BUILDERS = {
    "alfven_eigenmode": alfven_eigenmode,
    "fishbone": fishbone,
    "minimum_safety_factor": minimum_safety_factor,
    "sawtooth_oscillation": sawtooth_oscillation,
}


def build(
    event: str,
    shot: int,
    *,
    t_range: tuple[float, float] | None = None,
    paths: Paths | None = None,
) -> list[Panel]:
    """The panels for one shot of one event, over one window."""
    builder = BUILDERS.get(event, _generic)
    return list(builder.panels(int(shot), t_range=t_range, paths=paths))


def guidance(event: str) -> str:
    """What a reviewer of this event needs to be told, as HTML."""
    return BUILDERS.get(event, _generic).GUIDANCE
```

Note the relative imports are three dots deep (`...config`) because this is a
sub-package of `labeler.events`.

- [ ] **Step 4: Stub the three bespoke modules so the package imports**

Create each of `alfven_eigenmode.py`, `fishbone.py`, `minimum_safety_factor.py`, `sawtooth_oscillation.py` under `src/labeler/events/panels/` with exactly:

```python
GUIDANCE = "TASK 6/7 fills this in."


def panels(shot, *, t_range=None, paths=None):
    raise NotImplementedError
```

These are replaced in full by Tasks 6 and 7; they exist here only so `__init__` imports and the registry test can run.

- [ ] **Step 5: Run it and watch it pass**

Run: `pixi run -e labelmaker pytest tests/labeler/test_events_panels.py -q`
Expected: PASS, 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/labeler/events/panels tests/labeler/test_events_panels.py
git commit -m "labeler: make the per-event panel choice a registry, not a notebook cell"
```

---

### Task 6: The alfven_eigenmode builder

This is the acceptance target. Every other event is scaffolding beside it.

**Files:**
- Modify: `src/labeler/events/panels/alfven_eigenmode.py` (replace the stub)
- Test: `tests/labeler/test_events_panels.py` (append)

**Interfaces:**
- Consumes: `raw_signal` (Task 3); `Panel`, `CO2_CHORDS` from `verify`.
- Produces: `crosspower(time_ms, a, b, *, nperseg=2048, max_khz=300.0, max_bins=1000) -> tuple[np.ndarray, np.ndarray, np.ndarray]` returning `(freq_khz, t_ms, log_power)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/labeler/test_events_panels.py`:

```python
from labeler.events.panels import alfven_eigenmode as ae


def test_crosspower_takes_its_rate_from_the_span_not_a_median_diff():
    """A float32 time vector quantises its spacing at t ~ 3 s.

    Successive differences of such a vector are wrong by percents and in a
    biased direction; the span is exact. A 120 kHz tone must land at 120 kHz
    on a time base that starts at 3000 ms, not merely on one starting at 0.
    """
    rate = 1_000_000.0
    n = 200_000
    t_ms = (3000.0 + np.arange(n) / rate * 1000.0).astype("float32")
    tone = np.sin(2 * np.pi * 120_000.0 * np.arange(n) / rate).astype("float32")
    freq_khz, _, power = ae.crosspower(t_ms.astype("float64"), tone, tone)
    peak = freq_khz[np.argmax(power.mean(axis=1))]
    assert abs(peak - 120.0) < 1.0


def test_crosspower_caps_its_time_bins():
    rate = 1_000_000.0
    n = 400_000
    t_ms = np.arange(n) / rate * 1000.0
    noise = np.random.default_rng(0).normal(size=n).astype("float32")
    _, t_out, power = ae.crosspower(t_ms, noise, noise, max_bins=250)
    assert power.shape[1] <= 250
    assert len(t_out) == power.shape[1]


def test_crosspower_averages_power_before_taking_the_log():
    """A geometric mean would be dragged down by the quiet bins in a block.

    That is exactly what suppresses a short burst - the thing the panel
    exists to show. One loud block among quiet ones must survive averaging.
    """
    rate = 1_000_000.0
    n = 200_000
    t_ms = np.arange(n) / rate * 1000.0
    rng = np.random.default_rng(1)
    signal = rng.normal(scale=1e-3, size=n)
    signal[100_000:110_000] += 5.0 * np.sin(
        2 * np.pi * 120_000.0 * np.arange(10_000) / rate
    )
    freq_khz, t_out, power = ae.crosspower(t_ms, signal, signal, max_bins=40)
    band = (freq_khz > 110.0) & (freq_khz < 130.0)
    profile = power[band].mean(axis=0)
    burst = np.argmax(profile)
    assert 95.0 < t_out[burst] < 115.0
    assert profile[burst] > np.median(profile) + 1.0


def test_alfven_panels_are_three_crosspower_heatmaps(monkeypatch):
    from labeler.events import panels as registry
    from labeler.features.store import FeatureArray

    rate = 1_000_000.0
    n = 60_000
    fake = FeatureArray(
        x=np.arange(n) / rate * 1000.0,
        y=np.random.default_rng(2).normal(size=(4, n)).astype("float32"),
        attrs={"tier": "cache"},
    )
    monkeypatch.setattr(ae, "raw_signal", lambda *a, **k: fake)
    built = registry.build("alfven_eigenmode", 178642)
    assert len(built) == 3
    assert {p.kind for p in built} == {"heatmap"}
    assert all(p.bands == [(80.0, 250.0)] for p in built)
    assert all(p.ylabel == "kHz" for p in built)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pixi run -e labelmaker pytest tests/labeler/test_events_panels.py -q -k "crosspower or alfven"`
Expected: FAIL, `AttributeError: module ... has no attribute 'crosspower'`

- [ ] **Step 3: Implement**

Replace `src/labeler/events/panels/alfven_eigenmode.py` entirely:

```python
"""Alfven eigenmodes, as coherent narrowband activity on more than one chord.

Lifted from `data/events/alfven_eigenmode/verification.ipynb`, which this
replaces. The two comments inside `crosspower` are load-bearing and were
each written after the corresponding bug.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from ..raw import raw_signal
from ..verify import CO2_CHORDS, Panel

#: Where TAEs and RSAEs live. Marked on the figure, not filtered for.
AE_BAND = (80.0, 250.0)

#: R0xV1, R0xV2, R0xV3 - the reference chord against each vertical one.
#: A feature on ONE pair only is more likely chord-specific noise than a
#: mode, which is the whole reason three pairs are drawn rather than one.
CHORD_PAIRS = ((0, 1), (0, 2), (0, 3))

#: A full-rate spectrogram of a 2 s window is ~3300 columns, and three of
#: those as plotly heatmaps is tens of megabytes of JSON and a wedged
#: browser tab. The server re-renders on zoom instead, so a window is always
#: this sharp however wide it is.
MAX_TIME_BINS = 1000

GUIDANCE = (
    "<b>What you are looking for:</b> coherent narrowband activity inside "
    "80-250 kHz (between the dotted lines) that shows up on <b>more than one "
    "chord pair</b>. A feature on a single pair only is more likely "
    "chord-specific noise than a real mode. TAEs sit near the low end of the "
    "band; RSAEs sweep upward as the safety factor evolves through the shot."
    "<br><br>No AE-annotated shot is in the corpus - the 180 annotated shots "
    "span 170659-178879 and the corpus starts at 185601 - so every shot here "
    "is fetched live over PTDATA. The first open of a shot moves ~240 MB and "
    "takes several minutes; after that it is cached and reopening is instant."
)


def crosspower(time_ms, a, b, *, nperseg=2048, max_khz=300.0, max_bins=MAX_TIME_BINS):
    """Log |S_a . conj(S_b)| for two chords, as `(freq_khz, t_ms, z)`.

    The rate comes from the SPAN, never from a median of successive
    differences: a float32 time vector quantises its spacing at t ~ 3 s, and
    the resulting rate is wrong by percents, in a biased direction, with
    nothing on screen to give it away.
    """
    time_ms = np.asarray(time_ms, dtype="float64")
    rate = (len(time_ms) - 1) / ((time_ms[-1] - time_ms[0]) / 1000.0)
    kwargs = {
        "fs": rate,
        "nperseg": min(nperseg, len(time_ms)),
        "noverlap": min(nperseg, len(time_ms)) // 2,
        "mode": "complex",
    }
    freq, times_s, spec_a = scipy_signal.spectrogram(np.asarray(a), **kwargs)
    _, _, spec_b = scipy_signal.spectrogram(np.asarray(b), **kwargs)
    cross = spec_a * np.conj(spec_b)

    keep = freq <= max_khz * 1000.0
    freq_khz = freq[keep] / 1000.0
    magnitude = np.abs(cross[keep])
    t_ms = times_s * 1000.0 + time_ms[0]

    # The average is taken over the POWER and the log comes AFTER it.
    # Averaging the log instead is a geometric mean, which is pulled down by
    # the quiet bins in a block and so suppresses exactly the short bursts
    # this panel exists to show.
    if magnitude.shape[1] > max_bins:
        width = magnitude.shape[1] // max_bins
        usable = (magnitude.shape[1] // width) * width
        magnitude = (
            magnitude[:, :usable].reshape(len(freq_khz), -1, width).mean(axis=2)
        )
        t_ms = t_ms[:usable].reshape(-1, width).mean(axis=1)
    return freq_khz, t_ms, np.log10(magnitude + 1e-30)


def panels(shot, *, t_range=None, paths=None):
    """One crosspower heatmap per chord pair, over the window asked for."""
    co2 = raw_signal(int(shot), "co2", t_range=t_range, paths=paths)
    built = []
    for reference, vertical in CHORD_PAIRS:
        freq_khz, t_ms, power = crosspower(co2.x, co2.y[reference], co2.y[vertical])
        built.append(
            Panel(
                title=f"CO2 crosspower {CO2_CHORDS[reference]} x {CO2_CHORDS[vertical]}",
                kind="heatmap",
                x=t_ms,
                y=freq_khz,
                z=power,
                ylabel="kHz",
                bands=[AE_BAND],
            )
        )
    return built
```

- [ ] **Step 4: Run it and watch it pass**

Run: `pixi run -e labelmaker pytest tests/labeler/test_events_panels.py -q`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/labeler/events/panels/alfven_eigenmode.py tests/labeler/test_events_panels.py
git commit -m "labeler: move the AE crosspower panels out of the notebook"
```

---

### Task 7: The three remaining bespoke builders

**Files:**
- Modify: `src/labeler/events/panels/{fishbone,sawtooth_oscillation,minimum_safety_factor}.py`
- Test: `tests/labeler/test_events_panels.py` (append)

**Interfaces:**
- Consumes: `raw_signal`; `Panel`; `read_feature`; `Paths.features_file`.
- Produces: `panels(...)` and `GUIDANCE` on each module, matching the registry contract from Task 5.

- [ ] **Step 1: Write the failing test**

Append to `tests/labeler/test_events_panels.py`:

```python
def test_fishbone_draws_power_and_cross_phase(monkeypatch):
    from labeler.events import panels as registry
    from labeler.events.panels import fishbone
    from labeler.features.store import FeatureArray

    rate = 500_000.0
    n = 60_000
    fake = FeatureArray(
        x=1500.0 + np.arange(n) / rate * 1000.0,
        y=np.random.default_rng(3).normal(size=(2, n)).astype("float32"),
        attrs={},
    )
    monkeypatch.setattr(fishbone, "raw_signal", lambda *a, **k: fake)
    built = registry.build("fishbone", 192238)
    assert len(built) == 2
    assert "spectrogram" in built[0].title
    assert "cross-phase" in built[1].title
    # Cross-phase is an angle: it must span roughly -pi to pi, which a
    # spectrogram of a complex signal built from the two probes would not.
    assert built[1].z.min() < -3.0 and built[1].z.max() > 3.0


def test_sawtooth_draws_four_rows_of_four_adjacent_ece_channels(monkeypatch):
    from labeler.events import panels as registry
    from labeler.events.panels import sawtooth_oscillation as saw
    from labeler.features.store import FeatureArray

    def fake_raw_signal(shot, group, *, channels=None, t_range=None, paths=None):
        rows = list(channels)
        return FeatureArray(
            x=np.arange(500.0),
            y=np.zeros((len(rows), 500), dtype="float32"),
            attrs={},
        )

    monkeypatch.setattr(saw, "raw_signal", fake_raw_signal)
    built = registry.build("sawtooth_oscillation", 192238)
    assert len(built) == 4
    assert all(p.y.shape[0] == 4 for p in built)
    assert built[0].title == "ECE ch 20-23"


def test_minimum_safety_factor_draws_its_three_class_thresholds(monkeypatch, tmp_path):
    from labeler.config import Paths
    from labeler.events import panels as registry
    from labeler.events.panels import minimum_safety_factor as msf
    from labeler.features.store import FeatureArray

    features = {
        "qmin": FeatureArray(x=np.arange(10.0), y=np.ones((1, 10)), attrs={}),
        "qpsi": FeatureArray(x=np.arange(10.0), y=np.ones((5, 10)), attrs={}),
        "ip": FeatureArray(x=np.arange(10.0), y=np.ones((1, 10)), attrs={}),
    }
    monkeypatch.setattr(msf, "read_feature", lambda path, name: features[name])
    built = registry.build("minimum_safety_factor", 1, paths=Paths(root=tmp_path))
    assert [p.title for p in built][0] == "qmin (EFIT01 aeqdsk)"
    assert list(built[0].hlines) == [0.95, 1.5, 2.0]
    assert built[1].kind == "heatmap"
    assert built[1].ylabel == "psi_n"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pixi run -e labelmaker pytest tests/labeler/test_events_panels.py -q -k "fishbone or sawtooth or safety"`
Expected: FAIL, `NotImplementedError`

- [ ] **Step 3: Implement fishbone**

Replace `src/labeler/events/panels/fishbone.py`:

```python
"""Fishbones, as a coherent n = 1 burst locked across two Mirnov probes."""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from ..raw import raw_signal
from ..verify import Panel

#: B1 and B5, the pair the cross-phase is taken over.
PROBES = (0, 4)
NPERSEG = 4096
MAX_HZ = 40_000.0
FISHBONE_BAND = (2.0, 30.0)

GUIDANCE = (
    "<b>What you are looking for:</b> bursts in 2-30 kHz (the shaded band) "
    "that chirp DOWNWARD over a few milliseconds, with a cross-phase that "
    "stays flat across the burst - that flatness is what says the two probes "
    "are seeing one coherent mode rather than two patches of turbulence."
)


def panels(shot, *, t_range=None, paths=None):
    mhr = raw_signal(
        int(shot), "mhr", channels=list(PROBES), t_range=t_range, paths=paths
    )
    # The rate comes from the SPAN, never from a median diff: xdata is
    # float32 and its spacing quantises at t ~ 3 s.
    rate = (mhr.x.shape[0] - 1) / ((mhr.x[-1] - mhr.x[0]) / 1000.0)
    nperseg = min(NPERSEG, mhr.x.shape[0])
    kwargs = {
        "fs": rate,
        "nperseg": nperseg,
        "noverlap": nperseg // 2,
        "mode": "complex",
    }
    # Cross-phase needs each probe's OWN complex spectrum: phase(spec_a *
    # conj(spec_b)) is the cross-spectrum's phase, the quantity that locks
    # to a coherent n = 1 mode. The phase of a spectrogram taken of a complex
    # signal built from the two real probes (a + i*b) is a different,
    # meaningless quantity, so each probe is transformed on its own.
    freq, times, spec_a = scipy_signal.spectrogram(mhr.y[0], **kwargs)
    _, _, spec_b = scipy_signal.spectrogram(mhr.y[1], **kwargs)
    cross_phase = np.angle(spec_a * np.conj(spec_b))
    # B1's own power, out of the complex spectrogram already computed rather
    # than a third `mode="psd"` pass over the same two million samples.
    # scipy's one-sided psd doubles every bin but DC and Nyquist, so this is
    # the same panel shifted by a constant log10(2) - and it is log-scaled.
    power = np.abs(spec_a) ** 2
    keep = freq <= MAX_HZ
    t_ms = times * 1000.0 + mhr.x[0]
    khz = freq[keep] / 1000.0
    return [
        Panel(
            title=f"mhr B{PROBES[0] + 1} spectrogram",
            kind="heatmap",
            x=t_ms,
            y=khz,
            z=np.log10(power[keep] + 1e-30),
            ylabel="kHz",
            bands=[FISHBONE_BAND],
        ),
        Panel(
            title=f"cross-phase B{PROBES[0] + 1} x B{PROBES[1] + 1}",
            kind="heatmap",
            x=t_ms,
            y=khz,
            z=cross_phase[keep],
            ylabel="kHz",
            bands=[FISHBONE_BAND],
        ),
    ]
```

- [ ] **Step 4: Implement sawtooth_oscillation**

Replace `src/labeler/events/panels/sawtooth_oscillation.py`:

```python
"""Sawteeth, as the inversion of adjacent ECE channels across the q = 1 surface."""

from __future__ import annotations

from ..raw import raw_signal
from ..verify import Panel

#: Four rows of four ADJACENT channels covering 20-35, sixteen in all. The
#: flip a sawtooth crash makes is a RELATIVE thing - inner channels drop as
#: outer ones rise - so channels are overplotted in adjacent groups rather
#: than drawn one per panel.
CHANNEL_ROWS = ((20, 21, 22, 23), (24, 25, 26, 27), (28, 29, 30, 31), (32, 33, 34, 35))

GUIDANCE = (
    "<b>What you are looking for:</b> a sawtooth ramp on the inner channels "
    "that collapses abruptly while the outer channels jump up at the same "
    "instant - the inversion across the q = 1 surface. A rise or fall that "
    "moves every channel the same way is not a sawtooth."
    "<br><br>A shot outside the corpus is fetched live, 48 ECE channels over "
    "MDSplus, which is slow the first time and cached afterwards."
)


def panels(shot, *, t_range=None, paths=None):
    built = []
    for row in CHANNEL_ROWS:
        array = raw_signal(
            int(shot), "ece", channels=list(row), t_range=t_range, paths=paths
        )
        built.append(
            Panel(
                title=f"ECE ch {row[0]}-{row[-1]}",
                x=array.x,
                y=array.y,
                ylabel="keV",
                legend=[f"ch {c}" for c in row],
            )
        )
    return built
```

- [ ] **Step 5: Implement minimum_safety_factor**

Replace `src/labeler/events/panels/minimum_safety_factor.py`:

```python
"""qmin, against the rule's own class thresholds."""

from __future__ import annotations

import numpy as np

from ...config import Paths
from ...features.store import read_feature
from ..verify import Panel

#: The rule's class boundaries, drawn as dashed lines. A `bands` entry 0.01
#: tall at 8% opacity, on an axis spanning ~0.8-3, is invisible; a threshold
#: is a line, so `hlines` draws one.
THRESHOLDS = [0.95, 1.5, 2.0]

GUIDANCE = (
    "<b>What you are looking for:</b> where the qmin trace crosses 0.95, 1.5 "
    "and 2.0 (the dashed lines), and whether the label boundary sits on the "
    "crossing. The q profile below is on normalized psi, <b>not rho</b>."
)


def panels(shot, *, t_range=None, paths=None):
    paths = Paths.from_env() if paths is None else paths
    features = paths.features_file(int(shot))
    qmin = read_feature(features, "qmin")
    qpsi = read_feature(features, "qpsi")
    ip = read_feature(features, "ip")

    def window(x, y):
        """Seconds on disk, milliseconds on screen, clipped to the window."""
        x = np.asarray(x) * 1000.0
        if t_range is None:
            return x, y
        keep = (x >= t_range[0]) & (x <= t_range[1])
        return x[keep], y[:, keep]

    qmin_x, qmin_y = window(qmin.x, qmin.y)
    qpsi_x, qpsi_y = window(qpsi.x, qpsi.y)
    ip_x, ip_y = window(ip.x, ip.y)
    return [
        Panel(title="qmin (EFIT01 aeqdsk)", x=qmin_x, y=qmin_y, ylabel="q",
              hlines=THRESHOLDS),
        Panel(
            title="q profile",
            kind="heatmap",
            x=qpsi_x,
            y=np.linspace(0.0, 1.0, qpsi_y.shape[0]),
            z=qpsi_y,
            # normalized psi, NOT rho - see features/namespace.py's note on
            # the qpsi radial axis
            ylabel="psi_n",
        ),
        Panel(title="ip", x=ip_x, y=ip_y, ylabel="A"),
    ]
```

- [ ] **Step 6: Run it and watch it pass**

Run: `pixi run -e labelmaker pytest tests/labeler/test_events_panels.py -q`
Expected: PASS, 11 passed

- [ ] **Step 7: Commit**

```bash
git add src/labeler/events/panels tests/labeler/test_events_panels.py
git commit -m "labeler: move the fishbone, sawtooth and qmin panels into the registry"
```

---

### Task 8: The app, its gate, and the two listing endpoints

**Files:**
- Create: `src/labeler/events/ui/__init__.py`, `src/labeler/events/ui/app.py`
- Test: `tests/labeler/test_events_ui.py`

**Interfaces:**
- Consumes: `panels.BUILDERS`, `panels.guidance`; `rosters.read_roster`, `rosters.roster_path`; `verify.corrections_for`.
- Produces: `create_app(paths: Paths | None = None, token: str | None = None) -> FastAPI`; `COOKIE = "labeler_verify_token"`.

- [ ] **Step 1: Write the failing test**

Create `tests/labeler/test_events_ui.py`:

```python
"""The browser review surface: its gate, its reads, and what it refuses to write."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from labeler.config import Paths
from labeler.events.ui.app import COOKIE, create_app

ROSTER = (
    "shot,tier,holdout,reviewers,verified_on,notes\n"
    "170815,gold,false,alice,2026-01-01,\n"
    "178642,unverified,false,,,\n"
)


@pytest.fixture
def tables(tmp_path):
    """A label_tables root holding one event with a roster."""
    event = tmp_path / "alfven_eigenmode"
    (event / "format").mkdir(parents=True)
    (event / "shots.csv").write_text(ROSTER)
    return tmp_path


@pytest.fixture
def client(tables, tmp_path):
    paths = Paths(
        label_tables=tables,
        corpus=tmp_path / "corpus",
        raw_cache=tmp_path / "cache",
    )
    app = create_app(paths=paths, token="secret")
    transport = TestClient(app)
    transport.cookies.set(COOKIE, "secret")
    return transport


def test_no_cookie_and_no_token_is_refused(tables, tmp_path):
    app = create_app(paths=Paths(label_tables=tables), token="secret")
    assert TestClient(app).get("/api/events").status_code == 401


def test_a_wrong_token_is_refused(tables, tmp_path):
    app = create_app(paths=Paths(label_tables=tables), token="secret")
    response = TestClient(app).get("/api/events?token=wrong")
    assert response.status_code == 401


def test_the_right_token_sets_the_cookie(tables, tmp_path):
    app = create_app(paths=Paths(label_tables=tables), token="secret")
    response = TestClient(app, follow_redirects=False).get("/api/events?token=secret")
    assert response.status_code == 303
    assert response.cookies[COOKIE] == "secret"


def test_events_lists_what_is_on_disk(client):
    payload = client.get("/api/events").json()
    names = [row["event"] for row in payload["events"]]
    assert "alfven_eigenmode" in names
    row = next(r for r in payload["events"] if r["event"] == "alfven_eigenmode")
    assert row["builder"] == "alfven_eigenmode", "not the generic fallback"
    assert row["n_shots"] == 2


def test_shots_returns_the_roster_with_correction_counts(client, tables):
    review = tables / "alfven_eigenmode" / "review"
    review.mkdir()
    (review / "178642__nc1514__20260918T120000Z.csv").write_text(
        "shot,category,t_start,t_end,notes\n178642,1,350,1300,\n"
    )
    rows = client.get("/api/shots?event=alfven_eigenmode").json()["shots"]
    by_shot = {row["shot"]: row for row in rows}
    assert by_shot[170815]["reviewers"] == ["alice"]
    assert by_shot[170815]["n_corrections"] == 0
    assert by_shot[178642]["n_corrections"] == 1
    assert by_shot[178642]["tier"] == "unverified"


def test_guidance_comes_back_with_the_event(client):
    payload = client.get("/api/events").json()
    row = next(r for r in payload["events"] if r["event"] == "alfven_eigenmode")
    assert "80-250 kHz" in row["guidance"]
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pixi run -e labelmaker pytest tests/labeler/test_events_ui.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'labeler.events.ui'`

- [ ] **Step 3: Implement**

Create `src/labeler/events/ui/__init__.py`:

```python
"""The browser review surface."""
```

Create `src/labeler/events/ui/app.py`:

```python
"""Browser transport for label verification.

The token gate and the static-app structure come from `shot_design/ui/app.py`,
which took them from shot-recommender-system. Panels come from the panel
registry and corrections go through `verify.write_corrections`, which is the
same append-only path the notebooks used.

This app READS `shots.csv` and never writes it. That file is how a person
tracks what has been processed, and it holds the hand-set `tier` and
`holdout` calls; a surface that edited it on a button press is a surface that
can destroy somebody's curation by accident. Note in particular that
`ReviewSession.save()` is NOT used here - it calls `record_review`, which
writes that file.
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from ...config import Paths
from .. import panels as registry
from .. import rosters
from ..verify import corrections_for

STATIC = Path(__file__).parent / "static"
COOKIE = "labeler_verify_token"


def _json(value, **kwargs) -> Response:
    return Response(
        json.dumps(value, default=str), media_type="application/json", **kwargs
    )


def _unauthorized(request: Request, why: str) -> Response:
    return JSONResponse({"error": why}, status_code=401)


def _events(paths: Paths) -> list[dict]:
    """Every event directory that has a roster, with its builder and guidance."""
    rows = []
    for directory in sorted(p for p in paths.label_tables.iterdir() if p.is_dir()):
        roster = directory / rosters.ROSTER_NAME
        if not roster.is_file():
            continue
        event = directory.name
        frame = rosters.read_roster(roster)
        rows.append(
            {
                "event": event,
                # "generic" here is the honest answer and the page says so:
                # a reviewer should know the panels in front of them were not
                # chosen for this phenomenon.
                "builder": event if event in registry.BUILDERS else "generic",
                "guidance": registry.guidance(event),
                "n_shots": int(len(frame)),
            }
        )
    return rows


def create_app(paths: Paths | None = None, token: str | None = None) -> FastAPI:
    paths = Paths.from_env() if paths is None else paths
    app = FastAPI(
        title="labeler verify", docs_url=None, redoc_url=None, openapi_url=None
    )
    app.state.paths = paths
    app.state.token = token or secrets.token_hex(16)

    @app.middleware("http")
    async def gate(request: Request, call_next):
        expected = app.state.token.encode("utf-8")
        given = request.query_params.get("token")
        if given is not None:
            if not secrets.compare_digest(given.encode("utf-8"), expected):
                return _unauthorized(request, "bad token")
            url = request.url.remove_query_params("token")
            path = url.path if not url.path.startswith("//") else "/"
            response = RedirectResponse(
                path + (f"?{url.query}" if url.query else ""), 303
            )
            response.set_cookie(COOKIE, app.state.token, httponly=True, samesite="lax")
            return response
        cookie = request.cookies.get(COOKIE, "")
        if not cookie or not secrets.compare_digest(cookie.encode("utf-8"), expected):
            return _unauthorized(
                request,
                "no token: reopen the link printed by the verify server",
            )
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/events")
    def events():
        return _json({"events": _events(app.state.paths)})

    @app.get("/api/shots")
    def shots(event: str):
        paths = app.state.paths
        frame = rosters.read_roster(rosters.roster_path(event, root=paths.label_tables))
        rows = []
        for record in frame.to_dict("records"):
            shot = int(record["shot"])
            rows.append(
                {
                    "shot": shot,
                    "tier": record.get("tier", ""),
                    "holdout": bool(record.get("holdout", False)),
                    "reviewers": rosters.split_reviewers(record.get("reviewers", "")),
                    "verified_on": record.get("verified_on", ""),
                    "notes": record.get("notes", ""),
                    # Files on disk, counted fresh: a review saved a moment
                    # ago shows here against an unrecorded reviewer, which is
                    # the prompt to go edit shots.csv by hand.
                    "n_corrections": len(
                        corrections_for(event, shot, root=paths.label_tables)
                    ),
                }
            )
        return _json({"event": event, "shots": rows})

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app
```

Create the static directory so the mount succeeds:

```bash
mkdir -p src/labeler/events/ui/static
printf '<!doctype html><title>labeler verify</title>\n' > src/labeler/events/ui/static/index.html
```

- [ ] **Step 4: Run it and watch it pass**

Run: `pixi run -e labelmaker pytest tests/labeler/test_events_ui.py -q`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/labeler/events/ui tests/labeler/test_events_ui.py
git commit -m "labeler: serve the event roster behind a token, read-only"
```

---

### Task 9: The panels endpoint and its window

**Files:**
- Modify: `src/labeler/events/ui/app.py`
- Test: `tests/labeler/test_events_ui.py` (append)

**Interfaces:**
- Produces: `GET /api/panels?event=&shot=&t0=&t1=` returning `{"panels": [...], "t_range": [t0, t1], "note": str}`; `_panel_json(panel) -> dict`.

- [ ] **Step 1: Write the failing test**

Append to `tests/labeler/test_events_ui.py`:

```python
def test_panels_honours_the_window_it_is_given(client, monkeypatch):
    from labeler.events.panels import alfven_eigenmode as ae
    from labeler.features.store import FeatureArray

    seen = {}

    def fake_raw_signal(shot, group, *, channels=None, t_range=None, paths=None):
        seen["t_range"] = t_range
        rate = 1_000_000.0
        n = 40_000
        start = 0.0 if t_range is None else t_range[0]
        return FeatureArray(
            x=start + np.arange(n) / rate * 1000.0,
            y=np.random.default_rng(4).normal(size=(4, n)).astype("float32"),
            attrs={},
        )

    monkeypatch.setattr(ae, "raw_signal", fake_raw_signal)
    payload = client.get(
        "/api/panels?event=alfven_eigenmode&shot=178642&t0=1200&t1=1500"
    ).json()
    assert seen["t_range"] == (1200.0, 1500.0)
    assert payload["t_range"] == [1200.0, 1500.0]
    assert len(payload["panels"]) == 3
    first = payload["panels"][0]
    assert first["kind"] == "heatmap"
    assert len(first["z"]) == len(first["y"])
    assert len(first["z"][0]) == len(first["x"])
    assert first["bands"] == [[80.0, 250.0]]


def test_panels_without_a_window_asks_for_the_whole_shot(client, monkeypatch):
    from labeler.events.panels import alfven_eigenmode as ae
    from labeler.features.store import FeatureArray

    seen = {}

    def fake_raw_signal(shot, group, *, channels=None, t_range=None, paths=None):
        seen["t_range"] = t_range
        return FeatureArray(
            x=np.arange(40_000.0) / 1000.0,
            y=np.zeros((4, 40_000), dtype="float32"),
            attrs={},
        )

    monkeypatch.setattr(ae, "raw_signal", fake_raw_signal)
    client.get("/api/panels?event=alfven_eigenmode&shot=178642")
    assert seen["t_range"] is None


def test_panels_reports_a_missing_signal_as_a_message_not_a_500(client):
    """The corpus and cache are both empty and there is no fdp here."""
    response = client.get("/api/panels?event=alfven_eigenmode&shot=999999")
    assert response.status_code == 502
    assert "999999" in response.json()["error"]


def test_panels_says_when_there_is_no_label_row(client, monkeypatch):
    from labeler.events.panels import alfven_eigenmode as ae
    from labeler.features.store import FeatureArray

    monkeypatch.setattr(
        ae, "raw_signal",
        lambda *a, **k: FeatureArray(
            x=np.arange(40_000.0) / 1000.0,
            y=np.zeros((4, 40_000), dtype="float32"),
            attrs={},
        ),
    )
    payload = client.get("/api/panels?event=alfven_eigenmode&shot=178642").json()
    assert payload["note"] == "NO LABEL ROW"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pixi run -e labelmaker pytest tests/labeler/test_events_ui.py -q -k panels`
Expected: FAIL, 404 from the static mount

- [ ] **Step 3: Implement**

In `src/labeler/events/ui/app.py`, add to the imports:

```python
import warnings

import numpy as np

from ..verify import NoDataError, label_panel
```

Add before `create_app`:

```python
def _panel_json(panel) -> dict:
    """One `Panel` as the plain arrays plotly.js wants.

    NaN is not JSON, and a label grid is full of it where a cell is unknown.
    `json.dumps` would emit a bare `NaN` token that `JSON.parse` rejects, so
    every non-finite value becomes `null` - which plotly draws as a gap,
    which is what an unknown cell is.
    """

    def clean(array):
        values = np.asarray(array, dtype="float64")
        return np.where(np.isfinite(values), values, None).tolist()

    payload = {
        "title": panel.title,
        "kind": panel.kind,
        "ylabel": panel.ylabel,
        "x": clean(panel.x),
        "bands": [[float(low), float(high)] for low, high in panel.bands],
        "hlines": [float(level) for level in panel.hlines],
        "zmin": None if panel.zmin is None else float(panel.zmin),
        "zmax": None if panel.zmax is None else float(panel.zmax),
    }
    if panel.kind == "heatmap":
        payload["y"] = clean(panel.y)
        payload["z"] = [clean(row) for row in panel.z]
    else:
        payload["y"] = [clean(row) for row in panel.y]
        payload["legend"] = list(panel.legend or
                                 [f"ch {i}" for i in range(len(panel.y))])
    return payload
```

Add the endpoint inside `create_app`, before the static mount:

```python
    @app.get("/api/panels")
    def panels_for(event: str, shot: int, t0: float | None = None,
                   t1: float | None = None):
        paths = app.state.paths
        t_range = None if t0 is None or t1 is None else (float(t0), float(t1))
        try:
            built = registry.build(event, shot, t_range=t_range, paths=paths)
        except (NoDataError, OSError, FileNotFoundError) as error:
            # A shot the corpus does not have and fdp cannot reach is an
            # ordinary outcome here, not a bug: say so at the top of the
            # page rather than dropping a traceback in the log.
            return _json({"error": str(error)}, status_code=502)

        # The label row is appended the way `verify.review` appends it, and
        # its absence is said on the page rather than only in a warning on
        # stderr, which is not where the reviewer is looking.
        note = ""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                built.append(
                    label_panel(event, shot, source="format/shots",
                                root=paths.label_tables)
                )
        except OSError:
            note = "NO LABEL ROW"

        return _json(
            {
                "event": event,
                "shot": int(shot),
                "t_range": None if t_range is None else list(t_range),
                "note": note,
                "panels": [_panel_json(p) for p in built],
            }
        )
```

- [ ] **Step 4: Run it and watch it pass**

Run: `pixi run -e labelmaker pytest tests/labeler/test_events_ui.py -q`
Expected: PASS, 11 passed

- [ ] **Step 5: Commit**

```bash
git add src/labeler/events/ui/app.py tests/labeler/test_events_ui.py
git commit -m "labeler: render panels over whatever window the browser asks for"
```

---

### Task 10: Saving corrections, and not saving anything else

**Files:**
- Modify: `src/labeler/events/ui/app.py`
- Test: `tests/labeler/test_events_ui.py` (append)

**Interfaces:**
- Produces: `POST /api/save` taking `{"event": str, "shot": int, "marks": [{"t_start": float, "t_end": float, "category": int}], "reviewer": str | None}`, returning `{"written": "<filename>", "n": int}`.

- [ ] **Step 1: Write the failing test**

Append to `tests/labeler/test_events_ui.py`:

```python
def test_save_writes_one_corrections_file(client, tables):
    response = client.post("/api/save", json={
        "event": "alfven_eigenmode",
        "shot": 178642,
        "reviewer": "nc1514",
        "marks": [{"t_start": 350.0, "t_end": 1300.0, "category": 1}],
    })
    assert response.status_code == 200
    written = tables / "alfven_eigenmode" / "review" / response.json()["written"]
    assert written.is_file()
    assert "178642,1,350" in written.read_text().replace(".0", "")


def test_a_second_save_writes_a_second_file_and_destroys_nothing(client, tables):
    body = {
        "event": "alfven_eigenmode",
        "shot": 178642,
        "reviewer": "nc1514",
        "marks": [{"t_start": 350.0, "t_end": 1300.0, "category": 1}],
    }
    first = client.post("/api/save", json=body).json()["written"]
    body["marks"] = [{"t_start": 400.0, "t_end": 900.0, "category": 0}]
    second = client.post("/api/save", json=body).json()["written"]
    assert first != second
    review = tables / "alfven_eigenmode" / "review"
    assert len(list(review.glob("178642__*.csv"))) == 2
    assert "350" in (review / first).read_text()


def test_save_does_not_touch_shots_csv(client, tables):
    """The guarantee the reviewer is given, asserted on the bytes.

    An accidental `ReviewSession.save()` here would call `record_review` and
    break this without any other test noticing.
    """
    roster = tables / "alfven_eigenmode" / "shots.csv"
    before = roster.read_bytes()
    client.post("/api/save", json={
        "event": "alfven_eigenmode",
        "shot": 178642,
        "reviewer": "nc1514",
        "marks": [{"t_start": 350.0, "t_end": 1300.0, "category": 1}],
    })
    assert roster.read_bytes() == before


def test_save_with_no_marks_is_refused(client):
    response = client.post("/api/save", json={
        "event": "alfven_eigenmode", "shot": 178642,
        "reviewer": "nc1514", "marks": [],
    })
    assert response.status_code == 400
    assert "no marks" in response.json()["error"]


def test_save_refuses_an_interval_that_runs_backwards(client):
    response = client.post("/api/save", json={
        "event": "alfven_eigenmode", "shot": 178642, "reviewer": "nc1514",
        "marks": [{"t_start": 1300.0, "t_end": 350.0, "category": 1}],
    })
    assert response.status_code == 400
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pixi run -e labelmaker pytest tests/labeler/test_events_ui.py -q -k save`
Expected: FAIL, 405 or 404

- [ ] **Step 3: Implement**

Add to `src/labeler/events/ui/app.py` imports:

```python
import os

from ..verify import correction_path, write_corrections
```

Add the endpoint inside `create_app`, before the static mount:

```python
    @app.post("/api/save")
    async def save(request: Request):
        import pandas as pd

        from ..interval_tables import INTERVAL_COLUMNS

        body = await request.json()
        event = str(body["event"])
        shot = int(body["shot"])
        marks = body.get("marks") or []
        if not marks:
            return _json(
                {"error": "no marks to save; drag a range and mark it first"},
                status_code=400,
            )
        rows = []
        for mark in marks:
            t_start, t_end = float(mark["t_start"]), float(mark["t_end"])
            if t_end < t_start:
                return _json(
                    {"error": f"t_end {t_end} precedes t_start {t_start}"},
                    status_code=400,
                )
            rows.append([shot, int(mark["category"]), t_start, t_end, ""])

        reviewer = body.get("reviewer") or os.environ.get("USER", "unknown")
        paths = app.state.paths
        # `correction_path` mints a path that is free and `write_corrections`
        # refuses one that is not, so two reviewers cannot collide and a
        # second save cannot erase a first. Deliberately NOT
        # `ReviewSession.save()`: that also calls `record_review`, which
        # edits shots.csv, and shots.csv is a file people edit by hand.
        target = correction_path(
            event, shot, reviewer=reviewer, root=paths.label_tables
        )
        frame = pd.DataFrame(rows, columns=list(INTERVAL_COLUMNS))
        try:
            write_corrections(frame, target)
        except (ValueError, FileExistsError, OSError) as error:
            return _json({"error": str(error)}, status_code=400)
        return _json({"written": target.name, "n": len(rows)})
```

- [ ] **Step 4: Run it and watch it pass**

Run: `pixi run -e labelmaker pytest tests/labeler/test_events_ui.py -q`
Expected: PASS, 16 passed

- [ ] **Step 5: Commit**

```bash
git add src/labeler/events/ui/app.py tests/labeler/test_events_ui.py
git commit -m "labeler: save corrections from the browser, and nothing else"
```

---

### Task 11: The page

**Files:**
- Create: `src/labeler/events/ui/static/index.html` (replace the stub), `static/app.js`, `static/style.css`
- Modify: `src/labeler/events/ui/app.py` (serve plotly.min.js)

**Interfaces:**
- Consumes: `/api/events`, `/api/shots`, `/api/panels`, `/api/save`.
- Produces: `GET /vendor/plotly.min.js`.

There is no unit test for the page; Task 13's acceptance run is its test.

- [ ] **Step 1: Serve plotly from the installed package**

The machine this runs on reaches the browser through an SSH forward and may
have no route to a CDN, so plotly is served off disk. Add to
`src/labeler/events/ui/app.py`, inside `create_app` before the static mount:

```python
    @app.get("/vendor/plotly.min.js")
    def plotly_js():
        # plotly's conda package ships the bundle it renders with. Serving
        # that file is what lets this page work on a node with no route off
        # the cluster, which is every node it will actually run on.
        import plotly

        bundle = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
        if not bundle.is_file():
            return _json({"error": f"no plotly bundle at {bundle}"}, status_code=500)
        return Response(
            bundle.read_bytes(),
            media_type="application/javascript",
            headers={"Cache-Control": "max-age=86400"},
        )
```

- [ ] **Step 2: Write the page**

Replace `src/labeler/events/ui/static/index.html`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>labeler verify</title>
<link rel="stylesheet" href="/style.css">
<script src="/vendor/plotly.min.js"></script>
</head>
<body>
<header>
  <h1>verify</h1>
  <label>event <select id="event"></select></label>
  <label>shot <select id="shot"></select></label>
  <span id="builder" class="tag"></span>
  <span id="tier" class="tag"></span>
</header>

<p id="guidance" class="guidance"></p>
<p id="status" class="status"></p>

<div id="figure"></div>

<section class="controls">
  <button id="present">Mark present</button>
  <button id="absent">Mark absent</button>
  <button id="save">Save</button>
  <span id="marks"></span>
</section>

<details class="what">
  <summary>What Save writes</summary>
  <p>
    One new file, <code>review/&lt;shot&gt;__&lt;reviewer&gt;__&lt;stamp&gt;.csv</code>,
    in the same five-column schema as <code>format/</code>. Corrections are
    append-only: every press writes its own file and nothing under
    <code>review/</code> is ever overwritten or deleted, so a second reviewer
    cannot destroy the first's work.
  </p>
  <p>
    It writes <b>nothing else</b>. Not <code>shots.csv</code> &mdash; that is
    edited by hand, and <code>tier</code> and <code>holdout</code> in it are
    curation calls nothing here derives. Not <code>format/</code> &mdash;
    merging rows into it is a manual step. And no raw data is promoted to
    EKOLEMEN; that is <code>labeler-raw promote</code>, on purpose.
  </p>
</details>

<section class="roster">
  <h2>roster</h2>
  <table id="roster"><tbody></tbody></table>
</section>

<script src="/app.js"></script>
</body>
</html>
```

Create `src/labeler/events/ui/static/style.css`:

```css
:root { --ink: #1a1a1a; --quiet: #666; --rule: #ddd; --warn: #b2182b; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 1rem 1.5rem; color: var(--ink);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
header { display: flex; gap: 1rem; align-items: baseline; flex-wrap: wrap; }
h1 { font-size: 1.1rem; margin: 0 1rem 0 0; font-weight: 600; }
h2 { font-size: .95rem; margin: 1.5rem 0 .5rem; font-weight: 600; }
select { font: inherit; padding: .2rem; }
.tag {
  font-size: .8rem; color: var(--quiet); border: 1px solid var(--rule);
  border-radius: 3px; padding: 0 .4rem;
}
.guidance {
  max-width: 62rem; border-left: 3px solid var(--rule);
  padding-left: .8rem; color: #333;
}
.status { min-height: 1.5rem; color: var(--quiet); }
.status.error { color: var(--warn); font-weight: 600; }
.controls { display: flex; gap: .5rem; align-items: center; margin: .5rem 0; }
button { font: inherit; padding: .35rem .8rem; cursor: pointer; }
button:disabled { opacity: .5; cursor: default; }
.what { max-width: 62rem; margin: 1rem 0; color: #333; }
.what summary { cursor: pointer; color: var(--quiet); }
table { border-collapse: collapse; font-size: .85rem; }
td, th { border-bottom: 1px solid var(--rule); padding: .25rem .7rem .25rem 0;
         text-align: left; }
tr.current { font-weight: 600; }
code { background: #f4f4f4; padding: 0 .2rem; }
```

Create `src/labeler/events/ui/static/app.js`:

```javascript
"use strict";

// State the page keeps: the marks not yet saved, and the window last drawn.
const marks = [];
let currentEvent = null;
let currentShot = null;
let rendering = false;
let pending = null;

const $ = (id) => document.getElementById(id);

function say(text, isError) {
  const status = $("status");
  status.textContent = text;
  status.classList.toggle("error", Boolean(isError));
}

async function getJSON(url) {
  const response = await fetch(url);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || response.statusText);
  return body;
}

function traceFor(panel, row) {
  // Each row gets its own legend, so a multi-row figure shows names beside
  // their own row instead of one merged box at the top.
  const legend = `legend${row}`;
  if (panel.kind === "heatmap") {
    return [{
      type: "heatmap", x: panel.x, y: panel.y, z: panel.z,
      zmin: panel.zmin, zmax: panel.zmax,
      showscale: false, showlegend: false,
      xaxis: row === 1 ? "x" : `x${row}`, yaxis: row === 1 ? "y" : `y${row}`,
    }];
  }
  return panel.y.map((values, i) => ({
    type: "scattergl", mode: "lines", x: panel.x, y: values,
    name: (panel.legend || [])[i] || `ch ${i}`, legend,
    xaxis: row === 1 ? "x" : `x${row}`, yaxis: row === 1 ? "y" : `y${row}`,
  }));
}

function draw(payload) {
  const panels = payload.panels;
  const traces = [];
  const layout = {
    title: `${payload.event} - shot ${payload.shot}` +
           (payload.note ? ` - ${payload.note}` : ""),
    height: 200 * panels.length + 120,
    dragmode: "select",
    selectdirection: "h",
    showlegend: true,
    margin: { l: 60, r: 20, t: 60, b: 40 },
    shapes: [],
    annotations: [],
    grid: { rows: panels.length, columns: 1, pattern: "independent" },
  };
  panels.forEach((panel, index) => {
    const row = index + 1;
    traces.push(...traceFor(panel, row));
    const ykey = row === 1 ? "yaxis" : `yaxis${row}`;
    const xkey = row === 1 ? "xaxis" : `xaxis${row}`;
    layout[ykey] = { title: { text: panel.ylabel } };
    layout[xkey] = { matches: "x", title: { text: row === panels.length ? "Time (ms)" : "" } };
    layout.annotations.push({
      text: panel.title, showarrow: false, xref: "paper", yref: `${row === 1 ? "y" : "y" + row} domain`,
      x: 0, y: 1, yanchor: "bottom", xanchor: "left", font: { size: 12 },
    });
    // A filled rect over a spectrogram washes the very image under review;
    // darker reads as less power on every sequential colormap, biasing the
    // reviewer toward under-calling the mode. Boundary lines mark the band
    // without touching what is inside it.
    panel.bands.forEach(([low, high]) => {
      [low, high].forEach((edge) => layout.shapes.push({
        type: "line", xref: "paper", x0: 0, x1: 1,
        yref: row === 1 ? "y" : `y${row}`, y0: edge, y1: edge,
        line: { width: 1, dash: "dot", color: "black" },
      }));
    });
    panel.hlines.forEach((level) => layout.shapes.push({
      type: "line", xref: "paper", x0: 0, x1: 1,
      yref: row === 1 ? "y" : `y${row}`, y0: level, y1: level,
      line: { width: 1, dash: "dash", color: "#b2182b" },
    }));
  });

  const figure = $("figure");
  Plotly.react(figure, traces, layout, { displaylogo: false, responsive: true });
  figure.removeAllListeners?.("plotly_relayout");
  figure.on("plotly_relayout", onRelayout);
}

// Panning and zooming re-render the visible window server-side, so a window
// is as sharp as the screen can show however wide it is. Debounced, because
// a drag emits relayout continuously.
let debounce = null;
function onRelayout(event) {
  const t0 = event["xaxis.range[0]"];
  const t1 = event["xaxis.range[1]"];
  if (t0 === undefined || t1 === undefined) return;
  clearTimeout(debounce);
  debounce = setTimeout(() => loadPanels(Number(t0), Number(t1)), 200);
}

async function loadPanels(t0, t1) {
  if (rendering) { pending = [t0, t1]; return; }
  rendering = true;
  const window = (t0 === undefined || t1 === undefined) ? "" : `&t0=${t0}&t1=${t1}`;
  say(t0 === undefined ? "loading the whole shot - a first fetch takes minutes"
                       : `rendering ${Math.round(t0)}-${Math.round(t1)} ms`);
  try {
    const payload = await getJSON(
      `/api/panels?event=${currentEvent}&shot=${currentShot}${window}`);
    draw(payload);
    say(payload.note ? payload.note : "");
  } catch (error) {
    say(String(error.message || error), true);
  } finally {
    rendering = false;
    if (pending) { const next = pending; pending = null; loadPanels(...next); }
  }
}

function selectedRange() {
  const figure = $("figure");
  const selections = figure.layout && figure.layout.selections;
  if (!selections || !selections.length) {
    throw new Error("drag a time range on the figure first");
  }
  // The modebar offers Lasso Select beside Box Select. A lasso writes
  // {type: "path"} with x0/x1 both null, so reject anything that is not a
  // plain rectangle before touching x0/x1.
  const box = selections[selections.length - 1];
  if (box.type !== "rect" || box.x0 === null || box.x0 === undefined) {
    throw new Error("use the Box Select tool, not Lasso");
  }
  return [Math.min(box.x0, box.x1), Math.max(box.x0, box.x1)];
}

function mark(category) {
  return () => {
    let range;
    try { range = selectedRange(); }
    catch (error) { say(error.message, true); return; }
    marks.push({ t_start: range[0], t_end: range[1], category });
    // Clear the drag so a second click cannot silently record the same
    // interval twice.
    Plotly.relayout($("figure"), { selections: [] });
    showMarks();
  };
}

function showMarks() {
  $("marks").textContent = marks.length
    ? `${marks.length} correction(s) not yet saved`
    : "no corrections yet";
  $("save").disabled = marks.length === 0;
}

async function save() {
  const response = await fetch("/api/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      event: currentEvent, shot: currentShot, marks,
    }),
  });
  const body = await response.json();
  if (!response.ok) { say(body.error, true); return; }
  marks.length = 0;
  showMarks();
  say(`saved ${body.n} correction(s) to ${body.written}`);
  loadShots(currentEvent, currentShot);
}

async function loadShots(event, keepShot) {
  const body = await getJSON(`/api/shots?event=${event}`);
  const select = $("shot");
  select.innerHTML = "";
  const rows = $("roster").querySelector("tbody");
  rows.innerHTML = "<tr><th>shot</th><th>tier</th><th>holdout</th>" +
                   "<th>reviewers</th><th>corrections on disk</th></tr>";
  body.shots.forEach((row) => {
    const option = document.createElement("option");
    option.value = row.shot;
    option.textContent = `${row.shot} (${row.tier})`;
    select.appendChild(option);
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td>${row.shot}</td><td>${row.tier}</td><td>${row.holdout}</td>` +
      `<td>${row.reviewers.join("; ") || "-"}</td><td>${row.n_corrections}</td>`;
    if (Number(row.shot) === Number(keepShot)) tr.className = "current";
    rows.appendChild(tr);
  });
  if (keepShot) select.value = String(keepShot);
  currentShot = Number(select.value);
  $("tier").textContent =
    (body.shots.find((r) => Number(r.shot) === currentShot) || {}).tier || "";
}

async function start() {
  const body = await getJSON("/api/events");
  const select = $("event");
  body.events.forEach((row) => {
    const option = document.createElement("option");
    option.value = row.event;
    option.textContent = row.event;
    option.dataset.guidance = row.guidance;
    option.dataset.builder = row.builder;
    select.appendChild(option);
  });

  async function pickEvent() {
    currentEvent = select.value;
    const option = select.selectedOptions[0];
    $("guidance").innerHTML = option.dataset.guidance;
    $("builder").textContent = option.dataset.builder === "generic"
      ? "generic panels" : "panels for this event";
    marks.length = 0;
    showMarks();
    await loadShots(currentEvent);
    loadPanels();
  }

  select.addEventListener("change", pickEvent);
  $("shot").addEventListener("change", () => {
    currentShot = Number($("shot").value);
    marks.length = 0;
    showMarks();
    loadPanels();
  });
  $("present").addEventListener("click", mark(1));
  $("absent").addEventListener("click", mark(0));
  $("save").addEventListener("click", save);
  await pickEvent();
}

start().catch((error) => say(String(error.message || error), true));
```

- [ ] **Step 3: Check the page is served and parses**

Run:

```bash
pixi run -e labelmaker python -c "
from fastapi.testclient import TestClient
from labeler.config import Paths
from labeler.events.ui.app import COOKIE, create_app
app = create_app(token='t')
c = TestClient(app); c.cookies.set(COOKIE, 't')
for path in ('/', '/app.js', '/style.css', '/vendor/plotly.min.js'):
    r = c.get(path)
    print(path, r.status_code, len(r.content))
    assert r.status_code == 200, path
"
```

Expected: four lines, each `200` with a non-zero length; the plotly bundle is megabytes.

- [ ] **Step 4: Run the whole suite**

Run: `pixi run -e labelmaker pytest tests/labeler -q`
Expected: PASS, no failures

- [ ] **Step 5: Commit**

```bash
git add src/labeler/events/ui
git commit -m "labeler: draw the panels, take the marks, say what Save writes"
```

---

### Task 12: The launcher

**Files:**
- Create: `src/labeler/events/ui/serve.py`, `src/labeler/events/ui/__main__.py`
- Modify: `pyproject.toml`
- Test: `tests/labeler/test_events_ui.py` (append)

**Interfaces:**
- Produces: `main(host: str | None = None, port: int | None = None, token: str | None = None) -> int`; `DEFAULT_PORT = 8811`.

- [ ] **Step 1: Write the failing test**

Append to `tests/labeler/test_events_ui.py`:

```python
def test_serve_refuses_any_host_but_the_loopback():
    from labeler.events.ui import serve

    with pytest.raises(ValueError, match="127.0.0.1"):
        serve.main(host="0.0.0.0")


def test_serve_prints_the_token_link_and_the_forward(monkeypatch, capsys):
    from labeler.events.ui import serve

    monkeypatch.setattr(serve, "_run", lambda app, host, port: 0)
    serve.main(token="secret", port=9999)
    printed = capsys.readouterr().out
    assert "http://127.0.0.1:9999/?token=secret" in printed
    assert "ssh -L 9999:localhost:9999" in printed
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pixi run -e labelmaker pytest tests/labeler/test_events_ui.py -q -k serve`
Expected: FAIL, `ImportError: cannot import name 'serve'`

- [ ] **Step 3: Implement**

Create `src/labeler/events/ui/serve.py`:

```python
"""Local launcher. Binds the loopback; remote users connect over SSH."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlencode

DEFAULT_PORT = 8811

#: The wrapper every live fdp fetch has to run under. Checked here rather
#: than at the first fetch, because the first fetch is minutes into a review
#: of a shot the corpus does not have, and failing there wastes all of it.
FDP_COMMAND = "pixi run -e labelmaker fdp run python -m labeler.events.ui"


def _run(app, host: str, port: int) -> int:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info", access_log=False)
    return 0


def main(
    host: str | None = None,
    port: int | None = None,
    token: str | None = None,
) -> int:
    from .app import create_app

    host = "127.0.0.1" if host is None else host
    port = DEFAULT_PORT if port is None else int(port)
    if host != "127.0.0.1":
        raise ValueError(
            "the verify server binds only to 127.0.0.1; use an SSH forward"
        )
    app = create_app(token=token)
    query = urlencode({"token": app.state.token})
    print(f"verify: http://127.0.0.1:{port}/?{query}", flush=True)
    print(f"ssh -L {port}:localhost:{port} stellar", flush=True)
    if not os.environ.get("FDP_WRAPPED") and not Path("/usr/local/mdsplus").exists():
        print(
            "note: this process does not look like it is under the fdp "
            f"wrapper. A shot outside the corpus will fail to fetch "
            f"(PTSERVER / TREE-E-FOPENR). Restart as:\n  {FDP_COMMAND}",
            flush=True,
        )
    return _run(app, host, port)
```

Create `src/labeler/events/ui/__main__.py`:

```python
"""Run the browser verification surface."""

import argparse

from .serve import DEFAULT_PORT, main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="labeler.events.ui")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()
    raise SystemExit(main(port=args.port, token=args.token))
```

- [ ] **Step 4: Add the dependencies and the task**

In `pyproject.toml`, under `[tool.pixi.feature.labelmaker.dependencies]`, beside the existing `plotly` pin:

```toml
# The verification surface is a FastAPI app on the loopback, reached over an
# SSH forward. Same ranges as feature.ideate, which runs shot_design's UI.
fastapi = ">=0.115,<1"
uvicorn = ">=0.30,<1"
```

Under `[tool.pixi.feature.labelmaker.tasks]`, beside `labeler-raw`:

```toml
# `pixi run -e labelmaker fdp run labeler-verify`: the browser review
# surface. The fdp wrapper is what lets a shot outside the corpus fetch.
labeler-verify = "python -m labeler.events.ui"
```

- [ ] **Step 5: Install and run it and watch it pass**

Run:

```bash
pixi install -e labelmaker
pixi run -e labelmaker pytest tests/labeler -q
```

Expected: PASS, no failures

- [ ] **Step 6: Commit**

```bash
git add src/labeler/events/ui pyproject.toml pixi.lock tests/labeler/test_events_ui.py
git commit -m "labeler: launch the verify server on the loopback behind a token"
```

---

### Task 13: Retire the notebooks, and prove it on a real shot

**Files:**
- Delete: `data/events/*/verification.ipynb` (16 files), `data/events/alfven_eigenmode/review/_cache/`
- Modify: `data/events/*/README.md` (16 files), `data/events/alfven_eigenmode/example.ipynb`, `.gitignore`
- Modify: `CLAUDE.md` if it mentions the notebooks

**Interfaces:** none; this task ships.

- [ ] **Step 1: Run the acceptance before deleting anything**

This is the point of the whole plan, and it is the only check the unit tests
cannot make. Start the server under the wrapper:

```bash
pixi run -e labelmaker fdp run labeler-verify
```

Forward the port from your machine with the `ssh -L` line it prints, open
the token link, and confirm each of these against **alfven_eigenmode, shot
178642** (a measured-good fetch case, with a label grid under
`format/shots`; its labels run 0-1950 ms, present 350-1300 ms):

- [ ] the page opens and the event picker lists all sixteen events
- [ ] picking `alfven_eigenmode` shows the 80-250 kHz guidance, not the generic text
- [ ] the shot draws three crosspower panels plus the label row, **over the whole shot** rather than 0-2000 ms
- [ ] `.cache/raw/178642_processed.h5` now exists and is roughly 240 MB:
      `ls -lh .cache/raw/`
- [ ] reloading the page redraws in seconds and does not refetch
- [ ] zooming into 350-500 ms re-renders sharper than the whole-shot view
- [ ] box-select 350-1300 ms, *Mark present*, *Save* writes a file:
      `ls data/events/alfven_eigenmode/review/`
- [ ] pressing *Save* again writes a **second** file and the first still has its rows
- [ ] `shots.csv` is unchanged: `git diff --stat data/events/alfven_eigenmode/shots.csv` is empty
- [ ] `pixi run -e labelmaker labeler-raw promote 178642` refuses and names the missing groups
- [ ] `pixi run -e labelmaker labeler-raw promote 178642 --partial` moves it, and the page still draws that shot

If any of these fail, fix it before continuing. Do not delete the notebooks
against a surface that does not work.

- [ ] **Step 2: Delete the notebooks**

```bash
git rm data/events/*/verification.ipynb
git rm -r --cached data/events/alfven_eigenmode/review/_cache 2>/dev/null || true
rm -rf data/events/alfven_eigenmode/review/_cache
```

- [ ] **Step 3: Point each README at the server**

For every directory under `data/events/` that has a `README.md`, replace any
paragraph describing `verification.ipynb` with:

```markdown
## Verification

```bash
pixi run -e labelmaker fdp run labeler-verify
```

Open the token link it prints (over the `ssh -L` forward it also prints),
pick this event and a shot, drag a time range, and press *Mark present* or
*Mark absent* then *Save*.

Save writes one new file, `review/<shot>__<reviewer>__<stamp>.csv`, and
nothing else. Corrections are append-only: every press writes its own file
and nothing under `review/` is ever overwritten, so a second reviewer cannot
destroy the first's work. Merging those rows into `format/` is a manual step,
and `shots.csv` — including the hand-set `tier` and `holdout` — is edited by
hand.

Which traces this event shows is decided in
`src/labeler/events/panels/<event>.py`. An event with no module there gets
generic `ip`/`betan`/`pinj_total` panels, which show that the shot exists but
not whether the phenomenon happened; the page says so.

A shot outside the corpus is fetched live and cached under `.cache/raw/`,
about 240 MB apiece with no cap. `pixi run -e labelmaker labeler-raw clean`
empties it; `labeler-raw promote <shot>` moves one into
`/scratch/gpfs/EKOLEMEN/foundation_model`, where bulk raw data lives
long-term.
```

- [ ] **Step 4: Cut the foreign cache path out of example.ipynb**

In `data/events/alfven_eigenmode/example.ipynb`, the cell defining `TS_CACHE
= Path("/scratch/gpfs/nc1514/aemodes/data/.cache/ae_timeseries")` and
`read_co2` reads a different project's feather cache. Replace that function
and constant with the one raw path:

```python
from labeler.events.raw import raw_signal

def read_co2(shot):
    """The four CO2 chords and their rate, from wherever the shot lives."""
    co2 = raw_signal(int(shot), "co2")
    # The rate comes from the SPAN, never from a median diff: a float32 time
    # vector quantises its spacing at t ~ 3 s.
    rate = (len(co2.x) - 1) / ((co2.x[-1] - co2.x[0]) / 1000.0)
    return co2.y, rate
```

Delete the `import pyarrow.feather as feather` line and the `TS_CACHE`
constant. Run the notebook top to bottom and confirm the AE figure still
renders.

- [ ] **Step 5: Confirm the suite and the tree**

```bash
pixi run -e labelmaker pytest tests/labeler -q
git status --short | grep -c verification.ipynb
grep -rn "aemodes/data/.cache" data/ src/ || echo "no foreign cache paths left"
grep -rn "review/_cache" data/ src/ || echo "no npz cache references left"
```

Expected: the suite passes; the notebook count is 16 deletions; both greps
report their "none left" message.

- [ ] **Step 6: Commit**

```bash
git add -A data/events src/labeler CLAUDE.md
git commit -m "labeler: retire the verification notebooks for the browser surface

The sixteen notebooks held panel code that could not be imported or tested,
a window fixed at 0-2000 ms, and three unrelated raw-data paths - one of them
pointing into a different project's scratch directory. All three now live in
labeler.events.{raw,panels,ui}, verified end to end on alfven_eigenmode
shot 178642."
```

---

## Self-Review

**Spec coverage.** `raw.py` three tiers → Tasks 2-3. `.cache/raw` root → Task 1.
Additive and atomic writes → Task 2. `promote` with the partial refusal and
`clean` → Task 4. Panel registry with `_generic` fallback and `guidance` →
Task 5. The four bespoke builders → Tasks 6-7. Token gate, `/api/events`,
`/api/shots` with correction counts → Task 8. `/api/panels` with the window →
Task 9. `/api/save`, corrections only, plus the byte-level `shots.csv`
assertion → Task 10. The page, re-render on zoom, the lasso and
clear-selection guards, the "what Save writes" block → Task 11. Loopback
launcher, token link, fdp warning, `fastapi`/`uvicorn`, pixi tasks → Task 12.
Notebook deletion, READMEs, the `aemodes` path, acceptance → Task 13.

**Deviations from the spec, deliberate.** The spec's `/api/shots` unions the
roster with shots found in `format/`; Task 8 serves the roster alone, because
`rosters.read_roster` is the validated reader and a union needs a
`format/`-scanning function that does not exist yet. `alfven_eigenmode/shots.csv`
currently holds three example rows, so **the shot picker will be empty of real
shots until those rows are replaced by hand** — which is consistent with
shots.csv being a manual file, but means Task 13's acceptance needs 178642
added to that roster first. Flagged rather than silently designed around.

**Type consistency.** `panels(shot, *, t_range, paths)` and `GUIDANCE` are the
builder contract in Tasks 5, 6 and 7. `raw_signal(shot, group, *, channels,
t_range, paths)` is called identically in Tasks 2, 3, 6 and 7. `Paths` is
passed explicitly everywhere rather than read from the environment inside a
request.
