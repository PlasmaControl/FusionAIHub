"""`write_blurbs(..., workers=N)`: a thread pool for the `agy` backfill, writing rows
back in shot order regardless of thread scheduling, with the same per-shot failure
handling as the in-process (`workers=1`) path.

There is no `tmp_db` fixture in conftest.py; the smallest equivalent is built here from
the existing `shot_record`/`write_db` helpers (the same ones `shot_design_db` uses),
reusing the `paths` fixture for the on-disk layout.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from shot_design.shotdb.build import write_blurbs

from .conftest import shot_record, write_db

GOOD = (
    "The session ran as planned. It completed without incident. "
    "No notable findings were logged."
)


class FakeClient:
    """Stands in for `LLMClient`: a gate-passing reply after a short sleep, tracking
    the peak number of calls in flight so the test can tell a thread pool ran."""

    def __init__(self, content: str = GOOD, fail_shots: frozenset[int] = frozenset()):
        self.cfg = {
            "provider": "fake",
            "models": {"quality": "gemini-fake"},
            "blurb": {"model": "quality", "max_words": 90, "prompt_version": 6},
        }
        self.content = content
        self.fail_shots = fail_shots
        self._lock = threading.Lock()
        self._in_flight = 0
        self.max_concurrent = 0
        self.calls = 0

    def available(self) -> tuple[bool, str]:
        return True, ""

    def chat(self, messages, tools=None, model=None, temperature=None, max_tokens=None,
              cache=None):
        with self._lock:
            self._in_flight += 1
            self.max_concurrent = max(self.max_concurrent, self._in_flight)
            self.calls += 1
        try:
            time.sleep(0.05)
            # The shot number is the first line of the user message
            # (`blurb.source_text`'s "Shot <n>..." header).
            shot = int(messages[1]["content"].split()[1].rstrip(",()."))
            if shot in self.fail_shots:
                raise RuntimeError(f"synthetic failure for shot {shot}")
            return SimpleNamespace(content=self.content)
        finally:
            with self._lock:
                self._in_flight -= 1


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def tmp_db(paths):
    """Six shots with default (template) blurbs, so `write_blurbs` has enough concurrent
    work for a 4-worker pool to overlap -- built straight from `write_db`/`shot_record`
    rather than the full `build.build()` pipeline, which needs raw HDF5 files this test
    does not care about.
    """
    records = [
        shot_record(100 + i, "r1", 1.0e6 + i * 1e4, 5.0e6, f"log entry {i}")
        for i in range(6)
    ]
    write_db(paths.db_dir, records)
    return SimpleNamespace(paths=paths)


def test_write_blurbs_workers_writes_every_row(tmp_db, fake_client):
    n = write_blurbs(tmp_db.paths, fake_client, workers=4)
    df = pd.read_parquet(tmp_db.paths.db_dir / "shots.parquet")
    assert n == len(df) and (df["blurb_source"] == "llm").all()
    assert fake_client.max_concurrent >= 2


def test_write_blurbs_workers_write_order_matches_single_worker(tmp_db):
    """The parquet a 4-worker run writes must be byte-identical to what `workers=1`
    writes for the same inputs: results are collected from the futures and then written
    back in shot order, not completion order."""
    single_dir = tmp_db.paths.db_dir
    write_blurbs(tmp_db.paths, FakeClient(), workers=1)
    single_bytes = (single_dir / "shots.parquet").read_bytes()

    # A second, identically-seeded database (same shots, same order), backfilled with
    # workers=4 -- `.model_copy` reuses every other Paths field untouched.
    multi_dir = single_dir.parent / "db_multi"
    multi_dir.mkdir()
    records = [
        shot_record(100 + i, "r1", 1.0e6 + i * 1e4, 5.0e6, f"log entry {i}")
        for i in range(6)
    ]
    write_db(multi_dir, records)
    multi_paths = tmp_db.paths.model_copy(update={"db_dir": multi_dir})
    write_blurbs(multi_paths, FakeClient(), workers=4)
    multi_bytes = (multi_dir / "shots.parquet").read_bytes()

    assert single_bytes == multi_bytes


def test_write_blurbs_workers_one_shot_failure_does_not_kill_the_run(tmp_db):
    """A `client.chat` exception on one shot must fall back to the template for that
    shot only (the same per-shot handling `_blurb.make` already gives `workers=1`), and
    every other shot must still get its `llm` blurb."""
    client = FakeClient(fail_shots=frozenset({103}))
    n = write_blurbs(tmp_db.paths, client, workers=4)
    df = pd.read_parquet(tmp_db.paths.db_dir / "shots.parquet")
    assert n == len(df) - 1
    assert df.loc[103, "blurb_source"] == "template"
    assert set(df.drop(index=103)["blurb_source"]) == {"llm"}
