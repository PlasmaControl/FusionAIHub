"""`build` times each of its phases, logs them and records them in the manifest.

Three SLURM rebuilds ran at 65-68 % CPU on 3-6 cores and nobody could say which phase left the
cores idle, because a build reported one number: total elapsed. The per-phase wall times are the
measurement that turns that into a question with an answer, and they belong in the manifest so
the answer survives the job's log.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from ideate.shotdb import build, text


@pytest.fixture
def timed_db(paths, staged_shot_a, staged_shot_b, text_fixtures, monkeypatch):
    monkeypatch.setattr(
        text, "embed_texts", lambda texts: np.zeros((len(texts), 384), np.float32)
    )
    return paths


def run_build(paths):
    build.build(
        [900001, 900002], paths, build.load_build_cfg(), workers=1, encode=False,
        shot_source="list:recommender_v1",
    )
    return json.loads((paths.db_dir / "manifest.json").read_text())


def test_the_manifest_carries_one_wall_time_per_phase_in_the_order_they_ran(timed_db):
    phases = run_build(timed_db)["phase_seconds"]
    assert list(phases) == list(build.PHASES)
    assert all(type(v) is float and v >= 0.0 for v in phases.values())


def test_the_published_manifest_times_the_publish_itself(timed_db):
    """The publish is the last thing that happens, so its time cannot be in the manifest that
    the publish moves -- the published one has to be the one that carries it."""
    phases = run_build(timed_db)["phase_seconds"]
    assert phases["publish"] > 0.0
    assert phases["read_records"] > 0.0


def test_the_phase_times_are_logged_when_the_build_ends(timed_db, caplog):
    with caplog.at_level(logging.INFO, logger="ideate.shotdb.build"):
        run_build(timed_db)
    logged = [r.getMessage() for r in caplog.records if "phase" in r.getMessage()]
    assert logged, "the build logged no phase timing line"
    assert all(phase in logged[-1] for phase in build.PHASES)
