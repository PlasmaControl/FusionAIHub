"""What `build` stages outside the database directory, and what a crash in a rename window
leaves behind.

Both cases here are consequences of the publish guard: the text subset is now built into a
staging directory so a refused rebuild cannot have touched the cache, and the manifest is now
written a second time -- through a `.part` file renamed over the published one -- so the publish
can time itself. Each new rename is a new way to fail.
"""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import numpy as np
import pytest

from shot_design.shotdb import build, text


@pytest.fixture
def staging_db(paths, staged_shot_a, text_fixtures, monkeypatch):
    monkeypatch.setattr(
        text, "embed_texts", lambda texts: np.zeros((len(texts), 384), np.float32)
    )
    return paths


def test_the_staged_text_subset_lands_when_the_cache_is_on_another_filesystem(
    staging_db, staged_shot_a, monkeypatch
):
    """The staging directory is a sibling of db_dir, but an `SHOT_DESIGN_PATHS` file -- the scratch
    database workflow docs/SHOT_DESIGN.md recommends -- is free to put `text_cache_dir` on a different
    mount, and `os.replace` across mounts raises EXDEV. The subset must still land."""
    subset = text.subset_path(staging_db)
    real = os.replace

    def replace(src, dst, *args, **kwargs):
        if Path(dst) == subset:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    build.build(
        [staged_shot_a], staging_db, build.load_build_cfg(), workers=1,
        encode=False, shot_source="shots:1",
    )
    assert subset.exists(), "the staged text subset was lost on a cross-device move"
    shots = {json.loads(line)["shot"] for line in subset.read_text().splitlines() if line}
    assert staged_shot_a in shots


def test_publish_prunes_a_manifest_part_orphaned_by_a_crash(tmp_path):
    """A crash between writing `manifest.json.part` and renaming it over `manifest.json` leaves
    the staging file in the published database. It is build-owned, so the next publish is what
    has to remove it -- nothing else ever will. A file this module does not write is untouched,
    as always."""
    db, staging = tmp_path / "db", tmp_path / "db.tmp"
    db.mkdir()
    staging.mkdir()
    (db / "manifest.json.part").write_text('{"half": "written"}')
    (db / "corpus_coverage.parquet").write_text("another producer's file")
    (staging / "manifest.json").write_text('{"n_shots": 1}')
    build._publish(staging, db)
    assert not (db / "manifest.json.part").exists()
    assert (db / "corpus_coverage.parquet").exists()
    assert json.loads((db / "manifest.json").read_text()) == {"n_shots": 1}
