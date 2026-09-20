"""`shotdb.build.frame_codes_dirs`: production's cache first, then our own writes, then
the pinned v4 bundle -- one source of truth `build`'s `has_frame_codes` column, `corpus
select`'s preferred-shots set and `program_reference._cache_path` all read, so they
cannot disagree about the same shot (see the controller amendment in task-C3-brief.md's
Step 0).
"""

from __future__ import annotations

from pathlib import Path

from shot_design.shotdb import build, ignite


def test_frame_codes_dirs_production_cache_first_then_v4_bundle(paths, monkeypatch):
    # Evaluated BEFORE the patch and captured by closure -- `lambda: ignite.model_cfg()`
    # inside its own replacement would recurse forever once `ignite.model_cfg` names
    # this lambda.
    cfg = {**ignite.model_cfg(), "frame_codes_cache": "/prod/frame_codes"}
    monkeypatch.setattr(ignite, "model_cfg", lambda: cfg)
    dirs = build.frame_codes_dirs(paths)
    assert dirs[0] == Path("/prod/frame_codes")
    assert dirs[1] == Path(paths.data_root) / "frame_codes"
    assert dirs[2] == ignite.bundle_dir(paths) / "frame_codes"
    assert Path(paths.models_dir) / "IGNITE" / "frame_codes" not in dirs


def test_frame_codes_dirs_without_a_production_cache(paths, monkeypatch):
    cfg = {k: v for k, v in ignite.model_cfg().items() if k != "frame_codes_cache"}
    monkeypatch.setattr(ignite, "model_cfg", lambda: cfg)
    dirs = build.frame_codes_dirs(paths)
    assert dirs == (
        Path(paths.data_root) / "frame_codes",
        ignite.bundle_dir(paths) / "frame_codes",
    )


def test_tests_never_see_the_production_cache():
    """Pins `conftest.py`'s autouse `_no_production_frame_codes_cache` fixture: every
    test gets the pinned yaml WITHOUT `frame_codes_cache`, so a real shot number in the
    production range cannot silently read production's read-only corpus instead of a
    test's own tmp tree. (This assertion has to live in a collected test module, not in
    conftest.py itself -- pytest's default `python_files` pattern does not collect
    `conftest.py` as a test file.)
    """
    assert "frame_codes_cache" not in ignite.model_cfg()
