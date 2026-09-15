"""A pilot rebuild cannot silently replace a larger or differently selected DB."""

import json

import numpy as np
import pytest

from shot_design import cli
from shot_design.shotdb import build, text


@pytest.fixture
def guard_db(paths, staged_shot_a, staged_shot_b, text_fixtures, monkeypatch):
    monkeypatch.setattr(
        text, "embed_texts", lambda texts: np.zeros((len(texts), 384), np.float32)
    )
    build.build(
        [staged_shot_a, staged_shot_b], paths, build.load_build_cfg(),
        workers=1, encode=False, shot_source="list:recommender_v1",
    )
    return paths


def snapshot(root):
    return {
        str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    "source,shots", [("shots:1", [900001]), ("list:other", [900001, 900002]),
                     ("list:recommender_v1", [900001])],
)
def test_guard_refuses_without_touching_existing_files(guard_db, source, shots):
    before = snapshot(guard_db.data_root)
    with pytest.raises(ValueError) as exc:
        build.build(
            shots, guard_db, build.load_build_cfg(), workers=1,
            encode=False, shot_source=source,
        )
    message = str(exc.value)
    for value in ("list:recommender_v1", source, "2", str(len(shots)), "--force"):
        assert value in message
    assert snapshot(guard_db.data_root) == before


@pytest.mark.parametrize("shots", [[900001, 900002], [900001, 900002, 900003]])
def test_same_source_equal_or_larger_rebuild_still_publishes(guard_db, shots):
    build.build(
        shots, guard_db, build.load_build_cfg(), workers=1,
        encode=False, shot_source="list:recommender_v1",
    )
    manifest = json.loads((guard_db.db_dir / "manifest.json").read_text())
    assert manifest["n_shots"] == len(shots)
    assert "forced_over" not in manifest


def test_guard_uses_successfully_built_count(guard_db, monkeypatch):
    real = build.build_record

    def failed(shot, *args, **kwargs):
        if shot != 900001:
            raise ValueError("synthetic record failure")
        return real(shot, *args, **kwargs)

    monkeypatch.setattr(build, "build_record", failed)
    before = snapshot(guard_db.data_root)
    with pytest.raises(ValueError, match="--force"):
        build.build(
            [900001, 900002, 900003], guard_db, build.load_build_cfg(), workers=1,
            encode=False, shot_source="list:recommender_v1",
        )
    assert snapshot(guard_db.data_root) == before


@pytest.mark.parametrize("blob", [b"{broken", b"\xff", b"[]", b"{}",
                                  b'{"shot_source": "list:recommender_v1", "n_shots": "2"}'])
def test_corrupt_manifest_requires_force_and_records_unknown_history(guard_db, blob):
    mf = guard_db.db_dir / "manifest.json"
    mf.write_bytes(blob)
    before = snapshot(guard_db.data_root)
    with pytest.raises(ValueError, match="--force"):
        build.build(
            [900001], guard_db, build.load_build_cfg(), workers=1,
            encode=False, shot_source="shots:1",
        )
    assert snapshot(guard_db.data_root) == before
    build.build(
        [900001], guard_db, build.load_build_cfg(), workers=1,
        encode=False, shot_source="shots:1", force=True,
    )
    history = json.loads(mf.read_text())["forced_over"]
    assert set(history) == {"shot_source", "n_shots", "built_at"}


def test_unreadable_manifest_refuses_before_writing(guard_db, monkeypatch):
    mf = guard_db.db_dir / "manifest.json"
    real = type(mf).read_text

    def unreadable(path, *args, **kwargs):
        if path == mf:
            raise PermissionError("unreadable manifest")
        return real(path, *args, **kwargs)

    before = snapshot(guard_db.data_root)
    monkeypatch.setattr(type(mf), "read_text", unreadable)
    with pytest.raises(ValueError, match="--force"):
        build.build(
            [900001], guard_db, build.load_build_cfg(), workers=1,
            encode=False, shot_source="shots:1",
        )
    assert snapshot(guard_db.data_root) == before


def test_cli_force_records_previous_database_and_add_remains_an_upsert(guard_db, capsys):
    mf = guard_db.db_dir / "manifest.json"
    old = json.loads(mf.read_text())
    args = ["build", "--shots", "900001", "--workers", "1", "--no-encode"]
    assert cli.main(args) == 1
    error = capsys.readouterr().err
    assert "list:recommender_v1" in error and "shots:1" in error and "--force" in error
    assert cli.main([*args, "--force"]) == 0
    new = json.loads(mf.read_text())
    assert new["n_shots"] == 1
    assert new["forced_over"] == {
        key: old[key] for key in ("shot_source", "n_shots", "built_at")
    }
    assert cli.main(["add", "900002", "--workers", "1"]) == 0
    assert json.loads(mf.read_text())["n_shots"] == 2
