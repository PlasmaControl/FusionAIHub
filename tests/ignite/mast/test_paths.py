from pathlib import Path

import pytest

from tokamak_foundation_model.ignite.mast import paths


def test_defaults(monkeypatch):
    monkeypatch.delenv("IGNITE_MAST_ROOT", raising=False)
    monkeypatch.delenv("IGNITE_MAST_TOKAMARK", raising=False)
    assert paths.mast_root() == Path("/models/intermediate/mast")
    assert paths.tokamark_zarr_root() == Path("/models/mast/tokamark/v1")
    assert paths.cache_dir() == Path("/models/intermediate/mast/cache/v1")


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path))
    assert paths.stage0_dir() == tmp_path / "stage0"


def test_refuses_root_inside_repo():
    repo = Path(paths.__file__).resolve().parents[4]
    with pytest.raises(ValueError, match="inside the repository"):
        paths.prepare_write_root(repo / "scratch_out")


def test_prepare_write_root_announces_and_creates(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path))
    out = paths.prepare_write_root(paths.stage0_dir())
    assert out.is_dir() and "IGNITE_MAST_ROOT set" in capsys.readouterr().out
