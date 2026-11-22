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


def test_the_announcement_does_not_credit_the_root_variable_for_a_path_outside_the_root(monkeypatch, tmp_path, capsys):
    """A path that is not under mast_root() did not come from IGNITE_MAST_ROOT or from the default root."""
    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path / "root"))
    paths.prepare_write_root(tmp_path / "elsewhere" / "out")
    said = capsys.readouterr().out
    assert "not under the stage root" in said
    assert "IGNITE_MAST_ROOT set" not in said and "(default)" not in said


@pytest.fixture
def no_mkdir(tmp_path, monkeypatch):  # tmp_path first: it is made with mkdir
    """Fail loudly on any directory creation, so that a refusal that is missing cannot write into a protected tree.

    (A first version of these tests, run before the refusal existed, left an empty directory under
    /models/tokamind: prepare_write_root creates what it is given. That directory was removed.)
    """
    def refuse(self, *args, **kwargs):
        raise AssertionError(f"mkdir({self}) was reached: the path should have been refused first")

    monkeypatch.setattr(Path, "mkdir", refuse)


def test_refuses_a_root_inside_tokamind(no_mkdir):
    """/models/tokamind holds the replication queue; nothing of this stage may be written there.

    The refusal is asserted only: nothing is created under /models/tokamind.
    """
    with pytest.raises(ValueError, match="tokamind"):
        paths.prepare_write_root(Path("/models/tokamind/scratch_out"))
    with pytest.raises(ValueError, match="tokamind"):
        paths.prepare_write_root(Path("/models/tokamind"))


def test_refuses_a_root_inside_the_tokamark_data(no_mkdir, monkeypatch, tmp_path):
    """TokaMark's Zarr store is read-only data. Asserted by the refusal only; nothing is created there."""
    store = tmp_path / "tokamark_store"
    monkeypatch.setenv("IGNITE_MAST_TOKAMARK", str(store))
    with pytest.raises(ValueError, match="TokaMark"):
        paths.prepare_write_root(store / "scores")
    with pytest.raises(ValueError, match="TokaMark"):
        paths.prepare_write_root(store)
    assert not store.exists()


def test_a_link_into_tokamind_does_not_get_round_the_refusal(no_mkdir, tmp_path):
    """The check is on the resolved path. The link is made in tmp_path; nothing is created under /models/tokamind."""
    if not Path("/models/tokamind").is_dir():
        pytest.skip("/models/tokamind is not here")
    link = tmp_path / "link"
    link.symlink_to("/models/tokamind")
    with pytest.raises(ValueError, match="tokamind"):
        paths.prepare_write_root(link / "scratch_out")


def test_a_sibling_whose_name_starts_alike_is_allowed(monkeypatch, tmp_path):
    """Only the directory itself and what is inside it are refused, not a name that begins the same way."""
    monkeypatch.setenv("IGNITE_MAST_TOKAMARK", str(tmp_path / "tokamark_store"))
    assert paths.prepare_write_root(tmp_path / "tokamark_store_scores").is_dir()
