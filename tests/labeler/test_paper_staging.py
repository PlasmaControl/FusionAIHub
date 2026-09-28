"""The paper's swap: the owner's entries of `out` are moved, never copied or
followed, and nothing of theirs is deleted when a swap fails."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from labeler.config import Paths
from labeler.paper import build, staging

from . import ae_tree

OWNED = build.OWNED


@pytest.fixture(autouse=True)
def paths(tmp_path, monkeypatch):
    """A temporary Paths, set before every call, though the swap reads none."""
    paths = Paths(
        root=tmp_path / "root",
        label_tables=tmp_path / "events",
        corpus=tmp_path / "corpus",
    )
    ae_tree.env(monkeypatch, paths)
    return paths


def _outputs(base: Path) -> tuple[Path, Path]:
    """An old output at `base/paper`, with the owner's `notes.txt`, and a new
    one staged beside it."""
    out = base / "paper"
    out.mkdir(parents=True)
    (out / "fig_scores.pdf").write_text("old")
    (out / "manifest.json").write_text('{"old": true}')
    (out / "notes.txt").write_text("the owner's")
    staged = staging.staging_dir(out)
    (staged / "fig_scores.pdf").write_text("new")
    (staged / "manifest.json").write_text('{"new": true}')
    return out, staged


def _tree(root: Path) -> dict[str, str | None]:
    """Every entry under `root`, never following a link: a file's text, a
    link's target, or None for a directory."""
    found = {}
    for top, dirs, files in os.walk(root):
        for name in [*dirs, *files]:
            p = Path(top, name)
            key = p.relative_to(root).as_posix()
            if p.is_symlink():
                found[key] = f"-> {os.readlink(p)}"
            else:
                found[key] = None if p.is_dir() else p.read_text()
    return found


def _beside(out: Path) -> list[str]:
    """`out` and any staging or holding directory beside it."""
    return sorted(p.name for p in out.parent.iterdir() if out.name in p.name)


def test_the_new_output_takes_the_owners_entries(tmp_path):
    out, staged = _outputs(tmp_path)
    (out / "drafts").mkdir()
    (out / "drafts" / "one.txt").write_text("a draft")
    staging.swap(staged, out, owned=OWNED)
    assert _tree(out) == {
        "drafts": None,
        "drafts/one.txt": "a draft",
        "fig_scores.pdf": "new",
        "manifest.json": '{"new": true}',
        "notes.txt": "the owner's",
    }
    assert _beside(out) == ["paper"]


def test_an_entry_written_as_the_swap_starts_survives(tmp_path, monkeypatch):
    """The reviewer's swap_probe case 2: the owner writes into `out` just
    before it is renamed aside."""
    out, staged = _outputs(tmp_path)
    rename = Path.rename

    def late(self, target):
        if Path(self) == out:
            (out / "late_notes.txt").write_text("written during the swap")
        return rename(self, target)

    with monkeypatch.context() as patched:
        patched.setattr(Path, "rename", late)
        staging.swap(staged, out, owned=OWNED)
    assert (out / "late_notes.txt").read_text() == "written during the swap"
    assert (out / "notes.txt").read_text() == "the owner's"
    assert (out / "fig_scores.pdf").read_text() == "new"
    assert _beside(out) == ["paper"]


def _write_at(held: int, name: str, text: str) -> None:
    """Write `name` through the directory handle `held`, not through a path."""
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644, dir_fd=held)
    with os.fdopen(fd, "w") as file:
        file.write(text)


def _swap_with_a_late_write(out, staged, monkeypatch, name, text) -> Path | None:
    """The reviewer's probe B: a writer holding the old output (a shell's cwd
    in `out`, say) writes `name` into it after the swap has listed it, just
    before the new output's rename. The swap's answer."""
    held = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
    rename = Path.rename

    def writes_late(self, target):
        if Path(self) == staged:
            _write_at(held, name, text)
        return rename(self, target)

    try:
        with monkeypatch.context() as patched:
            patched.setattr(Path, "rename", writes_late)
            return staging.swap(staged, out, owned=OWNED)
    finally:
        os.close(held)


NEW = {
    "fig_scores.pdf": "new",
    "manifest.json": '{"new": true}',
    "notes.txt": "the owner's",
}


def test_an_entry_written_late_through_a_held_handle_survives(tmp_path, monkeypatch):
    """Only the build's names are deleted from the old output: `late.txt`,
    written into it after the listing, is moved into the new output, and no
    holder is left."""
    out, staged = _outputs(tmp_path)
    late = "written after the listing"
    kept = _swap_with_a_late_write(out, staged, monkeypatch, "late.txt", late)
    assert _tree(out) == NEW | {"late.txt": late}
    assert kept is None
    assert _beside(out) == ["paper"]


def test_a_late_entry_whose_name_is_taken_keeps_the_holder(tmp_path, monkeypatch):
    """The late entry is a `notes.txt`, which the new output already holds (the
    owner's, moved there): it is neither deleted nor moved over the owner's, so
    the old output's directory is kept, holding it alone, and the swap names
    its holder."""
    out, staged = _outputs(tmp_path)
    late = "saved again, late"
    kept = _swap_with_a_late_write(out, staged, monkeypatch, "notes.txt", late)
    assert _tree(out) == NEW
    assert kept is not None and kept.name.startswith(".paper.old-")
    assert _tree(kept) == {"paper": None, "paper/notes.txt": late}
    assert _beside(out) == sorted(["paper", kept.name])


def test_the_old_outputs_own_names_go_and_no_link_is_followed(tmp_path):
    """Of the old output only the build's names are deleted: one that is a real
    directory with what it holds, one that is a link as itself, its target
    untouched."""
    out, staged = _outputs(tmp_path)
    (out / "fig_scores.pdf").unlink()
    (out / "fig_scores.pdf").mkdir()
    (out / "fig_scores.pdf" / "inside.txt").write_text("in an owned directory")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep.txt").write_text("not the build's")
    (out / "fig_mhd.pdf").symlink_to(elsewhere, target_is_directory=True)
    assert staging.swap(staged, out, owned=OWNED) is None
    assert _tree(out) == NEW
    assert _tree(elsewhere) == {"keep.txt": "not the build's"}, "never followed"
    assert _beside(out) == ["paper"]


def test_a_symlink_stays_a_symlink(tmp_path):
    """The reviewer's swap_probe case 1, with a link to a file and a broken
    one: each is moved as itself, never followed or copied."""
    out, staged = _outputs(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "big.dat").write_text("x" * 1000)
    links = {
        "linked": elsewhere,
        "filelink": elsewhere / "big.dat",
        "broken": tmp_path / "nowhere",
    }
    for name, target in links.items():
        (out / name).symlink_to(target, target_is_directory=name == "linked")
    staging.swap(staged, out, owned=OWNED)
    for name, target in links.items():
        assert (out / name).is_symlink(), name
        assert os.readlink(out / name) == str(target), name
    assert _tree(elsewhere) == {"big.dat": "x" * 1000}, "the target is untouched"
    assert _beside(out) == ["paper"]


def test_a_failed_first_rename_leaves_out_and_no_holder(tmp_path, monkeypatch):
    """`out.rename(holder)` itself fails: the error is the rename's, `out` is
    as it was, and the holder is removed (the staged output is the caller's)."""
    out, staged = _outputs(tmp_path)
    before = _tree(out)
    rename = Path.rename

    def refuses_out(self, target):
        if Path(self) == out:
            raise OSError(errno.EBUSY, "out is busy")
        return rename(self, target)

    with monkeypatch.context() as patched:
        patched.setattr(Path, "rename", refuses_out)
        with pytest.raises(OSError, match="out is busy") as caught:
            staging.swap(staged, out, owned=OWNED)
    assert not isinstance(caught.value, staging.Stranded)
    assert _tree(out) == before
    assert _beside(out) == [staged.name, "paper"], "no holder is left"


def test_a_failed_swap_moves_the_owners_entries_back(tmp_path, monkeypatch):
    """The new output's rename fails: the owner's entries, a link among them,
    are moved back, and `out` is as it was."""
    out, staged = _outputs(tmp_path)
    (tmp_path / "elsewhere").mkdir()
    (out / "linked").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    before = _tree(out)
    rename = Path.rename

    def refuses_staged(self, target):
        if Path(self) == staged:
            raise OSError(errno.EIO, "the new output's rename failed")
        return rename(self, target)

    with monkeypatch.context() as patched:
        patched.setattr(Path, "rename", refuses_staged)
        with pytest.raises(OSError, match="the new output's rename failed"):
            staging.swap(staged, out, owned=OWNED)
    assert _tree(out) == before
    assert set(_tree(staged)) == {"fig_scores.pdf", "manifest.json"}


def test_a_signal_between_the_renames_puts_the_old_output_back(tmp_path, monkeypatch):
    """A signal just after `out` is renamed aside: one `try` covers both
    renames, so the old output is put back and no holder is left unnamed."""
    out, staged = _outputs(tmp_path)
    before = _tree(out)
    rename = Path.rename

    def interrupted(self, target):
        moved = rename(self, target)
        if Path(self) == out:
            raise KeyboardInterrupt
        return moved

    with monkeypatch.context() as patched:
        patched.setattr(Path, "rename", interrupted)
        with pytest.raises(KeyboardInterrupt):
            staging.swap(staged, out, owned=OWNED)
    assert _tree(out) == before
    assert _beside(out) == [staged.name, "paper"]


def test_a_signal_mid_move_still_moves_the_entry_back(tmp_path, monkeypatch):
    """A signal after an entry's rename, before the swap records the move: the
    entry is found in `staged` and moved back all the same."""
    out, staged = _outputs(tmp_path)
    before = _tree(out)
    rename = Path.rename

    def interrupted(self, target):
        moved = rename(self, target)
        if Path(target) == staged / "notes.txt":
            raise KeyboardInterrupt
        return moved

    with monkeypatch.context() as patched:
        patched.setattr(Path, "rename", interrupted)
        with pytest.raises(KeyboardInterrupt):
            staging.swap(staged, out, owned=OWNED)
    assert _tree(out) == before
    assert set(_tree(staged)) == {"fig_scores.pdf", "manifest.json"}


def test_stranded_carries_the_errno_and_names_where_each_entry_is(
    tmp_path, monkeypatch
):
    """The new output's rename fails and so does a move back: nothing more is
    done, and `Stranded` has the failing call's errno and strerror and names
    where each of the owner's entries is."""
    out, staged = _outputs(tmp_path)
    (out / "zz_notes.txt").write_text("the owner's too")
    rename = Path.rename

    def fails(self, target):
        if Path(self) == staged:
            raise OSError(errno.EIO, "the new output's rename failed")
        if Path(self) == staged / "notes.txt":
            raise OSError(errno.EACCES, "the move back failed")
        return rename(self, target)

    with monkeypatch.context() as patched:
        patched.setattr(Path, "rename", fails)
        with pytest.raises(staging.Stranded) as caught:
            staging.swap(staged, out, owned=OWNED)
    error = caught.value
    assert (error.errno, error.strerror) == (errno.EACCES, "the move back failed")
    assert error.entries == {
        "notes.txt": staged / "notes.txt",
        "zz_notes.txt": error.old / "zz_notes.txt",
    }
    assert (staged / "notes.txt").read_text() == "the owner's"
    assert (error.old / "zz_notes.txt").read_text() == "the owner's too"
    assert (error.old / "fig_scores.pdf").read_text() == "old"
    for named in (out, error.old, staged, "notes.txt", "the move back failed"):
        assert str(named) in str(error), named
    assert not os.path.lexists(out)


def test_a_failed_build_deletes_only_its_own_files(tmp_path):
    """After a failure the build deletes the staging directory only when it
    holds nothing but the build's names (and their `.tmp` files)."""
    _, staged = _outputs(tmp_path)
    (staged / "fig_scores.png.tmp").write_text("half written")
    staging.discard(staged, owned=OWNED)
    assert not os.path.lexists(staged)
    _, kept = _outputs(tmp_path / "again")
    (kept / "notes.txt").write_text("the owner's, not moved back")
    staging.discard(kept, owned=OWNED)
    assert _tree(kept) == {"notes.txt": "the owner's, not moved back"}
