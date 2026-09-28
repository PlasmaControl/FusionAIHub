"""Where the paper's build draws, and how its output takes the old one's place.

The build draws into a fresh directory beside `out` (`staging_dir`) and swaps
it in whole at the end (`swap`), the products and the manifest together, so a
failure leaves `out` as it was. The swap claims only the names it is given (the
build's `OWNED`). Once the old output is renamed aside, every other entry of it
is renamed into the new output: moved, never copied, and a symlink moves as
itself, never followed. The old output is deleted only once the new one is in
place. A swap that fails moves those entries back and renames the old output
back; one that cannot (another build's output landed at `out` meanwhile, say)
deletes nothing and raises `Stranded`, which names where everything is. After
a failure the build deletes the staging directory only if nothing but its own
files is left in it (`discard`).
"""

from __future__ import annotations

import contextlib
import errno
import os
import shutil
import tempfile
from collections.abc import Collection, Mapping
from pathlib import Path


def staging_dir(out: Path) -> Path:
    """A fresh directory beside `out` to draw into."""
    for k in range(1000):
        staged = out.with_name(f".{out.name}.staging-{os.getpid()}-{k}")
        try:
            staged.mkdir(parents=True)
            return staged
        except FileExistsError:
            continue
    raise FileExistsError(f"{out}: no free staging directory beside it")


class Stranded(OSError):
    """The swap could not finish, nor put the old output back (another build
    put its own at `out` meanwhile, say). Nothing was deleted: `old` holds the
    old output, `new` the new one, and `entries` maps each of the owner's
    entries of `out` to where it is. `errno` and `strerror` are the failing
    call's; the message names all of it."""

    def __init__(
        self,
        out: Path,
        old: Path,
        new: Path,
        *,
        entries: Mapping[str, Path] | None = None,
        cause: BaseException | None = None,
    ):
        super().__init__(
            getattr(cause, "errno", None), getattr(cause, "strerror", None)
        )
        self.out, self.old, self.new = out, old, new
        self.entries = dict(entries or {})
        moved = sorted(n for n, p in self.entries.items() if p.parent == new)
        self.message = (
            f"{out}: neither the new output nor the old one could be put there, "
            f"so nothing was deleted: the old output is in {old}, the new one "
            f"in {new}"
        )
        if moved:
            rest = f" and the rest in {old}" if len(moved) < len(self.entries) else ""
            self.message += (
                f"; of the other entries of {out}, {', '.join(moved)} "
                f"{'is' if len(moved) == 1 else 'are'} in {new}{rest}"
            )
        if self.strerror:
            self.message += f" ({self.strerror})"

    def __str__(self) -> str:
        return self.message


def _move(src: Path, dst: Path) -> None:
    """Rename `src` to `dst`, which must not exist; a symlink moves as itself."""
    if os.path.lexists(dst):
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(dst))
    src.rename(dst)


def swap(staged: Path, out: Path, *, owned: Collection[str]) -> None:
    """Put `staged` where `out` is. `out` is renamed into a holder beside it,
    each of its entries not in `owned` is renamed from there into `staged`,
    and `staged` is renamed to `out`, all under one `try`; only then is the
    holder, with the old products and manifest, deleted. On a failure the
    entries are moved back and the old output renamed back to `out`; if that
    fails too, everything stays where `Stranded` says."""
    if not os.path.lexists(out):
        staged.rename(out)
        return
    holder = Path(tempfile.mkdtemp(prefix=f".{out.name}.old-", dir=out.parent))
    old = holder / out.name
    theirs: list[str] = []
    moved: set[str] = set()
    try:
        out.rename(old)
        theirs = sorted(p.name for p in old.iterdir() if p.name not in owned)
        for name in theirs:
            _move(old / name, staged / name)
            moved.add(name)
        staged.rename(out)
    except BaseException:
        _undo(staged, out, old, theirs, moved)
        raise
    shutil.rmtree(holder, ignore_errors=True)


def _undo(
    staged: Path, out: Path, old: Path, theirs: list[str], moved: set[str]
) -> None:
    """Undo a failed `swap`: the owner's entries back from `staged` into the
    old output, then the old output back to `out`."""
    holder = old.parent

    def in_staged(name: str) -> bool:
        """The owner's `name` is in `staged`: recorded as moved, or moved just
        before a signal kept the record from being made."""
        return os.path.lexists(staged / name) and (
            name in moved or not os.path.lexists(old / name)
        )

    if not os.path.lexists(old):  # `out` never moved
        with contextlib.suppress(OSError):
            holder.rmdir()
        return
    if not os.path.lexists(staged):  # the new output reached `out` after all
        shutil.rmtree(holder, ignore_errors=True)
        return
    try:
        for name in reversed(theirs):
            if in_staged(name):
                _move(staged / name, old / name)
        old.rename(out)
    except BaseException as undo:
        entries = {n: (staged if in_staged(n) else old) / n for n in theirs}
        raise Stranded(out, old, staged, entries=entries, cause=undo) from undo
    with contextlib.suppress(OSError):
        holder.rmdir()  # empty again: the old output is back at `out`


def discard(staged: Path, *, owned: Collection[str]) -> None:
    """Delete a failed build's staging directory: its files in `owned` and
    their `.tmp` siblings, then the directory, which stays if anything else is
    left in it (an owner's entry a failed swap could not move back, say)."""
    for name in owned:
        for path in (staged / name, staged / f"{name}.tmp"):
            with contextlib.suppress(OSError):
                path.unlink()
    with contextlib.suppress(OSError):
        staged.rmdir()
