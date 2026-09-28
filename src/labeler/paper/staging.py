"""Where the paper's build draws, and how its output takes the old one's place.

The build draws into a fresh directory beside `out` (`staging_dir`) and swaps
it in whole at the end (`swap`), the products and the manifest together, so a
failure leaves `out` as it was. The swap claims only the names it is given (the
build's `OWNED`); every other entry of `out` is copied into the new output
first, so it stays. The old output is deleted only once the new one is in
place; a swap that can neither finish nor put the old output back (another
build's output landed at `out` meanwhile, say) deletes nothing and raises
`Stranded`, which names where each one is.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from collections.abc import Collection
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
    """The new output could not be put at `out`, nor the old one back (another
    build put its own there meanwhile, say). Nothing was deleted: `old` holds
    the old output and `new` the new one, and the message names both."""

    def __init__(self, out: Path, old: Path, new: Path):
        self.out, self.old, self.new = out, old, new
        super().__init__(
            f"{out}: neither the new output nor the old one could be put there, "
            f"so nothing was deleted: the old output is in {old}, the new one "
            f"in {new}"
        )


def swap(staged: Path, out: Path, *, owned: Collection[str]) -> None:
    """Put `staged` where `out` is, in two renames. The old products and
    manifest leave together; every other entry of `out` (not in `owned`) is
    copied over first. The old output is deleted only once the new one is in
    place: if the second rename fails, the old one is renamed back, and if that
    fails too, both stay where `Stranded` says."""
    if not out.exists():
        staged.rename(out)
        return
    for kept in out.iterdir():
        if kept.name not in owned and not (staged / kept.name).exists():
            if kept.is_dir():
                shutil.copytree(kept, staged / kept.name, symlinks=True)
            else:
                shutil.copy2(kept, staged / kept.name, follow_symlinks=False)
    holder = Path(tempfile.mkdtemp(prefix=f".{out.name}.old-", dir=out.parent))
    old = holder / out.name
    try:
        out.rename(old)
    except BaseException:
        with contextlib.suppress(OSError):
            holder.rmdir()  # never used: `out` did not move
        raise
    try:
        staged.rename(out)
    except BaseException:
        try:
            old.rename(out)
        except BaseException as undo:
            raise Stranded(out, old, staged) from undo
        with contextlib.suppress(OSError):
            holder.rmdir()  # empty again: the old output is back at `out`
        raise
    shutil.rmtree(holder, ignore_errors=True)
