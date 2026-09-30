"""Each file the paper's build reads, read once and pinned by its sha256.

A `Snapshot` reads an input's bytes once, hashes them, and parses them from
those bytes, so the sha256 the manifest pins is that of the bytes drawn. A
second read of a key is refused: it could pin other bytes than the ones drawn.
At the end of the build `changed` hashes every input again, and one that changed
while the build ran goes in the manifest's `changed_during_build`.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pandas as pd

from ..ae.xpower.train import read_split
from ..config import sha256_of
from ..events.review import labels
from . import shots


class Snapshot:
    """Each input read once: its bytes hashed, then parsed from those bytes, so
    the manifest pins what was drawn. A second read of a key is refused, so its
    pin stays the bytes drawn. `changed` hashes every input again."""

    def __init__(self, scratch: Path):
        self.scratch = scratch  # for a parser that needs a file: the bytes' copy
        self.pinned: dict[str, tuple[Path, str]] = {}

    def read(self, key: str, path: Path) -> bytes:
        """`path`'s bytes, pinned under `key`; a `ValueError` if `key` was read."""
        if key in self.pinned:
            raise ValueError(
                f"{key}: already read, from {self.pinned[key][0]}; a second read "
                "could pin other bytes than the ones drawn"
            )
        data = Path(path).read_bytes()
        self.pinned[key] = (Path(path), hashlib.sha256(data).hexdigest())
        return data

    def sha(self, key: str) -> str | None:
        return self.pinned[key][1] if key in self.pinned else None

    def json(self, key: str, path: Path) -> dict:
        return json.loads(self.read(key, path))

    def csv(self, key: str, path: Path) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(self.read(key, path)))

    def split(self, key: str, path: Path) -> dict[int, str]:
        return read_split(path, data=self.read(key, path))

    def copy(self, key: str, path: Path) -> Path:
        """`path`'s bytes, pinned under `key`, as a file of the same name in the
        scratch directory, for a parser that needs a path."""
        copy = self.scratch / key / Path(path).name
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(self.read(key, path))
        return copy

    def labels(self, key: str, path: Path) -> dict:
        """A `review/labels.csv`, parsed as the review parses it."""
        event = self.scratch / key
        copy = labels.labels_path(event)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(self.read(key, path))
        return labels.read_saved(event)

    def model(self, key: str, path: Path, split) -> shots.Model:
        return shots.Model.load(io.BytesIO(self.read(key, path)), split)

    def segmentation(self, key: str, path: Path) -> shots.Segmentation:
        return shots.Segmentation.load(io.BytesIO(self.read(key, path)))

    def changed(self) -> dict[str, dict]:
        """The inputs whose bytes are no longer the ones drawn."""
        found = {}
        for key, (path, drawn) in self.pinned.items():
            now = sha256_of(path) if path.is_file() else None
            if now != drawn:
                found[key] = {"path": str(path), "drawn": drawn, "now": now}
        return found
