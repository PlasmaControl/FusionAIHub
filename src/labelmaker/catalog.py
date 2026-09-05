"""Which shots exist and which of them a run should touch.

The proof-of-concept pool is the intersection of the corpus with the
tearing-mode training archive: only there can a reconstructed feature row be
compared against the row the model was actually trained on, which is what
makes Task 15 and Task 16 possible.
"""
from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np

from .config import Paths

#: Filtered training arrays of the Phase 1 model: x0 (11 scalars), x1
#: (33 x 5 profiles), y (betan, tm_label), z (shot per row).
TM_ARCHIVE = Path("/projects/EKOLEMEN/tm_data")

_SHOT_FILE = re.compile(r"^(\d+)_processed\.h5$")


def corpus_shots(paths: Paths) -> list[int]:
    """Every shot with a corpus file, sorted."""
    out = []
    for p in paths.corpus.iterdir():
        m = _SHOT_FILE.match(p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def corpus_groups(path) -> dict[str, int]:
    """Group name -> number of time samples, for one corpus file.

    A length of 1 is the corpus' "signal absent" sentinel (a `(C, 1)`
    placeholder), documented in
    src/tokamak_foundation_model/data/multi_file_dataset.py:845-861.
    """
    with h5py.File(path, "r") as f:
        return {
            name: int(f[name]["ydata"].shape[-1])
            for name in f
            if "ydata" in f[name]
        }


def available_groups(path) -> set[str]:
    """Corpus groups that actually carry a series."""
    return {name for name, n in corpus_groups(path).items() if n > 1}


def archive_shots(archive: Path = TM_ARCHIVE) -> np.ndarray:
    """Sorted unique shots present in the tearing-mode training arrays."""
    z = np.load(archive / "z.npy")
    return np.unique(z.astype(np.int64))


def overlap_shots(paths: Paths, archive: Path = TM_ARCHIVE) -> list[int]:
    """Shots in both the corpus and the training archive."""
    return sorted(set(corpus_shots(paths)) & set(archive_shots(archive).tolist()))


def sample_shots(shots, n: int, seed: int) -> list[int]:
    """A reproducible sample of the distinct shots; all of them when `n` is larger."""
    shots = sorted({int(s) for s in shots})
    if n >= len(shots):
        return shots
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(shots), size=n, replace=False)
    return sorted(shots[i] for i in idx)


def write_shot_file(path, shots) -> None:
    """One shot per line, sorted - the unit of a reproducible run."""
    Path(path).write_text("\n".join(str(s) for s in sorted(shots)) + "\n")


def read_shot_file(path) -> list[int]:
    """Shots from a file, ignoring blank lines and `#` comments."""
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(int(line))
    return sorted(out)
