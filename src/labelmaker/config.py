"""Every filesystem root labelmaker reads or writes.

Nothing else in the package hard-codes a path, so pointing labelmaker at a
different data root (a scratch copy, a test fixture) is one environment
variable. Defaults are group storage: Nathan's own scratch is near quota.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
DEFAULT_CORPUS = Path("/scratch/gpfs/EKOLEMEN/foundation_model")


@dataclass(frozen=True)
class Paths:
    """Where labelmaker's inputs and outputs live."""

    root: Path = DEFAULT_ROOT
    corpus: Path = DEFAULT_CORPUS

    @classmethod
    def from_env(cls) -> Paths:
        return cls(
            root=Path(os.environ.get("LABELMAKER_ROOT", str(DEFAULT_ROOT))),
            corpus=Path(os.environ.get("LABELMAKER_CORPUS", str(DEFAULT_CORPUS))),
        )

    @property
    def features(self) -> Path:
        return self.root / "features"

    @property
    def labels(self) -> Path:
        return self.root / "labels"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def validation(self) -> Path:
        return self.root / "validation"

    @property
    def labels_index(self) -> Path:
        return self.root / "labels_index.parquet"

    def features_file(self, shot: int) -> Path:
        return self.features / f"{shot}_features.h5"

    def labels_file(self, shot: int) -> Path:
        return self.labels / f"{shot}_labels.h5"

    def corpus_file(self, shot: int) -> Path:
        return self.corpus / f"{shot}_processed.h5"

    def mkdirs(self) -> None:
        for d in (self.features, self.labels, self.models, self.runs, self.validation):
            d.mkdir(parents=True, exist_ok=True)


def git_sha() -> str:
    """Short sha of the checkout that produced an artifact, or 'unknown'."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"
