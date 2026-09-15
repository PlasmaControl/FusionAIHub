"""Every filesystem root labelmaker reads or writes.

Nothing else in the package hard-codes a path, so pointing labelmaker at a
different data root (a scratch copy, a test fixture) is one environment
variable. Defaults are group storage: Nathan's own scratch is near quota.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
DEFAULT_CORPUS = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
#: The per-shot operator-text bundles `events/text_weak.py` reads. A third
#: read-only input root beside the corpus, and separate from it because it
#: is a different group's directory and covers a different set of shots.
DEFAULT_TEXT = Path(
    "/scratch/gpfs/EKOLEMEN/big_d3d_data/foundation_model_text"
    "/shotsummary/processed/per_shot_txt"
)
#: The logbook dump `events/text_weak.py` takes its SHOT-scope text from:
#: one 616 MB file, one JSON record per line, every line beginning
#: `{"shot": <digits>,`. Read-only, like the corpus and the bundles; what
#: labelmaker writes is the subset of it for the shots in hand, under
#: `text_cache`.
DEFAULT_LOGS_JSONL = Path(
    "/scratch/gpfs/EKOLEMEN/big_d3d_data/foundation_model_text/sql/logs.jsonl"
)
#: The curated label tables `events/databases.py` reads: a `tables.yaml`
#: manifest and one directory of CSVs per phenomenon. Label DATA, so it
#: lives under `data/events/` and never under `src/`, and the default
#: resolves relative to this FILE rather than to the caller's cwd - a run
#: from a SLURM scratch directory finds the committed manifest the same way
#: a run from the repo does. That resolution is a SOURCE CHECKOUT's (which
#: is every way labelmaker is run today: `PYTHONPATH=$PWD/src`, or an
#: editable install); from a non-editable wheel the tables are outside the
#: package and `LABELMAKER_LABEL_TABLES` is the answer. Overridable anyway,
#: because a table too large or too restricted to commit lives on /scratch.
DEFAULT_LABEL_TABLES = Path(__file__).resolve().parents[2] / "data" / "events"


@dataclass(frozen=True)
class Paths:
    """Where labelmaker's inputs and outputs live."""

    root: Path = DEFAULT_ROOT
    corpus: Path = DEFAULT_CORPUS
    text_root: Path = DEFAULT_TEXT
    logs_jsonl: Path = DEFAULT_LOGS_JSONL
    label_tables: Path = DEFAULT_LABEL_TABLES

    @classmethod
    def from_env(cls) -> Paths:
        return cls(
            root=Path(os.environ.get("LABELMAKER_ROOT", str(DEFAULT_ROOT))),
            corpus=Path(os.environ.get("LABELMAKER_CORPUS", str(DEFAULT_CORPUS))),
            text_root=Path(os.environ.get("LABELMAKER_TEXT_ROOT",
                                          str(DEFAULT_TEXT))),
            logs_jsonl=Path(os.environ.get("LABELMAKER_LOGS_JSONL",
                                           str(DEFAULT_LOGS_JSONL))),
            label_tables=Path(os.environ.get("LABELMAKER_LABEL_TABLES",
                                             str(DEFAULT_LABEL_TABLES))),
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
    def events(self) -> Path:
        return self.root / "events"

    @property
    def masks(self) -> Path:
        return self.root / "masks"

    @property
    def annotate(self) -> Path:
        return self.root / "annotate"

    @property
    def text_cache(self) -> Path:
        """Where the shots-in-hand slice of `logs_jsonl` is kept.

        An OUTPUT: streaming 616 MB to find five records is a thing to do
        once, and everything downstream reads the subset instead.
        """
        return self.root / "text"

    @property
    def logs_subset(self) -> Path:
        return self.text_cache / "logs_subset.jsonl"

    @property
    def logs_subset_missing(self) -> Path:
        """The shots `logs_jsonl` was searched for and did not have.

        One shot per line, beside the subset. A shot with no record can
        never enter the subset, so without this file it is searched for
        again on every call - a full pass over 616 MB to learn the same
        nothing, once per record-less shot per pass over a shot list.

        A miss is a fact about the source at the time, not forever, so
        there are two ways back: `text_weak.build_logs_subset(
        refresh_missing=True)` - which `python -m labelmaker.run events
        --refresh-text` is the production owner of - re-asks for the shots
        listed here, and DELETING this file forgets every recorded miss.
        """
        return self.text_cache / "logs_subset.missing"

    @property
    def labels_index(self) -> Path:
        return self.root / "labels_index.parquet"

    @property
    def events_index(self) -> Path:
        return self.root / "events_index.parquet"

    def features_file(self, shot: int) -> Path:
        return self.features / f"{shot}_features.h5"

    def labels_file(self, shot: int) -> Path:
        return self.labels / f"{shot}_labels.h5"

    def events_file(self, shot: int) -> Path:
        return self.events / f"{shot}_events.parquet"

    def sources_file(self, shot: int) -> Path:
        """Which sources RAN on this shot, and over what coverage.

        Beside the events file and not inside it, because it is a
        different claim: an events file says what was found, and a shot on
        which every detector ran and found nothing has an EMPTY one. This
        is what tells that apart from a shot nothing has been run on.
        """
        return self.events / f"{shot}_sources.parquet"

    def masks_file(self, shot: int) -> Path:
        return self.masks / f"{shot}_masks.npz"

    def corpus_file(self, shot: int) -> Path:
        return self.corpus / f"{shot}_processed.h5"

    def text_file(self, shot: int) -> Path:
        return self.text_root / f"shot_{shot}.txt"

    def mkdirs(self) -> None:
        for d in (self.features, self.labels, self.models, self.runs,
                  self.validation, self.events, self.masks, self.annotate,
                  self.text_cache):
            d.mkdir(parents=True, exist_ok=True)


@contextmanager
def atomic_path(path):
    """Yield a temporary sibling to write, then rename it into place.

    The rename is what makes a write atomic: a reader sees either the whole
    old file or the whole new one, never a half-written one. If the body
    raises - a bad dtype, an unserialisable attribute, a keyboard interrupt
    mid-run - the temporary file is removed, because nothing else ever
    would: a stray `.tmp` sibling sits in the data root until some later
    write to the very same path happens to truncate it, and over 16,909
    shots that is a slow leak of files nobody will recognise.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    done = False
    try:
        yield tmp
        done = True
    finally:
        # `finally` rather than `except`, so an interrupt is covered too.
        if done:
            tmp.replace(path)
        else:
            tmp.unlink(missing_ok=True)


def sha256_of(path) -> str:
    """Hex digest of a file, read in 1 MiB blocks.

    Lives here beside `git_sha` because both answer the same question about
    an artifact: exactly which bytes produced this output.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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
