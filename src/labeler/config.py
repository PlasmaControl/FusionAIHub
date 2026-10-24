"""Every filesystem root labeler reads or writes.

Nothing else in the package hard-codes a path, so pointing labeler at a
different data root (a scratch copy, a test fixture) is one environment
variable. Defaults are group storage: Nathan's own scratch is near quota.
"""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .env import getenv

DEFAULT_ROOT = Path("/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler")
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
#: labeler writes is the subset of it for the shots in hand, under
#: `text_cache`.
DEFAULT_LOGS_JSONL = Path(
    "/scratch/gpfs/EKOLEMEN/big_d3d_data/foundation_model_text/sql/logs.jsonl"
)
#: The curated label tables `events/databases.py` reads: one directory of
#: CSVs per phenomenon (the manifest that names them is `events.yaml` under
#: `config_dir`). Label DATA, so it lives in a folder of its own, `events`, beside
#: the data roots and outside the repository: per event `raw/`, `format/`,
#: `extend_*/`, `review/` and `shots.csv`, plus `catalog/` and `shotlists/`.
#: `LABELER_LABEL_TABLES` (`Paths.label_tables`) is the one setting that names it
#: - set it where the folder lives on another machine (StellarAI keeps it next to
#: its raw-data folder) - and every reader and writer of a label table goes
#: through it. A fresh machine restores the folder from the Hugging Face release
#: with `scripts/labeler/hf_publish.py pull`.
DEFAULT_LABEL_TABLES = Path("/scratch/gpfs/EKOLEMEN/D3D/events")
#: The labeler's hand-kept configuration: `events.yaml` (the taxonomy, the label
#: grid and the dataset manifest) and `best_models.yaml` (event -> best model ->
#: inputs -> score). Source, not data, so it is tracked under `configs/labeler/`
#: and resolves relative to this file like the label tables; `LABELER_CONFIG_DIR`
#: overrides it (a test fixture, a scratch copy of the manifest).
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "labeler"
#: Review events whose rows are another event's: the same panels, built once.
#: `store_event` also resolves a subcategory (`<event>.<key>`) to its parent's.
STORE_EVENTS = {
    "confinement": "high_confinement_mode",
    "mhd_n2": "mhd_n1",
    "mhd_n3": "mhd_n1",
}


def store_event(event: str) -> str:
    """The event whose review store, panel recipe and version `event` uses.

    An explicit `STORE_EVENTS` entry wins. Otherwise a subcategory (`<parent>.<key>`)
    has its own store only when a panel or review builder is registered for its id,
    and shows its parent's rows when not. Every other event is its own store.
    """
    if event in STORE_EVENTS:
        return STORE_EVENTS[event]
    parent, dot, _ = event.partition(".")
    if not dot:
        return event
    # Lazy and read with getattr: the builders import this module, and a panel
    # module may not exist yet.
    from .events import panels
    from .events.review import build

    if event in getattr(panels, "BUILDERS", {}) or event in build.BUILDERS:
        return event
    return store_event(parent)


@dataclass(frozen=True)
class Paths:
    """Where labeler's inputs and outputs live."""

    root: Path = DEFAULT_ROOT
    corpus: Path = DEFAULT_CORPUS
    text_root: Path = DEFAULT_TEXT
    logs_jsonl: Path = DEFAULT_LOGS_JSONL
    label_tables: Path = DEFAULT_LABEL_TABLES
    config_dir: Path = DEFAULT_CONFIG_DIR
    #: Where a live fetch parks a shot's raw record. None means `<root>/raw`, beside the
    #: prefetched AE shots; deleting it is safe, the next read refetches.
    raw_cache: Path | None = None
    #: One on-demand shot's directory in the temporary cache (`ondemand`), set by
    #: `events.ondemand.overlay_paths` while the review server builds a shot no
    #: protected list holds: its review store, features and fetched raw groups are
    #: written under it and not under the data root. None everywhere else.
    overlay: Path | None = None
    #: Whether `features_file` and `spectrogram_file` fall back to the temporary cache
    #: where the data root holds no file. Only the review server's `Paths` set it
    #: (`ui.app.create_app`), so that it serves a shot it built on demand; the
    #: pipelines, the prebuild CLI and everything that reads or writes production
    #: data (`Paths.from_env()`) see the data root alone.
    read_cache: bool = False

    def __post_init__(self) -> None:
        if self.raw_cache is None:
            object.__setattr__(self, "raw_cache", self.root / "raw")

    @classmethod
    def from_env(cls) -> Paths:
        raw_cache = getenv("LABELER_RAW_CACHE")
        return cls(
            root=Path(getenv("LABELER_ROOT", str(DEFAULT_ROOT))),
            corpus=Path(getenv("LABELER_CORPUS", str(DEFAULT_CORPUS))),
            text_root=Path(getenv("LABELER_TEXT_ROOT", str(DEFAULT_TEXT))),
            logs_jsonl=Path(getenv("LABELER_LOGS_JSONL", str(DEFAULT_LOGS_JSONL))),
            label_tables=Path(
                getenv("LABELER_LABEL_TABLES", str(DEFAULT_LABEL_TABLES))
            ),
            config_dir=Path(getenv("LABELER_CONFIG_DIR", str(DEFAULT_CONFIG_DIR))),
            raw_cache=Path(raw_cache) if raw_cache else None,
        )

    @property
    def events_manifest(self) -> Path:
        """`events.yaml`: the taxonomy, the label grid and the dataset manifest.

        Its `raw_datasets` paths and category names are relative to `label_tables`.
        """
        return self.config_dir / "events.yaml"

    @property
    def best_models(self) -> Path:
        """`best_models.yaml`: the best model per event, its inputs and its score."""
        return self.config_dir / "best_models.yaml"

    @property
    def shotlists_manifest(self) -> Path:
        """`shotlists.yaml`: the named shot lists and the ones each event recommends
        (`events.shotlists`)."""
        return self.config_dir / "shotlists.yaml"

    @property
    def shotlists(self) -> Path:
        """The named shot lists, `<name>.csv` or `<name>.txt` each, beside the label
        tables they name shots of. Gold and silver stay at their own paths."""
        return self.label_tables / "shotlists"

    @property
    def features(self) -> Path:
        return self.root / "features"

    @property
    def ondemand(self) -> Path:
        """The review server's temporary cache: one `<shot>/` directory per shot it
        built on demand (its store, features and raw groups), capped and evicted by
        `events.ondemand`. Safe to delete; the next open rebuilds."""
        return self.root / "cache" / "ondemand"

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
        refresh_missing=True)` - which `python -m labeler.run events
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

    def _temp(self, shot: int, *parts: str) -> Path:
        return self.ondemand / str(int(shot)) / Path(*parts)

    def features_file(self, shot: int) -> Path:
        """The shot's feature file: the overlay's while one is set; else the data
        root's, or with `read_cache` the temporary cache's when only that holds it."""
        name = f"{shot}_features.h5"
        if self.overlay is not None:
            return self.overlay / "features" / name
        main = self.features / name
        if self.read_cache and not main.exists() and (
                self._temp(shot, "features", name).exists()):
            return self._temp(shot, "features", name)
        return main

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

    @property
    def spectrograms(self) -> Path:
        """The review store: one rows file per shot, `<event>/<shot>.h5`."""
        return self.root / "spectrograms"

    @property
    def literature(self) -> Path:
        """The literature cache: probe hits, full texts and the links they give."""
        return self.root / "literature"

    @property
    def catalog(self) -> Path:
        """The catalog's derived data: the pool, the Ip log, drafts of its tables."""
        return self.root / "catalog"

    def spectrogram_file(self, event: str, shot: int) -> Path:
        """The shot's review store: the overlay's while one is set; else the data
        root's, or with `read_cache` the temporary cache's when only that holds it."""
        where = Path(store_event(event)) / f"{int(shot)}.h5"
        if self.overlay is not None:
            return self.overlay / "spectrograms" / where
        main = self.spectrograms / where
        if self.read_cache and not main.exists() and (
                self._temp(shot, "spectrograms", *where.parts).exists()):
            return self._temp(shot, "spectrograms", *where.parts)
        return main

    def mkdirs(self) -> None:
        for d in (
            self.features,
            self.labels,
            self.models,
            self.runs,
            self.validation,
            self.events,
            self.masks,
            self.annotate,
            self.text_cache,
        ):
            d.mkdir(parents=True, exist_ok=True)


def events_ref(path, label_tables: Path | None = None) -> str:
    """`path` as provenance text: `<events>/...` inside the label tables, else as given.

    A record that names a label table must not carry the folder's absolute path, which
    differs between machines (`LABELER_LABEL_TABLES`). `<events>` stands for
    `Paths.label_tables`; `label_tables` defaults to the environment's.
    """
    root = Paths.from_env().label_tables if label_tables is None else Path(label_tables)
    try:
        return f"<events>/{Path(path).relative_to(root).as_posix()}"
    except ValueError:
        return str(path)


class NamedBytes(io.BytesIO):
    """Make a reader's refusal name the file the bytes came from."""

    def __init__(self, data: bytes, path):
        super().__init__(data)
        self.name = str(path)

    def __str__(self) -> str:
        return self.name


@contextmanager
def atomic_path(path, *, unique: bool = False):
    """Yield a temporary sibling to write, then rename it into place. With `unique`
    the sibling's name carries the process and thread ids, so writers of one file
    that do not share a lock never write into the same temporary file.

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
    tmp = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp" if unique
        else path.name + ".tmp")
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


def git_sha(full: bool = False) -> str:
    """Checkout sha (short by default, full on request), or 'unknown'."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", *([] if full else ["--short"]), "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def git_dirty() -> bool | None:
    """Whether tracked files differ from HEAD, or None when git cannot answer."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.stdout.strip())
