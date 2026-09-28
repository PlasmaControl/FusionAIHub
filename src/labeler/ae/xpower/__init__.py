"""AE from cross-power: the owner's reviewed AE labels, learned and extended.

`data` turns the review page's AE rows and the owner's labels into 10 ms frames,
`model` is the network, `train` fits it, `evaluate` scores it against the earlier
methods on the held-out shots, `gallery` draws one JPEG per shot, and `extend`
runs it over shots nobody has labelled, as suggestions. `cv` cross-validates the
candidates of a version chosen that way (v2).

**Label snapshots.** v1 read the owner's live `review/labels.csv` and copied it
into each model directory. A version in `LABEL_SNAPSHOTS` reads its labels only
from its frozen snapshot, `models/ae_xpower/<version>/review/labels.csv`, and
refuses to run when that file's sha256 is not the one recorded here.

**Binding.** Every command's `--version` must be its models directory's name and
the version its checkpoints record (`check_bound`). A full scoring is never made
under `runs/`, and a pilot scoring there is 20 test shots at most (`check_full`);
the extension's gate is never read from there.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from ...config import Paths
from ...events.review import labels

METHOD = "ae_xpower"
VERSION = "v1"
EVENT = "alfven_eigenmode"
#: The owner's labels as the controller froze them for a version (the ledger's
#: Deviation 11: 180 shots as of 2026-09-27 22:05 EDT, for v2), by sha256.
LABEL_SNAPSHOTS = {
    "v2": "5f52a26831cd9173f3d67b1e540ba23ee4aa1adab9159b1312a663cf88eee1de",
}
#: Versions whose candidate and threshold `cv` chooses, not `evaluate --choose`.
CV_VERSIONS = frozenset({"v2"})
#: The pilot rule: a pilot is 20 shots or fewer.
PILOT_MAX = 20


def pilot_area(directory: Path, runs: Path) -> bool:
    """Only descendants of runs, resolving symlinks, can replace pilot records."""
    return runs.resolve() in directory.resolve().parents


def check_limit(paths: Paths, models: Path, limit: int) -> None:
    if limit < 0 or (limit > 0 and not pilot_area(models, paths.runs)):
        raise ValueError(
            f"{models}: --limit must be nonnegative and requires a directory "
            f"under {paths.runs} when positive"
        )


def check_full(paths: Paths, models: Path, limit: int) -> None:
    """A full scoring (`--limit 0`) is the version's one look at its test shots:
    never under `runs/`, where records may be replaced and the look repeated. A
    pilot there scores the first `PILOT_MAX` test shots at most (the pilot rule),
    so no repeatable scoring there covers a real test split."""
    if not pilot_area(models, paths.runs):
        return
    if limit == 0:
        raise ValueError(
            f"{models}: a full scoring (--limit 0) is never made under "
            f"{paths.runs}; a pilot there takes --limit 1 to {PILOT_MAX}"
        )
    if limit > PILOT_MAX:
        raise ValueError(
            f"{models}: a pilot scoring under {paths.runs} is at most "
            f"{PILOT_MAX} test shots, not --limit {limit}"
        )


def blob_version(blob: dict) -> str:
    """A checkpoint's recorded version; v1's released checkpoints predate the
    key, so a checkpoint without one is v1's."""
    return blob.get("version", "v1")


def check_bound(
    version: str, models: Path, blob: dict | None = None, file=None
) -> None:
    """`--version` is the models directory's name and, given a checkpoint, the
    version it records; otherwise refuse, naming both."""
    if models.name != version:
        raise ValueError(
            f"{models}: --version {version} differs from the models directory's "
            f"name {models.name}"
        )
    if blob is not None and blob_version(blob) != version:
        raise ValueError(
            f"{file}: --version {version} differs from the checkpoint's version "
            f"{blob_version(blob)}"
        )


def model_dir(paths: Paths, version: str = VERSION) -> Path:
    """The trained model, its split and its scores."""
    return paths.models / METHOD / version


def snapshot_file(paths: Paths, version: str) -> Path:
    """Where a version's frozen labels are: its models directory's `review/`."""
    return labels.labels_path(model_dir(paths, version))


def parse_labels(data: bytes) -> dict[int, labels.Label]:
    """`labels.read_saved` of these bytes, with the review parser's checks."""
    with TemporaryDirectory(prefix="ae-labels-") as directory:
        file = labels.labels_path(directory)
        file.parent.mkdir()
        file.write_bytes(data)
        return labels.read_saved(directory)


def check_snapshot(digest: str, version: str, where) -> None:
    """Refuse labels whose sha256 is not the version's snapshot's."""
    expected = LABEL_SNAPSHOTS[version]
    if digest != expected:
        raise ValueError(
            f"{where}: labels sha256 {digest} is not version {version}'s "
            f"snapshot {expected}"
        )


def read_snapshot(paths: Paths, version: str) -> tuple[bytes, dict]:
    """The version's snapshot bytes and labels, after checking its sha256."""
    if version not in LABEL_SNAPSHOTS:
        raise ValueError(f"version {version} has no label snapshot")
    file = snapshot_file(paths, version)
    data = file.read_bytes()
    check_snapshot(hashlib.sha256(data).hexdigest(), version, file)
    return data, parse_labels(data)


def suggestions_dir(paths: Paths, version: str = VERSION) -> Path:
    """The extension's suggestion table (v1 spec §3: suggestions, not labels)."""
    return paths.root / "suggestions" / METHOD / version


def gallery_dir(paths: Paths, version: str = VERSION) -> Path:
    """One JPEG per shot, in `reviewed/`, `unreviewed/` and `extension/`."""
    return paths.root / "gallery" / EVENT / f"{METHOD}-{version}"


def tokeye_masks(paths: Paths) -> Path:
    """TokEye's cleaned coherent masks for the AE180 shots, `<shot>_<split>_*`."""
    return paths.root / "ae" / "masks"


def seldnet_dir(paths: Paths) -> Path:
    """The earlier detector, `d3d_ae_activity_seldnet`, and its inputs' home."""
    return paths.models / "d3d_ae_activity_seldnet"


def event_dir(paths: Paths) -> Path:
    """The owner's AE review: roster, source table and `review/labels.csv`."""
    return paths.label_tables / EVENT
