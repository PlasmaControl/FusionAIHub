"""AE from cross-power: the owner's reviewed AE labels, learned and extended.

`data` turns the review page's AE rows and the owner's labels into 10 ms frames,
`model` is the network, `train` fits it, `evaluate` scores it against the earlier
methods on the held-out shots, `gallery` draws one JPEG per shot, and `extend`
runs it over shots nobody has labelled, as suggestions.
"""

from __future__ import annotations

from pathlib import Path

from ...config import Paths

METHOD = "ae_xpower"
VERSION = "v1"
EVENT = "alfven_eigenmode"


def pilot_area(directory: Path, runs: Path) -> bool:
    """Only descendants of runs, resolving symlinks, can replace pilot records."""
    return runs.resolve() in directory.resolve().parents


def check_limit(paths: Paths, models: Path, limit: int) -> None:
    if limit < 0 or (limit > 0 and not pilot_area(models, paths.runs)):
        raise ValueError(
            f"{models}: --limit must be nonnegative and requires a directory "
            f"under {paths.runs} when positive"
        )


def model_dir(paths: Paths, version: str = VERSION) -> Path:
    """The trained model, its split and its scores."""
    return paths.models / METHOD / version


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
