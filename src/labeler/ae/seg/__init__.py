"""AE time-frequency segmentation: where in the spectrogram the mode is.

`labeler.ae.xpower` says, per 10 ms frame, whether an Alfven eigenmode is
present. This package says where it is: which pixels of the review page's
cross-power rows, 80-250 kHz, belong to the mode. It is semantic, not instance,
segmentation (one "AE" class; a region is a connected set of AE pixels), so a
large, many-branched mode is still one answer.

- `pseudo`: training masks without drawing, TokEye's coherent mask inside the
  frames the owner labelled AE, inside 80-250 kHz;
- `regions`: a mask's regions for the review page, and the reviewer's
  accept or reject of each (`review/masks.jsonl`);
- `model`, `train`: a small U-Net trained on the reviewed pseudo-masks;
- `poi`: points of interest, one per predicted region, for a shot viewer.

v2 (`SEG_VERSIONS["v2"]`) is pseudo-v2 and SegNet v2: over 0-250 kHz and the
owner's whole window, from TokEye's whole-shot masks (`ae/masks-full`) and
ae_xpower v3's label snapshot, with the MHD-line rules of `mhdlines`. v1 is
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...config import Paths

METHOD = "ae_seg"
VERSION = "v1"
PSEUDO = "pseudo-v1"
EVENT = "alfven_eigenmode"


@dataclass(frozen=True)
class SegVersion:
    """What a segmentation version is made of."""

    pseudo: str  # the masks' directory name
    band_khz: tuple[float, float]  # the band SegNet trains on and draws over
    labels: str | None  # the ae_xpower snapshot the masks read; None: the live file
    ae_version: str  # the ae_xpower version: TokEye's masks and the chosen model
    whole_window: bool  # scored over the owner's whole window (else 0-2 s)


SEG_VERSIONS = {
    "v1": SegVersion("pseudo-v1", (80.0, 250.0), None, "v1", False),
    "v2": SegVersion("pseudo-v2", (0.0, 250.0), "v3", "v3", True),
}


def pseudo_dir(paths: Paths, version: str = VERSION) -> Path:
    return paths.root / "segmentation" / EVENT / SEG_VERSIONS[version].pseudo


def model_dir(paths: Paths, version: str = VERSION) -> Path:
    return paths.root / "models" / METHOD / version


def poi_dir(paths: Paths, version: str = VERSION) -> Path:
    return paths.root / "poi" / EVENT / f"{METHOD}-{version}"
