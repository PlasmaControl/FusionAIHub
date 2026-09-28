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
"""

from __future__ import annotations

from ...config import Paths

METHOD = "ae_seg"
VERSION = "v1"
PSEUDO = "pseudo-v1"
EVENT = "alfven_eigenmode"


def pseudo_dir(paths: Paths):
    return paths.root / "segmentation" / EVENT / PSEUDO


def model_dir(paths: Paths):
    return paths.root / "models" / METHOD / VERSION


def poi_dir(paths: Paths):
    return paths.root / "poi" / EVENT / f"{METHOD}-{VERSION}"
