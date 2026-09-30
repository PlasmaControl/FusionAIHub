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

v3 (`SEG_VERSIONS["v3"]`) is pseudo-v3 and SegNet v3: v2's inputs and split,
with the per-line MHD markers of `markers` in place of `mhdlines`' rules, and
below 80 kHz an owner-absent frame IGNORED but for the pixels they take. Its
masks are written only when the markers' gate passes (`gated`), and its test
reuses SegNet v2's test shots after v2's breakdown was seen (`test_of`,
`reuse_note`). v2 is unchanged.

v4 (`SEG_VERSIONS["v4"]`) is pseudo-v4 and SegNet v4: SegNet v1 with the band
at 60-250 kHz, the band the owner's review page draws AE from (2026-09-30).
pseudo-v1's rules, TokEye's 0-2 s masks, the live labels file and the chosen
ae_xpower v1 model's split, as v1's; the blob records its band. Its test
reuses SegNet v1's test shots after v1's test was scored (`test_of`,
`reuse_note`). v1-v3 are unchanged.
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
    gated: bool = False  # training needs the pseudo-masks' rules.json gate passed
    test_of: str | None = None  # the earlier version whose test shots it reuses


SEG_VERSIONS = {
    "v1": SegVersion("pseudo-v1", (80.0, 250.0), None, "v1", False),
    "v2": SegVersion("pseudo-v2", (0.0, 250.0), "v3", "v3", True),
    "v3": SegVersion(
        "pseudo-v3", (0.0, 250.0), "v3", "v3", True, gated=True, test_of="v2"
    ),
    "v4": SegVersion("pseudo-v4", (60.0, 250.0), None, "v1", False, test_of="v1"),
}

# What each `test_of` version's design is, and what of the earlier version's
# test had been seen when it was made: (design, seen, record).
REUSE = {
    "v3": ("pseudo-v3's markers", "test breakdown had been seen", "diagnosis.md"),
    "v4": ("the 60-250 kHz band", "test had been scored", "evaluation.json"),
}


def reuse_note(version: str, n_test: int | None = None) -> str | None:
    """The sentence a version's records carry when its test is a second use of
    an earlier version's test shots (`test_of`); None when it is not."""
    earlier = SEG_VERSIONS[version].test_of
    if earlier is None:
        return None
    shots = "test shots" if n_test is None else f"{n_test} test shots"
    design, seen, record = REUSE[version]
    return (
        f"SegNet {version}'s test is a second use of SegNet {earlier}'s {shots}, "
        f"and its design ({design}) was made after SegNet "
        f"{earlier}'s {seen} (models/ae_seg/{earlier}/"
        f"{record}): its test scores are not an unbiased estimate. Tier: "
        "suggestions."
    )


def pseudo_dir(paths: Paths, version: str = VERSION) -> Path:
    return paths.root / "segmentation" / EVENT / SEG_VERSIONS[version].pseudo


def model_dir(paths: Paths, version: str = VERSION) -> Path:
    return paths.root / "models" / METHOD / version


def poi_dir(paths: Paths, version: str = VERSION) -> Path:
    return paths.root / "poi" / EVENT / f"{METHOD}-{version}"
