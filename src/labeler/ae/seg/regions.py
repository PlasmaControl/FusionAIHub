"""A pseudo-mask's regions, and the reviewer's word on each.

A region is an 8-connected set of AE pixels (value 1) of a pseudo-mask,
numbered 1, 2, ... in `scipy.ndimage.label` order, which is fixed by the mask:
the same file always gives the same numbers. The review page draws each region
over the AE rows and a click rejects it (not the mode: a harmonic of an MHD
mode, a line of pickup) or takes the rejection back.

Each save appends one line to `data/events/alfven_eigenmode/review/masks.jsonl`:
`{"shot", "pseudo", "pseudo_sha256", "rejected", "reviewer", "name", "saved_at"}`.
The last
line of a shot is its decision; the earlier ones are its history. A decision
made on another version of the pseudo-mask (another sha256) is stale: the page
says so and training ignores it.

Training takes the pseudo-mask with every rejected region set to background
(`reviewed_mask`). A decision names in `pseudo` the masks it was made on, and
the review page draws pseudo-v1-full (`whole.REVIEW`), not the masks a SegNet
version trains on. Such a decision still reaches training through the
version's own mask, on the same store grid (`transfer`), as long as the file
clicked is still the one there (`clicked_mask`, by its sha256): the pixels of
the regions it rejects on the mask clicked are background where the version's
mask scores them, and stay unscored where it does not; the rest of each of the
version's regions they touch, outside the regions the page drew (below 60 kHz
in pseudo-v2 and v3), is unscored too.
"""

from __future__ import annotations

import getpass
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy import ndimage

from ...events.review.labels import REVIEW
from . import PSEUDO, SEG_VERSIONS, VERSION, pseudo_dir, whole
from .pseudo import EIGHT, IGNORE, PseudoMask

LOG = "masks.jsonl"
MAX_REGIONS = 5000


def log_path(event_dir) -> Path:
    return Path(event_dir) / REVIEW / LOG


def pseudo_file(paths, shot: int, version: str = VERSION) -> Path:
    """The shot's mask among `version`'s pseudo-masks; the review page reads v1's."""
    return pseudo_dir(paths, version) / f"{int(shot)}.npz"


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label_regions(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """`(labels, count)`: each AE pixel's region number, 0 elsewhere."""
    return ndimage.label(np.asarray(mask) == 1, structure=EIGHT)


def runs(region: np.ndarray) -> list[list[int]]:
    """`[row, first column, length]` for each horizontal run of a region's pixels."""
    out = []
    for row in np.flatnonzero(region.any(axis=1)):
        edges = np.flatnonzero(np.diff(np.r_[0, region[row].astype(np.int8), 0]))
        out += [[int(row), int(a), int(b - a)] for a, b in zip(edges[::2], edges[1::2])]
    return out


def describe(pm: PseudoMask) -> list[dict]:
    """Every region: its number, pixel count, box (ms, kHz) and runs."""
    labelled, count = label_regions(pm.mask)
    found = []
    for k, (rows, cols) in enumerate(ndimage.find_objects(labelled), start=1):
        region = labelled[rows, cols] == k
        found.append(
            {
                "id": k,
                "pixels": int(region.sum()),
                "t0_ms": pm.t0_ms + cols.start * pm.dt_ms,
                "t1_ms": pm.t0_ms + cols.stop * pm.dt_ms,
                "f0_khz": pm.y0_khz + (rows.start - 0.5) * pm.dy_khz,
                "f1_khz": pm.y0_khz + (rows.stop - 0.5) * pm.dy_khz,
                "runs": [
                    [r + rows.start, c + cols.start, n] for r, c, n in runs(region)
                ],
            }
        )
    if count > MAX_REGIONS:
        raise ValueError(f"{pm.shot}: {count} regions, more than {MAX_REGIONS}")
    return found


def shot_history(event_dir, shot: int | None = None) -> list[dict]:
    """Saved lines, optionally for one shot, including stale decisions."""
    path = log_path(event_dir)
    if not path.is_file():
        return []
    found = []
    for line in path.read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            if shot is None or int(entry["shot"]) == int(shot):
                found.append(entry)
    return found


def read_decisions(event_dir) -> dict[int, dict]:
    """The last decision per shot."""
    return {int(entry["shot"]): entry for entry in shot_history(event_dir)}


def save_decision(
    event_dir,
    shot: int,
    rejected,
    *,
    pseudo_sha256: str,
    name: str | None = None,
    pseudo: str = PSEUDO,
) -> dict:
    """Append one decision, made on the masks named `pseudo`; the line written."""
    entry = {
        "shot": int(shot),
        "pseudo": pseudo,
        "pseudo_sha256": pseudo_sha256,
        "rejected": sorted({int(k) for k in rejected}),
        "name": name,
        "reviewer": getpass.getuser(),
        "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path = log_path(event_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def rejected_pixels(pm: PseudoMask, decision: dict) -> np.ndarray:
    """`(n_y, n)` bool: the pixels of the regions `decision` rejects on `pm`."""
    labelled, _ = label_regions(pm.mask)
    return np.isin(labelled, decision["rejected"]) & (labelled > 0)


def reviewed_mask(pm: PseudoMask, decision: dict | None, sha256: str) -> np.ndarray:
    """The mask training uses: rejected regions set to background. A decision
    made on another version of the file is ignored here (see `transfer`)."""
    mask = pm.mask.copy()
    if not decision or decision.get("pseudo_sha256") != sha256:
        return mask
    mask[rejected_pixels(pm, decision)] = 0
    return mask


def decided_file(paths, shot: int, decision: dict) -> Path:
    """The file a decision was made on, by the mask set it names: the page's
    (`whole.REVIEW`) or a pseudo-mask version's."""
    name = decision.get("pseudo", PSEUDO)
    if name == whole.REVIEW:
        return whole.review_file(paths, shot)
    for version, spec in SEG_VERSIONS.items():
        if spec.pseudo == name:
            return pseudo_file(paths, shot, version)
    raise ValueError(f"{shot}: a decision on an unknown mask set {name!r}")


def clicked_name(shot: int, decision: dict) -> str:
    """How a model's `pseudo_masks.json` names the file a decision was made on."""
    return f"{decision.get('pseudo', PSEUDO)}/{int(shot)}.npz"


def clicked_mask(
    paths, shot: int, decision: dict | None, own: str
) -> tuple[Path | None, str | None]:
    """The file a decision made on another mask set than `own` was clicked on:
    `(path, None)` while it is that file (its sha256), `(None, why)` when it is
    not, and `(None, None)` without such a decision."""
    if not decision or decision.get("pseudo", PSEUDO) == own:
        return None, None
    try:
        path = decided_file(paths, shot, decision)
    except ValueError as error:
        return None, str(error)
    if not path.is_file():
        return None, f"{shot}: {path} is gone"
    if file_sha256(path) != decision.get("pseudo_sha256"):
        return None, f"{shot}: made on an earlier {path}"
    return path, None


def transfer(
    target: np.ndarray, pm: PseudoMask, clicked: PseudoMask, decision: dict
) -> np.ndarray:
    """`target`, on `pm`'s grid, with the regions `decision` rejects on
    `clicked` (the mask the reviewer clicked, the page's regions) taken out.

    A rejection clears AE; it never scores a pixel `target` does not score. A
    rejected pixel is background where `target` scores it (0 or 1) and stays
    IGNORE where it does not (pseudo-v1 after 2 s). Each of `target`'s own
    regions (`label_regions`) holding a rejected pixel is a line the reviewer
    said is not the mode: its pixels outside every region of `clicked` (the
    page never drew them, say below 60 kHz) are IGNORE, and those under a page
    region the reviewer kept stay as they are. Regions no rejection touches
    stay as they are."""
    grid = ("t0_ms", "dt_ms", "y0_khz", "dy_khz")
    if clicked.mask.shape != pm.mask.shape or any(
        abs(getattr(clicked, k) - getattr(pm, k)) > 1e-6 for k in grid
    ):
        raise ValueError(
            f"{pm.shot}: the mask clicked is not on the grid of the one trained on"
        )
    target = np.asarray(target)
    rejected = rejected_pixels(clicked, decision)
    own, _ = label_regions(target)
    touched = np.isin(own, np.unique(own[rejected])) & (own > 0)
    out = target.copy()
    out[touched & (np.asarray(clicked.mask) != 1)] = IGNORE
    out[rejected & (target != IGNORE)] = 0
    return out


def shot_view(
    paths, event_dir, shot: int, *, path=None, pseudo: str = PSEUDO
) -> dict | None:
    """What the review page draws for one shot, or None without a pseudo-mask.

    `path` is the shot's file among the masks named `pseudo` (default pseudo-v1's).
    """
    path = pseudo_file(paths, shot) if path is None else Path(path)
    if not path.is_file():
        return None
    pm = PseudoMask.load(path)
    sha = file_sha256(path)
    history = shot_history(event_dir, shot)
    decision = history[-1] if history else None
    current = bool(decision) and decision.get("pseudo_sha256") == sha
    return {
        "shot": int(shot),
        "revision": len(history),
        "pseudo": pseudo,
        "pseudo_sha256": sha,
        "grid": {"t0_ms": pm.t0_ms, "dt_ms": pm.dt_ms, "n": int(pm.mask.shape[1])},
        "y0_khz": pm.y0_khz,
        "dy_khz": pm.dy_khz,
        "regions": describe(pm),
        "rejected": decision["rejected"] if current else [],
        "stale": bool(decision) and not current,
        "last_save": decision,
    }
