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
(`reviewed_mask`). SegNet trains on pseudo-v1, and the review page draws
pseudo-v1-full (`whole.REVIEW`): a decision made there names that set in
`pseudo`, so training on pseudo-v1 cannot use it and says so
(`other_decisions`).
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
from . import PSEUDO, pseudo_dir
from .pseudo import EIGHT, PseudoMask

LOG = "masks.jsonl"
MAX_REGIONS = 5000


def log_path(event_dir) -> Path:
    return Path(event_dir) / REVIEW / LOG


def pseudo_file(paths, shot: int) -> Path:
    return pseudo_dir(paths) / f"{int(shot)}.npz"


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


def reviewed_mask(pm: PseudoMask, decision: dict | None, sha256: str) -> np.ndarray:
    """The mask training uses: rejected regions set to background. A decision
    made on another version of the file is ignored."""
    mask = pm.mask.copy()
    if not decision or decision.get("pseudo_sha256") != sha256:
        return mask
    labelled, _ = label_regions(pm.mask)
    mask[np.isin(labelled, decision["rejected"]) & (labelled > 0)] = 0
    return mask


def other_decisions(decisions: dict, pseudo: str = PSEUDO) -> list[int]:
    """The shots whose last decision was made on another mask set than `pseudo`."""
    return sorted(s for s, d in decisions.items() if d.get("pseudo", PSEUDO) != pseudo)


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
