#!/usr/bin/env python
"""Pack the plasma_tv TangTV inversions into one small npz per shot.

The inversions are IDL save files (`emission_structure_*.sav`, Chen 2026's C-III
tomographic inversions of the lower-divertor camera). A shot may exist in several
copies (the catalog's `raw/` folder, plasma_tv's `raw/all`, `issues`, `external`);
the first copy found in `SEARCH` wins. Output, per shot, under
`$LABELER_ROOT/round4/detach/inversions/<shot>.npz`:

    frames   (T, nZ, nR) float16, emissivity (arbitrary units)
    times_ms (T,)        float32, shot time of each inverted frame
    radii    (nR,)       float32, metres
    elevation (nZ,)      float32, metres, index 0 is the LOWEST row
    source   str         the .sav the frames came from

The raw video (`VID`, 420 frames of 240x720 uint8 every 16.7 ms) is NOT copied;
the figure reads it from the save file directly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from scipy.io import readsav

SEARCH = (
    "/scratch/gpfs/nc1514/FusionAIHub/data/events/detachment/raw",
    "/scratch/gpfs/nc1514/plasma_tv/data/raw/all",
    "/scratch/gpfs/nc1514/plasma_tv/data/raw/issues",
    "/scratch/gpfs/nc1514/plasma_tv/data/external/TangTV",
    "/scratch/gpfs/nc1514/plasma_tv/data/2026_trainset_new",
    "/scratch/gpfs/nc1514/plasma_tv/data/2026_trainset",
)


def find_saves() -> dict[int, Path]:
    """Shot -> the preferred inversion .sav (the plain file over its `_raw` twin)."""
    found: dict[int, Path] = {}
    for root in SEARCH:
        for path in sorted(Path(root).glob("emission_structure_*.sav")):
            match = re.search(r"_(\d{5,6})(_raw)?\.sav$", path.name)
            if not match:
                continue
            shot = int(match.group(1))
            if shot not in found:
                found[shot] = path
    return found


def pack(path: Path) -> dict:
    record = readsav(str(path))["emission_structure"][0]
    inverted, radii, elevation, _frames, times = (record[i] for i in range(5))
    if abs(radii - radii[0]).max() > 1e-6 or abs(elevation - elevation[0]).max() > 1e-6:
        raise ValueError(f"{path.name}: the grid changes between frames")
    if inverted.shape[1] != 201 and inverted.shape[1] != 281:
        raise ValueError(f"{path.name}: unexpected grid {inverted.shape}")
    return {
        "frames": np.asarray(inverted, dtype="float16"),
        "times_ms": np.asarray(times, dtype="float32"),
        "radii": np.asarray(radii[0], dtype="float32"),
        "elevation": np.asarray(elevation[0], dtype="float32"),
        "source": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shots", default="", help="comma list; default all found")
    args = parser.parse_args()
    out_dir = Path(os.environ["LABELER_ROOT"]) / "round4" / "detach" / "inversions"
    out_dir.mkdir(parents=True, exist_ok=True)
    found = find_saves()
    wanted = [int(s) for s in args.shots.split(",") if s] or sorted(found)
    index = {}
    for shot in wanted:
        target = out_dir / f"{shot}.npz"
        if not target.is_file():
            data = pack(found[shot])
            tmp = target.with_name(f".{target.name}.tmp.npz")
            np.savez(tmp, **data)
            tmp.replace(target)
        with np.load(target) as npz:
            index[shot] = {
                "source": str(npz["source"]),
                "frames": list(npz["frames"].shape),
                "t0_ms": float(npz["times_ms"][0]),
                "t1_ms": float(npz["times_ms"][-1]),
            }
        print(shot, index[shot]["frames"], index[shot]["source"], flush=True)
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
