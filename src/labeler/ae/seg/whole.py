"""TokEye over the whole shot on the review page: the mask and a layer under it.

    python -m labeler.ae.seg.whole [--shots S ...]

pseudo-v1 reads TokEye's first masks (`ae/masks`), which end at 2 s, so on the
review page its cyan stopped there even where the owner's AE frames run to 5 s.
TokEye's whole-shot masks (`$LABELER_ROOT/ae/masks-full`, 0-6 s) cover the
record. From them this module builds two things per shot, on the review store's
level-8 grid:

- **the review mask**, `pseudo-v1-full`: pseudo-v1's rules (`pseudo.build`:
  TokEye's lines inside the owner's AE frames, 80-250 kHz) over the owner's whole
  window. The page draws its regions and saves the reviewer's decisions on it.
  pseudo-v1 is unchanged and SegNet v1 still trains on it, so to v1 a decision
  made here is stale (another sha256) and ignored, as it is to v2 and v3;
- **the layer**, `tokeye-full`: a picture, not a label. A pixel is lit where
  TokEye lights at least two of the four chords, 0-250 kHz, AE or not
  (`pseudo.tokeye_rows`, `pseudo.pool_columns`). Nothing trains on it, scores it
  or saves a decision on it.

**Output.** Under `$LABELER_ROOT/segmentation/alfven_eigenmode/`:
`pseudo-v1-full/<shot>.npz` (a `PseudoMask`, for each shot the owner has saved)
with `index.csv`; `tokeye-full/<shot>.npz` (`bits`, the `(n_y, n)` lit pixels
bit-packed along time; `n`, `t0_ms`, `dt_ms`, `y0_khz`, `dy_khz`). Each
directory's `meta.json` names every shot's TokEye file by its sha256, and the
mask's names the labels it read.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import numpy as np

from ...config import Paths, atomic_path, git_sha, sha256_of
from ...events.review import labels
from ...events.review.rows import Grid
from ..xpower import event_dir
from ..xpower.data import clean_path, store_rows, tokeye_clean
from . import EVENT
from .pseudo import INDEX_COLUMNS, LEVEL, PseudoMask, pool_columns, summary, tokeye_rows
from .pseudo import build as build_mask

LAYER = "tokeye-full"
REVIEW = "pseudo-v1-full"


def masks_full(paths: Paths) -> Path:
    """TokEye's whole-shot cleaned masks, `<shot>_<split>_clean.npz`."""
    return paths.root / "ae" / "masks-full"


def layer_dir(paths: Paths) -> Path:
    return paths.root / "segmentation" / EVENT / LAYER


def layer_file(paths: Paths, shot: int) -> Path:
    return layer_dir(paths) / f"{int(shot)}.npz"


def review_dir(paths: Paths) -> Path:
    return paths.root / "segmentation" / EVENT / REVIEW


def review_file(paths: Paths, shot: int) -> Path:
    """The mask the review page draws and saves decisions on."""
    return review_dir(paths) / f"{int(shot)}.npz"


def build(grid: Grid, n_y: int, tokeye) -> np.ndarray:
    """`(n_y, grid.n)`: where TokEye lights two chords; `tokeye` is `tokeye_clean`'s."""
    t_ms, clean, _ = tokeye
    return pool_columns(tokeye_rows(clean), t_ms, grid)[:n_y]


def save(path, grid: Grid, y0: float, dy: float, lit: np.ndarray) -> None:
    with atomic_path(Path(path)) as tmp, open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            bits=np.packbits(np.asarray(lit, dtype=bool), axis=1),
            n=np.int64(grid.n),
            t0_ms=grid.t0_ms,
            dt_ms=grid.dt_ms,
            y0_khz=float(y0),
            dy_khz=float(dy),
        )


def view(paths: Paths, shot: int) -> dict | None:
    """What the review page draws for one shot's layer, or None without one.

    `bits` is base64 of the packed rows: row j (bin `y0_khz + j * dy_khz`) takes
    `ceil(n / 8)` bytes, column k in bit `7 - k % 8` of byte `k // 8`.
    """
    path = layer_file(paths, shot)
    if not path.is_file():
        return None
    with np.load(path) as z:
        bits = z["bits"]
        return {
            "shot": int(shot),
            "layer": LAYER,
            "grid": {
                "t0_ms": float(z["t0_ms"]),
                "dt_ms": float(z["dt_ms"]),
                "n": int(z["n"]),
            },
            "y0_khz": float(z["y0_khz"]),
            "dy_khz": float(z["dy_khz"]),
            "n_y": int(bits.shape[0]),
            "bits": base64.b64encode(np.ascontiguousarray(bits).tobytes()).decode(),
        }


def make(
    paths: Paths, shot: int, label: labels.Label | None, tokeye_bytes: bytes
) -> tuple[np.ndarray, Grid, float, float, PseudoMask | None]:
    """One shot's layer, its grid, y0 and dy, and its review mask (None unlabelled)."""
    grid, values, y0, dy = store_rows(paths.spectrogram_file(EVENT, shot), LEVEL)
    n_y = values.shape[1]
    tokeye = tokeye_clean(BytesIO(tokeye_bytes))
    mask = None
    if label is not None:
        mask = build_mask(shot, label, grid, n_y, y0, dy, tokeye)
    return build(grid, n_y, tokeye), grid, y0, dy, mask


def _meta(directory: Path, extra: dict, sources: dict, failed: list) -> None:
    """Write `meta.json`; a run over some shots keeps the others' sources."""
    file = directory / "meta.json"
    if file.is_file():
        sources = {**json.loads(file.read_text())["tokeye"], **sources}
    meta = {
        **extra,
        "shots": len(sources),
        "failed": failed,
        "tokeye": sources,
        "git_sha": git_sha(),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with atomic_path(file) as tmp:
        tmp.write_text(json.dumps(meta, indent=1) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--shots", type=int, nargs="*", help="default: every TokEye shot")
    args = p.parse_args(argv)
    paths = Paths.from_env()
    source = masks_full(paths)
    directory = event_dir(paths)
    saved = labels.read_saved(directory)
    shots = args.shots or sorted(
        {int(f.name.split("_")[0]) for f in source.glob("*_clean.npz")}
    )
    layers, masks, rows, failed = {}, {}, [], []
    for shot in shots:
        try:
            tokeye = clean_path(source, shot)
            if tokeye is None:
                raise FileNotFoundError(f"{shot} has no TokEye mask in {source}")
            data = tokeye.read_bytes()
            lit, grid, y0, dy, mask = make(paths, shot, saved.get(shot), data)
        except (KeyError, OSError, ValueError) as error:
            failed.append(shot)
            print(f"{shot}: {type(error).__name__}: {error}", flush=True)
            continue
        named = {"file": tokeye.name, "sha256": hashlib.sha256(data).hexdigest()}
        save(layer_file(paths, shot), grid, y0, dy, lit)
        layers[str(shot)] = named
        if mask is not None:
            mask.save(review_file(paths, shot))
            masks[str(shot)] = named
            rows.append(summary(mask, f"{shot}.npz"))
    origin = {"source": str(source)}
    _meta(layer_dir(paths), {"layer": LAYER, **origin}, layers, failed)
    labels_file = labels.labels_path(directory)
    _meta(
        review_dir(paths),
        {
            "pseudo": REVIEW,
            "rules": "pseudo-v1",
            **origin,
            "labels": str(labels_file),
            "labels_sha256": sha256_of(labels_file) if labels_file.is_file() else None,
        },
        masks,
        failed,
    )
    index = review_dir(paths) / "index.csv"
    if index.is_file():  # as meta.json: a run over some shots keeps the others' rows
        done = {str(r["shot"]) for r in rows}
        with open(index, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["shot"] not in done] + rows
    rows.sort(key=lambda r: int(r["shot"]))
    with atomic_path(index) as tmp, open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"wrote {len(layers)} layers to {layer_dir(paths)} and {len(masks)} masks "
        f"to {review_dir(paths)}; {len(failed)} failed"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
