"""TokEye's lines over the whole shot: the layer the review page draws under the
pseudo-mask.

    python -m labeler.ae.seg.whole [--shots S ...]

The pseudo-mask (`pseudo`) lies inside the owner's label windows, which end at
2 s, and inside 80-250 kHz. TokEye's whole-shot masks
(`$LABELER_ROOT/ae/masks-full`, 0-6 s) cover the record. This layer is a
picture, not a label: a pixel is lit where TokEye lights at least two of the four
chords, 0-250 kHz, AE or not, on the review store's level-8 grid (the pseudo-mask's
`tokeye_rows` and `pool_columns`). Nothing trains on it, scores it or saves a
decision on it.

**Output.** `$LABELER_ROOT/segmentation/alfven_eigenmode/tokeye-full/<shot>.npz`
(`bits`, the `(n_y, n)` lit pixels bit-packed along time; `n`, `t0_ms`, `dt_ms`,
`y0_khz`, `dy_khz`) and `meta.json` beside them, which names each shot's TokEye
file by its sha256.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import numpy as np

from ...config import Paths, atomic_path, git_sha
from ...events.review.rows import Grid
from ..xpower.data import clean_path, store_rows, tokeye_clean
from . import EVENT
from .pseudo import LEVEL, pool_columns, tokeye_rows

LAYER = "tokeye-full"


def masks_full(paths: Paths) -> Path:
    """TokEye's whole-shot cleaned masks, `<shot>_<split>_clean.npz`."""
    return paths.root / "ae" / "masks-full"


def layer_dir(paths: Paths) -> Path:
    return paths.root / "segmentation" / EVENT / LAYER


def layer_file(paths: Paths, shot: int) -> Path:
    return layer_dir(paths) / f"{int(shot)}.npz"


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
    """What the review page draws for one shot, or None without a layer.

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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--shots", type=int, nargs="*", help="default: every TokEye shot")
    args = p.parse_args(argv)
    paths = Paths.from_env()
    source = masks_full(paths)
    shots = args.shots or sorted(
        {int(f.name.split("_")[0]) for f in source.glob("*_clean.npz")}
    )
    hashes, failed = {}, []
    for shot in shots:
        try:
            tokeye = clean_path(source, shot)
            if tokeye is None:
                raise FileNotFoundError(f"{shot} has no TokEye mask in {source}")
            data = tokeye.read_bytes()
            grid, values, y0, dy = store_rows(
                paths.spectrogram_file(EVENT, shot), LEVEL
            )
            lit = build(grid, values.shape[1], tokeye_clean(BytesIO(data)))
        except (KeyError, OSError, ValueError) as error:
            failed.append(shot)
            print(f"{shot}: {type(error).__name__}: {error}", flush=True)
            continue
        save(layer_file(paths, shot), grid, y0, dy, lit)
        hashes[str(shot)] = {
            "file": tokeye.name,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    meta = {
        "layer": LAYER,
        "source": str(source),
        "shots": len(hashes),
        "failed": failed,
        "tokeye": hashes,
        "git_sha": git_sha(),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with atomic_path(layer_dir(paths) / "meta.json") as tmp:
        tmp.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {len(hashes)} layers to {layer_dir(paths)}; {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
