"""A whole-shot AE tree for the round-three tests: `ae_tree`'s, to 3 s.

`build` makes `ae_tree.build`'s tree, then gives every shot what v3 reads:
- a review store to 3200 ms, with the AE line (bin 150, ~146 kHz) over `AE_MS`
  and again over `LATE_AE_MS`, after 2 s, and the MHD mode over `MHD_MS`;
- TokEye's whole-shot masks (`ae/masks-full/<shot>_<split>_clean.npz`) to
  3000 ms, lit on bin 299 over both AE spans and on bins 39 and 199 over
  `MHD_MS`, with UCI's annotation over `AE_MS` (it ends at 2 s);
- SELDNet's whole-shot input (`ae/dataset-full/<shot>_<split>.npz`, zeros);
- the owner's saved labels over 0-3000 ms (`SPANS`), present on both AE spans.

`manifests` then lists both directories as one finished job wrote them.

The source table (`format/`) and v1's masks and dataset stay `ae_tree`'s, 0-2 s.
`tokeye_dt=0.256` gives both TokEye records the real column spacing, as the
segmentation tests need (a coarser record leaves store columns unlit).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from labeler.ae import full
from labeler.ae.xpower import data
from labeler.config import Paths
from labeler.events.review import rows

from . import ae_tree
from .ae_tree import AE_MS, DY, MHD_MS, TOKEYE_DT

EVENT = "alfven_eigenmode"
LATE_AE_MS = (2200, 2600)
END_MS = 3000
AE_SPANS = (AE_MS, LATE_AE_MS)
SPANS = [
    (0, AE_MS[0], 0),
    (*AE_MS, 1),
    (AE_MS[1], LATE_AE_MS[0], 0),
    (*LATE_AE_MS, 1),
    (LATE_AE_MS[1], END_MS, 0),
]


def ann_until(dt: float = TOKEYE_DT) -> float:
    """v1's last TokEye column in `ae_tree` (its record runs to 2002 ms)."""
    return -0.768 + dt * int(2002.8 // dt)


ANN_UNTIL_MS = ann_until()
#: Frames (10 ms) of 0-3000 ms where the owner says present, and the MHD ones.
AE_FRAMES = [*range(30, 90), *range(220, 260)]
MHD_FRAMES = list(range(120, 150))


def _inside(t, spans) -> np.ndarray:
    return np.any([(t >= a) & (t < b) for a, b in spans], axis=0)


def _store(path: Path, rng) -> None:
    grid = rows.Grid(t0_ms=-100.0, dt_ms=0.256, n=int((END_MS + 300) / 0.256))
    t = grid.t0_ms + (np.arange(grid.n) + 0.5) * grid.dt_ms
    ae, mhd = _inside(t, AE_SPANS), _inside(t, [MHD_MS])
    built = []
    for name in data.CROSS_ROWS:
        values = rng.integers(0, 40, (257, grid.n), dtype=np.uint8)
        values[150, ae] = 220  # 146 kHz
        for k in (1, 5, 6):  # 19.5 kHz and its 98 and 117 kHz harmonics
            values[20 * k, mhd] = 200
        built.append(
            rows.ImageRow(
                name,
                name,
                values,
                y0=0.0,
                dy=DY,
                y_units="kHz",
                z_lo=-3.0,
                z_hi=27.0,
                z_units="dB",
                band=(80.0, 250.0),
            )
        )
    rows.write(path, grid, built, event=EVENT, shot=int(path.stem))


def _masks_full(paths: Paths, shot: int, split: str, dt: float) -> None:
    t = -0.768 + dt * np.arange(int((END_MS + 2.8) // dt) + 1)
    clean = np.zeros((4, 512, len(t)), dtype=bool)
    clean[:, 299, _inside(t, AE_SPANS)] = True  # 146 kHz
    mhd = _inside(t, [MHD_MS])
    clean[:, 39, mhd] = True  # 19.5 kHz
    clean[:, 199, mhd] = True  # 98 kHz, the fifth harmonic
    ann = _inside(t, [AE_MS]) & (t <= ann_until(dt))
    masks = full.masks_full_dir(paths)
    masks.mkdir(parents=True, exist_ok=True)
    np.savez(
        masks / f"{shot}_{split}_clean.npz",
        t_ms=t,
        mask_clean=np.packbits(clean, axis=-1),
        ann=ann.astype(np.uint8),
        ann_until_ms=np.float64(ann_until(dt)),
        t0_input_ms=np.float64(0.0),
        fs_hz=np.float64(500_000.0),
        tile_ms=np.float64(0.0),
    )
    dataset = full.dataset_full_dir(paths)
    dataset.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dataset / f"{shot}_{split}.npz",
        spec=np.zeros((4, 348, len(t)), dtype=np.float16),
    )


def _labels(path: Path, shots) -> None:
    lines = ["shot,category,t_start,t_end,confidence"]
    for shot in shots:
        lines += [f"{shot},{c},{a},{b}," for a, b, c in SPANS]
    path.write_text("\n".join(lines) + "\n")


def build(
    tmp_path: Path,
    splits: dict[int, str],
    seed: int = 0,
    reviewed=None,
    tokeye_dt: float = TOKEYE_DT,
) -> Paths:
    """`ae_tree.build`'s tree, then the whole-shot store, masks-full and
    dataset-full of every shot in `splits`, and the saved labels over 0-3 s of
    `reviewed` (default: every shot)."""
    paths = ae_tree.build(
        tmp_path, splits, seed=seed, reviewed=reviewed, tokeye_dt=tokeye_dt
    )
    rng = np.random.default_rng(seed + 1)
    for shot, split in sorted(splits.items()):
        _store(paths.spectrogram_file(EVENT, shot), rng)
        _masks_full(paths, shot, split, tokeye_dt)
    saved = sorted(splits) if reviewed is None else sorted(reviewed)
    _labels(paths.label_tables / EVENT / "review" / "labels.csv", saved)
    return paths


def job_log(
    paths: Paths, job: str, stems, start: float, end: float, root=None
) -> Path:
    """`runs/slurm/<job>.out` as `ae_masks_full.sbatch` writes it: the start
    line, one line per stem written, the finish line (times in s, UTC)."""
    stems = list(stems)
    lines = [
        (
            f"job {job} on node01 at "
            f"{datetime.fromtimestamp(math.floor(start), UTC).isoformat()}, "
            f"root {root or paths.root}"
        ),
        f"device=cpu shots={len(stems)} batch=2 tile_ms=0 workers=1",
        *(
            f"[{i}/{len(stems)}] {s} frames=300 clean pixels=0 model and write 1.0s"
            for i, s in enumerate(stems, 1)
        ),
        f"finished at {datetime.fromtimestamp(math.ceil(end), UTC).isoformat()}",
    ]
    log = paths.runs / "slurm" / f"{job}.out"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("\n".join(lines) + "\n")
    return log


def manifests(paths: Paths, job: str = "7001", commit: str = "abc1234") -> None:
    """Both directories' manifests, as one finished job (`job_log`, from a
    minute before the first file to a minute after the last) wrote them."""
    dirs = (full.masks_full_dir(paths), full.dataset_full_dir(paths))
    stems = sorted(p.name.removesuffix(".npz") for p in dirs[1].glob("*.npz"))
    times = [p.stat().st_mtime for d in dirs for p in d.iterdir()]
    job_log(paths, job, stems, min(times) - 60, max(times) + 60)
    written = [full.read_job(paths, job, commit)]
    for directory in dirs:
        full.write_manifest(paths, directory, written)


class AllFire(torch.nn.Module):
    """A stand-in SELDNet firing on every column, whatever the record's length."""

    def forward(self, x):
        logit = torch.full((x.shape[2],), 5.0)
        return torch.stack([logit, torch.zeros_like(logit)], dim=-1)[None]
