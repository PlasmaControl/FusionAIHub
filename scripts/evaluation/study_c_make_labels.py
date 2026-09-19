"""Study C label pass (CPU-only): ELM + magnetics band-power per window.

Labels come from raw-HDF5 signals the model never sees as input
(filterscopes D-alpha peaks for ELMs; mirnov/mhr band power as the n=1
activity proxy), on the same fixed window grid as the latent extraction
(``t = warmup + k·stride``, k = 0..43). Analyses join to latents on
``(shot, t_start_s)`` — no dependency on dataset length bookkeeping, and
out-of-range windows are flagged, not dropped.

Outputs per split under ``<out-root>/labels/``:
  * ``windows_<split>.csv`` — one row per (shot, window)
  * ``shots_<split>.csv``   — shot-level meta (presence, ELM rate, gates)

Runs in ~6 min for both splits with ``--procs 6`` on a login node.
"""

import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "src"))
sys.path.insert(0, str(_HERE.parent))

import h5py  # noqa: E402

from tfm_eval.ckpt import DEFAULT_CKPT, load_ckpt  # noqa: E402
from tfm_eval.data import (  # noqa: E402
    assert_val_matches_training_cache,
    matched_train_subset,
    resolve_split,
)
from tfm_eval.labels import (  # noqa: E402
    band_power_labels,
    elm_labels,
    modality_present,
)

WINDOW_S = 0.05
WARMUP_S = 1.0
STRIDE_S = 0.25
N_WINDOWS = 44  # t in [1.0, 11.75] — matches max_duration_s=12.0 grids

GRID = (WARMUP_S + STRIDE_S * np.arange(N_WINDOWS)).astype(np.float64)

WINDOW_COLS = [
    "shot", "t_start_s", "elm_now", "elm_next", "n_peaks_now",
    "dalpha_max_now", "elm_in_range", "mirnov_bp", "mhr_bp",
]
SHOT_COLS = [
    "shot", "filterscopes_present", "mirnov_present", "mhr_present",
    "n_peaks_total", "elm_rate_hz",
]


def one_shot(path_str: str):
    path = Path(path_str)
    shot = path.name.replace("_processed.h5", "")
    with h5py.File(path, "r") as h5:
        present = {
            name: modality_present(h5, name)
            for name in ("filterscopes", "mirnov", "mhr")
        }
    elm = elm_labels(path, GRID, window_s=WINDOW_S) if present["filterscopes"] else None
    bp = band_power_labels(path, GRID, window_s=WINDOW_S)

    nan = np.full(N_WINDOWS, np.nan, dtype=np.float32)
    false = np.zeros(N_WINDOWS, dtype=bool)
    rows = []
    for j in range(N_WINDOWS):
        rows.append([
            shot,
            f"{GRID[j]:.2f}",
            int((elm["elm_now"] if elm else false)[j]),
            int((elm["elm_next"] if elm else false)[j]),
            int((elm["n_peaks_now"] if elm else false)[j]),
            f"{(elm['dalpha_max_now'] if elm else nan)[j]:.6g}",
            int((elm["in_range"] if elm else false)[j]),
            f"{bp.get('mirnov_bp', nan)[j]:.6f}",
            f"{bp.get('mhr_bp', nan)[j]:.6f}",
        ])
    meta = [
        shot,
        int(present["filterscopes"]),
        int(present["mirnov"]),
        int(present["mhr"]),
        int(elm["n_peaks_total"]) if elm else 0,
        f"{elm['elm_rate_hz']:.3f}" if elm else "nan",
    ]
    return rows, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--max-shots", type=int, default=None)
    ap.add_argument(
        "--out-root",
        default=str(_HERE.parents[2] / "data" / "outputs" / "eval_suite"
                    / "e2e_stage1_best"),
    )
    args = ap.parse_args()
    out_dir = Path(args.out_root) / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ck_args = load_ckpt(args.checkpoint).get("args", {}) or {}
    train_files, val_files = resolve_split(
        ck_args.get("data_dir", "/lustre/orion/fus187/proj-shared/foundation_model"),
        seed=ck_args.get("seed", 42),
        val_fraction=ck_args.get("val_fraction", 0.1),
    )
    assert_val_matches_training_cache(val_files)
    splits = {
        "val": list(val_files),
        "train": matched_train_subset(train_files, len(val_files)),
    }

    for split, files in splits.items():
        if args.max_shots:
            files = files[: args.max_shots]
        print(f"[{time.time()-t0:7.1f}s] {split}: {len(files)} shots "
              f"({args.procs} procs)", flush=True)
        with Pool(args.procs) as pool:
            results = []
            for i, res in enumerate(
                pool.imap(one_shot, [str(f) for f in files], chunksize=8)
            ):
                results.append(res)
                if (i + 1) % 100 == 0:
                    print(f"[{time.time()-t0:7.1f}s]   {i+1}/{len(files)}",
                          flush=True)

        win_path = out_dir / f"windows_{split}.csv"
        shot_path = out_dir / f"shots_{split}.csv"
        with open(win_path.with_suffix(".tmp"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(WINDOW_COLS)
            for rows, _ in results:
                w.writerows(rows)
        Path(win_path.with_suffix(".tmp")).rename(win_path)
        with open(shot_path.with_suffix(".tmp"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(SHOT_COLS)
            w.writerows(meta for _, meta in results)
        Path(shot_path.with_suffix(".tmp")).rename(shot_path)

        n_elmy = sum(1 for _, m in results if int(m[4]) > 0)
        print(f"[{time.time()-t0:7.1f}s] {split} done → {win_path.name}: "
              f"{n_elmy}/{len(results)} shots with ELMs", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
