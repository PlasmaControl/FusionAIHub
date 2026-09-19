"""Map omega-fetched n1rms/n2rms onto the Study C window grid.

Input: mdsh5 output from ``scripts/data_fetching_omega/n1rms_fetch/`` — one
HDF5 file (or a directory of them), laid out ``<shot>/MHD/<signal>/{data,
dim0}`` with ``dim0`` the time axis. Signal groups are located by substring
("N1RMS"/"N2RMS") so the exact MDSplus path escaping doesn't matter.

Output: ``<labels-dir>/n1rms_windows_{val,train}.csv`` on the same fixed
window grid as ``study_c_make_labels.py`` (t_start = 1.0 + 0.25·k, k<44;
join key ``(shot, t_start_s)``): per-window mean amplitude for the current
and next 50 ms windows, raw and log10. Rows are emitted for every shot in
the split (NaN where the fetch is missing) so joins stay trivial.

Usage::

    python scripts/evaluation/ingest_n1rms.py /path/to/n1rms_raw \
        --labels-dir data/outputs/eval_suite/e2e_stage1_best/labels
"""

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np

# Window grid — MUST match study_c_make_labels.py (kept in sync by the
# assertion below rather than an import, which would drag in torch).
WINDOW_S = 0.05
WARMUP_S = 1.0
STRIDE_S = 0.25
N_WINDOWS = 44
GRID = (WARMUP_S + STRIDE_S * np.arange(N_WINDOWS)).astype(np.float64)

COLS = [
    "shot", "t_start_s", "n1rms_now", "n1rms_next", "n2rms_now",
    "n2rms_next", "log10_n1rms_now", "log10_n1rms_next", "n1rms_present",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw_path", help="mdsh5 output file or directory of .h5 files")
    ap.add_argument(
        "--labels-dir",
        default="data/outputs/eval_suite/e2e_stage1_best/labels",
        help="directory with shots_{val,train}.csv; output lands here too",
    )
    return ap.parse_args()


def find_signal(shot_grp, key: str):
    """Return (t_s, y) for the signal whose group name contains `key`."""
    hits = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and name.split("/")[-1] == "data" \
                and key in name.upper():
            hits.append(name)

    shot_grp.visititems(visit)
    if not hits:
        return None
    data = shot_grp[hits[0]]
    dim0 = shot_grp[hits[0].rsplit("/", 1)[0] + "/dim0"]
    t = np.asarray(dim0, dtype=np.float64).ravel()
    y = np.asarray(data, dtype=np.float64).ravel()
    if t.size != y.size or t.size == 0:
        return None
    if t.max() > 100.0:  # DIII-D MDSplus times are in ms
        t = t / 1000.0
    return t, y


def window_means(t, y, grid, window_s):
    """Mean of y over [g, g+window_s) for each grid start; NaN if empty."""
    out = np.full(grid.shape, np.nan)
    order = np.argsort(t)
    t, y = t[order], y[order]
    lo = np.searchsorted(t, grid, side="left")
    hi = np.searchsorted(t, grid + window_s, side="left")
    for j in range(grid.size):
        if hi[j] > lo[j]:
            out[j] = np.nanmean(y[lo[j]:hi[j]])
    return out


def read_split_shots(labels_dir: Path, split: str):
    with open(labels_dir / f"shots_{split}.csv") as f:
        return [int(row["shot"]) for row in csv.DictReader(f)]


def main() -> int:
    args = parse_args()
    labels_dir = Path(args.labels_dir)
    raw = Path(args.raw_path)
    files = sorted(raw.glob("*.h5")) if raw.is_dir() else [raw]
    if not files:
        print(f"no .h5 files under {raw}")
        return 1

    # shot -> {"n1": (t, y), "n2": (t, y)}
    fetched = {}
    for fp in files:
        with h5py.File(fp, "r") as f:
            for shot_key in f:
                try:
                    shot = int(shot_key)
                except ValueError:
                    continue
                n1 = find_signal(f[shot_key], "N1RMS")
                n2 = find_signal(f[shot_key], "N2RMS")
                if n1 is not None or n2 is not None:
                    fetched[shot] = {"n1": n1, "n2": n2}
    print(f"{len(fetched)} shots with n1rms/n2rms data in {len(files)} file(s)")

    grid_next = GRID + WINDOW_S
    for split in ("val", "train"):
        shots = read_split_shots(labels_dir, split)
        n_present = 0
        out_path = labels_dir / f"n1rms_windows_{split}.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLS)
            for shot in shots:
                sig = fetched.get(shot)
                nan = np.full(GRID.shape, np.nan)
                if sig and sig["n1"] is not None:
                    n_present += 1
                    n1_now = window_means(*sig["n1"], GRID, WINDOW_S)
                    n1_next = window_means(*sig["n1"], grid_next, WINDOW_S)
                else:
                    n1_now = n1_next = nan
                if sig and sig["n2"] is not None:
                    n2_now = window_means(*sig["n2"], GRID, WINDOW_S)
                    n2_next = window_means(*sig["n2"], grid_next, WINDOW_S)
                else:
                    n2_now = n2_next = nan
                with np.errstate(divide="ignore", invalid="ignore"):
                    l1_now = np.where(n1_now > 0, np.log10(n1_now), np.nan)
                    l1_next = np.where(n1_next > 0, np.log10(n1_next), np.nan)
                present = int(sig is not None and sig["n1"] is not None)
                for j in range(GRID.size):
                    w.writerow([
                        shot, f"{GRID[j]:.2f}",
                        f"{n1_now[j]:.6g}", f"{n1_next[j]:.6g}",
                        f"{n2_now[j]:.6g}", f"{n2_next[j]:.6g}",
                        f"{l1_now[j]:.6f}", f"{l1_next[j]:.6f}",
                        present,
                    ])
        print(f"{split}: {n_present}/{len(shots)} shots present -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
