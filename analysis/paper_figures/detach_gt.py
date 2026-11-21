"""Paper figures: measured gas (act_global, per valve) and filterscopes (raw, frame means over 1-6 s)

    python analysis/paper_figures/detach_gt.py 201079 201080 201081 201082 201083 201084 201085
"""
import os
import sys
from pathlib import Path
import h5py, numpy as np, torch
R = "/scratch/gpfs/EKOLEMEN/ps9551/ignite/models/ignite_prod_v4"
RAW = "/tigerdata/EKOLEMEN/d3d_fusion_data"
VALVES = ("GASA", "GASB", "GASC", "GASD", "GASE", "LOB1", "LOB2", "PFX1", "PFX2", "PFX3", "UOB")
FS = [f"FS{i:02d}" for i in range(1, 9)]
T0, FRAME, N = 1.0, 0.05, 100
out = Path(os.environ.get("PAPER_FIG_DATA", "/scratch/gpfs/EKOLEMEN/ps9551/ignite/paper_figures/data"))
for s in sys.argv[1:]:
    a = torch.load(f"{R}/act_global/{s}.pt", weights_only=False)
    raw = np.asarray(a["actuators_raw"], dtype=np.float32)[:N]
    gas = raw[:, 36:47]
    rec = {"gas": gas, "valves": np.array(VALVES), "pinj": raw[:, 12:20].sum(1),
           "ech": raw[:, :12].sum(1), "tinj": raw[:, 28:36].sum(1)}
    fs = np.full((N, len(FS)), np.nan, np.float32)
    with h5py.File(f"{RAW}/{s}.h5", "r") as f:
        g = f[s]
        for j, name in enumerate(FS):
            node = None
            for tree in g:
                k = f"\\SPECTROSCOPY::{name}"
                if k in g[tree]:
                    node = g[tree][k]; break
            if node is None:
                continue
            y = np.asarray(node["data"], dtype=np.float64).ravel()
            t = np.asarray(node["dim0"], dtype=np.float64).ravel()
            t = t / 1e3 if t.max() > 100 else t  # ms -> s
            edges = T0 + FRAME * np.arange(N + 1)
            idx = np.searchsorted(t, edges)
            fs[:, j] = [np.nanmean(y[i0:i1]) if i1 > i0 else np.nan for i0, i1 in zip(idx[:-1], idx[1:])]
    rec["fs"], rec["fs_names"] = fs, np.array(FS)
    np.savez_compressed(out / f"detach_{s}.npz", **rec)
    print(s, "gas total mean", float(gas[20:].sum(1).mean()), "fs live", int(np.isfinite(fs).all(0).sum()))
