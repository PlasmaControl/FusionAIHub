"""Diagnostic (left cut face) and actuator (right cut face) marker geometry in (R, Z) [m] for the DIII-D render.

Real positions: magnetic probes (EFIT machine file, mhdin.dat XMP2/YMP2), CER chord radii (chord tables).
Schematic positions (as in overview_machine.png): Thomson, ECE, CO2 interferometer, NBI, ECH, 3D coils, gas valve.
Face 1 = left cut face (diagnostics), face 0 = right cut face (actuators).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nstx_geometry"))
from mhdin import read_mhdin  # noqa: E402

DIAG, ACT = 1, 0


def build():
    m = read_mhdin("/p/efit/support_files/DIII-D/green/181292/mhdin.dat")
    pts = []
    add = lambda name, R, Z, face: pts.append(dict(name=name, R=float(R), Z=float(Z), face=face))
    for i, (r, z) in enumerate(zip(m["XMP2"], m["YMP2"])):                    # magnetic probes (real)
        add(f"mag{i}", r, z, DIAG)
    for i, z in enumerate(np.linspace(-0.65, 0.75, 15)):                        # Thomson, vertical at R = 1.94
        add(f"ts{i}", 1.94, z, DIAG)
    for i, r in enumerate(np.linspace(1.62, 2.24, 14)):                         # ECE, midplane
        add(f"ece{i}", r, 0.0, DIAG)
    rows = list(csv.DictReader(l for l in open(ROOT / "d3d_geometry" / "cer_chords_202400.csv") if not l.startswith("#")))
    for i, r in enumerate(sorted({float(x["R_30LT"]) for x in rows if x["type"] == "T" and x["R_30LT"]})):
        add(f"cer{i}", r / 100, -0.03, DIAG)                                     # CER radii (real), midplane
    for i, r in enumerate((1.48, 1.94, 2.10)):                                  # CO2 interferometer, vertical chords
        add(f"co2top{i}", r, 1.25, DIAG)
        add(f"co2bot{i}", r, -1.25, DIAG)
    for i, z in enumerate((0.12, 0.0, -0.12)):                                  # NBI bands, from the wall inwards
        add(f"nbi_in{i}", 1.78, z, ACT)
        add(f"nbi_out{i}", 2.45, z, ACT)
    add("ech_src", 2.40, 0.90, ACT)                                              # ECH launcher, beams to the core
    for i, (r, z) in enumerate(((1.80, 0.30), (1.86, 0.18), (1.92, 0.08), (1.76, 0.40))):
        add(f"ech_tgt{i}", r, z, ACT)
    add("icoil_u", 2.30, 0.80, ACT)                                              # 3D (I-) coils, upper/lower outboard
    add("icoil_l", 2.30, -0.80, ACT)
    add("gas_src", 2.15, -1.25, ACT)                                             # gas valve, lower outboard
    add("gas_tgt", 1.95, -1.00, ACT)
    for f, nm in ((DIAG, "diag"), (ACT, "act")):                                 # label anchors (outboard edge)
        add(f"{nm}_label", 2.45, -1.55, f)
    return pts


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "markers.json"
    out.write_text(json.dumps(build()))
    print(len(build()), "points ->", out)
