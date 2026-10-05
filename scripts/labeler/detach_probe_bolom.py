#!/usr/bin/env python
"""List the BOLOM tree's nodes for one shot, to find the chord geometry.

    pixi run --frozen -e labelmaker fdp run python \\
        scripts/labeler/detach_probe_bolom.py --shot 189057

Run on the login node under the fdp wrapper, one connection. Writes
`$LABELER_ROOT/round4/detach/bolom_nodes.txt`: every node of the tree, its usage and
(for a numeric node holding a small array) its shape. The detachment figure draws the
bolometer chords over the flux surfaces when the geometry is found here and falls
back to chord profiles when it is not. An authentication failure stops the run.

It also records whether a 2D bolometer emissivity exists on the shot, the object the
views figure would overlay with EFIT: the `BOLFIT01.TWOD` structure's children, the
1D fit's children and the corpus `bolo` group's shape, in
`docs/labeler/results/detachment_bolometer_availability.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECORD = REPO / "docs/labeler/results/detachment_bolometer_availability.json"
CORPUS = Path("/scratch/gpfs/EKOLEMEN/foundation_model")

AUTH_WORDS = ("auth", "token", "login", "credential", "401", "403", "permission")
#: Nodes whose name suggests geometry or calibration; their values are read.
KEYS = ("R", "Z", "ANGLE", "GEOM", "CHORD", "PHI", "LOS", "RAD", "THETA", "XY", "POS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shot", type=int, required=True)
    args = parser.parse_args()
    import MDSplus

    out = Path(os.environ["LABELER_ROOT"]) / "round4" / "detach" / "bolom_nodes.txt"
    lines = []
    try:
        tree = MDSplus.Tree("bolom", args.shot, "readonly")
        for node in tree.getNodeWild("***"):
            path = str(node.path)
            usage = str(node.usage)
            info = ""
            upper = path.upper()
            if usage == "NUMERIC" and any(k in upper.split(":")[-1] for k in KEYS):
                try:
                    value = node.data()
                    info = f" shape={getattr(value, 'shape', ())}"
                    if getattr(value, "size", 99) <= 64:
                        info += f" value={value.tolist()}"
                except Exception as error:  # noqa: BLE001  a node may hold nothing
                    info = f" read-error={type(error).__name__}"
            lines.append(f"{path} {usage}{info}")
    except Exception as error:  # noqa: BLE001
        text = f"{type(error).__name__}: {str(error)[:200]}"
        print("ERROR", text)
        if any(word in text.lower() for word in AUTH_WORDS):
            print("AUTH ERROR: stopping, the login has lapsed", file=sys.stderr)
            return 3
        return 1
    out.write_text("\n".join(lines) + "\n")
    print(len(lines), "nodes ->", out)
    write_record(args.shot, tree)
    return 0


def write_record(shot: int, tree) -> None:
    """Does a 2D bolometer emissivity (a spatial reconstruction) exist here?"""
    children = {}
    for name in ("ONED", "TWOD"):
        node = tree.getNode(rf"\BOLOM::TOP.BOLFIT01.{name}")
        children[name] = [str(n.node_name) for n in node.getDescendants()]
    corpus = {}
    try:
        import h5py

        with h5py.File(CORPUS / f"{shot}_processed.h5", "r") as handle:
            corpus = {
                "bolo_ydata_shape": list(handle["bolo"]["ydata"].shape),
                "bolo_xdata_length": int(handle["bolo"]["xdata"].shape[0]),
            }
    except Exception as error:  # noqa: BLE001  the corpus file may be absent
        corpus = {"error": f"{type(error).__name__}: {str(error)[:120]}"}
    record = {
        "shot": shot,
        "bolfit01_twod_children": children["TWOD"],
        "bolfit01_oned_children": children["ONED"],
        "corpus_bolo": corpus,
        "has_2d_emissivity": bool(children["TWOD"]),
        "conclusion": (
            "no 2D bolometer emissivity: the corpus `bolo` group is 48 raw chord "
            "voltages (channels by time) and the BOLFIT01.TWOD structure has no "
            "children; the detach-ui source has no other bolometer image. The views "
            "figure shows PRAD_DIVL and PRAD_TOT traces instead."
            if not children["TWOD"]
            else "BOLFIT01.TWOD has children; read them before drawing"
        ),
    }
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    print("wrote", RECORD)


if __name__ == "__main__":
    raise SystemExit(main())
