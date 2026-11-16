"""Fetch a DIII-D EFIT equilibrium (gEQDSK) over the Fusion Data Platform.

Reads every node of ``\\<TREE>::TOP.RESULTS.GEQDSK`` for one shot through
FDP's streamed MDSplus access (a few MB, no staging), saves all time slices
to ``geqdsk_<shot>.npz`` and writes the slice nearest ``--time`` as a
standard text g-file ``g<shot>.<ms>``. Both go to ``PAPER_FIG_DATA``.

Run inside the FDP project (a login node is enough)::

    E=/scratch/gpfs/EKOLEMEN/ps9551/fdp-demo/.pixi/envs/default/bin
    BEARER_TOKEN_FILE=~/.fdp/token $E/fdp run $E/python \\
        analysis/paper_figures/fetch_geqdsk.py 204011 --time 3.0
"""
import argparse
import os
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get(
    "PAPER_FIG_DATA", "/scratch/gpfs/EKOLEMEN/ps9551/ignite/paper_figures/data"))
# Scalar and profile records of a g-file, in file order, by their MDSplus
# node names (XDIM is the g-file's RDIM, RZERO its RCENTR).
HEADER = ("XDIM", "ZDIM", "RZERO", "RGRID1", "ZMID", "RMAXIS", "ZMAXIS",
          "SSIMAG", "SSIBRY", "BCENTR", "CPASMA")
PROFILES = ("FPOL", "PRES", "FFPRIM", "PPRIME")


def read_tree(tree_name, shot):
    """All readable GEQDSK nodes of one shot.

    Parameters
    ----------
    tree_name : str
        EFIT tree, e.g. ``"EFIT01"``.
    shot : int
        Shot number.

    Returns
    -------
    dict of str to ndarray
        Node name to data; nodes that fail to read are left out.
    """
    import MDSplus

    tree = MDSplus.Tree(tree_name, shot)
    top = tree.getNode(f"\\{tree_name}::TOP.RESULTS.GEQDSK")
    out = {}
    for node in list(top.getMembers()) + list(top.getChildren()):
        name = str(node.getNodeName()).strip()
        try:
            out[name] = np.asarray(node.data())
        except Exception:
            continue
    return out


def _block(values):
    """Values in the g-file layout: five per line, ``%16.9E``."""
    values = np.ravel(values)
    lines = []
    for i in range(0, len(values), 5):
        lines.append("".join(f"{v:16.9E}" for v in values[i:i + 5]))
    return "\n".join(lines) + "\n"


def write_gfile(g, k, path, shot, tree_name):
    """Write time slice ``k`` of ``g`` as a text gEQDSK file.

    Parameters
    ----------
    g : dict of str to ndarray
        Output of :func:`read_tree`.
    k : int
        Time-slice index.
    path : Path
        Destination file.
    shot : int
        Shot number, for the header.
    tree_name : str
        EFIT tree, for the header.
    """
    psi = np.asarray(g["PSIRZ"][k])           # (nh, nw), Z major
    nh, nw = psi.shape
    ms = int(round(float(g["GTIME"][k])))
    head = [float(np.ravel(g[n])[k]) if np.size(g[n]) > 1 else float(g[n])
            for n in HEADER]
    rdim, zdim, rcentr, rleft, zmid, rmaxis, zmaxis, simag, sibry, bcentr, \
        current = head
    text = f"  {tree_name} {shot} {ms}ms" .ljust(48) + f"{0:4d}{nw:4d}{nh:4d}\n"
    text += _block([rdim, zdim, rcentr, rleft, zmid])
    text += _block([rmaxis, zmaxis, simag, sibry, bcentr])
    text += _block([current, simag, 0.0, rmaxis, 0.0])
    text += _block([zmaxis, 0.0, sibry, 0.0, 0.0])
    for n in PROFILES:
        text += _block(g[n][k])
    text += _block(psi)
    text += _block(g["QPSI"][k])
    nb = int(np.ravel(g["NBBBS"])[k])
    lim = np.asarray(g["LIM"]).reshape(-1, 2)
    text += f"{nb:5d}{len(lim):5d}\n"
    text += _block(np.column_stack([g["RBBBS"][k][:nb], g["ZBBBS"][k][:nb]]))
    text += _block(lim)
    path.write_text(text)


def main():
    """Fetch the equilibrium and write the npz and the g-file."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("shot", type=int)
    p.add_argument("--time", type=float, default=3.0,
                   help="time (s) of the slice written as a g-file")
    p.add_argument("--tree", default="EFIT01")
    a = p.parse_args()
    g = read_tree(a.tree, a.shot)
    OUT.mkdir(parents=True, exist_ok=True)
    npz = OUT / f"geqdsk_{a.shot}.npz"
    np.savez_compressed(npz, tree=a.tree, shot=a.shot, **g)
    k = int(np.argmin(np.abs(np.asarray(g["GTIME"]) - 1e3 * a.time)))
    gfile = OUT / f"g{a.shot:06d}.{int(round(float(g['GTIME'][k]))):05d}"
    write_gfile(g, k, gfile, a.shot, a.tree)
    print(f"wrote {npz} ({len(g)} nodes) and {gfile}")


if __name__ == "__main__":
    main()
