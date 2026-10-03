#!/usr/bin/env python
"""Probe processed diagnostics and geometry; stop immediately on auth failure."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from labeler.features import resolve_fdp

AUTH = ("auth", "token", "login", "credential", "401", "403", "permission")


def main():
    import MDSplus

    out = Path(os.environ["LABELER_ROOT"]) / "round4/detach"
    records = []
    # Processed probe tree names, processed DTS, and thermal camera trees.
    for tree_name in ("LANGMUIR", "LANG", "PROBES", "DTS", "THOMSON", "IRTV"):
        try:
            tree = MDSplus.Tree(tree_name, 189057, "readonly")
            nodes = [str(n.path) for n in tree.getNodeWild("***")]
            (out / f"probe_{tree_name.lower()}_nodes.txt").write_text("\n".join(nodes))
            records.append(
                {
                    "tree": tree_name,
                    "status": "ok",
                    "nodes": len(nodes),
                    "matches": [
                        n
                        for n in nodes
                        if any(
                            k in n.upper()
                            for k in ("JSAT", "ISAT", "POS", "HEAT", "FLUX")
                        )
                    ],
                }
            )
        except Exception as error:  # noqa: BLE001  diagnostic absence is recorded
            message = f"{type(error).__name__}: {error}"[:300]
            records.append({"tree": tree_name, "status": message})
            if any(w in message.lower() for w in AUTH):
                (out / "processed_probe.json").write_text(json.dumps(records, indent=1))
                raise SystemExit(3)
        time.sleep(1)
    for node in ("AFRAC", "DOD", "IRTQ", "IRTVQ", "QPERP"):
        try:
            rec = resolve_fdp._fetch_ptdata(node, 189057)
            records.append({"point": node, "status": "ok", "samples": len(rec["data"])})
        except Exception as error:  # noqa: BLE001  diagnostic absence is recorded
            message = f"{type(error).__name__}: {error}"[:300]
            records.append({"point": node, "status": message})
            if any(w in message.lower() for w in AUTH):
                (out / "processed_probe.json").write_text(json.dumps(records, indent=1))
                raise SystemExit(3)
        time.sleep(1)
    (out / "processed_probe.json").write_text(json.dumps(records, indent=1))
    print(json.dumps(records, indent=1))


if __name__ == "__main__":
    main()
