#!/usr/bin/env python
"""Audit filterscope view metadata and FS01 availability without changing inputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

from labeler.config import Paths, sha256_of
from labeler.elm import labels, prepare

AUTH_WORDS = ("auth", "credential", "kerberos", "permission denied", "expired", "kinit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--shot", type=int, default=190643)
    parser.add_argument(
        "--tree", choices=("SPECTROSCOPY", "D3D"), default="SPECTROSCOPY"
    )
    args = parser.parse_args()
    paths = Paths.from_env()
    source = prepare.review_csv(paths)
    shots = sorted(map(int, labels.review_table(source).shot.unique()))
    availability = {}
    for shot in shots:
        with h5py.File(paths.corpus / f"{shot}_processed.h5") as handle:
            row = np.asarray(handle["filterscopes/ydata"][0])
            availability[str(shot)] = {
                "finite_samples": int(np.isfinite(row).sum()),
                "samples": int(row.size),
            }
    record = {
        "producer": "scripts/labeler/elm_filterscope_metadata.py",
        "script_sha256": sha256_of(Path(__file__)),
        "review_source": str(source),
        "review_source_sha256": sha256_of(source),
        "fs01_corpus_availability": availability,
        "fs01_finite_shots": sum(
            v["finite_samples"] > 0 for v in availability.values()
        ),
        "used_channels": ["FS02", "FS03", "FS04"],
        "fs01_exclusion": (
            "The original native-rate ELM-O cache contains FS02-FS04 only. "
            "Existing U-Net fits inherited that cache; FS01 was not trained or "
            "evaluated. Corpus availability is reported separately and does not "
            "justify declaring FS01 universally unavailable."
        ),
        "metadata_shot": args.shot,
        "metadata_tree": args.tree,
        "metadata": {},
        "views": {
            channel: {
                "pmt": f"PMT{int(channel[-2:]) + 9:02d}",
                "view": None,
                "status": "divertor versus midplane not identified in accessible metadata",
            }
            for channel in ("FS01", "FS02", "FS03", "FS04")
        },
        "view_limit": (
            "The filterscope metadata nodes expose calibration and photon-flux "
            "signals, not a sightline/location field. Neither corpus attributes "
            "nor retained native-rate cache metadata identify divertor/midplane "
            "views. PMT mappings are verified tree aliases, not inferred geometry."
        ),
        "auth_stopped": False,
    }
    if args.fetch:
        from toksearch import MdsSignal
        from toksearch.signal.mds import MdsTreeRegistry

        for channel in ("FS01", "FS02", "FS03", "FS04"):
            node = (
                rf"\SPECTROSCOPY::{channel}"
                if args.tree == "SPECTROSCOPY"
                else rf"\D3D::TOP.SPECTROSCOPY.FILTERSCOPE.PMT{int(channel[-2:]) + 9:02d}"
            )
            time.sleep(1)
            try:
                alias_signal = MdsSignal(
                    rf"\SPECTROSCOPY::{channel}",
                    "SPECTROSCOPY",
                    dims=(),
                    fetch_units=False,
                )
                alias_backend = alias_signal.sig
                alias_tree = MdsTreeRegistry().open_tree(
                    alias_backend.treename, args.shot, treepath=alias_backend.treepath
                )
                alias_node = alias_tree.getNode(rf"\SPECTROSCOPY::{channel}")
                pmt = str(alias_node.getParent().getNodeName()).strip()
                record["views"][channel]["pmt"] = pmt
                if args.tree == "D3D":
                    node = rf"\D3D::TOP.SPECTROSCOPY.FILTERSCOPE.{pmt}"
                signal = MdsSignal(node, args.tree, dims=(), fetch_units=False)
                backend = signal.sig
                tree = MdsTreeRegistry().open_tree(
                    backend.treename, args.shot, treepath=backend.treepath
                )
                handle = tree.getNode(node)
                parent = handle if args.tree == "D3D" else handle.getParent()
                members = list(handle.getMembers()) + list(parent.getMembers())
                descendants = list(parent.getChildren())
                for child in list(descendants):
                    descendants.extend(list(child.getMembers()))
                    descendants.extend(list(child.getChildren()))
                members.extend(descendants)
                record["metadata"][channel] = {
                    "alias_fullpath": str(alias_node.getFullPath()),
                    "fullpath": str(handle.getFullPath()),
                    "parent": str(parent.getFullPath()),
                    "members": [str(member.getFullPath()) for member in members],
                    "text_members": {
                        str(member.getFullPath()): str(member.record)
                        for member in members
                        if str(member.getUsage()) == "TEXT"
                    },
                    "descendant_usages": {
                        str(member.getFullPath()): str(member.getUsage())
                        for member in descendants
                    },
                }
                if channel == "FS01":
                    scope = (
                        r"\D3D::TOP.SPECTROSCOPY.FILTERSCOPE.***"
                        if args.tree == "D3D"
                        else r"\SPECTROSCOPY::TOP.FILTERSCOPE.***"
                    )
                    text_nodes = tree.getNodeWild(scope, "TEXT")
                    record["filterscope_text_nodes"] = {}
                    for item in text_nodes:
                        try:
                            value = str(item.record)
                        except Exception as error:  # noqa: BLE001 -- external metadata
                            value = f"{type(error).__name__}: {error}"
                            if any(word in value.lower() for word in AUTH_WORDS):
                                record["auth_stopped"] = True
                                break
                        record["filterscope_text_nodes"][str(item.getFullPath())] = (
                            value
                        )
            except Exception as error:  # noqa: BLE001 -- audit external metadata errors
                message = f"{type(error).__name__}: {error}"
                record["metadata"][channel] = {"error": message}
                if any(word in message.lower() for word in AUTH_WORDS):
                    record["auth_stopped"] = True
            if record["auth_stopped"]:
                break
        MdsTreeRegistry().close_all_trees()
    target = Path("outputs/labeler/elm/filterscope_metadata.json")
    target.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record["metadata"], indent=1))
    print("text nodes:", record.get("filterscope_text_nodes"))
    print("FS01 finite shots:", record["fs01_finite_shots"])
    print("auth stopped:", record["auth_stopped"])
    return 1 if record["auth_stopped"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
