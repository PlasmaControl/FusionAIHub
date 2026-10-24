"""Resolve environment settings for Python readers and shell launchers.

Three data roots differ from one cluster to the next, so each cluster has a row
in `CLUSTERS`: `LABELER_ROOT`, `SHOT_DESIGN_DATA_ROOT` and `LABELER_LABEL_TABLES`.
A variable the caller exports always wins; the cluster's row is what the variable
is when nothing exports it. Stdlib only: shell launchers run this file with
whichever interpreter they have.
"""

from __future__ import annotations

import os
import socket
import sys

#: One row per cluster: `hosts` are hostname prefixes (the longest match wins, so
#: `stellarai` is tried before `stellar`), `env` is what each variable is on that
#: cluster. Add a cluster here and nowhere else (docs/clusters/adding-a-cluster.md).
#: Each cluster keeps its own label tables for now; a folder that both mount and
#: share, later, is a change to these rows only.
CLUSTERS = {
    "stellar": {
        "hosts": ("stellar",),
        "env": {
            "LABELER_ROOT": "/scratch/gpfs/EKOLEMEN/nc1514/labeler",
            "SHOT_DESIGN_DATA_ROOT": "/scratch/gpfs/EKOLEMEN/nc1514/shot_design",
            "LABELER_LABEL_TABLES": "/scratch/gpfs/EKOLEMEN/nc1514/events",
        },
    },
    "stellarai": {
        "hosts": ("stellarai", "sai-"),
        "env": {
            "LABELER_ROOT": "/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler",
            "SHOT_DESIGN_DATA_ROOT": (
                "/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/shot_design"
            ),
            "LABELER_LABEL_TABLES": "/scratch/gpfs/EKOLEMEN/D3D/events",
        },
    },
}


#: Values of `LABELER_CLUSTER` already reported as no cluster, so each is noted once per process.
_NOTED: set[str] = set()


def getenv(name: str, default: str | None = None) -> str | None:
    """Return the environment value of `name`, or `default` when it is unset.

    An empty value counts as set. Shell launchers call this module as a script.
    """
    return os.environ.get(name, default)


def cluster() -> str | None:
    """The cluster this process runs on, a key of `CLUSTERS`, or None.

    `LABELER_CLUSTER` decides when it is set; an empty or unknown value means no
    cluster, which is how a test pins the defaults (a non-empty unknown value, most
    likely a typo, is noted once on stderr). Otherwise the hostname's prefix
    (longest first), then Slurm's `SLURM_CLUSTER_NAME` when it is a key.
    """
    named = os.environ.get("LABELER_CLUSTER")
    if named is not None:
        if named and named not in CLUSTERS and named not in _NOTED:
            _NOTED.add(named)
            print(
                f"labeler.env: LABELER_CLUSTER={named!r} is not one of {sorted(CLUSTERS)}: "
                "no cluster, so the built-in defaults apply",
                file=sys.stderr,
            )
        return named if named in CLUSTERS else None
    host = socket.gethostname().lower()
    prefixes = sorted(
        ((prefix, name) for name, row in CLUSTERS.items() for prefix in row["hosts"]),
        key=lambda item: -len(item[0]),
    )
    for prefix, name in prefixes:
        if host.startswith(prefix):
            return name
    slurm = os.environ.get("SLURM_CLUSTER_NAME", "")
    return slurm if slurm in CLUSTERS else None


def default(name: str) -> str | None:
    """What `name` is on this cluster when nothing exports it, or None."""
    return CLUSTERS.get(cluster() or "", {}).get("env", {}).get(name)


def resolve(name: str, fallback: str | None = None) -> str | None:
    """The environment value of `name`, else this cluster's, else `fallback`."""
    return getenv(name, default(name) or fallback)


if __name__ == "__main__":
    if sys.argv[1] == "--cluster":
        print(cluster() or "")
    else:
        print(resolve(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""))
