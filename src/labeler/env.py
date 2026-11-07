"""Resolve environment settings for Python readers and shell launchers.

Three data roots differ from one cluster to the next, so each cluster has a row
in `CLUSTERS`: `LABELER_ROOT`, `SHOT_DESIGN_DATA_ROOT` and `LABELER_LABEL_TABLES`.
A variable the caller exports always wins; the cluster's row is what the variable
is when nothing exports it. Stdlib only: shell launchers run this file with
whichever interpreter they have.

The label tables can also be one folder both clusters share: `SHARED` names it, on
`/projects`. Where that folder is mounted and switched on (it holds `ACTIVE`, which
`labeler-sync activate` writes), it is the label tables. Anywhere else (a compute
node, which does not mount `/projects`, or any node before the switch) the
cluster's row names the cluster's own folder, which after the switch is a read-only
mirror that `labeler-sync mirror` refreshes from a login node
(docs/labeler/shared_folder.md).
"""

from __future__ import annotations

import os
import socket
import sys
from functools import cache

#: One row per cluster: `hosts` are hostname prefixes (the longest match wins, so
#: `stellarai` is tried before `stellar`), `env` is what each variable is on that
#: cluster. Add a cluster here and nowhere else (docs/clusters/adding-a-cluster.md).
#: `LABELER_LABEL_TABLES` here is the cluster's own folder: the label tables until
#: the shared folder is switched on, its mirror after.
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

#: Variables whose folder every cluster shares where it is mounted and switched on.
SHARED = {"LABELER_LABEL_TABLES": "/projects/EKOLEMEN/tokamaksi/diiid/events"}
#: The file in a shared folder that switches it on (`labeler-sync activate`).
ACTIVE = ".shared"


#: Values of `LABELER_CLUSTER` already reported as no cluster, so each is noted once per process.
_NOTED: set[str] = set()


def getenv(name: str, default: str | None = None) -> str | None:
    """Return the environment value of `name`, or `default` when it is unset.

    An empty value counts as set. Shell launchers call this module as a script.
    """
    return os.environ.get(name, default)


def _note(text: str) -> None:
    """Say `text` on stderr, once per process."""
    if text not in _NOTED:
        _NOTED.add(text)
        print(f"labeler.env: {text}", file=sys.stderr)


def cluster() -> str | None:
    """The cluster this process runs on, a key of `CLUSTERS`, or None.

    `LABELER_CLUSTER` decides when it is set; an empty or unknown value means no
    cluster, which is how a test pins the defaults (a non-empty unknown value, most
    likely a typo, is noted once on stderr). Otherwise the hostname's prefix
    (longest first), then Slurm's `SLURM_CLUSTER_NAME` when it is a key.
    """
    named = os.environ.get("LABELER_CLUSTER")
    if named is not None:
        if named and named not in CLUSTERS:
            _note(
                f"LABELER_CLUSTER={named!r} is not one of {sorted(CLUSTERS)}: "
                "no cluster, so the built-in defaults apply"
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


def local(name: str) -> str | None:
    """The cluster's own value of `name` (for a shared folder, its mirror), or None."""
    return CLUSTERS.get(cluster() or "", {}).get("env", {}).get(name)


@cache
def switched_on(folder: str) -> bool:
    """Whether `folder` is mounted here and holds `ACTIVE`. Asked once per process,
    so a long run does not change folders when the switch happens under it."""
    return os.path.isfile(os.path.join(folder, ACTIVE))


def shared(name: str) -> str | None:
    """The shared folder `name` resolves to here, or None: `name` is not shared, this
    is no cluster, or the folder is not mounted here or not switched on."""
    folder = SHARED.get(name)
    if folder is None or cluster() is None:
        return None
    return folder if switched_on(folder) else None


def default(name: str) -> str | None:
    """What `name` is here when nothing exports it, or None: the shared folder where
    it is mounted and switched on, else the cluster's row."""
    return shared(name) or local(name)


def resolve(name: str, fallback: str | None = None) -> str | None:
    """The environment value of `name`, else this node's default, else `fallback`.

    One exception to "an export wins": an exported shared folder this node does not
    mount (a job that inherited a login node's environment) means the cluster's
    mirror, noted once on stderr."""
    value = getenv(name)
    if value is None:
        return default(name) or fallback
    folder = SHARED.get(name)
    if folder and value.rstrip("/") == folder and not os.path.isdir(folder):
        mirror = local(name)
        if mirror:
            _note(f"{name}={value} is not mounted on this node: reading the mirror {mirror}")
            return mirror
    return value


if __name__ == "__main__":
    if sys.argv[1] == "--cluster":
        print(cluster() or "")
    else:
        print(resolve(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""))
