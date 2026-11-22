"""Data roots of the IGNITE-on-MAST work.

Everything big (caches, checkpoints, results) lives under ``/models``, never in the
repository. Each root can be moved with an environment variable; the defaults are the
node13 layout. Commands call ``prepare_write_root`` before their first write.
"""

from __future__ import annotations

import os
from pathlib import Path

_TOKAMARK_DEFAULT = "/models/mast/tokamark/v1"
_ROOT_DEFAULT = "/models/intermediate/mast"
_TOKAMIND_RUNS_DEFAULT = "/models/tokamind/repo/runs"

# The checkout that holds this file: .../<repo>/src/tokamak_foundation_model/ignite/mast/paths.py
_REPO = Path(__file__).resolve().parents[4]

_TOKAMIND = Path("/models/tokamind").resolve()

_announced: set[Path] = set()


def _inside(path: Path, directory: Path) -> bool:
    """``path`` is ``directory`` or lies under it (both resolved)."""
    return path == directory or directory in path.parents


def tokamark_zarr_root() -> Path:
    """TokaMark's MAST data (Zarr v2), read-only."""
    return Path(os.environ.get("IGNITE_MAST_TOKAMARK", _TOKAMARK_DEFAULT))


def mast_root() -> Path:
    """Root of everything this stage writes."""
    return Path(os.environ.get("IGNITE_MAST_ROOT", _ROOT_DEFAULT))


def cache_dir(version: str = "v1") -> Path:
    return mast_root() / "cache" / version


def stage0_dir() -> Path:
    return mast_root() / "stage0"


def tokamind_runs_dir() -> Path:
    """TokaMind's replication runs, read-only."""
    return Path(os.environ.get("IGNITE_MAST_TOKAMIND_RUNS", _TOKAMIND_RUNS_DEFAULT))


def prepare_write_root(path: Path) -> Path:
    """Check that ``path`` may be written, announce it, create it and return it.

    Raises ``ValueError`` if ``path`` resolves inside the git checkout that holds this
    module, inside ``/models/tokamind`` (the TokaMind replication queue runs there) or
    inside TokaMark's data (`tokamark_zarr_root`, read-only). The check is on the resolved
    path, so a link into one of those trees does not get round it. The root and where it
    came from are printed once per process for each distinct resolved path; a path that is
    not under `mast_root()` is announced as such, because neither the default nor
    ``IGNITE_MAST_ROOT`` named it.
    """
    resolved = Path(path).resolve()
    if _inside(resolved, _REPO):
        raise ValueError(
            f"refusing to write inside the repository: {resolved} is inside the "
            f"repository {_REPO}; data goes under {_ROOT_DEFAULT} "
            "(or the root named by IGNITE_MAST_ROOT)"
        )
    for protected, what in ((_TOKAMIND, "the TokaMind tree (/models/tokamind)"),
                            (tokamark_zarr_root().resolve(), "TokaMark's data (read-only)")):
        if _inside(resolved, protected):
            raise ValueError(
                f"refusing to write inside {what}: {resolved} is inside {protected}; "
                f"data goes under {_ROOT_DEFAULT} (or the root named by IGNITE_MAST_ROOT)"
            )
    if resolved not in _announced:
        _announced.add(resolved)
        if not _inside(resolved, mast_root().resolve()):
            source = f"not under the stage root {mast_root()}"
        else:
            source = "IGNITE_MAST_ROOT set" if "IGNITE_MAST_ROOT" in os.environ else "default"
        print(f"[ignite.mast] writing under {resolved} ({source})")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
