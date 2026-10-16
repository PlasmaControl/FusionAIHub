"""Stage a shot's FDP objects locally, then let MDSplus and PTDATA read them.

Why stage at all, when the FDP stack can read over the network?  Because a
federated read costs a round trip and a config is dense.  Measured from
Stellar against the Princeton cache, reading the same signal both ways:

=========================================  ==========  ==============
signal                                     streamed    staged + local
=========================================  ==========  ==============
``\\D3D::TOP.ELECTRONS.TS...CORE:TEMP``      49.6 s      0.003 s
``\\D3D::TOP.ELECTRONS.TS...CORE:DENSITY``   48.1 s      0.004 s
=========================================  ==========  ==============

Each streamed read re-traverses the tree structure over the network, and
having the tree already open does not help.  Staging pays one bulk transfer
per file (13-80 MB/s) and then reads at memory speed.  For ~1900 signals over
~20 objects that is the difference between hours and minutes.

This is the strategy ``ptdata.fsspec_provider.FsspecReader`` documents --
*"download-and-cache ... for situations where streaming small-range reads is
impractical (e.g. cold caches, slow-start storage)"* -- but its transfer goes
through ``fsspec``, whose http backend buffers a whole object in memory
(measured: 1.68 GB resident for a 502 MB file, nothing written after 6
minutes).  So the strategy is kept and the transfer is done with
``pelican object get`` instead, reading the same objects from the same cache.

Nothing is staged to Stellar's disks: the default destination is a
subdirectory of ``/dev/shm``, which is RAM, and it is removed after the shot.
Peak use is ~4 GB per concurrent shot.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Optional

#: MDSplus digit-substitution letters, mapped to their position from the right
#: of the shot number (``~c`` is the third digit from the right).  These appear
#: in the ``default_tree_path`` templates that ``fdp run`` exports.
_DIGIT_TOKENS = {chr(ord("c") + i): 3 + i for i in range(20)}


#: Pointname references inside a TDI record.  The first argument of any of the
#: PTDATA/PTHEAD family is a pointname, given either as a literal
#: (``PTDATA2("gaionoffa", $SHOT)``) or indirectly through a tree node, which
#: may be absolute (``PTDATA2(\ECE::ECPTNAMF[0], $SHOT, 4)``) or relative
#: (``PTHEAD_IFIX(.NB.NB21L:HEADPOINT, .NB:SHOT)``).  Both the modern ``*2``
#: functions and the older ones matter: NB uses ``PTHEAD_IFIX`` while ECE uses
#: ``PTDATA2``.
POINTNAME_CALL = re.compile(
    r'PT(?:DATA|NPTS|HEAD|CHAR)2?(?:_[A-Z0-9]+)?\s*\(\s*'
    r'(?:"([^"]*)"|((?:\\|\.)[A-Za-z0-9_:.]+))',
    re.IGNORECASE,
)


#: Same as :data:`POINTNAME_CALL`, but also captures the *shot* argument, which
#: decides whether this shot's index can rule on the pointname.  A literal
#: ``$SHOT`` means yes; a node reference means the call reads a different shot
#: (``PTDATA("15RS_I_ACC", .NB.NB15R:NBSHOT, _error)`` reads the beam's own shot)
#: and this shot's index says nothing about it.
LITERAL_POINTNAME_CALL = re.compile(
    r'PT(?:DATA|NPTS|HEAD|CHAR)2?(?:_[A-Z0-9]+)?\s*\(\s*'
    r'"([^"]*)"\s*,\s*([^,)]+)',
    re.IGNORECASE,
)

#: Shot arguments that mean "the shot being fetched".
CURRENT_SHOT_ARGS = frozenset({"$SHOT", "_SHOT"})


class StageError(RuntimeError):
    """Raised when an object could not be staged."""


def pelican_binary() -> str:
    """Return the ``pelican`` executable that ships with the FDP environment."""
    import sys

    candidate = Path(sys.executable).parent / "pelican"
    if candidate.is_file():
        return str(candidate)
    from shutil import which

    found = which("pelican")
    if not found:
        raise StageError(
            "Could not find the `pelican` client. It ships with the FDP pixi "
            "environment; run under `pixi run fdp run ...`."
        )
    return found


# --------------------------------------------------------------------------
# URL construction
# --------------------------------------------------------------------------
def tree_templates() -> list[str]:
    """Return the ``pelican://`` tree-path templates from the environment.

    ``fdp run`` exports ``default_tree_path`` as a ``;``-separated list of
    templates covering codes, usershots, models and shots.  Reading them from
    the environment rather than hard-coding paths means the layout stays the
    federation's business, not ours.

    Returns
    -------
    list of str
        Templates, in the order MDSplus would search them.

    Raises
    ------
    StageError
        If ``default_tree_path`` is unset, which means this is not running
        under ``fdp run``.
    """
    raw = os.environ.get("default_tree_path")
    if not raw:
        raise StageError(
            "default_tree_path is not set: run under `fdp run` so the "
            "federated tree locations are available."
        )
    delimiter = os.environ.get("SYS_D3_DELIM", ";")
    return [t for t in raw.split(delimiter) if t.strip()]


def expand_template(template: str, tree: str, shot: int) -> str:
    """Substitute MDSplus ``~`` tokens in a tree-path template.

    ``~t`` becomes the tree name; the letter tokens become shot digits counted
    from the right, so ``~f~e`` is digits six and five.  Verified against the
    archive for shots 133221, 165000, 195000 and 200000 under both the
    ``shots`` and ``codes`` layouts.

    Parameters
    ----------
    template : str
        One entry from :func:`tree_templates`.
    tree : str
        Tree name.
    shot : int
        Shot number.

    Returns
    -------
    str
        Directory URL for that tree and shot.
    """
    digits = str(shot)
    out = template.replace("~t", tree.lower())
    for token, position in _DIGIT_TOKENS.items():
        if f"~{token}" not in out:
            continue
        digit = digits[-position] if position <= len(digits) else "0"
        out = out.replace(f"~{token}", digit)
    return out


def mds_object_urls(tree: str, shot: int,
                    templates: Optional[list[str]] = None) -> list[str]:
    """Return candidate URLs for one tree's three files, across all layouts.

    .. warning::
       Do not stage these URLs into one directory.  **More than one layout can
       hold the same tree** -- the assumption that only one does was wrong, and
       since every layout spells the three files identically, a flat concurrent
       transfer lets structure and data land from different layouts.  Measured
       on a two-layout fixture: 2 distinct outcomes from 12 identical runs,
       one of them a mixed set.  Use :func:`stage_trees`, which picks one
       layout per tree.  This function remains for enumerating candidates.

    Parameters
    ----------
    tree : str
        Tree name.
    shot : int
        Shot number.
    templates : list of str, optional
        Defaults to :func:`tree_templates`.

    Returns
    -------
    list of str
        Object URLs to attempt.
    """
    templates = templates if templates is not None else tree_templates()
    urls = []
    for template in templates:
        directory = expand_template(template, tree, shot)
        for extension in ("tree", "characteristics", "datafile"):
            urls.append(f"{directory}/{tree.lower()}_{shot}.{extension}")
    return urls


# --------------------------------------------------------------------------
# PTDATA index
# --------------------------------------------------------------------------
def fetch_shot_index(shot: int, dest_dir: Path,
                     snapshot: Optional[str] = None) -> dict:
    """Download and parse the PTDATA index for one shot.

    The index is the federation's own answer to "which object holds this
    pointname": one JSON per shot, holding ``pointname_ext`` (pointname to
    extension) and ``ext_location`` (extension to object URL).  For shot
    200000 it lists 18007 pointnames across 102 objects and weighs ~500 kB, so
    one read replaces any local mapping we might otherwise maintain.

    Parameters
    ----------
    shot : int
        Shot number.
    dest_dir : Path
        Directory to download the JSON into.
    snapshot : str, optional
        Index snapshot directory name.  Defaults to the newest one, which
        sorts last because the names are dated.

    Returns
    -------
    dict
        The parsed index.

    Raises
    ------
    StageError
        If the index cannot be located or downloaded.
    """
    root = os.environ.get("PTDATA_JSON_INDEX_DIR")
    if not root:
        raise StageError(
            "PTDATA_JSON_INDEX_DIR is not set: run under `fdp run`.")

    if snapshot is None:
        snapshot = _newest_snapshot(root)

    url = f"{root}/{snapshot}/{str(shot)[:4]}/{shot}.json"
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / f"{shot}.json"

    # The index is a single critical object, so a transient failure is worth
    # retrying rather than losing the shot to it.
    failures = []
    for _ in range(3):
        failures = stage_objects([url], dest_dir)
        if target.is_file():
            break
    if not target.is_file():
        detail = failures[0][1] if failures else "no error reported"
        raise StageError(
            f"Could not stage the PTDATA index for {shot}: {url} ({detail})")

    with open(target) as handle:
        return json.load(handle)


def _newest_snapshot(root: str) -> str:
    """Return the newest index snapshot directory name under ``root``."""
    pattern = os.environ.get("PTDATA_JSON_INDEX_PATTERN", "json_indexes_*")
    prefix = pattern.rstrip("*")
    result = subprocess.run(
        [pelican_binary(), "object", "ls", root],
        capture_output=True, text=True, timeout=300,
    )
    names = [
        line.strip().rstrip("/").split("/")[-1]
        for line in result.stdout.splitlines()
        if prefix in line
    ]
    if not names:
        raise StageError(
            f"No index snapshots matching {pattern!r} under {root} "
            f"(pelican ls said: {result.stderr.strip()[:200]})"
        )
    return sorted(names)[-1]


def ptdata_objects(index: dict, pointnames: Iterable[str]) -> tuple:
    """Resolve pointnames to the objects that hold them.

    Parameters
    ----------
    index : dict
        As returned by :func:`fetch_shot_index`.
    pointnames : iterable of str
        Pointnames the config asks for.

    Returns
    -------
    urls : list of str
        Object URLs to stage, de-duplicated.
    sources : dict
        Pointname to extension (e.g. ``".MAG"``), for the reader.
    missing : list of str
        Pointnames the index has no entry for, i.e. absent for this shot.
    """
    by_name = {k.lower(): v for k, v in (index.get("pointname_ext") or {}).items()}
    locations = index.get("ext_location") or {}

    sources: dict[str, str] = {}
    missing: list[str] = []
    for pointname in pointnames:
        extension = by_name.get(pointname.lower())
        if extension is None:
            missing.append(pointname)
        else:
            sources[pointname] = extension

    urls = []
    for extension in dict.fromkeys(sources.values()):
        url = locations.get(extension)
        if url and url not in urls:
            urls.append(url)
    return urls, sources, missing


# --------------------------------------------------------------------------
# Transfer
# --------------------------------------------------------------------------
def stage_objects(
    urls: Iterable[str],
    dest_dir: Path,
    workers: int = 6,
    timeout: float = 1800.0,
    skip_existing: bool = True,
) -> list[tuple[str, str]]:
    """Copy objects into a local directory with ``pelican object get``.

    Parameters
    ----------
    urls : iterable of str
        ``pelican://`` object URLs.
    dest_dir : Path
        Local destination; created if absent.  Objects keep their basenames,
        which is what MDSplus and PTDATA expect.
    workers : int, default 6
        Concurrent transfers.
    timeout : float, default 1800
        Per-object timeout in seconds.
    skip_existing : bool, default True
        Skip objects already present with non-zero size.

    Returns
    -------
    list of (str, str)
        ``(url, message)`` for each object that could not be staged.  Missing
        objects are normal: a tree is tried under several layouts and a shot
        legitimately lacks many diagnostics.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    binary = pelican_binary()

    def fetch(url: str) -> Optional[tuple[str, str]]:
        target = dest_dir / url.rsplit("/", 1)[-1]
        if skip_existing and target.is_file() and target.stat().st_size > 0:
            return None
        try:
            result = subprocess.run(
                [binary, "object", "get", url, str(dest_dir) + "/"],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return url, f"timeout after {timeout:.0f}s"
        if result.returncode != 0 or not target.is_file():
            message = (result.stderr or result.stdout or "").strip().splitlines()
            return url, message[-1][:200] if message else f"rc={result.returncode}"
        return None

    urls = list(urls)
    if not urls:
        return []

    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for outcome in pool.map(fetch, urls):
            if outcome is not None:
                failures.append(outcome)
    return failures


def _fetch_to_targets(pairs, workers: int = 6,
                      timeout: float = 600.0) -> None:
    """Fetch ``(url, destination directory)`` pairs in parallel, ignoring misses.

    Unlike :func:`stage_objects` every pair names its own directory, which is
    what lets the same basename be fetched from several layouts without the
    copies overwriting one another.  A miss is normal here -- most layouts do
    not hold most trees -- so nothing is reported.
    """
    binary = pelican_binary()

    def fetch(pair) -> None:
        url, target = pair
        try:
            subprocess.run([binary, "object", "get", url, str(target) + "/"],
                           capture_output=True, text=True, timeout=timeout)
        except Exception:  # noqa: BLE001
            return

    pairs = list(pairs)
    if not pairs:
        return
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(fetch, pairs))


def stage_trees(shot: int, trees: Iterable[str], dest_dir: Path,
                workers: int = 6,
                templates: Optional[list[str]] = None) -> dict:
    """Stage each tree's files, taking all of them from a single layout.

    A tree is three files -- ``.tree`` (structure), ``.characteristics`` and
    ``.datafile`` (data) -- and more than one archive layout can hold a given
    tree for a given shot.  Every layout yields the *same three basenames*, so
    fetching all layouts' URLs concurrently into one directory lets the three
    files race independently and land from **different layouts**: a structure
    file from one, a data file from another.  MDSplus then opens the tree
    happily and reports ``TreeNODATA`` for nodes whose data lives in the
    datafile that did not win -- which is exactly the ECE symptom on shot
    146918, where ``\\ECE::ECPTNAMF`` read empty locally while the federation
    served it intact.  Being a thread race it is also nondeterministic, which
    is why the same shot behaved differently across runs.

    So layout choice is made once per tree and the whole set comes from it.
    The two small files are fetched from every candidate layout first (a
    ``.tree`` is kilobytes), the first layout in template order that yields
    both is selected, and only then is that layout's ``.datafile`` -- which can
    be gigabytes -- transferred.  Picking by template order rather than by
    arrival makes the result reproducible.

    Parameters
    ----------
    shot : int
        Shot number.
    trees : iterable of str
        Tree names.
    dest_dir : Path
        Staging directory.
    workers : int, default 6
        Concurrent transfers.
    templates : list of str, optional
        Defaults to :func:`tree_templates`.

    Returns
    -------
    dict
        ``tree -> layout directory`` for each tree that staged, so the choice
        is visible rather than implicit.
    """
    templates = templates if templates is not None else tree_templates()
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    probe_root = dest_dir / ".layout_probe"
    # Materialised and lower-cased: the loops below walk `trees` more than once,
    # and expand_template lower-cases the name, so the basenames must match.
    trees = [t.lower() for t in dict.fromkeys(trees)]

    # Phase 1: the two small files from every candidate layout, in parallel.
    probe_jobs: list[tuple[str, int, str, Path]] = []
    for tree in trees:
        for number, template in enumerate(templates):
            directory = expand_template(template, tree, shot)
            target = probe_root / tree / str(number)
            for extension in ("tree", "characteristics"):
                probe_jobs.append(
                    (tree, number,
                     f"{directory}/{tree}_{shot}.{extension}", target))

    for _, _, _, target in probe_jobs:
        target.mkdir(parents=True, exist_ok=True)
    _fetch_to_targets([(url, target) for _, _, url, target in probe_jobs],
                      workers=workers)

    # Phase 2: rank each tree's layouts by template order -- MDSplus's own
    # search order -- so the choice is reproducible rather than whichever
    # transfer happened to finish first.
    candidates: dict[str, list[int]] = {}
    for tree in trees:
        for number in range(len(templates)):
            structure = probe_root / tree / str(number) / f"{tree}_{shot}.tree"
            if structure.is_file() and structure.stat().st_size > 0:
                candidates.setdefault(tree, []).append(number)

    # Phase 3: take the first candidate layout whose datafile also lands, and
    # take the whole set from it.  A layout holding a structure file but no
    # usable datafile is precisely the state that yields TreeNODATA, so it is
    # skipped rather than accepted.
    chosen: dict[str, str] = {}
    for tree, numbers in candidates.items():
        for number in numbers:
            directory = expand_template(templates[number], tree, shot)
            failures = stage_objects(
                [f"{directory}/{tree}_{shot}.datafile"], dest_dir,
                workers=1, skip_existing=False)
            datafile = dest_dir / f"{tree}_{shot}.datafile"
            if failures or not (datafile.is_file()
                                and datafile.stat().st_size > 0):
                datafile.unlink(missing_ok=True)
                continue
            folder = probe_root / tree / str(number)
            for item in folder.glob(f"{tree}_{shot}.*"):
                shutil.move(str(item), str(dest_dir / item.name))
            chosen[tree] = directory
            break

    shutil.rmtree(probe_root, ignore_errors=True)
    return chosen


def stage_shot(
    shot: int,
    trees: Iterable[str],
    pointnames: Iterable[str],
    dest_dir: Path,
    workers: int = 6,
) -> dict:
    """Stage everything one shot needs, and report what landed.

    Parameters
    ----------
    shot : int
        Shot number.
    trees : iterable of str
        MDSplus trees to stage.
    pointnames : iterable of str
        PTDATA pointnames the config asks for.
    dest_dir : Path
        Staging directory (RAM-backed by default; see module docstring).
    workers : int, default 6
        Concurrent transfers.

    Returns
    -------
    dict
        ``index`` (the parsed shot index, for resolving further pointnames),
        ``available_trees``, ``ptdata_sources``, ``missing_pointnames``,
        ``bytes``, ``seconds``.
    """
    dest_dir = Path(dest_dir)
    started = time.time()
    trees = [t.lower() for t in dict.fromkeys(trees)]

    index = fetch_shot_index(shot, dest_dir)
    urls, sources, missing = ptdata_objects(index, pointnames)

    templates = tree_templates()
    # PTDATA objects have unique basenames so they can all go at once; trees
    # cannot, because every layout spells them the same -- see stage_trees.
    stage_objects(urls, dest_dir, workers=workers)
    layouts = stage_trees(shot, trees, dest_dir, workers=workers,
                          templates=templates)

    available = {t for t in trees if (dest_dir / f"{t}_{shot}.tree").is_file()}
    staged_bytes = sum(p.stat().st_size for p in dest_dir.glob("*") if p.is_file())

    return {
        "index": index,
        "available_trees": available,
        "ptdata_sources": sources,
        "missing_pointnames": missing,
        "bytes": staged_bytes,
        "seconds": time.time() - started,
        "tree_layouts": layouts,
    }


def referenced_pointnames(shot: int, trees: dict, signal_tree) -> list[str]:
    """Collect pointnames the config's MDSplus nodes read internally.

    Most of a DIII-D config is not raw data but TDI expressions over PTDATA, so
    the pointnames that need staging are mostly *not* the ones in the config's
    ``ptdata`` section: the ECE nodes read ``ecevs*``, GASFLOW timebases read
    ``gaionoff*``, and the NB beam programs read ``BMC*`` and header pointnames
    like ``NB21LEFT``.  Missing one means that signal fails once PTDATA is
    pointed at staged files, so this scan is what keeps coverage intact.

    Trees must already be staged locally -- reading ~1900 records over the
    network would cost more than it saves.

    Parameters
    ----------
    shot : int
        Shot number.
    trees : dict
        Config tree-key to signal-list mapping.
    signal_tree : callable
        ``(expression, default) -> tree name``; pass
        ``mds_reader.signal_tree``.

    Returns
    -------
    list of str
        Referenced pointnames, de-duplicated.
    """
    import MDSplus as mds
    import numpy as np

    found: list[str] = []
    open_trees: dict[str, object] = {}

    def remember(name: str) -> None:
        if name and name not in found:
            found.append(name)

    for key, signals in trees.items():
        if key.lower() == "ptdata":
            continue
        for signal in signals:
            tree_name = signal_tree(signal, key)
            try:
                if tree_name not in open_trees:
                    open_trees[tree_name] = mds.Tree(tree_name, shot)
                record = str(open_trees[tree_name].getNode(signal).getRecord())
            except Exception:  # noqa: BLE001
                continue

            for literal, indirect in POINTNAME_CALL.findall(record):
                if literal:
                    remember(literal)
                    continue
                try:
                    values = open_trees[tree_name].getNode(indirect).data()
                except Exception:  # noqa: BLE001
                    continue
                for value in np.atleast_1d(np.asarray(values)).ravel():
                    text = value.decode() if isinstance(value, bytes) else str(value)
                    remember(text.strip())

    for tree in open_trees.values():
        try:
            tree.close()
        except Exception:  # noqa: BLE001
            pass

    return found


def split_unreadable_nodes(shot: int, entries, index: dict,
                           signal_tree) -> tuple[list, list]:
    """Split deferred signals into worth-retrying and provably-unreadable.

    A signal deferred by the MDSplus pass is an expression node that failed
    because that pass runs without the federation index.  Most of them would
    fail again *with* the index, because the pointname they read does not exist
    for the shot -- and the shot index can say so without touching the network.

    The test is only valid when the record names a literal pointname **and**
    reads it for this shot.  Measured on shot 115401, the 160 deferred signals
    split three ways:

    ==========================================  =====  =========
    category                                    count  recovered
    ==========================================  =====  =========
    literal pointname, ``$SHOT``, not in index    120          0
    no PTDATA call (reads sibling nodes)           24          6
    shot argument is a node, not ``$SHOT``         16          6
    ==========================================  =====  =========

    Only the first category is dropped, so nothing recoverable is lost: the
    retry resolves such a node through the very index lookup this performs, and
    all 120 were observed to fail when actually retried.  Note this is *not* a
    claim that the index is authoritative about absence in general -- it is not,
    ``MPI79FA322D`` is absent from it yet reads fine through ``PtDataSignal`` --
    only that it decides this particular resolution path.

    Trees must already be staged; this reads records locally.

    Parameters
    ----------
    shot : int
        Shot number.
    entries : iterable of (str, str)
        ``(config_key, signal)`` pairs, as collected for a retry.
    index : dict
        The shot index, for its ``pointname_ext`` keys.
    signal_tree : callable
        ``(expression, default) -> tree name``; pass ``mds_reader.signal_tree``.

    Returns
    -------
    keep : list of (str, str)
        Entries that could still succeed.
    unreadable : list of str
        Signal paths dropped, for the caller to log.  Never discard silently.
    """
    import MDSplus as mds

    known = {name.upper() for name in (index.get("pointname_ext") or {})}
    if not known:
        # No index means no grounds to drop anything.
        return list(entries), []

    keep: list = []
    unreadable: list[str] = []
    open_trees: dict[str, object] = {}

    for key, signal in entries:
        tree_name = signal_tree(signal, key)
        try:
            if tree_name not in open_trees:
                open_trees[tree_name] = mds.Tree(tree_name, shot)
            record = str(open_trees[tree_name].getNode(signal).getRecord())
        except Exception:  # noqa: BLE001
            # Cannot read the record, so cannot rule it out.
            keep.append((key, signal))
            continue

        match = LITERAL_POINTNAME_CALL.search(record)
        if match is None:
            keep.append((key, signal))
            continue
        pointname, shot_arg = match.group(1).strip(), match.group(2).strip()
        if shot_arg.upper() not in CURRENT_SHOT_ARGS:
            keep.append((key, signal))
        elif pointname.upper() in known:
            keep.append((key, signal))
        else:
            unreadable.append(signal)

    for tree in open_trees.values():
        try:
            tree.close()
        except Exception:  # noqa: BLE001
            pass

    return keep, unreadable


def use_staged_ptdata(stage_dir: Path) -> dict:
    """Point PTDATA at staged shot files and report what changed.

    ``SYS_D3`` alone is not enough: with ``PTDATA_JSON_INDEX_DIR`` set, the
    ptdata 2.0 engine resolves through the federation index and reads over the
    network even when the objects sit locally (measured: ECE ``TECEF01`` 18.9 s
    with the index, 1.34 s without).  So the index is removed from the
    environment for the local pass and restored by :func:`use_indexed_ptdata`
    for the retry pass.

    Parameters
    ----------
    stage_dir : Path
        Directory holding the staged ``<shot>.<EXT>`` files.

    Returns
    -------
    dict
        The removed variables, to hand back to :func:`use_indexed_ptdata`.
    """
    saved = {k: os.environ[k] for k in ("PTDATA_JSON_INDEX_DIR",)
             if k in os.environ}
    os.environ.pop("PTDATA_JSON_INDEX_DIR", None)
    os.environ["SYS_D3"] = str(stage_dir)
    return saved


def use_indexed_ptdata(saved: dict) -> None:
    """Restore federated PTDATA routing after a staged pass.

    Parameters
    ----------
    saved : dict
        As returned by :func:`use_staged_ptdata`.
    """
    os.environ.update(saved)
    os.environ.pop("SYS_D3", None)


def load_cached_index(shot: int, stage_dir: Path) -> dict:
    """Load a shot index that is already staged.

    Parameters
    ----------
    shot : int
        Shot number.
    stage_dir : Path
        Directory holding ``<shot>.json``.

    Returns
    -------
    dict
        The parsed index.
    """
    with open(Path(stage_dir) / f"{shot}.json") as handle:
        return json.load(handle)
