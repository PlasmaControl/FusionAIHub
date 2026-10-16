"""Fetch one DIII-D shot from the FDP into an HDF5 file.

Must be run under ``fdp run``, which injects the environment that points
MDSplus and PTDATA at the federation.  Without it PTDATA falls back to
contacting a live ptserver, which Stellar cannot reach, and fails with
``getservbyname failed for task 'PTSERVER'``.  ``fetch_shot.sh`` does the
wrapping; run it by hand as::

    cd /scratch/gpfs/ps9551/fdp-demo
    pixi run fdp run python <path>/fetch_shot.py --shot 200000 \\
        --config <path>/config_fdp.yaml --output-dir <dir>

There is no staging step: reads are HTTP range requests against a Pelican
cache, so a signal costs roughly what it is worth rather than what the file
around it weighs.

Individual signals are allowed to fail.  A config asks for signals that exist
only in some eras, and for plenty of nodes that are empty or scalar; both are
counted separately from real failures, and neither aborts the shot.  The exit
status reflects whether the shot produced a usable file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

import fdp_stage
import mds_reader

#: Config key holding PTDATA pointnames rather than MDSplus paths.
PTDATA_KEY = "ptdata"

#: Variable that only ``fdp run`` sets; used to fail fast with a clear message.
#: Deliberately not ``PTDATA_JSON_INDEX_DIR``: the MDSplus pass unsets that on
#: purpose so PTDATA-backed TDI nodes read staged objects.
FDP_MARKER = "default_tree_path"

#: Error kinds worth a second, federated attempt.  These are what a PTDATA read
#: raises when the object holding a pointname was not staged.  Deliberately
#: excludes ``TreeNNF`` and ``TreeNODATA``, which mean the node genuinely does
#: not exist for the shot: retrying those over the network costs a round trip
#: each and recovers nothing (a full config produces hundreds of them).
#: ``DegenerateRead`` is ours, not MDSplus's: see :func:`_collapsed_expression`.
RETRYABLE = frozenset({"TdiABORT", "PtDataError", "NotInIndex", "TdiUNKNOWN_VAR",
                       "DegenerateRead"})


#: Per-worker state, built once by :func:`_init_worker` and reused across
#: signals so each process opens a tree at most once.
_WORKER: dict = {}


def _init_worker(stage_dir: str, trees: list, ptdata_sources: dict) -> None:
    """Prepare a worker process for local reads.

    Each worker configures its own MDSplus tree paths and its own PTDATA
    reader.  Nothing is shared with the parent beyond the staged files on disk,
    which is what makes the pool safe: MDSplus keeps per-process state and h5py
    cannot take concurrent writers, so workers only read and the parent writes.

    Parameters
    ----------
    stage_dir : str
        Directory holding the staged objects.
    trees : list of str
        Tree names to make resolvable.
    ptdata_sources : dict
        Pointname to shot-file extension, from the shot index.
    """
    from pathlib import Path as _Path

    mds_reader.configure_local_reads(_Path(stage_dir), trees)
    os.environ["SYS_D3"] = stage_dir
    _WORKER.clear()
    _WORKER["stage"] = stage_dir
    _WORKER["sources"] = ptdata_sources
    _WORKER["trees"] = {}
    _WORKER["ptdata"] = None
    _WORKER["shot"] = int(os.environ["FDP_FETCH_SHOT"])
    _WORKER["keep_scalars"] = os.environ.get("FDP_KEEP_SCALARS") == "1"

    # The shard is opened and closed around each write rather than held open:
    # a handle left open when a pool worker exits leaves the file unflushed, and
    # the parent then fails to read it ("bad object header version number").
    # Opening a RAM-backed file costs ~1 ms, which is noise next to a read.
    _WORKER["shard_path"] = str(_Path(stage_dir) / f"shard_{os.getpid()}.h5")


def _repair_ptdata(pointname: str, payload: dict, defect: str, shot: int) -> tuple:
    """Re-read a defective PTDATA payload through the TDI ``PTDATA2()`` function.

    Repairs are surgical, because the two readers each know something the other
    does not.  ``PtDataReader`` reports the real ``units`` where TDI reports
    blanks, so a bad timebase costs only its ``dim0``; an uncalibrated read is
    wrong in the values themselves, and its ``b"vo"`` units are a lie about the
    scaling, so the whole TDI payload replaces it.

    A failed or degenerate TDI read is discarded rather than trusted: with the
    federation index absent ``PTDATA2()`` returns shape ``(1,)``, which must
    never be written over a good staged read.

    Parameters
    ----------
    pointname : str
        PTDATA pointname.
    payload : dict
        The suspect payload from :func:`mds_reader.gather_ptdata_staged`.
    defect : str
        From :func:`mds_reader.ptdata_defect`.
    shot : int
        Shot number.

    Returns
    -------
    payload : dict
        Repaired if the TDI read was usable, else the original unchanged.
    repaired : str or None
        The defect that was fixed, for accounting; None if nothing changed.
    """
    try:
        fixed = mds_reader.gather_ptdata_tdi(pointname, shot)
    except Exception:  # noqa: BLE001
        return payload, None

    replacement = fixed.get("data")
    original = payload.get("data")
    if replacement is None or replacement.ndim != 1:
        return payload, None
    if original is not None and replacement.size < original.size:
        return payload, None

    if defect == "timebase":
        axis = fixed.get("dim0")
        if axis is None or (original is not None and axis.shape != original.shape):
            return payload, None
        return dict(payload, dim0=axis), defect

    return fixed, defect


def _collapsed_expression(tree, signal: str, payload: dict) -> bool:
    """Is this a PTDATA-backed node that collapsed rather than read?

    The MDSplus pass runs with ``PTDATA_JSON_INDEX_DIR`` unset, and for some
    shots that makes a node the expression depends on unreadable.  ECE is the
    case that motivated this: ``TECEF01`` is
    ``PTDATA2(\\ECE::ECPTNAMF[0], $SHOT, 4), Build_Signal(...)``, and with the
    index absent ``\\ECE::ECPTNAMF`` is ``TreeNODATA``, so the whole expression
    evaluates to a single zero.  **It raises nothing.**  Measured on shot 146918:
    ``shape=(1,)`` in that pass against ``shape=(4096000,)`` once the index is
    present, for all 40 channels.

    So a one-sample result from a node whose record calls the PTDATA family is
    treated as a failure and deferred to the pass that has the index.  Being
    wrong here is cheap: a node that really does hold one sample is simply read
    again later and written the same way.  Being wrong the other way writes a
    flat zero over a 4-million-sample diagnostic.

    Only consulted when the payload is already suspicious, so the extra record
    read costs nothing on the overwhelming majority of signals.

    Parameters
    ----------
    tree : MDSplus.Tree
        Open tree holding ``signal``.
    signal : str
        Node path.
    payload : dict
        From :func:`mds_reader.gather_mds`.

    Returns
    -------
    bool
        True if this looks collapsed and should be retried with the index.
    """
    data = payload.get("data")
    if data is None or data.ndim < 1 or data.size > 1:
        return False
    return _record_reads_ptdata(tree, signal)


def _record_reads_ptdata(tree, signal: str) -> bool:
    """Does this node's TDI record call the PTDATA family?

    Used to decide whether a failure is plausibly the index's or the staged
    trees' fault rather than genuine absence.  Reading the record is a local
    operation once the trees are staged.
    """
    if os.environ.get("PTDATA_JSON_INDEX_DIR"):
        # The index is already present, so a later pass has nowhere better to
        # look and deferring would just loop.
        return False
    if tree is None:
        return False
    try:
        record = str(tree.getNode(signal).getRecord())
    except Exception:  # noqa: BLE001
        return False
    return fdp_stage.POINTNAME_CALL.search(record) is not None


def _read_one(task: tuple) -> tuple:
    """Read one signal in a worker process.

    Parameters
    ----------
    task : tuple
        ``(config_key, signal)``.

    Returns
    -------
    tuple
        ``(config_key, signal, status, error_kind, note)``; exactly one of
        ``status`` and ``error_kind`` is None.  ``note`` carries a PTDATA
        repair or an unrepaired defect, and is None otherwise.
    """
    key, signal = task
    note = None
    try:
        if key.lower() == PTDATA_KEY:
            source = _WORKER["sources"].get(signal)
            if source is None:
                return key, signal, None, "NotInIndex", note
            if _WORKER["ptdata"] is None:
                from pathlib import Path as _Path
                _WORKER["ptdata"] = mds_reader.make_staged_ptdata_reader(
                    _Path(_WORKER["stage"]))
            payload = mds_reader.gather_ptdata_staged(
                _WORKER["ptdata"], signal, _WORKER["shot"], source)
            # The staged reader mis-reconstructs multi-rate timebases and
            # silently skips calibration on a few pointnames; TDI gets both
            # right but needs the index, so this only fires on the PTDATA pass.
            defect = mds_reader.ptdata_defect(payload)
            if defect:
                note = f"defect:{defect}"
                if os.environ.get("PTDATA_JSON_INDEX_DIR"):
                    payload, repaired = _repair_ptdata(
                        signal, payload, defect, _WORKER["shot"])
                    if repaired:
                        note = f"repaired:{repaired}"
        else:
            import MDSplus as mds

            tree_name = mds_reader.signal_tree(signal, key)
            if tree_name not in _WORKER["trees"]:
                _WORKER["trees"][tree_name] = mds.Tree(tree_name,
                                                       _WORKER["shot"])
            payload = mds_reader.gather_mds(_WORKER["trees"][tree_name], signal)
            # Not written if it collapsed: pass 3 writes with overwrite=False,
            # so a degenerate value left here would survive the retry.
            if _collapsed_expression(_WORKER["trees"][tree_name], signal,
                                     payload):
                return key, signal, None, "DegenerateRead", note

        data = payload.get("data")
        if not _WORKER["keep_scalars"] and (
                data is None or data.ndim == 0 or data.size == 0):
            return key, signal, "empty", None, note

        import h5py

        with h5py.File(_WORKER["shard_path"], "a") as shard:
            group = shard.require_group(mds_reader.tree_group_name(key))
            mds_reader.write_signal(
                group, mds_reader.signal_group_name(key, signal), payload)
        return key, signal, "written", None, note
    except Exception as error:  # noqa: BLE001
        kind = type(error).__name__
        # TreeNODATA is excluded from RETRYABLE because it usually means the node
        # genuinely holds nothing for the shot -- a full config raises hundreds.
        # But it is also what a PTDATA-backed expression raises when a node it
        # depends on is missing from the *staged* trees while the federation has
        # it: shot 146918's staged `ece` tree has no data in \ECE::ECPTNAMF, so
        # all 40 TECEF channels raise TreeNODATA here, yet read 4096000 samples
        # each once the retry unpins the tree paths. Reclassify only when the
        # record calls PTDATA, so the hundreds of genuine ones stay excluded.
        if kind in ("TreeNODATA", "TreeNNF") and key.lower() != PTDATA_KEY:
            tree_name = mds_reader.signal_tree(signal, key)
            if _record_reads_ptdata(_WORKER["trees"].get(tree_name), signal):
                kind = "DegenerateRead"
        return key, signal, None, kind, note


def require_fdp_environment() -> None:
    """Abort unless the process is running under ``fdp run``.

    Raises
    ------
    SystemExit
        If the federation environment is absent, with the fix in the message.
        Catching this here turns a confusing PTDATA network error deep in a
        1900-signal run into an immediate, actionable failure.
    """
    if os.environ.get(FDP_MARKER):
        return
    raise SystemExit(
        f"{FDP_MARKER} is not set, so this is not running under `fdp run`.\n"
        f"MDSplus would have no federated tree locations and PTDATA would fall "
        f"back to a live ptserver, which Stellar cannot reach. Re-run as:\n"
        f"    cd /scratch/gpfs/ps9551/fdp-demo && pixi run fdp run python "
        f"{Path(__file__).name} ...\n"
        f"and load the cache module (or export PELICAN_CLIENT_PREFERREDCACHES) "
        f"so reads are served from Princeton."
    )


def config_trees(config: dict, trees: dict) -> list[str]:
    """Return the MDSplus trees to stage for a config.

    A signal names its own tree via its ``TREE::`` prefix, but ``\\D3D::TOP.``
    paths read through subtree files whose names are not derivable from the
    path (``SPECTROSCOPY.FILTERSCOPE`` lives in ``fscope``), so those come from
    the config's ``prefetch_trees`` list.

    Parameters
    ----------
    config : dict
        Raw parsed config, for its ``prefetch_trees`` entry.
    trees : dict
        Tree-key to signal-list mapping.

    Returns
    -------
    list of str
        Lower-case tree names, de-duplicated.
    """
    needed: list[str] = []
    for key, signals in trees.items():
        if key.lower() == PTDATA_KEY:
            continue
        for candidate in [key] + [mds_reader.signal_tree(s, key) for s in signals]:
            name = candidate.lower()
            if name and name not in needed:
                needed.append(name)
    for tree in config.get("prefetch_trees") or []:
        if tree.lower() not in needed:
            needed.append(tree.lower())
    return needed


def load_config(paths: list[Path]) -> dict:
    """Load and merge one or more signal configs.

    Parameters
    ----------
    paths : list of Path
        YAML config files in the omega pipeline's format: a ``trees`` mapping
        of tree name to signal list, where the ``ptdata`` key holds bare
        pointnames.  ``server`` and ``prefetch_trees`` keys, if present, are
        ignored -- there is one federated source now, and nothing to prefetch.

    Returns
    -------
    trees : dict
        Mapping of tree key to list of signals, merged with duplicates removed.
    config : dict
        Merged non-signal settings, currently just ``prefetch_trees``.
    """
    trees: dict[str, list[str]] = {}
    merged: dict = {"prefetch_trees": []}
    for path in paths:
        with open(path) as handle:
            config = yaml.safe_load(handle) or {}
        for key, signals in (config.get("trees") or {}).items():
            existing = trees.setdefault(key, [])
            for signal in signals or []:
                if signal not in existing:
                    existing.append(signal)
        for tree in config.get("prefetch_trees") or []:
            if tree not in merged["prefetch_trees"]:
                merged["prefetch_trees"].append(tree)
    return trees, merged


def fetch_shot(
    shot: int,
    config_paths: list[Path],
    output_dir: Path,
    overwrite: bool = False,
    only_trees: Optional[list[str]] = None,
    skip_trees: Optional[list[str]] = None,
    keep_scalars: bool = False,
    progress_every: int = 200,
    stage_dir: Optional[Path] = None,
    stage_workers: int = 6,
    keep_stage: bool = False,
    no_retry: bool = False,
    reuse_stage: bool = False,
    scan_expressions: bool = True,
    stage_only: bool = False,
    workers: int = 8,
    write_retry_manifest: Optional[Path] = None,
    retry_manifest: Optional[Path] = None,
    defer_filter: bool = True,
) -> int:
    """Fetch one shot end to end.

    Parameters
    ----------
    shot : int
        Shot number.
    config_paths : list of Path
        Signal configs to merge.
    output_dir : Path
        Directory for ``<shot>.h5``.
    overwrite : bool, default False
        Rewrite signals already present in the output file.
    only_trees, skip_trees : list of str, optional
        Restrict or exclude config tree keys.
    keep_scalars : bool, default False
        Write nodes whose data is empty or 0-dimensional.  The old pipeline
        omitted them -- neither a reference nor a production file contains a
        single one -- so they are dropped by default.
    progress_every : int, default 200
        Emit a progress line every N signals; a full config takes a while and
        silence is hard to distinguish from a hang.
    stage_dir : Path, optional
        Parent directory for the per-shot staging area.  Defaults to a
        subdirectory of ``/dev/shm``, i.e. RAM: staged objects never touch
        Stellar's disks and are removed when the shot finishes.  Pass a
        filesystem path only if RAM is scarce.
    stage_workers : int, default 6
        Concurrent object transfers.
    keep_stage : bool, default False
        Leave the staged objects behind, for debugging.
    reuse_stage : bool, default False
        Assume ``stage_dir`` is already populated (the parent staged it) and do
        not transfer anything.
    stage_only : bool, default False
        Stage the shot's objects (including those the TDI expressions read) and
        return without reading anything.  This is the first of three passes:
        staging needs the federation index, the MDSplus pass needs it absent,
        and the PTDATA pass needs it back.  See ``fetch_shot_fast.sh``.
    workers : int, default 8
        Worker processes for the read phase.  Reads are independent, so wall
        time scales with this until the staged objects or the CPU saturate.
        Use 1 to read in-process.
    scan_expressions : bool, default True
        Scan node records for the pointnames TDI expressions read, and stage
        their objects too.  Only useful on the MDSplus pass.
    no_retry : bool, default False
        Skip the federated retry pass. Faster, but signals whose PTDATA object
        was not staged are then lost rather than merely slow.
    write_retry_manifest : Path, optional
        Write this pass's retryable failures here instead of retrying them, for
        a later pass to pick up with ``retry_manifest``.  The MDSplus pass
        needs this: it runs with the federation index unset, and a
        PTDATA-backed TDI expression node cannot be read at all in that
        environment, so retrying in place recovers nothing.
    retry_manifest : Path, optional
        Read signals deferred by an earlier pass and retry them here, in
        addition to this pass's own failures.  Missing file means nothing to do.
    defer_filter : bool, default True
        Drop deferred signals that :func:`fdp_stage.split_unreadable_nodes`
        proves cannot be read, which on shot 115401 is 120 of 160 and costs
        three quarters of the retry's wall time for nothing.  The dropped names
        are always written alongside the manifest.

    Returns
    -------
    int
        Number of signals written.
    """
    import shutil

    started = time.time()
    trees, config = load_config(config_paths)

    if only_trees:
        wanted = {t.lower() for t in only_trees}
        trees = {k: v for k, v in trees.items() if k.lower() in wanted}
    if skip_trees:
        unwanted = {t.lower() for t in skip_trees}
        trees = {k: v for k, v in trees.items() if k.lower() not in unwanted}
    if not trees:
        raise SystemExit("No trees left to fetch after --only-tree/--skip-tree")

    total = sum(len(v) for v in trees.values())
    print(f"shot {shot}: {total} signals across {len(trees)} config trees",
          flush=True)
    cache = os.environ.get("PELICAN_CLIENT_PREFERREDCACHES", "director-chosen")
    print(f"  cache: {cache}", flush=True)

    # ---- stage ----
    default_root = Path("/dev/shm") / f"fdp_{os.getpid()}"
    stage_root = Path(stage_dir) if stage_dir else default_root
    shot_stage = stage_root / str(shot)
    pointnames = [n for k, v in trees.items()
                  if k.lower() == PTDATA_KEY for n in v]
    mds_trees = config_trees(config, trees)

    if reuse_stage:
        staged = {
            "index": fdp_stage.load_cached_index(shot, shot_stage),
            "available_trees": {t for t in mds_trees
                                if (shot_stage / f"{t}_{shot}.tree").is_file()},
            "bytes": sum(p.stat().st_size for p in shot_stage.glob("*")
                         if p.is_file()),
            "seconds": 0.0,
        }
        staged["ptdata_sources"], staged["missing_pointnames"] = (
            fdp_stage.ptdata_objects(staged["index"], pointnames)[1:])
    else:
        staged = fdp_stage.stage_shot(shot, mds_trees, pointnames, shot_stage,
                                      workers=stage_workers)
    print(f"  staged: {staged['bytes']/1e9:.2f} GB in {staged['seconds']:.0f}s "
          f"({len(staged['available_trees'])}/{len(mds_trees)} trees, "
          f"{len(set(staged['ptdata_sources'].values()))} PTDATA objects) "
          f"-> {shot_stage}", flush=True)
    if staged["missing_pointnames"]:
        print(f"  {len(staged['missing_pointnames'])} pointnames absent from "
              f"the shot index, e.g. {staged['missing_pointnames'][:4]}",
              flush=True)

    mds_reader.configure_local_reads(shot_stage, mds_trees)
    # With the federation index absent (the MDSplus pass), PTDATA must be told
    # where the staged objects are, or the TDI expression nodes have nothing to
    # read. Harmless when the index is present: it is then ignored.
    os.environ["SYS_D3"] = str(shot_stage)

    # Most signals are TDI expressions over PTDATA, and they read pointnames
    # that are not in the config. Scan the (now local) records for those and
    # stage their objects too, or they fail once PTDATA is pointed at the
    # staging directory.
    t_scan = time.time()
    referenced = [] if not scan_expressions else fdp_stage.referenced_pointnames(
        shot, trees, mds_reader.signal_tree)
    index = staged["index"]
    extra_urls, extra_sources, _ = fdp_stage.ptdata_objects(index, referenced)
    if extra_urls:
        fdp_stage.stage_objects(extra_urls, shot_stage, workers=stage_workers)
    staged_bytes = sum(p.stat().st_size for p in shot_stage.glob("*")
                       if p.is_file())
    print(f"  expressions reference {len(referenced)} pointnames -> "
          f"{len(extra_urls)} more objects; stage now "
          f"{staged_bytes/1e9:.2f} GB ({time.time()-t_scan:.0f}s)", flush=True)

    if stage_only:
        print(f"shot {shot}: staged only, {staged_bytes/1e9:.2f} GB ready in "
              f"{time.time()-started:.0f}s", flush=True)
        return 1

    # Point PTDATA at the staged objects. SYS_D3 alone is not enough: with the
    # federation index set, ptdata 2.0 reads over the network even when the
    # object is local (ECE TECEF01: 18.9 s indexed vs 1.34 s staged).
    ptdata_reader = mds_reader.make_staged_ptdata_reader(shot_stage)
    ptdata_sources = staged["ptdata_sources"]
    retry: list[tuple[str, str]] = []

    import MDSplus as mds

    output_path = Path(output_dir) / f"{shot}.h5"
    handle, shot_group = mds_reader.open_output(output_path, shot)

    written = failed = empty = seen = 0
    failures: dict[str, int] = {}
    notes: dict[str, int] = {}
    #: Deferred separately from `retry`: exempt from the defer filter, because a
    #: degenerate read produced a value and dropping it would lose the signal.
    degenerate: list[tuple[str, str]] = []
    retry: list[tuple[str, str]] = []
    groups: dict[str, object] = {}

    tasks = [(key, signal) for key, signals in trees.items()
             for signal in signals]
    os.environ["FDP_FETCH_SHOT"] = str(shot)

    def record(key: str, signal: str, status, error_kind, note=None) -> None:
        """Account for one result; the data is already in a worker shard."""
        nonlocal written, failed, empty
        if note:
            notes[note] = notes.get(note, 0) + 1
        if error_kind is not None:
            failed += 1
            failures[error_kind] = failures.get(error_kind, 0) + 1
            if error_kind == "DegenerateRead":
                # Must bypass the defer filter. That filter drops a node whose
                # literal pointname is absent from the index, which is sound for
                # a node that raised -- but a degenerate read *produced* a value,
                # so dropping it loses a signal that would otherwise have been
                # written. Observed on \D3D::TOP.NEUTRALS.GASFLOW.PFX3:PCS.
                degenerate.append((key, signal))
            elif error_kind in RETRYABLE:
                retry.append((key, signal))
        elif status == "empty":
            empty += 1
        else:
            written += 1

    os.environ["FDP_KEEP_SCALARS"] = "1" if keep_scalars else "0"

    # Remove shards from an interrupted earlier pass: a half-written HDF5 file
    # cannot be reopened ("bad object header version number") and would abort
    # the merge for signals this pass read perfectly well.
    for stale in shot_stage.glob("shard_*.h5"):
        stale.unlink(missing_ok=True)

    t_read = time.time()
    try:
        if workers > 1:
            # Reads are independent and the cost is a mix of local I/O and
            # PTDATA float conversion, so they scale with processes. The parent
            # does all writing: h5py has no concurrent-writer support, and
            # MDSplus state is per process anyway.
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor

            # "spawn", not the default "fork": a forked child inherits HDF5's
            # global state, and workers then fail to exit at pool teardown --
            # every worker idle, the parent stuck in wait4, after all the data
            # was already read. Spawn costs ~1 s per worker at startup.
            print(f"  reading with {workers} workers", flush=True)
            with ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_init_worker,
                    initargs=(str(shot_stage), mds_trees,
                              staged["ptdata_sources"])) as pool:
                for key, signal, status, error_kind, note in pool.map(
                        _read_one, tasks, chunksize=4):
                    seen += 1
                    record(key, signal, status, error_kind, note)
                    if progress_every and seen % progress_every == 0:
                        print(f"    [{seen}/{len(tasks)}] {written} written, "
                              f"{failed} unavailable, {empty} empty "
                              f"({time.time()-t_read:.0f}s)", flush=True)
        else:
            _init_worker(str(shot_stage), mds_trees, staged["ptdata_sources"])
            for task in tasks:
                seen += 1
                record(*_read_one(task))
                if progress_every and seen % progress_every == 0:
                    print(f"    [{seen}/{len(tasks)}] {written} written, "
                          f"{failed} unavailable, {empty} empty "
                          f"({time.time()-t_read:.0f}s)", flush=True)
    finally:
        handle.close()

    # Merge the worker shards into the output. Groups are created only when a
    # shard actually has signals for them, so a diagnostic absent for this shot
    # leaves no empty group behind.
    t_merge = time.time()
    shards = sorted(shot_stage.glob("shard_*.h5"))
    if shards:
        import h5py

        handle, shot_group = mds_reader.open_output(output_path, shot)
        try:
            for shard_path in shards:
                with h5py.File(shard_path, "r") as shard:
                    for tree_name in shard:
                        target = shot_group.require_group(tree_name)
                        for signal_name in shard[tree_name]:
                            if signal_name in target:
                                if not overwrite:
                                    continue
                                del target[signal_name]
                            shard.copy(shard[tree_name][signal_name], target,
                                       signal_name)
        finally:
            handle.close()
        for shard_path in shards:
            shard_path.unlink(missing_ok=True)
        print(f"  merged {len(shards)} shards in {time.time()-t_merge:.0f}s",
              flush=True)

    # Retrying in this process is useless for a whole class of failures. The
    # MDSplus pass unsets PTDATA_JSON_INDEX_DIR on purpose, and PTDATA-backed
    # TDI expression nodes (the NB beam CURRENT/VOLTAGE/PERVEANCE/F0 nodes, for
    # one) then raise TdiABORT no matter how often they are asked -- staged
    # objects do not help, because the index is what routes the read. Writing
    # them to a manifest hands them to a pass that has the index instead.
    if write_retry_manifest is not None:
        manifest = Path(write_retry_manifest)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        unreadable: list[str] = []
        if defer_filter:
            retry, unreadable = fdp_stage.split_unreadable_nodes(
                shot, retry, staged["index"], mds_reader.signal_tree)
        retry = retry + degenerate
        manifest.write_text(json.dumps(retry))
        print(f"  deferred {len(retry)} signals to {manifest}"
              f" ({len(degenerate)} degenerate)", flush=True)
        if unreadable:
            # Dropping work is only acceptable if it leaves a record, so the
            # names go next to the manifest rather than only into a count.
            skipped = manifest.with_suffix(".skipped.json")
            skipped.write_text(json.dumps(unreadable))
            print(f"  skipped {len(unreadable)} nodes whose pointname is absent "
                  f"from the shot index (listed in {skipped.name}), e.g. "
                  f"{[s.split(':')[-1] for s in unreadable[:3]]}", flush=True)
        retry = []

    deferred: list[tuple[str, str]] = []
    if retry_manifest is not None and Path(retry_manifest).is_file():
        deferred = [tuple(entry) for entry
                    in json.loads(Path(retry_manifest).read_text())]

    # Anything the staged pass could not read is retried over the federation,
    # so a pointname whose object was not staged is slow rather than lost. Only
    # failures get here, so the cost is bounded.
    todo = ([] if no_retry else retry + degenerate) + deferred
    if todo:
        recovered = deferred_recovered = 0
        t_retry = time.time()
        # A deferred signal's tree is usually not among this pass's own trees
        # (the PTDATA pass carries no d3d signals of its own), so its staged
        # tree file has to be made resolvable before the read.
        extra_trees = sorted({mds_reader.signal_tree(s, k) for k, s in todo
                              if k.lower() != PTDATA_KEY} - {""})

        # Make this pass actually federated.  configure_local_reads has pinned
        # every <tree>_path to the staging directory, so without this the
        # "retry over the federation" silently re-reads the same staged trees
        # and gets the same answer.  It matters: the staged `ece` tree for shot
        # 146918 has no data in \ECE::ECPTNAMF, so every TECEF channel fails
        # TreeNODATA against it, while the federated copy reads all 40 at
        # 4096000 samples.  MDSplus falls back to default_tree_path once the
        # per-tree override is gone.
        pinned = {}
        for tree_name in set(mds_trees) | set(extra_trees):
            variable = f"{tree_name.lower()}_path"
            if variable in os.environ:
                pinned[variable] = os.environ.pop(variable)

        handle, shot_group = mds_reader.open_output(output_path, shot)
        try:
            for index_in_todo, (key, signal) in enumerate(todo):
                try:
                    if key.lower() == PTDATA_KEY:
                        payload = mds_reader.gather_ptdata(signal, shot)
                        defect = mds_reader.ptdata_defect(payload)
                        if defect:
                            payload, _ = _repair_ptdata(
                                signal, payload, defect, shot)
                    else:
                        tree_name = mds_reader.signal_tree(signal, key)
                        tree = mds.Tree(tree_name, shot)
                        payload = mds_reader.gather_mds(tree, signal)
                    data = payload.get("data")
                    if data is None or data.ndim == 0 or data.size == 0:
                        continue
                    group = shot_group.require_group(
                        mds_reader.tree_group_name(key))
                    mds_reader.write_signal(
                        group, mds_reader.signal_group_name(key, signal),
                        payload, overwrite=overwrite)
                    recovered += 1
                    if index_in_todo >= len(todo) - len(deferred):
                        deferred_recovered += 1
                except Exception as retry_error:  # noqa: BLE001
                    # Swallowing these by default is deliberate -- a retry that
                    # fails is not a shot failure -- but it makes a broken retry
                    # indistinguishable from one never attempted, which cost
                    # three misdiagnosed rounds on the ECE channels. Set
                    # FDP_DEBUG_RETRY=1 to see them.
                    if os.environ.get("FDP_DEBUG_RETRY"):
                        print(f"    retry fail {signal[-24:]}: "
                              f"{type(retry_error).__name__}: "
                              f"{str(retry_error)[:80]}", flush=True)
                    continue
        finally:
            handle.close()
            os.environ.update(pinned)
        written += recovered
        # Deferred signals failed in an earlier pass, so they were never in
        # this pass's failure count and must not be subtracted from it.
        failed -= recovered - deferred_recovered
        if deferred:
            print(f"  deferred signals recovered: "
                  f"{deferred_recovered}/{len(deferred)}", flush=True)
        print(f"  retry over the federation: {recovered}/{len(todo)} "
              f"recovered in {time.time()-t_retry:.0f}s", flush=True)

    if not keep_stage:
        shutil.rmtree(stage_root, ignore_errors=True)

    elapsed = time.time() - started
    print(f"  read: {written} written, {failed} unavailable, "
          f"{empty} empty/scalar dropped in {elapsed:.0f}s", flush=True)
    if failures:
        summary = ", ".join(f"{k}x{v}" for k, v in
                            sorted(failures.items(), key=lambda kv: -kv[1]))
        print(f"  failure kinds: {summary}", flush=True)
    if notes:
        # A leftover "defect:" line means a bad timebase or an uncalibrated
        # read was written as-is, which is worth seeing rather than inferring.
        summary = ", ".join(f"{k}x{v}" for k, v in
                            sorted(notes.items(), key=lambda kv: -kv[1]))
        print(f"  PTDATA fixups: {summary}", flush=True)
    print(f"shot {shot}: done in {elapsed:.0f}s -> {output_path}", flush=True)
    return written


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch one DIII-D shot from the FDP into an HDF5 file")
    parser.add_argument("--shot", type=int, required=True, help="Shot number")
    parser.add_argument("--config", type=Path, action="append", required=True,
                        help="Signal config YAML (repeatable)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write <shot>.h5 into")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rewrite signals already present in the output")
    parser.add_argument("--only-tree", action="append", default=None,
                        help="Fetch only this config tree key (repeatable)")
    parser.add_argument("--skip-tree", action="append", default=None,
                        help="Skip this config tree key (repeatable)")
    parser.add_argument("--keep-scalars", action="store_true",
                        help="Write empty and 0-d nodes too; the old pipeline "
                             "omitted them and so do we by default")
    parser.add_argument("--stage-dir", type=Path, default=None,
                        help="Parent dir for staging (default: /dev/shm, i.e. RAM)")
    parser.add_argument("--stage-workers", type=int, default=6,
                        help="Concurrent object transfers (default: 6)")
    parser.add_argument("--keep-stage", action="store_true",
                        help="Do not delete the staged objects afterwards")
    parser.add_argument("--workers", type=int, default=8,
                        help="Worker processes for reading (default: 8)")
    parser.add_argument("--stage-only", action="store_true",
                        help="Stage the shot's objects and exit (pass 1 of 3)")
    parser.add_argument("--reuse-stage", action="store_true",
                        help="Staging directory is already populated")
    parser.add_argument("--no-scan", action="store_true",
                        help="Skip the TDI record scan (PTDATA-only passes)")
    parser.add_argument("--no-retry", action="store_true",
                        help="Skip the federated retry pass for signals the "
                             "staged pass could not read")
    parser.add_argument("--write-retry-manifest", type=Path, default=None,
                        help="Write this pass's retryable failures here for a "
                             "later pass instead of retrying them in place")
    parser.add_argument("--retry-manifest", type=Path, default=None,
                        help="Retry signals deferred by an earlier pass")
    parser.add_argument("--no-defer-filter", action="store_true",
                        help="Defer every failure, including nodes whose "
                             "pointname the shot index says does not exist")
    parser.add_argument("--allow-no-fdp", action="store_true",
                        help="Skip the `fdp run` check (for debugging only)")
    args = parser.parse_args()

    if not args.allow_no_fdp:
        require_fdp_environment()

    written = fetch_shot(
        args.shot,
        args.config,
        args.output_dir,
        overwrite=args.overwrite,
        only_trees=args.only_tree,
        skip_trees=args.skip_tree,
        keep_scalars=args.keep_scalars,
        stage_dir=args.stage_dir,
        stage_workers=args.stage_workers,
        keep_stage=args.keep_stage,
        no_retry=args.no_retry,
        reuse_stage=args.reuse_stage,
        scan_expressions=not args.no_scan,
        stage_only=args.stage_only,
        workers=args.workers,
        write_retry_manifest=args.write_retry_manifest,
        retry_manifest=args.retry_manifest,
        defer_filter=not args.no_defer_filter,
    )

    sys.exit(0 if written else 1)


if __name__ == "__main__":
    main()
