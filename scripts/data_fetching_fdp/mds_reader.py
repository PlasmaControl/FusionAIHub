"""Read DIII-D signals through FDP and write them in the mdsh5 layout.

Reads go through the supported FDP stack, which needs no local staging: ``fdp
run`` points MDSplus and PTDATA at ``pelican://`` URLs, ``libfdpio2`` turns
their file operations into object reads, and ``libXrdCl`` issues HTTP range
requests against a Pelican cache.  So a signal is fetched by asking for it --
there is nothing to copy first.

Two consequences for this module:

* No environment setup here.  ``fdp run`` supplies ``default_tree_path``,
  ``PTDATA_JSON_INDEX_DIR``, ``PTDATA_PLUGIN_LIB``, ``MDS_PATH`` and the
  bearer token; the ``d3d-cache/1.0`` module (or the three variables it sets)
  supplies the cache preference.  This module only reads.
* PTDATA pointname lookup is the index plugin's job.  ``libjson_index_plugin``
  consults the JSON index to find which object holds a pointname, so no
  pointname-to-file map is needed.

The HDF5 layout reproduces what ``mdsh5``'s ``read_mds`` produced on omega,
so files from either pipeline are interchangeable downstream::

    <shot>/
      <TREE>/                            # upper case, e.g. D3D, EFIT01, PTDATA
        <signal path>/                   # verbatim from the config
          data                          # signal values
          dim0 ... dimN                 # one per data dimension
          units                         # scalar byte string

Datasets are contiguous and uncompressed, matching the reference files.  Nodes
whose data is empty or 0-dimensional are not written: the old pipeline omitted
them, and a config asks for plenty of both.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def configure_local_reads(stage_dir: Path, trees: Iterable[str]) -> None:
    """Point MDSplus at staged tree files instead of the federation.

    Sets ``<tree>_path`` for each tree, which MDSplus prefers over
    ``default_tree_path``.  Must be called before any tree is opened.

    Parameters
    ----------
    stage_dir : Path
        Directory holding the staged ``<tree>_<shot>.*`` files.
    trees : iterable of str
        Tree names to redirect.
    """
    for tree in trees:
        os.environ[f"{tree.lower()}_path"] = str(stage_dir)


def make_staged_ptdata_reader(stage_dir: Path):
    """Return a PTDATA reader that reads staged shot files.

    This is what ``ptdata.fsspec_provider.FsspecReader`` does internally once
    it has downloaded an object: construct a ``PtDataReader`` over a local
    directory.  We stage with ``pelican object get`` instead of ``fsspec``
    (see :mod:`fdp_stage`), so the reader is built directly.

    Parameters
    ----------
    stage_dir : Path
        Directory holding the staged ``<shot>.<EXT>`` files.

    Returns
    -------
    ptdata.PtDataReader
        Reader restricted to that directory.
    """
    import ptdata

    return ptdata.PtDataReader(sys_d3_paths=[str(stage_dir)])


def gather_mds(tree, expression: str) -> dict:
    """Read one MDSplus node, with all its dimensions and units.

    Parameters
    ----------
    tree : MDSplus.Tree
        An open tree.  Opening it requires no treepath argument: MDSplus falls
        back to ``default_tree_path``, which ``fdp run`` sets to the federated
        ``pelican://`` locations.
    expression : str
        Node path or TDI expression, e.g.
        ``r"\\D3D::TOP.ELECTRONS.TS.BLESSED.CORE:TEMP"``.

    Returns
    -------
    dict
        ``{"data": ndarray, "dim0": ndarray, ..., "units": bytes}``.  Only
        dimensions that could be read are present.

    Raises
    ------
    Exception
        Whatever MDSplus raises for an absent or empty node (``TreeNNF``,
        ``TreeNODATA``, ``TdiABORT``, ...).  Callers treat these as per-signal
        failures, not fatal.
    """
    node = tree.getNode(expression)
    data = _as_array(node.data())
    payload: dict = {"data": data}

    for axis in range(data.ndim):
        try:
            payload[f"dim{axis}"] = _as_array(node.getDimensionAt(axis).data())
        except Exception:
            # Plenty of nodes carry fewer dimension records than data axes;
            # the old pipeline simply omitted them.
            break

    try:
        payload["units"] = _as_bytes(node.getUnits())
    except Exception:
        pass

    return payload


def gather_ptdata_staged(reader, pointname: str, shot: int, source: str,
                         ical: int = 1, as_float32: bool = True) -> dict:
    """Read one PTDATA pointname from a staged shot file.

    Parameters
    ----------
    reader : ptdata.PtDataReader
        From :func:`make_staged_ptdata_reader`.
    pointname : str
        PTDATA pointname.
    shot : int
        Shot number.
    source : str
        Shot-file extension holding it, e.g. ``".MAG"``, from the index.
    ical : int, default 1
        Calibration flag: 1 is physics units.
    as_float32 : bool, default True
        Store the data as ``float32``, which is what the old pipeline did.  The
        reader hands back ``float64`` while PTDATA's own TDI path returns
        ``float32``, so the extra width carries no information and doubles the
        file size.

    Returns
    -------
    dict
        ``{"data": ndarray, "dim0": ndarray, "units": bytes}``.
    """
    import ptdata

    params = ptdata.ExtractionParams()
    if hasattr(params, "ical"):
        params.ical = ical
    result = reader.fetch(pointname, shot, params, source)

    if not isinstance(result, dict):
        result = {
            "data": getattr(result, "data", None),
            "times": getattr(result, "times", None),
            "units": getattr(result, "units", None),
        }

    data = np.asarray(result["data"])
    if as_float32 and data.dtype == np.float64:
        data = data.astype(np.float32)
    payload: dict = {"data": data}
    times = result.get("times")
    if times is not None:
        payload["dim0"] = np.asarray(times)
    units = result.get("units")
    if isinstance(units, dict):
        units = units.get("data")
    if units is not None:
        payload["units"] = _as_bytes(units)
    return payload


#: Sample count above which a staged PTDATA read is not trusted without a TDI
#: cross-check.  Calibrated on two shots: the largest read observed to *agree*
#: with the reference is exactly 2**23 (the BES fast channels on shot 146918,
#: 128 of them), and the smallest to disagree is 14,992,000 (the CO2 channels on
#: 144300).  So the boundary lies in (2**23, 14.99e6] and this is the largest
#: value the evidence supports.
#:
#: Do not lower this on one shot's evidence.  It was first set to 2**21 from
#: shot 144300 alone, where nothing fell in between and the largest good read
#: happened to be 2**21; on 146918 that would have re-read 128 correct channels
#: and 1.07e9 samples over the federation for nothing.
BULK_SAMPLE_LIMIT = 2 ** 23

#: ``units`` seen only on staged reads that disagreed with the reference.  On
#: shot 144300 it appears on all four bad reads and on none of the 31 good ones.
#: Whether it is causal or merely correlated with the record size is unresolved.
SUSPECT_UNITS = frozenset({b"f"})


def ptdata_defect(payload: dict) -> Optional[str]:
    """Return the defect in a ``PtDataReader`` payload, or None if it is sound.

    ``ptdata.PtDataReader`` gets two things wrong that GA's TDI ``PTDATA2()``
    gets right, and both are detectable from the payload alone:

    ``"uncalibrated"``
        ``units == b"vo"`` means the calibration was not applied and the values
        are raw volts.  Measured on shot 115401, ``IPTIPP`` was off by 5e5x and
        ``DSTDENP`` by 1.7e16x.
    ``"timebase"``
        Multi-rate pointnames come back with a ``dim0`` that duplicates a
        sample at each rate change, then lags by one and jumps backwards.  A
        physical timebase is strictly increasing, so non-monotonicity is proof
        of the defect and cannot false-positive.  ``data`` is unaffected.

    The check is O(n) on an array already in memory, so it is run on every
    pointname.  See :func:`gather_ptdata_tdi` for the repair.

    Parameters
    ----------
    payload : dict
        As returned by :func:`gather_ptdata_staged`.

    Returns
    -------
    str or None
        ``"uncalibrated"``, ``"timebase"``, or None.
    """
    units = payload.get("units")
    if units is not None and bytes(units).strip().lower() == b"vo":
        return "uncalibrated"

    # Third class: the values are neither mis-scaled nor mis-timed but simply
    # wrong.  On shot 144300 the four CO2 channels DENV1UF/V2UF/V3UF/R0UF come
    # back spanning [-7.1e12, 1.6e14] where the reference and TDI both give
    # exactly zero -- a magnitude no interpretation of the channel admits (ADC
    # counts, volts, fringes, and density all rule it out, the last because it
    # would need to be negative).  Unlike the two cases above there is no
    # self-validating signature, so this leans on two correlated markers and
    # accepts false positives: a needless TDI read costs seconds, while a missed
    # one writes 1e14 into a density channel.  :func:`gather_ptdata_tdi` is
    # authoritative here -- it reproduced the reference exactly for all eight
    # affected channels across shots 144300 and 146918.
    data = payload.get("data")
    if data is not None and getattr(data, "size", 0) > BULK_SAMPLE_LIMIT:
        return "unverifiable_bulk"
    if units is not None and bytes(units).strip().lower() in SUSPECT_UNITS:
        return "unverifiable_bulk"

    times = payload.get("dim0")
    if times is not None:
        axis = np.asarray(times)
        if axis.ndim == 1 and axis.size > 2 and axis.dtype.kind in "fiu":
            if not np.all(np.diff(axis.astype("f8")) > 0):
                return "timebase"
    return None


def gather_ptdata_tdi(pointname: str, shot: int,
                      as_float32: bool = True) -> dict:
    """Read one PTDATA pointname through GA's TDI ``PTDATA2()`` function.

    This is the path the omega pipeline used, and it reconstructs multi-rate
    timebases and applies calibration correctly: on shot 115401 it reproduced
    the reference file exactly (``dim0`` and ``data`` both max|d| = 0) for the
    six pointnames :func:`ptdata_defect` flags.

    Two costs make this a fallback rather than the default.  It needs
    ``PTDATA_JSON_INDEX_DIR`` present -- with the index absent it returns a
    degenerate shape ``(1,)`` even when the objects are staged -- and because
    reads then go over the network it takes 2.5-5.4 s per pointname against
    milliseconds for the staged reader.  It also reports ``units`` as blank,
    where the staged reader returns the real ones.

    Parameters
    ----------
    pointname : str
        PTDATA pointname.
    shot : int
        Shot number.
    as_float32 : bool, default True
        Store the data as ``float32``, matching :func:`gather_ptdata_staged`.

    Returns
    -------
    dict
        ``{"data": ndarray, "dim0": ndarray, "units": bytes}``.

    Raises
    ------
    Exception
        Whatever MDSplus raises if the expression cannot be evaluated.
    """
    import MDSplus as mds

    signal = mds.Data.execute(f'PTDATA2("{pointname}",{shot})')

    data = _as_array(signal.data())
    if as_float32 and data.dtype == np.float64:
        data = data.astype(np.float32)
    payload: dict = {"data": data}

    try:
        payload["dim0"] = _as_array(signal.dim_of().data())
    except Exception:
        # A pointname with no dimension record is not worth failing over; the
        # caller keeps the staged timebase in that case.
        pass

    try:
        payload["units"] = _as_bytes(signal.units)
    except Exception:
        pass

    return payload


def gather_ptdata(pointname: str, shot: int, ical: int = 1) -> dict:
    """Read one PTDATA pointname through the federation.

    Parameters
    ----------
    pointname : str
        PTDATA pointname, e.g. ``"MPI1A322D"``.
    shot : int
        Shot number.
    ical : int, default 1
        PTDATA calibration flag: 1 is physics units, 0 raw counts.

    Returns
    -------
    dict
        ``{"data": ndarray, "dim0": ndarray, "units": bytes}``.

    Raises
    ------
    Exception
        ``PtDataError`` if the index has no entry for this pointname and shot.

    Notes
    -----
    ``PtDataSignal(remote=...)`` is deliberately not passed.  It sets
    ``PTDATA_LOC``, which is a no-op in the ptdata 2.0 engine -- routing is
    decided by ``PTDATA_JSON_INDEX_DIR`` instead.

    A ``units`` of ``b"vo"`` means the calibration was not applied and the
    values are raw volts; check it before trusting a pointname's scaling.
    """
    from toksearch_d3d import PtDataSignal

    result = PtDataSignal(pointname, ical=ical).gather(shot)

    payload: dict = {"data": np.asarray(result["data"])}
    if "times" in result:
        payload["dim0"] = np.asarray(result["times"])

    units = (result.get("units") or {}).get("data")
    if units is not None:
        payload["units"] = _as_bytes(units)

    return payload


def _as_array(value) -> np.ndarray:
    """Coerce an MDSplus value to an array, matching the mdsh5 conventions.

    Text-valued nodes are stored by the old pipeline as ``|S1`` arrays of
    single characters rather than as a scalar string, so a 19-character node
    becomes shape ``(19,)``.  Keeping that means files from either pipeline
    compare equal.

    Parameters
    ----------
    value : object
        Whatever ``node.data()`` returned.

    Returns
    -------
    numpy.ndarray
        Array form of the value.
    """
    if isinstance(value, (str, bytes)):
        raw = value.encode("utf-8", "replace") if isinstance(value, str) else value
        return np.frombuffer(raw, dtype="S1")

    array = np.asarray(value)
    if array.ndim == 0 and array.dtype.kind in ("U", "S"):
        raw = array.item()
        if isinstance(raw, str):
            raw = raw.encode("utf-8", "replace")
        return np.frombuffer(raw, dtype="S1")

    return array


def _as_bytes(value) -> bytes:
    """Coerce an MDSplus/PTDATA units value to bytes for HDF5 storage."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8", "replace")
    try:
        return str(value.data()).encode("utf-8", "replace")
    except Exception:
        return str(value).encode("utf-8", "replace")


def write_signal(parent, name: str, payload: dict, overwrite: bool = False) -> None:
    """Write one signal's payload as a group of datasets.

    Parameters
    ----------
    parent : h5py.Group
        The tree group, e.g. ``f["200000/D3D"]``.
    name : str
        Group name: the config's signal path verbatim, or the upper-cased
        pointname for PTDATA.
    payload : dict
        As returned by :func:`gather_mds` or :func:`gather_ptdata`.
    overwrite : bool, default False
        Replace an existing group instead of leaving it alone.
    """
    if name in parent:
        if not overwrite:
            return
        del parent[name]

    group = parent.create_group(name)
    for key, value in payload.items():
        if key == "units":
            group.create_dataset("units", data=np.bytes_(value))
        else:
            group.create_dataset(key, data=value)


def open_output(path: Path, shot: int) -> tuple:
    """Open (or create) the per-shot output file and its shot group.

    Parameters
    ----------
    path : Path
        Output HDF5 file path.
    shot : int
        Shot number, used as the top-level group name.

    Returns
    -------
    handle : h5py.File
        The open file, in append mode.
    shot_group : h5py.Group
        The ``<shot>`` group.
    """
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(path, "a")
    return handle, handle.require_group(str(shot))


def tree_group_name(config_key: str) -> str:
    """Return the HDF5 group name for a config tree key (upper case)."""
    return config_key.upper()


def signal_group_name(config_key: str, signal: str) -> str:
    """Return the HDF5 group name for one signal.

    MDSplus paths are stored verbatim; PTDATA pointnames are upper-cased,
    which is what the old pipeline did (a config asking for ``ip`` produced a
    group ``IP``).

    Parameters
    ----------
    config_key : str
        Tree key the signal came from.
    signal : str
        Signal path or pointname as written in the config.

    Returns
    -------
    str
        Group name to use in the HDF5 file.
    """
    if config_key.lower() == "ptdata":
        return signal.upper()
    return signal


def signal_tree(expression: str, default: Optional[str] = None) -> str:
    """Return the MDSplus tree to open for a signal expression.

    Parameters
    ----------
    expression : str
        MDSplus signal path.
    default : str, optional
        Fallback when the expression carries no ``TREE::`` prefix.

    Returns
    -------
    str
        Lower-case tree name.
    """
    if "::" in expression:
        prefix = expression.split("::", 1)[0].lstrip("\\")
        if prefix:
            return prefix.lower()
    return (default or "").lower()
