"""Features from fdp: PTDATA points and MDSplus trees.

This is the path that scales. The archive store covers 3,246 of the corpus'
16,909 shots; everything else, and every shot recorded after 191450, has to
be fetched. The access patterns are copied from
/scratch/gpfs/nc1514/fdp/scripts/omnimode.py, a working fetch script against
this cluster's Pelican/OSDF route:

  * `MdsSignal(expr, tree).fetch(shot)` -> {"data", "units", *dims}
  * `PtDataSignal(name).fetch(shot)` for PTDATA
  * fdp has no retries and no caching; one retry here, then a recorded miss
  * toksearch_d3d's ptserver reader is not fork-safe, so a worker pool must
    be forked before the first fetch (see the runner)

Every fetch must run under the `fdp run` wrapper, which supplies the server
configuration. Without it PTDATA fails with `getservbyname failed for task
'PTSERVER'` and MDSplus with `TREE-E-FOPENR`.

Imports of toksearch are deferred into `available()` and the fetch helpers,
so importing labeler in an environment without the `fdp` feature never
touches it.

MEASUREMENT RECORD - node probe, 2026-09-04, shot 185945, fdp 0.5.1 /
toksearch 2.11.1 / toksearch_d3d 0.10.1:

    PTDATA:
      ip       keys=['data','n_over','n_under','times','units'] data=(30720,) float64 dims={'times':(30720,)} units={'data':'a','times':'ms'}
      bt       keys=['data','n_over','n_under','times','units'] data=(50176,) float64 dims={'times':(50176,)} units={'data':'t','times':'ms'}
    efit01 aeqdsk scalars:
      kappa    keys=['data','times','units'] data=(305,) float32 dims={'times':(305,)} units={'data':' ','times':'ms'}
      tritop   keys=['data','times','units'] data=(305,) float32 dims={'times':(305,)} units={'data':' ','times':'ms'}
      tribot   keys=['data','times','units'] data=(305,) float32 dims={'times':(305,)} units={'data':' ','times':'ms'}
      gapin    keys=['data','times','units'] data=(305,) float32 dims={'times':(305,)} units={'data':'m','times':'ms'}
      betan    keys=['data','times','units'] data=(305,) float32 dims={'times':(305,)} units={'data':' ','times':'ms'}
      atime    keys=['data','times','units'] data=(305,) float32 dims={'times':(305,)} units={'data':'ms','times':'ms'}
      time     keys=['data','times','units'] data=(305,) float32 dims={'times':(305,)} units={'data':'V s / rad','times':'ms'}
    efit01 geqdsk:
      gtime    keys=['data','times','units'] data=(305,) float32 dims={'times':(305,)} units={'data':'ms','times':'ms'}
      rmaxis   keys=['data','times','units'] data=(305,) float32 dims={'times':(305,)} units={'data':'m','times':'ms'}
      qpsi     keys=['data','times','units'] data=(305,65) float32 dims={'times':(65,)} units={'data':' ','times':'normalized psi'}
      pres     keys=['data','times','units'] data=(305,65) float32 dims={'times':(65,)} units={'data':'N / m^2','times':'normalized psi'}
      rhovn    keys=['data','times','units'] data=(305,65) float32 dims={'times':(65,)} units={'data':' ','times':'normalized psi'}
    zipfit01 profiles (dims=['x','t_ms']):
      EDENSFIT keys=['data','t_ms','units','x'] data=(288,121) float32 dims={'t_ms':(288,),'x':(121,)} units={'data':'10^19 m^-3','x':'rho','t_ms':'ms'}
      ETEMPFIT keys=['data','t_ms','units','x'] data=(291,121) float32 dims={'t_ms':(291,),'x':(121,)} units={'data':'keV','x':'rho','t_ms':'ms'}
      TROTFIT  keys=['data','t_ms','units','x'] data=(233,121) float32 dims={'t_ms':(233,),'x':(121,)} units={'data':'kHz','x':'rho','t_ms':'ms'}

What that record settles, and the two traps in it:

  * every 1-D record carries its own time axis, so no separate time-node
    fetch is needed for the scalars. `aeqdsk:time` is a TRAP - its units are
    `V s / rad`, so it is poloidal flux; `atime` is the time-valued leaf.
  * for the geqdsk 2-D nodes `rec["times"]` is NOT time: it is the RADIAL
    axis, 65 points of normalized psi. Their time axis is absent from the
    record and comes from `gtime`. Every axis below is therefore identified
    by its units and its length, never by its position.
  * PTDATA units are SI: `ip` in amps, `bt` in tesla, and no unit factor is
    needed against the archive (measured below).
  * `rot_zipfit` is in kHz, `ne_zipfit` in 1e19 m^-3, `te_zipfit` in keV -
    read off the `units` field, which magnitude alone could not have
    decided.

THE PROFILE COORDINATE, MEASURED - the archive is the oracle here, because
its columns are the model's own training inputs. The geqdsk profiles arrive
on 65 points of normalized psi and `rhovn` (65 points, per time slice) is
the psi->rho map, so the obvious reading is that they need converting before
they can share a 33-point rho grid. They do not, and the archive says so.
On shot 185945, archive rows aligned to EFIT slices EXACTLY (the 55 rows
whose time, lagged as below, coincides with a `gtime` sample to better than
1e-6 ms - so none of this is a sampling artefact):

      coordinate passed to the 33-point grid | qpsi     | pres
      uniform normalized psi                 | 9.7e-03  | 2.3e-02
      rhovn (the psi->rho map)               | 6.9e-02  | 2.4e-01
      sqrt(normalized psi)                   | 1.2e-01  | 5.3e-01

(median relative difference against `qpsi_EFIT01` / `pres_EFIT01`.) The psi
reading wins by 7x on qpsi and 10x on pres, so the archive's 33-point EFIT
profile axis is uniform normalized psi and NOT rho, whatever `RHO_GRID` is
called. It is in fact the even indices of the geqdsk axis: 65 = 2*32 + 1, so
`linspace(0, 1, 33)[i] == linspace(0, 1, 65)[2i]` and no interpolation
happens at all on this path. Converting to rho would move every off-axis
point of the model's input by up to 0.22 in coordinate - the measured
`max|rhovn - psi|` on that shot - which is exactly why this had to be
measured rather than reasoned about. `GEQDSK_RHO` is kept below as the
record of the alternative that was tried and rejected.

`rhovn` itself is well behaved, which the interpolation would have needed:
on shot 185945 all 305 slices are strictly increasing from 0 to 1 with a
minimum step of 8.7e-3, and over the 120-shot random sample below every one
of 30,052 time slices is strictly increasing. `_to_rho_grid` sorts its
coordinate anyway and NaNs anything outside its range, so a slice that was
not monotonic would be resampled from a sorted copy rather than silently
producing a folded profile.

THE ARCHIVE LAG, MEASURED - the archive's rows are not at the times this
repo's `resolve_archive` assigns them. Scanning the offset that best matches
each archive row against the raw record on shot 185945:

  * `ip` (median relative difference against `ip`): a 25 ms window mean is
    best at a -37.5 ms window start, i.e. centred on t - 25 ms, at 1.19e-03,
    against 3.40e-03 for the same window at zero offset, 4.58e-03 for the
    best nearest-sample offset (-19 ms) and 6.51e-03 for nearest-sample at
    zero offset.
  * on-axis `pres` (coordinate-free, so this is a pure time result): best at
    -25 ms, 1.07e-02, against 3.81e-02 at zero offset, monotonically worse
    in both directions.

Both say the same thing: archive row k holds the value at 25*(k-1) ms, one
row earlier than `STEP_S * arange(n)` claims, and it is a 50 ms window
statistic rather than a point sample (see `models.base.ARCHIVE_WINDOW_S`,
`2 * STEP_S`). `ARCHIVE_LAG_S` records the offset for the comparison
scripts; nothing in the resolver depends on it, and correcting
`resolve_archive` is not this module's business (it would move every feature
Task 15 compares).

FDP AGAINST THE ARCHIVE, MEASURED UNDER A SUPERSEDED CONVENTION - a RANDOM
120 of the 3,246 corpus/archive overlap shots (`catalog.sample_shots(overlap,
120, seed=12)`; not the first 120 by number, which would have been the early
shot range only). Each archive row was compared at `t + ARCHIVE_LAG_S`: the
two PTDATA points against a 25 ms window mean centred there, everything else
against the nearest source slice with `max_gap` half a grid step, so nothing
was compared to a clamped edge. "exact" repeats the comparison on the subset
of rows whose time coincides with a source sample to 1e-9 s, which removes
the remaining sampling difference for the 20 ms EFIT and ZIPFIT nodes.

**This table's own convention no longer matches what `build()` uses**:
`InputSpec.build` now samples EVERY fdp-resolved field
- not just the two PTDATA points - with the archive's own 50 ms window ending
at `t` (`ARCHIVE_WINDOW_S`, via `ns.SAMPLING_BY_SOURCE`/`sample_by_resolver`).
Kept below for the unit-factor and profile-axis conclusions, which do not
depend on the windowing convention; superseded for accuracy purposes by the
re-measurement immediately after it.

    feature      med rel   p90 rel   med ratio  shots   points | exact med rel
    ip          1.08e-03  4.81e-03    1.000034     39     9360 |            -
    bt          1.03e-03  2.58e-03    0.999992     39     9360 |            -
    r0          7.87e-04  3.05e-03    1.000000    104     5555 |     8.38e-04
    kappa       1.22e-03  5.82e-03    1.000000    104     5555 |     1.30e-03
    tritop      3.19e-03  6.98e-02    1.000000    104     5555 |     3.13e-03
    tribot      2.77e-03  6.93e-02    1.000000    104     5555 |     2.96e-03
    gapin       7.33e-03  6.54e-02    1.000000    104     5554 |     7.68e-03
    betan       1.31e-02  5.66e-02    1.000000    104     5555 |     1.18e-02
    qpsi        6.93e-03  3.24e-02    0.999416    118   186351 |     6.38e-03
    pres        3.00e-02  2.36e-01    1.000000    118   180704 |     2.89e-02
    ne_zipfit   1.26e-02  4.94e-02    0.999352     92   143616 |     1.18e-02
    te_zipfit   1.24e-02  6.34e-02    0.999438     92   156222 |     1.24e-02
    rot_zipfit  1.09e-02  7.13e-02    1.000000     93   124806 |     1.19e-02

NO UNIT FACTOR IS NEEDED ON ANY OF THE THIRTEEN. Every median ratio is 1.000
to within 6e-4, so there is no scale constant in this module: `ip` in amps
and `bt` in tesla are what the archive - and therefore the model - was
trained on, and the three ZIPFIT profiles in 1e19 m^-3, keV and kHz likewise.
Had any of these wanted a power of ten it would have shown here as 1e3 or
1e-3, not as a spread near 1. What is left is a spread near 1: a sampling and
provenance difference, biggest on `pres` (3.0e-2, p90 2.4e-1) which is the
steepest-in-time profile of the set, and it does not shrink much on the
exactly-coincident subset, so it is not purely a sampling artefact - the
offline EFIT01 tree has been rerun at least once since the store was built.
Everything is far inside the plan's 5e-2 gate.

FDP AGAINST THE ARCHIVE, RE-MEASURED UNDER THE CONVENTION `build()` ACTUALLY
USES (Task 15) - all thirteen features, each source's array
windowed with `window_mean(x, y, t - ARCHIVE_WINDOW_S, ARCHIVE_WINDOW_S)`
against the archive column at the archive's own grid times `t`, exactly what
`InputSpec.build` does for a `"fdp"`-resolved field. A RANDOM 10 of the
overlap shots (`catalog.sample_shots(catalog.overlap_shots(paths), 10,
seed=12)`; script `scan_fdp_price_table.py`, not committed - see the Task 15
report):

    186154, 186644, 186727, 186743, 187255, 189059, 189681, 189744, 190835,
    190915

    feature      med rel     p90 rel     shots   points | this table's old med rel
    ip          9.80e-05    1.39e-03        5     1200  |           1.08e-03
    bt          7.19e-05    1.86e-04        5     1200  |           1.03e-03
    r0          3.26e-08    1.99e-04       10     2094  |           7.87e-04
    kappa       3.00e-08    4.32e-04       10     2094  |           1.22e-03
    tritop      3.39e-08    5.49e-04       10     2094  |           3.19e-03
    tribot      3.22e-08    6.44e-04       10     2094  |           2.77e-03
    gapin       3.19e-08    1.73e-03       10     2094  |           7.33e-03
    betan       3.33e-08    2.94e-03       10     2094  |           1.31e-02
    qpsi        3.56e-08    5.11e-03       10    69069  |           6.93e-03
    pres        3.50e-08    1.92e-02       10    66976  |           3.00e-02
    ne_zipfit   2.45e-08    2.80e-03        9    59565  |           1.26e-02
    te_zipfit   2.37e-08    3.22e-03        9    59532  |           1.24e-02
    rot_zipfit  2.23e-08    2.19e-03        9    40755  |           1.09e-02

Every one of the thirteen improves, several by four to five orders of
magnitude: the eleven EFIT/ZIPFIT features that were already sampled
nearest-sample in the old table land at ~3e-8 median relative error once
windowed the same way as the two PTDATA points, because EFIT's own ~20 ms
cadence lines a 50 ms window up with whole EFIT slices almost every time -
the same effect `validate`'s module docstring records for `kappa` on shot
185945. `ip` and `bt` (the two features the old table already windowed, just
centred rather than ending at `t`) improve by roughly 11x and 14x. This is
NOT twelve of the thirteen becoming bit-identical to the archive: `p90 rel`
stays in the 1e-4 to 2e-2 range, so the tail - the rows where an EFIT/ZIPFIT
slice does not land inside the window, or the offline tree was rerun since
the store was built (see `pres`'s note above) - is not fixed by this
convention change, only the typical row. `n_shots`/`points` are lower here
than the old table's 120-shot scan (a 10-shot re-measurement, not a
120-shot one); the direction and rough magnitude of the improvement is what
matters here, not a fourth-decimal-place match to a larger sample.

`shots` is below 120 for two different reasons, both recorded rather than
worked around:

  * the ARCHIVE lacks the column: `ip` and `bt` are absent on 81 of these
    120 shots, and over the WHOLE store on 2,192 of 5,000 (the columns are
    present on 2,808, 56.2%, against 99.6% for the EFIT columns and 72.8%
    / 74.9% for the ZIPFIT ones). fdp served `ip` and `bt` on all 120, so
    on this pair the fdp path is not merely a fallback - it covers shots
    the archive cannot.
  * fdp could not serve it. Over the same 120 shots: `efit01` unreachable
    on 1 shot (`TreeFOPENR`, 190830) and reduced to a SINGLE time slice on
    1 more (190199, where every aeqdsk and geqdsk node comes back length 1);
    `TROTFIT` absent on 26 and single-sliced on 1 (22.5% of shots between
    them); `EDENSFIT`/`ETEMPFIT` absent on 10 and single-sliced on 1 (9.2%).
    A single-slice record is a `ShortRecord` miss, not a stored feature: a
    one-sample group is the corpus' absent-signal sentinel, so storing one
    would read as absent downstream anyway (see `store.write_features`).

OUTSIDE THE ARCHIVE, which is the point of this module: the corpus has
16,909 shots, 13,663 of them outside the store and 13,288 above its last
shot, 191450. On a RANDOM 5 of those 13,663 (seed 7 - 199200, 199872,
200680, 203594, 204243) fdp served all 13 of its features on three shots, 12
on one (`TROTFIT` absent) and 10 on one (all three ZIPFIT profiles absent).

That is not the model's whole input set. `d3d_tearing_onset_cnn1d` wants 16
canonical features: fdp supplies 12 of them, `resolve_corpus` supplies
`pinj_total`, `tinj_total` and `ech_power_total`, and `ech_rho` - the ECH
deposition location - has NO source but the archive. So a non-archive shot
reaches at best 15 of 16, with the deposition location always absent, which
the adapter's `unknown_when_active` pair turns into an invalid row wherever
ECH power was flowing. On 2 of those 5 shots the corpus' `pinj`/`tinj`
groups were the absent-signal sentinel too, taking the total to 12 or 13 of
16. None of that is this module's to fix: `pinj_total` and `tinj_total` have
no fdp locator by design, and the ECH questions are on hold.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ..timebase import decimate_to_step
from . import namespace as ns
from .store import FeatureArray

SOURCE = "fdp"

EFIT_TREE = "efit01"
ZIPFIT_TREE = "zipfit01"

#: The aeqdsk time node. Only a fallback: every 1-D record measured carries
#: its own `times`. NOT `:time`, which is poloidal flux (see the record).
AEQDSK_TIME = r"\efit01::top.results.aeqdsk:atime"
#: The geqdsk time node. NOT a fallback but the only source of time for the
#: 2-D geqdsk nodes, whose own `times` dim is the radial axis.
GEQDSK_TIME = r"\efit01::top.results.geqdsk:gtime"
#: The psi->rho map, 65 points per time slice. DELIBERATELY UNUSED: applying
#: it disagrees with the model's own training input by 6.9e-2 on qpsi and
#: 2.4e-1 on pres, against 9.7e-3 and 2.3e-2 for leaving the coordinate as
#: normalized psi. See the module docstring.
GEQDSK_RHO = r"\efit01::top.results.geqdsk:rhovn"

MS_PER_S = 1000.0

#: Offset to add to a `resolve_archive` time axis to reach the time the row
#: actually holds. MEASURED, see the module docstring; used by the
#: comparison scripts and the live tests, not by the resolver.
ARCHIVE_LAG_S = -0.025

#: Dim units that mean "this axis is time", and their factor to seconds.
_TIME_SCALE = {"ms": 1.0 / MS_PER_S, "s": 1.0, "sec": 1.0, "secs": 1.0,
               "second": 1.0, "seconds": 1.0}

#: Bound on a recorded miss cause: it is written into an HDF5 attribute and
#: into the JSON `missing` blob (`store.MISSING_ATTR`), both read back on
#: every later run, so an unbounded traceback-length string must not land
#: there. 200 matches `run._guarded`'s existing `str(exc)[:200]` bound on a
#: per-shot error detail.
_CAUSE_MAX_LEN = 200


class ShortRecord(ValueError):
    """A node that came back with fewer than two time samples.

    Its own miss class rather than a bare ValueError, because scanning
    thousands of recorded misses it matters whether the tree held one time
    slice (this) or whether two axes disagreed. MEASURED on a random 120
    overlap shots: `efit01` is reduced to a single slice on 1 of them and
    ZIPFIT nodes on 1-2, so a bulk run meets it.
    """


def _import_diagnosis() -> str | None:
    """None if toksearch imports cleanly; otherwise which module failed, and
    why.

    Checked as two separate imports so the diagnosis names which one broke
    rather than collapsing to "toksearch is unavailable" - that collapse is
    exactly what made this regression (Task 16b) silent. The root cause was
    `import torch` binding the SYSTEM `/lib64/libstdc++.so.6` ahead of the
    pixi env's own copy, so every compiled extension needing a newer GLIBCXX
    symbol - `toksearch_d3d` among them, via `fdp` -> `pyxrootd` - raised a
    bare `ImportError`. `available()` swallowing that into a plain `False`
    meant the recorded miss cause was the string `"ToksearchUnavailable"`
    forever, indistinguishable from an environment that simply lacks the
    package, and a re-run reproduced it identically without ever attempting
    a fetch. See `pyproject.toml`'s `tool.pixi.feature.fdp` activation table
    for the fix to the loader ordering itself; this only makes the next
    occurrence of *any* import failure here self-diagnosing instead of
    requiring the same investigation again.
    """
    try:
        import toksearch  # noqa: F401
    except ImportError as exc:
        return f"toksearch: {type(exc).__name__}: {exc}"
    try:
        import toksearch_d3d  # noqa: F401
    except ImportError as exc:
        return f"toksearch_d3d: {type(exc).__name__}: {exc}"
    return None


def available() -> bool:
    """True when toksearch can be imported (the `labeler`/`fdp` envs).

    A bool, not the diagnosis: kept as the simple gate `resolve()` and any
    other caller can branch on. `resolve()` uses `_import_diagnosis()`
    directly so a miss it records carries the reason, not just this verdict.
    """
    return _import_diagnosis() is None


def _fetch_ptdata(name: str, shot: int) -> dict:
    """One PTDATA point, as toksearch returns it."""
    from toksearch_d3d import PtDataSignal

    return PtDataSignal(name).fetch(int(shot))


def _fetch_mds(expr: str, tree: str, shot: int, dims: Sequence[str] = ()) -> dict:
    """One MDSplus node. `dims` uses toksearch's own dimension mechanism.

    `dim_of(...)` TDI strings do not work through either signal class.
    """
    from toksearch import MdsSignal

    signal = MdsSignal(expr, tree, dims=list(dims)) if dims else MdsSignal(expr, tree)
    return signal.fetch(int(shot))


def _as_f64(values) -> np.ndarray:
    """Fetched values in float64, without the signalling-NaN cast warning.

    MEASURED: shot 199872's `betan` comes back float32 carrying a SIGNALLING
    NaN, and widening one raises the hardware FP invalid flag, which numpy
    reports as `RuntimeWarning: invalid value encountered in cast`. Under
    `-W error` that becomes an exception, and `resolve`'s per-signal `except`
    would have turned a perfectly readable record into a recorded miss - a
    silent data loss, not a crash. `np.errstate` does suppress this one,
    because it comes from the FP flags and not from `warnings` (unlike
    `nanmean`'s "Mean of empty slice", which needs a mask).

    The NaN itself is preserved, quietened; every reduction downstream masks
    on `isfinite` already.
    """
    with np.errstate(invalid="ignore"):
        return np.asarray(values, dtype=np.float64)


def _dims(rec: dict) -> dict[str, tuple[np.ndarray, str]]:
    """The record's non-data entries as `(values, lowercased units)`.

    `rec["units"]` is a dict keyed by `"data"` and by each dim name; a dim
    whose units are absent gets `""`. Entries that are not numeric arrays
    (PTDATA's `n_over`/`n_under` scalars) are kept only if they convert.
    """
    units = rec.get("units")
    units = units if isinstance(units, dict) else {}
    out: dict[str, tuple[np.ndarray, str]] = {}
    for key, value in rec.items():
        if key in ("data", "units"):
            continue
        try:
            arr = _as_f64(value).ravel()
        except (TypeError, ValueError):
            continue
        out[key] = (arr, str(units.get(key, "")).strip().lower())
    return out


def _time_axis_s(rec: dict, n: int) -> np.ndarray | None:
    """The record's own time axis in seconds, or None if it has none.

    Identified by units, so the geqdsk 2-D nodes - whose `times` dim is 65
    points of normalized psi - correctly report that they carry no time.
    """
    for arr, unit in _dims(rec).values():
        if unit in _TIME_SCALE and arr.size == n:
            return arr * _TIME_SCALE[unit]
    return None


def _checked_time(t_s: np.ndarray, locator: str) -> np.ndarray:
    """A time axis that can be binned: finite and strictly increasing."""
    t_s = _as_f64(t_s).ravel()
    if t_s.size < 2:
        raise ShortRecord(f"{locator}: {t_s.size} time samples")
    if not np.isfinite(t_s).all() or not (np.diff(t_s) > 0).all():
        raise ValueError(f"{locator}: non-monotonic or non-finite time axis")
    return t_s


def _scalar_axes(
    rec: dict, locator: str, fallback: Callable[[int], np.ndarray] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """`(data, t_s)` for a 1-D record.

    `fallback(n)` supplies a time axis for a record that carries none. It is
    a callable so that the extra node is fetched only when it is needed -
    every 1-D node measured carries its own `times`.
    """
    data = _as_f64(rec["data"])
    if data.ndim != 1:
        raise ValueError(f"{locator}: expected a 1-D node, got {data.shape}")
    t_s = _time_axis_s(rec, data.size)
    if t_s is None and fallback is not None:
        t_s = fallback(data.size)
    if t_s is None:
        raise ValueError(f"{locator}: no time axis in {sorted(_dims(rec))}")
    if t_s.size != data.size:
        raise ValueError(f"{locator}: {data.size} samples vs {t_s.size} times")
    return data, _checked_time(t_s, locator)


def _profile_axes(
    rec: dict, locator: str, fallback: Callable[[int], np.ndarray] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """`(data (n_t, n_x), t_s, coord, coord_units)` for a 2-D record.

    Both axes are identified from the dims' units and lengths, and the
    orientation of `data` from those lengths - never from position. That is
    what makes the same code safe for zipfit, whose dims come back in the
    transpose of the order they were requested in, and for geqdsk, whose
    only dim is radial.
    """
    data = _as_f64(rec["data"])
    if data.ndim != 2:
        raise ValueError(f"{locator}: expected a 2-D node, got {data.shape}")
    dims = _dims(rec)
    t_key, t_s = None, None
    for key, (arr, unit) in dims.items():
        if unit in _TIME_SCALE and arr.size in data.shape:
            t_key, t_s = key, arr * _TIME_SCALE[unit]
            break
    coord, coord_units = None, ""
    for key, (arr, unit) in dims.items():
        if key != t_key and arr.size in data.shape:
            coord, coord_units = arr, unit
            break
    if coord is None:
        raise ValueError(f"{locator}: no radial axis among {sorted(dims)}")
    if t_s is None and fallback is not None:
        n_t = data.shape[1] if data.shape[0] == coord.size else data.shape[0]
        t_s = fallback(n_t)
    if t_s is None:
        raise ValueError(f"{locator}: no time axis among {sorted(dims)}")
    if data.shape == (coord.size, t_s.size) and data.shape[0] != data.shape[1]:
        data = data.T
    elif data.shape != (t_s.size, coord.size):
        raise ValueError(
            f"{locator}: data {data.shape} matches neither "
            f"(t={t_s.size}, x={coord.size}) nor its transpose"
        )
    return data, _checked_time(t_s, locator), coord, coord_units


def _to_rho_grid(
    values: np.ndarray, coord: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Resample `(n_t, n_x)` profiles onto `grid`, returning `(n_grid, n_t)`.

    `coord` is either one shared axis `(n_x,)` - zipfit's `x`, geqdsk's psi -
    or one axis per time slice `(n_t, n_x)`, which is the shape `rhovn` has.
    It is sorted per slice, so a coordinate that arrives descending or
    slightly out of order is still resampled rather than folded.

    Outside a slice's coordinate range the result is NaN rather than an edge
    value: an extrapolated profile edge is not a measurement.

    The name says rho because that is what `grid` is called; on the geqdsk
    path the coordinate is normalized psi on both sides, measured - see the
    module docstring.
    """
    values = np.atleast_2d(_as_f64(values))
    coord = _as_f64(coord)
    n_t, n_x = values.shape
    if coord.ndim == 1:
        if coord.size != n_x:
            raise ValueError(
                f"coord {coord.shape} does not match values {values.shape}"
            )
        coord = np.broadcast_to(coord, (n_t, n_x))
    elif coord.shape != values.shape:
        raise ValueError(f"coord {coord.shape} does not match values {values.shape}")
    out = np.full((np.size(grid), n_t), np.nan)
    for i in range(n_t):
        x, y = coord[i], values[i]
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum() < 2:
            continue
        x, y = x[good], y[good]
        order = np.argsort(x)
        x, y = x[order], y[order]
        inside = (grid >= x[0]) & (grid <= x[-1])
        out[inside, i] = np.interp(grid[inside], x, y)
    return out


def _series(
    data: np.ndarray, t_s: np.ndarray, spec, locator: str, units: str
) -> FeatureArray:
    """A 1-D record -> a `(1, T)` feature, decimated only if asked.

    A spec with `step == 0.0` keeps the native rate, as the archive resolver
    does. Decimating an EFIT scalar to 1 ms would leave 19 of every 20 bins
    empty and call the result a measurement.
    """
    attrs = {"resolver": SOURCE, "locator": locator, "units_from_source": units}
    y = _as_f64(data)[None, :]
    if spec.step and spec.step > 0.0:
        x, y = decimate_to_step(t_s, y, spec.step)
        attrs["decimated_to_s"] = str(spec.step)
    else:
        x = t_s
    return FeatureArray(x=x, y=y, attrs=attrs)


def _profile(
    data: np.ndarray,
    t_s: np.ndarray,
    coord: np.ndarray,
    coord_units: str,
    spec,
    locator: str,
    units: str,
) -> FeatureArray:
    """A 2-D record -> a `(33, T)` feature on the canonical radial grid."""
    y = _to_rho_grid(data, coord, ns.RHO_GRID)
    return FeatureArray(
        x=t_s,
        y=y,
        attrs={
            "resolver": SOURCE,
            "locator": locator,
            "units_from_source": units,
            "radial_coordinate": coord_units or "unknown",
            "resampled_from": f"{coord.shape[-1]} points of "
                              f"{coord_units or 'unknown'}",
        },
    )


def _data_units(rec: dict) -> str:
    """The record's units for `data`, as the source spells them."""
    units = rec.get("units")
    if isinstance(units, dict):
        return str(units.get("data", "")).strip()
    return str(units or "").strip()


def _resolve_one(spec, locator: str, shot: int, cached) -> FeatureArray:
    """One feature, or an exception the caller records as a miss."""
    if spec.name == "efc_n1_ka":
        components = []
        for name in ("efc_a1_c_ka", "efc_a1_iu_ka", "efc_a1_il_ka"):
            component = ns.by_name(name)
            components.append(cached(
                f"feature:{name}",
                lambda component=component: _resolve_one(
                    component, component.locator_for(SOURCE), shot, cached
                ),
            ))
        t_s = _checked_time(components[0].x, locator)
        if any(not np.array_equal(t_s, c.x) for c in components[1:]):
            raise ValueError(f"{locator}: component clocks disagree")
        values = np.stack([c.y[0] for c in components])
        data = np.max(values, axis=0)
        data[~np.isfinite(values).all(axis=0)] = np.nan
        return _series(data, t_s, spec, locator, "kA")

    if not locator.startswith("\\"):
        rec = _fetch_ptdata(locator, shot)
        data, t_s = _scalar_axes(rec, locator)
        data = _actuator_units(data, spec, _data_units(rec))
        return _series(data, t_s, spec, locator, _data_units(rec))

    # The qualified node owns its tree, including rf/operations/pellet/d3d.
    # Only EFIT has the measured separate aeqdsk/geqdsk time-node fallback.
    tree = locator.lstrip("\\").split("::", 1)[0].lower()
    dims = ("x", "t_ms") if tree == ZIPFIT_TREE else ()
    rec = _fetch_mds(locator, tree, shot, dims=dims)

    def time_node(node: str) -> Callable[[int], np.ndarray]:
        """A time axis from a separate node, fetched once per shot.

        The node's own `data` is the axis, in ms; its length is required to
        match the record it is standing in for, so a tree whose time node
        and profile node disagree is a recorded miss and not a silent
        broadcast.
        """
        def fetch(n: int) -> np.ndarray:
            axis = cached(node, lambda: _fetch_mds(node, EFIT_TREE, shot))
            t_s = _scalar_axes(axis, node)[0] / MS_PER_S
            if t_s.size != n:
                raise ValueError(
                    f"{locator}: {n} samples vs {t_s.size} times from {node}"
                )
            return t_s

        return fetch

    if spec.kind == "scalar":
        node = AEQDSK_TIME if "aeqdsk" in locator.lower() else GEQDSK_TIME
        fallback = time_node(node) if tree == EFIT_TREE else None
        data, t_s = _scalar_axes(rec, locator, fallback=fallback)
        data = _actuator_units(data, spec, _data_units(rec))
        return _series(data, t_s, spec, locator, _data_units(rec))

    fallback = time_node(GEQDSK_TIME) if tree == EFIT_TREE else None
    data, t_s, coord, coord_units = _profile_axes(rec, locator, fallback=fallback)
    return _profile(
        data, t_s, coord, coord_units, spec, locator, _data_units(rec)
    )


def _actuator_units(data: np.ndarray, spec, units: str) -> np.ndarray:
    """Apply only the LC2 conversions established by source unit metadata."""
    if spec.name == "lh_power":
        if units.lower() != "kw":
            raise ValueError(f"{spec.name}: expected kW, got {units!r}")
    elif spec.name in ("ecoil_a", "efc_a1_c_ka", "efc_a1_iu_ka", "efc_a1_il_ka"):
        if units.lower() not in ("a", "amp", "amps"):
            raise ValueError(f"{spec.name}: expected amps, got {units!r}")
        if spec.units == "kA":
            return data / 1000.0
    return data


def resolve(
    shot: int,
    names: Sequence[str],
    *,
    retries: int = 1,
) -> tuple[dict[str, FeatureArray], dict[str, str]]:
    """Fetch the requested canonical features for one shot through fdp.

    Returns `(arrays, missing)`; `missing` maps a feature name to a short
    cause, so one unreachable signal costs a shot that signal and nothing
    else. Asking for a feature with no fdp source is a KeyError, raised
    before any fetch: that is a programming error, not a per-shot gap.
    """
    specs = [ns.by_name(n) for n in names]
    locators = {spec.name: spec.locator_for(SOURCE) for spec in specs}
    diagnosis = _import_diagnosis()
    if diagnosis is not None:
        # "ToksearchUnavailable" is kept as a literal prefix - not just a
        # class name - because `store.TRANSIENT_CAUSES`/`is_transient`
        # matches it as a substring, and that classification (retryable by
        # a plain re-run) is correct and must survive the diagnosis being
        # appended.
        cause = f"ToksearchUnavailable: {diagnosis}"[:_CAUSE_MAX_LEN]
        return {}, dict.fromkeys(names, cause)

    arrays: dict[str, FeatureArray] = {}
    missing: dict[str, str] = {}
    records = {}  # time nodes and resolved components, once per shot

    def cached(key: str, thunk):
        if key not in records:
            records[key] = thunk()
        return records[key]

    for spec in specs:
        locator = locators[spec.name]
        for _ in range(max(retries, 0) + 1):
            try:
                arrays[spec.name] = cached(
                    f"feature:{spec.name}",
                    lambda spec=spec, locator=locator: _resolve_one(
                        spec, locator, shot, cached
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - per-signal isolation
                # fdp raises whatever MDSplus, ptserver or the OSDF route
                # raised, none of it a documented type, and one bad signal
                # must never cost the shot its other features.
                missing[spec.name] = type(exc).__name__
            else:
                missing.pop(spec.name, None)
                break
    # A component requested before its maximum may fail initially and then
    # recover while the maximum resolves. Return every requested recovery.
    for name in names:
        if f"feature:{name}" in records:
            arrays[name] = records[f"feature:{name}"]
            missing.pop(name, None)
    return arrays, missing
