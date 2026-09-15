"""Read raw per-shot signals from the two d3d_fusion_data-layout locations.

A shot's raw data is the union of the read-only staged file
(/scratch/gpfs/EKOLEMEN/d3d_fusion_data/<shot>.h5) and our own fetched file
(<data_root>/raw/<shot>.h5). Both are pandas HDFStore fixed-format files; ours wins on conflict.
Time axes are milliseconds in both. We read with h5py directly: it is fast, needs no pandas
version agreement, and lets us inspect a column without loading a 500 kHz group.

This module is the ONE reader of that layout. shotdb.ignite converts the same files into the
IGNITE input layout and imports the opener, the group reader, the torn-write rule and the
location precedence from here rather than carrying its own copies: the two had already drifted
(one opened with the HDF5 lock disabled and one did not; one read float32 and one float64) before
they were folded together.


Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from ..config import Paths, SignalSpec
from ..schema import Status

# `Signal` is defined in reader.py, where the interface that hands it around is defined, and is
# imported here under its old name because it has always been `legacy_raw.Signal` to every caller
# (features.py, build.py, the tests) and it is the same class either way.
from .reader import ShotFailed, Signal, Unavailable

_log = logging.getLogger(__name__)

# The file-level marker scripts/fetch_shots.py stamps on every file it writes (its SCHEMA). It is
# duplicated here rather than imported because the fetcher runs in the fdp environment and cannot
# import ideate; tests/test_fetch_plan.py pins the two equal.
SCHEMA = "ideate-raw-v1"

# The dtype a column is read as. Every data-bearing column in BOTH stores is float32 on disk: our
# fetcher writes `df.astype(np.float32)` (all 160 blocks of the first 20 raw files checked), and
# the staged d3d_fusion_data files are float32 too except for their (1,1) NaN placeholder groups
# (cer_vb, cer_vb_error and i_dens_fit on 160904 and 161172 are float64 and hold no sample). So a
# float32 read is exact for every real sample and is what the IGNITE processed layout stores
# anyway; float64 would only double the memory of a 500 kHz column. legacy_raw used to read float32
# and ignite float64 -- this is the one decision, made once.
DTYPE = np.float32

# The most one selection in Frame.columns materialises before it is split into per-column
# arrays. 256 MiB covers every registry group in one read (the largest, the staged rmp_current,
# is 90 MiB) while a 500 kHz IGNITE modality is read in slices of a few channels.
_BATCH_BYTES = 256 * 2**20

# Paths we've already warned about in this process. Dozens of signal specs per shot means one bad
# file could otherwise print hundreds of identical lines across a full-database build, burying
# anything new. This is per-process bookkeeping, not a cache of file contents: it is never
# consulted to decide behavior, never cleared when a file is fixed, and holds only paths, never
# data.
_warned_paths: set[Path] = set()


def _warn_once(path: Path, error: OSError | KeyError | RuntimeError) -> None:
    if path in _warned_paths:
        return
    _warned_paths.add(path)
    _log.warning("could not read %s: %s", path, error)


def staged_path(shot: int, paths: Paths) -> Path:
    return paths.staged_raw_dir / f"{shot}.h5"


def our_path(shot: int, paths: Paths) -> Path:
    return paths.raw_dir / f"{shot}.h5"


def source_paths(shot: int, paths: Paths) -> list[Path]:
    """Where to look for `shot`, in precedence order: our fetched file, then the staged copy --
    only the ones that exist. Every reader of the raw store (read_signal, signal_status,
    read_shot) takes its locations from this one list, so a shot present
    in both files reads identically everywhere."""
    return [p for p in (our_path(shot, paths), staged_path(shot, paths)) if p.exists()]


def open_h5(path: Path) -> h5py.File:
    """Open read-only WITHOUT taking an HDF5 file lock.

    scripts/fetch_shots.py holds a write lock on a file in raw_dir for as long as it is writing
    that shot, and a plain `h5py.File(path, "r")` against a locked file raises BlockingIOError
    instead of reading it -- measured while a 198-shot fetch was in flight. Before this opener was
    shared, legacy_raw still opened with the lock: BlockingIOError is an OSError, `_warn_once`
    swallowed it, and `ideate build` during a fetch silently built the shot with zero signals. We
    only ever read here, and the torn-write hazard the lock would guard against is covered by the
    fetcher's `complete = True` attr, stamped last and checked by `_incomplete` below.
    """
    return h5py.File(path, "r", locking=False)


def _decode(items) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in items]


def _attr(g: h5py.Group | h5py.File, key: str):
    v = g.attrs.get(key)
    if v is None or isinstance(v, h5py.Empty):
        return None
    return v.decode() if isinstance(v, bytes) else v


def is_ours(f: h5py.File, g: h5py.Group | None = None) -> bool:
    """Was this file -- or, failing a file-level answer, this group of it -- written by
    scripts/fetch_shots.py?

    Decided from the markers the fetcher itself writes, never from the directory the file was
    found in: `stamp_file` puts schema="ideate-raw-v1" on the file, and `write_group` puts
    source="toksearch", fetcher_version and (last) complete on every group. A fetched file copied
    into the staged directory, or a d3d_fusion_data copy placed under raw_dir, therefore reads
    exactly as it would in its home location. The staged files carry none of these: their root
    attrs are PyTables' CLASS/PYTABLES_FORMAT_VERSION/TITLE/VERSION and their group attrs pandas'
    own plus units/missing_channels/sampling_frequency_kHz/start_time_ms/end_time_ms/
    r_coordinates/z_coordinates (checked on 160904 and 161172). All 200 of our raw files carry
    the schema attr and all 3,690 of their groups carry all three group markers (2026-09-04).

    The group-level markers matter for one case only: a group torn between `store.put` and the
    attr stamps in a file the fetcher has not stamped yet. The fetcher now stamps the file before
    its first group, so that case is closed going forward; the group check keeps files written
    by the older stamp-last order honest.
    """
    if _attr(f, "schema") == SCHEMA:
        return True
    if g is None:
        return False
    return _attr(g, "source") == "toksearch" or any(
        k in g.attrs for k in ("fetcher_version", "complete")
    )


def _incomplete(f: h5py.File, g: h5py.Group) -> bool:
    """Is this group of *our* file a write that never finished?

    scripts/fetch_shots.py replaces a group wholesale -- `store.remove(group)` then `store.put`
    then the attrs -- and stamps `complete = True` LAST, precisely so that an interrupted or
    failed write is identifiable afterwards: `fetch_shot` also swallows a per-group exception and
    moves on, deliberately leaving the group without the attr so the next `--plan` lists it
    again. Either way the group's frame may be absent, truncated or half-rewritten, so the only
    safe reading of one of our groups with no `complete = True` is "not there yet" -- fall
    through to the staged copy rather than hand back samples from a frame that is still being
    written. This matters concretely: a bulk fetch writes into raw_dir while databases are built
    from the same paths.

    Only for groups `is_ours` claims. The staged d3d_fusion_data files come from another producer
    entirely and carry no `complete` attribute at all (checked on all 936 of them), so requiring
    it there would read every staged shot as empty.
    """
    if not is_ours(f, g):
        return False
    # `complete = True` goes in through PyTables and comes back out through h5py as
    # numpy.uint8(1), not Python's True (HDF5 has no bool type; PyTables writes uint8 plus a
    # type-hint attr, and h5py does not consult that hint). Measured on a live fetched file. So
    # this is a truthiness test, never an `is True` identity test, which would read every
    # correctly-finished group in raw_dir as mid-write.
    return not bool(_attr(g, "complete"))


def _missing(g: h5py.Group) -> set[str]:
    # Our own fetcher writes missing_channels as an array of individual names, e.g.
    # np.array([b"eclukfpwrc"]). The real staged files instead write one comma-joined bytes
    # scalar (confirmed on d3d_fusion_data/161172.h5's q_psi group: b"q0,q95,qmin") -- splitting
    # every decoded entry on "," covers both conventions; it is a no-op for names with no comma.
    v = g.attrs.get("missing_channels")
    if v is None or isinstance(v, h5py.Empty):
        return set()
    names = (part.strip() for entry in _decode(np.atleast_1d(v)) for part in entry.split(","))
    return {n for n in names if n}


@dataclass
class Frame:
    """One group of a raw file, decoded once: its time axis and, per block, the column names and
    the values dataset. `column()` slices a single column out of the block dataset rather than
    reading the block, so a 500 kHz 64-channel group costs one channel of memory, not 1.3 GB.

    The dataset handles belong to the open file; use a Frame only inside the `with open_h5(...)`
    that produced it. `t` is None for a placeholder group -- no `axis1`, or fewer than two samples
    (the (1,1) NaN frame the staged producer writes for a modality it did not record) -- and
    `column()` then answers None for every name, while `missing` still says what the group's own
    missing_channels attr recorded.
    """

    t: np.ndarray | None
    blocks: list[tuple[list[str], h5py.Dataset]]
    ours: bool
    missing: set[str]
    units: str | None

    def column(self, col: str) -> np.ndarray | None:
        """`col` on `t` as float32, or None when the group does not carry it, its block is not
        the (n_time, n_cols) / (n_cols, n_time) shape, or it holds no finite sample."""
        return self.columns([col]).get(col)

    def columns(self, cols: list[str]) -> dict[str, np.ndarray]:
        """The subset of `cols` this group carries with at least one finite sample, each as a
        contiguous float32 array on `t`.

        The wanted columns of one block are read in as few selections as possible rather than
        one per column. Both producers write a block as one contiguous (n_time, n_cols) dataset
        (chunks=None), so a single column is a strided read across the whole block: on the staged
        160904's rmp_current (1,311,173 x 18, 90 MiB) eighteen `vals[:, j]` reads take 538 ms and
        one `vals[:, [j0, ..., j17]]` takes 53 ms, holding only the eighteen columns and not the
        block; p_inj (8 columns, 40 MiB) is 152 ms against 24 ms. The selection is capped at
        _BATCH_BYTES so a 500 kHz spectrogram group (ece is 3.1 M samples x 40 channels, ~500 MB)
        is read a few columns at a time instead of all at once.
        """
        out: dict[str, np.ndarray] = {}
        if self.t is None or not cols:
            return out
        wanted = list(dict.fromkeys(cols))
        for items, vals in self.blocks:
            here = [c for c in wanted if c in items]
            if not here:
                continue
            wanted = [c for c in wanted if c not in here]  # a name lives in exactly one block
            if vals.ndim != 2:
                continue
            if vals.shape[0] == self.t.size:
                time_first = True
            elif vals.shape[1] == self.t.size:
                time_first = False
            else:
                continue
            js = sorted(items.index(c) for c in here)  # h5py wants an increasing selection
            per_read = max(1, _BATCH_BYTES // (self.t.size * vals.dtype.itemsize))
            for start in range(0, len(js), per_read):
                batch = js[start : start + per_read]
                sel = vals[:, batch] if time_first else vals[batch, :].T
                sel = np.asarray(sel, dtype=DTYPE)
                for i, j in enumerate(batch):
                    y = np.ascontiguousarray(sel[:, i])
                    if np.isfinite(y).any():
                        out[items[j]] = y
            if not wanted:
                break
        return out


def frame(f: h5py.File, group: str) -> Frame | None:
    """`group` of the open file `f`, decoded, or None when the group is absent or is a torn
    write of ours (see `_incomplete`). Raises h5py's OSError/KeyError/RuntimeError on a corrupt
    file exactly like any other access would; callers wrap one group at a time (see _read_specs
    for why the wrap has to cover the containment check too)."""
    if group not in f:
        return None
    g = f[group]
    if _incomplete(f, g):
        return None
    t = None
    if "axis1" in g:
        t = np.asarray(g["axis1"][()], dtype=np.float64)
        if t.ndim != 1 or t.size < 2:
            t = None
    blocks: list[tuple[list[str], h5py.Dataset]] = []
    if t is not None:
        for k in range(int(g.attrs.get("nblocks", 1))):
            items_key, vals_key = f"block{k}_items", f"block{k}_values"
            if items_key in g and vals_key in g:
                blocks.append((_decode(g[items_key][()]), g[vals_key]))
    units = _attr(g, "units")
    return Frame(t, blocks, is_ours(f, g), _missing(g), str(units) if units is not None else None)


def _signal(fr: Frame, y: np.ndarray, spec: SignalSpec) -> Signal:
    # Order matters. null_value is one or more literal constants EFIT/PTDATA write verbatim (e.g.
    # zxpt1's -9.99 "no X-point found" marker, or drsep's symmetric +-0.4 m saturation as a list
    # of two), never a computed result, so an exact equality comparison against each one is the
    # right test -- and it must run against the raw stored sample, before scaling, because
    # scaling would move the sample away from the constant(s) the registry declares and the
    # sentinel would no longer match. scale runs next, converting the column into the unit the
    # registry declares (e.g. ip's stored megaamps -> declared amps). abs continues to run last,
    # against the scaled value, exactly as it already did before scale/null_value existed.
    if spec.null_value is not None:
        nulls = spec.null_value if isinstance(spec.null_value, list) else [spec.null_value]
        for nv in nulls:
            y = np.where(y == nv, np.nan, y)
    if spec.scale != 1.0:
        y = y * spec.scale
    if spec.abs:
        y = np.abs(y)
    return Signal(fr.t, y, fr.units, "fetched" if fr.ours else "staged", spec.group, spec.col)


def _read_specs(
    shot: int, specs: list[SignalSpec], paths: Paths
) -> tuple[dict[str, Signal | None], set[str]]:
    """Every spec's signal from the first location that has it, in ONE pass over the files: each
    location is opened once and each group's time axis and column names are decoded once, however
    many specs share the group (rmp_current is 18 of them). Also returns the names OUR file
    settles as unavailable -- a finished group of ours lists the column in missing_channels --
    which only matters for a spec no location could supply.

    Every failure point falls back to the next location the same way. A file whose bytes are
    damaged without changing its size can open and list its groups fine, then fail once something
    is actually read or looked up, in one of three shapes depending on exactly what got damaged
    (all three confirmed against this project's h5py, see tests/test_legacy_raw.py):

    - OSError: the file fails to open at all -- garbage bytes, a truncated file, or damage severe
      enough to corrupt its own superblock -- or, once open, a damaged dataset's raw chunk
      data/B-tree fails the instant that dataset is actually read ("Can't synchronously read data
      (wrong B-tree signature)").
    - KeyError: corruption reaches a dataset's own object header. Opening a member by name has to
      read its object header to hand back the object, and a header h5py cannot parse looks to its
      name-lookup code exactly like a name that failed to resolve, so it raises KeyError the same
      way a missing key would ("Unable to synchronously open object (bad object header version
      number)").
    - RuntimeError: h5py's error table maps known HDF5 error codes to specific Python exceptions
      and falls back to RuntimeError for any code it has no specific mapping for. Confirmed to
      fire reliably when corruption reaches a *group's* own link/symbol-table structures -- shared
      plumbing that a plain containment check (`group not in f`) must consult to answer "does this
      name exist" ("Unable to synchronously check link existence (bad symbol table node
      signature)"). Because this is a generic fallback rather than a specific mapping, it is not
      tied to that one call site or one message: the same damage class can surface RuntimeError
      from other HDF5 calls too, at other corruption offsets, with other messages -- treat it as
      "h5py had no better exception for this", not as "a link-existence check failed".

    So the open is one try, and each group's decode-and-read is another, catching all three
    types, and a group that fails leaves the others of the same file readable. Guarding only the
    open call, or only some of these types, (as earlier versions of this module did) lets an error
    propagate straight out and abort every other signal in the same shot's extraction, including
    ones that live entirely in the other location. The tuple stops at these three: TypeError and
    ValueError are deliberately left uncaught, because they are far more likely to mean a bug in
    our own code -- a wrong argument, a bad type -- than damage in someone else's file, and a
    handler that swallows our own bugs would be as harmful as one that lets corruption through.
    """
    by_group: dict[str, list[SignalSpec]] = {}
    for s in specs:
        by_group.setdefault(s.group, []).append(s)
    signals: dict[str, Signal | None] = {s.name: None for s in specs}
    settled: set[str] = set()
    for path in source_paths(shot, paths):
        try:
            f = open_h5(path)
        except (OSError, KeyError, RuntimeError) as e:
            _warn_once(path, e)
            continue
        with f:
            for group, group_specs in by_group.items():
                todo = [s for s in group_specs if signals[s.name] is None]
                if not todo:
                    continue
                try:
                    fr = frame(f, group)
                    if fr is None:
                        continue
                    got = fr.columns([s.col for s in todo])
                    for s in todo:
                        y = got.get(s.col)
                        if y is not None:
                            signals[s.name] = _signal(fr, y, s)
                        elif fr.ours and s.col in fr.missing:
                            # Only OUR fetcher's missing_channels means "DIII-D has no data": it
                            # is written after an attempt against the real archive. A staged
                            # file's missing_channels says only that the staged producer did not
                            # record the column, which is a different claim -- every
                            # d3d_fusion_data file lists "q0,q95,qmin" (verified on 160904,
                            # 161172, 163119) yet all three fetch cleanly from EFIT01, which is
                            # exactly what the fetch plan does for the breadth shots. Leaving
                            # those `pending` is what lets `ideate fetch` back-fill them instead
                            # of writing them off forever.
                            settled.add(s.name)
                except (OSError, KeyError, RuntimeError) as e:
                    _warn_once(path, e)
    return signals, settled


def read_shot(
    shot: int, specs: list[SignalSpec], paths: Paths
) -> tuple[dict[str, Signal | None], dict[str, Status]]:
    """What build_record needs for one shot, in one pass over its files: every installed spec's
    signal (None for a spec not installed on this shot, which is never read) and every spec's
    coverage status.

    Presence is derived from the read itself -- a signal that came back is `present` -- so the
    two dicts cannot disagree, and the files are not opened a second time to say so. Before this
    the build read every column once for the values and once more for the status: measured on
    shot 160904 (78 specs over 19 groups), 1073 ms for the values and 883 ms for the statuses, 36
    opens of rmp_current alone for its 18 coils.
    """
    installed = [s for s in specs if s.installed]
    found, settled = _read_specs(shot, installed, paths)
    signals: dict[str, Signal | None] = {}
    coverage: dict[str, Status] = {}
    for s in specs:
        sig = found.get(s.name)
        signals[s.name] = sig
        if not s.installed:
            coverage[s.name] = "not_installed"
        elif sig is not None:
            coverage[s.name] = "present"
        elif s.name in settled or s.fetch is None:
            coverage[s.name] = "unavailable"
        else:
            coverage[s.name] = "pending"
    return signals, coverage


def read_signal(shot: int, spec: SignalSpec, paths: Paths) -> Signal | None:
    return _read_specs(shot, [spec], paths)[0][spec.name]


def signal_status(shot: int, spec: SignalSpec, paths: Paths) -> Status:
    return read_shot(shot, [spec], paths)[1][spec.name]


def pending_specs(shot: int, specs: list[SignalSpec], paths: Paths) -> list[SignalSpec]:
    coverage = read_shot(shot, specs, paths)[1]
    return [s for s in specs if coverage[s.name] == "pending"]


def list_groups(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with open_h5(path) as f:
        for name, g in f.items():
            cols: list[str] = []
            for k in range(int(g.attrs.get("nblocks", 1))):
                if f"block{k}_items" in g:
                    cols += _decode(g[f"block{k}_items"][()])
            out[name] = cols
    return out


class LegacyReader:
    """This module as a `SignalReader`: the same functions with `paths` bound, plus the file-level
    half of the protocol.

    An adapter and nothing more. Every spec-level answer is the module function's answer, byte
    for byte -- the point is that `build` can be handed a corpus reader instead without a single
    behaviour of the legacy path changing meanwhile. The file-level half (`groups`, `read`,
    `coverage`) is new API, not a rename of anything, and is written on top of `frame()` so the
    two locations, the torn-write rule and the precedence order are the ones this module already
    applies everywhere else.

    Where the protocol says "raise", this class raises: `Unavailable` when no location carries
    the group, `ShotFailed` when no location could be read at all. That is a different discipline
    from `read_signal`, which warns once and returns None, and it is deliberate -- the spec-level
    half answers for a registry of dozens of signals at once, where one damaged file must not
    abort the other thirty, while the file-level half answers about one named group and has no
    other way to say "the file is broken" than to say it.
    """

    def __init__(self, paths: Paths):
        self.paths = paths

    def __repr__(self) -> str:
        return f"LegacyReader({str(self.paths.raw_dir)!r})"

    # ---------------------------------------------------------------------- the spec level

    def read_shot(
        self, shot: int, specs: list[SignalSpec]
    ) -> tuple[dict[str, Signal | None], dict[str, Status]]:
        return read_shot(shot, specs, self.paths)

    def read_signal(self, shot: int, spec: SignalSpec) -> Signal | None:
        return read_signal(shot, spec, self.paths)

    def signal_status(self, shot: int, spec: SignalSpec) -> Status:
        return signal_status(shot, spec, self.paths)

    # ---------------------------------------------------------------------- the file level

    def path(self, shot: int) -> Path:
        """The location this shot reads from: ours if it exists, else the staged copy, else
        where ours would be written."""
        found = source_paths(shot, self.paths)
        return found[0] if found else our_path(shot, self.paths)

    def available(self, shot: int) -> bool:
        for p in source_paths(shot, self.paths):
            try:
                with open_h5(p):
                    return True
            except (OSError, KeyError, RuntimeError) as e:
                _warn_once(p, e)
        return False

    def groups(self, shot: int) -> list[str]:
        """Every group either location carries with a real time axis, sorted. A placeholder group
        -- the (1,1) NaN frame the staged producer writes for a modality it did not record -- has
        no usable time axis and is not one this shot has."""
        found, error = set(), None
        for p in self._locations(shot):
            try:
                with open_h5(p) as f:
                    for name in f:
                        fr = frame(f, name)
                        if fr is not None and fr.t is not None:
                            found.add(name)
            except (OSError, KeyError, RuntimeError) as e:
                error = e
        if not found and error is not None:
            raise ShotFailed(f"shot {shot}: {error}") from error
        return sorted(found)

    def read(
        self, shot: int, group: str, channels: Sequence[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """`group` as `(t_ms float64 (n,), y float32 (C, n))`.

        `channels` index into the group's own column order (the order `list_groups` reports),
        which is what the protocol's unnamed-channel vocabulary means here. A column the group
        carries but that holds no finite sample comes back as a row of NaN rather than being
        dropped, so the returned row order always matches the indices asked for.
        """
        return self._one_group(shot, group, lambda fr: self._rows(fr, group, channels))

    def coverage(self, shot: int, group: str) -> tuple[float, float]:
        return self._one_group(shot, group, lambda fr: (float(fr.t[0]), float(fr.t[-1])))

    # ---------------------------------------------------------------------- internals

    def _locations(self, shot: int) -> list[Path]:
        found = source_paths(shot, self.paths)
        if not found:
            raise ShotFailed(
                f"shot {shot}: no file in {self.paths.raw_dir} or {self.paths.staged_raw_dir}"
            )
        return found

    def _one_group(self, shot: int, group: str, fn):
        """`fn` applied to `group`'s frame at the first location that has it, ours first.

        A location that fails to open or read does not end the search -- that is the same
        fall-through `_read_specs` does, and it is what keeps a torn file of ours from hiding the
        staged copy -- but if no location ends up answering and one of them failed, the failure
        is what the caller hears about, not "not recorded".
        """
        error = None
        for p in self._locations(shot):
            try:
                with open_h5(p) as f:
                    fr = frame(f, group)
                    if fr is None or fr.t is None:
                        continue
                    return fn(fr)
            except (OSError, KeyError, RuntimeError) as e:
                _warn_once(p, e)
                error = e
        if error is not None:
            raise ShotFailed(f"shot {shot}: {group}: {error}") from error
        raise Unavailable(f"shot {shot} has no group {group!r}")

    @staticmethod
    def _rows(fr: Frame, group: str, channels: Sequence[int] | None) -> tuple:
        names = [c for items, _ in fr.blocks for c in items]
        if channels is None:
            idx = list(range(len(names)))
        else:
            idx = [int(c) for c in channels]
            bad = [c for c in idx if not 0 <= c < len(names)]
            if bad:
                raise IndexError(f"{group} has {len(names)} columns, asked for {bad}")
        want = [names[i] for i in idx]
        got = fr.columns(want)
        n = fr.t.size
        y = np.empty((len(want), n), dtype=DTYPE)
        for i, name in enumerate(want):
            col = got.get(name)
            y[i] = np.full(n, np.nan, DTYPE) if col is None else col
        return np.asarray(fr.t, dtype=np.float64), y
