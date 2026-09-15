"""Ported from shot-recommender-system (shotrec) @565d548."""

# tests/test_legacy_raw.py
import logging

import h5py
import numpy as np
import pandas as pd
import pytest

from ideate import config
from ideate.shotdb import legacy_raw

from .conftest import stamp_ours, write_frame


def spec(name="ip", group="ip", col="ipsip", **kw):
    # setdefault, not a plain kwarg: a literal `fetch={...}, **kw` collides ("multiple values
    # for keyword argument 'fetch'") whenever a caller overrides fetch explicitly, which
    # test_statuses does below (fetch=None, for a signal with no fetch method at all).
    kw.setdefault("fetch", {"kind": "ptdata", "node": col})
    return config.SignalSpec(name=name, group=group, col=col, **kw)


def test_reads_staged_and_fetched_identically(paths, staged_shot_a, our_shot_c):
    # scale=1.0e6: ipsip is stored in megaamps (see signals.yaml's ip.scale comment); the
    # fixtures model that real on-disk convention, so reading it back into amps needs the same
    # scale the production registry declares.
    a = legacy_raw.read_signal(staged_shot_a, spec(abs=True, scale=1.0e6), paths)
    c = legacy_raw.read_signal(our_shot_c, spec(abs=True, scale=1.0e6), paths)
    assert a.source == "staged" and c.source == "fetched"
    assert a.t_ms.dtype == np.float64 and a.y.dtype == np.float32
    assert a.t_ms.size == a.y.size == 7000
    assert abs(float(a.y[(a.t_ms >= 1000) & (a.t_ms <= 4000)].mean()) - 1.2e6) < 1.0
    assert c.units == "A" and a.units is None


def test_placeholder_and_missing_group_are_none(paths, staged_shot_b):
    assert legacy_raw.read_signal(staged_shot_b, spec("pech_LEIA", "ech", "ecleifpwrc"), paths) is None
    assert legacy_raw.read_signal(staged_shot_b, spec("gas_GASA", "gas", "gasa"), paths) is None


def test_two_block_frame(paths, staged_shot_b):
    betap = legacy_raw.read_signal(staged_shot_b, spec("betap", "beta", "betap"), paths)
    assert betap is not None and abs(float(betap.y[0]) - 0.7) < 1e-6


def test_statuses(paths, our_shot_c):
    assert legacy_raw.signal_status(our_shot_c, spec(), paths) == "present"
    assert (
        legacy_raw.signal_status(our_shot_c, spec("pech_LUKE", "ech", "eclukfpwrc"), paths)
        == "unavailable"
    )
    assert legacy_raw.signal_status(our_shot_c, spec("gas_GASA", "gas", "gasa"), paths) == "pending"
    assert (
        legacy_raw.signal_status(
            our_shot_c, spec("ne0", "e_dens_fit", "edensfit0.00", fetch=None), paths
        )
        == "unavailable"
    )
    assert (
        legacy_raw.signal_status(
            our_shot_c, spec("pech_BORIS", "ech", "ecborfpwrc", installed=False), paths
        )
        == "not_installed"
    )
    assert [
        s.name
        for s in legacy_raw.pending_specs(our_shot_c, [spec(), spec("gas_GASA", "gas", "gasa")], paths)
    ] == ["gas_GASA"]


def test_list_groups(paths, staged_shot_a):
    groups = legacy_raw.list_groups(legacy_raw.staged_path(staged_shot_a, paths))
    assert groups["ip"] == ["ipsip", "iptipp"] and "pinjf_15l" in groups["p_inj"]


# --- Supplementary tests beyond the brief's literal 5 -----------------------------------
#
# Both probes below are motivated by inspecting the real staged file
# (/scratch/gpfs/EKOLEMEN/d3d_fusion_data/161172.h5) during implementation; see
# task-6-report.md for the full comparison against the brief's layout description.


def test_transposed_block_values_are_read_correctly(paths):
    """The brief's _read_col accepts block{k}_values in either (n_time, n_cols) or
    (n_cols, n_time) orientation. In practice neither the real staged file (written by pandas
    0.15.2) nor our own fixtures (written by pandas 3.0.5) ever produce the (n_cols, n_time)
    case -- both write (n_time, n_cols) for every group sampled, single- and multi-block alike.
    So that branch is never exercised by pandas-written fixtures; author one group directly
    with h5py to prove the transposition logic itself is correct, not just defensively present.
    """
    shot, path = 900006, paths.raw_dir / "900006.h5"
    t = np.arange(0.0, 10.0, 1.0)
    with h5py.File(path, "w") as f:
        g = f.create_group("ip")
        g.attrs["nblocks"] = 1
        g.attrs["complete"] = True  # raw_dir file: our fetcher's finished-write marker
        g.create_dataset("axis1", data=t)
        g.create_dataset("block0_items", data=np.array([b"ipsip", b"iptipp"]))
        # (n_cols, n_time) = (2, 10): the transposed orientation, opposite of what pandas writes.
        g.create_dataset(
            "block0_values",
            data=np.array([np.arange(0.0, 10.0), np.arange(100.0, 110.0)], dtype=np.float32),
        )
    a = legacy_raw.read_signal(shot, spec("ipsip", "ip", "ipsip"), paths)
    b = legacy_raw.read_signal(shot, spec("iptipp", "ip", "iptipp"), paths)
    np.testing.assert_allclose(a.y, np.arange(0.0, 10.0))
    np.testing.assert_allclose(b.y, np.arange(100.0, 110.0))


def test_missing_channels_comma_joined_string_is_split(paths):
    """Our own fetcher (Task 11) records missing_channels as an array of names, e.g.
    np.array([b"eclukfpwrc"]) -- what our_shot_c uses. The real staged files instead record it
    as a single comma-joined bytes scalar: q_psi on shot 161172 carries
    missing_channels == b"q0,q95,qmin", not three separate array entries. signal_status must
    recognize "q95" inside that joined string, not treat the whole string as one (unmatchable)
    channel name.

    Which status that produces depends on WHICH file listed the column, and the two claims are
    different. Our fetcher writes missing_channels after actually asking the archive, so it means
    "DIII-D has no data" -> unavailable. A staged file's only means the staged producer did not
    record the column; q0/q95/qmin are listed on every real d3d_fusion_data file yet all three
    fetch cleanly from EFIT01, so from a staged file the honest answer is `pending` -- there is
    something left for `ideate fetch` to do. Both directions are pinned below.
    """
    te = np.arange(100.0, 5800.0, 25.0)
    df = pd.DataFrame({"qpsi0.00": np.full(te.size, 9.4, dtype=np.float32)}, index=pd.Index(te))

    staged_shot, staged_file = 900005, paths.staged_raw_dir / "900005.h5"
    with pd.HDFStore(staged_file, mode="a") as store:
        store.put("q_psi", df, format="fixed")
        store.get_storer("q_psi").attrs.missing_channels = b"q0,q95,qmin"
    # split happened (else "q95" would not match the joined scalar at all) and, being staged,
    # resolves to pending rather than being written off:
    assert legacy_raw.signal_status(staged_shot, spec("q0", "q_psi", "q0"), paths) == "pending"
    assert legacy_raw.signal_status(staged_shot, spec("q95", "q_psi", "q95"), paths) == "pending"
    assert (
        legacy_raw.signal_status(staged_shot, spec("qpsi0", "q_psi", "qpsi0.00"), paths) == "present"
    )
    # ... and a column with no fetch method at all stays unavailable, since nothing can back-fill it
    no_fetch = spec("q95", "q_psi", "q95", fetch=None)
    assert legacy_raw.signal_status(staged_shot, no_fetch, paths) == "unavailable"

    our_shot, our_file = 900006, paths.raw_dir / "900006.h5"
    with pd.HDFStore(our_file, mode="a") as store:
        store.put("q_psi", df, format="fixed")
        st = store.get_storer("q_psi")
        st.attrs.missing_channels = b"q0,q95,qmin"
        st.attrs.source = "toksearch"
        st.attrs.complete = True
    assert legacy_raw.signal_status(our_shot, spec("q95", "q_psi", "q95"), paths) == "unavailable"
    assert legacy_raw.signal_status(our_shot, spec("qpsi0", "q_psi", "qpsi0.00"), paths) == "present"


# --- Fix wave 1: unreadable files, our-file precedence, a meaningful abs test --------------


def test_unreadable_our_file_falls_back_to_staged(paths, staged_shot_a, caplog):
    """A corrupt/truncated file at the 'ours' location (~292 of ~4,014 real archive files fail
    to open at all) must not abort the whole shot -- read_signal should still find the signal in
    the other location, and signal_status must still report it present."""
    bad = legacy_raw.our_path(staged_shot_a, paths)
    bad.write_bytes(b"not an hdf5 file")
    with caplog.at_level(logging.WARNING):
        # scale=1.0e6: ipsip is megaamps on disk (see signals.yaml's ip.scale comment).
        sig = legacy_raw.read_signal(staged_shot_a, spec(scale=1.0e6), paths)
        status = legacy_raw.signal_status(staged_shot_a, spec(), paths)
    assert sig is not None and sig.source == "staged"
    assert abs(float(sig.y[(sig.t_ms >= 1000) & (sig.t_ms <= 4000)].mean()) - 1.2e6) < 1.0
    assert status == "present"
    assert str(bad) in caplog.text


def test_only_unreadable_file_is_pending_not_none(paths, caplog):
    """When the only file for a shot is unreadable, that is 'we tried to fetch this and failed',
    not 'DIII-D has no such data' -- read_signal must return None (not raise) and signal_status
    must report pending (refetch is the right remedy), never present or unavailable."""
    shot = 900008
    bad = legacy_raw.our_path(shot, paths)
    bad.write_bytes(b"not an hdf5 file")
    with caplog.at_level(logging.WARNING):
        sig = legacy_raw.read_signal(shot, spec(), paths)
        status = legacy_raw.signal_status(shot, spec(), paths)
    assert sig is None
    assert status == "pending"
    assert str(bad) in caplog.text


def test_our_file_wins_a_real_conflict(paths, dual_shot_d):
    """staged_shot_a/staged_shot_b/our_shot_c each put their shot in exactly one location, so
    none of them can catch a regression that swapped or dropped the our-file-wins precedence.
    dual_shot_d puts ip/ipsip in both locations with different values; assert on the array
    values themselves (not just .source) so this fails if the wrong file's data comes back even
    under a correctly-labeled source."""
    # scale=1.0e6: ipsip is megaamps on disk (see signals.yaml's ip.scale comment); dual_shot_d's
    # fixture values are 1 MA (staged) / 2 MA (fetched).
    sig = legacy_raw.read_signal(dual_shot_d, spec(scale=1.0e6), paths)
    assert sig.source == "fetched"
    np.testing.assert_allclose(sig.y, 2.0e6)


def test_falls_through_to_staged_when_our_file_lacks_the_column(paths, dual_shot_d):
    """dual_shot_d's our-file 'gas' group has gasb but not gasa; the staged copy has gasa.
    Finding the group but not the column must fall through to the other location rather than
    returning None."""
    sig = legacy_raw.read_signal(dual_shot_d, spec("gas_GASA", "gas", "gasa"), paths)
    assert sig.source == "staged"
    np.testing.assert_allclose(sig.y, 5.0)


def test_abs_flag_flips_a_negative_signal_positive(paths):
    """Plasma current is recorded in either sign convention depending on campaign (co- vs.
    counter-current relative to Bt), so SignalSpec.abs exists to compare magnitudes regardless of
    which convention a given shot used. Every Ip fixture elsewhere in this suite is already
    non-negative, so abs=True there is a no-op that would not catch a broken np.abs."""
    shot, path = 900007, paths.staged_raw_dir / "900007.h5"
    t = np.arange(0.0, 10.0, 1.0)
    df = pd.DataFrame({"ipsip": np.full(t.size, -1.2e6, dtype=np.float32)}, index=pd.Index(t))
    df.to_hdf(path, key="ip", mode="a", format="fixed")
    sig = legacy_raw.read_signal(shot, spec(abs=True), paths)
    assert sig is not None
    assert np.all(sig.y > 0)
    np.testing.assert_allclose(sig.y, 1.2e6)


# --- Fix wave 2: corruption discovered while reading, and warn-once-per-path --------------


# Three corruption bands, all produced by the same construction below (_write_corrupt): a group
# with a chunked block0_values, built directly with h5py rather than through write_frame/to_hdf,
# because pandas' "fixed" format writes contiguous (unchunked) datasets in this environment --
# h5py.Dataset.chunks is None on every column checked -- and contiguous storage has no B-tree to
# corrupt in the first place. Real large DIII-D signals are chunked (this module's own docstring
# notes "without loading a 500 kHz group"); this reproduces that shape, then flips every bit
# (`^= 0xFF`) in the file's trailing `1 - frac` fraction, in place, without changing its length.
#
# Empirically swept against this project's pinned h5py (3.16.0/HDF5 2.0.0) across fresh,
# byte-identical builds (no randomness in the construction, so repeat builds always land in the
# same band) -- not a guarantee for other h5py/HDF5 versions, only a record of what this exact
# pinned version does with this exact fixture shape:
#
#   frac=0.7  (trailing ~30-45% corrupted): lands in block0_values' raw chunk data/B-tree.
#             OSError "Can't synchronously read data (wrong B-tree signature)" the instant that
#             dataset is actually read; the open, the group listing, and the axis1/block0_items
#             reads all still succeed untouched. Used by the read-time-corruption tests below.
#   frac=0.3  (trailing ~54-98% corrupted): lands in the *objects'* own headers -- reached via
#             g["block0_items"], which _read_col opens one line before block0_values, so that is
#             the call site this band actually exercises. KeyError "Unable to synchronously open
#             object (bad object header version number)": opening a member by name has to read
#             its object header, and a header h5py can't parse looks to its name-lookup code
#             exactly like a name that failed to resolve. The *link* itself still resolves
#             ("block0_items" in g is True); only opening the object fails. Used by the
#             object-header-corruption tests below.
#   frac=0.008 (trailing ~99.2% corrupted, stable across roughly 98.8%-99.4%): lands in the
#             *group's own* link/symbol-table structures -- shared plumbing a plain containment
#             check ("x in group") must consult, rather than any one dataset's storage. Raises
#             RuntimeError "Unable to synchronously check link existence (bad symbol table node
#             signature)" from the containment check itself -- confirmed at both `"ip" in f` and
#             `"block0_items" in g`. This is h5py's fallback exception for HDF5 error codes it has
#             no specific mapping for, not a dedicated one for this failure: fractions just above
#             this band raise RuntimeError too but with other messages ("address of object past
#             end of allocation", "bad local heap signature"), and fractions just below it land
#             back in a KeyError instead (via `f["ip"]` itself rather than a lookup within it) --
#             confirming RuntimeError is not tied to one call site or one message. A previous
#             version of this comment mis-identified this fraction's failure as a KeyError with
#             this same message; it is not a KeyError at all -- corrected here. Used by the
#             link-existence-corruption tests below.
#
# Corrupting less than the first band leaves the chunk's raw data untouched and the file reads
# back with no error at all. Corrupting the entire file (frac=0) damages the superblock itself,
# and the file fails to open in the first place -- OSError, but a different case entirely (see
# test_unreadable_our_file_falls_back_to_staged, which uses plain garbage bytes instead).
_FRAC_RAW_CHUNK_DATA = 0.7
_FRAC_OBJECT_HEADER = 0.3
_FRAC_LINK_EXISTENCE = 0.008


def _write_corrupt(path, frac: float, group: str = "ip", col: str = "ipsip") -> None:
    """Write a real HDF5 file with a chunked dataset, then corrupt its trailing `1 - frac`
    fraction in place (same length) -- the shared construction behind all three corruption bands
    named in the comment above. `frac` selects which failure shape results; see that comment for
    the three values this suite uses and why each lands where it does.
    """
    n = 7000
    t = np.arange(0.0, float(n), 1.0, dtype=np.float64)
    vals = np.stack([np.arange(n, dtype=np.float32), np.arange(n, dtype=np.float32) * 2.0], axis=1)
    with h5py.File(path, "w") as f:
        g = f.create_group(group)
        g.attrs["nblocks"] = 1
        # This file lives in raw_dir, so it has to look like a group our fetcher finished writing
        # (`complete` last) or legacy_raw skips it as mid-write before any corruption is reached --
        # which would test the wrong thing entirely. The group's own object header, where attrs
        # live, survives intact in both corruption bands that get as far as this check.
        g.attrs["complete"] = True
        g.create_dataset("axis1", data=t)
        g.create_dataset("block0_items", data=np.array([col.encode(), b"other"]))
        g.create_dataset("block0_values", data=vals, chunks=(64, 2))
    raw = bytearray(path.read_bytes())
    start = int(len(raw) * frac)
    for i in range(start, len(raw)):
        raw[i] ^= 0xFF
    path.write_bytes(bytes(raw))


def test_read_time_corruption_falls_back_to_staged(paths, staged_shot_a, caplog):
    """A file can open fine and still raise OSError once a chunked dataset inside it is actually
    read (see the corruption-bands comment and _write_corrupt above). That must fall back to the
    other location exactly like a file that fails to open at all does in
    test_unreadable_our_file_falls_back_to_staged -- _read_signal_at/_status_at wrap the open and
    the read in one try precisely so this works."""
    bad = legacy_raw.our_path(staged_shot_a, paths)
    _write_corrupt(bad, _FRAC_RAW_CHUNK_DATA)
    # Confirm this is genuinely a read-time failure, not an open-time one: if h5py's behaviour
    # ever changed so this exact corruption were instead caught at open, this setup check would
    # fail loudly here rather than letting the test below silently degrade into re-testing the
    # already-covered open-time path.
    with h5py.File(bad, "r") as f:
        assert "ip" in f
        with pytest.raises(OSError):
            f["ip"]["block0_values"][()]
    with caplog.at_level(logging.WARNING):
        # scale=1.0e6: ipsip is megaamps on disk (see signals.yaml's ip.scale comment).
        sig = legacy_raw.read_signal(staged_shot_a, spec(scale=1.0e6), paths)
        status = legacy_raw.signal_status(staged_shot_a, spec(), paths)
    assert sig is not None and sig.source == "staged"
    assert abs(float(sig.y[(sig.t_ms >= 1000) & (sig.t_ms <= 4000)].mean()) - 1.2e6) < 1.0
    assert status == "present"
    assert str(bad) in caplog.text


def test_only_read_time_corrupt_file_is_pending_not_none(paths, caplog):
    """Same corruption as above, but as the only file for a shot: this is 'we tried to fetch
    this and failed', not 'DIII-D has no such data' -- read_signal must return None (never raise)
    and signal_status must report pending, mirroring
    test_only_unreadable_file_is_pending_not_none for the read-time failure mode instead of the
    open-time one."""
    shot = 900009
    bad = legacy_raw.our_path(shot, paths)
    _write_corrupt(bad, _FRAC_RAW_CHUNK_DATA)
    with caplog.at_level(logging.WARNING):
        sig = legacy_raw.read_signal(shot, spec(), paths)
        status = legacy_raw.signal_status(shot, spec(), paths)
    assert sig is None
    assert status == "pending"
    assert str(bad) in caplog.text


def test_bad_file_warns_only_once_per_path(paths, caplog):
    """The warning must not flood the log: dozens of signal specs per shot, times both
    read_signal and signal_status opening independently, means one bad file could otherwise
    print hundreds of identical lines across a full-database build, burying anything new."""
    legacy_raw._warned_paths.clear()
    # This test's own paths are already unique (a fresh tmp_path per test gives every shot number
    # here a distinct absolute path), so clearing isn't needed for isolation from other tests --
    # doing it anyway makes that independence explicit rather than relied upon, per this module's
    # deliberate lack of a public reset API for _warned_paths.
    shot_a, shot_b = 900010, 900011
    bad_a, bad_b = legacy_raw.our_path(shot_a, paths), legacy_raw.our_path(shot_b, paths)
    bad_a.write_bytes(b"not an hdf5 file")
    bad_b.write_bytes(b"not an hdf5 file")
    with caplog.at_level(logging.WARNING):
        legacy_raw.read_signal(shot_a, spec(), paths)
        legacy_raw.signal_status(shot_a, spec(), paths)  # same path, other entry point: still one
        legacy_raw.read_signal(shot_b, spec(), paths)  # a different path: gets its own warning
    for_a = [r for r in caplog.records if str(bad_a) in r.getMessage()]
    for_b = [r for r in caplog.records if str(bad_b) in r.getMessage()]
    assert len(for_a) == 1
    assert len(for_b) == 1


# --- Fix wave 3: corruption reaching the object header raises KeyError, not OSError -------


def test_object_header_corruption_falls_back_to_staged(paths, staged_shot_a, caplog):
    """A file can open fine, list its groups fine, and even confirm a column's link exists, and
    still raise a bare KeyError (not OSError) the instant that column's object header is opened
    -- see the corruption-bands comment and _write_corrupt above (frac=0.3, the object-header
    band). This is the same defect class as the read-time OSError covered by
    test_read_time_corruption_falls_back_to_staged: a location that cannot be used for either
    reason must fall back to the other location exactly the same way, instead of aborting every
    signal in the shot's extraction."""
    bad = legacy_raw.our_path(staged_shot_a, paths)
    _write_corrupt(bad, _FRAC_OBJECT_HEADER)
    # Confirm this is genuinely the object-header failure and not one of the already-covered
    # cases, at the exact call site _read_col hits first (g[items_key], before it ever reaches
    # block0_values): the link resolves fine (the containment check _read_col relies on before
    # opening anything succeeds), but opening the object itself raises KeyError, not OSError.
    with h5py.File(bad, "r") as f:
        assert "ip" in f
        g = f["ip"]
        assert "block0_items" in g
        with pytest.raises(KeyError):
            g["block0_items"]
    with caplog.at_level(logging.WARNING):
        # scale=1.0e6: ipsip is megaamps on disk (see signals.yaml's ip.scale comment).
        sig = legacy_raw.read_signal(staged_shot_a, spec(scale=1.0e6), paths)
        status = legacy_raw.signal_status(staged_shot_a, spec(), paths)
    assert sig is not None and sig.source == "staged"
    assert abs(float(sig.y[(sig.t_ms >= 1000) & (sig.t_ms <= 4000)].mean()) - 1.2e6) < 1.0
    assert status == "present"
    assert str(bad) in caplog.text


def test_only_object_header_corrupt_file_is_pending_not_none(paths, caplog):
    """Same corruption as above, but as the only file for a shot: this is 'we tried to fetch
    this and failed', not 'DIII-D has no such data' -- read_signal must return None (never raise)
    and signal_status must report pending, mirroring
    test_only_read_time_corrupt_file_is_pending_not_none for the KeyError failure mode instead of
    the OSError one."""
    shot = 900012
    bad = legacy_raw.our_path(shot, paths)
    _write_corrupt(bad, _FRAC_OBJECT_HEADER)
    with caplog.at_level(logging.WARNING):
        sig = legacy_raw.read_signal(shot, spec(), paths)
        status = legacy_raw.signal_status(shot, spec(), paths)
    assert sig is None
    assert status == "pending"
    assert str(bad) in caplog.text


# --- Fix wave 4: h5py's generic-fallback RuntimeError, not just OSError/KeyError -----------


def test_link_existence_corruption_falls_back_to_staged(paths, staged_shot_a, caplog):
    """A file can open fine and still raise a bare RuntimeError -- neither OSError nor KeyError --
    the instant a plain containment check ("x in group") touches it -- see the corruption-bands
    comment and _write_corrupt above (frac=0.008, the link-existence band). h5py's error table
    maps known HDF5 error codes to specific Python exceptions and falls back to RuntimeError for
    codes it has none for, so this can come from any HDF5 call on a damaged file, not only the
    `spec.group not in f` check this fixture happens to hit first; it is that generic fallback,
    not a fourth failure mode to special-case. A location that fails this way must fall back to
    the other location exactly like the OSError and KeyError cases above, instead of aborting
    every signal in the shot's extraction."""
    bad = legacy_raw.our_path(staged_shot_a, paths)
    _write_corrupt(bad, _FRAC_LINK_EXISTENCE)
    # Confirm this is genuinely the link-existence failure and not one of the already-covered
    # cases: the file itself still opens (an open-time OSError would fail here), and the exact
    # call site _read_signal_at/_status_at hit first -- `spec.group not in f` -- raises
    # RuntimeError, not KeyError or OSError. If h5py's behaviour ever changed so this exact
    # corruption were instead caught one of the already-covered ways, this setup check would fail
    # loudly here rather than letting the test below silently degrade into re-testing an
    # already-covered path.
    with h5py.File(bad, "r") as f, pytest.raises(RuntimeError):
        _ = "ip" in f
    with caplog.at_level(logging.WARNING):
        # scale=1.0e6: ipsip is megaamps on disk (see signals.yaml's ip.scale comment).
        sig = legacy_raw.read_signal(staged_shot_a, spec(scale=1.0e6), paths)
        status = legacy_raw.signal_status(staged_shot_a, spec(), paths)
    assert sig is not None and sig.source == "staged"
    assert abs(float(sig.y[(sig.t_ms >= 1000) & (sig.t_ms <= 4000)].mean()) - 1.2e6) < 1.0
    assert status == "present"
    assert str(bad) in caplog.text


def test_only_link_existence_corrupt_file_is_pending_not_none(paths, caplog):
    """Same corruption as above, but as the only file for a shot: this is 'we tried to fetch this
    and failed', not 'DIII-D has no such data' -- read_signal must return None (never raise) and
    signal_status must report pending, mirroring
    test_only_object_header_corrupt_file_is_pending_not_none for the RuntimeError failure mode
    instead of the KeyError one."""
    shot = 900013
    bad = legacy_raw.our_path(shot, paths)
    _write_corrupt(bad, _FRAC_LINK_EXISTENCE)
    with caplog.at_level(logging.WARNING):
        sig = legacy_raw.read_signal(shot, spec(), paths)
        status = legacy_raw.signal_status(shot, spec(), paths)
    assert sig is None
    assert status == "pending"
    assert str(bad) in caplog.text


# --- Units fix: per-column scale and null-value sentinels ---------------------------------
#
# Motivated by a sweep of the real archive file (/scratch/gpfs/EKOLEMEN/d3d_fusion_data/
# 161172.h5, see task-7-units-report.md): ip/ipsip is stored in megaamps although its registry
# entry declares amps, and divertor_geo/zxpt1 uses -9.99 as EFIT's "no X-point found" sentinel.
# SignalSpec.scale and .null_value fix both in configuration. These tests exercise the reader
# mechanism directly through the spec() helper rather than via signals.yaml, so they pin the
# behavior independent of any one signal's registry entry.


def test_scale_multiplies_values_and_unscaled_spec_is_unaffected(paths):
    """A signal read with a scale comes back multiplied by it; the identical raw file read through
    a spec with no scale (the field's default, 1.0) is unchanged -- scale is opt-in per spec, not
    a reinterpretation of the column itself."""
    shot, path = 900014, paths.staged_raw_dir / "900014.h5"
    t = np.arange(0.0, 10.0, 1.0)
    df = pd.DataFrame({"ipsip": np.full(t.size, 1.037, dtype=np.float32)}, index=pd.Index(t))
    df.to_hdf(path, key="ip", mode="a", format="fixed")
    scaled = legacy_raw.read_signal(shot, spec(scale=1.0e6), paths)
    unscaled = legacy_raw.read_signal(shot, spec(), paths)
    assert scaled is not None and unscaled is not None
    np.testing.assert_allclose(scaled.y, 1.037e6, atol=1.0)
    np.testing.assert_allclose(unscaled.y, 1.037, atol=1e-4)


def test_null_value_becomes_nan_at_exact_sentinel_only(paths):
    """A sample exactly equal to null_value becomes NaN; real values elsewhere -- including one
    merely close to the sentinel but not an exact bit-for-bit match -- are left alone. The
    comparison is exact equality, not a tolerance, because the sentinel (zxpt1's -9.99 EFIT
    marker) is a literal constant the writer stamps in, not a computed value that could land
    nearby by rounding."""
    shot, path = 900015, paths.staged_raw_dir / "900015.h5"
    t = np.arange(0.0, 5.0, 1.0)
    y = np.array([-9.99, -1.5, -9.989, -2.3, -9.99], dtype=np.float32)
    df = pd.DataFrame({"zxpt1": y}, index=pd.Index(t))
    df.to_hdf(path, key="divertor_geo", mode="a", format="fixed")
    sig = legacy_raw.read_signal(shot, spec("zxpt1", "divertor_geo", "zxpt1", null_value=-9.99), paths)
    assert sig is not None
    assert np.isnan(sig.y[0]) and np.isnan(sig.y[4])
    assert not np.isnan(sig.y[2])  # -9.989 is close to the sentinel but must not be nulled
    np.testing.assert_allclose(sig.y[[1, 2, 3]], [-1.5, -9.989, -2.3], atol=1e-4)


def test_null_value_list_nulls_each_member_independently(paths):
    """drsep's fix: a symmetric +-clamp can't be expressed as one scalar, so null_value also
    accepts a list and nulls a sample matching ANY entry in it. Uses the exact on-disk float32
    saturation values confirmed against the real archive (see signals.yaml's drsep comment) --
    Python's own naive float32(0.4)/float32(-0.4) are each one ULP away from what EFIT actually
    writes and would silently null nothing on a real file. A value merely near one of the two, or
    exactly at neither, is left alone, same as the single-sentinel case above."""
    shot, path = 900017, paths.staged_raw_dir / "900017.h5"
    lo, hi = -0.3999999761581421, 0.3999999761581421
    y = np.array([lo, -0.2, 0.0, 0.2, hi], dtype=np.float32)
    t = np.arange(0.0, 5.0, 1.0)
    df = pd.DataFrame({"drsep": y}, index=pd.Index(t))
    df.to_hdf(path, key="divertor_geo", mode="a", format="fixed")
    sig = legacy_raw.read_signal(
        shot, spec("drsep", "divertor_geo", "drsep", null_value=[lo, hi]), paths
    )
    assert sig is not None
    assert np.isnan(sig.y[0]) and np.isnan(sig.y[4])
    np.testing.assert_allclose(sig.y[[1, 2, 3]], [-0.2, 0.0, 0.2], atol=1e-6)


def test_scale_and_abs_compose_for_a_negative_input(paths):
    """scale and abs must both actually apply, in sequence (scale first, then abs -- see the
    comment in _read_signal_at): a negative raw value scaled by a factor and then made positive
    must equal abs(raw * scale), which only holds if neither step is skipped."""
    shot, path = 900016, paths.staged_raw_dir / "900016.h5"
    t = np.arange(0.0, 5.0, 1.0)
    df = pd.DataFrame({"ipsip": np.full(t.size, -2.0, dtype=np.float32)}, index=pd.Index(t))
    df.to_hdf(path, key="ip", mode="a", format="fixed")
    sig = legacy_raw.read_signal(shot, spec(scale=3.0, abs=True), paths)
    assert sig is not None
    assert np.all(sig.y > 0)
    np.testing.assert_allclose(sig.y, 6.0, atol=1e-4)


def test_a_fetched_group_without_complete_is_treated_as_absent(paths, dual_shot_d):
    """A bulk fetch writes into raw_dir while databases are built from the same paths.

    scripts/fetch_shots.write_group replaces a group wholesale and stamps `complete = True` LAST,
    so a fetched group without it is a write that was interrupted or that failed -- its frame may
    be absent, truncated or half-rewritten. Reading it anyway would hand back a partial trace
    under the fetched label; the only safe reading is "not there yet", i.e. fall through to the
    staged copy. dual_shot_d normally has our 2 MA file win; strip the attr and the staged 1 MA
    copy must win instead, with the status following the same rule.

    The file is stamped as the fetcher stamps every file it writes: without a marker of our own
    on it, a group with no `complete` is indistinguishable from a staged group, which never has
    one -- see legacy_raw.is_ours.
    """
    stamp_ours(paths.raw_dir / "900004.h5", dual_shot_d)
    with h5py.File(paths.raw_dir / "900004.h5", "a") as f:
        del f["ip"].attrs["complete"]
    sig = legacy_raw.read_signal(dual_shot_d, spec(scale=1.0e6), paths)
    assert sig.source == "staged"
    np.testing.assert_allclose(sig.y, 1.0e6)
    assert legacy_raw.signal_status(dual_shot_d, spec(), paths) == "present"  # from the staged copy


def test_an_incomplete_fetched_group_with_no_staged_copy_is_pending(paths, our_shot_c):
    """our_shot_c exists only in raw_dir, so there is nothing to fall through to: an unfinished
    group must read as absent-but-fetchable, never as a signal."""
    with h5py.File(paths.raw_dir / "900003.h5", "a") as f:
        del f["ech"].attrs["complete"]
    ech = spec("pech_LEIA", "ech", "ecleifpwrc")
    assert legacy_raw.read_signal(our_shot_c, ech, paths) is None
    assert legacy_raw.signal_status(our_shot_c, ech, paths) == "pending"
    # the *other* group of the same file is untouched and still readable
    assert legacy_raw.read_signal(our_shot_c, spec(abs=True, scale=1.0e6), paths).source == "fetched"


def test_staged_groups_are_never_required_to_be_complete(paths, staged_shot_a):
    """No staged d3d_fusion_data group carries a `complete` attribute; requiring one would read
    every staged shot as empty."""
    with h5py.File(paths.staged_raw_dir / "900001.h5", "r") as f:
        assert "complete" not in f["ip"].attrs
    assert legacy_raw.read_signal(staged_shot_a, spec(abs=True, scale=1.0e6), paths) is not None
    assert legacy_raw.signal_status(staged_shot_a, spec(), paths) == "present"


# --- One reader: the lock, the file's own markers, and one pass per shot -------------------


def _recording_h5(monkeypatch) -> list[bool | None]:
    """Swap h5py.File for a subclass that records the `locking` kwarg of every read-only open."""
    opens: list[bool | None] = []
    real = h5py.File

    class Recording(real):
        def __init__(self, *args, **kwargs):
            mode = kwargs.get("mode", args[1] if len(args) > 1 else "r")
            if mode == "r":
                opens.append(kwargs.get("locking"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(h5py, "File", Recording)
    return opens


def test_every_read_only_open_disables_the_hdf5_lock(paths, dual_shot_d, monkeypatch):
    """scripts/fetch_shots.py holds a write lock on the file it is writing, and a plain
    `h5py.File(path, "r")` against it raises BlockingIOError -- an OSError, which _warn_once
    swallowed, so a `ideate build` during a fetch silently built the shot with zero signals. Every
    entry point has to go through legacy_raw.open_h5 (locking=False); this pins all of them, on a shot
    that exists in both locations so both files are opened."""
    opens = _recording_h5(monkeypatch)
    legacy_raw.read_signal(dual_shot_d, spec("gas_GASA", "gas", "gasa"), paths)  # falls to staged
    legacy_raw.signal_status(dual_shot_d, spec(), paths)
    legacy_raw.read_shot(dual_shot_d, [spec(), spec("gas_GASA", "gas", "gasa")], paths)
    legacy_raw.pending_specs(dual_shot_d, [spec()], paths)
    legacy_raw.list_groups(legacy_raw.our_path(dual_shot_d, paths))
    assert len(opens) >= 6 and all(lk is False for lk in opens), opens


def test_is_ours_reads_the_fetchers_markers_not_the_directory(paths):
    """A d3d_fusion_data file has none of the fetcher's attrs; one of ours has the file-level
    schema and, per group, source/fetcher_version/complete. Either alone is enough."""
    t = np.arange(0.0, 5.0)
    staged_like = paths.raw_dir / "900040.h5"  # a staged-layout file, deliberately in raw_dir
    write_frame(staged_like, "ip", t, {"ipsip": t})
    with h5py.File(staged_like, "r") as f:
        assert legacy_raw.is_ours(f) is False and legacy_raw.is_ours(f, f["ip"]) is False

    stamped = paths.staged_raw_dir / "900041.h5"  # ours by its stamp, in the staged dir
    write_frame(stamped, "ip", t, {"ipsip": t})
    stamp_ours(stamped, 900041)
    with h5py.File(stamped, "r") as f:
        assert legacy_raw.is_ours(f) is True and legacy_raw.is_ours(f, f["ip"]) is True

    by_group = paths.staged_raw_dir / "900042.h5"  # unstamped file, groups marked by write_group
    for group, attrs in (
        ("a", {"source": "toksearch"}),
        ("b", {"fetcher_version": "1"}),
        ("c", {"complete": True}),
        ("d", {}),
    ):
        write_frame(by_group, group, t, {"x": t}, attrs)
    with h5py.File(by_group, "r") as f:
        assert legacy_raw.is_ours(f) is False
        assert [legacy_raw.is_ours(f, f[g]) for g in "abcd"] == [True, True, True, False]


def test_a_fetched_file_in_the_staged_directory_keeps_its_own_semantics(paths):
    """One of our files copied under staged_raw_dir: its torn group must still read as 'not
    there yet' and its missing_channels must still settle a column as unavailable -- the two rules
    that used to switch on the directory and so were both lost on a mislocated file."""
    shot, path = 900043, paths.staged_raw_dir / "900043.h5"
    t = np.arange(0.0, 10.0)
    ours = {"source": "toksearch", "fetcher_version": "1", "missing_channels": np.array([], "S")}
    write_frame(path, "ip", t, {"ipsip": np.full_like(t, 2.0)}, ours)  # torn: no `complete`
    write_frame(
        path,
        "ech",
        t,
        {"ecleifpwrc": np.full_like(t, 1.0)},
        {**ours, "complete": True, "missing_channels": np.array([b"eclukfpwrc"], "S")},
    )
    stamp_ours(path, shot)
    assert legacy_raw.read_signal(shot, spec(), paths) is None
    assert legacy_raw.signal_status(shot, spec(), paths) == "pending"
    leia = legacy_raw.read_signal(shot, spec("pech_LEIA", "ech", "ecleifpwrc"), paths)
    assert leia is not None and leia.source == "fetched"  # labelled by producer, not location
    luke = spec("pech_LUKE", "ech", "eclukfpwrc")
    assert legacy_raw.signal_status(shot, luke, paths) == "unavailable"


def test_a_staged_file_in_raw_dir_is_read_as_the_staged_producers(paths):
    """The mirror case: a d3d_fusion_data copy placed under raw_dir carries no `complete` on any
    group and a comma-joined missing_channels. Requiring `complete` there would read every group
    as torn, and honouring its missing_channels would write q95 off for ever."""
    shot, path = 900044, paths.raw_dir / "900044.h5"
    t = np.arange(0.0, 10.0)
    ip = pd.DataFrame({"ipsip": np.full(t.size, 1.5, np.float32)}, index=t)
    q = pd.DataFrame({"qpsi0.00": np.full(t.size, 3.0, np.float32)}, index=t)
    with pd.HDFStore(path, mode="a") as store:
        store.put("ip", ip, format="fixed")
        store.put("q_psi", q, format="fixed")
        store.get_storer("q_psi").attrs.missing_channels = np.bytes_(b"q0,q95,qmin")
    sig = legacy_raw.read_signal(shot, spec(scale=1.0e6), paths)
    assert sig is not None and sig.source == "staged"
    np.testing.assert_allclose(sig.y, 1.5e6)
    assert legacy_raw.signal_status(shot, spec(), paths) == "present"
    assert legacy_raw.signal_status(shot, spec("q95", "q_psi", "q95"), paths) == "pending"


def test_read_shot_is_one_pass_and_agrees_with_the_per_spec_api(
    paths, dual_shot_d, our_shot_c, monkeypatch
):
    """build_record used to read every column twice -- once for the value, once more to say
    `present`. read_shot answers both from one pass: values identical to read_signal, statuses
    identical to signal_status, `present` exactly when a signal came back, not-installed specs
    never read, and each existing file opened once."""
    specs = [
        spec(scale=1.0e6),
        spec("gas_GASA", "gas", "gasa"),
        spec("gas_GASB", "gas", "gasb"),
        spec("pech_LEIA", "ech", "ecleifpwrc"),
        spec("ne0", "e_dens_fit", "edensfit0.00", fetch=None),
        spec("pech_BORIS", "ech", "ecborfpwrc", installed=False),
    ]
    for shot in (dual_shot_d, our_shot_c):
        opens = _recording_h5(monkeypatch)
        signals, coverage = legacy_raw.read_shot(shot, specs, paths)
        assert len(opens) == len(legacy_raw.source_paths(shot, paths))
        assert list(signals) == list(coverage) == [s.name for s in specs]
        for s in specs:
            assert coverage[s.name] == legacy_raw.signal_status(shot, s, paths), s.name
            assert (coverage[s.name] == "present") == (signals[s.name] is not None), s.name
            single = legacy_raw.read_signal(shot, s, paths) if s.installed else None
            got = signals[s.name]
            if single is None:
                assert got is None, s.name
            else:
                assert got.source == single.source and got.units == single.units
                np.testing.assert_array_equal(got.t_ms, single.t_ms)
                np.testing.assert_array_equal(got.y, single.y)
        monkeypatch.undo()
    d_signals, d_cov = legacy_raw.read_shot(dual_shot_d, specs, paths)
    assert d_signals["gas_GASA"].source == "staged" and d_signals["gas_GASB"].source == "fetched"
    assert d_cov["pech_BORIS"] == "not_installed" and d_cov["ne0"] == "unavailable"
    assert d_cov["pech_LEIA"] == "pending"
    c_cov = legacy_raw.read_shot(our_shot_c, [spec("pech_LUKE", "ech", "eclukfpwrc")], paths)[1]
    assert c_cov == {"pech_LUKE": "unavailable"}


def test_a_corrupt_group_leaves_the_other_groups_of_the_same_file_readable(paths, caplog):
    """The per-group try in _read_specs: damage confined to one group's chunk data must not take
    down the signals that live in the file's other groups."""
    shot = 900045
    path = legacy_raw.our_path(shot, paths)
    _write_corrupt(path, _FRAC_RAW_CHUNK_DATA)  # `ip` is now unreadable past its header
    t = np.arange(0.0, 10.0)
    write_frame(path, "gas", t, {"gasa": np.full_like(t, 4.0)}, {"complete": True})
    with caplog.at_level(logging.WARNING):
        signals, coverage = legacy_raw.read_shot(
            shot, [spec(), spec("gas_GASA", "gas", "gasa")], paths
        )
    assert signals["ip"] is None and coverage["ip"] == "pending"
    assert coverage["gas_GASA"] == "present"
    np.testing.assert_allclose(signals["gas_GASA"].y, 4.0)
    assert str(path) in caplog.text
