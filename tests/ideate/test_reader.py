"""The `Reader` protocol and `legacy_raw`'s adapter onto it.

`shotdb/reader.py` exists so that `build` names an interface rather than the d3d_fusion_data
layout: iteration 0 builds from `legacy_raw`, iteration 1 from `corpus.CorpusReader`, and the
swap has to be a constructor argument rather than an import edit. What is pinned here is that the
adapter is exactly that -- an adapter: every spec-level answer is the module function's answer,
unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from ideate import config
from ideate.shotdb import build, legacy_raw
from ideate.shotdb.corpus import CorpusReader
from ideate.shotdb.reader import Reader, ShotFailed, SignalReader, Unavailable


def spec(name="ip", group="ip", col="ipsip", **kw) -> config.SignalSpec:
    return config.SignalSpec(name=name, group=group, col=col, fetch={"kind": "ptdata"}, **kw)


# ------------------------------------------------------------------------------- conformance


def test_legacy_reader_satisfies_both_protocols(paths):
    r = legacy_raw.LegacyReader(paths)
    assert isinstance(r, Reader) and isinstance(r, SignalReader)


def test_a_corpus_reader_is_a_reader_but_not_yet_a_signal_reader(tmp_path):
    # The spec-level half of the interface arrives with the corpus build (Task I6). Until it
    # does, `isinstance` says so rather than a NotImplementedError saying it at call time.
    r = CorpusReader(tmp_path)
    assert isinstance(r, Reader) and not isinstance(r, SignalReader)


def test_build_reads_through_the_reader_it_is_handed(paths, staged_shot_a, text_fixtures):
    """`build_record` must go through the reader argument, not through the module."""
    seen: list[int] = []

    class Spy(legacy_raw.LegacyReader):
        def read_shot(self, shot, specs):
            seen.append(shot)
            return super().read_shot(shot, specs)

    rec, _ = build.build_record(staged_shot_a, paths, build.load_build_cfg(), reader=Spy(paths))
    assert seen == [staged_shot_a] and rec.coverage["ip"] == "present"


# ------------------------------------------------------------------------- the spec-level half


def test_the_adapter_answers_exactly_as_the_module_functions_do(paths, staged_shot_a, our_shot_c):
    for shot in (staged_shot_a, our_shot_c):
        r = legacy_raw.LegacyReader(paths)
        specs = config.expand_registry(shot, include_not_installed=True)
        signals, coverage = r.read_shot(shot, specs)
        want_signals, want_coverage = legacy_raw.read_shot(shot, specs, paths)
        assert coverage == want_coverage
        assert sorted(k for k, v in signals.items() if v is not None) == sorted(
            k for k, v in want_signals.items() if v is not None
        )
        assert r.signal_status(shot, spec()) == legacy_raw.signal_status(shot, spec(), paths)
        got, want = r.read_signal(shot, spec()), legacy_raw.read_signal(shot, spec(), paths)
        np.testing.assert_array_equal(got.y, want.y)


# ------------------------------------------------------------------------- the file-level half


def test_the_adapter_reads_the_legacy_layout_in_milliseconds(paths, staged_shot_a):
    r = legacy_raw.LegacyReader(paths)
    assert r.path(staged_shot_a) == legacy_raw.staged_path(staged_shot_a, paths)
    assert r.available(staged_shot_a) and not r.available(999999)
    assert "ip" in r.groups(staged_shot_a) and "p_inj" in r.groups(staged_shot_a)
    t_ms, y = r.read(staged_shot_a, "ip")
    assert y.shape == (2, t_ms.size) and t_ms.dtype == np.float64 and y.dtype == np.float32
    assert (t_ms[0], t_ms[-1]) == (-500.0, 6499.0)
    assert r.coverage(staged_shot_a, "ip") == pytest.approx((-500.0, 6499.0))
    _, one = r.read(staged_shot_a, "ip", channels=[0])
    np.testing.assert_array_equal(one[0], y[0])


def test_the_adapter_raises_unavailable_and_shot_failed(paths, staged_shot_a, our_shot_c):
    r = legacy_raw.LegacyReader(paths)
    with pytest.raises(Unavailable):
        r.read(staged_shot_a, "no_such_group")
    with pytest.raises(ShotFailed):
        r.read(999999, "ip")  # no file in either location
    legacy_raw._warned_paths.clear()
    (paths.raw_dir / f"{our_shot_c}.h5").write_bytes(b"not an HDF5 file at all")
    with pytest.raises(ShotFailed) as e:
        r.read(our_shot_c, "ip")  # the only location this shot has, and it is unreadable
    assert isinstance(e.value.__cause__, OSError)


def test_an_unreadable_location_falls_through_to_the_other_one(paths, staged_shot_a):
    """Precedence is legacy_raw's, unchanged: a damaged file of ours does not lose the staged
    copy's groups -- which is why an unreadable location is not by itself a failed shot."""
    legacy_raw._warned_paths.clear()
    (paths.raw_dir / f"{staged_shot_a}.h5").write_bytes(b"not an HDF5 file at all")
    t_ms, y = legacy_raw.LegacyReader(paths).read(staged_shot_a, "ip")
    assert y.shape == (2, t_ms.size) and t_ms.size > 1000
