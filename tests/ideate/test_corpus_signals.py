"""`CorpusSignalReader` -- the signal registry resolved against the FAITH corpus + labelmaker.

Three things are pinned here, and they are the three ways a corpus build can quietly lie:

* **which channel** an address names. The corpus's channel arrays are unnamed, so a wrong index
  is not an error, it is a plausible number from the wrong diagnostic. Every fixture channel
  holds a distinct constant, so a wrong channel is an arithmetic mismatch.
* **which reduction** it uses. `sum` over eight beams and `mean` over eight accelerating voltages
  are not interchangeable, and neither is either with `first`.
* **what an absence means.** A corpus group that is not there is `unavailable` -- DIII-D did not
  record it. A labelmaker feature that is not there yet is `pending` -- fdp can still fetch it.
  Neither is ever 0, and the two are never each other.
"""

from __future__ import annotations

import numpy as np
import pytest

from ideate import config
from ideate.shotdb import legacy_raw
from ideate.shotdb.corpus_signals import CorpusSignalReader
from ideate.shotdb.reader import Reader, ShotFailed, SignalReader

from .conftest import CORPUS_BARE_SHOT, write_feature_file


def registry(shot: int) -> list[config.SignalSpec]:
    return config.expand_registry(shot, include_not_installed=True)


def spec_named(shot: int, name: str) -> config.SignalSpec:
    return next(s for s in registry(shot) if s.name == name)


# ---------------------------------------------------------------------------------- conformance


def test_a_corpus_signal_reader_is_a_signal_reader(paths):
    r = CorpusSignalReader(paths)
    assert isinstance(r, Reader) and isinstance(r, SignalReader)


# ------------------------------------------------------------------- the corpus address kind


def test_an_actuator_member_reads_its_own_named_channel(paths, signal_corpus):
    """`pnbi_15L` is channel 0 of `pinj` and `pnbi_33R` is channel 7, by name, not by luck."""
    r = CorpusSignalReader(paths)
    first = r.read_signal(signal_corpus, spec_named(signal_corpus, "pnbi_15L"))
    last = r.read_signal(signal_corpus, spec_named(signal_corpus, "pnbi_33R"))
    assert first is not None and last is not None
    assert np.allclose(first.y, 1.0e5) and np.allclose(last.y, 8.0e5)
    # seconds on disk (float32, as the real files are), milliseconds here
    assert first.t_ms[0] == 0.0 and abs(first.t_ms[-1] - 6000.0) < 1e-2


def test_a_member_is_found_by_name_even_when_the_channel_order_differs(paths, signal_corpus):
    """The corpus orders its gyrotrons alphabetically and the registry does not. Only the
    ech_power channel the fixture powered (LEIA, index 5) may come back non-zero."""
    r = CorpusSignalReader(paths)
    powered = {
        m: r.read_signal(signal_corpus, spec_named(signal_corpus, f"pech_{m}"))
        for m in ("LEIA", "LUKE", "BORIS", "SCARECROW")
    }
    assert np.allclose(powered["LEIA"].y, 1.0e6)
    for m in ("LUKE", "BORIS", "SCARECROW"):
        assert np.allclose(powered[m].y, 0.0)


def test_the_system_total_is_the_sum_over_the_corpus_channels(paths, signal_corpus):
    """What `build` gets out of `features.system_totals` over the members read above."""
    from ideate.shotdb import features

    r = CorpusSignalReader(paths)
    specs = registry(signal_corpus)
    signals, _ = r.read_shot(signal_corpus, specs)
    totals = features.system_totals(signals, config.actuator_systems(signal_corpus))
    assert np.allclose(totals["pnbi_total"].y, 3.6e6)  # (1+..+8) * 1e5
    assert np.allclose(totals["pech_total"].y, 1.0e6)
    assert np.allclose(totals["gas_total"].y, 2.0)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("nbi_torque_total", 36.0),  # sum of 1..8
        ("nbi_voltage_mean", 4.5e3),  # mean of 1e3..8e3
        ("rmp_total", 30.0),  # sum of |+10| and |-20| over the coils
        ("ne_line", 5.0e13),  # co2's V2 chord, not its first channel
        ("neutrons", 7.0e14),  # neutron_rate's total channel
        ("dalpha", 4.0),  # mean of filterscopes 1..7; 0 is unrecorded, 8-9 excluded
    ],
)
def test_each_reduction_over_the_corpus_channels(paths, signal_corpus, name, expected):
    r = CorpusSignalReader(paths)
    sig = r.read_signal(signal_corpus, spec_named(signal_corpus, name))
    assert sig is not None, name
    assert np.allclose(sig.y, expected), (name, sig.y[:3])


def test_filterscope_channels_above_seven_are_never_included(paths, signal_corpus):
    """Channels 8+ of `filterscopes` are other photomultipliers, not D-alpha. Including them
    would read 204.4 here instead of 4.0 -- a number, not an error."""
    r = CorpusSignalReader(paths)
    sig = r.read_signal(signal_corpus, spec_named(signal_corpus, "dalpha"))
    assert not np.isclose(float(sig.y[0]), 204.4)


def test_a_group_the_corpus_did_not_record_is_unavailable_and_never_zero(paths, signal_corpus):
    r = CorpusSignalReader(paths)
    specs = registry(CORPUS_BARE_SHOT)
    signals, coverage = r.read_shot(CORPUS_BARE_SHOT, specs)
    for name in ("pnbi_15L", "dalpha", "ne_line", "neutrons", "nbi_torque_total"):
        assert signals[name] is None, name
        assert coverage[name] == "unavailable", name


def test_a_signal_with_no_corpus_address_is_unavailable_not_pending(paths, signal_corpus):
    """q95 lives in EFIT and neither the corpus nor labelmaker carries it. `pending` would
    promise `ideate fetch` could get it, and on this reader nothing can."""
    r = CorpusSignalReader(paths)
    _, coverage = r.read_shot(signal_corpus, registry(signal_corpus))
    assert coverage["q95"] == "unavailable"


def test_an_uninstalled_member_is_not_installed(paths, signal_corpus):
    """A member the machine did not have yet is never read and never `unavailable`: the corpus
    says nothing about it either way. (No registry member currently carries a `since:`, so the
    spec is made here rather than taken from a shot's registry.)"""
    spec = config.SignalSpec(
        name="pnbi_15L", system="nbi", member="15L", installed=False, units="W"
    )
    signals, coverage = CorpusSignalReader(paths).read_shot(signal_corpus, [spec])
    assert signals["pnbi_15L"] is None and coverage["pnbi_15L"] == "not_installed"


def test_one_read_per_corpus_group_however_many_specs_share_it(paths, signal_corpus, monkeypatch):
    """Eight beams share `pinj`. Reading the group once per spec would be eight opens of the same
    file for one shot, and the registry has thirty-odd corpus specs."""
    from ideate.shotdb import corpus

    seen: list[str] = []
    real = corpus.CorpusReader.read

    def spy(self, shot, group, channels=None):
        seen.append(group)
        return real(self, shot, group, channels)

    monkeypatch.setattr(corpus.CorpusReader, "read", spy)
    CorpusSignalReader(paths).read_shot(signal_corpus, registry(signal_corpus))
    assert len(seen) == len(set(seen)), seen


def test_a_corpus_file_that_cannot_be_read_is_a_failed_shot_not_an_absent_diagnostic(paths):
    from .conftest import CORPUS_SIGNAL_SHOT

    d = paths.foundation_model_processed_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{CORPUS_SIGNAL_SHOT}_processed.h5").write_bytes(b"not an hdf5 file at all")
    with pytest.raises(ShotFailed):
        CorpusSignalReader(paths).read_shot(CORPUS_SIGNAL_SHOT, registry(CORPUS_SIGNAL_SHOT))


# --------------------------------------------------------------- the labelmaker address kind


def test_a_stored_feature_is_read_with_its_resolver(paths, signal_corpus, labelmaker_features):
    r = CorpusSignalReader(paths)
    bt = r.read_signal(signal_corpus, spec_named(signal_corpus, "bt"))
    assert bt is not None
    assert np.allclose(bt.y, 2.05)  # the spec's abs, applied as it is on the legacy layout
    assert bt.t_ms[0] == 0.0 and abs(bt.t_ms[-1] - 5975.0) < 1e-6  # seconds -> ms
    assert bt.source == "labelmaker" and bt.resolver == "labelmaker:archive"


def test_a_labelmaker_scale_is_not_the_legacy_one(paths, signal_corpus, labelmaker_features):
    """`ip`'s registry `scale: 1e6` corrects a d3d_fusion_data storage convention (megaamps in a
    column declared amps). labelmaker's `ip` is already amps; applying that scale would report a
    1.2 MA shot as 1.2 TA."""
    r = CorpusSignalReader(paths)
    ip = r.read_signal(signal_corpus, spec_named(signal_corpus, "ip"))
    assert 1.0e6 < float(np.nanmax(ip.y)) < 2.0e6


def test_segments_are_found_from_a_labelmaker_ip(paths, signal_corpus, labelmaker_features):
    from ideate.shotdb import build, features

    r = CorpusSignalReader(paths)
    ip = r.read_signal(signal_corpus, spec_named(signal_corpus, "ip"))
    segs = features.find_segments(ip, build.load_build_cfg()["segments"])
    flat = next(s for s in segs if s.name == "flat_top")
    assert flat.t1_ms - flat.t0_ms > 3000.0  # the fixture's trapezoid is flat 500-4500 ms


def test_a_missing_feature_file_is_pending_never_zero(paths, signal_corpus, labelmaker_features):
    """`CORPUS_BARE_SHOT` has no feature file at all: fdp can still fetch every one of them."""
    r = CorpusSignalReader(paths)
    signals, coverage = r.read_shot(CORPUS_BARE_SHOT, registry(CORPUS_BARE_SHOT))
    for name in ("ip", "bt", "betan", "li", "aminor"):
        assert signals[name] is None, name
        assert coverage[name] == "pending", name


def test_the_recorded_cause_of_a_miss_decides_pending_from_unavailable(
    paths, signal_corpus, labelmaker_features
):
    r = CorpusSignalReader(paths)
    _, coverage = r.read_shot(signal_corpus, registry(signal_corpus))
    assert coverage["qmin"] == "pending"  # fdp:TreeFOPENR -- worth another attempt
    assert coverage["volume"] == "unavailable"  # corpus:SignalAbsent -- it will fail identically
    assert coverage["li"] == "pending"  # never attempted


def test_a_feature_with_no_finite_sample_is_unavailable(paths, signal_corpus, labelmaker_features):
    r = CorpusSignalReader(paths)
    signals, coverage = r.read_shot(signal_corpus, registry(signal_corpus))
    assert signals["kappa"] is None and coverage["kappa"] == "unavailable"


@pytest.mark.parametrize(
    ("reduce", "expected"), [("core", 10.0), ("edge", 1.0), ("peak", 12.0)]
)
def test_a_profile_feature_is_reduced_to_a_scalar(
    paths, signal_corpus, labelmaker_features, reduce, expected
):
    """`ne_zipfit` is (33 rho, T). core is rho = 0, edge is rho = 1, peak is the largest of the
    33 -- the fixture puts its peak at rho index 16 so the three cannot alias."""
    spec = config.SignalSpec(
        name="ne_probe", labelmaker={"feature": "ne_zipfit", "reduce": reduce}
    )
    sig = CorpusSignalReader(paths).read_signal(signal_corpus, spec)
    assert sig is not None and np.allclose(sig.y, expected)


def test_the_registry_reduces_a_profile_the_same_way(paths, signal_corpus, labelmaker_features):
    """`ne0` is the on-axis density: the same quantity the staged `edensfit0.00` column is."""
    r = CorpusSignalReader(paths)
    assert np.allclose(r.read_signal(signal_corpus, spec_named(signal_corpus, "ne0")).y, 10.0)


def test_a_signal_status_matches_the_read(paths, signal_corpus, labelmaker_features):
    r = CorpusSignalReader(paths)
    for name in ("bt", "qmin", "pnbi_15L", "q95"):
        spec = spec_named(signal_corpus, name)
        got, status = r.read_signal(signal_corpus, spec), r.signal_status(signal_corpus, spec)
        assert (got is not None) == (status == "present"), name


# ---------------------------------------------------------------- the shape of what is returned


def test_read_shot_answers_in_the_same_shape_the_legacy_reader_does(
    paths, signal_corpus, labelmaker_features, staged_shot_a
):
    """Same keys, same value types, same status vocabulary -- which is why `build_record` needs
    no branch on which reader it was handed."""
    corpus_signals, corpus_cov = CorpusSignalReader(paths).read_shot(
        signal_corpus, registry(signal_corpus)
    )
    legacy_signals, legacy_cov = legacy_raw.LegacyReader(paths).read_shot(
        staged_shot_a, registry(staged_shot_a)
    )
    assert set(corpus_signals) == set(legacy_signals)
    assert set(corpus_cov) == set(legacy_cov)
    assert set(corpus_cov.values()) <= set(legacy_cov.values()) | {"present", "unavailable"}
    for sig in corpus_signals.values():
        if sig is not None:
            assert sig.t_ms.dtype == np.float64 and sig.y.dtype == np.float32
            assert sig.t_ms.shape == sig.y.shape


def test_a_feature_file_written_for_another_shot_is_not_read(paths, signal_corpus, tmp_path):
    """The store is one file per shot; a reader that globbed would hand shot A's `ip` to shot B."""
    features_dir = tmp_path / "elsewhere" / "features"
    write_feature_file(features_dir, 999999, {"bt": ([0.0, 0.025], [[1.0, 1.0]], "archive")}, {})
    r = CorpusSignalReader(paths, features_dir=features_dir)
    assert r.signal_status(signal_corpus, spec_named(signal_corpus, "bt")) == "pending"
