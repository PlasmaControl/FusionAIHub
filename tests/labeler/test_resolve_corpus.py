"""The corpus resolver: channel sums, units, decimation, absent groups."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from labeler.features import resolve_corpus as rc

CORPUS = Path("/scratch/gpfs/EKOLEMEN/foundation_model")


def _fake_corpus(tmp_path, *, ech_absent=False, ech_nan_channel=True):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    n = 2000                                  # 2 s at 1 kHz, for speed
    t = np.linspace(0.0, 2.0, n, dtype=np.float32)
    with h5py.File(corpus / "190000_processed.h5", "w") as f:
        g = f.create_group("pinj")            # 8 beams, 1 MW each, in W
        g.create_dataset("xdata", data=t)
        g.create_dataset("ydata", data=np.full((8, n), 1.0e6, dtype=np.float32))
        g = f.create_group("tinj")            # 8 beams, 1 N m each
        g.create_dataset("xdata", data=t)
        g.create_dataset("ydata", data=np.full((8, n), 1.0, dtype=np.float32))
        g = f.create_group("ech_power")
        g.create_dataset("xdata", data=np.array([0.0], dtype=np.float32)
                         if ech_absent else t)
        y = np.full((12, 1 if ech_absent else n), 1.0e5, dtype=np.float32)
        if ech_nan_channel and not ech_absent:
            y[3] = np.nan
        g.create_dataset("ydata", data=y)
    return corpus


def test_beam_channels_are_summed_and_scaled_to_the_model_units(tmp_path):
    corpus = _fake_corpus(tmp_path)
    got, missing = rc.resolve(190000, ["pinj_total", "tinj_total"], corpus=corpus)
    assert missing == {}
    # 8 beams x 1 MW = 8e6 W = 8000 kW
    np.testing.assert_allclose(np.nanmax(got["pinj_total"].y), 8000.0, rtol=1e-5)
    np.testing.assert_allclose(np.nanmax(got["tinj_total"].y), 8.0, rtol=1e-5)
    assert got["pinj_total"].attrs["resolver"] == "corpus"
    assert got["pinj_total"].attrs["scale_to_canonical"] == "0.001"


def test_nan_channels_do_not_poison_the_total(tmp_path):
    corpus = _fake_corpus(tmp_path)
    got, _ = rc.resolve(190000, ["ech_power_total"], corpus=corpus)
    # 11 finite channels x 1e5 W; the NaN channel is skipped, not propagated
    np.testing.assert_allclose(np.nanmax(got["ech_power_total"].y), 11.0e5, rtol=1e-5)
    assert got["ech_power_total"].attrs["nan_channels"] == "1"


def test_series_are_decimated_to_the_declared_step(tmp_path):
    corpus = _fake_corpus(tmp_path)
    got, _ = rc.resolve(190000, ["pinj_total"], corpus=corpus)
    x = got["pinj_total"].x
    assert x.size == 2001                      # 2 s at the 1 ms declared step
    np.testing.assert_allclose(np.diff(x), 0.001)


def test_an_absent_group_is_recorded_not_raised(tmp_path):
    corpus = _fake_corpus(tmp_path, ech_absent=True)
    got, missing = rc.resolve(190000, ["pinj_total", "ech_power_total"], corpus=corpus)
    assert set(got) == {"pinj_total"}
    assert missing == {"ech_power_total": "SignalAbsent"}


def test_a_missing_shot_file_misses_everything(tmp_path):
    corpus = _fake_corpus(tmp_path)
    got, missing = rc.resolve(999999, ["pinj_total"], corpus=corpus)
    assert got == {}
    assert missing == {"pinj_total": "FileNotFoundError"}


def test_a_feature_with_no_corpus_source_is_refused(tmp_path):
    corpus = _fake_corpus(tmp_path)
    with pytest.raises(KeyError, match="corpus"):
        rc.resolve(190000, ["ne_zipfit"], corpus=corpus)


@pytest.mark.skipif(not CORPUS.exists(), reason=f"corpus not available: {CORPUS}")
def test_real_corpus_shot_resolves_all_three_actuator_totals():
    got, missing = rc.resolve(
        185945, ["pinj_total", "tinj_total", "ech_power_total"], corpus=CORPUS
    )
    assert missing == {}
    for name in ("pinj_total", "tinj_total", "ech_power_total"):
        assert got[name].y.shape[0] == 1
    # MEASURED grid origins on this shot: the beam groups start at t=0, and
    # ech_power starts a quarter-second early. Asserted per group rather than
    # as one disjunction, which passed whatever the origins turned out to be.
    assert got["pinj_total"].x[0] == pytest.approx(0.0, abs=1e-3)
    assert got["tinj_total"].x[0] == pytest.approx(0.0, abs=1e-3)
    assert got["ech_power_total"].x[0] == pytest.approx(-0.25, abs=1e-3)
    # 10 MW-class beam power on this shot, expressed in kW
    assert 5_000.0 < np.nanmax(got["pinj_total"].y) < 30_000.0


def test_the_scales_are_the_measured_ones():
    """pinj is W where the model wants kW; ECH is W in both sources.

    The ECH scale was settled by comparing the corpus channel sum against
    the archive's own total column `EC.PECH`, time-aligned onto the 25 ms
    grid over a RANDOM 60 overlap shots (n=5,186): global median ratio
    0.779, per-shot medians 0.444 to 1.406. That is order 1, not 1e-3 or
    1e3, so no scaling - but the two are NOT interchangeable, and the
    ~22% median shortfall is a content difference (the corpus's 12
    channels miss gyrotrons `EC.PECH` counts), not a unit one. The
    archive's `ech_pwr` is a single gyrotron and cannot answer the
    question at all; see the note on the `ech_power_total` spec.
    """
    assert rc.SCALE_TO_CANONICAL["pinj_total"] == 1e-3
    assert rc.SCALE_TO_CANONICAL["tinj_total"] == 1.0
    assert rc.SCALE_TO_CANONICAL["ech_power_total"] == 1.0


def test_every_summed_corpus_feature_has_a_scale():
    """A corpus feature with no entry would raise KeyError mid-run, per shot.

    `resolve` indexes SCALE_TO_CANONICAL directly on the summing path, so the
    failure would land in a bulk run rather than here. One assertion moves it
    to test time.

    Only the SUMMING path indexes it. A `waveform` feature is returned whole,
    a `profile` feature is kept per channel, and a `scalar` whose locator names
    a channel (`gas_raw#0`) is that channel alone: none of the three converts
    units, so each must be ABSENT from the dict rather than present with a 1.0.
    An entry there would say a scale had been decided for a path that has none.
    """
    from labeler.features import namespace as ns

    corpus_features = ns.by_source("corpus")

    def summing(spec):
        _, channel = rc.split_locator(spec.locator_for("corpus"))
        return spec.kind == "scalar" and channel is None

    summed = {s.name for s in corpus_features if summing(s)}
    unscaled = {s.name for s in corpus_features if not summing(s)}
    assert summed == set(rc.SCALE_TO_CANONICAL), (
        f"corpus features without a scale: {summed - set(rc.SCALE_TO_CANONICAL)}; "
        f"scales for non-corpus features: {set(rc.SCALE_TO_CANONICAL) - summed}"
    )
    assert {s.name for s in corpus_features if s.kind == "waveform"}, (
        "expected at least one waveform corpus feature (co2)"
    )
    assert unscaled and not (unscaled & set(rc.SCALE_TO_CANONICAL))


def test_a_time_with_no_finite_channel_is_unknown_not_zero(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    n = 2000
    t = np.linspace(0.0, 2.0, n, dtype=np.float32)
    y = np.full((12, n), 1.0e5, dtype=np.float32)
    y[:, 1000:1100] = np.nan                  # every gyrotron dark, briefly
    with h5py.File(corpus / "190000_processed.h5", "w") as f:
        g = f.create_group("ech_power")
        g.create_dataset("xdata", data=t)
        g.create_dataset("ydata", data=y)
    got, _ = rc.resolve(190000, ["ech_power_total"], corpus=corpus)
    out = got["ech_power_total"]
    dark = (out.x >= t[1000]) & (out.x < t[1099])
    assert dark.any()
    assert np.isnan(out.y[0][dark]).all()     # not 0.0, which nansum returns
    assert got["ech_power_total"].attrs["nan_channels"] == "0"


def test_a_truncated_corpus_file_is_recorded_not_raised(tmp_path):
    # About 0.7% of the corpus is truncated on disk (2 of a random 300,
    # 186419 and 186800 among them); h5py raises OSError on open.
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "190000_processed.h5").write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)
    got, missing = rc.resolve(190000, ["pinj_total", "tinj_total"], corpus=corpus)
    assert got == {}
    assert missing == {"pinj_total": "OSError", "tinj_total": "OSError"}


def test_a_group_that_is_not_channels_by_time_is_recorded(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    with h5py.File(corpus / "190000_processed.h5", "w") as f:
        g = f.create_group("pinj")            # 1-D: no channel axis
        g.create_dataset("xdata", data=np.linspace(0.0, 2.0, 100, dtype=np.float32))
        g.create_dataset("ydata", data=np.full(100, 1.0e6, dtype=np.float32))
        g = f.create_group("tinj")            # x and y disagree in length
        g.create_dataset("xdata", data=np.linspace(0.0, 2.0, 100, dtype=np.float32))
        g.create_dataset("ydata", data=np.full((8, 99), 1.0, dtype=np.float32))
    got, missing = rc.resolve(190000, ["pinj_total", "tinj_total"], corpus=corpus)
    assert got == {}
    assert missing == {"pinj_total": "ShapeError(ndim=1)", "tinj_total": "ShapeError"}


# --- the `co2` waveform feature (task 7c) ------------------------------------


def _fake_co2(tmp_path, *, n=200_000, channels=4, absent=False):
    """A synthetic corpus file carrying a `co2` group at the real 500 kHz.

    Small on purpose - 200,000 samples is 0.4 s, not the corpus' 4.5e6 - so
    the test is about the SHAPE of the path, not about moving 72 MB.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    if absent:
        t = np.array([0.0], dtype=np.float32)
        y = np.zeros((channels, 1), dtype=np.float32)
    else:
        t = (-1.45 + np.arange(n, dtype=np.float64) / 5.0e5).astype(np.float32)
        y = np.stack([
            np.sin(2.0 * np.pi * (120.0 + 5.0 * c) * 1e3 * t.astype(np.float64))
            for c in range(channels)
        ]).astype(np.float32)
    with h5py.File(corpus / "199000_processed.h5", "w") as f:
        g = f.create_group("co2")
        g.create_dataset("xdata", data=t)
        g.create_dataset("ydata", data=y)
    return corpus


def test_a_waveform_is_returned_whole_never_summed_or_decimated(tmp_path):
    corpus = _fake_co2(tmp_path)
    got, missing = rc.resolve(199000, ["co2"], corpus=corpus)
    assert missing == {}
    arr = got["co2"]
    # Four channels, every sample, native rate: nothing here reduces it.
    assert arr.y.shape == (4, 200_000)
    assert arr.y.dtype == np.float32
    assert arr.x.size == 200_000
    assert arr.attrs["resolver"] == "corpus"
    assert arr.attrs["locator"] == "co2"
    assert arr.attrs["n_channels"] == "4"
    assert arr.attrs["native_rate"] == "1"
    assert arr.attrs["scale_to_canonical"] == "1.0"
    assert "decimated_to_s" not in arr.attrs


def test_the_waveform_sample_rate_is_read_off_the_span(tmp_path):
    # From a median diff of the float32 `xdata` this reads 524,288 Hz; from
    # the span it reads 500 kHz, which is the number the band depends on.
    corpus = _fake_co2(tmp_path)
    got, _ = rc.resolve(199000, ["co2"], corpus=corpus)
    assert float(got["co2"].attrs["sample_rate_hz"]) == pytest.approx(5.0e5, rel=1e-4)


def test_an_absent_co2_group_is_the_standard_miss_not_an_error(tmp_path):
    corpus = _fake_co2(tmp_path, absent=True)
    got, missing = rc.resolve(199000, ["co2"], corpus=corpus)
    assert got == {}
    assert missing == {"co2": "SignalAbsent"}


def test_a_shot_with_no_corpus_file_misses_co2_like_anything_else(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    got, missing = rc.resolve(199000, ["co2"], corpus=corpus)
    assert got == {}
    assert missing == {"co2": "FileNotFoundError"}


# ---- channel-selected and per-channel features (d3d_elm_time_to_event_dsm) ----


def _elm_corpus(tmp_path, *, co2_absent=False, ece_channels=48):
    """A corpus file holding the groups the ELM model's features read."""
    corpus = tmp_path / "elm_corpus"
    corpus.mkdir()
    n = 2000                                  # 2 s at 1 kHz, for speed
    t = np.linspace(0.0, 2.0, n, dtype=np.float32)
    with h5py.File(corpus / "190000_processed.h5", "w") as f:
        g = f.create_group("gas_raw")         # 11 valves; only channel 0 is `gas`
        g.create_dataset("xdata", data=t)
        y = np.zeros((11, n), dtype=np.float32)
        y[0] = 0.5
        y[1] = 99.0                           # a decoy a channel sum would eat
        g.create_dataset("ydata", data=y)
        g = f.create_group("ece")
        g.create_dataset("xdata", data=t)
        g.create_dataset("ydata", data=np.arange(
            ece_channels, dtype=np.float32)[:, None] * np.ones((1, n), np.float32))
        g = f.create_group("co2")
        g.create_dataset("xdata", data=np.array([0.0], dtype=np.float32)
                         if co2_absent else t)
        chords = np.array([1.0e14, 2.0e14, 3.0e14, 4.0e14], dtype=np.float32)
        g.create_dataset("ydata", data=chords[:, None] * np.ones(
            (1, 1 if co2_absent else n), np.float32))
    return corpus


def test_a_channel_selected_feature_takes_that_channel_and_never_the_sum(tmp_path):
    corpus = _elm_corpus(tmp_path)
    got, missing = rc.resolve(190000, ["gas", "co2_v2"], corpus=corpus)
    assert missing == {}
    np.testing.assert_allclose(np.nanmax(got["gas"].y), 0.5, rtol=1e-5)
    np.testing.assert_allclose(np.nanmax(got["co2_v2"].y), 3.0e14, rtol=1e-5)
    assert got["gas"].attrs["locator"] == "gas_raw#0"
    assert got["gas"].attrs["n_channels"] == "1"
    assert got["gas"].attrs["scale_to_canonical"] == "1.0"


def test_a_profile_feature_keeps_every_channel_in_order(tmp_path):
    corpus = _elm_corpus(tmp_path)
    got, _ = rc.resolve(190000, ["ece"], corpus=corpus)
    y = got["ece"].y
    assert y.shape == (48, 2001)               # 2 s at the 1 ms declared step
    np.testing.assert_allclose(y[:, 0], np.arange(48.0))
    assert got["ece"].attrs["locator"] == "ece"


def test_a_locator_naming_a_channel_the_group_lacks_is_a_per_shot_miss(tmp_path):
    corpus = _elm_corpus(tmp_path)
    with h5py.File(corpus / "190000_processed.h5", "a") as f:
        del f["co2"]
        g = f.create_group("co2")              # only two chords this shot
        g.create_dataset("xdata", data=np.linspace(0.0, 2.0, 100, dtype=np.float32))
        g.create_dataset("ydata", data=np.ones((2, 100), dtype=np.float32))
    got, missing = rc.resolve(190000, ["co2_r0", "co2_v3"], corpus=corpus)
    assert set(got) == {"co2_r0"}
    assert missing == {"co2_v3": "ChannelMissing(3/2)"}


def test_an_absent_co2_group_misses_every_chord(tmp_path):
    corpus = _elm_corpus(tmp_path, co2_absent=True)
    names = ["co2_r0", "co2_v1", "co2_v2", "co2_v3", "gas"]
    got, missing = rc.resolve(190000, names, corpus=corpus)
    assert set(got) == {"gas"}
    assert set(missing) == {"co2_r0", "co2_v1", "co2_v2", "co2_v3"}
    assert set(missing.values()) == {"SignalAbsent"}
