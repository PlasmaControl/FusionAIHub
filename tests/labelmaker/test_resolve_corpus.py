"""The corpus resolver: channel sums, units, decimation, absent groups."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from labelmaker.features import resolve_corpus as rc

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


def test_every_corpus_sourced_feature_has_a_scale():
    """A corpus feature with no entry would raise KeyError mid-run, per shot.

    `resolve` indexes SCALE_TO_CANONICAL directly, so the failure would land
    in a bulk run rather than here. One assertion moves it to test time.
    """
    from labelmaker.features import namespace as ns

    declared = {spec.name for spec in ns.by_source("corpus")}
    assert declared == set(rc.SCALE_TO_CANONICAL), (
        f"corpus features without a scale: {declared - set(rc.SCALE_TO_CANONICAL)}; "
        f"scales for non-corpus features: {set(rc.SCALE_TO_CANONICAL) - declared}"
    )


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
