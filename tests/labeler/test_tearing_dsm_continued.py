"""d3d_tearing_time_to_event_dsm_continued: the base model's contract, other weights.

The variant exists to be compared against the checkpoint it continues, so what
these tests guard is that nothing except the weight file differs: the same
input spec object, the same 20 output columns, the same loader code, the same
card contract. A copied `load` would satisfy the first two and drift on the
third, so the identity of the code is asserted too.
"""
import pickle

import numpy as np

from labeler.models import registry
from labeler.models.base import BuiltInputs
from labeler.models.d3d_tearing_time_to_event_dsm import spec as base
from labeler.models.d3d_tearing_time_to_event_dsm_continued import spec as cont


def _built(n=7, seed=0):
    rng = np.random.default_rng(seed)
    scalars = np.abs(rng.normal(size=(n, 14)))
    profiles = np.abs(rng.normal(size=(n, 33, 6))) + 1.0
    return BuiltInputs(t=0.025 * np.arange(n), scalars=scalars, profiles=profiles,
                       valid=np.ones(n, bool), missing=(), resolvers={})


def test_the_variant_reuses_the_base_specs_and_names_its_own_weights():
    assert cont.SLUG == "d3d_tearing_time_to_event_dsm_continued"
    assert cont.CARD_ID == "plasmacontrol/d3d-tearing-time-to-event-dsm-continued"
    assert cont.INPUT_SPEC is base.INPUT_SPEC
    assert cont.OUTPUT_SPEC is base.OUTPUT_SPEC
    assert cont.preprocess is base.preprocess
    assert cont.HORIZONS_MS == base.HORIZONS_MS
    assert cont.ARTIFACTS == ("rt_fixed_rot_continued.pkl", "rt_normalizations_dict.pkl")
    assert cont.ARTIFACTS[0] != base.ARTIFACTS[0]
    assert cont.ARTIFACTS[1] == base.ARTIFACTS[1]     # same normalisation constants
    assert cont.ADAPTER.time_step_ms == base.ADAPTER.time_step_ms == 25.0
    assert cont.ADAPTER.ensemble_n == 1
    assert cont.ADAPTER.framework == "dsm_pickle"
    assert str(cont.UPSTREAM).endswith(cont.SLUG)
    # The continuation ran on the same rows, so it inherits the same training
    # shots - not a copy of the list, the same object.
    assert cont.ADAPTER.training_shots is base.ADAPTER.training_shots


def test_card_matches_the_spec_and_declares_the_same_columns_as_the_base_card():
    assert registry.card_discrepancies(cont.SLUG) == []
    assert cont.SLUG in registry.implemented()
    got = registry.read_card(cont.SLUG)["labelmaker"]
    want = registry.read_card(base.SLUG)["labelmaker"]
    assert got["outputs"] == want["outputs"]
    assert got["inputs"] == want["inputs"]
    assert got["upstream"]["training_code"] == "scripts/labeler/retrain_tearing_dsm.py"


def test_load_reads_the_continued_weights_and_leaves_isotonic_columns_nan(
        tmp_path, monkeypatch):
    """Same predictions as the base loader on the same graph, from a file of
    the variant's own name, with no calibration.json in the directory.

    The shipped model directory does now hold one - `validate` runs
    `calibration_study` for this slug too - so this is the no-map path, not a
    statement that the variant never publishes isotonic columns.
    """
    from .test_dsm_pickle import _fake_pickle

    # _fake_pickle seeds every fake torch model with a fixed seed.
    path, _ = _fake_pickle(tmp_path, monkeypatch)
    path.rename(tmp_path / cont.ARTIFACTS[0])
    (tmp_path / cont.ARTIFACTS[1]).write_bytes(pickle.dumps({}))
    x = np.random.default_rng(17).normal(size=(7, 38))
    # `make_load` closes over the base module's globals, which is the point:
    # patching the base module's `preprocess` reaches the variant's predictor.
    monkeypatch.setattr(base, "preprocess", lambda built, norm: x)

    predict = cont.load(tmp_path)
    got = predict(_built())
    assert got.shape == (1, 7, 20)
    assert np.isfinite(got[0, :, :17]).all()
    assert np.isnan(got[0, :, 17:]).all()
    assert predict.output_spec is base.OUTPUT_SPEC

    # Byte-identical to what the base loader gives for the same graph: only
    # the file name it reads differs.
    (tmp_path / base.ARTIFACTS[0]).write_bytes((tmp_path / cont.ARTIFACTS[0]).read_bytes())
    np.testing.assert_array_equal(got, base.load(tmp_path)(_built()))


def test_a_missing_continued_weight_file_is_an_error_not_a_fallback(tmp_path):
    """Naming the base model's file in the variant's directory must not load."""
    (tmp_path / "rt_fixed_rot.pkl").write_bytes(pickle.dumps([]))
    (tmp_path / cont.ARTIFACTS[1]).write_bytes(pickle.dumps({}))
    try:
        cont.load(tmp_path)
    except (FileNotFoundError, OSError):
        return
    raise AssertionError("loading without rt_fixed_rot_continued.pkl should fail")


def test_the_variant_is_scored_against_the_same_archived_truth_as_the_base():
    """Without these entries the variant's published labels are unscoreable:
    `_pooled_onset_rows` only reads labels that have an `ARCHIVE_TRUTH` rule."""
    from labeler import validate

    for name in ("tm_risk_250ms", "tm_risk_500ms", "tm_risk_1s", "tm_time_p50"):
        assert (validate.ARCHIVE_TRUTH[f"{cont.SLUG}/{name}"]
                == validate.ARCHIVE_TRUTH[f"{base.SLUG}/{name}"])
