"""Ground truth for one shot, and scoring a published label against it.

The tearing archive is the only ground truth labelmaker has, and it carries no
time axis of its own - its rows are placed on our grid by the same match
`validate` uses to score a pool. These helpers expose that per shot, so
`analyze` can say how a label did on the shot in front of you.
"""
import numpy as np
import pytest

from labelmaker import validate
from labelmaker.config import Paths

N = 6


class _FakeBuilt:
    def __init__(self):
        self.t = 0.025 * np.arange(8, dtype=np.float64)
        self.scalars = np.zeros((8, 9))
        self.profiles = np.zeros((8, 33, 0))
        self.valid = np.ones(8, bool)
        self.missing = ()
        self.resolvers = {}
        self.invalid_reasons = {}


def _fake_match(monkeypatch, tm_label, betan=None, index=(0, 1, 2, 4, 5, 6)):
    """Six archived rows landing on our timesteps `index`."""
    index = np.asarray(index)
    y = np.stack([
        np.asarray(betan if betan is not None else np.full(N, 2.0), dtype=float),
        np.asarray(tm_label, dtype=float),
    ], axis=1)
    got = {"x0": np.zeros((N, 9)), "x1": np.zeros((N, 33, 0)), "y": y,
           "rows": np.arange(N)}
    info = {"index": index, "passed": True, "n_archived_rows": N, "n_matched": N,
            "n_unique_matched": N, "median_distance": 0.0, "fail_reason": None}

    def fake(shot, spec, paths, archive):
        if shot != 190000:
            return validate._ShotMatch(skip_reason="no archived rows")
        return validate._ShotMatch(got=got, built=_FakeBuilt(), info=info, features={})

    monkeypatch.setattr(validate, "_matched_shot", fake)


def test_archived_truth_puts_the_archive_rows_on_our_time_axis(monkeypatch):
    _fake_match(monkeypatch, tm_label=[0, 0, 0, 1, 1, 0], betan=[1.0, 1.5, 2.0, 2.5, 3.0, 2.0])
    got = validate.archived_truth(190000, Paths.from_env())
    assert got["available"] is True
    np.testing.assert_array_equal(got["index"], [0, 1, 2, 4, 5, 6])
    np.testing.assert_allclose(got["t"], [0.0, 0.025, 0.05, 0.1, 0.125, 0.15])
    np.testing.assert_array_equal(got["tm_label"], [False, False, False, True, True, False])
    np.testing.assert_allclose(got["betan"], [1.0, 1.5, 2.0, 2.5, 3.0, 2.0])
    # the first row the archive calls a mode
    assert got["onset_s"] == pytest.approx(0.1)
    assert got["n_rows"] == 6


def test_archived_truth_reports_no_onset_when_the_label_never_turns_on(monkeypatch):
    _fake_match(monkeypatch, tm_label=[0] * N)
    got = validate.archived_truth(190000, Paths.from_env())
    assert got["available"] is True and got["onset_s"] is None


def test_archived_truth_says_why_a_shot_has_none(monkeypatch):
    _fake_match(monkeypatch, tm_label=[0] * N)
    got = validate.archived_truth(199597, Paths.from_env())
    assert got["available"] is False and got["reason"] == "no archived rows"
    assert "t" not in got


def test_a_binary_label_is_scored_against_the_archived_label_with_its_lead_time(monkeypatch):
    _fake_match(monkeypatch, tm_label=[0, 0, 0, 1, 1, 0])
    truth = validate.archived_truth(190000, Paths.from_env())
    # a label that crosses 0.5 two steps before the archived onset at 0.100 s
    y = np.array([0.1, 0.2, 0.6, 0.9, 0.8, 0.7, 0.2, 0.1])
    got = validate.score_against_truth(
        "d3d_tearing_onset_cnn1d/tm_prob", y, np.ones(8, bool), truth, threshold=0.5
    )
    assert got["scored"] is True and got["kind"] == "archived label"
    assert got["n"] == 6 and got["n_positive"] == 2
    assert got["auroc"] == pytest.approx(1.0)      # both positives rank above every negative
    # at 0.5 the label also flags the 0.6 row, which the archive calls quiet
    assert got["recall"] == pytest.approx(1.0) and got["precision"] == pytest.approx(2 / 3)
    assert got["onset_s"] == pytest.approx(0.1)
    assert got["first_above_threshold"] == pytest.approx(0.05)
    assert got["lead_time_s"] == pytest.approx(0.05)


def test_a_regression_label_is_scored_against_its_archived_column(monkeypatch):
    _fake_match(monkeypatch, tm_label=[0] * N, betan=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    truth = validate.archived_truth(190000, Paths.from_env())
    y = np.full(8, 1.5)
    got = validate.score_against_truth(
        "d3d_tearing_onset_cnn1d/betan", y, np.ones(8, bool), truth, threshold=None
    )
    assert got["scored"] is True and got["kind"] == "archived column"
    assert got["rmse"] == pytest.approx(0.5) and got["bias"] == pytest.approx(0.5)
    assert got["n"] == 6 and "auroc" not in got


def test_a_survival_label_is_scored_on_pre_onset_rows_within_its_horizon(monkeypatch):
    # onset at 0.100 s; rows at 0.000, 0.025, 0.050 are pre-onset. The
    # horizon is overridden to 50 ms so the boundary falls inside this
    # six-row fixture; a caller passes the label's own horizon.
    _fake_match(monkeypatch, tm_label=[0, 0, 0, 1, 1, 0])
    truth = validate.archived_truth(190000, Paths.from_env())
    y = np.array([0.05, 0.1, 0.3, 0.9, 0.9, 0.9, 0.1, 0.1])
    got = validate.score_against_truth(
        "d3d_tearing_time_to_event_dsm/tm_risk_250ms", y, np.ones(8, bool), truth,
        threshold=0.2, horizon_s=0.05,
    )
    assert got["scored"] is True and got["kind"] == "onset within 0.05 s"
    # rows at 0.050 (onset - t = 0.050 <= 0.05) is positive; 0.000 and 0.025 are not
    assert got["n"] == 3 and got["n_positive"] == 1
    assert got["auroc"] == pytest.approx(1.0)
    assert got["lead_time_s"] == pytest.approx(0.05)


def test_a_label_the_archive_cannot_serve_is_left_unscored(monkeypatch):
    _fake_match(monkeypatch, tm_label=[0] * N)
    truth = validate.archived_truth(190000, Paths.from_env())
    got = validate.score_against_truth("fake_model/score", np.zeros(8), np.ones(8, bool),
                                       truth, threshold=0.5)
    assert got["scored"] is False and "fake_model/score" in got["reason"]


def test_scoring_an_unavailable_truth_repeats_its_reason(monkeypatch):
    _fake_match(monkeypatch, tm_label=[0] * N)
    truth = validate.archived_truth(199597, Paths.from_env())
    got = validate.score_against_truth("d3d_tearing_onset_cnn1d/tm_prob", np.zeros(8),
                                       np.ones(8, bool), truth, threshold=0.5)
    assert got["scored"] is False and got["reason"] == "no archived rows"


def test_only_rows_labelmaker_would_publish_are_scored(monkeypatch):
    """A row the validity mask rejects is not a prediction labelmaker stands
    behind, so it must not count for or against the model."""
    _fake_match(monkeypatch, tm_label=[0, 0, 0, 1, 1, 0])
    truth = validate.archived_truth(190000, Paths.from_env())
    y = np.array([0.1, 0.2, 0.6, 0.9, 0.8, 0.7, 0.2, 0.1])
    valid = np.ones(8, bool)
    valid[4] = False                                   # drops one positive row
    got = validate.score_against_truth(
        "d3d_tearing_onset_cnn1d/tm_prob", y, valid, truth, threshold=0.5
    )
    assert got["n"] == 5 and got["n_positive"] == 1
    assert got["n_invalid_dropped"] == 1


def test_reaching_for_the_truth_never_raises(monkeypatch):
    """The archive may be unmounted, or hold rows this shot's match cannot
    place. Either way the caller keeps its analysis and gets a reason."""
    def boom(shot, spec, paths, archive):
        raise OSError("archive not mounted")

    monkeypatch.setattr(validate, "_matched_shot", boom)
    got = validate.archived_truth(190000, Paths.from_env())
    assert got["available"] is False and "not mounted" in got["reason"]
