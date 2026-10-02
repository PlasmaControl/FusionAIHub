import numpy as np
import pandas as pd

from labeler.confinement.elm import joint_intervals, onset_exposure, summarize


def _confinement():
    return pd.DataFrame(
        {
            "shot": [1, 1],
            "t_start": [0.0, 100.0],
            "t_end": [100.0, 200.0],
            "regimes": ["QH", "H"],
            "label": [1, 1],
        }
    )


def test_elm_unknown_uncertain_and_not_observable_are_not_absence():
    elm = pd.DataFrame(
        {
            "shot": [1, 1, 1],
            "t_start": [20.0, 50.0, 80.0],
            "t_end": [50.0, 80.0, 120.0],
            "category": [0, 2, 3],
        }
    )
    joint = joint_intervals(_confinement(), elm)
    summary = summarize(joint)
    qh = summary.loc[summary.regime == "QH"].iloc[0]
    assert qh.assessed_ms == 30
    assert qh.unknown_ms == 20 and qh.uncertain_ms == 30
    assert qh.not_observable_ms == 20


def test_qh_elm_conflict_is_exact_overlap_and_guards_do_not_bridge_gaps():
    elm = pd.DataFrame(
        {"shot": [1], "t_start": [90.0], "t_end": [110.0], "category": [1]}
    )
    joint = joint_intervals(_confinement(), elm)
    assert (
        joint.loc[(joint.regime == "QH") & (joint.elm_state == 1), "duration_ms"].sum()
        == 10
    )
    guarded = joint_intervals(_confinement(), elm, guard_ms=20)
    assert not (guarded.elm_state == 1).any()


def test_overlapping_review_states_are_excluded_without_double_count():
    elm = pd.DataFrame(
        {
            "shot": [1, 1],
            "t_start": [0.0, 25.0],
            "t_end": [100.0, 75.0],
            "category": [0, 1],
        }
    )
    joint = joint_intervals(_confinement().iloc[:1], elm)
    assert joint.duration_ms.sum() == 100
    assert joint.loc[joint.elm_state == 2, "duration_ms"].sum() == 50


def test_legacy_onsets_measure_event_rate_and_preserve_sample_gaps():
    confinement = _confinement().iloc[:1].assign(t_end=5.0)
    traces = [(1, np.array([0.0, 1.0, 4.0]), np.array([1, 0, 1]))]
    result = onset_exposure(confinement, traces)
    assert result.iloc[0].exposure_ms == 3
    assert result.iloc[0].onsets == 2
