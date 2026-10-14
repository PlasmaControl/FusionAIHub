"""The tm_mhd review source's merge rule (round 5, decisions S5 as amended, S6, EQ4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from labeler.tearing import source as S

A, P, U, N = S.ABSENT, S.PRESENT, S.UNCERTAIN, S.NOT_OBSERVABLE


def tracks(*rows):
    """(family_id, status, label, why) per track; modelled when `why` is blank; n from
    the label (1 without one)."""
    return pd.DataFrame(
        [
            {
                "family_id": fid,
                "n": int(label.split("/")[1]) if "/" in label else 1,
                "status": status,
                "label": label,
                "why": why,
                "modelled": why == "",
                "p_res": 0.9,
            }
            for fid, status, label, why in rows
        ]
    )


def frames(spans, freq=None):
    """Frames every 4 ms over `{family_id: (t0, t1)}`, at `freq[family_id]` kHz (10)."""
    freq = freq or {}
    return pd.DataFrame(
        [
            {"family_id": fid, "t_ms": t, "f0": freq.get(fid, 10.0)}
            for fid, (a, b) in spans.items()
            for t in np.arange(a, b + 1, 4.0)
        ]
    )


def test_track_classes():
    T = tracks(
        (1, "m/n", "2/1", ""),
        (2, "m/n", "4/2", ""),
        (3, "m/n", "1/1", ""),
        (4, "m/n", "2/2", ""),
        (5, "not resonant", "", ""),
        (6, "no rotation", "", ""),
        (7, "q_min above", "", ""),
        (8, "", "", "no equilibrium slice within 30 ms"),
        (9, "", "", "harmonic"),
        (10, "", "", "line"),
        (11, "m/n", "5/2", ""),
        (12, "m/n", "4/1", ""),
        (13, "m/n", "7/3", ""),
        (14, "m/n", "5/3", ""),
        (15, "m/n", "3/2", ""),
        (16, "m/n", "4/3", ""),
        (17, "m/n", "3/1", ""),
    )
    assert list(S.track_classes(T)) == [
        "resonant",  # 2/1
        "resonant",  # 4/2, an island at q = 2
        "q1",
        "q1",
        "not_resonant",
        "unsure",
        "unsure",
        "unsure",
        "",
        "",
        "unsure",  # 5/2, 4/1, 7/3, 5/3: outside the named set
        "unsure",
        "unsure",
        "unsure",
        "resonant",
        "resonant",
        "resonant",
    ]
    detail = S.track_classes(T, detail=True)
    assert list(detail[10:14]) == ["unsure:label"] * 4
    assert detail[5] == "unsure" and detail[0] == "resonant"
    # an out-of-set label is uncertain, never absent and never present
    fr = frames({11: (100, 300), 3: (100, 300)})
    masks = S.track_masks(T, fr, 0, 400)
    got = S.merge(
        np.full(400, U),
        np.ones(400, bool),
        masks["present"],
        masks["seen"],
        masks["unsure"],
    )
    assert not masks["present"].any() and (got[100:300] == U).all()
    assert masks["cls"]["unsure:label"][100:300].all()


def test_n2_tracks_at_n_times_a_simultaneous_1_1_frequency_are_unsure():
    T = tracks(
        (1, "m/n", "1/1", ""),  # 8 kHz over 100-300
        (2, "m/n", "3/2", ""),  # 16.2 kHz: 2 x 8 within 0.3 kHz -> 1/1 harmonic
        (3, "m/n", "4/3", ""),  # 24 kHz over 100-500: 3 x 8 on 51 of 101 frames
        (4, "m/n", "3/2", ""),  # 17 kHz: 1 kHz off 2 x 8
        (5, "m/n", "2/1", ""),  # n = 1 at the 1/1's frequency: not tested
        (6, "m/n", "3/2", ""),  # 16 kHz over 100-700: 51 of 151 frames, under half
    )
    spans = {1: (100, 300), 2: (100, 300), 3: (100, 500), 4: (100, 300)}
    spans.update({5: (100, 300), 6: (100, 700)})
    freq = {1: 8.0, 2: 16.2, 3: 24.0, 4: 17.0, 5: 8.0, 6: 16.0}
    fr = frames(spans, freq)
    assert list(S.q1_harmonics(T, fr)) == [False, True, True, False, False, False]
    detail = S.track_classes(T, fr, detail=True)
    assert list(detail) == ["q1"] + ["unsure:harmonic"] * 2 + ["resonant"] * 3
    assert list(S.track_classes(T, fr)[1:3]) == ["unsure", "unsure"]
    assert list(S.track_classes(T)[1:3]) == ["resonant", "resonant"]  # no frames
    # The 1/1 alone resolves rule-uncertain time to absent; its harmonic keeps it
    # uncertain and makes nothing present.
    lo, hi = 0, 400
    rule, observed = np.full(hi, U), np.ones(hi, bool)
    pair = frames({1: (100, 300), 2: (100, 300)}, freq)
    for kept, expect in ((T.iloc[:2], U), (T.iloc[:1], A)):
        masks = S.track_masks(kept, pair[pair.family_id.isin(kept.family_id)], lo, hi)
        got = S.merge(rule, observed, masks["present"], masks["seen"], masks["unsure"])
        assert not masks["present"].any() and (got[110:290] == expect).all()


def test_rule_cells_precedence_and_points():
    rows = pd.DataFrame(
        {
            "category": [A, P, U, N, P],
            "t_start": [0.0, 20.0, 30.0, 45.0, 10.0],
            "t_end": [60.0, 50.0, 55.0, 48.0, 10.0],  # the last row is an onset point
        }
    )
    got = S.rule_cells(rows, 0, 60)
    assert (got[:20] == A).all()  # the point at 10 ms paints nothing
    assert (got[20:45] == P).all()  # present wins over the overlapping uncertain row
    assert (got[45:48] == N).all() and (got[48:50] == P).all()
    assert (got[50:55] == U).all() and (got[55:] == A).all()


def test_rule_uncertain_resolved_only_where_the_mhd_saw_only_1_1_or_not_resonant():
    lo, hi = 0, 1000
    rule = np.full(hi, U)
    rule[:100] = A
    T = tracks(
        (1, "m/n", "1/1", ""),  # 100-300: 1/1 only -> absent
        (2, "not resonant", "", ""),  # 300-400 -> absent
        (3, "uncertain", "", ""),  # 350-450 overlaps: stays uncertain
        (4, "m/n", "3/2", ""),  # 600-700 resonant >= 50 ms -> present
        (5, "m/n", "2/1", ""),  # 800-820: too short -> no present, stays uncertain
        (6, "m/n", "1/1", ""),  # 900-1000: 1/1 only, but not observed
    )
    fr = frames(
        {1: (100, 300), 2: (300, 400), 3: (350, 450), 4: (600, 700), 5: (800, 820)}
        | {6: (900, 1000)}
    )
    masks = S.track_masks(T, fr, lo, hi)
    observed = np.ones(hi, bool)
    observed[900:] = False
    got = S.merge(rule, observed, masks["present"], masks["seen"], masks["unsure"])
    everywhere = S.merge(
        rule, np.ones(hi, bool), masks["present"], masks["seen"], masks["unsure"]
    )
    assert (got[:100] == A).all()  # rule absent kept
    assert (got[110:340] == A).all()  # 1/1 and not-resonant cover
    assert (got[360:440] == U).all()  # an uncertain track
    assert (got[470:590] == U).all()  # no track
    assert (got[600:700] == P).all()
    assert (got[800:820] == U).all()  # a resonant track too short for a span
    assert (got[900:] == U).all()  # not observed: the 1/1 there does not count
    assert (everywhere[900:] == A).all()  # observed, the same 1/1 makes it absent
    assert (everywhere[:900] == got[:900]).all()
    assert np.isclose(masks["conf"][650], 0.9) and np.isnan(masks["conf"][500])


def test_resonant_spans_join_gaps_up_to_50_ms_and_beat_rule_absent():
    T = tracks((1, "m/n", "2/1", ""), (2, "m/n", "2/1", ""), (3, "m/n", "3/1", ""))
    # 100-130 and 170-200 join across 40 ms; 400-430 alone is too short.
    fr = frames({1: (100, 130), 2: (170, 200), 3: (400, 430)})
    masks = S.track_masks(T, fr, 0, 500)
    rule = np.full(500, A)
    got = S.merge(
        rule, np.ones(500, bool), masks["present"], masks["seen"], masks["unsure"]
    )
    assert (got[100:200] == P).all()
    assert (got[400:430] == A).all()  # rule absent stays absent under a short track


def test_without_rule_labels_the_mhd_decides():
    T = tracks((1, "m/n", "3/2", ""), (2, "no rotation", "", ""))
    fr = frames({1: (100, 200), 2: (300, 340)})
    masks = S.track_masks(T, fr, 0, 600)
    rule = np.full(600, S.NO_RULE)
    observed = np.ones(600, bool)
    observed[500:] = False
    got = S.merge(rule, observed, masks["present"], masks["seen"], masks["unsure"])
    assert (got[100:200] == P).all()
    assert (got[210:290] == A).all() and (got[300:340] == U).all()
    assert (got[500:] == U).all()


def test_observed_needs_mirnov_tree_slice_and_rotation():
    info = {"window": [0, 400], "mirnov_ms": [0, 400], "mirnov_gaps": [[100, 150]]}
    info["eq_t"] = [20.0, 40.0, 60.0, 300.0]
    got = S.observed_cells(info, True, "efit02", [0.0, 50.0, 300.0], 0, 400)
    assert got[50] and not got[120] and not got[200] and got[310]
    assert not S.observed_cells(info, True, None, [50.0], 0, 400).any()
    assert not S.observed_cells(info, False, "efit01", [50.0], 0, 400).any()


def test_resolved_bins_change_only_rule_uncertain_bins():
    rule = pd.DataFrame(
        {
            "category": [A, U, N, U],
            "t_start": [0.0, 100.0, 200.0, 300.0],
            "t_end": [400.0, 200.0, 300.0, 400.0],
        }
    )
    src = pd.DataFrame(
        {
            "category": [P, P, A, U],
            "t_start": [0, 100, 150, 300],
            "t_end": [100, 150, 300, 400],
        }
    )
    centres = np.arange(5.0, 400.0, 10.0)
    y, valid = S.resolved_bins(rule, src, centres)
    base_y, base_valid = S.scoring.label_bins(rule, centres)
    first = centres < 100  # rule absent: the source's present does not change it
    assert (y[first] == 0).all() and valid[first].all()
    assert (y[(centres > 100) & (centres < 150)] == 1).all()
    assert (y[(centres > 150) & (centres < 200)] == 0).all()
    assert valid[(centres > 100) & (centres < 200)].all()
    assert not valid[(centres > 200)].any()  # not observable, still uncertain
    assert (y[~base_valid & ~valid] == base_y[~base_valid & ~valid]).all()


def test_rule_locked_phases_are_never_made_absent():
    rows = pd.DataFrame(
        {
            "category": [A, U, U, U],
            "t_start": [0.0, 100.0, 200.0, 300.0],
            "t_end": [100.0, 200.0, 300.0, 400.0],
            "attrs": [
                np.nan,
                '{"n": 1, "reason": "confirmed_locked_phase"}',
                '{"n": 1, "reason": "post_collapse_lock_unknown"}',
                '{"n": 2, "reason": "coherent_sub_seed"}',
            ],
        }
    )
    lo, hi = 0, 400
    rule, locked = S.rule_cells(rows, lo, hi), S.locked_cells(rows, lo, hi)
    assert locked[100:300].all() and not locked[:100].any() and not locked[300:].any()
    assert list(S.reasons(rows)) == [
        "",
        "confirmed_locked_phase",
        "post_collapse_lock_unknown",
        "coherent_sub_seed",
    ]
    assert "locked_unseeded" in S.LOCKED_REASONS  # the rule's own list
    T = tracks((1, "m/n", "1/1", ""), (2, "m/n", "2/1", ""))
    fr = frames({1: (0, 400), 2: (240, 300)})  # 1/1 throughout, a 2/1 over 240-300
    masks = S.track_masks(T, fr, lo, hi)
    observed = np.ones(hi, bool)
    args = (rule, observed, masks["present"], masks["seen"], masks["unsure"])
    got = S.merge(*args, locked=locked)
    assert (got[:100] == A).all()
    assert (got[100:238] == U).all()  # a locked phase under the 1/1 alone
    assert (got[240:300] == P).all()  # a resonant span still makes it present
    assert (got[305:] == A).all()  # a sub-seed line under the 1/1 alone
    assert (S.merge(*args)[100:238] == A).all()  # without the locked mask


def test_without_rule_labels_unread_mirnov_time_is_not_observable():
    info = {"window": [0, 600], "mirnov_ms": [50, 600], "mirnov_gaps": [[300, 340]]}
    read = S.read_cells(info, True, 0, 600)
    assert not read[:50].any() and read[50:300].all() and not read[300:340].any()
    assert read[340:].all() and not S.read_cells(info, False, 0, 600).any()
    T = tracks((1, "m/n", "3/2", ""))
    fr = frames({1: (280, 380)})  # a resonant span across the gap
    masks = S.track_masks(T, fr, 0, 600)
    observed = read.copy()
    observed[500:] = False  # read, but no rotation profile
    args = (observed, masks["present"], masks["seen"], masks["unsure"])
    got = S.merge(np.full(600, S.NO_RULE), *args, read=read)
    assert (got[:50] == N).all()  # before the record's finite stretch
    assert (got[300:340] == N).all()  # a NaN gap, under a resonant span too
    assert (got[280:300] == P).all() and (got[340:380] == P).all()
    assert (got[50:270] == A).all() and (got[390:500] == A).all()
    assert (got[500:] == U).all()  # read, not observed
    rule = np.full(600, S.NO_RULE)
    rule[:50] = U  # a rule label keeps its state where the Mirnov was not read
    assert (S.merge(rule, *args, read=read)[:50] == U).all()


def test_table_rows_are_runs_of_state_and_confidence():
    state = np.array([A] * 3 + [P] * 4 + [U] * 2)
    conf = np.array([np.nan] * 3 + [0.91234, 0.9123, 0.8, 0.8] + [np.nan] * 2)
    assert S.table_rows(7, 100, state, conf) == [
        [7, A, 100, 103, ""],
        [7, P, 103, 105, 0.912],  # equal at 3 decimals: one run
        [7, P, 105, 107, 0.8],
        [7, U, 107, 109, ""],
    ]
    assert S.table_rows(7, 100, [], []) == []
