"""The context rule: a shot number counts only with the words that say so."""

from __future__ import annotations

import pytest

from labeler.literature.context import REACH, mentions, normalise


def _found(text, shots=None):
    return [(m.shot, m.match_type) for m in mentions(text, shots)]


@pytest.mark.parametrize(
    "text",
    [
        "In DIII-D shot 189631 the mode locked.",
        "The 189631 discharge disrupted.",
        "See #189631.",
        "DIII–D 189631",  # an en dash, as pypdf often gives
        "dis-\ncharge 189631",  # a word broken across a line
    ],
)
def test_a_number_with_a_keyword_in_reach_counts(text):
    assert _found(text) == [(189631, "exact")]


@pytest.mark.parametrize(
    "text",
    [
        "The grant 189631 paid for it.",
        "snapshot 189631",  # 'shot' must start a word
        "shot 1896310",  # seven digits
        "shot 0.189631",
        "shot 189631.5",
    ],
)
def test_no_keyword_or_no_standalone_number_does_not(text):
    assert _found(text) == []


def test_reach_is_sixty_characters_either_way():
    assert REACH == 60
    assert _found("shot" + "x" * 60 + "189631") == [(189631, "exact")]
    assert _found("shot" + "x" * 61 + "189631") == []
    assert _found("189631" + "x" * 59 + " shot") == [(189631, "exact")]
    assert _found("189631" + "x" * 60 + " shot") == []


def test_a_run_of_numbers_shares_its_context():
    run = ", ".join(str(s) for s in range(189600, 189610))
    text = f"Shots {run} and 189615 were repeats."
    assert [s for s, _ in _found(text)] == [*range(189600, 189610), 189615]
    far = "shot 189600 " + "word " * 20 + "189700"
    assert _found(far) == [(189600, "exact")]


def test_ranges_expand_inside_their_ends():
    assert _found("discharges 189600-189603") == [
        (189600, "exact"),
        (189601, "range"),
        (189602, "range"),
        (189603, "exact"),
    ]
    assert _found("shots 189600 to 189602") == [
        (189600, "exact"),
        (189601, "range"),
        (189602, "exact"),
    ]
    assert _found("shots 189600–02") == [
        (189600, "exact"),
        (189601, "range"),
        (189602, "range"),
    ]


def test_wide_or_unclear_ranges_are_not_expanded():
    assert _found("shots 189600-189700") == [(189600, "exact"), (189700, "exact")]
    assert _found("shot 189631 to 40 ms") == [(189631, "exact")]
    assert _found("shot 189631-5") == [(189631, "exact")]  # one closing digit


def test_a_range_needs_a_keyword_too():
    assert _found("pages 189600-189603") == []


def test_another_machines_shot_does_not_count():
    assert _found("NSTX-U discharge 204112 disrupted.") == []
    assert _found("LHD shots 189600-189603") == []
    # the name before a number decides, even when one after it is nearer
    assert _found("DIII-D shot 190000 and NSTX-U shot 204112") == [(190000, "exact")]
    assert _found("NSTX-U differs; DIII-D shot 190000") == [(190000, "exact")]
    assert _found("shot 190000 on DIII-D, unlike LHD") == [(190000, "exact")]
    run = ", ".join(str(s) for s in range(190000, 190012))
    assert len(_found(f"DIII-D shots {run} and NSTX-U")) == 12  # a run is one


def test_only_the_shots_asked_for():
    assert _found("shots 189600-189603", {189602, 200000}) == [(189602, "range")]


def test_context_is_the_normalised_text_in_reach():
    (m,) = mentions("In   DIII-D\nshot 189631 the mode locked.")
    assert m.context == "In DIII-D shot 189631 the mode locked."
    assert normalise("a­b \t c") == "ab c"
