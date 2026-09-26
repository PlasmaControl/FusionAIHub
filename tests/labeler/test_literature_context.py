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


@pytest.mark.parametrize(
    "paper, context",
    [
        ("NSTX-U NSTX-U DIII-D", "during shot 204100"),
        ("NSTX NSTX DIII-D", "during shot 204100"),
        ("NSTX-U NSTX DIII-D", "shots 203648, 203681"),
        ("NSTX-U NSTX-U DIII-D", "discharges 204098-204101"),
        ("LHD LHD DIII-D", "In shot 189894, a Li granule"),
        ("LHD LHD DIII-D", "#186636 no IPD #186638"),
        ("NSTX-U NSTX LHD LHD DIII-D", "during shot 204100"),
        ("NSTX-U LHD LHD DIII-D", "In shot 189894, a Li granule"),
        ("NSTX–U NSTX–U DIII–D", "during shot 204100"),
    ],
)
def test_paper_machine_owns_numbers_far_from_device_names(paper, context):
    text = paper + ". " + "Plasma behaviour is discussed here. " * 3 + context
    assert _found(text) == []


def test_paper_machine_counts_names_after_the_number_too():
    text = "during shot 204100. " + "Plasma behaviour is discussed here. " * 3
    assert _found(text + "NSTX-U NSTX-U DIII-D") == []


@pytest.mark.parametrize("machine", ["NSTX-U", "LHD"])
@pytest.mark.parametrize(
    "context",
    ["the DIII-D reference shot 190904", "reference shot 190904 on DIII-D"],
)
def test_paper_machine_yields_to_a_device_name_in_reach(machine, context):
    text = f"{machine} {machine} {machine}. "
    text += "Plasma behaviour is discussed here. " * 3 + context
    assert _found(text) == [(190904, "exact")]


@pytest.mark.parametrize(
    "paper",
    [
        "DIII-D DIII-D NSTX-U",
        "DIII-D NSTX-U",
        "DIII-D LHD",
        "DIII-D DIII-D NSTX-U NSTX LHD LHD",
        "DIII-D nstx-u nstx-u lhd lhd",
        "DIII-D xNSTX-Ux xNSTXx xLHDx xLHDx",
        "diii - d DIII–D NSTX-U",
    ],
)
def test_paper_default_stays_diii_d_without_a_larger_other_count(paper):
    text = paper + ". " + "Plasma behaviour is discussed here. " * 3
    assert _found(text + "discharge #204184") == [(204184, "exact")]


@pytest.mark.parametrize("machine", ["KSTAR", "TCV", "JET"])
def test_paper_default_ignores_machines_with_other_shot_numbering(machine):
    text = f"{machine} {machine} {machine}. "
    text += "the DIII-D cases with standard (#198955). "
    text += "Plasma behaviour is discussed here. " * 3 + "#191366 t = 1774.9 ms"
    assert _found(text) == [(198955, "exact"), (191366, "exact")]


def test_only_the_shots_asked_for():
    assert _found("shots 189600-189603", {189602, 200000}) == [(189602, "range")]


def test_context_is_the_normalised_text_in_reach():
    (m,) = mentions("In   DIII-D\nshot 189631 the mode locked.")
    assert m.context == "In DIII-D shot 189631 the mode locked."
    assert normalise("a­b \t c") == "ab c"
