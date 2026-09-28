"""The context rule: a shot number counts only with the words that say so."""

from __future__ import annotations

import pytest

from labeler.literature import context
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


@pytest.mark.parametrize(
    "text",
    [
        "#:~:text=Primary%20energy%20consumption%20in%20China%202018%2D2023",
        "shot %20189631",
        "shot %189631",
        "shot %189600-02",
    ],
)
def test_url_encoding_is_not_a_shot_or_range(text):
    assert _found(text) == []


@pytest.mark.parametrize(
    "citation",
    [
        "Phys. Rev. Lett. 127, 185001",
        "Phys.Rev.Lett.127,185001",
        "Phys. Rev. E 104, 205001",
        "Physical Review Letters 127, 185001",
        "Phys. Rev. Research 3, 193001",
        "Phys. Plasmas ... Phys. Rev. X 11, 190001",
        "Phys. Rev. 127, 185001",
        "Physical Review 127, 185001",
        "Phys. Rev. A 104, 190001",
        "Phys. Rev. B. 104, 190001",
        "Phys. Rev. C 104, 190001",
        "Phys. Rev. D. 104, 190001",
        "Phys. Rev. E. 104, 190001",
        "Phys. Rev. X. 11, 190001",
        "Phys. Rev. Applied 12, 190001",
        "Phys. Rev. Fluids 4, 190001",
        "Phys. Rev. Accel. Beams 24, 190001",
        "Phys.Rev.Accel.Beams1234,190001",
        "Physical\nReview\tLetters 127 , 185001",
    ],
)
def test_physical_review_article_numbers_are_not_shots(citation):
    assert _found("DIII-D " + citation) == []


def test_physical_review_article_is_not_named_by_the_next_reference():
    text = (
        "et al 2023 Phys. Rev. Lett. 131 195101 [3] Thome K E et al 2024 "
        "Overview of results from the 2023 DIII-D negative triangularity campaign"
    )
    assert _found(text) == []


def test_physical_review_article_number_does_not_start_a_range():
    assert _found("shot Phys. Rev. Lett. 131 195101-03") == []


@pytest.mark.parametrize("number", ["190001", "190001-03"])
@pytest.mark.parametrize(
    "citation",
    [
        "Physical Review Accel. Beams 1234, ",
        "Physical\n\tReview   Accel.\nBeams 1234 \t,\n ",
    ],
)
def test_article_lookback_covers_the_longest_normalised_prefix(citation, number):
    assert _found("DIII-D " + citation + number) == []


@pytest.mark.parametrize("number", ["190001", "190001-03"])
@pytest.mark.parametrize("before", [" ", "x"])
def test_article_lookback_accepts_glued_citations_at_its_edge(
    monkeypatch, number, before
):
    citation = "Physical Review Accel. Beams 1234 , "
    # Tighten the span to put the journal exactly at its edge, without the margin.
    monkeypatch.setattr(context, "_ARTICLE_SPAN", len(citation), raising=False)
    text = "DIII-D" + before + citation + number
    assert _found(text) == []


@pytest.mark.parametrize(
    "number, expected",
    [
        ("190001", [(190001, "exact")]),
        (
            "190001-03",
            [(190001, "exact"), (190002, "range"), (190003, "range")],
        ),
    ],
)
def test_article_lookback_does_not_exclude_shots_after_distant_citations(
    number, expected
):
    text = "Phys. Rev. Lett. 131 195103. " + "Other text. " * 10
    assert _found(text + "DIII-D shot " + number) == expected


def test_excluded_article_between_real_shots_keeps_both():
    text = "DIII-D shot 190000, Phys. Rev. Lett. 131 195103, 190002"
    assert _found(text) == [(190000, "exact"), (190002, "exact")]


@pytest.mark.parametrize(
    "text, last",
    [
        ("shot %189600-189602", 189602),
        ("shot Phys. Rev. Lett. 131 195101-195103", 195103),
    ],
)
def test_excluded_range_start_does_not_exclude_its_written_end(text, last):
    assert _found(text) == [(last, "exact")]


@pytest.mark.parametrize("first", ["%202018", "Phys. Rev. Lett. 131 195101"])
def test_excluded_number_cannot_give_a_run_its_context(first):
    text = "shot " + first + ", " * 35 + "190000"
    assert _found(text) == []


def test_excluded_article_number_leaves_the_real_shot():
    assert _found("shot 190000, Phys. Rev. Lett. 131 195101") == [(190000, "exact")]


def test_excluded_article_rule_keeps_a_genuine_shot_run():
    assert _found("DIII-D shots 195101 and 195140") == [
        (195101, "exact"),
        (195140, "exact"),
    ]


def test_excluded_article_rule_keeps_numbers_after_other_numbers():
    assert _found("shot 190000 at 3.3 s, 108 192275 to 192276") == [
        (190000, "exact"),
        (192275, "exact"),
        (192276, "exact"),
    ]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("shot Phys. Plasmas 108 192275", []),
        ("shot Phys. Rev. Lett. 12345, 192275", [(192275, "exact")]),
    ],
)
def test_excluded_article_rule_requires_the_journal_and_a_short_volume(text, expected):
    assert _found(text) == expected


TICKS = (
    "ems. To overcome these limitations, the DGPA model has been proposed [9], "
    "which integrates the expressive capacity of 2 160000 165000 170000 175000 "
    "180000 185000 190000 195000 Shot Numbers 0.01 0.02 0.03 0.04 0.05 0.06 0.07"
)
CONTRACT = (
    "248.80 per Item(s). No Building 6/10/2022 $2,248.80 $0.00 $2,248.80 6/8/22 "
    "Oleta Coach Lines Contract# I.C.C. MC-192798. 846.08 per 55 passenger bus. . "
    "1 Item(s) @ 846.08 per Item"
)
TABLE_HASH_HEADER = (
    "ection, perturbed divertor and main chamber gas species ( Γδ and Γbg), the "
    "fundamental frequency f 0, and modes lk. # BT Γδ Γbg f 0 [Hz] lk LSN 192043 "
    "unfav. N2 D2 1000/441 1, 3 192044 unfav. N2 D2 1000/441 1, "
)
GLUED_CITATION = (
    "get plates during type-i edge-localized modesPhys. Rev. Lett. 91 195003 "
    "[302] Knolker M., Evans T.E., Wingen A."
)
POSTAL = (
    "al University, Seoul, Republic of Korea 17 Ioffe Institute, St. Petersburg "
    "194021, Russia 18 Institution Project Center ITER, Rosatom, Moscow"
)


@pytest.mark.parametrize(
    "text, expected",
    [
        (TICKS, []),
        (CONTRACT, []),
        (TABLE_HASH_HEADER, [(192043, "exact")]),
        (GLUED_CITATION, []),
        ("DIII-D " + GLUED_CITATION, []),
        (POSTAL, []),
        (POSTAL + " DIII-D", []),
    ],
    ids=["ticks", "contract", "table", "glued", "glued_near", "postal", "postal_near"],
)
def test_d23_real_non_shots_and_table(text, expected):
    assert _found(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("195000 190000 185000 180000 175000 170000 165000 160000 Shot Numbers", []),
        ("Shot 190000 190500 191000", []),
        ("shots 190904, 160000, 165000, 170000", [190904]),
        ("shots 190001, 190002 and 190003", [190001, 190002, 190003]),
        ("shots 190001, 190003 and 190005", [190001, 190003, 190005]),
        ("DIII-D shots 190101, 190201 and 190301", [190101, 190201, 190301]),
        ("shots 190000, 195000", [190000, 195000]),
        ("shots 190000, 190099, 190198", [190000, 190099, 190198]),
        ("shots 190000, 190000, 190000", [190000, 190000, 190000]),
        ("shots 190000, 190500, 191000, 191001, 192000, 193000, 194000", [191001]),
        ("shots 190000, 190500, 191000-191002", [191002]),
        ("shot 190000, 195000, 200000" + ", " * 35 + "190001", []),
    ],
)
def test_d23_round_ticks_only(text, expected):
    assert _found(text) == [(s, "exact") for s in expected]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Contract# MC-192798", []),
        ("Contract# z-192798-99", []),
        ("Contract# MC-192798-192800", [(192800, "exact")]),
        ("#-192798", [(192798, "exact")]),
        ("shot -192798", [(192798, "exact")]),
        ("#192221-#192233", [(192221, "exact"), (192233, "exact")]),
        (
            "shots 192221-192233",
            [(192221, "exact")]
            + [(s, "range") for s in range(192222, 192233)]
            + [(192233, "exact")],
        ),
        ("shot MC-192798" + ", " * 35 + "190001", []),
    ],
)
def test_d23_identifiers_and_genuine_hyphens(text, expected):
    assert _found(text) == expected


@pytest.mark.parametrize(
    "journal",
    [
        "Appl. Phys. Lett.",
        "Applied Physics Letters",
        "J. Appl. Phys.",
        "Journal of Applied Physics",
        "J. Chem. Phys.",
        "Journal of Chemical Physics",
        "Phys. Plasmas",
        "Physics of Plasmas",
        "Rev. Sci. Instrum.",
        "Review of Scientific Instruments",
        "Nucl. Fusion",
        "Nuclear Fusion",
        "Plasma Phys. Control. Fusion",
        "Plasma Physics and Controlled Fusion",
        "J. Phys. D: Appl. Phys.",
        "Journal of Physics D: Applied Physics",
        "New J. Phys.",
        "New Journal of Physics",
        "Plasma Sources Sci. Technol.",
        "Plasma Sources Science and Technology",
        "Phys. Scr.",
        "Physica Scripta",
        "Meas. Sci. Technol.",
        "Measurement Science and Technology",
        "J. Phys.: Conf. Ser.",
        "Journal of Physics: Conference Series",
    ],
)
@pytest.mark.parametrize("comma", ["", ","])
@pytest.mark.parametrize("space", [" ", "", " ", " "])
def test_d23_journal_articles_are_not_tokens_or_range_starts(journal, comma, space):
    citation = space.join(journal.split()) + space + "1234" + comma + space
    assert _found("DIII-D " + citation + "194101") == []
    assert _found("DIII-D " + citation + "194101-03") == []


@pytest.mark.parametrize(
    "text, expected",
    [
        ("DIII-D modesPhys. Rev. Lett. 91 195003", []),
        ("DIII-D Phys. Rev. Lett. 91 195003", []),
        ("DIII-D Phys. Rev. Lett. 91 195003", []),
        ("DIII-D modesPhysical Reviewer Letters 91 195003", [(195003, "exact")]),
        ("shot 190000, Appl. Phys. Lett. 118, 194101", [(190000, "exact")]),
        ("shot Appl. Phys. Lett. 118, 194101" + ", " * 35 + "190001", []),
    ],
)
def test_d23_glued_citations_and_controls(text, expected):
    assert _found(text) == expected


def test_d23_article_span_has_a_margin_over_the_longest_prefix():
    prefix = "Journal of Physics D: Applied Physics 1234 , "
    assert len(normalise(prefix)) == 45
    assert context._ARTICLE_SPAN >= len(normalise(prefix)) + 16
    assert _found("DIII-D " + prefix + "194101") == []


@pytest.mark.parametrize(
    "country",
    [
        "China",
        "P. R. China",
        "PR China",
        "P.R. China",
        "Russia",
        "Russian Federation",
        "India",
        "Kazakhstan",
        "Singapore",
    ],
)
def test_d23_postal_countries(country):
    assert _found(f"Shanghai 200240, {country}. DIII-D National Fusion Facility") == []
    assert _found(f"DIII-D 194021 St. Petersburg, {country}") == []
    assert _found(f"DIII-D 194021 to 23, {country}") == [(194021, "exact")]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("194021 St. Petersburg, Russia, DIII-D", []),
        ("DIII-D shot 194021, Russian and Chinese teams", [(194021, "exact")]),
        ("DIII-D shot 194021 (Russia collaboration)", [(194021, "exact")]),
        ("DIII-D shot 194021, Chinas", [(194021, "exact")]),
        ("DIII-D shot 194021, 2, Russia", [(194021, "exact")]),
        ("shot 194021, Russia" + ", " * 35 + "190001", []),
        ("shot 194021" + "x" * 33 + ", China", []),
        ("shot 194021" + "x" * 33 + ", Chinas", [(194021, "exact")]),
        ("shot 194021" + "x" * 34 + ", China", [(194021, "exact")]),
    ],
)
def test_d23_postal_reach_and_controls(text, expected):
    assert _found(text) == expected


RANGE_END = (
    "nts seem to better representΘredep at lower pressure. 9 3 RESULTS Electron "
    "Pressure (Pa) DIII-D shots Large dot Diameter (mm) 67.3 148679 to 148682 "
    "[23] 10 108 192275 to 192276 [38] 8 144 184948 to 184951 1"
)
CUT_KEYWORD = (
    "n DIII-D 3 1.0 1.5 2.0 2.5 R (m) -1.5 -1.0 -0.5 0.0 0.5 1.0 1.5 z (m) "
    "193806@3.00s Figure 1. A representative equilibrium of the discharges "
    "used for this work, all of which"
)
RUN_FAR_KEYWORD = (
    " in DIII-D 6 0.0 0.2 0.4 0.6 0.8 1.0 ρ 0 20 40 60 80 100 120Vϕ (km/s) "
    "193807-4.000s 194479-2.800s 194479-3.800s 4.5 Nm Intrinsic Intrinsic "
    "Figure 5. Proﬁles of impurity toroidal rotation for th"
)


def test_d23_real_range_written_end():
    assert _found(RANGE_END) == [
        (148679, "exact"),
        (148680, "range"),
        (148681, "range"),
        (148682, "exact"),
        (192275, "exact"),
        (192276, "exact"),
    ]


@pytest.mark.parametrize("keyword_before", [True, False])
@pytest.mark.parametrize("words", [11, 12])
def test_d23_written_ends_share_range_context_once(keyword_before, words):
    filler = " ".join(["word"] * words)
    text = (
        "DIII-D shots " + filler + " 189600 to 189603 were run."
        if keyword_before
        else "189600 to 189603 " + filler + " shots on DIII-D."
    )
    expected = [
        (189600, "exact"),
        (189601, "range"),
        (189602, "range"),
        (189603, "exact"),
    ]
    # The brief's twelve words put both versions beyond REACH (62 / 61).
    assert _found(text) == (expected if words == 11 else [])
    assert _found(text, {189603}) == ([(189603, "exact")] if words == 11 else [])


@pytest.mark.parametrize(
    "text, expected",
    [
        ("DIII-D shots " + "word " * 20 + "189600-189603", []),
        ("LHD shots " + "word " * 12 + "189600-189603", []),
        ("NSTX-U shots 189600-189603", []),
        (
            "DIII-D shots 189600-03",
            [
                (189600, "exact"),
                (189601, "range"),
                (189602, "range"),
                (189603, "range"),
            ],
        ),
        (
            "DIII-D shots 189600-189650",
            [(189600, "exact")]
            + [(s, "range") for s in range(189601, 189650)]
            + [(189650, "exact")],
        ),
        ("DIII-D shots 189600-189651", [(189600, "exact"), (189651, "exact")]),
        ("DIII-D shots 189600-51", [(189600, "exact")]),
        ("DIII-D shots 189650-189600", [(189650, "exact"), (189600, "exact")]),
        (
            "DIII-D shots " + "word " * 11 + "189600-189651",
            [(189600, "exact")],
        ),
        (
            "DIII-D shots " + "word " * 11 + "189650-189600",
            [(189650, "exact")],
        ),
    ],
)
def test_d23_range_limits_and_guards(text, expected):
    assert _found(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        CUT_KEYWORD,
        RUN_FAR_KEYWORD,
        "DIII-D shots " + ", ".join(str(s) for s in range(190001, 190021)),
        "padding " * 20 + "DIII-D shots 189600 to 189603 " + "trailing " * 20,
        "padding " * 20 + "DIII-D shots 189600-03 " + "trailing " * 20,
    ],
)
def test_d23_context_covers_number_and_whole_nearest_keyword(text):
    text = normalise(text)
    found = mentions(text)
    assert found
    keywords = list(context.KEYWORDS.finditer(text))
    for m in found:
        keyword = min(
            keywords,
            key=lambda k: max(m.start - k.end(), k.start() - m.end, 0),
        )
        a, b = min(m.start, keyword.start()), max(m.end, keyword.end())
        expected = text[max(0, a - REACH) : b + REACH]
        assert context.KEYWORDS.search(m.context)
        assert m.context == expected
        assert len(m.context) <= b - a + 2 * REACH
        if m.match_type == "exact":
            assert str(m.shot) in m.context
        else:
            assert text[m.start : m.end] in m.context
    if text == normalise(CUT_KEYWORD):
        assert "discharges" in found[0].context


def test_d23_context_uses_the_nearest_eligible_keyword():
    # The nearer "shot" supplies each context, not the distant "#".
    text = "#" + "word " * 13 + "shots 190001, 190002, 190003"
    for m in mentions(text):
        a = text.index("shots")
        assert m.context == text[max(0, a - REACH) : m.end + REACH]


def test_d23_range_end_context_uses_its_range_keyword():
    text = "padding " * 20 + "shots " + "word " * 11 + "189600 to 189603"
    found = mentions(text)
    assert len(found) == 4
    for m in found:
        assert m.context == text[text.index("shots") - REACH : m.end + REACH]
    assert [(m.start, m.end) for m in found if m.match_type == "exact"] == [
        (text.index("189600"), text.index("189600") + 6),
        (text.index("189603"), text.index("189603") + 6),
    ]


@pytest.mark.parametrize("abbreviated", [False, True])
@pytest.mark.parametrize("distance", [50, 51])
def test_range_cap_counts_difference_of_ends(abbreviated, distance):
    end = str(distance) if abbreviated else str(189600 + distance)
    found = _found(f"DIII-D shots 189600-{end}")
    assert ((189601, "range") in found) == (distance == 50)
    if distance == 50:
        assert len(found) == 51
    else:
        assert found == (
            [(189600, "exact")]
            if abbreviated
            else [(189600, "exact"), (189651, "exact")]
        )


@pytest.mark.parametrize("space", ["\u00a0", "\u2009"])
def test_unicode_spaces_keep_shot_mentions(space):
    assert normalise(f"shot{space}189631") == "shot 189631"
    assert _found(f"shot{space}189631") == [(189631, "exact")]


@pytest.mark.parametrize(
    "numbers", ["190000 195000", "195000 190000", "190000\n195000", "190000 191000"]
)
def test_round_axis_pairs_are_not_shots(numbers):
    assert _found(f"DIII-D {numbers} Shot Numbers") == []


@pytest.mark.parametrize(
    "numbers", ["190000 and 195000", "190000, 195000", "190000 190500", "190001 195001"]
)
def test_axis_pair_rule_keeps_shot_lists(numbers):
    assert len(_found(f"DIII-D shots {numbers}")) == 2


@pytest.mark.parametrize(
    "country",
    [
        "People's Republic of China",
        "People’s Republic of China",
        "Peoples Republic of China",
        "P. R. China",
        "China",
    ],
)
def test_peoples_republic_postal_address_is_not_a_shot(country):
    text = (
        f"Institute of Plasma Physics, DIII-D collaboration, Shanghai 200240, {country}"
    )
    assert _found(text) == []


def test_journal_of_physics_d_real_article_number_is_not_a_shot():
    text = (
        "J. van Dijk, G. M. W. Kroesen, and A. Bogaerts, "
        '"Plasma modelling and numerical simulation," J. Phys. D: Appl. Phys. '
        "42, 190301 (2009). DIII-D"
    )
    assert _found(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "see report#runs 195123 for details",
        (
            "https://www.statista.com/statistics/265612/primary-energy-consumption-"
            "in-china-by-fuel-type-in-oil-\nequivalent/#:~:text=Primary%20energy%20"
            "consumption%20in%20China%202018%2D2023%2C%20by%20fuel&text\n"
            "=Coal%20is%20by%20far%20the,exajoules"
        ),
    ],
)
def test_url_fragment_hash_does_not_supply_context(text):
    assert _found(text) == []


@pytest.mark.parametrize(
    "text, shot",
    [
        ("shot #195123", 195123),
        ("discharge # 189051", 189051),
        ("shot#189051", 189051),
        ("# Ip (MA)\n192043 1.2", 192043),
    ],
)
def test_hash_shot_keywords_are_retained(text, shot):
    assert _found(text) == [(shot, "exact")]


@pytest.mark.parametrize(
    "text, shots",
    [
        (
            "fluctuations on DIII-D during QRE shot 190 604. The observed modes",
            [190604],
        ),
        (
            "Overview of UEDGE grids based on DIII-D discharge # 189 051 at 3250 ms",
            [189051],
        ),
        (
            (
                "The channel #12 of ECE and channel #1308 of ECEI "
                "of shot 191 506 at 3000 ms"
            ),
            [191506],
        ),
        (
            (
                "successfully avoided TMs. Shots 199 600 and 199 601 "
                "used additional ECH power"
            ),
            [199600, 199601],
        ),
        ("Shots 199 600 and 199601", [199600, 199601]),
        ("Shots 199600 and 199 601", [199600, 199601]),
        (
            "the LLAMA diagnostic array for DIII-D shot number 189 337 @ time slice",
            [189337],
        ),
    ],
)
def test_d26_real_split_shot_numbers(text, shots):
    assert _found(text) == [(shot, "exact") for shot in shots]
    for mention in mentions(text):
        assert str(mention.shot) in mention.context
        assert normalise(text)[mention.start : mention.end] == str(mention.shot)


def test_d26_real_split_discharge_range():
    text = "Over the 13 repeated discharges (discharges 187 214-187 226) Te,ped"
    assert _found(text) == (
        [(187214, "exact")]
        + [(shot, "range") for shot in range(187215, 187226)]
        + [(187226, "exact")]
    )


@pytest.mark.parametrize("space", ["\u2009", "\u00a0", "\n"])
def test_d26_collapses_whitespace_before_joining(space):
    assert normalise(f"shot 190{space}604") == "shot 190604"
    assert _found(f"shot 190{space}604") == [(190604, "exact")]


@pytest.mark.parametrize(
    "anchor",
    [
        "shot ",
        "SHOTS ",
        "discharge ",
        "Discharges ",
        "#",
        "# ",
        "shot no. ",
        "shots nos. ",
        "shot number ",
        "shots numbers ",
        "discharge # ",
        "shot#",
        "shots: ",
        "# no. ",
    ],
)
def test_d26_anchors_join_only_the_number(anchor):
    text = f"before {anchor}190 604 at 3250 ms; unrelated 189 051 after"
    expected = f"before {anchor}190604 at 3250 ms; unrelated 189 051 after"
    assert normalise(text) == expected
    assert normalise(expected) == expected


@pytest.mark.parametrize("gap", [", ", "; ", "/", " & ", " and ", " or ", " "])
def test_d26_follows_the_whole_run(gap):
    assert _found(f"Shots 199 601{gap}199603{gap}199 607") == [
        (199601, "exact"),
        (199603, "exact"),
        (199607, "exact"),
    ]


@pytest.mark.parametrize("gap", ["-", " to ", " through ", " thru "])
def test_d26_follows_ranges(gap):
    assert _found(f"Shots 190 604{gap}190 606") == [
        (190604, "exact"),
        (190605, "range"),
        (190606, "exact"),
    ]


@pytest.mark.parametrize(
    "text",
    [
        (
            "0.4 0.6 0.8 0 200 400 600 800 0.4 0.6 0.8 0 200 400 600 800 "
            "Indices within Shots Normalized Coil Deflection"
        ),
        "DIII-D Shots 200 400 600 800",
        "DIII-D Shots 800 600 400 200 0",
        (
            "an H-mode (189 051 at 3250 ms), an L-mode (174 237 at 3500 ms), "
            "and an I-mode (189 381 at 2850 ms)"
        ),
        "see report#runs 190 604 for details",
        "see equivalent/#:~:text=190 604",
        "DIII-D 190 604",
        "snapshot 190 604",
        "shotgun 190 604",
        "shot %190 604",
        "shot .190 604",
        "shot 1190 604",
        "shot 190 6040",
        "shot 190 604.5",
        "shot 190 604.5 and 190 605",
        "shot (190 604)",
    ],
)
def test_d26_does_not_join_ticks_unanchored_or_non_numbers(text):
    assert normalise(text) == text
    assert _found(text) == []


def test_d26_tick_guard_still_joins_a_later_real_shot_in_the_chain():
    text = "Shots 200 400 600 800 and 190 604"
    assert normalise(text) == "Shots 200 400 600 800 and 190604"
    assert _found(text) == [(190604, "exact")]
