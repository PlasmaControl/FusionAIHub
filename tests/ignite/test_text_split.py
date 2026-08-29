"""split_bundle: pre-experiment vs. whole-bundle text split, plus anchor fallback rules."""

from tokamak_foundation_model.ignite.text_embed import (
    ANCHOR_GENERAL,
    ANCHOR_PLANNED,
    ANCHOR_SESSION_SUMMARIES,
    ANCHOR_SHOT_SPECIFIC,
    split_bundle,
)

HEADER = "# DIII-D per-shot text bundle\n\nRUN_ID: 20240401\n\nSHOT: 158103\n"
GENERAL_INFO = "GENERAL_INFO_SENTINEL\nMETADATA_SENTINEL\n"
SUMMARIES = "SUMMARIES_SENTINEL\n"
PLANNED_TEXT = "PLANNED_SENTINEL\n"
SHOT_SPECIFIC = "SHOT_SPECIFIC_SENTINEL\n"


def _bundle(
    header=HEADER,
    general=True,
    summaries=True,
    planned=True,
    shot_specific=True,
):
    parts = [header]
    if general:
        parts.append(ANCHOR_GENERAL + "\n" + GENERAL_INFO)
    if summaries:
        parts.append(ANCHOR_SESSION_SUMMARIES + "\n" + SUMMARIES)
    if planned:
        parts.append(ANCHOR_PLANNED + "\n" + PLANNED_TEXT)
        if shot_specific:
            parts.append(ANCHOR_SHOT_SPECIFIC + "\n" + SHOT_SPECIFIC)
        else:
            # no shot-specific anchor: trailing content has no boundary, so it stays
            # part of the (unterminated) planned section through EOF
            parts.append(SHOT_SPECIFIC)
    return "".join(parts)


def test_full_bundle_includes_pre_experiment_content_only():
    text = _bundle()
    input_text, total_text = split_bundle(text)

    assert total_text == text

    assert "RUN_ID: 20240401" in input_text
    assert "SHOT: 158103" in input_text
    assert "GENERAL_INFO_SENTINEL" in input_text
    assert "METADATA_SENTINEL" in input_text
    assert "PLANNED_SENTINEL" in input_text

    assert "SUMMARIES_SENTINEL" not in input_text
    assert "SHOT_SPECIFIC_SENTINEL" not in input_text


def test_missing_session_summaries_extends_general_slice_to_planned():
    text = _bundle(summaries=False)
    input_text, _ = split_bundle(text)

    assert "GENERAL_INFO_SENTINEL" in input_text
    assert "METADATA_SENTINEL" in input_text
    assert "PLANNED_SENTINEL" in input_text
    assert "SUMMARIES_SENTINEL" not in input_text  # section wasn't there at all
    assert "SHOT_SPECIFIC_SENTINEL" not in input_text


def test_missing_planned_yields_header_plus_general_only():
    text = _bundle(planned=False)
    input_text, _ = split_bundle(text)

    assert "GENERAL_INFO_SENTINEL" in input_text
    assert "PLANNED_SENTINEL" not in input_text  # section wasn't there
    assert "SHOT_SPECIFIC_SENTINEL" not in input_text
    assert "SUMMARIES_SENTINEL" not in input_text


def test_missing_shot_specific_planned_section_runs_to_eof():
    text = _bundle(shot_specific=False)
    input_text, _ = split_bundle(text)

    assert "PLANNED_SENTINEL" in input_text
    # with no shot-specific anchor, the planned section absorbs the unbounded tail,
    # so the trailing sentinel (no anchor line in front of it) is included too
    assert "SHOT_SPECIFIC_SENTINEL" in input_text
    assert "SUMMARIES_SENTINEL" not in input_text


def test_missing_general_anchor_yields_empty_input():
    text = _bundle(general=False)
    input_text, total_text = split_bundle(text)

    assert input_text == ""
    assert total_text == text


def test_missing_summaries_and_planned_excludes_shot_specific():
    # compound-missing case: BOTH ANCHOR_SESSION_SUMMARIES and ANCHOR_PLANNED are absent, but
    # ANCHOR_SHOT_SPECIFIC is present. The general slice must stop at shot-specific, not
    # silently swallow it (and everything after) by running unbounded to EOF.
    text = (
        HEADER
        + ANCHOR_GENERAL
        + "\n"
        + GENERAL_INFO
        + ANCHOR_SHOT_SPECIFIC
        + "\n"
        + SHOT_SPECIFIC
    )

    input_text, total_text = split_bundle(text)

    assert total_text == text
    assert "GENERAL_INFO_SENTINEL" in input_text
    assert "SHOT_SPECIFIC_SENTINEL" not in input_text


def test_all_three_later_anchors_missing_yields_header_plus_general_to_eof():
    # only the general-section anchor is present; header + general slice run to EOF (no
    # planned/summaries/shot-specific anchors exist to bound anything further).
    text = HEADER + ANCHOR_GENERAL + "\n" + GENERAL_INFO

    input_text, total_text = split_bundle(text)

    assert total_text == text
    assert input_text == text  # nothing to exclude: the whole doc is pre-experiment content


def test_duplicate_session_summaries_anchor_uses_first_occurrence_without_crashing():
    # two SESSION-WIDE SUMMARIES lines; the general slice must end at the FIRST one, so
    # content between the two occurrences (and after) stays excluded from input_text.
    text = (
        HEADER
        + ANCHOR_GENERAL
        + "\n"
        + GENERAL_INFO
        + ANCHOR_SESSION_SUMMARIES
        + "\nFIRST_SUMMARIES_TAIL_SENTINEL\n"
        + ANCHOR_SESSION_SUMMARIES
        + "\nSECOND_SUMMARIES_TAIL_SENTINEL\n"
        + ANCHOR_PLANNED
        + "\n"
        + PLANNED_TEXT
        + ANCHOR_SHOT_SPECIFIC
        + "\n"
        + SHOT_SPECIFIC
    )

    input_text, total_text = split_bundle(text)

    assert total_text == text
    assert "GENERAL_INFO_SENTINEL" in input_text
    assert "PLANNED_SENTINEL" in input_text
    assert "FIRST_SUMMARIES_TAIL_SENTINEL" not in input_text
    assert "SECOND_SUMMARIES_TAIL_SENTINEL" not in input_text


def test_counters_increment_on_missing_anchor():
    # a missing anchor silently changes the causality-critical INPUT slice, so a caller
    # processing many bundles must be able to tally how often each fallback fired.
    text = _bundle(summaries=False)
    counters = {}
    split_bundle(text, counters=counters)
    assert counters == {"missing_summaries_anchor": 1}


def test_counters_untouched_on_complete_bundle():
    text = _bundle()
    counters = {}
    split_bundle(text, counters=counters)
    assert counters == {}


def test_counters_param_optional():
    # omitting counters entirely must stay valid (no behavior change to the returned slices)
    text = _bundle(general=False)
    input_text, total_text = split_bundle(text)
    assert input_text == ""
    assert total_text == text
