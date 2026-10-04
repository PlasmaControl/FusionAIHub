"""Paper table regression checks for reference scope and metric conditioning."""

import copy
import json
from pathlib import Path

from labeler.elm import facts, swap_tex

SOURCE = (
    Path(__file__).resolve().parents[2] / "outputs/labeler/elm/swap/evaluation.json"
)


def record():
    return json.loads(SOURCE.read_text())


def test_main_swap_keeps_numeric_f1_without_stacked_precision_recall():
    data = record()
    text = swap_tex.main_table(data["swap"]["overlap"], data, "evaluation.json")
    assert "0.720" in text  # saved elm-ours reviewed F1, retained numerically
    assert "P=" not in text and "R=" not in text
    assert "200" not in text
    assert r"\begin{tabular}{@{}lccccc@{}}" in text and r"$\Delta$AUROC" in text


def test_compact_ranking_marks_refit_and_omits_historical_initializations():
    data = record()
    text = swap_tex.ranking_table(data, "evaluation.json", "auroc")
    assert r"elm-dsm-survival$^{\ddagger}$" in text
    assert "elm-dsm detection init" not in text
    assert "_detection_common" not in text


def test_bes_point_order_crossing_distinguishes_a_stable_order():
    def reference(clean):
        return {
            "methods": {
                "elm-dsm-detect": {"point": {"auroc": clean}},
                "elm-elmo": {"point": {"auroc": 0.8}},
            }
        }

    data = {
        "swap": {
            "overlap_bes": {
                "reviewed": reference(0.7),
                "legacy": reference(0.75),
                "occupancy": {"gap_100ms": {"legacy": reference(0.85)}},
            }
        }
    }
    assert swap_tex.bes_point_order_crosses(data)
    data["swap"]["overlap_bes"]["occupancy"]["gap_100ms"]["legacy"] = reference(0.79)
    assert not swap_tex.bes_point_order_crosses(data)


def test_generated_captions_are_short_and_table_specific(tmp_path):
    import re

    swap_tex.write(record(), tmp_path)
    files = list(tmp_path.glob("*.tex"))
    assert len(files) == 4
    for path in files:
        cap = 170 if path.name == "table_elm_swap.tex" else 150
        for caption in re.findall(
            r"\\caption\{(.*?)\}\n\\label", path.read_text(), re.DOTALL
        ):
            assert len(caption.split()) <= cap, path.name
            assert "source\\_formatters" not in caption and "Hiro" not in caption
            assert "source exposure" not in caption
            if "rankings" in path.name:
                assert "Brackets" not in caption and "Numerical F1" not in caption
    caption = (tmp_path / "table_elm_swap.tex").read_text()
    one = swap_tex.finding_one(record())
    assert "inconclusive" in caption
    assert "eight" in caption and "seven with BES" in caption
    assert f"{one['cells']} known-majority cells" in caption
    assert f"{one['recall']:.3f}" in caption and f"{one['precision']:.3f}" in caption
    assert f"$|M|={one['M']}$" in caption and f"$|P|={one['P']}$" in caption
    assert f"{one['strict_bins']} strict interior bins" in caption
    assert "AUROC" in caption and "review-tuned" in caption
    assert "no AE Finding-2 analogue is established" in caption
    assert "paired legacy-minus-review AUROC change" in caption
    assert "No ranking change" in caption and "in-sample on 5/8 shots" in caption
    assert "upstream WPQH PCPHD02/03 export" in caption
    assert "domain shift" not in caption


def test_finding_one_is_read_on_the_known_majority_cells_first():
    one = swap_tex.finding_one(record())
    assert one["cells"] == 782 and (one["M"], one["P"]) == (60, 14)
    assert round(one["recall"], 3) == 0.692 and round(one["precision"], 3) == 0.906
    assert one["strict_bins"] == 641 and (one["strict_M"], one["strict_P"]) == (40, 4)


def test_swap_caption_reads_the_elmo_shift_numbers_from_the_records():
    data = record()
    ours = json.loads((SOURCE.parent.parent / "ours/evaluation.json").read_text())
    text = swap_tex.main_caption(data, ours)
    panel = data["swap"]["overlap_bes"]["reviewed"]["methods"]["elm-elmo"]
    whole = ours["sets"]["bes73"]["methods"]["elm-elmo"]
    lo, hi = panel["ci95"]["auroc"]
    assert f"{panel['point']['auroc']:.3f}" in text
    assert f"{lo:.3f}, {hi:.3f}" in text
    assert f"{whole['point']['auroc']:.3f}" in text
    assert lo <= whole["point"]["auroc"] <= hi
    assert "no shift is established" in text and "domain shift" not in text
    change = data["swap"]["overlap"]["comparison"]["auroc_change_paired"]
    assert f"{change['elm-dsm']['value']:+.3f}" in text


def test_swap_caption_branches_on_whether_the_interval_excludes_the_bes73_value():
    data = record()
    ours = json.loads((SOURCE.parent.parent / "ours/evaluation.json").read_text())
    inside = swap_tex.main_caption(data, ours)
    assert "the interval contains that value, so no shift is established" in inside
    panel = data["swap"]["overlap_bes"]["reviewed"]["methods"]["elm-elmo"]
    moved = copy.deepcopy(ours)
    moved["sets"]["bes73"]["methods"]["elm-elmo"]["point"]["auroc"] = (
        panel["ci95"]["auroc"][1] + 0.05
    )
    outside = swap_tex.main_caption(data, moved)
    assert (
        "the interval excludes that value, so a shift from the bes73 value" in outside
    )
    assert "is indicated" in outside and "no shift is established" not in outside


def test_swap_caption_reports_photodiode_agreement_only_with_the_record():
    data = record()
    fresh = {"records": 16, "identical": 16}
    text = swap_tex.main_caption(data, None, fresh)
    assert "equals a fresh fetch in 16 of 16 records" in text
    assert "All overlap shots use the upstream WPQH" in swap_tex.main_caption(data)


def test_swap_caption_counts_the_intervals_that_exclude_zero():
    data = record()
    total, hits = swap_tex.paired_intervals(data)
    named = [
        key
        for tag in ("overlap", "overlap_bes")
        for key, row in data["swap"][tag]["comparison"]["auroc_change_paired"].items()
        if row["ci95"] and (row["ci95"][0] > 0 or row["ci95"][1] < 0)
    ]
    assert len(hits) == len(named)
    text = swap_tex.excluding_zero(data)
    if hits:
        assert text.startswith(
            f"{swap_tex.count_word(len(hits)).capitalize()} of {total} unadjusted "
        )
        assert " zero: " in text
        _, _, row = hits[0]
        assert f"{row['value']:+.3f}" in text
        assert "domain shift" not in text
    else:
        assert text == (
            f"All {swap_tex.count_word(total)} unadjusted paired intervals "
            "include zero. "
        )
    quiet = {
        "swap": {
            tag: {
                "n_shots": 7,
                "comparison": {
                    "auroc_change_paired": {"elm-dsm": {"value": 0.0, "ci95": [-1, 1]}}
                },
            }
            for tag in ("overlap", "overlap_bes")
        }
    }
    assert swap_tex.excluding_zero(quiet) == (
        "All two unadjusted paired intervals include zero. "
    )


def test_descriptive_metric_cell_suppresses_interval():
    result = {
        "point": {"auroc": 0.993},
        "ci95": {"auroc": None},
        "descriptive_only": True,
    }
    assert swap_tex.metric_cell(result, "auroc") == "0.993"


def paper_module():
    import importlib.util

    path = SOURCE.parents[4] / "scripts/labeler/elm_paper_tables.py"
    spec = importlib.util.spec_from_file_location("elm_paper_tables", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def protocol_module():
    import importlib.util
    import sys

    path = SOURCE.parents[4] / "scripts/labeler/elm_protocol.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("elm_protocol", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_patch_models_lists_the_dsm_adaptation_once_under_every_old_label():
    module = protocol_module()
    old = (
        "## Models\n**stable**: x | 2026_09_06\n\n**latest**: elm-ours | 2026_09_13"
        "\n\n**all**:\n"
        "- elm-dsm (60-input 1×128 refit, detection) | 2026_10_01 | AUROC: 0.1\n"
        "- elm-dsm (detection) | 2026_09_30 | AUROC: 0.2\n"
        "- elm-dsm-detect | 2026_09_29 | AUROC: 0.3\n"
        f"- {module.DSM_LABEL} | 2026_10_02 | AUROC: 0.4\n"
        "- elm_clock | 2026_09_13 | F1: 0.5\n\nOld note.\n\n## Inputs\n"
    )
    text = module.patch_models(
        old, [f"- {module.DSM_LABEL} | 2026_10_03 | AUROC: 0.9"], "N."
    )
    entries = [line for line in text.splitlines() if line.startswith("- ")]
    assert [e.split(" | ")[0] for e in entries] == [
        f"- {module.DSM_LABEL}",
        "- elm-clock",
    ]
    assert "AUROC: 0.9" in entries[0]


def benchmark_inputs():
    root = SOURCE.parent.parent
    ours = json.loads((root / "ours/evaluation.json").read_text())
    dsm = json.loads((root / "dsm/evaluation.json").read_text())
    feature = json.loads((root / "ours/feature_only.json").read_text())
    fx = facts.facts(facts.load(root))
    return ours, dsm, feature, fx


def test_main_occupancy_table_uses_primary_panels_and_omits_unavailable_elmo():
    ours, dsm, feature, fx = benchmark_inputs()
    text = paper_module().benchmark_table(ours, dsm, feature, fx)
    assert text.count("primary bins") == 2
    # two primary panels, plus the bes73 moved-boundary block for the three
    # methods scored there
    for name, count in (
        ("elm-ours", 3),
        ("elm-clock", 3),
        ("always-present", 2),
        ("elm-feature", 2),
        ("elm-elmo", 2),
    ):
        assert text.count(name + " & ") == count, name
    assert "elm-dsm" not in text and "ELM-O" not in text
    # the guard25 alarm sits beside the raw alarm
    assert "Absent-span alarm" in text and "Alarm, guard25" in text
    for key in ("all119/elm-ours", "bes73/elm-elmo"):
        alarm = fx["alarm"][key]
        assert f"{alarm['span_alarm'][0]:.3f}" in text
        assert f"{alarm['span_alarm_guard25'][0]:.3f}" in text


def test_main_caption_is_short_and_carries_only_the_four_required_points():
    ours, dsm, feature, fx = benchmark_inputs()
    text = paper_module().benchmark_table(ours, dsm, feature, fx)
    caption = text.split(r"\caption{", 1)[1].split(r"\label", 1)[0]
    assert len(caption.split()) <= 120
    for tag in ("all119", "bes73"):
        assert f"{ours['sets'][tag]['bins']:,}" in caption
    assert "developmental shot-grouped CV on the 119 reviewed shots" in caption
    assert "no blind-split score" in caption
    assert "ELMy-period occupancy, not onsets" in caption
    assert f"{100 * fx['crowd_share']:.0f}\\% of positive bins" in caption
    assert "only interior 50 ms bins are scored" in caption
    assert "seeded by the D-alpha clock" in caption
    assert "favours D-alpha-input models over elm-elmo" in caption
    assert "Paired elm-ours minus elm-elmo on bes73: AUROC" in caption
    assert "every interval includes zero" in caption
    # diagnostics live in the notes, not in the caption
    for stale in ("Fold ", "zero-based", "seeds", "moved", "non-crowd", "cv2"):
        assert stale not in caption, stale


def test_table_notes_hold_the_diagnostics_the_caption_dropped():
    _, _, _, fx = benchmark_inputs()
    root = SOURCE.parent.parent
    note = paper_module().appendix_note(
        fx, json.loads((root / "swap/evaluation.json").read_text())
    )
    folds = fx["ours_folds"]
    early = min(folds, key=lambda r: r["epoch"])
    assert f"Fold {early['fold']} (zero-based)" in note
    thr = [r["threshold"] for r in folds]
    assert f"{min(thr):.3f}--{max(thr):.3f}" in note
    moved = fx["moved"]
    assert f"{moved['present_spans']} of {moved['all_present_spans']} present" in note
    quiet = fx["non_crowd_only"]
    assert f"{quiet['shots']} shots/{quiet['bins']:,} bins" in note
    assert f"{quiet['precision'][0]:.3f} [" in note
    assert "mean probability" in note and "any touching detected span" in note
    assert "FPR is the fraction of absent bins" in note
    assert "guard25 trims 25 ms from each absent-span edge" in note
    # the guard25 denominator: empty interiors stay in as not alarmed
    alarm = fx["alarm"]
    ours_all = alarm["all119/elm-ours"]
    assert "stays in the denominator as not alarmed" in note
    assert (
        f"{ours_all['guard25_empty']} of {ours_all['absent_spans']} on all119" in note
    )
    bes_ours = alarm["bes73/elm-ours"]
    assert f"{bes_ours['guard25_empty']} of {bes_ours['absent_spans']} on bes73" in note
    rule = "/".join(
        f"{alarm[f'{tag}/always present']['span_alarm_guard25'][0]:.3f}"
        for tag in ("all119", "bes73")
    )
    assert f"the always-present rule scores {rule} rather than 1" in note
    for key, name in (
        ("all119/elm-ours", "elm-ours"),
        ("all119/elm-clock", "clock"),
        ("bes73/elm-elmo", "elm-elmo"),
    ):
        interior = alarm[key]["span_alarm_interior_guard25"][0]
        assert f"{interior:.3f} ({name})" in note
    assert "inner-validation F1" in note and "shot-bootstrap" in note
    share = fx["clock_share"]
    assert f"{100 * share['both']:.0f}\\% of both" in note
    # the known-majority glossary entry says what swap.coverage_bins does
    assert "at least 25 ms of legacy coverage" in note
    assert "at least 25 ms of one reviewed state" in note
    assert "reviewed present occupancy" not in note
    assert "ELM-O" in note and "\\texttt{elm-elmo}" in note
    assert "no sightline list is retained" in note
    assert "8 of 8" not in note and "domain shift" not in note


def test_smith_table_has_no_mismatched_frozen_rows_or_duplicate_tolerances():
    smith = json.loads((SOURCE.parent.parent / "smith/evaluation.json").read_text())
    text = paper_module().smith_table(smith)
    assert "elm-ours &" not in text and "elm-ours (onset" not in text
    assert "elm-ours-onset &" in text
    assert r"$\pm$5 ms" not in text
    assert "target mismatch" in text


def test_multiline_method_request_still_uses_single_line_with_exposure_marker():
    text = swap_tex.method_label("elm-dsm-detect-exposed", multiline=True)
    assert r"\shortstack" not in text
    assert text.endswith(r"$^{\ddagger}$")


def test_native_paper_table_omits_four_shot_exact_export():
    native = json.loads(
        (SOURCE.parent.parent / "dsm/native_evaluation.json").read_text()
    )
    text = paper_module().native_table(native)
    assert "exact native exports" not in text and "Reviewed occupancy" not in text
    assert "within horizon" in text and "Non-crowd starts" in text


def test_native_detection_caption_keeps_latex_multiplication_command():
    native = json.loads(
        (SOURCE.parent.parent / "dsm/native_detection.json").read_text()
    )
    text = paper_module().native_detection_table(native)
    assert r"60-input $1\times128$ adaptation" in text
    assert "\t" not in text


def test_native_detection_caption_states_training_sizes_and_the_matched_row():
    _, _, _, fx = benchmark_inputs()
    native = json.loads(
        (SOURCE.parent.parent / "dsm/native_detection.json").read_text()
    )
    text = paper_module().native_detection_table(native, fx)
    caption = text.split(r"\caption{", 1)[1].split(r"\label", 1)[0]
    assert len(caption.split()) <= 170
    sizes = fx["native_detection"]["folds"]
    own = fx["ours_training_sizes"]
    low, high = min(f["train"] for f in sizes), max(f["train"] for f in sizes)
    assert f"{low}--{high}" in caption
    assert f"elm-ours {min(f['train'] for f in own)}--" in caption
    assert "train/inner-validation" in caption
    # elm-dsm-detect trains on the full folds; the gate is a condition, not a finding
    assert "elm-dsm-detect 81--82/14, both on the full 119-shot folds" in caption
    assert f"({native['shots_at_least_90_percent']} shots pass)" in caption
    assert "audit gate" in caption and "is met" in caption
    assert "audit finds" not in caption
    # the paired interval comes from the saved predictions, not the marginal ones
    pairs = fx["native_detection"]["paired_native_folds"]["all119"]
    assert (
        "Paired elm-ours (native-fold training shots) minus elm-dsm-native-detect, "
        f"all119: {facts.delta_text(pairs)}" in caption
    )
    assert "elm-ours (native-fold training shots)" in text
    assert "elm-dsm-detect (60-input $1\\times128$)" in text
    assert "elm-dsm-native-detect (124-input [100,1000])" in text
    assert "elm-dsm refit" not in text and "ELM-O" not in text
    # lists of shots never reach the caption
    assert "185857" not in caption


def test_smith_and_per_kind_captions_have_specific_limits():
    root = SOURCE.parent.parent
    smith = json.loads((root / "smith/evaluation.json").read_text())
    ours = json.loads((root / "ours/evaluation.json").read_text())
    module = paper_module()
    text = module.smith_table(smith)
    assert "Conditional precision" not in text and "Conditional F1" not in text
    assert "Event precision and F1 are not shown" in text
    assert "10 ms minimum peak separation fixes the head's in-window precision" in text
    assert "conditional on selected windows" in text
    assert "continuous-discharge precision and F1 are unavailable" in text
    assert "folds grouped by run day (none of 31 Smith run days crosses)" in text
    assert "unknown because event membership" in text
    assert "domain shift" not in text and "192721:" not in text
    assert "[0.98, 0.98]" not in text
    kinds = module.per_kind_table(ours)
    assert "Span-touch recall does not measure onset timing" in kinds
    assert "5 ms before" not in kinds and "domain shift" not in kinds
