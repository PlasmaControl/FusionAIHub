"""Paper table regression checks for reference scope and metric conditioning."""

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
    assert r"\begin{tabular}{lccccc}" in text and r"$\Delta$AUROC" in text


def test_compact_ranking_marks_refit_and_omits_historical_initializations():
    data = record()
    text = swap_tex.ranking_table(data, "evaluation.json", "auroc")
    assert r"elm-dsm refit$^{\ddagger}$" in text
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
        for caption in re.findall(
            r"\\caption\{(.*?)\}\n\\label", path.read_text(), re.DOTALL
        ):
            assert len(caption.split()) <= 150, path.name
            assert "source\\_formatters" not in caption and "Hiro" not in caption
            assert "source exposure" not in caption
            if "rankings" in path.name:
                assert "Brackets" not in caption and "Numerical F1" not in caption
    caption = (tmp_path / "table_elm_swap.tex").read_text()
    assert "inconclusive" in caption
    assert "eight" in caption and "three shots (2 with BES)" in caption
    assert "AUROC" in caption and "review-tuned" in caption
    assert "no AE Finding-2 analogue is established" in caption
    assert "paired legacy-minus-review AUROC change" in caption
    assert "No ranking change" in caption and "in-sample on 5/8 shots" in caption
    assert "upstream WPQH PCPHD02/03 export" in caption


def test_swap_caption_reads_the_domain_shift_numbers_from_the_records():
    data = record()
    ours = json.loads((SOURCE.parent.parent / "ours/evaluation.json").read_text())
    text = swap_tex.main_caption(data, ours)
    panel = data["swap"]["overlap_bes"]["reviewed"]["methods"]["elm-elmo"]
    whole = ours["sets"]["bes73"]["methods"]["elm-elmo"]
    assert f"{panel['point']['auroc']:.3f}" in text
    assert f"{whole['point']['auroc']:.3f}" in text and "domain shift" in text
    change = data["swap"]["overlap"]["comparison"]["auroc_change_paired"]
    assert f"{change['elm-dsm']['value']:+.3f}" in text


def test_swap_caption_names_only_the_intervals_that_exclude_zero():
    data = record()
    overlap, bes = data["swap"]["overlap"], data["swap"]["overlap_bes"]
    panels = (("on all overlap shots", overlap), ("on the BES shots", bes))
    hits = [
        (scope, key)
        for scope, panel in panels
        for key, row in panel["comparison"]["auroc_change_paired"].items()
        if row["ci95"][0] > 0 or row["ci95"][1] < 0
    ]
    text = swap_tex.excluding_zero(*panels)
    if hits:
        assert text.startswith("The paired interval excludes zero only for ")
        assert all(scope in text for scope, _ in hits)
        row = bes["comparison"]["auroc_change_paired"]["elm-elmo"]
        assert f"{row['value']:+.3f}" in text
    else:
        assert text == "Every paired interval includes zero. "
    quiet = {
        "comparison": {
            "auroc_change_paired": {"elm-dsm": {"value": 0.0, "ci95": [-1, 1]}}
        }
    }
    assert (
        swap_tex.excluding_zero(("x", quiet)) == "Every paired interval includes zero. "
    )
    assert "excludes zero" in swap_tex.main_caption(data) or not hits


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


def test_main_occupancy_table_uses_primary_panels_and_omits_unavailable_elmo():
    root = SOURCE.parent.parent
    ours = json.loads((root / "ours/evaluation.json").read_text())
    dsm = json.loads((root / "dsm/evaluation.json").read_text())
    feature = json.loads((root / "ours/feature_only.json").read_text())
    fx = facts.facts(facts.load(root))
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
    assert "elm-dsm (60-input" not in text
    caption = text.split(r"\caption{", 1)[1].split(r"\label", 1)[0]
    for tag in ("all119", "bes73"):
        assert f"{ours['sets'][tag]['bins']:,}" in caption
    assert "developmental shot-grouped CV on the 119 reviewed shots" in caption
    assert "blind split awaits review of those shots" in caption
    assert "ELMy-period occupancy, not onsets" in caption
    assert f"{100 * fx['crowd_share']:.0f}\\% of positive bins" in caption
    assert "wholly inside reviewed spans" in caption and "task is easier" in caption
    assert "review seeded by the D-alpha clock" in caption
    assert "favours D-alpha-input models over ELM-O" in caption
    share = fx["clock_share"]
    assert f"{100 * share['both']:.0f}\\% of both" in caption
    assert "inner-validation F1" in caption and "shot-bootstrap" in caption
    assert "mean probability" in caption and "any touching detected span" in caption
    assert "FPR is the fraction of absent bins" in caption
    assert "alarm is the fraction of absent spans" in caption
    assert "Paired elm-ours minus ELM-O on bes73: AUROC" in caption
    folds = fx["ours_folds"]
    early = min(folds, key=lambda r: r["epoch"])
    assert (
        f"Fold {early['fold']} (zero-based) selected epoch {early['epoch']}" in caption
    )
    thr = [r["threshold"] for r in folds]
    assert f"{min(thr):.3f}--{max(thr):.3f}" in caption
    quiet = fx["non_crowd_only"]
    assert f"{quiet['shots']} shots/{quiet['bins']:,} bins" in caption
    assert f"{quiet['precision'][0]:.3f} [" in caption
    moved = fx["moved"]
    assert (
        f"{moved['present_spans']} of {moved['all_present_spans']} present" in caption
    )
    assert "cv2" not in caption and "unresolved" not in caption
    assert "domain shift" not in text and "192721:" not in text
    # the alarm rates sit beside the false-alarm bin rate in the table body
    alarm = fx["alarm"]["all119/elm-ours"]["span_alarm"][0]
    assert f"{alarm:.3f}" in text


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
    assert "21 of 31 Smith run days crossing folds" in text
    assert "unknown because event membership" in text
    assert "domain shift" not in text and "192721:" not in text
    assert "[0.98, 0.98]" not in text
    kinds = module.per_kind_table(ours)
    assert "Span-touch recall does not measure onset timing" in kinds
    assert "5 ms before" not in kinds and "domain shift" not in kinds
