"""Paper table regression checks for reference scope and metric conditioning."""

import json
from pathlib import Path

from labeler.elm import swap_tex

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
    assert r"\begin{tabular}{lcccc}" in text


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
    assert "eight" in caption and "3 shots; 2 with BES" in caption
    assert "AUROC" in caption and "review-tuned" in caption
    assert "AE Finding-2 analogue" in caption
    assert r"dalpha\_wpqh.pkl" in caption and "PCPHD02/03" in caption
    assert "0.815" in caption and "0.846" in caption


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


def test_main_occupancy_table_has_two_panels_and_only_six_core_rows():
    root = SOURCE.parent.parent
    ours = json.loads((root / "ours/evaluation.json").read_text())
    dsm = json.loads((root / "dsm/evaluation.json").read_text())
    feature = json.loads((root / "ours/feature_only.json").read_text())
    text = paper_module().benchmark_table(ours, dsm, feature)
    assert text.count("common bins") == 2
    visible_rows = (
        text.replace(r"\shortstack[l]{", "")
        .replace(r"\\", " ")
        .replace("} &", " &")
    )
    for name in (
        "elm-ours",
        "elm-elmo",
        r"elm-dsm (60-input $1\times128$ refit, detection)",
        "elm-clock",
        "always-present",
        "elm-feature",
    ):
        assert visible_rows.count(name + " & ") == 2
    assert "detection init" not in text and "source exposure" not in text
    caption = text.split(r"\caption{", 1)[1].rsplit(r"}\label", 1)[0]
    # Decimal points are not sentence breaks; the caption has four sentences.
    import re

    assert len(re.split(r"(?<=[.!?])\s+", caption)) == 4
    for tag in ("all119", "bes73"):
        assert f"{dsm['sets'][tag]['bins']:,}" in caption
    assert "50 ms bins wholly inside reviewed spans" in caption
    assert "inner-validation F1" in caption and "shot-bootstrap" in caption
    assert "bin-mean probability" in caption and r"$\geq$ threshold" in caption
    assert "aligned score" in caption and "input means" in caption
    assert "any detected-span touch" in caption
    assert "Review was seeded by the clock" in caption
    for metric, label in (("auroc", "AUROC"), ("auprc", "AUPRC"), ("f1", "F1")):
        delta = dsm["sets"]["bes73"]["paired"][f"elm-ours - elm-elmo: {metric}"]
        lo, hi = delta["ci95"]
        assert f"{label} {delta['value']:+.3f} [{lo:.3f}, {hi:.3f}]" in caption
    assert "cv2" not in caption and "unresolved" not in caption
    assert "domain shift" not in text and "192721:" not in text


def test_native_paper_table_omits_four_shot_exact_export():
    native = json.loads(
        (SOURCE.parent.parent / "dsm/native_evaluation.json").read_text()
    )
    text = paper_module().native_table(native)
    assert "exact native exports" not in text and "Reviewed occupancy" not in text
    assert "within horizon" in text and "Non-crowd starts" in text


def test_smith_and_per_kind_captions_have_specific_limits():
    root = SOURCE.parent.parent
    smith = json.loads((root / "smith/evaluation.json").read_text())
    ours = json.loads((root / "ours/evaluation.json").read_text())
    module = paper_module()
    text = module.smith_table(smith)
    assert "Conditional precision" in text and "Conditional F1" in text
    assert "conditional on selected windows" in text
    assert "continuous-discharge precision/F1 are unavailable for every method" in text
    assert "21 of 31 Smith run days crossing folds" in text
    assert "unknown because event membership" in text
    assert "domain shift" not in text and "192721:" not in text
    assert "[0.98, 0.98]" not in text
    kinds = module.per_kind_table(ours)
    assert "Span-touch recall does not measure onset timing" in kinds
    assert "5 ms before" not in kinds and "domain shift" not in kinds
