"""Paper table regression checks for reference scope and source exposure."""

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


def test_ranking_marks_source_initialized_and_refit_rows():
    data = record()
    text = swap_tex.ranking_table(data, "evaluation.json", "auroc")
    assert r"elm-dsm refit$^{\ddagger}$" in text
    assert r"elm-dsm detection init$^{\ddagger}$" in text


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
    assert len(files) >= 30
    for path in files:
        for caption in re.findall(
            r"\\caption\{(.*?)\}\n\\label", path.read_text(), re.DOTALL
        ):
            assert len(caption.split()) <= 80, path.name
            assert "source\\_formatters" not in caption and "Hiro" not in caption
            if "rankings" in path.name:
                assert "Brackets" not in caption and "Numerical F1" not in caption
    caption = (tmp_path / "table_elm_swap.tex").read_text()
    assert "inconclusive" in caption
    assert "eight" in caption and "three" in caption and "two" in caption
    assert "AUROC" in caption and "review-tuned" in caption
