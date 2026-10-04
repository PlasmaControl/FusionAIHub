"""The ELM captions and documents quote numbers read from the JSON records."""

import copy
import math
import re
from pathlib import Path

import pytest

from labeler.elm import facts

ROOT = Path(__file__).resolve().parents[2]
RECORDS = ROOT / "outputs/labeler/elm"
GENERATORS = (
    "scripts/labeler/elm_paper_tables.py",
    "scripts/labeler/elm_protocol.py",
    "src/labeler/elm/swap_tex.py",
)


@pytest.fixture(scope="module")
def records():
    return facts.load(RECORDS)


def test_fmt_and_delta_text_format_points_intervals_and_signs():
    assert facts.fmt(0.9412, [0.9061, 0.9669]) == "0.941 [0.906, 0.967]"
    assert facts.fmt(None) == "--" and facts.fmt(float("nan")) == "--"
    assert facts.fmt(0.0234, [-0.0141, 0.0672], signed=True) == (
        "+0.023 [-0.014, +0.067]"
    )
    text = facts.delta_text(
        {"auroc": (0.02, [-0.01, 0.05]), "f1": (-0.01, [-0.05, 0.03])}
    )
    assert text == "AUROC +0.020 [-0.010, +0.050], F1 -0.010 [-0.050, +0.030]"


def test_spread_reports_mean_range_and_sample_sd():
    out = facts.spread([1.0, 2.0, 3.0])
    assert out["mean"] == 2.0 and (out["min"], out["max"]) == (1.0, 3.0)
    assert out["sd"] == pytest.approx(1.0)
    assert math.isnan(facts.spread([5.0])["sd"])


def test_facts_follow_the_records(records):
    fx = facts.facts(records)
    own = records["ours"]["sets"]["all119"]["methods"]["elm-ours"]
    assert fx["bins"]["all119"] == records["ours"]["sets"]["all119"]["bins"]
    assert fx["crowd_share"] == pytest.approx(
        own["counts"]["crowd_bins"] / (own["counts"]["tp"] + own["counts"]["fn"])
    )
    shares = records["strata"]["clock_boundary_identity"]["shares"]
    assert fx["clock_share"]["both"] == shares["both_within_1ms"]
    assert len(fx["ours_folds"]) == len(fx["dsm_folds"]) == 5
    assert all(isinstance(fx["offsets"][k], int) for k in ("median", "p25", "p75"))
    assert set(fx["paired_primary"]) == {"auroc", "auprc", "f1"}
    # changing a record changes the fact: nothing is cached or typed in
    edited = copy.deepcopy(records)
    edited["strata"]["clock_boundary_identity"]["shares"]["both_within_1ms"] = 0.5
    assert facts.facts(edited)["clock_share"]["both"] == 0.5


def test_optional_records_default_to_none(records):
    bare = {k: v for k, v in records.items() if k in facts.REQUIRED}
    fx = facts.facts(bare)
    assert fx["moved"] is None and fx["tiled"] is None
    assert fx["run_day"] is None and fx["dsm_baselines"] is None
    assert facts.tiled_facts(None) is None and facts.run_day_facts({}) is None


def test_sensitivity_facts_read_their_records(records):
    fx = facts.facts(records)
    tiled = records["tiled"]
    assert fx["tiled"]["tile_ms"] == tiled["tile_ms"]
    assert fx["tiled"]["serving_threshold"] == pytest.approx(
        sum(fx["tiled"]["fold_thresholds"]) / len(fx["tiled"]["fold_thresholds"])
    )
    run_day = fx["run_day"]
    assert (
        run_day["crossing"] == 0 and run_day["days"] == records["run_day"]["run_days"]
    )
    for tag in ("all119", "bes73"):
        assert set(run_day[tag]["headline"]) == {"auroc", "auprc", "f1"}
    seeds = fx["dsm_baselines"]
    for key in ("reduced", "native"):
        row = seeds[key]
        assert row["all119"]["auroc"]["min"] <= row["all119"]["auroc"]["max"]
        assert row["warmup_epochs"] < row["total_epochs"]


def test_generators_do_not_type_in_record_numbers():
    """Figures that the JSON records hold must not appear as literals."""
    literals = (
        r"\b56\\?%",
        r"\b44\\?%",
        r"\b33\\?%",
        r"\b0\.596\b",
        r"\b0\.105\b",
        r"\b0\.440\b",
        r"\b0\.930\b",
        r"\b0\.82 ms\b",
        r"\b30,436\b",
        r"\b2,316\b",
        r"\b0\.986\b",
        r"\b16 (review )?days\b",
        r"\b21 of 31\b",
        r"\b641 (bins|strict)\b",
        r"\b12,409\b",
        r"\b6,843\b",
    )
    for name in GENERATORS:
        text = (ROOT / name).read_text()
        for pattern in literals:
            assert not re.search(pattern, text), (name, pattern)


def test_training_sizes_and_native_comparator_facts_come_from_the_records(records):
    fx = facts.facts(records)
    own, native = fx["ours_training_sizes"], fx["native_detection"]
    assert all(isinstance(f["train"], int) for f in own)
    assert sum(f["test"] for f in own) == fx["shots"]["all119"]
    assert all(f["train"] < 30 for f in native["folds"])
    assert min(f["train"] for f in own) > 2 * max(f["train"] for f in native["folds"])
    assert native["has_ours_native_folds"]
    photodiode = fx["photodiode"]
    assert photodiode["identical"] == photodiode["records"] == 2 * photodiode["shots"]


def test_photodiode_facts_are_absent_without_the_record():
    assert facts.photodiode_facts(None) is None
