"""Metrics, the published-vs-all-matched split, and the card that records them.

Task 16. Where the brief's example code and the addendum disagree with the
current source, the addendum wins - see its eight numbered items, referenced
by number in the tests below that exist because of it.
"""
import numpy as np
import pytest

from labelmaker import validate
from labelmaker.catalog import TM_ARCHIVE
from labelmaker.config import Paths
from labelmaker.models import registry
from labelmaker.models.base import (
    InputField,
    InputSpec,
    ModelAdapter,
    OutputField,
    OutputSpec,
)


def _fake_tearing_adapter(ensemble_n=1):
    """A stand-in for d3d_tearing_onset_cnn1d: the same scalar order at
    MATCH_COLUMNS (indices 0, 1, 6, 7, 8 -> bt, ip, tritop, tribot, gapin)
    and the same output column order at _TRUTH_COLUMNS (betan=0,
    tm_prob=1), but with no real weights to load - for tests that exercise
    label_quality's per-shot loop without `/projects/EKOLEMEN` mounted.
    """
    fields = (
        InputField("bt", "bt", lag="t+dt"),
        InputField("ip", "ip", lag="t+dt"),
        InputField("pinj", "pinj_total", lag="t+dt"),
        InputField("tinj", "tinj_total", lag="t+dt"),
        InputField("r0", "r0", lag="t+dt"),
        InputField("kappa", "kappa", lag="t+dt"),
        InputField("tritop_EFIT01", "tritop", lag="t+dt"),
        InputField("tribot_EFIT01", "tribot", lag="t+dt"),
        InputField("gapin_EFIT01", "gapin", lag="t+dt"),
    )

    def predict(built):
        return np.zeros((ensemble_n, built.t.size, 2))

    return ModelAdapter(
        slug="d3d_tearing_onset_cnn1d", card_id="test/fake", framework="none",
        time_step_ms=25.0, artifacts=(), upstream="none",
        input_spec=InputSpec(fields=fields, dt_s=0.025),
        output_spec=OutputSpec(fields=(
            OutputField("betan", "regression", column=0, activation="none"),
            OutputField("tm_prob", "binary", column=1, activation="sigmoid"),
        )),
        load=lambda model_dir: predict,
        ensemble_n=ensemble_n,
    )


def test_auroc_matches_hand_computed_cases():
    truth = np.array([0, 0, 1, 1])
    assert validate.binary_metrics(np.array([0.1, 0.2, 0.8, 0.9]), truth)["auroc"] == 1.0
    assert validate.binary_metrics(np.array([0.9, 0.8, 0.2, 0.1]), truth)["auroc"] == 0.0
    assert validate.binary_metrics(np.array([0.5, 0.5, 0.5, 0.5]), truth)["auroc"] == 0.5
    # one swapped pair out of four -> 0.75
    got = validate.binary_metrics(np.array([0.1, 0.85, 0.8, 0.9]), truth)["auroc"]
    assert abs(got - 0.75) < 1e-12


def test_f1_and_brier_and_calibration():
    truth = np.array([0, 0, 1, 1, 1])
    prob = np.array([0.2, 0.7, 0.9, 0.6, 0.3])
    got = validate.binary_metrics(prob, truth, bins=2)
    # at 0.5: predictions 0,1,1,1,0 -> tp=2, fp=1, fn=1 -> f1 = 2*2/(2*2+1+1)
    assert abs(got["f1_at_0.5"] - 4 / 6) < 1e-12
    assert abs(got["brier"] - np.mean((prob - truth) ** 2)) < 1e-12
    assert got["n"] == 5 and got["n_positive"] == 3
    assert len(got["calibration"]) == 2
    assert 0.0 <= got["ece"] <= 1.0


def test_metrics_refuse_a_degenerate_truth_vector():
    got = validate.binary_metrics(np.array([0.1, 0.2]), np.array([0, 0]))
    assert got["auroc"] is None and got["n_positive"] == 0


def test_metrics_refuse_an_empty_input():
    got = validate.binary_metrics(np.array([]), np.array([]))
    assert got["n"] == 0 and got["auroc"] is None and got["calibration"] == []


def test_regression_metrics():
    got = validate.regression_metrics(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
    assert abs(got["rmse"] - 0.5) < 1e-12
    assert abs(got["bias"] + 0.5) < 1e-12


@pytest.mark.parametrize("seed", range(6))
def test_rankdata_matches_scipy_oracle_with_ties(seed):
    """Addendum item 1: `_rankdata` replaces `scipy.stats.rankdata` at
    runtime (scipy must not be imported by validate.py - see the module's
    reasoning on `_ks_statistic`, which this mirrors). This test is the
    proof: it imports scipy itself, as an independent oracle, exactly the
    way `test_ks_statistic.py` does, and is only able to because some
    test-collection plugin loads a compatible libstdc++ before torch does -
    an environment quirk that holds under pytest and nowhere else.
    """
    scipy_stats = pytest.importorskip("scipy.stats", reason="scipy not importable here")
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 80))
    a = rng.integers(0, 10, size=n).astype(np.float64)  # small range -> heavy ties
    got = validate._rankdata(a)
    want = scipy_stats.rankdata(a, method="average")
    np.testing.assert_allclose(got, want)


def test_rankdata_matches_scipy_oracle_continuous():
    scipy_stats = pytest.importorskip("scipy.stats", reason="scipy not importable here")
    rng = np.random.default_rng(7)
    a = rng.normal(size=113)
    np.testing.assert_allclose(validate._rankdata(a), scipy_stats.rankdata(a, method="average"))


def test_score_field_separates_published_from_all_matched_rows():
    """Addendum item 6: labelmaker masks invalid rows out of what it
    publishes, so scoring every matched row measures something the package
    never emits. The two invalid rows here are deliberately mis-scored so
    `all_matched` and `published` diverge, not merely differ in `n`.
    """
    truth = np.array([0, 0, 1, 1, 1, 0])
    pred = np.array([0.1, 0.2, 0.8, 0.9, 0.05, 0.95])
    valid = np.array([True, True, True, True, False, False])
    got = validate._score_field("binary", pred, truth, valid)
    assert got["n_valid"] == 4
    assert got["n_invalid"] == 2
    assert got["published"]["auroc"] == 1.0
    assert got["all_matched"]["auroc"] < 1.0
    assert got["all_matched"]["n"] == 6
    assert got["published"]["n"] == 4


def test_score_field_all_rows_valid_matches_all_matched():
    truth = np.array([0.0, 1.0, 1.0, 0.0])
    pred = np.array([0.2, 0.8, 0.6, 0.3])
    valid = np.array([True, True, True, True])
    got = validate._score_field("binary", pred, truth, valid)
    assert got["published"] == got["all_matched"]
    assert got["n_invalid"] == 0


def test_label_quality_asserts_the_truth_column_mapping(monkeypatch):
    """Addendum item 5: `_TRUTH_COLUMNS` indexes `y.npy`'s column order,
    which agrees with `OUTPUT_SPEC`'s own `column` only because upstream
    happened to build both in the same order. If that coincidence ever
    broke, scoring would silently compare predictions to the wrong truth
    column and still look plausible - so this is checked once, loudly,
    before any shot is touched (and before the model's weights are loaded,
    so this test needs no real weights on disk).
    """
    import labelmaker.validate as validate_mod

    original = dict(validate_mod._TRUTH_COLUMNS)
    validate_mod._TRUTH_COLUMNS = {"tm_prob": 0, "betan": 1}  # swapped
    try:
        with pytest.raises(ValueError, match="truth column"):
            validate.label_quality("d3d_tearing_onset_cnn1d", [], Paths.from_env())
    finally:
        validate_mod._TRUTH_COLUMNS = original


def test_label_quality_asserts_the_match_column_mapping():
    """I10, shared with `reconstruction_fidelity`: MATCH_COLUMNS indexes
    d3d_tearing_onset_cnn1d's scalar order specifically, and label_quality
    performs the same row alignment reconstruction_fidelity does, so it
    needs the same loud guard - checked before adapter.load, so this test
    needs no real weights on disk either.
    """
    import labelmaker.validate as validate_mod

    original = validate_mod.MATCH_COLUMNS
    validate_mod.MATCH_COLUMNS = (1, 0, 6, 7, 8)  # bt/ip swapped
    try:
        with pytest.raises(ValueError, match="MATCH_COLUMNS"):
            validate.label_quality("d3d_tearing_onset_cnn1d", [], Paths.from_env())
    finally:
        validate_mod.MATCH_COLUMNS = original


def test_label_quality_isolates_a_per_shot_crash(tmp_path, monkeypatch):
    """C1, shared with `reconstruction_fidelity`: one shot's bad data must
    not abort the whole validation run. A shot missing every one of the five
    `MATCH_COLUMNS` features makes `match_rows`' variance guard raise
    `ValueError` on a constant column - reachable per-shot, so it must cost
    only that shot.
    """
    import h5py

    n = 3
    fake_archive = {
        "x0": np.zeros((n, 11)),
        "x1": np.zeros((n, 33, 5)),
        "y": np.zeros((n, 2)),
        "rows": np.arange(n),
    }

    def fake_archive_rows(shot, archive=TM_ARCHIVE):
        return fake_archive if shot == 111 else None

    monkeypatch.setattr(validate, "archive_rows", fake_archive_rows)
    monkeypatch.setattr(registry, "load_adapter", lambda slug: _fake_tearing_adapter())

    features_dir = tmp_path / "features"
    features_dir.mkdir()
    with h5py.File(features_dir / "111_features.h5", "w"):
        pass  # every field absent

    report = validate.label_quality(
        "d3d_tearing_onset_cnn1d", [111, 222], Paths(root=tmp_path)
    )
    assert report["n_shots_requested"] == 2
    assert report["n_shots_used"] == 0
    assert report["skipped"]["222"] == "no archived rows"
    assert "ValueError" in report["skipped"]["111"]
    assert "constant" in report["skipped"]["111"]


def test_label_quality_counts_shots_before_the_generator_is_consumed():
    def shots():
        yield 999999999  # not in the archive; must still be consumed

    report = validate.label_quality(
        "d3d_tearing_onset_cnn1d", shots(), Paths.from_env()
    )
    assert report["n_shots_requested"] == 1
    assert report["skipped"]["999999999"] == "no archived rows"


def test_model_index_results_shape():
    results = validate.model_index_results(
        {
            "adapter_fidelity": {"passed": True, "max_abs_diff": 1e-7},
            "label_quality": {
                "n_shots_used": 42,
                "archived_inputs": {"tm_prob": {"auroc": 0.9, "f1_at_0.5": 0.5}},
                "reconstructed_inputs": {"tm_prob": {"auroc": 0.8, "f1_at_0.5": 0.4}},
            },
        }
    )
    names = {r["metrics"][0]["name"] for r in results}
    assert "auroc (archived inputs)" in names
    assert "auroc (reconstructed inputs)" in names
    for r in results:
        assert r["task"]["type"] and r["dataset"]["name"]
        assert isinstance(r["metrics"][0]["value"], float)


def test_model_index_results_reads_the_published_reconstructed_score():
    """The headline number must be `reconstructed_inputs` (already the
    valid-row-only, published score - addendum item 6), not
    `reconstructed_inputs_all_matched`. This locks that choice down: if a
    future edit swapped the source dict, this test would catch the wrong
    number reaching the card.
    """
    results = validate.model_index_results(
        {
            "label_quality": {
                "n_shots_used": 10,
                "archived_inputs": {"tm_prob": {"auroc": 0.9}},
                "reconstructed_inputs": {"tm_prob": {"auroc": 0.42}},
                "reconstructed_inputs_all_matched": {"tm_prob": {"auroc": 0.11}},
            }
        }
    )
    recon = next(r for r in results if r["task"]["name"] == "tm_prob"
                 and any("reconstructed" in m["name"] for m in r["metrics"]))
    value = next(m["value"] for m in recon["metrics"] if "reconstructed" in m["name"])
    assert value == pytest.approx(0.42)


def test_update_model_index_rewrites_only_the_results(tmp_path, monkeypatch):
    card = tmp_path / "README.md"
    card.write_text(
        "---\n"
        "library_name: keras\n"
        "model-index:\n"
        "  - name: d3d-tearing-onset-cnn1d\n"
        "    results: []\n"
        "labelmaker:\n"
        "  status: implemented\n"
        "  slug: s\n"
        "---\n\n# Title\n\nProse that must survive.\n"
    )
    monkeypatch.setattr(registry, "card_path", lambda slug: card)
    registry.update_model_index(
        "s",
        [
            {
                "task": {"type": "tabular-classification"},
                "dataset": {"name": "d3d overlap shots", "type": "d3d"},
                "metrics": [{"name": "auroc (archived inputs)", "type": "roc_auc",
                             "value": 0.91}],
            }
        ],
    )
    text = card.read_text()
    assert "Prose that must survive." in text
    parsed = registry.parse_card(text)
    assert parsed["model-index"][0]["results"][0]["metrics"][0]["value"] == 0.91
    assert parsed["labelmaker"]["status"] == "implemented"
    assert parsed["library_name"] == "keras"


def test_update_model_index_leaves_card_discrepancies_empty(monkeypatch, tmp_path):
    """Addendum item 7: after a real card write, `card_discrepancies` must
    still return `[]` - it has previously caught a stray edit to this card.
    Runs against a copy of the real `d3d_tearing_onset_cnn1d` card, not a
    minimal fixture, since that is the one this rewrite actually touches in
    production.
    """
    real = registry.card_path("d3d_tearing_onset_cnn1d")
    copy = tmp_path / "README.md"
    copy.write_text(real.read_text())
    monkeypatch.setattr(
        registry, "card_path",
        lambda slug: copy if slug == "d3d_tearing_onset_cnn1d" else registry.MODELS_DIR / slug / "README.md",
    )
    results = validate.model_index_results(
        {
            "label_quality": {
                "n_shots_used": 1,
                "archived_inputs": {"tm_prob": {"auroc": 0.9, "f1_at_0.5": 0.5}},
                "reconstructed_inputs": {"tm_prob": {"auroc": 0.8, "f1_at_0.5": 0.4}},
            }
        }
    )
    registry.update_model_index("d3d_tearing_onset_cnn1d", results)
    assert registry.card_discrepancies("d3d_tearing_onset_cnn1d") == []


@pytest.mark.skipif(not TM_ARCHIVE.exists(), reason="tm archive not available")
@pytest.mark.skipif(
    not (Paths.from_env().features / "185945_features.h5").exists(),
    reason="run `features` on shot 185945 first",
)
def test_label_quality_report_on_one_real_shot():
    report = validate.label_quality("d3d_tearing_onset_cnn1d", [185945], Paths.from_env())
    assert report["n_shots_used"] == 1
    for key in ("archived_inputs", "reconstructed_inputs",
                "reconstructed_inputs_all_matched", "row_counts"):
        assert "tm_prob" in report[key] or "betan" in report[key]
    counts = report["row_counts"]["betan"]
    assert counts["archived"] == counts["reconstructed_all_matched"]
    assert counts["reconstructed_published_valid"] + counts["reconstructed_published_invalid"] \
        == counts["reconstructed_all_matched"]
