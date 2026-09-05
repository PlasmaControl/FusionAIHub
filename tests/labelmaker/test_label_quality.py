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
    # Pin the exact value, independently hand-verified
    # for these inputs - bin0 (prob < 0.5: 0.2, 0.3) n=2, mean_prob 0.25 vs
    # observed 0.5 -> contributes 2/5 * 0.25 = 0.10; bin1 (0.7, 0.9, 0.6) n=3,
    # mean_prob 0.73333 vs observed 0.66667 -> contributes 3/5 * 0.06667 =
    # 0.04. Total 0.14. A wrong denominator (bin count instead of N) or a
    # sign error would survive the loose `0.0 <= ece <= 1.0` bound above but
    # not this.
    assert abs(got["ece"] - 0.14) < 1e-9


def test_ece_all_ones_truth_returns_auroc_none():
    """The `n_neg == 0` branch: all-positive truth means no
    negative to rank against, so AUROC is undefined (`None`), not 1.0 or 0.0
    - and the function must not raise dividing by a zero `n_neg`.
    """
    got = validate.binary_metrics(np.array([0.6, 0.7, 0.8]), np.array([1, 1, 1]))
    assert got["auroc"] is None
    assert got["n_positive"] == 3
    assert got["calibration"] == []


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
    reasoning on `_ks_statistic`, which this mirrors: a loader-ordering
    problem shared with labelmaker's fdp scaling path, Task 16b - not a
    defect unique to scipy). This test is the proof: it imports scipy
    itself, as an independent oracle, exactly the way `test_ks_statistic.py`
    does, guarded with `importorskip` so it degrades gracefully wherever
    that import is not available (now unconditionally importable in this
    environment after Task 16b's `pyproject.toml` activation fix).
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


def test_label_quality_skip_reasons_histogram_and_warning(tmp_path, monkeypatch):
    """Many shots skipping for the identical underlying
    cause must collapse into one histogram bucket with a warning, not vanish
    into `skipped`'s per-shot dict of numpy-repr strings - which is exactly
    how the dead fdp scaling path went undiagnosed from
    this report alone. Also checks the per-shot diagnosis (`skip_diagnosis`)
    is populated, not thrown away, at skip time.
    """
    import h5py

    n = 3
    fake_archive = {
        "x0": np.zeros((n, 11)), "x1": np.zeros((n, 33, 5)),
        "y": np.zeros((n, 2)), "rows": np.arange(n),
    }
    shots = [111, 222, 333, 444, 555]

    monkeypatch.setattr(validate, "archive_rows", lambda shot, archive=TM_ARCHIVE: fake_archive)
    monkeypatch.setattr(registry, "load_adapter", lambda slug: _fake_tearing_adapter())

    features_dir = tmp_path / "features"
    features_dir.mkdir()
    for shot in shots:
        with h5py.File(features_dir / f"{shot}_features.h5", "w"):
            pass  # every field absent -> the identical "constant" ValueError each time

    report = validate.label_quality(
        "d3d_tearing_onset_cnn1d", shots, Paths(root=tmp_path)
    )
    assert report["n_shots_used"] == 0
    assert len(report["skipped"]) == 5
    histogram = report["skip_reasons"]["histogram"]
    # All five collapse into one bucket despite each raw message carrying a
    # distinct numpy array repr in its "(std=...)" suffix.
    assert len(histogram) == 1
    ((cause, count),) = histogram.items()
    assert count == 5
    assert "std=" not in cause  # the numeric detail was stripped
    assert any("5 of 5" in w for w in report["skip_reasons"]["warnings"])
    assert all(str(s) in report["skip_diagnosis"] for s in shots)


def test_label_quality_asserts_truth_column_shapes(tmp_path, monkeypatch):
    """The model-side truth-column assertion cannot see
    an ARCHIVE-side column swap. Built here with `betan` and `tm_prob`
    swapped: column 0 binary, column 1 continuous - the opposite of what
    `_TRUTH_COLUMNS` (`betan: 0, tm_prob: 1`) declares.
    """
    n = 500
    archive = tmp_path / "archive"
    archive.mkdir()
    rng = np.random.default_rng(0)
    np.save(archive / "z.npy", np.zeros(n, dtype=np.int64))
    np.save(archive / "x0.npy", np.zeros((n, 11)))
    np.save(archive / "x1.npy", np.zeros((n, 33, 5)))
    swapped_y = np.stack(
        [rng.integers(0, 2, n).astype(np.float64), rng.uniform(0.0, 5.0, n)], axis=1
    )
    np.save(archive / "y.npy", swapped_y)
    monkeypatch.setattr(registry, "load_adapter", lambda slug: _fake_tearing_adapter())
    validate._archive_shot_ids.cache_clear()
    with pytest.raises(ValueError, match="column swap|not binary"):
        validate.label_quality("d3d_tearing_onset_cnn1d", [], Paths.from_env(), archive=archive)
    validate._archive_shot_ids.cache_clear()


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
                "n_shots_requested": 100,
                "row_counts_total": {"valid": 800, "invalid": 200},
                "archived_inputs_valid": {"tm_prob": {"auroc": 0.9, "f1_at_0.5": 0.5}},
                "reconstructed_inputs_valid": {"tm_prob": {"auroc": 0.8, "f1_at_0.5": 0.4}},
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
    """The headline number must be `reconstructed_inputs_valid` (the
    row-matched, valid-row-only, published score), not
    `reconstructed_inputs_all`. This locks that choice down: if a future
    edit swapped the source dict, this test would catch the wrong number
    reaching the card.
    """
    results = validate.model_index_results(
        {
            "label_quality": {
                "n_shots_used": 10,
                "n_shots_requested": 10,
                "row_counts_total": {"valid": 90, "invalid": 10},
                "archived_inputs_valid": {"tm_prob": {"auroc": 0.9}},
                "reconstructed_inputs_valid": {"tm_prob": {"auroc": 0.42}},
                "reconstructed_inputs_all": {"tm_prob": {"auroc": 0.11}},
            }
        }
    )
    recon = next(r for r in results if r["task"]["name"] == "tm_prob"
                 and any("reconstructed" in m["name"] for m in r["metrics"]))
    value = next(m["value"] for m in recon["metrics"] if "reconstructed" in m["name"])
    assert value == pytest.approx(0.42)


def test_model_index_results_dataset_name_states_shots_and_rows():
    """A card-only reader must be able to see the
    row denominators and that labelmaker's validity mask was applied,
    without opening the JSON - "d3d overlap shots (n=31)" alone hid that
    the headline numbers were a 31-of-100-shot, valid-rows-only measurement.
    """
    results = validate.model_index_results(
        {
            "label_quality": {
                "n_shots_used": 31,
                "n_shots_requested": 100,
                "row_counts_total": {"valid": 1748, "invalid": 558},
                "archived_inputs_valid": {"tm_prob": {"auroc": 0.9}},
                "reconstructed_inputs_valid": {"tm_prob": {"auroc": 0.8}},
            }
        }
    )
    name = results[0]["dataset"]["name"]
    assert "31" in name and "100" in name
    assert "1748" in name and "2306" in name  # 1748 + 558


def test_model_index_results_dataset_name_falls_back_without_row_counts():
    """No `row_counts_total` (e.g. a report shape from before C1) must not
    put a bare "None" into the published dataset name.
    """
    results = validate.model_index_results(
        {
            "label_quality": {
                "n_shots_used": 7,
                "archived_inputs_valid": {"tm_prob": {"auroc": 0.9}},
                "reconstructed_inputs_valid": {"tm_prob": {"auroc": 0.8}},
            }
        }
    )
    assert "None" not in results[0]["dataset"]["name"]
    assert "7" in results[0]["dataset"]["name"]


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
                "n_shots_requested": 1,
                "row_counts_total": {"valid": 30, "invalid": 5},
                "archived_inputs_valid": {"tm_prob": {"auroc": 0.9, "f1_at_0.5": 0.5}},
                "reconstructed_inputs_valid": {"tm_prob": {"auroc": 0.8, "f1_at_0.5": 0.4}},
            }
        }
    )
    registry.update_model_index("d3d_tearing_onset_cnn1d", results)
    assert registry.card_discrepancies("d3d_tearing_onset_cnn1d") == []


def test_update_model_index_preserves_approximations_content(monkeypatch, tmp_path):
    """`card_discrepancies` never inspects
    `labelmaker.approximations`, so a future `safe_dump` change that mangled
    those 12 prose entries would pass the whole suite silently. A
    `yaml.safe_load` before/after equality test on that block closes it
    cheaply, against the real card - the one this rewrite actually touches.
    """
    real = registry.card_path("d3d_tearing_onset_cnn1d")
    text_before = real.read_text()
    approximations_before = registry.parse_card(text_before)["labelmaker"][
        "approximations"
    ]
    assert len(approximations_before) == 12

    copy = tmp_path / "README.md"
    copy.write_text(text_before)
    monkeypatch.setattr(
        registry, "card_path",
        lambda slug: copy if slug == "d3d_tearing_onset_cnn1d" else registry.MODELS_DIR / slug / "README.md",
    )
    results = validate.model_index_results(
        {
            "label_quality": {
                "n_shots_used": 1,
                "n_shots_requested": 1,
                "row_counts_total": {"valid": 30, "invalid": 5},
                "archived_inputs_valid": {"tm_prob": {"auroc": 0.9}},
                "reconstructed_inputs_valid": {"tm_prob": {"auroc": 0.8}},
            }
        }
    )
    registry.update_model_index("d3d_tearing_onset_cnn1d", results)
    approximations_after = registry.parse_card(copy.read_text())["labelmaker"][
        "approximations"
    ]
    assert approximations_after == approximations_before


@pytest.mark.skipif(not TM_ARCHIVE.exists(), reason="tm archive not available")
@pytest.mark.skipif(
    not (Paths.from_env().features / "185945_features.h5").exists(),
    reason="run `features` on shot 185945 first",
)
def test_label_quality_report_on_one_real_shot():
    report = validate.label_quality("d3d_tearing_onset_cnn1d", [185945], Paths.from_env())
    assert report["n_shots_used"] == 1
    for key in ("archived_inputs_all", "archived_inputs_valid",
                "reconstructed_inputs_valid", "reconstructed_inputs_all", "row_counts"):
        assert "tm_prob" in report[key] or "betan" in report[key]
    counts = report["row_counts"]["betan"]
    assert counts["matched"] == report["row_counts_total"]["valid"] + report["row_counts_total"]["invalid"]
    assert counts["valid"] + counts["invalid"] == counts["matched"]
    # The penalty is computed from the row-matched pair and says so.
    penalty = report["reconstruction_penalty"]["betan"]
    a_v = report["archived_inputs_valid"]["betan"]["rmse"]
    o_v = report["reconstructed_inputs_valid"]["betan"]["rmse"]
    assert abs(penalty["rmse"] - (o_v - a_v)) < 1e-9
    assert penalty["n_rows"] == counts["valid"]
    assert "row-matched" in penalty["computed_from"]


def test_reconstruction_penalty_is_computed_row_matched_not_from_all_matched(monkeypatch):
    """The two headline scores must come from the same
    row set. `theirs` (archived, `bt=0`) and `ours` (reconstructed, `bt=2`)
    predict different baseline values, and the last two of six rows are
    marked invalid and predicted wildly wrong on BOTH sides - so
    `archived_inputs_all` and `archived_inputs_valid` disagree, and a
    penalty computed from the wrong pair would land far from the row-matched
    one, not merely differ in the last decimal place.
    """
    import labelmaker.validate as validate_mod
    from labelmaker.models import registry as registry_mod

    n = 6
    n_invalid = 2
    fake_archive = {
        "x0": np.zeros((n, 9)),
        "x1": np.zeros((n, 33, 0)),
        "y": np.tile(np.array([2.0, 0.0]), (n, 1)),  # betan truth: 2.0 everywhere
        "rows": np.arange(n),
    }

    class FakeBuilt:
        def __init__(self):
            self.t = np.arange(n, dtype=np.float64) * 0.025
            self.scalars = np.tile(
                np.array([2.0, 1.0e6, 0.0, 0.0, 0.0, 0.0, 0.1, -0.1, 1.0]), (n, 1)
            )
            self.profiles = np.zeros((n, 33, 0))
            self.valid = np.array([True] * (n - n_invalid) + [False] * n_invalid)
            self.missing = ()
            self.resolvers = {"bt": "archive"}

    def fake_matched_shot(shot, spec, paths, archive):
        if shot != 111:
            return validate_mod._ShotMatch(skip_reason="no archived rows")
        built = FakeBuilt()
        info = {
            "index": np.arange(n), "distance": np.zeros(n), "median_distance": 0.0,
            "max_distance": 0.0, "monotonic": True, "n_archived_rows": n,
            "n_matched": n, "n_unique_matched": n, "fail_reason": None,
            "passed": True,
        }
        return validate_mod._ShotMatch(
            got=fake_archive, built=built, info=info, features={}
        )

    def fake_predict(built):
        bt = np.asarray(built.scalars)[:, 0]
        pred = np.where(bt > 1.0, 3.0, 2.0)  # ours (bt=2) -> 3.0, theirs (bt=0) -> 2.0
        pred = pred.copy()
        pred[n - n_invalid:] = 99.0  # both sides, wildly wrong on the invalid rows
        return pred.reshape(1, n, 1)

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
    adapter = ModelAdapter(
        slug="d3d_tearing_onset_cnn1d", card_id="test/fake", framework="none",
        time_step_ms=25.0, artifacts=(), upstream="none",
        input_spec=InputSpec(fields=fields, dt_s=0.025),
        output_spec=OutputSpec(fields=(
            OutputField("betan", "regression", column=0, activation="none"),
        )),
        load=lambda model_dir: fake_predict,
        ensemble_n=1,
    )
    monkeypatch.setattr(registry_mod, "load_adapter", lambda slug: adapter)
    monkeypatch.setattr(validate_mod, "_matched_shot", fake_matched_shot)

    report = validate_mod.label_quality(
        "d3d_tearing_onset_cnn1d", [111], Paths.from_env()
    )

    a_all = report["archived_inputs_all"]["betan"]["rmse"]
    a_valid = report["archived_inputs_valid"]["betan"]["rmse"]
    assert a_all > 10.0  # dominated by the two 99.0-vs-2.0 invalid rows
    assert a_valid == pytest.approx(0.0)  # theirs matches truth exactly on valid rows

    o_valid = report["reconstructed_inputs_valid"]["betan"]["rmse"]
    assert o_valid == pytest.approx(1.0)  # |3.0 - 2.0| on every valid row

    penalty = report["reconstruction_penalty"]["betan"]
    assert penalty["rmse"] == pytest.approx(o_valid - a_valid)
    assert abs(penalty["rmse"] - (o_valid - a_all)) > 1.0
    assert penalty["n_rows"] == n - n_invalid
    assert "row-matched" in penalty["computed_from"]
