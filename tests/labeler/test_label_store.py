"""Label files are corpus-shaped, provenanced, and indexable."""
import h5py
import numpy as np
import pandas as pd
import pytest

from labeler.labels.schema import LabelSpec, artifact_digest, group_path
from labeler.labels.store import (
    append_index,
    index_rows,
    labelled,
    read_label,
    write_labels,
)
from labeler.models.base import Decoded

SLUG = "d3d_tearing_onset_cnn1d"
T = 0.025 * np.arange(6)


def _specs():
    return (
        LabelSpec(
            name="tm_prob", task="binary", activation="sigmoid", units="",
            classes=("no_tearing", "tearing"), slug=SLUG,
            card_id="plasmacontrol/d3d-tearing-onset-cnn1d",
            time_step_ms=25.0, ensemble_n=10, artifact_sha256="abc123",
        ),
        LabelSpec(
            name="betan", task="regression", activation="none", units="",
            classes=(), slug=SLUG,
            card_id="plasmacontrol/d3d-tearing-onset-cnn1d",
            time_step_ms=25.0, ensemble_n=10, artifact_sha256="abc123",
        ),
    )


def _decoded():
    p = np.linspace(0.1, 0.6, 6)
    return {
        "tm_prob": Decoded(mean=p, lo=p - 0.05, hi=p + 0.05),
        "betan": Decoded(mean=p * 5, lo=p * 5 - 0.2, hi=p * 5 + 0.2),
    }


def _write(path, valid=None):
    valid = np.ones(6, bool) if valid is None else valid
    write_labels(
        path, 190000, T, _decoded(), _specs(), valid,
        run_id="run-test", features_sha256="f" * 64,
    )


def test_group_path_and_artifact_digest_are_stable():
    assert group_path(SLUG, "tm_prob") == f"{SLUG}/tm_prob"
    a = artifact_digest({"m0.h5": "aa", "m1.h5": "bb"})
    b = artifact_digest({"m1.h5": "bb", "m0.h5": "aa"})
    assert a == b and len(a) == 64            # order-independent, sha256 hex
    assert a != artifact_digest({"m0.h5": "aa"})


def test_round_trip_in_corpus_layout(tmp_path):
    p = tmp_path / "190000_labels.h5"
    _write(p)
    got = read_label(p, SLUG, "tm_prob")
    np.testing.assert_allclose(got.x, T)
    assert got.y.shape == (1, 6)
    np.testing.assert_allclose(got.y[0], np.linspace(0.1, 0.6, 6), rtol=1e-6)
    assert got.attrs["task"] == "binary"
    assert got.attrs["card_id"] == "plasmacontrol/d3d-tearing-onset-cnn1d"
    assert labelled(p) == {f"{SLUG}/tm_prob", f"{SLUG}/betan"}


def test_spread_and_validity_companions(tmp_path):
    p = tmp_path / "190000_labels.h5"
    valid = np.array([1, 1, 0, 0, 1, 1], dtype=bool)
    _write(p, valid)
    with h5py.File(p, "r") as f:
        g = f[SLUG]
        assert g["tm_prob_spread"]["ydata"].shape == (2, 6)
        np.testing.assert_allclose(
            g["tm_prob_spread"]["ydata"][0], np.linspace(0.1, 0.6, 6) - 0.05,
            rtol=1e-6,
        )
        v = g["tm_prob_valid"]["ydata"]
        assert v.shape == (1, 6) and v.dtype == np.uint8
        assert v[0].tolist() == [1, 1, 0, 0, 1, 1]


def test_provenance_attrs_on_the_file(tmp_path):
    p = tmp_path / "190000_labels.h5"
    _write(p)
    with h5py.File(p, "r") as f:
        assert f.attrs["shot"] == 190000
        assert f.attrs["run_id"] == "run-test"
        assert f.attrs["features_sha256"] == "f" * 64
        assert f.attrs["labelmaker_version"] and f.attrs["git_sha"]
        assert f[SLUG]["tm_prob"].attrs["ensemble_n"] == 10
        assert f[SLUG]["tm_prob"].attrs["time_step_ms"] == 25.0
        assert f[SLUG]["tm_prob"].attrs["artifact_sha256"] == "abc123"
        assert list(f[SLUG]["tm_prob"].attrs["classes"]) == ["no_tearing", "tearing"]


def test_write_is_atomic_and_merges_other_models(tmp_path):
    p = tmp_path / "190000_labels.h5"
    _write(p)
    other = (
        LabelSpec(
            name="elm_hazard", task="regression", activation="none", units="1/s",
            classes=(), slug="d3d_elm_time_to_event_dsm", card_id="x/y",
            time_step_ms=50.0, ensemble_n=1, artifact_sha256="def456",
        ),
    )
    d = {"elm_hazard": Decoded(mean=np.zeros(6), lo=np.zeros(6), hi=np.zeros(6))}
    write_labels(p, 190000, T, d, other, np.ones(6, bool),
                 run_id="run-2", features_sha256="f" * 64)
    assert labelled(p) == {
        f"{SLUG}/tm_prob", f"{SLUG}/betan",
        "d3d_elm_time_to_event_dsm/elm_hazard",
    }
    assert list(tmp_path.iterdir()) == [p]


def test_index_rows_summarise_each_label(tmp_path):
    p = tmp_path / "190000_labels.h5"
    _write(p, np.array([1, 1, 0, 0, 1, 1], bool))
    rows = {r["label"]: r for r in index_rows(p)}
    assert set(rows) == {"tm_prob", "betan"}
    r = rows["tm_prob"]
    assert r["shot"] == 190000 and r["slug"] == SLUG and r["task"] == "binary"
    assert r["n_total"] == 6 and r["n_valid"] == 4
    assert 0.0 <= r["mean_valid"] <= 1.0
    assert r["run_id"] == "run-test"


def test_append_index_can_clear_a_replaced_shot_with_no_new_events(tmp_path):
    import pandas as pd

    idx = tmp_path / "index.parquet"
    keys = ["shot", "source", "phenomenon"]
    rows = [{"shot": shot, "source": "elm_clock", "phenomenon": "elm"}
            for shot in (100, 101)]
    append_index(idx, rows, keys=keys)
    append_index(idx, [], keys=keys, replace_shots=[100])
    assert pd.read_parquet(idx).shot.tolist() == [101]
    append_index(idx, [], keys=keys, replace_shots=[101])
    assert pd.read_parquet(idx).empty


def test_append_index_is_idempotent_per_shot_and_label(tmp_path):
    p = tmp_path / "190000_labels.h5"
    _write(p)
    idx = tmp_path / "labels_index.parquet"
    append_index(idx, index_rows(p))
    append_index(idx, index_rows(p))          # same shot again
    df = pd.read_parquet(idx)
    assert len(df) == 2                        # not 4
    assert set(df["label"]) == {"tm_prob", "betan"}
    assert list(tmp_path.iterdir()).count(idx) == 1


def test_append_index_keeps_the_run_id_of_the_last_write(tmp_path):
    # The default keys are (shot, slug, label): a re-run replaces its own row.
    p = tmp_path / "190000_labels.h5"
    _write(p)
    idx = tmp_path / "labels_index.parquet"
    append_index(idx, index_rows(p))
    rows = [dict(r, run_id="run-2") for r in index_rows(p)]
    append_index(idx, rows)
    df = pd.read_parquet(idx)
    assert df["run_id"].tolist() == ["run-2", "run-2"]
    assert df[["shot", "slug", "label"]].values.tolist() == [
        [190000, SLUG, "betan"], [190000, SLUG, "tm_prob"],
    ]                                          # sorted by the key columns


def test_append_index_takes_other_key_columns(tmp_path):
    # The events index is keyed by (shot, source, phenomenon) instead.
    idx = tmp_path / "events_index.parquet"
    keys = ["shot", "source", "phenomenon"]
    first = [
        {"shot": 190000, "source": "tokeye_track", "phenomenon": "eho",
         "n_events": 3},
        {"shot": 190000, "source": "ece_sawtooth", "phenomenon": "sawtooth",
         "n_events": 45},
    ]
    append_index(idx, first, keys=keys)
    append_index(idx, [dict(first[0], n_events=7)], keys=keys)
    df = pd.read_parquet(idx)
    assert len(df) == 2
    assert df["source"].tolist() == ["ece_sawtooth", "tokeye_track"]  # key sort
    assert df.set_index("source")["n_events"].to_dict() == {
        "ece_sawtooth": 45, "tokeye_track": 7,
    }


def test_a_failed_write_leaves_no_temp_file(tmp_path):
    # decoded is missing the spec's label, so the loop raises after the temp
    # file is open. See the features-store counterpart for why this matters.
    p = tmp_path / "190000_labels.h5"
    with pytest.raises(KeyError):
        write_labels(p, 190000, T, {}, _specs(), np.ones(6, bool),
                     run_id="r", features_sha256="f" * 64)
    assert list(tmp_path.iterdir()) == []


def test_specs_must_share_one_slug(tmp_path):
    # write_labels writes everything into specs[0]'s group, so a mixed batch
    # would silently file the rest under the wrong model.
    p = tmp_path / "190000_labels.h5"
    other = LabelSpec(
        name="x", task="binary", activation="none", units="", classes=(),
        slug="another_model", card_id="a/b", time_step_ms=25.0, ensemble_n=1,
        artifact_sha256="ab",
    )
    with pytest.raises(ValueError, match="one slug"):
        write_labels(p, 190000, T, _decoded(), (*_specs(), other),
                     np.ones(6, bool), run_id="r", features_sha256="f" * 64)


def test_merge_false_replaces_the_whole_file(tmp_path):
    p = tmp_path / "190000_labels.h5"
    _write(p)
    other = (
        LabelSpec(
            name="elm_hazard", task="regression", activation="none", units="1/s",
            classes=(), slug="d3d_elm_time_to_event_dsm", card_id="x/y",
            time_step_ms=50.0, ensemble_n=1, artifact_sha256="def456",
        ),
    )
    d = {"elm_hazard": Decoded(mean=np.zeros(6), lo=np.zeros(6), hi=np.zeros(6))}
    write_labels(p, 190000, T, d, other, np.ones(6, bool),
                 run_id="run-2", features_sha256="f" * 64, merge=False)
    assert labelled(p) == {"d3d_elm_time_to_event_dsm/elm_hazard"}


def test_read_label_raises_for_an_absent_label(tmp_path):
    p = tmp_path / "190000_labels.h5"
    _write(p)
    with pytest.raises(KeyError):
        read_label(p, SLUG, "no_such_label")


def test_output_field_attrs_round_trip(tmp_path):
    from dataclasses import replace

    from labeler.labels.schema import specs_for
    from labeler.models import registry
    from labeler.models.base import OutputField, OutputSpec

    adapter = replace(registry.load_adapter(SLUG), output_spec=OutputSpec((
        OutputField("tm_prob", "binary", 0, attrs=(("calibration_fit_on", '{"shots":[1]}'),)),
    )))
    path = tmp_path / "labels.h5"
    write_labels(path, 1, T, _decoded(), specs_for(adapter, "abc"), np.ones(6, bool),
                 run_id="test", features_sha256="abc")
    assert read_label(path, SLUG, "tm_prob").attrs["calibration_fit_on"] == '{"shots":[1]}'
