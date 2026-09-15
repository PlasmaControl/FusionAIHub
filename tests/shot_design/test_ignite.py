"""IGNITE encoding, missing-data behavior and production frame-code parity.

Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shot_design.config import load_paths
from shot_design.env import getenv
from shot_design.shotdb import build, ignite


def test_input_defaults_to_existing_processed_corpus(paths, tmp_path):
    import h5py

    paths.foundation_model_processed_dir.mkdir()
    processed = paths.foundation_model_processed_dir / "900001_processed.h5"
    with h5py.File(processed, "w") as file:
        group = file.create_group("ts_core_density")
        group["xdata"] = np.array([0.0, 0.05], np.float32)
        group["ydata"] = np.array([[1.0, 2.0], [np.nan, np.nan]], np.float32)
    before = processed.read_bytes()
    directory, filled = ignite._resolve_input(900001, paths, None)
    assert directory == paths.foundation_model_processed_dir
    assert filled == {"ts_core_density": 1}
    assert processed.read_bytes() == before
    with pytest.raises(FileNotFoundError, match="no processed IGNITE input"):
        ignite._resolve_input(900001, paths, tmp_path)
    assert list(paths.ignite_inputs_dir.iterdir()) == []


def test_encode_records_reports_missing_processed_inputs(paths, monkeypatch):
    from types import SimpleNamespace

    calls = []

    def encode(shot, codecs, paths, **kwargs):
        ignite._resolve_input(shot, paths, None)
        calls.append(shot)

    monkeypatch.setattr(ignite, "encode_shot", encode)
    embeddings, errors = ignite.encode_records([SimpleNamespace(shot=900001)], {}, paths)
    assert embeddings == {} and calls == []
    assert "no processed IGNITE input" in errors[900001]


def test_load_codecs_says_what_to_do_when_the_directory_is_absent(tmp_path):
    with pytest.raises(ignite.CheckpointMissing) as e:
        ignite.load_codecs(tmp_path / "nope")
    msg = str(e.value)
    assert "shot_design model --download" in msg and "models_dir" in msg


def test_load_codecs_distinguishes_an_empty_directory_from_an_absent_one(tmp_path):
    empty = tmp_path / "models"
    empty.mkdir()
    with pytest.raises(ignite.CheckpointMissing, match="holds no codec checkpoints"):
        ignite.load_codecs(empty)


def test_encode_db_reports_not_installed_instead_of_raising(paths):
    """`shot_design build` must survive a missing model: a scalar-only database is fully usable and
    the manifest has to make the difference visible."""
    out = ignite.encode_db(paths.db_dir, [], paths)
    assert out["status"] == "not_installed" and out["channels"] == []
    assert "ignite" in out["reason"]


def test_build_still_takes_the_not_installed_path_now_that_this_module_exists(paths):
    """Before this module existed, build._encode synthesised `not_installed` from its own
    ImportError. Creating shot_design.shotdb.ignite stops that branch firing, so the status now has to
    come from encode_db -- and it must be the same status, or every built manifest changes
    meaning."""
    out = build._encode(paths.db_dir, [], paths)
    assert out["status"] == "not_installed"
    assert "ignite" in out["reason"]


def test_segment_embedding_is_the_mean_over_the_segments_frames():
    emb = ignite.ShotEmbedding(
        shot=1,
        modalities=("a",),
        dims=(2,),
        features=np.array([[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]], np.float32),
        mask=np.ones((3, 1), bool),
        t0=1.0,
    )
    assert np.allclose(emb.frame_times, [1.0, 1.05, 1.10])
    assert np.allclose(emb.segment(1.0, 1.10), [2.0, 3.0])  # first two frames
    assert np.all(np.isnan(emb.segment(5.0, 6.0)))  # no frames in range


def test_segment_embedding_ignores_a_modality_absent_from_some_frames():
    """A modality missing for part of a segment must not drag its mean toward zero, and one
    missing for the whole segment must stay NaN rather than silently become a real value."""
    emb = ignite.ShotEmbedding(
        shot=1,
        modalities=("a", "b"),
        dims=(1, 1),
        features=np.array([[1.0, np.nan], [3.0, np.nan]], np.float32),
        mask=np.array([[True, False], [True, False]]),
        t0=0.0,
    )
    seg = emb.segment(0.0, 1.0)
    assert seg[0] == pytest.approx(2.0) and np.isnan(seg[1])
    assert emb.present == ("a",)


def test_windows_are_cut_on_a_fixed_grid_from_t0_and_keep_a_partial_tail():
    feats = np.arange(7, dtype=np.float32).reshape(7, 1)  # 7 frames = 350 ms
    emb = ignite.ShotEmbedding(
        shot=1, modalities=("a",), dims=(1,), features=feats, mask=np.ones((7, 1), bool), t0=0.0
    )
    wins = emb.windows(0.25)  # 5 frames per window
    assert [(round(a, 3), round(b, 3)) for a, b, _ in wins] == [(0.0, 0.25), (0.25, 0.35)]
    assert wins[0][2][0] == pytest.approx(2.0) and wins[1][2][0] == pytest.approx(5.5)


def test_segment_matrix_follows_the_segment_table_order_and_nans_unencoded_shots():
    from shot_design.schema import Segment

    class Rec:
        def __init__(self, shot, segs):
            self.shot, self.segments = shot, segs

    emb = ignite.ShotEmbedding(
        shot=7,
        modalities=("a",),
        dims=(1,),
        features=np.array([[1.0], [3.0]], np.float32),
        mask=np.ones((2, 1), bool),
        t0=0.0,
    )
    recs = [
        Rec(7, [Segment(name="flat_top", t0_ms=0.0, t1_ms=100.0)]),
        Rec(8, [Segment(name="flat_top", t0_ms=0.0, t1_ms=100.0)]),
    ]
    mat = ignite.segment_matrix({7: emb}, recs, ["8:flat_top", "7:flat_top"], 1)
    assert np.isnan(mat[0, 0]) and mat[1, 0] == pytest.approx(2.0)


def test_window_table_ids_carry_the_shot_and_start_time():
    emb = ignite.ShotEmbedding(
        shot=7,
        modalities=("a",),
        dims=(1,),
        features=np.ones((6, 1), np.float32),
        mask=np.ones((6, 1), bool),
        t0=0.0,
    )
    df, mat = ignite.window_table({7: emb}, 0.25, 1)
    assert list(df.index) == ["7:0", "7:250"] and mat.shape == (2, 1) and mat.dtype == np.float16
    assert list(df["t1_ms"]) == [250.0, 300.0]


def test_coverage_is_read_off_the_final_matrix_per_shot():
    mat = np.array(
        [
            [1.0, 1.0, np.nan],  # shot 7, first segment: modality a only
            [1.0, 1.0, 2.0],  # shot 7, second segment: both
            [np.nan, np.nan, np.nan],  # shot 8: never encoded
            [np.nan, np.nan, 3.0],  # shot 9: modality b only
        ],
        np.float32,
    )
    cov, n = ignite.coverage_from_matrix(mat, [7, 7, 8, 9], [2, 1], ["a", "b"])
    assert cov == {"a": 1, "b": 2} and n == 2


# --- the parts that genuinely need the trained weights ---------------------------------------

_bundle = Path(getenv("SHOT_DESIGN_IGNITE_CKPT", str(ignite.bundle_dir(load_paths()))))
needs_weights = pytest.mark.skipif(
    not ignite.codec_manifest(_bundle).exists(),
    reason=f"no IGNITE bundle at {_bundle} (shot_design model --download)",
)
FM_DIR = Path("/scratch/gpfs/EKOLEMEN/foundation_model")


@needs_weights
@pytest.mark.real_data
def test_codecs_expose_the_encode_then_quantize_contract():
    """`encode(x) -> (B, n_tok, d_model)` PRE-FSQ features and `quantize(feats) -> (quant, codes)`
    -- read off SpectroCodec/SlowTSCodec/FastTSCodec, which all three encodable families share.
    The video codecs are in the bundle but are not loaded: nothing here holds camera frames."""
    codecs = ignite.load_codecs(_bundle)
    assert set(codecs) == {
        "ece",
        "bes",
        "mhr",
        "co2",
        "filterscopes",
        "mse",
        "cer_ti",
        "cer_rot",
        "ts_core_density",
        "ts_core_temp",
        "ts_tangential_density",
        "ts_tangential_temp",
    }
    for name, (codec, cfg, family) in codecs.items():
        assert hasattr(codec, "encode") and hasattr(codec, "quantize"), name
        assert not any(p.requires_grad for p in codec.parameters()), name
        assert not codec.training, name
        assert family in ignite.ENCODABLE_FAMILIES and cfg.d_model > 0, name


@needs_weights
@pytest.mark.real_data
@pytest.mark.skipif(not (FM_DIR / "190090_processed.h5").exists(), reason="no official 190090")
@pytest.mark.skipif(not (_bundle / "frame_codes" / "190090.pt").exists(), reason="no production frame codes for 190090")
def test_frame_codes_reproduce_the_production_cache_bit_for_bit():
    """The bundle ships the production frame codes for ten shots. Our loader + the same codecs
    must give the same integers -- this is what pins the frame origin (0.0 s, not 1.0), the
    standardisation, the STFT and the channel selection all at once. Full run measured: 239 frames
    x 12 modalities all identical on 190090; the test keeps a 20-frame, 3-family slice."""
    import torch

    ref = torch.load(_bundle / "frame_codes" / "190090.pt", map_location="cpu", weights_only=False)
    # The current corpus has no co2 samples on this shot; mhr exercises spectro
    # encoding with real data, alongside slowts and fastts.
    codecs = ignite.load_codecs(_bundle, names=["ts_core_density", "filterscopes", "mhr"])
    got = ignite.frame_codes(
        190090, codecs, load_paths(), data_dir=FM_DIR, max_frames=20, workers=2
    )
    assert set(got) == set(codecs)
    for name, codes in got.items():
        theirs = ref["codes"][name][:20].numpy().astype(np.int64)
        assert codes.shape == theirs.shape, name
        assert (codes == theirs).all(), f"{name}: {(codes != theirs).mean():.3f} of tokens differ"


@needs_weights
@pytest.mark.real_data
@pytest.mark.skipif(not (FM_DIR / "185601_processed.h5").exists(), reason="no official 185601")
def test_an_absent_modality_is_nan_not_the_constant_frames_the_dataset_would_yield():
    """185601's `bes` group is the (64, 1) placeholder. CodecPairDataset still yields 239 frames
    of a constant spectrogram for it (measured: std 0.0), which the codec would encode into a
    finite, meaningless vector. The embedding has to decide absence from the file instead."""
    codecs = ignite.load_codecs(_bundle, names=["bes", "ts_core_density"])
    emb = ignite.encode_shot(185601, codecs, load_paths(), data_dir=FM_DIR, max_frames=6, workers=2)
    assert emb.modalities == ("bes", "ts_core_density") and emb.features.shape == (6, sum(emb.dims))
    assert emb.present == ("ts_core_density",)
    assert np.isnan(emb.features[:, : emb.dims[0]]).all()
    assert np.isfinite(emb.features[:, emb.dims[0] :]).all()


# ---------------------------------------------------------------------------------------------
