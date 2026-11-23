"""Frames of 5 ms and the linear units the transforms work in."""

import shutil
from pathlib import Path

import numpy as np

from tokamak_foundation_model.ignite.mast import paths, tokenizers, transform
from tokamak_foundation_model.ignite.mast.headroom import cheap
from tokamak_foundation_model.ignite.mast.tokenizers import frames

NONE = transform.ChannelStats(compress="none", scale=np.ones(2), mean=np.array([1.0, -2.0]), std=np.array([2.0, 4.0]))
ASINH = transform.ChannelStats(compress="asinh", scale=np.array([0.5, 2.0]), mean=np.array([0.1, 0.2]),
                               std=np.array([1.5, 0.7]))
X = np.array([[1.0, 3.0, np.nan], [-2.0, 2.0, 6.0]])           # 2 channels, 3 samples, physical


def test_frames_are_stage0s_frames():
    assert frames.FRAME_S == cheap.FRAME_S == 0.005
    for dt, n in ((0.005, 1), (0.00025, 20), (0.0002, 25), (0.00002, 250)):
        assert frames.frame_len(dt) == cheap.frame_len(dt) == n


def test_frame_index_of_whole_frames():
    assert frames.frame_index(100, 0.00025).tolist() == np.repeat(np.arange(5), 20).tolist()
    assert frames.frame_index(3, 0.005).tolist() == [0, 1, 2]
    assert frames.frame_index(5000, 0.00002).max() == 19 and frames.frame_index(5000, 0.00002).dtype == np.int64


def test_frame_index_of_spectrogram_columns():
    idx = frames.frame_index(98, 0.001024)                      # 100 ms of TokaMark's Mirnov columns
    assert np.bincount(idx).tolist() == [5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5]
    assert (np.diff(idx) >= 0).all()


def test_linear_units_undo_the_compression_and_keep_nan():
    u = frames.to_linear(X, NONE)
    assert np.allclose(u[0, :2], [0.0, 1.0]) and np.isnan(u[0, 2]) and np.allclose(u[1], [0.0, 1.0, 2.0])
    u = frames.to_linear(X, ASINH)
    assert np.allclose(u[0, :2], [2.0, 6.0]) and np.isnan(u[0, 2]) and np.allclose(u[1], [-1.0, 1.0, 3.0])


def test_the_conversions_are_inverses_and_agree_with_the_cache():
    for st in (NONE, ASINH):
        u = frames.to_linear(X, st)
        assert np.allclose(frames.from_linear(u, st), X, equal_nan=True)
        z = transform.to_model(X, st)
        assert np.allclose(frames.linear_from_model(z, st), u, equal_nan=True)
        assert np.allclose(frames.model_from_linear(u, st), z, equal_nan=True)
        assert frames.to_linear(np.stack([X, X, X]), st).shape == (3, 2, 3)      # a leading batch axis


def test_the_package_hash_changes_with_its_files(tmp_path):
    first = tokenizers.code_sha256()
    assert len(first) == 64 and first == tokenizers.code_sha256()
    package = Path(tokenizers.__file__).parent
    for f in package.glob("*.py"):
        shutil.copy(f, tmp_path / f.name)
    assert tokenizers.sha256_of_py_files(tmp_path) == first
    victim = tmp_path / "frames.py"
    victim.write_bytes(victim.read_bytes() + b"#")                      # one changed byte
    assert tokenizers.sha256_of_py_files(tmp_path) != first
    victim.write_bytes(victim.read_bytes()[:-1])
    assert tokenizers.sha256_of_py_files(tmp_path) == first
    (tmp_path / "learned").mkdir()
    (tmp_path / "learned" / "x.py").write_text("x = 1\n")               # a sub-package is not hashed
    assert tokenizers.sha256_of_py_files(tmp_path) == first
