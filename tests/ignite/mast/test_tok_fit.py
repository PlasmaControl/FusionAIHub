"""Ranges and bases fitted from cached values (synthetic here; the cache itself in the trial)."""

import numpy as np
import pytest

from tokamak_foundation_model.ignite.mast import tasks, transform
from tokamak_foundation_model.ignite.mast.tokenizers import assign, codecs, fit

RNG = np.random.default_rng(0)
IP = tasks.SignalSpec("summary-ip", 0.00025, (1,), 3.5e5, 3.0e5)
LCFS = tasks.SignalSpec("equilibrium-lcfs_r", 0.005, (30,), 0.9, 0.4)


def st(compress, shape):
    return transform.ChannelStats(compress=compress, scale=np.ones(shape), mean=np.zeros(shape), std=np.ones(shape))


def series(n=3, t=400):
    return [RNG.standard_normal((1, t)).astype(np.float32) for _ in range(n)]


def test_the_fit_skips_and_counts_a_shot_without_the_signal():
    f = fit.fit_signal(IP, st("none", 1), [*series(2), None], kind="frame_dct")
    assert (f.n_shots, f.n_skipped, f.n_examples, f.partly_valid) == (2, 1, 40, False)
    lo, hi, width = f.ranges["linear"]
    assert lo.shape == hi.shape == width.shape == (1, 16) and (lo < hi).all() and (width > 0).all()
    assert set(f.ranges) == {"linear"} and f.pca == {}


def test_a_signal_no_shot_has_is_reported_and_not_fitted():
    f = fit.fit_signal(IP, st("none", 1), [None, None], kind="frame_dct")
    assert (f.n_shots, f.n_skipped, f.n_examples) == (0, 2, 0) and f.ranges == {} and f.pca == {}


def test_an_asinh_signal_is_fitted_in_both_domains():
    f = fit.fit_signal(IP, st("asinh", 1), series(), kind="frame_dct")
    assert set(f.ranges) == {"linear", "compressed"}


def test_a_profile_fit_has_a_basis_and_score_ranges_per_domain():
    shots = [RNG.standard_normal((30, 50)).astype(np.float32) for _ in range(4)]
    shots[0][::2, :10] = np.nan                                              # partly valid profiles
    f = fit.fit_signal(LCFS, st("none", 30), shots, kind="linear")
    assert set(f.pca) == set(f.ranges) == {"signal", "linear"} and f.partly_valid and f.n_examples == 200
    mean, basis, eig, total = f.pca["signal"]
    assert basis.shape == (30, 30) and f.ranges["signal"][0].shape == (1, 30)


def test_build_codec_makes_the_codec_of_the_setting():
    f = fit.fit_signal(IP, st("none", 1), series(), kind="frame_dct")
    c = fit.build_codec(IP, st("none", 1), f, assign.Setting("frame_dct", (4,)))
    assert isinstance(c, codecs.FrameDCT) and (c.n_tok, c.n_val) == (1, 4)
    assert c.grids.lo.shape == (1, 4) and c.grids.n_bins == 64 and np.array_equal(c.grids.lo, f.ranges["linear"][0][:, :4])
    with pytest.raises(ValueError, match="compressed"):
        fit.build_codec(IP, st("none", 1), f, assign.Setting("frame_dct", (4,), domain="compressed"))
    with pytest.raises(ValueError, match="frame_klt"):
        fit.build_codec(IP, st("none", 1), f, assign.Setting("frame_klt", (4,)))


def test_build_codec_groups_components_into_tokens_and_sets_the_ridge():
    shots = [RNG.standard_normal((30, 50)).astype(np.float32) for _ in range(4)]
    whole = fit.fit_signal(LCFS, st("none", 30), shots, kind="linear")
    c = fit.build_codec(LCFS, st("none", 30), whole, assign.Setting("linear", (8, 4), "signal"))
    assert isinstance(c, codecs.LinearCodec) and (c.n_tok, c.n_val, c.ridge) == (2, 4, 0.0)
    assert c.grids.lo.shape == (2, 4) and (c.signal_mean, c.signal_std) == (0.9, 0.4)
    assert np.array_equal(c.grids.lo.ravel(), whole.ranges["signal"][0][0, :8])
    shots[0][::2, :10] = np.nan
    holes = fit.fit_signal(LCFS, st("none", 30), shots, kind="linear")
    assert fit.build_codec(LCFS, st("none", 30), holes, assign.Setting("linear", (8, 4), "signal")).ridge > 0.0


def test_a_fit_is_stored_and_read_back(tmp_path):
    fits = {"summary-ip": fit.fit_signal(IP, st("asinh", 1), series(), kind="frame_dct"),
            "equilibrium-lcfs_r": fit.fit_signal(LCFS, st("none", 30), [RNG.standard_normal((30, 50)).astype(np.float32)],
                                                 kind="linear")}
    fit.save_fit(fits, tmp_path, {"n_shots": 3})
    back, meta = fit.load_fit(tmp_path)
    assert meta["n_shots"] == 3 and set(back) == set(fits) and len(fit.fit_sha256(tmp_path)) == 64
    assert np.array_equal(back["summary-ip"].ranges["compressed"][1], fits["summary-ip"].ranges["compressed"][1])
    assert np.array_equal(back["equilibrium-lcfs_r"].pca["signal"][1], fits["equilibrium-lcfs_r"].pca["signal"][1])
    assert back["equilibrium-lcfs_r"].n_examples == 50
