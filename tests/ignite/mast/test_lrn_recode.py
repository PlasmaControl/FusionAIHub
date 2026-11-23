"""The linear recoder, the recoded codec, and the codecs of an assignment that names a recoder (plan 1c)."""

import numpy as np
import pytest

from tokamak_foundation_model.ignite.mast import tasks, transform
from tokamak_foundation_model.ignite.mast.tokenizers import assign, codecs, fit, grid
from tokamak_foundation_model.ignite.mast.tokenizers.learned import build, identity, recode, store

RNG = np.random.default_rng(0)
SPEC = tasks.SignalSpec("m", 0.00025, (3,), 0.0, 1.0)
META = {"signals": {"m": {"kind": "frame_dct", "domain": "linear", "n_streams": 3, "values_per_token": 1}}}
TABLE = identity.StreamTable({"m": (0, 3)}, np.zeros((3, identity.N_FEAT), dtype=np.float32))


def stats(shape=(3,)):
    return transform.ChannelStats(compress="none", scale=np.ones(shape), mean=np.zeros(shape), std=np.ones(shape))


def low_rank(m=4000, d=16, scales=(3.0, 2.0, 1.0), noise=0.01):
    q, _ = np.linalg.qr(RNG.standard_normal((d, d)))
    return (RNG.standard_normal((m, len(scales))) * np.array(scales)) @ q[:, :len(scales)].T \
        + noise * RNG.standard_normal((m, d))


def window(n=6):
    """`n` windows of 3 channels and 100 samples (5 frames); neighbouring channels are correlated."""
    w = RNG.standard_normal((n, 3, 100)).cumsum(axis=-1) * 0.1
    return w + 0.5 * np.roll(w, 1, axis=1)


def pieces(concat=False, n_values=16, n_val=None):
    front = codecs.FrameDCT("m", 0.00025, (3,), stats(), 16)
    fv = front.encode(window(40)).reshape(-1, 3, 16)                 # 200 frames
    norm = store.fit_norm(fv)
    e = norm.apply(fv)
    r = recode.fit_linear(e.reshape(len(e), -1) if concat else e.reshape(-1, 16), n_values, concat=concat, n_val=n_val)
    return front, norm, r, np.arange(3)


def test_a_linear_recoder_finds_the_directions_of_the_data():
    e = low_rank()
    r = recode.fit_linear(e, 4)
    assert (r.kind, r.concat, r.d_in, r.n_tok, r.n_val, r.n_values) == ("linear", False, 16, 1, 4, 4)
    ident = np.zeros(len(e), dtype=np.int64)
    v = r.encode(e, ident)
    assert v.shape == (4000, 4) and v.dtype == np.float32
    assert np.sqrt(np.mean((r.decode(v, ident) - e) ** 2)) < 0.02          # three directions hold all but the noise
    assert np.allclose(v.var(axis=0)[:3], [9.0, 4.0, 1.0], rtol=0.15)       # ordered by variance
    with pytest.raises(ValueError):
        recode.fit_linear(e, 6, n_val=4)


def test_a_prefix_of_the_values_decodes_with_an_error_that_falls_with_its_length():
    e = low_rank()
    r = recode.fit_linear(e, 16)
    ident = np.zeros(len(e), dtype=np.int64)
    v = r.encode(e, ident)
    errors = []
    for keep in (1, 2, 3, 16):
        cut = v.copy()
        cut[:, keep:] = 0.0
        errors.append(float(np.sqrt(np.mean((r.decode(cut, ident) - e) ** 2))))
    assert errors[0] > errors[1] > errors[2] > errors[3] and errors[3] < 1e-5
    assert errors[0] == pytest.approx(np.sqrt(5.0 / 16), rel=0.1)           # directions 2 and 3 are left: 4 + 1 over 16


def test_the_fitted_basis_is_never_worse_than_keeping_the_first_coordinates():
    e = low_rank(scales=(3.0, 2.0, 1.0, 0.5), noise=0.05)
    r = recode.fit_linear(e, 16)
    ident = np.zeros(len(e), dtype=np.int64)
    v = r.encode(e, ident)
    for keep in (1, 2, 4, 8):
        cut = v.copy()
        cut[:, keep:] = 0.0
        fitted = np.mean((r.decode(cut, ident) - e) ** 2)
        first = e[:, keep:].var(axis=0).sum() / 16                          # coordinates from `keep` on replaced by their mean
        assert fitted <= first + 1e-6


def test_a_full_recoder_changes_nothing_and_fewer_values_give_a_coarser_window():
    front, norm, r, ident = pieces()
    x = window()
    full = recode.RecodedCodec(front, norm, r, ident, (16,))
    assert (full.n_tok, full.n_val, full.domain, full.name, full.values_per_frame) == (3, 16, "linear", "m", 48)
    assert full.encode(x).shape == (6, 5, 3, 16) and np.allclose(full.roundtrip(x), front.roundtrip(x), atol=1e-4)
    four, one = (recode.RecodedCodec(front, norm, r, ident, (k,)) for k in (4, 1))
    assert four.encode(x).shape == (6, 5, 3, 4) and four.values_per_frame == 12
    err = lambda c: float(np.sqrt(np.mean((c.roundtrip(x) - x) ** 2)))
    assert err(full) < err(four) < err(one)
    with pytest.raises(ValueError):
        recode.RecodedCodec(front, norm, r, ident, (17,))


def test_a_missing_channel_has_no_token_and_changes_no_other_token():
    front, norm, r, ident = pieces()
    c = recode.RecodedCodec(front, norm, r, ident, (4,))
    x = window()
    clean = c.encode(x)
    x[2, 1, 20:40] = np.nan                                                  # window 2, channel 1, frame 1
    holed = c.encode(x)
    assert np.isnan(holed[2, 1, 1]).all()
    others = np.ones(clean.shape, dtype=bool)
    others[2, 1, 1] = False
    assert np.allclose(holed[others], clean[others], rtol=0, atol=1e-6)
    back = c.roundtrip(x)
    assert np.isnan(back[2, 1, 20:40]).all() and np.isfinite(back[2, 0]).all() and np.isfinite(back[2, 1, :20]).all()


def test_the_concatenated_codec_fills_a_missing_channel():
    front, norm, r, ident = pieces(concat=True, n_values=24, n_val=4)
    assert (r.concat, r.d_in, r.n_tok, r.n_val) == (True, 48, 6, 4)
    c = recode.RecodedCodec(front, norm, r, ident, (12,))
    assert (c.n_tok, c.n_val, c.values_per_frame) == (3, 4, 12)
    x = window()
    clean = c.roundtrip(x)
    x[2, 1, 20:40] = np.nan
    values = c.encode(x)
    assert values.shape == (6, 5, 3, 4) and np.isfinite(values).all()
    back = c.roundtrip(x)
    assert np.isfinite(back).all()                                           # the missing channel is decoded from the others
    assert not np.allclose(back[2, 0, 20:40], clean[2, 0, 20:40], rtol=0, atol=1e-6)   # and the others pay for it
    x[2, :, 20:40] = np.nan
    assert np.isnan(c.encode(x)[2, 1]).all() and np.isnan(c.roundtrip(x)[2, :, 20:40]).all()
    with pytest.raises(ValueError, match="multiple"):
        recode.RecodedCodec(front, norm, r, ident, (6,))


def test_a_band_front_keeps_one_number_of_values_per_band():
    shape = (2, 256)
    st = transform.ChannelStats(compress="asinh", scale=np.full(shape, 2.0), mean=np.zeros(shape), std=np.ones(shape))
    front = codecs.BandDCT("s", 0.001024, shape, st, (32,) * 8)
    x = RNG.gamma(2.0, 1.0, (4, 2, 256, 98))
    fv = front.encode(x).reshape(-1, 16, 32)
    norm = store.fit_norm(fv)
    r = recode.fit_linear(norm.apply(fv).reshape(-1, 32), 32)
    c = recode.RecodedCodec(front, norm, r, np.arange(16), (8, 2, 2, 2, 2, 2, 2, 2))
    values = c.encode(x)
    assert (c.n_tok, c.n_val, c.values_per_frame) == (16, 8, 2 * (8 + 7 * 2)) and values.shape == (4, 20, 16, 8)
    assert values[:, :, 0, 2:].any() and not values[:, :, 1, 2:].any() and not values[:, :, 9, 2:].any()
    assert c.roundtrip(x).shape == x.shape
    with pytest.raises(ValueError):
        recode.RecodedCodec(front, norm, r, np.arange(16), (8, 2, 2))


def test_a_recoder_is_saved_once_with_what_it_was_fitted_on(tmp_path):
    front, norm, r, ident = pieces()
    recode.save_recoder(tmp_path / "klt.x", r, {"m": norm}, TABLE, {**META, "date": "2026-11-22"})
    got = recode.load_recoder(tmp_path / "klt.x")
    e = RNG.standard_normal((5, 16)).astype(np.float32)
    zero = np.zeros(5, dtype=np.int64)
    assert np.allclose(got.recoder.encode(e, zero), r.encode(e, zero), atol=1e-6)
    assert np.array_equal(got.norms["m"].scale, norm.scale) and got.meta["signals"]["m"]["domain"] == "linear"
    assert got.table.index_of("m", 3).tolist() == [0, 1, 2] and got.ranges == {}
    recode.save_recoder(tmp_path / "klt.y", recode.LinearRecoder(r.mean + 1.0, r.basis), {"m": norm}, TABLE, META)
    first = recode.recoder_sha256(tmp_path / "klt.x")
    assert len(first) == 64 and first != recode.recoder_sha256(tmp_path / "klt.y")
    with pytest.raises(FileExistsError):
        recode.save_recoder(tmp_path / "klt.x", r, {"m": norm}, TABLE, META)


def test_build_codec_reads_the_recoder_an_assignment_names(tmp_path):
    st = stats()
    sfit = fit.fit_signal(SPEC, st, [RNG.standard_normal((3, 200)).astype(np.float32) for _ in range(4)], kind="frame_dct")
    front, norm, r, ident = pieces()
    e = norm.apply(front.encode(window(40)).reshape(-1, 3, 16))
    values = r.encode(e.reshape(-1, 16), np.zeros(len(e) * 3, dtype=np.int64)).reshape(-1, 3, 16)
    recode.save_recoder(tmp_path / "klt.x", r, {"m": norm}, TABLE, META, {"m": grid.fit_ranges(values)})
    plain = assign.Setting("frame_dct", (4,))
    assert isinstance(build.build_codec(SPEC, st, sfit, plain, tmp_path), codecs.FrameDCT)
    named = assign.Setting("recoded", (4,), "linear", source="klt.x")
    c = build.build_codec(SPEC, st, sfit, named, tmp_path)
    assert isinstance(c, recode.RecodedCodec) and (c.n_tok, c.n_val) == (3, 4)
    assert c.grids.lo.shape == (3, 4) and c.grids.n_bins == 64
    assert np.isfinite(c.roundtrip(window(), "nearest")).all()
    with pytest.raises(ValueError, match="domain"):
        build.build_codec(SPEC, st, sfit, assign.Setting("recoded", (4,), "compressed", source="klt.x"), tmp_path)
    with pytest.raises(ValueError, match="concat"):
        build.build_codec(SPEC, st, sfit, assign.Setting("recoded_concat", (4,), "linear", source="klt.x"), tmp_path)
    other = tasks.SignalSpec("other", 0.00025, (3,), 0.0, 1.0)
    with pytest.raises(ValueError, match="other"):
        build.build_codec(other, st, sfit, named, tmp_path)
    assert build.sources({"m": named, "n": plain}) == ["klt.x"] and build.sources({"n": plain}) == []
    assert build.sources_sha256({"m": named}, tmp_path) != build.sources_sha256({"n": plain}, tmp_path)



def test_a_band_codec_with_uneven_keeps_reads_its_dropped_positions_as_zero_when_quantized():
    shape = (2, 256)
    st = transform.ChannelStats(compress="asinh", scale=np.full(shape, 2.0), mean=np.zeros(shape), std=np.ones(shape))
    front = codecs.BandDCT("s", 0.001024, shape, st, (32,) * 8)
    x = RNG.gamma(2.0, 1.0, (4, 2, 256, 98))
    fv = front.encode(x).reshape(-1, 16, 32)
    norm = store.fit_norm(fv)
    r = recode.fit_linear(norm.apply(fv).reshape(-1, 32), 32)
    keep = (8, 2, 2, 2, 2, 2, 2, 2)
    free = r.encode(norm.apply(fv).reshape(-1, 32), np.zeros(len(fv) * 16, dtype=np.int64)).reshape(-1, 16, 32)[:, :, :8]
    grids = grid.GridSet(*grid.fit_ranges(free))                              # ranges of the unmasked values: 0 is not a centre
    c = recode.RecodedCodec(front, norm, r, np.arange(16), keep, grids)
    assert c.k == keep
    encoded = c.encode(x)
    dropped = np.arange(8)[None, :] >= np.tile(keep, 2)[:, None]               # (n_tok, n_val)
    assert not dropped[0].any() and dropped[1, 2:].all()
    assert codecs.Codec.read(c, encoded, "nearest")[..., dropped].any()          # the grid read alone puts energy back
    for mode in ("ranged", "nearest"):
        read = c.read(encoded, mode)
        assert (read[..., dropped] == 0.0).all() and np.isfinite(read).all() and read[..., ~dropped].any()
    assert c.roundtrip(x, "nearest").shape == x.shape
