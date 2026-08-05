"""CPU TDD spec for the IGNITE Phase-A slow-TS codec family (Thomson / CER / MSE).

Mirrors the spectro / video codec tests but for the slow-TS pieces
(docs/IGNITE_DESIGN.md §4.3 — the "lightest touch" smooth-profile codec):

    slow_ts_nets.SlowTSEncoder(cfg)(x: (B,C,T)) -> feats (B, n_tok, d_model)
    slow_ts_nets.SlowTSDecoder(cfg)(quant: (B,n_tok,d_model)) -> recon (B,C,T)
    slow_ts_codec.SlowTSCodec(cfg).forward(x) -> dict(recon, feats, quant, codes)
    slow_ts_codec.SlowTSCodec.generator_losses(x, cfg, step, mask) -> dict(total,...)
    gate.slowts_decode_fidelity(recon, target, mask) -> dict(envelope_corr, peak_f1, sharpness)

    train_codec.SlowTSCodecPairDataset  (subclass of TokamakMultiFileDataset)
    train_codec.slowts_compute_gate / train_codec.train_slowts_codec

Small synthetic tensors + tiny SYNTHETIC HDF5 slow-TS shots only. CPU. No SLURM / GPU / real
data. There is NO δ-shift pair and NO discriminator anywhere (the slow-TS codec is a masked
reconstruction + entropy codec).

Run:
    .pixi/envs/default/bin/python -m pytest tests/ignite/test_slowts_codec.py -q
"""
from __future__ import annotations

import inspect
import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from tokamak_foundation_model.data.multi_file_dataset import TokamakMultiFileDataset
from tokamak_foundation_model.ignite import gate
from tokamak_foundation_model.ignite import train_codec as tc
from tokamak_foundation_model.ignite.config import (
    SLOWTS_SIGNALS,
    SlowTSCodecConfig,
    slowts_patch_for,
)
from tokamak_foundation_model.ignite.slow_ts_codec import SlowTSCodec
from tokamak_foundation_model.ignite.slow_ts_nets import SlowTSDecoder, SlowTSEncoder


# --------------------------------------------------------------------------------------- #
# tiny configs
# --------------------------------------------------------------------------------------- #
def _small_cfg(signal: str = "ts_core_density", channels: int = 12) -> SlowTSCodecConfig:
    """Small transformer + a position×time grid that patches into >1 token (for gate seqs)."""
    return SlowTSCodecConfig(
        signal=signal,
        channels=channels,
        time_steps=5,
        n_zones=2,      # -> 2 radial-zone position patches
        patch_c=6,      # ceil(12 / 2) = 6 positions per zone
        patch_t=5,      # -> 1 time patch  => n_tok = 2
        d_model=32,
        enc_depth=1,
        dec_depth=1,
        heads=2,
        fsq_levels=[4, 4, 3],
    )


# --------------------------------------------------------------------------------------- #
# config geometry
# --------------------------------------------------------------------------------------- #
def test_config_geometry_and_tokens():
    cfg = _small_cfg()
    assert cfg.n_pos_patch == 2 and cfg.n_time_patch == 1
    assert cfg.n_tok == 2
    assert cfg.fsq_dim == 3 and cfg.codebook_size == 4 * 4 * 3
    assert cfg.window_samples == 5  # 50 ms @ 100 Hz

    # production default (DESIGNED per-frame budget: 4 radial-zone tokens per window @
    # [8,5,5,5]=1000). C=44 is divisible by 4 so patch_c=11 and no padding.
    prod = tc.slowts_codec_cfg("ts_core_density", 44)
    assert prod.n_tok == 4
    assert prod.n_zones == 4 and prod.n_pos_patch == 4 and prod.n_time_patch == 1
    assert prod.patch_c == 11 and prod.padded_channels == 44  # 44 = 4*11, no padding
    assert prod.codebook_size == 1000 and prod.fsq_dim == 4
    assert prod.channels == 44 and prod.time_steps == 5


@pytest.mark.parametrize(
    "channels,patch_c,padded",
    [(44, 11, 44), (10, 3, 12), (48, 12, 48), (69, 18, 72)],
)
def test_slowts_always_4_tokens_incl_nondivisible(channels, patch_c, padded):
    """Every slow-TS signal yields EXACTLY 4 tokens; non-÷4 counts pad + mask the tail."""
    cfg = tc.slowts_codec_cfg("ts_core_density", channels)
    assert cfg.n_tok == 4 and cfg.n_zones == 4
    assert cfg.patch_c == patch_c, "patch_c must be ceil(channels/4)"
    assert cfg.padded_channels == padded
    assert cfg.padded_channels >= cfg.channels
    assert cfg.padded_channels == 4 * cfg.patch_c


def test_config_rejects_impossible_zone_split():
    # patch_c too SMALL to hold the profile in n_zones zones (channels > padded_channels).
    with pytest.raises(AssertionError):
        SlowTSCodecConfig(channels=10, n_zones=4, patch_c=2)  # padded 8 < 10
    # patch_c so LARGE a whole zone is empty (channels <= (n_zones-1)*patch_c).
    with pytest.raises(AssertionError):
        SlowTSCodecConfig(channels=10, n_zones=4, patch_c=4)  # (4-1)*4 = 12 >= 10 -> empty zone
    with pytest.raises(AssertionError):
        SlowTSCodecConfig(time_steps=5, patch_t=2)  # 5 % 2 != 0


def test_zero_is_missing_matches_loader_policy():
    """The cfg missingness flag mirrors data_loader.SignalConfig.zero_is_missing per signal."""
    from tokamak_foundation_model.data.data_loader import TokamakH5Dataset

    for sig in SLOWTS_SIGNALS:
        loader_cfg = next(c for c in TokamakH5Dataset.SIGNAL_CONFIGS if c.name == sig)
        codec_cfg = tc.slowts_codec_cfg(sig, loader_cfg.num_channels)
        assert codec_cfg.zero_is_missing == loader_cfg.zero_is_missing, sig
    # concretely: Thomson zero_is_missing, CER/MSE not (built via the 4-zone helper).
    assert tc.slowts_codec_cfg("ts_core_temp", 44).zero_is_missing is True
    assert tc.slowts_codec_cfg("cer_ti", 48).zero_is_missing is False
    assert tc.slowts_codec_cfg("mse", 69).zero_is_missing is False


def test_fit_window_masks_nonfinite_channels_instead_of_rejecting():
    """inf/nan raw channels (e.g. mse's ~2 garbage channels) are masked invalid + zeroed, NOT
    rejected wholesale. Regression for the mse +0.55/-0.52 oscillation (windows touching the bad
    channels were dropped; leaked ones poisoned the recon)."""
    cfg = tc.slowts_codec_cfg("mse", 69)
    ds = tc.SlowTSCodecPairDataset.__new__(tc.SlowTSCodecPairDataset)
    ds.codec_cfg = cfg
    T = cfg.time_steps
    torch.manual_seed(0)
    raw = torch.randn(69, T)
    raw[3] = float("inf")           # bad channel (inf)
    raw[7] = float("nan")           # bad channel (nan)
    nan_mask = torch.zeros(69, T)   # loader did NOT flag them (the failure mode)

    signal, valid = ds._fit_window(raw, nan_mask)

    assert torch.isfinite(signal).all()                       # no inf/nan leaks (was the reject trigger)
    assert float(valid[3].sum()) == 0.0                        # bad channels excluded from loss
    assert float(valid[7].sum()) == 0.0
    assert float(signal[3].abs().sum()) == 0.0                 # bad channels zeroed in the input
    assert float(signal[7].abs().sum()) == 0.0
    assert float(valid[0].sum()) > 0.0                         # good channels retained


# --------------------------------------------------------------------------------------- #
# nets round-trip
# --------------------------------------------------------------------------------------- #
def test_encoder_decoder_shapes():
    cfg = _small_cfg()
    enc, dec = SlowTSEncoder(cfg), SlowTSDecoder(cfg)
    B = 3
    x = torch.randn(B, cfg.channels, cfg.time_steps)
    feats = enc(x)
    assert feats.shape == (B, cfg.n_tok, cfg.d_model)
    recon = dec(feats)
    assert recon.shape == x.shape
    assert torch.isfinite(recon).all()


def test_nets_gradient_flows_end_to_end():
    cfg = _small_cfg()
    enc, dec = SlowTSEncoder(cfg), SlowTSDecoder(cfg)
    x = torch.randn(2, cfg.channels, cfg.time_steps, requires_grad=True)
    dec(enc(x)).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_decoder_last_layer_exposed():
    cfg = _small_cfg()
    dec = SlowTSDecoder(cfg)
    assert dec.last_layer is dec.to_signal.weight
    assert isinstance(dec.last_layer, torch.nn.Parameter)


# --------------------------------------------------------------------------------------- #
# SlowTSCodec forward + codes at the DEFAULT FSQ size, all 7 signals
# --------------------------------------------------------------------------------------- #
def test_codec_forward_shapes_and_codes_in_range():
    cfg = _small_cfg()
    codec = SlowTSCodec(cfg)
    B = 2
    x = torch.randn(B, cfg.channels, cfg.time_steps) * 3.0
    out = codec(x)
    assert set(out) == {"recon", "feats", "quant", "codes"}
    assert out["recon"].shape == x.shape
    assert out["feats"].shape == (B, cfg.n_tok, cfg.d_model)
    assert out["quant"].shape == (B, cfg.n_tok, cfg.d_model)
    assert out["codes"].shape == (B, cfg.n_tok, cfg.fsq_dim)
    assert out["codes"].dtype == torch.long
    levels = torch.tensor(cfg.fsq_levels)
    assert (out["codes"] >= 0).all() and (out["codes"] < levels).all()
    assert torch.isfinite(out["recon"]).all()
    assert codec.codebook_size == cfg.codebook_size


@pytest.mark.parametrize(
    "signal,channels",
    [("ts_tangential_density", 10), ("ts_core_density", 44), ("cer_ti", 48), ("mse", 69)],
)
def test_codec_roundtrip_at_default_fsq_size(signal, channels):
    """encode -> quantize -> decode at the right-sized default FSQ ([8,5,5,5]=1000, 4 dims).

    EXACTLY 4 tokens per window for every signal (incl. the non-÷4 counts 10 and 69, which pad
    up to padded_channels 12 and 72). The codec patchifies over ``padded_channels``.
    """
    cfg = tc.slowts_codec_cfg(signal, channels)
    # keep the tiny transformer for a fast test.
    cfg.d_model, cfg.enc_depth, cfg.dec_depth, cfg.heads = 32, 1, 1, 2
    assert cfg.fsq_dim == 4 and cfg.codebook_size == 1000
    assert cfg.n_tok == 4
    codec = SlowTSCodec(cfg)
    # the encoder sees padded_channels positions (the dataset pads + masks the zone tail).
    x = torch.randn(2, cfg.padded_channels, cfg.time_steps) * 5.0
    out = codec(x)
    assert out["recon"].shape == x.shape == (2, cfg.padded_channels, cfg.time_steps)
    assert out["codes"].shape == (2, 4, 4)
    levels = torch.tensor(cfg.fsq_levels)
    assert (out["codes"] >= 0).all() and (out["codes"] < levels).all()
    assert torch.isfinite(out["recon"]).all()


# --------------------------------------------------------------------------------------- #
# generator_losses — finite, no consistency/adversarial term, single x + mask signature
# --------------------------------------------------------------------------------------- #
def test_generator_losses_keys_finite_and_lightest_touch():
    cfg = _small_cfg()
    codec = SlowTSCodec(cfg)
    x = torch.randn(2, cfg.channels, cfg.time_steps)
    out = codec.generator_losses(x, cfg, step=0)
    for k in ("total", "pixel", "entropy"):
        assert k in out and out[k].dim() == 0 and torch.isfinite(out[k]).all()
    # "lightest touch": NO adversarial, NO feature-matching, NO consistency terms.
    assert "consistency" not in out
    assert "adversarial" not in out
    assert "feature_matching" not in out
    assert out["adaptive_weight"] == 1.0  # no adaptive-adv weight
    assert out["recon"].shape == x.shape
    assert out["codes"].shape == (2, cfg.n_tok, cfg.fsq_dim)


def test_generator_losses_signature_no_x_shift_no_disc():
    """The slow-TS loss takes a SINGLE x + mask (no δ-shift pair, no discriminator arg)."""
    params = list(inspect.signature(SlowTSCodec.generator_losses).parameters)
    assert "x_shift" not in params, "slow-TS codec must not take a shift pair"
    assert "disc" not in params, "slow-TS codec has no discriminator"
    assert params[1] == "x" and "mask" in params


def test_masked_recon_ignores_missing_samples():
    """The masked reconstruction MAE must IGNORE masked-out (missing) samples.

    Construct a recon that is EXACT on the valid samples but arbitrarily wrong on the missing
    ones. With the mask the MAE must be ~0; without it the MAE must be large. This is the
    critical missingness contract (don't train the codec toward the zero/NaN-fill).
    """
    cfg = _small_cfg()
    codec = SlowTSCodec(cfg)
    B = 2
    x = torch.randn(B, cfg.channels, cfg.time_steps)
    # validity mask: mark half the positions missing.
    mask = torch.ones(B, cfg.channels, cfg.time_steps)
    mask[:, : cfg.channels // 2, :] = 0.0

    # a "recon" equal to x on valid, hugely wrong on missing.
    recon = x.clone()
    recon[mask < 0.5] += 1000.0

    masked = SlowTSCodec._masked_recon_mae(recon, x, mask)
    unmasked = SlowTSCodec._masked_recon_mae(recon, x, None)
    assert float(masked) < 1e-5, float(masked)         # missing samples ignored -> ~0
    assert float(unmasked) > 1.0                        # unmasked sees the huge missing error
    # all-invalid mask falls back to the unmasked MAE (stays finite).
    allzero = torch.zeros_like(mask)
    fb = SlowTSCodec._masked_recon_mae(recon, x, allzero)
    assert torch.isfinite(fb) and torch.allclose(fb, unmasked)


def test_masked_recon_accepts_bt_and_bc_masks():
    cfg = _small_cfg()
    x = torch.randn(2, cfg.channels, cfg.time_steps)
    recon = torch.randn_like(x)
    bt = torch.ones(2, cfg.time_steps)            # (B, T)
    bc = torch.ones(2, cfg.channels)              # (B, C)
    for m in (bt, bc):
        v = SlowTSCodec._masked_recon_mae(recon, x, m)
        assert torch.isfinite(v)
    with pytest.raises(ValueError):
        SlowTSCodec._masked_recon_mae(recon, x, torch.ones(3, 3, 3, 3))  # bad ndim/shape


def test_generator_step_reduces_masked_recon():
    """Repeated Adam steps on ONE window drive the (masked) reconstruction MAE down."""
    torch.manual_seed(0)
    cfg = _small_cfg()
    codec = SlowTSCodec(cfg)
    opt = torch.optim.Adam(codec.parameters(), lr=1e-2)
    x = torch.randn(2, cfg.channels, cfg.time_steps)
    mask = torch.ones_like(x)
    mask[:, 0, :] = 0.0  # one position genuinely missing

    pix = []
    for step in range(30):
        g = codec.generator_losses(x, cfg, step=step, mask=mask)
        opt.zero_grad(set_to_none=True)
        g["total"].backward()
        opt.step()
        pix.append(float(g["pixel"].detach()))
        assert math.isfinite(pix[-1])
    assert sum(pix[-5:]) / 5 < sum(pix[:5]) / 5, (pix[0], pix[-1])


# --------------------------------------------------------------------------------------- #
# gate.slowts_decode_fidelity
# --------------------------------------------------------------------------------------- #
def test_slowts_decode_fidelity_keys_and_perfect_match():
    x = torch.randn(2, 12, 5)
    dm = gate.slowts_decode_fidelity(x, x)  # recon == target
    for k in ("envelope_corr", "peak_f1", "sharpness", "hf_energy_recon", "hf_energy_target"):
        assert k in dm
    assert dm["envelope_corr"] > 0.999
    assert abs(dm["sharpness"] - 1.0) < 1e-6
    assert 0.0 <= dm["peak_f1"] <= 1.0


def test_slowts_decode_fidelity_over_smoothed_is_less_sharp():
    torch.manual_seed(0)
    x = torch.randn(2, 12, 5)
    smooth = x.mean(dim=-1, keepdim=True).expand_as(x)  # collapse the time variation
    dm = gate.slowts_decode_fidelity(smooth, x)
    assert dm["sharpness"] < 1.0


def test_slowts_decode_fidelity_mask_used():
    """A validity mask makes the profile statistic average over valid samples only."""
    x = torch.randn(2, 12, 5)
    mask = torch.ones(2, 12, 5)
    mask[:, :, 0] = 0.0  # drop the first time sample from the profile average
    dm = gate.slowts_decode_fidelity(x, x, mask=mask)
    assert dm["envelope_corr"] > 0.999  # recon==target still perfect regardless
    with pytest.raises(ValueError):
        gate.slowts_decode_fidelity(torch.randn(2, 4), torch.randn(2, 4))  # wrong ndim
    with pytest.raises(ValueError):
        gate.slowts_decode_fidelity(x, x, mask=torch.ones(2, 12))  # bad mask shape


# --------------------------------------------------------------------------------------- #
# synthetic slow-TS HDF5 shots + SlowTSCodecPairDataset
# --------------------------------------------------------------------------------------- #
def _write_slowts_shot(path: Path, signal: str, channels: int, duration_s: float, seed: int,
                       zero_is_missing: bool) -> None:
    """Write a tiny {signal}/{xdata,ydata} HDF5 shot in the loader's format.

    ydata is (C, T) with a smooth per-position profile so windows are non-degenerate; some
    samples are made "missing" the way real data is: zeros for zero_is_missing (Thomson)
    signals, NaN otherwise (CER/MSE beam-off gaps).
    """
    rng = np.random.default_rng(seed)
    fps = 100.0  # native == target_fs for slow-TS
    T = int(round(duration_s * fps))
    t = np.linspace(0.0, duration_s, T)
    # smooth position profile that drifts slowly in time (real slow-TS is smooth).
    # keep it strictly positive so a zero is unambiguously the "missing" fill (Thomson).
    pos = np.linspace(1.0, 5.0, channels)[:, None]
    drift = 1.0 + 0.1 * np.sin(np.linspace(0, 2 * np.pi, T))[None, :]
    y = (pos * drift + 0.05 * np.abs(rng.standard_normal((channels, T)))).astype(np.float32)
    # inject missingness in the WINDOWED region (windows start at t0_start=1.0 s; each is
    # 50 ms). Put a block on a couple of positions inside a window at ~1.075 s so a sampled
    # 50 ms window actually covers it.
    miss_val = 0.0 if zero_is_missing else np.nan
    i0 = int(round(1.05 * fps))  # ~1.05 s, inside the first few windowed 50 ms clips
    y[0, i0 : i0 + 2] = miss_val
    y[channels // 2, i0 + 3 : i0 + 5] = miss_val
    with h5py.File(path, "w") as f:
        g = f.create_group(signal)
        g.create_dataset("xdata", data=t)
        g.create_dataset("ydata", data=y)


@pytest.fixture
def slowts_shots(tmp_path):
    """4 tiny ts_core_density shots (zero_is_missing) with room for several 50 ms windows."""
    signal, channels = "ts_core_density", 12
    duration_s = 1.0 + 8 * 0.05 + 0.1
    shots = []
    for i in range(4):
        sid = f"70000{i}"
        _write_slowts_shot(tmp_path / f"{sid}_processed.h5", signal, channels,
                           duration_s, seed=i, zero_is_missing=True)
        shots.append(sid)
    return {"dir": tmp_path, "shots": shots, "signal": signal, "channels": channels}


@pytest.fixture
def cer_shots(tmp_path):
    """4 tiny cer_ti shots (NaN-missing) — the CER/MSE missingness policy."""
    signal, channels = "cer_ti", 12
    duration_s = 1.0 + 8 * 0.05 + 0.1
    shots = []
    for i in range(4):
        sid = f"71000{i}"
        _write_slowts_shot(tmp_path / f"{sid}_processed.h5", signal, channels,
                           duration_s, seed=i + 100, zero_is_missing=False)
        shots.append(sid)
    return {"dir": tmp_path, "shots": shots, "signal": signal, "channels": channels}


def _tiny_slowts_cfg(signal="ts_core_density", channels=12) -> SlowTSCodecConfig:
    return SlowTSCodecConfig(
        signal=signal, channels=channels, time_steps=5, n_zones=2, patch_c=6, patch_t=5,
        d_model=32, enc_depth=1, dec_depth=1, heads=2, fsq_levels=[4, 4, 3],
    )


def test_modality_channels_slowts():
    # real loader channel counts for the 7 signals.
    assert tc.modality_channels("ts_core_density") == 44
    assert tc.modality_channels("ts_tangential_density") == 10
    assert tc.modality_channels("cer_ti") == 48
    assert tc.modality_channels("cer_rot") == 48
    assert tc.modality_channels("mse") == 69
    assert set(tc.SLOWTS_MODALITIES) == set(SLOWTS_SIGNALS)


def test_slowts_dataset_reuses_parent_index_machinery(slowts_shots):
    cfg = _tiny_slowts_cfg()
    ds = tc.SlowTSCodecPairDataset(
        slowts_shots["signal"], slowts_shots["shots"], cfg,
        data_dir=slowts_shots["dir"], seed=0,
    )
    # IS a TokamakMultiFileDataset (reuses idx-map + LRU handles + length cache + pickling).
    assert isinstance(ds, TokamakMultiFileDataset)
    assert len(ds) > 1
    # overrides ONLY the transform hook — NOT __getitem__ / the searchsorted index map.
    assert "_getitem_standard" in vars(tc.SlowTSCodecPairDataset)
    assert "__getitem__" not in vars(tc.SlowTSCodecPairDataset)
    src = inspect.getsource(tc.SlowTSCodecPairDataset)
    assert "searchsorted" not in src, "must reuse the parent's binary-search index map"
    assert int(ds._cumulative_lengths[-1]) == len(ds)


def test_slowts_dataset_shapes_and_mask(slowts_shots):
    cfg = _tiny_slowts_cfg()
    ds = tc.SlowTSCodecPairDataset(
        slowts_shots["signal"], slowts_shots["shots"], cfg,
        data_dir=slowts_shots["dir"], seed=0,
    )
    for i in range(min(6, len(ds))):
        signal, mask = ds[i]
        assert signal.shape == (cfg.channels, cfg.time_steps)
        assert mask.shape == (cfg.channels, cfg.time_steps)
        assert torch.isfinite(signal).all()
        # this is NOT a δ-pair: a single window tensor + a per-sample validity mask.
        assert mask.dim() == 2
        # mask is a {0,1} validity indicator with at least one valid sample.
        assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}
        assert float(mask.sum()) > 0.0


def test_slowts_dataset_zero_is_missing_mask(slowts_shots):
    """For a zero_is_missing (Thomson) signal, samples that are 0 in the window are invalid."""
    cfg = _tiny_slowts_cfg()  # ts_core_density -> zero_is_missing True
    assert cfg.zero_is_missing
    ds = tc.SlowTSCodecPairDataset(
        slowts_shots["signal"], slowts_shots["shots"], cfg,
        data_dir=slowts_shots["dir"], seed=0,
    )
    seen_missing = False
    for i in range(len(ds)):
        signal, mask = ds[i]
        # every masked-invalid sample must be exactly zero (the missing fill), and every
        # exact-zero sample must be masked invalid (the zero_is_missing contract).
        inv = mask < 0.5
        if inv.any():
            seen_missing = True
            assert torch.all(signal[inv] == 0.0)
        assert torch.all((signal == 0.0) == inv)
    assert seen_missing, "the synthetic shots inject zero-missing blocks; expected some"


def test_slowts_dataset_nan_missing_mask(cer_shots):
    """For a CER/MSE signal (NaN-missing), invalid samples come from the loader's NaN mask."""
    cfg = _tiny_slowts_cfg(signal="cer_ti", channels=cer_shots["channels"])
    assert not cfg.zero_is_missing
    ds = tc.SlowTSCodecPairDataset(
        cer_shots["signal"], cer_shots["shots"], cfg, data_dir=cer_shots["dir"], seed=0,
    )
    seen_missing = False
    for i in range(len(ds)):
        signal, mask = ds[i]
        assert torch.isfinite(signal).all()  # NaNs are zero-filled by the loader
        if (mask < 0.5).any():
            seen_missing = True
    assert seen_missing, "the synthetic CER shots inject NaN-missing blocks; expected some"


# --------------------------------------------------------------------------------------- #
# SCALE FIX — per-signal standardization of the codec input (log_standardize / standardize).
# The dataset must feed the codec the O(1) input the FM model consumes (each signal's
# SignalConfig.preprocess.method applied) instead of the unstandardized raw that collapses the
# ~1e19-scale Thomson density signals. Mirrors data_loader._apply_preprocessing EXACTLY.
# --------------------------------------------------------------------------------------- #
def _standardize_via_ds(raw, *, signal, method, mean, std, channels):
    """Run SlowTSCodecPairDataset._standardize on ``raw`` (C,T) via a cfg with the given stats.

    Builds the cfg + a dataset instance WITHOUT touching HDF5 (we only call the pure transform
    method) so this is a fast unit test of the exact standardize math.
    """
    # n_zones=1 (whole profile is one zone) — this test exercises the pure standardize math on a
    # raw (C,T), not the zone patchify, so a single-zone cfg keeps it valid for any C.
    cfg = SlowTSCodecConfig(signal=signal, channels=channels, time_steps=raw.shape[1],
                            n_zones=1, patch_c=channels, patch_t=raw.shape[1])
    cfg.preprocess_method = method
    cfg.channel_mean = mean
    cfg.channel_std = std
    ds = tc.SlowTSCodecPairDataset.__new__(tc.SlowTSCodecPairDataset)  # no __init__ / no HDF5
    ds.h5_file = None  # so the parent __del__ (which touches self.h5_file) is a no-op
    ds.codec_cfg = cfg
    return ds._standardize(raw)


def test_standardize_1e19_signal_becomes_O1_log_standardize():
    """A synthetic ~1e19-scale (Thomson density) signal is O(1) after the log_standardize path."""
    C, T = 4, 5
    torch.manual_seed(0)
    # electron density ~1e19 m^-3 scale, strictly positive.
    raw = (torch.rand(C, T) * 4.0 + 1.0) * 1e19
    assert raw.std() > 1e18  # confirm the pre-fix scale is huge
    # log-space stats (the loader reads the 'log' sub-dict for log_standardize): log10(1e19)~19.
    log_raw = torch.log10(raw.clamp(min=-0.99) + 1.0)
    mean = log_raw.mean(dim=1).tolist()
    std = log_raw.std(dim=1).clamp(min=1e-3).tolist()
    out = _standardize_via_ds(raw, signal="ts_core_density", method="log_standardize",
                              mean=mean, std=std, channels=C)
    assert torch.isfinite(out).all()
    # O(1): overall std ~1 and no 1e19-scale values survive.
    assert out.abs().max() < 20.0
    assert 0.3 < out.std().item() < 3.0
    # EXACT: matches data_loader's log10(clip(x,-0.99)+1) then (arr-mean)/std.clamp(1e-3).
    m = torch.tensor(mean).reshape(C, 1)
    s = torch.tensor(std).reshape(C, 1).clamp(min=1e-3)
    expected = (log_raw - m) / s
    assert torch.allclose(out, expected, atol=1e-5)


def test_standardize_path_is_x_minus_mean_over_std():
    """The 'standardize' (CER/MSE) path is EXACTLY (x - mean) / std.clamp(min=1e-3), RAW-space."""
    C, T = 3, 5
    torch.manual_seed(1)
    raw = torch.randn(C, T) * 400.0 + 100.0  # cer_ti-like raw scale (~hundreds)
    mean = raw.mean(dim=1).tolist()
    std = raw.std(dim=1).tolist()
    out = _standardize_via_ds(raw, signal="cer_ti", method="standardize",
                              mean=mean, std=std, channels=C)
    m = torch.tensor(mean).reshape(C, 1)
    s = torch.tensor(std).reshape(C, 1).clamp(min=1e-3)
    assert torch.allclose(out, (raw - m) / s, atol=1e-5)
    # per-channel ~ zero-mean unit-std (it was standardized by its own window stats here).
    assert out.mean().abs() < 1e-4


def test_standardize_std_clamp_matches_loader():
    """A tiny per-channel std is clamped at 1e-3 exactly like data_loader (no divide-by-~0 blowup)."""
    C, T = 2, 5
    raw = torch.zeros(C, T)
    raw[0] = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])   # ch0 constant -> std ~0
    raw[1] = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    mean = [1.0, 2.0]
    std = [0.0, 1.0]  # ch0 std 0 -> must clamp to 1e-3, not divide by 0
    out = _standardize_via_ds(raw, signal="cer_ti", method="standardize",
                              mean=mean, std=std, channels=C)
    assert torch.isfinite(out).all()
    # ch0: (1 - 1)/clamp(0,1e-3) = 0/1e-3 = 0
    assert torch.allclose(out[0], torch.zeros(T), atol=1e-6)


def test_none_stats_is_byte_identical_to_prefix():
    """None stats / method None => IDENTITY (byte-identical to the pre-fix raw path)."""
    C, T = 4, 5
    raw = torch.randn(C, T) * 1e18
    # no method, no stats -> identity
    out0 = _standardize_via_ds(raw, signal="ts_core_density", method=None,
                               mean=None, std=None, channels=C)
    assert torch.equal(out0, raw)
    # method set but stats None -> still identity
    out1 = _standardize_via_ds(raw, signal="ts_core_density", method="log_standardize",
                               mean=None, std=None, channels=C)
    assert torch.equal(out1, raw)
    # method "none" with stats present -> identity
    out2 = _standardize_via_ds(raw, signal="ts_core_density", method="none",
                               mean=[0.0] * C, std=[1.0] * C, channels=C)
    assert torch.equal(out2, raw)


def test_standardize_wrong_shape_stats_rejected():
    """Stats of length != C are rejected loud (a mis-pointed stats file must fail, not skip)."""
    C, T = 4, 5
    raw = torch.randn(C, T)
    with pytest.raises(ValueError):
        _standardize_via_ds(raw, signal="cer_ti", method="standardize",
                            mean=[0.0] * (C - 1), std=[1.0] * (C - 1), channels=C)
    with pytest.raises(ValueError):
        _standardize_via_ds(raw, signal="ts_core_density", method="bogus_method",
                            mean=[0.0] * C, std=[1.0] * C, channels=C)


def test_slowts_dataset_zero_is_missing_mask_preserved_after_standardize(slowts_shots):
    """With stats injected, the zero_is_missing mask still matches the RAW zeros (mask semantics
    intact); the codec input is standardized (values move OFF the raw zero/scale)."""
    channels = slowts_shots["channels"]
    # build the pre-fix (identity) dataset to read the RAW windows + masks.
    cfg_raw = _tiny_slowts_cfg(channels=channels)
    ds_raw = tc.SlowTSCodecPairDataset(
        slowts_shots["signal"], slowts_shots["shots"], cfg_raw,
        data_dir=slowts_shots["dir"], seed=0,
    )
    # build the FIXED dataset with log_standardize stats (matched to the synthetic channel count).
    cfg_fix = _tiny_slowts_cfg(channels=channels)
    cfg_fix.preprocess_method = "log_standardize"
    cfg_fix.channel_mean = [1.0] * channels
    cfg_fix.channel_std = [0.5] * channels
    ds_fix = tc.SlowTSCodecPairDataset(
        slowts_shots["signal"], slowts_shots["shots"], cfg_fix,
        data_dir=slowts_shots["dir"], seed=0,
    )
    seen_missing = False
    for i in range(len(ds_raw)):
        raw_sig, raw_mask = ds_raw[i]
        fix_sig, fix_mask = ds_fix[i]
        # 1. mask is IDENTICAL (built from raw before standardization).
        assert torch.equal(raw_mask, fix_mask)
        # 2. masked-invalid positions were raw-zero (the missing fill) — the loss ignores them.
        inv = fix_mask < 0.5
        if inv.any():
            seen_missing = True
            assert torch.all(raw_sig[inv] == 0.0)
        # 3. the standardized input differs from raw wherever raw != 0 (input actually rescaled).
        present = fix_mask > 0.5
        if present.any():
            assert not torch.allclose(fix_sig[present], raw_sig[present])
    assert seen_missing, "synthetic shots inject zero-missing blocks; expected some"


# --------------------------------------------------------------------------------------- #
# INPUT-MASK FIX (Bug B) — after standardization a MISSING position (raw 0) maps to a
# LARGE-NEGATIVE artifact (~-25 for ts_core_density); the encoder sees it (the loss is masked,
# the INPUT is not) and with a ⅔-missing signal the input is dominated by -25 -> collapse.
# The dataset must ZERO the missing positions in the codec INPUT (0 == standardized neutral),
# while keeping the loss masking (missing still excluded from the recon loss).
# --------------------------------------------------------------------------------------- #
def _fit_window_via_ds(raw, nan_mask, *, signal, method, mean, std, channels,
                       zero_is_missing_ok, n_zones=1):
    """Run SlowTSCodecPairDataset._fit_window on a synthetic (raw, nan_mask) WITHOUT HDF5.

    ``n_zones=1`` (whole profile is one zone; padded_channels == channels) isolates the
    standardize + input-masking behaviour these tests check. The zone-padding + mask behaviour
    for the 4-zone production layout is covered separately in
    ``test_slowts_zone_padding_masked_and_4_tokens``.
    """
    T = raw.shape[1]
    patch_c = -(-channels // n_zones)               # ceil(channels / n_zones)
    cfg = SlowTSCodecConfig(signal=signal, channels=channels, time_steps=T,
                            n_zones=n_zones, patch_c=patch_c, patch_t=T)
    cfg.preprocess_method = method
    cfg.channel_mean = mean
    cfg.channel_std = std
    ds = tc.SlowTSCodecPairDataset.__new__(tc.SlowTSCodecPairDataset)  # no __init__ / no HDF5
    ds.h5_file = None
    ds.codec_cfg = cfg
    return ds._fit_window(raw, nan_mask)


def test_slowts_input_missing_positions_zeroed_after_standardize():
    """A ⅔-missing (raw-0) Thomson-density window: present values O(1), MISSING positions == 0.

    Before the fix, missing positions standardized to ~-25 (pp(0)-mean)/std and dominated the
    encoder input (std ~7.4). After the fix they are 0 (neutral) and the whole-window std is O(1).
    """
    C, T = 6, 5
    torch.manual_seed(0)
    # ts_core_density-like: strictly-positive ~1e19 present values; a raw 0 == missing.
    raw = (torch.rand(C, T) * 4.0 + 1.0) * 1e19
    raw[: (2 * C) // 3] = 0.0                     # ⅔ of the positions missing (raw 0)
    nan_mask = torch.zeros(C, T)                  # no NaNs; zero_is_missing carries the missingness
    log_raw = torch.log10(raw.clamp(min=-0.99) + 1.0)
    # GLOBAL per-channel stats (as preprocessing_stats.pt carries — computed across the whole
    # dataset where a channel IS mostly present, NOT per-window): every channel's log-mean ~19.4
    # for a ~1e19 density. This is the regime that turns a missing raw-0 into the -25 artifact:
    # pp(0)=log10(1)=0 -> (0 - 19.4)/0.7 ~ -27. (A per-WINDOW fallback would hide it.)
    mean = [19.4] * C
    std = [0.7] * C
    signal, valid = _fit_window_via_ds(
        raw, nan_mask, signal="ts_core_density", method="log_standardize",
        mean=mean, std=std, channels=C, zero_is_missing_ok=True,
    )
    inv = valid < 0.5
    pres = valid > 0.5
    assert inv.any() and pres.any()
    # MISSING positions are EXACTLY 0 (not the ~-25 large-negative artifact).
    assert torch.all(signal[inv] == 0.0)
    # if we had NOT masked the input, those positions would be pp(0)-mean/std ~ large-negative.
    m = torch.tensor(mean).reshape(C, 1); s = torch.tensor(std).reshape(C, 1).clamp(min=1e-3)
    unmasked = (log_raw - m) / s
    assert float(unmasked[inv].min()) < -5.0, "sanity: unmasked missing IS a large-negative artifact"
    # PRESENT values are O(1) (standardized), not the 1e19 raw scale.
    assert torch.isfinite(signal).all()
    assert float(signal[pres].abs().max()) < 20.0
    # whole-window input std is O(1) now (was ~7.4 dominated by the -25 mass).
    assert float(signal.std()) < 5.0


def test_slowts_input_mask_semantics_preserved_and_loss_still_masks_missing():
    """Zeroing the input does NOT change the validity mask, and the recon loss still ignores
    missing (a codec whose recon is 0 at missing but exact at present has ~0 masked recon)."""
    C, T = 6, 5
    torch.manual_seed(1)
    raw = (torch.rand(C, T) * 4.0 + 1.0) * 1e19
    raw[:2] = 0.0                                  # first 2 positions missing
    nan_mask = torch.zeros(C, T)
    log_raw = torch.log10(raw.clamp(min=-0.99) + 1.0)
    mean = [float(log_raw[c][raw[c] != 0].mean()) if (raw[c] != 0).any() else 0.0 for c in range(C)]
    std = [float(log_raw[c][raw[c] != 0].std().clamp(min=1e-3)) if (raw[c] != 0).any() else 1.0
           for c in range(C)]
    signal, valid = _fit_window_via_ds(
        raw, nan_mask, signal="ts_core_density", method="log_standardize",
        mean=mean, std=std, channels=C, zero_is_missing_ok=True,
    )
    # validity mask matches the RAW zero_is_missing policy (raw != 0), unchanged by input-zeroing.
    assert torch.equal(valid, (raw != 0.0).to(torch.float32))
    # the loss (masked recon MAE) ignores missing: recon == signal everywhere except missing,
    # where recon is arbitrary -> masked MAE is 0 regardless of the missing recon value.
    cfg = SlowTSCodecConfig(signal="ts_core_density", channels=C, time_steps=T,
                            n_zones=1, patch_c=C, patch_t=T)
    recon = signal.clone().unsqueeze(0)
    recon[:, valid < 0.5] = 999.0                  # garbage at missing positions
    mae = SlowTSCodec._masked_recon_mae(recon, signal.unsqueeze(0), valid.unsqueeze(0))
    assert float(mae) < 1e-6, "masked recon must ignore the garbage at missing positions"


def test_slowts_input_masking_noop_when_stats_absent():
    """Without stats (pre-fix / stat-less path) the input is NOT re-zeroed (byte-identical)."""
    C, T = 6, 5
    torch.manual_seed(2)
    raw = torch.randn(C, T) * 1e18
    raw[:2] = 0.0
    nan_mask = torch.zeros(C, T)
    signal, valid = _fit_window_via_ds(
        raw, nan_mask, signal="ts_core_density", method=None,
        mean=None, std=None, channels=C, zero_is_missing_ok=True,
    )
    # identity standardization + no input-zeroing -> exactly the raw window.
    assert torch.equal(signal, raw)


# --------------------------------------------------------------------------------------- #
# RADIAL-ZONE PADDING (the 4-token layout) — a non-÷4 channel count is padded up to
# padded_channels and the pad tail is MASKED missing + input-zeroed (never in loss/encoder).
# --------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("channels,padded", [(10, 12), (69, 72), (44, 44), (48, 48)])
def test_slowts_zone_padding_masked_and_4_tokens(channels, padded):
    """_fit_window pads C up to padded_channels; the pad tail is input-0 + mask-invalid."""
    T = 5
    torch.manual_seed(channels)
    raw = (torch.rand(channels, T) * 4.0 + 1.0) * 1e19   # all present (positive) Thomson density
    nan_mask = torch.zeros(channels, T)
    log_raw = torch.log10(raw + 1.0)
    mean = log_raw.mean(dim=1).tolist()
    std = log_raw.std(dim=1).clamp(min=1e-3).tolist()
    signal, valid = _fit_window_via_ds(
        raw, nan_mask, signal="ts_core_density", method="log_standardize",
        mean=mean, std=std, channels=channels, zero_is_missing_ok=True, n_zones=4,
    )
    # padded to padded_channels = 4 * ceil(C/4); patches into EXACTLY 4 radial-zone tokens.
    assert signal.shape == (padded, T) and valid.shape == (padded, T)
    n_pad = padded - channels
    if n_pad > 0:
        # pad tail: input-zeroed (neutral -> absent to the encoder) AND mask-invalid (out of loss).
        assert torch.all(signal[channels:] == 0.0)
        assert torch.all(valid[channels:] == 0.0)
    # the REAL profile rows are present + standardized to O(1) (unaffected by padding).
    assert torch.all(valid[:channels] == 1.0)
    assert float(signal[:channels].abs().max()) < 20.0
    # feed the padded window through the production-sized codec -> exactly 4 tokens.
    cfg = tc.slowts_codec_cfg("ts_core_density", channels)
    cfg.d_model, cfg.enc_depth, cfg.dec_depth, cfg.heads = 32, 1, 1, 2
    assert cfg.padded_channels == padded and cfg.n_tok == 4
    out = SlowTSCodec(cfg)(signal.unsqueeze(0))
    assert out["codes"].shape == (1, 4, 4)
    assert out["recon"].shape == (1, padded, T)


def test_load_slowts_channel_stats_method_and_length():
    """load_slowts_channel_stats returns the FM's per-signal method + a C-length mean/std, reading
    the 'log' sub-dict for log_standardize signals and 'raw' for standardize signals."""
    stats_path = "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt"
    if not Path(stats_path).exists():
        pytest.skip("canonical preprocessing_stats.pt not present")
    expect_method = {
        "ts_core_density": "log_standardize", "ts_core_temp": "log_standardize",
        "ts_tangential_density": "log_standardize", "ts_tangential_temp": "log_standardize",
        "cer_ti": "standardize", "cer_rot": "standardize", "mse": "standardize",
    }
    for sig, method in expect_method.items():
        m, mean, std = tc.load_slowts_channel_stats(sig, stats_path)
        assert m == method
        C = tc.modality_channels(sig)
        assert len(mean) == C and len(std) == C
        # NaN / inf handling: every returned stat is finite (loader maps bad stats to 0/1).
        assert all(math.isfinite(v) for v in mean)
        assert all(math.isfinite(v) for v in std)
    with pytest.raises(ValueError):
        tc.load_slowts_channel_stats("not_a_slowts_signal", stats_path)


def test_slowts_loader_num_workers_0_and_2(slowts_shots):
    cfg = _tiny_slowts_cfg()
    ds = tc.SlowTSCodecPairDataset(
        slowts_shots["signal"], slowts_shots["shots"], cfg,
        data_dir=slowts_shots["dir"], seed=1,
    )
    loader0 = tc.make_slowts_loader(ds, batch_size=2, num_workers=0, seed=1)
    sig, mask = next(iter(loader0))
    assert sig.shape == (2, cfg.channels, cfg.time_steps)
    assert mask.shape == (2, cfg.channels, cfg.time_steps)

    loader2 = tc.make_slowts_loader(ds, batch_size=2, num_workers=2, seed=2)
    seen = 0
    for sig, mask in loader2:
        assert sig.shape[1:] == (cfg.channels, cfg.time_steps)
        assert torch.isfinite(sig).all()
        seen += 1
        if seen >= 2:
            break
    assert seen >= 1


def test_slowts_loader_ddp_sampler_shards(slowts_shots):
    cfg = _tiny_slowts_cfg()
    ds = tc.SlowTSCodecPairDataset(
        slowts_shots["signal"], slowts_shots["shots"], cfg,
        data_dir=slowts_shots["dir"], seed=0,
    )
    l0 = tc.make_slowts_loader(ds, batch_size=1, num_workers=0, rank=0, world_size=2, seed=0)
    l1 = tc.make_slowts_loader(ds, batch_size=1, num_workers=0, rank=1, world_size=2, seed=0)
    assert isinstance(l0.sampler, tc.DistributedTwoLevelSampler)
    assert set(iter(l0.sampler)).isdisjoint(set(iter(l1.sampler)))


# --------------------------------------------------------------------------------------- #
# slowts_compute_gate — finite for a reconstructing input, -inf score on NaN
# --------------------------------------------------------------------------------------- #
def test_slowts_nuisance_preserves_structure_but_differs():
    cfg = _small_cfg()
    x = torch.randn(1, cfg.channels, cfg.time_steps)
    y = tc.slowts_nuisance(x, seed=0)
    assert y.shape == x.shape
    # a nonzero amplitude jitter moves the realization (seed 0 gives a nonzero scale here).
    assert not torch.allclose(x, y)


def test_slowts_compute_gate_finite_for_reconstructing_input():
    from tokamak_foundation_model.ignite import spike

    cfg = _small_cfg()
    cfg.gate_recon_floor = -1.0  # never disqualify: prove finiteness of the plumbing
    codec = SlowTSCodec(cfg)
    B, n_win = 2, 4
    windows = [torch.randn(B, cfg.channels, cfg.time_steps) for _ in range(2)]
    masks = [torch.ones(B, cfg.channels, cfg.time_steps) for _ in range(2)]
    frame_seq = torch.randn(B, n_win, cfg.channels, cfg.time_steps)
    g = tc.slowts_compute_gate(codec, windows, masks, frame_seq, cfg)
    assert 0.0 <= g["stability"] <= 1.0
    assert 0.0 <= g["persistence"] <= 1.0
    assert isinstance(g["pass_stability"], bool) and isinstance(g["pass_persistence"], bool)
    for k in ("envelope_corr", "peak_f1", "sharpness"):
        assert math.isfinite(g["decode"][k])
    score = spike.gate_score(g, recon_floor=cfg.gate_recon_floor)
    assert math.isfinite(score)


def test_slowts_gate_score_minus_inf_on_nan_recon():
    from tokamak_foundation_model.ignite import spike

    g = {
        "forecastability": {"margin_transition": 0.0},
        "decode": {"envelope_corr": float("nan"), "peak_f1": 0.0, "sharpness": 1.0},
        "utilization": {"min_dim_entropy": 0.0, "frac_of_observable": 0.0},
    }
    assert spike.gate_score(g, recon_floor=0.2) == float("-inf")


# --------------------------------------------------------------------------------------- #
# per-step + full train_slowts_codec loop (single-process CPU)
# --------------------------------------------------------------------------------------- #
def test_slowts_train_step_reduces_masked_recon(slowts_shots):
    torch.manual_seed(0)
    cfg = _tiny_slowts_cfg()
    codec = SlowTSCodec(cfg)
    opt_g = torch.optim.Adam(codec.parameters(), lr=1e-2)

    ds = tc.SlowTSCodecPairDataset(
        slowts_shots["signal"], slowts_shots["shots"], cfg,
        data_dir=slowts_shots["dir"], seed=0,
    )
    items = [ds[i] for i in range(min(4, len(ds)))]
    x = torch.stack([a for a, _ in items], dim=0)
    m = torch.stack([b for _, b in items], dim=0)

    pix = []
    for step in range(30):
        g_terms = tc.slowts_codec_train_step(codec, opt_g, x, m, cfg, step=step)
        pix.append(float(g_terms["pixel"].detach()))
        assert math.isfinite(pix[-1])
    assert sum(pix[-5:]) / 5 < sum(pix[:5]) / 5, (pix[0], pix[-1])


def test_train_slowts_codec_loop_writes_artifacts(slowts_shots, tmp_path, monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    cfg = _tiny_slowts_cfg()
    out_dir = tmp_path / "srun"
    final_gate = tc.train_slowts_codec(
        cfg,
        slowts_shots["signal"],
        slowts_shots["shots"],
        slowts_shots["shots"],  # eval on the same tiny pool (test only)
        steps=3,
        eval_every=1,
        batch_size=2,
        num_workers=0,
        data_dir=slowts_shots["dir"],
        lr=1e-3,
        eval_batches=2,
        eval_batch_size=2,
        eval_frames=3,
        out_dir=out_dir,
        seed=0,
    )
    assert final_gate["steps"] == 3 and final_gate["global_step"] == 3
    assert 0.0 <= final_gate["stability"] <= 1.0
    assert 0.0 <= final_gate["persistence"] <= 1.0
    assert sorted(out_dir.glob("gate_*.json"))
    assert (out_dir / "codec_last.pt").exists()


def test_train_slowts_codec_cli_main(slowts_shots, tmp_path, monkeypatch):
    """argparse main() dispatches --modality ts_core_density to the slow-TS trainer."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    out_dir = tmp_path / "scli"

    # shrink the codec built inside main() to a tiny transformer.
    orig = tc.slowts_codec_cfg

    def _small(signal, channels):
        cfg = orig(signal, channels)
        cfg.d_model, cfg.enc_depth, cfg.dec_depth, cfg.heads = 32, 1, 1, 2
        cfg.fsq_levels = [4, 4, 3]
        # single token per window (patch_c == channels) so codebook math holds.
        return cfg

    monkeypatch.setattr(tc, "slowts_codec_cfg", _small)

    argv = [
        "--modality", slowts_shots["signal"],
        "--shots", ",".join(slowts_shots["shots"]),
        "--eval_n_shots", "1",
        "--n_shots", "3",
        "--steps", "2",
        "--eval_every", "1",
        "--batch_size", "2",
        "--num_workers", "0",
        "--eval_batches", "2",
        "--eval_batch_size", "2",
        "--eval_frames", "3",
        "--out_dir", str(out_dir),
        "--data_dir", str(slowts_shots["dir"]),
        # --stats_path='' DISABLES the standardization SCALE FIX (raw path) so this synthetic-shot
        # smoke test does not depend on the real preprocessing_stats.pt being on disk (the fix's
        # own math is unit-tested above; a dedicated CLI test below exercises the ON path).
        "--stats_path", "",
        "--seed", "0",
    ]
    gate_dict = tc.main(argv)
    assert gate_dict["steps"] == 2
    assert (out_dir / "summary.json").exists()
    assert sorted(out_dir.glob("gate_*.json"))


def test_slowts_cli_main_standardization_on_injects_stats(slowts_shots, tmp_path, monkeypatch):
    """The ON dispatch path: main() loads the per-signal method + stats and INJECTS them onto the
    cfg the trainer builds (so the codec input is standardized). Uses synthetic stats matched to
    the tiny shot channel count via a monkeypatched loader — no dependency on the real stats file."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    out_dir = tmp_path / "scli_on"
    # the codec operates on the LOADER's channel count (44 for ts_core_density; the synthetic shot
    # is zero-padded up to it), so the injected stats must be that length.
    channels = tc.modality_channels(slowts_shots["signal"])

    captured = {}
    orig = tc.slowts_codec_cfg

    def _small(signal, chans):
        cfg = orig(signal, chans)
        cfg.d_model, cfg.enc_depth, cfg.dec_depth, cfg.heads = 32, 1, 1, 2
        cfg.fsq_levels = [4, 4, 3]
        captured["cfg"] = cfg
        return cfg

    def _fake_stats(signal, stats_path=None):
        # synthetic log_standardize stats matched to the loader's channel count.
        return "log_standardize", [1.0] * channels, [0.5] * channels

    monkeypatch.setattr(tc, "slowts_codec_cfg", _small)
    monkeypatch.setattr(tc, "load_slowts_channel_stats", _fake_stats)

    argv = [
        "--modality", slowts_shots["signal"],
        "--shots", ",".join(slowts_shots["shots"]),
        "--eval_n_shots", "1", "--n_shots", "3",
        "--steps", "1", "--eval_every", "1",
        "--batch_size", "2", "--num_workers", "0",
        "--eval_batches", "2", "--eval_batch_size", "2", "--eval_frames", "3",
        "--out_dir", str(out_dir), "--data_dir", str(slowts_shots["dir"]),
        "--stats_path", "/does/not/need/to/exist.pt",  # loader is monkeypatched
        "--seed", "0",
    ]
    gate_dict = tc.main(argv)
    assert gate_dict["steps"] == 1
    # the fix was injected onto the cfg the trainer actually used.
    cfg = captured["cfg"]
    assert cfg.preprocess_method == "log_standardize"
    assert cfg.channel_mean == [1.0] * channels
    assert cfg.channel_std == [0.5] * channels


def test_slowts_rejects_consistency_weight_cli(slowts_shots, tmp_path, monkeypatch):
    """--consistency_weight is invalid for a slow-TS modality (no shift-consistency term)."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    argv = [
        "--modality", slowts_shots["signal"],
        "--shots", ",".join(slowts_shots["shots"]),
        "--steps", "1", "--out_dir", str(tmp_path / "x"),
        "--data_dir", str(slowts_shots["dir"]),
        "--consistency_weight", "1.0",
    ]
    with pytest.raises(SystemExit):
        tc.main(argv)
