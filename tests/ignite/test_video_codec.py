"""CPU TDD spec for the IGNITE Phase-A tangtv **video** codec family.

Mirrors the spectro codec's tests but for the video pieces (docs/IGNITE_DESIGN.md §4.3):

    video_nets.VideoEncoder(cfg)(x: (B,C,T,H,W)) -> feats (B, n_tok, d_model)
    video_nets.VideoDecoder(cfg)(quant: (B,n_tok,d_model)) -> recon (B,C,T,H,W)
    video_codec.VideoCodec(cfg).forward(x) -> dict(recon, feats, quant, codes)
    video_codec.VideoCodec.generator_losses(x, disc, cfg, step, frame_mask) -> dict(total,...)
    video_discriminator.FramePatchGAN(cfg)(x) -> list of per-frame patch-score maps
    gate.video_decode_fidelity(recon, target) -> dict(envelope_corr, peak_f1, sharpness)

    train_codec.VideoCodecPairDataset  (subclass of TokamakMultiFileDataset)
    train_codec.video_compute_gate / train_codec.train_video_codec

Small synthetic tensors + tiny SYNTHETIC HDF5 tangtv shots only. CPU. No SLURM / GPU / real
data. NO δ-shift pair anywhere (the video codec has no consistency term).

Run:
    .pixi/envs/default/bin/python -m pytest tests/ignite/test_video_codec.py -q
"""
from __future__ import annotations

import inspect
import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from tokamak_foundation_model.data.multi_file_dataset import TokamakMultiFileDataset
from tokamak_foundation_model.ignite import gate
from tokamak_foundation_model.ignite import train_codec as tc
from tokamak_foundation_model.ignite.config import VideoCodecConfig
from tokamak_foundation_model.ignite.video_codec import VideoCodec
from tokamak_foundation_model.ignite.video_discriminator import FramePatchGAN
from tokamak_foundation_model.ignite.video_nets import VideoDecoder, VideoEncoder


# --------------------------------------------------------------------------------------- #
# tiny configs
# --------------------------------------------------------------------------------------- #
def _small_cfg(channels: int = 2) -> VideoCodecConfig:
    """Small transformer + small space-time grid (divisible patching)."""
    return VideoCodecConfig(
        channels=channels,
        frames=4,
        height=16,
        width=24,
        patch_t=2,
        patch_h=8,
        patch_w=8,
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
    assert cfg.n_time_patch == 2 and cfg.n_height_patch == 2 and cfg.n_width_patch == 3
    assert cfg.n_tok == 2 * 2 * 3 == 12
    assert cfg.fsq_dim == 3 and cfg.codebook_size == 4 * 4 * 3

    # production default: 108 tokens/window, 1000-code FSQ.
    prod = VideoCodecConfig()
    assert prod.n_tok == 108
    assert prod.codebook_size == 1000 and prod.fsq_dim == 4
    assert (prod.channels, prod.frames, prod.height, prod.width) == (2, 5, 120, 360)


def test_config_rejects_indivisible_patch_and_bad_divertor():
    with pytest.raises(AssertionError):
        VideoCodecConfig(width=25, patch_w=8)  # 25 % 8 != 0
    with pytest.raises(AssertionError):
        VideoCodecConfig(divertor="middle")


# --------------------------------------------------------------------------------------- #
# nets round-trip
# --------------------------------------------------------------------------------------- #
def test_encoder_decoder_shapes():
    cfg = _small_cfg()
    enc, dec = VideoEncoder(cfg), VideoDecoder(cfg)
    B = 3
    x = torch.randn(B, cfg.channels, cfg.frames, cfg.height, cfg.width)
    feats = enc(x)
    assert feats.shape == (B, cfg.n_tok, cfg.d_model)
    recon = dec(feats)
    assert recon.shape == x.shape
    assert torch.isfinite(recon).all()


def test_nets_gradient_flows_end_to_end():
    cfg = _small_cfg()
    enc, dec = VideoEncoder(cfg), VideoDecoder(cfg)
    x = torch.randn(2, cfg.channels, cfg.frames, cfg.height, cfg.width, requires_grad=True)
    dec(enc(x)).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_decoder_last_layer_exposed_for_adaptive_weight():
    cfg = _small_cfg()
    dec = VideoDecoder(cfg)
    # last_layer must be the to_pixels weight Parameter (the VQGAN adaptive-weight anchor).
    assert dec.last_layer is dec.to_pixels.weight
    assert isinstance(dec.last_layer, torch.nn.Parameter)


# --------------------------------------------------------------------------------------- #
# VideoCodec forward + codes at the DEFAULT FSQ size
# --------------------------------------------------------------------------------------- #
def test_codec_forward_shapes_and_codes_in_range():
    cfg = _small_cfg()
    codec = VideoCodec(cfg)
    B = 2
    x = torch.randn(B, cfg.channels, cfg.frames, cfg.height, cfg.width) * 3.0
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


def test_codec_roundtrip_at_default_fsq_size():
    """encode -> quantize -> decode at the right-sized default FSQ ([8,5,5,5]=1000, 4 dims)."""
    cfg = VideoCodecConfig(
        channels=2, frames=4, height=16, width=24, patch_t=2, patch_h=8, patch_w=8,
        d_model=32, enc_depth=1, dec_depth=1, heads=2,
    )  # default fsq_levels [8,5,5,5]
    assert cfg.fsq_dim == 4 and cfg.codebook_size == 1000
    codec = VideoCodec(cfg)
    x = torch.randn(2, cfg.channels, cfg.frames, cfg.height, cfg.width) * 5.0
    out = codec(x)
    assert out["recon"].shape == x.shape
    assert out["codes"].shape == (2, cfg.n_tok, 4)
    levels = torch.tensor(cfg.fsq_levels)
    assert (out["codes"] >= 0).all() and (out["codes"] < levels).all()
    assert torch.isfinite(out["recon"]).all()


# --------------------------------------------------------------------------------------- #
# FramePatchGAN discriminator
# --------------------------------------------------------------------------------------- #
def test_frame_patchgan_multiscale_raw_scores():
    cfg = _small_cfg()
    torch.manual_seed(0)
    disc = FramePatchGAN(cfg)
    x = torch.randn(3, cfg.channels, cfg.frames, cfg.height, cfg.width)
    out = disc(x)
    assert isinstance(out, list) and len(out) >= 2, "expected multi-scale output"
    # per-frame scoring -> batch axis is B*T
    for m in out:
        assert m.dim() == 4 and m.shape[0] == 3 * cfg.frames
        assert torch.isfinite(m).all()
    # different scales -> different spatial resolution
    assert len({(m.shape[-2], m.shape[-1]) for m in out}) >= 2
    # raw hinge scores (no sigmoid) span negative & positive with random weights
    cat = torch.cat([m.reshape(-1) for m in out])
    assert (cat < 0).any() and (cat > 0).any()


def test_frame_patchgan_return_features_and_grad():
    cfg = _small_cfg()
    disc = FramePatchGAN(cfg)
    x = torch.randn(2, cfg.channels, cfg.frames, cfg.height, cfg.width, requires_grad=True)
    scores, feats = disc(x, return_features=True)
    assert isinstance(scores, list) and isinstance(feats, list) and len(feats) > 0
    # default forward path byte-identical to the score list of return_features
    plain = disc(x)
    for a, b in zip(plain, scores):
        assert torch.allclose(a, b)
    loss = sum(m.mean() for m in scores)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# --------------------------------------------------------------------------------------- #
# generator_losses — finite, no consistency term, adaptive weight finite/clamped, masking
# --------------------------------------------------------------------------------------- #
def test_generator_losses_keys_finite_and_no_consistency():
    cfg = _small_cfg()
    codec = VideoCodec(cfg)
    disc = FramePatchGAN(cfg)
    x = torch.randn(2, cfg.channels, cfg.frames, cfg.height, cfg.width)
    out = codec.generator_losses(x, disc, cfg, step=0)
    for k in ("total", "adversarial", "pixel", "feature_matching", "entropy"):
        assert k in out and out[k].dim() == 0 and torch.isfinite(out[k]).all()
    # NO consistency term for the video codec (the deliberate difference from spectro).
    assert "consistency" not in out
    assert out["recon"].shape == x.shape
    assert out["codes"].shape == (2, cfg.n_tok, cfg.fsq_dim)


def test_generator_losses_signature_has_no_x_shift():
    """The video generator loss takes a SINGLE x (no δ-shift nuisance pair)."""
    params = list(inspect.signature(VideoCodec.generator_losses).parameters)
    assert "x_shift" not in params, "video codec must not take a shift pair"
    assert params[1] == "x" and "frame_mask" in params


def test_adaptive_adv_weight_finite_and_clamped():
    cfg = _small_cfg()
    cfg.adaptive_adv_weight = True
    cfg.adaptive_adv_clamp = 2.0
    codec = VideoCodec(cfg)
    disc = FramePatchGAN(cfg)
    x = torch.randn(2, cfg.channels, cfg.frames, cfg.height, cfg.width)
    out = codec.generator_losses(x, disc, cfg, step=0)
    lam = out["adaptive_weight"]
    assert math.isfinite(lam)
    assert 0.0 <= lam <= cfg.adaptive_adv_clamp + 1e-6


def test_generator_step_reduces_pixel_anchor():
    """Repeated Adam steps on ONE clip drive the pixel reconstruction anchor down."""
    torch.manual_seed(0)
    cfg = _small_cfg()
    codec = VideoCodec(cfg)
    disc = FramePatchGAN(cfg)
    opt = torch.optim.Adam(codec.parameters(), lr=1e-2)
    x = torch.randn(2, cfg.channels, cfg.frames, cfg.height, cfg.width)

    pix = []
    for step in range(20):
        codec.train()
        g = codec.generator_losses(x, disc, cfg, step=step)
        opt.zero_grad(set_to_none=True)
        g["total"].backward()
        opt.step()
        pix.append(float(g["pixel"].detach()))
        assert math.isfinite(pix[-1])
    assert sum(pix[-5:]) / 5 < sum(pix[:5]) / 5, (pix[0], pix[-1])


def test_frame_mask_restricts_pixel_anchor():
    """A per-frame validity mask must make the pixel anchor ignore masked-out frames."""
    cfg = _small_cfg()
    codec = VideoCodec(cfg)
    disc = FramePatchGAN(cfg)
    torch.manual_seed(1)
    x = torch.randn(2, cfg.channels, cfg.frames, cfg.height, cfg.width)
    # all-valid vs all-invalid masks: an all-invalid mask falls back to the full MAE (finite),
    # while a partial mask changes the pixel term. Assert the masked pixel is finite and that a
    # mask selecting a single frame differs from the unmasked mean.
    full = codec.generator_losses(x, disc, cfg, step=0)["pixel"]
    m = torch.zeros(2, cfg.frames)
    m[:, 0] = 1.0  # keep only frame 0
    masked = codec.generator_losses(x, disc, cfg, step=0, frame_mask=m)["pixel"]
    assert torch.isfinite(masked)
    assert not torch.allclose(full, masked)


# --------------------------------------------------------------------------------------- #
# gate.video_decode_fidelity
# --------------------------------------------------------------------------------------- #
def test_video_decode_fidelity_keys_and_perfect_match():
    x = torch.randn(2, 2, 4, 8, 10)
    dm = gate.video_decode_fidelity(x, x)  # recon == target
    for k in ("envelope_corr", "peak_f1", "sharpness", "hf_energy_recon", "hf_energy_target"):
        assert k in dm
    assert dm["envelope_corr"] > 0.999           # identical intensity map -> corr 1
    assert abs(dm["sharpness"] - 1.0) < 1e-6      # identical HF energy -> ratio 1
    assert 0.0 <= dm["peak_f1"] <= 1.0


def test_video_decode_fidelity_blurred_is_less_sharp():
    torch.manual_seed(0)
    x = torch.randn(2, 2, 4, 12, 12)
    blur = x.mean(dim=(-1, -2), keepdim=True).expand_as(x)  # spatial mean collapse
    dm = gate.video_decode_fidelity(blur, x)
    assert dm["sharpness"] < 1.0, "a mean-collapsed recon must have less HF gradient energy"


def test_video_decode_fidelity_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        gate.video_decode_fidelity(torch.randn(2, 2, 8, 10), torch.randn(2, 2, 8, 10))


# --------------------------------------------------------------------------------------- #
# synthetic tangtv HDF5 shots + VideoCodecPairDataset
# --------------------------------------------------------------------------------------- #
def _write_tangtv_shot(path: Path, duration_s: float, seed: int,
                       raw_hw=(30, 40), off=(1, 3, 5)) -> None:
    """Write a tiny tangtv/{xdata,ydata} HDF5 shot in the loader's format.

    ydata is (7, T, H, W) with modes-in-brightness so windows are non-degenerate; the off
    filters (default ch 1,3,5) are stored as fully-NaN slabs, exactly like real tangtv.
    """
    rng = np.random.default_rng(seed)
    C = 7
    fps = 200.0  # native fps; loader resamples to target_fps=100
    T = int(round(duration_s * fps))
    H, W = raw_hw
    t = np.linspace(0.0, duration_s, T)
    y = rng.random((C, T, H, W)).astype(np.float32)
    for ch in off:
        y[ch] = np.nan
    with h5py.File(path, "w") as f:
        g = f.create_group("tangtv")
        g.create_dataset("xdata", data=t)
        g.create_dataset("ydata", data=y)


@pytest.fixture
def tangtv_shots(tmp_path):
    shots = []
    duration_s = 1.0 + 5 * 0.05 + 0.1  # room for several 50 ms windows past t0_start=1.0
    for i in range(4):
        sid = f"80000{i}"
        _write_tangtv_shot(tmp_path / f"{sid}_processed.h5", duration_s, seed=i)
        shots.append(sid)
    return {"dir": tmp_path, "shots": shots}


def _tiny_video_cfg() -> VideoCodecConfig:
    # small spatial grid; the dataset resizes the loader's 120x360 down to these (defensive fit)
    return VideoCodecConfig(
        channels=tc.modality_channels("tangtv_lower"),
        frames=5, height=20, width=30, patch_t=5, patch_h=10, patch_w=10,
        d_model=32, enc_depth=1, dec_depth=1, heads=2, fsq_levels=[4, 4, 3],
    )


def test_modality_channels_and_divertor_mapping():
    assert tc.modality_channels("tangtv_lower") == 2
    assert tc.modality_channels("tangtv_upper") == 2
    assert tc.video_divertor("tangtv_lower") == "lower"
    assert tc.video_divertor("tangtv_upper") == "upper"
    with pytest.raises(ValueError):
        tc.video_divertor("ece")


def test_video_dataset_reuses_parent_index_machinery(tangtv_shots):
    cfg = _tiny_video_cfg()
    ds = tc.VideoCodecPairDataset(
        "tangtv_lower", tangtv_shots["shots"], cfg, data_dir=tangtv_shots["dir"], seed=0
    )
    # IS a TokamakMultiFileDataset (reuses idx-map + LRU handles + length cache + pickling).
    assert isinstance(ds, TokamakMultiFileDataset)
    assert len(ds) > 1
    # overrides ONLY the transform hook — NOT __getitem__ / the searchsorted index map.
    assert "_getitem_standard" in vars(tc.VideoCodecPairDataset)
    assert "__getitem__" not in vars(tc.VideoCodecPairDataset)
    src = inspect.getsource(tc.VideoCodecPairDataset)
    assert "searchsorted" not in src, "must reuse the parent's binary-search index map"
    assert int(ds._cumulative_lengths[-1]) == len(ds)


def test_video_dataset_shapes_and_mask(tangtv_shots):
    cfg = _tiny_video_cfg()
    ds = tc.VideoCodecPairDataset(
        "tangtv_lower", tangtv_shots["shots"], cfg, data_dir=tangtv_shots["dir"], seed=0
    )
    for i in range(min(6, len(ds))):
        frames, frame_mask = ds[i]
        assert frames.shape == (cfg.channels, cfg.frames, cfg.height, cfg.width)
        assert frame_mask.shape == (cfg.frames,)
        assert torch.isfinite(frames).all()
        # this is NOT a δ-pair: a single frame tensor + a per-frame mask (no second view).
        assert frame_mask.dim() == 1


def test_video_loader_num_workers_0_and_2(tangtv_shots):
    cfg = _tiny_video_cfg()
    ds = tc.VideoCodecPairDataset(
        "tangtv_lower", tangtv_shots["shots"], cfg, data_dir=tangtv_shots["dir"], seed=1
    )
    loader0 = tc.make_video_loader(ds, batch_size=2, num_workers=0, seed=1)
    frames, mask = next(iter(loader0))
    assert frames.shape == (2, cfg.channels, cfg.frames, cfg.height, cfg.width)
    assert mask.shape == (2, cfg.frames)

    loader2 = tc.make_video_loader(ds, batch_size=2, num_workers=2, seed=2)
    seen = 0
    for frames, mask in loader2:
        assert frames.shape[1:] == (cfg.channels, cfg.frames, cfg.height, cfg.width)
        assert torch.isfinite(frames).all()
        seen += 1
        if seen >= 2:
            break
    assert seen >= 1


def test_video_loader_ddp_sampler_shards(tangtv_shots):
    cfg = _tiny_video_cfg()
    ds = tc.VideoCodecPairDataset(
        "tangtv_lower", tangtv_shots["shots"], cfg, data_dir=tangtv_shots["dir"], seed=0
    )
    l0 = tc.make_video_loader(ds, batch_size=1, num_workers=0, rank=0, world_size=2, seed=0)
    l1 = tc.make_video_loader(ds, batch_size=1, num_workers=0, rank=1, world_size=2, seed=0)
    assert isinstance(l0.sampler, tc.DistributedTwoLevelSampler)
    assert set(iter(l0.sampler)).isdisjoint(set(iter(l1.sampler)))


# --------------------------------------------------------------------------------------- #
# video_compute_gate — finite for a reconstructing input, -inf score on NaN
# --------------------------------------------------------------------------------------- #
def test_video_compute_gate_finite_for_reconstructing_input():
    """A codec that reconstructs its input yields finite gate metrics + a finite score.

    We fake a perfect reconstructor by monkey-hooking a trivially-reconstructing codec: use a
    tiny codec but assert only finiteness + ranges (a fresh codec won't hit envelope_corr high,
    so we drive the recon-floor to 0 to keep the score finite and prove the plumbing).
    """
    from tokamak_foundation_model.ignite import spike

    cfg = _small_cfg()
    cfg.gate_recon_floor = -1.0  # never disqualify: prove finiteness of the plumbing
    codec = VideoCodec(cfg)
    B, n_win = 2, 4
    clips = [torch.randn(B, cfg.channels, cfg.frames, cfg.height, cfg.width) for _ in range(2)]
    frame_seq = torch.randn(B, n_win, cfg.channels, cfg.frames, cfg.height, cfg.width)
    g = tc.video_compute_gate(codec, clips, frame_seq, cfg)
    assert 0.0 <= g["stability"] <= 1.0
    assert 0.0 <= g["persistence"] <= 1.0
    assert isinstance(g["pass_stability"], bool) and isinstance(g["pass_persistence"], bool)
    for k in ("envelope_corr", "peak_f1", "sharpness"):
        assert math.isfinite(g["decode"][k])
    score = spike.gate_score(g, recon_floor=cfg.gate_recon_floor)
    assert math.isfinite(score)


def test_video_gate_score_minus_inf_on_nan_recon():
    """gate_score must be -inf when the video decode envelope_corr is NaN (recon failure)."""
    from tokamak_foundation_model.ignite import spike

    g = {
        "forecastability": {"margin_transition": 0.0},
        "decode": {"envelope_corr": float("nan"), "peak_f1": 0.0, "sharpness": 1.0},
        "utilization": {"min_dim_entropy": 0.0, "frac_of_observable": 0.0},
    }
    assert spike.gate_score(g, recon_floor=0.2) == float("-inf")


def test_video_nuisance_is_brightness_gain_only():
    """The stability nuisance is a BRIGHTNESS/GAIN jitter only — no spatial roll.

    The divertor camera is FIXED, so a spatial shift is not a physical nuisance; the only
    realization noise is a camera gain fluctuation. So ``video_nuisance`` must be an affine
    per-clip level change (multiplicative gain + small additive offset) and must NOT roll the
    spatial axes — i.e. it must equal ``x * gain + bias`` for scalar ``gain``/``bias``.
    """
    cfg = _small_cfg()
    x = torch.randn(1, cfg.channels, cfg.frames, cfg.height, cfg.width)
    y = tc.video_nuisance(x, seed=0)
    assert y.shape == x.shape
    assert not torch.allclose(x, y)  # the level moved (realization changed)

    # It must be exactly an affine level change (gain*x + bias) — recover the scalars by
    # least squares and confirm the residual is ~0 (no per-pixel spatial roll would survive).
    xf = x.flatten()
    yf = y.flatten()
    var = float(((xf - xf.mean()) ** 2).mean())
    gain = float(((xf - xf.mean()) * (yf - yf.mean())).mean() / (var + 1e-12))
    bias = float(yf.mean() - gain * xf.mean())
    assert torch.allclose(y, x * gain + bias, atol=1e-5), (
        "video_nuisance must be a pure brightness/gain affine transform (no spatial roll)"
    )

    # A pure spatial roll (the DROPPED nuisance) is NOT affine in x, so its residual would be
    # large — sanity-check the discriminating power of the assertion above.
    rolled = torch.roll(x, shifts=(1, 1), dims=(-2, -1))
    rvar = float(((xf - xf.mean()) ** 2).mean())
    rg = float(((xf - xf.mean()) * (rolled.flatten() - rolled.mean())).mean() / (rvar + 1e-12))
    rb = float(rolled.mean() - rg * xf.mean())
    assert not torch.allclose(rolled, x * rg + rb, atol=1e-5)


def test_video_nuisance_no_spatial_roll_signature():
    """``video_nuisance`` no longer exposes a ``shift`` (spatial-roll) parameter."""
    import inspect
    params = inspect.signature(tc.video_nuisance).parameters
    assert "shift" not in params, "the spatial-roll 'shift' param must be gone"
    assert "bright" in params and "offset" in params  # brightness/gain only


# --------------------------------------------------------------------------------------- #
# per-step + full train_video_codec loop (single-process CPU)
# --------------------------------------------------------------------------------------- #
def test_video_train_step_reduces_pixel_anchor(tangtv_shots):
    torch.manual_seed(0)
    cfg = _tiny_video_cfg()
    codec = VideoCodec(cfg)
    disc_raw = FramePatchGAN(cfg)
    opt_g = torch.optim.Adam(codec.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(disc_raw.parameters(), lr=1e-3)

    ds = tc.VideoCodecPairDataset(
        "tangtv_lower", tangtv_shots["shots"], cfg, data_dir=tangtv_shots["dir"], seed=0
    )
    clips = [ds[i][0] for i in range(min(4, len(ds)))]
    x = torch.stack(clips, dim=0)
    m = torch.ones(x.shape[0], cfg.frames)

    pix = []
    for step in range(20):
        g_terms, d_loss = tc.video_codec_train_step(
            codec, disc_raw, opt_g, opt_d, x, m, cfg, step=step
        )
        pix.append(float(g_terms["pixel"].detach()))
        assert math.isfinite(pix[-1]) and math.isfinite(float(d_loss.detach()))
    assert sum(pix[-5:]) / 5 < sum(pix[:5]) / 5, (pix[0], pix[-1])


def test_train_video_codec_loop_writes_artifacts(tangtv_shots, tmp_path, monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    cfg = _tiny_video_cfg()
    out_dir = tmp_path / "vrun"
    final_gate = tc.train_video_codec(
        cfg,
        "tangtv_lower",
        tangtv_shots["shots"],
        tangtv_shots["shots"],  # eval on the same tiny pool (test only)
        steps=3,
        eval_every=1,
        batch_size=2,
        num_workers=0,
        data_dir=tangtv_shots["dir"],
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


def test_train_video_codec_cli_main(tangtv_shots, tmp_path, monkeypatch):
    """argparse main() dispatches --modality tangtv_lower to the video trainer."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    out_dir = tmp_path / "vcli"

    orig = tc.VideoCodecConfig

    def _small(**kw):
        kw.update(
            frames=5, height=20, width=30, patch_t=5, patch_h=10, patch_w=10,
            d_model=32, enc_depth=1, dec_depth=1, heads=2, fsq_levels=[4, 4, 3],
        )
        return orig(**kw)

    monkeypatch.setattr(tc, "VideoCodecConfig", _small)

    argv = [
        "--modality", "tangtv_lower",
        "--shots", ",".join(tangtv_shots["shots"]),
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
        "--data_dir", str(tangtv_shots["dir"]),
        "--seed", "0",
    ]
    gate_dict = tc.main(argv)
    assert gate_dict["steps"] == 2
    assert (out_dir / "summary.json").exists()
    assert sorted(out_dir.glob("gate_*.json"))
