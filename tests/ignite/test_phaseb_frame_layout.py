"""Phase-B foundation: DynamicsConfig frame layout + FrameTokenizer embed/logits round-trip."""

import pytest
import torch

from tokamak_foundation_model.ignite.dynamics_config import (
    FROZEN_MODALITIES,
    DynamicsConfig,
    ModalitySpec,
)
from tokamak_foundation_model.ignite.frame_layout import FrameTokenizer


def test_frame_layout_totals_1017():
    cfg = DynamicsConfig()
    assert cfg.tokens_per_frame == 1017           # 192*4 + 108*2 + 4*7 + 5*1 (fast-TS)
    assert cfg.n_modalities == 14                 # 4 spectro + 2 video + 7 slow-TS + 1 fast-TS
    assert cfg.max_frames == cfg.k0_seed + cfg.n_predict == 100  # 20 + 80


def test_confirmed_hyperparams():
    cfg = DynamicsConfig()
    assert cfg.d_model == 1024
    assert cfg.depth == 24            # start value; configurable for the depth study
    assert cfg.k0_seed == 20
    assert cfg.maskgit_decode_steps == 10
    assert cfg.ss_ramp_final_frac == pytest.approx(0.15)


def test_depth_is_a_free_knob():
    # the depth study must be a one-line sweep
    for k in (12, 24, 36, 48):
        assert DynamicsConfig(depth=k).depth == k


def test_modality_slices_are_contiguous_and_cover_the_frame():
    cfg = DynamicsConfig()
    slices = cfg.modality_token_slices()
    assert slices[0][0] == 0
    assert slices[-1][1] == cfg.tokens_per_frame
    for (a0, a1), (b0, b1) in zip(slices, slices[1:]):
        assert a1 == b0                          # no gaps / overlaps
    assert [e - s for s, e in slices] == [m.n_tok for m in cfg.modalities]


def _codes(cfg, B=2, F=3):
    return {
        m.name: torch.randint(0, m.codebook_size, (B, F, m.n_tok))
        for m in cfg.modalities
    }


def test_embed_shape_and_frame_offset():
    cfg = DynamicsConfig()
    tok = FrameTokenizer(cfg)
    codes = _codes(cfg, B=2, F=3)
    emb = tok.embed(codes, frame_offset=5)
    assert emb.shape == (2, 3, cfg.tokens_per_frame, cfg.d_model)
    # frame_offset beyond max_frames must raise
    with pytest.raises(ValueError):
        tok.embed(_codes(cfg, B=1, F=4), frame_offset=cfg.max_frames - 2)


def test_embed_missing_modality_raises():
    cfg = DynamicsConfig()
    tok = FrameTokenizer(cfg)
    codes = _codes(cfg)
    codes.pop("mse")
    with pytest.raises(ValueError):
        tok.embed(codes)


def test_logits_split_matches_per_modality_vocab():
    cfg = DynamicsConfig()
    tok = FrameTokenizer(cfg)
    B, F = 2, 3
    h = torch.randn(B, F, cfg.tokens_per_frame, cfg.d_model)
    logits = tok.logits(h)
    assert set(logits) == {m.name for m in cfg.modalities}
    for m in cfg.modalities:
        assert logits[m.name].shape == (B, F, m.n_tok, m.codebook_size)


def test_position_embeddings_distinguish_tokens():
    # two identical codes at different within-modality positions must embed differently
    cfg = DynamicsConfig(modalities=(ModalitySpec("x", "spectro", 4, 8),), d_model=16)
    tok = FrameTokenizer(cfg)
    codes = {"x": torch.zeros(1, 1, 4, dtype=torch.long)}  # same code at all 4 positions
    emb = tok.embed(codes)[0, 0]                            # (4, d)
    # positions differ -> rows differ (pos_embed breaks the tie)
    assert not torch.allclose(emb[0], emb[1])


def test_all_four_codec_families_present():
    names = [m.name for m in FROZEN_MODALITIES]
    assert "mse" in names and "co2" in names and "filterscopes" in names
    assert "tangtv_lower" in names and "tangtv_upper" in names          # video included
    assert len([m for m in FROZEN_MODALITIES if m.family == "spectro"]) == 4
    assert len([m for m in FROZEN_MODALITIES if m.family == "video"]) == 2
    assert len([m for m in FROZEN_MODALITIES if m.family == "slowts"]) == 7
    assert len([m for m in FROZEN_MODALITIES if m.family == "fastts"]) == 1   # the missed 4th family


def test_logits_last_matches_full_logits_last_frame():
    from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
    from tokamak_foundation_model.ignite.frame_layout import FrameTokenizer
    import torch

    cfg = DynamicsConfig(modalities=(ModalitySpec("a", "spectro", 3, 5),
                                     ModalitySpec("b", "slowts", 2, 4)),
                         d_model=16, depth=2, n_heads=2, k0_seed=2, n_predict=3)
    torch.manual_seed(0)
    tok = FrameTokenizer(cfg).eval()
    h = torch.randn(2, 4, cfg.tokens_per_frame, cfg.d_model)
    full = tok.logits(h)
    last = tok.logits_last(h)
    for m in cfg.modalities:
        assert last[m.name].shape == (2, m.n_tok, m.codebook_size)
        assert torch.allclose(last[m.name], full[m.name][:, -1], atol=0, rtol=0)
