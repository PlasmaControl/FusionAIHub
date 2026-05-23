"""§5.4 tests for the Phase C tube-patch video tokenizer.

The Perceiver-pool design (16 / 32 global queries) was abandoned after
three iterations plateaued at ~0.62 ratio on plasma channels with
featureless reconstructions — global tokens cannot encode unbounded
local spatial structure with bounded count, regardless of decoder
shape.

The tube-patch design (VideoMAE-style) replaces the global pool with
local patches: a 3D conv with kernel and stride equal to the patch
size produces one token per spatiotemporal patch. With patch
``(3, 12, 12)`` over ``(C, T=3, H=120, W=360)`` input, this yields
``(120/12) * (360/12) = 300`` tokens per camera per 50 ms window. Each
token represents a bounded ``7 x 3 x 12 x 12 = 3024`` pixel region.

Contract:

1. **Shape**: ``(B, 2, 3, 120, 360) -> (B, 300, 256)``.
2. **Spatial selectivity**: a bright patch on one side is encoded
   distinguishably from an identical input without it.
3. **Motion detection**: a moving object yields different tokens from
   the same object held static across frames.
4. **Reconstruction round-trip**: tokenizer + output head are an
   approximate inverse. At init, recon shape matches input shape and
   gradients flow.
5. **Memory (OOM)**: full-batch forward+backward fits on an A100 40 GB.
   GPU-only.
6. **Missing camera**: ``mask=False`` -> learned ``missing_token``.
7. **Modality embedding distinctness**: changing only ``modality_emb``
   changes the output.
8. **Patch locality**: modifying a corner of the input only changes
   the corner-region tokens, not far-away tokens — this is the
   structural property that makes per-patch reconstruction work.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tokamak_foundation_model.e2e.output_heads import VideoOutputHead
from tokamak_foundation_model.e2e.tokenizers.video import VideoTokenizer


# Plan-locked architecture defaults.
N_CHANNELS = 2
N_FRAMES = 3
PATCH_SIZE = (3, 12, 12)            # (T, H, W)
SPATIAL_HW = (120, 360)
N_H = SPATIAL_HW[0] // PATCH_SIZE[1]   # 10
N_W = SPATIAL_HW[1] // PATCH_SIZE[2]   # 30
N_TOKENS = N_H * N_W                   # 300
D_MODEL = 256


def _make_tokenizer() -> VideoTokenizer:
    return VideoTokenizer(
        n_channels=N_CHANNELS,
        n_frames=N_FRAMES,
        patch_size=PATCH_SIZE,
        d_model=D_MODEL,
        spatial_size=SPATIAL_HW,
    )


def _make_output_head() -> VideoOutputHead:
    return VideoOutputHead(
        n_channels=N_CHANNELS,
        n_frames=N_FRAMES,
        patch_size=PATCH_SIZE,
        d_model=D_MODEL,
        spatial_size=SPATIAL_HW,
    )


def _zero_input(batch: int = 1) -> torch.Tensor:
    return torch.zeros(batch, N_CHANNELS, N_FRAMES, *SPATIAL_HW)


# ── Test 1 — Shape contract ──────────────────────────────────────────────


def test_tokenizer_output_shape():
    """tangtv ``(B, 2, 3, 120, 360) -> (B, 300, 256)``."""
    tok = _make_tokenizer()
    x = torch.randn(2, N_CHANNELS, N_FRAMES, *SPATIAL_HW)
    out = tok(x)
    assert out.shape == (2, N_TOKENS, D_MODEL)
    assert out.dtype == x.dtype


# ── Test 2 — Spatial selectivity ────────────────────────────────────────


def test_spatial_selectivity():
    """A bright patch on one side gives distinguishable tokens from a
    plain frame. With local patches the test is decisive: at most a
    handful of tokens should change, and the global cosine should drop
    well below 1.0.
    """
    tok = _make_tokenizer().eval()
    bright = _zero_input()
    bright[:, :, :, :60, :180] = 1.0   # top-left quadrant bright

    plain = _zero_input()

    with torch.no_grad():
        out_bright = tok(bright)
        out_plain = tok(plain)

    cos = F.cosine_similarity(
        out_bright.flatten(1), out_plain.flatten(1), dim=1
    ).item()
    assert cos < 0.85, (
        f"Spatial selectivity failed: cos_sim(bright, plain) = {cos:.3f}"
    )


# ── Test 3 — Motion detection ────────────────────────────────────────────


def test_motion_detection():
    """Tokens for a moving object differ from the same object held
    static. Each tube token convolves over 3 frames, so different
    temporal content is directly encoded into each token.
    """
    tok = _make_tokenizer().eval()

    static = _zero_input()
    static[:, :, :, 24:36, 24:36] = 1.0   # same square in all 3 frames

    moving = _zero_input()
    moving[:, :, 0, 24:36, 24:36] = 1.0
    moving[:, :, 1, 24:36, 60:72] = 1.0
    moving[:, :, 2, 24:36, 96:108] = 1.0

    with torch.no_grad():
        out_static = tok(static)
        out_moving = tok(moving)

    cos = F.cosine_similarity(
        out_static.flatten(1), out_moving.flatten(1), dim=1
    ).item()
    assert cos < 0.9, (
        f"Motion detection failed: cos_sim(static, moving) = {cos:.3f}"
    )


# ── Test 4 — Reconstruction round-trip ──────────────────────────────────


def test_reconstruction_pipeline():
    """Tokenizer + output head are a differentiable encode/decode pipe.

    With local-patch architecture the inverse is structurally clean:
    ``Conv3d(stride=p)`` followed by ``ConvTranspose3d(stride=p)``.
    We require shape match, finite output, and nonzero gradients on
    the tokenizer.
    """
    tok = _make_tokenizer()
    head = _make_output_head()
    x = torch.randn(1, N_CHANNELS, N_FRAMES, *SPATIAL_HW, requires_grad=False)

    tokens = tok(x)
    recon = head(tokens)

    expected_shape = (1, N_FRAMES, N_CHANNELS, *SPATIAL_HW)
    assert recon.shape == expected_shape, (
        f"recon.shape = {recon.shape}, expected {expected_shape}"
    )
    assert torch.isfinite(recon).all()

    loss = (recon - x.permute(0, 2, 1, 3, 4)).abs().mean()
    loss.backward()
    grad_seen = any(
        (p.grad is not None) and (p.grad.abs().sum() > 0)
        for p in tok.parameters()
    )
    assert grad_seen, "No nonzero gradient flowed back to the tokenizer."


# ── Test 5 — Full-size forward+backward fits on A100 40 GB ──────────────


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="OOM gate is GPU-only; run on a node with a 40 GB A100.",
)
def test_full_size_forward_no_oom():
    """batch=128 forward+backward through tokenizer+head must not OOM."""
    device = torch.device("cuda")
    tok = _make_tokenizer().to(device)
    head = _make_output_head().to(device)
    x = torch.randn(
        128, N_CHANNELS, N_FRAMES, *SPATIAL_HW,
        device=device, requires_grad=False,
    )
    tokens = tok(x)
    recon = head(tokens)
    loss = recon.mean()
    loss.backward()
    torch.cuda.synchronize()


# ── Test 6 — Missing-camera token ───────────────────────────────────────


def test_missing_camera_returns_learned_token():
    """``mask=False`` -> learned ``missing_token`` (not zeros, not data)."""
    tok = _make_tokenizer().eval()
    with torch.no_grad():
        tok.missing_token.copy_(torch.randn_like(tok.missing_token) * 0.5)

    x = torch.randn(2, N_CHANNELS, N_FRAMES, *SPATIAL_HW)
    mask_all_present = torch.ones(2, dtype=torch.bool)
    mask_all_missing = torch.zeros(2, dtype=torch.bool)

    with torch.no_grad():
        out_present = tok(x, mask=mask_all_present)
        out_missing = tok(x, mask=mask_all_missing)

    assert not torch.allclose(out_missing, torch.zeros_like(out_missing))
    assert torch.allclose(out_missing[0], out_missing[1])
    expected = tok.missing_token.expand(2, -1, -1)
    assert torch.allclose(out_missing, expected)
    cos = F.cosine_similarity(
        out_present.flatten(1), out_missing.flatten(1), dim=1
    ).mean().item()
    assert cos < 0.99, (
        f"Missing token too close to data-driven output: cos = {cos:.3f}"
    )


# ── Test 7 — Modality embedding distinctness ────────────────────────────


def test_modality_embedding_changes_output():
    """Changing only ``modality_emb`` changes the tokenizer output."""
    torch.manual_seed(0)
    tok_a = _make_tokenizer().eval()
    tok_b = _make_tokenizer().eval()
    tok_b.load_state_dict(tok_a.state_dict())

    with torch.no_grad():
        tok_b.modality_emb.copy_(
            torch.randn_like(tok_b.modality_emb) * 0.1
        )

    x = torch.randn(2, N_CHANNELS, N_FRAMES, *SPATIAL_HW)
    with torch.no_grad():
        out_a = tok_a(x)
        out_b = tok_b(x)

    cos = F.cosine_similarity(
        out_a.flatten(1), out_b.flatten(1), dim=1
    ).mean().item()
    assert cos < 0.99, (
        f"Modality embedding had no effect on output: cos = {cos:.3f}"
    )


# ── Test 8 — Patch locality ─────────────────────────────────────────────


def test_patch_locality():
    """Modifying a single corner patch should not perturb far-away tokens.

    This is the structural property that makes per-patch reconstruction
    work. With ``Conv3d(stride=patch)`` patch embedding the receptive
    field of each output token is exactly one ``(T, H, W)`` patch, so
    a perturbation in patch (0, 0) cannot affect token at index
    (n_h - 1, n_w - 1) — and so on.
    """
    tok = _make_tokenizer().eval()
    base = _zero_input()
    perturbed = base.clone()
    perturbed[:, :, :, : PATCH_SIZE[1], : PATCH_SIZE[2]] = 1.0

    with torch.no_grad():
        tokens_base = tok(base).reshape(1, N_H, N_W, D_MODEL)
        tokens_pert = tok(perturbed).reshape(1, N_H, N_W, D_MODEL)

    diff = (tokens_base - tokens_pert).abs().sum(dim=-1)   # (1, n_h, n_w)
    diff = diff[0]

    # The (0, 0) token *must* see the change — non-trivial difference.
    assert diff[0, 0].item() > 1e-3, (
        "Top-left token did not change when its own patch was perturbed."
    )
    # Tokens far from the perturbation must be untouched (modulo the
    # shared modality embedding offset which is constant). We test
    # against the (n_h - 1, n_w - 1) token, the farthest corner.
    far_diff = diff[N_H - 1, N_W - 1].item()
    assert far_diff < 1e-5, (
        f"Far token changed by {far_diff:.3e} when only the opposite "
        "corner patch was perturbed — patch locality is violated."
    )
