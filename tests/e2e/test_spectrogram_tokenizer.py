"""Step 2 (Phase B spectrogram tokenizer) tests.

Tests the contract for ``SpectrogramTokenizer`` (Step 3) and
``SpectrogramOutputHead`` (Step 4) before either is implemented. Tests
will fail with ``ImportError`` until those modules land — that is the
TDD signal.

Architecture (plan-locked):

* Input: ``(B, C, F=512, T=98)`` for a 50 ms STFT window
  (n_fft=1024, hop=256, fs=500 kHz, DC dropped). Time axis is
  truncated to 96 internally for clean division by patch_t=8.
* Tokenizer: ``Conv2d(C, d_model, kernel=(patch_f, patch_t),
  stride=(patch_f, patch_t))`` matching layout (B, C, F, T). Each
  token has bounded receptive field (one patch). Add learned spatial
  PE per token + learned modality embedding.
* Output head: ``ConvTranspose2d(d_model, C, kernel=(patch_f, patch_t),
  stride=(patch_f, patch_t))``. Reconstructs ``(B, C, 512, 96)``
  (truncated time, not original 98).
* Per-modality patch sizes:
    - CO2: (F=64, T=8) → 8 × 12 = 96 tokens
    - ECE: (F=32, T=8) → 16 × 12 = 192 tokens
    - BES: (F=32, T=8) → 16 × 12 = 192 tokens
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tokamak_foundation_model.e2e.output_heads import SpectrogramOutputHead
from tokamak_foundation_model.e2e.tokenizers.spectrogram import (
    SpectrogramTokenizer,
)


# Plan-locked architecture defaults.
D_MODEL = 256
FREQ_BINS = 512
TIME_FRAMES = 98
TRUNC_T = 96   # time truncated to multiple of patch_t

# Per-modality config (channels, patch_f, patch_t).
MODALITIES = {
    "co2": dict(C=4,  patch_f=64, patch_t=8),
    "ece": dict(C=40, patch_f=32, patch_t=8),
    "bes": dict(C=16, patch_f=32, patch_t=8),
}


def _make_tokenizer(modality: str) -> SpectrogramTokenizer:
    cfg = MODALITIES[modality]
    return SpectrogramTokenizer(
        n_channels=cfg["C"],
        d_model=D_MODEL,
        patch_f=cfg["patch_f"],
        patch_t=cfg["patch_t"],
        freq_bins=FREQ_BINS,
        time_frames=TIME_FRAMES,
    )


def _make_output_head(modality: str) -> SpectrogramOutputHead:
    cfg = MODALITIES[modality]
    n_patches_f = FREQ_BINS // cfg["patch_f"]
    n_patches_t = TRUNC_T // cfg["patch_t"]
    return SpectrogramOutputHead(
        n_channels=cfg["C"],
        d_model=D_MODEL,
        patch_f=cfg["patch_f"],
        patch_t=cfg["patch_t"],
        n_patches_f=n_patches_f,
        n_patches_t=n_patches_t,
    )


def _expected_n_tokens(modality: str) -> int:
    cfg = MODALITIES[modality]
    return (FREQ_BINS // cfg["patch_f"]) * (TRUNC_T // cfg["patch_t"])


# ── Test 1 — Shape contract ──────────────────────────────────────────────


@pytest.mark.parametrize("modality", ["co2", "ece", "bes"])
def test_tokenizer_output_shape(modality):
    """Per-modality: ``(B, C, 512, 98) -> (B, n_tokens, 256)``.

    CO2: (4 → 96 tokens), ECE/BES: (40/16 → 192 tokens).
    """
    tok = _make_tokenizer(modality)
    cfg = MODALITIES[modality]
    x = torch.randn(2, cfg["C"], FREQ_BINS, TIME_FRAMES)
    out = tok(x)
    n_tokens = _expected_n_tokens(modality)
    assert out.shape == (2, n_tokens, D_MODEL), (
        f"{modality}: got {tuple(out.shape)}, expected (2, {n_tokens}, {D_MODEL})"
    )
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all()


# ── Test 2 — Frequency selectivity ──────────────────────────────────────


def test_frequency_selectivity():
    """Tokens for a narrowband 50 kHz signal differ from 200 kHz.

    Build a synthetic spectrogram with energy concentrated in one
    narrow frequency band; compare against the same shape with energy
    in a different band. With local F-patch tokenization, only a
    bounded set of tokens should change → cosine similarity well
    below 1.
    """
    cfg = MODALITIES["ece"]
    tok = _make_tokenizer("ece").eval()
    # Frequency axis: bin i ≈ (i+1) * fs/n_fft = (i+1) * 488 Hz (DC dropped).
    # 50 kHz ≈ bin 102, 200 kHz ≈ bin 409.
    spec_50k = torch.zeros(1, cfg["C"], FREQ_BINS, TIME_FRAMES)
    spec_50k[:, :, 100:104, :] = 1.0
    spec_200k = torch.zeros(1, cfg["C"], FREQ_BINS, TIME_FRAMES)
    spec_200k[:, :, 407:411, :] = 1.0
    with torch.no_grad():
        t_50 = tok(spec_50k)
        t_200 = tok(spec_200k)
    cos = F.cosine_similarity(
        t_50.flatten(1), t_200.flatten(1), dim=1
    ).item()
    assert cos < 0.9, (
        f"Frequency selectivity failed: cos_sim(50kHz, 200kHz) = {cos:.3f}"
    )


# ── Test 3 — Reconstruction round-trip ──────────────────────────────────


@pytest.mark.parametrize("modality", ["co2", "ece", "bes"])
def test_reconstruction_pipeline(modality):
    """Tokenizer + output head form a differentiable encode/decode pipe.

    Output reconstructs to ``(B, C, 512, 96)`` (truncated time, not 98).
    Gradients flow back into the tokenizer.
    """
    tok = _make_tokenizer(modality)
    head = _make_output_head(modality)
    cfg = MODALITIES[modality]
    x = torch.randn(1, cfg["C"], FREQ_BINS, TIME_FRAMES, requires_grad=False)

    tokens = tok(x)
    recon = head(tokens)

    expected = (1, cfg["C"], FREQ_BINS, TRUNC_T)
    assert recon.shape == expected, (
        f"{modality}: recon.shape = {tuple(recon.shape)}, expected {expected}"
    )
    assert torch.isfinite(recon).all()

    # Compare against truncated input so the loss is well-defined.
    target = x[..., :TRUNC_T]
    loss = (recon - target).abs().mean()
    loss.backward()
    grad_ok = any(
        (p.grad is not None) and (p.grad.abs().sum() > 0)
        for p in tok.parameters()
    )
    assert grad_ok, f"{modality}: no gradient reached the tokenizer"


# ── Test 4 — Memory gate (GPU only) ─────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU only")
def test_memory_gate_gpu():
    """All three spectrogram tokenizers + heads at batch=128 on a single
    GPU forward + backward without OOM.

    Per the plan's full-config attention budget (1178 tokens), each
    spectrogram tokenizer alone is small — this test guards the
    spectrogram-pipeline contribution to memory, not the full model.
    """
    device = torch.device("cuda")
    B = 128
    total_loss = torch.zeros((), device=device)
    for modality in ("co2", "ece", "bes"):
        tok = _make_tokenizer(modality).to(device)
        head = _make_output_head(modality).to(device)
        cfg = MODALITIES[modality]
        x = torch.randn(B, cfg["C"], FREQ_BINS, TIME_FRAMES, device=device)
        tokens = tok(x)
        recon = head(tokens)
        total_loss = total_loss + (recon - x[..., :TRUNC_T]).abs().mean()
    total_loss.backward()
    assert torch.isfinite(total_loss).item()


# ── Test 5 — Modality embedding distinctness ────────────────────────────


def test_modality_embeddings_distinct():
    """Two SpectrogramTokenizer instances initialise their
    ``modality_embed`` parameters to different values (independent
    Gaussian draws). Smoke-test on the same modality config so any
    distinctness comes from initialisation noise, not config diffs.
    """
    a = _make_tokenizer("ece")
    b = _make_tokenizer("ece")
    # The plan's init is ``nn.init.normal_(std=0.02)``; two
    # independent draws should be approximately orthogonal.
    cos = F.cosine_similarity(
        a.modality_embed.flatten().unsqueeze(0),
        b.modality_embed.flatten().unsqueeze(0),
        dim=1,
    ).item()
    assert abs(cos) < 0.5, (
        f"Modality embeddings unexpectedly aligned: cos = {cos:.3f}"
    )


# ── Test 6 — Time truncation ────────────────────────────────────────────


def test_time_truncation_invariance():
    """The last 2 time frames of the input ([..., 96:98]) must not
    influence the output — the tokenizer truncates to 96 before
    Conv2d. Replacing those frames with anything (zeros, noise) gives
    identical tokens.
    """
    tok = _make_tokenizer("ece").eval()
    cfg = MODALITIES["ece"]
    x = torch.randn(2, cfg["C"], FREQ_BINS, TIME_FRAMES)
    with torch.no_grad():
        out_a = tok(x)
        x_b = x.clone()
        x_b[..., TRUNC_T:] = 999.0   # garbage in the truncated region
        out_b = tok(x_b)
    assert torch.allclose(out_a, out_b), (
        "Tokens depend on truncated time region — truncation is leaking"
    )


# ── Test 7 — Missing-modality token (mirrors Phase C VideoTokenizer) ────


def test_missing_modality_token_replaces_absent_rows():
    """When ``mask=False`` for a row, the tokenizer outputs the learned
    ``missing_token`` for that row, identical to what a fully-missing
    batch would produce. Present rows match the no-mask path.
    """
    cfg = MODALITIES["ece"]
    tok = _make_tokenizer("ece").eval()
    x = torch.randn(3, cfg["C"], FREQ_BINS, TIME_FRAMES)
    # Mixed batch: row 0 present, row 1 absent, row 2 present.
    mask = torch.tensor([True, False, True])

    with torch.no_grad():
        no_mask_out = tok(x)             # all-present reference
        mixed_out = tok(x, mask=mask)
        # Reference: the learned missing_token expanded to a single row.
        missing_row = tok.missing_token.unsqueeze(0)   # (1, n_tokens, d_model)

    # Absent row equals the learned token.
    assert torch.allclose(mixed_out[1:2], missing_row), (
        "mask=False row should equal missing_token, not the encoded value"
    )
    # Present rows go through the encoder unchanged.
    assert torch.allclose(mixed_out[0:1], no_mask_out[0:1])
    assert torch.allclose(mixed_out[2:3], no_mask_out[2:3])


def test_all_absent_returns_only_missing_token():
    """``mask=all-False`` short-circuits the Conv2d path — all rows
    return the learned ``missing_token`` regardless of input."""
    cfg = MODALITIES["co2"]
    tok = _make_tokenizer("co2").eval()
    x = torch.randn(4, cfg["C"], FREQ_BINS, TIME_FRAMES)
    mask = torch.zeros(4, dtype=torch.bool)
    with torch.no_grad():
        out = tok(x, mask=mask)
    expected = tok.missing_token.expand(4, -1, -1)
    assert torch.allclose(out, expected)


def test_mask_none_equals_all_true():
    """``mask=None`` (default) is byte-identical to ``mask=all-True``,
    preserving backwards compatibility with code paths that don't pass
    a mask."""
    cfg = MODALITIES["bes"]
    tok = _make_tokenizer("bes").eval()
    x = torch.randn(2, cfg["C"], FREQ_BINS, TIME_FRAMES)
    mask = torch.ones(2, dtype=torch.bool)
    with torch.no_grad():
        a = tok(x)
        b = tok(x, mask=mask)
    assert torch.allclose(a, b)