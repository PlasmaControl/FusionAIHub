"""Regression guard: EVERY IGNITE codec family's encoder input must be sane (finite + O(1)).

A systematic audit found several codec data paths feeding the encoder inputs that did NOT
match the FM's per-modality preprocessing — raw pixels (video, std ~5-40), raw ~1e15
filterscope counts (fast-TS), raw ~1e19 Thomson density (slow-TS), a large-negative
standardized-missing artifact (~-25, slow-TS missing positions), FLT_MAX sentinels (co2).
Each is a "scale leak" that collapses the codec. This module is the standing guard so that
class of bug cannot recur silently: it asserts, on synthetic REPRESENTATIVE inputs, that the
tensor the encoder actually receives for each family is

    * finite (no NaN / inf), and
    * O(1)-bounded: ``abs-max < ABS_MAX_CEIL`` and present-value ``std ∈ [STD_LO, STD_HI]``.

A raw-scale leak (1e14 / 1e19 / raw camera pixels / the -25 missing artifact) FAILS the
``assert_encoder_input_sane`` check; the standardized/O(1) input PASSES. The parametrized
family list (:data:`ENCODER_INPUT_BUILDERS`) is the extension point: adding a new modality =
add one ``(name, builder)`` entry, and it is checked automatically.

Reuse boundary (§7): only ``torch`` + sibling ``ignite`` modules (data transforms + codecs) +
``e2e.multimodal`` (the video-standardize MECHANISM the codec mirrors). No FAITH *model* code
is trained/loaded; no GPU / SLURM / real data (small synthetic tensors only).

Run:
    .pixi/envs/default/bin/python -m pytest tests/ignite/test_encoder_input_scale.py -q
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import pytest
import torch

from tokamak_foundation_model.ignite import data
from tokamak_foundation_model.ignite import train_codec as tc
from tokamak_foundation_model.ignite.config import (
    FastTSCodecConfig,
    SlowTSCodecConfig,
    SpectroCodecConfig,
    VideoCodecConfig,
)
from tokamak_foundation_model.ignite.video_codec import VideoCodec


# ------------------------------------------------------------------------------------- #
# the O(1) sanity band (the guard). A raw-scale leak blows through ABS_MAX_CEIL; a
# collapsed-constant input has ~0 std; a healthy standardized input sits comfortably inside.
# ------------------------------------------------------------------------------------- #
ABS_MAX_CEIL: float = 50.0      # |input| must stay well under this (raw 1e14/1e19/pixels fail)
STD_LO: float = 0.1             # present-value std floor (a collapsed constant fails)
STD_HI: float = 5.0             # present-value std ceiling (a raw-scale spread fails)


def assert_encoder_input_sane(
    x: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
    name: str = "input",
) -> None:
    """Assert ``x`` (the tensor the encoder receives) is finite + O(1)-bounded.

    ``valid`` (optional, same shape as ``x``, 1 = present) restricts the std check to PRESENT
    positions — missing positions are expected to be neutral (0) and would otherwise deflate the
    std. The abs-max / finiteness checks always cover the WHOLE tensor (a missing-position
    artifact like the slow-TS -25 must still fail abs-max via the present-only std is not enough).
    """
    assert torch.isfinite(x).all(), f"{name}: encoder input has non-finite values"
    abs_max = float(x.abs().max())
    assert abs_max < ABS_MAX_CEIL, (
        f"{name}: encoder input abs-max {abs_max:.3g} >= {ABS_MAX_CEIL} — a RAW-SCALE LEAK "
        f"(unstandardized raw / a large-negative missing artifact) is reaching the encoder"
    )
    vals = x if valid is None else x[valid > 0.5]
    if vals.numel() > 1:
        std = float(vals.std())
        assert STD_LO <= std <= STD_HI, (
            f"{name}: present-value std {std:.3g} outside [{STD_LO}, {STD_HI}] — either a "
            f"collapsed-constant input (std~0) or a raw-scale spread (std>>1)"
        )


# ------------------------------------------------------------------------------------- #
# per-family synthetic REPRESENTATIVE encoder-input builders. Each returns
# ``(encoder_input, valid_or_None)`` — the exact tensor the codec's encoder consumes, built by
# the SAME code path production uses (data transform / codec standardize / dataset _fit_window).
# ------------------------------------------------------------------------------------- #
def _spectro_input() -> Tuple[torch.Tensor, None]:
    """Spectro encoder input = data.log_power_stft(raw) — log-power, O(1) when present."""
    cfg = SpectroCodecConfig(channels=1)
    torch.manual_seed(0)
    # a realistic bounded raw signal at STFT_FS (a few-Hz-ish oscillation + noise), |raw| << 1e20.
    W = cfg.window_samples
    t = torch.linspace(0, 1, W)
    raw = (torch.sin(2 * torch.pi * 40.0 * t) + 0.1 * torch.randn(W)).reshape(1, 1, W)
    return data.log_power_stft(raw, cfg), None


def _video_input() -> Tuple[torch.Tensor, None]:
    """Video encoder input = VideoCodec.standardize_input(raw_pixels) — per-(B,C) z-score."""
    cfg = VideoCodecConfig(channels=2, frames=4, height=16, width=24,
                           patch_t=2, patch_h=8, patch_w=8)
    torch.manual_seed(0)
    # RAW camera pixels: per-(B,C) mean ~130, std ~40 (uint8-ish regime) — NOT O(1).
    base = torch.rand(2, cfg.channels, 1, 1, 1) * 100.0 + 80.0
    raw = base + torch.randn(2, cfg.channels, cfg.frames, cfg.height, cfg.width) * 40.0
    return VideoCodec.standardize_input(raw), None


def _slowts_input() -> Tuple[torch.Tensor, torch.Tensor]:
    """Slow-TS encoder input = _fit_window(raw, nan_mask)[0] — standardized + missing→0."""
    C, T = 6, 5
    torch.manual_seed(0)
    raw = (torch.rand(C, T) * 4.0 + 1.0) * 1e19        # ~1e19 Thomson density
    raw[: (2 * C) // 3] = 0.0                            # ⅔ missing (raw 0)
    nan_mask = torch.zeros(C, T)
    log_raw = torch.log10(raw.clamp(min=-0.99) + 1.0)
    present = raw != 0.0
    mean, std = [], []
    for c in range(C):
        v = log_raw[c][present[c]]
        mean.append(float(v.mean()) if v.numel() else 0.0)
        std.append(float(v.std().clamp(min=1e-3)) if v.numel() else 1.0)
    cfg = SlowTSCodecConfig(signal="ts_core_density", channels=C, time_steps=T,
                            n_zones=1, patch_c=C, patch_t=T)
    cfg.preprocess_method = "log_standardize"
    cfg.channel_mean = mean
    cfg.channel_std = std
    ds = tc.SlowTSCodecPairDataset.__new__(tc.SlowTSCodecPairDataset)
    ds.h5_file = None
    ds.codec_cfg = cfg
    signal, valid = ds._fit_window(raw, nan_mask)
    return signal, valid


def _fastts_input() -> Tuple[torch.Tensor, None]:
    """Fast-TS encoder input = data.elm_envelope(raw, cfg) — standardized ELM envelope."""
    C = 8
    cfg = FastTSCodecConfig(channels=C)
    torch.manual_seed(0)
    W = cfg.window_samples
    # RAW filterscope counts ~1e14 with ELM bursts that GROW across the window (so the standardized
    # per-bin RMS envelope has real spread — mirrors the documented real-data env std p50 ~0.45,
    # NOT a flat saturated line). Bursts recur ~every 16 samples with a rising amplitude ramp.
    raw = (torch.randn(1, C, W).abs() * 1e13)
    ramp = torch.linspace(0.2, 4.0, W)                   # activity ramps up over the 50 ms window
    burst = torch.zeros(W)
    burst[::16] = 1.0
    raw += (burst * ramp).reshape(1, 1, W) * 5e14        # ELM bursts, growing amplitude
    # real per-channel stats (as the loader's preprocessing_stats.pt would carry).
    std = raw[0].std(dim=-1)
    mean = raw[0].mean(dim=-1)
    cfg.channel_mean = mean.tolist()
    cfg.channel_std = std.tolist()
    return data.elm_envelope(raw, cfg), None


ENCODER_INPUT_BUILDERS: Dict[str, Callable[[], Tuple[torch.Tensor, torch.Tensor | None]]] = {
    "spectro": _spectro_input,
    "video": _video_input,
    "slow_ts": _slowts_input,
    "fast_ts": _fastts_input,
}


# ------------------------------------------------------------------------------------- #
# THE GUARD — every codec family's encoder input is finite + O(1) on representative inputs.
# ------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", list(ENCODER_INPUT_BUILDERS))
def test_encoder_input_is_sane_for_every_family(family):
    x, valid = ENCODER_INPUT_BUILDERS[family]()
    assert_encoder_input_sane(x, valid=valid, name=family)


# ------------------------------------------------------------------------------------- #
# NEGATIVE controls — the guard MUST FAIL on a raw-scale leak (so it actually protects us).
# ------------------------------------------------------------------------------------- #
def test_guard_fails_on_raw_pixel_leak():
    """Raw camera pixels (std ~40, range [16, 240]) — the pre-fix video leak — must FAIL."""
    raw = torch.rand(2, 2, 4, 16, 24) * 224.0 + 16.0
    with pytest.raises(AssertionError):
        assert_encoder_input_sane(raw, name="raw_pixels")


def test_guard_fails_on_1e19_slowts_leak():
    """Unstandardized ~1e19 Thomson density — the pre-fix slow-TS leak — must FAIL."""
    raw = (torch.rand(6, 5) * 4.0 + 1.0) * 1e19
    with pytest.raises(AssertionError):
        assert_encoder_input_sane(raw, name="raw_1e19")


def test_guard_fails_on_1e14_fastts_leak():
    """Unstandardized ~1e14 filterscope counts — the pre-fix fast-TS leak — must FAIL."""
    raw = torch.randn(1, 8, 500).abs() * 1e14
    with pytest.raises(AssertionError):
        assert_encoder_input_sane(raw, name="raw_1e14")


def test_guard_fails_on_large_negative_missing_artifact():
    """The slow-TS standardized-missing artifact (~-25 mass) — pre-fix input — must FAIL abs-max."""
    x = torch.randn(6, 5) * 0.8            # present values O(1)
    x[:4] = -25.0                          # ⅔ positions at the -25 missing artifact
    with pytest.raises(AssertionError):
        assert_encoder_input_sane(x, name="missing_artifact")


def test_guard_fails_on_collapsed_constant():
    """A collapsed near-constant input (std ~0) must FAIL the std floor."""
    x = torch.full((2, 4, 8), 0.3) + 1e-6 * torch.randn(2, 4, 8)
    with pytest.raises(AssertionError):
        assert_encoder_input_sane(x, name="collapsed")


def test_guard_passes_on_clean_standardized_input():
    """A clean O(1) standardized input PASSES (the guard is not over-tight)."""
    x = torch.randn(2, 4, 8)               # ~N(0,1)
    assert_encoder_input_sane(x, name="clean")
